#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/cuda_execution_domain_fixture.h"
#include "../transform/cuda_image_fixture.h"

#include <cuda_runtime_api.h>
#include <fatbinary_section.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/syscall.h>

// Exact NVIDIA compiler callback ABI, also exercised by the ordinary Runtime fixture.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** __cudaRegisterFatBinary(void*);
extern "C" void __cudaRegisterFatBinaryEnd(void**);
extern "C" void __cudaUnregisterFatBinary(void**);
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                       dim3*, dim3*, int*);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" ssize_t __real_send(int, const void*, std::size_t, int);

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kBlocks = 8, kGuard = 0x91ae73c5U;
constexpr std::size_t kOutputBytes = (kBlocks + 2) * sizeof(std::uint32_t);

struct Scenario {
    std::string_view name;
    bool is_runtime{}, is_cached{}, is_packed{}, is_prepared{}, checks_capacity{};
    bool fails_issue{}, fails_preparation{}, resets_elsewhere{}, holds_physical{};
    bool loses_reply{}, uses_legacy_thread{}, cancels_admission{};
    unsigned int commands{2};
};
struct alignas(64) HostArena {
    std::array<std::uint64_t, 8> before{};
    std::array<std::byte, 192> bytes{};
    std::array<std::uint64_t, 8> after{};
    CUcontext context{};
    bool is_live{};
};
struct State {
    std::atomic<bool> callback_entered{}, callback_release{}, callback_returned{};
    std::atomic<bool> has_watchdog{}, early_returns{}, cleanup_entered{}, fault_observed{};
    std::atomic<bool> all_issued{}, physical_release{};
    std::atomic<bool> error_tail_pending{}, error_observed_while_pending{};
    std::atomic<bool> legacy_routes_owned{true};
    CUevent error_tail_event{};
    std::atomic<bool> batch_ready{}, counts_observations{};
    std::atomic<unsigned int> local_error_sends{};
    std::atomic<bool> holds_batch_reply{}, batch_reply_held{}, batch_sync_observed{};
    ExecutionAsyncFrontierKey batch_observed_key;
    unsigned int batch_sync_remote_queries{};
    std::int32_t batch_sync_remote_status{CUDA_ERROR_UNKNOWN};
    std::vector<std::byte> delayed_batch_reply;
    std::size_t delayed_batch_offset{};
    int delayed_batch_descriptor{-1};
    bool batch_reply_delivered{};
    std::atomic<unsigned int> returned{};
    std::atomic<bool> capacity_call_started{}, capacity_returned_early{};
    std::atomic<bool> capacity_busy_reply{}, capacity_cancel_returned{};
    std::atomic<int> capacity_result{CUDA_ERROR_UNKNOWN};
    std::atomic<bool> held_receipt{}, lost_reply_checked{}, invalid_loss_frame{};
    unsigned int held_sends{}, lost_sends{};
    std::uint64_t loss_sequence{}, loss_frontier{};
    const std::byte* native_parameters{};
    std::array<std::byte, 12> parameter_snapshot{};
    unsigned int numerical_allocations{}, numerical_values{}, successful_launches{};
    unsigned int cleanup_before_callback{}, cleanup_before_physical{}, native_attempts{},
        preparation_failures{};
    unsigned int batch_queries{}, batch_releases{}, single_queries{}, single_releases{},
        queried_items{}, released_items{};
    bool invalid_batch_reply{};
    std::array<std::uint32_t, kBlocks + 2> expected{};
    std::array<HostArena, 4> hosts{};
    Scenario scenario;
} state;
thread_local bool observes_local_error_query{};
struct LocalErrorQueryScope {
    LocalErrorQueryScope() {
        observes_local_error_query = true;
    }
    ~LocalErrorQueryScope() {
        observes_local_error_query = false;
    }
    LocalErrorQueryScope(const LocalErrorQueryScope&) = delete;
    LocalErrorQueryScope& operator=(const LocalErrorQueryScope&) = delete;
};

using ThreadCall = cudaError_t(CUDARTAPI*)();
ThreadCall legacy_thread_call(bool is_reset) {
    const auto* name = is_reset ? "cudaThreadExit" : "cudaThreadSynchronize";
    auto* selected = ::dlsym(RTLD_DEFAULT, name);
    const auto modern = is_reset ? &cudaDeviceReset : &cudaDeviceSynchronize;
    Dl_info candidate{}, owned{};
    if (!selected || !::dladdr(selected, &candidate) ||
        !::dladdr(reinterpret_cast<void*>(modern), &owned) ||
        candidate.dli_fbase != owned.dli_fbase) {
        // Test-only RED cleanup: never invoke an unknown client-native entry.
        // The normal final oracle still rejects this missing owned route.
        state.legacy_routes_owned.store(false, std::memory_order_release);
        return modern;
    }
    return reinterpret_cast<ThreadCall>(selected);
}
cudaError_t reset_runtime() {
    return state.scenario.uses_legacy_thread ? legacy_thread_call(true)() : cudaDeviceReset();
}

void check(CUresult value) {
    if (value != CUDA_SUCCESS)
        throw std::runtime_error("public Driver status=" + std::to_string(value));
}
void check(cudaError_t value) {
    if (value != cudaSuccess)
        throw std::runtime_error("public Runtime status=" + std::to_string(value));
}
void wait_for(const std::atomic<bool>& flag) {
    const auto until = Clock::now() + std::chrono::seconds{5};
    while (!flag.load(std::memory_order_acquire)) {
        expect(Clock::now() < until, "bounded actual callback observation");
        std::this_thread::yield();
    }
}
void CUDART_CB barrier(void*) {
    state.callback_entered.store(true, std::memory_order_release);
    wait_for(state.callback_release);
    state.callback_returned.store(true, std::memory_order_release);
}
std::string source() {
    std::string result(kSource);
    const auto position = result.find(" add.u32 %r4, %r4, %r2;");
    expect(position != std::string::npos, "original fixture has the exact affine insertion site");
    result.insert(position, " mul.lo.u32 %r4, %r4, 3;\n");
    return result;
}
void verify_output(const FakeAllocation& allocation) {
    if (allocation.bytes != kOutputBytes) return;
    std::array<std::uint32_t, kBlocks + 2> observed{};
    std::memcpy(observed.data(), allocation.data.data(), kOutputBytes);
    expect(observed == state.expected, "all non-idempotent values and both guards match exactly");
    ++state.numerical_allocations;
    state.numerical_values += static_cast<unsigned int>(observed.size());
}
CudaSessionDispatch dispatch() {
    auto result = pair_dispatch();
    result.host_allocate = [](void** output, std::size_t bytes, unsigned int) {
        const auto found =
            std::ranges::find_if(state.hosts, [](const auto& host) { return !host.is_live; });
        expect(driver.current && bytes == 192 && found != state.hosts.end(),
               "callback plus two kernels own three bounded aligned HostSlots");
        found->before.fill(kGuard);
        found->after.fill(kGuard);
        found->bytes.fill(std::byte{});
        found->context = driver.current;
        found->is_live = true;
        *output = found->bytes.data();
        return CUDA_SUCCESS;
    };
    result.host_map = [](CUdeviceptr* output, void* pointer, unsigned int) {
        expect(std::ranges::any_of(state.hosts,
                                   [pointer](const auto& host) {
                                       return host.is_live && host.context == driver.current &&
                                              host.bytes.data() == pointer;
                                   }),
               "map the exact owned three-slot host arena");
        *output = reinterpret_cast<CUdeviceptr>(pointer);
        return CUDA_SUCCESS;
    };
    result.host_release = [](void* pointer) {
        const auto found = std::ranges::find_if(state.hosts, [pointer](const auto& host) {
            return host.is_live && host.context == driver.current && host.bytes.data() == pointer;
        });
        expect(found != state.hosts.end() &&
                   std::ranges::all_of(found->before, [](auto value) { return value == kGuard; }) &&
                   std::ranges::all_of(found->after, [](auto value) { return value == kGuard; }),
               "retire each exact three-slot arena once with both boundary guards unchanged");
        found->is_live = false;
        ++driver.host_frees;
        return CUDA_SUCCESS;
    };
    // Exact CUDA callback ABI cannot regroup its adjacent scalar arguments.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    result.occupancy = [](int* blocks, CUfunction function, int threads, std::size_t shared) {
        if (state.scenario.fails_preparation && state.callback_entered.load() &&
            state.preparation_failures == 0) {
            ++state.preparation_failures;
            state.fault_observed.store(true, std::memory_order_release);
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        }
        return occupancy(blocks, function, threads, shared);
    };
    result.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        const auto status = allocate(output, bytes);
        if (status == CUDA_SUCCESS && bytes == kOutputBytes) {
            state.expected.fill(kGuard);
            for (unsigned int index = 0; index != kBlocks; ++index)
                state.expected[index + 1] = 11U + index * 17U;
            std::memcpy(device_bytes(*output, bytes), state.expected.data(), bytes);
        }
        return status;
    };
    result.release = [](CUdeviceptr pointer) {
        const auto& owned = allocation(pointer, 0);
        verify_output(owned);
        if (owned.bytes == kOutputBytes && !state.callback_returned.load())
            ++state.cleanup_before_callback;
        if (owned.bytes == kOutputBytes && state.scenario.holds_physical &&
            !state.physical_release.load())
            ++state.cleanup_before_physical;
        return release(pointer);
    };
    result.context_destroy = [](CUcontext context) {
        for (const auto& owned : driver.allocations)
            if (owned.is_live && owned.context == context) verify_output(owned);
        if (!state.callback_returned.load()) ++state.cleanup_before_callback;
        if (state.scenario.holds_physical && !state.physical_release.load())
            ++state.cleanup_before_physical;
        return context_destroy(context);
    };
    result.module_unload = [](CUmodule module) {
        if (state.cleanup_entered.load() && !state.callback_returned.load())
            ++state.cleanup_before_callback;
        if (state.cleanup_entered.load() && state.scenario.holds_physical &&
            !state.physical_release.load())
            ++state.cleanup_before_physical;
        return module_unload(module);
    };
    result.stream_destroy = [](CUstream stream) {
        if (state.cleanup_entered.load() && !state.callback_returned.load())
            ++state.cleanup_before_callback;
        if (state.cleanup_entered.load() && state.scenario.holds_physical &&
            !state.physical_release.load())
            ++state.cleanup_before_physical;
        return stream_destroy(stream);
    };
    result.execution.event_create = [](CUevent* event, unsigned int flags) {
        const auto available =
            std::ranges::find_if(driver.events, [](const auto& value) { return !value.is_live; });
        expect(available != driver.events.end(), "bounded event arena can reuse retired slots");
        driver.next_event = static_cast<std::size_t>(available - driver.events.begin());
        return event_create(event, flags);
    };
    result.execution.event_query = [](CUevent event) {
        const auto status = event_query(event);
        if (status == CUDA_SUCCESS && event == state.error_tail_event &&
            !state.physical_release.load()) {
            state.error_tail_pending.store(true, std::memory_order_release);
            return CUDA_ERROR_NOT_READY;
        }
        if (status == CUDA_SUCCESS && reinterpret_cast<FakeEvent*>(event)->launch != 0 &&
            state.scenario.holds_physical && !state.physical_release.load())
            return CUDA_ERROR_NOT_READY;
        return status;
    };
    result.execution.event_record = [](CUevent event, CUstream stream) {
        const auto status = event_record(event, stream);
        if (status == CUDA_SUCCESS && state.scenario.holds_physical && state.scenario.fails_issue &&
            state.fault_observed.load(std::memory_order_acquire) && !state.error_tail_event)
            state.error_tail_event = event;
        return status;
    };
    // Exact CUDA callback ABI, including the mutually exclusive parameter forms.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    result.execution.launch_kernel = [](CUfunction function, unsigned int workers, unsigned int gy,
                                        unsigned int gz, unsigned int bx, unsigned int by,
                                        unsigned int bz, unsigned int shared, CUstream stream,
                                        void** arguments, void** extra) {
        ++state.native_attempts;
        expect(state.callback_returned.load(std::memory_order_acquire),
               "admission never crosses the actual CPU callback dependency");
        expect(extra && arguments == nullptr, "native issue retains its packed parameter owner");
        const auto* bytes = static_cast<const std::byte*>(extra[1]);
        if (state.scenario.loses_reply) {
            state.native_parameters = bytes;
            std::memcpy(state.parameter_snapshot.data(), bytes, state.parameter_snapshot.size());
        }
        CUdeviceptr output{};
        std::uint32_t add{};
        std::memcpy(&output, bytes, sizeof(output));
        std::memcpy(&add, bytes + sizeof(output), sizeof(add));
        const auto expected_add = state.successful_launches % 2 == 0 ? 3U : 7U;
        expect(add == expected_add, "native issue uses the pre-return scalar snapshot");
        const auto* selected = reinterpret_cast<const FakeFunction*>(function);
        expect(workers == (selected->is_private ? 320U : kBlocks),
               "actual original/private selection has explicit geometry");
        if (state.scenario.fails_issue && state.native_attempts == 1)
            driver.next_launch_status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        else
            for (unsigned int index = 0; index != kBlocks; ++index) {
                auto* value =
                    device_bytes(output + index * sizeof(std::uint32_t), sizeof(std::uint32_t));
                std::uint32_t previous{};
                std::memcpy(&previous, value, sizeof(previous));
                previous *= 3;
                std::memcpy(value, &previous, sizeof(previous));
            }
        const auto status =
            launch(function, workers, gy, gz, bx, by, bz, shared, stream, arguments, extra);
        if (status == CUDA_SUCCESS) {
            ++state.successful_launches;
            for (unsigned int index = 0; index != kBlocks; ++index)
                state.expected[index + 1] = state.expected[index + 1] * 3U + expected_add;
        } else
            state.fault_observed.store(true, std::memory_order_release);
        if (state.native_attempts == state.scenario.commands)
            state.all_issued.store(true, std::memory_order_release);
        return status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return result;
}

struct Application {
    CUcontext context{};
    CUmodule module{};
    CUfunction function{};
    CUstream stream{};
    CUdeviceptr memory{};
    void** registration{};
    int host_identity{};
    using DriverLaunch = decltype(&cuLaunchKernel);
    using RuntimeLaunch = cudaError_t(CUDARTAPI*)(const void*, dim3, dim3, void**, std::size_t,
                                                  cudaStream_t);
    DriverLaunch driver_launch{&cuLaunchKernel};
    RuntimeLaunch runtime_launch{&cudaLaunchKernel};

    void open() {
        if (state.scenario.is_runtime) {
            check(cudaSetDevice(0));
            check(cuCtxGetCurrent(&context));
            auto image = glimmer::test::cuda_test_fatbin({}, source());
            __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                        reinterpret_cast<const unsigned long long*>(image.data()),
                                        nullptr};
            registration = __cudaRegisterFatBinary(&wrapper);
            expect(registration != nullptr, "actual Runtime compiler image is registered");
            char name[] = "kernel";
            __cudaRegisterFunction(registration, reinterpret_cast<const char*>(&host_identity),
                                   name, name, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
            __cudaRegisterFatBinaryEnd(registration);
            // Explicit lookup prepares the ordinary binding before holding its
            // stream; asynchronous launch still owns all later issue preparation.
            cudaFuncAttributes attributes{};
            check(cudaFuncGetAttributes(&attributes, &host_identity));
            void* pointer{};
            check(cudaMalloc(&pointer, kOutputBytes));
            memory = reinterpret_cast<CUdeviceptr>(pointer);
        } else {
            check(cuCtxCreate(&context, 0, 0));
            const auto ptx = source();
            check(cuModuleLoadData(&module, ptx.c_str()));
            check(cuModuleGetFunction(&function, module, "kernel"));
            check(cuMemAlloc(&memory, kOutputBytes));
        }
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        if (state.scenario.is_cached) {
            driver_launch = reinterpret_cast<DriverLaunch>(::dlsym(RTLD_DEFAULT, "cuLaunchKernel"));
            runtime_launch =
                reinterpret_cast<RuntimeLaunch>(::dlsym(RTLD_DEFAULT, "cudaLaunchKernel"));
            expect(driver_launch && runtime_launch, "cache both actual public launch symbols");
        }
    }
    CUresult submit(unsigned int index) {
        CUdeviceptr pointer = memory + sizeof(std::uint32_t);
        std::uint32_t value = index % 2 == 0 ? 3U : 7U;
        void* arguments[]{&pointer, &value};
        CUresult result{};
        if (state.scenario.is_runtime) {
            const auto returned = runtime_launch(&host_identity, dim3{kBlocks}, dim3{32}, arguments,
                                                 0, reinterpret_cast<cudaStream_t>(stream));
            result = returned == cudaSuccess ? CUDA_SUCCESS : static_cast<CUresult>(returned);
        } else if (state.scenario.is_packed) {
            std::array<std::byte, 12> packed{};
            std::memcpy(packed.data(), &pointer, sizeof(pointer));
            std::memcpy(packed.data() + sizeof(pointer), &value, sizeof(value));
            std::size_t bytes = packed.size();
            void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                          CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
            result = driver_launch(function, kBlocks, 1, 1, 32, 1, 1, 0, stream, nullptr, extra);
            packed.fill(std::byte{0xff});
        } else {
            result =
                driver_launch(function, kBlocks, 1, 1, 32, 1, 1, 0, stream, arguments, nullptr);
        }
        // These automatic objects cease to exist before the callback is released.
        pointer = 0;
        value = UINT32_MAX;
        arguments[0] = nullptr;
        return result;
    }
    void close() {
        if (state.scenario.is_runtime) {
            // Driver and Runtime share the exact virtual UVA in this fixture.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            check(cudaFree(reinterpret_cast<void*>(memory)));
            check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
            if (registration) __cudaUnregisterFatBinary(registration);
            check(cudaGetLastError());
            check(reset_runtime());
        } else {
            if (memory) check(cuMemFree(memory));
            if (stream) check(cuStreamDestroy(stream));
            if (module) check(cuModuleUnload(module));
            check(cuCtxDestroy(context));
        }
    }
};

void failed_application() {
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    check(app.submit(0));
    state.returned.store(1, std::memory_order_release);
    state.early_returns.store(!state.has_watchdog.load(), std::memory_order_release);
    if (!state.scenario.fails_preparation)
        expect(cudaPeekAtLastError() == cudaSuccess,
               "pending accepted native issue is not a Runtime error");
    state.callback_release.store(true, std::memory_order_release);
    wait_for(state.fault_observed);
    const bool is_history = state.scenario.name == "error-history";
    const bool is_cross_api = state.scenario.name == "error-cross-api";
    if (state.scenario.resets_elsewhere) {
        auto resetter = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(app.context));
            expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
                   "another thread cannot observe or steal the launch-origin error notification");
            check(cudaDeviceReset());
            expect(cudaGetLastError() == cudaSuccess,
                   "foreign reset preserves the old epoch error for its original thread");
        });
        resetter.get();
    } else if (state.scenario.name == "error-reset" || state.scenario.fails_preparation) {
        check(cudaDeviceReset());
    } else {
        const auto until = Clock::now() + std::chrono::seconds{3};
        if (!is_cross_api)
            while (cudaPeekAtLastError() == cudaSuccess)
                expect(Clock::now() < until, "bounded nonblocking first-launch error observation");
        auto other = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(app.context));
            return cudaGetLastError();
        });
        expect(other.get() == cudaSuccess, "another current-context thread does not consume error");
        if (is_history) {
            expect(cudaGetLastError() == cudaErrorLaunchOutOfResources &&
                       cudaGetLastError() == cudaSuccess,
                   "consume the first Runtime notification without erasing its failed prefix");
            check(app.submit(0));
            state.returned.fetch_add(1, std::memory_order_release);
        }
        if (is_cross_api) {
            expect(cuStreamSynchronize(app.stream) == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES &&
                       cuCtxSynchronize() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
                   "Driver synchronization reports the exact Runtime-submitted failure");
            expect(cudaPeekAtLastError() == cudaErrorLaunchOutOfResources &&
                       cudaGetLastError() == cudaErrorLaunchOutOfResources &&
                       cudaGetLastError() == cudaSuccess,
                   "Driver error observation cannot discard the original Runtime notification");
        } else {
            expect(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(app.stream)) ==
                       cudaErrorLaunchOutOfResources,
                   "synchronization retains the first SDK failure across later admission/reaping");
        }
        check(cudaDeviceReset());
    }
    if (is_cross_api)
        expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
               "reset cannot replay an already consumed Runtime notification");
    else
        expect(cudaPeekAtLastError() == cudaErrorLaunchOutOfResources &&
                   cudaPeekAtLastError() == cudaErrorLaunchOutOfResources &&
                   cudaGetLastError() == cudaErrorLaunchOutOfResources &&
                   cudaGetLastError() == cudaSuccess,
               "Peek preserves and Get consumes exactly the original thread's retired-epoch error");
    __cudaUnregisterFatBinary(app.registration);
    check(cudaGetLastError());
    check(cudaSetDevice(0));
    expect(cudaPeekAtLastError() == cudaSuccess,
           "a new Runtime context generation cannot inherit consumed old command failures");
    check(cudaDeviceReset());
}
void lost_reply_application() {
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    expect(app.submit(0) == CUDA_ERROR_UNKNOWN,
           "a genuinely lost admitted reply returns uncertainty, never fabricated success");
    state.returned.store(1, std::memory_order_release);
    const auto later_launch = app.submit(1);
    const auto later_free = cuMemFree(app.memory);
    const auto later_destroy = cuCtxDestroy(app.context);
    std::cout << "Lost-reply follow-up rejection: launch=" << static_cast<int>(later_launch)
              << "; free=" << static_cast<int>(later_free)
              << "; destroy=" << static_cast<int>(later_destroy) << '\n'
              << std::flush;
    expect(later_launch == CUDA_ERROR_INVALID_HANDLE && later_free == CUDA_ERROR_INVALID_VALUE &&
               later_destroy == CUDA_ERROR_UNKNOWN,
           "the uncertain context cannot replay or release its still-owned command");
    state.lost_reply_checked.store(true, std::memory_order_release);
}
void batch_race_application() {
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    check(app.submit(0));
    state.returned.store(1, std::memory_order_release);
    state.early_returns.store(!state.has_watchdog.load(), std::memory_order_release);
    state.holds_batch_reply.store(true, std::memory_order_release);
    state.callback_release.store(true, std::memory_order_release);
    wait_for(state.batch_ready);
    auto synchronizer = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(app.context));
        wait_for(state.batch_reply_held);
        expect(cuCtxSynchronize() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES &&
                   cudaGetLastError() == cudaSuccess,
               "foreign Driver synchronization sees the real error before batch delivery without "
               "stealing the Runtime origin notification");
        state.batch_sync_observed.store(true, std::memory_order_release);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    cudaError_t observed{};
    while ((observed = cudaPeekAtLastError()) == cudaSuccess) {
        expect(Clock::now() < deadline, "bounded eventual originating Runtime error");
        std::this_thread::yield();
    }
    expect(observed == cudaErrorLaunchOutOfResources,
           "the delayed exact batch reply still reports the original Runtime error");
    synchronizer.get();
    expect(cudaGetLastError() == cudaErrorLaunchOutOfResources &&
               cudaGetLastError() == cudaSuccess &&
               cuCtxSynchronize() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
           "origin notification is once-only while active-generation synchronization retains it");
    check(cudaDeviceReset());
    __cudaUnregisterFatBinary(app.registration);
    check(cudaGetLastError());
    check(cudaSetDevice(0));
    check(cudaDeviceReset());
}
void error_tail_application() {
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    check(app.submit(0));
    state.returned.store(1, std::memory_order_release);
    state.early_returns.store(!state.has_watchdog.load(), std::memory_order_release);
    state.callback_release.store(true, std::memory_order_release);
    wait_for(state.fault_observed);
    const auto failure = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    const auto runtime_failure = cudaErrorLaunchOutOfResources;
    const bool keeps_notification = state.scenario.name.ends_with("-reset");
    if (state.scenario.is_runtime) {
        const auto until = Clock::now() + std::chrono::seconds{3};
        cudaError_t observed{};
        while ((observed = cudaPeekAtLastError()) == cudaSuccess)
            expect(Clock::now() < until, "late error query cannot wait for the held physical tail");
        expect(observed == runtime_failure && cudaPeekAtLastError() == runtime_failure,
               "the original Runtime SDK error is visible without consuming its notification");
    }
    auto other = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(app.context));
        expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
               "the other context thread cannot acquire an origin Runtime notification");
        if (state.scenario.uses_legacy_thread)
            expect(legacy_thread_call(false)() == runtime_failure &&
                       cudaGetLastError() == runtime_failure && cudaGetLastError() == cudaSuccess,
                   "legacy synchronization reports this caller's prefix error without stealing "
                   "the Runtime origin notification");
        expect(cuStreamSynchronize(app.stream) == failure && cuCtxSynchronize() == failure,
               "another thread observes the exact Driver prefix error before physical retirement");
    });
    other.get();
    wait_for(state.error_tail_pending);
    expect(!state.physical_release.load() && !state.cleanup_entered.load(),
           "error queries and synchronization did not manufacture physical release");
    state.error_observed_while_pending.store(true, std::memory_order_release);
    if (state.scenario.is_runtime) {
        expect(cudaPeekAtLastError() == runtime_failure,
               "Driver observation preserves the original Runtime error while its tail is held");
        if (!keeps_notification)
            expect(cudaGetLastError() == runtime_failure && cudaGetLastError() == cudaSuccess,
                   "Get consumes the exact pending Runtime notification only once");
    } else {
        expect(cudaGetLastError() == cudaSuccess,
               "Driver-origin failure cannot become a fabricated Runtime notification");
    }
    state.cleanup_entered.store(true, std::memory_order_release);
    if (state.scenario.is_runtime) {
        check(reset_runtime());
        expect(state.physical_release.load(), "reset waits for independently proven physical stop");
        if (keeps_notification)
            expect(cudaPeekAtLastError() == runtime_failure &&
                       cudaPeekAtLastError() == runtime_failure &&
                       cudaGetLastError() == runtime_failure && cudaGetLastError() == cudaSuccess,
                   "reset retains unconsumed origin error through actual physical teardown");
        else
            expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
                   "reset cannot replay an error notification already consumed while pending");
        __cudaUnregisterFatBinary(app.registration);
        check(cudaGetLastError());
        check(cudaSetDevice(0));
        check(reset_runtime());
    } else {
        check(cuMemFree(app.memory));
        app.memory = 0;
        expect(state.physical_release.load(), "free cannot overtake the retained failed SDK tail");
        app.close();
    }
}
void cancellation_application() {
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    for (unsigned int index = 0; index != 2; ++index) {
        check(app.submit(index));
        state.returned.fetch_add(1, std::memory_order_release);
    }
    state.early_returns.store(!state.has_watchdog.load(), std::memory_order_release);
    const auto name = state.scenario.name;
    auto retirement = std::async(std::launch::async, [&] {
        wait_for(state.capacity_busy_reply);
        check(cuCtxSetCurrent(app.context));
        state.cleanup_entered.store(true, std::memory_order_release);
        if (name == "capacity-close")
            check(cuCtxDestroy(app.context));
        else if (name == "capacity-reset")
            check(cudaDeviceReset());
        else if (name == "capacity-unload")
            check(cuModuleUnload(app.module));
        else if (name == "capacity-stream-destroy")
            check(cuStreamDestroy(app.stream));
        else {
            __cudaUnregisterFatBinary(app.registration);
            check(cudaGetLastError());
        }
    });
    state.capacity_call_started.store(true, std::memory_order_release);
    const auto result = app.submit(2);
    state.capacity_returned_early.store(!state.callback_release.load(), std::memory_order_release);
    state.capacity_result.store(result, std::memory_order_release);
    state.capacity_cancel_returned.store(true, std::memory_order_release);
    retirement.get();
    if (state.scenario.is_runtime)
        expect(cudaGetLastError() == static_cast<cudaError_t>(result) &&
                   cudaGetLastError() == cudaSuccess,
               "Runtime cancellation reports the originating failure once across retirement");
    if (name == "capacity-close") return;  // The exact owner already retired all resources.
    if (name == "capacity-reset") {
        __cudaUnregisterFatBinary(app.registration);
        check(cudaGetLastError());
        check(cudaSetDevice(0));
        check(cudaDeviceReset());
        return;
    }
    if (name == "capacity-unload") app.module = nullptr;
    if (name == "capacity-stream-destroy") app.stream = nullptr;
    if (name == "capacity-unregister") app.registration = nullptr;
    app.close();
}
void application() {
    if (state.scenario.cancels_admission) {
        cancellation_application();
        return;
    }
    if (state.scenario.fails_issue && state.scenario.holds_physical) {
        error_tail_application();
        return;
    }
    if (state.scenario.name == "batch-race" || state.scenario.name == "batch-release-race") {
        batch_race_application();
        return;
    }
    if (state.scenario.loses_reply) {
        lost_reply_application();
        return;
    }
    if (state.scenario.fails_issue || state.scenario.fails_preparation) {
        failed_application();
        return;
    }
    Application app;
    app.open();
    check(cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(app.stream), barrier, nullptr));
    wait_for(state.callback_entered);
    const bool keeps_runtime_error =
        state.scenario.name == "runtime-prior-error" || state.scenario.name == "runtime-capacity";
    if (keeps_runtime_error)
        expect(cudaMalloc(nullptr, 1) == cudaErrorInvalidValue,
               "establish a genuine local Runtime error before successful admissions");
    for (unsigned int index = 0; index != 2; ++index) {
        check(app.submit(index));
        state.returned.fetch_add(1, std::memory_order_release);
    }
    state.early_returns.store(!state.has_watchdog.load(), std::memory_order_release);
    if (state.scenario.checks_capacity) {
        state.capacity_call_started.store(true, std::memory_order_release);
        const auto result = app.submit(2);
        state.capacity_returned_early.store(!state.callback_release.load(),
                                            std::memory_order_release);
        state.capacity_result.store(result, std::memory_order_release);
    }
    if (keeps_runtime_error)
        expect(cudaPeekAtLastError() == cudaErrorInvalidValue &&
                   cudaGetLastError() == cudaErrorInvalidValue &&
                   cudaPeekAtLastError() == cudaSuccess,
               "successful admissions preserve an existing thread-local error exactly once");
    const auto name = state.scenario.name;
    if (name == "free" || name == "unload" || name == "stream-destroy" || name == "reset") {
        state.cleanup_entered.store(true, std::memory_order_release);
        // The owner releases the held callback only after observing this request
        // and bounded progress. Cleanup must retain both accepted commands.
        if (name == "free") {
            check(cuMemFree(app.memory));
            app.memory = 0;
        } else if (name == "unload") {
            check(cuModuleUnload(app.module));
            app.module = nullptr;
        } else if (name == "stream-destroy") {
            check(cuStreamDestroy(app.stream));
            app.stream = nullptr;
        } else {
            check(cudaDeviceReset());
            __cudaUnregisterFatBinary(app.registration);
            check(cudaGetLastError());
            // A fresh Runtime generation must not inherit old receipts or handles.
            check(cudaSetDevice(0));
            check(cudaDeviceReset());
            return;
        }
    } else {
        state.callback_release.store(true, std::memory_order_release);
    }
    if (name == "batch-count") {
        wait_for(state.batch_ready);
        state.counts_observations.store(true, std::memory_order_release);
        const LocalErrorQueryScope local_query;
        expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
               "one originating Runtime observation proves both successful exact records");
        state.counts_observations.store(false, std::memory_order_release);
    }
    if (state.scenario.holds_physical) {
        wait_for(state.all_issued);
        expect(cuStreamQuery(app.stream) == CUDA_ERROR_NOT_READY,
               "two actual SDK successes cannot complete their held physical event prefix");
        state.cleanup_entered.store(true, std::memory_order_release);
        check(cuMemFree(app.memory));
        app.memory = 0;
    }
    if (state.scenario.uses_legacy_thread)
        check(legacy_thread_call(false)());
    else
        check(cuCtxSynchronize());
    for (unsigned int index = state.scenario.checks_capacity ? 3U : 2U;
         index != state.scenario.commands; ++index) {
        check(app.submit(index));
        check(cuStreamSynchronize(app.stream));
    }
    app.close();
}

Scenario scenario(std::string_view name) {
    Scenario result{.name = name};
    if (name == "driver" || name == "free" || name == "unload" || name == "stream-destroy")
        return result;
    if (name == "driver-cached")
        result.is_cached = true;
    else if (name == "packed")
        result.is_packed = true;
    else if (name == "prepared")
        result.is_prepared = true;
    else if (name == "runtime" || name == "reset" || name == "runtime-prior-error" ||
             name == "batch-count" || name == "legacy-thread") {
        result.is_runtime = true;
        result.uses_legacy_thread = name == "legacy-thread";
    } else if (name == "runtime-cached")
        result.is_runtime = result.is_cached = true;
    else if (name == "capacity" || name == "frontier-capacity" || name == "byte-capacity") {
        result.checks_capacity = true;
        result.commands = 3;
    } else if (name == "runtime-capacity") {
        result.is_runtime = result.checks_capacity = true;
        result.commands = 3;
    } else if (name == "capacity-close" || name == "capacity-reset" || name == "capacity-unload" ||
               name == "capacity-stream-destroy" || name == "capacity-unregister") {
        result.is_runtime = name == "capacity-reset" || name == "capacity-unregister";
        result.cancels_admission = result.checks_capacity = true;
        result.commands = 3;
    } else if (name == "lost-reply") {
        result.loses_reply = result.holds_physical = true;
        result.commands = 1;
    } else if (name == "reuse")
        result.commands = 24;
    else if (name == "physical")
        result.holds_physical = true;
    else if (name == "error-physical" || name == "error-physical-reset" ||
             name == "driver-error-physical" || name == "legacy-thread-error-physical" ||
             name == "legacy-thread-error-physical-reset") {
        result.is_runtime = name != "driver-error-physical";
        result.uses_legacy_thread = name.starts_with("legacy-thread-");
        result.holds_physical = result.fails_issue = true;
        result.commands = 1;
    } else if (name == "error" || name == "error-reset" || name == "error-other-reset" ||
               name == "preparation-error" || name == "error-history" ||
               name == "error-cross-api" || name == "batch-race" || name == "batch-release-race") {
        result.is_runtime = true;
        result.commands = name == "error-history" ? 2U : 1U;
        result.fails_preparation = name == "preparation-error";
        result.fails_issue = !result.fails_preparation;
        result.is_prepared = result.fails_preparation;
        result.resets_elsewhere = name == "error-other-reset";
    } else
        throw std::runtime_error("unknown async launch scenario");
    return result;
}
void actual_owner(const std::string& path, const Scenario& selected) {
    state.scenario = selected;
    driver = {};
    driver.has_profile_function_metadata = true;
    driver.next_claimed = kBlocks;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 8;
    config.session_limits = limits(1000);
    // The independently held host callback occupies one command alongside the
    // two accepted kernels. The third must wait for real capacity recovery.
    config.session_limits.submissions.max_commands = 3;
    config.session_limits.max_objects = 8;
    if (selected.name == "frontier-capacity") {
        config.session_limits.submissions.max_commands = 4;
        config.session_limits.max_objects = 3;
    }
    if (selected.name == "byte-capacity") {
        config.session_limits.submissions.max_commands = 4;
        config.session_limits.submissions.max_parameter_bytes = 24;
    }
    config.persistent.workers = selected.is_prepared ? 320 : 0;
    ExecutionService service(std::move(config), dispatch());
    expect(service.start(), "start actual asynchronous admission service");
    const auto owner = current_process_identity();
    expect(owner.has_value(), "actual authenticated owner process identity");
    const auto pid = std::to_string(owner->pid), start = std::to_string(owner->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind public frontend exclusively to this actual service");
    auto client = std::async(std::launch::async, application);
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    std::optional<Clock::time_point> callback_time, cleanup_time, capacity_time;
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        if (state.batch_sync_observed.load(std::memory_order_acquire) &&
            !state.batch_reply_delivered) {
            const auto remaining = state.delayed_batch_reply.size() - state.delayed_batch_offset;
            expect(state.delayed_batch_descriptor >= 0 && remaining && Clock::now() < deadline,
                   "deliver only the exact real batch frame after independent synchronization");
            const auto sent =
                __real_send(state.delayed_batch_descriptor,
                            state.delayed_batch_reply.data() + state.delayed_batch_offset,
                            remaining, MSG_NOSIGNAL | MSG_DONTWAIT);
            if (sent > 0)
                state.delayed_batch_offset += static_cast<std::size_t>(sent);
            else
                expect(sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR),
                       "held real response survives only bounded nonblocking transport retries");
            state.batch_reply_delivered =
                state.delayed_batch_offset == state.delayed_batch_reply.size();
            if (!state.batch_reply_delivered) continue;  // Never interleave a partial frame.
        }
        expect(service.step(1) && Clock::now() < deadline,
               "bounded real frontend/service progress");
        const auto now = Clock::now();
        if ((selected.name == "batch-count" || selected.name == "batch-race" ||
             selected.name == "batch-release-race") &&
            state.all_issued.load(std::memory_order_acquire) &&
            std::ranges::none_of(driver.events, [](const auto& event) { return event.is_live; }))
            state.batch_ready.store(true, std::memory_order_release);
        if (selected.loses_reply && state.held_receipt.load(std::memory_order_acquire))
            state.callback_release.store(true, std::memory_order_release);
        if (state.callback_entered.load() && !callback_time) callback_time = now;
        if (state.cleanup_entered.load() && !cleanup_time) cleanup_time = now;
        if (cleanup_time && !selected.cancels_admission &&
            now - *cleanup_time >= std::chrono::milliseconds{20}) {
            state.callback_release.store(true, std::memory_order_release);
            state.physical_release.store(true, std::memory_order_release);
        }
        if (callback_time && !state.callback_release.load() &&
            now - *callback_time >= std::chrono::milliseconds{500}) {
            state.has_watchdog.store(true, std::memory_order_release);
            state.callback_release.store(true, std::memory_order_release);
        }
        if (state.capacity_call_started.load(std::memory_order_acquire) && !capacity_time)
            capacity_time = now;
        if (capacity_time && !selected.cancels_admission &&
            now - *capacity_time >= std::chrono::milliseconds{20})
            state.callback_release.store(true, std::memory_order_release);
        if (selected.cancels_admission && state.capacity_cancel_returned.load())
            state.callback_release.store(true, std::memory_order_release);
    }
    client.get();
    if (selected.loses_reply) {
        expect(state.lost_reply_checked.load() && state.held_receipt.load() &&
                   !state.has_watchdog.load() && !state.invalid_loss_frame.load() &&
                   state.held_sends != 0 && state.lost_sends == 1 && state.loss_sequence != 0 &&
                   state.loss_frontier != 0 && state.native_attempts == 1 &&
                   state.successful_launches == 1 && state.all_issued.load() &&
                   !state.physical_release.load() && driver.context_creates == 1 &&
                   driver.context_destroys == 0 && driver.release_calls == 0 &&
                   driver.live_resources() != 0 && glimmer_execution_lookup_count(2) == 0 &&
                   state.native_parameters &&
                   std::memcmp(state.native_parameters, state.parameter_snapshot.data(),
                               state.parameter_snapshot.size()) == 0,
               "lost actual admission retains physical resources and copied parameters without "
               "replay");
        for (const auto& allocation : driver.allocations)
            if (allocation.is_live) verify_output(allocation);
        expect(state.numerical_allocations == 1 && state.numerical_values == kBlocks + 2,
               "the single actual issue preserves every non-idempotent value and both guards");
        std::cout << "Actual admission reply lost: sdk_issues=1; checked_values="
                  << state.numerical_values << "; physical_tail_pending=true; replay=false; "
                  << "parameters_retained=true; healthy_cleanup=false\n"
                  << std::flush;
        // Intentional uncertainty isolation: do not call request_stop/destruct
        // the retained service or misreport process exit as healthy cleanup.
        _exit(EXIT_SUCCESS);
    }
    expect(service.request_stop(), "normal native owner and socket shutdown");
    const auto commands = selected.checks_capacity && state.capacity_result.load() != CUDA_SUCCESS
                              ? 2U
                              : selected.commands;
    const auto successful = selected.fails_issue         ? commands - 1U
                            : selected.fails_preparation ? 0U
                                                         : commands;
    const auto attempts = selected.fails_preparation ? 0U : commands;
    expect(driver.context_creates == driver.context_destroys && driver.live_resources() == 0 &&
               std::ranges::none_of(state.hosts, [](const auto& host) { return host.is_live; }) &&
               state.numerical_allocations == 1 && state.numerical_values == kBlocks + 2 &&
               state.successful_launches == successful && state.native_attempts == attempts &&
               state.preparation_failures == (selected.fails_preparation ? 1U : 0U) &&
               state.cleanup_before_callback == 0 && state.cleanup_before_physical == 0 &&
               glimmer_execution_lookup_count(2) == 0,
           "all values, guards, attempts and resources retire normally without native fallback");
    if (selected.cancels_admission) {
        const auto expected = selected.name == "capacity-close" || selected.name == "capacity-reset"
                                  ? CUDA_ERROR_CONTEXT_IS_DESTROYED
                                  : CUDA_ERROR_INVALID_HANDLE;
        expect(state.capacity_busy_reply.load() && state.capacity_result.load() == expected &&
                   state.capacity_returned_early.load() && state.native_attempts == 2 &&
                   !state.has_watchdog.load(),
               "after normal cleanup, retirement cancels only the unaccepted busy launch before "
               "callback recovery");
    } else if (selected.checks_capacity)
        expect(state.capacity_result.load() == CUDA_SUCCESS &&
                   !state.capacity_returned_early.load() && state.native_attempts == 3 &&
                   !state.has_watchdog.load(),
               "after exact cleanup, temporary capacity waits and issues the third command once");
    std::cout << "Async launch normal cleanup: case=" << selected.name << "; commands=" << commands
              << "; checked_values=" << state.numerical_values
              << "; contexts=" << driver.context_destroys << "; replay=false\n"
              << std::flush;
    // The old DSO uses its unchanged empty-tail contract. Its bounded release
    // above permits every command and all normal cleanup before this true RED.
    expect(state.early_returns.load() && !state.has_watchdog.load(),
           "same application thread must finish admission before native issue is permitted");
    if (selected.uses_legacy_thread)
        expect(state.legacy_routes_owned.load(std::memory_order_acquire),
               "legacy thread sync/reset must select owned public wrappers after normal cleanup");
    if (selected.fails_issue && selected.holds_physical)
        expect(state.error_tail_event && state.error_tail_pending.load() &&
                   state.error_observed_while_pending.load() && state.physical_release.load(),
               "combined SDK error, pending exact event, early notification and proven cleanup "
               "were all independently exercised");
    if (selected.name == "batch-count") {
        std::cout << "Actual async observation RPCs after normal cleanup: batch_queries="
                  << state.batch_queries << "; batch_releases=" << state.batch_releases
                  << "; single_queries=" << state.single_queries
                  << "; single_releases=" << state.single_releases
                  << "; queried_items=" << state.queried_items
                  << "; released_items=" << state.released_items << '\n'
                  << std::flush;
        expect(
            !state.invalid_batch_reply && state.local_error_sends.load() == 0,
            "origin Get/Peek sends no request while independent background replies remain valid");
    } else if (selected.name == "batch-race" || selected.name == "batch-release-race") {
        expect(state.batch_reply_held.load() && state.batch_sync_observed.load() &&
                   state.batch_reply_delivered && !state.delayed_batch_reply.empty(),
               "actual service error remains visible across a held batch response and normal "
               "cleanup");
        if (selected.name == "batch-release-race")
            expect(state.batch_observed_key.id && state.batch_observed_key.sequence &&
                       state.batch_sync_remote_queries == 1 &&
                       state.batch_sync_remote_status == CUDA_SUCCESS,
                   "after actual service deletion only the already cached frontend history "
                   "supplies the original error during the held Release acknowledgement");
    }
}
int run(std::string_view name, const char* parent_directory = nullptr) {
    char pattern[] = "/tmp/glimmer-async-launch-XXXXXX";
    const auto* directory = parent_directory ? parent_directory : ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    std::jthread watchdog([](const std::stop_token& stop) {
        const auto deadline = Clock::now() + std::chrono::seconds{12};
        while (!stop.stop_requested()) {
            if (Clock::now() >= deadline) _exit(98);
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    });
    int status = EXIT_SUCCESS;
    try {
        actual_owner(std::string(directory) + "/service.sock", scenario(name));
    } catch (const std::exception& error) {
        std::cerr << "Async launch case " << name << ": " << error.what() << '\n';
        status = EXIT_FAILURE;
    }
    watchdog.request_stop();
    if (::rmdir(directory) != 0) status = EXIT_FAILURE;
    return status;
}
int isolated_lost_reply(const char* executable) {
    char pattern[] = "/tmp/glimmer-async-loss-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    const auto child = ::fork();
    if (child < 0) {
        (void)::rmdir(directory);
        return EXIT_FAILURE;
    }
    if (child == 0) {
        ::execl("/proc/self/exe", executable, "lost-reply", directory, static_cast<char*>(nullptr));
        _exit(99);
    }
    int status{};
    const bool passed = ::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                        WEXITSTATUS(status) == EXIT_SUCCESS;
    // This exact test-owned filesystem path is reclaimed only after its
    // deliberately retained child has exited; no live owner is signalled.
    const auto socket = std::string(directory) + "/service.sock";
    const bool removed = ::unlink(socket.c_str()) == 0 || errno == ENOENT;
    return passed && removed && ::rmdir(directory) == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
void count_observation_reply(const void* pointer, std::size_t bytes) {
    const auto frame = std::span{static_cast<const std::byte*>(pointer), bytes};
    if (frame.size() < kExecutionHeaderBytes) return;
    const auto header = decode_execution_header(frame.first(kExecutionHeaderBytes));
    if (!header || header->kind != ExecutionMessageKind::kReply) return;
    if (header->operation == ExecutionOpcode::kQueryFrontier)
        ++state.single_queries;
    else if (header->operation == ExecutionOpcode::kReleaseFrontier)
        ++state.single_releases;
    else if (header->operation == ExecutionOpcode::kQueryAsyncFrontiers ||
             header->operation == ExecutionOpcode::kReleaseAsyncFrontiers) {
        const auto reply = frame.size() == kExecutionHeaderBytes + header->payload_bytes
                               ? decode_execution_reply(frame.subspan(kExecutionHeaderBytes))
                               : std::nullopt;
        if (!reply || reply->disposition != ExecutionWireDisposition::kComplete || reply->status ||
            reply->original_status || reply->id || reply->address || reply->parameter_bytes ||
            !reply->arguments.empty()) {
            state.invalid_batch_reply = true;
            return;
        }
        if (header->operation == ExecutionOpcode::kQueryAsyncFrontiers) {
            ++state.batch_queries;
            const auto observed = decode_execution_async_frontier_observations(reply->data);
            if (!observed || !std::ranges::all_of(*observed, [](const auto& item) {
                    return item.status == CUDA_SUCCESS ||
                           (item.status == CUDA_ERROR_NOT_READY && !item.is_retired);
                }))
                state.invalid_batch_reply = true;
            else
                state.queried_items += static_cast<unsigned int>(observed->size());
        } else {
            ++state.batch_releases;
            const auto released = decode_execution_async_frontier_batch(reply->data);
            if (!released)
                state.invalid_batch_reply = true;
            else
                state.released_items += static_cast<unsigned int>(released->size());
        }
    }
}
ssize_t send_with_loss(int descriptor, const void* pointer, std::size_t bytes, int flags) {
    if (state.scenario.cancels_admission) {
        const auto sent = __real_send(descriptor, pointer, bytes, flags);
        const auto frame = std::span{static_cast<const std::byte*>(pointer), bytes};
        const auto header = frame.size() >= kExecutionHeaderBytes
                                ? decode_execution_header(frame.first(kExecutionHeaderBytes))
                                : std::nullopt;
        if (sent == static_cast<ssize_t>(bytes) && header &&
            header->kind == ExecutionMessageKind::kReply &&
            header->operation == ExecutionOpcode::kLaunch &&
            frame.size() == kExecutionHeaderBytes + header->payload_bytes) {
            const auto reply = decode_execution_reply(frame.subspan(kExecutionHeaderBytes));
            if (reply && reply->disposition == ExecutionWireDisposition::kAdmissionBusy)
                state.capacity_busy_reply.store(true, std::memory_order_release);
        }
        return sent;  // Observe only the real original send; never fabricate a receipt.
    }
    if (!state.scenario.loses_reply) {
        std::optional<std::int32_t> sync_remote_status;
        if (state.holds_batch_reply.load(std::memory_order_acquire)) {
            const auto frame = std::span{static_cast<const std::byte*>(pointer), bytes};
            const auto header = frame.size() >= kExecutionHeaderBytes
                                    ? decode_execution_header(frame.first(kExecutionHeaderBytes))
                                    : std::nullopt;
            const bool is_held = state.batch_reply_held.load(std::memory_order_acquire);
            if (header && header->kind == ExecutionMessageKind::kReply && is_held &&
                !state.batch_sync_observed.load(std::memory_order_acquire) &&
                header->operation == ExecutionOpcode::kQueryFrontier) {
                const auto reply =
                    frame.size() == kExecutionHeaderBytes + header->payload_bytes
                        ? decode_execution_reply(frame.subspan(kExecutionHeaderBytes))
                        : std::nullopt;
                expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                           !reply->original_status && !reply->id && !reply->address &&
                           reply->parameter_bytes == 1 && reply->arguments.empty() &&
                           reply->data.empty(),
                       "inspect only the real retired foreign-thread synchronization response");
                sync_remote_status = reply->status;
            }
            if (header && header->kind == ExecutionMessageKind::kReply && !is_held &&
                (header->operation == ExecutionOpcode::kQueryAsyncFrontiers ||
                 header->operation == ExecutionOpcode::kReleaseAsyncFrontiers)) {
                const auto reply =
                    frame.size() == kExecutionHeaderBytes + header->payload_bytes
                        ? decode_execution_reply(frame.subspan(kExecutionHeaderBytes))
                        : std::nullopt;
                expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                           !reply->status && !reply->original_status && !reply->id &&
                           !reply->address && !reply->parameter_bytes && reply->arguments.empty(),
                       "delay only an exact real complete batch envelope");
                if (header->operation == ExecutionOpcode::kQueryAsyncFrontiers) {
                    const auto observations =
                        decode_execution_async_frontier_observations(reply->data);
                    expect(observations && observations->size() == 1,
                           "inspect the exact one-command observation carrier");
                    if (observations->front().status != CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ||
                        !observations->front().is_retired)
                        return __real_send(descriptor, pointer, bytes, flags);
                    state.batch_observed_key = observations->front().key;
                } else {
                    const auto keys = decode_execution_async_frontier_batch(reply->data);
                    expect(keys && keys->size() == 1 && state.batch_observed_key.id &&
                               keys->front() == state.batch_observed_key,
                           "real Release acknowledgement echoes the preceding exact error owner");
                }
                const auto held_operation = state.scenario.name == "batch-release-race"
                                                ? ExecutionOpcode::kReleaseAsyncFrontiers
                                                : ExecutionOpcode::kQueryAsyncFrontiers;
                if (header->operation == held_operation) {
                    state.delayed_batch_reply.assign(frame.begin(), frame.end());
                    state.delayed_batch_descriptor = descriptor;
                    state.batch_reply_held.store(true, std::memory_order_release);
                    // Test-only complete-frame delivery reordering. Query keeps
                    // service metadata; Release has already deleted it. Later
                    // sync replies pass while the original caller still waits.
                    return static_cast<ssize_t>(bytes);
                }
            }
        }
        const bool counts = state.counts_observations.load(std::memory_order_acquire);
        const auto sent = __real_send(descriptor, pointer, bytes, flags);
        if (sent > 0 && sync_remote_status) {
            ++state.batch_sync_remote_queries;
            state.batch_sync_remote_status = *sync_remote_status;
        }
        if (sent > 0 && counts) count_observation_reply(pointer, bytes);
        return sent;
    }
    try {
        const auto frame = std::span{static_cast<const std::byte*>(pointer), bytes};
        if (frame.size() < kExecutionHeaderBytes)
            return __real_send(descriptor, pointer, bytes, flags);
        const auto header = decode_execution_header(frame.first(kExecutionHeaderBytes));
        if (!header || header->kind != ExecutionMessageKind::kReply ||
            header->operation != ExecutionOpcode::kLaunch)
            return __real_send(descriptor, pointer, bytes, flags);
        const auto reply = frame.size() == kExecutionHeaderBytes + header->payload_bytes
                               ? decode_execution_reply(frame.subspan(kExecutionHeaderBytes))
                               : std::nullopt;
        if (!reply || reply->disposition != ExecutionWireDisposition::kAdmissionAccepted ||
            reply->status || !reply->id || !reply->address || reply->original_status ||
            reply->parameter_bytes || !reply->arguments.empty() || !reply->data.empty()) {
            state.invalid_loss_frame.store(true);
            errno = EIO;
            return -1;
        }
        if (!state.held_receipt.load()) {
            state.loss_sequence = reply->id;
            state.loss_frontier = reply->address;
            state.held_receipt.store(true, std::memory_order_release);
        } else if (state.loss_sequence != reply->id || state.loss_frontier != reply->address) {
            state.invalid_loss_frame.store(true);
        }
        if (!state.all_issued.load(std::memory_order_acquire)) {
            ++state.held_sends;
            errno = EAGAIN;
        } else {
            ++state.lost_sends;
            errno = EPIPE;
        }
        return -1;
    } catch (...) {
        state.invalid_loss_frame.store(true);
        errno = EIO;
        return -1;
    }
}
}  // namespace
extern "C" ssize_t send(int descriptor, const void* pointer, std::size_t bytes, int flags) {
    if (observes_local_error_query) state.local_error_sends.fetch_add(1, std::memory_order_relaxed);
    return static_cast<ssize_t>(
        ::syscall(SYS_sendto, descriptor, pointer, bytes, flags, nullptr, 0));
}
// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" ssize_t __wrap_send(int descriptor, const void* pointer, std::size_t bytes, int flags) {
    return send_with_loss(descriptor, pointer, bytes, flags);
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "lost-reply") return run(argv[1], argv[2]);
    if (argc == 2 && std::string_view(argv[1]) == "lost-reply") return isolated_lost_reply(argv[0]);
    if (argc == 2) return run(argv[1]);
    if (argc != 1) return EXIT_FAILURE;
    for (const auto* mode : {"driver",
                             "driver-cached",
                             "packed",
                             "prepared",
                             "runtime",
                             "runtime-cached",
                             "capacity",
                             "frontier-capacity",
                             "byte-capacity",
                             "runtime-capacity",
                             "capacity-close",
                             "capacity-reset",
                             "capacity-unload",
                             "capacity-stream-destroy",
                             "capacity-unregister",
                             "reuse",
                             "free",
                             "unload",
                             "stream-destroy",
                             "reset",
                             "error",
                             "error-reset",
                             "error-other-reset",
                             "preparation-error",
                             "physical",
                             "runtime-prior-error",
                             "error-history",
                             "error-cross-api",
                             "error-physical",
                             "error-physical-reset",
                             "driver-error-physical",
                             "legacy-thread",
                             "legacy-thread-error-physical",
                             "legacy-thread-error-physical-reset",
                             "batch-count",
                             "batch-race",
                             "batch-release-race"}) {
        const auto child = ::fork();
        if (child < 0) return EXIT_FAILURE;
        if (child == 0) {
            ::execl("/proc/self/exe", argv[0], mode, static_cast<char*>(nullptr));
            _exit(99);
        }
        int status{};
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return EXIT_FAILURE;
    }
    if (isolated_lost_reply(argv[0]) != EXIT_SUCCESS) return EXIT_FAILURE;
    std::cout << "Actual frontend/service async launch suite passed: healthy_cases=37; "
                 "isolated_lost_reply_cases=1\n";
    return EXIT_SUCCESS;
}
