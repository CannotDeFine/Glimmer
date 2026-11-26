#include "runtime.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::backend::cuda;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status) {
    expect(status == CUDA_SUCCESS, "owned Driver event call succeeds");
}
void check(cudaError_t status) {
    expect(status == cudaSuccess, "owned Runtime event call succeeds");
}
struct Event {
    unsigned int flags{};
    std::uint64_t serial{};
    CUstream stream{};
    CUcontext context{};
    bool is_live{}, is_recorded{}, is_ready{};
};
struct Native {
    static thread_local CUcontext current;
    std::array<Event, 64> events{};
    std::array<char, 4> streams{};
    std::array<char, 16> contexts{};
    std::uint64_t next_event{1};
    unsigned int stream_count{}, context_count{}, event_destroys{}, context_destroys{},
        synchronizes{}, elapsed{};
    std::vector<std::uint64_t> record_generations, waited_generations;
    std::vector<unsigned int> record_flags;
    std::atomic<unsigned int> phase{}, waits{};
    bool should_fail_event_destroy{};
    unsigned int event_destroy_attempts{};
} native;
thread_local CUcontext Native::current{};
Event& event(CUevent value) {
    auto found = std::ranges::find_if(native.events, [value](const auto& item) {
        return reinterpret_cast<CUevent>(const_cast<Event*>(&item)) == value && item.is_live;
    });
    expect(found != native.events.end(), "only live original event handles reach injected SDK");
    return *found;
}
CUresult CUDAAPI event_create(CUevent* output, unsigned int flags) {
    if (flags & ~3U) return CUDA_ERROR_INVALID_VALUE;
    auto found =
        std::ranges::find_if(native.events, [](const auto& item) { return !item.is_live; });
    if (found == native.events.end()) return CUDA_ERROR_OUT_OF_MEMORY;
    *found = {
        .flags = flags, .serial = native.next_event++, .context = native.current, .is_live = true};
    *output = reinterpret_cast<CUevent>(&*found);
    return CUDA_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native CUDA ABI.
CUresult CUDAAPI event_record_flags(CUevent value, CUstream stream, unsigned int flags) {
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    auto& selected = event(value);
    expect(selected.context == native.current, "original record uses its actual context owner");
    selected.is_recorded = true;
    selected.is_ready = false;
    selected.stream = stream;
    if (!(selected.flags & CU_EVENT_DISABLE_TIMING)) {
        native.record_generations.push_back(selected.serial);
        native.record_flags.push_back(flags);
    }
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult CUDAAPI event_record(CUevent value, CUstream stream) {
    return event_record_flags(value, stream, 0);
}
CUresult CUDAAPI event_query(CUevent value) {
    const auto& selected = event(value);
    expect(selected.context == native.current,
           "original query never borrows another current context");
    return !selected.is_recorded || selected.is_ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    ++native.event_destroy_attempts;
    expect(event(value).context == native.current,
           "original destroy runs only on its owning context");
    if (native.should_fail_event_destroy) {
        native.should_fail_event_destroy = false;
        return CUDA_ERROR_UNKNOWN;
    }
    event(value).is_live = false;
    ++native.event_destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_sync(CUevent value) {
    ++native.synchronizes;
    expect(event_query(value) == CUDA_SUCCESS, "native event sync never blocks the owner loop");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI elapsed(float* output, CUevent start, CUevent end) {
    ++native.elapsed;
    if (!event(start).is_recorded || !event(end).is_recorded ||
        (event(start).flags & CU_EVENT_DISABLE_TIMING) ||
        (event(end).flags & CU_EVENT_DISABLE_TIMING))
        return CUDA_ERROR_INVALID_HANDLE;
    *output = 42.125F;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI wait_event(CUstream, CUevent value, unsigned int flags) {
    expect(flags == 0, "original wait flags are unchanged");
    native.waited_generations.push_back(event(value).serial);
    ++native.waits;
    return CUDA_SUCCESS;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch value;
    value.execution.context_get_current = [](CUcontext* output) {
        *output = native.current;
        return CUDA_SUCCESS;
    };
    value.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int, unsigned int,
                                       CUstream, void**,
                                       void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.execution.event_create = event_create;
    value.execution.event_record = event_record;
    value.execution.event_query = event_query;
    value.execution.event_destroy = event_destroy;
    value.execution.stream_synchronize = [](CUstream) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    value.device_get = [](CUdevice* output, int) {
        *output = 0;
        return CUDA_SUCCESS;
    };
    value.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(native.context_count < native.contexts.size(), "bounded distinct original contexts");
        native.current = reinterpret_cast<CUcontext>(&native.contexts[native.context_count++]);
        *output = native.current;
        return CUDA_SUCCESS;
    };
    value.context_set_current = [](CUcontext selected) {
        native.current = selected;
        return CUDA_SUCCESS;
    };
    value.context_destroy = [](CUcontext selected) {
        native.current = nullptr;
        for (auto& item : native.events)
            if (item.context == selected) item.is_live = false;
        ++native.context_destroys;
        return CUDA_SUCCESS;
    };
    value.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.function_get = [](CUfunction*, CUmodule, const char*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.allocate = [](CUdeviceptr*, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.release = [](CUdeviceptr) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.upload = [](CUdeviceptr, const void*, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.download = [](void*, CUdeviceptr, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.stream_create = [](CUstream* output, unsigned int flags) {
        expect(flags == CU_STREAM_NON_BLOCKING && native.stream_count < native.streams.size(),
               "owned nonblocking stream geometry");
        *output = reinterpret_cast<CUstream>(&native.streams[native.stream_count++]);
        return CUDA_SUCCESS;
    };
    value.stream_destroy = [](CUstream) { return CUDA_SUCCESS; };
    value.host_register = [](void*, std::size_t, unsigned int) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.host_unregister = [](void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.host_map = [](CUdeviceptr*, void*, unsigned int) { return CUDA_ERROR_NOT_SUPPORTED; };
    value.event_record_flags = event_record_flags;
    value.stream_wait_event = wait_event;
    value.event_elapsed_time = elapsed;
    value.event_synchronize = event_sync;
    return value;
}
void application() {
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    CUstream first{}, second{};
    check(cuStreamCreate(&first, CU_STREAM_NON_BLOCKING));
    check(cuStreamCreate(&second, CU_STREAM_NON_BLOCKING));
    CUevent selected{}, untimed{};
    auto unchanged = reinterpret_cast<CUevent>(1);
    expect(cuEventCreate(&unchanged, 128) == CUDA_ERROR_INVALID_VALUE &&
               unchanged == reinterpret_cast<CUevent>(1),
           "original event-create errors preserve the caller output");
    check(cuEventCreate(&selected, 0));
    check(cuEventCreate(&untimed, CU_EVENT_DISABLE_TIMING));
    check(cuEventQuery(selected));
    check(cudaEventQuery(reinterpret_cast<cudaEvent_t>(untimed)));
    float milliseconds = -99;
    expect(cuEventElapsedTime(&milliseconds, selected, selected) == CUDA_ERROR_INVALID_HANDLE &&
               milliseconds == -99,
           "original unrecorded elapsed-time error leaves output unchanged");
    check(cuEventRecord(selected, nullptr));
    expect(cuEventQuery(selected) == CUDA_ERROR_NOT_READY, "logical record stays pending");
    check(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(first),
                              reinterpret_cast<cudaEvent_t>(selected), 0));
    check(cudaEventRecord(reinterpret_cast<cudaEvent_t>(selected),
                          reinterpret_cast<cudaStream_t>(second)));
    native.phase = 1;
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!native.waits.load()) {
        expect(Clock::now() < deadline, "old-generation wait progresses before rerecorded event");
        std::this_thread::sleep_for(std::chrono::microseconds{100});
    }
    expect(cuEventQuery(selected) == CUDA_ERROR_NOT_READY,
           "rerecorded generation remains pending after old-generation wait issues");
    for (;;) {
        const auto status = cuStreamQuery(first);
        if (status == CUDA_SUCCESS) break;
        expect(status == CUDA_ERROR_NOT_READY && Clock::now() < deadline,
               "old-generation wait and its stream settle independently");
    }
    check(cuEventDestroy(selected));
    expect(
        cudaEventQuery(reinterpret_cast<cudaEvent_t>(selected)) == cudaErrorInvalidResourceHandle &&
            cudaGetLastError() == cudaErrorInvalidResourceHandle,
        "destroy invalidates the public token without waiting for pending native work");
    CUevent begin{}, end{};
    check(cuEventCreate(&begin, 0));
    check(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&end)));
    check(cuEventRecord(begin, first));
    expect(
        cuEventRecordWithFlags(end, first, CU_EVENT_RECORD_EXTERNAL) == CUDA_ERROR_INVALID_VALUE &&
            cudaEventRecordWithFlags(reinterpret_cast<cudaEvent_t>(end),
                                     reinterpret_cast<cudaStream_t>(first),
                                     CU_EVENT_RECORD_EXTERNAL) == cudaErrorInvalidValue &&
            cudaGetLastError() == cudaErrorInvalidValue,
        "external record flags reject outside capture without replacing the old generation");
    check(cuEventQuery(end));
    check(cudaEventRecordWithFlags(reinterpret_cast<cudaEvent_t>(end),
                                   reinterpret_cast<cudaStream_t>(first), 0));
    check(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(end)));
    check(cuEventElapsedTime(&milliseconds, begin, end));
    expect(milliseconds == 42.125F,
           "elapsed time is the exact original SDK float, not host timing");
    check(cudaEventElapsedTime(&milliseconds, reinterpret_cast<cudaEvent_t>(begin),
                               reinterpret_cast<cudaEvent_t>(end)));
    expect(milliseconds == 42.125F, "Runtime and Driver share the same owned event timestamps");
    milliseconds = -99;
    expect(cudaEventElapsedTime(&milliseconds, reinterpret_cast<cudaEvent_t>(untimed),
                                reinterpret_cast<cudaEvent_t>(end)) ==
                   cudaErrorInvalidResourceHandle &&
               milliseconds == -99 && cudaGetLastError() == cudaErrorInvalidResourceHandle,
           "disabled timing preserves original error and output bytes");
    expect(::dlsym(RTLD_DEFAULT, "cuEventRecord") == reinterpret_cast<void*>(&cuEventRecord) &&
               ::dlsym(RTLD_DEFAULT, "cudaEventQuery") == reinterpret_cast<void*>(&cudaEventQuery),
           "global event lookups cannot escape to caller-native resource paths");
    check(cuCtxSetCurrent(nullptr));
    expect(cuEventQuery(begin) == CUDA_ERROR_INVALID_HANDLE, "event owner context is required");
    check(cuCtxSetCurrent(context));
    const auto child = ::fork();
    expect(child >= 0, "fork event ownership guard");
    if (!child) _exit(cuEventQuery(begin) == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "inherited event access rejects before touching native state");
    check(cuEventDestroy(begin));
    check(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(end)));
    check(cuEventDestroy(untimed));
    native.phase = 2;
    check(cuCtxSynchronize());
    native.phase = 3;
    CUevent cross_event{};
    check(cuEventCreate(&cross_event, 0));
    check(cuEventRecord(cross_event, nullptr));
    expect(cuEventQuery(cross_event) == CUDA_ERROR_NOT_READY, "cross-context producer is pending");
    CUcontext target_context{};
    CUstream target_stream{};
    check(cuCtxCreate(&target_context, 0, 0));
    check(cuStreamCreate(&target_stream, CU_STREAM_NON_BLOCKING));
    const auto previous_waits = native.waits.load();
    check(cuStreamWaitEvent(target_stream, cross_event, 0));
    expect(cuStreamQuery(target_stream) == CUDA_ERROR_NOT_READY,
           "foreign wait cannot satisfy an incomplete source generation");
    check(cuCtxSetCurrent(context));
    check(cuEventDestroy(cross_event));
    check(cuCtxSetCurrent(target_context));
    native.phase = 4;
    const auto cross_deadline = Clock::now() + std::chrono::seconds{5};
    while (native.waits.load() == previous_waits) {
        expect(Clock::now() < cross_deadline,
               "foreign wait progresses after source logical completion");
        std::this_thread::sleep_for(std::chrono::microseconds{100});
    }
    expect(cuStreamQuery(target_stream) == CUDA_ERROR_NOT_READY,
           "native wait issue is not its target-stream physical retirement");
    native.phase = 5;
    check(cuStreamSynchronize(target_stream));
    check(cuCtxDestroy(target_context));
    check(cuCtxSetCurrent(context));
    std::array<CUevent, 32> bounded{};
    std::size_t created{};
    for (; created < bounded.size(); ++created) {
        auto output = reinterpret_cast<CUevent>(1);
        const auto result = cuEventCreate(&output, 0);
        if (result == CUDA_ERROR_OUT_OF_MEMORY) {
            expect(output == reinterpret_cast<CUevent>(1),
                   "capacity rejection leaves output untouched");
            break;
        }
        check(result);
        bounded[created] = output;
    }
    expect(created == 16, "event objects and native generations obey the configured owner bound");
    for (std::size_t index = 0; index < created; ++index) check(cuEventDestroy(bounded[index]));
    check(cuEventCreate(&selected, 0));
    check(cuEventDestroy(selected));
    check(cuCtxDestroy(context));
}
void run(const std::string& path) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual event service identity");
    const auto pid = std::to_string(identity->pid),
               ticks = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", ticks.c_str(), 1) == 0,
           "authenticate before frontend initialization");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    config.session_limits.max_objects = 16;
    ExecutionService service(std::move(config), dispatch());
    expect(service.start(), "start real event service/backend composition");
    auto client = std::async(std::launch::async, application);
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        const auto phase = native.phase.load();
        if (phase == 1 || phase == 2 || phase == 4 || phase == 5)
            for (auto& item : native.events)
                if (item.is_live &&
                    (phase == 2 || phase == 5 ||
                     (phase == 4 &&
                      item.context == reinterpret_cast<CUcontext>(&native.contexts[0])) ||
                     (phase == 1 && item.stream != reinterpret_cast<CUstream>(&native.streams[1]))))
                    item.is_ready = true;
        expect(service.step(1) && Clock::now() < deadline, "bounded event owner progress");
    }
    client.get();
    expect(native.record_generations.size() == 5 && native.waited_generations.size() == 2 &&
               native.waited_generations[0] == native.record_generations[0] &&
               native.record_generations[0] != native.record_generations[1] &&
               native.waited_generations[1] == native.record_generations[4] &&
               native.record_flags[3] == 0 && native.elapsed == 4 && native.synchronizes == 1 &&
               native.context_destroys == 2 &&
               std::ranges::none_of(native.events, [](const auto& item) { return item.is_live; }) &&
               service.request_stop(),
           "old-generation identity, original timestamps/flags and exactly-once owner cleanup");
}
void foreign_request_validation(const std::string& path) {
    using namespace glimmer::control;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "foreign request service identity");
    const auto waits_before = native.waits.load();
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch());
    expect(service.start(), "start authenticated foreign-event rejection endpoint");
    auto client = std::async(std::launch::async, [&] {
        const auto success = [](std::optional<ExecutionWireReply> reply,
                                std::source_location location = std::source_location::current()) {
            if (!reply || reply->status)
                throw std::runtime_error(
                    "raw owned operation failed at line " + std::to_string(location.line()) +
                    ": status=" + (reply ? std::to_string(reply->status) : "no reply"));
            return *reply;
        };
        const auto request = [](std::uint64_t source, std::uint64_t event, std::uint64_t stream = 0,
                                bool has_extra = false, unsigned int wait_flags = 0) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(source) && input.write_u64(event) && input.write_u64(stream) &&
                       input.write_u32(wait_flags) && (!has_extra || input.write_u32(0)),
                   "encode foreign context capability without native handles");
            return input.take();
        };
        const auto rejected = [](std::optional<ExecutionWireReply> reply, CUresult expected) {
            expect(reply && reply->status == expected && !reply->original_status,
                   "untrusted foreign event rejects before original SDK issue");
        };
        ExecutionSocketClient source({path, *identity, 5000});
        ExecutionSocketClient target({path, *identity, 5000});
        success(source.open());
        ExecutionPayloadWriter flags;
        expect(flags.write_u32(0), "encode ordinary event flags");
        const auto event = success(source.exchange(ExecutionOpcode::kCreateEvent, flags.take()));
        success(target.open());
        rejected(target.exchange(ExecutionOpcode::kWaitForeignEvent, request(UINT64_MAX, event.id)),
                 CUDA_ERROR_INVALID_HANDLE);
        rejected(target.exchange(ExecutionOpcode::kWaitForeignEvent,
                                 request(source.session(), UINT64_MAX)),
                 CUDA_ERROR_INVALID_HANDLE);
        rejected(target.exchange(ExecutionOpcode::kWaitForeignEvent,
                                 request(source.session(), event.id, UINT64_MAX)),
                 CUDA_ERROR_INVALID_HANDLE);
        rejected(target.exchange(ExecutionOpcode::kWaitForeignEvent,
                                 request(source.session(), event.id, 0, true)),
                 CUDA_ERROR_INVALID_VALUE);
        rejected(
            target.exchange(ExecutionOpcode::kWaitForeignEvent,
                            request(source.session(), event.id, 0, false, CU_EVENT_WAIT_EXTERNAL)),
            CUDA_ERROR_INVALID_VALUE);
        ExecutionSocketClient host({path, *identity, 5000});
        ExecutionPayloadWriter role;
        expect(role.write_u32(kExecutionHostOwnerRole) && role.write_u32(0),
               "encode isolated host role");
        success(host.open(role.take()));
        rejected(
            target.exchange(ExecutionOpcode::kWaitForeignEvent, request(host.session(), event.id)),
            CUDA_ERROR_INVALID_HANDLE);
        success(host.exchange(ExecutionOpcode::kClose));
        const auto source_id = source.session();
        const auto child = ::fork();
        expect(child >= 0, "fork distinct authenticated foreign-event sender");
        if (!child) {
            try {
                ExecutionSocketClient foreign({path, *identity, 5000});
                success(foreign.open());
                rejected(foreign.exchange(ExecutionOpcode::kWaitForeignEvent,
                                          request(source_id, event.id)),
                         CUDA_ERROR_INVALID_HANDLE);
                success(foreign.exchange(ExecutionOpcode::kClose));
                _exit(EXIT_SUCCESS);
            } catch (...) {
                _exit(EXIT_FAILURE);
            }
        }
        int status{};
        expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
               "same-UID child cannot borrow another kernel-authenticated peer's event");
        success(target.exchange(ExecutionOpcode::kWaitForeignEvent, request(source_id, event.id)));
        success(source.exchange(ExecutionOpcode::kClose));
        rejected(target.exchange(ExecutionOpcode::kWaitForeignEvent, request(source_id, event.id)),
                 CUDA_ERROR_INVALID_HANDLE);
        success(target.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        for (auto& item : native.events)
            if (item.is_live) item.is_ready = true;
        expect(service.step(1) && Clock::now() < deadline, "bounded foreign capability validation");
    }
    client.get();
    expect(native.waits.load() == waits_before + 1 && service.request_stop(),
           "only one authenticated live-source wait reaches the native SDK");
}
void backend_borrow_lifetime() {
    const auto destroyed_before = native.context_destroys;
    CudaSessionLimits limits;
    limits.max_objects = 16;
    CudaExecutionSession source({71, 1}, 71, dispatch(), limits);
    CudaExecutionSession target({73, 1}, 73, dispatch(), limits);
    check(source.initialize(0).status);
    const auto source_context = native.current;
    const auto created = source.create_event(0);
    check(created.status);
    const auto recorded = source.record_event(1, created.id, 0);
    expect(recorded.status == CUDA_SUCCESS && !recorded.original_status &&
               source.issue_next().state == CudaExecutionIssueState::kIssued,
           "start original source record without claiming admission as SDK success");
    check(source.deactivate().status);
    check(target.initialize(0).status);
    const auto target_context = native.current;
    auto captured = source.capture_event(created.id);
    expect(captured.status == CUDA_SUCCESS && native.current == target_context &&
               captured.event.observe() == CUDA_ERROR_NOT_READY,
           "foreign capture/observation is CPU-only while another context is current");
    auto other_capture = source.capture_event(created.id);
    expect(other_capture.status == CUDA_SUCCESS, "separate bounded CPU borrow");
    auto other_thread =
        std::async(std::launch::async, [token = std::move(other_capture.event)]() mutable {
            const auto status = token.observe();
            token = {};  // Release on this worker, not in future.get().
            return status;
        });
    expect(other_thread.get() == CUDA_ERROR_INVALID_CONTEXT,
           "foreign-thread observation rejects and CPU lease release performs no SDK cleanup");
    const auto child = ::fork();
    expect(child >= 0, "fork captured source-generation guard");
    if (!child) _exit(captured.event.observe() == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork cannot convert an inherited generation into physical completion");
    check(target.wait_event(1, captured.event, 0).status);
    expect(!target.snapshot().has_ready_work &&
               target.issue_next().state == CudaExecutionIssueState::kIdle,
           "dependency-blocked queue is not runnable GPU demand");
    check(source.activate().status);
    check(source.destroy_event(created.id).status);
    expect(source.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               native.context_destroys == destroyed_before,
           "source destroy defers while a foreign command owns its immutable generation");
    for (auto& item : native.events)
        if (item.is_live && item.context == source_context) item.is_ready = true;
    expect(source.poll().retired == 1 && captured.event.observe() == CUDA_SUCCESS &&
               target.snapshot().has_ready_work,
           "real source retirement, not an owner switch or intent, enables the consumer");
    check(target.activate().status);
    expect(target.issue_next().state == CudaExecutionIssueState::kIssued,
           "foreign native wait issues once after source completion");
    captured.event = {};
    check(source.activate().status);
    expect(source.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               native.context_destroys == destroyed_before,
           "SDK wait return alone cannot release the foreign native context");
    check(target.activate().status);
    for (auto& item : native.events)
        if (item.is_live && item.context == target_context) item.is_ready = true;
    expect(target.poll().retired == 1, "target private event proves actual wait retirement");
    check(source.activate().status);
    check(source.shutdown().status);
    check(target.activate().status);
    check(target.shutdown().status);
    expect(native.context_destroys == destroyed_before + 2 &&
               std::ranges::none_of(native.events, [](const auto& item) { return item.is_live; }),
           "last CPU lease release permits exactly-once native context teardown");
    CudaExecutionSession uncertain({79, 1}, 79, dispatch(), limits);
    check(uncertain.initialize(0).status);
    const auto valid = uncertain.create_event(0), failing = uncertain.create_event(0);
    check(valid.status);
    check(failing.status);
    const auto cleanup_before = native.event_destroy_attempts;
    native.should_fail_event_destroy = true;
    expect(uncertain.destroy_event(failing.id).status == CUDA_ERROR_UNKNOWN &&
               uncertain.capture_event(valid.id).status == CUDA_ERROR_UNKNOWN,
           "uncertain source cleanup cannot authorize a new foreign event borrow");
    check(uncertain.shutdown().status);
    expect(native.event_destroy_attempts == cleanup_before + 1 &&
               std::ranges::none_of(native.events, [](const auto& item) { return item.is_live; }),
           "context recovery settles ambiguous event destruction without replaying it");
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-owned-events-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock");
        foreign_request_validation(std::string(directory) + "/foreign.sock");
        backend_borrow_lifetime();
        std::cout << "Owned event ABI, generation, dependency and timestamp checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
