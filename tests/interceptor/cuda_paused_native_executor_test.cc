#include "internal/cuda_paused_native_executor.h"

#include <atomic>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
const auto kStream = reinterpret_cast<CUstream>(0x2000);
thread_local CUcontext current_context{};
thread_local bool is_native{};
thread_local CUstreamCaptureMode capture_mode{CU_STREAM_CAPTURE_MODE_GLOBAL};
std::atomic<bool> is_complete{false};
std::atomic<unsigned int> creates{}, records{}, ptds_records{}, queries{}, destroys{};
unsigned int failure_mode{};
unsigned int mode_failure{};
struct Event {
    bool was_recorded{};
    std::chrono::steady_clock::time_point next_query{};
};
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
CUresult create(CUevent* output, unsigned int flags) {
    if (mode_failure == 3) {
        throw std::runtime_error("private event creation exception");
    }
    if (!is_native || current_context != kContext || flags != CU_EVENT_DISABLE_TIMING ||
        capture_mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    ++creates;
    *output = std::bit_cast<CUevent>(new Event{});
    return CUDA_SUCCESS;
}
CUresult record(CUevent event, CUstream stream) {
    ++records;
    if (!is_native || current_context != kContext || stream != kStream ||
        capture_mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (failure_mode == 1) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    std::bit_cast<Event*>(event)->was_recorded = true;
    return CUDA_SUCCESS;
}
CUresult record_ptds(CUevent event, CUstream stream) {
    ++ptds_records;
    return record(event, stream);
}
CUresult query(CUevent event) {
    ++queries;
    if (!is_native || current_context != kContext || !std::bit_cast<Event*>(event)->was_recorded ||
        capture_mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (failure_mode == 2) {
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    auto& native = *std::bit_cast<Event*>(event);
    const auto now = std::chrono::steady_clock::now();
    if (now < native.next_query) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!is_complete.load()) {
        native.next_query = now + std::chrono::milliseconds{1};
        return CUDA_ERROR_NOT_READY;
    }
    return CUDA_SUCCESS;
}
CUresult destroy(CUevent event) {
    ++destroys;
    if (!is_native || current_context != kContext ||
        capture_mode != CU_STREAM_CAPTURE_MODE_RELAXED) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (failure_mode == 3) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    delete std::bit_cast<Event*>(event);
    return CUDA_SUCCESS;
}
bool enter() noexcept {
    const bool old = is_native;
    is_native = true;
    return old;
}
void leave(bool old) noexcept {
    is_native = old;
}
CUresult exchange_mode(CUstreamCaptureMode* mode) {
    if ((mode_failure == 1 && *mode == CU_STREAM_CAPTURE_MODE_RELAXED) ||
        (mode_failure == 2 && *mode != CU_STREAM_CAPTURE_MODE_RELAXED)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::swap(capture_mode, *mode);
    return CUDA_SUCCESS;
}
// The private mode callbacks are shared by clean and contained-failure cases.
CudaPausedNativeDispatch dispatch() {
    return {create, record, record_ptds, query, destroy, enter, leave, exchange_mode};
}
CUresult in_context(CUcontext context, const CudaPausedNativeExecutor::NativeOperation& operation) {
    const auto original = current_context;
    current_context = context;
    try {
        const auto result = operation();
        current_context = original;
        return result;
    } catch (...) {
        current_context = original;
        throw;
    }
}
void private_mode_failure(unsigned int mode) {
    const pid_t child = fork();
    require(child >= 0, "fork private capture-mode failure containment");
    if (child == 0) {
        mode_failure = mode;
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
        CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
        CudaPausedNativePermit permit;
        if (executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit) !=
            CUDA_SUCCESS) {
            _exit(1);
        }
        unsigned int calls{};
        const auto result = executor.issue(permit, [&] {
            ++calls;
            return CUDA_SUCCESS;
        });
        const auto expected = mode == 3 ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_VALUE;
        if (result.status != expected || result.tracking_status != expected || result.was_invoked ||
            calls != 0 || executor.drain() != expected || executor.stats().retained != 1 ||
            capture_mode !=
                (mode == 2 ? CU_STREAM_CAPTURE_MODE_RELAXED : CU_STREAM_CAPTURE_MODE_GLOBAL) ||
            is_native || current_context) {
            _exit(2);
        }
        _exit(0);
    }
    int status{};
    require(
        waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "checked mode exchange/restore and private exceptions quarantine without original issue");
}
void thread_local_mode_preservation() {
    capture_mode = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
    CudaPausedNativePermit permit;
    require(executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit) ==
                CUDA_SUCCESS,
            "acquire thread-local source mode");
    const auto issued = executor.issue(permit, [] {
        return capture_mode == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL ? CUDA_SUCCESS
                                                                   : CUDA_ERROR_INVALID_VALUE;
    });
    require(issued.status == CUDA_SUCCESS && executor.drain() == CUDA_SUCCESS &&
                capture_mode == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            "private mode scopes restore nondefault application thread mode");
    capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
}
template <typename Predicate>
void wait(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline,
                "bounded deterministic state observation");
        std::this_thread::yield();
    }
}
void success_and_async_admission() {
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
    CudaPausedNativeExecutor foreign(gate, dispatch(), in_context);
    auto resources = std::make_shared<int>(1);
    std::weak_ptr<void> retained = resources;
    CudaPausedNativePermit first;
    require(
        executor.acquire(kContext, kStream, false, resources, {}, first) == CUDA_SUCCESS && first,
        "original issue requires a real shared execution grant");
    resources.reset();
    require(!retained.expired(), "entry owns caller resources beyond API return");
    unsigned int original_calls{};
    const auto original = [&] {
        require(capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL,
                "application operation preserves original capture mode");
        ++original_calls;
        wait([&] { return executor.stats().heartbeats > 0; });
        require(queries == 0,
                "issuing source keeps its grant alive without polling an unrecorded event");
        return CUDA_SUCCESS;
    };
    require(
        foreign.issue(first, original).status == CUDA_ERROR_INVALID_VALUE && original_calls == 0,
        "foreign permit cannot invoke original CUDA");
    const auto issued = executor.issue(first, original);
    require(issued.status == CUDA_SUCCESS && issued.tracking_status == CUDA_SUCCESS &&
                issued.was_invoked && original_calls == 1 && records == 1 &&
                executor.stats().completed == 0,
            "native issue returns before its actual fence completes");
    wait([&] { return queries.load() > 0 && executor.stats().heartbeats > 0; });
    require(executor.issue(first, original).status == CUDA_ERROR_INVALID_VALUE &&
                executor.retire_unissued(first) == CUDA_ERROR_INVALID_VALUE && original_calls == 1,
            "duplicate issue or cancellation cannot replay or release running work");
    std::latch acquiring(1);
    std::atomic<bool> was_admitted{false};
    CUresult next_result{};
    std::thread next([&] {
        CudaPausedNativePermit permit;
        acquiring.count_down();
        next_result =
            executor.acquire(kContext, kStream, true, std::make_shared<int>(2), {}, permit);
        if (next_result == CUDA_SUCCESS) {
            was_admitted.store(true);
            next_result = executor.issue(permit, [] { return CUDA_ERROR_INVALID_VALUE; }).status;
        }
    });
    acquiring.wait();
    wait([&] { return executor.stats().retained == 2; });
    require(!was_admitted.load(),
            "API return and heartbeat do not free physical execution capacity");
    is_complete.store(true);
    next.join();
    require(next_result == CUDA_ERROR_INVALID_VALUE && was_admitted.load() &&
                executor.drain() == CUDA_SUCCESS,
            "physical proof admits the next source and preserves its original native error");
    const auto stats = executor.stats();
    require(stats.grants == 2 && stats.completed == 1 && stats.failed == 1 && stats.retained == 0 &&
                creates == 2 && records == 2 && ptds_records == 1 && destroys == 2 && !is_native &&
                !current_context && capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL,
            "exact terminal reporting, native ABI and context/scope restoration");
    first = {};
    require(retained.expired(), "retired permit releases its metadata resource exactly once");

    CudaPausedNativePermit unused;
    require(executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, unused) ==
                    CUDA_SUCCESS &&
                executor.retire_unissued(unused) == CUDA_SUCCESS &&
                executor.drain() == CUDA_SUCCESS &&
                executor.issue(unused, original).status == CUDA_ERROR_INVALID_VALUE && creates == 2,
            "reclassified construction retires an unissued grant without any private CUDA");
}
void pending_admission_failure() {
    const pid_t child = fork();
    require(child >= 0, "fork pending-admission failure containment");
    if (child == 0) {
        failure_mode = 0;
        is_complete.store(false);
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
        CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
        CudaPausedNativePermit first;
        if (executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, first) !=
            CUDA_SUCCESS) {
            _exit(1);
        }
        std::atomic<CUresult> result{CUDA_SUCCESS};
        std::atomic<bool> returned{};
        std::thread pending([&] {
            CudaPausedNativePermit permit;
            result.store(
                executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit));
            returned.store(true);
        });
        wait([&] { return executor.stats().retained == 2; });
        // Set before monitor can query this first, still unissued entry.
        failure_mode = 2;
        if (executor.issue(first, [] { return CUDA_SUCCESS; }).status != CUDA_SUCCESS) {
            _exit(2);
        }
        wait([&] { return returned.load(); });
        pending.join();
        if (result.load() != CUDA_ERROR_LAUNCH_FAILED || executor.stats().retained != 2 ||
            executor.stats().completed != 0 || capture_mode != CU_STREAM_CAPTURE_MODE_GLOBAL) {
            _exit(3);
        }
        // Uncertain ownership is intentionally retained until process exit.
        _exit(0);
    }
    int status{};
    require(
        waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "another entry's failure wakes pending caller without native issue or capacity release");
}
void source_failure_wakes_admission() {
    const auto child = fork();
    require(child >= 0, "fork source failure containment");
    if (child == 0) {
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
        CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
        CudaPausedNativePermit first;
        if (executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, first) !=
            CUDA_SUCCESS) {
            _exit(1);
        }
        std::atomic<CUresult> result{CUDA_SUCCESS};
        std::atomic<bool> returned{};
        std::thread pending([&] {
            CudaPausedNativePermit permit;
            result.store(
                executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit));
            returned.store(true);
        });
        wait([&] { return executor.stats().retained == 2; });
        const auto before_creates = creates.load();
        const auto before_records = records.load();
        const auto before_queries = queries.load();
        executor.fail_closed(CUDA_ERROR_LAUNCH_FAILED);
        executor.fail_closed(CUDA_ERROR_INVALID_HANDLE);
        wait([&] { return returned.load(); });
        pending.join();
        unsigned int original_calls{};
        const auto issued = executor.issue(first, [&] {
            ++original_calls;
            return CUDA_SUCCESS;
        });
        CudaPausedNativePermit rejected;
        if (result.load() != CUDA_ERROR_LAUNCH_FAILED ||
            executor.status() != CUDA_ERROR_LAUNCH_FAILED ||
            executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, rejected) !=
                CUDA_ERROR_LAUNCH_FAILED ||
            rejected || issued.status != CUDA_ERROR_LAUNCH_FAILED || issued.was_invoked ||
            original_calls != 0 || executor.drain() != CUDA_ERROR_LAUNCH_FAILED ||
            executor.stats().retained != 2 || executor.stats().completed != 0 ||
            creates.load() != before_creates || records.load() != before_records ||
            queries.load() != before_queries) {
            _exit(2);
        }
        // Neither entry was issued. Source quarantine retains CPU/control
        // ownership and makes no private CUDA call or guessed terminal report.
        _exit(0);
    }
    int status{};
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "source failure wakes pending admission and rejects cached permits without CUDA");
}

void bounded_and_caller_bound() {
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaPausedNativeExecutor executor(gate, dispatch(), in_context, 1);
    CudaPausedNativePermit first, rejected;
    require(executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, first) ==
                    CUDA_SUCCESS &&
                executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {},
                                 rejected) == CUDA_ERROR_OUT_OF_MEMORY &&
                !rejected,
            "bounded entry rejection precedes native issue");
    CudaPausedNativeResult wrong;
    std::thread foreign([&] { wrong = executor.issue(first, [] { return CUDA_SUCCESS; }); });
    foreign.join();
    require(wrong.status == CUDA_ERROR_INVALID_VALUE && !wrong.was_invoked &&
                executor.retire_unissued(first) == CUDA_SUCCESS && executor.drain() == CUDA_SUCCESS,
            "a permit cannot reinterpret another caller's PTDS or Runtime error state");
}
void unrelated_wakeups_preserve_poll_deadlines() {
    is_complete.store(false);
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaPausedNativeExecutor executor(gate, dispatch(), in_context);
    CudaPausedNativePermit permit;
    const auto before_queries = queries.load();
    require(executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit) ==
                    CUDA_SUCCESS &&
                executor.issue(permit, [] { return CUDA_SUCCESS; }).status == CUDA_SUCCESS,
            "wake regression issues exactly one admitted native operation");
    wait([&] { return queries.load() > before_queries; });
    for (unsigned int count = 0; count < 512; ++count) {
        require(
            gate->claim(999999).get().status == glimmer::control::CooperativeClaimStatus::kFailed,
            "unrelated metadata publication must retain its actual control result");
    }
    require(executor.status() == CUDA_SUCCESS && executor.stats().completed == 0 &&
                executor.stats().retained == 1,
            "publication storms cannot bypass NOT_READY backoff or release physical capacity");
    is_complete.store(true);
    require(executor.drain() == CUDA_SUCCESS && executor.stats().completed == 1 &&
                executor.stats().retained == 0,
            "actual native proof and terminal acknowledgment still retire the entry");
}
void retained_failure(unsigned int mode) {
    const pid_t child = fork();
    require(child >= 0, "fork failure containment test");
    if (child == 0) {
        failure_mode = mode;
        is_complete.store(true);
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
        std::weak_ptr<glimmer::control::AsyncCooperativeLaunchGate> held = gate;
        auto executor = std::make_unique<CudaPausedNativeExecutor>(gate, dispatch(), in_context);
        gate.reset();
        CudaPausedNativePermit permit;
        if (executor->acquire(kContext, kStream, false, std::make_shared<int>(0), {}, permit) !=
            CUDA_SUCCESS) {
            _exit(1);
        }
        unsigned int calls{};
        const auto before_queries = queries.load();
        const auto before_destroys = destroys.load();
        const auto result = executor->issue(permit, [&]() -> CUresult {
            ++calls;
            if (mode == 4) {
                throw std::runtime_error("uncertain native source issue");
            }
            return CUDA_SUCCESS;
        });
        if (mode == 1 || mode == 4) {
            if (result.tracking_status == CUDA_SUCCESS) {
                _exit(2);
            }
        } else if (result.status != CUDA_SUCCESS) {
            _exit(3);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        while (executor->status() == CUDA_SUCCESS && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (executor->status() == CUDA_SUCCESS || executor->drain() == CUDA_SUCCESS ||
            executor
                ->issue(permit,
                        [&] {
                            ++calls;
                            return CUDA_SUCCESS;
                        })
                .was_invoked ||
            calls != 1) {
            _exit(4);
        }
        if ((mode == 1 || mode == 4) && queries != before_queries) {
            _exit(6);
        }
        if (mode == 2 && queries != before_queries + 1) {
            _exit(7);
        }
        if (mode == 3 && destroys != before_destroys + 1) {
            _exit(8);
        }
        executor.reset();
        if (held.expired()) {
            _exit(5);
        }
        _exit(0);
    }
    int status{};
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "uncertain record/query/destroy/issue retains capacity and never replays native CUDA");
}
}  // namespace
int main() {
    try {
        success_and_async_admission();
        bounded_and_caller_bound();
        unrelated_wakeups_preserve_poll_deadlines();
        pending_admission_failure();
        source_failure_wakes_admission();
        thread_local_mode_preservation();
        for (unsigned int mode = 1; mode <= 3; ++mode) {
            private_mode_failure(mode);
        }
        for (unsigned int mode = 1; mode <= 4; ++mode) {
            retained_failure(mode);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
