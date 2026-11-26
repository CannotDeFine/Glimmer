#include "internal/launch_completion_tracker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>

namespace {

using glimmer::interceptor::DriverDispatch;
using glimmer::interceptor::DriverFunctionTable;
using glimmer::interceptor::LaunchCompletionTracker;

std::mutex g_mutex;
std::condition_variable g_condition;
bool g_destroy_entered = false;
bool g_destroy_released = false;
CUresult g_destroy_result = CUDA_SUCCESS;
std::array<int, 4> g_event_storage{};
std::size_t g_created = 0;
std::atomic<unsigned int> g_destroyed{0};
std::atomic<unsigned int> g_queries{0};
CUresult g_create_result = CUDA_SUCCESS;
CUresult g_record_result = CUDA_SUCCESS;
CUresult g_query_result = CUDA_SUCCESS;
unsigned int g_not_ready_queries = 0;
thread_local CUstreamCaptureMode g_thread_capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
std::atomic<bool> g_capture_invalidated{false};
std::atomic<unsigned int> g_mode_entries{0};
std::atomic<unsigned int> g_mode_restores{0};
bool g_has_global_capture = false;
thread_local bool g_scope_armed = false;
CUresult g_entry_result = CUDA_SUCCESS;
CUresult g_restore_result = CUDA_SUCCESS;
CUresult g_resolution_result = CUDA_SUCCESS;
bool g_null_resolution = false;
bool g_throw_resolution = false;
bool g_throw_query = false;
bool g_captured_event = false;
std::atomic<bool> g_missing_guard{false};
std::atomic<bool> g_incorrect_mode{false};

class DiagnosticCapture final {
   public:
    DiagnosticCapture() : file_(std::tmpfile(), &std::fclose), saved_(::dup(STDERR_FILENO)) {
        redirected_ = file_ && saved_ >= 0 && ::dup2(::fileno(file_.get()), STDERR_FILENO) >= 0;
    }
    ~DiagnosticCapture() {
        if (redirected_ && ::dup2(saved_, STDERR_FILENO) < 0) std::abort();
        if (saved_ >= 0 && ::close(saved_) != 0) std::abort();
    }
    DiagnosticCapture(const DiagnosticCapture&) = delete;
    DiagnosticCapture& operator=(const DiagnosticCapture&) = delete;
    [[nodiscard]] bool valid() const {
        return redirected_;
    }
    [[nodiscard]] std::string read() {
        if (!redirected_ || std::fseek(file_.get(), 0, SEEK_SET) != 0) return {};
        std::string result;
        std::array<char, 512> bytes{};
        while (const auto count = std::fread(bytes.data(), 1, bytes.size(), file_.get())) {
            result.append(bytes.data(), count);
        }
        return std::ferror(file_.get()) == 0 ? result : std::string{};
    }

   private:
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file_;
    int saved_ = -1;
    bool redirected_ = false;
};

CUresult exchange_capture_mode(CUstreamCaptureMode* mode) {
    if (!glimmer::interceptor::is_inside_driver_call()) g_missing_guard = true;
    if (mode == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_scope_armed) {
        ++g_mode_entries;
        if (*mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL ||
            (g_restore_result == CUDA_SUCCESS &&
             g_thread_capture_mode != CU_STREAM_CAPTURE_MODE_GLOBAL)) {
            g_incorrect_mode = true;
        }
        if (g_entry_result != CUDA_SUCCESS) return g_entry_result;
        g_scope_armed = true;
    } else {
        ++g_mode_restores;
        if (g_restore_result == CUDA_SUCCESS &&
            (*mode != CU_STREAM_CAPTURE_MODE_GLOBAL ||
             g_thread_capture_mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)) {
            g_incorrect_mode = true;
        }
        g_scope_armed = false;
        if (g_restore_result != CUDA_SUCCESS) return g_restore_result;
    }
    std::swap(*mode, g_thread_capture_mode);
    return CUDA_SUCCESS;
}

CUresult resolve_capture_mode(const char* symbol, void** pointer, int, cuuint64_t) {
    if (g_throw_resolution) throw std::runtime_error("injected resolution failure");
    if (symbol == nullptr || pointer == nullptr ||
        std::string_view(symbol) != "cuThreadExchangeStreamCaptureMode") {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *pointer = g_null_resolution ? nullptr : reinterpret_cast<void*>(&exchange_capture_mode);
    return g_resolution_result;
}

CUresult create_event(CUevent* event, unsigned int) {
    if (g_create_result == CUDA_SUCCESS && g_created < g_event_storage.size()) {
        *event = reinterpret_cast<CUevent>(&g_event_storage[g_created++]);
    }
    return g_create_result;
}

CUresult record_event(CUevent, CUstream) {
    return g_record_result;
}

CUresult query_event(CUevent) {
    if (g_has_global_capture && g_thread_capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL) {
        g_capture_invalidated = true;
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }
    const auto count = g_queries.fetch_add(1);
    if (g_throw_query) throw std::runtime_error("injected query failure");
    if (g_captured_event) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    return count < g_not_ready_queries ? CUDA_ERROR_NOT_READY : g_query_result;
}

CUresult destroy_event(CUevent) {
    if (g_has_global_capture && g_thread_capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL) {
        g_capture_invalidated = true;
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }
    std::unique_lock lock(g_mutex);
    g_destroy_entered = true;
    g_condition.notify_all();
    g_condition.wait(lock, [] { return g_destroy_released; });
    ++g_destroyed;
    return g_destroy_result;
}

bool close_during_destroy(CUresult destroy_result) {
    {
        std::scoped_lock lock(g_mutex);
        g_destroy_entered = false;
        g_destroy_released = false;
        g_destroy_result = destroy_result;
        g_created = 0;
    }
    DriverFunctionTable functions;
    functions.event_create = create_event;
    functions.event_record = record_event;
    functions.event_query = query_event;
    functions.event_destroy = destroy_event;
    functions.get_proc_address = resolve_capture_mode;
    DriverDispatch driver(functions);
    glimmer::control::LaunchGate gate;
    LaunchCompletionTracker tracker(driver, gate, false);
    const auto task = gate.acquire();
    if (!task || !tracker.start() || !tracker.track(*task, nullptr)) {
        return false;
    }
    bool entered = false;
    {
        std::unique_lock lock(g_mutex);
        entered =
            g_condition.wait_for(lock, std::chrono::seconds(5), [] { return g_destroy_entered; });
    }
    const bool closed = tracker.close_batch(*task);
    const auto during_destroy = gate.stats();
    {
        std::scoped_lock lock(g_mutex);
        g_destroy_released = true;
    }
    g_condition.notify_all();
    // Admission of the next task is the synchronization boundary for the
    // terminal transition. No arbitrary sleep or fixed timing assertion.
    const auto next = gate.acquire(std::chrono::seconds(5));
    const auto after_destroy = gate.stats();
    if (next) {
        static_cast<void>(gate.complete(*next));
    }
    // The local gate forgets terminal tasks. Only the newly admitted task
    // should remain; remote integration tests retain terminal outcomes.
    const bool expected_terminal =
        after_destroy.total_task_count == 1 && after_destroy.running_task_count == 1;
    if (!entered || !closed || !expected_terminal || during_destroy.running_task_count != 1) {
        std::cerr << "entered=" << entered << " closed=" << closed
                  << " running=" << during_destroy.running_task_count
                  << " completed=" << after_destroy.completed_task_count
                  << " failed=" << after_destroy.failed_task_count << '\n';
    }
    return entered && closed && during_destroy.running_task_count == 1 &&
           during_destroy.completed_task_count == 0 && during_destroy.failed_task_count == 0 &&
           next.has_value() && expected_terminal;
}

bool failure_and_cleanup_paths() {
    DriverFunctionTable functions;
    functions.event_create = create_event;
    functions.event_record = record_event;
    functions.event_query = query_event;
    functions.event_destroy = destroy_event;
    functions.get_proc_address = resolve_capture_mode;
    // Creation failure, recording failure, query failure, not-ready progress,
    // multiple events in a batch, and shutdown with pending events.
    for (int scenario = 0; scenario < 6; ++scenario) {
        g_created = 0;
        g_destroyed = 0;
        g_queries = 0;
        g_destroy_released = true;
        g_destroy_result = CUDA_SUCCESS;
        g_create_result = scenario == 0 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
        g_record_result = scenario == 1 ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
        g_query_result = scenario == 2 ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
        g_not_ready_queries = scenario == 3 ? 20 : 0;
        DriverDispatch driver(functions);
        glimmer::control::LaunchGate gate;
        const auto task = gate.acquire();
        if (!task) {
            return false;
        }
        {
            LaunchCompletionTracker tracker(driver, gate, false);
            if (tracker.track(0, nullptr) || tracker.close_batch(0) || tracker.fail(0)) {
                return false;
            }
            const bool tracked = tracker.track(*task, nullptr);
            if (tracked != (scenario >= 2)) {
                return false;
            }
            if (scenario >= 4 && !tracker.track(*task, nullptr)) {
                return false;
            }
            if (scenario >= 2 && scenario != 5) {
                if (!tracker.close_batch(*task) || !tracker.start() || tracker.start()) {
                    return false;
                }
                const auto next = gate.acquire(std::chrono::seconds(5));
                if (!next || !gate.complete(*next)) {
                    return false;
                }
            }
        }
        const auto expected_destroyed = scenario == 0 ? 0U : (scenario >= 4 ? 2U : 1U);
        if (g_destroyed != expected_destroyed || gate.stats().total_task_count != 0) {
            return false;
        }
    }
    return true;
}

bool capture_mode_regression() {
    g_created = 0;
    g_queries = 0;
    g_destroyed = 0;
    g_destroy_released = true;
    g_create_result = CUDA_SUCCESS;
    g_record_result = CUDA_SUCCESS;
    g_query_result = CUDA_SUCCESS;
    g_destroy_result = CUDA_SUCCESS;
    g_not_ready_queries = 0;
    g_has_global_capture = true;
    g_capture_invalidated = false;
    g_mode_entries = 0;
    g_mode_restores = 0;
    g_incorrect_mode = false;
    DriverFunctionTable functions;
    functions.event_create = create_event;
    functions.event_record = record_event;
    functions.event_query = query_event;
    functions.event_destroy = destroy_event;
    functions.get_proc_address = resolve_capture_mode;
    DriverDispatch driver(functions);
    glimmer::control::LaunchGate gate;
    const auto task = gate.acquire();
    bool finished = false;
    {
        LaunchCompletionTracker tracker(driver, gate, false);
        if (!task || !tracker.start() || !tracker.track(*task, nullptr) ||
            !tracker.close_batch(*task)) {
            return false;
        }
        const auto next = gate.acquire(std::chrono::seconds{5});
        finished = next && gate.complete(*next);
    }
    g_has_global_capture = false;
    return finished && !g_capture_invalidated && !g_incorrect_mode && g_queries == 1 &&
           g_destroyed == 1 && g_mode_entries == 2 && g_mode_restores == 2 &&
           g_thread_capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL;
}

bool mode_failure_paths() {
    // A failed resolver, null original and throwing injected resolver must
    // reject startup before event/lease side effects or a live monitor.
    for (int scenario = 0; scenario < 4; ++scenario) {
        DriverFunctionTable functions;
        functions.get_proc_address = scenario == 0 ? nullptr : resolve_capture_mode;
        DriverDispatch driver(functions);
        glimmer::control::LaunchGate gate;
        LaunchCompletionTracker tracker(driver, gate, false);
        g_resolution_result = scenario == 1 ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
        g_null_resolution = scenario == 2;
        g_throw_resolution = scenario == 3;
        if (tracker.start() || gate.stats().total_task_count != 0) return false;
    }
    g_resolution_result = CUDA_SUCCESS;
    g_null_resolution = false;
    g_throw_resolution = false;

    for (int scenario = 0; scenario < 5; ++scenario) {
        g_created = 0;
        g_queries = 0;
        g_destroyed = 0;
        g_destroy_released = true;
        g_capture_invalidated = false;
        g_mode_entries = 0;
        g_mode_restores = 0;
        g_missing_guard = false;
        g_incorrect_mode = false;
        g_entry_result = scenario == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
        g_restore_result = scenario == 1 ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        g_captured_event = scenario == 2;
        g_throw_query = scenario == 3;
        g_not_ready_queries = scenario == 4 ? 7 : 0;
        g_has_global_capture = true;
        DriverFunctionTable functions;
        functions.event_create = create_event;
        functions.event_record = record_event;
        functions.event_query = query_event;
        functions.event_destroy = destroy_event;
        functions.get_proc_address = resolve_capture_mode;
        DriverDispatch driver(functions);
        glimmer::control::LaunchGate gate;
        const auto task = gate.acquire();
        DiagnosticCapture diagnostic;
        if (!task || !diagnostic.valid()) return false;
        {
            LaunchCompletionTracker tracker(driver, gate, true);
            if (!tracker.start() || !tracker.track(*task, nullptr) || !tracker.close_batch(*task))
                return false;
            const auto next = gate.acquire(std::chrono::seconds{5});
            if (!next || !gate.complete(*next)) return false;
        }
        const auto output = diagnostic.read();
        const bool should_succeed = scenario == 4;
        const bool terminal_matches =
            output.find(should_succeed ? " completed=1\n" : " completed=0\n") != std::string::npos;
        const auto expected_queries = scenario == 0 ? 0U : (scenario == 4 ? 8U : 1U);
        const auto expected_destroyed = scenario == 0 ? 0U : 1U;
        if (!terminal_matches || g_queries != expected_queries ||
            g_destroyed != expected_destroyed || g_capture_invalidated || g_missing_guard ||
            g_incorrect_mode || g_thread_capture_mode != CU_STREAM_CAPTURE_MODE_GLOBAL ||
            g_mode_entries != (scenario == 0 ? 2U : expected_queries + 1) ||
            g_mode_restores != (scenario == 0 ? 0U : expected_queries + 1)) {
            return false;
        }
    }
    g_entry_result = CUDA_SUCCESS;
    g_restore_result = CUDA_SUCCESS;
    g_captured_event = false;
    g_throw_query = false;
    g_not_ready_queries = 0;
    g_has_global_capture = false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--capture-mode") {
        if (!capture_mode_regression()) {
            std::cerr << "private event polling invalidated caller GLOBAL capture\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    if (!close_during_destroy(CUDA_SUCCESS) || !close_during_destroy(CUDA_ERROR_INVALID_HANDLE)) {
        std::cerr << "batch closed before event cleanup completed\n";
        return EXIT_FAILURE;
    }
    if (!failure_and_cleanup_paths()) {
        std::cerr << "event failure, polling, or cleanup handling failed\n";
        return EXIT_FAILURE;
    }
    if (!capture_mode_regression()) {
        std::cerr << "private event polling invalidated caller GLOBAL capture\n";
        return EXIT_FAILURE;
    }
    if (!mode_failure_paths()) {
        std::cerr << "private capture-mode failure, restore, or terminal reporting failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
