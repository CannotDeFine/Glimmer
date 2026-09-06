#include "internal/launch_completion_tracker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>

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
    return g_queries.fetch_add(1) < g_not_ready_queries ? CUDA_ERROR_NOT_READY : g_query_result;
}

CUresult destroy_event(CUevent) {
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

}  // namespace

int main() {
    if (!close_during_destroy(CUDA_SUCCESS) || !close_during_destroy(CUDA_ERROR_INVALID_HANDLE)) {
        std::cerr << "batch closed before event cleanup completed\n";
        return EXIT_FAILURE;
    }
    if (!failure_and_cleanup_paths()) {
        std::cerr << "event failure, polling, or cleanup handling failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
