#include "internal/cuda_queue_worker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
thread_local CUcontext current_context{};
thread_local unsigned int thread_native_calls{};
std::atomic<unsigned int> native_calls{};
std::atomic<CUresult> push_error{CUDA_SUCCESS};
std::atomic<CUresult> pop_error{CUDA_SUCCESS};
std::atomic<CUresult> query_error{CUDA_SUCCESS};
thread_local bool is_native_scope{};
std::atomic<unsigned int> scope_enters{};
std::atomic<unsigned int> scope_leaves{};

bool enter_scope() noexcept {
    const auto previous = is_native_scope;
    is_native_scope = true;
    ++scope_enters;
    return previous;
}
void leave_scope(bool previous) noexcept {
    is_native_scope = previous;
    ++scope_leaves;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void native_call() {
    ++thread_native_calls;
    require(current_context == kContext, "native work must execute with owner context");
    ++native_calls;
}
CUresult push(CUcontext context) {
    ++thread_native_calls;
    if (push_error != CUDA_SUCCESS) {
        return push_error;
    }
    current_context = context;
    return CUDA_SUCCESS;
}
CUresult pop(CUcontext* context) {
    ++thread_native_calls;
    *context = current_context;
    current_context = nullptr;
    return pop_error;
}
CUresult current(CUcontext* context) {
    ++thread_native_calls;
    ++native_calls;
    *context = current_context;
    return CUDA_SUCCESS;
}
CUresult create(CUevent* event, unsigned int) {
    native_call();
    *event = reinterpret_cast<CUevent>(new int{0});
    return CUDA_SUCCESS;
}
CUresult record(CUevent, CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult query(CUevent) {
    native_call();
    return query_error;
}
CUresult destroy(CUevent event) {
    native_call();
    delete reinterpret_cast<int*>(event);
    return CUDA_SUCCESS;
}
CUresult drain(CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, CUstream, void**, void**) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult host_to_device(CUdeviceptr, const void*, std::size_t, CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult device_to_host(void*, CUdeviceptr, std::size_t, CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult device_to_device(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult free_async(CUdeviceptr, CUstream) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult wait_event(CUstream, CUevent, unsigned int) {
    native_call();
    return CUDA_SUCCESS;
}
CUresult synchronize_event(CUevent) {
    native_call();
    return CUDA_SUCCESS;
}
CudaQueueWorkerDispatch dispatch() {
    return {{current, launch, host_to_device, device_to_host, device_to_device, free_async, create,
             record, query, destroy, drain, wait_event, synchronize_event},
            push,
            pop};
}
class Coordinator final : public CudaQueueCoordinator {
   public:
    std::atomic<unsigned int> steps{};
    std::atomic<unsigned int> stops{};
    std::atomic<CUresult> stop_result{CUDA_ERROR_NOT_READY};
    std::atomic<bool> has_control_work{};
    CUresult native_boundary_status() const noexcept override {
        return has_control_work ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    CUresult progress(CudaOperationQueue& queue) noexcept override {
        ++steps;
        return queue.pump();
    }
    void stopped(CudaOperationQueue&, CUresult result) noexcept override {
        stop_result = result;
        ++stops;
    }
};
void wait_for(const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(predicate(), "background action must finish within test deadline");
}

void signaled_progress_and_command_delivery() {
    auto coordinator = std::make_shared<Coordinator>();
    auto signal = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate::ProgressSignal>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator,
                           {.poll_interval = std::chrono::seconds{10}, .progress_signal = signal});
    wait_for([&] { return coordinator->steps.load() != 0; });
    // No command or timer can advance the next scan; an early hint must persist
    // across the owner's predicate-to-wait interval.
    const auto before = coordinator->steps.load();
    signal->notify();
    bool was_awakened{};
    try {
        wait_for([&] { return coordinator->steps.load() > before; });
        was_awakened = true;
    } catch (...) {
        // Stop before asserting so the old timer-only worker cannot delay unwind.
    }
    unsigned int accepted{};
    const auto delivered = worker.invoke([&](CudaOperationQueue&) {
        ++accepted;
        return CUDA_SUCCESS;
    });
    const auto stopped = worker.shutdown();
    require(was_awakened && delivered == CUDA_SUCCESS && accepted == 1 && stopped == CUDA_SUCCESS &&
                coordinator->stops == 1,
            "CPU hints, accepted commands and closure wake the owner independently of its timer");
}

void submillisecond_polling() {
    require(CudaQueueWorkerOptions{}.poll_interval == std::chrono::milliseconds{1},
            "generic workers retain their previous default interval");
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator,
                           {.poll_interval = std::chrono::microseconds{100}});
    wait_for([&] { return coordinator->steps.load() >= 2; });
    require(worker.shutdown() == CUDA_SUCCESS && coordinator->stops == 1,
            "positive submillisecond timer progresses and shuts down normally");
}

void checkpoint_failure_retains_ownership() {
    for (bool fails_activation : {false, true}) {
        const auto child = ::fork();
        require(child >= 0, "fork isolated uncertain checkpoint owner");
        if (child == 0) {
            try {
                auto coordinator = std::make_shared<Coordinator>();
                auto worker = std::make_unique<CudaQueueWorker>(kContext, dispatch(), coordinator);
                auto resources = std::make_shared<int>(1);
                const std::weak_ptr<int> retained = resources;
                require(worker->invoke([&](auto& queue) {
                    return queue.bind_event(reinterpret_cast<CUevent>(0x3000), resources)
                               ? CUDA_SUCCESS
                               : CUDA_ERROR_UNKNOWN;
                }) == CUDA_SUCCESS,
                        "bind uncertain checkpoint lifetime");
                resources.reset();
                CudaWorkerCheckpoint checkpoint;
                if (fails_activation) {
                    require(worker->checkpoint(checkpoint) == CUDA_SUCCESS,
                            "establish checkpoint before activation failure");
                    current_context = kContext;
                    push_error = CUDA_ERROR_LAUNCH_FAILED;
                    require(worker->resume_checkpoint(checkpoint) == CUDA_ERROR_LAUNCH_FAILED &&
                                worker->status() == CUDA_ERROR_LAUNCH_FAILED,
                            "failed native activation preserves exact error and joins");
                } else {
                    pop_error = CUDA_ERROR_INVALID_CONTEXT;
                    require(worker->checkpoint(checkpoint) == CUDA_ERROR_INVALID_CONTEXT &&
                                !checkpoint && worker->status() == CUDA_ERROR_INVALID_CONTEXT,
                            "failed native restoration publishes no successful checkpoint");
                }
                const auto calls = native_calls.load();
                require(worker->resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                            native_calls == calls && worker->shutdown() == CUDA_ERROR_NOT_READY,
                        "native failure cannot automatically restart or issue cleanup");
                worker.reset();
                require(!retained.expired() && native_calls == calls && coordinator->stops == 0,
                        "uncertain checkpoint destruction retains resources without CUDA retry");
                // Intentional uncertainty quarantine, not ordinary leak-free cleanup.
                ::_exit(EXIT_SUCCESS);
            } catch (...) {
                ::_exit(EXIT_FAILURE);
            }
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
                "uncertain checkpoint fixture must verify retained ownership before process exit");
    }
}

void checkpoint_preserves_bindings() {
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator);
    OrderedStreamToken stream;
    CudaQueueEventToken event;
    CudaOperationToken recorded;
    require(worker.invoke([&](CudaOperationQueue& queue) {
        auto resources = std::make_shared<int>(1);
        stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                   OrderedStreamKind::kNonBlocking, resources);
        event = queue.bind_event(reinterpret_cast<CUevent>(0x3000), resources);
        recorded = queue.submit(stream, CudaQueuedEventRecord{event});
        return recorded ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "accept event on checkpoint queue");
    wait_for([&] {
        return worker.invoke([&](auto& queue) { return queue.query(recorded); }) == CUDA_SUCCESS;
    });
    CudaWorkerCheckpoint stale;
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
        coordinator->has_control_work = true;
        CudaWorkerCheckpoint checkpoint;
        require(worker.checkpoint(checkpoint) == CUDA_ERROR_NOT_READY && !checkpoint,
                "empty work does not checkpoint before terminal control acknowledgment");
        coordinator->has_control_work = false;
        wait_for([&] { return checkpoint || worker.checkpoint(checkpoint) == CUDA_SUCCESS; });
        require(checkpoint && worker.owns_checkpoint(checkpoint) && coordinator->stops == 0 &&
                    worker.shutdown() == CUDA_ERROR_NOT_READY &&
                    worker.invoke([](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_DEINITIALIZED,
                "joined checkpoint preserves queue and rejects cleanup or new commands");
        require(worker.checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
                    worker.resume_checkpoint({}) == CUDA_ERROR_INVALID_VALUE,
                "occupied and empty checkpoint values reject without ownership changes");
        if (stale) {
            require(worker.resume_checkpoint(stale) == CUDA_ERROR_INVALID_VALUE,
                    "older checkpoint generation cannot resume a fresh handoff");
            const auto before_retire = native_calls.load();
            require(
                worker.retire_checkpoint_after_context_destroy(stale) == CUDA_ERROR_INVALID_VALUE &&
                    worker.owns_checkpoint(checkpoint) && native_calls == before_retire,
                "stale generation cannot retire a newer joined owner or issue CUDA");
        }
        const auto parked_steps = coordinator->steps.load();
        const auto parked_calls = native_calls.load();
        require(worker.resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_CONTEXT &&
                    worker.owns_checkpoint(checkpoint) && coordinator->steps == parked_steps &&
                    native_calls == parked_calls + 1,
                "wrong caller rejects before activation without queue polling");
        current_context = kContext;
        require(worker.resume_checkpoint(checkpoint) == CUDA_SUCCESS &&
                    !worker.owns_checkpoint(checkpoint) &&
                    worker.resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE,
                "trusted bound caller resumes once and consumes exact generation");
        current_context = nullptr;
        require(worker.invoke([&](auto& queue) {
            require(queue.query_event(event) == CUDA_SUCCESS &&
                        queue.query(recorded) == CUDA_SUCCESS &&
                        queue.event_snapshot(event) == std::optional{recorded},
                    "new worker preserves completed event and immutable prefix identity");
            recorded = queue.submit(stream, CudaQueuedEventRecord{event});
            return recorded ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        }) == CUDA_SUCCESS,
                "same stream/event bindings accept a fresh record after handoff");
        wait_for([&] {
            return worker.invoke([&](auto& queue) { return queue.query(recorded); }) ==
                   CUDA_SUCCESS;
        });
        stale = checkpoint;
    }
    CudaQueueWorker foreign(kContext, dispatch(), std::make_shared<Coordinator>());
    require(foreign.resume_checkpoint(stale) == CUDA_ERROR_INVALID_VALUE,
            "foreign worker cannot consume original checkpoint identity");
    require(foreign.shutdown() == CUDA_SUCCESS && worker.shutdown() == CUDA_SUCCESS &&
                coordinator->stops == 1,
            "ordinary shutdown retires preserved queue and acknowledges stop exactly once");
}

void checkpoint_retirement_is_cuda_free() {
    auto coordinator = std::make_shared<Coordinator>();
    auto worker = std::make_unique<CudaQueueWorker>(kContext, dispatch(), coordinator);
    // The ordinary worker is still polling. Observe only this caller until a
    // successful checkpoint independently proves global native quiescence.
    const auto before = thread_native_calls;
    require(worker->retire_checkpoint_after_context_destroy({}) == CUDA_ERROR_INVALID_VALUE &&
                thread_native_calls == before && coordinator->stops == 0,
            "unproven retirement rejects without native work or stop acknowledgment");
    CudaWorkerCheckpoint checkpoint;
    wait_for([&] { return checkpoint || worker->checkpoint(checkpoint) == CUDA_SUCCESS; });
    {
        CudaQueueWorker foreign(kContext, dispatch(), std::make_shared<Coordinator>());
        CudaWorkerCheckpoint foreign_checkpoint;
        wait_for([&] {
            return foreign_checkpoint || foreign.checkpoint(foreign_checkpoint) == CUDA_SUCCESS;
        });
        const auto foreign_parked = native_calls.load();
        require(worker->retire_checkpoint_after_context_destroy(foreign_checkpoint) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    foreign.retire_checkpoint_after_context_destroy(checkpoint) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    worker->owns_checkpoint(checkpoint) &&
                    foreign.owns_checkpoint(foreign_checkpoint) && native_calls == foreign_parked,
                "foreign joined identities cannot exchange destruction proof or issue CUDA");
        require(
            foreign.retire_checkpoint_after_context_destroy(foreign_checkpoint) == CUDA_SUCCESS &&
                native_calls == foreign_parked,
            "foreign owner retires only its own exact checkpoint without CUDA");
    }
    const auto parked = native_calls.load();
    coordinator->has_control_work = true;
    require(worker->retire_checkpoint_after_context_destroy(checkpoint) == CUDA_ERROR_NOT_READY &&
                worker->owns_checkpoint(checkpoint) && native_calls == parked &&
                coordinator->stops == 0,
            "retirement cannot bypass terminal control ownership");
    coordinator->has_control_work = false;
    require(worker->retire_checkpoint_after_context_destroy(checkpoint) == CUDA_SUCCESS &&
                !worker->owns_checkpoint(checkpoint) && native_calls == parked &&
                coordinator->stops == 1 && coordinator->stop_result == CUDA_SUCCESS,
            "independently proven destruction consumes joined CPU ownership exactly once");
    require(
        worker->retire_checkpoint_after_context_destroy(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            worker->resume_checkpoint(checkpoint) == CUDA_ERROR_INVALID_VALUE &&
            worker->shutdown() == CUDA_SUCCESS && native_calls == parked && coordinator->stops == 1,
        "retired checkpoint never reactivates or retries native cleanup");
    worker.reset();
    require(native_calls == parked && coordinator->stops == 1,
            "retired queue and worker destructors perform no destroyed-context CUDA work");
}

void autonomous_progress() {
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator);
    require(worker.status() == CUDA_SUCCESS, "worker must initialize");
    const auto caller = std::this_thread::get_id();
    CudaOperationToken token;
    require(worker.invoke([&](CudaOperationQueue& queue) {
        require(std::this_thread::get_id() != caller, "command must move to owner thread");
        auto resources = std::make_shared<int>(1);
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                              OrderedStreamKind::kNonBlocking, resources);
        token = queue.submit(stream, CudaQueuedMarker{});
        require(static_cast<bool>(token), "marker must enqueue");
        require(worker.invoke([](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_INVALID_VALUE,
                "owner reentry must reject without deadlock");
        require(worker.shutdown() == CUDA_ERROR_INVALID_VALUE, "owner cannot join itself");
        return CUDA_SUCCESS;
    }) == CUDA_SUCCESS,
            "metadata command must return");
    wait_for([&] { return native_calls >= 3; });
    CUresult completion = CUDA_ERROR_NOT_READY;
    wait_for([&] {
        require(worker.invoke([&](auto& queue) {
            completion = queue.query(token);
            return CUDA_SUCCESS;
        }) == CUDA_SUCCESS,
                "query metadata command must succeed");
        return completion == CUDA_SUCCESS;
    });
    require(current_context == nullptr, "caller context stack must remain untouched");
    require(worker.shutdown() == CUDA_SUCCESS && coordinator->stops == 1 &&
                coordinator->stop_result == CUDA_SUCCESS,
            "worker must report actual queue drain once");
    require(worker.invoke([](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_DEINITIALIZED,
            "closed mailbox must reject");
    require(worker.shutdown() == CUDA_SUCCESS, "shutdown must be idempotent");
}

void bounded_mailbox() {
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator, {.max_queued_commands = 1});
    std::mutex mutex;
    std::condition_variable condition;
    bool is_entered = false;
    bool should_release = false;
    auto first = std::async(std::launch::async, [&] {
        return worker.invoke([&](auto&) {
            std::unique_lock lock(mutex);
            is_entered = true;
            condition.notify_all();
            condition.wait(lock, [&] { return should_release; });
            return CUDA_SUCCESS;
        });
    });
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return is_entered; });
    }
    std::atomic<unsigned int> accepted{};
    std::atomic<unsigned int> rejected{};
    std::array<std::future<CUresult>, 4> contenders;
    for (auto& contender : contenders) {
        contender = std::async(std::launch::async, [&] {
            const auto value = worker.invoke([&](auto&) {
                ++accepted;
                return CUDA_SUCCESS;
            });
            if (value == CUDA_ERROR_OUT_OF_MEMORY) {
                ++rejected;
            }
            return value;
        });
    }
    wait_for([&] { return rejected == 3; });
    {
        const std::lock_guard lock(mutex);
        should_release = true;
    }
    condition.notify_all();
    require(first.get() == CUDA_SUCCESS, "first command must complete");
    for (auto& contender : contenders) {
        const auto value = contender.get();
        require(value == CUDA_SUCCESS || value == CUDA_ERROR_OUT_OF_MEMORY,
                "bounded rejection must preserve accepted commands");
    }
    require(accepted == 1 && worker.shutdown() == CUDA_SUCCESS,
            "only one queued command may be accepted");
}

void native_worker_scope_is_balanced() {
    auto native = dispatch();
    native.enter_native_scope = enter_scope;
    native.leave_native_scope = leave_scope;
    {
        CudaQueueWorker worker(kContext, native, std::make_shared<Coordinator>());
        require(worker.invoke([](auto&) {
            return is_native_scope ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        }) == CUDA_SUCCESS &&
                    !is_native_scope && worker.shutdown() == CUDA_SUCCESS,
                "private worker work must enter native scope without changing caller scope");
    }
    require(scope_enters == 1 && scope_leaves == 1,
            "native scope must span context push, queue work and checked teardown once");
    native.leave_native_scope = nullptr;
    CudaQueueWorker invalid(kContext, native, std::make_shared<Coordinator>());
    require(invalid.status() == CUDA_ERROR_INVALID_VALUE && scope_enters == 1 && scope_leaves == 1,
            "an unpaired native dispatch scope must reject before native entry");
}

void native_boundary_waits_for_control_acknowledgment() {
    auto coordinator = std::make_shared<Coordinator>();
    coordinator->has_control_work = true;
    CudaQueueWorker worker(kContext, dispatch(), coordinator);
    CudaNativeBoundaryToken boundary;
    require(worker.try_pause_native(boundary) == CUDA_ERROR_NOT_READY && !boundary,
            "an empty GPU queue is not proof that the global slot was acknowledged");
    coordinator->has_control_work = false;
    require(worker.try_pause_native(boundary) == CUDA_SUCCESS && boundary,
            "control acknowledgment must permit a native boundary");
    require(worker.finish_native(boundary, [](auto&) { return CUDA_SUCCESS; }) == CUDA_SUCCESS &&
                worker.shutdown() == CUDA_SUCCESS,
            "the checked boundary must return normal worker ownership");
}

void native_boundary_pause() {
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator);
    CudaNativeBoundaryToken token;
    query_error = CUDA_ERROR_NOT_READY;
    require(worker.invoke([](auto& queue) {
        auto resources = std::make_shared<int>(1);
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                              OrderedStreamKind::kNonBlocking, resources);
        return queue.submit(stream, CudaQueuedMarker{}) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "native-boundary fixture must accept logical work");
    require(worker.try_pause_native(token) == CUDA_ERROR_NOT_READY && !token,
            "pending logical work must not pause priority progress");
    query_error = CUDA_SUCCESS;
    CUresult pause_result = CUDA_ERROR_NOT_READY;
    wait_for([&] {
        if (pause_result != CUDA_SUCCESS) {
            pause_result = worker.try_pause_native(token);
        }
        return pause_result == CUDA_SUCCESS;
    });
    require(static_cast<bool>(token), "successful boundary must return ownership proof");
    const auto steps = coordinator->steps.load();
    const auto calls = native_calls.load();
    unsigned int command_calls = 0;
    for (unsigned int count = 0; count < 100; ++count) {
        require(worker.invoke([&](auto&) {
            ++command_calls;
            return CUDA_SUCCESS;
        }) == CUDA_ERROR_NOT_READY,
                "paused ordinary commands must reject without running");
    }
    CudaNativeBoundaryToken duplicate;
    require(worker.try_pause_native(duplicate) == CUDA_ERROR_NOT_READY && !duplicate &&
                worker.shutdown() == CUDA_ERROR_NOT_READY && command_calls == 0 &&
                coordinator->steps == steps && native_calls == calls,
            "paused owner must not poll CUDA, progress policy or enter teardown");
    CudaNativeBoundaryToken invalid;
    require(worker.finish_native(invalid,
                                 [&](auto&) {
                                     ++command_calls;
                                     return CUDA_SUCCESS;
                                 }) == CUDA_ERROR_INVALID_VALUE &&
                command_calls == 0,
            "invalid tokens must never execute an import");
    CudaQueueWorker foreign(kContext, dispatch(), std::make_shared<Coordinator>());
    CudaNativeBoundaryToken foreign_token;
    require(foreign.try_pause_native(foreign_token) == CUDA_SUCCESS &&
                foreign.finish_native(token, [](auto&) { return CUDA_SUCCESS; }) ==
                    CUDA_ERROR_INVALID_VALUE,
            "native tokens must be bound to one worker identity");
    require(
        foreign.finish_native(foreign_token, [](auto&) { return CUDA_SUCCESS; }) == CUDA_SUCCESS &&
            foreign.shutdown() == CUDA_SUCCESS,
        "foreign fixture must finish its own boundary");
    require(worker.finish_native(token, [](auto&) { return CUDA_ERROR_OUT_OF_MEMORY; }) ==
                    CUDA_ERROR_OUT_OF_MEMORY &&
                worker.shutdown() == CUDA_ERROR_NOT_READY,
            "failed import must leave the native boundary paused");
    require(worker.finish_native(token,
                                 [](auto&) -> CUresult {
                                     throw std::runtime_error("injected native import exception");
                                 }) == CUDA_ERROR_UNKNOWN &&
                worker.shutdown() == CUDA_ERROR_NOT_READY,
            "import exception must not trigger CUDA teardown during native capture");
    require(worker.finish_native(token, [](auto&) { return CUDA_SUCCESS; }) == CUDA_SUCCESS,
            "successful trusted import must finish the pause");
    require(
        worker.finish_native(token, [](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_INVALID_VALUE,
        "completed token must be stale");
    CudaNativeBoundaryToken next;
    pause_result = CUDA_ERROR_NOT_READY;
    wait_for([&] {
        if (pause_result != CUDA_SUCCESS) {
            pause_result = worker.try_pause_native(next);
        }
        return pause_result == CUDA_SUCCESS;
    });
    require(worker.finish_native(token, [](auto&) { return CUDA_SUCCESS; }) ==
                    CUDA_ERROR_INVALID_VALUE &&
                worker.finish_native(next, [](auto&) { return CUDA_SUCCESS; }) == CUDA_SUCCESS,
            "a later boundary must reject prior generations");
    require(worker.shutdown() == CUDA_SUCCESS, "resumed worker must support normal teardown");
}

void native_reservation_pause() {
    query_error = CUDA_ERROR_NOT_READY;
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator);
    CudaOperationToken physical;
    CudaOperationToken reservation;
    auto resources = std::make_shared<int>(1);
    require(worker.invoke([&](CudaOperationQueue& queue) {
        const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                                OrderedStreamKind::kNonBlocking, resources);
        const auto priority = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                                OrderedStreamKind::kNonBlocking, resources);
        physical = queue.submit(producer, CudaQueuedMarker{});
        reservation = queue.submit(priority, CudaQueuedNativeReservation{resources});
        return physical && reservation ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "physical-boundary fixture must submit metadata");
    wait_for([&] {
        bool is_running = false;
        require(worker.invoke([&](CudaOperationQueue& queue) {
            is_running = queue.state(physical) == OrderedOperationState::kRunning;
            return CUDA_SUCCESS;
        }) == CUDA_SUCCESS,
                "physical marker must stay queryable");
        return is_running;
    });
    CudaNativeBoundaryToken boundary;
    unsigned int grant_checks{};
    unsigned int grant_steps{};
    const auto caller = std::this_thread::get_id();
    const auto acquire_grant = [&](CudaOperationQueue&) {
        require(std::this_thread::get_id() != caller && current_context == kContext,
                "grant prerequisite must run on the bound CUDA owner");
        ++grant_checks;
        grant_steps = coordinator->steps.load();
        return CUDA_SUCCESS;
    };
    require(worker.try_pause_native_reservation(reservation, boundary, {}) ==
                    CUDA_ERROR_INVALID_VALUE &&
                !boundary,
            "missing grant prerequisite cannot authorize pause");
    for (const auto status : {CUDA_ERROR_NOT_READY, CUDA_ERROR_INVALID_HANDLE}) {
        require(worker.try_pause_native_reservation(reservation, boundary,
                                                    [status](auto&) { return status; }) == status &&
                    !boundary && worker.status() == CUDA_SUCCESS,
                "pending or failed grant preserves exact status without pausing the owner");
    }
    require(worker.try_pause_native_reservation(reservation, boundary, acquire_grant) ==
                    CUDA_ERROR_NOT_READY &&
                !boundary && grant_checks == 1,
            "successful grant check is not proof of physical completion");
    require(worker.try_pause_native_reservation(reservation, boundary) == CUDA_ERROR_NOT_READY &&
                !boundary,
            "ready reservation cannot pause unproven physical work");
    query_error = CUDA_SUCCESS;
    CUresult paused = CUDA_ERROR_NOT_READY;
    wait_for([&] {
        if (paused != CUDA_SUCCESS) {
            paused = worker.try_pause_native_reservation(reservation, boundary, acquire_grant);
        }
        return paused == CUDA_SUCCESS;
    });
    const auto calls = native_calls.load();
    const auto steps = coordinator->steps.load();
    const auto checks = grant_checks;
    CudaNativeBoundaryToken duplicate;
    require(steps == grant_steps &&
                worker.try_pause_native_reservation(reservation, duplicate, acquire_grant) ==
                    CUDA_ERROR_NOT_READY &&
                !duplicate && grant_checks == checks,
            "successful grant and pause share one owner turn; paused owner cannot recheck grant");
    require(worker.invoke([](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_NOT_READY &&
                worker.shutdown() == CUDA_ERROR_NOT_READY && native_calls == calls &&
                coordinator->steps == steps,
            "reservation pause must stop all native progress and retain worker ownership");
    require(worker.finish_native(boundary,
                                 [&](CudaOperationQueue& queue) {
                                     return queue.replace_native_reservation(
                                                reservation,
                                                {reinterpret_cast<CUevent>(0x7000), resources})
                                                ? CUDA_SUCCESS
                                                : CUDA_ERROR_UNKNOWN;
                                 }) == CUDA_SUCCESS,
            "same reservation token must become the actual completion fence");
    wait_for([&] {
        bool is_complete = false;
        require(worker.invoke([&](CudaOperationQueue& queue) {
            is_complete = queue.query(reservation) == CUDA_SUCCESS;
            return CUDA_SUCCESS;
        }) == CUDA_SUCCESS,
                "replaced native token must stay queryable");
        return is_complete;
    });
    require(worker.shutdown() == CUDA_SUCCESS, "replaced reservation must physically drain");
}

void owner_driven_native_reservation() {
    query_error = CUDA_ERROR_NOT_READY;
    auto coordinator = std::make_shared<Coordinator>();
    auto signal = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate::ProgressSignal>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator,
                           {.poll_interval = std::chrono::seconds{10}, .progress_signal = signal});
    auto resources = std::make_shared<int>(1);
    CudaOperationToken reservation;
    require(worker.invoke([&](auto& queue) {
        const auto physical = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                                OrderedStreamKind::kNonBlocking, resources);
        const auto priority = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                                OrderedStreamKind::kNonBlocking, resources);
        const auto marker = queue.submit(physical, CudaQueuedMarker{});
        reservation = queue.submit(priority, CudaQueuedNativeReservation{resources});
        return marker && reservation ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "prepare distinct physical work and native reservation");
    std::atomic<bool> has_grant{};
    std::atomic<unsigned int> checks{};
    std::atomic<bool> can_prepare_pause{};
    std::atomic<unsigned int> preparations{};
    CudaNativeBoundaryToken boundary;
    auto result = std::async(std::launch::async, [&] {
        return worker.await_native_reservation(
            reservation, boundary,
            [&](auto&) {
                require(current_context == kContext, "native handoff must run on its CUDA owner");
                ++checks;
                return has_grant ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
            },
            [&](auto&) {
                require(has_grant && query_error == CUDA_SUCCESS && current_context == kContext,
                        "frontend preparation must follow exact grant and physical proof");
                ++preparations;
                return can_prepare_pause ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
            });
    });
    wait_for([&] { return checks.load() != 0; });
    CudaNativeBoundaryToken duplicate;
    unsigned int duplicate_checks{};
    require(worker.await_native_reservation(reservation, duplicate,
                                            [&](auto&) {
                                                ++duplicate_checks;
                                                return CUDA_SUCCESS;
                                            }) == CUDA_ERROR_NOT_READY &&
                duplicate_checks == 0,
            "a duplicate cannot replace or execute beside the accepted native wait");
    for (unsigned int index = 0; index < 8; ++index) {
        const auto before = checks.load();
        signal->notify();
        wait_for([&] { return checks.load() > before; });
        require(result.wait_for(std::chrono::seconds{0}) == std::future_status::timeout,
                "unrelated notifications cannot authorize native issue");
    }
    auto before = checks.load();
    has_grant = true;
    signal->notify();
    wait_for([&] { return checks.load() > before; });
    require(preparations == 0 &&
                result.wait_for(std::chrono::seconds{0}) == std::future_status::timeout,
            "a real grant still cannot replace physical stop proof");
    query_error = CUDA_SUCCESS;
    signal->notify();
    wait_for([&] { return preparations.load() != 0; });
    require(result.wait_for(std::chrono::seconds{0}) == std::future_status::timeout,
            "busy frontend metadata must retain the wait without publishing a pause");
    can_prepare_pause = true;
    signal->notify();
    wait_for([&] { return result.wait_for(std::chrono::seconds{0}) == std::future_status::ready; });
    require(result.get() == CUDA_SUCCESS && boundary && worker.native_boundary_is_paused(boundary),
            "one producer submission must independently reach its exact physical pause");
    const auto native_before = native_calls.load();
    require(worker.await_native_reservation(reservation, duplicate,
                                            [&](auto&) {
                                                ++duplicate_checks;
                                                return CUDA_SUCCESS;
                                            }) == CUDA_ERROR_NOT_READY &&
                duplicate_checks == 0 && worker.shutdown() == CUDA_ERROR_NOT_READY &&
                native_calls == native_before,
            "a paused owner cannot accept another wait or tear down live caller work");
    require(worker.finish_native(boundary,
                                 [&](auto& queue) {
                                     return queue.replace_native_reservation(
                                                reservation,
                                                {reinterpret_cast<CUevent>(0x7000), resources})
                                                ? CUDA_SUCCESS
                                                : CUDA_ERROR_UNKNOWN;
                                 }) == CUDA_SUCCESS,
            "the exact waited reservation must retain its real trailing fence");
    require(worker.shutdown() == CUDA_SUCCESS, "handed-off native work must drain normally");
}

void native_wait_capacity_and_shutdown() {
    auto coordinator = std::make_shared<Coordinator>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator, {.max_queued_commands = 1});
    CudaOperationToken reservation;
    auto resources = std::make_shared<int>(1);
    require(worker.invoke([&](auto& queue) {
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                              OrderedStreamKind::kNonBlocking, resources);
        reservation = queue.submit(stream, CudaQueuedNativeReservation{resources});
        return reservation ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "prepare bounded native wait");
    std::atomic<unsigned int> checks{};
    CudaNativeBoundaryToken boundary;
    auto result = std::async(std::launch::async, [&] {
        return worker.await_native_reservation(reservation, boundary, [&](auto&) {
            ++checks;
            return CUDA_ERROR_NOT_READY;
        });
    });
    wait_for([&] { return checks.load() != 0; });
    CudaWorkerCheckpoint checkpoint;
    require(worker.invoke([](auto&) { return CUDA_SUCCESS; }) == CUDA_ERROR_OUT_OF_MEMORY &&
                worker.checkpoint(checkpoint) == CUDA_ERROR_OUT_OF_MEMORY && !checkpoint,
            "a retained wait counts against mailbox capacity and cannot be checkpointed away");
    require(
        worker.shutdown() == CUDA_SUCCESS && result.get() == CUDA_ERROR_DEINITIALIZED &&
            !boundary && coordinator->stops == 1,
        "shutdown must settle the accepted waiter without granting or dropping queue ownership");
}

void native_wait_rejection_and_exception() {
    for (const bool should_throw : {false, true}) {
        auto coordinator = std::make_shared<Coordinator>();
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        CudaOperationToken reservation;
        auto resources = std::make_shared<int>(1);
        require(worker.invoke([&](auto& queue) {
            const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                                  OrderedStreamKind::kNonBlocking, resources);
            reservation = queue.submit(stream, CudaQueuedNativeReservation{resources});
            return reservation ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        }) == CUDA_SUCCESS,
                "prepare rejected native wait");
        CudaNativeBoundaryToken boundary;
        require(worker.await_native_reservation({}, boundary, [](auto&) { return CUDA_SUCCESS; }) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    worker.await_native_reservation(reservation, boundary, {}) ==
                        CUDA_ERROR_INVALID_VALUE,
                "invalid native waits must reject before acceptance");
        unsigned int calls{};
        const auto expected = should_throw ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_HANDLE;
        require(worker.await_native_reservation(
                    reservation, boundary,
                    [&](auto&) -> CUresult {
                        ++calls;
                        CudaNativeBoundaryToken nested;
                        require(worker.await_native_reservation(
                                    reservation, nested, [](auto&) { return CUDA_SUCCESS; }) ==
                                    CUDA_ERROR_INVALID_VALUE,
                                "a CUDA owner cannot synchronously wait on itself");
                        if (should_throw) throw std::runtime_error("injected native prerequisite");
                        return CUDA_ERROR_INVALID_HANDLE;
                    }) == expected &&
                    calls == 1 && !boundary,
                "failed prerequisites must publish exactly once without manufacturing pause");
        require(worker.shutdown() == (should_throw ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS),
                "exception containment must retain the original worker failure");
    }
}

void native_pause_preparation_failure() {
    for (const bool should_throw : {false, true}) {
        auto coordinator = std::make_shared<Coordinator>();
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        auto resources = std::make_shared<int>(1);
        CudaOperationToken reservation;
        require(worker.invoke([&](auto& queue) {
            const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                                  OrderedStreamKind::kNonBlocking, resources);
            reservation = queue.submit(stream, CudaQueuedNativeReservation{resources});
            return reservation ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        }) == CUDA_SUCCESS,
                "prepare a reservation with a rejected frontend handoff");
        CudaNativeBoundaryToken boundary;
        unsigned int preparations{};
        const auto expected = should_throw ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_HANDLE;
        require(worker.await_native_reservation(
                    reservation, boundary, [](auto&) { return CUDA_SUCCESS; },
                    [&](auto&) -> CUresult {
                        ++preparations;
                        if (should_throw) throw std::runtime_error("injected pause preparation");
                        return CUDA_ERROR_INVALID_HANDLE;
                    }) == expected &&
                    preparations == 1 && !boundary,
                "failed frontend preparation must settle once without manufacturing pause");
        require(worker.shutdown() == (should_throw ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS),
                "frontend preparation exceptions must use normal worker containment");
    }
}

void native_waiters_do_not_block_ready_successors() {
    auto coordinator = std::make_shared<Coordinator>();
    auto signal = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate::ProgressSignal>();
    CudaQueueWorker worker(kContext, dispatch(), coordinator,
                           {.poll_interval = std::chrono::seconds{10}, .progress_signal = signal});
    auto resources = std::make_shared<int>(1);
    std::array<CudaOperationToken, 2> reservations;
    const std::array streams{reinterpret_cast<CUstream>(0x2000),
                             reinterpret_cast<CUstream>(0x2001)};
    require(worker.invoke([&](auto& queue) {
        for (unsigned int index = 0; index < reservations.size(); ++index) {
            const auto stream =
                queue.bind_stream(streams[index], OrderedStreamKind::kNonBlocking, resources);
            reservations[index] = queue.submit(stream, CudaQueuedNativeReservation{resources});
        }
        return reservations[0] && reservations[1] ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "prepare two independent reservation waiters");
    std::atomic<bool> has_first_grant{};
    std::atomic<unsigned int> first_checks{};
    std::atomic<unsigned int> second_checks{};
    CudaNativeBoundaryToken first_boundary;
    auto first = std::async(std::launch::async, [&] {
        return worker.await_native_reservation(reservations[0], first_boundary, [&](auto&) {
            ++first_checks;
            return has_first_grant ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
        });
    });
    wait_for([&] { return first_checks.load() != 0; });
    CudaNativeBoundaryToken second_boundary;
    const auto second =
        worker.await_native_reservation(reservations[1], second_boundary, [&](auto&) {
            ++second_checks;
            return CUDA_SUCCESS;
        });
    const bool first_is_pending =
        first.wait_for(std::chrono::seconds{0}) == std::future_status::timeout;
    if (second == CUDA_SUCCESS) {
        require(worker.finish_native(second_boundary,
                                     [&](auto& queue) {
                                         return queue.replace_native_reservation(
                                                    reservations[1],
                                                    {reinterpret_cast<CUevent>(0x7000), resources})
                                                    ? CUDA_SUCCESS
                                                    : CUDA_ERROR_UNKNOWN;
                                     }) == CUDA_SUCCESS,
                "complete the independently granted successor before releasing the first grant");
    }
    has_first_grant = true;
    signal->notify();
    require(first.get() == CUDA_SUCCESS && first_boundary,
            "the previously pending waiter must remain live after successor completion");
    require(worker.finish_native(first_boundary,
                                 [&](auto& queue) {
                                     return queue.replace_native_reservation(
                                                reservations[0],
                                                {reinterpret_cast<CUevent>(0x7001), resources})
                                                ? CUDA_SUCCESS
                                                : CUDA_ERROR_UNKNOWN;
                                 }) == CUDA_SUCCESS &&
                worker.shutdown() == CUDA_SUCCESS,
            "both original reservations must retain and drain their own fences");
    require(second == CUDA_SUCCESS && second_checks == 1 && first_is_pending,
            "a pending reservation must not block a separately granted successor");
}

void failure_paths() {
    push_error = CUDA_ERROR_INVALID_CONTEXT;
    auto coordinator = std::make_shared<Coordinator>();
    {
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        require(worker.status() == CUDA_ERROR_INVALID_CONTEXT && worker.invoke([](auto&) {
            return CUDA_SUCCESS;
        }) == CUDA_ERROR_INVALID_CONTEXT &&
                    worker.shutdown() == CUDA_ERROR_INVALID_CONTEXT && coordinator->stops == 0,
                "activation failure must preserve error and never run queue");
    }
    push_error = CUDA_SUCCESS;
    {
        CudaQueueWorker worker(kContext, dispatch(), coordinator, {.max_queued_commands = 0});
        require(worker.shutdown() == CUDA_ERROR_INVALID_VALUE, "invalid options must reject");
    }
    for (const auto interval : {std::chrono::microseconds{0}, std::chrono::microseconds{-1}}) {
        const auto before = native_calls.load();
        CudaQueueWorker worker(kContext, dispatch(), coordinator, {.poll_interval = interval});
        require(worker.status() == CUDA_ERROR_INVALID_VALUE &&
                    worker.shutdown() == CUDA_ERROR_INVALID_VALUE && native_calls == before,
                "nonpositive timer rejects before native context or queue initialization");
    }
    {
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        require(worker.invoke([](auto&) -> CUresult { throw std::runtime_error("injected"); }) ==
                        CUDA_ERROR_UNKNOWN &&
                    worker.shutdown() == CUDA_ERROR_UNKNOWN,
                "command exception must contain failure and drain");
    }
    pop_error = CUDA_ERROR_INVALID_CONTEXT;
    {
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        require(worker.shutdown() == CUDA_ERROR_INVALID_CONTEXT,
                "context restoration error must remain visible");
    }
    pop_error = CUDA_SUCCESS;
    query_error = CUDA_ERROR_LAUNCH_FAILED;
    {
        CudaQueueWorker worker(kContext, dispatch(), coordinator);
        require(worker.invoke([](auto& queue) {
            auto resources = std::make_shared<int>(1);
            const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x2000),
                                                  OrderedStreamKind::kNonBlocking, resources);
            return queue.submit(stream, CudaQueuedMarker{}) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        }) == CUDA_SUCCESS,
                "failure fixture must accept marker");
        wait_for([&] { return worker.status() == CUDA_ERROR_LAUNCH_FAILED; });
        require(worker.shutdown() == CUDA_ERROR_LAUNCH_FAILED,
                "autonomous polling failure must preserve original error");
    }
    query_error = CUDA_SUCCESS;
}
}  // namespace

int main() {
    try {
        native_waiters_do_not_block_ready_successors();
        checkpoint_failure_retains_ownership();
        autonomous_progress();
        signaled_progress_and_command_delivery();
        submillisecond_polling();
        checkpoint_preserves_bindings();
        checkpoint_retirement_is_cuda_free();
        bounded_mailbox();
        native_worker_scope_is_balanced();
        native_boundary_waits_for_control_acknowledgment();
        native_boundary_pause();
        native_reservation_pause();
        owner_driven_native_reservation();
        native_wait_capacity_and_shutdown();
        native_wait_rejection_and_exception();
        native_pause_preparation_failure();
        failure_paths();
        std::cout << "Context-owned worker checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
