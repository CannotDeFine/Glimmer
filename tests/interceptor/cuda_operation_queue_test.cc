#include "internal/cuda_operation_queue.h"
#include "internal/cuda_admission_coordinator.h"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace glimmer::interceptor;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct FakeEvent {
    bool is_ready{};
    bool is_user_event{};
};
struct Fake {
    static inline Fake* active{};
    CUcontext context{reinterpret_cast<CUcontext>(0x1000)};
    std::vector<std::unique_ptr<FakeEvent>> events;
    std::vector<char> issued;
    std::vector<CUstream> records;
    std::vector<CUevent> recorded_events;
    CUresult create_error{CUDA_SUCCESS};
    CUresult issue_error{CUDA_SUCCESS};
    CUresult record_error{CUDA_SUCCESS};
    CUresult query_error{CUDA_SUCCESS};
    CUresult destroy_error{CUDA_SUCCESS};
    CUresult drain_error{CUDA_SUCCESS};
    CUresult wait_error{CUDA_SUCCESS};
    CUresult prefix_drain_error{CUDA_SUCCESS};
    bool should_publish_create_error{};
    bool should_return_null_event{};
    bool should_release_on_destroy_error{};
    int destroy_calls{};
    int prefix_drains{};
    unsigned int argument_value{};
    int drains{};
    int public_records{};
    CUevent waited_event{};
    std::chrono::steady_clock::time_point now{};
    static std::chrono::steady_clock::time_point clock_now() noexcept {
        return active->now;
    }
    Fake() {
        require(active == nullptr, "fake dispatch must have one owner");
        active = this;
    }
    ~Fake() {
        active = nullptr;
    }
    void ready() {
        for (auto& event : events) {
            event->is_ready = true;
        }
    }
    static CUresult current(CUcontext* context) {
        *context = active->context;
        return CUDA_SUCCESS;
    }
    static CUresult create(CUevent* event, unsigned int flags) {
        require(flags == CU_EVENT_DISABLE_TIMING, "queue events must not collect timing");
        if (active->create_error != CUDA_SUCCESS && !active->should_publish_create_error) {
            return active->create_error;
        }
        if (active->should_return_null_event) {
            *event = nullptr;
            return CUDA_SUCCESS;
        }
        auto owned = std::make_unique<FakeEvent>();
        *event = reinterpret_cast<CUevent>(owned.get());
        active->events.push_back(std::move(owned));
        return active->create_error;
    }
    static CUresult record(CUevent event, CUstream stream) {
        active->records.push_back(stream);
        active->recorded_events.push_back(event);
        for (const auto& owned : active->events) {
            if (reinterpret_cast<CUevent>(owned.get()) == event && owned->is_user_event) {
                ++active->public_records;
            }
        }
        return active->record_error;
    }
    static CUresult query(CUevent event) {
        if (active->query_error != CUDA_SUCCESS) {
            return active->query_error;
        }
        return reinterpret_cast<FakeEvent*>(event)->is_ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    static CUresult destroy(CUevent event) {
        ++active->destroy_calls;
        if (active->destroy_error != CUDA_SUCCESS && !active->should_release_on_destroy_error) {
            return active->destroy_error;
        }
        require(std::erase_if(active->events,
                              [&](const auto& owned) {
                                  return owned.get() == reinterpret_cast<FakeEvent*>(event);
                              }) == 1,
                "owned event must be destroyed exactly once");
        return active->destroy_error;
    }
    static CUresult drain(CUstream) {
        ++active->drains;
        return active->drain_error;
    }
    static CUresult launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int, CUstream, void** arguments,
                           void**) {
        active->issued.push_back('k');
        active->argument_value = *static_cast<unsigned int*>(arguments[0]);
        return active->issue_error;
    }
    static CUresult host_to_device(CUdeviceptr, const void*, std::size_t, CUstream) {
        active->issued.push_back('h');
        return active->issue_error;
    }
    static CUresult device_to_host(void*, CUdeviceptr, std::size_t, CUstream) {
        active->issued.push_back('d');
        return active->issue_error;
    }
    static CUresult device_to_device(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
        active->issued.push_back('c');
        return active->issue_error;
    }
    static CUresult free(CUdeviceptr, CUstream) {
        active->issued.push_back('f');
        return active->issue_error;
    }
    static CUresult wait(CUstream, CUevent event, unsigned int flags) {
        require(flags == 0, "native fence import must retain default wait flags");
        active->issued.push_back('w');
        active->waited_event = event;
        return active->wait_error;
    }
    static CudaQueueDispatch dispatch() {
        return {current, launch, host_to_device,   device_to_host, device_to_device,
                free,    create, record,           query,          destroy,
                drain,   wait,   synchronize_event};
    }
    static CUresult synchronize_event(CUevent event) {
        ++active->prefix_drains;
        if (active->prefix_drain_error == CUDA_SUCCESS) {
            reinterpret_cast<FakeEvent*>(event)->is_ready = true;
        }
        return active->prefix_drain_error;
    }
};

struct Continuation final : CudaQueueContinuation {
    bool is_complete{};
    int preparations{};
    int inspections{};
    int requests{};
    CUresult preparation_error{CUDA_SUCCESS};
    CUresult staging_error{CUDA_SUCCESS};
    CUresult request_error{CUDA_SUCCESS};
    CUresult inspection_error{CUDA_SUCCESS};
    CUresult prepare_launch(std::array<unsigned int, 3>& grid,
                            OwnedKernelArguments&) noexcept override {
        ++preparations;
        grid = {3, 1, 1};
        return preparation_error;
    }
    CUresult request_retreat() noexcept override {
        ++requests;
        return request_error;
    }
    CUresult stage_completion(CUstream) noexcept override {
        return staging_error;
    }
    CUresult inspect_stopped(bool& complete) noexcept override {
        ++inspections;
        complete = is_complete;
        return inspection_error;
    }
};
struct ReleaseObserver final : CudaQueueReleaseObserver {
    int confirmations{};
    int context_destructions{};
    void confirmed_release(CUdeviceptr) noexcept override {
        ++confirmations;
    }
    void context_destroyed(CUdeviceptr) noexcept override {
        ++context_destructions;
    }
};
struct RecordGuard final : CudaQueueEventRecordGuard {
    bool is_current{true};
    CUresult result{CUDA_SUCCESS};
    int calls{};
    CUresult should_write(bool& output) noexcept override {
        ++calls;
        output = is_current;
        return result;
    }
};

CudaQueuedKernel kernel(std::shared_ptr<void> resources,
                        std::shared_ptr<Continuation> continuation = {}) {
    unsigned int value = 42;
    std::array<void*, 1> addresses{&value};
    const std::array<PtxParameterLayout, 1> layout{{{4, 4}}};
    auto owned = OwnedKernelArguments::copy(layout, addresses.data());
    require(owned.has_value(), "argument storage must be copied");
    value = 99;
    return {reinterpret_cast<CUfunction>(0x2000),
            {105, 1, 1},
            {64, 1, 1},
            0,
            std::move(*owned),
            std::move(resources),
            std::move(continuation)};
}
void step(Fake& fake, CudaOperationQueue& queue) {
    fake.ready();
    require(queue.pump() == CUDA_SUCCESS, "physical completion pump must succeed");
}

void continuation_and_owned_operations() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    std::weak_ptr<void> retained = resources;
    auto progress = std::make_shared<Continuation>();
    auto release_observer = std::make_shared<ReleaseObserver>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto consumer = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto work = queue.submit(producer, kernel(resources, progress));
    const auto record = queue.submit(producer, CudaQueuedMarker{});
    const std::array dependency{record};
    const auto wait = queue.submit(consumer, CudaQueuedMarker{}, dependency);
    const auto copy =
        queue.submit(consumer, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToDevice, 0x6000, 0x5000,
                                              nullptr, 16, resources});
    const auto release =
        queue.submit(consumer, CudaQueuedFree{0x5000, resources, release_observer});
    resources.reset();
    require(work && record && wait && copy && release && !retained.expired(),
            "all payload and stream resources must stay owned");
    require(queue.pump() == CUDA_SUCCESS && fake.argument_value == 42 &&
                fake.issued == std::vector<char>{'k'},
            "launch must use owned bytes and defer dependent CUDA operations");
    require(
        queue.pump() == CUDA_SUCCESS && progress->inspections == 0 && !queue.cancel_stopped(work),
        "NOT_READY and cancellation cannot publish physical stop");
    for (int iteration = 0; iteration < 4; ++iteration) {
        require(queue.request_retreat(work) == CUDA_SUCCESS, "running continuation can retreat");
        step(fake, queue);
        require(queue.state(work) == OrderedOperationState::kSuspended &&
                    queue.query(record) == CUDA_ERROR_NOT_READY &&
                    queue.query(wait) == CUDA_ERROR_NOT_READY && queue.pending_count() == 5 &&
                    fake.issued.size() == static_cast<std::size_t>(iteration) + 1,
                "physical retreat must not unblock record/wait/copy/free");
        require(queue.resume(work) && !queue.resume(work), "resume must occur exactly once");
        require(queue.pump() == CUDA_SUCCESS, "resumed generation must launch");
    }
    progress->is_complete = true;
    step(fake, queue);
    for (int iteration = 0; iteration < 4; ++iteration) {
        step(fake, queue);
    }
    require(queue.pending_count() == 0 && queue.query(release) == CUDA_SUCCESS &&
                release_observer->confirmations == 1 &&
                fake.issued == std::vector<char>({'k', 'k', 'k', 'k', 'k', 'c', 'f'}),
            "final completion must release exactly one copy and free");
    require(queue.shutdown() == CUDA_SUCCESS && retained.expired() && fake.events.empty(),
            "shutdown must release terminal payload and stream ownership");
}

void implicit_completion_snapshots_retain_default_domains() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto blocking = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kBlocking, resources);
    const auto thread = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                          OrderedStreamKind::kPerThreadDefault, resources);
    const auto independent = queue.bind_stream(reinterpret_cast<CUstream>(0x5000),
                                               OrderedStreamKind::kNonBlocking, resources);
    const auto legacy = queue.bind_stream(nullptr, OrderedStreamKind::kLegacyDefault, resources);
    const auto first = queue.submit(blocking, CudaQueuedMarker{});
    const auto second = queue.submit(thread, CudaQueuedMarker{});
    const auto excluded = queue.submit(independent, CudaQueuedMarker{});
    require(first && second && excluded && queue.close_stream(blocking),
            "retain a closed pending blocking tail");
    const auto snapshot = queue.implicit_completion_snapshot(true);
    require(snapshot.size() == 2 && std::ranges::find(snapshot, first) != snapshot.end() &&
                std::ranges::find(snapshot, second) != snapshot.end() &&
                queue.implicit_completion_snapshot(false).empty() &&
                !queue.completion_snapshot(blocking),
            "legacy domain includes closed blocking and PTDS but excludes nonblocking work");
    const auto barrier = queue.submit(legacy, CudaQueuedMarker{});
    const auto thread_snapshot = queue.completion_snapshot(thread);
    const auto independent_snapshot = queue.completion_snapshot(independent);
    require(barrier && thread_snapshot && thread_snapshot->size() == 2 &&
                std::ranges::find(*thread_snapshot, barrier) != thread_snapshot->end() &&
                independent_snapshot && independent_snapshot->size() == 1 &&
                independent_snapshot->front() == excluded && fake.events.empty() &&
                queue.query_native_boundary(snapshot) == CUDA_ERROR_NOT_READY,
            "blocking snapshots observe legacy tail without creating or polling a native fence");
    for (unsigned int iteration = 0; iteration < 10; ++iteration) {
        require(queue.pump() == CUDA_SUCCESS, "pump immutable completion domains");
        step(fake, queue);
    }
    require(queue.query(barrier) == CUDA_SUCCESS &&
                queue.query_native_boundary(snapshot) == CUDA_SUCCESS &&
                queue.shutdown() == CUDA_SUCCESS && fake.events.empty(),
            "captured default domain completes after exact physical prefixes and cleans up");
}

void immutable_snapshots_and_default_order() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    auto progress = std::make_shared<Continuation>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto consumer = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    auto record = queue.submit(producer, CudaQueuedMarker{});
    const auto old_record = record;
    const std::array dependency{record};
    const auto old_wait = queue.submit(consumer, CudaQueuedMarker{}, dependency);
    const auto snapshot = queue.context_snapshot();
    const auto later = queue.submit(producer, kernel(resources, progress));
    record = queue.submit(producer, CudaQueuedMarker{});
    const std::array next_dependency{record};
    const auto next_wait = queue.submit(consumer, CudaQueuedMarker{}, next_dependency);
    require(queue.pump() == CUDA_SUCCESS, "old record must start");
    step(fake, queue);
    step(fake, queue);
    require(queue.query(old_record) == CUDA_SUCCESS && queue.query(old_wait) == CUDA_SUCCESS &&
                queue.state(later) == OrderedOperationState::kSuspended &&
                queue.query(next_wait) == CUDA_ERROR_NOT_READY &&
                queue.stream_snapshot(producer) == record,
            "re-recording must not retarget an earlier wait");
    for (const auto& token : snapshot) {
        require(queue.query(token) == CUDA_SUCCESS, "context snapshot excludes later work");
    }
    require(queue.cancel_stopped(later) && queue.query(record) == CUDA_ERROR_UNKNOWN &&
                queue.query(next_wait) == CUDA_ERROR_UNKNOWN,
            "stopped cancellation must fail, never issue, dependent operations");

    const auto legacy = queue.bind_stream(nullptr, OrderedStreamKind::kLegacyDefault, resources);
    const auto blocking = queue.bind_stream(reinterpret_cast<CUstream>(0x7000),
                                            OrderedStreamKind::kBlocking, resources);
    const auto per_thread =
        queue.bind_stream(CU_STREAM_PER_THREAD, OrderedStreamKind::kPerThreadDefault, resources);
    const auto a = queue.submit(blocking, CudaQueuedMarker{});
    const auto b = queue.submit(per_thread, CudaQueuedMarker{});
    const auto c = queue.submit(legacy, CudaQueuedMarker{});
    const auto d = queue.submit(blocking, CudaQueuedMarker{});
    require(queue.pump() == CUDA_SUCCESS && queue.state(a) == OrderedOperationState::kRunning &&
                queue.state(b) == OrderedOperationState::kRunning &&
                queue.state(c) == OrderedOperationState::kQueued,
            "legacy must await both blocking and per-thread default work");
    step(fake, queue);
    require(queue.state(c) == OrderedOperationState::kRunning &&
                queue.state(d) == OrderedOperationState::kQueued,
            "later blocking work must await legacy completion");
    step(fake, queue);
    step(fake, queue);
    require(queue.query(d) == CUDA_SUCCESS && queue.shutdown() == CUDA_SUCCESS,
            "default dependencies must drain");
}

void failures_retain_physical_ownership() {
    for (int stage = 0; stage < 5; ++stage) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto work = queue.submit(stream, kernel(resources));
        const auto dependent = queue.submit(stream, CudaQueuedMarker{});
        const auto error = CUDA_ERROR_LAUNCH_FAILED;
        if (stage == 0) {
            fake.create_error = error;
        } else if (stage == 1) {
            fake.issue_error = error;
        } else if (stage == 2) {
            fake.record_error = error;
        }
        const auto initial = queue.pump();
        if (stage >= 3) {
            require(initial == CUDA_SUCCESS, "pre-fault issue must succeed");
            if (stage == 3) {
                fake.query_error = error;
            } else {
                fake.ready();
                fake.destroy_error = error;
            }
            require(queue.pump() == error, "query/destroy fault must remain visible");
        } else {
            require(initial == (stage == 0 ? CUDA_SUCCESS : error),
                    "event preparation errors are stopped; issue errors poison queue");
        }
        require(queue.query(work) == error && queue.query(dependent) != CUDA_SUCCESS,
                "native error and logical dependency must be retained");
        if (stage != 0 && stage != 4) {
            fake.drain_error = CUDA_ERROR_UNKNOWN;
            require(queue.shutdown() == CUDA_ERROR_UNKNOWN && queue.pending_count() == 2 &&
                        queue.state(work) == OrderedOperationState::kRunning,
                    "failed drain cannot free physical execution ownership");
        }
        fake.drain_error = CUDA_SUCCESS;
        fake.destroy_error = CUDA_SUCCESS;
        if (stage == 4) {
            require(queue.shutdown() == error && fake.destroy_calls == 1,
                    "uncertain destruction must not be retried after resetting a fake error");
            fake.events.clear();  // Simulate actual native context destruction.
            fake.context = nullptr;
            queue.retire_after_context_destroy();
            require(queue.shutdown() == CUDA_SUCCESS && queue.query(work) == error &&
                        queue.query(dependent) == CUDA_ERROR_CONTEXT_IS_DESTROYED,
                    "only trusted context destruction retires uncertain event ownership");
            continue;
        }
        require(queue.shutdown() == CUDA_SUCCESS && queue.query(work) == error &&
                    queue.query(dependent) == CUDA_ERROR_UNKNOWN && fake.events.empty() &&
                    fake.issued.size() <= 1,
                "confirmed drain must fail dependents without retrying the kernel");
    }
}

void uncertain_event_cleanup_never_replays() {
    for (const bool should_fail_creation : {false, true}) {
        Fake fake;
        auto resources = std::make_shared<int>(1);
        const std::weak_ptr<void> retained = resources;
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto token = queue.submit(stream, kernel(resources));
        resources.reset();
        require(queue.close_stream(stream), "uncertain cleanup must retain a closed owner");
        if (should_fail_creation) {
            fake.create_error = CUDA_ERROR_OUT_OF_MEMORY;
            fake.should_publish_create_error = true;
            require(queue.pump() == CUDA_ERROR_OUT_OF_MEMORY && fake.issued.empty() &&
                        !fake.events.empty(),
                    "failed create publishing a handle must retain it without GPU issue");
        } else {
            require(queue.pump() == CUDA_SUCCESS, "destroy-after-release fixture must issue once");
            fake.ready();
            fake.destroy_error = CUDA_ERROR_LAUNCH_FAILED;
            fake.should_release_on_destroy_error = true;
            require(queue.pump() == CUDA_ERROR_LAUNCH_FAILED && fake.events.empty() &&
                        fake.destroy_calls == 1,
                    "native destroy can release a handle before reporting an asynchronous error");
        }
        const auto expected_error =
            should_fail_creation ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_LAUNCH_FAILED;
        fake.destroy_error = CUDA_SUCCESS;
        const auto calls = fake.destroy_calls;
        require(queue.shutdown() == expected_error && queue.shutdown() == expected_error &&
                    fake.destroy_calls == calls && !retained.expired() &&
                    queue.query(token) == expected_error,
                "uncertain private handles must not be destroyed or replayed during retry");
        fake.events.clear();  // Simulate actual native context destruction.
        fake.context = nullptr;
        queue.retire_after_context_destroy();
        require(queue.shutdown() == CUDA_SUCCESS && retained.expired() &&
                    fake.destroy_calls == calls && queue.query(token) == expected_error,
                "trusted context destruction retires owners without touching stale event handles");
    }
    Fake fake;
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    auto resources = std::make_shared<int>(1);
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto token = queue.submit(stream, kernel(resources));
    fake.should_return_null_event = true;
    require(queue.pump() == CUDA_ERROR_UNKNOWN && fake.issued.empty() &&
                queue.query(token) == CUDA_ERROR_UNKNOWN && queue.shutdown() == CUDA_SUCCESS,
            "successful event create with a null handle must reject before native GPU issue");
}

void rejection_context_and_lifetime() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch(),
                             {.pending_operations = 1, .streams = 2});
    CudaOperationQueue foreign(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto other = foreign.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                           OrderedStreamKind::kNonBlocking, resources);
    const auto alien = foreign.submit(other, CudaQueuedMarker{});
    const std::array alien_dependency{alien};
    require(!queue.submit(stream, CudaQueuedMarker{}, alien_dependency) &&
                queue.query(alien) == CUDA_ERROR_INVALID_HANDLE && !queue.resume(alien) &&
                !queue.bind_stream(nullptr, OrderedStreamKind::kPerThreadDefault, resources),
            "foreign tokens and ambiguous PTDS identity must be rejected");
    const auto marker = queue.submit(stream, CudaQueuedMarker{});
    require(!queue.submit(stream, CudaQueuedMarker{}) && queue.stream_snapshot(stream) == marker,
            "bounded rejection must not publish a new tail");
    CUresult wrong_thread = CUDA_SUCCESS;
    std::thread thread([&] { wrong_thread = queue.pump(); });
    thread.join();
    require(wrong_thread == CUDA_ERROR_INVALID_CONTEXT && fake.events.empty(),
            "PTDS work must never be pumped on a different thread");
    const auto context = fake.context;
    fake.context = nullptr;
    require(queue.pump() == CUDA_ERROR_INVALID_CONTEXT, "wrong current context must reject issue");
    fake.context = context;
    require(queue.close_stream(stream) && !queue.submit(stream, CudaQueuedMarker{}) &&
                queue.pump() == CUDA_SUCCESS,
            "closing a stream must retain its pending physical operation");
    step(fake, queue);
    const auto reused = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    require(reused && reused != stream && queue.query(marker) == CUDA_SUCCESS,
            "recycled native stream must acquire a fresh identity");
    const auto work = queue.submit(reused, kernel(resources));
    require(queue.pump() == CUDA_SUCCESS, "new identity must execute normally");
    // Simulate successful context destruction: the native owner destroys all
    // events and GPU work before asking the adapter to retire its descriptors.
    fake.events.clear();
    queue.retire_after_context_destroy();
    require(queue.query(work) == CUDA_ERROR_CONTEXT_IS_DESTROYED && queue.pending_count() == 0 &&
                queue.shutdown() == CUDA_SUCCESS && foreign.shutdown() == CUDA_SUCCESS,
            "only a trusted context-destruction proof can replace the stream drain");
}

void continuation_errors_and_typed_copies() {
    for (int stage = 0; stage < 4; ++stage) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        auto progress = std::make_shared<Continuation>();
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto work = queue.submit(stream, kernel(resources, progress));
        const auto error = CUDA_ERROR_INVALID_VALUE;
        if (stage == 0) {
            progress->preparation_error = error;
        } else if (stage == 1) {
            progress->staging_error = error;
        } else if (stage == 2) {
            progress->inspection_error = error;
        } else {
            progress->request_error = error;
        }
        require(queue.pump() == (stage < 2 ? error : CUDA_SUCCESS),
                "continuation preparation/readback errors must reject further issue");
        if (stage == 2) {
            step(fake, queue);
            require(queue.state(work) == OrderedOperationState::kFailed,
                    "inspection failure after proven stop must fail logical work");
        } else if (stage == 3) {
            require(queue.request_retreat(work) == error && queue.shutdown() == error &&
                        queue.state(work) == OrderedOperationState::kRunning &&
                        queue.pending_count() == 1,
                    "failed retreat signaling cannot authorize a physical stop");
            progress->request_error = CUDA_SUCCESS;
        }
        require(queue.query(work) == error && queue.shutdown() == CUDA_SUCCESS &&
                    fake.issued.size() == static_cast<std::size_t>(stage != 0),
                "continuation errors must retain native status without duplicate launches");
    }
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    int host_value = 0;
    require(!queue.submit(stream, CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice, 0x5000, 0,
                                                 nullptr, 4, resources}) &&
                !queue.submit(stream, CudaQueuedFree{0x5000, {}, {}}) && queue.pending_count() == 0,
            "invalid payload ownership must reject transactionally");
    const auto write = queue.submit(stream, CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice,
                                                           0x5000, 0, &host_value, 4, resources});
    const auto read = queue.submit(stream, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToHost, 0,
                                                          0x5000, &host_value, 4, resources});
    require(write && read && queue.pump() == CUDA_SUCCESS && fake.issued == std::vector<char>{'h'},
            "typed host write must precede host read");
    step(fake, queue);
    step(fake, queue);
    require(queue.query(read) == CUDA_SUCCESS && fake.issued == std::vector<char>({'h', 'd'}) &&
                queue.shutdown() == CUDA_SUCCESS,
            "typed host copies must preserve native dispatch and completion");
}

void bounded_history_reclamation() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch(),
                             {.pending_operations = 1, .streams = 1});
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    for (int iteration = 0; iteration < 10000; ++iteration) {
        const auto marker = queue.submit(stream, CudaQueuedMarker{});
        require(marker && queue.pump() == CUDA_SUCCESS, "one-slot history must admit next marker");
        step(fake, queue);
        require(queue.pending_count() == 0 && queue.query(marker) == CUDA_SUCCESS &&
                    fake.events.empty(),
                "terminal payloads and events must not accumulate with stream history");
    }
    require(queue.shutdown() == CUDA_SUCCESS, "bounded history must shut down cleanly");
}

void asynchronous_free_confirmation_and_ambiguity() {
    for (int stage = 0; stage < 5; ++stage) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        auto release_observer = std::make_shared<ReleaseObserver>();
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto release =
            queue.submit(stream, CudaQueuedFree{0x5000, resources, release_observer});
        const auto error = CUDA_ERROR_LAUNCH_FAILED;
        if (stage == 0) {
            fake.issue_error = error;
        } else if (stage == 1) {
            fake.record_error = error;
        } else if (stage == 4) {
            fake.create_error = error;
        }
        require(queue.pump() == (stage < 2 ? error : CUDA_SUCCESS),
                "free issue/record error must remain observable");
        if (stage == 2) {
            fake.query_error = error;
            require(queue.pump() == error, "free query error must poison the queue");
        } else if (stage == 3) {
            fake.ready();
            fake.destroy_error = error;
            require(queue.pump() == error && release_observer->confirmations == 1,
                    "proven release must be published even when event destruction fails");
            fake.destroy_error = CUDA_SUCCESS;
            require(
                queue.shutdown() == error && fake.destroy_calls == 1,
                "proven memory release does not authorize retrying uncertain event destruction");
            fake.events.clear();  // Simulate actual native context destruction.
            fake.context = nullptr;
            queue.retire_after_context_destroy();
        }
        if (stage == 0) {
            require(queue.shutdown() == error && queue.pending_count() == 1 &&
                        release_observer->confirmations == 0,
                    "ambiguous native free must retain its owner even after physical drain");
            fake.events.clear();
            queue.retire_after_context_destroy();
            require(release_observer->context_destructions == 1 && queue.query(release) == error,
                    "context recovery must notify ownership without hiding the original error");
        } else {
            require(queue.shutdown() == CUDA_SUCCESS && queue.query(release) == error &&
                        release_observer->confirmations == (stage == 4 ? 0 : 1) &&
                        release_observer->context_destructions == 0,
                    "successful free enqueue plus stop proof must notify retirement exactly once");
        }
        require(fake.events.empty() && fake.issued.size() <= 1,
                "error cleanup must not retry free or retain a completed event");
    }
}

void native_event_generations_and_boundaries() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    auto progress = std::make_shared<Continuation>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto consumer = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    CUevent native = nullptr;
    require(Fake::create(&native, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS, "create owned event");
    reinterpret_cast<FakeEvent*>(native)->is_user_event = true;
    auto event_owner = std::shared_ptr<void>(native, [](void* pointer) {
        if (Fake::destroy(reinterpret_cast<CUevent>(pointer)) != CUDA_SUCCESS) {
            std::terminate();
        }
    });
    const std::weak_ptr<void> retained = event_owner;
    auto event = queue.bind_event(native, event_owner);
    require(event && queue.query_event(event) == CUDA_SUCCESS &&
                queue.event_snapshot(event).has_value() && !*queue.event_snapshot(event) &&
                !queue.bind_event(native, event_owner),
            "unrecorded event must be complete and duplicate binding rejected");
    const auto empty_wait = queue.wait_event(consumer, event);
    const auto first = queue.submit(producer, CudaQueuedEventRecord{event});
    const auto first_wait = queue.wait_event(consumer, event);
    const auto later = queue.submit(producer, kernel(resources, progress));
    const auto second = queue.submit(producer, CudaQueuedEventRecord{event});
    const auto second_wait = queue.wait_event(consumer, event);
    const std::array first_boundary{first, first_wait};
    require(empty_wait && first && first_wait && later && second && second_wait &&
                queue.event_snapshot(event) == second &&
                queue.query_native_boundary(first_boundary) == CUDA_ERROR_NOT_READY &&
                fake.drains == 0,
            "admission must snapshot each wait and never native-sync a boundary");
    require(queue.pump() == CUDA_SUCCESS, "first event record must issue");
    step(fake, queue);
    step(fake, queue);
    require(queue.query_native_boundary(first_boundary) == CUDA_SUCCESS &&
                queue.state(later) == OrderedOperationState::kSuspended &&
                queue.query_event(event) == CUDA_ERROR_NOT_READY &&
                queue.query(second_wait) == CUDA_ERROR_NOT_READY && fake.public_records == 1,
            "rerecord must not retarget the old wait or issue behind suspension");
    const auto complete_snapshot = queue.context_snapshot();
    require(queue.query_native_boundary(complete_snapshot) == CUDA_ERROR_NOT_READY &&
                queue.close_event(event) && !queue.close_event(event) &&
                !queue.event_snapshot(event) && !queue.wait_event(consumer, event) &&
                !queue.submit(producer, CudaQueuedEventRecord{event}) &&
                !queue.bind_event(native, event_owner),
            "closing must reject lookup and reuse without dropping pending records");
    event_owner.reset();
    event = {};
    require(!retained.expired(), "queued record must retain a closed event owner");
    progress->is_complete = true;
    require(queue.resume(later) && queue.pump() == CUDA_SUCCESS, "resume producer");
    for (int iteration = 0; iteration < 4; ++iteration) {
        step(fake, queue);
    }
    require(queue.query_native_boundary(complete_snapshot) == CUDA_SUCCESS && retained.expired() &&
                fake.public_records == 2 && fake.drains == 0,
            "final logical completion must record twice and release closed event ownership");
    require(queue.shutdown() == CUDA_SUCCESS &&
                queue.query_native_boundary(first_boundary) == CUDA_ERROR_CONTEXT_IS_DESTROYED,
            "shutdown prohibits future native boundary use");
}

void guarded_deferred_event_records() {
    for (const bool should_supersede : {false, true}) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        auto progress = std::make_shared<Continuation>();
        auto guard = std::make_shared<RecordGuard>();
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                                OrderedStreamKind::kNonBlocking, resources);
        const auto consumer = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                                OrderedStreamKind::kNonBlocking, resources);
        CUevent native = nullptr;
        require(Fake::create(&native, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS,
                "create guarded original event");
        reinterpret_cast<FakeEvent*>(native)->is_user_event = true;
        const auto event = queue.bind_event(native, resources);
        const auto work = queue.submit(producer, kernel(resources, progress));
        const auto record = queue.submit(producer, CudaQueuedEventRecord{event, guard});
        const auto wait = queue.wait_event(consumer, event);
        require(work && record && wait && queue.pump() == CUDA_SUCCESS,
                "accept producer, guarded record and immutable wait");
        step(fake, queue);
        require(queue.state(work) == OrderedOperationState::kSuspended && guard->calls == 0,
                "guard cannot run ahead of suspended producer");
        if (should_supersede) {
            guard->is_current = false;
            require(Fake::record(native, reinterpret_cast<CUstream>(0x5000)) == CUDA_SUCCESS,
                    "original native owner records while producer is physically stopped");
        }
        require(queue.query(record) == CUDA_ERROR_NOT_READY &&
                    queue.query(wait) == CUDA_ERROR_NOT_READY && guard->calls == 0,
                "native supersession cannot complete historical producer or wait");
        progress->is_complete = true;
        require(queue.resume(work) && queue.pump() == CUDA_SUCCESS,
                "resume the original producer without dropping dependencies");
        step(fake, queue);
        require(queue.state(record) == OrderedOperationState::kRunning &&
                    queue.query(wait) == CUDA_ERROR_NOT_READY && guard->calls == 1 &&
                    fake.public_records == 1,
                "current write or superseded marker still needs its actual private fence");
        for (int iteration = 0; iteration < 3; ++iteration) {
            step(fake, queue);
        }
        require(queue.query(record) == CUDA_SUCCESS && queue.query(wait) == CUDA_SUCCESS &&
                    guard->calls == 1 && fake.public_records == 1 &&
                    queue.shutdown() == CUDA_SUCCESS && Fake::destroy(native) == CUDA_SUCCESS,
                "exact historical completion must not rewrite the newer original event");
    }
    for (const auto error :
         {CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_OUT_OF_MEMORY}) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        auto guard = std::make_shared<RecordGuard>();
        guard->result = error;
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto event = queue.bind_event(reinterpret_cast<CUevent>(0x7000), resources);
        const auto record = queue.submit(stream, CudaQueuedEventRecord{event, guard});
        require(record && queue.pump() == error && queue.query(record) == error &&
                    queue.pump() == error && guard->calls == 1 && fake.recorded_events.empty() &&
                    queue.shutdown() == CUDA_SUCCESS,
                "guard failure preserves exact error and forbids original record replay");
    }
    for (const bool should_fail_create : {false, true}) {
        Fake fake;
        auto resources = std::make_shared<int>(0);
        auto guard = std::make_shared<RecordGuard>();
        guard->is_current = false;
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto native = reinterpret_cast<CUevent>(0x7000);
        const auto event = queue.bind_event(native, resources);
        const auto record = queue.submit(stream, CudaQueuedEventRecord{event, guard});
        const auto error = should_fail_create ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_LAUNCH_FAILED;
        if (should_fail_create) {
            fake.create_error = error;
        } else {
            fake.record_error = error;
        }
        const auto pump_result = should_fail_create ? CUDA_SUCCESS : error;
        require(record && queue.pump() == pump_result && queue.query(record) == error &&
                    queue.pump() == pump_result &&
                    guard->calls == static_cast<int>(!should_fail_create) &&
                    fake.recorded_events.size() == static_cast<std::size_t>(!should_fail_create) &&
                    std::ranges::count(fake.recorded_events, native) == 0 &&
                    queue.shutdown() == CUDA_SUCCESS,
                "suppressed user write cannot hide private fence allocation/record failure");
    }
}

void cross_stream_rerecord_serialization() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    auto progress = std::make_shared<Continuation>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto consumer = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto event = queue.bind_event(reinterpret_cast<CUevent>(0x7000), resources);
    const auto work = queue.submit(producer, kernel(resources, progress));
    const auto first = queue.submit(producer, CudaQueuedEventRecord{event});
    const auto second = queue.submit(consumer, CudaQueuedEventRecord{event});
    require(work && first && second && queue.pump() == CUDA_SUCCESS,
            "cross-stream records must admit");
    step(fake, queue);
    require(queue.state(work) == OrderedOperationState::kSuspended &&
                queue.state(first) == OrderedOperationState::kQueued &&
                queue.state(second) == OrderedOperationState::kQueued &&
                std::ranges::count(fake.recorded_events, reinterpret_cast<CUevent>(0x7000)) == 0,
            "rerecord on independent stream must not overtake an earlier deferred record");
    CUresult wrong_thread = CUDA_SUCCESS;
    std::thread thread([&] { wrong_thread = queue.query_native_boundary({}); });
    thread.join();
    require(wrong_thread == CUDA_ERROR_INVALID_CONTEXT,
            "native boundary must reject a non-owner host thread");
    progress->is_complete = true;
    require(queue.resume(work) && queue.pump() == CUDA_SUCCESS, "resume cross-stream producer");
    step(fake, queue);
    require(queue.state(first) == OrderedOperationState::kRunning &&
                queue.state(second) == OrderedOperationState::kQueued,
            "physical event handle reuse waits for previous logical record");
    step(fake, queue);
    step(fake, queue);
    require(queue.query_event(event) == CUDA_SUCCESS && queue.shutdown() == CUDA_SUCCESS,
            "both native records must complete in order");
}

void event_rejection_and_error_retention() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch(),
                             {.pending_operations = 1, .streams = 2});
    CudaOperationQueue foreign(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto foreign_event = foreign.bind_event(reinterpret_cast<CUevent>(0x6000), resources);
    auto event = queue.bind_event(reinterpret_cast<CUevent>(0x7000), resources);
    require(event && !queue.bind_event(reinterpret_cast<CUevent>(0x8000), resources) &&
                !queue.wait_event(stream, foreign_event) &&
                !queue.submit(stream, CudaQueuedEventRecord{foreign_event}) &&
                queue.query_event(foreign_event) == CUDA_ERROR_INVALID_HANDLE,
            "event bounds and foreign handles must reject transactionally");
    const auto first = queue.submit(stream, CudaQueuedEventRecord{event});
    require(first && !queue.submit(stream, CudaQueuedEventRecord{event}) &&
                queue.event_snapshot(event) == first && queue.stream_snapshot(stream) == first,
            "full queue must preserve previous event and stream generations");
    fake.record_error = CUDA_ERROR_LAUNCH_FAILED;
    require(queue.pump() == CUDA_ERROR_LAUNCH_FAILED &&
                queue.query_event(event) == CUDA_ERROR_LAUNCH_FAILED &&
                queue.query_native_boundary({}) == CUDA_ERROR_LAUNCH_FAILED,
            "native record failure must poison even an empty native boundary");
    fake.drain_error = CUDA_ERROR_INVALID_CONTEXT;
    require(queue.shutdown() == CUDA_ERROR_INVALID_CONTEXT &&
                queue.query(first) == CUDA_ERROR_LAUNCH_FAILED,
            "failed drain must preserve the original native record error");
    fake.record_error = CUDA_SUCCESS;
    fake.drain_error = CUDA_SUCCESS;
    require(queue.shutdown() == CUDA_SUCCESS, "event error recovery must drain ownership");

    CudaOperationQueue errors(fake.context, Fake::dispatch());
    const auto a = errors.bind_stream(reinterpret_cast<CUstream>(0x9000),
                                      OrderedStreamKind::kNonBlocking, resources);
    const auto b = errors.bind_stream(reinterpret_cast<CUstream>(0xa000),
                                      OrderedStreamKind::kNonBlocking, resources);
    const auto pending = errors.submit(a, CudaQueuedMarker{});
    const auto failed = errors.submit(b, CudaQueuedMarker{});
    require(errors.cancel_stopped(failed) && errors.close_stream(b), "retain failed closed tail");
    const auto snapshot = errors.context_snapshot();
    require(errors.query_native_boundary(snapshot) == CUDA_ERROR_UNKNOWN,
            "terminal errors take precedence over a pending earlier stream");
    const auto foreign_work =
        foreign.submit(foreign.bind_stream(reinterpret_cast<CUstream>(0xb000),
                                           OrderedStreamKind::kNonBlocking, resources),
                       CudaQueuedMarker{});
    const std::array foreign_snapshot{pending, foreign_work};
    require(errors.query_native_boundary(foreign_snapshot) == CUDA_ERROR_INVALID_HANDLE,
            "foreign boundary snapshots must reject, not appear pending");
    fake.context = reinterpret_cast<CUcontext>(0xc000);
    require(errors.query_native_boundary(snapshot) == CUDA_ERROR_INVALID_CONTEXT,
            "native boundary must check owning current context");
    fake.context = reinterpret_cast<CUcontext>(0x1000);
    require(errors.shutdown() == CUDA_SUCCESS && foreign.shutdown() == CUDA_SUCCESS,
            "error snapshots must clean up without replay");
    CudaOperationQueue bounded(fake.context, Fake::dispatch(),
                               {.pending_operations = 1, .streams = 1});
    for (int iteration = 0; iteration < 10000; ++iteration) {
        const auto binding = bounded.bind_event(reinterpret_cast<CUevent>(0xd000), resources);
        require(binding && bounded.close_event(binding),
                "closed unrecorded bindings must reclaim capacity");
    }
    const auto bounded_stream = bounded.bind_stream(reinterpret_cast<CUstream>(0xe000),
                                                    OrderedStreamKind::kNonBlocking, resources);
    const auto bounded_event = bounded.bind_event(reinterpret_cast<CUevent>(0xd000), resources);
    for (int iteration = 0; iteration < 10000; ++iteration) {
        const auto record = bounded.submit(bounded_stream, CudaQueuedEventRecord{bounded_event});
        require(record && bounded.pump() == CUDA_SUCCESS, "bounded rerecord must issue");
        step(fake, bounded);
        require(bounded.query_event(bounded_event) == CUDA_SUCCESS &&
                    bounded.pending_count() == 0 && fake.events.empty(),
                "rerecord history must not retain private events or payloads");
    }
}

void controlled_kernel_admission() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    auto progress = std::make_shared<Continuation>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    CudaOperationQueue foreign(fake.context, Fake::dispatch());
    const auto low_stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
    const auto high_stream = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                               OrderedStreamKind::kNonBlocking, resources);
    const auto low = queue.submit(low_stream, kernel(resources, progress));
    const auto dependent = queue.submit(low_stream, kernel(resources));
    const auto marker = queue.submit(low_stream, CudaQueuedMarker{});
    const auto high = queue.submit(high_stream, kernel(resources));
    require(queue.kernel_candidates().size() == 2 && queue.pump_admitted({}) == CUDA_SUCCESS &&
                fake.issued.empty() && fake.events.empty(),
            "blocked kernels are not candidates and no unadmitted kernel may issue");
    const std::array low_selection{low};
    require(queue.pump_admitted(low_selection) == CUDA_SUCCESS &&
                queue.state(low) == OrderedOperationState::kRunning && fake.issued.size() == 1,
            "exactly the selected ready kernel must issue");
    fake.ready();
    const std::array duplicate{high, high};
    const std::array non_kernel{high, marker};
    const std::array blocked{dependent};
    const auto other =
        foreign.submit(foreign.bind_stream(reinterpret_cast<CUstream>(0x5000),
                                           OrderedStreamKind::kNonBlocking, resources),
                       kernel(resources));
    const std::array foreign_selection{high, other};
    require(queue.pump_admitted(duplicate) == CUDA_ERROR_INVALID_VALUE &&
                queue.pump_admitted(non_kernel) == CUDA_ERROR_INVALID_VALUE &&
                queue.pump_admitted(blocked) == CUDA_ERROR_INVALID_VALUE &&
                queue.pump_admitted(foreign_selection) == CUDA_ERROR_INVALID_VALUE &&
                progress->inspections == 0 && fake.issued.size() == 1,
            "invalid selection must reject atomically before native queries or issue");
    require(queue.pump_admitted({}) == CUDA_SUCCESS &&
                queue.state(low) == OrderedOperationState::kSuspended &&
                queue.kernel_candidates().size() == 2 &&
                queue.pump_admitted(low_selection) == CUDA_ERROR_INVALID_VALUE,
            "polling permits stop proof, not implicit admission of a continuation");
    const std::array high_selection{high};
    require(queue.pump_admitted(high_selection) == CUDA_SUCCESS && fake.issued.size() == 2,
            "higher-priority admitted work may issue while low remains stopped");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.query(high) == CUDA_SUCCESS &&
                queue.state(low) == OrderedOperationState::kSuspended,
            "high completion must not resume unadmitted low work");
    progress->is_complete = true;
    require(queue.resume(low) && queue.pump_admitted(low_selection) == CUDA_SUCCESS,
            "explicitly resumed and admitted low work may issue");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.query(low) == CUDA_SUCCESS &&
                queue.state(dependent) == OrderedOperationState::kQueued,
            "newly ready dependent kernels remain gated by admission");
    const std::array dependent_selection{dependent};
    require(queue.pump_admitted(dependent_selection) == CUDA_SUCCESS, "admit dependent kernel");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS &&
                queue.state(marker) == OrderedOperationState::kRunning,
            "non-kernel logical dependencies progress without kernel admission");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.query(marker) == CUDA_SUCCESS &&
                queue.kernel_candidates().empty() && queue.shutdown() == CUDA_SUCCESS &&
                foreign.shutdown() == CUDA_SUCCESS,
            "controlled queue must complete without replay or retained candidates");
}

struct NativeCompletion final : CudaQueueNativeFenceObserver {
    int calls{};
    CUresult status{CUDA_SUCCESS};
    CUresult confirmed_native_completion() noexcept override {
        ++calls;
        return status;
    }
};

void imported_native_fences() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    auto fence_owner = std::make_shared<int>(1);
    const std::weak_ptr<void> retained = fence_owner;
    auto native_event = std::make_unique<FakeEvent>();
    native_event->is_user_event = true;
    const auto event = reinterpret_cast<CUevent>(native_event.get());
    fake.events.push_back(std::move(native_event));
    auto completion = std::make_shared<NativeCompletion>();
    const auto fence = queue.submit(stream, CudaQueuedNativeFence{event, fence_owner, completion});
    const auto dependent = queue.submit(stream, kernel(resources));
    fence_owner.reset();
    require(fence && dependent && !retained.expired() && queue.kernel_candidates().empty() &&
                queue.pump_admitted({}) == CUDA_SUCCESS && fake.issued == std::vector<char>{'w'} &&
                fake.waited_event == event && queue.query(fence) == CUDA_ERROR_NOT_READY &&
                completion->calls == 0,
            "import must retain an owned CUDA-visible event dependency before later issue");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && retained.expired() &&
                queue.query(fence) == CUDA_SUCCESS && queue.kernel_candidates().size() == 1 &&
                fake.issued == std::vector<char>{'w'} && completion->calls == 1,
            "native physical proof releases the fence, not an unadmitted dependent kernel");
    const std::array selection{dependent};
    require(queue.pump_admitted(selection) == CUDA_SUCCESS,
            "admit imported-fence dependent kernel");
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.shutdown() == CUDA_SUCCESS &&
                completion->calls == 1,
            "native-fence import must clean up private ownership");

    auto missing = Fake::dispatch();
    missing.stream_wait_event = nullptr;
    CudaOperationQueue unavailable(fake.context, missing);
    const auto missing_stream = unavailable.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                                        OrderedStreamKind::kNonBlocking, resources);
    require(!unavailable.submit(missing_stream, CudaQueuedNativeFence{event, resources}) &&
                unavailable.stream_snapshot(missing_stream).has_value() &&
                !*unavailable.stream_snapshot(missing_stream),
            "missing native wait support rejects without publishing a stream tail");
    missing = Fake::dispatch();
    missing.event_synchronize = nullptr;
    CudaOperationQueue no_drain(fake.context, missing);
    const auto no_drain_stream = no_drain.bind_stream(reinterpret_cast<CUstream>(0x8000),
                                                      OrderedStreamKind::kNonBlocking, resources);
    require(!no_drain.submit(no_drain_stream, CudaQueuedNativeFence{event, resources}) &&
                !*no_drain.stream_snapshot(no_drain_stream),
            "missing native prefix proof must reject before publishing ownership");
    CudaOperationQueue errors(fake.context, Fake::dispatch());
    const auto error_stream = errors.bind_stream(reinterpret_cast<CUstream>(0x5000),
                                                 OrderedStreamKind::kNonBlocking, resources);
    auto error_owner = std::make_shared<int>(2);
    const std::weak_ptr<void> error_retained = error_owner;
    auto failed_completion = std::make_shared<NativeCompletion>();
    const auto failed =
        errors.submit(error_stream, CudaQueuedNativeFence{event, error_owner, failed_completion});
    error_owner.reset();
    fake.wait_error = CUDA_ERROR_LAUNCH_FAILED;
    require(failed && errors.pump_admitted({}) == CUDA_ERROR_LAUNCH_FAILED &&
                errors.query(failed) == CUDA_ERROR_LAUNCH_FAILED && !error_retained.expired(),
            "native wait error retains ambiguous physical ownership and original status");
    fake.drain_error = CUDA_ERROR_INVALID_CONTEXT;
    fake.prefix_drain_error = CUDA_ERROR_INVALID_CONTEXT;
    require(errors.shutdown() == CUDA_ERROR_INVALID_CONTEXT && !error_retained.expired() &&
                failed_completion->calls == 0,
            "failed native-fence drain cannot release ownership");
    fake.drain_error = CUDA_SUCCESS;
    fake.prefix_drain_error = CUDA_SUCCESS;
    require(errors.shutdown() == CUDA_SUCCESS && error_retained.expired() &&
                errors.query(failed) == CUDA_ERROR_LAUNCH_FAILED && failed_completion->calls == 1,
            "proven recovery releases the imported fence without retrying native work");

    CudaOperationQueue unissued(fake.context, Fake::dispatch());
    const auto unissued_stream = unissued.bind_stream(reinterpret_cast<CUstream>(0x9000),
                                                      OrderedStreamKind::kNonBlocking, resources);
    auto unissued_owner = std::make_shared<int>(3);
    const std::weak_ptr<void> unissued_retained = unissued_owner;
    auto unissued_completion = std::make_shared<NativeCompletion>();
    const auto unissued_fence = unissued.submit(
        unissued_stream, CudaQueuedNativeFence{event, unissued_owner, unissued_completion});
    unissued_owner.reset();
    require(
        unissued_fence && !unissued.cancel_stopped(unissued_fence) && !unissued_retained.expired(),
        "an unissued import is not proof that its external native prefix stopped");
    fake.prefix_drain_error = CUDA_ERROR_INVALID_CONTEXT;
    require(unissued.shutdown() == CUDA_ERROR_INVALID_CONTEXT && !unissued_retained.expired() &&
                unissued_completion->calls == 0,
            "failed unissued-prefix drain must retain its external owner");
    fake.prefix_drain_error = CUDA_SUCCESS;
    require(unissued.shutdown() == CUDA_SUCCESS && unissued_retained.expired() &&
                unissued_completion->calls == 1,
            "successful prefix event proof must permit explicit owner release");

    CudaOperationQueue create_failure(fake.context, Fake::dispatch());
    const auto create_stream = create_failure.bind_stream(
        reinterpret_cast<CUstream>(0xa000), OrderedStreamKind::kNonBlocking, resources);
    auto create_owner = std::make_shared<int>(4);
    const std::weak_ptr<void> create_retained = create_owner;
    const auto create_fence =
        create_failure.submit(create_stream, CudaQueuedNativeFence{event, create_owner});
    create_owner.reset();
    fake.create_error = CUDA_ERROR_OUT_OF_MEMORY;
    require(create_fence && create_failure.pump_admitted({}) == CUDA_ERROR_OUT_OF_MEMORY &&
                !create_retained.expired(),
            "private event allocation failure must not release an already-running native prefix");
    fake.create_error = CUDA_SUCCESS;
    fake.prefix_drain_error = CUDA_ERROR_INVALID_CONTEXT;
    require(create_failure.shutdown() == CUDA_ERROR_INVALID_CONTEXT && !create_retained.expired(),
            "event-create failure still requires actual native prefix proof");
    fake.prefix_drain_error = CUDA_SUCCESS;
    require(create_failure.shutdown() == CUDA_SUCCESS && create_retained.expired() &&
                create_failure.query(create_fence) == CUDA_ERROR_OUT_OF_MEMORY,
            "prefix recovery must preserve the original create error without replay");
}

void native_completion_errors() {
    Fake fake;
    auto resources = std::make_shared<int>(0);
    auto native_event = std::make_unique<FakeEvent>();
    native_event->is_user_event = true;
    const auto event = reinterpret_cast<CUevent>(native_event.get());
    fake.events.push_back(std::move(native_event));
    auto completion = std::make_shared<NativeCompletion>();
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto token = queue.submit(stream, CudaQueuedNativeFence{event, resources, completion});
    require(token && queue.pump_admitted({}) == CUDA_SUCCESS && completion->calls == 0,
            "native wait import does not notify completion");
    fake.query_error = CUDA_ERROR_LAUNCH_FAILED;
    require(queue.pump_admitted({}) == CUDA_ERROR_LAUNCH_FAILED && completion->calls == 0,
            "failed physical query supplies no proof");
    fake.query_error = CUDA_SUCCESS;
    fake.prefix_drain_error = CUDA_ERROR_INVALID_CONTEXT;
    require(queue.shutdown() == CUDA_ERROR_INVALID_CONTEXT && completion->calls == 0,
            "failed explicit native prefix proof cannot notify");
    fake.prefix_drain_error = CUDA_SUCCESS;
    require(queue.shutdown() == CUDA_SUCCESS && completion->calls == 1 &&
                queue.query(token) == CUDA_ERROR_LAUNCH_FAILED,
            "actual recovery proof notifies once while retaining original operation error");

    auto rejected = std::make_shared<NativeCompletion>();
    rejected->status = CUDA_ERROR_INVALID_HANDLE;
    CudaOperationQueue metadata_error(fake.context, Fake::dispatch());
    const auto other = metadata_error.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                                  OrderedStreamKind::kNonBlocking, resources);
    const auto failed =
        metadata_error.submit(other, CudaQueuedNativeFence{event, resources, rejected});
    require(failed && metadata_error.pump_admitted({}) == CUDA_SUCCESS,
            "issue native import before injecting completion metadata failure");
    fake.ready();
    require(metadata_error.pump_admitted({}) == CUDA_ERROR_INVALID_HANDLE && rejected->calls == 1 &&
                metadata_error.pump_admitted({}) == CUDA_ERROR_INVALID_HANDLE &&
                metadata_error.shutdown() == CUDA_SUCCESS && rejected->calls == 1 &&
                metadata_error.query(failed) == CUDA_ERROR_INVALID_HANDLE,
            "notification failure retains exact error without callback or native replay");

    auto cleanup = std::make_shared<NativeCompletion>();
    CudaOperationQueue cleanup_error(fake.context, Fake::dispatch());
    const auto cleanup_stream = cleanup_error.bind_stream(
        reinterpret_cast<CUstream>(0x5000), OrderedStreamKind::kNonBlocking, resources);
    const auto cleanup_token =
        cleanup_error.submit(cleanup_stream, CudaQueuedNativeFence{event, resources, cleanup});
    require(cleanup_token && cleanup_error.pump_admitted({}) == CUDA_SUCCESS,
            "issue native import before private event cleanup failure");
    fake.ready();
    fake.destroy_error = CUDA_ERROR_UNKNOWN;
    require(cleanup_error.pump_admitted({}) == CUDA_ERROR_UNKNOWN && cleanup->calls == 1,
            "proven prefix completion precedes failed private event cleanup");
    cleanup_error.retire_after_context_destroy();
    fake.destroy_error = CUDA_SUCCESS;
    require(cleanup_error.shutdown() == CUDA_SUCCESS && cleanup->calls == 1,
            "trusted context recovery cannot repeat completion notification");

    auto retired = std::make_shared<NativeCompletion>();
    CudaOperationQueue destroyed_context(fake.context, Fake::dispatch());
    const auto retired_stream = destroyed_context.bind_stream(
        reinterpret_cast<CUstream>(0x6000), OrderedStreamKind::kNonBlocking, resources);
    const auto retired_token =
        destroyed_context.submit(retired_stream, CudaQueuedNativeFence{event, resources, retired});
    require(static_cast<bool>(retired_token),
            "retain unissued native prefix observer until trusted context recovery");
    destroyed_context.retire_after_context_destroy();
    require(destroyed_context.shutdown() == CUDA_SUCCESS && retired->calls == 0,
            "context destruction retires owners without publishing native event completion");
}

template <typename Predicate>
void coordinate_until(CudaAdmissionCoordinator& coordinator, CudaOperationQueue& queue,
                      Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        // Progress is a simulated scheduler tick, not elapsed host wall time.
        // Retry/heartbeat deadlines must see monotonic time advance explicitly.
        Fake::active->now += std::chrono::milliseconds{1};
        require(coordinator.progress(queue) == CUDA_SUCCESS, "cooperative progress must succeed");
        std::this_thread::yield();
    }
    require(predicate(), "cooperative lifecycle must reach expected state");
}

void asynchronous_execution_coordination() {
    Fake fake;
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaAdmissionCoordinator coordinator(gate, {.pending_limit = 3, .clock_now = Fake::clock_now});
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto priority = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    auto progress = std::make_shared<Continuation>();
    const auto low = queue.submit(producer, kernel(resources, progress));
    const auto blocked = queue.submit(producer, kernel(resources));
    require(coordinator.track(queue, low, {.relative_priority = 1}) &&
                coordinator.track(queue, blocked, {.relative_priority = 99}) &&
                !coordinator.track(queue, low),
            "only new queued tokens may register once");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(low) == OrderedOperationState::kRunning; });
    require(fake.issued == std::vector<char>{'k'},
            "blocked priority kernel cannot enter policy before dependency readiness");
    const auto heartbeats_before = coordinator.stats().acknowledged_heartbeats;
    fake.now += std::chrono::milliseconds{100};
    coordinate_until(coordinator, queue, [&] {
        return coordinator.stats().acknowledged_heartbeats == heartbeats_before + 1;
    });
    const auto high = queue.submit(priority, kernel(resources));
    require(coordinator.track(queue, high, {.relative_priority = 9}),
            "ready high kernel must register");
    const auto overflow = queue.submit(priority, kernel(resources));
    require(!coordinator.track(queue, overflow) && queue.cancel_stopped(overflow),
            "bounded registration failure must allow unissued transactional rollback");
    coordinate_until(coordinator, queue, [&] { return progress->requests == 1; });
    require(queue.state(low) == OrderedOperationState::kRunning && fake.issued.size() == 1 &&
                coordinator.stats().acknowledged_suspensions == 0,
            "retreat request alone cannot admit priority work or acknowledge stop");
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(high) == OrderedOperationState::kRunning; });
    require(queue.state(low) == OrderedOperationState::kSuspended &&
                coordinator.stats().acknowledged_suspensions == 1 && fake.issued.size() == 2,
            "proven physical stop and suspend acknowledgement must precede priority issue");
    progress->is_complete = true;
    fake.ready();
    coordinate_until(coordinator, queue, [&] { return coordinator.stats().logical_resumes == 1; });
    require(
        queue.query(high) == CUDA_SUCCESS && queue.state(low) == OrderedOperationState::kRunning,
        "resumption must follow high completion and a new execution grant");
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(blocked) == OrderedOperationState::kRunning; });
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return coordinator.stats().acknowledged_completions == 3; });
    require(queue.query(low) == CUDA_SUCCESS && queue.query(blocked) == CUDA_SUCCESS &&
                coordinator.stats().granted_executions == 4 &&
                coordinator.stats().retreat_requests == 1 && fake.issued.size() == 4,
            "complete logical operations once despite repeated physical generation");
    require(queue.shutdown() == CUDA_SUCCESS, "coordinated physical queue must drain");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS, "terminal reports must be acknowledged");
}

void stopped_combined_submission_preserves_ownership() {
    for (const bool has_external_owner : {false, true}) {
        Fake fake;
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
        const auto owner = has_external_owner ? gate->submit_and_claim().get()
                                              : glimmer::control::CooperativeSubmission{};
        CudaAdmissionCoordinator coordinator(gate, {.clock_now = Fake::clock_now});
        auto resources = std::make_shared<int>(0);
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto token = queue.submit(stream, CudaQueuedNativeReservation{resources});
        require(
            token && coordinator.track(queue, token) && coordinator.progress(queue) == CUDA_SUCCESS,
            "combined submission must enter the asynchronous lifecycle");
        require(
            gate->claim(999999).get().status == glimmer::control::CooperativeClaimStatus::kFailed &&
                coordinator.stats().granted_executions == 0 && fake.issued.empty(),
            "mailbox result alone cannot issue CUDA or publish a consumed grant");
        require(queue.shutdown() == CUDA_SUCCESS, "never-issued queue must physically stop");
        coordinator.stopped(queue, CUDA_SUCCESS);
        require(coordinator.control_status() == CUDA_SUCCESS && fake.issued.empty() &&
                    coordinator.stats().acknowledged_failures == (has_external_owner ? 0 : 1),
                "stopped teardown must distinguish pending cancellation from granted failure");
        const auto next = gate->submit_and_claim().get();
        require(next.task_id != 0 &&
                    next.claim.status == (has_external_owner
                                              ? glimmer::control::CooperativeClaimStatus::kPending
                                              : glimmer::control::CooperativeClaimStatus::kGranted),
                "stopped cleanup must release its own grant, never external ownership");
        require(has_external_owner ? gate->cancel_pending(next.task_id).get() &&
                                         gate->complete(owner.claim.execution).get()
                                   : gate->complete(next.claim.execution).get(),
                "fixture terminal ownership must clean up exactly once");
    }
}

void pending_claim_deadline_is_not_a_grant(bool is_native, bool is_saturated = false,
                                           bool should_rearm = false) {
    Fake fake;
    if (is_saturated) {
        fake.now = std::chrono::steady_clock::time_point::max() - std::chrono::microseconds{10};
    }
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    const auto occupied = gate->submit().get();
    require(occupied.has_value(), "external physical owner must submit");
    const auto owner = gate->claim(*occupied).get();
    require(owner.status == glimmer::control::CooperativeClaimStatus::kGranted,
            "external physical owner must retain the only slot");
    CudaAdmissionCoordinator coordinator(gate, {.clock_now = Fake::clock_now});
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto pending = is_native ? queue.submit(stream, CudaQueuedNativeReservation{resources})
                                   : queue.submit(stream, kernel(resources));
    require(pending && coordinator.track(queue, pending), "pending work must register");
    const auto is_granted = [&] {
        return is_native ? coordinator.native_reservation_grant(queue, pending).has_value()
                         : queue.state(pending) == OrderedOperationState::kRunning;
    };
    for (unsigned int scan = 0; scan < 4; ++scan) {
        require(
            coordinator.progress(queue) == CUDA_SUCCESS && gate->heartbeat(owner.execution).get(),
            "ordered barriers settle submission and pending claim without changing fake time");
    }
    if (should_rearm) {
        fake.now += std::chrono::microseconds{25};
        for (unsigned int scan = 0; scan < 4; ++scan) {
            require(coordinator.progress(queue) == CUDA_SUCCESS &&
                        gate->heartbeat(owner.execution).get() && !is_granted(),
                    "a repeated Pending reply rearms another positive native deadline");
        }
    }
    require(!is_granted() && gate->complete(owner.execution).get(),
            "an ungranted reservation does not release the external physical owner");
    for (unsigned int scan = 0; scan < 32; ++scan) {
        require(coordinator.progress(queue) == CUDA_SUCCESS &&
                    gate->claim(999999).get().status ==
                        glimmer::control::CooperativeClaimStatus::kFailed &&
                    !is_granted(),
                "unrelated ready results cannot bypass the frozen pending-claim deadline");
    }
    fake.now += std::chrono::microseconds{is_saturated ? 9 : 24};
    for (unsigned int scan = 0; scan < 4; ++scan) {
        require(coordinator.progress(queue) == CUDA_SUCCESS &&
                    gate->claim(999999).get().status ==
                        glimmer::control::CooperativeClaimStatus::kFailed &&
                    !is_granted(),
                "even ready capacity cannot bypass the positive native retry boundary");
    }
    fake.now += std::chrono::microseconds{1};
    for (unsigned int scan = 0; scan < 4; ++scan) {
        require(coordinator.progress(queue) == CUDA_SUCCESS &&
                    gate->claim(999999).get().status ==
                        glimmer::control::CooperativeClaimStatus::kFailed,
                "exact deadline permits only a fresh asynchronous claim");
    }
    require(is_granted() == (is_native || is_saturated),
            "native uses 25 us, kernels one ms; both honor the saturated deadline");
    if (!is_native) {
        if (!is_saturated) {
            fake.now += std::chrono::microseconds{975};
            for (unsigned int scan = 0; scan < 4; ++scan) {
                require(coordinator.progress(queue) == CUDA_SUCCESS &&
                            gate->claim(999999).get().status ==
                                glimmer::control::CooperativeClaimStatus::kFailed,
                        "legacy kernel retry remains asynchronous at one ms");
            }
        }
        require(is_granted() && fake.issued == std::vector<char>{'k'},
                "legacy kernel issues only under its real one-ms grant");
        fake.ready();
    }
    require((!is_native || fake.issued.empty()) && queue.shutdown() == CUDA_SUCCESS,
            "elapsed native retry obtains authority but never issues original CUDA");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS,
            "unissued granted ownership retires only after actual stopped teardown");
}

void native_claim_retry_options() {
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    require(CudaAdmissionOptions{}.native_request_retry_interval == std::chrono::microseconds{25} &&
                CudaAdmissionOptions{}.request_retry_interval == std::chrono::milliseconds{1},
            "native and ordinary control retries have separate explicit defaults");
    for (const auto interval : {std::chrono::microseconds{-1}, std::chrono::microseconds{0},
                                std::chrono::microseconds{1000001}}) {
        bool was_rejected{};
        try {
            CudaAdmissionCoordinator invalid(gate, {.native_request_retry_interval = interval});
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "invalid native retry intervals reject before coordination");
    }
    for (const auto interval : {std::chrono::microseconds{1}, std::chrono::microseconds{1000000}}) {
        CudaAdmissionCoordinator valid(gate, {.native_request_retry_interval = interval});
        require(valid.control_status() == CUDA_SUCCESS, "native retry limits are inclusive");
    }
}

void false_retreat_response_backs_off() {
    Fake fake;
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaAdmissionCoordinator coordinator(gate, {.clock_now = Fake::clock_now});
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    auto continuation = std::make_shared<Continuation>();
    const auto low = queue.submit(stream, kernel(resources, continuation));
    require(coordinator.track(queue, low, {.relative_priority = 1}), "low work must register");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(low) == OrderedOperationState::kRunning; });
    for (unsigned int scan = 0; scan < 4; ++scan) {
        require(coordinator.progress(queue) == CUDA_SUCCESS &&
                    gate->claim(999999).get().status ==
                        glimmer::control::CooperativeClaimStatus::kFailed,
                "settle a false retreat response at a frozen clock");
    }
    const auto high = gate->submit({.relative_priority = 9}).get();
    require(high && gate->claim(*high).get().status ==
                        glimmer::control::CooperativeClaimStatus::kPending,
            "actual high arrival must wait for physical low stop");
    for (unsigned int scan = 0; scan < 32; ++scan) {
        require(coordinator.progress(queue) == CUDA_SUCCESS &&
                    gate->claim(999999).get().status ==
                        glimmer::control::CooperativeClaimStatus::kFailed &&
                    continuation->requests == 0,
                "unrelated publication cannot bypass the false-retreat retry deadline");
    }
    fake.now += std::chrono::milliseconds{1};
    coordinate_until(coordinator, queue, [&] { return continuation->requests == 1; });
    require(coordinator.stats().acknowledged_suspensions == 0 && fake.issued.size() == 1,
            "a retry can request retreat but cannot acknowledge physical stop");
    fake.ready();
    require(queue.shutdown() == CUDA_SUCCESS, "actual physical stop must precede teardown");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS && gate->cancel_pending(*high).get(),
            "stopped low ownership and never-issued high work retire exactly once");
    for (const auto interval : {std::chrono::milliseconds{0}, std::chrono::milliseconds{1001}}) {
        bool was_rejected{};
        try {
            CudaAdmissionCoordinator invalid(gate, {.request_retry_interval = interval});
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "invalid retry intervals reject before coordination");
    }
}

void coordination_failure_and_unproven_stop() {
    for (const bool should_close_control : {false, true}) {
        Fake fake;
        glimmer::control::LaunchGateOptions options;
        options.tenant_weight = should_close_control ? 1 : 0;
        auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
        if (should_close_control) {
            gate->shutdown();
        }
        CudaAdmissionCoordinator coordinator(gate);
        auto resources = std::make_shared<int>(0);
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                              OrderedStreamKind::kNonBlocking, resources);
        const auto token = queue.submit(stream, kernel(resources));
        require(coordinator.track(queue, token), "rejection fixture must register metadata");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (coordinator.progress(queue) == CUDA_SUCCESS &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        require(coordinator.control_status() == CUDA_ERROR_UNKNOWN && fake.issued.empty() &&
                    fake.events.empty(),
                "control rejection must fail before native issue, not replay");
        require(queue.shutdown() == CUDA_SUCCESS, "unissued rejected queue must clean up");
        coordinator.stopped(queue, CUDA_SUCCESS);
    }
    Fake fake;
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaAdmissionCoordinator coordinator(gate);
    auto resources = std::make_shared<int>(0);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto low = queue.submit(stream, kernel(resources, std::make_shared<Continuation>()));
    require(coordinator.track(queue, low, {.relative_priority = 1}),
            "failed-drain low must register");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(low) == OrderedOperationState::kRunning; });
    const auto high = gate->submit({.relative_priority = 9}).get();
    require(high.has_value(), "external priority task must submit");
    fake.drain_error = CUDA_ERROR_INVALID_CONTEXT;
    const auto failed = queue.shutdown();
    coordinator.stopped(queue, failed);
    require(
        failed == CUDA_ERROR_INVALID_CONTEXT &&
            gate->claim(*high).get().status == glimmer::control::CooperativeClaimStatus::kPending,
        "failed physical drain cannot acknowledge stop or release execution capacity");
    fake.drain_error = CUDA_SUCCESS;
    require(queue.shutdown() == CUDA_SUCCESS, "actual physical recovery must drain");
    coordinator.stopped(queue, CUDA_SUCCESS);
    const auto granted = gate->claim(*high).get();
    require(granted.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                gate->complete(granted.execution).get(),
            "only real recovery permits capacity release and priority claim");
}

void reserved_native_priority_insertion(bool should_fail) {
    Fake fake;
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaAdmissionCoordinator coordinator(gate, {.clock_now = Fake::clock_now});
    auto resources = std::make_shared<int>(1);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto producer = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                            OrderedStreamKind::kNonBlocking, resources);
    const auto priority = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                            OrderedStreamKind::kNonBlocking, resources);
    auto continuation = std::make_shared<Continuation>();
    const auto low = queue.submit(producer, kernel(resources, continuation));
    const auto blocked = queue.submit(producer, CudaQueuedNativeReservation{resources});
    require(coordinator.track(queue, low, {.relative_priority = 1}) &&
                coordinator.track(queue, blocked, {.relative_priority = 99}),
            "blocked native priority must register without requesting a grant");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(low) == OrderedOperationState::kRunning; });
    auto owner = std::make_shared<int>(2);
    const std::weak_ptr<void> retained = owner;
    const auto high = queue.submit(priority, CudaQueuedNativeReservation{owner});
    owner.reset();
    const auto dependent = queue.submit(priority, CudaQueuedMarker{});
    require(high && dependent && coordinator.track(queue, high, {.relative_priority = 9}) &&
                queue.is_native_reservation(blocked) && !queue.native_reservation_is_ready(blocked),
            "reservation must retain invocation and same-stream dependent snapshot");
    coordinate_until(coordinator, queue, [&] { return continuation->requests == 1; });
    require(queue.query_native_reservation_boundary(high) == CUDA_ERROR_NOT_READY &&
                !coordinator.native_reservation_grant(queue, high) &&
                !coordinator.native_reservation_grant(queue, blocked) &&
                fake.issued == std::vector<char>{'k'},
            "retreat request is not permission for native insertion");
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return coordinator.native_reservation_grant(queue, high).has_value(); });
    const auto grant = *coordinator.native_reservation_grant(queue, high);
    require(queue.query_native_reservation_boundary(high) == CUDA_SUCCESS &&
                queue.state(low) == OrderedOperationState::kSuspended &&
                queue.state(high) == OrderedOperationState::kQueued &&
                queue.query(dependent) == CUDA_ERROR_NOT_READY && fake.issued.size() == 1,
            "physical-stop pause must retain suspended low without issuing the reservation");
    fake.now += std::chrono::milliseconds{100};
    coordinate_until(coordinator, queue,
                     [&] { return coordinator.stats().acknowledged_heartbeats >= 1; });
    auto native = std::make_unique<FakeEvent>();
    const auto event = reinterpret_cast<CUevent>(native.get());
    fake.events.push_back(std::move(native));
    require(!coordinator.commit_native_reservation(queue, high, {grant.task_id, 0},
                                                   {event, resources}) &&
                !coordinator.commit_native_reservation(queue, high, grant, {nullptr, resources}) &&
                queue.is_native_reservation(high) && !retained.expired(),
            "invalid handoff must retain reservation, resources and grant");
    // Simulate one caller-issued operation; pump must only import its fence.
    fake.issued.push_back('n');
    require(coordinator.commit_native_reservation(
                queue, high, grant, {event, resources},
                should_fail ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS) &&
                !coordinator.commit_native_reservation(queue, high, grant, {event, resources}) &&
                queue.native_fence_is_ready(high) && !retained.expired(),
            "same-token replacement must retain invocation and reject duplicate issue");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(high) == OrderedOperationState::kRunning; });
    require(queue.state(low) == OrderedOperationState::kSuspended &&
                queue.query(dependent) == CUDA_ERROR_NOT_READY &&
                fake.issued == std::vector<char>({'k', 'n', 'w'}) && queue.cancel_stopped(blocked),
            "native fence owns capacity until completion, not merely import");
    continuation->is_complete = true;
    fake.ready();
    coordinate_until(coordinator, queue, [&] { return coordinator.stats().logical_resumes == 1; });
    require(retained.expired() && queue.state(low) == OrderedOperationState::kRunning,
            "real native completion permits owner retirement and changed-generation resume");
    fake.ready();
    coordinate_until(coordinator, queue, [&] {
        const auto stats = coordinator.stats();
        return queue.query(dependent) == CUDA_SUCCESS &&
               stats.acknowledged_completions == (should_fail ? 1U : 2U) &&
               stats.acknowledged_failures == (should_fail ? 1U : 0U);
    });
    require(fake.issued == std::vector<char>({'k', 'n', 'w', 'k'}) &&
                coordinator.stats().granted_executions == 3,
            "native priority insertion must not replay original launch or admit blocked work");
    require(queue.shutdown() == CUDA_SUCCESS, "reserved native fixture must physically drain");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS, "reserved native ownership must settle");
}

void generation_bound_native_execution() {
    Fake fake;
    auto native = std::make_unique<FakeEvent>();
    native->is_user_event = true;
    const auto event = reinterpret_cast<CUevent>(native.get());
    fake.events.push_back(std::move(native));
    glimmer::control::LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>(options);
    CudaAdmissionCoordinator coordinator(gate, {.clock_now = Fake::clock_now});
    auto resources = std::make_shared<int>(1);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto other = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                         OrderedStreamKind::kNonBlocking, resources);
    const auto independent = queue.bind_stream(reinterpret_cast<CUstream>(0x5000),
                                               OrderedStreamKind::kNonBlocking, resources);
    const auto ordinary = queue.submit(other, kernel(resources));
    const auto blocked = queue.submit(other, CudaQueuedNativeFence{event, resources});
    const auto first = queue.submit(stream, CudaQueuedNativeFence{event, resources});
    const auto duplicate_task = queue.submit(independent, CudaQueuedNativeFence{event, resources});
    const auto task = gate->submit({.relative_priority = 1}).get();
    require(task.has_value(), "caller native task must submit on the shared gate");
    const auto granted = gate->claim(*task).get();
    require(granted.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                !coordinator.adopt_native_execution(queue, ordinary, granted.execution) &&
                !coordinator.adopt_native_execution(queue, blocked, granted.execution) &&
                !coordinator.adopt_native_execution(queue, first, {*task, 0}) &&
                coordinator.adopt_native_execution(queue, first, granted.execution) &&
                !coordinator.adopt_native_execution(queue, first, granted.execution) &&
                !coordinator.adopt_native_execution(queue, duplicate_task, granted.execution),
            "adoption requires one ready native fence and one exact nonduplicate generation");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(first) == OrderedOperationState::kRunning; });
    const auto heartbeats_before = coordinator.stats().acknowledged_heartbeats;
    fake.now += std::chrono::milliseconds{100};
    coordinate_until(coordinator, queue, [&] {
        return coordinator.stats().acknowledged_heartbeats == heartbeats_before + 1;
    });
    const auto high = gate->submit({.relative_priority = 9}).get();
    require(
        high &&
            gate->claim(*high).get().status == glimmer::control::CooperativeClaimStatus::kPending &&
            coordinator.stats().retreat_requests == 0 &&
            coordinator.stats().acknowledged_suspensions == 0,
        "caller-issued opaque execution is nonpreemptible and retains capacity");
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return coordinator.stats().acknowledged_completions == 1; });
    const auto high_grant = gate->claim(*high).get();
    require(high_grant.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                gate->complete(high_grant.execution).get() &&
                coordinator.stats().granted_executions == 1 &&
                std::ranges::find(fake.issued, 'k') == fake.issued.end(),
            "only actual native completion releases the original grant without a kernel replay");
    const auto failed_task = gate->submit({}).get();
    require(failed_task.has_value(), "failed native launch fixture must submit");
    const auto failed_grant = gate->claim(*failed_task).get();
    const auto failed_fence = queue.submit(stream, CudaQueuedNativeFence{event, resources});
    require(failed_grant.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                coordinator.adopt_native_execution(queue, failed_fence, failed_grant.execution,
                                                   CUDA_ERROR_LAUNCH_FAILED),
            "a native error still requires its completion fence before terminal reporting");
    coordinate_until(coordinator, queue,
                     [&] { return queue.state(failed_fence) == OrderedOperationState::kRunning; });
    fake.ready();
    coordinate_until(coordinator, queue,
                     [&] { return coordinator.stats().acknowledged_failures == 1; });
    require(coordinator.stats().acknowledged_completions == 1 &&
                coordinator.stats().acknowledged_failures == 1,
            "a stopped failed native launch must not be reported as successful execution");
    require(queue.shutdown() == CUDA_SUCCESS, "native execution queue must drain");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS,
            "adopted native terminal ownership must be acknowledged");
}

void native_execution_failed_prefix_proof() {
    Fake fake;
    auto native = std::make_unique<FakeEvent>();
    const auto event = reinterpret_cast<CUevent>(native.get());
    fake.events.push_back(std::move(native));
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaAdmissionCoordinator coordinator(gate);
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    auto resources = std::make_shared<int>(1);
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    const auto token = queue.submit(stream, CudaQueuedNativeFence{event, resources});
    const auto task = gate->submit({}).get();
    require(task.has_value(), "prefix proof fixture must submit");
    const auto execution = gate->claim(*task).get();
    require(execution.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                coordinator.adopt_native_execution(queue, token, execution.execution),
            "native caller must hand its granted execution to coordination");
    const auto high = gate->submit({.relative_priority = 9}).get();
    require(high.has_value(), "waiting external work must submit");
    fake.prefix_drain_error = CUDA_ERROR_INVALID_CONTEXT;
    const auto failed = queue.shutdown();
    coordinator.stopped(queue, failed);
    require(
        failed == CUDA_ERROR_INVALID_CONTEXT &&
            gate->claim(*high).get().status == glimmer::control::CooperativeClaimStatus::kPending,
        "unissued queue import cannot release already-issued native execution on failed proof");
    fake.prefix_drain_error = CUDA_SUCCESS;
    require(queue.shutdown() == CUDA_SUCCESS, "actual imported-prefix stop must prove recovery");
    coordinator.stopped(queue, CUDA_SUCCESS);
    const auto recovered = gate->claim(*high).get();
    require(recovered.status == glimmer::control::CooperativeClaimStatus::kGranted &&
                gate->complete(recovered.execution).get() && fake.issued.empty() &&
                coordinator.stats().acknowledged_failures == 1,
            "only native event stop proof authorizes capacity release without reissuing CUDA");
}

void failed_unsubmitted_dependency_reclaims_registration() {
    Fake fake;
    auto gate = std::make_shared<glimmer::control::AsyncCooperativeLaunchGate>();
    CudaAdmissionCoordinator coordinator(gate, {.pending_limit = 1});
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    auto resources = std::make_shared<int>(1);
    const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                          OrderedStreamKind::kNonBlocking, resources);
    std::uint32_t zero = 0;
    const auto initialize =
        queue.submit(stream, CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice, 0x6000, 0, &zero,
                                            sizeof(zero), resources});
    const auto kernel_token = queue.submit(stream, kernel(resources));
    require(initialize && kernel_token && coordinator.track(queue, kernel_token),
            "failed initialization fixture must register a dependent kernel");
    fake.create_error = CUDA_ERROR_OUT_OF_MEMORY;
    require(coordinator.progress(queue) == CUDA_SUCCESS &&
                queue.query(initialize) == CUDA_ERROR_OUT_OF_MEMORY &&
                queue.state(kernel_token) == OrderedOperationState::kFailed &&
                coordinator.stats().granted_executions == 0 && fake.issued.empty(),
            "failed predecessor must never enter control or GPU execution");
    fake.create_error = CUDA_SUCCESS;
    const auto other = queue.bind_stream(reinterpret_cast<CUstream>(0x4000),
                                         OrderedStreamKind::kNonBlocking, resources);
    const auto independent = queue.submit(other, kernel(resources));
    require(independent && coordinator.track(queue, independent),
            "failed unsubmitted metadata must not permanently consume registration capacity");
    require(queue.shutdown() == CUDA_SUCCESS, "unissued registration fixture must drain");
    coordinator.stopped(queue, CUDA_SUCCESS);
    require(coordinator.control_status() == CUDA_SUCCESS,
            "no admission was issued for the failed dependency or unissued teardown");
}

void unproven_shutdown_quarantines_resources() {
    const auto child = fork();
    require(child >= 0, "quarantine regression must create an isolated child");
    if (child == 0) {
        try {
            Fake fake;
            auto resources = std::make_shared<int>(0);
            std::weak_ptr<void> retained = resources;
            {
                CudaOperationQueue queue(fake.context, Fake::dispatch());
                const auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x3000),
                                                      OrderedStreamKind::kNonBlocking, resources);
                const auto work = queue.submit(stream, kernel(resources));
                require(work && queue.pump() == CUDA_SUCCESS, "quarantine work must issue once");
                fake.drain_error = CUDA_ERROR_UNKNOWN;
            }
            resources.reset();
            // _exit deliberately avoids process-exit leak reporting: this one
            // failure contract REQUIRES process-lifetime retention. Ordinary
            // recovery and ownership tests still run with leak detection.
            _exit(!retained.expired() && fake.drains == 1 && fake.issued.size() == 1 ? 0 : 1);
        } catch (...) {
            _exit(1);
        }
    }
    int status = 0;
    pid_t result = 0;
    do {
        result = waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);
    require(result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "failed destructor drain must quarantine, not release, live inputs");
}
}  // namespace

int main() {
    try {
        continuation_and_owned_operations();
        immutable_snapshots_and_default_order();
        implicit_completion_snapshots_retain_default_domains();
        failures_retain_physical_ownership();
        uncertain_event_cleanup_never_replays();
        rejection_context_and_lifetime();
        continuation_errors_and_typed_copies();
        bounded_history_reclamation();
        asynchronous_free_confirmation_and_ambiguity();
        native_event_generations_and_boundaries();
        guarded_deferred_event_records();
        cross_stream_rerecord_serialization();
        event_rejection_and_error_retention();
        controlled_kernel_admission();
        imported_native_fences();
        native_completion_errors();
        stopped_combined_submission_preserves_ownership();
        pending_claim_deadline_is_not_a_grant(true);
        pending_claim_deadline_is_not_a_grant(false);
        pending_claim_deadline_is_not_a_grant(true, true);
        pending_claim_deadline_is_not_a_grant(false, true);
        pending_claim_deadline_is_not_a_grant(true, false, true);
        native_claim_retry_options();
        false_retreat_response_backs_off();
        asynchronous_execution_coordination();
        coordination_failure_and_unproven_stop();
        generation_bound_native_execution();
        reserved_native_priority_insertion(false);
        reserved_native_priority_insertion(true);
        native_execution_failed_prefix_proof();
        failed_unsubmitted_dependency_reclaims_registration();
        unproven_shutdown_quarantines_resources();
        std::cout << "CUDA operation queue ownership and ordering checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
