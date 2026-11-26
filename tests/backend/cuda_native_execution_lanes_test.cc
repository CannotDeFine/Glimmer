#include "internal/cuda_native_execution_lanes.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using Phase = CudaNativeExecutionCallPhase;
constexpr auto kDeadline = std::chrono::seconds{5};

void expect(bool value, std::string_view message) {
    if (value) return;
    // A failed concurrency assertion must not unwind through a blocked worker's
    // referenced fixture storage. Successful cases still run normal teardown.
    std::cerr << "native lanes: " << message << '\n';
    std::_Exit(1);
}

template <typename Predicate>
void await(Predicate predicate, std::string_view message) {
    const auto deadline = std::chrono::steady_clock::now() + kDeadline;
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::yield();
    }
}

class Gate {
   public:
    void enter() {
        std::unique_lock lock(mutex_);
        is_entered_ = true;
        condition_.notify_all();
        expect(condition_.wait_for(lock, kDeadline, [&] { return is_open_; }),
               "bounded fake callback gate was released by its owner");
    }
    void wait_entered() {
        std::unique_lock lock(mutex_);
        expect(condition_.wait_for(lock, kDeadline, [&] { return is_entered_; }),
               "worker reached the independent callback handshake");
    }
    void open() {
        const std::lock_guard lock(mutex_);
        is_open_ = true;
        condition_.notify_all();
    }

   private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_entered_{}, is_open_{};
};

struct NativeContext {
    std::uint64_t id{};
    int device{};
};
struct Packet {
    int native_status{-1};
    std::uint64_t value{};
};
enum class NativeCall { kSet, kGet, kInvoke };
struct Visit {
    NativeCall call{};
    CUcontext context{};
    std::uint64_t incarnation{};
    unsigned step{};
};
enum class Fault {
    kNone,
    kSet,
    kPreGet,
    kPreMismatch,
    kCarrier,
    kThrow,
    kPostGet,
    kPostMismatch,
    kCarrierAndPost,
    kThrowAndPost,
};
struct Driver {
    std::mutex mutex;
    std::vector<Visit> visits;
    std::atomic<unsigned> calls{}, exits{};
    Fault fault{};
    CUcontext fault_context{};
    std::shared_ptr<Gate> set_gate, post_gate, exit_gate;

    void record(Visit entry) {
        const std::lock_guard lock(mutex);
        visits.push_back(entry);
        if (entry.call != NativeCall::kInvoke) ++calls;
    }
    std::vector<Visit> copy() {
        const std::lock_guard lock(mutex);
        return visits;
    }
};
std::shared_ptr<Driver> active_driver;
std::atomic<std::uint64_t> next_incarnation{1};
struct ThreadState {
    const std::uint64_t incarnation{next_incarnation.fetch_add(1)};
    CUcontext context{};
    unsigned gets{};
    int depth{};
    std::shared_ptr<Driver> driver;
    ~ThreadState() {
        if (!driver) return;
        if (driver->exit_gate) driver->exit_gate->enter();
        ++driver->exits;
    }
};
thread_local ThreadState thread_state;

CUresult CUDAAPI set_current(CUcontext context) {
    const auto driver = std::atomic_load(&active_driver);
    thread_state.driver = driver;
    thread_state.gets = 0;
    driver->record({NativeCall::kSet, context, thread_state.incarnation});
    if (context == driver->fault_context && driver->set_gate) driver->set_gate->enter();
    if (context == driver->fault_context && driver->fault == Fault::kSet)
        return CUDA_ERROR_INVALID_VALUE;
    thread_state.context = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* output) {
    const auto driver = thread_state.driver;
    ++thread_state.gets;
    driver->record({NativeCall::kGet, thread_state.context, thread_state.incarnation});
    *output = thread_state.context;
    if (thread_state.context != driver->fault_context) return CUDA_SUCCESS;
    const bool is_before = thread_state.gets == 1;
    if (!is_before && driver->post_gate) driver->post_gate->enter();
    if ((is_before && driver->fault == Fault::kPreGet) ||
        (!is_before &&
         (driver->fault == Fault::kPostGet || driver->fault == Fault::kCarrierAndPost ||
          driver->fault == Fault::kThrowAndPost)))
        return CUDA_ERROR_INVALID_HANDLE;
    if ((is_before && driver->fault == Fault::kPreMismatch) ||
        (!is_before && driver->fault == Fault::kPostMismatch))
        *output = nullptr;
    return CUDA_SUCCESS;
}
CudaNativeExecutionLaneDispatch dispatch() {
    return {.context_set_current = set_current, .context_get_current = get_current};
}
std::shared_ptr<Driver> fresh_driver() {
    auto driver = std::make_shared<Driver>();
    std::atomic_store(&active_driver, driver);
    return driver;
}
std::shared_ptr<NativeContext> context(NativeContext identity) {
    return std::make_shared<NativeContext>(identity);
}
CUcontext native(const std::shared_ptr<NativeContext>& value) {
    return reinterpret_cast<CUcontext>(value.get());
}
CudaNativeExecutionLaneKey key(std::uint64_t thread) {
    return {.peer = 7, .thread = {thread, 0}};
}
CudaNativeExecutionCall call(std::uint64_t sequence, const std::shared_ptr<NativeContext>& owner,
                             const std::shared_ptr<Packet>& packet,
                             std::function<CUresult()> invoke) {
    return {.sequence = sequence,
            .device_ordinal = owner->device,
            .context = {owner->id, native(owner), owner},
            .arguments_lifetime = packet,
            .invoke = std::move(invoke)};
}
CudaNativeExecutionTicket accepted(CudaNativeExecutionAcceptance result) {
    expect(result.status == CUDA_SUCCESS, "trusted native call is admitted");
    return std::move(result.ticket);
}
CudaNativeExecutionCallObservation terminal(const CudaNativeExecutionTicket& ticket) {
    await(
        [&] {
            const auto phase = ticket.observe().phase;
            return phase == Phase::kReturned || phase == Phase::kCancelled ||
                   phase == Phase::kUncertain;
        },
        "call reached a terminal host state without a fabricated GPU proof");
    return ticket.observe();
}
void returned(const CudaNativeExecutionTicket& ticket) {
    const auto observed = terminal(ticket);
    expect(observed.phase == Phase::kReturned && observed.status == CUDA_SUCCESS &&
               observed.was_invoked,
           "exact successful carrier return preserves actual invocation");
}
void stop(CudaNativeExecutionLanes& lanes) {
    expect(lanes.request_stop() == CUDA_SUCCESS, "owner requests stop without native cleanup");
    await([&] { return lanes.shutdown() == CUDA_SUCCESS; }, "independently exited workers join");
    expect(lanes.shutdown() == CUDA_SUCCESS && lanes.request_stop() == CUDA_SUCCESS,
           "completed shutdown and stop are idempotent");
}
void release(CudaNativeExecutionLanes& lanes, const CudaNativeExecutionTicket& ticket) {
    expect(lanes.release(ticket) == CUDA_SUCCESS, "owner explicitly releases known host storage");
    expect(lanes.release(ticket) == CUDA_ERROR_INVALID_VALUE,
           "duplicate release cannot refund another registration");
}

void tls_fifo_and_native_status() {
    const auto driver = fresh_driver();
    CudaNativeExecutionLanes lanes(dispatch());
    const auto first = context({1, 0}), second = context({2, 1}), independent = context({3, 2});
    auto packet = std::make_shared<Packet>();
    const auto gate = std::make_shared<Gate>();
    auto one = accepted(
        lanes.submit(key(11), call(1, first, packet, [=] {
                         driver->record({NativeCall::kInvoke, thread_state.context,
                                         thread_state.incarnation, 1});
                         expect(thread_state.context == native(first) && thread_state.depth == 0,
                                "first callback starts on its exact context and fresh real TLS");
                         thread_state.depth = 1;
                         gate->enter();
                         packet->native_status = 7;
                         return CUDA_SUCCESS;
                     })));
    gate->wait_entered();
    const auto entered = one.observe();
    expect(entered.phase == Phase::kEntered && entered.was_invoked &&
               entered.status == CUDA_ERROR_NOT_READY,
           "a blocked native callback is entered, not returned");
    auto two = accepted(
        lanes.submit(key(11), call(2, second, packet, [=] {
                         driver->record({NativeCall::kInvoke, thread_state.context,
                                         thread_state.incarnation, 2});
                         expect(thread_state.context == native(second) && thread_state.depth == 1,
                                "context switching preserves the lane's real native TLS");
                         thread_state.depth = 2;
                         return CUDA_SUCCESS;
                     })));
    auto three = accepted(
        lanes.submit(key(11), call(3, first, packet, [=] {
                         driver->record({NativeCall::kInvoke, thread_state.context,
                                         thread_state.incarnation, 3});
                         expect(thread_state.context == native(first) && thread_state.depth == 2,
                                "group-like callbacks retain FIFO order across devices");
                         thread_state.depth = 0;
                         return CUDA_SUCCESS;
                     })));
    const CudaNativeExecutionLaneKey other{.peer = 7, .thread = {0, 12}};
    auto four = accepted(lanes.submit(
        other, call(1, independent, packet, [=] {
            driver->record(
                {NativeCall::kInvoke, thread_state.context, thread_state.incarnation, 4});
            expect(thread_state.depth == 0, "different authenticated caller has independent TLS");
            return CUDA_SUCCESS;
        })));
    returned(four);
    const CudaNativeExecutionLaneKey foreign_peer{.peer = 8, .thread = {11, 0}};
    auto five = accepted(lanes.submit(
        foreign_peer, call(1, independent, packet, [=] {
            driver->record(
                {NativeCall::kInvoke, thread_state.context, thread_state.incarnation, 5});
            expect(thread_state.depth == 0,
                   "identical thread token belonging to another peer cannot share native TLS");
            return CUDA_SUCCESS;
        })));
    returned(five);
    expect(two.observe().phase == Phase::kPending && !two.observe().was_invoked &&
               three.observe().phase == Phase::kPending && lanes.retained_jobs() == 5,
           "owner remains responsive while the FIFO head is blocked");
    expect(lanes.release(one) == CUDA_ERROR_NOT_READY &&
               lanes.release(two) == CUDA_ERROR_NOT_READY &&
               lanes.shutdown() == CUDA_ERROR_NOT_READY,
           "pending/entered work cannot release and shutdown requires explicit stop");
    gate->open();
    returned(one);
    returned(two);
    returned(three);
    expect(packet->native_status == 7,
           "requested native InProgress is packet data, not a failed CUDA carrier");
    std::vector<Visit> bodies;
    bodies.reserve(5);
    for (const auto& entry : driver->copy())
        if (entry.call == NativeCall::kInvoke) bodies.push_back(entry);
    expect(bodies.size() == 5 && bodies[0].step == 1 && bodies[1].step == 4 &&
               bodies[2].step == 5 && bodies[3].step == 2 && bodies[4].step == 3 &&
               bodies[0].incarnation == bodies[3].incarnation &&
               bodies[0].incarnation == bodies[4].incarnation &&
               bodies[0].incarnation != bodies[1].incarnation &&
               bodies[0].incarnation != bodies[2].incarnation &&
               bodies[1].incarnation != bodies[2].incarnation,
           "actual thread-local identity, not transported labels, proves stable lanes");
    for (const auto& ticket : {one, two, three, four, five}) release(lanes, ticket);
    expect(lanes.retained_jobs() == 0, "all five explicit releases refund the exact quota");
    stop(lanes);
    expect(driver->exits == 3 && driver->calls == 15,
           "three real workers exit with exactly set/pre/post proofs and no cleanup SDK");
}

void device_exclusion_and_context_proofs() {
    const auto driver = fresh_driver();
    const auto first = context({10, 0}), same_device = context({11, 0}),
               other_device = context({12, 1});
    driver->fault_context = native(first);
    driver->set_gate = std::make_shared<Gate>();
    driver->post_gate = std::make_shared<Gate>();
    const auto body_gate = std::make_shared<Gate>();
    const auto packet = std::make_shared<Packet>();
    CudaNativeExecutionLanes lanes(dispatch());
    auto one = accepted(lanes.submit(key(1), call(1, first, packet, [=] {
                                         body_gate->enter();
                                         return CUDA_SUCCESS;
                                     })));
    driver->set_gate->wait_entered();
    expect(one.observe().phase == Phase::kEntered && !one.observe().was_invoked,
           "entered proof chain is distinguishable from requested callable invocation");
    auto two =
        accepted(lanes.submit(key(2), call(1, same_device, packet, [] { return CUDA_SUCCESS; })));
    auto three =
        accepted(lanes.submit(key(3), call(1, other_device, packet, [] { return CUDA_SUCCESS; })));
    returned(three);
    expect(two.observe().phase == Phase::kPending && !two.observe().was_invoked,
           "same-device contender cannot enter even while the first context proof blocks");
    driver->set_gate->open();
    body_gate->wait_entered();
    expect(two.observe().phase == Phase::kPending,
           "same-device native callbacks exclude one another across different lanes");
    body_gate->open();
    driver->post_gate->wait_entered();
    expect(one.observe().phase == Phase::kEntered && one.observe().was_invoked &&
               two.observe().phase == Phase::kPending,
           "device exclusion includes the post-call native context proof");
    for (const auto& entry : driver->copy())
        expect(entry.context != native(same_device),
               "waiting device contender has not made even a context setter callback");
    driver->post_gate->open();
    returned(one);
    returned(two);
    for (const auto& ticket : {one, two, three}) release(lanes, ticket);
    stop(lanes);
    expect(driver->calls == 9 && driver->exits == 3,
           "independent device made progress without extra SDK operations");
}

void validation_and_limits() {
    const auto driver = fresh_driver();
    for (unsigned index = 0; index < 6; ++index) {
        auto sdk = dispatch();
        CudaNativeExecutionLaneLimits limits;
        if (index == 0) sdk.context_set_current = nullptr;
        if (index == 1) sdk.context_get_current = nullptr;
        if (index == 2) limits.maximum_lanes = 0;
        if (index == 3) limits.maximum_jobs = 0;
        if (index == 4) limits.maximum_jobs_per_lane = 0;
        if (index == 5) limits.maximum_devices = 0;
        bool rejected{};
        try {
            CudaNativeExecutionLanes invalid(sdk, limits);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid dispatch and zero limits reject before native work");
    }
    const auto owner = context({20, 0}), other = context({21, 1});
    const auto packet = std::make_shared<Packet>();
    CudaNativeExecutionLanes lanes(dispatch(), {2, 3, 2});
    for (unsigned index = 0; index < 10; ++index) {
        auto request = call(1, owner, packet, [] { return CUDA_SUCCESS; });
        auto identity = key(1);
        if (index == 0) identity.peer = 0;
        if (index == 1) identity.thread = {};
        if (index == 2) request.sequence = 0;
        if (index == 3) request.sequence = 2;
        if (index == 4) request.context.native = nullptr;
        if (index == 5) request.context.id = 0;
        if (index == 6) request.context.lifetime.reset();
        if (index == 7) request.arguments_lifetime.reset();
        if (index == 8) request.invoke = {};
        if (index == 9) request.device_ordinal = -1;
        expect(lanes.submit(identity, std::move(request)).status == CUDA_ERROR_INVALID_VALUE,
               "malformed key/call/context has no admission or native side effect");
    }
    expect(driver->calls == 0 && lanes.retained_jobs() == 0,
           "invalid requests do not allocate a live native job");
    const auto gate = std::make_shared<Gate>();
    auto one = accepted(lanes.submit(key(1), call(1, owner, packet, [=] {
                                         gate->enter();
                                         return CUDA_SUCCESS;
                                     })));
    gate->wait_entered();
    expect(lanes.submit(key(1), call(1, owner, packet, [] { return CUDA_SUCCESS; })).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               lanes.submit(key(1), call(3, owner, packet, [] { return CUDA_SUCCESS; })).status ==
                   CUDA_ERROR_INVALID_VALUE,
           "duplicate and skipped sequences reject without advancing FIFO identity");
    auto two = accepted(lanes.submit(key(1), call(2, owner, packet, [] { return CUDA_SUCCESS; })));
    expect(lanes.submit(key(1), call(3, owner, packet, [] { return CUDA_SUCCESS; })).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "per-lane bound includes its entered and pending jobs");
    auto three =
        accepted(lanes.submit(key(2), call(1, other, packet, [] { return CUDA_SUCCESS; })));
    returned(three);
    expect(lanes.submit(key(2), call(2, other, packet, [] { return CUDA_SUCCESS; })).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "returned but unreleased job still consumes the global bound");
    release(lanes, three);
    auto four = accepted(lanes.submit(key(2), call(2, other, packet, [] { return CUDA_SUCCESS; })));
    returned(four);
    release(lanes, four);
    expect(lanes.submit(key(3), call(1, other, packet, [] { return CUDA_SUCCESS; })).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "an empty stable lane is not recycled into a new caller incarnation");
    gate->open();
    returned(one);
    returned(two);
    release(lanes, one);
    release(lanes, two);
    auto five = accepted(lanes.submit(key(1), call(3, owner, packet, [] { return CUDA_SUCCESS; })));
    returned(five);
    release(lanes, five);
    stop(lanes);
    CudaNativeExecutionLanes independent_limits(dispatch(), {1, 1, 9});
    stop(independent_limits);
    expect(lanes.retained_jobs() == 0,
           "per-lane bound may exceed global bound; all limits remain independent");

    CudaNativeExecutionLanes device_bound(dispatch(), {2, 3, 3, 1});
    auto known =
        accepted(device_bound.submit(key(1), call(1, owner, packet, [] { return CUDA_SUCCESS; })));
    returned(known);
    release(device_bound, known);
    expect(
        device_bound.submit(key(1), call(2, other, packet, [] { return CUDA_SUCCESS; })).status ==
            CUDA_ERROR_OUT_OF_MEMORY,
        "distinct device registry stays bounded even after its jobs are released");
    auto reused =
        accepted(device_bound.submit(key(1), call(2, owner, packet, [] { return CUDA_SUCCESS; })));
    returned(reused);
    release(device_bound, reused);
    stop(device_bound);
}

void ticket_storage_and_owner_boundaries() {
    const auto driver = fresh_driver();
    CudaNativeExecutionLanes lanes(dispatch()), foreign(dispatch());
    const auto absent = CudaNativeExecutionTicket{}.observe();
    expect(absent.phase == Phase::kCancelled && absent.status == CUDA_ERROR_INVALID_VALUE &&
               !absent.was_invoked,
           "default ticket has no native result or registration authority");
    auto owner = context({30, 0});
    auto packet = std::make_shared<Packet>();
    const std::weak_ptr<NativeContext> weak_context = owner;
    const std::weak_ptr<Packet> weak_packet = packet;
    auto ticket = accepted(lanes.submit(key(1), call(1, owner, packet, [packet] {
                                            packet->value = 123;
                                            return CUDA_SUCCESS;
                                        })));
    returned(ticket);
    auto copy = ticket;
    owner.reset();
    packet.reset();
    expect(!weak_context.expired() && !weak_packet.expired() && lanes.retained_jobs() == 1,
           "host return and ticket copies do not release immutable argument/context leases");
    expect(foreign.release(ticket) == CUDA_ERROR_INVALID_VALUE &&
               lanes.release({}) == CUDA_ERROR_INVALID_VALUE,
           "foreign and empty tickets cannot release authenticated jobs");
    std::thread wrong([&] {
        expect(copy.observe().phase == Phase::kReturned,
               "ticket observation is permitted on another thread");
        expect(lanes.request_stop() == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.shutdown() == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.release(copy) == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.submit(key(2), {}).status == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.retained_jobs() == 0,
               "every owner operation rejects wrong thread before SDK or locks");
    });
    wrong.join();
    const auto child = ::fork();
    expect(child >= 0, "fork owner-boundary child");
    if (child == 0) {
        const auto inherited = copy.observe();
        expect(inherited.phase == Phase::kUncertain &&
                   inherited.status == CUDA_ERROR_INVALID_CONTEXT && !inherited.was_invoked,
               "forked ticket rejects before inherited synchronization state");
        expect(lanes.submit(key(2), {}).status == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.release(copy) == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.request_stop() == CUDA_ERROR_INVALID_CONTEXT &&
                   lanes.shutdown() == CUDA_ERROR_INVALID_CONTEXT && lanes.retained_jobs() == 0,
               "fork rejects before inherited mutex/native ownership");
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated fork ownership rejection succeeds");
    release(lanes, ticket);
    ticket = {};
    copy = {};
    expect(weak_context.expired() && weak_packet.expired(),
           "explicit known release and final ticket drop release only CPU storage");
    stop(lanes);
    stop(foreign);
    expect(driver->calls == 3 && driver->exits == 1,
           "foreign/thread/fork controls perform no native callback");
}

void stop_cancels_only_unentered() {
    const auto driver = fresh_driver();
    CudaNativeExecutionLanes lanes(dispatch());
    const auto owner = context({40, 0}), waiting_owner = context({41, 0});
    const auto packet = std::make_shared<Packet>();
    const auto gate = std::make_shared<Gate>();
    auto one = accepted(lanes.submit(key(1), call(1, owner, packet, [=] {
                                         gate->enter();
                                         return CUDA_SUCCESS;
                                     })));
    gate->wait_entered();
    auto two = accepted(
        lanes.submit(key(1), call(2, owner, packet, [] {
                         expect(false, "stop must cancel queued same-lane call before entry");
                         return CUDA_SUCCESS;
                     })));
    auto three =
        accepted(lanes.submit(key(2), call(1, waiting_owner, packet, [] {
                                  expect(false, "stop must cancel device-gate waiter before entry");
                                  return CUDA_SUCCESS;
                              })));
    expect(lanes.request_stop() == CUDA_SUCCESS && lanes.request_stop() == CUDA_SUCCESS,
           "stop never waits for the running original callback");
    for (const auto& ticket : {two, three}) {
        const auto observed = terminal(ticket);
        expect(observed.phase == Phase::kCancelled && !observed.was_invoked,
               "only never-entered jobs become known cancelled");
        release(lanes, ticket);
    }
    expect(one.observe().phase == Phase::kEntered && one.observe().was_invoked &&
               lanes.shutdown() == CUDA_ERROR_NOT_READY &&
               lanes.release(one) == CUDA_ERROR_NOT_READY,
           "running callback remains retained and shutdown remains nonblocking");
    expect(lanes.submit(key(1), call(3, owner, packet, [] { return CUDA_SUCCESS; })).status ==
               CUDA_ERROR_DEINITIALIZED,
           "stop closes admission without another native call");
    gate->open();
    returned(one);
    release(lanes, one);
    stop(lanes);
    expect(driver->calls == 3, "cancelled jobs never select or query any context");
}

void callback_cannot_reenter_owner() {
    const auto driver = fresh_driver();
    CudaNativeExecutionLanes lanes(dispatch());
    const auto owner = context({50, 0});
    const auto packet = std::make_shared<Packet>();
    auto ticket = accepted(lanes.submit(
        key(1), call(1, owner, packet, [&] {
            expect(lanes.submit(key(2), {}).status == CUDA_ERROR_INVALID_CONTEXT &&
                       lanes.release({}) == CUDA_ERROR_INVALID_CONTEXT &&
                       lanes.request_stop() == CUDA_ERROR_INVALID_CONTEXT &&
                       lanes.shutdown() == CUDA_ERROR_INVALID_CONTEXT && lanes.retained_jobs() == 0,
                   "native worker cannot reenter owner admission, cleanup or shutdown");
            return CUDA_SUCCESS;
        })));
    returned(ticket);
    release(lanes, ticket);
    stop(lanes);
    expect(driver->calls == 3, "worker reentry produces no nested native callbacks");
}

void independent_thread_exit() {
    const auto driver = fresh_driver();
    driver->exit_gate = std::make_shared<Gate>();
    CudaNativeExecutionLanes lanes(dispatch());
    const auto owner = context({60, 0});
    const auto packet = std::make_shared<Packet>();
    auto ticket =
        accepted(lanes.submit(key(1), call(1, owner, packet, [] { return CUDA_SUCCESS; })));
    returned(ticket);
    release(lanes, ticket);
    expect(lanes.request_stop() == CUDA_SUCCESS, "request native worker termination");
    driver->exit_gate->wait_entered();
    expect(
        driver->exits == 0 && lanes.shutdown() == CUDA_ERROR_NOT_READY,
        "body return is not thread exit: a blocked SDK TLS destructor must not block owner join");
    driver->exit_gate->open();
    stop(lanes);
    expect(driver->exits == 1 && driver->calls == 3,
           "actual exit confirmation precedes successful nonblocking join");
}

CUresult fault_status(Fault fault) {
    switch (fault) {
        case Fault::kSet:
        case Fault::kCarrier:
        case Fault::kCarrierAndPost:
            return CUDA_ERROR_INVALID_VALUE;
        case Fault::kPreGet:
        case Fault::kPostGet:
            return CUDA_ERROR_INVALID_HANDLE;
        case Fault::kPreMismatch:
        case Fault::kPostMismatch:
            return CUDA_ERROR_INVALID_CONTEXT;
        case Fault::kThrow:
        case Fault::kThrowAndPost:
            return CUDA_ERROR_UNKNOWN;
        case Fault::kNone:
            return CUDA_SUCCESS;
    }
    return CUDA_ERROR_UNKNOWN;
}

void uncertainty(Fault fault) {
    const auto driver = fresh_driver();
    auto owner = context({70, 0});
    const auto same_device = context({71, 0}), independent = context({72, 1});
    auto packet = std::make_shared<Packet>();
    const auto other_packet = std::make_shared<Packet>();
    const std::weak_ptr<NativeContext> weak_context = owner;
    const std::weak_ptr<Packet> weak_packet = packet;
    driver->fault = fault;
    driver->fault_context = native(owner);
    driver->set_gate = std::make_shared<Gate>();
    auto lanes = std::make_unique<CudaNativeExecutionLanes>(dispatch());
    auto one = accepted(
        lanes->submit(key(1), call(1, owner, packet, [packet, fault] {
                          packet->native_status = 6;
                          if (fault == Fault::kThrow || fault == Fault::kThrowAndPost)
                              throw std::runtime_error("injected native-call carrier throw");
                          return fault == Fault::kCarrier || fault == Fault::kCarrierAndPost
                                     ? CUDA_ERROR_INVALID_VALUE
                                     : CUDA_SUCCESS;
                      })));
    driver->set_gate->wait_entered();
    auto two = accepted(lanes->submit(
        key(1), call(2, same_device, other_packet, [] {
            expect(false, "unknown device must not invoke a queued same-lane callback");
            return CUDA_SUCCESS;
        })));
    auto three = accepted(lanes->submit(
        key(2), call(1, same_device, other_packet, [] {
            expect(false, "unknown device must not invoke another lane's queued callback");
            return CUDA_SUCCESS;
        })));
    driver->set_gate->open();
    const auto observed = terminal(one);
    const bool was_invoked = fault == Fault::kCarrier || fault == Fault::kThrow ||
                             fault == Fault::kPostGet || fault == Fault::kPostMismatch ||
                             fault == Fault::kCarrierAndPost || fault == Fault::kThrowAndPost;
    expect(observed.phase == Phase::kUncertain && observed.status == fault_status(fault) &&
               observed.was_invoked == was_invoked &&
               packet->native_status == (was_invoked ? 6 : -1),
           "proof/carrier failures preserve exact status and separate requested native provenance");
    for (const auto& ticket : {two, three}) {
        const auto cancelled = terminal(ticket);
        expect(cancelled.phase == Phase::kCancelled && !cancelled.was_invoked &&
                   cancelled.status == fault_status(fault),
               "sticky device failure cancels only known-unentered jobs with its exact error");
        release(*lanes, ticket);
    }
    const auto before_cancelled = driver->calls.load();
    auto late = accepted(lanes->submit(
        key(1), call(3, same_device, other_packet, [] {
            expect(false, "already quarantined device cannot invoke a later valid request");
            return CUDA_SUCCESS;
        })));
    const auto late_observation = late.observe();
    expect(late_observation.phase == Phase::kCancelled && !late_observation.was_invoked &&
               late_observation.status == fault_status(fault) && driver->calls == before_cancelled,
           "quarantined device admits exact next sequence as immediate CPU-only cancellation");
    release(*lanes, late);
    auto four = accepted(
        lanes->submit(key(3), call(1, independent, other_packet, [] { return CUDA_SUCCESS; })));
    returned(four);
    release(*lanes, four);
    const unsigned failing_callbacks = fault == Fault::kSet ? 1U : was_invoked ? 3U : 2U;
    expect(driver->calls == failing_callbacks + 3U,
           "post-context proof still runs after carrier error/throw, first failure remains exact");
    expect(lanes->release(one) == CUDA_ERROR_NOT_READY && lanes->retained_jobs() == 1,
           "uncertain job is never releasable or discounted");
    for (const auto& entry : driver->copy())
        expect(entry.context != native(same_device),
               "failed device never receives another SDK entry");
    stop(*lanes);
    owner.reset();
    packet.reset();
    one = {};
    const auto calls = driver->calls.load();
    lanes.reset();
    expect(driver->calls == calls && !weak_context.expired() && !weak_packet.expired(),
           "unknown job keeps context/call packet after shutdown and destructor without SDK retry");
    // Deliberate uncertainty is process-retained, not a leak-free recovery claim.
}

void destructor_does_not_block_or_release(bool has_returned) {
    const auto driver = fresh_driver();
    auto owner = context({80, 0});
    auto packet = std::make_shared<Packet>();
    const std::weak_ptr<NativeContext> weak_context = owner;
    const std::weak_ptr<Packet> weak_packet = packet;
    const auto gate = std::make_shared<Gate>();
    auto lanes = std::make_unique<CudaNativeExecutionLanes>(dispatch());
    auto ticket = accepted(lanes->submit(key(1), call(1, owner, packet, [=] {
                                             if (!has_returned) gate->enter();
                                             return CUDA_SUCCESS;
                                         })));
    if (has_returned) {
        returned(ticket);
        stop(*lanes);
    } else {
        gate->wait_entered();
    }
    owner.reset();
    packet.reset();
    ticket = {};
    const auto calls = driver->calls.load();
    lanes.reset();
    expect(driver->calls == calls && !weak_context.expired() && !weak_packet.expired(),
           "destructor neither joins a blocked callback nor implicitly releases unacknowledged "
           "host storage");
    // A live callback is intentionally left blocked until this child exits.
}

void multi_device_atomic_exclusion() {
    const auto driver = fresh_driver();
    const auto zero = context({100, 0}), one = context({101, 1}), two = context({102, 2});
    const auto packet = std::make_shared<Packet>();
    const auto gate = std::make_shared<Gate>();
    CudaNativeExecutionLanes lanes(dispatch());
    auto request = call(1, zero, packet, [gate] {
        gate->enter();
        return CUDA_SUCCESS;
    });
    request.device_ordinals = {1, 0};
    auto held = accepted(lanes.submit(key(1), std::move(request)));
    gate->wait_entered();
    auto blocked_zero =
        accepted(lanes.submit(key(2), call(1, zero, packet, [] { return CUDA_SUCCESS; })));
    auto blocked_one =
        accepted(lanes.submit(key(3), call(1, one, packet, [] { return CUDA_SUCCESS; })));
    auto independent =
        accepted(lanes.submit(key(4), call(1, two, packet, [] { return CUDA_SUCCESS; })));
    returned(independent);
    expect(blocked_zero.observe().phase == Phase::kPending &&
               blocked_one.observe().phase == Phase::kPending,
           "one multi-device callback excludes every member while an outside device progresses");
    for (const auto& entry : driver->copy())
        expect(entry.context != native(one),
               "additional reserved device cannot even select context");
    gate->open();
    for (const auto& ticket : {held, blocked_zero, blocked_one, independent}) {
        returned(ticket);
        release(lanes, ticket);
    }
    stop(lanes);
    expect(driver->calls == 12 && driver->exits == 4,
           "multi-device permission adds no fabricated SDK calls or context switches");
}

void multi_device_no_partial_acquisition() {
    const auto driver = fresh_driver();
    const auto zero = context({110, 0}), one = context({111, 1});
    const auto packet = std::make_shared<Packet>();
    const auto gate = std::make_shared<Gate>();
    CudaNativeExecutionLanes lanes(dispatch());
    auto held = accepted(lanes.submit(key(1), call(1, one, packet, [gate] {
                                          gate->enter();
                                          return CUDA_SUCCESS;
                                      })));
    gate->wait_entered();
    auto request = call(1, zero, packet, [] {
        expect(false, "stopped waiting device set never enters its callable");
        return CUDA_SUCCESS;
    });
    request.device_ordinals = {0, 1};
    auto waiting = accepted(lanes.submit(key(2), std::move(request)));
    auto available_zero =
        accepted(lanes.submit(key(3), call(1, zero, packet, [] { return CUDA_SUCCESS; })));
    returned(available_zero);
    expect(waiting.observe().phase == Phase::kPending && !waiting.observe().was_invoked,
           "waiting for device1 does not partially reserve otherwise-free device0");
    expect(lanes.request_stop() == CUDA_SUCCESS && lanes.shutdown() == CUDA_ERROR_NOT_READY,
           "stop stays responsive with a blocked member and a pending device set");
    const auto cancelled = waiting.observe();
    expect(cancelled.phase == Phase::kCancelled && !cancelled.was_invoked,
           "whole unentered set cancels without selecting a context");
    release(lanes, waiting);
    release(lanes, available_zero);
    gate->open();
    returned(held);
    release(lanes, held);
    stop(lanes);
    expect(driver->calls == 6, "only the two actual singleton calls reached SDK context proof");
}

void multi_device_validation_and_quota() {
    const auto driver = fresh_driver();
    const auto zero = context({120, 0}), one = context({121, 1});
    const auto packet = std::make_shared<Packet>();
    CudaNativeExecutionLanes lanes(dispatch(), {4, 8, 8, 2});
    for (const auto& devices : std::vector<std::vector<int>>{{1}, {0, -1}, {0, 0}, {0, 1, 2}}) {
        auto request = call(1, zero, packet, [] { return CUDA_SUCCESS; });
        request.device_ordinals = devices;
        expect(
            lanes.submit(key(1), std::move(request)).status == CUDA_ERROR_INVALID_VALUE,
            "device sets reject missing selected device, negatives, duplicates and raw overflow");
    }
    expect(driver->calls == 0 && lanes.retained_jobs() == 0,
           "invalid device sets consume neither a sequence nor SDK authority");
    auto first = call(1, zero, packet, [] { return CUDA_SUCCESS; });
    first.device_ordinals = {1, 0};
    auto accepted_first = accepted(lanes.submit(key(1), std::move(first)));
    returned(accepted_first);
    release(lanes, accepted_first);
    auto oversized = call(2, zero, packet, [] { return CUDA_SUCCESS; });
    oversized.device_ordinals = {0, 2};
    expect(lanes.submit(key(1), std::move(oversized)).status == CUDA_ERROR_OUT_OF_MEMORY,
           "all new device identities are preflighted against lifetime capacity");
    auto reused = call(2, one, packet, [] { return CUDA_SUCCESS; });
    reused.device_ordinals = {0, 1};
    auto accepted_reuse = accepted(lanes.submit(key(1), std::move(reused)));
    returned(accepted_reuse);
    release(lanes, accepted_reuse);
    stop(lanes);
    expect(driver->calls == 6, "rejected aggregate capacity did not advance sequence or call SDK");
}

void multi_device_quarantine_scope() {
    const auto driver = fresh_driver();
    const auto zero = context({130, 0}), one = context({131, 1}), two = context({132, 2}),
               three = context({133, 3});
    auto packet = std::make_shared<Packet>();
    driver->fault = Fault::kPostMismatch;
    driver->fault_context = native(zero);
    driver->set_gate = std::make_shared<Gate>();
    auto lanes = std::make_unique<CudaNativeExecutionLanes>(dispatch());
    auto request = call(1, zero, packet, [] { return CUDA_SUCCESS; });
    request.device_ordinals = {0, 1};
    auto uncertain = accepted(lanes->submit(key(1), std::move(request)));
    driver->set_gate->wait_entered();
    auto superset = call(1, two, packet, [] {
        expect(false, "intersecting unentered set is cancelled rather than replayed");
        return CUDA_SUCCESS;
    });
    superset.device_ordinals = {1, 2};
    auto cancelled = accepted(lanes->submit(key(2), std::move(superset)));
    driver->set_gate->open();
    const auto failed = terminal(uncertain);
    expect(failed.phase == Phase::kUncertain && failed.status == CUDA_ERROR_INVALID_CONTEXT,
           "unknown callback quarantines its entire entered device set");
    expect(terminal(cancelled).phase == Phase::kCancelled && !cancelled.observe().was_invoked,
           "unentered intersecting set is only cancelled");
    release(*lanes, cancelled);
    for (const auto& owner : {zero, one}) {
        auto rejected =
            accepted(lanes->submit(key(owner->id), call(1, owner, packet, [] {
                                       expect(false, "every entered device remains quarantined");
                                       return CUDA_SUCCESS;
                                   })));
        expect(rejected.observe().phase == Phase::kCancelled,
               "primary and additional entered device both reject native entry");
        release(*lanes, rejected);
    }
    auto healthy =
        accepted(lanes->submit(key(3), call(1, two, packet, [] { return CUDA_SUCCESS; })));
    returned(healthy);
    release(*lanes, healthy);
    auto mixed = call(1, three, packet, [] {
        expect(false, "future intersecting set also cancels before SDK");
        return CUDA_SUCCESS;
    });
    mixed.device_ordinals = {0, 3};
    auto future_cancelled = accepted(lanes->submit(key(4), std::move(mixed)));
    expect(future_cancelled.observe().phase == Phase::kCancelled,
           "a future set intersecting a quarantined device cancels CPU-only");
    release(*lanes, future_cancelled);
    auto still_healthy =
        accepted(lanes->submit(key(5), call(1, three, packet, [] { return CUDA_SUCCESS; })));
    returned(still_healthy);
    release(*lanes, still_healthy);
    expect(driver->calls == 9 && lanes->release(uncertain) == CUDA_ERROR_NOT_READY,
           "cancellation never spreads quarantine to unentered healthy devices");
    stop(*lanes);
    const std::weak_ptr<Packet> retained = packet;
    packet.reset();
    uncertain = {};
    const auto count = driver->calls.load();
    lanes.reset();
    expect(driver->calls == count && !retained.expired(),
           "uncertain multi-device ownership is retained without cleanup SDK calls");
}

template <typename Function>
void isolated(Function body, std::string_view message) {
    const auto child = ::fork();
    expect(child >= 0, "create isolated permanent-retention process");
    if (child == 0) {
        body();
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           message);
}
}  // namespace

int main() {
    tls_fifo_and_native_status();
    device_exclusion_and_context_proofs();
    validation_and_limits();
    ticket_storage_and_owner_boundaries();
    stop_cancels_only_unentered();
    callback_cannot_reenter_owner();
    independent_thread_exit();
    multi_device_atomic_exclusion();
    multi_device_no_partial_acquisition();
    multi_device_validation_and_quota();
    for (const auto fault :
         {Fault::kSet, Fault::kPreGet, Fault::kPreMismatch, Fault::kCarrier, Fault::kThrow,
          Fault::kPostGet, Fault::kPostMismatch, Fault::kCarrierAndPost, Fault::kThrowAndPost})
        isolated([=] { uncertainty(fault); }, "isolated exact native/proof failure");
    isolated([] { destructor_does_not_block_or_release(false); }, "blocked destructor retention");
    isolated([] { destructor_does_not_block_or_release(true); }, "unreleased returned storage");
    isolated(multi_device_quarantine_scope,
             "entered-set quarantine does not spread on cancellation");
    std::cout << "native execution lanes: 10 healthy groups, 12 isolated retained-owner cases; "
                 "GPU_initialized=false\n";
    return 0;
}
