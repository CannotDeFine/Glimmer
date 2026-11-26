#include "internal/cuda_external_execution_bridge.h"
#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;

void expect(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void complete(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "independent source setup/cleanup succeeds");
}
void deferred(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kDeferred &&
               result.status == CUDA_ERROR_NOT_READY && !result.original_status,
           "retained physical owner cannot be cleaned or given original provenance");
}

enum class Call { kContext, kCreate, kRecord, kWait, kQuery, kDestroy, kWork };
struct Context {
    bool alive{};
    int device{};
};
struct Stream {
    bool alive{};
    CUcontext context{};
};
struct Event {
    bool alive{}, recorded{}, ready{};
    CUcontext context{};
    CUstream stream{};
};
struct Module {
    bool alive{};
    CUcontext context{};
    std::array<std::byte, 32> global{};
    char function{};
};
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
struct Entry {
    Call call{};
    CUcontext context{};
    CUstream stream{};
    CUevent event{};
    std::thread::id thread;
};
thread_local CUcontext current_context{};
struct Driver {
    std::array<Context, 4> contexts{};
    std::array<Stream, 8> streams{};
    std::array<Event, 16> events{};
    std::array<Module, 4> modules{};
    std::map<CUdeviceptr, Memory> memory;
    std::vector<Entry> entries;
    std::size_t next_context{}, next_stream{}, next_event{}, next_module{};
    std::atomic<unsigned> calls{};
    unsigned releases{}, unloads{}, stream_destroys{}, context_destroys{}, synchronizes{};
    std::function<CUresult(Call)> injection;
    bool partial_create{}, null_create{};
    CUevent alias_create{};

    CUresult visit(Call call, CUstream stream = nullptr, CUevent event = nullptr) {
        ++calls;
        entries.push_back({call, current_context, stream, event, std::this_thread::get_id()});
        return injection ? injection(call) : CUDA_SUCCESS;
    }
    void reset() {
        contexts = {};
        streams = {};
        events = {};
        modules = {};
        memory.clear();
        entries.clear();
        injection = {};
        next_context = next_stream = next_event = next_module = 0;
        calls = releases = unloads = stream_destroys = context_destroys = synchronizes = 0;
        partial_create = null_create = false;
        alias_create = nullptr;
        current_context = nullptr;
    }
} driver;

CUresult CUDAAPI get_current(CUcontext* output) {
    const auto result = driver.visit(Call::kContext);
    *output = current_context;
    return result;
}
CUresult CUDAAPI create_event(CUevent* output, unsigned flags) {
    expect(flags == CU_EVENT_DISABLE_TIMING && current_context,
           "private bridge event is untimed and belongs to the current domain");
    const auto result = driver.visit(Call::kCreate);
    if (driver.alias_create) {
        *output = driver.alias_create;
        driver.entries.back().event = *output;
        return result;
    }
    if (!driver.null_create && (result == CUDA_SUCCESS || driver.partial_create)) {
        expect(driver.next_event < driver.events.size(), "bounded fake event inventory");
        auto& event = driver.events[driver.next_event++];
        event = {.alive = true, .context = current_context};
        *output = reinterpret_cast<CUevent>(&event);
        driver.entries.back().event = *output;
    }
    return result;
}
CUresult CUDAAPI record_event(CUevent handle, CUstream stream) {
    auto& event = *reinterpret_cast<Event*>(handle);
    expect(event.alive && !event.recorded && event.context == current_context &&
               (!stream || reinterpret_cast<Stream*>(stream)->context == current_context),
           "event recording never crosses contexts or replays an event");
    const auto result = driver.visit(Call::kRecord, stream, handle);
    if (result == CUDA_SUCCESS) {
        event.recorded = true;
        event.stream = stream;
    }
    return result;
}
CUresult CUDAAPI wait_event(CUstream stream, CUevent handle, unsigned flags) {
    const auto& event = *reinterpret_cast<Event*>(handle);
    expect(flags == 0 && event.alive && event.recorded &&
               (!stream || reinterpret_cast<Stream*>(stream)->context == current_context),
           "cross-context dependency waits only on a genuinely recorded live event");
    return driver.visit(Call::kWait, stream, handle);
}
CUresult CUDAAPI query_event(CUevent handle) {
    const auto& event = *reinterpret_cast<Event*>(handle);
    expect(event.alive && event.recorded && event.context == current_context,
           "only actual recorded source tail may be queried");
    const auto result = driver.visit(Call::kQuery, event.stream, handle);
    return result != CUDA_SUCCESS ? result : event.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI destroy_event(CUevent handle) {
    auto& event = *reinterpret_cast<Event*>(handle);
    expect(event.alive && event.context == current_context,
           "private event is destroyed once on its own native context");
    const auto result = driver.visit(Call::kDestroy, event.stream, handle);
    if (result == CUDA_SUCCESS) event.alive = false;
    return result;
}
CudaExternalExecutionBridgeDispatch bridge_dispatch() {
    return {get_current, create_event, record_event, destroy_event, wait_event};
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact fake CUDA ABI callbacks.
CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch result;
    result.initialize = [](unsigned) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    result.device_get = [](CUdevice* output, int device) {
        ++driver.calls;
        *output = device;
        return CUDA_SUCCESS;
    };
    result.context_create = [](CUcontext* output, unsigned, CUdevice device) {
        ++driver.calls;
        expect(driver.next_context < driver.contexts.size(), "bounded context inventory");
        auto& context = driver.contexts[driver.next_context++];
        context = {true, device};
        current_context = *output = reinterpret_cast<CUcontext>(&context);
        return CUDA_SUCCESS;
    };
    result.context_set_current = [](CUcontext context) {
        ++driver.calls;
        current_context = context;
        return CUDA_SUCCESS;
    };
    result.context_destroy = [](CUcontext context) {
        ++driver.calls;
        auto& value = *reinterpret_cast<Context*>(context);
        expect(value.alive && context == current_context,
               "context cleanup uses exact live source owner");
        value.alive = false;
        ++driver.context_destroys;
        current_context = nullptr;
        return CUDA_SUCCESS;
    };
    result.module_load = [](CUmodule* output, const void*) {
        ++driver.calls;
        expect(driver.next_module < driver.modules.size(), "bounded module inventory");
        auto& module = driver.modules[driver.next_module++];
        module.alive = true;
        module.context = current_context;
        *output = reinterpret_cast<CUmodule>(&module);
        return CUDA_SUCCESS;
    };
    result.module_unload = [](CUmodule handle) {
        ++driver.calls;
        auto& module = *reinterpret_cast<Module*>(handle);
        expect(module.alive && module.context == current_context,
               "module lease settles before unload");
        module.alive = false;
        ++driver.unloads;
        return CUDA_SUCCESS;
    };
    result.module_global = [](CUdeviceptr* output, std::size_t* bytes, CUmodule handle,
                              const char*) {
        ++driver.calls;
        auto& module = *reinterpret_cast<Module*>(handle);
        *output = reinterpret_cast<CUdeviceptr>(module.global.data());
        *bytes = module.global.size();
        return CUDA_SUCCESS;
    };
    result.function_get = [](CUfunction* output, CUmodule handle, const char*) {
        ++driver.calls;
        *output = reinterpret_cast<CUfunction>(&reinterpret_cast<Module*>(handle)->function);
        return CUDA_SUCCESS;
    };
    result.parameter_info = [](CUfunction, std::size_t index, std::size_t* offset,
                               std::size_t* size) {
        ++driver.calls;
        expect(index == 0, "independent public kernel ABI has one pointer");
        *offset = 0;
        *size = 8;
        return CUDA_SUCCESS;
    };
    result.allocate = [](CUdeviceptr* output, std::size_t size) {
        ++driver.calls;
        Memory memory{current_context, std::vector<std::byte>(size)};
        *output = reinterpret_cast<CUdeviceptr>(memory.bytes.data());
        expect(driver.memory.emplace(*output, std::move(memory)).second, "distinct live memory");
        return CUDA_SUCCESS;
    };
    result.release = [](CUdeviceptr address) {
        ++driver.calls;
        const auto found = driver.memory.find(address);
        expect(found != driver.memory.end() && found->second.context == current_context,
               "resource owner frees exact live allocation once");
        driver.memory.erase(found);
        ++driver.releases;
        return CUDA_SUCCESS;
    };
    result.upload = [](CUdeviceptr, const void*, std::size_t) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    result.download = [](void*, CUdeviceptr, std::size_t) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    result.stream_create = [](CUstream* output, unsigned) {
        ++driver.calls;
        expect(driver.next_stream < driver.streams.size(), "bounded stream inventory");
        auto& stream = driver.streams[driver.next_stream++];
        stream = {true, current_context};
        *output = reinterpret_cast<CUstream>(&stream);
        return CUDA_SUCCESS;
    };
    result.stream_destroy = [](CUstream handle) {
        ++driver.calls;
        auto& stream = *reinterpret_cast<Stream*>(handle);
        expect(stream.alive && stream.context == current_context,
               "stream lease settles before cleanup");
        stream.alive = false;
        ++driver.stream_destroys;
        return CUDA_SUCCESS;
    };
    result.execution.context_get_current = get_current;
    result.execution.event_create = create_event;
    result.execution.event_record = record_event;
    result.execution.event_query = query_event;
    result.execution.event_destroy = destroy_event;
    result.execution.stream_synchronize = [](CUstream) {
        ++driver.calls;
        ++driver.synchronizes;
        return CUDA_ERROR_NOT_READY;
    };
    result.execution.launch_kernel = [](CUfunction, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, CUstream, void**, void**) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Requests are serialized test operations, not a production execution lane.
// The stable real thread deliberately owns only the immutable worker endpoint.
class Lane {
   public:
    explicit Lane(CUcontext context)
        : thread_([this, context] {
              current_context = context;
              std::unique_lock lock(mutex_);
              for (;;) {
                  condition_.wait(lock, [&] { return stopping_ || pending_; });
                  if (stopping_) return;
                  auto task = std::move(task_);
                  lock.unlock();
                  CUresult result{};
                  std::exception_ptr error;
                  try {
                      result = task();
                  } catch (...) {
                      error = std::current_exception();
                  }
                  lock.lock();
                  result_ = result;
                  error_ = error;
                  pending_ = false;
                  completed_ = true;
                  condition_.notify_all();
              }
          }) {}
    ~Lane() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            condition_.notify_all();
        }
        thread_.join();
    }
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;
    void submit(std::function<CUresult()> task) {
        std::lock_guard lock(mutex_);
        expect(!pending_, "one test job per stable worker");
        task_ = std::move(task);
        completed_ = false;
        pending_ = true;
        condition_.notify_all();
    }
    CUresult finish() {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return completed_; });
        if (error_) std::rethrow_exception(error_);
        return result_;
    }
    CUresult run(std::function<CUresult()> task) {
        submit(std::move(task));
        return finish();
    }

   private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool pending_{}, completed_{}, stopping_{};
    CUresult result_{};
    std::exception_ptr error_;
    std::function<CUresult()> task_;
    std::thread thread_;
};

constexpr std::string_view kPtx =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .global .align 4 .u32 global_data[8];\n"
    ".visible .entry kernel(.param .u64 pointer) { ret; }\n";
struct Fixture {
    CudaExecutionSession source{{101, 109}, 17, session_dispatch()};
    CUcontext source_context{}, execution_context{};
    CUstream source_stream{}, execution_stream{};
    std::uint64_t allocation{}, module{}, global{}, stream_id{};
    std::shared_ptr<const void> execution_context_owner = std::make_shared<int>(41);
    std::shared_ptr<const void> execution_stream_owner = std::make_shared<int>(43);
    CudaSessionExternalCommand capsule;
    std::shared_ptr<CudaSessionExternalCommand> boundary_owner;
    CudaExecutionExternalAcceptance tail;
    std::unique_ptr<CudaExternalExecutionBridge> bridge;
    CudaExternalExecutionWorker worker;
    std::unique_ptr<Lane> lane;

    explicit Fixture(bool uses_default = false) {
        complete(source.initialize(5));
        source_context = current_context;
        const auto memory = source.allocate(32);
        complete(memory);
        allocation = memory.id;
        const auto loaded = source.load_ptx(kPtx);
        complete(loaded);
        module = loaded.id;
        const auto symbol = source.get_global(module, "global_data");
        complete(symbol);
        global = symbol.id;
        if (!uses_default) {
            const auto stream = source.create_stream(0);
            complete(stream);
            stream_id = stream.id;
        }
        const std::array<CudaSessionExternalRangeRequest, 2> ranges{
            {{{allocation, 4}, 16}, {{global, 8}, 8}}};
        auto captured = source.capture_external_command(stream_id, ranges);
        expect(captured.status == CUDA_SUCCESS,
               "capture exact source allocation and global-parent owners");
        capsule = std::move(captured.command);
        source_stream = capsule.stream();
        boundary_owner = std::make_shared<CudaSessionExternalCommand>(capsule);
        tail = source.accept_external_barrier(stream_id, boundary_owner);
        expect(tail.status == CUDA_SUCCESS &&
                   source.begin_external_barrier(tail.barrier) == CUDA_SUCCESS,
               "actual source dispatcher accepts and begins the exact pending external tail");
        auto sdk = session_dispatch();
        expect(sdk.context_create(&execution_context, 0, 5) == CUDA_SUCCESS &&
                   sdk.stream_create(&execution_stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
               "trusted execution domain is a distinct context and explicit real stream");
        current_context = source_context;
        lane = std::make_unique<Lane>(execution_context);
    }
    CudaExternalExecutionDomain domain() const {
        return {{29, execution_context, execution_context_owner},
                5,
                execution_stream,
                execution_stream_owner};
    }
    void construct() {
        bridge = std::make_unique<CudaExternalExecutionBridge>(capsule, tail.frontier, domain(),
                                                               bridge_dispatch());
        worker = bridge->worker();
    }
    CUresult prepare() {
        return source.prepare_external_bridge(*bridge);
    }
    CUresult join() {
        return source.join_external_bridge(*bridge);
    }
    CUresult clean_source() {
        return source.cleanup_external_bridge(*bridge);
    }
    CUresult begin_worker() {
        return lane->run([this] { return worker.begin(); });
    }
    CUresult record_worker() {
        return lane->run([this] { return worker.record_enqueue(); });
    }
    CUresult clean_worker() {
        return lane->run([this] { return worker.cleanup(); });
    }
    void ready_tail() {
        driver.events[0].ready = true;
        expect(source.poll().retired == 1 && tail.frontier.observe() == CUDA_SUCCESS,
               "only actual recorded immutable source tail completion permits bridge cleanup");
    }
    void drop_local_capsules() {
        capsule = {};
        boundary_owner.reset();
    }
    void release_source() {
        complete(source.release(allocation));
        complete(source.unload_module(module));
        if (stream_id) complete(source.destroy_stream(stream_id));
        complete(source.shutdown());
    }
    void release_execution() {
        lane.reset();
        current_context = execution_context;
        auto sdk = session_dispatch();
        expect(sdk.stream_destroy(execution_stream) == CUDA_SUCCESS &&
                   sdk.context_destroy(execution_context) == CUDA_SUCCESS,
               "trusted execution owner cleans its stream and context only after bridge release");
        execution_stream_owner.reset();
        execution_context_owner.reset();
    }
};

std::vector<Entry> operations() {
    std::vector<Entry> result;
    for (const auto& entry : driver.entries)
        if (entry.call != Call::kContext) result.push_back(entry);
    return result;
}

void healthy_two_sided_dependencies(bool uses_default) {
    driver.reset();
    Fixture fixture(uses_default);
    const auto owner_thread = std::this_thread::get_id();
    const auto calls = driver.calls.load();
    fixture.construct();
    expect(driver.calls == calls && fixture.bridge->failure() == CUDA_SUCCESS &&
               !fixture.bridge->completed(),
           "construction is CPU-only and grants no native enqueue or completion");
    const std::weak_ptr<const void> execution_context_owner = fixture.execution_context_owner;
    const std::weak_ptr<const void> execution_stream_owner = fixture.execution_stream_owner;
    fixture.execution_context_owner.reset();
    fixture.execution_stream_owner.reset();
    expect(!execution_context_owner.expired() && !execution_stream_owner.expired(),
           "bridge independently retains the trusted execution context and stream owners");
    fixture.drop_local_capsules();
    deferred(fixture.source.release(fixture.allocation));
    deferred(fixture.source.unload_module(fixture.module));
    if (fixture.stream_id) deferred(fixture.source.destroy_stream(fixture.stream_id));
    driver.entries.clear();
    expect(fixture.join() == CUDA_ERROR_NOT_READY && fixture.clean_source() == CUDA_ERROR_NOT_READY,
           "a ready logical source boundary is not an incoming or outgoing GPU dependency");
    expect(fixture.prepare() == CUDA_SUCCESS && fixture.prepare() == CUDA_ERROR_INVALID_VALUE,
           "source event is recorded exactly once before execution begins");
    expect(fixture.begin_worker() == CUDA_SUCCESS &&
               fixture.begin_worker() == CUDA_ERROR_INVALID_VALUE &&
               fixture.clean_worker() == CUDA_ERROR_NOT_READY &&
               fixture.join() == CUDA_ERROR_NOT_READY,
           "worker incoming wait is not completed vendor enqueue or outgoing source ordering");
    expect(fixture.lane->run([&] {
        const auto observed = operations();
        expect(observed.size() == 4 && observed.back().call == Call::kWait &&
                   observed.back().context == fixture.execution_context &&
                   observed.back().stream == fixture.execution_stream,
               "independent native work starts only after the execution stream actually waits on "
               "source prefix");
        return driver.visit(Call::kWork, fixture.execution_stream);
    }) == CUDA_SUCCESS,
           "one actual external work marker, not bridge-minted host status");
    expect(fixture.record_worker() == CUDA_SUCCESS &&
               fixture.record_worker() == CUDA_ERROR_INVALID_VALUE &&
               fixture.tail.frontier.observe() == CUDA_ERROR_NOT_READY &&
               fixture.clean_source() == CUDA_ERROR_NOT_READY &&
               fixture.clean_worker() == CUDA_ERROR_NOT_READY,
           "recorded execution event still grants no source physical completion");
    expect(fixture.join() == CUDA_SUCCESS && fixture.join() == CUDA_ERROR_INVALID_VALUE &&
               fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS &&
               fixture.source.poll().retired == 0 && fixture.clean_source() == CUDA_ERROR_NOT_READY,
           "source stream waits on execution event before tail publication; pending tail forbids "
           "cleanup");
    fixture.ready_tail();
    deferred(fixture.source.release(fixture.allocation));
    expect(fixture.clean_worker() == CUDA_ERROR_NOT_READY &&
               fixture.clean_source() == CUDA_SUCCESS && fixture.clean_worker() == CUDA_SUCCESS &&
               fixture.bridge->completed() && fixture.bridge->failure() == CUDA_SUCCESS,
           "real tail stop precedes source cleanup, which precedes original-worker cleanup");
    const auto observed = operations();
    constexpr std::array expected{Call::kCreate, Call::kRecord, Call::kCreate,  Call::kWait,
                                  Call::kWork,   Call::kRecord, Call::kWait,    Call::kRecord,
                                  Call::kQuery,  Call::kQuery,  Call::kDestroy, Call::kDestroy,
                                  Call::kDestroy};
    expect(observed.size() == expected.size(),
           "complete independent event dependency/cleanup ledger");
    for (std::size_t index = 0; index != expected.size(); ++index)
        expect(observed[index].call == expected[index],
               "exact two-sided native event operation order");
    const auto source_event = observed[0].event;
    const auto execution_event = observed[2].event;
    expect(
        source_event && execution_event && source_event != execution_event &&
            observed[1].event == source_event && observed[1].context == fixture.source_context &&
            observed[1].stream == fixture.source_stream && observed[3].event == source_event &&
            observed[3].context == fixture.execution_context &&
            observed[3].stream == fixture.execution_stream &&
            observed[5].event == execution_event &&
            observed[5].context == fixture.execution_context &&
            observed[6].event == execution_event && observed[6].context == fixture.source_context &&
            observed[6].stream == fixture.source_stream && observed[11].event == source_event &&
            observed[12].event == execution_event && observed[3].thread != owner_thread &&
            observed[3].thread == observed[5].thread && observed[5].thread == observed[12].thread &&
            observed[1].thread == owner_thread && observed[6].thread == owner_thread &&
            observed[11].thread == owner_thread && driver.synchronizes == 0,
        "both actual cross-context waits use correct domain events, streams and stable worker "
        "ownership");
    const auto native_calls = driver.calls.load();
    fixture.worker = {};
    fixture.bridge.reset();
    expect(driver.calls == native_calls && fixture.source.snapshot().external_command_leases == 0 &&
               fixture.source.snapshot().external_context_leases == 0 &&
               execution_context_owner.expired() && execution_stream_owner.expired(),
           "completed endpoint destruction is CPU-only and releases real capsule registrations");
    fixture.release_source();
    fixture.release_execution();
    expect(driver.releases == 1 && driver.unloads == 1 && driver.context_destroys == 2 &&
               driver.stream_destroys == (uses_default ? 1U : 2U),
           "all exact allocation/module/stream/context owners clean once after both dependency "
           "directions settle");
}

void finish_healthy(Fixture& fixture) {
    expect(fixture.prepare() == CUDA_SUCCESS && fixture.begin_worker() == CUDA_SUCCESS &&
               fixture.record_worker() == CUDA_SUCCESS && fixture.join() == CUDA_SUCCESS &&
               fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS,
           "complete healthy actual two-sided bridge");
    fixture.ready_tail();
    expect(fixture.clean_source() == CUDA_SUCCESS && fixture.clean_worker() == CUDA_SUCCESS &&
               fixture.bridge->completed(),
           "both native owners settle after the exact tail");
    fixture.drop_local_capsules();
    fixture.worker = {};
    fixture.bridge.reset();
    fixture.release_source();
    fixture.release_execution();
}

void constructor_rejections() {
    driver.reset();
    Fixture fixture;
    const auto completed_frontier = fixture.source.capture_frontier(0);
    expect(completed_frontier.status == CUDA_SUCCESS &&
               completed_frontier.frontier.observe() == CUDA_SUCCESS,
           "empty unrelated legacy frontier has no pending external tail");
    for (unsigned which = 0; which != 17; ++which) {
        auto source = fixture.capsule;
        auto tail = fixture.tail.frontier;
        auto domain = fixture.domain();
        auto sdk = bridge_dispatch();
        switch (which) {
            case 0:
                source = {};
                break;
            case 1:
                tail = {};
                break;
            case 2:
                tail = completed_frontier.frontier;
                break;
            case 3:
                domain.context.id = 0;
                break;
            case 4:
                domain.context.native = nullptr;
                break;
            case 5:
                domain.context.lifetime.reset();
                break;
            case 6:
                domain.device_ordinal = -1;
                break;
            case 7:
                domain.device_ordinal = 6;
                break;
            case 8:
                domain.stream = nullptr;
                break;
            case 9:
                domain.stream = CU_STREAM_LEGACY;
                break;
            case 10:
                domain.stream = CU_STREAM_PER_THREAD;
                break;
            case 11:
                domain.stream_lifetime.reset();
                break;
            case 12:
                sdk.context_get_current = nullptr;
                break;
            case 13:
                sdk.event_record = nullptr;
                break;
            case 14:
                sdk.stream_wait_event = nullptr;
                break;
            case 15:
                sdk.event_create = nullptr;
                break;
            case 16:
                sdk.event_destroy = nullptr;
                break;
            default:
                break;
        }
        const auto calls = driver.calls.load();
        bool rejected{};
        try {
            CudaExternalExecutionBridge invalid(std::move(source), std::move(tail),
                                                std::move(domain), sdk);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.calls == calls,
               "invalid source/tail/domain/dispatch rejects without native side effects or partial "
               "leases");
    }
    fixture.construct();
    finish_healthy(fixture);
}

void endpoint_owner_boundaries() {
    driver.reset();
    Fixture fixture;
    fixture.construct();
    CudaExternalExecutionWorker empty;
    expect(empty.begin() == CUDA_ERROR_INVALID_HANDLE &&
               empty.record_enqueue() == CUDA_ERROR_INVALID_HANDLE &&
               empty.cleanup() == CUDA_ERROR_INVALID_HANDLE,
           "empty worker has no native authority");
    const auto first_calls = driver.calls.load();
    expect(fixture.worker.begin() == CUDA_ERROR_INVALID_CONTEXT && driver.calls == first_calls,
           "source owner cannot become the native worker and bypass session exclusion");
    expect(fixture.begin_worker() == CUDA_ERROR_NOT_READY,
           "early worker entry cannot bind or replace the missing actual source prefix");
    // An early unsuccessful entry on one incarnation must not claim the worker.
    fixture.lane.reset();
    fixture.lane = std::make_unique<Lane>(fixture.execution_context);
    const auto calls = driver.calls.load();
    expect(fixture.lane->run([&] { return fixture.prepare(); }) == CUDA_ERROR_INVALID_CONTEXT &&
               driver.calls == calls,
           "source facade cannot be used by the native worker");
    expect(fixture.prepare() == CUDA_SUCCESS && fixture.begin_worker() == CUDA_SUCCESS,
           "only first admissible worker begin binds its actual thread incarnation");
    auto copy = fixture.worker;
    const auto before = driver.calls.load();
    expect(copy.begin() == CUDA_ERROR_INVALID_CONTEXT &&
               copy.record_enqueue() == CUDA_ERROR_INVALID_CONTEXT &&
               copy.cleanup() == CUDA_ERROR_INVALID_CONTEXT && driver.calls == before,
           "source owner cannot adopt an already-bound worker endpoint");
    const auto child = fork();
    expect(child >= 0, "fork endpoint ownership probe");
    if (child == 0) {
        const bool valid = fixture.prepare() == CUDA_ERROR_INVALID_CONTEXT &&
                           fixture.join() == CUDA_ERROR_INVALID_CONTEXT &&
                           fixture.clean_source() == CUDA_ERROR_INVALID_CONTEXT &&
                           copy.begin() == CUDA_ERROR_INVALID_CONTEXT &&
                           copy.record_enqueue() == CUDA_ERROR_INVALID_CONTEXT &&
                           copy.cleanup() == CUDA_ERROR_INVALID_CONTEXT && driver.calls == before;
        std::_Exit(valid ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before inherited state/native access");
    expect(fixture.record_worker() == CUDA_SUCCESS && fixture.join() == CUDA_SUCCESS &&
               fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS,
           "rejected foreign owner attempts leave original worker usable");
    fixture.ready_tail();
    expect(fixture.clean_source() == CUDA_SUCCESS &&
               fixture.clean_source() == CUDA_ERROR_INVALID_VALUE &&
               fixture.clean_worker() == CUDA_SUCCESS &&
               fixture.clean_worker() == CUDA_ERROR_INVALID_VALUE,
           "each exact owner cleans once; repeated cleanup never retries SDK destruction");
    copy = {};
    fixture.worker = {};
    fixture.bridge.reset();
    fixture.drop_local_capsules();
    fixture.release_source();
    fixture.release_execution();
}

void source_session_identity() {
    driver.reset();
    Fixture fixture;
    fixture.construct();
    CudaExecutionSession foreign{{101, 109}, 23, session_dispatch()};
    complete(fixture.source.deactivate());
    complete(foreign.initialize(5));
    const auto calls = driver.calls.load();
    expect(foreign.prepare_external_bridge(*fixture.bridge) == CUDA_ERROR_INVALID_HANDLE &&
               foreign.join_external_bridge(*fixture.bridge) == CUDA_ERROR_INVALID_HANDLE &&
               foreign.cleanup_external_bridge(*fixture.bridge) == CUDA_ERROR_INVALID_HANDLE &&
               driver.calls == calls,
           "same peer/device is not authority for another session's source bridge");
    complete(foreign.shutdown());
    complete(fixture.source.activate());
    finish_healthy(fixture);
}

void source_callback_reentry() {
    driver.reset();
    Fixture fixture;
    fixture.construct();
    bool correct = true;
    unsigned callbacks{};
    driver.injection = [&](Call) {
        if (current_context != fixture.source_context) return CUDA_SUCCESS;
        ++callbacks;
        const auto calls = driver.calls.load();
        const auto released = fixture.source.release(fixture.allocation);
        const auto unloaded = fixture.source.unload_module(fixture.module);
        const auto destroyed = fixture.source.destroy_stream(fixture.stream_id);
        const auto captured = fixture.source.capture_external_command(fixture.stream_id, {});
        correct &=
            fixture.prepare() == CUDA_ERROR_NOT_READY && fixture.join() == CUDA_ERROR_NOT_READY &&
            fixture.clean_source() == CUDA_ERROR_NOT_READY &&
            fixture.bridge->failure() == CUDA_ERROR_NOT_READY && !fixture.bridge->completed() &&
            released.disposition == CudaSessionDisposition::kDeferred &&
            released.status == CUDA_ERROR_NOT_READY &&
            unloaded.disposition == CudaSessionDisposition::kDeferred &&
            unloaded.status == CUDA_ERROR_NOT_READY &&
            destroyed.disposition == CudaSessionDisposition::kDeferred &&
            destroyed.status == CUDA_ERROR_NOT_READY && captured.status == CUDA_ERROR_NOT_READY &&
            !captured.command.context().native() && driver.calls == calls;
        return CUDA_SUCCESS;
    };
    expect(fixture.prepare() == CUDA_SUCCESS && fixture.begin_worker() == CUDA_SUCCESS &&
               fixture.record_worker() == CUDA_SUCCESS && fixture.join() == CUDA_SUCCESS,
           "source bridge facade spans every source native callback with session exclusion");
    driver.injection = {};
    expect(fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS,
           "trusted actual joined work now records the source tail");
    fixture.ready_tail();
    driver.injection = [&](Call) {
        ++callbacks;
        const auto calls = driver.calls.load();
        correct &= fixture.prepare() == CUDA_ERROR_NOT_READY &&
                   fixture.join() == CUDA_ERROR_NOT_READY &&
                   fixture.clean_source() == CUDA_ERROR_NOT_READY &&
                   fixture.source.capture_external_command(0, {}).status == CUDA_ERROR_NOT_READY &&
                   driver.calls == calls;
        return CUDA_SUCCESS;
    };
    expect(fixture.clean_source() == CUDA_SUCCESS, "source event cleanup also excludes reentry");
    driver.injection = {};
    expect(fixture.clean_worker() == CUDA_SUCCESS && correct && callbacks >= 6,
           "callback reentry never issues native calls, captures a dying resource or changes "
           "bridge phase");
    fixture.drop_local_capsules();
    fixture.worker = {};
    fixture.bridge.reset();
    fixture.release_source();
    fixture.release_execution();
}

void worker_reentry_and_concurrent_source() {
    driver.reset();
    Fixture fixture;
    fixture.construct();
    expect(fixture.prepare() == CUDA_SUCCESS,
           "source prefix exists before worker callback controls");
    bool correct = true;
    unsigned callbacks{};
    std::mutex mutex;
    std::condition_variable condition;
    bool blocked{}, release{};
    driver.injection = [&](Call call) {
        if (current_context != fixture.execution_context) return CUDA_SUCCESS;
        ++callbacks;
        const auto before = driver.calls.load();
        correct &= fixture.worker.begin() == CUDA_ERROR_NOT_READY &&
                   fixture.worker.record_enqueue() == CUDA_ERROR_NOT_READY &&
                   fixture.worker.cleanup() == CUDA_ERROR_NOT_READY && driver.calls == before;
        if (call == Call::kWait) {
            std::unique_lock lock(mutex);
            blocked = true;
            condition.notify_all();
            condition.wait(lock, [&] { return release; });
        }
        return CUDA_SUCCESS;
    };
    fixture.lane->submit([&] { return fixture.worker.begin(); });
    bool reached{};
    {
        std::unique_lock lock(mutex);
        reached = condition.wait_for(lock, std::chrono::seconds(5), [&] { return blocked; });
    }
    const auto calls = driver.calls.load();
    const bool excluded = reached && fixture.prepare() == CUDA_ERROR_NOT_READY &&
                          fixture.join() == CUDA_ERROR_NOT_READY &&
                          fixture.clean_source() == CUDA_ERROR_NOT_READY &&
                          fixture.bridge->failure() == CUDA_ERROR_NOT_READY &&
                          !fixture.bridge->completed() && driver.calls == calls;
    {
        std::lock_guard lock(mutex);
        release = true;
        condition.notify_all();
    }
    const auto begun = fixture.lane->finish();
    expect(excluded && begun == CUDA_SUCCESS && fixture.record_worker() == CUDA_SUCCESS,
           "original worker continues after rejected concurrent source attempts");
    driver.injection = {};
    expect(fixture.join() == CUDA_SUCCESS &&
               fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS,
           "outgoing source join remains necessary after worker callback completion");
    fixture.ready_tail();
    expect(fixture.clean_source() == CUDA_SUCCESS, "source cleanup precedes worker cleanup");
    driver.injection = [&](Call) {
        ++callbacks;
        const auto before = driver.calls.load();
        correct &= fixture.worker.begin() == CUDA_ERROR_NOT_READY &&
                   fixture.worker.record_enqueue() == CUDA_ERROR_NOT_READY &&
                   fixture.worker.cleanup() == CUDA_ERROR_NOT_READY && driver.calls == before;
        return CUDA_SUCCESS;
    };
    expect(fixture.clean_worker() == CUDA_SUCCESS && correct && callbacks >= 6,
           "worker context/create/wait/record/destroy callbacks all exclude recursive entry");
    driver.injection = {};
    fixture.drop_local_capsules();
    fixture.worker = {};
    fixture.bridge.reset();
    fixture.release_source();
    fixture.release_execution();
}

enum class Phase { kPrepare, kBegin, kRecord, kJoin, kSourceCleanup, kWorkerCleanup };
CUresult phase_call(Fixture& fixture, Phase phase) {
    switch (phase) {
        case Phase::kPrepare:
            return fixture.prepare();
        case Phase::kBegin:
            return fixture.begin_worker();
        case Phase::kRecord:
            return fixture.record_worker();
        case Phase::kJoin:
            return fixture.join();
        case Phase::kSourceCleanup:
            return fixture.clean_source();
        case Phase::kWorkerCleanup:
            return fixture.clean_worker();
    }
    return CUDA_ERROR_INVALID_VALUE;
}
void prepare_phase(Fixture& fixture, Phase phase) {
    fixture.construct();
    fixture.drop_local_capsules();
    if (phase == Phase::kPrepare) return;
    expect(fixture.prepare() == CUDA_SUCCESS, "fault prefix: source dependency recorded");
    if (phase == Phase::kBegin) return;
    expect(fixture.begin_worker() == CUDA_SUCCESS, "fault prefix: execution incoming wait");
    if (phase == Phase::kRecord) return;
    expect(fixture.record_worker() == CUDA_SUCCESS,
           "fault prefix: actual execution enqueue recorded");
    if (phase == Phase::kJoin) return;
    expect(fixture.join() == CUDA_SUCCESS &&
               fixture.source.record_external_enqueue(fixture.tail.barrier) == CUDA_SUCCESS,
           "fault prefix: exact source outgoing wait and recorded dispatcher tail");
    fixture.ready_tail();
    if (phase == Phase::kWorkerCleanup)
        expect(fixture.clean_source() == CUDA_SUCCESS,
               "fault prefix: source event cleaned after real tail");
}

enum class FaultMode { kReturnError, kChangeContext, kPartialCreate, kNullCreate, kAliasCreate };
struct Fault {
    Phase phase;
    Call call;
    unsigned occurrence{1};
    FaultMode mode{FaultMode::kReturnError};
};
std::vector<Fault> faults() {
    std::vector<Fault> result;
    result.reserve(40);
    constexpr std::array phases{Phase::kPrepare, Phase::kBegin,         Phase::kRecord,
                                Phase::kJoin,    Phase::kSourceCleanup, Phase::kWorkerCleanup};
    for (const auto phase : phases) {
        const unsigned proofs = phase == Phase::kPrepare || phase == Phase::kBegin ? 4 : 2;
        for (unsigned occurrence = 1; occurrence <= proofs; ++occurrence)
            result.push_back({phase, Call::kContext, occurrence});
        std::vector<Call> callbacks;
        switch (phase) {
            case Phase::kPrepare:
                callbacks = {Call::kCreate, Call::kRecord};
                break;
            case Phase::kBegin:
                callbacks = {Call::kCreate, Call::kWait};
                break;
            case Phase::kRecord:
                callbacks = {Call::kRecord};
                break;
            case Phase::kJoin:
                callbacks = {Call::kWait};
                break;
            case Phase::kSourceCleanup:
            case Phase::kWorkerCleanup:
                callbacks = {Call::kDestroy};
                break;
        }
        for (const auto callback : callbacks) {
            result.push_back({phase, callback});
            result.push_back({phase, callback, 1, FaultMode::kChangeContext});
        }
    }
    for (const auto phase : {Phase::kPrepare, Phase::kBegin}) {
        result.push_back({phase, Call::kCreate, 1, FaultMode::kPartialCreate});
        result.push_back({phase, Call::kCreate, 1, FaultMode::kNullCreate});
    }
    result.push_back({Phase::kBegin, Call::kCreate, 1, FaultMode::kAliasCreate});
    return result;
}

void permanent_failure(const Fault& fault) {
    driver.reset();
    auto fixture = std::make_unique<Fixture>();
    const std::weak_ptr<const void> execution_context_owner = fixture->execution_context_owner;
    const std::weak_ptr<const void> execution_stream_owner = fixture->execution_stream_owner;
    prepare_phase(*fixture, fault.phase);
    fixture->execution_context_owner.reset();
    fixture->execution_stream_owner.reset();
    driver.entries.clear();
    unsigned matches{};
    CUresult failure = fault.call == Call::kContext || fault.mode == FaultMode::kChangeContext
                           ? CUDA_ERROR_INVALID_CONTEXT
                           : CUDA_ERROR_UNKNOWN;
    if (fault.mode == FaultMode::kPartialCreate) driver.partial_create = true;
    if (fault.mode == FaultMode::kNullCreate) driver.null_create = true;
    if (fault.mode == FaultMode::kAliasCreate)
        driver.alias_create = reinterpret_cast<CUevent>(&driver.events[1]);
    driver.injection = [&](Call call) {
        if (call != fault.call || ++matches != fault.occurrence) return CUDA_SUCCESS;
        if (fault.mode == FaultMode::kChangeContext) {
            current_context = nullptr;
            return CUDA_SUCCESS;
        }
        if (fault.mode == FaultMode::kNullCreate || fault.mode == FaultMode::kAliasCreate)
            return CUDA_SUCCESS;
        return failure;
    };
    const auto result = phase_call(*fixture, fault.phase);
    driver.injection = {};
    driver.partial_create = false;
    driver.null_create = false;
    driver.alias_create = nullptr;
    current_context = fixture->source_context;
    expect(fixture->lane->run([&] {
        current_context = fixture->execution_context;
        return CUDA_SUCCESS;
    }) == CUDA_SUCCESS,
           "test restores owner contexts without claiming failed bridge recovery");
    expect(matches >= fault.occurrence && result == failure &&
               fixture->bridge->failure() == failure && !fixture->bridge->completed() &&
               !execution_context_owner.expired() && !execution_stream_owner.expired(),
           "every failed native/proof boundary permanently retains exact bridge owners and first "
           "failure");
    const auto native_calls = driver.calls.load();
    for (int repeat = 0; repeat != 2; ++repeat) {
        expect(fixture->prepare() == failure && fixture->join() == failure &&
                   fixture->clean_source() == failure && fixture->begin_worker() == failure,
               "later owner calls expose the original bridge failure without native replay");
        (void)fixture->record_worker();
        (void)fixture->clean_worker();
    }
    expect(driver.calls == native_calls && driver.synchronizes == 0,
           "failure retries never record, wait, query, synchronize or destroy any bridge event");
    if (fault.phase < Phase::kSourceCleanup)
        expect(
            fixture->source.fail_external_barrier(fixture->tail.barrier, failure) == CUDA_SUCCESS,
            "trusted composition marks the real still-started source tail uncertain");
    const auto retained = [&](const CudaSessionResult& result) {
        if (fault.phase < Phase::kSourceCleanup) {
            expect(result.disposition == CudaSessionDisposition::kRejected &&
                       result.status == CUDA_ERROR_UNKNOWN && !result.original_status,
                   "failed external tail quarantines ordinary source cleanup without calling "
                   "native cleanup");
        } else {
            deferred(result);
        }
    };
    retained(fixture->source.release(fixture->allocation));
    retained(fixture->source.unload_module(fixture->module));
    retained(fixture->source.destroy_stream(fixture->stream_id));
    const auto before_drop = driver.calls.load();
    fixture->worker = {};
    fixture->bridge.reset();
    fixture->lane.reset();
    expect(
        driver.calls == before_drop && !execution_context_owner.expired() &&
            !execution_stream_owner.expired() &&
            fixture->source.snapshot().external_command_leases == 1 &&
            fixture->source.snapshot().external_context_leases == 1 && driver.releases == 0 &&
            driver.unloads == 0,
        "destruction never repairs uncertainty or drops source/execution physical-resource owners");
    fixture.reset();
    expect(!execution_context_owner.expired() && !execution_stream_owner.expired() &&
               driver.calls == before_drop,
           "unsettled endpoint and session destruction perform no native cleanup");
}

void expired_worker_incarnation() {
    driver.reset();
    auto fixture = std::make_unique<Fixture>();
    fixture->construct();
    fixture->drop_local_capsules();
    expect(fixture->prepare() == CUDA_SUCCESS && fixture->begin_worker() == CUDA_SUCCESS,
           "first actual worker incarnation owns the incoming dependency");
    fixture->lane.reset();
    fixture->lane = std::make_unique<Lane>(fixture->execution_context);
    const auto calls = driver.calls.load();
    expect(fixture->begin_worker() == CUDA_ERROR_INVALID_CONTEXT &&
               fixture->record_worker() == CUDA_ERROR_INVALID_CONTEXT &&
               fixture->clean_worker() == CUDA_ERROR_INVALID_CONTEXT && driver.calls == calls,
           "a replacement native thread cannot inherit a dead worker's authority even if its "
           "thread ID is reused");
    expect(fixture->source.fail_external_barrier(fixture->tail.barrier, CUDA_ERROR_UNKNOWN) ==
               CUDA_SUCCESS,
           "lost lane does not grant enqueue or completion proof");
    fixture->worker = {};
    fixture->bridge.reset();
    fixture->lane.reset();
    deferred(fixture->source.shutdown());
}

void failed_tail_is_not_cleanup_authority() {
    driver.reset();
    auto fixture = std::make_unique<Fixture>();
    prepare_phase(*fixture, Phase::kJoin);
    expect(fixture->join() == CUDA_SUCCESS &&
               fixture->source.fail_external_barrier(fixture->tail.barrier,
                                                     CUDA_ERROR_LAUNCH_FAILED) == CUDA_SUCCESS,
           "independently failed accepted tail remains a failure after outgoing join");
    const auto calls = driver.calls.load();
    expect(fixture->clean_source() == CUDA_ERROR_LAUNCH_FAILED &&
               fixture->bridge->failure() == CUDA_ERROR_LAUNCH_FAILED &&
               fixture->clean_worker() == CUDA_ERROR_LAUNCH_FAILED && driver.calls == calls,
           "failed immutable frontier never authorizes either event destruction");
    fixture->worker = {};
    fixture->bridge.reset();
    fixture->lane.reset();
    deferred(fixture->source.shutdown());
}

template <typename Function>
void isolated(std::string_view label, Function function) {
    const auto child = fork();
    expect(child >= 0, "fork isolated permanent-retention case");
    if (child == 0) {
        try {
            function();
            std::_Exit(0);
        } catch (const std::exception& error) {
            std::cerr << label << ": " << error.what() << '\n';
            std::_Exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           std::string("isolated bridge case: ") + std::string(label));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto failure_cases = faults();
        if (argc == 3 && std::string_view(argv[1]) == "--failure-case") {
            const auto which = std::stoul(argv[2]);
            expect(which < failure_cases.size(), "bounded failure case index");
            permanent_failure(failure_cases[which]);
            std::_Exit(0);
        }
        expect(argc == 1, "no unexpected test arguments");
        healthy_two_sided_dependencies(false);
        healthy_two_sided_dependencies(true);
        constructor_rejections();
        endpoint_owner_boundaries();
        source_session_identity();
        source_callback_reentry();
        worker_reentry_and_concurrent_source();
        for (std::size_t index = 0; index != failure_cases.size(); ++index)
            isolated("native/proof " + std::to_string(index),
                     [&] { permanent_failure(failure_cases[index]); });
        isolated("expired worker incarnation", expired_worker_incarnation);
        isolated("failed physical tail", failed_tail_is_not_cleanup_authority);
        std::cout
            << "external execution bridge: 7 healthy groups, " << failure_cases.size()
            << " native/proof failures plus 2 isolated lifecycle failures; GPU_initialized=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
