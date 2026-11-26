#include "internal/cuda_execution_session.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Native {
    CUcontext current{};
    CUmemoryPool pool{reinterpret_cast<CUmemoryPool>(0x1000)};
    CUmemoryPool failed_selection{};
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::map<CUevent, bool> events;
    bool ready{};
    CUresult callback_status{CUDA_SUCCESS}, record_status{CUDA_SUCCESS};
    unsigned int creates{}, destroys{}, allocations{}, frees{}, fills{}, callbacks{};
    std::thread callback;
} native;
CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device(CUdevice* value, int ordinal) {
    *value = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current(CUcontext* value) {
    *value = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_context(CUcontext* value, unsigned int, CUdevice) {
    *value = reinterpret_cast<CUcontext>(new int(0));
    ++native.creates;
    native.current = *value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI set_context(CUcontext value) {
    native.current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_context(CUcontext value) {
    delete reinterpret_cast<int*>(value);
    native.current = nullptr;
    ++native.destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_create(CUevent* value, unsigned int) {
    *value = reinterpret_cast<CUevent>(new int(0));
    native.events.emplace(*value, false);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent value, CUstream) {
    native.events.at(value) = true;
    return native.record_status;
}
CUresult CUDAAPI event_query(CUevent value) {
    return native.ready && native.events.at(value) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    expect(native.events.erase(value) == 1, "destroy original event once");
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI load(CUmodule*, const void*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unload(CUmodule) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI function(CUfunction*, CUmodule, const char*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI parameter(CUfunction, std::size_t, std::size_t*, std::size_t*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI allocate(CUdeviceptr* output, std::size_t bytes) {
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    ++native.allocations;
    if (!bytes) {
        *output = 0;
        return CUDA_SUCCESS;
    }
    std::vector<std::byte> data(bytes);
    *output = reinterpret_cast<CUdeviceptr>(data.data());
    native.memory.emplace(*output, std::move(data));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr value) {
    expect(native.current && native.memory.erase(value) == 1,
           "free actual memory in live current context");
    ++native.frees;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload(CUdeviceptr target, const void* input, std::size_t bytes) {
    auto& storage = native.memory.at(target);
    expect(bytes <= storage.size(), "upload fits actual original allocation");
    std::memcpy(storage.data(), input, bytes);
    return CUDA_SUCCESS;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): original CUDA ABI order.
CUresult CUDAAPI download(void* output, CUdeviceptr source, std::size_t bytes) {
    const auto& storage = native.memory.at(source);
    expect(bytes <= storage.size(), "download fits actual original allocation");
    std::memcpy(output, storage.data(), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_stream(CUstream* value, unsigned int) {
    *value = reinterpret_cast<CUstream>(new int(0));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_stream(CUstream value) {
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI pool_query(CUmemoryPool* output, CUdevice) {
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    *output = native.pool;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI pool_select(CUdevice, CUmemoryPool value) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    if (value == native.failed_selection) return CUDA_ERROR_UNKNOWN;
    native.pool = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI allocate_async(CUdeviceptr* output, std::size_t bytes, CUstream) {
    return allocate(output, bytes);
}
CUresult CUDAAPI allocate_from(CUdeviceptr* output, std::size_t bytes, CUmemoryPool,
                               CUstream stream) {
    return allocate_async(output, bytes, stream);
}
cudaError_t CUDARTAPI trim(cudaMemPool_t, std::size_t) {
    return cudaSuccess;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): original CUDA ABI order.
CUresult CUDAAPI fill(CUdeviceptr target, unsigned int value, std::size_t count, CUstream) {
    auto& storage = native.memory.at(target);
    expect(count <= storage.size() / sizeof(value), "fill fits actual original allocation");
    ++native.fills;
    for (std::size_t index = 0; index != count; ++index)
        std::memcpy(storage.data() + index * sizeof(value), &value, sizeof(value));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI callback(CUstream, CUhostFn function, void* data) {
    ++native.callbacks;
    native.callback = std::thread([=] { function(data); });
    return native.callback_status;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {current, launch, event_create, event_record, event_query, event_destroy, synchronize},
        initialize,
        device,
        create_context,
        set_context,
        destroy_context,
        load,
        unload,
        function,
        parameter,
        allocate,
        release,
        upload,
        download,
        create_stream,
        destroy_stream};
    result.execution.fill32_async = fill;
    result.process_memory = {initialize,     current,       pool_query, pool_query, pool_select,
                             allocate_async, allocate_from, trim,       release};
    result.launch_host_function = callback;
    return result;
}
void run() {
    const auto functions = dispatch();
    auto device_owner = std::make_shared<CudaProcessMemoryDevice>(functions.process_memory,
                                                                  [] { return CUDA_SUCCESS; });
    auto quota = std::make_shared<glimmer::core::QuotaLedger>(128);
    auto process = std::make_shared<CudaProcessMemoryBackend>(CudaProcessMemoryConfig{1, 0}, quota,
                                                              device_owner);
    CudaExecutionSession first({1, 1}, 1, functions), second({2, 1}, 2, functions);
    expect(first.attach_process_memory(process) == CUDA_SUCCESS &&
               second.attach_process_memory(process) == CUDA_SUCCESS,
           "both contexts share one process owner before native creation");
    expect(first.initialize(0).status == CUDA_SUCCESS, "initialize source context");
    const auto classic = first.allocate(96);
    expect(classic.status == CUDA_SUCCESS && quota->usage().used_bytes() == 96,
           "classic charge uses shared process budget");
    expect(!first
                .allocate_process_memory(
                    1, {CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 64})
                .acceptance.accepted(),
           "pooled admission cannot overcommit classic bytes");
    expect(first.release(classic.id).status == CUDA_SUCCESS, "classic actual free refunds budget");
    auto gate = std::make_shared<CudaSessionHostFunction>();
    auto boundary = first.accept_external_barrier(0, gate);
    expect(boundary.status == CUDA_SUCCESS &&
               first.enqueue_host_function(boundary.barrier, 0, gate) == CUDA_SUCCESS,
           "native callback and tail enqueue on original stream");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!gate->entered() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    expect(gate->entered(), "actual native callback enters while CPU service continues");
    const auto accepted = first.allocate_process_memory(
        1, {CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 64});
    expect(accepted.acceptance.accepted(), "async allocation admitted after pending callback");
    expect(first.issue_next().sequence == 1,
           "actual allocation SDK host issue does not wait callback physical tail");
    const auto result = accepted.operation->reply();
    expect(result && result->original_status == CUDA_SUCCESS && result->id && result->output,
           "actual SDK output returned while preceding callback remains blocked");
    const auto consumer = first.accept_memory(2, {.kind = CudaMemoryKind::kFill32,
                                                  .destination = {result->id, 0},
                                                  .count = 16,
                                                  .value = 0x1728394a,
                                                  .is_asynchronous = true});
    expect(consumer.accepted() && first.issue_next().sequence == 2 && native.fills == 1,
           "same-stream consumer host enqueue needs no allocator readiness wait");
    expect(first.acknowledge_result(accepted.acceptance.receipt) &&
               first.acknowledge_result(consumer.receipt),
           "host receipts are acknowledged without dropping GPU leases");
    expect(first.shutdown().status != CUDA_SUCCESS && quota->usage().used_bytes() == 64,
           "source teardown cannot overtake native callback or allocator/consumer tails");
    gate->finish();
    native.callback.join();
    native.ready = true;
    expect(first.poll().status == CUDA_SUCCESS,
           "all actual fake-native tails retire after callback return");
    expect(process->progress() == CUDA_SUCCESS,
           "allocator source lease retires without freeing pooled allocation");
    expect(first.shutdown().status == CUDA_SUCCESS && native.memory.contains(result->output) &&
               native.destroys == 1 && quota->usage().used_bytes() == 64,
           "context A destroy preserves process allocation and charge");
    expect(second.initialize(0).status == CUDA_SUCCESS, "create unrelated consumer context B");
    std::array<std::uint32_t, 16> actual{};
    expect(second.download({result->id, 0}, std::as_writable_bytes(std::span{actual})).status ==
               CUDA_SUCCESS,
           "new context resolves process-owned range");
    for (const auto value : actual)
        expect(value == 0x1728394a, "all allocated values survive source destroy");
    expect(second.release(result->id).status == CUDA_SUCCESS && native.frees == 2 &&
               quota->usage().used_bytes() == 0,
           "actual free in B once after every consumer tail");
    expect(second.shutdown().status == CUDA_SUCCESS && native.destroys == 2 &&
               native.events.empty() && native.memory.empty() && process->close() == CUDA_SUCCESS &&
               process->shutdown() == CUDA_SUCCESS,
           "all contexts events memory and quota cleanly retired");
}
void retained_callback_failures() {
    for (const bool record_failure : {false, true}) {
        const auto child = ::fork();
        expect(child >= 0, "create isolated retained-resource failure control");
        if (!child) {
            try {
                native.current = nullptr;
                native.ready = false;
                native.callbacks = 0;
                native.record_status = record_failure ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
                native.callback_status = record_failure ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
                auto gate = std::make_shared<CudaSessionHostFunction>();
                std::weak_ptr<CudaSessionHostFunction> retained = gate;
                {
                    CudaExecutionSession session({11, 1}, 11, dispatch());
                    expect(session.initialize(0).status == CUDA_SUCCESS, "partial control context");
                    auto barrier = session.accept_external_barrier(0, gate);
                    expect(
                        barrier.status == CUDA_SUCCESS &&
                            session.enqueue_host_function(barrier.barrier, 0, gate) != CUDA_SUCCESS,
                        "partial native callback or event failure is observed");
                    expect(
                        session.enqueue_host_function(barrier.barrier, 0, gate) != CUDA_SUCCESS &&
                            native.callbacks == 1,
                        "uncertain callback is never replayed");
                    gate.reset();
                    expect(!retained.expired() && session.shutdown().status != CUDA_SUCCESS,
                           "actual native userdata remains retained after caller lease loss");
                    // This private test releases only CPU wait so its child can
                    // finish. It never claims CUDA retirement or recovery.
                    retained.lock()->finish();
                    native.callback.join();
                }
                expect(!retained.expired() && !native.events.empty(),
                       "unknown physical event and callback owner remain quarantined");
                std::_Exit(0);
            } catch (...) {
                std::_Exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "bounded isolated partial callback controls preserve unknown owners");
    }
}
void shared_device_fence_blocks_other_owner() {
    const auto child = ::fork();
    expect(child >= 0, "create isolated device uncertainty control");
    if (!child) {
        try {
            native.current = nullptr;
            native.ready = false;
            native.record_status = native.callback_status = CUDA_SUCCESS;
            const auto functions = dispatch();
            auto device_owner = std::make_shared<CudaProcessMemoryDevice>(
                functions.process_memory, [] { return CUDA_SUCCESS; });
            auto first = std::make_shared<CudaProcessMemoryBackend>(
                CudaProcessMemoryConfig{21, 0}, std::make_shared<glimmer::core::QuotaLedger>(128),
                device_owner);
            auto second = std::make_shared<CudaProcessMemoryBackend>(
                CudaProcessMemoryConfig{22, 0}, std::make_shared<glimmer::core::QuotaLedger>(128),
                device_owner);
            CudaExecutionSession session({22, 1}, 22, functions);
            expect(session.attach_process_memory(second) == CUDA_SUCCESS &&
                       session.initialize(0).status == CUDA_SUCCESS,
                   "second process owns an independent live context");
            const auto allocation = session.allocate(64);
            expect(allocation.status == CUDA_SUCCESS, "second process classic allocation");
            CudaSessionMemory operation{.kind = CudaMemoryKind::kFill32,
                                        .destination = {allocation.id, 0},
                                        .count = 16,
                                        .value = 0x12345678,
                                        .is_asynchronous = true};
            const auto issued = session.accept_memory(1, operation);
            expect(issued.accepted() && session.issue_next().sequence == 1,
                   "second owner issues one physical command before uncertainty");
            const auto physical = session.capture_command_frontier(1);
            const auto queued = session.accept_memory(2, operation);
            expect(physical.status == CUDA_SUCCESS && queued.accepted(),
                   "another owner's unissued command is already accepted");
            const auto baseline =
                first->pool_control({CudaProcessMemoryOperation::kGetDefaultPool, 0, true});
            expect(baseline.status == CUDA_SUCCESS, "first process logical pool initialized");
            native.pool = reinterpret_cast<CUmemoryPool>(0x2000);
            native.failed_selection = native.pool;
            const auto failure =
                first->pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true});
            expect(failure.is_quarantined && failure.status != CUDA_SUCCESS &&
                       second->native_health() == CUDA_ERROR_UNKNOWN &&
                       !second->owner()->snapshot().is_quarantined,
                   "unproven restore fences the shared device, not only its failing process");
            const auto fills = native.fills;
            const auto allocations = native.allocations;
            const auto frees = native.frees;
            const auto destroys = native.destroys;
            const auto callbacks = native.callbacks;
            const auto events = native.events.size();
            const auto denied = session.issue_next();
            expect(denied.state == CudaExecutionIssueState::kError &&
                       denied.native_status == CUDA_ERROR_UNKNOWN && native.fills == fills,
                   "already accepted command cannot issue after another owner's failure");
            expect(session.allocate(16).status == CUDA_ERROR_UNKNOWN &&
                       native.allocations == allocations,
                   "cold native entry is fenced across process owners");
            auto gate = std::make_shared<CudaSessionHostFunction>();
            const auto barrier = session.accept_external_barrier(0, gate);
            expect(
                barrier.status == CUDA_SUCCESS &&
                    session.enqueue_host_function(barrier.barrier, 0, gate) == CUDA_ERROR_UNKNOWN &&
                    native.callbacks == callbacks && native.events.size() == events,
                "new external native actor cannot receive a grant after uncertainty");
            expect(session.query_frontier(physical.frontier) == CUDA_ERROR_NOT_READY &&
                       session.poll().status == CUDA_SUCCESS && native.frees == frees &&
                       native.destroys == destroys,
                   "device fence cannot fabricate physical stop or release live resources");
            expect(session.reject_queued(CUDA_ERROR_UNKNOWN) == 1 &&
                       session.host_result(queued.receipt).has_value(),
                   "CPU-known unissued command receives an error without native issue");
            native.ready = true;
            expect(session.poll().status == CUDA_SUCCESS &&
                       session.query_frontier(physical.frontier) == CUDA_SUCCESS &&
                       native.fills == fills && native.frees == frees &&
                       native.destroys == destroys,
                   "real physical completion remains observable under the sticky issue fence");
            // No cleanup authority is inferred from the unknown device pool.
            // Retained resources are intentionally confined to this child.
            std::_Exit(0);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            std::_Exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "shared device fence regression passes in isolated child");
}
}  // namespace
int main() {
    try {
        run();
        retained_callback_failures();
        shared_device_fence_blocks_other_owner();
        std::cout << "Process memory Session ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        // A failed assertion must not destroy a still-joinable callback thread.
        if (native.callback.joinable()) std::_Exit(1);
        return 1;
    }
}
