#include "runtime.h"
#include "host_stage_metrics.h"
#include "async_audit_log.h"
#include "business_owner_lease.h"
#include "internal/process_identity.h"
#include "internal/cuda_gemm_signal_layout.h"
#include "internal/cuda_sgemm_image.h"
#include "glimmer/control/execution_blas_protocol.h"
#include "glimmer/control/execution_function_protocol.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/transform/cuda_module_image.h"
#include "../transform/cuda_image_fixture.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/ostream_sink.h>

namespace poll_wait_fixture {
// Only the serialized service owner arms observation. Other callers still
// delegate unchanged, and no injected readiness or completion can release work.
thread_local bool captures_next{};
thread_local std::int64_t observed_wait_ns{-1};
thread_local int disconnect_descriptor{-1};
thread_local unsigned int polls_before_disconnect{};
thread_local bool disconnected{};
void observe_wait(const timespec* timeout) {
    if (polls_before_disconnect && --polls_before_disconnect == 0)
        disconnected = ::shutdown(disconnect_descriptor, SHUT_RDWR) == 0;
    if (!captures_next) return;
    captures_next = false;
    observed_wait_ns =
        timeout ? static_cast<std::int64_t>(timeout->tv_sec) * 1000000000 + timeout->tv_nsec : -1;
}
}  // namespace poll_wait_fixture
// GNU ld --wrap requires these exact linker symbols, not public C++ naming.
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __real_ppoll(pollfd*, nfds_t, const timespec*, const sigset_t*);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __wrap_ppoll(pollfd* descriptors, nfds_t count, const timespec* timeout,
                            const sigset_t* signals) {
    poll_wait_fixture::observe_wait(timeout);
    return __real_ppoll(descriptors, count, timeout, signals);
}
#ifdef __GLIBC__
// Keep the optimized libc buffer-size check; observe without changing any argument.
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __real___ppoll_chk(pollfd*, nfds_t, const timespec*, const sigset_t*, std::size_t);
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __wrap___ppoll_chk(pollfd* descriptors, nfds_t count, const timespec* timeout,
                                  const sigset_t* signals, std::size_t bytes) {
    poll_wait_fixture::observe_wait(timeout);
    return __real___ppoll_chk(descriptors, count, timeout, signals, bytes);
}
#endif

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void business_owner_lease_contract() {
    std::weak_ptr<int> weak;
    std::weak_ptr<int> expired_lease;
    {
        auto resource = std::make_shared<int>(42);
        weak = resource;
        glimmer::app::BusinessOwnerLease<int> owner;
        owner.bind(resource);
        auto observer = resource;
        expect(!owner.has_leases(), "ordinary observation copies cannot block owner retirement");
        auto first = owner.acquire();
        auto second = first;
        expired_lease = first;
        expect(owner.has_leases() && first.get() == resource.get(),
               "an explicit business lease retains the exact owner");
        first.reset();
        expect(owner.has_leases(), "copied leases retain their shared business obligation");
        std::thread release([lease = std::move(second)]() mutable { lease.reset(); });
        release.join();
        expect(!owner.has_leases(), "the final business lease releases only its own obligation");
        resource.reset();
        observer.reset();
        expect(!weak.expired(), "the registry anchor retains the resource until owner teardown");
    }
    expect(weak.expired() && expired_lease.expired(),
           "weak lease references cannot retain the resource after owner teardown");
    std::weak_ptr<int> surviving_weak;
    std::shared_ptr<int> outstanding;
    {
        auto resource = std::make_shared<int>(73);
        surviving_weak = resource;
        glimmer::app::BusinessOwnerLease<int> original;
        original.bind(resource);
        outstanding = original.acquire();
        auto moved = std::move(original);
        expect(moved.has_leases(), "moving the registry preserves its outstanding business lease");
        resource.reset();
    }
    expect(!surviving_weak.expired() && *outstanding == 73,
           "an outstanding lease retains its resource after registry teardown");
    std::thread release([lease = std::move(outstanding)]() mutable { lease.reset(); });
    release.join();
    expect(surviving_weak.expired(), "the last external lease releases the retired resource");
}
class FailingAuditSink final : public spdlog::sinks::base_sink<std::mutex> {
   public:
    bool fails_log{}, fails_flush{};
    unsigned int fail_on_flush{}, flush_calls{};

   private:
    void sink_it_(const spdlog::details::log_msg&) override {
        if (fails_log) throw std::runtime_error("injected audit write failure");
    }
    void flush_() override {
        ++flush_calls;
        if (fails_flush || flush_calls == fail_on_flush)
            throw std::runtime_error("injected audit flush failure");
    }
};
void async_audit_contract() {
    std::ostringstream captured;
    auto logger = spdlog::default_logger();
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(captured);
    sink->set_pattern("%v");
    logger->sinks().push_back(sink);
    {
        auto audit = std::make_unique<glimmer::app::AsyncAuditLog>();
        expect(audit->info("audit-order={}", 1) && audit->info("audit-order={}", 2),
               "bounded audit enqueue accepts ordinary records");
        const auto child = ::fork();
        expect(child >= 0, "fork async audit ownership control");
        if (!child) {
            const bool invalid = audit->lost();
            audit.reset();
            ::_exit(invalid ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork child discards inherited worker without using its locks");
        audit->flush();
        const auto output = captured.str();
        const auto first = output.find("audit-order=1");
        const auto second = output.find("audit-order=2");
        expect(first != std::string::npos && second != std::string::npos && first < second &&
                   !audit->lost(),
               "audit drain preserves FIFO order without loss");
        const std::string oversized(4 * 1024 * 1024 + 1, 'x');
        expect(!audit->info("{}", oversized) && audit->lost(),
               "audit overflow marks evidence lost without an unbounded queue");
    }
    logger->sinks().pop_back();
    for (const bool fails_flush : {false, true}) {
        auto failure = std::make_shared<FailingAuditSink>();
        failure->fails_log = !fails_flush;
        failure->fails_flush = fails_flush;
        logger->sinks().push_back(failure);
        {
            glimmer::app::AsyncAuditLog audit;
            expect(audit.info("injected audit sink failure"), "enqueue remains independent of I/O");
            audit.flush();
            expect(audit.lost(), "internally handled sink write or flush errors invalidate audit");
        }
        logger->sinks().pop_back();
    }
}
struct FakeNative {
    CUcontext current{};
    CUresult launch_status{CUDA_SUCCESS};
    CUresult query_failure{CUDA_SUCCESS};
    bool is_ready{};
    unsigned int launches{}, destroys{}, event_destroys{}, synchronizes{};
    unsigned int driver_initializations{}, context_creates{};
    std::vector<std::unique_ptr<int>> events;
    bool is_prepared{};
    char private_function{};
    std::vector<std::byte> counters;
    void* host{};
    unsigned int initializations{}, host_frees{};
    unsigned int context_flags{};
    bool is_multiplexed{}, has_released{}, is_dependent_ready{};
    CUresult dependent_status{CUDA_SUCCESS};
    std::array<char, 2> streams{};
    unsigned int stream_count{}, stream_destroys{}, context_reclaimed_streams{};
    std::vector<CUstream> issued_streams;
    std::string expected_module;
    std::vector<std::byte> expected_binary;
    const char* module_source{};
    unsigned int module_loads{}, module_unloads{};
    Kind kind{Kind::kPreemptive};
    bool is_cross_profile{};
    std::array<char, 5> profile_functions{};
    std::vector<unsigned int> profiled_kinds;
    std::vector<std::array<std::uint32_t, 2>> slices;
    bool is_ordinary_chain{};
    IssueDepthMetrics* ordinary_depth{};
    unsigned int peak_unretired{}, overlapping_batches{};
    unsigned int ordinary_intake_width{1};
} native;
struct CacheNative {
    ExecutionService* service{};
    std::array<char, 2> contexts{}, originals{}, privates{};
    std::array<const void*, 2> images{};
    unsigned int creates{}, function_queries{}, parameter_queries{};
} cache_native;
CUresult CUDAAPI initialize(unsigned int) {
    ++native.driver_initializations;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* device, int ordinal) {
    *device = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current(CUcontext* result) {
    *result = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* result, unsigned int flags, CUdevice) {
    ++native.context_creates;
    native.context_flags = flags;
    native.current = reinterpret_cast<CUcontext>(&native);
    *result = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext value) {
    native.current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext value) {
    expect(value == reinterpret_cast<CUcontext>(&native), "destroy only owned native context");
    native.current = nullptr;
    native.context_reclaimed_streams += native.stream_count;
    native.stream_count = 0;
    ++native.destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_load(CUmodule* result, const void* image) {
    if (!native.expected_binary.empty()) {
        native.module_source = static_cast<const char*>(image);
        expect(
            std::memcmp(image, native.expected_binary.data(), native.expected_binary.size()) == 0,
            "original binary callback receives all owned large image bytes");
    }
    if (!native.expected_module.empty()) {
        native.module_source = static_cast<const char*>(image);
        expect(std::string_view(native.module_source) == native.expected_module,
               "actual original module intake receives every immutable large source byte");
    }
    ++native.module_loads;
    *result = reinterpret_cast<CUmodule>(&native);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_unload(CUmodule) {
    ++native.module_unloads;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI function_get(CUfunction* result, CUmodule, const char* name) {
    if (!native.expected_binary.empty())
        expect(std::memcmp(native.module_source, native.expected_binary.data(),
                           native.expected_binary.size()) == 0,
               "binary source lifetime is independent of received descriptor/mapping lifetime");
    if (!native.expected_module.empty())
        expect(std::string_view(native.module_source) == native.expected_module,
               "session source remains owned after received file and mapping retire");
    *result = std::string_view(name).find("__glimmer_ptb") == std::string_view::npos
                  ? reinterpret_cast<CUfunction>(&native)
                  : reinterpret_cast<CUfunction>(&native.private_function);
    return CUDA_SUCCESS;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact native CUDA ABI.
CUresult CUDAAPI parameter_info(CUfunction function, std::size_t index, std::size_t* offset,
                                std::size_t* bytes) {
    if (native.is_prepared && function == reinterpret_cast<CUfunction>(&native.private_function)) {
        constexpr std::array<std::size_t, 5> offsets{0, 4, 8, 16, 24};
        expect(index < offsets.size(), "query only known generated arguments");
        *offset = offsets[index];
        *bytes = index < 3 ? 4 : 8;
        return CUDA_SUCCESS;
    }
    throw std::runtime_error("zero-parameter PTX must not probe an invalid SDK index");
}
CUresult CUDAAPI allocate(CUdeviceptr* pointer, std::size_t bytes) {
    if (native.is_prepared) {
        native.counters.resize(bytes);
        *pointer = 0x1000;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult CUDAAPI release(CUdeviceptr pointer) {
    if (native.is_prepared && pointer == 0x1000) {
        native.counters.clear();
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI upload(CUdeviceptr, const void*, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI download(void*, CUdeviceptr, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI stream_create(CUstream* result, unsigned int) {
    if (native.is_multiplexed && native.stream_count < native.streams.size()) {
        *result = reinterpret_cast<CUstream>(&native.streams[native.stream_count++]);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult CUDAAPI stream_destroy(CUstream) {
    if (native.is_multiplexed) {
        ++native.stream_destroys;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI event_create(CUevent* result, unsigned int) {
    native.events.push_back(std::make_unique<int>(0));
    *result = reinterpret_cast<CUevent>(native.events.back().get());
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent event, CUstream) {
    *reinterpret_cast<int*>(event) = static_cast<int>(native.launches);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent event) {
    if (native.query_failure != CUDA_SUCCESS) return native.query_failure;
    if (native.is_multiplexed) {
        const auto issued = *reinterpret_cast<int*>(event);
        if (issued == 1) return native.has_released ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
        if (issued == 3) return native.is_dependent_ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    return native.is_ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent) {
    if (native.is_ordinary_chain)
        expect(native.launches % 32 == 0,
               "ordinary batch cannot retire before every original has issued");
    ++native.event_destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_sync(CUstream) {
    ++native.synchronizes;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI launch(CUfunction function, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream stream, void**,
                        void** extra) {
    if (native.is_ordinary_chain) {
        expect(function == reinterpret_cast<CUfunction>(&native) && !stream,
               "ordinary chain issues original functions on exactly one legacy stream");
        if (native.ordinary_depth && native.launches % 32 == 0) {
            std::uint64_t admitted{};
            for (std::size_t index = 0; index < IssueDepthMetrics::bucket_capacity; ++index)
                admitted += native.ordinary_depth->bucket(index).admitted;
            expect(native.ordinary_depth->valid() &&
                       admitted == native.launches + native.ordinary_intake_width,
                   "single-peer intake admits exactly its bounded round width before issue");
            ++native.overlapping_batches;
        }
        expect(native.launches - native.event_destroys == native.launches % 32,
               "every successor retains all earlier host-issued batch predecessors");
        native.peak_unretired =
            std::max(native.peak_unretired, native.launches + 1 - native.event_destroys);
        expect(native.peak_unretired <= 32, "ordinary chain has bounded retained predecessors");
    }
    ++native.launches;
    if (native.is_multiplexed) {
        native.issued_streams.push_back(stream);
        if (stream == reinterpret_cast<CUstream>(&native.streams[1])) native.has_released = true;
        if (!stream) return native.dependent_status;
    }
    if (function == reinterpret_cast<CUfunction>(&native.private_function)) {
        expect(extra && *static_cast<std::size_t*>(extra[3]) == 32, "actual generated private ABI");
        CUdeviceptr counter{};
        std::memcpy(&counter, static_cast<const std::byte*>(extra[1]) + 16, 8);
        const std::uint64_t claimed = 1;
        std::memcpy(native.counters.data() + (counter - 0x1000), &claimed, sizeof(claimed));
    }
    return native.launch_status;
}
CUresult CUDAAPI host_allocate(void** pointer, std::size_t bytes, unsigned int) {
    native.host = ::operator new (bytes, std::align_val_t{64});
    *pointer = native.host;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_release(void* pointer) {
    ++native.host_frees;
    ::operator delete (pointer, std::align_val_t{64});
    native.host = nullptr;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_map(CUdeviceptr* device, void* pointer, unsigned int) {
    *device = reinterpret_cast<CUdeviceptr>(pointer);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload_async(CUdeviceptr device, const void* pointer, std::size_t bytes,
                              CUstream) {
    ++native.initializations;
    std::memcpy(native.counters.data() + (device - 0x1000), pointer, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download_async(void* pointer, CUdeviceptr device, std::size_t bytes, CUstream) {
    std::memcpy(pointer, native.counters.data() + (device - 0x1000), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI occupancy(int* active, CUfunction, int, std::size_t) {
    *active = 4;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_get_flags(unsigned int* flags) {
    *flags = native.context_flags;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set_flags(unsigned int flags) {
    native.context_flags = flags;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI copy_device(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes) {
    expect(!destination && !source && !bytes, "service forwards the validated zero-copy metadata");
    ++native.launches;
    return native.launch_status;
}
const CudaSessionDispatch kDispatch{
    {current, launch, event_create, event_record, event_query, event_destroy, stream_sync,
     copy_device},
    initialize,
    device_get,
    context_create,
    context_set,
    context_destroy,
    module_load,
    module_unload,
    function_get,
    parameter_info,
    allocate,
    release,
    upload,
    download,
    stream_create,
    stream_destroy,
    host_allocate,
    host_release,
    host_map,
    upload_async,
    download_async,
    occupancy,
    nullptr,
    context_get_flags,
    context_set_flags,
    [](CUfunction, CUfunction_attribute attribute, int value) {
        expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
                   value == CU_SHAREDMEM_CARVEOUT_MAX_SHARED,
               "configure only the owned function preference");
        return CUDA_SUCCESS;
    }};
void audit_terminal_failure_contract(const std::string& path) {
    auto logger = spdlog::default_logger();
    std::ostringstream captured;
    auto capture = std::make_shared<spdlog::sinks::ostream_sink_mt>(captured);
    capture->set_pattern("%v");
    auto failure = std::make_shared<FailingAuditSink>();
    failure->fail_on_flush = 2;
    logger->sinks().push_back(capture);
    logger->sinks().push_back(failure);
    {
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.should_audit_native = true;
        ExecutionService service(std::move(config), kDispatch);
        expect(service.start() && !service.audit_healthy() && service.request_stop(),
               "audit sink failure cannot change normal service cleanup");
        expect(failure->flush_calls >= 2 && !service.audit_healthy() &&
                   captured.str().find("\"family\":2,\"event\":2") != std::string::npos,
               "a written terminal marker cannot hide a final flush failure");
    }
    logger->sinks().pop_back();
    logger->sinks().pop_back();
}
ExecutionWireReply success(std::optional<ExecutionWireReply> result) {
    expect(result && result->disposition == ExecutionWireDisposition::kComplete &&
               result->status == 0 && result->original_status == 0,
           "only actual native operation can report original SDK success");
    return std::move(*result);
}
namespace lt_query_fixture {
// The service owner alone touches fake native objects. Client-side controls and
// observations are atomic; every context transition is an actual socket request.
struct State {
    std::array<char, 2> contexts{}, handles{};
    char operation{}, layout{}, preference{};
    unsigned int creates{}, handle_creates{}, handle_destroys{}, metadata_destroys{};
    std::atomic<unsigned int> activations{}, versions{}, preferences{}, heuristics{}, checks{};
    std::atomic<bool> fails_activation{};
    std::atomic<cublasStatus_t> query_status{CUBLAS_STATUS_SUCCESS};
    std::atomic<int> current_index{-1};
} state;

std::size_t current_index() {
    for (std::size_t index = 0; index < state.contexts.size(); ++index)
        if (native.current == reinterpret_cast<CUcontext>(&state.contexts[index])) return index;
    throw std::runtime_error("Lt callback requires one of two actual owned contexts");
}
std::array<std::byte, 64> algorithm(std::size_t index) {
    std::array<std::byte, 64> bytes{};
    bytes.fill(static_cast<std::byte>(0x31U + index));
    return bytes;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact vendor descriptor ABI.
void validate(cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
              cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d) {
    const auto index = current_index();
    expect(handle == reinterpret_cast<cublasLtHandle_t>(&state.handles[index]) &&
               operation == reinterpret_cast<cublasLtMatmulDesc_t>(&state.operation) &&
               a == reinterpret_cast<cublasLtMatrixLayout_t>(&state.layout) && a == b && a == c &&
               a == d,
           "Lt query uses its own native handle/current context and authenticated metadata");
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaSessionDispatch dispatch() {
    auto result = kDispatch;
    result.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(state.creates < 2 && !native.current, "create two distinct detached native owners");
        state.current_index = static_cast<int>(state.creates);
        *output = native.current = reinterpret_cast<CUcontext>(&state.contexts[state.creates++]);
        return CUDA_SUCCESS;
    };
    result.context_set_current = [](CUcontext context) {
        ++state.activations;
        if (context == reinterpret_cast<CUcontext>(&state.contexts[0]) &&
            state.fails_activation.exchange(false))
            return CUDA_ERROR_INVALID_VALUE;
        expect(!context || context == reinterpret_cast<CUcontext>(&state.contexts[0]) ||
                   context == reinterpret_cast<CUcontext>(&state.contexts[1]),
               "activate only a retained context identity");
        native.current = context;
        state.current_index = context ? static_cast<int>(current_index()) : -1;
        return CUDA_SUCCESS;
    };
    result.context_destroy = [](CUcontext context) {
        expect(context && context == native.current, "context cleanup activates its actual owner");
        (void)current_index();
        ++native.destroys;
        native.current = nullptr;
        state.current_index = -1;
        return CUDA_SUCCESS;
    };
    result.context_get_flags = [](unsigned int* output) {
        (void)current_index();
        *output = 0;
        return CUDA_SUCCESS;
    };
    result.lt.create = [](cublasLtHandle_t* output) {
        *output = reinterpret_cast<cublasLtHandle_t>(&state.handles[current_index()]);
        ++state.handle_creates;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.destroy = [](cublasLtHandle_t handle) {
        expect(handle == reinterpret_cast<cublasLtHandle_t>(&state.handles[current_index()]),
               "destroy native Lt handle in its actual context");
        ++state.handle_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native layout factory ABI.
    result.lt.layout_create = [](cublasLtMatrixLayout_t* output, cudaDataType type,
                                 std::uint64_t rows, std::uint64_t columns, std::int64_t leading) {
        expect(type == CUDA_R_32F && rows == 2 && columns == 2 && leading == 2,
               "exact layout creation through independent metadata owner");
        *output = reinterpret_cast<cublasLtMatrixLayout_t>(&state.layout);
        return CUBLAS_STATUS_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    result.lt.operation_create = [](cublasLtMatmulDesc_t* output, cublasComputeType_t compute,
                                    cudaDataType scale) {
        expect(compute == CUBLAS_COMPUTE_32F && scale == CUDA_R_32F,
               "exact native operation creation");
        *output = reinterpret_cast<cublasLtMatmulDesc_t>(&state.operation);
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.preference_create = [](cublasLtMatmulPreference_t* output) {
        *output = reinterpret_cast<cublasLtMatmulPreference_t>(&state.preference);
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.layout_destroy = [](cublasLtMatrixLayout_t object) {
        expect(object == reinterpret_cast<cublasLtMatrixLayout_t>(&state.layout),
               "destroy only the owned layout");
        ++state.metadata_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.operation_destroy = [](cublasLtMatmulDesc_t object) {
        expect(object == reinterpret_cast<cublasLtMatmulDesc_t>(&state.operation),
               "destroy only the owned operation");
        ++state.metadata_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.preference_destroy = [](cublasLtMatmulPreference_t object) {
        expect(object == reinterpret_cast<cublasLtMatmulPreference_t>(&state.preference),
               "destroy only the owned preference");
        ++state.metadata_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.layout_set = [](cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void*,
                              std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.operation_set = [](cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*,
                                 std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.preference_set = [](cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                                  const void*, std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.layout_get = [](cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void*,
                              std::size_t, std::size_t*) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.operation_get = [](cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void*,
                                 std::size_t, std::size_t*) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.preference_get = [](cublasLtMatmulPreference_t object,
                                  cublasLtMatmulPreferenceAttributes_t attribute, void* output,
                                  std::size_t bytes, std::size_t* written) {
        (void)current_index();
        expect(object == reinterpret_cast<cublasLtMatmulPreference_t>(&state.preference) &&
                   attribute == CUBLASLT_MATMUL_PREF_SEARCH_MODE && bytes == sizeof(std::uint32_t),
               "private preference observation is bounded and context-bound");
        const std::uint32_t mode{};
        std::memcpy(output, &mode, sizeof(mode));
        *written = sizeof(mode);
        ++state.preferences;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.version = []() -> std::size_t {
        (void)current_index();
        ++state.versions;
        return 120604;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original Lt query ABIs.
    result.lt.heuristic = [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                             cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
                             cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
                             cublasLtMatmulPreference_t preference, int requested,
                             cublasLtMatmulHeuristicResult_t* output, int* count) {
        validate(handle, operation, a, b, c, d);
        expect(preference == reinterpret_cast<cublasLtMatmulPreference_t>(&state.preference) &&
                   requested == 1,
               "heuristic preserves the requested preference and count");
        ++state.heuristics;
        const auto status = state.query_status.load();
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        *output = {};
        const auto bytes = algorithm(current_index());
        std::memcpy(&output->algo, bytes.data(), bytes.size());
        output->workspaceSize = 256;
        output->wavesCount = 1.5F;
        *count = 1;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.algorithm_check =
        [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
           cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
           const cublasLtMatmulAlgo_t* candidate, cublasLtMatmulHeuristicResult_t* output) {
            validate(handle, operation, a, b, c, d);
            const auto expected = algorithm(current_index());
            expect(std::memcmp(candidate, expected.data(), expected.size()) == 0,
                   "algorithm check consumes only this exact native handle's recorded algorithm");
            ++state.checks;
            const auto status = state.query_status.load();
            if (status != CUBLAS_STATUS_SUCCESS) return status;
            *output = {};
            output->algo = *candidate;
            output->workspaceSize = 512;
            output->wavesCount = 2.5F;
            return CUBLAS_STATUS_SUCCESS;
        };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return result;
}

struct Query {
    std::uint64_t handle{}, metadata{}, operation{}, layout{}, preference{};
    std::array<std::byte, 64> algorithm{};
    std::vector<std::byte> encode(bool is_check) const {
        ExecutionPayloadWriter writer;
        expect(
            writer.write_u64(handle) && writer.write_u64(metadata) && writer.write_u64(operation),
            "encode authenticated Lt query owners");
        for (unsigned int index = 0; index < 4; ++index)
            expect(writer.write_u64(layout), "encode immutable positional layouts");
        expect(is_check ? writer.write_blob(algorithm)
                        : writer.write_u64(preference) && writer.write_u32(1),
               "encode exact Lt query suffix");
        return writer.take();
    }
};
std::uint64_t create_metadata(ExecutionSocketClient& client, CudaLtObjectKind kind) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(static_cast<std::uint32_t>(kind)) && writer.write_u32(CUDA_R_32F) &&
               writer.write_u32(CUBLAS_COMPUTE_32F) && writer.write_u32(CUDA_R_32F) &&
               writer.write_u64(2) && writer.write_u64(2) && writer.write_u64(2),
           "encode exact metadata factory input");
    const auto reply = success(client.exchange(ExecutionOpcode::kCreateLtMetadata, writer.take()));
    expect(reply.id != 0 && !reply.parameter_bytes, "factory publishes a healthy owned identity");
    return reply.id;
}
void rejected(std::optional<ExecutionWireReply> reply, CUresult status) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
               reply->status == status && !reply->original_status && !reply->id &&
               !reply->parameter_bytes && reply->data.empty(),
           "pre-native query rejection has no invented SDK status or quarantine");
}
void verify_result(std::optional<ExecutionWireReply> response, const Query& query, bool is_check) {
    if (!response || response->disposition != ExecutionWireDisposition::kComplete ||
        response->status != 0 || response->original_status != 0)
        throw std::runtime_error(
            "Lt owner query failed: driver_status=" +
            (response ? std::to_string(response->status) : "missing") + "; original_status=" +
            (response && response->original_status ? std::to_string(*response->original_status)
                                                   : "absent") +
            "; heuristic_calls=" + std::to_string(state.heuristics.load()) +
            "; check_calls=" + std::to_string(state.checks.load()));
    const auto reply = success(std::move(response));
    expect(reply.address == 120604 && !reply.parameter_bytes, "exact healthy native SDK identity");
    ExecutionPayloadReader input(reply.data);
    std::uint32_t count{}, status{}, waves{}, reserved{};
    std::uint64_t workspace{};
    std::span<const std::byte> bytes;
    expect(input.read_u32(count) && count == 1 && input.read_blob(bytes) &&
               std::ranges::equal(bytes, query.algorithm) && input.read_u64(workspace) &&
               workspace == (is_check ? 512U : 256U) && input.read_u32(status) && status == 0 &&
               input.read_u32(waves) &&
               waves == std::bit_cast<std::uint32_t>(is_check ? 2.5F : 1.5F),
           "native query result carries exact algorithm bytes, workspace, state and waves");
    for (unsigned int index = 0; index < 4; ++index)
        expect(input.read_u32(reserved) && reserved == 0, "native reserved fields remain exact");
    expect(input.empty(), "no unvalidated algorithm result suffix");
}
void run_client(const std::string& path, TaskPeerIdentity peer) {
    ExecutionSocketClient first({path, peer, 3000}), second({path, peer, 3000}),
        metadata({path, peer, 3000});
    success(first.open());
    success(second.open());
    ExecutionPayloadWriter role;
    expect(role.write_u32(kExecutionLtMetadataRole) && role.write_u32(0), "encode metadata role");
    const auto opened = metadata.open(role.take());
    expect(opened && opened->disposition == ExecutionWireDisposition::kComplete &&
               opened->status == 0 && !opened->original_status &&
               decode_execution_bootstrap_ownership(opened->data) ==
                   ExecutionBootstrapOwnership::kMetadataRetained &&
               metadata.session() != first.session() && metadata.session() != second.session(),
           "authenticated independent metadata owner has no native context");
    Query a, b;
    a.handle = success(first.exchange(ExecutionOpcode::kCreateLt)).id;
    b.handle = success(second.exchange(ExecutionOpcode::kCreateLt)).id;
    a.metadata = b.metadata = metadata.session();
    a.operation = b.operation = create_metadata(metadata, CudaLtObjectKind::kOperation);
    a.layout = b.layout = create_metadata(metadata, CudaLtObjectKind::kLayout);
    a.preference = b.preference = create_metadata(metadata, CudaLtObjectKind::kPreference);
    a.algorithm = algorithm(0);
    b.algorithm = algorithm(1);
    const auto switch_to = [](ExecutionSocketClient& owner, int index) {
        success(owner.exchange(ExecutionOpcode::kGetContextFlags));
        expect(state.current_index == index, "awaited opposite-client RPC establishes current");
    };
    for (const bool is_check : {false, true}) {
        const auto opcode =
            is_check ? ExecutionOpcode::kLtAlgorithmCheck : ExecutionOpcode::kLtHeuristic;
        switch_to(second, 1);
        verify_result(first.exchange(opcode, a.encode(is_check)), a, is_check);
        switch_to(first, 0);
        verify_result(second.exchange(opcode, b.encode(is_check)), b, is_check);

        switch_to(second, 1);
        const auto activations = state.activations.load();
        const auto versions = state.versions.load();
        const auto heuristics = state.heuristics.load();
        const auto checks = state.checks.load();
        for (unsigned int invalid = 0; invalid < (is_check ? 4U : 5U); ++invalid) {
            auto bad = a;
            if (invalid == 0) bad.metadata = second.session();
            if (invalid == 1) bad.operation = a.layout;
            if (invalid == 2) bad.layout = 0;
            if (invalid == 3) bad.metadata = 0;
            if (invalid == 4) bad.preference = a.layout;
            rejected(first.exchange(opcode, bad.encode(is_check)), CUDA_ERROR_INVALID_HANDLE);
        }
        auto trailing = a.encode(is_check);
        trailing.push_back(std::byte{});
        rejected(first.exchange(opcode, trailing), CUDA_ERROR_INVALID_VALUE);
        expect(state.activations == activations && state.versions == versions &&
                   state.heuristics == heuristics && state.checks == checks &&
                   state.current_index == 1,
               "invalid borrowed owners/kinds/payloads have no activation or SDK side effects");

        state.fails_activation = true;
        rejected(first.exchange(opcode, a.encode(is_check)), CUDA_ERROR_INVALID_VALUE);
        expect(!state.fails_activation && state.activations == activations + 1 &&
                   state.versions == versions && state.heuristics == heuristics &&
                   state.checks == checks && state.current_index == 1,
               "failed activation rejects before any original or private Lt callback");
        verify_result(first.exchange(opcode, a.encode(is_check)), a, is_check);

        switch_to(second, 1);
        state.query_status = CUBLAS_STATUS_NOT_SUPPORTED;
        const auto failure = first.exchange(opcode, a.encode(is_check));
        expect(failure && failure->disposition == ExecutionWireDisposition::kComplete &&
                   failure->status == CUBLAS_STATUS_NOT_SUPPORTED &&
                   failure->original_status == CUBLAS_STATUS_NOT_SUPPORTED &&
                   !failure->parameter_bytes && failure->data.empty(),
               "actual Lt callback failure remains exact SDK provenance without quarantine");
        state.query_status = CUBLAS_STATUS_SUCCESS;
    }
    switch_to(second, 1);
    auto foreign_algorithm = a;
    foreign_algorithm.algorithm = b.algorithm;
    const auto checks = state.checks.load();
    const auto heuristics = state.heuristics.load();
    rejected(first.exchange(ExecutionOpcode::kLtAlgorithmCheck, foreign_algorithm.encode(true)),
             CUDA_ERROR_NOT_SUPPORTED);
    expect(state.checks == checks && state.heuristics == heuristics,
           "another valid handle's algorithm never reaches the original query callback");

    success(second.exchange(ExecutionOpcode::kClose));
    expect(state.current_index == -1, "closing other owner leaves no native current context");
    verify_result(first.exchange(ExecutionOpcode::kLtHeuristic, a.encode(false)), a, false);
    verify_result(first.exchange(ExecutionOpcode::kLtAlgorithmCheck, a.encode(true)), a, true);
    const auto closed = metadata.exchange(ExecutionOpcode::kClose);
    expect(closed && closed->disposition == ExecutionWireDisposition::kComplete &&
               closed->status == 0 && !closed->parameter_bytes,
           "metadata borrows release after each query and permit normal cleanup");
    success(first.exchange(ExecutionOpcode::kClose));
}
}  // namespace lt_query_fixture
void lt_query_context_activation(const std::string& path) {
    native = {};
    const auto peer = current_process_identity();
    expect(peer.has_value(), "authenticate actual same-process Lt clients");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    config.uses_original_kernels = true;
    // Ranked metadata-only requests do not schedule an idle GPU session. This
    // prevents unrelated per-pass scheduling from masking missing activation.
    config.client_priorities = {{*peer, 0}};
    ExecutionService service(std::move(config), lt_query_fixture::dispatch());
    expect(service.start(), "start real Lt query service");
    auto client =
        std::async(std::launch::async, [&] { lt_query_fixture::run_client(path, *peer); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded real-service Lt context regression");
    client.get();
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded normal Lt context cleanup");
    const auto& state = lt_query_fixture::state;
    expect(state.creates == 2 && state.handle_creates == 2 && state.handle_destroys == 2 &&
               state.metadata_destroys == 3 && state.heuristics == 5 && state.checks == 5 &&
               state.versions == 11 && state.preferences == 5 && native.destroys == 2 &&
               !native.current && !native.launches && !native.synchronizes &&
               !native.event_destroys && !native.module_loads && service.request_stop(),
           "two native owners and all metadata clean once, without compute/preparation/drain");
}
void shared_preparation_cache(const std::string& path, bool enabled, bool restores = false) {
    native = {};
    native.is_prepared = true;
    cache_native = {};
    auto dispatch = kDispatch;
    dispatch.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(cache_native.creates < 2, "cache fixture owns exactly two distinct native contexts");
        *output = native.current =
            reinterpret_cast<CUcontext>(&cache_native.contexts[cache_native.creates++]);
        return CUDA_SUCCESS;
    };
    dispatch.context_destroy = [](CUcontext context) {
        expect(context == native.current &&
                   (context == reinterpret_cast<CUcontext>(&cache_native.contexts[0]) ||
                    context == reinterpret_cast<CUcontext>(&cache_native.contexts[1])),
               "cached CPU code cannot substitute another session's native context");
        ++native.destroys;
        native.current = nullptr;
        return CUDA_SUCCESS;
    };
    dispatch.module_load = [](CUmodule* output, const void* image) {
        expect(cache_native.service && cache_native.service->ptx_cache_snapshot().is_valid,
               "native SDK callback may observe cache metadata without cache exclusion");
        const auto index =
            native.current == reinterpret_cast<CUcontext>(&cache_native.contexts[0]) ? 0U : 1U;
        expect(native.current == reinterpret_cast<CUcontext>(&cache_native.contexts[index]) &&
                   std::string_view(static_cast<const char*>(image))
                           .find("kernel__glimmer_ptb_preemptive") != std::string_view::npos,
               "each actual native owner separately loads the immutable transformed code");
        cache_native.images[index] = image;
        *output = reinterpret_cast<CUmodule>(native.current);
        ++native.module_loads;
        return CUDA_SUCCESS;
    };
    dispatch.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        const auto index =
            native.current == reinterpret_cast<CUcontext>(&cache_native.contexts[0]) ? 0U : 1U;
        expect(module == reinterpret_cast<CUmodule>(native.current),
               "cached source never imports another native module handle");
        *output = reinterpret_cast<CUfunction>(std::string_view(name).find("__glimmer_ptb") ==
                                                       std::string_view::npos
                                                   ? &cache_native.originals[index]
                                                   : &cache_native.privates[index]);
        ++cache_native.function_queries;
        return CUDA_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver metadata ABI.
    dispatch.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                                 std::size_t* bytes) {
        const auto context =
            native.current == reinterpret_cast<CUcontext>(&cache_native.contexts[0]) ? 0U : 1U;
        constexpr std::array<std::size_t, 5> offsets{0, 4, 8, 16, 24};
        expect(function == reinterpret_cast<CUfunction>(&cache_native.privates[context]) &&
                   index < offsets.size(),
               "cache hit independently revalidates the native ABI in its actual context");
        *offset = offsets[index];
        *bytes = index < 3 ? 4 : 8;
        ++cache_native.parameter_queries;
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.persistent.workers = 320;
    if (!enabled) config.ptx_cache.max_bytes = 0;
    if (enabled) {
        if (restores)
            config.ptx_cache_input = path + ".ptx";
        else
            config.ptx_cache_output = path + ".ptx";
    }
    ExecutionService service(std::move(config), dispatch);
    cache_native.service = &service;
    expect(!service.save_ptx_cache(), "archive publication requires proven service shutdown");
    expect(service.start(), "start actual service-owned CPU preparation cache");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real cache service peer identity");
    for (unsigned int round = 0; round < 2; ++round) {
        auto client = std::async(std::launch::async, [&] {
            ExecutionSocketClient connection({path, *identity, 3000});
            success(connection.open());
            ExecutionPayloadWriter source;
            expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n.visible "
                                       ".entry kernel() { ret; }"),
                   "identical original source in independent sessions");
            const auto module =
                success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
            ExecutionPayloadWriter lookup;
            expect(lookup.write_u64(module.id) && lookup.write_string("kernel"),
                   "only owned original function identity");
            const auto function =
                success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
            expect(function.id && function.arguments.empty() && function.parameter_bytes == 0,
                   "cached private ABI is never exposed to the application");
            success(connection.exchange(ExecutionOpcode::kClose));
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
            expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded real cache composition progress");
        client.get();
        while (native.destroys != round + 1)
            expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                   "cache borrow retires with its actual native context");
    }
    const auto cache = service.ptx_cache_snapshot();
    expect(cache.is_valid && cache.lookups == 2 &&
               cache.transforms == (enabled && !restores ? 1U : 0U) &&
               cache.hits == (enabled ? (restores ? 2U : 1U) : 0U) &&
               cache.loaded_entries == (restores ? 1U : 0U) &&
               cache.entries == (enabled ? 1U : 0U) &&
               (enabled ? cache.retained_bytes > 0
                        : cache.retained_bytes == 0 && cache.bypasses == 2) &&
               cache_native.creates == 2 && native.module_loads == 2 &&
               cache_native.function_queries == 4 && cache_native.parameter_queries == 10 &&
               native.host_frees == 2 && native.launches == 0 && service.request_stop(),
           "shared preparation reuses CPU transformation only, preserving all native ownership and "
           "metadata work");
    if (enabled)
        expect(cache_native.images[0] == cache_native.images[1],
               "actual SDK loads borrow the same immutable image");
    expect(service.save_ptx_cache(), "publish complete CPU artifacts after actual owner cleanup");
    if (enabled && !restores)
        expect(!service.save_ptx_cache(),
               "repeated publication cannot replace the existing archive");
    if (restores)
        expect(::unlink((path + ".ptx").c_str()) == 0,
               "remove only the owned restart fixture archive");
    cache_native.service = nullptr;
}

void completed_issue_events(const IssueDepthMetrics& metrics, std::uint64_t expected) {
    expect(metrics.event_count() == expected, "one complete event chain per ordinary compute");
    for (std::size_t index = 0; index < metrics.event_count(); ++index) {
        const auto& event = metrics.event(index);
        expect(event.connection && event.sequence && event.opcode && event.plan_serial &&
                   event.admitted_ns && event.admitted_ns <= event.first_observed_ns &&
                   event.first_observed_ns <= event.matched_observed_ns &&
                   event.matched_observed_ns <= event.authorized_ns &&
                   event.authorized_ns <= event.issue_begin_ns &&
                   event.issue_begin_ns <= event.issue_end_ns &&
                   event.issue_end_ns <= event.retired_ns && event.plan_positive_observations &&
                   event.positive_observations >= event.plan_positive_observations &&
                   event.first_positive_observation_generation >
                       event.admission_observation_generation &&
                   event.preparation_required <= 1 &&
                   (event.preparation_required
                        ? event.opcode == static_cast<std::uint32_t>(ExecutionOpcode::kLaunch) &&
                              event.admitted_ns <= event.service_observed_prepared_ns &&
                              event.service_observed_prepared_ns <= event.first_observed_ns
                        : event.service_observed_prepared_ns == 0),
               "actual owner event chain is ordered and same-plan authorized");
    }
}
void selected_variants_execute_through_service(const std::string& path, Kind kind, bool uses_naive,
                                               bool should_profile = false,
                                               bool should_restore = false,
                                               bool observes_depth = false,
                                               bool records_events = false) {
    native = {};
    native.is_prepared = true;
    native.kind = kind;
    native.is_cross_profile = should_profile;
    if (should_restore) native.is_ready = true;
    auto dispatch = kDispatch;
    dispatch.function_get = [](CUfunction* output, CUmodule, const char* name) {
        const std::string_view symbol{name};
        if (symbol != "kernel") {
            constexpr std::array suffixes{"", "_block_static", "_dynamic", "_preemptive",
                                          "_sliced"};
            if (native.is_cross_profile) {
                for (std::size_t index = 0; index < suffixes.size(); ++index)
                    if (symbol == std::string("kernel__glimmer_ptb") + suffixes[index]) {
                        *output = reinterpret_cast<CUfunction>(&native.profile_functions[index]);
                        return CUDA_SUCCESS;
                    }
                expect(false, "only explicitly prepared families acquire native identities");
            }
            expect(symbol == std::string("kernel__glimmer_ptb") +
                                 suffixes.at(static_cast<std::size_t>(native.kind)),
                   "trusted service kind selects the actual generated native function");
        }
        *output = symbol == "kernel" ? reinterpret_cast<CUfunction>(&native)
                                     : reinterpret_cast<CUfunction>(&native.private_function);
        return CUDA_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                                 std::size_t* bytes) {
        if (native.is_cross_profile)
            for (std::size_t family = 0; family < native.profile_functions.size(); ++family)
                if (function == reinterpret_cast<CUfunction>(&native.profile_functions[family]))
                    native.kind = static_cast<Kind>(family);
        const auto count = native.kind == Kind::kStatic        ? 1U
                           : native.kind == Kind::kBlockStatic ? 3U
                           : native.kind == Kind::kDynamic     ? 4U
                                                               : 5U;
        const auto expected_function =
            native.is_cross_profile
                ? &native.profile_functions[static_cast<std::size_t>(native.kind)]
                : &native.private_function;
        expect(function == reinterpret_cast<CUfunction>(expected_function) && index < count,
               "native metadata matches the selected family and empty application ABI");
        const bool wide = native.kind != Kind::kSliced && index >= 3;
        *offset = wide ? 16 + (index - 3) * 8 : index * 4;
        *bytes = wide ? 8 : 4;
        return CUDA_SUCCESS;
    };
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int grid_x,
                                          unsigned int grid_y, unsigned int grid_z,
                                          unsigned int block_x, unsigned int, unsigned int,
                                          unsigned int, CUstream, void**, void** extra) {
        if (native.is_cross_profile) {
            if (function == reinterpret_cast<CUfunction>(&native)) {
                expect(grid_x == 8 && grid_y == 1 && grid_z == 1 && block_x == 32,
                       "profile original preserves the application geometry");
                native.profiled_kinds.push_back(5);
                ++native.launches;
                return CUDA_SUCCESS;
            }
            for (std::size_t family = 0; family < native.profile_functions.size(); ++family)
                if (function == reinterpret_cast<CUfunction>(&native.profile_functions[family]))
                    native.kind = static_cast<Kind>(family);
            native.profiled_kinds.push_back(static_cast<unsigned int>(native.kind));
        }
        const auto bytes = native.kind == Kind::kStatic        ? 4U
                           : native.kind == Kind::kBlockStatic ? 12U
                           : native.kind == Kind::kDynamic     ? 24U
                           : native.kind == Kind::kSliced      ? 20U
                                                               : 32U;
        const auto expected_function =
            native.is_cross_profile
                ? &native.profile_functions[static_cast<std::size_t>(native.kind)]
                : &native.private_function;
        expect(function == reinterpret_cast<CUfunction>(expected_function) && grid_x == 3 &&
                   grid_y == 1 && grid_z == 1 && block_x == 32 && extra &&
                   *static_cast<std::size_t*>(extra[3]) == bytes,
               "service issues actual private family ABI with explicit worker geometry");
        ++native.launches;
        const auto* input = static_cast<const std::byte*>(extra[1]);
        if (native.kind == Kind::kSliced) {
            std::array<std::uint32_t, 2> slice{};
            std::memcpy(slice.data(), input + 12, 8);
            native.slices.push_back(slice);
        } else if (native.kind == Kind::kDynamic || native.kind == Kind::kPreemptive) {
            CUdeviceptr counter{};
            std::memcpy(&counter, input + 16, 8);
            const std::uint64_t completed = 8;
            std::memcpy(native.counters.data() + (counter - 0x1000), &completed, 8);
        }
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.persistent = {3, false, kind};
    auto depth = observes_depth ? std::make_shared<IssueDepthMetrics>(
                                      IssueDepthMetrics::system_clock,
                                      records_events ? IssueDepthMetrics::event_capacity_limit : 0)
                                : nullptr;
    config.issue_depth_metrics = depth;
    config.scheduling_mode = uses_naive ? glimmer::core::ExecutionSchedulingMode::kNaive
                                        : glimmer::core::ExecutionSchedulingMode::kPriority;
    if (should_profile) {
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
        config.profile.workers = {3};
        config.profile.kinds = {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic,
                                Kind::kPreemptive, Kind::kSliced};
        config.profile.warmup_calls = 0;
        config.profile.samples_per_candidate = 1;
        if (should_restore) {
            config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
            config.profile_input = path + ".cache";
        } else {
            config.profile_output = path + ".cache";
        }
        dispatch.device_uuid = [](CUuuid* uuid, CUdevice) {
            *uuid = {};
            uuid->bytes[0] = 1;
            return CUDA_SUCCESS;
        };
        dispatch.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
    }
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual selected-variant service without a priority table");
    expect(!service.save_profile(), "profile publication requires proven complete shutdown");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated variant service identity");
    std::atomic<bool> has_closed{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "application supplies only its original source");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "original lookup");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        for (unsigned int command = 0; command < (should_profile ? 7U : 2U); ++command) {
            ExecutionPayloadWriter launch_input;
            expect(launch_input.write_u64(function.id) && launch_input.write_u64(0),
                   "owned launch");
            for (const unsigned int dimension : {8U, 1U, 1U, 32U, 1U, 1U})
                expect(launch_input.write_u32(dimension), "original geometry");
            expect(launch_input.write_u32(0) && launch_input.write_blob({}), "original empty ABI");
            const auto reply =
                success(connection.exchange(ExecutionOpcode::kLaunch, launch_input.take()));
            expect(reply.id == command + 1, "one logical reply per application launch");
        }
        success(connection.exchange(ExecutionOpcode::kClose));
        has_closed = true;
    });
    unsigned int waits{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded service scheduling and automatic continuation");
        if (native.launches && !native.is_ready) {
            expect(native.launches == 1 && !has_closed && native.event_destroys == 0,
                   "host reply cannot advance a slice or its same-stream successor before stop");
            if (++waits == 20) native.is_ready = true;
        }
    }
    client.get();
    while (!native.destroys)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "selected variant resources retire after final logical completion");
    if (should_profile) {
        const auto snapshot = service.profile_snapshot();
        if (should_restore) {
            expect(
                snapshot.loaded_cases == 1 && !snapshot.is_learning && snapshot.samples == 0 &&
                    snapshot.hits == 7 && snapshot.cases == 1 && !native.profiled_kinds.empty() &&
                    std::ranges::all_of(
                        native.profiled_kinds,
                        [&](auto selected) { return selected == native.profiled_kinds.front(); }) &&
                    native.event_destroys == native.launches && native.host_frees == 1 &&
                    waits == 0 && service.request_stop() && service.save_profile(),
                "fresh ordinary IPC service consumes the saved configuration without profiling "
                "or replay");
            expect(::unlink((path + ".cache").c_str()) == 0,
                   "remove only the owned fixture archive");
            return;
        }
        const std::vector<unsigned int> first_candidates{5, 0, 1, 2, 3, 4, 4, 4};
        expect(native.profiled_kinds.size() > first_candidates.size() &&
                   std::equal(first_candidates.begin(), first_candidates.end(),
                              native.profiled_kinds.begin()) &&
                   snapshot.samples == 6 && snapshot.trained_cases == 1 && snapshot.hits == 1 &&
                   snapshot.rejected_samples == 0 && snapshot.cancelled == 0 &&
                   native.event_destroys == native.launches && native.host_frees == 1 &&
                   waits == 20 && service.request_stop() && service.save_profile() &&
                   !service.save_profile(),
               "actual authenticated IPC trains every native family once and issues a learned "
               "candidate while retaining physical events and automatic slices");
        return;
    }
    const auto expected_launches = kind == Kind::kSliced ? 6U : 2U;
    expect(waits == 20 && native.launches == expected_launches &&
               native.event_destroys == expected_launches && native.host_frees == 1 &&
               native.initializations ==
                   (kind == Kind::kDynamic || kind == Kind::kPreemptive ? 2U : 0U) &&
               service.request_stop(),
           "unranked and naive modes execute every logical command and clean up once");
    if (kind == Kind::kSliced)
        expect(native.slices ==
                   std::vector<std::array<std::uint32_t, 2>>{
                       {0, 3}, {3, 3}, {6, 2}, {0, 3}, {3, 3}, {6, 2}},
               "actual IPC service automatically resumes both disjoint slice sequences");
    if (depth) {
        depth->finish();
        expect(depth->valid(), "configured preemptive compute without pause/resume is observed");
        std::uint64_t admitted{}, submitted{}, retired{};
        for (std::size_t index = 0; index < IssueDepthMetrics::bucket_capacity; ++index) {
            admitted += depth->bucket(index).admitted;
            submitted += depth->bucket(index).submitted;
            retired += depth->bucket(index).retired;
        }
        expect(admitted == 2 && submitted == 2 && retired == 2,
               "persistent commands retain one admission, submission and actual retirement each");
        completed_issue_events(*depth, records_events ? admitted : 0);
    }
}
void immutable_module_files(const std::string& path, bool is_binary) {
    native = {};
    if (is_binary)
        native.expected_binary = glimmer::test::cuda_test_cubin({}, std::size_t{17} * 1024 * 1024);
    else {
        native.expected_module =
            ".version 7.0\n.target sm_70\n.address_size 64\n"
            ".visible .entry kernel() { ret; }\n//";
        native.expected_module.resize(std::size_t{17} * 1024 * 1024, 'x');
    }
    const std::string_view contents =
        is_binary ? std::string_view(reinterpret_cast<const char*>(native.expected_binary.data()),
                                     native.expected_binary.size())
                  : std::string_view(native.expected_module);
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.session_limits.max_image_bytes = contents.size() + 1024;
    config.exit_when_idle = true;
    auto dispatch = kDispatch;
    dispatch.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice) {
        expect(attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ||
                   attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
               "large binary uses original compute-capability metadata only");
        *output = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 7 : 0;
        return CUDA_SUCCESS;
    };
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start real service for immutable module capabilities");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "module endpoint identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 5000});
        success(connection.open());
        const auto make_file = [](std::string_view contents, int seals) {
            ExecutionOwnedDescriptor file(
                ::memfd_create("glimmer-module-fixture", MFD_CLOEXEC | MFD_ALLOW_SEALING));
            expect(file.get() >= 0 &&
                       ::ftruncate(file.get(), static_cast<off_t>(contents.size())) == 0 &&
                       ::pwrite(file.get(), contents.data(), contents.size(), 0) ==
                           static_cast<ssize_t>(contents.size()),
                   "create exact module backing");
            if (seals) expect(::fcntl(file.get(), F_ADD_SEALS, seals) == 0, "seal fixture backing");
            return file;
        };
        const auto request = [&](int descriptor, std::uint64_t bytes, std::uint32_t kind,
                                 bool has_extra = false) {
            ExecutionPayloadWriter input;
            expect(input.write_u32(kind) && input.write_u64(bytes) &&
                       (!has_extra || input.write_u32(0)),
                   "encode module capability without process pointers");
            return connection.exchange(ExecutionOpcode::kLoadModuleFile, input.take(), descriptor);
        };
        constexpr auto required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK;
        const auto image_kind = static_cast<std::uint32_t>(
            is_binary ? ExecutionModuleImageKind::kBinary : ExecutionModuleImageKind::kPtx);
        const auto reject = [](const std::optional<ExecutionWireReply>& reply) {
            expect(reply && reply->status == CUDA_ERROR_INVALID_VALUE && !reply->original_status,
                   "malformed module capability rejects before any original SDK call");
        };
        auto backing = make_file(contents, required_seals | F_SEAL_SEAL);
        const auto bytes = contents.size();
        reject(request(backing.get(), bytes - 1, image_kind));
        reject(request(backing.get(), bytes + 1, image_kind));
        reject(request(backing.get(), 0, image_kind));
        reject(request(backing.get(), glimmer::transform::kMaxCudaModuleBytes + 1, image_kind));
        reject(request(backing.get(), bytes, 0));
        reject(request(backing.get(), bytes, image_kind, true));
        for (const auto seals : {0, F_SEAL_GROW | F_SEAL_SHRINK, F_SEAL_WRITE | F_SEAL_GROW,
                                 F_SEAL_FUTURE_WRITE | F_SEAL_GROW | F_SEAL_SHRINK}) {
            auto mutable_file = make_file(".version 7.0\n", seals);
            reject(request(mutable_file.get(), 13, image_kind));
        }
        ExecutionOwnedDescriptor nonregular(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(nonregular.get() >= 0, "open controlled nonregular descriptor");
        reject(request(nonregular.get(), 13, image_kind));
        const auto module = success(request(backing.get(), bytes, image_kind));
        const auto over_budget = request(backing.get(), bytes, image_kind);
        expect(over_budget && over_budget->status == CUDA_ERROR_OUT_OF_MEMORY &&
                   !over_budget->original_status,
               "descriptor transport cannot bypass the session's charged image budget");
        backing = ExecutionOwnedDescriptor{};
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"),
               "lookup retained code");
        success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        ExecutionPayloadWriter unload;
        expect(unload.write_u64(module.id), "unload exact owned module");
        success(connection.exchange(ExecutionOpcode::kUnloadModule, unload.take()));
        // A successful unload must actually refund the image charge.
        auto second = make_file(contents, required_seals);
        success(request(second.get(), bytes, image_kind));
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1), "progress bounded module-file service");
        expect(std::chrono::steady_clock::now() < deadline, "bounded module-file control");
    }
    client.get();
    expect(native.module_loads == 2 && native.module_unloads == 1 && native.destroys == 1 &&
               native.launches == 0 && service.request_stop(),
           "two original loads retire through one explicit unload and one context destruction");
}
void cold_context_purpose(const std::string& path) {
    struct Case {
        std::vector<std::byte> payload;
        CUresult expected{CUDA_SUCCESS};
        bool is_authorized{true};
    };
    std::vector<Case> cases{{{}}};
    ExecutionPayloadWriter legacy;
    expect(legacy.write_u32(CU_CTX_SCHED_YIELD), "encode original flags-only Hello");
    cases.push_back({legacy.take()});
    for (const auto purpose :
         {ExecutionContextPurpose::kExplicitUser, ExecutionContextPurpose::kProcessPrimary,
          ExecutionContextPurpose::kRuntimePrimary}) {
        const auto encoded = encode_execution_context_bootstrap(
            {purpose, CU_CTX_SCHED_YIELD, std::numeric_limits<std::uint64_t>::max()});
        expect(encoded.has_value(), "encode public cold context purpose");
        cases.push_back({{encoded->begin(), encoded->end()}});
    }
    const auto valid = cases.back().payload;
    for (const auto offset :
         {std::size_t{0}, std::size_t{3}, std::size_t{4}, std::size_t{12}, std::size_t{15}}) {
        auto invalid = valid;
        invalid[offset] = std::byte{255};
        cases.push_back({std::move(invalid), CUDA_ERROR_INVALID_VALUE});
    }
    auto invalid = valid;
    std::fill(invalid.begin() + 16, invalid.end(), std::byte{});
    cases.push_back({std::move(invalid), CUDA_ERROR_INVALID_VALUE});
    for (const auto length : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{16},
                              std::size_t{23}, std::size_t{25}}) {
        invalid = valid;
        invalid.resize(length);
        cases.push_back({std::move(invalid), CUDA_ERROR_INVALID_VALUE});
    }
    // The eight-byte helper grammar must not treat CTX1 as a host role.
    ExecutionPayloadWriter confused;
    expect(confused.write_u32(kExecutionContextRole) && confused.write_u32(0),
           "encode deliberately confused helper role");
    cases.push_back({confused.take(), CUDA_ERROR_INVALID_VALUE});
    cases.push_back({valid, CUDA_ERROR_NOT_PERMITTED, false});
    for (const auto& test : cases) {
        native = {};
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.uses_original_kernels = true;
        auto metrics = std::make_shared<HostStageMetrics>(HostStageMetrics::system_clock, 64);
        config.host_stage_metrics = metrics;
        const auto identity = current_process_identity();
        expect(identity.has_value(), "authenticated purpose fixture identity");
        if (!test.is_authorized) {
            auto other = *identity;
            ++other.start_time_ticks;
            config.client_priorities.push_back({other, 1});
        }
        ExecutionService service(std::move(config), kDispatch);
        expect(service.start(), "start actual service cold-purpose intake");
        auto client = std::async(std::launch::async, [&] {
            ExecutionSocketClient connection({path, *identity, 3000});
            const auto reply = connection.open(test.payload);
            expect(reply && reply->status == test.expected,
                   "exact authenticated cold purpose success/rejection status");
            if (test.expected == CUDA_SUCCESS) {
                const auto descriptor = decode_execution_context_bootstrap(test.payload);
                const auto flags = success(connection.exchange(ExecutionOpcode::kGetContextFlags));
                expect(descriptor && flags.id == descriptor->flags,
                       "versioned and legacy flags reach their own original native factory");
                success(connection.exchange(ExecutionOpcode::kClose));
            } else
                expect(!reply->original_status &&
                           decode_execution_bootstrap_ownership(reply->data) ==
                               ExecutionBootstrapOwnership::kNone &&
                           !connection.open(test.payload),
                       "invalid/unauthorized purpose proves no resources without SDK/retry");
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(service.step(1), "progress actual authenticated purpose exchange");
            expect(std::chrono::steady_clock::now() < deadline, "bounded cold context intake");
        }
        client.get();
        const auto count = test.expected == CUDA_SUCCESS ? 1U : 0U;
        // The service and isolated Session both retain their existing cuInit
        // boundaries; neither may run at all for rejected cold intake.
        expect(native.driver_initializations == 2 * count && native.context_creates == count &&
                   native.destroys == count && !native.current && !native.launches &&
                   service.request_stop(),
               "purpose never grants sharing: successful isolated context retires exactly once; "
               "malformed/unauthorized input enters no SDK");
        expect(metrics->valid() && metrics->intake_count() == std::size_t{3} * count,
               "actual Hello, flags query and Close retain exact ordered intake timestamps; "
               "rejected bootstrap fabricates no retained request");
    }
}

// The helper owns a separate native thread. Its current context must not leak
// into the service owner's injected Driver view.
thread_local CUcontext helper_planning_context{};
// The Driver callback fixes the flags/device order and their convertible types.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI helper_planning_create(CUcontext* output, unsigned int flags, CUdevice device) {
    const auto result = context_create(output, flags, device);
    helper_planning_context = *output;
    return result;
}
void idle_helper_planning(const std::string& path) {
    using Reason = HostStageMetrics::SlotPlanReason;
    struct Case {
        bool enabled{};
        std::uint32_t priority{};
        bool disconnects{};
    };
    for (const auto test : {Case{}, Case{true}, Case{true, 1}, Case{true, 0, true}}) {
        native = {};
        const auto identity = current_process_identity();
        expect(identity.has_value(), "authenticated helper planning identity");
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.persistent.workers = 320;
        config.client_priorities.push_back({*identity, test.priority});
        config.should_skip_idle_helper_plans = test.enabled;
        auto metrics = std::make_shared<HostStageMetrics>();
        config.host_stage_metrics = metrics;
        auto dispatch = kDispatch;
        dispatch.host_register = [](void*, std::size_t, unsigned int) -> CUresult {
            throw std::runtime_error("idle helper fixture must never register memory");
        };
        dispatch.host_unregister = [](void*) -> CUresult {
            throw std::runtime_error("idle helper fixture must never unregister memory");
        };
        dispatch.execution.context_get_current = [](CUcontext* output) {
            *output = helper_planning_context;
            return CUDA_SUCCESS;
        };
        dispatch.context_create = helper_planning_create;
        dispatch.context_set_current = [](CUcontext context) {
            helper_planning_context = context;
            return context_set(context);
        };
        dispatch.context_destroy = [](CUcontext context) {
            helper_planning_context = nullptr;
            return context_destroy(context);
        };
        ExecutionService service(std::move(config), dispatch);
        expect(service.start(), "start helper planning service");
        std::atomic<bool> is_ready{}, may_close{};
        auto client = std::async(std::launch::async, [&] {
            const auto original = [](std::optional<ExecutionWireReply> reply, const char* label) {
                if (!reply || reply->status != CUDA_SUCCESS ||
                    reply->original_status != CUDA_SUCCESS ||
                    reply->disposition != ExecutionWireDisposition::kComplete)
                    throw std::runtime_error(std::string{label} + ": status=" +
                                             std::to_string(reply ? reply->status : -1));
            };
            ExecutionSocketClient helper({path, *identity, 3000});
            ExecutionPayloadWriter role;
            expect(role.write_u32(kExecutionHostOwnerRole) && role.write_u32(0),
                   "encode authenticated host helper");
            original(helper.open(role.take()), "helper bootstrap");
            ExecutionSocketClient compute({path, *identity, 3000});
            original(compute.open(), "compute bootstrap");
            is_ready = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!may_close.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            expect(may_close.load(), "bounded owner-side idle helper checks");
            // A real post-idle request must still acquire a fresh plan.
            original(compute.exchange(ExecutionOpcode::kGetContextFlags), "post-idle flags");
            original(compute.exchange(ExecutionOpcode::kClose), "compute close");
            if (!test.disconnects) {
                const auto closed = helper.exchange(ExecutionOpcode::kClose);
                expect(closed && closed->status == CUDA_SUCCESS && !closed->original_status &&
                           closed->disposition == ExecutionWireDisposition::kComplete,
                       "helper Close preserves its composite cleanup result");
            }
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{8};
        while (!is_ready.load() &&
               client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(service.step(1), "progress both authenticated context factories");
            expect(std::chrono::steady_clock::now() < deadline, "bounded helper bootstrap");
        }
        if (!is_ready.load()) client.get();
        expect(is_ready.load(), "both helper and compute bootstrap complete");
        const auto counts = [&] {
            std::array<std::uint64_t, 7> result{};
            for (std::size_t index = 0; index < HostStageMetrics::bucket_count; ++index) {
                const auto& bucket = metrics->bucket(index);
                result[0] +=
                    bucket.stages[static_cast<std::size_t>(HostStageMetrics::Stage::kPlanSlot)]
                        .calls;
                result[1] += bucket.queues[0].samples;
                result[2] += bucket.queues[1].samples;
                for (std::size_t reason = 0; reason < 4; ++reason)
                    result[3 + reason] += bucket.slot_plans[reason];
            }
            return result;
        };
        const auto before = counts();
        constexpr std::uint64_t steps = 32;
        for (std::uint64_t step = 0; step < steps; ++step)
            expect(service.step(0), "progress quiescent helper without blocking");
        const auto after = counts();
        const bool skips = test.enabled && test.priority == 0;
        expect(
            after[0] - before[0] == steps * (skips ? 1 : 2) && after[1] > before[1] &&
                after[2] > before[2],
            "only the unused helper-local plan disappears; every global plan retains both peers");
        expect(after[3 + static_cast<std::size_t>(Reason::kSkippedHelper)] -
                       before[3 + static_cast<std::size_t>(Reason::kSkippedHelper)] ==
                   (skips ? steps : 0),
               "exact helper skip count and default/nonzero-priority fallback");
        may_close = true;
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(service.step(1), "progress actual post-idle requests and close");
            expect(std::chrono::steady_clock::now() < deadline, "bounded helper close");
        }
        client.get();
        while (!service.request_stop()) {
            expect(service.step(1), "retire disconnected helper through its original owner");
            expect(std::chrono::steady_clock::now() < deadline, "bounded helper retirement");
        }
        const auto completed = counts();
        expect(
            metrics->valid() && native.context_creates == 2 && native.destroys == 2 &&
                !native.current && !native.launches &&
                (!skips || completed[3 + static_cast<std::size_t>(Reason::kLocalObligation)]),
            "real helper requests and retirement keep their plans and exactly-once native cleanup");
    }
}

struct PreparationCapture {
    std::ostringstream messages;
    std::shared_ptr<spdlog::logger> logger{spdlog::default_logger()};
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink{
        std::make_shared<spdlog::sinks::ostream_sink_mt>(messages)};
    PreparationCapture() {
        sink->set_pattern("%v");
        logger->sinks().push_back(sink);
    }
    ~PreparationCapture() {
        std::erase(logger->sinks(), sink);
    }
    PreparationCapture(const PreparationCapture&) = delete;
    PreparationCapture& operator=(const PreparationCapture&) = delete;
};
void disconnect_waits_for_gpu(const std::string& path, CUresult launch_status,
                              bool should_fail_query = false, bool is_memory = false,
                              bool observes_depth = false, bool records_events = false) {
    native = {};
    native.launch_status = launch_status;
    if (should_fail_query) native.query_failure = CUDA_ERROR_ILLEGAL_ADDRESS;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.session_limits.max_objects = 2;
    config.exit_when_idle = true;
    std::unique_ptr<PreparationCapture> capture;
    std::shared_ptr<IssueDepthMetrics> depth;
    if (observes_depth) {
        depth = std::make_shared<IssueDepthMetrics>(
            IssueDepthMetrics::system_clock,
            records_events ? IssueDepthMetrics::event_capacity_limit : 0);
        config.issue_depth_metrics = depth;
        config.should_audit_native = true;
        capture = std::make_unique<PreparationCapture>();
    }
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start injectable real service loop");
    poll_wait_fixture::captures_next = true;
    expect(service.step(1) && !poll_wait_fixture::captures_next &&
               poll_wait_fixture::observed_wait_ns == 1000000,
           "idle owner preserves the exact requested transport wait without native work");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted fixture server identity");
    std::atomic<bool> submits_valid_command{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        ExecutionPayloadWriter bootstrap;
        expect(bootstrap.write_u32(CU_CTX_SCHED_YIELD), "encode optional context flags");
        success(connection.open(bootstrap.take()));
        const auto flags = success(connection.exchange(ExecutionOpcode::kGetContextFlags));
        expect(flags.id == CU_CTX_SCHED_YIELD, "bootstrap reaches actual context creation");
        ExecutionPayloadWriter update;
        expect(update.write_u32(CU_CTX_SCHED_SPIN), "encode safe native flag mutation");
        success(connection.exchange(ExecutionOpcode::kSetContextFlags, update.take()));
        expect(
            success(connection.exchange(ExecutionOpcode::kGetContextFlags)).id == CU_CTX_SCHED_SPIN,
            "query observes the actual session's flag mutation");
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "encode PTX");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "encode lookup");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        const auto submit = [&](std::uint64_t id) {
            ExecutionPayloadWriter input;
            if (is_memory) {
                expect(input.write_u32(static_cast<std::uint32_t>(ExecutionMemoryKind::kCopy)) &&
                           input.write_u32(0) && input.write_u64(id) && input.write_u64(0) &&
                           input.write_u64(0) && input.write_u64(0) && input.write_u64(0) &&
                           input.write_u32(0) && input.write_u64(0),
                       "encode owned fixed-width device-copy command");
                return connection.exchange(ExecutionOpcode::kMemory, input.take());
            }
            expect(input.write_u64(id) && input.write_u64(0), "encode legacy stream handles");
            for (unsigned int dimension = 0; dimension < 6; ++dimension)
                expect(input.write_u32(1), "encode geometry");
            expect(input.write_u32(0) && input.write_blob({}), "encode zero-parameter ABI");
            return connection.exchange(ExecutionOpcode::kLaunch, input.take());
        };
        const auto invalid = submit(UINT64_MAX);
        expect(invalid && invalid->disposition == ExecutionWireDisposition::kRejected &&
                   !invalid->original_status,
               "invalid ID never claims original native execution");
        if (is_memory) {
            for (const unsigned int kind : {0U, 4U}) {
                ExecutionPayloadWriter malformed;
                expect(malformed.write_u32(kind) && malformed.write_u32(2),
                       "encode truncated/invalid owned memory metadata");
                const auto rejected =
                    connection.exchange(ExecutionOpcode::kMemory, malformed.take());
                expect(rejected && rejected->disposition == ExecutionWireDisposition::kRejected &&
                           !rejected->original_status,
                       "malformed memory commands claim neither acceptance nor native results");
            }
        }
        submits_valid_command.store(true, std::memory_order_release);
        const auto actual = submit(is_memory ? 0 : function.id);
        expect(actual && actual->disposition == ExecutionWireDisposition::kComplete &&
                   actual->status == launch_status && actual->original_status == launch_status &&
                   actual->id == 1,
               "actual SDK error is preserved, not retried or consumed by reject");
        if (!should_fail_query) {
            const auto capture = [&] {
                ExecutionPayloadWriter input;
                expect(input.write_u32(0) && input.write_u64(0),
                       "encode exact context frontier scope");
                return connection.exchange(ExecutionOpcode::kCaptureFrontier, input.take());
            };
            const auto by_id = [&](ExecutionOpcode operation, std::uint64_t id) {
                ExecutionPayloadWriter input;
                expect(input.write_u64(id), "encode nonrecycled session-bound capability");
                return connection.exchange(operation, input.take());
            };
            const auto first = capture(), second = capture(), full = capture();
            expect(first && second && first->status == 0 && second->status == 0 &&
                       first->id != second->id && !first->original_status && full &&
                       full->status == CUDA_ERROR_OUT_OF_MEMORY && !full->original_status,
                   "bounded wire frontiers do not invent SDK results");
            const auto pending = by_id(ExecutionOpcode::kQueryFrontier, first->id);
            expect(pending && pending->status == CUDA_ERROR_NOT_READY && !pending->original_status,
                   "real pending native event cannot be acknowledged as physical completion");
            const auto released = by_id(ExecutionOpcode::kReleaseFrontier, first->id);
            const auto stale = by_id(ExecutionOpcode::kQueryFrontier, first->id);
            expect(released && released->status == 0 && !released->original_status && stale &&
                       stale->status == CUDA_ERROR_INVALID_HANDLE && !stale->original_status,
                   "explicit release invalidates only frontier metadata");
            const auto next = capture();
            expect(next && next->status == 0 && next->id > second->id,
                   "a released capability ID is never reused");
            const auto release_second = by_id(ExecutionOpcode::kReleaseFrontier, second->id);
            const auto release_next = by_id(ExecutionOpcode::kReleaseFrontier, next->id);
            expect(release_second && release_next && release_second->status == 0 &&
                       release_next->status == 0,
                   "release all temporary CPU frontier capabilities");
        }
        if (should_fail_query) {
            const auto drained = connection.exchange(ExecutionOpcode::kDrain);
            expect(drained && drained->status == CUDA_ERROR_ILLEGAL_ADDRESS &&
                       !drained->original_status,
                   "preserve real async failure without inventing an original sync call");
        }
        // Destroy the peer with a real launch still physically outstanding.
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    bool deferred_ready_command{};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1), "owner progresses real transport and native dispatcher");
        if (submits_valid_command.load(std::memory_order_acquire) && !native.launches &&
            service.pending_priority_requests(0))
            deferred_ready_command = true;
        expect(std::chrono::steady_clock::now() < deadline, "bounded fixture exchange");
    }
    client.get();
    for (unsigned int iteration = 0; iteration < 20; ++iteration)
        expect(service.step(0), "observe disconnect while GPU is still running");
    expect(native.launches == 1 && native.destroys == 0 && native.event_destroys == 0 &&
               native.synchronizes == 0 && !service.finished(),
           "host reply, SDK error and peer death are not physical completion");
    bool has_active_progress_wait = true;
    poll_wait_fixture::captures_next = true;
    expect(service.step(should_fail_query ? 1 : 2000),
           "owner progresses actual held trailing native event or contained failure");
    has_active_progress_wait =
        !poll_wait_fixture::captures_next &&
        poll_wait_fixture::observed_wait_ns == (should_fail_query ? 1000000 : 0);
    // Uncertain-state retention is deliberately isolated from ordinary leak checks.
    if (should_fail_query) {
        expect(has_active_progress_wait, "quarantine cannot become active progress permission");
        _exit(EXIT_SUCCESS);
    }
    native.is_ready = true;
    while (!service.finished()) {
        expect(service.step(0), "retire only after actual trailing completion proof");
        expect(std::chrono::steady_clock::now() < deadline, "bounded physical cleanup");
    }
    poll_wait_fixture::captures_next = true;
    expect(service.step(1) && !poll_wait_fixture::captures_next &&
               poll_wait_fixture::observed_wait_ns == 1000000,
           "physical retirement restores idle transport wait without a busy loop");
    expect(native.launches == 1 && native.event_destroys == 1 && native.destroys == 1 &&
               native.synchronizes == 0 && service.request_stop(),
           "exactly-once native execution and resource teardown after disconnect");
    expect(has_active_progress_wait,
           "active GPU ownership must not sleep waiting for unrelated socket traffic");
    expect(!deferred_ready_command,
           "ready ordinary command reaches actual SDK issue in its accepting owner turn");
    if (depth) {
        depth->finish();
        expect(depth->valid() == (is_memory || launch_status == CUDA_SUCCESS),
               "diagnostic errors never change native errors or completion ownership");
        std::uint64_t admitted{}, submitted{}, retired{};
        for (std::size_t index = 0; index < IssueDepthMetrics::bucket_capacity; ++index) {
            admitted += depth->bucket(index).admitted;
            submitted += depth->bucket(index).submitted;
            retired += depth->bucket(index).retired;
        }
        expect(admitted == (is_memory ? 0U : 1U) &&
                   (!depth->valid() || (submitted == admitted && retired == admitted)),
               "only compute contributes and the actual passive retirement closes it exactly once");
        completed_issue_events(*depth, records_events && depth->valid() ? admitted : 0);
        expect(service.audit_healthy() &&
                   capture->messages.str().find("issued=1, retained=0, retired=1") !=
                       std::string::npos,
               "depth and existing native audit observe the same actual retirement");
    }
}

void prepared_dependent_launch_waits_without_disconnect(const std::string& path) {
    native = {};
    native.is_prepared = true;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = 320;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start service-owned persistent preparation");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted service identity");
    std::atomic<bool> second_requested{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "unchanged original PTX");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"),
               "original function name");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        expect(function.parameter_bytes == 0 && function.arguments.empty(),
               "private ABI is never exposed");
        const auto submit = [&] {
            ExecutionPayloadWriter input;
            expect(input.write_u64(function.id) && input.write_u64(0), "logical stream");
            for (unsigned int axis = 0; axis < 6; ++axis)
                expect(input.write_u32(1), "original geometry");
            expect(input.write_u32(0) && input.write_blob({}), "original empty parameters");
            return success(connection.exchange(ExecutionOpcode::kLaunch, input.take()));
        };
        expect(submit().id == 1, "initial real SDK result arrives before physical completion");
        second_requested.store(true, std::memory_order_release);
        expect(submit().id == 2,
               "dependent accepted launch keeps its connection until actual SDK issue");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    unsigned int pending_steps{};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1), "advance prepared real service transport");
        if (second_requested.load(std::memory_order_acquire) && !native.is_ready) {
            expect(native.launches == 1 && native.destroys == 0,
                   "logical dependency must not early-issue or drop the peer");
            if (++pending_steps == 20) native.is_ready = true;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded dependent service fixture");
    }
    client.get();
    expect(pending_steps == 20,
           "second request must remain pending across real owner-loop progress");
    while (!service.finished()) {
        expect(service.step(1), "retire only actual prepared physical events");
        expect(std::chrono::steady_clock::now() < deadline, "bounded prepared service cleanup");
    }
    expect(
        native.launches == 2 && native.initializations == 2 && native.host_frees == 1 &&
            native.destroys == 1 && native.synchronizes == 0 && service.request_stop(),
        "two actual prepared launches drain with no fake reply, retry or leaked private resources");
}

enum class MultiplexScenario { kSuccess, kSdkError, kDisconnect };
void independent_submission_releases_dependent(const std::string& path,
                                               MultiplexScenario scenario) {
    const auto dependent_status =
        scenario == MultiplexScenario::kSdkError ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    const bool should_disconnect = scenario == MultiplexScenario::kDisconnect;
    native = {};
    native.is_multiplexed = true;
    native.is_prepared = true;
    native.dependent_status = dependent_status;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 4;
    config.session_limits.submissions.max_commands = 3;
    config.persistent.workers = 320;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start real multiplexed execution owner");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "multiplexed service trusted identity");
    std::atomic<bool> should_complete_dependent{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry producer() { ret; }\n"
                                   ".visible .entry helper() { bar.sync 15; ret; }"),
               "owned original zero-argument source");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        const auto lookup = [&](const char* name) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(module.id) && input.write_string(name), "original function");
            return success(connection.exchange(ExecutionOpcode::kGetFunction, input.take())).id;
        };
        const auto producer_function = lookup("producer");
        const auto helper_function = lookup("helper");
        const auto create_stream = [&](unsigned int flags) {
            ExecutionPayloadWriter input;
            expect(input.write_u32(flags), "original stream flags");
            return success(connection.exchange(ExecutionOpcode::kCreateStream, input.take())).id;
        };
        const auto producer = create_stream(CU_STREAM_DEFAULT);
        const auto independent = create_stream(CU_STREAM_NON_BLOCKING);
        const auto kernel = [&](std::uint64_t stream, bool is_producer = false) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(is_producer ? producer_function : helper_function) &&
                       input.write_u64(stream),
                   "owned launch handles");
            for (unsigned int axis = 0; axis < 6; ++axis)
                expect(input.write_u32(1), "unchanged zero-argument geometry");
            expect(input.write_u32(0) && input.write_blob({}), "owned empty parameter ABI");
            return input.take();
        };
        expect(
            success(connection.exchange(ExecutionOpcode::kLaunch, kernel(producer, true))).id == 1,
            "producer has a real SDK result, not physical completion");
        const auto dependent = connection.submit(ExecutionOpcode::kLaunch, kernel(0));
        ExecutionPayloadWriter capture;
        expect(capture.write_u32(0) && capture.write_u64(0), "capture exact received prefix");
        const auto captured = connection.submit(ExecutionOpcode::kCaptureFrontier, capture.take());
        expect(dependent.status == ExecutionSocketSubmitStatus::kAccepted &&
                   captured.status == ExecutionSocketSubmitStatus::kAccepted,
               "dependent and following capture are independently owned");
        std::optional<ExecutionWireReply> prefix;
        while (!(prefix = connection.take_reply(captured.sequence))) {
            expect(connection.poll_for(1), "capture progresses past pending dependent SDK issue");
            expect(!connection.take_reply(dependent.sequence),
                   "dependent receipt cannot invent early original SDK success");
        }
        expect(prefix->status == 0 && prefix->id && !prefix->original_status,
               "captured CPU membership does not claim original SDK execution");
        if (should_disconnect) return;  // The accepted dependent original has never been issued.
        const auto release = connection.submit(ExecutionOpcode::kLaunch, kernel(independent));
        expect(release.status == ExecutionSocketSubmitStatus::kAccepted,
               "later nonblocking stream enters the same authenticated session");
        bool has_release{}, has_dependent{};
        while (!has_release || !has_dependent) {
            expect(connection.poll_for(1), "independent original releases queued predecessor");
            if (const auto reply = connection.take_reply(release.sequence)) {
                expect(reply->status == 0 && reply->original_status == 0 && reply->id == 3,
                       "releasing original is issued once before the dependent original");
                has_release = true;
            }
            if (const auto reply = connection.take_reply(dependent.sequence)) {
                expect(
                    has_release && reply->status == dependent_status &&
                        reply->original_status == dependent_status && reply->id == 2,
                    "real deferred SDK success/error arrives exactly once, out of request order");
                has_dependent = true;
            }
        }
        const auto query = [&] {
            ExecutionPayloadWriter input;
            expect(input.write_u64(prefix->id), "exact immutable frontier capability");
            return connection.exchange(ExecutionOpcode::kQueryFrontier, input.take());
        };
        const auto pending = query();
        expect(pending && pending->status == CUDA_ERROR_NOT_READY && !pending->original_status,
               "captured prefix includes the previously received dependent launch");
        should_complete_dependent = true;
        for (;;) {
            const auto result = query();
            expect(result && !result->original_status, "physical prefix observation remains owned");
            if (result->status == CUDA_SUCCESS) break;
            expect(result->status == CUDA_ERROR_NOT_READY,
                   "only actual original events settle prefix");
        }
        ExecutionPayloadWriter input;
        expect(input.write_u64(prefix->id), "release CPU-only membership token");
        const auto released = connection.exchange(ExecutionOpcode::kReleaseFrontier, input.take());
        expect(released && released->status == 0 && !released->original_status,
               "token release cannot finish still-running independent work");
        // Disconnect while the later independent event is still physically pending.
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1), "advance actual authenticated multi-request service");
        if (should_complete_dependent) native.is_dependent_ready = true;
        expect(std::chrono::steady_clock::now() < deadline, "bounded dependency/release fixture");
    }
    client.get();
    for (unsigned int step = 0; step != 20; ++step)
        expect(service.step(0), "observe peer death without inventing stop");
    const auto expected_launches = should_disconnect ? 1U : 3U;
    const auto expected_streams =
        should_disconnect
            ? std::vector<CUstream>{reinterpret_cast<CUstream>(&native.streams[0])}
            : std::vector<CUstream>{reinterpret_cast<CUstream>(&native.streams[0]),
                                    reinterpret_cast<CUstream>(&native.streams[1]), nullptr};
    expect(native.launches == expected_launches && native.issued_streams == expected_streams &&
               native.event_destroys == (should_disconnect ? 0U : 2U) && native.destroys == 0 &&
               native.stream_count == 2 && native.host_frees == 0 && native.synchronizes == 0 &&
               !service.finished(),
           "host acknowledgments preserve the actual later event and its context after disconnect");
    native.is_ready = true;
    native.has_released = true;  // The fake device independently supplies its actual stop proof.
    while (!service.finished()) {
        expect(service.step(0), "settle final independent native event before teardown");
        expect(std::chrono::steady_clock::now() < deadline,
               "bounded physical multi-request cleanup");
    }
    expect(native.launches == expected_launches && native.event_destroys == expected_launches &&
               native.destroys == 1 && native.context_reclaimed_streams == 2 &&
               native.stream_destroys == 0 && native.host_frees == 1 && native.counters.empty() &&
               native.synchronizes == 0 && service.request_stop(),
           "all originals and physical lifetimes retire once, without SDK replay or fake drain");
}
void owned_pointer_metadata(const std::string& path) {
    native = {};
    native.is_prepared = true;
    struct PointerState {
        std::array<char, 2> contexts{};
        std::atomic<unsigned int> creates{}, queries{};
    } pointers;
    static PointerState* active{};
    active = &pointers;
    auto dispatch = kDispatch;
    dispatch.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        const auto index = active->creates.fetch_add(1);
        expect(index < active->contexts.size() && !native.current, "two detached real owners");
        *output = native.current = reinterpret_cast<CUcontext>(&active->contexts[index]);
        return CUDA_SUCCESS;
    };
    dispatch.context_destroy = [](CUcontext context) {
        expect(context && context == native.current, "destroy only activated pointer-query owner");
        native.current = nullptr;
        ++native.destroys;
        return CUDA_SUCCESS;
    };
    dispatch.module_global = [](CUdeviceptr* output, std::size_t* bytes, CUmodule, const char*) {
        *output = 0x2000;
        *bytes = 16;
        return CUDA_SUCCESS;
    };
    dispatch.runtime.version = [](int* output) {
        *output = 12060;
        return cudaSuccess;
    };
    dispatch.pointer_get_attribute = [](void* output, CUpointer_attribute selector,
                                        CUdeviceptr value) {
        ++active->queries;
        const bool is_known =
            (value >= 0x1000 && value < 0x1020) || (value >= 0x2000 && value < 0x2010);
        if (!is_known) return CUDA_ERROR_INVALID_VALUE;
        if (selector == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL) {
            *static_cast<int*>(output) = value == 0x1007 ? -17 : 0;
            return value == 0x1007 ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        }
        expect(selector == CU_POINTER_ATTRIBUTE_DEVICE_POINTER, "only actual pinned selectors");
        const bool is_accessible =
            native.current &&
            (value < 0x2000 || native.current == reinterpret_cast<CUcontext>(&active->contexts[0]));
        *static_cast<CUdeviceptr*>(output) = is_accessible ? value : 0;
        return !native.current ? CUDA_ERROR_INVALID_CONTEXT
               : is_accessible ? CUDA_SUCCESS
                               : CUDA_ERROR_INVALID_VALUE;
    };
    dispatch.runtime.pointer_attributes = [](cudaPointerAttributes* output, const void* value) {
        ++active->queries;
        const auto address = reinterpret_cast<std::uintptr_t>(value);
        if (address == 0x1009) {
            const int raw_type = -91;
            std::memcpy(&output->type, &raw_type, sizeof(raw_type));
            output->device = -17;
            return cudaErrorInvalidValue;
        }
        const bool is_known =
            (address >= 0x1000 && address < 0x1020) || (address >= 0x2000 && address < 0x2010);
        const bool is_accessible =
            native.current && (address < 0x2000 ||
                               native.current == reinterpret_cast<CUcontext>(&active->contexts[0]));
        *output = {is_known ? cudaMemoryTypeDevice : cudaMemoryTypeUnregistered, is_known ? 0 : -2,
                   is_known && is_accessible ? const_cast<void*>(value) : nullptr,
                   is_known ? nullptr : const_cast<void*>(value)};
        return cudaSuccess;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    config.uses_original_kernels = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual pointer metadata service");
    const auto peer = current_process_identity();
    expect(peer.has_value(), "pointer metadata authenticated server");
    std::atomic<bool> allow_stop{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient carrier({path, *peer, 5000});
        ExecutionPayloadWriter hello;
        expect(hello.write_u32(kExecutionPointerMetadataRole) && hello.write_u32(0), "PMD1 role");
        const auto opened = carrier.open(hello.take());
        expect(opened && !opened->status && !opened->original_status &&
                   decode_execution_bootstrap_ownership(opened->data) ==
                       ExecutionBootstrapOwnership::kMetadataRetained &&
                   pointers.creates == 0,
               "metadata bootstrap creates no application context or fake original success");
        const auto exchange = [&](const ExecutionPointerQuery& query) {
            ExecutionPayloadWriter input;
            expect(write_execution_pointer_query(input, query), "canonical pointer request");
            return carrier.exchange(ExecutionOpcode::kPointerMetadata, input.take());
        };
        const auto observation = [&](const ExecutionPointerQuery& query, int original) {
            const auto reply = exchange(query);
            expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                       reply->status == original && reply->original_status == original &&
                       !reply->id && !reply->address && !reply->parameter_bytes &&
                       reply->arguments.empty() && reply->data.size() == 20,
                   "actual native status and independent effects survive real service framing");
            ExecutionPayloadReader input(reply->data);
            ExecutionPointerObservation result;
            expect(read_execution_pointer_observation(input, result) && input.empty(),
                   "canonical complete observation");
            return result;
        };
        const auto rejected = [&](const ExecutionPointerQuery& query) {
            const auto calls = pointers.queries.load();
            const auto reply = exchange(query);
            expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                       reply->status && !reply->original_status && reply->data.empty() &&
                       pointers.queries == calls,
                   "untrusted source/query identity rejects before requested native callback");
        };
        ExecutionPointerQuery unknown{ExecutionPointerDomain::kRuntime, 0,
                                      ExecutionPointerKind::kUnregistered};
        expect(observation(unknown, 0) ==
                       ExecutionPointerObservation{15, 0, -2, ExecutionPointerAlias::kNull,
                                                   ExecutionPointerAlias::kInput} &&
                   pointers.creates == 0,
               "unregistered Runtime query neither reads caller bits nor creates context");
        ExecutionSocketClient source({path, *peer, 5000}), querying({path, *peer, 5000});
        success(source.open());
        success(querying.open());
        ExecutionPayloadWriter allocate_input;
        expect(allocate_input.write_u64(32), "owned positive device extent");
        const auto allocation =
            success(source.exchange(ExecutionOpcode::kAllocate, allocate_input.take()));
        ExecutionPointerQuery device{ExecutionPointerDomain::kDriver,
                                     3,
                                     ExecutionPointerKind::kDevice,
                                     source.session(),
                                     source.session(),
                                     allocation.id,
                                     3};
        expect(observation(device, 0).device_alias == ExecutionPointerAlias::kInput,
               "authenticated original owned interior device alias");
        device.query_session = 0;
        expect(observation(device, CUDA_ERROR_INVALID_CONTEXT) == ExecutionPointerObservation{4},
               "no-current native Driver error legitimately writes null");
        device.selector = 9;
        expect(observation(device, 0) == ExecutionPointerObservation{2},
               "allocation ordinal is independent of current context");
        device.offset = 7;
        expect(observation(device, CUDA_ERROR_UNKNOWN).device == -17,
               "native error signed scalar is not narrowed by the domainless codec");
        device.domain = ExecutionPointerDomain::kRuntime;
        device.selector = 0;
        device.offset = 9;
        expect(
            observation(device, cudaErrorInvalidValue) == ExecutionPointerObservation{3, -91, -17},
            "Runtime error publishes only actually changed raw signed fields");
        device.offset = 0;
        expect(observation(device, 0) == ExecutionPointerObservation{15, 2, 0},
               "Runtime no-current device observation keeps both aliases unavailable");
        auto invalid = device;
        invalid.query_session = carrier.session();
        rejected(invalid);
        invalid = device;
        invalid.source_session = carrier.session();
        rejected(invalid);
        invalid = device;
        invalid.kind = ExecutionPointerKind::kHost;
        rejected(invalid);
        invalid = device;
        invalid.offset = 32;
        rejected(invalid);
        ExecutionPayloadWriter module_input;
        expect(module_input.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                         ".visible .global .align 4 .u32 token[4];\n"
                                         ".visible .entry kernel() { ret; }"),
               "owned global source");
        const auto module =
            success(source.exchange(ExecutionOpcode::kLoadPtx, module_input.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("token"),
               "module global identity");
        const auto global = success(source.exchange(ExecutionOpcode::kGetGlobal, lookup.take()));
        auto global_query = device;
        global_query.query_session = querying.session();
        global_query.allocation = global.id;
        expect(observation(global_query, 0) == ExecutionPointerObservation{15, 2, 0},
               "other-context Runtime global remains device memory with unavailable aliases");
        global_query.domain = ExecutionPointerDomain::kDriver;
        global_query.selector = 3;
        expect(
            observation(global_query, CUDA_ERROR_INVALID_VALUE) == ExecutionPointerObservation{4},
            "cross-context native Driver error write-null is retained");
        ExecutionPayloadWriter kernel_lookup;
        expect(kernel_lookup.write_u64(module.id) && kernel_lookup.write_string("kernel"),
               "kernel name");
        const auto function =
            success(source.exchange(ExecutionOpcode::kGetFunction, kernel_lookup.take()));
        ExecutionPayloadWriter launch_input;
        expect(launch_input.write_u64(function.id) && launch_input.write_u64(0),
               "native launch identity");
        for (unsigned int dimension = 0; dimension < 6; ++dimension)
            expect(launch_input.write_u32(1), "original logical geometry");
        expect(launch_input.write_u32(0) && launch_input.write_blob({}), "no kernel parameters");
        success(source.exchange(ExecutionOpcode::kLaunch, launch_input.take()));
        device.query_session = source.session();
        expect(observation(device, 0).device_alias == ExecutionPointerAlias::kInput,
               "metadata completes while the native kernel event is still pending");
        allow_stop.store(true, std::memory_order_release);
        ExecutionPayloadWriter free_input;
        expect(free_input.write_u64(allocation.id), "exact allocation free");
        success(source.exchange(ExecutionOpcode::kRelease, free_input.take()));
        rejected(device);
        ExecutionPayloadWriter unload;
        expect(unload.write_u64(module.id), "owned global parent unload");
        success(source.exchange(ExecutionOpcode::kUnloadModule, unload.take()));
        rejected(global_query);
        success(source.exchange(ExecutionOpcode::kClose));
        success(querying.exchange(ExecutionOpcode::kClose));
        const auto closed = carrier.exchange(ExecutionOpcode::kClose);
        expect(closed && closed->status == 0 && !closed->original_status,
               "CPU metadata close fabricates no native callback status");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        if (allow_stop.load(std::memory_order_acquire)) native.is_ready = true;
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded actual pointer metadata service progress");
    }
    client.get();
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "pointer service cleanup");
    expect(pointers.creates == 2 && native.destroys == 2 && native.launches == 1 &&
               native.event_destroys == 1 && native.synchronizes == 0 && service.request_stop(),
           "two true owners clean up; metadata never drains or creates a hidden context");
    active = nullptr;
}
void owned_function_controls(const std::string& path) {
    native = {};
    struct Controls {
        int value{-17};
        unsigned int queries{}, attributes{}, caches{};
    } controls;
    // The injectable ABI uses one owner thread. Client observations are wire
    // replies; inspect counters only after joining that client.
    static Controls* active{};
    active = &controls;
    auto dispatch = kDispatch;
    dispatch.function_get_attribute = [](int* output, CUfunction_attribute attribute,
                                         CUfunction function) {
        expect(function == reinterpret_cast<CUfunction>(&native), "query actual original function");
        ++active->queries;
        if (attribute == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES ||
            attribute == CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES ||
            attribute == CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
            *output = 17 + static_cast<int>(attribute);
        else
            *output = attribute == static_cast<CUfunction_attribute>(31) ? 999 : active->value;
        return attribute == static_cast<CUfunction_attribute>(31) ? CUDA_ERROR_INVALID_VALUE
                                                                  : CUDA_SUCCESS;
    };
    dispatch.function_set_attribute = [](CUfunction function, CUfunction_attribute attribute,
                                         int value) {
        expect(function == reinterpret_cast<CUfunction>(&native), "set actual original function");
        ++active->attributes;
        if (attribute == static_cast<CUfunction_attribute>(31)) {
            expect(value == 123, "representable unknown attribute preserves signed SDK input");
            return CUDA_ERROR_INVALID_VALUE;
        }
        expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
               "typed function attribute reaches original ABI");
        active->value = value;
        return CUDA_SUCCESS;
    };
    dispatch.function_set_cache = [](CUfunction function, CUfunc_cache cache) {
        expect(
            function == reinterpret_cast<CUfunction>(&native) && cache == CU_FUNC_CACHE_PREFER_L1,
            "owned cache setter receives original function and exact enum");
        ++active->caches;
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual function-control service");
    const auto peer = current_process_identity();
    expect(peer.has_value(), "function-control authenticated server identity");
    std::atomic<bool> allow_stop{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *peer, 5000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "owned function source");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "function identity");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        const auto encode = [&](std::uint32_t operation, std::uint64_t id, std::uint32_t argument,
                                std::int32_t value) {
            ExecutionPayloadWriter input;
            expect(input.write_u32(operation) && input.write_u64(id) && input.write_u32(argument) &&
                       input.write_u32(std::bit_cast<std::uint32_t>(value)),
                   "canonical20-byte function-control payload");
            return input.take();
        };
        const auto exchange = [&](std::uint32_t operation, std::uint32_t argument = 0,
                                  std::int32_t value = 0) {
            return connection.exchange(ExecutionOpcode::kFunctionControl,
                                       encode(operation, function.id, argument, value));
        };
        const auto reject = [](const auto& reply) {
            expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                       reply->status != CUDA_SUCCESS && !reply->original_status && !reply->id &&
                       !reply->address && reply->data.empty(),
                   "malformed controls have no requested original SDK provenance");
        };
        const auto canonical = encode(1, function.id, 0, 0);
        for (std::size_t size = 0; size < canonical.size(); ++size)
            reject(connection.exchange(ExecutionOpcode::kFunctionControl,
                                       std::span{canonical}.first(size)));
        auto trailing = canonical;
        trailing.push_back(std::byte{});
        reject(connection.exchange(ExecutionOpcode::kFunctionControl, trailing));
        for (const auto operation : {0U, 5U, UINT32_MAX}) reject(exchange(operation));
        reject(exchange(4, 1));
        reject(exchange(4, 0, 1));
        reject(exchange(1, 0, 1));
        reject(exchange(3, 2, -1));
        reject(exchange(1, 32));
        reject(exchange(2, UINT32_MAX, -1));
        reject(exchange(3, 4));
        reject(connection.exchange(ExecutionOpcode::kFunctionControl, encode(1, UINT64_MAX, 0, 0)));
        const auto query = [&](int expected) {
            const auto reply = success(exchange(1));
            expect(reply.id == std::bit_cast<std::uint32_t>(expected) && !reply.address &&
                       !reply.parameter_bytes && reply.data.empty(),
                   "query publishes exactly original int bits, no process pointer or quarantine");
        };
        const auto aggregate = [&](int expected) {
            const auto reply = success(exchange(4));
            expect(!reply.id && !reply.address && !reply.parameter_bytes &&
                       reply.arguments.empty() && reply.data.size() == 76,
                   "aggregate publishes only a complete original documented snapshot");
            ExecutionPayloadReader input(reply.data);
            ExecutionFunctionAttributes attributes;
            expect(read_execution_function_attributes(input, attributes) && input.empty() &&
                       attributes.sizes == std::array<std::uint64_t, 3>{18, 19, 20},
                   "all three original size fields survive actual service transport");
            for (const auto value : attributes.values)
                expect(value == expected, "every signed original field preserves exact bits");
        };
        query(-17);
        aggregate(-17);
        for (const auto operation : {1U, 2U}) {
            const auto failed = exchange(operation, 31, operation == 2 ? 123 : 0);
            expect(failed && failed->disposition == ExecutionWireDisposition::kComplete &&
                       failed->status == CUDA_ERROR_INVALID_VALUE &&
                       failed->original_status == CUDA_ERROR_INVALID_VALUE && !failed->id &&
                       !failed->address && !failed->parameter_bytes && failed->data.empty(),
                   "original SDK errors survive transport without poisoned output or fake policy");
        }
        const auto changed =
            success(exchange(2, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, -1));
        expect(!changed.id && !changed.address && changed.data.empty(),
               "setter has no value output");
        query(-1);
        success(exchange(3, CU_FUNC_CACHE_PREFER_L1));
        ExecutionPayloadWriter launch_input;
        expect(launch_input.write_u64(function.id) && launch_input.write_u64(0),
               "owned kernel launch");
        for (unsigned int dimension = 0; dimension < 6; ++dimension)
            expect(launch_input.write_u32(1), "original geometry");
        expect(launch_input.write_u32(0) && launch_input.write_blob({}), "zero-parameter ABI");
        success(connection.exchange(ExecutionOpcode::kLaunch, launch_input.take()));
        query(-1);      // Read-only control must not wait for the pending event.
        aggregate(-1);  // Nor may an aggregate query acquire scheduling demand.
        const auto delayed = connection.submit(
            ExecutionOpcode::kFunctionControl,
            encode(2, function.id, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 17));
        expect(delayed.status == ExecutionSocketSubmitStatus::kAccepted, "submit mutation once");
        for (unsigned int attempt = 0; attempt < 20; ++attempt) {
            expect(connection.poll_for(1) && !connection.take_reply(delayed.sequence),
                   "mutation cannot complete while actual native work is outstanding");
        }
        allow_stop.store(true, std::memory_order_release);
        std::optional<ExecutionWireReply> completed;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!(completed = connection.take_reply(delayed.sequence)))
            expect(connection.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "deferred control completes without replay after physical stop");
        expect(!success(std::move(completed)).id, "single eventual mutation result");
        query(17);
        ExecutionPayloadWriter unload;
        expect(unload.write_u64(module.id), "unload actual parent module");
        success(connection.exchange(ExecutionOpcode::kUnloadModule, unload.take()));
        reject(exchange(1));
        reject(exchange(4));
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        if (allow_stop.load(std::memory_order_acquire)) native.is_ready = true;
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded owned function-control service progress");
    }
    client.get();
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "independent owned service cleanup");
    expect(controls.queries == 37 && controls.attributes == 3 && controls.caches == 1 &&
               controls.value == 17 && native.launches == 1 && native.event_destroys == 1 &&
               native.synchronizes == 0 && native.module_unloads == 1 && native.destroys == 1 &&
               service.request_stop(),
           "all real controls execute once, only after required physical stop, and clean up");
    active = nullptr;
}
}  // namespace
namespace {
struct LibraryNative {
    unsigned int creates{}, destroys{}, versions{}, extended{}, classic_batches{};
    char handle{};
    unsigned int lt_creates{}, lt_destroys{};
    std::array<char, 2> lt_handles{};
    CUdeviceptr workspace{};
    std::size_t workspace_bytes{};
    unsigned int workspace_sets{}, atomics_sets{}, sm_count_sets{};
    cublasAtomicsMode_t atomics{CUBLAS_ATOMICS_NOT_ALLOWED};
    int sm_count{};
    cublasPointerMode_t pointer_mode{CUBLAS_POINTER_MODE_HOST};
    unsigned int geams{}, gemvs{}, sgemm_ex_calls{};
} library_native;
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original GEAM ABI.
template <typename T>
cublasStatus_t service_geam(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                            int rows, int columns, const T* alpha, const T* a, int lda,
                            const T* beta, const T* b, int ldb, T* c, int ldc) {
    expect(native.current == reinterpret_cast<CUcontext>(&native) &&
               handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) &&
               ta == CUBLAS_OP_T && tb == CUBLAS_OP_T && columns == 3 && lda == 4 && ldb == 4 &&
               ldc == 3,
           "GEAM real IPC preserves the original ABI, signed shape, transpose and padded LDs");
    ++library_native.geams;
    if (rows < 0) {
        expect(!a && !b && !c, "invalid-shape unresolved pointers never cross IPC as addresses");
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!rows) {
        expect(!alpha && !beta && !a && !b && !c, "empty request preserves NULL presence");
        return CUBLAS_STATUS_SUCCESS;
    }
    expect(rows == 2 && reinterpret_cast<CUdeviceptr>(a) == 0x1000 && a == b,
           "ORT-style transpose retains exact shared input identity");
    if (a == c) return CUBLAS_STATUS_INVALID_VALUE;
    expect(reinterpret_cast<CUdeviceptr>(c) == 0x1100, "GEAM receives only owned output storage");
    if (library_native.pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        expect(alpha && beta && *alpha == T{1} && *beta == T{},
               "both host coefficient widths survive explicit service decoding");
    } else {
        expect(reinterpret_cast<CUdeviceptr>(alpha) == 0x1180 &&
                   reinterpret_cast<CUdeviceptr>(beta) == 0x1190,
               "fully owned DEVICE coefficients stay device pointers without a private copy");
    }
    return std::is_same_v<T, float> ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
void check_extension_scalars(cublasHandle_t handle, const float* alpha, const float* beta) {
    expect(native.current == reinterpret_cast<CUcontext>(&native) &&
               handle == reinterpret_cast<cublasHandle_t>(&library_native.handle),
           "BLAS extension issues only in its real owner context with the actual handle");
    if (library_native.pointer_mode == CUBLAS_POINTER_MODE_HOST)
        expect(alpha && beta && *alpha == 2.0F && *beta == 0.5F,
               "both exact extension APIs preserve four-byte host scalars");
    else
        expect(reinterpret_cast<CUdeviceptr>(alpha) == 0x1180 &&
                   reinterpret_cast<CUdeviceptr>(beta) == 0x1190,
               "extension device scalars retain their original owned addresses");
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original cuBLAS ABIs.
cublasStatus_t service_sgemv(cublasHandle_t handle, cublasOperation_t transpose, int rows,
                             int columns, const float* alpha, const float* a, int lda,
                             const float* x, int incx, const float* beta, float* y, int incy) {
    ++library_native.gemvs;
    if (rows <= 0) {
        expect(native.current == reinterpret_cast<CUcontext>(&native) &&
                   handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) &&
                   transpose == CUBLAS_OP_T && columns == 3 && lda == 4 && incx == -2 &&
                   incy == 3 && !alpha && !beta && !a && !x && !y,
               "metadata-only SGEMV retains original NULL arguments and actual owner");
        return CUBLAS_STATUS_SUCCESS;
    }
    check_extension_scalars(handle, alpha, beta);
    expect(transpose == CUBLAS_OP_T && rows == 2 && columns == 3 && lda == 4 && incx == -2 &&
               incy == 3 && reinterpret_cast<CUdeviceptr>(a) == 0x1000 &&
               reinterpret_cast<CUdeviceptr>(x) == 0x1080 &&
               reinterpret_cast<CUdeviceptr>(y) == 0x1100,
           "SGEMV IPC preserves exact signed increments, physical base and shape");
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t service_sgemm_ex(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                                int rows, int columns, int depth, const float* alpha, const void* a,
                                cudaDataType type_a, int lda, const void* b, cudaDataType type_b,
                                int ldb, const float* beta, void* c, cudaDataType type_c, int ldc) {
    ++library_native.sgemm_ex_calls;
    if (rows <= 0) {
        expect(native.current == reinterpret_cast<CUcontext>(&native) &&
                   handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) &&
                   ta == CUBLAS_OP_N && tb == CUBLAS_OP_C && columns == 3 && depth == 4 &&
                   lda == 4 && ldb == 4 && ldc == 3 && type_a == CUDA_R_16F &&
                   type_b == CUDA_R_16F && type_c == CUDA_R_32F && !alpha && !beta && !a && !b &&
                   !c,
               "metadata-only SgemmEx keeps signed shape, types and NULL presence");
        return rows < 0 ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
    }
    check_extension_scalars(handle, alpha, beta);
    if (!depth) {
        expect(ta == CUBLAS_OP_N && tb == CUBLAS_OP_C && rows == 2 && columns == 3 && lda == 4 &&
                   ldb == 4 && ldc == 3 && type_a == CUDA_R_16F && type_b == CUDA_R_16F &&
                   type_c == CUDA_R_32F && !a && !b && reinterpret_cast<CUdeviceptr>(c) == 0x1100,
               "zero-depth SgemmEx retains output and real scalar storage with NULL A/B");
        return CUBLAS_STATUS_SUCCESS;
    }
    expect(ta == CUBLAS_OP_N && tb == CUBLAS_OP_C && rows == 2 && columns == 3 && depth == 4 &&
               lda == 4 && ldb == 4 && ldc == 3 && type_a == CUDA_R_16F && type_b == CUDA_R_16F &&
               type_c == CUDA_R_32F && reinterpret_cast<CUdeviceptr>(a) == 0x1000 &&
               reinterpret_cast<CUdeviceptr>(b) == 0x1080 &&
               reinterpret_cast<CUdeviceptr>(c) == 0x1100,
           "SgemmEx IPC preserves its exact original ABI without invented compute/algo fields");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
void original_library_status_domains(const std::string& path) {
    native = {};
    native.is_prepared = true;
    library_native = {};
    std::atomic<bool> allow_extended_stop{};
    auto dispatch = kDispatch;
    dispatch.blas.sgeam = service_geam<float>;
    dispatch.blas.dgeam = service_geam<double>;
    dispatch.blas.sgemv = service_sgemv;
    dispatch.blas.sgemm_ex = service_sgemm_ex;
    dispatch.blas.get_stream = [](cublasHandle_t, cudaStream_t* output) {
        *output = nullptr;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_pointer_mode = [](cublasHandle_t, cublasPointerMode_t mode) {
        library_native.pointer_mode = mode;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.get_pointer_mode = [](cublasHandle_t, cublasPointerMode_t* output) {
        *output = library_native.pointer_mode;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.lt.create = [](cublasLtHandle_t* output) {
        expect(native.current == reinterpret_cast<CUcontext>(&native),
               "Lt factory runs only in its actual owned context");
        const auto index = library_native.lt_creates++;
        if (!index) return CUBLAS_STATUS_ALLOC_FAILED;
        expect(index <= library_native.lt_handles.size(), "bounded native Lt creation");
        *output = reinterpret_cast<cublasLtHandle_t>(&library_native.lt_handles[index - 1]);
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.lt.destroy = [](cublasLtHandle_t handle) {
        expect(native.current == reinterpret_cast<CUcontext>(&native) &&
                   (handle == reinterpret_cast<cublasLtHandle_t>(&library_native.lt_handles[0]) ||
                    handle == reinterpret_cast<cublasLtHandle_t>(&library_native.lt_handles[1])),
               "Lt cleanup uses only retained SDK handles before context teardown");
        ++library_native.lt_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.create = [](cublasHandle_t* handle) {
        expect(native.current == reinterpret_cast<CUcontext>(&native),
               "library create uses the actual context owner");
        if (++library_native.creates == 1) return CUBLAS_STATUS_ALLOC_FAILED;
        *handle = reinterpret_cast<cublasHandle_t>(&library_native.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.destroy = [](cublasHandle_t handle) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) && native.current,
               "native library cleanup precedes owned-context destruction");
        ++library_native.destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_stream = [](cublasHandle_t, cudaStream_t) {
        library_native.workspace = 0;
        library_native.workspace_bytes = 0;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_workspace = [](cublasHandle_t handle, void* pointer, std::size_t bytes) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) && native.current,
               "workspace configuration reaches only the owned original handle");
        ++library_native.workspace_sets;
        const auto address = reinterpret_cast<CUdeviceptr>(pointer);
        if (address % 256) return CUBLAS_STATUS_INVALID_VALUE;
        library_native.workspace = address;
        library_native.workspace_bytes = bytes;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_atomics_mode = [](cublasHandle_t, cublasAtomicsMode_t mode) {
        ++library_native.atomics_sets;
        library_native.atomics = mode;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.get_atomics_mode = [](cublasHandle_t, cublasAtomicsMode_t* output) {
        *output = library_native.atomics;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_sm_count_target = [](cublasHandle_t, int count) {
        ++library_native.sm_count_sets;
        if (count > 80) return CUBLAS_STATUS_INVALID_VALUE;
        library_native.sm_count = count;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.get_sm_count_target = [](cublasHandle_t, int* output) {
        *output = library_native.sm_count;
        return CUBLAS_STATUS_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original FP32 batch ABI.
    dispatch.blas.strided_sgemm = [](cublasHandle_t handle, cublasOperation_t ta,
                                     cublasOperation_t tb, int rows, int columns, int depth,
                                     const float* alpha, const float* a, int leading_a,
                                     long long stride_a, const float* b, int leading_b,
                                     long long stride_b, const float* beta, float* c, int leading_c,
                                     long long stride_c, int batches) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) &&
                   native.current && ta == CUBLAS_OP_N && tb == CUBLAS_OP_T && rows == 2 &&
                   columns == 2 && depth == 2 && leading_a == 2 && leading_b == 2 &&
                   leading_c == 2 && stride_a == 8 && stride_b == 8 && stride_c == 8 &&
                   batches == 3 && reinterpret_cast<CUdeviceptr>(a) == 0x1000 &&
                   reinterpret_cast<CUdeviceptr>(b) == 0x1080 &&
                   reinterpret_cast<CUdeviceptr>(c) == 0x1100 && *alpha == 2 && *beta == 0.5F,
               "classic FP32 batch uses exact original arguments and callback through real IPC");
        ++library_native.classic_batches;
        return CUBLAS_STATUS_NOT_SUPPORTED;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuBLAS ABI.
    dispatch.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                             const float*, const float*, int, const float*, int, const float*,
                             float*, int) { return CUBLAS_STATUS_SUCCESS; };
    dispatch.blas.gemm_ex = [](cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
                               int rows, int columns, int depth, const void* alpha, const void* a,
                               cudaDataType a_type, int leading_a, const void* b,
                               cudaDataType b_type, int leading_b, const void* beta, void* c,
                               cudaDataType c_type, int leading_c, cublasComputeType_t compute,
                               cublasGemmAlgo_t algorithm) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&library_native.handle) &&
                   native.current && ta == CUBLAS_OP_N && tb == CUBLAS_OP_T && rows == 2 &&
                   columns == 2 && depth == 2 && leading_a == 2 && leading_b == 2 &&
                   leading_c == 2 && a_type == CUDA_R_16F && b_type == CUDA_R_16F &&
                   c_type == CUDA_R_16F && compute == CUBLAS_COMPUTE_32F &&
                   algorithm == CUBLAS_GEMM_DEFAULT && reinterpret_cast<CUdeviceptr>(a) == 0x1000 &&
                   reinterpret_cast<CUdeviceptr>(b) == 0x1080 &&
                   reinterpret_cast<CUdeviceptr>(c) == 0x1100 &&
                   *static_cast<const float*>(alpha) == 2.0F &&
                   *static_cast<const float*>(beta) == 0.5F && !library_native.workspace &&
                   !library_native.workspace_bytes && !library_native.sm_count &&
                   library_native.atomics == CUBLAS_ATOMICS_ALLOWED,
               "real GemmEx service decoder preserves all native typed fields, owned addresses and "
               "raw scalars");
        return ++library_native.extended == 1 ? CUBLAS_STATUS_EXECUTION_FAILED
                                              : CUBLAS_STATUS_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    dispatch.blas.get_version = [](cublasHandle_t handle, int* output) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&library_native.handle),
               "query only the owned SDK handle");
        if (++library_native.versions == 1) return CUBLAS_STATUS_EXECUTION_FAILED;
        *output = 120604;
        return CUBLAS_STATUS_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual library protocol/native service");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "native service identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        success(connection.open());
        const auto lt_error = connection.exchange(ExecutionOpcode::kCreateLt);
        expect(lt_error && lt_error->disposition == ExecutionWireDisposition::kComplete &&
                   lt_error->status == CUBLAS_STATUS_ALLOC_FAILED &&
                   lt_error->original_status == CUBLAS_STATUS_ALLOC_FAILED && !lt_error->id &&
                   !lt_error->parameter_bytes,
               "Lt original known-empty failure crosses the real service without quarantine");
        const auto lt = success(connection.exchange(ExecutionOpcode::kCreateLt));
        expect(lt.id && !lt.parameter_bytes, "native Lt factory publishes only an owned identity");
        ExecutionPayloadWriter lt_destroy;
        expect(lt_destroy.write_u64(lt.id), "owned Lt destruction payload");
        const auto lt_bytes = lt_destroy.take();
        success(connection.exchange(ExecutionOpcode::kDestroyLt, lt_bytes));
        const auto stale_lt = connection.exchange(ExecutionOpcode::kDestroyLt, lt_bytes);
        expect(stale_lt && stale_lt->disposition == ExecutionWireDisposition::kRejected &&
                   !stale_lt->original_status && !stale_lt->parameter_bytes,
               "stale Lt handles cannot repeat SDK destruction");
        const auto implicit_lt = success(connection.exchange(ExecutionOpcode::kCreateLt));
        expect(implicit_lt.id > lt.id, "retired native addresses do not recycle public identities");
        const auto rejected = connection.exchange(ExecutionOpcode::kCreateBlas);
        expect(rejected && rejected->disposition == ExecutionWireDisposition::kComplete &&
                   rejected->status == CUBLAS_STATUS_ALLOC_FAILED &&
                   rejected->original_status == CUBLAS_STATUS_ALLOC_FAILED && !rejected->id,
               "original library create failure is neither a Driver error nor an admission ACK");
        const auto handle = success(connection.exchange(ExecutionOpcode::kCreateBlas));
        expect(handle.id, "actual native create publishes one owned identity");
        const auto query = [&] {
            ExecutionPayloadWriter input;
            expect(input.write_u64(handle.id) && input.write_u32(static_cast<std::uint32_t>(
                                                     ExecutionBlasAttribute::kVersion)),
                   "wire query contains only scoped identity and typed attribute");
            return connection.exchange(ExecutionOpcode::kGetBlasAttribute, input.take());
        };
        const auto failed = query();
        expect(
            failed && failed->disposition == ExecutionWireDisposition::kComplete &&
                failed->status == CUBLAS_STATUS_EXECUTION_FAILED &&
                failed->original_status == CUBLAS_STATUS_EXECUTION_FAILED,
            "real library error 13 crosses the actual native service without a Driver conversion");
        expect(success(query()).address == 120604,
               "a native metadata error permits a fresh successful query");
        ExecutionPayloadWriter allocation;
        expect(allocation.write_u64(512), "owned extended matrix backing request");
        const auto memory =
            success(connection.exchange(ExecutionOpcode::kAllocate, allocation.take()));
        const auto workspace = [&](std::uint64_t allocation_id, std::uint64_t offset,
                                   std::uint64_t bytes, bool trailing = false) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(handle.id) && input.write_u64(allocation_id) &&
                       input.write_u64(offset) && input.write_u64(bytes) &&
                       (!trailing || input.write_u32(0)),
                   "workspace payload carries an owned interval and exact original byte count");
            return connection.exchange(ExecutionOpcode::kSetBlasWorkspace, input.take());
        };
        success(workspace(memory.id, 0, 256));
        const auto alignment = workspace(memory.id, 1, 256);
        expect(alignment && alignment->disposition == ExecutionWireDisposition::kComplete &&
                   alignment->status == CUBLAS_STATUS_INVALID_VALUE &&
                   alignment->original_status == CUBLAS_STATUS_INVALID_VALUE,
               "original workspace alignment error keeps its actual library domain");
        for (const auto& invalid : {workspace(memory.id, 0, 256, true), workspace(UINT64_MAX, 0, 1),
                                    workspace(memory.id, 512, 1),
                                    workspace(memory.id, 0, UINT64_MAX), workspace(0, 0, 1)})
            expect(invalid && invalid->disposition == ExecutionWireDisposition::kRejected &&
                       !invalid->original_status,
                   "malformed, unowned and overflowing workspaces never reach the original SDK");
        success(workspace(0, 0, 0));
        success(workspace(memory.id, 256, 256));
        const auto attribute = [&](ExecutionBlasAttribute kind,
                                   std::optional<std::uint64_t> value) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(handle.id) &&
                       input.write_u32(static_cast<std::uint32_t>(kind)) &&
                       (!value || input.write_u64(*value)),
                   "handle policy uses typed values without native pointers");
            return connection.exchange(
                value ? ExecutionOpcode::kSetBlasAttribute : ExecutionOpcode::kGetBlasAttribute,
                input.take());
        };
        expect(
            success(attribute(ExecutionBlasAttribute::kAtomicsMode, std::nullopt)).address == 0 &&
                success(attribute(ExecutionBlasAttribute::kSmCountTarget, std::nullopt)).address ==
                    0,
            "original default atomics and SM hint remain independently queryable");
        success(attribute(ExecutionBlasAttribute::kAtomicsMode, 1));
        const auto bad_atomics = attribute(ExecutionBlasAttribute::kAtomicsMode, 2);
        expect(
            bad_atomics && !bad_atomics->original_status &&
                success(attribute(ExecutionBlasAttribute::kAtomicsMode, std::nullopt)).address == 1,
            "invalid raw atomics mode rejects without corrupting actual native policy");
        success(attribute(ExecutionBlasAttribute::kSmCountTarget, 7));
        const auto bad_sm = attribute(ExecutionBlasAttribute::kSmCountTarget, 81);
        const auto negative_sm = attribute(ExecutionBlasAttribute::kSmCountTarget, UINT64_MAX);
        expect(
            bad_sm && bad_sm->status == CUBLAS_STATUS_INVALID_VALUE &&
                bad_sm->original_status == CUBLAS_STATUS_INVALID_VALUE && negative_sm &&
                !negative_sm->original_status &&
                success(attribute(ExecutionBlasAttribute::kSmCountTarget, std::nullopt)).address ==
                    7,
            "native SM-hint rejection and preparation rejection have distinct provenance");
        success(attribute(ExecutionBlasAttribute::kSmCountTarget, 0));
        // Identical default stream still discards the explicit workspace.
        success(attribute(ExecutionBlasAttribute::kStream, 0));
        const auto extended = [&](bool malformed, bool classic = false, bool wrong_type = false) {
            ExecutionPayloadWriter input;
            if (classic) {
                for (int operand = 0; operand < 3; ++operand)
                    expect(input.write_u64(8), "original FP32 element stride");
                expect(input.write_u32(3), "original FP32 batch count");
            }
            const auto type = classic && !wrong_type ? CUDA_R_32F : CUDA_R_16F;
            for (const auto value :
                 {static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(type),
                  static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(CUBLAS_COMPUTE_32F),
                  UINT32_MAX})
                expect(input.write_u32(value), "typed extended header");
            expect(input.write_u64(handle.id) && input.write_u32(0) && input.write_u32(1),
                   "original extended handle and transpose");
            for (int field = 0; field < 6; ++field)
                expect(input.write_u32(2), "original extended shape");
            for (const auto offset : {0U, 128U, 256U})
                expect(input.write_u64(memory.id) && input.write_u64(offset),
                       "owned typed operand ranges");
            const float alpha = 2.0F, beta = 0.5F;
            const auto av = std::as_bytes(std::span{&alpha, 1});
            expect(input.write_u32(0) && input.write_blob(malformed ? av.first(2) : av) &&
                       input.write_blob(std::as_bytes(std::span{&beta, 1})),
                   "raw extended host coefficient snapshots");
            return connection.exchange(
                classic ? ExecutionOpcode::kStridedSgemm : ExecutionOpcode::kGemmEx, input.take());
        };
        const auto malformed = extended(true);
        expect(malformed && malformed->disposition == ExecutionWireDisposition::kRejected &&
                   !malformed->original_status,
               "wrong scalar width rejects without an original SDK result");
        const auto extended_error = extended(false);
        expect(extended_error &&
                   extended_error->disposition == ExecutionWireDisposition::kComplete &&
                   extended_error->status == CUBLAS_STATUS_EXECUTION_FAILED &&
                   extended_error->original_status == CUBLAS_STATUS_EXECUTION_FAILED,
               "actual original GemmEx error remains in the library domain across real "
               "authenticated IPC");
        expect(success(query()).address == 120604,
               "metadata progresses while the failed native library event remains pending");
        allow_extended_stop.store(true, std::memory_order_release);
        const auto drain = [&] {
            const auto reply = connection.exchange(ExecutionOpcode::kDrain);
            expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                       reply->status == CUDA_SUCCESS && !reply->original_status,
                   "logical drain requires actual prior event stops but invents no SDK call");
        };
        drain();
        success(extended(false));
        drain();
        const auto forged_classic = extended(false, true, true);
        expect(forged_classic &&
                   forged_classic->disposition == ExecutionWireDisposition::kRejected &&
                   !forged_classic->original_status,
               "forged classic non-FP32 metadata rejects before either native callback");
        const auto classic_error = extended(false, true);
        expect(classic_error && classic_error->disposition == ExecutionWireDisposition::kComplete &&
                   classic_error->status == CUBLAS_STATUS_NOT_SUPPORTED &&
                   classic_error->original_status == CUBLAS_STATUS_NOT_SUPPORTED,
               "classic batch returns its actual original SDK error, not a substituted Ex call");
        drain();
        std::uint64_t last_geam{};
        const auto geam_request = [&](ExecutionGeam problem, int expected, bool malformed = false) {
            ExecutionPayloadWriter input;
            expect(write_execution_geam(input, problem), "encode the canonical GEAM request");
            auto bytes = input.take();
            if (malformed) bytes.push_back(std::byte{});
            const auto reply = connection.exchange(ExecutionOpcode::kGeam, bytes);
            if (malformed) {
                expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                           reply->status && !reply->original_status && !reply->id &&
                           !reply->parameter_bytes && !reply->address && reply->arguments.empty() &&
                           reply->data.empty(),
                       "trailing GEAM bytes reject before admission without native provenance");
                return;
            }
            expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                       reply->status == expected && reply->original_status == expected &&
                       reply->id > last_geam && !reply->parameter_bytes && !reply->address &&
                       reply->arguments.empty() && reply->data.empty(),
                   "GEAM returns an admitted ID and exact original SDK status, including errors");
            last_geam = reply->id;
            drain();
        };
        for (const auto type : {ExecutionGeamType::kFloat, ExecutionGeamType::kDouble}) {
            ExecutionGeam problem;
            problem.type = type;
            problem.transpose_a = problem.transpose_b = CUBLAS_OP_T;
            problem.rows = 2;
            problem.columns = 3;
            problem.leading_a = problem.leading_b = 4;
            problem.leading_c = 3;
            problem.handle = handle.id;
            problem.presence = 31;
            problem.buffers[0] = problem.buffers[1] = {memory.id, 0};
            problem.buffers[2] = {memory.id, 256};
            const auto one = type == ExecutionGeamType::kFloat
                                 ? static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(1.0F))
                                 : std::bit_cast<std::uint64_t>(1.0);
            for (std::size_t index = 0; index < 8; ++index)
                problem.alpha[index] = static_cast<std::byte>((one >> (8 * index)) & 0xffU);
            const auto expected = type == ExecutionGeamType::kFloat ? CUBLAS_STATUS_SUCCESS
                                                                    : CUBLAS_STATUS_NOT_SUPPORTED;
            geam_request(problem, 0, true);
            geam_request(problem, expected);
            success(attribute(ExecutionBlasAttribute::kPointerMode, 1));
            problem.pointer_mode = 1;
            problem.alpha = {};
            problem.buffers[3] = {memory.id, 384};
            problem.buffers[4] = {memory.id, 400};
            geam_request(problem, expected);
            success(attribute(ExecutionBlasAttribute::kPointerMode, 0));
            problem.pointer_mode = 0;
            problem.buffers = {};
            problem.rows = -1;
            geam_request(problem, CUBLAS_STATUS_INVALID_VALUE);
            problem.rows = 0;
            problem.presence = 0;
            geam_request(problem, CUBLAS_STATUS_SUCCESS);
            problem.rows = 2;
            problem.presence = 28;
            problem.buffers[0] = problem.buffers[1] = problem.buffers[2] = {memory.id, 0};
            geam_request(problem, CUBLAS_STATUS_INVALID_VALUE);
        }
        for (const auto mode : {0U, 1U}) {
            success(attribute(ExecutionBlasAttribute::kPointerMode, mode));
            ExecutionSgemv gemv;
            gemv.transpose = CUBLAS_OP_T;
            gemv.rows = 2;
            gemv.columns = 3;
            gemv.leading_a = 4;
            gemv.increment_x = -2;
            gemv.increment_y = 3;
            ExecutionSgemmEx exact;
            exact.transpose_b = CUBLAS_OP_C;
            exact.type_a = exact.type_b = CUDA_R_16F;
            exact.type_c = CUDA_R_32F;
            exact.rows = 2;
            exact.columns = 3;
            exact.depth = 4;
            exact.leading_a = exact.leading_b = 4;
            exact.leading_c = 3;
            const auto initialize_problem = [&](auto& problem) {
                problem.handle = handle.id;
                problem.pointer_mode = mode;
                problem.presence = 31;
                for (std::size_t index = 0; index < 3; ++index)
                    problem.buffers[index] = {memory.id, index * 128};
                if (mode) {
                    problem.buffers[3] = {memory.id, 384};
                    problem.buffers[4] = {memory.id, 400};
                } else {
                    const float alpha = 2.0F, beta = 0.5F;
                    std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
                    std::memcpy(problem.beta.data(), &beta, sizeof(beta));
                }
            };
            initialize_problem(gemv);
            initialize_problem(exact);
            for (const auto operation : {ExecutionOpcode::kSgemv, ExecutionOpcode::kSgemmEx}) {
                ExecutionPayloadWriter input;
                expect(operation == ExecutionOpcode::kSgemv
                           ? write_execution_sgemv(input, gemv)
                           : write_execution_sgemm_ex(input, exact),
                       "encode distinct original-only extension request");
                const auto bytes = input.take();
                for (const bool trailing : {false, true}) {
                    auto invalid = bytes;
                    if (trailing)
                        invalid.push_back(std::byte{});
                    else
                        invalid.pop_back();
                    const auto rejected = connection.exchange(operation, invalid);
                    expect(rejected &&
                               rejected->disposition == ExecutionWireDisposition::kRejected &&
                               rejected->status == CUDA_ERROR_INVALID_VALUE && !rejected->id &&
                               !rejected->original_status && !rejected->parameter_bytes,
                           "truncated or trailing extension payload cannot enter the SDK");
                }
                const auto reply = connection.exchange(operation, bytes);
                const auto expected = operation == ExecutionOpcode::kSgemv
                                          ? CUBLAS_STATUS_SUCCESS
                                          : CUBLAS_STATUS_NOT_SUPPORTED;
                expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                           reply->status == expected && reply->original_status == expected &&
                           reply->id > last_geam && !reply->parameter_bytes && !reply->address &&
                           reply->data.empty() && reply->arguments.empty(),
                       "exact native extension status and admitted identity survive real IPC");
                last_geam = reply->id;
                drain();
            }
            const auto boundary_request = [&](const auto& problem, ExecutionOpcode operation,
                                              int expected) {
                ExecutionPayloadWriter input;
                if constexpr (std::is_same_v<std::decay_t<decltype(problem)>, ExecutionSgemv>)
                    expect(write_execution_sgemv(input, problem), "canonical boundary SGEMV");
                else
                    expect(write_execution_sgemm_ex(input, problem), "canonical boundary SgemmEx");
                const auto reply = connection.exchange(operation, input.take());
                expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                           reply->status == expected && reply->original_status == expected &&
                           reply->id > last_geam && !reply->parameter_bytes && !reply->address &&
                           reply->data.empty() && reply->arguments.empty(),
                       "boundary request retains admitted identity and exact native SDK status");
                last_geam = reply->id;
                drain();
            };
            for (const int rows : {0, -1}) {
                auto empty_vector = gemv;
                empty_vector.rows = rows;
                empty_vector.presence = 0;
                empty_vector.buffers = {};
                empty_vector.alpha = empty_vector.beta = {};
                boundary_request(empty_vector, ExecutionOpcode::kSgemv, CUBLAS_STATUS_SUCCESS);
                auto empty_matrix = exact;
                empty_matrix.rows = rows;
                empty_matrix.presence = 0;
                empty_matrix.buffers = {};
                empty_matrix.alpha = empty_matrix.beta = {};
                boundary_request(empty_matrix, ExecutionOpcode::kSgemmEx,
                                 rows ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS);
            }
            auto zero_depth = exact;
            zero_depth.depth = 0;
            zero_depth.presence = 19;
            zero_depth.buffers[0] = zero_depth.buffers[1] = {};
            boundary_request(zero_depth, ExecutionOpcode::kSgemmEx, CUBLAS_STATUS_SUCCESS);
        }
        ExecutionPayloadWriter free;
        expect(free.write_u64(memory.id), "owned extended backing release");
        success(connection.exchange(ExecutionOpcode::kRelease, free.take()));
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        if (allow_extended_stop.load(std::memory_order_acquire)) native.is_ready = true;
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded actual library service progress");
    }
    client.get();
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "library owner cleanup completes after disconnect");
    expect(library_native.creates == 2 && library_native.versions == 19 &&
               library_native.extended == 2 && library_native.classic_batches == 1 &&
               library_native.geams == 10 && library_native.gemvs == 6 &&
               library_native.sgemm_ex_calls == 8 && library_native.destroys == 1 &&
               library_native.lt_creates == 3 && library_native.lt_destroys == 2 &&
               library_native.workspace_sets == 4 && library_native.atomics_sets == 1 &&
               library_native.sm_count_sets == 3 && native.destroys == 1 && native.launches == 0 &&
               native.synchronizes == 0 && service.request_stop(),
           "actual SDK operations and implicit library teardown run exactly once without "
           "fabricated GPU work");
}
// A deliberately raw peer can prequeue Close/Hello boundaries that the ordinary
// client convenience API correctly refuses while it retains admission receipts.
class IntakePeer final {
   public:
    explicit IntakePeer(const std::string& path)
        : descriptor(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        expect(descriptor.get() >= 0 && path.size() < sizeof(address.sun_path),
               "create bounded raw intake peer");
        std::copy(path.begin(), path.end(), address.sun_path);
        expect(::connect(descriptor.get(), reinterpret_cast<const sockaddr*>(&address),
                         sizeof(address)) == 0,
               "connect actual intake socket");
    }
    void send(ExecutionOpcode operation, std::span<const std::byte> payload = {}, int fd = -1) {
        const auto encoded = encode_execution_message(
            {{ExecutionMessageKind::kRequest, operation, 0, session, next_request++, incarnation},
             {payload.begin(), payload.end()}});
        expect(encoded.has_value(), "encode exact raw intake frame");
        ssize_t sent{};
        if (fd < 0) {
            sent = ::send(descriptor.get(), encoded->data(), encoded->size(),
                          MSG_DONTWAIT | MSG_NOSIGNAL);
        } else {
            alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> ancillary{};
            iovec data{const_cast<std::byte*>(encoded->data()), encoded->size()};
            msghdr message{};
            message.msg_iov = &data;
            message.msg_iovlen = 1;
            message.msg_control = ancillary.data();
            message.msg_controllen = ancillary.size();
            auto* item = CMSG_FIRSTHDR(&message);
            item->cmsg_level = SOL_SOCKET;
            item->cmsg_type = SCM_RIGHTS;
            item->cmsg_len = CMSG_LEN(sizeof(int));
            std::memcpy(CMSG_DATA(item), &fd, sizeof(fd));
            sent = ::sendmsg(descriptor.get(), &message, MSG_DONTWAIT | MSG_NOSIGNAL);
        }
        expect(sent == static_cast<ssize_t>(encoded->size()),
               "complete frame is prequeued before service progress");
    }
    std::optional<ExecutionWireReply> take() {
        for (unsigned int attempt = 0; attempt < 8; ++attempt) {
            if (!header && input.size() == kExecutionHeaderBytes) {
                header = decode_execution_header(input);
                expect(header.has_value(), "decode complete service reply header");
            }
            const auto needed = kExecutionHeaderBytes + (header ? header->payload_bytes : 0);
            if (header && input.size() == needed) {
                expect(header->kind == ExecutionMessageKind::kReply &&
                           header->sequence == next_reply++,
                       "exact ordered raw peer reply");
                if (header->operation == ExecutionOpcode::kHello) {
                    session = header->session;
                    incarnation = header->incarnation;
                }
                auto result = decode_execution_reply(
                    std::span<const std::byte>{input}.subspan(kExecutionHeaderBytes));
                expect(result.has_value(), "decode complete actual service reply");
                input.clear();
                header.reset();
                return result;
            }
            std::array<std::byte, 4096> bytes{};
            const auto count = ::recv(descriptor.get(), bytes.data(),
                                      std::min(bytes.size(), needed - input.size()), MSG_DONTWAIT);
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return std::nullopt;
            expect(count > 0, "raw reply remains connected until normal Close");
            input.insert(input.end(), bytes.begin(), bytes.begin() + count);
        }
        return std::nullopt;
    }
    ExecutionWireReply wait(ExecutionService& service) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        for (;;) {
            if (const auto reply = take()) return *reply;
            expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
                   "bounded raw intake reply progress");
        }
    }
    std::vector<std::byte> prepare(ExecutionService& service) {
        send(ExecutionOpcode::kHello);
        success(wait(service));
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry helper() { bar.sync 15; ret; }"),
               "prepare an ordinary original function");
        send(ExecutionOpcode::kLoadPtx, source.take());
        const auto module = success(wait(service));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("helper"),
               "resolve raw function");
        send(ExecutionOpcode::kGetFunction, lookup.take());
        const auto function = success(wait(service));
        ExecutionPayloadWriter launch;
        expect(launch.write_u64(function.id) && launch.write_u64(0) && launch.write_u32(1) &&
                   launch.write_u32(1) && launch.write_u32(1) && launch.write_u32(32) &&
                   launch.write_u32(1) && launch.write_u32(1) && launch.write_u32(0) &&
                   launch.write_blob({}) && launch.write_u32(kExecutionAsyncLaunchMagic),
               "encode raw ordinary ASY1");
        return launch.take();
    }
    ExecutionOwnedDescriptor descriptor;
    std::uint64_t session{}, next_request{1}, next_reply{1};
    std::array<std::uint64_t, 2> incarnation{};

   private:
    std::vector<std::byte> input;
    std::optional<ExecutionWireHeader> header;
};
ExecutionServiceConfig intake_config(const std::string& path, bool enabled) {
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 8;
    config.uses_original_kernels = true;
    config.should_receive_async_burst = enabled;
    const auto peer = current_process_identity();
    expect(peer.has_value(), "authenticate intake fixture");
    config.client_priorities.push_back({*peer, 0, ExecutionClientKernelMode::kOriginal});
    return config;
}
void intake_count(const ExecutionService& service, bool enabled, std::uint64_t expected) {
    const auto count = service.async_intake_second_takes();
    expect(enabled ? count && *count == expected : !count,
           "only successful enabled second-round takes contribute to the aggregate");
}
void intake_admitted(const std::optional<ExecutionWireReply>& reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
               !reply->status && !reply->original_status && reply->id && reply->address,
           "ordinary frame has a real admission receipt, not an SDK result");
}
void poll_journal_service(const std::string& path) {
    native = {};
    auto journal = std::make_shared<ExecutionSocketPollJournal>(
        ExecutionSocketPollJournal::Limits{128, 1024, 256, 2048, 256});
    auto config = intake_config(path, false);
    config.transport.should_record_receive_time = true;
    ExecutionService service(journal, std::move(config), kDispatch);
    expect(service.start(), "start the explicit diagnostic runtime constructor");
    IntakePeer peer(path);
    const auto launch = peer.prepare(service);
    peer.send(ExecutionOpcode::kLaunch, launch);
    intake_admitted(peer.wait(service));
    expect(native.launches == 1 && !native.event_destroys && journal->valid(),
           "journal cannot manufacture physical retirement at admission");
    native.is_ready = true;
    peer.send(ExecutionOpcode::kSnapshot);
    const auto snapshot = peer.wait(service);
    expect(snapshot.disposition == ExecutionWireDisposition::kComplete && !snapshot.status &&
               !snapshot.original_status && snapshot.data.size() == 60,
           "owner snapshot completes without claiming a native SDK call");
    const std::array invalid{std::byte{0x35}};
    peer.send(ExecutionOpcode::kLaunch, invalid);
    const auto rejected = peer.wait(service);
    expect(rejected.status == CUDA_ERROR_INVALID_VALUE && !rejected.original_status &&
               native.launches == 1,
           "diagnostic transport preserves malformed launch rejection and once-only execution");
    peer.send(ExecutionOpcode::kClose);
    success(peer.wait(service));
    expect(service.request_stop() && native.event_destroys == 1 && native.destroys == 1 &&
               journal->valid() && journal->frames().size() == 7,
           "diagnostic runtime retains complete ordinary frames and native cleanup");
    const auto output = path + ".poll";
    expect(journal->publish(output) && ::unlink(output.c_str()) == 0,
           "publish and remove only the completed CPU runtime fixture journal");
}
std::vector<std::byte> intake_control_boundary(const std::string& path, bool enabled,
                                               ExecutionOpcode operation, int launch_suffix = 0) {
    native = {};
    ExecutionService service(intake_config(path, enabled), kDispatch);
    expect(service.start(), "start control-frame intake fixture");
    intake_count(service, enabled, 0);
    IntakePeer peer(path);
    const auto launch = peer.prepare(service);
    peer.send(ExecutionOpcode::kLaunch, launch);
    std::optional<std::vector<std::byte>> continuation;
    if (operation == ExecutionOpcode::kSourceContinuation) {
        const ExecutionSourceContinuation scope{0, peer.incarnation, 1, 1, peer.session, 0};
        continuation = encode_execution_source_continuation(scope, ExecutionOpcode::kSnapshot, {});
        expect(continuation.has_value(), "valid carrier with unauthorized continuation identity");
    } else if (operation == ExecutionOpcode::kLaunch) {
        continuation = launch;
        if (launch_suffix == 1)
            continuation->resize(continuation->size() - sizeof(std::uint32_t));
        else
            continuation->back() ^= std::byte{1};
    }
    ExecutionOwnedDescriptor fd(operation == ExecutionOpcode::kLoadModuleFile
                                    ? ::open("/dev/null", O_RDONLY | O_CLOEXEC)
                                    : -1);
    expect(operation != ExecutionOpcode::kLoadModuleFile || fd.get() >= 0,
           "own exact control-frame descriptor");
    peer.send(
        operation,
        continuation ? std::span<const std::byte>{*continuation} : std::span<const std::byte>{},
        fd.get());
    expect(service.step(0), "first ordinary intake round");
    intake_admitted(peer.take());
    expect(!peer.take() && service.pending_priority_requests(0) == 0 && native.launches == 1 &&
               !native.event_destroys && !native.destroys,
           "unsafe second frame is not consumed or executed before the next ordinary step");
    intake_count(service, enabled, 0);
    native.is_ready = true;
    const auto reply = peer.wait(service);
    const auto encoded_reply = encode_execution_reply(reply);
    expect(encoded_reply.has_value(),
           "retain the entire ordinary control reply for policy comparison");
    if (operation == ExecutionOpcode::kClose || operation == ExecutionOpcode::kSnapshot ||
        launch_suffix == 1)
        expect(!reply.status, "ordinary control succeeds after its original boundary");
    else
        expect(reply.status != 0, "invalid control retains its ordinary rejection");
    if (operation != ExecutionOpcode::kClose) {
        peer.send(ExecutionOpcode::kClose);
        success(peer.wait(service));
    }
    expect(service.request_stop() && native.event_destroys == (launch_suffix == 1 ? 2 : 1) &&
               native.destroys == 1,
           "control boundary preserves physical and context cleanup");
    intake_count(service, enabled, 0);
    return *encoded_reply;
}
void intake_disconnect_boundary(const std::string& path) {
    native = {};
    ExecutionService service(intake_config(path, true), kDispatch);
    expect(service.start(), "start extra-poll disconnect fixture");
    IntakePeer peer(path);
    const auto launch = peer.prepare(service);
    peer.send(ExecutionOpcode::kLaunch, launch);
    expect(service.step(0), "issue independently held earlier work");
    intake_admitted(peer.take());
    expect(native.launches == 1 && !native.event_destroys, "earlier physical event is held");
    intake_count(service, true, 0);
    peer.send(ExecutionOpcode::kLaunch, launch);
    poll_wait_fixture::disconnect_descriptor = peer.descriptor.get();
    poll_wait_fixture::polls_before_disconnect = 2;
    poll_wait_fixture::disconnected = false;
    const bool progressed = service.step(0);
    poll_wait_fixture::polls_before_disconnect = 0;
    poll_wait_fixture::disconnect_descriptor = -1;
    expect(progressed && poll_wait_fixture::disconnected && !native.event_destroys &&
               !native.destroys && !service.request_stop(),
           "second-poll disconnect cannot release or replay accepted physical ownership");
    intake_count(service, true, 0);
    const auto issued = native.launches;
    expect(issued >= 1 && issued <= 2, "uncertain unaccepted suffix creates no duplicate issue");
    native.is_ready = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!service.request_stop())
        expect(std::chrono::steady_clock::now() < deadline,
               "bounded physical disconnect retirement");
    expect(native.launches == issued && native.event_destroys == issued && native.destroys == 1,
           "actual stop permits exactly-once final ownership cleanup");
}
void intake_fairness(const std::string& path, bool enabled, bool unsafe_first) {
    native = {};
    auto config = intake_config(path, enabled);
    const auto metrics = std::make_shared<HostStageMetrics>(HostStageMetrics::system_clock, 512);
    config.host_stage_metrics = metrics;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start two-peer intake fixture");
    IntakePeer first(path), second(path);
    const auto first_launch = first.prepare(service);
    const auto second_launch = second.prepare(service);
    first.send(ExecutionOpcode::kLaunch, first_launch);
    second.send(ExecutionOpcode::kLaunch, second_launch);
    first.send(
        unsafe_first ? ExecutionOpcode::kSnapshot : ExecutionOpcode::kLaunch,
        unsafe_first ? std::span<const std::byte>{} : std::span<const std::byte>{first_launch});
    second.send(ExecutionOpcode::kLaunch, second_launch);
    expect(service.step(0), "first chance for both prequeued peers");
    intake_admitted(first.take());
    intake_admitted(second.take());
    if (enabled) {
        if (!unsafe_first)
            intake_admitted(first.take());
        else
            expect(!first.take(), "earlier unsafe peer remains unconsumed");
        intake_admitted(second.take());
    } else {
        expect(!first.take() && !second.take(), "default path takes one frame per peer");
    }
    expect(native.launches == (enabled ? (unsafe_first ? 3 : 4) : 2) && !native.event_destroys,
           "fixed per-peer round bound preserves every held physical event");
    intake_count(service, enabled, unsafe_first ? 1 : 2);
    native.is_ready = true;
    if (!enabled || unsafe_first) {
        const auto reply = first.wait(service);
        if (unsafe_first)
            expect(!reply.status, "retained unsafe frame uses next ordinary step");
        else
            intake_admitted(reply);
    }
    if (!enabled) intake_admitted(second.wait(service));
    first.send(ExecutionOpcode::kClose);
    success(first.wait(service));
    second.send(ExecutionOpcode::kClose);
    success(second.wait(service));
    expect(service.request_stop() && native.destroys == 2 &&
               native.launches == (unsafe_first ? 3 : 4) &&
               native.event_destroys == native.launches,
           "two ordinary peers clean up all actual work");
    const auto publication = path + ".json";
    expect(metrics->valid() && metrics->publish(publication),
           "publish existing receive chronology");
    std::ifstream file(publication);
    const std::string json((std::istreambuf_iterator<char>(file)), {});
    file.close();
    expect(::unlink(publication.c_str()) == 0, "remove consumed temporary lifecycle publication");
    const auto begin = json.find("\"events\":[");
    expect(begin != std::string::npos, "existing lifecycle array is present");
    std::vector<std::array<std::uint64_t, 2>> received;
    auto cursor = begin + std::string_view("\"events\":[").size();
    while (cursor < json.size() && json[cursor] != ']') {
        if (json[cursor] == ',') ++cursor;
        expect(json[cursor] == '[', "canonical lifecycle event row");
        const auto end = json.find(']', cursor);
        expect(end != std::string::npos, "complete lifecycle row");
        std::istringstream row(json.substr(cursor + 1, end - cursor - 1));
        std::array<std::uint64_t, 6> values{};
        for (std::size_t index = 0; index < values.size(); ++index) {
            expect(static_cast<bool>(row >> values[index]), "complete numeric lifecycle field");
            char comma{};
            if (index + 1 != values.size())
                expect(static_cast<bool>(row >> comma) && comma == ',',
                       "canonical field separator");
        }
        if (values[2] == static_cast<std::uint64_t>(HostStageMetrics::Lifecycle::kReceive) &&
            values[3] >= 4 && values[3] <= 5)
            received.push_back({values[1], values[3]});
        cursor = end + 1;
    }
    expect(received.size() == 4 && received[0][0] != received[1][0] && received[0][1] == 4 &&
               received[1][1] == 4 && received[2][1] == 5 && received[3][1] == 5,
           "both peers get their first frame before either receives a second");
    if (enabled && unsafe_first)
        expect(received[2][0] == received[1][0],
               "unsafe first peer cannot starve later eligible peer");
}
void intake_initialization_and_capacity(const std::string& path, bool enabled) {
    native = {};
    auto config = intake_config(path, enabled);
    config.transport.max_pending_requests = 1;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start one-credit intake control");
    IntakePeer peer(path);
    const auto payload = peer.prepare(service);
    peer.send(ExecutionOpcode::kLaunch, payload);
    peer.send(ExecutionOpcode::kLaunch, payload);
    expect(service.step(0), "progress one-credit first frame");
    intake_admitted(peer.take());
    expect(native.launches == 1 && !peer.take(), "second round cannot exceed pending capacity");
    intake_count(service, enabled, 0);
    intake_admitted(peer.wait(service));
    native.is_ready = true;
    peer.send(ExecutionOpcode::kClose);
    success(peer.wait(service));
    expect(service.request_stop() && native.launches == 2 && native.event_destroys == 2 &&
               native.destroys == 1,
           "capacity fallback retains normal cleanup");
    intake_count(service, enabled, 0);
    native = {};
    ExecutionService cold(intake_config(path + ".cold", enabled), kDispatch);
    expect(cold.start(), "start cold Hello boundary");
    IntakePeer uninitialized(path + ".cold");
    uninitialized.send(ExecutionOpcode::kHello);
    uninitialized.send(ExecutionOpcode::kSnapshot);
    expect(cold.step(0) && cold.step(0), "accept peer and retain only Hello");
    success(uninitialized.take());
    expect(native.context_creates == 1 && !native.destroys,
           "Hello initializes before the prequeued invalid next identity is examined");
    expect(cold.step(0) && cold.request_stop() && native.destroys == 1,
           "ordinary next-step validation rejects the pre-bootstrap session identity");
    intake_count(cold, enabled, 0);
}
void intake_boundaries(const std::string& path) {
    std::vector<std::vector<std::byte>> replies;
    for (const bool enabled : {false, true}) {
        std::size_t index{};
        for (const auto& [operation, suffix] : {std::pair{ExecutionOpcode::kClose, 0},
                                                {ExecutionOpcode::kSnapshot, 0},
                                                {ExecutionOpcode::kGraphControl, 0},
                                                {ExecutionOpcode::kSourceContinuation, 0},
                                                {ExecutionOpcode::kLoadModuleFile, 0},
                                                {ExecutionOpcode::kLaunch, 1},
                                                {ExecutionOpcode::kLaunch, 2}}) {
            const auto reply = intake_control_boundary(path, enabled, operation, suffix);
            if (!enabled)
                replies.push_back(reply);
            else
                expect(reply == replies.at(index), "OFF/ON preserve the complete control reply");
            ++index;
        }
        for (const bool unsafe_first : {false, true}) intake_fairness(path, enabled, unsafe_first);
        intake_initialization_and_capacity(path, enabled);
    }
    intake_disconnect_boundary(path);
    for (unsigned int excluded = 0; excluded < 10; ++excluded) {
        native = {};
        auto config = intake_config(path, true);
        if (excluded == 0) config.client_priorities.front().priority = 1;
        if (excluded == 1) config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
        if (excluded == 2) config.uses_nccl = true;
        if (excluded == 3) config.shared = ExecutionServiceSharedConfig{};
        if (excluded == 4) config.profile_input = "not-opened";
        if (excluded == 5) config.paired = ExecutionServicePairConfig{};
        if (excluded == 6)
            config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic;
        if (excluded == 7)
            config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
        if (excluded == 8) config.profile_output = "not-created";
        if (excluded == 9) config.persistent.uses_workload_agnostic = true;
        bool rejected{};
        try {
            ExecutionService invalid(std::move(config), kDispatch);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !native.driver_initializations && !native.context_creates,
               "excluded burst policy rejects before native entry");
    }
}
void sharded_signal_configuration(const std::string& path) {
    native = {};
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    expect(!config.persistent.uses_sharded_host_retreat &&
               config.session_limits.max_private_host_bytes == std::size_t{1024} * 1024,
           "ordinary service configuration retains scalar signal and its existing host budget");
    {
        ExecutionService ordinary(config, kDispatch);
        expect(!native.driver_initializations && !native.context_creates,
               "ordinary service construction acquires no native resources");
    }
    config.persistent.workers = 320;
    config.persistent.replace_sgemm = true;
    config.persistent.gemm_tile_side = 32;
    config.persistent.uses_sharded_host_retreat = true;
    constexpr auto per_slot = kPersistentProgressHostSlotBytes + kGemmSignalSlotBytes;
    config.session_limits.max_private_host_bytes =
        config.session_limits.submissions.max_commands * per_slot;
    if (!cuda_sgemm_sharded_image(32).empty()) {
        ExecutionService valid(config, kDispatch);
        expect(!native.driver_initializations && !native.context_creates,
               "bounded trusted shard configuration is admitted without native work");
    }
    for (unsigned int index = 0; index < 5; ++index) {
        auto invalid = config;
        if (index == 0) invalid.persistent.workers = 0;
        if (index == 1) invalid.persistent.replace_sgemm = false;
        if (index == 2) invalid.persistent.gemm_tile_side = 64;
        if (index == 3) invalid.persistent.kind = glimmer::transform::PtxExecutionKind::kDynamic;
        if (index == 4) --invalid.session_limits.max_private_host_bytes;
        bool rejected{};
        try {
            ExecutionService service(std::move(invalid), kDispatch);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !native.driver_initializations && !native.context_creates,
               "invalid sharded service configuration rejects before native initialization");
    }
}
struct OrdinaryAsyncHandshake {
    std::atomic<bool> setup_ready{}, may_close{};
    std::atomic<std::uint32_t> permitted_batch{}, ready_batch{}, completed_batch{};
};
struct OrdinaryAsyncSteps {
    std::uint64_t all{}, compute{};
};

// Keep a distinct owner-only sampling boundary; client, setup and teardown are excluded.
[[gnu::noinline]] void ordinary_async_owner_loop(ExecutionService& service,
                                                 OrdinaryAsyncHandshake& handshake,
                                                 std::future<void>& client, std::uint32_t batches,
                                                 std::chrono::steady_clock::time_point deadline,
                                                 OrdinaryAsyncSteps* steps = nullptr) {
    for (std::uint32_t batch = 1; batch <= batches; ++batch) {
        native.is_ready = false;
        handshake.permitted_batch.store(batch, std::memory_order_release);
        while (handshake.ready_batch.load(std::memory_order_acquire) != batch) {
            if (client.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready)
                client.get();
            expect(std::chrono::steady_clock::now() < deadline,
                   "client publishes the complete bounded batch before owner progress");
            std::this_thread::yield();
        }
        while (handshake.completed_batch.load(std::memory_order_acquire) != batch) {
            if (client.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready)
                client.get();
            if (steps) {
                ++steps->all;
                if (native.launches < batch * 32) ++steps->compute;
            }
            expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
                   "ordinary asynchronous chain makes bounded owner progress");
            if (!native.is_ready && native.launches == batch * 32) {
                expect(native.event_destroys == (batch - 1) * 32 &&
                           native.launches - native.event_destroys == 32,
                       "all 32 original issues coexist before any batch retirement");
                native.is_ready = true;
            }
        }
        expect(native.launches == batch * 32 && native.event_destroys == batch * 32,
               "each batch closes every issue and physical event before the next batch");
    }
}

void ordinary_async_chain(const std::string& path, bool checks_provenance,
                          bool receives_burst = false, bool measures_intake = false) {
    constexpr std::uint32_t batch_size = 32;
    const std::uint32_t commands = checks_provenance ? 96 : measures_intake ? 8192 : 100000;
    native = {};
    native.is_ordinary_chain = true;
    native.is_prepared = true;
    native.ordinary_intake_width = receives_burst ? 2 : 1;
    const auto peer = current_process_identity();
    expect(peer.has_value(), "authenticate ordinary asynchronous fixture");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 8;
    config.persistent.workers = 320;
    config.client_priorities.push_back({*peer, 0, ExecutionClientKernelMode::kConfigured});
    config.priority_idle_us = 2000;
    config.should_skip_idle_helper_plans = true;
    config.should_receive_async_burst = receives_burst;
    const auto depth = checks_provenance ? std::make_shared<IssueDepthMetrics>(
                                               IssueDepthMetrics::system_clock, commands)
                                         : nullptr;
    config.issue_depth_metrics = depth;
    native.ordinary_depth = depth.get();
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start ordinary asynchronous service fixture");
    if (checks_provenance) {
        intake_count(service, receives_burst, 0);
        auto foreign =
            std::async(std::launch::async, [&] { return service.async_intake_second_takes(); });
        expect(!foreign.get(), "foreign thread cannot read the owner aggregate");
    }
    OrdinaryAsyncHandshake handshake;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection(
            {path, *peer, 3000, kExecutionMaxPayloadBytes, batch_size});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry helper() { bar.sync 15; ret; }"),
               "use an original nontransformable zero-argument function");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("helper"), "look up original");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        ExecutionPayloadWriter input;
        expect(input.write_u64(function.id) && input.write_u64(0) && input.write_u32(1) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(32) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(0) &&
                   input.write_blob({}) && input.write_u32(kExecutionAsyncLaunchMagic),
               "encode ordinary nondeferred ASY1 launch");
        const auto payload = input.take();
        const auto complete = [&](ExecutionOpcode operation,
                                  std::span<const std::byte> bytes = {}) {
            auto reply = connection.exchange(operation, bytes);
            if (!reply || reply->disposition != ExecutionWireDisposition::kComplete ||
                reply->status || reply->original_status)
                throw std::runtime_error(
                    "ordinary chain completion failed: opcode=" +
                    std::to_string(static_cast<unsigned int>(operation)) +
                    ", status=" + (reply ? std::to_string(reply->status) : "missing"));
            return *reply;
        };
        handshake.setup_ready.store(true, std::memory_order_release);
        for (std::uint32_t batch = 1; batch <= commands / batch_size; ++batch) {
            while (handshake.permitted_batch.load(std::memory_order_acquire) != batch) {
                expect(std::chrono::steady_clock::now() < deadline, "bounded batch permission");
                std::this_thread::yield();
            }
            std::array<std::uint64_t, batch_size> requests{};
            std::array<ExecutionAsyncFrontierKey, batch_size> keys{};
            for (auto& request : requests) {
                bool fully_flushed{};
                const auto submitted = connection.submit_observed(ExecutionOpcode::kLaunch, payload,
                                                                  -1, nullptr, &fully_flushed);
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && fully_flushed,
                       "every ordinary frame reaches the socket before owner batch progress");
                request = submitted.sequence;
            }
            handshake.ready_batch.store(batch, std::memory_order_release);
            for (std::size_t index = 0; index < requests.size(); ++index) {
                std::optional<ExecutionWireReply> reply;
                while (!(reply = connection.take_reply(requests[index])))
                    expect(connection.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                           "collect each exact ordinary admission reply");
                expect(
                    reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                        !reply->status && !reply->original_status && reply->address &&
                        reply->id == static_cast<std::size_t>(batch - 1) * batch_size + index + 1,
                    "ordinary asynchronous commands have contiguous unique receipts");
                keys[index] = {reply->address, reply->id};
            }
            complete(ExecutionOpcode::kDrain);
            const auto encoded = encode_execution_async_frontier_batch(keys);
            expect(encoded.has_value(), "encode all exact batch frontiers");
            const auto queried = complete(ExecutionOpcode::kQueryAsyncFrontiers, *encoded);
            const auto observed = decode_execution_async_frontier_observations(queried.data);
            expect(observed && observed->size() == batch_size, "complete batch retirement receipt");
            for (std::size_t index = 0; index < keys.size(); ++index)
                expect((*observed)[index].key == keys[index] && (*observed)[index].is_retired &&
                           !(*observed)[index].status,
                       "every exact frontier independently reports successful retirement");
            expect(complete(ExecutionOpcode::kReleaseAsyncFrontiers, *encoded).data == *encoded,
                   "release exactly the fully retired batch capabilities");
            handshake.completed_batch.store(batch, std::memory_order_release);
        }
        while (!handshake.may_close.load(std::memory_order_acquire)) {
            expect(std::chrono::steady_clock::now() < deadline, "bounded teardown permission");
            std::this_thread::yield();
        }
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    while (!handshake.setup_ready.load(std::memory_order_acquire)) {
        if (client.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready)
            client.get();
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "ordinary fixture setup finishes outside sampled owner loop");
    }
    OrdinaryAsyncSteps steps;
    HostStageMetrics::Stamp begin{}, end{};
    if (measures_intake)
        expect(HostStageMetrics::system_clock(begin), "sample intake owner-loop start");
    ordinary_async_owner_loop(service, handshake, client, commands / batch_size, deadline,
                              measures_intake ? &steps : nullptr);
    if (measures_intake)
        expect(HostStageMetrics::system_clock(end), "sample intake owner-loop finish");
    handshake.may_close.store(true, std::memory_order_release);
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "ordinary fixture tears down outside sampled owner loop");
    client.get();
    expect(service.request_stop() && service.audit_healthy() && native.destroys == 1 &&
               native.launches == commands && native.event_destroys == commands &&
               native.peak_unretired == batch_size && !native.synchronizes &&
               native.host_frees == 1 && !native.initializations,
           "ordinary async chain closes without retries, SDK errors or private preparation");
    intake_count(service, receives_burst, commands / 2);
    if (depth) {
        depth->finish();
        expect(depth->valid() && native.overlapping_batches == commands / batch_size,
               "all correctness batches overlap before first issue");
        completed_issue_events(*depth, commands);
        std::vector<IssueDepthMetrics::Event> events;
        for (std::size_t index = 0; index < depth->event_count(); ++index)
            events.push_back(depth->event(index));
        std::ranges::sort(events, {}, &IssueDepthMetrics::Event::sequence);
        for (std::size_t index = 0; index < events.size(); ++index) {
            const auto& event = events[index];
            const auto last = index / batch_size * batch_size + batch_size - 1;
            expect(!event.preparation_required && !event.service_observed_prepared_ns &&
                       event.sequence == index + 1,
                   "every ordinary asynchronous command is nondeferred");
            expect(event.retired_ns >= events[last].issue_end_ns,
                   "every earlier batch command remains unretired through the last host issue");
            if (index % batch_size)
                expect(events[index - 1].issue_end_ns <= event.issue_begin_ns,
                       "successor follows an already host-issued predecessor");
        }
    }
    native.ordinary_depth = nullptr;
    std::cout << "Ordinary async chain: admitted=" << commands << ", issued=" << native.launches
              << ", retired=" << native.event_destroys
              << ", peak_unretired=" << native.peak_unretired << ", observers=" << checks_provenance
              << '\n';
    if (measures_intake)
        std::cout << "Ordinary async intake: enabled=" << receives_burst
                  << ", commands=" << commands << ", steps=" << steps.all
                  << ", compute_steps=" << steps.compute << ", wall_ns=" << end.wall - begin.wall
                  << ", cpu_ns=" << end.cpu - begin.cpu << '\n';
}

struct DepthFixture {
    bool observes{true}, audits{true}, has_unsupported_request{}, records_events{},
        records_selections{};
    std::uint32_t commands{1};
};
std::array<std::uint64_t, 2> issue_depth_deferred(const std::string& path, bool records_events,
                                                  std::uint32_t commands) {
    native = {};
    native.is_prepared = native.is_ready = true;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.persistent.workers = 320;
    const auto depth = std::make_shared<IssueDepthMetrics>(
        IssueDepthMetrics::system_clock,
        records_events ? IssueDepthMetrics::event_capacity_limit : 0);
    config.issue_depth_metrics = depth;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start actual deferred-admission issue-depth fixture");
    const auto peer = current_process_identity();
    expect(peer.has_value(), "authenticate deferred fixture");
    // Construct both ledgers before timing, including the event storage reservation.
    HostStageMetrics::Stamp begin{}, end{};
    expect(HostStageMetrics::system_clock(begin), "sample deferred fixture owner clocks");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *peer, 3000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "load the actual zero-argument source through the service");
        const auto module = success(connection.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"),
               "look up original entry");
        const auto function =
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        ExecutionPayloadWriter input;
        // The shared fake SDK reports exactly one completed logical block.
        expect(input.write_u64(function.id) && input.write_u64(0) && input.write_u32(1) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(32) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(0) &&
                   input.write_blob({}) && input.write_u32(kExecutionAsyncLaunchMagic),
               "encode actual ASY1 deferred kernel");
        const auto payload = input.take();
        for (std::uint32_t index = 0; index < commands; ++index) {
            const auto reply = connection.exchange(ExecutionOpcode::kLaunch, payload);
            expect(reply && reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                       !reply->status && reply->id == index + 1 && reply->address &&
                       !reply->original_status && !reply->parameter_bytes &&
                       reply->arguments.empty() && reply->data.empty(),
                   "asynchronous admission preserves exact receipt and frontier identities");
            const ExecutionAsyncFrontierKey key{reply->address, reply->id};
            const auto drained = connection.exchange(ExecutionOpcode::kDrain);
            expect(drained && drained->disposition == ExecutionWireDisposition::kComplete &&
                       !drained->status && !drained->original_status,
                   "drain proves retirement independently of asynchronous admission");
            const auto keys = encode_execution_async_frontier_batch(std::span{&key, 1});
            expect(keys.has_value(), "encode exact retired frontier");
            const auto queried = connection.exchange(ExecutionOpcode::kQueryAsyncFrontiers, *keys);
            expect(queried && queried->disposition == ExecutionWireDisposition::kComplete &&
                       !queried->status && !queried->original_status,
                   "query the exact asynchronous frontier after drain");
            const auto observed = decode_execution_async_frontier_observations(queried->data);
            expect(observed && observed->size() == 1 && observed->front().key == key &&
                       observed->front().is_retired && !observed->front().status,
                   "every command physically retires with its own identity");
            const auto released =
                connection.exchange(ExecutionOpcode::kReleaseAsyncFrontiers, *keys);
            expect(released && released->disposition == ExecutionWireDisposition::kComplete &&
                       !released->status && !released->original_status && released->data == *keys,
                   "release each exact frontier before the next bounded admission");
        }
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "bounded deferred observation progress");
    client.get();
    expect(service.request_stop() && service.audit_healthy() && native.destroys == 1 &&
               native.launches == commands && native.event_destroys == commands,
           "deferred observer preserves exactly-once issue and physical cleanup");
    expect(HostStageMetrics::system_clock(end), "sample final deferred fixture owner clocks");
    depth->finish();
    expect(depth->valid(), "every admitted deferred event closes without diagnostic errors");
    completed_issue_events(*depth, records_events ? commands : 0);
    for (std::size_t index = 0; index < depth->event_count(); ++index)
        expect(depth->event(index).preparation_required == 1 &&
                   depth->event(index).service_observed_prepared_ns,
               "every sensitivity event actually traversed the unique deferred preparation hook");
    return {end.wall - begin.wall, end.cpu - begin.cpu};
}
std::array<std::uint64_t, 2> issue_depth_blas(const std::string& path, DepthFixture options = {}) {
    native = {};
    native.is_prepared = native.is_ready = true;
    library_native = {};
    auto dispatch = kDispatch;
    dispatch.blas.create = [](cublasHandle_t* handle) {
        *handle = reinterpret_cast<cublasHandle_t>(&library_native.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.destroy = [](cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; };
    dispatch.blas.get_stream = [](cublasHandle_t, cudaStream_t* stream) {
        *stream = nullptr;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.get_pointer_mode = [](cublasHandle_t, cublasPointerMode_t* mode) {
        *mode = CUBLAS_POINTER_MODE_HOST;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.sgeam = service_geam<float>;
    ExecutionServiceConfig config;
    dispatch.blas.set_stream = [](cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; };
    dispatch.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                             const float*, const float*, int, const float*, int, const float*,
                             float*, int) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    config.transport.socket_path = path;
    config.uses_original_kernels = true;
    config.should_audit_native = options.audits;
    const auto depth =
        options.observes
            ? std::make_shared<IssueDepthMetrics>(
                  IssueDepthMetrics::system_clock,
                  options.records_events ? IssueDepthMetrics::event_capacity_limit : 0,
                  IssueDepthMetrics::SelectionConfig{
                      options.records_selections ? IssueDepthMetrics::selection_capacity_limit : 0})
            : nullptr;
    config.issue_depth_metrics = depth;
    PreparationCapture capture;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start native BLAS issue-depth fixture");
    const auto peer = current_process_identity();
    expect(peer.has_value(), "authenticate BLAS fixture");
    HostStageMetrics::Stamp begin{}, end{};
    expect(HostStageMetrics::system_clock(begin), "sample fixture owner clocks");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *peer, 3000});
        success(connection.open());
        const auto handle = success(connection.exchange(ExecutionOpcode::kCreateBlas));
        ExecutionPayloadWriter allocation;
        expect(allocation.write_u64(512), "encode owned matrix storage");
        const auto memory =
            success(connection.exchange(ExecutionOpcode::kAllocate, allocation.take()));
        ExecutionGeam problem;
        problem.type = ExecutionGeamType::kFloat;
        problem.transpose_a = problem.transpose_b = CUBLAS_OP_T;
        problem.rows = 2;
        problem.columns = 3;
        problem.leading_a = problem.leading_b = 4;
        problem.leading_c = 3;
        problem.handle = handle.id;
        problem.presence = 31;
        problem.buffers[0] = problem.buffers[1] = {memory.id, 0};
        problem.buffers[2] = {memory.id, 256};
        const float one = 1.0F;
        std::memcpy(problem.alpha.data(), &one, sizeof(one));
        ExecutionPayloadWriter wire;
        expect(write_execution_geam(wire, problem), "encode exact original BLAS computation");
        const auto payload = wire.take();
        for (std::uint32_t index = 0; index < options.commands; ++index)
            expect(success(connection.exchange(ExecutionOpcode::kGeam, payload)).id == index + 1,
                   "one accepted ordinary BLAS command per native call");
        const auto drained = connection.exchange(ExecutionOpcode::kDrain);
        expect(drained && drained->disposition == ExecutionWireDisposition::kComplete &&
                   drained->status == CUDA_SUCCESS && !drained->original_status,
               "owner drain proves retirement without inventing an SDK return value");
        if (options.has_unsupported_request) {
            const auto rejected = connection.exchange(ExecutionOpcode::kDnnCompute);
            expect(rejected && rejected->disposition == ExecutionWireDisposition::kRejected &&
                       rejected->status == CUDA_ERROR_NOT_SUPPORTED && !rejected->original_status,
                   "unsupported diagnostic scope preserves the original DNN profile gate");
        }
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "bounded BLAS observation progress");
    client.get();
    expect(service.request_stop() && service.audit_healthy() && native.destroys == 1 &&
               native.event_destroys == options.commands &&
               library_native.geams == options.commands && native.launches == 0,
           "function-zero BLAS retains exact native issue and physical cleanup counts");
    expect(HostStageMetrics::system_clock(end), "sample final fixture owner clocks");
    if (depth) {
        depth->finish();
        expect(depth->valid() != options.has_unsupported_request,
               "only unsupported diagnostic scope invalidates the completed BLAS ledger");
        std::uint64_t admitted{}, submitted{}, retired{};
        for (std::size_t index = 0; index < IssueDepthMetrics::bucket_capacity; ++index) {
            admitted += depth->bucket(index).admitted;
            submitted += depth->bucket(index).submitted;
            retired += depth->bucket(index).retired;
        }
        expect(admitted == options.commands && submitted == admitted && retired == admitted,
               "function-zero compute is counted independently of kernel handles");
        completed_issue_events(*depth, options.records_events ? admitted : 0);
        if (options.records_selections) {
            std::uint64_t expected{};
            for (std::size_t index = 0; index < IssueDepthMetrics::bucket_capacity; ++index) {
                const auto& bucket = depth->bucket(index);
                expect(!bucket.unknown, "selection fixture has no unknown observations");
                expected += bucket.ready_compute + bucket.not_ready_compute + bucket.other_pending;
            }
            expect(depth->selection_count() == expected && expected >= admitted,
                   "every nonzero pending observation is retained");
            for (std::size_t index = 0; index < depth->event_count(); ++index) {
                const auto& event = depth->event(index);
                std::size_t matching{};
                for (std::size_t row = 0; row < depth->selection_count(); ++row) {
                    const auto& selected = depth->selection(row);
                    if (selected.connection == event.connection &&
                        selected.observation_generation ==
                            event.first_positive_observation_generation) {
                        expect(
                            selected.pending_sequence == event.sequence &&
                                selected.observed_ns == event.first_observed_ns &&
                                selected.classification ==
                                    IssueDepthMetrics::SelectionClass::kTrackedReady,
                            "first positive event and sparse selection share the same observation");
                        ++matching;
                    }
                }
                expect(matching == 1, "each event has exactly one first-positive selection");
            }
        } else {
            expect(!depth->selection_count() && !depth->selection_storage_bytes(),
                   "selection journal is opt-in");
        }
    }
    if (options.audits)
        expect(capture.messages.str().find("issued=1, retained=0, retired=1") != std::string::npos,
               "native audit is not replaced by the depth observer");
    return {end.wall - begin.wall, end.cpu - begin.cpu};
}
void cold_preparation_audit(const std::string& path, bool is_enabled, bool fails_load) {
    native = {};
    native.is_prepared = true;
    PreparationCapture capture;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.exit_when_idle = true;
    config.persistent.workers = 320;
    config.should_audit_native = is_enabled;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
    auto dispatch = kDispatch;
    if (fails_load)
        dispatch.module_load = [](CUmodule*, const void*) {
            ++native.module_loads;
            return CUDA_ERROR_INVALID_IMAGE;
        };
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual cold-preparation audit service");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted preparation service identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        success(connection.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "preserve original PTX in the audited request");
        const auto module = connection.exchange(ExecutionOpcode::kLoadPtx, source.take());
        if (fails_load) {
            expect(module && module->status == CUDA_ERROR_INVALID_IMAGE &&
                       module->original_status == CUDA_ERROR_INVALID_IMAGE && !module->id,
                   "audit preserves the real original failure after optional preparation");
        } else {
            const auto loaded = success(module);
            ExecutionPayloadWriter lookup;
            expect(lookup.write_u64(loaded.id) && lookup.write_string("kernel"),
                   "request the original function only");
            success(connection.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        }
        const auto malformed = connection.exchange(ExecutionOpcode::kGetFunction);
        expect(malformed && malformed->status == CUDA_ERROR_INVALID_VALUE &&
                   !malformed->original_status,
               "request audit cannot fabricate SDK provenance for a parsing rejection");
        success(connection.exchange(ExecutionOpcode::kClose));
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded cold-preparation audit progress");
    client.get();
    expect(native.module_loads == (fails_load ? 2U : 1U) && native.destroys == 1 &&
               !native.launches && !native.synchronizes && service.request_stop(),
           "audit adds no SDK operation, work replay or physical completion authority");
    std::istringstream lines(capture.messages.str());
    std::vector<std::string> phases;
    unsigned int begins{}, ends{}, phase_records{};
    bool is_request_active{};
    for (std::string line; std::getline(lines, line);) {
        if (line.starts_with("Preparation request audit:")) {
            expect(is_enabled && line.find("observer_enabled=true") != std::string::npos,
                   "observer binding and reset remain valid on the service owner");
            if (line.find("event=begin,") != std::string::npos) {
                expect(!is_request_active && phases.empty(), "one cold request attempt at a time");
                is_request_active = true;
                ++begins;
            } else {
                expect(line.find("event=end,") != std::string::npos && is_request_active &&
                           phases.empty() && line.find("outcome=reply,") != std::string::npos,
                       "request completion follows all phases, even a failed native call");
                is_request_active = false;
                ++ends;
            }
        } else if (line.starts_with("Preparation phase audit:")) {
            expect(is_enabled && is_request_active, "phase belongs to an active cold request");
            const auto begin = line.find("phase=");
            const auto end = line.find(',', begin);
            expect(begin != std::string::npos && end != std::string::npos, "bounded phase name");
            const auto name = line.substr(begin, end - begin);
            if (line.find("event=begin,") != std::string::npos)
                phases.push_back(name);
            else {
                expect(line.find("event=end,") != std::string::npos && !phases.empty() &&
                           phases.back() == name,
                       "native load and nested function validation phases are balanced");
                phases.pop_back();
            }
            ++phase_records;
        }
    }
    expect(!is_request_active && phases.empty() && begins == ends &&
               begins == (is_enabled ? (fails_load ? 2U : 3U) : 0U) &&
               (is_enabled ? phase_records > 0 : phase_records == 0),
           "disabled audit is silent; enabled success/failure/rejection is complete and balanced");
}
namespace unified_copy_fixture {
void admission_status_controls() {
    using Status = glimmer::backend::ExecutionAcceptStatus;
    struct Case {
        Status status;
        CUresult default_result, alternate_result;
    };
    constexpr std::array cases{
        Case{Status::kAccepted, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
        Case{Status::kInvalid, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
        Case{Status::kWrongSequence, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
        Case{Status::kWrongProcess, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
        Case{Status::kFull, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
        Case{Status::kClosed, CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_CONTEXT_IS_DESTROYED},
        Case{Status::kQuarantined, CUDA_ERROR_UNKNOWN, CUDA_ERROR_UNKNOWN},
        Case{Status::kOutOfMemory, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_OUT_OF_MEMORY}};
    for (const auto& test : cases)
        expect(admission_rejection_status(test.status, CUDA_ERROR_INVALID_VALUE) ==
                       test.default_result &&
                   admission_rejection_status(test.status, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) ==
                       test.alternate_result,
               "every admission status preserves its exact policy error or caller fallback");
}
ExecutionWireReply synthetic_success(std::optional<ExecutionWireReply> result) {
    expect(result && result->disposition == ExecutionWireDisposition::kComplete &&
               result->status == 0 && !result->original_status && !result->parameter_bytes &&
               result->arguments.empty(),
           "pageable metadata completion cannot invent an original SDK operation");
    return std::move(*result);
}
// HostOwner uses a different native thread/context from application execution.
// Only native owners access Context; cross-thread fixture controls are atomic.
struct Context {
    std::map<CUdeviceptr, std::unique_ptr<std::byte[]>> allocations;
};
thread_local Context* current_context{};
struct State {
    std::atomic<unsigned int> creates{}, destroys{}, frees{}, copies{}, synchronous{},
        asynchronous{}, directions{}, registrations{}, unregistrations{}, event_creates{},
        event_destroys{}, event_queries{}, retired{}, owner_proofs{}, context_reclaims{};
    std::atomic<CUresult> copy_status{CUDA_SUCCESS};
    std::atomic<const char*> callback_failure{};
} state;
bool callback_check(bool condition, const char* message) {
    if (!condition) {
        const char* absent{};
        (void)state.callback_failure.compare_exchange_strong(absent, message);
    }
    return condition;
}
void* storage(CUdeviceptr address) {
    return std::bit_cast<void*>(address);
}
CUresult CUDAAPI copy(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes) {
    if (!callback_check(current_context != nullptr && bytes == 128,
                        "generic callback requires its actual owner and exact byte extent"))
        return CUDA_ERROR_INVALID_VALUE;
    const auto status = state.copy_status.load();
    if (status == CUDA_SUCCESS) std::memcpy(storage(destination), storage(source), bytes);
    ++state.copies;
    return status;
}
CudaSessionDispatch dispatch() {
    auto result = kDispatch;
    result.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    result.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    result.execution.context_get_current = [](CUcontext* output) {
        *output = reinterpret_cast<CUcontext>(current_context);
        // During the retained-copy phases the application owns its device
        // allocation, while the separate HostOwners have no device allocations.
        // Their proofs therefore establish actual helper work, not main-owner
        // polling of the still-pending physical event.
        if (current_context && current_context->allocations.empty()) ++state.owner_proofs;
        return CUDA_SUCCESS;
    };
    result.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        if (!callback_check(!current_context, "new native owner requires a detached thread"))
            return CUDA_ERROR_INVALID_CONTEXT;
        current_context = new Context;
        *output = reinterpret_cast<CUcontext>(current_context);
        ++state.creates;
        return CUDA_SUCCESS;
    };
    result.context_set_current = [](CUcontext context) {
        current_context = reinterpret_cast<Context*>(context);
        return CUDA_SUCCESS;
    };
    result.context_destroy = [](CUcontext context) {
        if (!callback_check(context && context == reinterpret_cast<CUcontext>(current_context),
                            "context teardown requires its exact current owner"))
            return CUDA_ERROR_INVALID_CONTEXT;
        // cuCtxDestroy legitimately reclaims allocations on an exception path.
        // Do not throw here and hide the client's original assertion. The
        // normal-path oracle below still requires one explicit free and zero
        // context-reclaimed allocations.
        state.context_reclaims += static_cast<unsigned int>(current_context->allocations.size());
        delete current_context;
        current_context = nullptr;
        ++state.destroys;
        return CUDA_SUCCESS;
    };
    result.context_get_flags = [](unsigned int* output) {
        if (!callback_check(current_context != nullptr, "metadata requires the native owner"))
            return CUDA_ERROR_INVALID_CONTEXT;
        *output = 0;
        return CUDA_SUCCESS;
    };
    result.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        if (!callback_check(current_context && bytes == 256, "exact owned device allocation"))
            return CUDA_ERROR_INVALID_VALUE;
        auto memory = std::make_unique<std::byte[]>(bytes);
        std::memset(memory.get(), 0xa5, bytes);
        *output = std::bit_cast<CUdeviceptr>(memory.get());
        current_context->allocations.emplace(*output, std::move(memory));
        return CUDA_SUCCESS;
    };
    result.release = [](CUdeviceptr address) {
        if (!callback_check(current_context && current_context->allocations.erase(address) == 1,
                            "release exactly one retained native device allocation"))
            return CUDA_ERROR_INVALID_VALUE;
        ++state.frees;
        return CUDA_SUCCESS;
    };
    result.execution.event_create = [](CUevent* output, unsigned int) {
        *output = reinterpret_cast<CUevent>(new unsigned int(0));
        ++state.event_creates;
        return CUDA_SUCCESS;
    };
    result.execution.event_record = [](CUevent event, CUstream stream) {
        if (!callback_check(!stream && state.copies > 0,
                            "tail must be recorded after actual generic issue"))
            return CUDA_ERROR_INVALID_HANDLE;
        *reinterpret_cast<unsigned int*>(event) = state.copies;
        return CUDA_SUCCESS;
    };
    result.execution.event_query = [](CUevent event) {
        ++state.event_queries;
        const auto sequence = *reinterpret_cast<unsigned int*>(event);
        return sequence != 0 && sequence <= state.retired ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    result.execution.event_destroy = [](CUevent event) {
        delete reinterpret_cast<unsigned int*>(event);
        ++state.event_destroys;
        return CUDA_SUCCESS;
    };
    result.execution.stream_synchronize = [](CUstream) {
        (void)callback_check(false,
                             "generic copies cannot replace independent events with a drain");
        return CUDA_ERROR_UNKNOWN;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original generic Driver copy ABIs.
    result.execution.copy_unified = [](CUdeviceptr destination, CUdeviceptr source,
                                       std::size_t bytes) {
        const auto status = copy(destination, source, bytes);
        ++state.synchronous;
        return status;
    };
    result.execution.copy_unified_async = [](CUdeviceptr destination, CUdeviceptr source,
                                             std::size_t bytes, CUstream stream) {
        if (!callback_check(!stream, "original generic Async requires the exact default stream"))
            return CUDA_ERROR_INVALID_HANDLE;
        const auto status = copy(destination, source, bytes);
        ++state.asynchronous;
        return status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    result.execution.copy_device = [](CUdeviceptr, CUdeviceptr, std::size_t) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.execution.copy_device_async = [](CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.upload = [](CUdeviceptr, const void*, std::size_t) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.download = [](void*, CUdeviceptr, std::size_t) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.upload_async = [](CUdeviceptr, const void*, std::size_t, CUstream) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.download_async = [](void*, CUdeviceptr, std::size_t, CUstream) {
        ++state.directions;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.execution.upload_async = result.upload_async;
    result.execution.download_async = result.download_async;
    result.host_register = [](void*, std::size_t, unsigned int) {
        ++state.registrations;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.host_unregister = [](void*) {
        ++state.unregistrations;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.host_map = [](CUdeviceptr* output, void*, unsigned int) {
        *output = 0;
        return CUDA_ERROR_INVALID_VALUE;
    };
    return result;
}
std::vector<std::byte> by_id(std::uint64_t id) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id), "encode retained resource identity");
    return writer.take();
}
std::vector<std::byte> payload(const ExecutionUnifiedCopy& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_unified_copy(writer, value), "encode canonical unified copy");
    return writer.take();
}
void rejected_copy(std::optional<ExecutionWireReply> reply, int status = CUDA_ERROR_INVALID_VALUE) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
               reply->status == status && !reply->original_status,
           "invalid unified-copy capabilities never claim actual native issue");
}
void run(const std::string& path, CUresult native_failure) {
    admission_status_controls();
    expect(native_failure == CUDA_ERROR_INVALID_VALUE || native_failure == CUDA_ERROR_OUT_OF_MEMORY,
           "fixed independent native SDK failure scenarios");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 4;
    config.exit_when_idle = true;
    ExecutionService service(config, dispatch());
    expect(service.start(), "start real authenticated unified-copy service");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated service process identity");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient application({path, *identity, 5000});
        success(application.open());
        const auto device = success(application.exchange(ExecutionOpcode::kAllocate, by_id(256)));
        ExecutionSocketClient pageable({path, *identity, 5000});
        ExecutionPayloadWriter bootstrap;
        expect(bootstrap.write_u32(kExecutionPageableOwnerRole) && bootstrap.write_u32(0),
               "explicit private unregistered role");
        success(pageable.open(bootstrap.take()));
        ExecutionOwnedDescriptor backing(
            ::memfd_create("glimmer-uva-service", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        expect(
            backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
            "size-sealed writable pageable backing");
        auto* mapping = static_cast<unsigned char*>(
            ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, backing.get(), 0));
        expect(mapping != MAP_FAILED, "map independently owned client image");
        std::array<unsigned char, 4096> expected{};
        expected.fill(0xd3);
        for (std::size_t index = 0; index != 128; ++index)
            expected[index + 64] = static_cast<unsigned char>((index * 17 + 3) & 255);
        std::memcpy(mapping, expected.data(), expected.size());
        const auto import = [&](unsigned int flags, int descriptor = -1) {
            ExecutionPayloadWriter writer;
            expect(writer.write_u64(256) && writer.write_u64(4096) && writer.write_u32(flags),
                   "exact bounded pageable import");
            return pageable.exchange(ExecutionOpcode::kImportHostMemory, writer.take(),
                                     descriptor < 0 ? backing.get() : descriptor);
        };
        rejected_copy(import(CU_MEMHOSTALLOC_PORTABLE));
        const auto resource = synthetic_success(import(0));
        expect(resource.id != 0 && resource.address == 0 && resource.data.size() == 4 &&
                   std::ranges::all_of(resource.data,
                                       [](std::byte value) { return value == std::byte{}; }),
               "PAG import exposes no device alias and retains a certain owner");
        ExecutionUnifiedCopy upload{
            {ExecutionCopyEndpointKind::kDevice, 0, device.id, 64},
            {ExecutionCopyEndpointKind::kPageable, pageable.session(), resource.id, 64},
            128,
            0,
            false};
        const auto invalid = [&](ExecutionUnifiedCopy value) {
            rejected_copy(application.exchange(ExecutionOpcode::kUnifiedCopy, payload(value)));
        };
        auto wrong = upload;
        wrong.source.connection = application.session();
        invalid(wrong);
        wrong = upload;
        wrong.source.connection = UINT64_MAX;
        invalid(wrong);
        wrong = upload;
        wrong.source.kind = ExecutionCopyEndpointKind::kHost;
        invalid(wrong);
        wrong = upload;
        wrong.source.offset = 129;
        invalid(wrong);
        wrong = upload;
        wrong.destination.offset = 129;
        invalid(wrong);
        wrong = upload;
        wrong.source.allocation = resource.id + 1;
        invalid(wrong);
        for (const auto role : {kExecutionHostOwnerRole, kExecutionStagingOwnerRole}) {
            ExecutionSocketClient other({path, *identity, 5000});
            ExecutionPayloadWriter role_input;
            expect(role_input.write_u32(role) && role_input.write_u32(0), "distinct helper role");
            success(other.open(role_input.take()));
            wrong = upload;
            wrong.source.connection = other.session();
            invalid(wrong);
            const auto closed = other.exchange(ExecutionOpcode::kClose);
            expect(closed && closed->status == 0, "empty non-PAG helper closes normally");
        }
        for (const auto operation :
             {ExecutionOpcode::kHostTransfer, ExecutionOpcode::kStagedHostTransfer}) {
            ExecutionPayloadWriter writer;
            expect(writer.write_u64(pageable.session()) && writer.write_u64(resource.id) &&
                       writer.write_u64(64) && writer.write_u64(device.id) &&
                       writer.write_u64(64) && writer.write_u64(128) && writer.write_u64(0) &&
                       writer.write_u32(1),
                   "PAG capability cannot masquerade as old HST/STG transfer authority");
            rejected_copy(application.exchange(operation, writer.take()));
        }
        rejected_copy(pageable.exchange(ExecutionOpcode::kUnifiedCopy, payload(upload)),
                      CUDA_ERROR_NOT_SUPPORTED);
        wrong = upload;
        wrong.destination = wrong.source;
        rejected_copy(application.exchange(ExecutionOpcode::kUnifiedCopy, payload(wrong)),
                      CUDA_ERROR_NOT_SUPPORTED);
        expect(state.copies == 0 && state.registrations == 0 && state.directions == 0,
               "all malformed/role controls reject before any native copy or registration");

        success(application.exchange(ExecutionOpcode::kUnifiedCopy, payload(upload)));
        expect(state.synchronous == 1 && state.asynchronous == 0 && state.retired == 0,
               "ordinary H2D returns its exact cuMemcpy result before the held tail");
        std::array<unsigned char, 256> device_expected{};
        device_expected.fill(0xa5);
        std::memcpy(device_expected.data() + 64, expected.data() + 64, 128);
        expect(std::memcmp(storage(device.address), device_expected.data(),
                           device_expected.size()) == 0 &&
                   std::memcmp(mapping, expected.data(), expected.size()) == 0,
               "first copy preserves every device and pageable guard byte");
        const auto idle = pageable.submit(ExecutionOpcode::kQueryStagingIdle, by_id(resource.id));
        expect(idle.status == ExecutionSocketSubmitStatus::kAccepted,
               "submit actual PAG idle query");
        for (unsigned int turn = 0; turn != 8; ++turn)
            expect(pageable.poll_for(1) && !pageable.take_reply(idle.sequence),
                   "SDK return cannot release the independent physical pageable lease");
        success(application.exchange(ExecutionOpcode::kGetContextFlags));
        expect(state.retired == 0 && state.event_queries > 0,
               "metadata progresses behind held pageable idle");
        state.retired = 1;
        const auto await = [](ExecutionSocketClient& connection, std::uint64_t sequence) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            std::optional<ExecutionWireReply> reply;
            while (!(reply = connection.take_reply(sequence)))
                expect(connection.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                       "bounded exact correlated reply");
            return reply;
        };
        const auto idle_reply = await(pageable, idle.sequence);
        expect(idle_reply && idle_reply->status == 0 && idle_reply->id == resource.id &&
                   !idle_reply->original_status,
               "idle receipt follows its actual physical tail without inventing an SDK call");

        auto download = upload;
        std::swap(download.destination, download.source);
        download.is_asynchronous = true;
        std::memset(mapping + 64, 0x71, 128);
        const auto pending = application.submit(ExecutionOpcode::kUnifiedCopy, payload(download));
        expect(pending.status == ExecutionSocketSubmitStatus::kAccepted,
               "submit actual generic download");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (state.asynchronous != 1)
            expect(application.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "observe original Async callback, not admission-only success");
        for (unsigned int turn = 0; turn != 8; ++turn)
            expect(application.poll_for(1) && !application.take_reply(pending.sequence),
                   "PAG download reply waits for physical completion after SDK return");
        expect(state.retired == 1 && std::memcmp(mapping, expected.data(), expected.size()) == 0,
               "private result and guards are complete but not yet public reply authority");
        state.retired = 2;
        success(await(application, pending.sequence));

        upload.is_asynchronous = true;
        state.copy_status = native_failure;
        const auto failed = application.exchange(ExecutionOpcode::kUnifiedCopy, payload(upload));
        expect(failed && failed->disposition == ExecutionWireDisposition::kComplete &&
                   failed->status == native_failure && failed->original_status == native_failure &&
                   failed->id == 3 && state.copies == 3 && state.retired == 2,
               "actual generic SDK error is returned once and does not prove tail retirement");
        expect(std::memcmp(storage(device.address), device_expected.data(),
                           device_expected.size()) == 0 &&
                   std::memcmp(mapping, expected.data(), expected.size()) == 0,
               "failed native copy retains complete no-write images");
        const auto proofs_before_release = state.owner_proofs.load();
        const auto releasing =
            pageable.submit(ExecutionOpcode::kReleaseHostMemory, by_id(resource.id));
        expect(releasing.status == ExecutionSocketSubmitStatus::kAccepted,
               "request exact mapping release");
        const auto release_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (state.owner_proofs == proofs_before_release)
            expect(pageable.poll_for(1) && !pageable.take_reply(releasing.sequence) &&
                       std::chrono::steady_clock::now() < release_deadline,
                   "real HostOwner release enters its native context before deferred result");
        for (unsigned int turn = 0; turn != 8; ++turn)
            expect(pageable.poll_for(1) && !pageable.take_reply(releasing.sequence),
                   "even an original SDK error retains mapping through its physical event");
        state.retired = 3;
        const auto released = synthetic_success(await(pageable, releasing.sequence));
        expect(!released.id && !released.address && released.data.empty(),
               "pageable release reports only synthetic metadata completion");
        rejected_copy(pageable.exchange(ExecutionOpcode::kQueryStagingIdle, by_id(resource.id)));
        invalid(upload);

        ExecutionOwnedDescriptor destination_backing(
            ::memfd_create("glimmer-h2h-service", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        expect(destination_backing.get() >= 0 &&
                   ::ftruncate(destination_backing.get(), 4096) == 0 &&
                   ::fcntl(destination_backing.get(), F_ADD_SEALS,
                           F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
               "independent sealed H2H destination extent");
        auto* host_output = static_cast<unsigned char*>(::mmap(
            nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, destination_backing.get(), 0));
        expect(host_output != MAP_FAILED, "map independent full H2H output image");
        const auto host_source = synthetic_success(import(0));
        const auto host_destination = synthetic_success(import(0, destination_backing.get()));
        state.copy_status = CUDA_SUCCESS;
        for (unsigned int mode = 0; mode != 2; ++mode) {
            std::memset(host_output, 0xa7, 4096);
            auto host_expected = expected;
            host_expected.fill(0xa7);
            std::copy_n(expected.begin() + 64, 128, host_expected.begin() + 64);
            const unsigned int sequence = 4 + mode;
            ExecutionUnifiedCopy h2h{
                {ExecutionCopyEndpointKind::kPageable, pageable.session(), host_destination.id, 64},
                {ExecutionCopyEndpointKind::kPageable, pageable.session(), host_source.id, 64},
                128,
                0,
                mode != 0};
            const auto submitted = application.submit(ExecutionOpcode::kUnifiedCopy, payload(h2h));
            expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
                   "submit one exact H2H request through real IPC");
            const auto copy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (state.copies != sequence)
                expect(application.poll_for(1) && std::chrono::steady_clock::now() < copy_deadline,
                       "observe exact original H2H callback");
            for (unsigned int turn = 0; turn != 8; ++turn)
                expect(application.poll_for(1) && !application.take_reply(submitted.sequence),
                       "H2H pageable output requires its independent physical tail");
            const auto source_idle =
                pageable.submit(ExecutionOpcode::kQueryStagingIdle, by_id(host_source.id));
            const auto destination_idle =
                pageable.submit(ExecutionOpcode::kQueryStagingIdle, by_id(host_destination.id));
            expect(source_idle.status == ExecutionSocketSubmitStatus::kAccepted &&
                       destination_idle.status == ExecutionSocketSubmitStatus::kAccepted,
                   "both independently owned endpoint leases are observable");
            for (unsigned int turn = 0; turn != 8; ++turn)
                expect(pageable.poll_for(1) && !pageable.take_reply(source_idle.sequence) &&
                           !pageable.take_reply(destination_idle.sequence),
                       "neither endpoint becomes idle from an SDK host return");
            expect(std::memcmp(host_output, host_expected.data(), host_expected.size()) == 0 &&
                       std::memcmp(mapping, expected.data(), expected.size()) == 0,
                   "H2H preserves complete source, destination and every guard byte");
            state.retired = sequence;
            success(await(application, submitted.sequence));
            synthetic_success(await(pageable, source_idle.sequence));
            synthetic_success(await(pageable, destination_idle.sequence));
        }
        synthetic_success(
            pageable.exchange(ExecutionOpcode::kReleaseHostMemory, by_id(host_source.id)));
        synthetic_success(
            pageable.exchange(ExecutionOpcode::kReleaseHostMemory, by_id(host_destination.id)));
        expect(::munmap(host_output, 4096) == 0, "release proven H2H destination client mapping");
        success(application.exchange(ExecutionOpcode::kRelease, by_id(device.id)));
        expect(state.frees == 1 && state.retired == 5,
               "explicit device free occurs only after every generic-copy physical tail retires");
        for (auto* connection : {&pageable, &application}) {
            const auto closed = connection->exchange(ExecutionOpcode::kClose);
            expect(closed && closed->status == 0 &&
                       closed->disposition == ExecutionWireDisposition::kComplete,
                   "physically retired contexts close normally despite a returned copy error");
        }
        expect(::munmap(mapping, 4096) == 0,
               "release independent client mapping after normal owner cleanup");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (!service.finished()) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded actual IPC/service/native-actor progress");
        if (client.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) break;
    }
    try {
        client.get();
    } catch (const std::exception& error) {
        std::cerr << "Unified-copy client failure: " << error.what() << '\n';
        if (const auto* failure = state.callback_failure.load())
            std::cerr << "Unified-copy native callback failure: " << failure << '\n';
        throw;
    }
    if (const auto* failure = state.callback_failure.load()) throw std::runtime_error(failure);
    expect(service.request_stop() && state.creates == 4 && state.destroys == 4 &&
               state.frees == 1 && state.context_reclaims == 0 && state.synchronous == 2 &&
               state.asynchronous == 3 && state.copies == 5 && state.event_creates == 5 &&
               state.event_destroys == 5 && state.directions == 0 && state.registrations == 0 &&
               state.unregistrations == 0 && !current_context,
           "all real owners, byte images and tails retire without directional fallback or "
           "registration");
}
}  // namespace unified_copy_fixture
}  // namespace
int main(int argc, char** argv) {
    char pattern[] = "/tmp/glimmer-exec-runtime-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sharded-signal-config") {
            sharded_signal_configuration(std::string(directory) + "/sharded-signal.sock");
            expect(::rmdir(directory) == 0, "empty shard configuration fixture directory");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--poll-journal") {
            poll_journal_service(std::string(directory) + "/poll-journal.sock");
            expect(::rmdir(directory) == 0, "empty diagnostic runtime directory");
            std::cout << "Poll journal runtime ownership checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--async-intake-boundaries") {
            intake_boundaries(std::string(directory) + "/intake-boundaries.sock");
            expect(::rmdir(directory) == 0, "empty intake boundary fixture directory");
            std::cout << "Bounded asynchronous intake ownership checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--async-intake") {
            const auto mode = std::string_view(argv[2]);
            const auto policy = std::string_view(argv[3]);
            expect((mode == "check" || mode == "bench") && (policy == "on" || policy == "off"),
                   "explicit asynchronous intake mode and policy");
            ordinary_async_chain(std::string(directory) + "/async-intake.sock", mode == "check",
                                 policy == "on", mode == "bench");
            expect(::rmdir(directory) == 0, "empty asynchronous intake directory");
            return EXIT_SUCCESS;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--ordinary-async-chain") {
            const auto mode = std::string_view(argv[2]);
            expect(mode == "check" || mode == "profile", "explicit ordinary async fixture mode");
            ordinary_async_chain(std::string(directory) + "/ordinary-async.sock", mode == "check");
            expect(::rmdir(directory) == 0, "empty ordinary async fixture directory");
            return EXIT_SUCCESS;
        }
        business_owner_lease_contract();
        async_audit_contract();
        audit_terminal_failure_contract(std::string(directory) + "/audit-failure.sock");
        if (argc == 2 && (std::string_view(argv[1]) == "--unified-copy" ||
                          std::string_view(argv[1]) == "--unified-copy-oom")) {
            const auto native_failure = std::string_view(argv[1]) == "--unified-copy-oom"
                                            ? CUDA_ERROR_OUT_OF_MEMORY
                                            : CUDA_ERROR_INVALID_VALUE;
            unified_copy_fixture::run(std::string(directory) + "/unified-copy.sock",
                                      native_failure);
            expect(::rmdir(directory) == 0, "empty unified-copy service fixture directory");
            std::cout << "Owned generic UVA copy service checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--context-purpose") {
            cold_context_purpose(std::string(directory) + "/context-purpose.sock");
            idle_helper_planning(std::string(directory) + "/helper-planning.sock");
            expect(::rmdir(directory) == 0, "empty context-purpose fixture directory");
            std::cout << "Authenticated cold context purpose checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--issue-depth") {
            for (const auto operation :
                 {ExecutionOpcode::kLaunch, ExecutionOpcode::kSgemm, ExecutionOpcode::kGemmEx,
                  ExecutionOpcode::kStridedGemmEx, ExecutionOpcode::kStridedSgemm,
                  ExecutionOpcode::kLtMatmul, ExecutionOpcode::kGeam, ExecutionOpcode::kSgemv,
                  ExecutionOpcode::kSgemmEx})
                expect(issue_depth_compute_class(operation) == IssueDepthComputeClass::kIncluded,
                       "explicit ordinary compute family classification");
            expect(issue_depth_compute_class(ExecutionOpcode::kDnnCompute) ==
                           IssueDepthComputeClass::kUnsupported &&
                       issue_depth_compute_class(ExecutionOpcode::kMemory) ==
                           IssueDepthComputeClass::kExcluded,
                   "unsupported compute is not silently treated as non-compute");
            for (const bool events : {false, true}) {
                (void)issue_depth_deferred(std::string(directory) + "/depth-deferred.sock", events,
                                           3);
                (void)issue_depth_blas(std::string(directory) + "/depth-blas.sock",
                                       {.records_events = events});
                (void)issue_depth_blas(std::string(directory) + "/depth-dnn.sock",
                                       {.has_unsupported_request = true, .records_events = events});
                selected_variants_execute_through_service(
                    std::string(directory) + "/depth-ptb.sock", Kind::kPreemptive, false, false,
                    false, true, events);
                disconnect_waits_for_gpu(std::string(directory) + "/depth.sock", CUDA_SUCCESS,
                                         false, false, true, events);
                disconnect_waits_for_gpu(std::string(directory) + "/depth-memory.sock",
                                         CUDA_SUCCESS, false, true, true, events);
                disconnect_waits_for_gpu(std::string(directory) + "/depth-error.sock",
                                         CUDA_ERROR_INVALID_VALUE, false, false, true, events);
            }
            (void)issue_depth_blas(
                std::string(directory) + "/depth-selections.sock",
                {.records_events = true, .records_selections = true, .commands = 3});
            expect(::rmdir(directory) == 0, "empty issue-depth service fixture directory");
            std::cout << "Passive issue-depth service checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--issue-events-overhead" ||
                          std::string_view(argv[1]) == "--preparation-events-overhead" ||
                          std::string_view(argv[1]) == "--issue-selections-overhead")) {
            const bool deferred = std::string_view(argv[1]) == "--preparation-events-overhead";
            const bool selections = std::string_view(argv[1]) == "--issue-selections-overhead";
            std::array<double, 3> wall_ratios{}, cpu_ratios{};
            for (std::size_t pair = 0; pair < 3; ++pair) {
                std::array<std::uint64_t, 2> off{}, on{};
                const bool on_first = pair == 1;
                for (const bool enabled : {on_first, !on_first}) {
                    const auto path = std::string(directory) + "/events-overhead.sock";
                    const auto measured =
                        deferred
                            ? issue_depth_deferred(path, enabled, 2000)
                            : issue_depth_blas(path, {.audits = false,
                                                      .records_events = selections || enabled,
                                                      .records_selections = selections && enabled,
                                                      .commands = 2000});
                    (enabled ? on : off) = measured;
                    std::cout << "Issue-event CPU sample: pair=" << pair << ", on=" << enabled
                              << ", wall_ns=" << measured[0] << ", cpu_ns=" << measured[1]
                              << ", commands=2000, retired=2000, events="
                              << (selections || enabled ? 2000 : 0)
                              << ", deferred_events=" << (enabled && deferred ? 2000 : 0)
                              << ", selection_first_positives="
                              << (enabled && selections ? 2000 : 0) << '\n';
                }
                expect(off[0] && off[1], "positive paired reference durations");
                wall_ratios[pair] = static_cast<double>(on[0]) / static_cast<double>(off[0]);
                cpu_ratios[pair] = static_cast<double>(on[1]) / static_cast<double>(off[1]);
            }
            std::ranges::sort(wall_ratios);
            std::ranges::sort(cpu_ratios);
            std::cout << "Issue-event median paired ratios: wall=" << wall_ratios[1]
                      << ", cpu=" << cpu_ratios[1] << '\n';
            expect(wall_ratios[1] <= 1.20 && cpu_ratios[1] <= 1.20,
                   "frozen owner-event CPU perturbation gate");
            expect(::rmdir(directory) == 0, "empty event overhead fixture directory");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--issue-depth-overhead") {
            std::array<std::uint64_t, 2> off{}, on{};
            for (unsigned int pair = 0; pair < 3; ++pair) {
                const bool on_first = pair == 1;
                for (const bool enabled : {on_first, !on_first}) {
                    const auto measured =
                        issue_depth_blas(std::string(directory) + "/depth-overhead.sock",
                                         {.observes = enabled, .audits = false, .commands = 2000});
                    auto& total = enabled ? on : off;
                    for (std::size_t index = 0; index < total.size(); ++index)
                        total[index] += measured[index];
                    std::cout << "Issue-depth CPU sample: pair=" << pair << ", on=" << enabled
                              << ", wall_ns=" << measured[0] << ", cpu_ns=" << measured[1] << '\n';
                }
            }
            expect(::rmdir(directory) == 0, "empty bounded overhead fixture directory");
            std::cout << "Issue-depth CPU ratio: wall="
                      << static_cast<double>(on[0]) / static_cast<double>(off[0])
                      << ", cpu=" << static_cast<double>(on[1]) / static_cast<double>(off[1])
                      << '\n';
            expect(on[0] * 5 <= off[0] * 6 && on[1] * 5 <= off[1] * 6,
                   "diagnostic CPU perturbation must be at most twenty percent");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pointer-metadata") {
            owned_pointer_metadata(std::string(directory) + "/pointer-metadata.sock");
            expect(::rmdir(directory) == 0, "empty pointer metadata service directory");
            std::cout << "Owned pointer metadata service checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-attributes") {
            owned_function_controls(std::string(directory) + "/runtime-attributes.sock");
            expect(::rmdir(directory) == 0, "empty aggregate function service fixture directory");
            std::cout << "Owned Runtime function aggregate service checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--lt-query-context") {
            lt_query_context_activation(std::string(directory) + "/lt-query.sock");
            expect(::rmdir(directory) == 0, "empty Lt query service fixture directory");
            std::cout << "Owned Lt query context activation checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--geam" ||
                          std::string_view(argv[1]) == "--blas-extension")) {
            original_library_status_domains(std::string(directory) + "/geam.sock");
            expect(::rmdir(directory) == 0, "empty GEAM service fixture directory");
            std::cout << "Owned GEAM service protocol checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--preparation-audit") {
            for (const bool enabled : {false, true})
                for (const bool failure : {false, true})
                    cold_preparation_audit(std::string(directory) + "/preparation.sock", enabled,
                                           failure);
            expect(::rmdir(directory) == 0, "empty preparation fixture directory");
            std::cout << "Cold-preparation service audit checks passed\n";
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown service fixture argument");
        cold_context_purpose(std::string(directory) + "/context-purpose.sock");
        idle_helper_planning(std::string(directory) + "/helper-planning.sock");
        owned_pointer_metadata(std::string(directory) + "/pointer-metadata.sock");
        lt_query_context_activation(std::string(directory) + "/lt-query.sock");
        for (const bool enabled : {false, true})
            for (const bool failure : {false, true})
                cold_preparation_audit(std::string(directory) + "/preparation.sock", enabled,
                                       failure);
        shared_preparation_cache(std::string(directory) + "/cache.sock", true);
        shared_preparation_cache(std::string(directory) + "/cache.sock", true, true);
        shared_preparation_cache(std::string(directory) + "/cache-disabled.sock", false);
        selected_variants_execute_through_service(std::string(directory) + "/profile-families.sock",
                                                  Kind::kPreemptive, false, true);
        selected_variants_execute_through_service(std::string(directory) + "/profile-families.sock",
                                                  Kind::kPreemptive, true, true, true);
        for (const auto kind :
             {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced})
            for (const bool naive : {false, true})
                selected_variants_execute_through_service(std::string(directory) + "/variants.sock",
                                                          kind, naive);
        immutable_module_files(std::string(directory) + "/module-ptx.sock", false);
        immutable_module_files(std::string(directory) + "/module-binary.sock", true);
        original_library_status_domains(std::string(directory) + "/library.sock");
        owned_function_controls(std::string(directory) + "/function-controls.sock");
        disconnect_waits_for_gpu(std::string(directory) + "/success.sock", CUDA_SUCCESS);
        disconnect_waits_for_gpu(std::string(directory) + "/error.sock", CUDA_ERROR_INVALID_VALUE);
        disconnect_waits_for_gpu(std::string(directory) + "/memory-success.sock", CUDA_SUCCESS,
                                 false, true);
        disconnect_waits_for_gpu(std::string(directory) + "/memory-error.sock",
                                 CUDA_ERROR_INVALID_VALUE, false, true);
        prepared_dependent_launch_waits_without_disconnect(std::string(directory) +
                                                           "/prepared.sock");
        independent_submission_releases_dependent(std::string(directory) + "/multiplex.sock",
                                                  MultiplexScenario::kSuccess);
        independent_submission_releases_dependent(std::string(directory) + "/multiplex-error.sock",
                                                  MultiplexScenario::kSdkError);
        independent_submission_releases_dependent(
            std::string(directory) + "/multiplex-disconnect.sock", MultiplexScenario::kDisconnect);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated unknown-physical-state fixture");
        if (child == 0) {
            try {
                disconnect_waits_for_gpu(std::string(directory) + "/uncertain.sock", CUDA_SUCCESS,
                                         true);
            } catch (...) {
                _exit(EXIT_FAILURE);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "async failure remains retained without retry or false native completion");
        // The child intentionally bypasses destructors; remove only its known socket inode.
        expect(::unlink((std::string(directory) + "/uncertain.sock").c_str()) == 0,
               "remove isolated stopped fixture socket");
        std::cout << "Execution service native-result and disconnect lifetime checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) {
        std::cerr << "Private service fixture directory was not empty\n";
        result = EXIT_FAILURE;
    }
    return result;
}
