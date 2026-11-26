#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/execution_socket.h"
#include "glimmer/control/execution_blas_protocol.h"
#include "glimmer/control/execution_function_protocol.h"
#include "glimmer/control/execution_jit_protocol.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/control/execution_process_memory_protocol.h"
#include "glimmer/control/execution_graph_protocol.h"
#include "driver_lookup.h"
#include "runtime_lookup.h"
#include "copy_audit.h"
#include "rpc_metrics.h"
#include "glimmer/transform/cuda_module_image.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>
#include <fatbinary_section.h>
#ifdef GLIMMER_HAS_NCCL
#include "glimmer/control/execution_nccl_protocol.h"
#include <nccl.h>
#endif
#ifdef GLIMMER_HAS_CUDNN
#include "glimmer/control/execution_dnn_protocol.h"
#include "glimmer/control/execution_dnn_legacy_protocol.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "../../internal/cudnn_sdk_profile.h"
#include <cudnn.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sys/uio.h>
#include <unistd.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif

namespace {
using namespace glimmer::control;
using glimmer::interceptor::execution::CopyAudit;
using glimmer::interceptor::execution::CopyRpcAudit;
using glimmer::interceptor::execution::RpcMetrics;
const pid_t kProcess = ::getpid();
constexpr std::size_t kMaxHandles = 16384;
constexpr std::size_t kMaxImageBytes = glimmer::transform::kMaxCudaModuleBytes;
constexpr std::size_t kMaxHostAllocations = 256;
constexpr std::size_t kMaxHostFunctions = 256;
constexpr std::size_t kMaxAsyncLaunches = 256;
constexpr std::size_t kAsyncReapBudget = 8;
static_assert(kMaxAsyncLaunches <= kExecutionMaxAsyncFrontiers);
constexpr std::size_t kMaxStagingBytes = std::size_t{64} * 1024 * 1024;
CUresult reject_constructor_only(std::string_view diagnostic) noexcept {
    // Actual invocation only: fixed text, no frontend state, loader or CUDA.
    // A failed diagnostic must not turn rejection into success or native work.
    while (!diagnostic.empty()) {
        const auto written = ::write(STDERR_FILENO, diagnostic.data(), diagnostic.size());
        if (written <= 0 || static_cast<std::size_t>(written) > diagnostic.size()) break;
        diagnostic.remove_prefix(static_cast<std::size_t>(written));
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}
using Exclusion = std::unique_lock<std::mutex>;
enum class Kind {
    kContext,
    kModule,
    kFunction,
    kStream,
    kEvent,
    kGraph,
    kGraphExecutable,
    kAllocation,
    kProcessAllocation,
    kMemoryPool,
    kProcessMemoryContext,
    kModuleGlobal,
    kHostContext,
    kPointerMetadataContext,
    kHostAllocation,
    kStagingContext,
    kPageableContext,
    kBlas,
    kLtMetadataContext,
    kLt,
    kRandom,
    kDnnMetadataContext,
    kDnn,
    kNcclContext
};
struct HostMapping {
    void* pointer{MAP_FAILED};
    std::size_t bytes{};
    ~HostMapping() {
        if (pointer != MAP_FAILED && ::munmap(pointer, bytes) != 0) {
            // Fixed best-effort ownership diagnostic: no loader/CUDA recursion.
            constexpr char message[] =
                "Glimmer: retaining failed local host unmap until process exit\n";
            const auto written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
            if (written != static_cast<ssize_t>(sizeof(message) - 1)) return;
        }
    }
    HostMapping() = default;
    HostMapping(const HostMapping&) = delete;
    HostMapping& operator=(const HostMapping&) = delete;
};
struct RuntimeImage;
struct RuntimeSymbolBinding {
    RuntimeImage* image{};
    std::size_t variable{};
};
struct RuntimeKernelBinding {
    RuntimeImage* image{};
    std::size_t kernel{};
};
#ifdef GLIMMER_HAS_NCCL
struct SourceCleanupBinding {
    ExecutionSourceContinuation origin;
    std::uint64_t helper_session{}, resource{}, next_sequence{1};
};
#endif
struct Entry {
    Kind kind{Kind::kContext};
    Entry* context{};
    std::uint64_t remote{}, parent{}, address{}, bytes{}, epoch{}, submissions{};
    std::uint64_t dnn_profile_epoch{};
    bool is_dnn_profile_binding{}, has_dnn_profile_mismatch{};
    std::uint64_t primary_references{}, active_calls{}, active_mutations{};
    // CPU retirement intent cancels only ordinary unaccepted pressure waits.
    // It does not seal in-flight mutation replies or grant native teardown.
    std::uint64_t retirement_requests{};
    // Transport reservations are distinct from resource pins and GPU ownership.
    std::uint64_t pending_requests{}, host_function_pins{};
    bool has_host_function_request{};
    CUresult host_function_error{CUDA_SUCCESS};
    unsigned int context_flags{};
    unsigned int host_flags{};
    unsigned int stream_flags{};
    int stream_priority{};
    std::uint64_t blas_stream{};
#ifdef GLIMMER_HAS_NCCL
    // Observed retained NCCL primary provenance, never a source-session ID.
    std::uint64_t nccl_primary_domain{};
    std::optional<SourceCleanupBinding> nccl_cleanup;
#endif
#ifdef GLIMMER_HAS_CUDNN
    std::uint64_t dnn_stream{};
    std::uintptr_t dnn_public_stream{};
    // Historical graph HANDLE output remains a value after context teardown.
    // This owner identity does not make a destroyed handle live again.
    std::uint64_t dnn_backend_owner{};
#endif
    unsigned int blas_pointer_mode{};
    std::vector<ExecutionWireArgument> arguments;
    std::unique_ptr<ExecutionSocketClient> connection;
    std::unique_ptr<HostMapping> host_mapping;
    std::unordered_map<const void*, RuntimeSymbolBinding> runtime_symbols;
    std::unordered_map<const void*, RuntimeKernelBinding> runtime_kernels;
    bool has_runtime_bindings{};
    bool is_alive{}, is_uncertain{}, is_primary{}, is_active{}, is_opening{}, is_closing{},
        is_retiring{};
};
// Exclusion is held at the first uncertainty boundary. Avoid logger callbacks,
// allocation and CUDA/loader recursion while reporting the exact transport cause.
void report_rpc_expiry(const Entry& context) noexcept {
    if (context.is_uncertain || !context.connection) return;
    const auto expired = context.connection->expired_request();
    if (!expired) return;
    std::array<char, 256> buffer{};
    constexpr std::string_view prefix = "Glimmer: execution RPC deadline expired; ";
    auto* cursor = std::copy(prefix.begin(), prefix.end(), buffer.begin());
    auto* const end = buffer.data() + buffer.size() - 1;
    constexpr std::array<std::string_view, 5> labels{
        "session=", ", sequence=", ", operation=", ", effective_operation=", ", timeout_ms="};
    const std::array<std::uint64_t, 5> values{
        expired->session, expired->sequence, static_cast<std::uint32_t>(expired->operation),
        static_cast<std::uint32_t>(expired->effective_operation), expired->timeout_ms};
    for (std::size_t index = 0; index < labels.size(); ++index) {
        if (labels[index].size() > static_cast<std::size_t>(end - cursor)) return;
        cursor = std::copy(labels[index].begin(), labels[index].end(), cursor);
        const auto converted = std::to_chars(cursor, end, values[index]);
        if (converted.ec != std::errc{}) return;
        cursor = converted.ptr;
    }
    *cursor++ = '\n';
    const auto saved_errno = errno;
    std::string_view message(buffer.data(), static_cast<std::size_t>(cursor - buffer.data()));
    while (!message.empty()) {
        const auto written = ::write(STDERR_FILENO, message.data(), message.size());
        if (written <= 0 || static_cast<std::size_t>(written) > message.size()) break;
        message.remove_prefix(static_cast<std::size_t>(written));
    }
    errno = saved_errno;
}
// CPU notification ownership is independent of CUDA/resource lifetime. A weak
// nonrecycled caller identity prevents another thread consuming these errors.
struct AsyncLaunch {
    Entry* context{};
    std::weak_ptr<const int> caller;
    std::uint64_t epoch{}, stream{}, sequence{}, frontier{};
    CUresult failure{CUDA_SUCCESS};
    CUresult synchronized_failure{CUDA_SUCCESS};
    bool is_runtime{}, is_nonblocking{}, is_preparing{true}, is_polling{}, is_notified{},
        has_retirement_proof{};
};
// State exclusion is held; only the originating Runtime synchronization may
// pre-consume this matching notification. Other statuses/callers stay independent.
void cache_async_failure(AsyncLaunch& launch, CUresult status) noexcept {
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY && launch.failure == CUDA_SUCCESS)
        launch.failure = status;
    if (launch.failure != CUDA_SUCCESS && launch.failure == launch.synchronized_failure)
        launch.is_notified = true;
}
struct LtEntry {
    Entry* context{};
    std::uint64_t token{}, remote{}, epoch{}, active_calls{};
    ExecutionLtObjectKind kind{};
    std::uint32_t scale_type{CUDA_R_32F}, pointer_mode{};
    bool is_alive{}, is_retiring{}, is_uncertain{};
};
#ifdef GLIMMER_HAS_NCCL
struct NcclEntry {
    std::uint64_t token{}, active_calls{};
    ExecutionNcclCommunicator remote;
    bool is_alive{}, is_uncertain{}, is_retiring{};
};
struct NcclCaller {
    std::array<std::uint64_t, 2> thread{};
    std::uint64_t sequence{1}, ordinary_sequence{1}, group{}, depth{};
    bool is_uncertain{};
};
enum class NcclOutputKind { kUniqueId, kCommunicator, kAsyncError };
enum class NcclOutputPhase { kPrepared, kInFlight, kDeferred, kComplete, kUncertain };
struct NcclOutput {
    ExecutionNcclScope scope;
    NcclOutputKind kind{};
    void* destination{};
    std::shared_ptr<NcclEntry> communicator;
    NcclOutputPhase phase{NcclOutputPhase::kPrepared};
};
thread_local NcclCaller* nccl_caller{};
#endif
#ifdef GLIMMER_HAS_CUDNN
struct DnnEntry {
    Entry* context{};
    std::uint64_t token{}, remote{}, epoch{}, active_calls{};
    ExecutionDnnObjectKind kind{};
    std::uint32_t backend_kind{};
    bool is_alive{}, is_retiring{}, is_uncertain{};
};
struct DnnBackendPointerSlot {
    Entry* owner{};
    std::uint64_t epoch{}, id{};
    std::uintptr_t address{};
};
#endif
struct StagingAllocation {
    Entry* origin{};
    Entry* owner{};
    std::uint64_t epoch{}, remote{}, release_sequence{}, idle_sequence{};
    std::size_t capacity_bytes{};
    HostMapping mapping;
#ifdef GLIMMER_HAS_NCCL
    std::optional<SourceCleanupBinding> nccl_cleanup;
#endif
    bool is_preparing{true}, is_uncertain{}, can_reuse{}, is_idle{}, should_release{};
};
struct UnifiedEndpoint {
    ExecutionCopyEndpoint wire;
    Entry* allocation{};
    Entry* parent{};
    std::uintptr_t address{};
};
enum class HostFunctionPhase { kPreparing, kWaiting, kRunning, kFinishing, kUncertain };
struct HostFunction {
    Entry* origin{};
    Entry* stream{};
    std::uint64_t epoch{}, remote{}, sequence{}, enqueue_sequence{};
    cudaHostFn_t function{};
    void* data{};
    std::vector<std::byte> query_payload, finish_payload, exception_payload;
    ExecutionSourceContinuation continuation;
    HostFunctionPhase phase{HostFunctionPhase::kPreparing};
    bool has_normal_return{true};
};
struct RuntimeKernel {
    const void* host{};
    std::string name;
};
struct RuntimeFunction {
    const void* host{};
    Entry* function{};
};
struct RuntimeResolvedSymbol {
    const void* host{};
    Entry* global{};
};
struct RuntimeVariable {
    const void* host{};
    std::string name;
    std::size_t bytes{};
    bool is_external{}, is_constant{}, is_global{};
};
struct RuntimeBinding {
    Entry* context{};
    Entry* module{};
    std::uint64_t epoch{};
    std::vector<RuntimeFunction> functions;
    std::vector<RuntimeResolvedSymbol> symbols;
    bool is_loading{}, is_preparing{};
};
struct RuntimeImage {
    std::vector<std::byte> bytes;
    std::vector<RuntimeKernel> kernels;
    std::vector<RuntimeVariable> variables;
    std::vector<std::unique_ptr<RuntimeBinding>> bindings;
    CUresult error{CUDA_SUCCESS};
    std::size_t retained_bytes{}, metadata_bytes{}, pending_lookups{};
    bool is_finalized{}, is_closing{}, is_alive{true};
};
thread_local std::vector<Entry*> context_stack;
thread_local bool is_entered{};
thread_local bool is_host_function_active{};
thread_local cudaError_t runtime_last_error{cudaSuccess};
thread_local std::shared_ptr<const int> async_caller;
thread_local cudaStreamCaptureMode runtime_capture_mode{cudaStreamCaptureModeGlobal};
struct DefaultStreamBinding {
    Entry* context{};
    std::uint64_t epoch{};
    Entry* stream{};
};
thread_local std::vector<DefaultStreamBinding> default_stream_bindings;

CUresult context_flags_status(unsigned int flags) noexcept {
    const auto scheduling = flags & CU_CTX_SCHED_MASK;
    if ((flags & ~static_cast<unsigned int>(CU_CTX_FLAGS_MASK)) ||
        (scheduling != CU_CTX_SCHED_AUTO && scheduling != CU_CTX_SCHED_SPIN &&
         scheduling != CU_CTX_SCHED_YIELD && scheduling != CU_CTX_SCHED_BLOCKING_SYNC))
        return CUDA_ERROR_INVALID_VALUE;
    if (flags & (CU_CTX_COREDUMP_ENABLE | CU_CTX_USER_COREDUMP_ENABLE))
        return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_SUCCESS;
}

bool self_copy(void* destination, const void* source, std::size_t bytes, bool is_write) noexcept {
    if (bytes == 0) return true;
    if (!destination || !source || bytes > static_cast<std::size_t>(SSIZE_MAX)) return false;
    iovec local{is_write ? const_cast<void*>(source) : destination, bytes};
    iovec remote{is_write ? destination : const_cast<void*>(source), bytes};
    const auto copied = is_write ? ::process_vm_writev(kProcess, &local, 1, &remote, 1, 0)
                                 : ::process_vm_readv(kProcess, &local, 1, &remote, 1, 0);
    return copied >= 0 && static_cast<std::size_t>(copied) == bytes;
}
bool writable_host_range(void* pointer, std::size_t bytes,
                         glimmer::interceptor::execution::RpcMetrics::HostTrace* trace = nullptr,
                         off_t* hint = nullptr) noexcept {
    using Stage = glimmer::interceptor::execution::RpcMetrics::HostStage;
    if (!bytes) return true;
    auto cursor = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || bytes > UINTPTR_MAX - cursor) return false;
    const auto target = cursor + bytes;
    ExecutionOwnedDescriptor maps(::open("/proc/self/maps", O_RDONLY | O_CLOEXEC));
    struct EnterCleanup {
        glimmer::interceptor::execution::RpcMetrics::HostTrace* trace;
        ~EnterCleanup() {
            if (trace) trace->transition(Stage::kCloseOther);
        }
    } enter_cleanup{trace};
    if (trace) trace->transition(Stage::kParse);
    if (maps.get() < 0) return false;
    std::array<char, 4096> buffer{};
    // This is a search position, never cached permission or mapping identity.
    // pread leaves the descriptor position at zero for the original fallback.
    if (hint && *hint > 0) {
        if (trace) trace->transition(Stage::kRead);
        const auto count = ::pread(maps.get(), buffer.data(), buffer.size(), *hint - 1);
        if (trace) {
            trace->transition(Stage::kParse);
            trace->read_result(count);
        }
        if (count > 1 && buffer[0] == '\n') {
            const char* begin = buffer.data() + 1;
            const char* limit = buffer.data() + count;
            const auto* end = std::find(begin, limit, '\n');
            if (end != limit) {
                std::uintptr_t first{}, last{};
                const auto start = std::from_chars(begin, end, first, 16);
                if (start.ec == std::errc{} && start.ptr != end && *start.ptr == '-') {
                    const auto finish = std::from_chars(start.ptr + 1, end, last, 16);
                    if (finish.ec == std::errc{} && end - finish.ptr >= 5 && *finish.ptr == ' ' &&
                        first < last && finish.ptr[2] == 'w' && first <= cursor && target <= last) {
                        if (trace) trace->hint_result(true);
                        return true;
                    }
                }
            }
        }
        if (trace) trace->hint_result(false);
    }
    std::array<char, 96> prefix{};
    std::size_t used{};
    const auto original = cursor;
    std::uint64_t offset{}, line{};
    for (;;) {
        if (trace) trace->transition(Stage::kRead);
        const auto count = ::read(maps.get(), buffer.data(), buffer.size());
        if (trace) {
            trace->transition(Stage::kParse);
            trace->read_result(count);
        }
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        for (ssize_t index = 0; index < count; ++index) {
            const auto* begin = buffer.data() + index;
            const auto* newline = static_cast<const char*>(
                std::memchr(begin, '\n', static_cast<std::size_t>(count - index)));
            const auto length = newline ? static_cast<std::size_t>(newline - begin)
                                        : static_cast<std::size_t>(count - index);
            const auto copied = std::min(length, prefix.size() - used);
            std::memcpy(prefix.data() + used, begin, copied);
            used += copied;
            index += static_cast<ssize_t>(length);
            if (!newline) break;
            const auto* end = prefix.data() + used;
            std::uintptr_t first{}, last{};
            const auto start = std::from_chars(prefix.data(), end, first, 16);
            if (start.ec != std::errc{} || start.ptr == end || *start.ptr != '-') return false;
            const auto finish = std::from_chars(start.ptr + 1, end, last, 16);
            if (finish.ec != std::errc{} || end - finish.ptr < 5 || *finish.ptr != ' ' ||
                first >= last)
                return false;
            // Only the header is retained; arbitrarily long pathname text
            // cannot grow allocation or be mistaken for another VMA header.
            used = 0;
            const auto current_line = line;
            if (hint) line = offset + static_cast<std::uint64_t>(index) + 1;
            if (last <= cursor) continue;
            if (first > cursor || finish.ptr[2] != 'w') return false;
            cursor = std::min(last, target);
            if (cursor == target) {
                if (hint && first <= original &&
                    current_line <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
                    *hint = static_cast<off_t>(current_line);
                return true;
            }
        }
        if (hint) offset += static_cast<std::uint64_t>(count);
    }
}
const void* pointer_slot(const void* base, std::size_t index) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    if (!base || index > (UINTPTR_MAX - address) / sizeof(void*)) return nullptr;
    // Checked numeric arithmetic avoids pointer UB before a self-process read.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<const void*>(address + index * sizeof(void*));
}
bool copy_launch_arguments(std::span<const ExecutionWireArgument> layout, void** arguments,
                           std::span<std::byte> output) noexcept {
    constexpr std::size_t batch_size = 64;
    for (std::size_t first = 0; first < layout.size();) {
        const auto count = std::min(batch_size, layout.size() - first);
        const auto* slots = pointer_slot(arguments, first);
        const auto pointer_bytes = count * sizeof(void*);
        if (!slots || pointer_bytes > UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(slots))
            return false;
        std::array<void*, batch_size> pointers;
        if (!self_copy(pointers.data(), slots, pointer_bytes, false)) return false;
        std::array<iovec, batch_size> local, remote;
        std::size_t used{}, expected{};
        for (std::size_t index = 0; index < count; ++index) {
            const auto& argument = layout[first + index];
            if (argument.offset > output.size() ||
                argument.bytes > output.size() - argument.offset ||
                argument.bytes > static_cast<std::size_t>(SSIZE_MAX) - expected)
                return false;
            if (!argument.bytes) continue;
            if (!pointers[index]) return false;
            const auto bytes = static_cast<std::size_t>(argument.bytes);
            local[used] = {output.data() + argument.offset, bytes};
            remote[used++] = {pointers[index], bytes};
            expected += bytes;
        }
        if (used) {
            const auto copied =
                ::process_vm_readv(kProcess, local.data(), used, remote.data(), used, 0);
            if (copied < 0 || static_cast<std::size_t>(copied) != expected) return false;
        }
        first += count;
    }
    return true;
}
template <typename Value>
bool writable(Value* output) noexcept {
    Value old{};
    const auto bytes = std::as_writable_bytes(std::span{&old, 1}).size();
    return self_copy(&old, output, bytes, false) && self_copy(output, &old, bytes, true);
}
template <typename Value>
bool publish(Value* output, Value value) noexcept {
    return self_copy(output, &value, std::as_bytes(std::span{&value, 1}).size(), true);
}
std::optional<std::string> owned_string(const char* input, std::size_t limit) {
    if (!input) return std::nullopt;
    std::string output;
    // Stop at page boundaries so a valid terminator never requires reading an
    // inaccessible following page. Reads only this process, never GPU memory.
    constexpr std::size_t chunk_bytes = 256;
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return std::nullopt;
    while (output.size() < limit) {
        const auto offset = output.size();
        const auto address = reinterpret_cast<std::uintptr_t>(input);
        if (offset > std::numeric_limits<std::uintptr_t>::max() - address) return std::nullopt;
        const auto current = address + offset;
        const auto page_left = static_cast<std::size_t>(page_size) - current % page_size;
        const auto count = std::min({chunk_bytes, page_left, limit - offset});
        std::array<char, chunk_bytes> buffer{};
        // The numeric address is checked above, then read through the kernel.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        if (!self_copy(buffer.data(), reinterpret_cast<const void*>(current), count, false))
            return std::nullopt;
        const auto end = std::find(buffer.begin(), buffer.begin() + count, '\0');
        output.append(buffer.begin(), end);
        if (end != buffer.begin() + count) return output;
    }
    return std::nullopt;
}
std::optional<std::uint64_t> environment_number(const char* name) {
    const auto value = owned_string(std::getenv(name), 32);
    if (!value || value->empty()) return std::nullopt;
    std::uint64_t number{};
    const auto parsed = std::from_chars(value->data(), value->data() + value->size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value->data() + value->size())
        return std::nullopt;
    return number;
}
bool yield_frontend(Exclusion& exclusion) {
    exclusion.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds{50});
    if (::getpid() != kProcess) return false;
    exclusion.lock();
    return true;
}
bool yield_admission(Exclusion& exclusion, unsigned int& backoff_us) {
    const auto delay_us = backoff_us;
    backoff_us = std::min(backoff_us * 2U, 1000U);
    exclusion.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds{delay_us});
    if (::getpid() != kProcess) return false;
    exclusion.lock();
    return true;
}
bool wait_frontend_reply(Exclusion& exclusion, ExecutionSocketClient& connection,
                         RpcMetrics::Scope& observation) {
    // Snapshot while serialized; wait without blocking another caller's I/O,
    // callback or error progress. Only the exact receipt can finish the RPC.
    const auto readiness = connection.readiness();
    RpcMetrics::Interval wait{}, fallback{};
    exclusion.unlock();
    bool should_sleep = !readiness;
    if (readiness) {
        wait.begin = observation.phase_clock();
        should_sleep = !readiness->wait_for_ns(50000);
        wait.end = observation.phase_clock();
    }
    if (should_sleep) {
        fallback.begin = observation.phase_clock();
        std::this_thread::sleep_for(std::chrono::microseconds{50});
        fallback.end = observation.phase_clock();
    }
    if (::getpid() != kProcess) return false;
    const auto lock_begin = observation.phase_clock();
    exclusion.lock();
    const auto lock_end = observation.phase_clock();
    if (readiness) observation.phase(RpcMetrics::Phase::kWaitCall, wait);
    if (should_sleep) observation.phase(RpcMetrics::Phase::kFallbackSleep, fallback);
    observation.phase(RpcMetrics::Phase::kMutexReacquire, {lock_begin, lock_end});
    return true;
}
struct CounterPin {
    std::uint64_t& count;
    Exclusion& exclusion;
    CounterPin(std::uint64_t& value, Exclusion& lock) : count(value), exclusion(lock) {
        ++count;
    }
    CounterPin(const CounterPin&) = delete;
    CounterPin& operator=(const CounterPin&) = delete;
    ~CounterPin() {
        if (::getpid() == kProcess && exclusion.owns_lock()) --count;
    }
};
struct FlagPin {
    bool& flag;
    Exclusion& exclusion;
    FlagPin(bool& value, Exclusion& lock) : flag(value), exclusion(lock) {
        flag = true;
    }
    FlagPin(const FlagPin&) = delete;
    FlagPin& operator=(const FlagPin&) = delete;
    ~FlagPin() {
        if (::getpid() == kProcess && exclusion.owns_lock()) flag = false;
    }
};
ExecutionWireReply local_failure(CUresult error) {
    ExecutionWireReply result;
    result.status = error;
    return result;
}
struct GraphCapture {
    Entry* context{};
    Entry* stream{};
    std::uint64_t epoch{}, caller{};
    cudaStreamCaptureMode mode{cudaStreamCaptureModeGlobal};
    cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
    bool is_transition{};
};
thread_local std::uint64_t graph_caller{};
thread_local bool is_graph_preparation{};
struct State;
std::optional<ExecutionWireReply> graph_rpc_gate(State&, Exclusion&, Entry&, ExecutionOpcode,
                                                 bool is_observation);
bool graph_quarantine_receipt(ExecutionOpcode, const ExecutionWireReply&) noexcept;
bool graph_capture_owner(const State&, const Entry*, const Entry* = nullptr) noexcept;
CUresult capture_graph_command(State&, Exclusion&, Entry&, Entry*, ExecutionGraphAction,
                               std::span<const std::byte>);
CUresult reject_graph_capture(State&, Exclusion&);
struct State {
    std::mutex mutex;
    std::unique_ptr<glimmer::interceptor::execution::RpcMetrics> rpc_metrics;
    bool should_seek_host_range{};
    std::vector<std::unique_ptr<Entry>> entries;
    std::vector<std::shared_ptr<AsyncLaunch>> async_launches;
    std::size_t async_reap_cursor{};
    std::optional<ExecutionSocketClientConfig> config;
    Entry* primary{};
    Entry* host_context{};
    Entry* pointer_metadata_context{};
    Entry* process_memory_context{};
    Entry* process_memory_pool_spare{};
    Entry* staging_context{};
    Entry* pageable_context{};
    Entry* lt_metadata_context{};
    std::vector<std::shared_ptr<LtEntry>> lt_entries;
    std::uint64_t next_lt_token{0x4c54000000000001ULL};
    bool has_lt_tokens{true};
#ifdef GLIMMER_HAS_NCCL
    Entry* nccl_context{};
    std::vector<std::shared_ptr<NcclEntry>> nccl_entries;
    std::vector<std::unique_ptr<NcclCaller>> nccl_callers;
    std::vector<std::shared_ptr<NcclOutput>> nccl_outputs;
    std::uint64_t next_nccl_token{0x4e43000000000001ULL}, next_nccl_caller{1}, next_nccl_group{1};
#endif
#ifdef GLIMMER_HAS_CUDNN
    Entry* dnn_metadata_context{};
    std::vector<std::shared_ptr<DnnEntry>> dnn_entries;
    // Stable caller-address bindings only. Neither allocation leases nor host
    // pointees are captured until an authenticated Execute preparation asks.
    std::vector<DnnBackendPointerSlot> dnn_backend_pointer_slots;
    std::unordered_map<std::uintptr_t, std::uint64_t> dnn_backend_pointer_index;
    std::uint64_t next_dnn_token{0x444e000000000001ULL};
    bool has_dnn_tokens{true};
#endif
    std::vector<Entry*> host_allocations;
    std::vector<std::unique_ptr<StagingAllocation>> staging_allocations;
    std::condition_variable staging_wake;
    std::thread staging_progress;
    bool should_stop_staging{};
    bool should_cache_staging{true};
    std::vector<std::unique_ptr<HostFunction>> host_functions;
    std::condition_variable host_function_wake;
    std::thread host_function_progress;
    bool should_stop_host_functions{};
    std::condition_variable async_status_wake;
    std::thread async_status_progress;
    bool should_stop_async_status{}, has_async_status_hint{};
    std::vector<std::unique_ptr<RuntimeImage>> runtime_images;
    std::size_t runtime_image_bytes{}, runtime_metadata_bytes{};
    bool runtime_owns_primary{};
    std::optional<GraphCapture> graph_capture;
    std::uint64_t next_graph_caller{1};
    bool has_driver_initialization{};
    State() {
        entries.reserve(kMaxHandles);
        async_launches.reserve(kMaxAsyncLaunches);
        lt_entries.reserve(kMaxHandles);
#ifdef GLIMMER_HAS_NCCL
        nccl_entries.reserve(kMaxHandles);
        nccl_callers.reserve(kMaxHandles);
        nccl_outputs.reserve(kMaxHandles);
#endif
#ifdef GLIMMER_HAS_CUDNN
        dnn_entries.reserve(kMaxHandles);
#endif
        host_allocations.reserve(kMaxHostAllocations);
        staging_allocations.reserve(kMaxHostAllocations);
        host_functions.reserve(kMaxHostFunctions);
        if (const auto* setting = std::getenv("GLIMMER_EXECUTION_STAGING_CACHE")) {
            const auto value = owned_string(setting, 2);
            if (!value || (*value != "0" && *value != "1")) return;
            should_cache_staging = *value == "1";
        }
        const auto path = owned_string(std::getenv("GLIMMER_EXECUTION_SOCKET"), 4096);
        const auto pid = environment_number("GLIMMER_EXECUTION_SERVER_PID");
        const auto start = environment_number("GLIMMER_EXECUTION_SERVER_START");
        std::optional<std::uint32_t> preparation_timeout;
        if (std::getenv("GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS")) {
            const auto parsed = environment_number("GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS");
            if (!parsed || !*parsed || *parsed > kExecutionMaxPreparationTimeoutMs) return;
            preparation_timeout = static_cast<std::uint32_t>(*parsed);
        }
        if (path && pid && *pid > 0 && *pid <= UINT32_MAX && start && *start > 0)
            config = ExecutionSocketClientConfig{
                *path,
                {static_cast<std::uint32_t>(*pid), static_cast<std::uint32_t>(::geteuid()), *start},
                30000};
        if (config) config->preparation_timeout_ms = preparation_timeout;
        const auto saved_errno = errno;
        std::array<char, 2> seek_setting{};
        const auto* seek_text = std::getenv("GLIMMER_EXECUTION_HOST_RANGE_SEEK");
        should_seek_host_range =
            seek_text && self_copy(seek_setting.data(), seek_text, seek_setting.size(), false) &&
            seek_setting[0] == '1' && seek_setting[1] == '\0';
        std::array<char, 2> copy_audit_setting{};
        const auto* copy_audit_text = std::getenv("GLIMMER_EXECUTION_COPY_AUDIT");
        glimmer::interceptor::execution::configure_copy_audit(
            copy_audit_text &&
            self_copy(copy_audit_setting.data(), copy_audit_text, copy_audit_setting.size(),
                      false) &&
            copy_audit_setting[0] == '1' && copy_audit_setting[1] == '\0');
        errno = saved_errno;
        if (const auto* output = std::getenv("GLIMMER_EXECUTION_RPC_OUTPUT_DIR")) {
            const auto saved = errno;
            try {
                const auto directory = owned_string(output, 4096);
                if (directory &&
                    glimmer::interceptor::execution::RpcMetrics::valid_directory(*directory)) {
                    const auto* detail_text = std::getenv("GLIMMER_EXECUTION_HOST_RANGE_DETAIL");
                    const auto detail = detail_text ? owned_string(detail_text, 2) : std::nullopt;
                    using Metrics = glimmer::interceptor::execution::RpcMetrics;
                    rpc_metrics = std::make_unique<Metrics>(*directory, Metrics::capacity_limit,
                                                            detail && *detail == "1");
                }
            } catch (...) {
                // Missing observation never changes frontend configuration.
            }
            errno = saved;
        }
    }
    ~State() {
        {
            std::lock_guard lock(mutex);
            should_stop_staging = true;
            should_stop_host_functions = true;
            should_stop_async_status = true;
        }
        staging_wake.notify_one();
        host_function_wake.notify_one();
        async_status_wake.notify_one();
        if (async_status_progress.joinable()) {
            if (async_status_progress.get_id() == std::this_thread::get_id())
                async_status_progress.detach();  // A user allocation handler may exit.
            else
                async_status_progress.join();
        }
        if (staging_progress.joinable()) staging_progress.join();
        if (host_function_progress.joinable()) {
            if (host_function_progress.get_id() == std::this_thread::get_id())
                host_function_progress.detach();  // A callback may terminate its process.
            else
                host_function_progress.join();
        }
        if (rpc_metrics) {
            {
                std::lock_guard lock(mutex);
                rpc_metrics->seal();
            }
            (void)rpc_metrics->publish();
        }
    }
    void ensure_staging_progress() {
        if (staging_progress.joinable()) return;
        staging_progress = std::thread([this] {
            // CPU transport only. This worker never binds a CUDA context,
            // invokes the SDK or touches application host memory. It keeps
            // asynchronous cleanup progressing even while the caller is idle.
            std::unique_lock lock(mutex);
            for (;;) {
                staging_wake.wait(lock, [this] {
                    return should_stop_staging ||
                           std::ranges::any_of(staging_allocations, [](const auto& stage) {
                               const auto* owner = stage->owner;
                               return owner && owner->is_active && !owner->is_opening &&
                                      !owner->is_uncertain && !stage->is_preparing &&
                                      !stage->is_uncertain &&
                                      (!stage->is_idle || stage->should_release);
                           });
                });
                if (should_stop_staging) return;
                poll_staging();
                staging_wake.wait_for(lock, std::chrono::milliseconds{2},
                                      [this] { return should_stop_staging; });
                if (should_stop_staging) return;
            }
        });
    }
    Entry* find(const void* handle, Kind kind) const noexcept {
        const auto found = std::ranges::find_if(entries, [&](const auto& value) {
            return value.get() == handle && value->kind == kind && value->is_alive;
        });
        return found == entries.end() ? nullptr : found->get();
    }
    Entry* context() const noexcept {
        auto* current = context_stack.empty() ? nullptr : context_stack.back();
        return current && current->is_alive && !current->is_uncertain ? current : nullptr;
    }
    Entry* allocate_entry(Kind kind, Entry* context) {
        if (entries.size() == kMaxHandles) return nullptr;
        auto value = std::make_unique<Entry>();
        value->kind = kind;
        value->context = context;
        auto* result = value.get();
        entries.push_back(std::move(value));
        return result;
    }
    CUresult await_context_transition(Exclusion& exclusion, Entry& context) {
        while (context.is_opening || context.is_closing) {
            if (context.is_uncertain) return CUDA_ERROR_UNKNOWN;
            if (context.is_closing && !context.is_alive) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        }
        return context.is_uncertain ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
    }
    std::optional<ExecutionSourceContinuation> source_continuation(const Entry& target) const {
#ifdef GLIMMER_HAS_NCCL
        if (target.kind == Kind::kNcclContext || !nccl_caller || !nccl_caller->depth)
            return std::nullopt;
        const auto* origin = target.kind == Kind::kContext ? &target : context();
        return ExecutionSourceContinuation{
            0,
            nccl_caller->thread,
            nccl_caller->group,
            nccl_caller->ordinary_sequence,
            origin && origin->is_active && origin->connection ? origin->connection->session() : 0,
            0};
#else
        (void)target;
        return std::nullopt;
#endif
    }
    bool source_continuation_healthy(
        const std::optional<ExecutionSourceContinuation>& scope) const {
#ifdef GLIMMER_HAS_NCCL
        return !scope || (!nccl_caller->is_uncertain && scope->ordinary_sequence && scope->group);
#else
        return !scope;
#endif
    }
    void submitted_source_continuation(const std::optional<ExecutionSourceContinuation>& scope) {
#ifdef GLIMMER_HAS_NCCL
        if (scope && !scope->flags)
            nccl_caller->ordinary_sequence =
                scope->ordinary_sequence == UINT64_MAX ? 0 : scope->ordinary_sequence + 1;
#else
        (void)scope;
#endif
    }
    void quarantine_source_continuation(const std::optional<ExecutionSourceContinuation>& scope) {
#ifdef GLIMMER_HAS_NCCL
        if (!scope) return;
        const auto caller = std::ranges::find_if(nccl_callers, [&](const auto& candidate) {
            return candidate->thread == scope->caller;
        });
        if (caller != nccl_callers.end()) (*caller)->is_uncertain = true;
#else
        (void)scope;
#endif
    }
    CUresult open_context(
        Exclusion& exclusion, Entry& context, unsigned int flags,
        ExecutionContextPurpose primary_purpose = ExecutionContextPurpose::kProcessPrimary) {
        const auto continuation = source_continuation(context);
        if (!source_continuation_healthy(continuation)) return CUDA_ERROR_UNKNOWN;
        const auto waited = await_context_transition(exclusion, context);
        if (waited != CUDA_SUCCESS) return waited;
        if (context.is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context.is_active) return CUDA_SUCCESS;
        if (graph_capture) return reject_graph_capture(*this, exclusion);
        if (context.is_uncertain || context.epoch == UINT64_MAX) return CUDA_ERROR_UNKNOWN;
        if (context.is_primary) flags = context.context_flags | CU_CTX_MAP_HOST;
        ExecutionPayloadWriter bootstrap;
        std::vector<std::byte> bytes;
        if (context.kind == Kind::kHostContext || context.kind == Kind::kStagingContext ||
            context.kind == Kind::kPageableContext || context.kind == Kind::kLtMetadataContext ||
            context.kind == Kind::kDnnMetadataContext ||
            context.kind == Kind::kPointerMetadataContext ||
            context.kind == Kind::kProcessMemoryContext || context.kind == Kind::kNcclContext) {
            const auto role =
#ifdef GLIMMER_HAS_NCCL
                context.kind == Kind::kNcclContext ? kExecutionNcclRole :
#endif
                context.kind == Kind::kLtMetadataContext        ? kExecutionLtMetadataRole
                : context.kind == Kind::kPointerMetadataContext ? kExecutionPointerMetadataRole
                : context.kind == Kind::kProcessMemoryContext   ? kExecutionProcessMemoryRole
                : context.kind == Kind::kDnnMetadataContext     ? kExecutionDnnMetadataRole
                : context.kind == Kind::kStagingContext         ? kExecutionStagingOwnerRole
                : context.kind == Kind::kPageableContext        ? kExecutionPageableOwnerRole
                                                                : kExecutionHostOwnerRole;
            if (!bootstrap.write_u32(role) || !bootstrap.write_u32(flags))
                return CUDA_ERROR_OUT_OF_MEMORY;
            bytes = bootstrap.take();
        } else {
            const auto encoded = encode_execution_context_bootstrap(
                {context.is_primary ? primary_purpose : ExecutionContextPurpose::kExplicitUser,
                 flags, context.epoch + 1});
            if (!encoded) return CUDA_ERROR_INVALID_VALUE;
            bytes.assign(encoded->begin(), encoded->end());
        }
        context.connection = std::make_unique<ExecutionSocketClient>(*config);
        FlagPin opening(context.is_opening, exclusion);
        const auto submitted = context.connection->submit_open(bytes, continuation);
        if (submitted.status == ExecutionSocketSubmitStatus::kAccepted)
            submitted_source_continuation(continuation);
        std::optional<ExecutionWireReply> result;
        while (submitted.status == ExecutionSocketSubmitStatus::kAccepted) {
            result = context.connection->take_reply(submitted.sequence);
            if (result) break;
            const bool can_progress = context.connection->poll_for(0);
            result = context.connection->take_reply(submitted.sequence);
            if (result || !can_progress) break;
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!result || result->status != 0 ||
            result->disposition != ExecutionWireDisposition::kComplete) {
            const auto ownership =
                result ? decode_execution_bootstrap_ownership(result->data) : std::nullopt;
            context.is_uncertain =
                !result || result->status == 0 || ownership != ExecutionBootstrapOwnership::kNone;
            if (context.is_uncertain) quarantine_source_continuation(continuation);
            if (!context.is_uncertain) context.connection.reset();
            return result && result->status ? static_cast<CUresult>(result->status)
                                            : CUDA_ERROR_UNKNOWN;
        }
        context.is_active = true;
        ++context.epoch;
        context.dnn_profile_epoch = 0;
        context.has_dnn_profile_mismatch = false;
        // A previous generation's known no-callback error remains observable
        // through teardown, but cannot contaminate a successfully opened owner.
        context.host_function_error = CUDA_SUCCESS;
        if (context.kind == Kind::kContext) has_driver_initialization = true;
        return CUDA_SUCCESS;
    }
    Entry* primary_entry() {
        if (!primary) {
            primary = allocate_entry(Kind::kContext, nullptr);
            if (primary) {
                primary->context = primary;
                primary->is_primary = primary->is_alive = true;
            }
        }
        return primary;
    }
    std::optional<ExecutionWireReply> rpc(
        Exclusion& exclusion, Entry& context, ExecutionOpcode operation,
        std::span<const std::byte> bytes, int borrowed_descriptor = -1,
        bool* was_submitted = nullptr,
        const std::optional<ExecutionSourceContinuation>& explicit_continuation = std::nullopt,
        ExecutionSourceContinuation* submitted_continuation = nullptr,
        std::uint64_t* submitted_sequence = nullptr, bool is_async_observation = false) {
        glimmer::interceptor::execution::RpcMetrics::Scope rpc_observation(
            rpc_metrics.get(), exclusion, {}, false,
            {context.connection ? context.connection->session() : 0,
             static_cast<std::uint32_t>(operation)});
        CopyRpcAudit copy_audit(static_cast<std::uint32_t>(operation),
                                context.connection ? context.connection->session() : 0);
        if (was_submitted) *was_submitted = false;
        if (submitted_sequence) *submitted_sequence = 0;
        if (submitted_continuation) *submitted_continuation = {};
        if (auto denied =
                graph_rpc_gate(*this, exclusion, context, operation, is_async_observation))
            return denied;
        if (is_async_observation && operation != ExecutionOpcode::kQueryFrontier &&
            operation != ExecutionOpcode::kReleaseFrontier &&
            operation != ExecutionOpcode::kQueryAsyncFrontiers &&
            operation != ExecutionOpcode::kReleaseAsyncFrontiers)
            return local_failure(CUDA_ERROR_INVALID_VALUE);
        if (operation == ExecutionOpcode::kBindDnnProfile && explicit_continuation)
            return local_failure(CUDA_ERROR_INVALID_VALUE);
        // CPU-only version agreement is not SDK execution or source authority.
        // The subsequent DNN call retains its original exact source scope.
        auto continuation = (is_async_observation || operation == ExecutionOpcode::kBindDnnProfile)
                                ? std::optional<ExecutionSourceContinuation>{}
                            : explicit_continuation ? explicit_continuation
                                                    : source_continuation(context);
        if (explicit_continuation &&
            explicit_continuation->flags != kExecutionSourceContinuationCleanup)
            return local_failure(CUDA_ERROR_INVALID_VALUE);
        if (!explicit_continuation && !source_continuation_healthy(continuation))
            return local_failure(CUDA_ERROR_UNKNOWN);
        if (context.is_uncertain || !context.connection) return std::nullopt;
        if (context.is_closing && !is_async_observation)
            return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        const auto epoch = context.epoch;
#ifdef GLIMMER_HAS_CUDNN
        if (execution_opcode_is_dnn(operation)) {
            if (borrowed_descriptor >= 0) return local_failure(CUDA_ERROR_NOT_SUPPORTED);
            while (context.is_dnn_profile_binding) {
                if (!yield_frontend(exclusion)) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
                if (!context.is_alive || context.epoch != epoch || context.is_closing)
                    return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
                if (context.is_uncertain) return std::nullopt;
            }
            if (context.has_dnn_profile_mismatch) return local_failure(CUDA_ERROR_NOT_SUPPORTED);
            if (context.dnn_profile_epoch != epoch) {
                FlagPin binding(context.is_dnn_profile_binding, exclusion);
                ExecutionPayloadWriter profile;
                if (!write_execution_dnn_profile(profile,
                                                 glimmer::internal::cudnn_profile::kHeaderVersion))
                    return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
                const auto value = profile.take();
                auto reply = rpc(exclusion, context, ExecutionOpcode::kBindDnnProfile, value);
                if (!reply) return std::nullopt;
                const bool empty_identity = !reply->id && !reply->address &&
                                            !reply->parameter_bytes && !reply->original_status &&
                                            reply->arguments.empty();
                if (empty_identity && reply->disposition == ExecutionWireDisposition::kRejected &&
                    reply->status == CUDA_ERROR_NOT_SUPPORTED && reply->data.empty()) {
                    context.has_dnn_profile_mismatch = true;
                    return reply;
                }
                std::uint32_t version{};
                ExecutionPayloadReader observed(reply->data);
                if (!empty_identity || reply->disposition != ExecutionWireDisposition::kComplete ||
                    reply->status || !read_execution_dnn_profile(observed, version) ||
                    version != glimmer::internal::cudnn_profile::kHeaderVersion ||
                    context.epoch != epoch || !context.is_alive || context.is_closing) {
                    context.is_uncertain = true;
                    return std::nullopt;
                }
                context.dnn_profile_epoch = epoch;
            }
        }
#endif
        std::optional<CounterPin> retiring;
        if (operation == ExecutionOpcode::kClose) {
            if (context.retirement_requests == kMaxHandles)
                return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
            retiring.emplace(context.retirement_requests, exclusion);
        }
        while (operation == ExecutionOpcode::kClose && context.active_mutations) {
            if (!yield_frontend(exclusion)) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
            if (!context.is_alive || context.epoch != epoch || context.is_closing)
                return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            if (context.is_uncertain) return std::nullopt;
        }
        if (context.active_calls == kMaxHandles) return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
        if (operation == ExecutionOpcode::kClose) context.is_closing = true;
        // Seal new calls after actual work replies, before waiting for finite
        // metadata queries. Requiring zero queries before sealing can starve
        // teardown forever while another thread polls a logical frontier.
        while (operation == ExecutionOpcode::kClose && context.active_calls) {
            if (!yield_frontend(exclusion)) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
            if (!context.is_alive || context.epoch != epoch)
                return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            if (context.is_uncertain) return std::nullopt;
        }
        if (operation == ExecutionOpcode::kClose && context.kind == Kind::kContext) {
            const auto observed = settle_async_launches(exclusion, context);
            if (observed != CUDA_SUCCESS) return local_failure(observed);
            const auto staged = release_origin_staging(exclusion, context);
            if (staged != CUDA_SUCCESS) return local_failure(staged);
            const auto released = release_origin_hosts(exclusion, context);
            if (released != CUDA_SUCCESS) return local_failure(released);
        }
        CounterPin pending(context.active_calls, exclusion);
        std::optional<CounterPin> mutation;
        if (operation != ExecutionOpcode::kClose &&
            operation != ExecutionOpcode::kCaptureFrontier &&
            operation != ExecutionOpcode::kQueryFrontier &&
            operation != ExecutionOpcode::kReleaseFrontier &&
            operation != ExecutionOpcode::kQueryAsyncFrontiers &&
            operation != ExecutionOpcode::kReleaseAsyncFrontiers &&
            operation != ExecutionOpcode::kQueryEvent &&
            operation != ExecutionOpcode::kSynchronizeEvent &&
            operation != ExecutionOpcode::kEventElapsedTime &&
            operation != ExecutionOpcode::kGetBlasAttribute &&
            operation != ExecutionOpcode::kGetLtAttribute &&
            operation != ExecutionOpcode::kLtAlgorithmCapability &&
            operation != ExecutionOpcode::kLtVersion &&
            !execution_is_dnn_descriptor_observation(operation) &&
            operation != ExecutionOpcode::kGetDnnStream)
            mutation.emplace(context.active_mutations, exclusion);
            // Close may first perform exact helper cleanup RPCs. Capture origin
            // before yielding, but assign the dense ordinal only after those calls.
#ifdef GLIMMER_HAS_NCCL
        if (continuation && !explicit_continuation) {
            continuation->ordinary_sequence = nccl_caller->ordinary_sequence;
            if (!source_continuation_healthy(continuation))
                return local_failure(CUDA_ERROR_UNKNOWN);
        }
#endif
        std::optional<std::vector<std::byte>> wrapped;
        if (continuation) {
            wrapped = encode_execution_source_continuation(*continuation, operation, bytes,
                                                           config->max_payload_bytes);
            if (!wrapped) return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
        }
        auto* connection = context.connection.get();
        ExecutionSocketSubmission accepted;
        for (;;) {
            // A callback completion must always fit behind ordinary requests
            // waiting on that callback. Only accepted transport requests count:
            // counting callers waiting here would prevent every waiter moving.
            if (context.host_function_pins &&
                context.pending_requests >= config->max_pending_requests - 1) {
                if (!connection->poll_for(0)) {
                    report_rpc_expiry(context);
                    context.is_uncertain = true;
                    quarantine_source_continuation(continuation);
                    return std::nullopt;
                }
                if (!yield_frontend(exclusion)) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
                if (!context.is_alive || context.epoch != epoch)
                    return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
                if (context.is_uncertain) return std::nullopt;
                continue;
            }
            rpc_observation.sending();
            const auto wire_operation = wrapped ? ExecutionOpcode::kSourceContinuation : operation;
            const auto wire_bytes = wrapped ? std::span<const std::byte>{*wrapped} : bytes;
            bool fully_flushed{};
            accepted =
                rpc_observation.observes_delivery()
                    ? connection->submit_observed(wire_operation, wire_bytes, borrowed_descriptor,
                                                  copy_audit.submission_timestamp(), &fully_flushed)
                    : connection->submit(wire_operation, wire_bytes, borrowed_descriptor,
                                         copy_audit.submission_timestamp());
            if (accepted.status == ExecutionSocketSubmitStatus::kAccepted) {
                rpc_observation.submitted(accepted.sequence, fully_flushed);
                copy_audit.submitted(accepted.sequence);
                if (was_submitted) *was_submitted = true;
                if (submitted_sequence) *submitted_sequence = accepted.sequence;
                if (submitted_continuation && continuation) *submitted_continuation = *continuation;
                submitted_source_continuation(continuation);
                break;
            }
            if (accepted.status == ExecutionSocketSubmitStatus::kInvalid)
                return local_failure(CUDA_ERROR_INVALID_VALUE);
            if (accepted.status != ExecutionSocketSubmitStatus::kFull || !connection->poll_for(0)) {
                report_rpc_expiry(context);
                context.is_uncertain = true;
                quarantine_source_continuation(continuation);
                return std::nullopt;
            }
            if (!yield_frontend(exclusion)) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
            if (!context.is_alive || context.epoch != epoch)
                return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
            if (context.is_uncertain) return std::nullopt;
        }
        CounterPin outstanding(context.pending_requests, exclusion);
        bool can_progress = true;
        unsigned int quick_probes_left{8};
        std::optional<ExecutionWireReply> result;
        for (;;) {
            if (!result) {
                const auto begin = rpc_observation.phase_clock();
                result = connection->take_reply(accepted.sequence);
                rpc_observation.phase(RpcMetrics::Phase::kPollTake,
                                      {begin, rpc_observation.phase_clock()});
            }
            if (result) {
                rpc_observation.received();
                if (result->status == 0 &&
                    result->disposition != ExecutionWireDisposition::kComplete &&
                    !(operation == ExecutionOpcode::kLaunch && !continuation &&
                      result->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                      !result->original_status && result->id && result->address &&
                      !result->parameter_bytes && result->arguments.empty() &&
                      result->data.empty()) &&
                    !(execution_allows_compute_replacement(operation) &&
                      result->disposition == ExecutionWireDisposition::kComputeReplacement &&
                      !result->original_status && result->id) &&
                    !(operation == ExecutionOpcode::kLaunch && !continuation &&
                      result->disposition == ExecutionWireDisposition::kAdmissionBusy &&
                      !result->original_status && !result->id && !result->address &&
                      !result->parameter_bytes && result->arguments.empty() &&
                      result->data.empty()) &&
                    !(!continuation && graph_quarantine_receipt(operation, *result))) {
                    context.is_uncertain = true;
                    quarantine_source_continuation(continuation);
#ifdef GLIMMER_HAS_NCCL
                    // The NCCL boundary preserves an actual original status
                    // independently of this malformed carrier/disposition.
                    // Return only to its dedicated validator; never replay.
                    if (operation == ExecutionOpcode::kNcclCall) return result;
#endif
                    return std::nullopt;
                }
                copy_audit.replied(result->status);
                return result;
            }
            if (!can_progress) {
                report_rpc_expiry(context);
                context.is_uncertain = true;
                quarantine_source_continuation(continuation);
                return std::nullopt;
            }
            const auto poll_begin = rpc_observation.phase_clock();
            can_progress = connection->poll_for(0);
            result = connection->take_reply(accepted.sequence);
            rpc_observation.phase(RpcMetrics::Phase::kPollTake,
                                  {poll_begin, rpc_observation.phase_clock()});
            // A ready exact receipt, including one retained after transport
            // poisoning, still passes the same validation before any yield.
            if (result || !can_progress) continue;
            if (quick_probes_left) {
                --quick_probes_left;
                // Yield exclusion on every bounded probe so other callers may
                // deliver callback/error progress. A poll is never a receipt.
                exclusion.unlock();
                const auto yield_begin = rpc_observation.phase_clock();
                std::this_thread::yield();
                const auto yield_end = rpc_observation.phase_clock();
                if (::getpid() != kProcess) return local_failure(CUDA_ERROR_INVALID_CONTEXT);
                const auto lock_begin = rpc_observation.phase_clock();
                exclusion.lock();
                const auto lock_end = rpc_observation.phase_clock();
                rpc_observation.phase(RpcMetrics::Phase::kQuickYield, {yield_begin, yield_end});
                rpc_observation.phase(RpcMetrics::Phase::kMutexReacquire, {lock_begin, lock_end});
            } else if (!wait_frontend_reply(exclusion, *connection, rpc_observation)) {
                return local_failure(CUDA_ERROR_INVALID_CONTEXT);
            }
            if (!context.is_alive || context.epoch != epoch)
                return local_failure(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        }
    }
    std::optional<ExecutionWireReply> call_owned(
        Exclusion& exclusion, Entry& context, ExecutionOpcode operation,
        std::span<const std::byte> bytes, int borrowed_descriptor = -1,
        const std::optional<ExecutionSourceContinuation>& explicit_continuation = std::nullopt,
        ExecutionSourceContinuation* submitted_continuation = nullptr) {
        if (context.is_primary && !context.is_active && !context.is_uncertain) {
            const auto opened =
                open_context(exclusion, context, context.context_flags | CU_CTX_MAP_HOST);
            if (opened != CUDA_SUCCESS) {
                ExecutionWireReply failed;
                failed.status = opened;
                failed.disposition = ExecutionWireDisposition::kRejected;
                return failed;  // The requested original operation was not called.
            }
        }
        return rpc(exclusion, context, operation, bytes, borrowed_descriptor, nullptr,
                   explicit_continuation, submitted_continuation);
    }
    std::optional<ExecutionWireReply> call(
        Exclusion& exclusion, Entry& context, ExecutionOpcode operation,
        ExecutionPayloadWriter input, int borrowed_descriptor = -1,
        const std::optional<ExecutionSourceContinuation>& explicit_continuation = std::nullopt,
        ExecutionSourceContinuation* submitted_continuation = nullptr) {
        const auto bytes = input.take();
        return call_owned(exclusion, context, operation, bytes, borrowed_descriptor,
                          explicit_continuation, submitted_continuation);
    }
    CUresult open_host_context(Exclusion& exclusion);
    CUresult observe_async_launch_batch(Exclusion& exclusion,
                                        std::span<const std::shared_ptr<AsyncLaunch>> launches,
                                        bool may_defer_successful_release = false);
    void observe_async_launches(Exclusion& exclusion,
                                std::span<const std::shared_ptr<AsyncLaunch>> snapshot,
                                bool may_defer_successful_release = false);
    CUresult flush_retired_async_launches(Exclusion& exclusion, Entry& context);
    void reap_async_launches(Exclusion& exclusion, bool is_full = false);
    void retire_async_notifications() noexcept;
    CUresult settle_async_launches(Exclusion& exclusion, Entry& context);
    CUresult retired_driver_error(Entry& context, std::uint64_t epoch) noexcept;
    bool has_async_status_work() const noexcept;
    void run_async_status_progress();
    void ensure_async_status_progress();
    void request_async_status_progress() noexcept {
        has_async_status_hint = true;
        async_status_wake.notify_one();
    }
    CUresult release_host(Exclusion& exclusion, Entry& entry);
    CUresult release_origin_hosts(Exclusion& exclusion, Entry& origin);
    void poll_staging();
    void poll_staging_owner(Entry& owner);
    void ensure_host_function_progress();
    void run_host_functions();
    HostFunction* ready_host_function();
    void quarantine_host_function(HostFunction& function) noexcept;
    bool release_host_function(HostFunction& function) noexcept;
    CUresult release_origin_staging(Exclusion& exclusion, Entry& origin);
    Entry* host_range(const void* pointer, std::size_t bytes, bool exact = false) const noexcept {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        for (auto* entry : host_allocations)
            if (entry->is_alive && !entry->is_retiring && !entry->is_uncertain &&
                address >= entry->address && address - entry->address < entry->bytes &&
                bytes <= entry->bytes - (address - entry->address) &&
                (!exact || address == entry->address))
                return entry;
        return nullptr;
    }
    Entry* range(CUdeviceptr pointer, std::size_t bytes, bool exact = false) const noexcept {
        for (const auto& entry : entries)
            if ((entry->kind == Kind::kAllocation || entry->kind == Kind::kModuleGlobal ||
                 entry->kind == Kind::kProcessAllocation) &&
                entry->is_alive && !entry->is_uncertain &&
                (entry->kind == Kind::kProcessAllocation
                     ? context() && process_memory_context &&
                           entry->context == process_memory_context &&
                           !process_memory_context->is_uncertain && !entry->is_retiring
                     : entry->context == context()) &&
                pointer >= entry->address && pointer - entry->address <= entry->bytes &&
                bytes <= entry->bytes - (pointer - entry->address) &&
                (!exact || pointer == entry->address))
                return entry.get();
        return nullptr;
    }
    bool range_is_retiring(const Entry& entry) const noexcept {
        return entry.is_retiring ||
               (entry.kind == Kind::kModuleGlobal &&
                std::ranges::any_of(entries, [&](const auto& parent) {
                    return parent->kind == Kind::kModule && parent->context == entry.context &&
                           parent->remote == entry.parent && parent->is_retiring;
                }));
    }
    CUresult classify_unified_endpoint(std::uintptr_t address, std::size_t bytes,
                                       UnifiedEndpoint& output) const noexcept {
        auto* origin = context();
        if (!origin) return CUDA_ERROR_INVALID_CONTEXT;
        if (bytes > UINTPTR_MAX - address) return CUDA_ERROR_INVALID_VALUE;
        if (!address) {
            if (bytes) return CUDA_ERROR_INVALID_VALUE;
            output = {};
            return CUDA_SUCCESS;
        }
        // A known interval must never become pageable merely because a full
        // owned-range lookup failed. Inspect both ends, including zero-byte
        // endpoints and the rounded mapping padding of public host allocations.
        const auto touches = [&](std::uint64_t base, std::uint64_t extent) {
            return address >= base ? address - base < extent || (!bytes && address - base == extent)
                                   : base - address < bytes;
        };
        UnifiedEndpoint candidate;
        candidate.address = address;
        bool has_known_range{};
        for (const auto& entry : entries) {
            const bool is_device = entry->kind == Kind::kAllocation ||
                                   entry->kind == Kind::kModuleGlobal ||
                                   entry->kind == Kind::kProcessAllocation;
            const bool is_host = entry->kind == Kind::kHostAllocation;
            if (!is_device && !is_host) continue;
            const auto extent =
                is_host && entry->host_mapping ? entry->host_mapping->bytes : entry->bytes;
            if (!touches(entry->address, extent)) {
                has_known_range = has_known_range ||
                                  (address >= entry->address && address - entry->address == extent);
                continue;
            }
            has_known_range = true;
            if (!entry->is_alive) continue;  // A newer live allocation may reuse this address.
            if (entry->is_uncertain ||
                (entry->kind == Kind::kProcessAllocation &&
                 (!process_memory_context || entry->context != process_memory_context ||
                  process_memory_context->is_uncertain || !process_memory_context->is_active)) ||
                (is_host && (!host_context || !host_context->is_active ||
                             !host_context->connection || host_context->is_uncertain)))
                return CUDA_ERROR_UNKNOWN;
            if ((is_device && entry->kind != Kind::kProcessAllocation &&
                 entry->context != origin) ||
                range_is_retiring(*entry) || address < entry->address ||
                address - entry->address > entry->bytes ||
                bytes > entry->bytes - (address - entry->address))
                return CUDA_ERROR_INVALID_VALUE;
            if (candidate.allocation) return CUDA_ERROR_INVALID_VALUE;
            candidate.allocation = entry.get();
            candidate.wire = {
                is_device ? ExecutionCopyEndpointKind::kDevice : ExecutionCopyEndpointKind::kHost,
                is_device ? 0 : host_context->connection->session(), entry->remote,
                address - entry->address};
            if (entry->kind == Kind::kModuleGlobal) {
                const auto parent = std::ranges::find_if(entries, [&](const auto& value) {
                    return value->kind == Kind::kModule && value->context == origin &&
                           value->remote == entry->parent && value->is_alive;
                });
                if (parent == entries.end()) return CUDA_ERROR_INVALID_VALUE;
                if ((*parent)->is_uncertain) return CUDA_ERROR_UNKNOWN;
                if ((*parent)->is_retiring) return CUDA_ERROR_INVALID_VALUE;
                candidate.parent = parent->get();
            }
        }
        if (!candidate.allocation) {
            if (has_known_range) return CUDA_ERROR_INVALID_VALUE;
            for (const auto& stage : staging_allocations)
                if (touches(reinterpret_cast<std::uintptr_t>(stage->mapping.pointer),
                            stage->mapping.bytes))
                    return stage->is_uncertain ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_VALUE;
            candidate.wire.kind = ExecutionCopyEndpointKind::kPageable;
        }
        output = candidate;
        return CUDA_SUCCESS;
    }
};
struct StateHolder {
    std::unique_ptr<State> value{std::make_unique<State>()};
    ~StateHolder() {
        if (::getpid() != kProcess) (void)value.release();
    }
};
State& state() {
    static StateHolder holder;
    return *holder.value;
}
struct Entered {
    Entered() {
        is_entered = true;
    }
    ~Entered() {
        is_entered = false;
    }
};
template <typename Function>
CUresult locked_boundary(Function&& function, bool needs_connection = true) noexcept {
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (is_host_function_active) return CUDA_ERROR_NOT_PERMITTED;
    if (is_entered) return CUDA_ERROR_NOT_SUPPORTED;
    Entered entered;
    try {
        auto& shared = state();
        const auto entered_at = shared.rpc_metrics
                                    ? shared.rpc_metrics->entry()
                                    : glimmer::interceptor::execution::RpcMetrics::Stamp{};
        std::unique_lock lock(shared.mutex);
        glimmer::interceptor::execution::RpcMetrics::Scope boundary_observation(
            shared.rpc_metrics.get(), lock, entered_at, true);
        CopyAudit::acquired();
        if (needs_connection && !shared.config) return CUDA_ERROR_NOT_INITIALIZED;
        return function(shared, lock);
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
template <typename Function>
CUresult boundary(Function&& function) noexcept {
    return locked_boundary(
        [&](State& shared, std::unique_lock<std::mutex>&) { return function(shared); });
}
CUresult status(const std::optional<ExecutionWireReply>& result) noexcept {
    return result ? static_cast<CUresult>(result->status) : CUDA_ERROR_UNKNOWN;
}
bool async_batch_receipt(const ExecutionWireReply& reply) noexcept {
    return reply.disposition == ExecutionWireDisposition::kComplete && !reply.status &&
           !reply.original_status && !reply.id && !reply.address && !reply.parameter_bytes &&
           reply.arguments.empty();
}
struct AsyncObservationPins {
    Entry& context;
    Exclusion& exclusion;
    std::span<const std::shared_ptr<AsyncLaunch>> launches;
    bool was_submitted{};
    int exceptions{std::uncaught_exceptions()};
    AsyncObservationPins(Entry& owner, Exclusion& lock,
                         std::span<const std::shared_ptr<AsyncLaunch>> selected)
        : context(owner), exclusion(lock), launches(selected) {
        for (const auto& launch : launches) launch->is_polling = true;
    }
    AsyncObservationPins(const AsyncObservationPins&) = delete;
    AsyncObservationPins& operator=(const AsyncObservationPins&) = delete;
    CUresult quarantine() noexcept {
        context.is_uncertain = true;
        for (const auto& launch : launches) cache_async_failure(*launch, CUDA_ERROR_UNKNOWN);
        return CUDA_ERROR_UNKNOWN;
    }
    ~AsyncObservationPins() {
        if (::getpid() != kProcess || !exclusion.owns_lock()) return;
        if (was_submitted && std::uncaught_exceptions() != exceptions) (void)quarantine();
        for (const auto& launch : launches) launch->is_polling = false;
    }
};
CUresult State::observe_async_launch_batch(Exclusion& exclusion,
                                           std::span<const std::shared_ptr<AsyncLaunch>> launches,
                                           bool may_defer_successful_release) {
    if (launches.empty() || launches.size() > kMaxAsyncLaunches) return CUDA_ERROR_INVALID_VALUE;
    auto& context = *launches.front()->context;
    const auto epoch = launches.front()->epoch;
    std::array<ExecutionAsyncFrontierKey, kMaxAsyncLaunches> keys{};
    std::array<std::size_t, kMaxAsyncLaunches> queried_indices{};
    std::array<ExecutionAsyncFrontierObservation, kMaxAsyncLaunches> observations{};
    std::size_t query_count{};
    for (std::size_t index = 0; index < launches.size(); ++index) {
        const auto& launch = launches[index];
        if (launch->context != &context || launch->epoch != epoch || launch->is_preparing ||
            launch->is_polling || !launch->frontier || !launch->sequence)
            return CUDA_ERROR_INVALID_VALUE;
        const ExecutionAsyncFrontierKey key{launch->frontier, launch->sequence};
        if (launch->has_retirement_proof) {
            observations[index] = {key, static_cast<std::int32_t>(launch->failure), true};
        } else {
            queried_indices[query_count] = index;
            keys[query_count++] = key;
        }
    }
    // A second active-call slot is needed by each RPC. The outer pin prevents
    // context reset/Close between Query publication and the Release receipt.
    if (context.active_calls >= kMaxHandles - 1) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin active(context.active_calls, exclusion);
    AsyncObservationPins polling(context, exclusion, launches);
    const auto is_current = [&] {
        return context.is_alive && context.is_active && context.epoch == epoch &&
               context.connection && !context.is_uncertain;
    };
    if (!is_current()) return polling.quarantine();
    if (query_count) {
        const auto selected = std::span{keys}.first(query_count);
        const auto query_bytes = encode_execution_async_frontier_batch(selected);
        if (!query_bytes) return CUDA_ERROR_OUT_OF_MEMORY;
        const auto observed =
            rpc(exclusion, context, ExecutionOpcode::kQueryAsyncFrontiers, *query_bytes, -1,
                &polling.was_submitted, std::nullopt, nullptr, nullptr, true);
        if (::getpid() != kProcess || !exclusion.owns_lock()) return CUDA_ERROR_INVALID_CONTEXT;
        if (!polling.was_submitted && observed && observed->status && !context.is_uncertain)
            return status(observed);  // Local rejection did not create remote uncertainty.
        if (!is_current() || !observed || !async_batch_receipt(*observed))
            return polling.quarantine();
        const auto queried = decode_execution_async_frontier_observations(observed->data);
        if (!queried || queried->size() != selected.size()) return polling.quarantine();
        for (std::size_t index = 0; index < query_count; ++index)
            if ((*queried)[index].key != selected[index]) return polling.quarantine();
        // Validate the entire carrier before publishing any error or stop proof.
        for (std::size_t index = 0; index < query_count; ++index)
            observations[queried_indices[index]] = (*queried)[index];
        polling.was_submitted = false;
    }

    std::array<ExecutionAsyncFrontierKey, kMaxAsyncLaunches> retired{};
    std::size_t retired_count{};
    bool has_error{};
    for (std::size_t index = 0; index < launches.size(); ++index) {
        const auto& observation = observations[index];
        auto& launch = *launches[index];
        cache_async_failure(launch, static_cast<CUresult>(observation.status));
        launch.has_retirement_proof = observation.is_retired;
        has_error = has_error || launch.failure != CUDA_SUCCESS;
        if (observation.is_retired) retired[retired_count++] = observation.key;
    }
    // Publish EVERY error before Release can delete any service metadata while
    // exclusion is yielded. Physical proof stays with this pinned local batch;
    // a frontier remains owned until its exact echoed Release receipt arrives.
    if (!retired_count) return CUDA_SUCCESS;
    // Only successful, physically proven metadata may wait for a small batch.
    // Keep its frontier and capacity charge until the exact Release echo arrives.
    if (may_defer_successful_release && !has_error && retired_count < kAsyncReapBudget)
        return CUDA_SUCCESS;
    const auto retired_keys = std::span{retired}.first(retired_count);
    const auto release_bytes = encode_execution_async_frontier_batch(retired_keys);
    if (!release_bytes) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto released =
        rpc(exclusion, context, ExecutionOpcode::kReleaseAsyncFrontiers, *release_bytes, -1,
            &polling.was_submitted, std::nullopt, nullptr, nullptr, true);
    if (::getpid() != kProcess || !exclusion.owns_lock()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!polling.was_submitted && released && released->status && !context.is_uncertain)
        return status(released);
    if (!is_current() || !released || !async_batch_receipt(*released)) return polling.quarantine();
    const auto echoed = decode_execution_async_frontier_batch(released->data);
    if (!echoed || !std::ranges::equal(*echoed, retired_keys)) return polling.quarantine();
    polling.was_submitted = false;
    for (std::size_t index = 0; index < launches.size(); ++index)
        if (observations[index].is_retired) launches[index]->frontier = 0;
    return CUDA_SUCCESS;
}
void State::observe_async_launches(Exclusion& exclusion,
                                   std::span<const std::shared_ptr<AsyncLaunch>> snapshot,
                                   bool may_defer_successful_release) {
    // Each selected context/generation is visited once, including when its
    // first record is already being polled by another thread. Strong references
    // keep the bounded selection stable across every unlocked transport wait.
    for (std::size_t index = 0; index < snapshot.size(); ++index) {
        const auto same_origin = [&](const auto& launch) {
            return launch->context == snapshot[index]->context &&
                   launch->epoch == snapshot[index]->epoch;
        };
        if (std::ranges::any_of(snapshot.first(index), same_origin)) continue;
        std::array<std::shared_ptr<AsyncLaunch>, kMaxAsyncLaunches> group;
        std::size_t count{};
        for (const auto& launch : snapshot.subspan(index))
            if (same_origin(launch) && !launch->is_preparing && !launch->is_polling &&
                launch->frontier)
                group[count++] = launch;
        if (count)
            (void)observe_async_launch_batch(exclusion, std::span{group}.first(count),
                                             may_defer_successful_release);
        if (::getpid() != kProcess || !exclusion.owns_lock()) return;
    }
}
CUresult State::flush_retired_async_launches(Exclusion& exclusion, Entry& context) {
    std::array<std::shared_ptr<AsyncLaunch>, kMaxAsyncLaunches> selected;
    std::size_t count{};
    for (const auto& launch : async_launches)
        if (launch->context == &context && launch->epoch == context.epoch &&
            launch->has_retirement_proof && launch->frontier && !launch->is_preparing &&
            !launch->is_polling)
            selected[count++] = launch;
    if (!count) return CUDA_SUCCESS;
    return observe_async_launch_batch(exclusion, std::span{selected}.first(count));
}
void State::reap_async_launches(Exclusion& exclusion, bool is_full) {
    // Strong snapshots survive another caller's erase while an RPC yields the
    // shared exclusion. Ordinary admission spends only a fixed query budget;
    // a pressure retry may inspect every bounded record and flush retirements.
    std::array<std::shared_ptr<AsyncLaunch>, kMaxAsyncLaunches> snapshot;
    const auto count =
        std::min(async_launches.size(), is_full ? kMaxAsyncLaunches : kAsyncReapBudget);
    for (std::size_t index = 0; index < count; ++index) {
        async_reap_cursor %= async_launches.size();
        snapshot[index] = async_launches[async_reap_cursor++];
    }
    observe_async_launches(exclusion, std::span{snapshot}.first(count), !is_full);
    if (::getpid() == kProcess && exclusion.owns_lock()) retire_async_notifications();
}
void State::retire_async_notifications() noexcept {
    std::erase_if(async_launches, [](const auto& launch) {
        const bool is_old_context = !launch->context->is_alive || !launch->context->is_active ||
                                    launch->context->epoch != launch->epoch;
        return !launch->is_preparing && !launch->frontier &&
               (launch->failure == CUDA_SUCCESS ||
                (is_old_context && (launch->is_notified || launch->caller.expired())));
    });
}
CUresult State::settle_async_launches(Exclusion& exclusion, Entry& context) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds{config ? config->timeout_ms : 30000};
    for (;;) {
        const auto flushed = flush_retired_async_launches(exclusion, context);
        if (flushed != CUDA_SUCCESS) return flushed;
        const auto snapshot = async_launches;
        std::array<std::shared_ptr<AsyncLaunch>, kMaxAsyncLaunches> selected;
        std::size_t selected_count{};
        for (const auto& launch : snapshot)
            if (launch->context == &context && launch->epoch == context.epoch)
                selected[selected_count++] = launch;
        observe_async_launches(exclusion, std::span{selected}.first(selected_count));
        bool has_pending{};
        for (const auto& launch : snapshot) {
            if (launch->context != &context || launch->epoch != context.epoch) continue;
            has_pending = has_pending || launch->is_preparing || launch->frontier;
        }
        if (context.is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (!has_pending) return CUDA_SUCCESS;
        if (std::chrono::steady_clock::now() >= deadline) {
            context.is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
}
CUresult State::retired_driver_error(Entry& context, std::uint64_t epoch) noexcept {
    CUresult first = CUDA_SUCCESS;
    for (const auto& launch : async_launches) {
        if (launch->context != &context || launch->epoch != epoch || launch->is_runtime ||
            launch->is_preparing || launch->frontier)
            continue;
        if (!launch->is_notified && first == CUDA_SUCCESS) first = launch->failure;
        // Destruction reports the first unreported Driver failure after actual
        // teardown. Runtime notifications instead survive for their own caller.
        launch->is_notified = true;
    }
    retire_async_notifications();
    return first;
}
bool State::has_async_status_work() const noexcept {
    return std::ranges::any_of(async_launches, [](const auto& launch) {
        const auto& context = *launch->context;
        return !launch->is_preparing && launch->frontier && !launch->has_retirement_proof &&
               context.is_alive && context.is_active && !context.is_uncertain &&
               context.epoch == launch->epoch;
    });
}
void State::run_async_status_progress() {
    // CPU metadata only: no CUDA context, SDK invocation or application buffer.
    std::unique_lock lock(mutex);
    for (;;) {
        async_status_wake.wait(
            lock, [this] { return should_stop_async_status || has_async_status_work(); });
        if (should_stop_async_status) return;
        has_async_status_hint = false;
        try {
            reap_async_launches(lock);
        } catch (const std::bad_alloc&) {
            // Pre-entry allocation failure retains every record for retry.
            // Submitted ambiguity is already quarantined by observation pins.
        } catch (...) {
            for (const auto& launch : async_launches)
                if (launch->frontier && !launch->has_retirement_proof) {
                    launch->context->is_uncertain = true;
                    cache_async_failure(*launch, CUDA_ERROR_UNKNOWN);
                }
        }
        async_status_wake.wait_for(lock, std::chrono::microseconds{200}, [this] {
            return should_stop_async_status || has_async_status_hint;
        });
        if (should_stop_async_status) return;
    }
}
void State::ensure_async_status_progress() {
    if (!async_status_progress.joinable())
        async_status_progress = std::thread([this] { run_async_status_progress(); });
}
bool host_function_receipt(const ExecutionWireReply& reply, std::uint64_t token,
                           std::uint64_t parameter_bytes = 0) noexcept {
    return reply.disposition == ExecutionWireDisposition::kComplete && !reply.original_status &&
           reply.id == token && !reply.address && reply.parameter_bytes == parameter_bytes &&
           reply.arguments.empty() && reply.data.empty();
}
void State::quarantine_host_function(HostFunction& function) noexcept {
    function.phase = HostFunctionPhase::kUncertain;
    function.origin->is_uncertain = true;
    if (function.continuation.group) quarantine_source_continuation(function.continuation);
}
bool State::release_host_function(HostFunction& function) noexcept {
    auto& origin = *function.origin;
    const auto found = std::ranges::find_if(
        host_functions, [&](const auto& candidate) { return candidate.get() == &function; });
    if (found == host_functions.end() || !origin.active_mutations || !origin.host_function_pins ||
        (function.stream && !function.stream->active_calls)) {
        quarantine_host_function(function);
        return false;
    }
    --origin.active_mutations;
    --origin.host_function_pins;
    if (function.stream) --function.stream->active_calls;
    host_functions.erase(found);
    return true;
}
HostFunction* State::ready_host_function() {
    for (std::size_t index = 0; index < host_functions.size();) {
        auto& function = *host_functions[index];
        if (function.phase != HostFunctionPhase::kWaiting &&
            function.phase != HostFunctionPhase::kFinishing) {
            ++index;
            continue;
        }
        auto& origin = *function.origin;
        if (!origin.is_alive || !origin.is_active || origin.epoch != function.epoch ||
            !origin.connection || (function.stream && !function.stream->is_alive)) {
            quarantine_host_function(function);
            ++index;
            continue;
        }
        auto& connection = *origin.connection;
        const bool can_progress = connection.poll_for(0);
        if (function.sequence) {
            if (const auto reply = connection.take_reply(function.sequence)) {
                if (!origin.pending_requests || !origin.has_host_function_request) {
                    quarantine_host_function(function);
                    ++index;
                    continue;
                }
                --origin.pending_requests;
                origin.has_host_function_request = false;
                function.sequence = 0;
                if (function.phase == HostFunctionPhase::kWaiting && reply->status > 0 &&
                    reply->status <= 999 && host_function_receipt(*reply, function.remote, 1)) {
                    // The owner explicitly proved that permission was never
                    // published and removed this CPU boundary. Retain the
                    // predecessor error independently of local pin retirement.
                    if (origin.host_function_error == CUDA_SUCCESS)
                        origin.host_function_error = static_cast<CUresult>(reply->status);
                    if (release_host_function(function)) continue;
                } else if (!reply->status && host_function_receipt(*reply, function.remote)) {
                    if (function.phase == HostFunctionPhase::kWaiting && !origin.is_uncertain &&
                        origin.host_function_error == CUDA_SUCCESS) {
                        function.phase = HostFunctionPhase::kRunning;
                        return &function;
                    }
                    if (function.phase == HostFunctionPhase::kFinishing &&
                        function.has_normal_return && release_host_function(function))
                        continue;
                }
                quarantine_host_function(function);
                ++index;
                continue;
            }
        }
        if (!can_progress ||
            (origin.is_uncertain && function.phase == HostFunctionPhase::kWaiting)) {
            quarantine_host_function(function);
            ++index;
            continue;
        }
        if (!function.sequence) {
            const bool is_waiting = function.phase == HostFunctionPhase::kWaiting;
            // Both readiness and completion may use the reserved final slot,
            // but only one callback control request per origin may be pending.
            // Wire submission order, not local vector order, decides which
            // callback is first when concurrent callers yield during enqueue.
            const bool has_predecessor =
                is_waiting && std::ranges::any_of(host_functions, [&](const auto& other) {
                    return other.get() != &function && other->origin == &origin &&
                           (other->phase == HostFunctionPhase::kRunning ||
                            other->phase == HostFunctionPhase::kFinishing ||
                            (other->enqueue_sequence &&
                             other->enqueue_sequence < function.enqueue_sequence &&
                             (other->phase == HostFunctionPhase::kPreparing ||
                              other->phase == HostFunctionPhase::kWaiting)));
                });
            if (!has_predecessor && !origin.has_host_function_request &&
                origin.pending_requests < config->max_pending_requests) {
                const auto& bytes = is_waiting                   ? function.query_payload
                                    : function.has_normal_return ? function.finish_payload
                                                                 : function.exception_payload;
                const auto submitted =
                    connection.submit(is_waiting ? ExecutionOpcode::kQueryHostFunction
                                                 : ExecutionOpcode::kFinishHostFunction,
                                      bytes);
                if (submitted.status == ExecutionSocketSubmitStatus::kAccepted) {
                    ++origin.pending_requests;
                    origin.has_host_function_request = true;
                    function.sequence = submitted.sequence;
                } else if (submitted.status != ExecutionSocketSubmitStatus::kFull) {
                    quarantine_host_function(function);
                }
            }
        }
        ++index;
    }
    return nullptr;
}
void State::run_host_functions() {
    std::unique_lock lock(mutex);
    for (;;) {
        host_function_wake.wait(lock, [this] {
            return should_stop_host_functions ||
                   std::ranges::any_of(host_functions, [](const auto& function) {
                       return function->phase == HostFunctionPhase::kWaiting ||
                              function->phase == HostFunctionPhase::kFinishing;
                   });
        });
        if (should_stop_host_functions) return;
        HostFunction* ready{};
        try {
            ready = ready_host_function();
        } catch (...) {
            for (auto& function : host_functions)
                if (function->phase == HostFunctionPhase::kWaiting ||
                    function->phase == HostFunctionPhase::kFinishing)
                    quarantine_host_function(*function);
        }
        if (ready) {
            // Ready records have stable storage and retain their origin/stream
            // pins. Only this worker removes admitted records. No frontend
            // exclusion or native execution grant survives into user code.
            lock.unlock();
            bool has_normal_return = true;
            is_host_function_active = true;
            is_entered = true;  // Library-specific boundaries reject CUDA reentry too.
            try {
                ready->function(ready->data);
            } catch (...) {
                has_normal_return = false;
            }
            is_entered = false;
            is_host_function_active = false;
            if (::getpid() != kProcess) return;  // Never lock inherited state after fork.
            lock.lock();
            ready->has_normal_return = has_normal_return;
            ready->phase = HostFunctionPhase::kFinishing;
            if (!has_normal_return) {
                ready->origin->is_uncertain = true;
                if (ready->continuation.group) quarantine_source_continuation(ready->continuation);
            }
            continue;
        }
        host_function_wake.wait_for(lock, std::chrono::milliseconds{2},
                                    [this] { return should_stop_host_functions; });
        if (should_stop_host_functions) return;
    }
}
void State::ensure_host_function_progress() {
    if (!host_function_progress.joinable())
        host_function_progress = std::thread([this] { run_host_functions(); });
}
CUresult State::open_host_context(Exclusion& exclusion) {
    if (!host_context) {
        host_context = allocate_entry(Kind::kHostContext, nullptr);
        if (!host_context) return CUDA_ERROR_OUT_OF_MEMORY;
        host_context->context = host_context;
        host_context->is_alive = true;
    }
    return open_context(exclusion, *host_context, CU_CTX_MAP_HOST);
}
#ifdef GLIMMER_HAS_NCCL
std::optional<ExecutionSourceContinuation> cleanup_continuation(const SourceCleanupBinding& binding,
                                                                const Entry& helper,
                                                                std::uint64_t resource) {
    if (!helper.connection || helper.connection->session() != binding.helper_session ||
        resource != binding.resource || !resource || !binding.next_sequence)
        return std::nullopt;
    auto scope = binding.origin;
    scope.flags = kExecutionSourceContinuationCleanup;
    scope.cleanup_parent = scope.ordinary_sequence;
    scope.ordinary_sequence = binding.next_sequence;
    return scope;
}
void advance_cleanup(SourceCleanupBinding& binding) {
    binding.next_sequence = binding.next_sequence == UINT64_MAX ? 0 : binding.next_sequence + 1;
}
#endif
void State::poll_staging() {
    if (staging_context) poll_staging_owner(*staging_context);
    if (pageable_context) poll_staging_owner(*pageable_context);
}
void State::poll_staging_owner(Entry& owner) {
    if (!owner.connection || owner.is_opening || !owner.is_active) return;
    auto& connection = *owner.connection;
    const bool can_progress = connection.poll_for(0);
    for (auto& allocation : staging_allocations) {
        if (allocation->owner != &owner) continue;
        if (allocation->idle_sequence && !allocation->is_uncertain) {
            const auto idle = connection.take_reply(allocation->idle_sequence);
            if (idle) {
                if (idle->status != CUDA_SUCCESS || idle->original_status ||
                    idle->disposition != ExecutionWireDisposition::kComplete ||
                    idle->id != allocation->remote || idle->address || idle->parameter_bytes ||
                    !idle->arguments.empty() || !idle->data.empty()) {
                    allocation->is_uncertain = owner.is_uncertain = true;
                    continue;
                }
                allocation->idle_sequence = 0;
                allocation->is_idle = true;
            }
        }
        if (!allocation->release_sequence || allocation->is_uncertain) continue;
        const auto released = connection.take_reply(allocation->release_sequence);
        if (!released) continue;
        if (released->status != CUDA_SUCCESS ||
            released->disposition != ExecutionWireDisposition::kComplete ||
            (owner.kind == Kind::kPageableContext &&
             (released->original_status || released->id || released->address ||
              released->parameter_bytes || !released->arguments.empty() ||
              !released->data.empty())) ||
            ::munmap(allocation->mapping.pointer, allocation->mapping.bytes) != 0) {
            allocation->is_uncertain = owner.is_uncertain = true;
            continue;
        }
        allocation->mapping.pointer = MAP_FAILED;
    }
    std::erase_if(staging_allocations, [](const auto& allocation) {
        return !allocation->is_preparing && !allocation->is_uncertain &&
               allocation->release_sequence && allocation->mapping.pointer == MAP_FAILED;
    });
    if (!can_progress) owner.is_uncertain = true;
    // Construction owns device-wide native exclusion. Consume already returned
    // CPU receipts above, but do not enqueue a new helper query/release here.
    if (graph_capture) return;
    for (auto& allocation : staging_allocations) {
        if (allocation->owner != &owner || allocation->is_preparing || allocation->is_uncertain ||
            allocation->release_sequence || allocation->idle_sequence ||
            (allocation->is_idle && !allocation->should_release))
            continue;
        if (owner.is_uncertain) {
            allocation->is_uncertain = true;
            continue;
        }
        ExecutionPayloadWriter input;
        if (!input.write_u64(allocation->remote)) continue;  // No native release was submitted.
        auto bytes = input.take();
        const bool should_query = allocation->can_reuse && !allocation->should_release;
        auto operation =
            should_query ? ExecutionOpcode::kQueryStagingIdle : ExecutionOpcode::kReleaseHostMemory;
#ifdef GLIMMER_HAS_NCCL
        if (allocation->nccl_cleanup) {
            const auto scope =
                cleanup_continuation(*allocation->nccl_cleanup, owner, allocation->remote);
            auto wrapped = scope ? encode_execution_source_continuation(*scope, operation, bytes,
                                                                        config->max_payload_bytes)
                                 : std::nullopt;
            if (!wrapped) {
                allocation->is_uncertain = owner.is_uncertain = true;
                continue;
            }
            bytes = std::move(*wrapped);
            operation = ExecutionOpcode::kSourceContinuation;
        }
#endif
        const auto submitted = connection.submit(operation, bytes);
        if (submitted.status == ExecutionSocketSubmitStatus::kAccepted) {
            if (should_query)
                allocation->idle_sequence = submitted.sequence;
            else
                allocation->release_sequence = submitted.sequence;
#ifdef GLIMMER_HAS_NCCL
            if (allocation->nccl_cleanup) advance_cleanup(*allocation->nccl_cleanup);
#endif
        } else if (submitted.status != ExecutionSocketSubmitStatus::kFull)
            allocation->is_uncertain = owner.is_uncertain = true;
    }
}
CUresult State::release_origin_staging(Exclusion& exclusion, Entry& origin) {
    for (;;) {
        for (auto& allocation : staging_allocations)
            if (allocation->origin == &origin && allocation->epoch == origin.epoch)
                allocation->should_release = true;
        poll_staging();
        const auto found = std::ranges::find_if(staging_allocations, [&](const auto& allocation) {
            return allocation->origin == &origin && allocation->epoch == origin.epoch;
        });
        if (found == staging_allocations.end()) return CUDA_SUCCESS;
        if ((*found)->is_uncertain || (*found)->owner->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
}
CUresult State::release_host(Exclusion& exclusion, Entry& entry) {
    if (!host_context || host_context->is_uncertain || entry.is_uncertain)
        return CUDA_ERROR_UNKNOWN;
    entry.is_retiring = true;
    while (entry.active_calls) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (host_context->is_uncertain || entry.is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry.remote)) {
        entry.is_retiring = false;  // No operation was admitted.
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    std::optional<ExecutionSourceContinuation> continuation;
#ifdef GLIMMER_HAS_NCCL
    if (entry.nccl_cleanup) {
        continuation = cleanup_continuation(*entry.nccl_cleanup, *host_context, entry.remote);
        if (!continuation) {
            entry.is_uncertain = host_context->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
    }
#endif
    bool was_submitted{};
    const auto result = rpc(exclusion, *host_context, ExecutionOpcode::kReleaseHostMemory,
                            input.take(), -1, &was_submitted, continuation);
#ifdef GLIMMER_HAS_NCCL
    if (was_submitted && entry.nccl_cleanup) advance_cleanup(*entry.nccl_cleanup);
#endif
    if (status(result) != CUDA_SUCCESS) {
        entry.is_uncertain = host_context->is_uncertain = true;
        return status(result);
    }
    if (entry.host_mapping && entry.host_mapping->pointer != MAP_FAILED) {
        if (::munmap(entry.host_mapping->pointer, entry.host_mapping->bytes) != 0) {
            entry.is_uncertain = host_context->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        entry.host_mapping->pointer = MAP_FAILED;
    }
    entry.is_alive = false;
    auto* retired = &entry;
    std::erase(host_allocations, retired);
    // A host entry is never an opaque public handle. All admission pins have
    // retired, so keeping an unreachable record would exhaust long-lived apps.
    std::erase_if(entries, [retired](const auto& candidate) { return candidate.get() == retired; });
    return CUDA_SUCCESS;
}
CUresult State::release_origin_hosts(Exclusion& exclusion, Entry& origin) {
    for (;;) {
        const auto found = std::ranges::find_if(host_allocations, [&](const auto* entry) {
            return entry->context == &origin && entry->epoch == origin.epoch;
        });
        if (found == host_allocations.end()) return CUDA_SUCCESS;
        auto* entry = *found;
        if (entry->is_uncertain || (host_context && host_context->is_uncertain))
            return CUDA_ERROR_UNKNOWN;
        if (entry->is_retiring) {
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
            continue;  // An already-issued explicit free owns the same resource.
        }
        const auto result = release_host(exclusion, *entry);
        if (result != CUDA_SUCCESS) return result;
    }
}
struct HostUsePins {
    std::array<Entry*, kMaxHostAllocations> entries{};
    std::size_t count{};
    Exclusion& exclusion;
    HostUsePins(State& shared, Exclusion& lock) : exclusion(lock) {
        if (shared.host_context && shared.host_context->is_uncertain &&
            !shared.host_allocations.empty())
            throw std::runtime_error("unproven host memory ownership");
        for (auto* entry : shared.host_allocations)
            if (entry->is_alive && !entry->is_retiring && !entry->is_uncertain) {
                if (entry->active_calls == kMaxHandles) throw std::bad_alloc();
            }
        for (auto* entry : shared.host_allocations)
            if (entry->is_alive && !entry->is_retiring && !entry->is_uncertain) {
                ++entry->active_calls;
                entries[count++] = entry;
            }
    }
    ~HostUsePins() {
        if (::getpid() == kProcess && exclusion.owns_lock())
            for (std::size_t index = 0; index < count; ++index) --entries[index]->active_calls;
    }
    HostUsePins(const HostUsePins&) = delete;
    HostUsePins& operator=(const HostUsePins&) = delete;
};
struct HostAdmission {
    State& shared;
    Entry* entry;
    Exclusion& exclusion;
    bool should_retain{};
    HostAdmission(State& value, Entry& resource, Exclusion& lock)
        : shared(value), entry(&resource), exclusion(lock) {
        shared.host_allocations.push_back(entry);
    }
    ~HostAdmission() {
        if (!should_retain && ::getpid() == kProcess && exclusion.owns_lock()) {
            std::erase(shared.host_allocations, entry);
            std::erase_if(shared.entries,
                          [this](const auto& candidate) { return candidate.get() == entry; });
        }
    }
    HostAdmission(const HostAdmission&) = delete;
    HostAdmission& operator=(const HostAdmission&) = delete;
};
CUresult host_allocate(State& shared, Exclusion& exclusion, void** output, std::size_t bytes,
                       unsigned int flags) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    auto* origin = shared.context();
    if (!origin) return CUDA_ERROR_INVALID_CONTEXT;
    if (origin->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (origin->active_mutations >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    if (flags & ~static_cast<unsigned int>(CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED))
        return CUDA_ERROR_INVALID_VALUE;
    CounterPin publishing(origin->active_mutations, exclusion);
    if (origin->is_primary && !origin->is_active) {
        const auto opened =
            shared.open_context(exclusion, *origin, origin->context_flags | CU_CTX_MAP_HOST);
        if (opened != CUDA_SUCCESS) return opened;
    }
    if (!bytes)
        return publish(output, static_cast<void*>(nullptr)) ? CUDA_SUCCESS
                                                            : CUDA_ERROR_INVALID_VALUE;
    if (flags & CU_MEMHOSTALLOC_WRITECOMBINED) return CUDA_ERROR_NOT_SUPPORTED;
    if (shared.host_allocations.size() >= kMaxHostAllocations) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0 ||
        bytes > std::numeric_limits<std::size_t>::max() - static_cast<std::size_t>(page) + 1)
        return CUDA_ERROR_INVALID_VALUE;
    const auto extent =
        ((bytes - 1) / static_cast<std::size_t>(page) + 1) * static_cast<std::size_t>(page);
    if (extent > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()))
        return CUDA_ERROR_INVALID_VALUE;
    auto* entry = shared.allocate_entry(Kind::kHostAllocation, origin);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    HostAdmission admission(shared, *entry, exclusion);
    entry->epoch = origin->epoch;
    entry->bytes = bytes;
    entry->host_flags = flags | CU_MEMHOSTALLOC_DEVICEMAP;  // Qualified Linux/V100 UVA profile.
    entry->host_mapping = std::make_unique<HostMapping>();
    entry->host_mapping->bytes = extent;
    const auto opened = shared.open_host_context(exclusion);
    if (opened != CUDA_SUCCESS) return opened;
    ExecutionOwnedDescriptor descriptor(
        ::memfd_create("glimmer-host", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (descriptor.get() < 0 || ::ftruncate(descriptor.get(), static_cast<off_t>(extent)) != 0 ||
        ::fcntl(descriptor.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0)
        return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter input;
    if (!input.write_u64(bytes) || !input.write_u64(extent) || !input.write_u32(flags))
        return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionSourceContinuation imported_origin;
    const auto result =
        shared.call(exclusion, *shared.host_context, ExecutionOpcode::kImportHostMemory,
                    std::move(input), descriptor.get(), std::nullopt, &imported_origin);
    if (status(result) != CUDA_SUCCESS) {
        std::uint32_t quarantine{};
        ExecutionPayloadReader ownership(result ? std::span<const std::byte>{result->data}
                                                : std::span<const std::byte>{});
        if (!result || !ownership.read_u32(quarantine) || !ownership.empty() || quarantine != 0) {
            shared.host_context->is_uncertain = entry->is_uncertain = true;
            admission.should_retain = true;
        }
        return status(result);
    }
    if (!result->id || !result->address || result->address % static_cast<std::size_t>(page) ||
        result->address > UINTPTR_MAX || extent - 1 > UINTPTR_MAX - result->address) {
        shared.host_context->is_uncertain = true;
        entry->is_uncertain = admission.should_retain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    entry->remote = result->id;
#ifdef GLIMMER_HAS_NCCL
    if (imported_origin.group)
        entry->nccl_cleanup = SourceCleanupBinding{
            imported_origin, shared.host_context->connection->session(), entry->remote};
#endif
    entry->address = result->address;
    const auto has_retiring_overlap = [&] {
        return std::ranges::any_of(shared.host_allocations, [&](const auto* other) {
            if (other == entry || !other->is_alive || !other->is_retiring || !other->host_mapping ||
                other->host_mapping->pointer == MAP_FAILED)
                return false;
            return entry->address >= other->address
                       ? entry->address - other->address < other->host_mapping->bytes
                       : other->address - entry->address < extent;
        });
    };
    while (has_retiring_overlap()) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (shared.host_context->is_uncertain) break;
    }
    if (shared.host_context->is_uncertain) {
        entry->is_uncertain = admission.should_retain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    // Only claim a previously empty CPU range; never overwrite application VM.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* requested = reinterpret_cast<void*>(static_cast<std::uintptr_t>(entry->address));
    auto* mapping = ::mmap(requested, extent, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_FIXED_NOREPLACE, descriptor.get(), 0);
    if (mapping == MAP_FAILED || mapping != requested) {
        if (mapping != MAP_FAILED) entry->host_mapping->pointer = mapping;
        const auto rollback = shared.release_host(exclusion, *entry);
        if (rollback != CUDA_SUCCESS) admission.should_retain = true;
        return rollback == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : rollback;
    }
    entry->host_mapping->pointer = mapping;
    entry->is_alive = true;
    admission.should_retain = true;
    if (!publish(output, mapping)) {
        const auto rollback = shared.release_host(exclusion, *entry);
        return rollback == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : rollback;
    }
    return CUDA_SUCCESS;
}
CUresult host_flags(State& shared, unsigned int* output, void* pointer) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
    auto* entry = shared.host_range(pointer, 0);
    if (!entry) return CUDA_ERROR_INVALID_VALUE;
    if (shared.host_context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    return publish(output, entry->host_flags) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult host_pointer(State& shared, CUdeviceptr* output, void* pointer, unsigned int flags) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    auto* entry = shared.host_range(pointer, 0);
    if (!entry) return CUDA_ERROR_INVALID_VALUE;
    if (shared.host_context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    return publish(output, static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer)))
               ? CUDA_SUCCESS
               : CUDA_ERROR_INVALID_VALUE;
}
template <typename Function>
Function original(const char* name) noexcept {
    // No virtual-resource operation uses this path. Only SDK initialization and
    // device/error metadata are allowed. Wrappers own their resolution or
    // observation cache; an absent late-load provider must not be cached.
    return reinterpret_cast<Function>(glimmer::interceptor::execution::native_driver_symbol(name));
}
CUresult ensure_driver_initialization(State& shared, Exclusion& exclusion) {
    if (shared.has_driver_initialization) return CUDA_SUCCESS;
    // Initialization metadata only. Never import the provider's context, and
    // do not cache a missing provider across later SDK loads.
    exclusion.unlock();
    const auto native = original<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
    CUcontext ignored{};
    const auto observed = native ? native(&ignored) : CUDA_ERROR_NOT_INITIALIZED;
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    exclusion.lock();
    if (!shared.has_driver_initialization && observed != CUDA_SUCCESS) return observed;
    shared.has_driver_initialization = true;
    return CUDA_SUCCESS;
}
struct ContextCreation {
    unsigned int flags{};
    CUdevice device{};
};
CUresult create_context(State& shared, Exclusion& exclusion, CUcontext* output,
                        ContextCreation options) {
    const auto flags = options.flags;
    const auto device = options.device;
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    if (device != 0) return CUDA_ERROR_NOT_SUPPORTED;
    const auto flags_status = context_flags_status(flags);
    if (flags_status != CUDA_SUCCESS) return flags_status;
    if (context_stack.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    context_stack.reserve(context_stack.size() + 1);
    auto* context = shared.allocate_entry(Kind::kContext, nullptr);
    if (!context) return CUDA_ERROR_OUT_OF_MEMORY;
    context->context = context;
    const auto opened = shared.open_context(exclusion, *context, flags);
    if (opened != CUDA_SUCCESS) return opened;
    context->context_flags = flags;
    context->is_alive = true;
    context_stack.push_back(context);
    if (!publish(output, reinterpret_cast<CUcontext>(context))) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}
ExecutionOwnedDescriptor immutable_module_file(std::span<const std::byte> bytes) noexcept {
    ExecutionOwnedDescriptor descriptor(
        ::memfd_create("glimmer-module", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (descriptor.get() < 0 ||
        ::ftruncate(descriptor.get(), static_cast<off_t>(bytes.size())) != 0)
        return ExecutionOwnedDescriptor{};
    std::size_t written{};
    while (written < bytes.size()) {
        const auto count = ::pwrite(descriptor.get(), bytes.data() + written,
                                    bytes.size() - written, static_cast<off_t>(written));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return ExecutionOwnedDescriptor{};
        written += static_cast<std::size_t>(count);
    }
    if (::fcntl(descriptor.get(), F_ADD_SEALS,
                F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0)
        return ExecutionOwnedDescriptor{};
    return descriptor;
}
bool host_ranges_overlap(const void* left, std::size_t left_bytes, const void* right,
                         std::size_t right_bytes) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(left);
    const auto b = reinterpret_cast<std::uintptr_t>(right);
    return !left || !right || left_bytes > UINTPTR_MAX - a || right_bytes > UINTPTR_MAX - b ||
           (a < b + right_bytes && b < a + left_bytes);
}
struct JitSubmission {
    struct ReadRange {
        const void* pointer{};
        std::size_t bytes{};
    };
    std::vector<ExecutionJitOption> options;
    std::vector<void*> pointers;
    std::vector<ReadRange> input_ranges;
    void** values{};
    bool has_byte_output(std::size_t index) const noexcept {
        const auto storage = execution_jit_storage(options[index].option);
        return storage == ExecutionJitStorage::kFloat || storage == ExecutionJitStorage::kLog;
    }
    void* output_pointer(std::size_t index) const noexcept {
        return options[index].option == 2 ? const_cast<void*>(pointer_slot(values, index))
                                          : pointers[index];
    }
    bool preflight_outputs() const noexcept {
        for (std::size_t index = 0; index < options.size(); ++index) {
            const auto& option = options[index];
            if (has_byte_output(index) &&
                !writable_host_range(output_pointer(index), option.data.size()))
                return false;
            if ((option.option == 1 || option.option == 4 || option.option == 6) &&
                !writable_host_range(const_cast<void*>(pointer_slot(values, index)), sizeof(void*)))
                return false;
        }
        return true;
    }
    bool publish_outputs(std::span<const ExecutionJitOption> actual) const noexcept {
        if (!preflight_outputs()) return false;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const auto& option = actual[index];
            if (has_byte_output(index) &&
                !self_copy(output_pointer(index), option.data.data(), option.data.size(), true))
                return false;
            if (option.option == 1 || option.option == 4 || option.option == 6) {
                const auto value = static_cast<std::uintptr_t>(option.value);
                if (!self_copy(const_cast<void*>(pointer_slot(values, index)), &value,
                               sizeof(value), true))
                    return false;
            }
        }
        return true;
    }
};
CUresult snapshot_jit_names(JitSubmission& submission, ExecutionJitOption& option,
                            std::size_t index, std::span<const std::uint32_t> ids) {
    const auto count_option = std::ranges::find(ids, option.option + 1);
    const auto count =
        count_option == ids.end()
            ? std::uintptr_t{}
            : reinterpret_cast<std::uintptr_t>(
                  submission.pointers[static_cast<std::size_t>(count_option - ids.begin())]);
    if (count > kExecutionMaxJitNames) return CUDA_ERROR_INVALID_VALUE;
    auto* pointer = submission.pointers[index];
    option.value = pointer ? 1 : 0;
    std::array<const char*, kExecutionMaxJitNames> names{};
    if (count && !self_copy(names.data(), pointer, count * sizeof(names[0]), false))
        return CUDA_ERROR_INVALID_VALUE;
    ExecutionPayloadWriter bytes(kExecutionMaxJitNamesBytes);
    if (!bytes.write_u32(static_cast<std::uint32_t>(count))) return CUDA_ERROR_OUT_OF_MEMORY;
    if (count) submission.input_ranges.push_back({pointer, count * sizeof(names[0])});
    for (std::size_t name = 0; name < count; ++name) {
        const auto value = owned_string(names[name], kExecutionMaxJitNameBytes);
        if (!value) return CUDA_ERROR_INVALID_VALUE;
        if (!bytes.write_blob(std::as_bytes(std::span{value->data(), value->size() + 1})))
            return CUDA_ERROR_OUT_OF_MEMORY;
        submission.input_ranges.push_back({names[name], value->size() + 1});
    }
    option.data = bytes.take();
    return CUDA_SUCCESS;
}
CUresult snapshot_jit(JitSubmission& submission, CUmodule* output, unsigned int count,
                      const CUjit_option* keys, void** values) {
    if (!count || count > kExecutionMaxJitOptions) return CUDA_ERROR_INVALID_VALUE;
    static_assert(sizeof(CUjit_option) == sizeof(std::uint32_t));
    std::array<std::uint32_t, kExecutionMaxJitOptions> ids{};
    if (!self_copy(ids.data(), keys, count * sizeof(ids[0]), false))
        return CUDA_ERROR_INVALID_VALUE;
    std::array<bool, kExecutionMaxJitOptions> seen{};
    for (const auto id : std::span(ids).first(count)) {
        if (execution_jit_storage(id) == ExecutionJitStorage::kUnsupported)
            return CUDA_ERROR_NOT_SUPPORTED;
        if (seen[id]) return CUDA_ERROR_INVALID_VALUE;
        seen[id] = true;
    }
    submission.pointers.resize(count);
    if (!self_copy(submission.pointers.data(), values, count * sizeof(void*), false))
        return CUDA_ERROR_INVALID_VALUE;
    submission.values = values;
    submission.options.reserve(count);
    CUmodule empty_handle{};
    const auto handle_bytes = std::as_bytes(std::span{&empty_handle, 1}).size();
    if (host_ranges_overlap(output, handle_bytes, keys, count * sizeof(CUjit_option)) ||
        host_ranges_overlap(output, handle_bytes, values, count * sizeof(void*)) ||
        host_ranges_overlap(keys, count * sizeof(CUjit_option), values, count * sizeof(void*)))
        return CUDA_ERROR_INVALID_VALUE;
    for (std::size_t index = 0; index < count; ++index) {
        ExecutionJitOption option{ids[index], 0, {}};
        const auto kind = execution_jit_storage(option.option);
        if (kind == ExecutionJitStorage::kScalar) {
            option.value = reinterpret_cast<std::uintptr_t>(submission.pointers[index]);
        } else if (kind == ExecutionJitStorage::kFloat) {
            // DataEx wall time occupies four bytes of the slot itself. Its
            // value is not a host address; retain client upper bytes locally.
            option.data.resize(sizeof(float));
            std::memcpy(option.data.data(), &submission.pointers[index], sizeof(float));
        } else if (kind == ExecutionJitStorage::kNames) {
            const auto copied =
                snapshot_jit_names(submission, option, index, std::span(ids).first(count));
            if (copied != CUDA_SUCCESS) return copied;
        } else {
            std::size_t bytes = 4;
            if (kind == ExecutionJitStorage::kLog) {
                const auto paired = std::find(ids.begin(), ids.begin() + count, option.option + 1);
                if (paired == ids.begin() + count) return CUDA_ERROR_INVALID_VALUE;
                const auto capacity = reinterpret_cast<std::uintptr_t>(
                    submission.pointers[static_cast<std::size_t>(paired - ids.begin())]);
                if (!capacity || capacity > kExecutionMaxJitLogBytes)
                    return CUDA_ERROR_INVALID_VALUE;
                bytes = capacity;
            }
            auto* pointer = submission.pointers[index];
            if (host_ranges_overlap(pointer, bytes, output, handle_bytes) ||
                host_ranges_overlap(pointer, bytes, keys, count * sizeof(CUjit_option)) ||
                host_ranges_overlap(pointer, bytes, values, count * sizeof(void*)))
                return CUDA_ERROR_INVALID_VALUE;
            for (std::size_t previous = 0; previous < submission.options.size(); ++previous)
                if (submission.has_byte_output(previous) &&
                    host_ranges_overlap(pointer, bytes, submission.output_pointer(previous),
                                        submission.options[previous].data.size()))
                    return CUDA_ERROR_INVALID_VALUE;
            option.data.resize(bytes);
            if (!self_copy(option.data.data(), pointer, bytes, false))
                return CUDA_ERROR_INVALID_VALUE;
        }
        submission.options.push_back(std::move(option));
    }
    for (const auto& range : submission.input_ranges) {
        if (host_ranges_overlap(range.pointer, range.bytes, output, handle_bytes) ||
            host_ranges_overlap(range.pointer, range.bytes, keys, count * sizeof(CUjit_option)) ||
            host_ranges_overlap(range.pointer, range.bytes, values, count * sizeof(void*)))
            return CUDA_ERROR_INVALID_VALUE;
        for (std::size_t index = 0; index < submission.options.size(); ++index)
            if (submission.has_byte_output(index) &&
                host_ranges_overlap(range.pointer, range.bytes, submission.output_pointer(index),
                                    submission.options[index].data.size()))
                return CUDA_ERROR_INVALID_VALUE;
    }
    return valid_execution_jit_input(submission.options) && submission.preflight_outputs()
               ? CUDA_SUCCESS
               : CUDA_ERROR_INVALID_VALUE;
}
CUresult load_module(State& shared, Exclusion& exclusion, CUmodule* output, const void* image,
                     const JitSubmission* jit = nullptr) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    std::array<std::byte, 4> signature{};
    if (!self_copy(signature.data(), image, signature.size(), false))
        return CUDA_ERROR_INVALID_IMAGE;
    constexpr std::array<std::byte, 4> elf{std::byte{0x7f}, std::byte{'E'}, std::byte{'L'},
                                           std::byte{'F'}};
    constexpr std::array<std::byte, 4> fatbin{std::byte{0x50}, std::byte{0xed}, std::byte{0x55},
                                              std::byte{0xba}};
    const bool is_binary = signature == elf || signature == fatbin;
    std::vector<std::byte> owned;
    std::optional<std::string> source;
    std::span<const std::byte> image_bytes;
    if (is_binary) {
        const auto prefix_bytes = signature == elf ? std::size_t{64} : std::size_t{16};
        owned.resize(prefix_bytes);
        if (!self_copy(owned.data(), image, owned.size(), false)) return CUDA_ERROR_INVALID_IMAGE;
        const auto tables = glimmer::transform::cuda_module_table_extent(owned);
        if (!tables) return CUDA_ERROR_INVALID_IMAGE;
        owned.resize(*tables);
        if (!self_copy(owned.data(), image, owned.size(), false)) return CUDA_ERROR_INVALID_IMAGE;
        const auto bytes = glimmer::transform::cuda_module_image_extent(owned);
        if (!bytes || *bytes > kMaxImageBytes) return CUDA_ERROR_INVALID_IMAGE;
        owned.resize(*bytes);
        if (!self_copy(owned.data(), image, owned.size(), false) ||
            glimmer::transform::cuda_module_image_extent(owned) != owned.size())
            return CUDA_ERROR_INVALID_IMAGE;
        image_bytes = owned;
    } else {
        source = owned_string(static_cast<const char*>(image), kMaxImageBytes);
        if (!source || source->empty()) return CUDA_ERROR_INVALID_IMAGE;
        image_bytes = std::as_bytes(std::span{source->data(), source->size()});
    }
    ExecutionPayloadWriter input;
    ExecutionOwnedDescriptor descriptor;
    auto operation = is_binary ? ExecutionOpcode::kLoadBinary : ExecutionOpcode::kLoadPtx;
    if (jit) {
        for (std::size_t index = 0; index < jit->options.size(); ++index)
            if (jit->has_byte_output(index) &&
                host_ranges_overlap(jit->output_pointer(index), jit->options[index].data.size(),
                                    image, image_bytes.size() + (is_binary ? 0 : 1)))
                return CUDA_ERROR_INVALID_VALUE;
        const auto options = encode_execution_jit_options(jit->options);
        descriptor = immutable_module_file(image_bytes);
        const auto kind =
            is_binary ? ExecutionModuleImageKind::kBinary : ExecutionModuleImageKind::kPtx;
        if (!options || descriptor.get() < 0 ||
            !input.write_u32(static_cast<std::uint32_t>(kind)) ||
            !input.write_u64(image_bytes.size()) || !input.write_blob(*options))
            return CUDA_ERROR_OUT_OF_MEMORY;
        operation = ExecutionOpcode::kLoadModuleJit;
    } else if (image_bytes.size() > kExecutionMaxPayloadBytes - sizeof(std::uint32_t)) {
        descriptor = immutable_module_file(image_bytes);
        const auto kind =
            is_binary ? ExecutionModuleImageKind::kBinary : ExecutionModuleImageKind::kPtx;
        if (descriptor.get() < 0 || !input.write_u32(static_cast<std::uint32_t>(kind)) ||
            !input.write_u64(image_bytes.size()))
            return CUDA_ERROR_OUT_OF_MEMORY;
        operation = ExecutionOpcode::kLoadModuleFile;
    } else if (!input.write_blob(image_bytes)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    auto* entry = shared.allocate_entry(Kind::kModule, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result =
        shared.call(exclusion, *context, operation, std::move(input), descriptor.get());
    const auto malformed = [&] {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    };
    if (!result || result->status < 0 || result->status > 999 || result->address ||
        result->parameter_bytes || !result->arguments.empty() || (!jit && !result->data.empty()))
        return malformed();
    std::optional<std::vector<ExecutionJitOption>> actual_jit;
    if (jit && result->original_status) {
        actual_jit = decode_execution_jit_options(result->data);
        if (!actual_jit || !execution_jit_output_matches(jit->options, *actual_jit))
            return malformed();
    } else if (!result->data.empty())
        return malformed();
    if (result->status != CUDA_SUCCESS) {
        if (result->id) return malformed();
        if (result->disposition == ExecutionWireDisposition::kRejected) {
            if (result->original_status) {
                // Native creation may have succeeded before private preparation
                // failed. A rejected carrier is not a no-resource proof.
                context->is_uncertain = true;
                if (*result->original_status < 0 || *result->original_status > 999)
                    return CUDA_ERROR_UNKNOWN;
            }
            if (actual_jit && !jit->publish_outputs(*actual_jit))
                return static_cast<CUresult>(result->status);
            return static_cast<CUresult>(result->status);
        }
        if (result->disposition != ExecutionWireDisposition::kComplete ||
            result->original_status != result->status)
            return malformed();
        if (actual_jit)
            (void)jit->publish_outputs(*actual_jit);  // Preserve the proven original error.
        return static_cast<CUresult>(result->status);
    }
    if (result->disposition != ExecutionWireDisposition::kComplete ||
        result->original_status != CUDA_SUCCESS || !result->id ||
        std::ranges::any_of(shared.entries, [&](const auto& existing) {
            return existing->kind == Kind::kModule && existing->is_alive &&
                   existing->context == context && existing->remote == result->id;
        }))
        return malformed();
    entry->remote = result->id;
    entry->is_alive = true;
    const auto handle = reinterpret_cast<CUmodule>(entry);
    if (actual_jit && (!writable_host_range(output, std::as_bytes(std::span{&handle, 1}).size()) ||
                       !jit->publish_outputs(*actual_jit)))
        return CUDA_ERROR_INVALID_VALUE;  // Known module stays owned by its context.
    return publish(output, handle) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult get_function(State& shared, Exclusion& exclusion, CUfunction* output, CUmodule module,
                      const char* name) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    auto* parent = shared.find(module, Kind::kModule);
    if (!context || !parent || parent->context != context || parent->is_retiring)
        return CUDA_ERROR_INVALID_HANDLE;
    if (parent->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin parent_use(parent->active_calls, exclusion);
    const auto symbol = owned_string(name, 4096);
    if (!symbol || symbol->empty()) return CUDA_ERROR_INVALID_VALUE;
    ExecutionPayloadWriter input;
    if (!input.write_u64(parent->remote) || !input.write_string(*symbol))
        return CUDA_ERROR_OUT_OF_MEMORY;
    auto result = shared.call(exclusion, *context, ExecutionOpcode::kGetFunction, std::move(input));
    if (status(result) != CUDA_SUCCESS) return status(result);
    if (result->parameter_bytes > kExecutionMaxPayloadBytes) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    for (const auto& existing : shared.entries) {
        if (existing->kind != Kind::kFunction || !existing->is_alive ||
            existing->context != context || existing->remote != result->id)
            continue;
        if (existing->parent != parent->remote || existing->bytes != result->parameter_bytes ||
            !std::equal(existing->arguments.begin(), existing->arguments.end(),
                        result->arguments.begin(), result->arguments.end(),
                        [](const auto& a, const auto& b) {
                            return a.offset == b.offset && a.bytes == b.bytes;
                        })) {
            context->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        return publish(output, reinterpret_cast<CUfunction>(existing.get()))
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    }
    if (!result->id) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    auto* entry = shared.allocate_entry(Kind::kFunction, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    entry->remote = result->id;
    entry->parent = parent->remote;
    entry->bytes = result->parameter_bytes;
    entry->arguments = std::move(result->arguments);
    entry->is_alive = true;
    return publish(output, reinterpret_cast<CUfunction>(entry)) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult function_control_locked(State& shared, Exclusion& exclusion,
                                 ExecutionFunctionControl operation, CUfunction handle,
                                 std::uint32_t argument, int value, int* output = nullptr,
                                 ExecutionFunctionAttributes* attributes = nullptr) {
    const bool is_query = operation == ExecutionFunctionControl::kGetAttribute;
    const bool is_aggregate = operation == ExecutionFunctionControl::kGetAttributes;
    if ((is_query && !writable(output)) || (is_aggregate && !attributes))
        return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    auto* function = shared.find(handle, Kind::kFunction);
    if (!context || !function || function->context != context) return CUDA_ERROR_INVALID_HANDLE;
    const auto parent = std::ranges::find_if(shared.entries, [&](const auto& entry) {
        return entry->kind == Kind::kModule && entry->is_alive && entry->context == context &&
               entry->remote == function->parent;
    });
    if (parent == shared.entries.end() || (*parent)->is_retiring || function->is_retiring)
        return CUDA_ERROR_INVALID_HANDLE;
    auto* module = parent->get();
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (module->is_uncertain || function->is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (context->active_calls >= kMaxHandles - 1 || context->active_mutations >= kMaxHandles - 1 ||
        module->active_calls == kMaxHandles || function->active_calls == kMaxHandles)
        return CUDA_ERROR_OUT_OF_MEMORY;
    // Keep both public identities alive across unlocked transport. The
    // connection preserves issue order; the owner defers mutations until
    // physical work retires, without adding a drain to original queries.
    CounterPin active(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    CounterPin module_use(module->active_calls, exclusion);
    CounterPin function_use(function->active_calls, exclusion);
    const auto epoch = context->epoch;
    ExecutionPayloadWriter input;
    if (!input.write_u32(static_cast<std::uint32_t>(operation)) ||
        !input.write_u64(function->remote) || !input.write_u32(argument) ||
        !input.write_u32(std::bit_cast<std::uint32_t>(value)))
        return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply =
        shared.call(exclusion, *context, ExecutionOpcode::kFunctionControl, std::move(input));
    const auto invalid = [&] {
        context->is_uncertain = module->is_uncertain = function->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    };
    if (!reply || reply->parameter_bytes > 1 || reply->address ||
        ((!is_aggregate || reply->status ||
          reply->disposition != ExecutionWireDisposition::kComplete || reply->parameter_bytes) &&
         !reply->data.empty()) ||
        !reply->arguments.empty() || reply->status < 0 || reply->status > 999 ||
        (reply->original_status && (*reply->original_status < 0 || *reply->original_status > 999)))
        return invalid();
    if (reply->parameter_bytes)
        context->is_uncertain = module->is_uncertain = function->is_uncertain = true;
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (!reply->status || reply->id || (reply->original_status && !reply->parameter_bytes))
            return invalid();
        return static_cast<CUresult>(reply->status);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || reply->parameter_bytes ||
        !reply->original_status || reply->status != *reply->original_status ||
        (!is_query && reply->id) || (is_query && reply->id > UINT32_MAX) ||
        (reply->status && reply->id) || context->is_uncertain || !context->is_alive ||
        context->epoch != epoch || !module->is_alive || !function->is_alive)
        return invalid();
    if (reply->status) return static_cast<CUresult>(reply->status);
    if (is_aggregate) {
        ExecutionFunctionAttributes snapshot;
        ExecutionPayloadReader reader(reply->data);
        if (reply->data.size() != kExecutionFunctionAttributesBytes ||
            !read_execution_function_attributes(reader, snapshot) || !reader.empty() ||
            std::ranges::any_of(snapshot.sizes, [](auto bytes) {
                return bytes > static_cast<std::uint64_t>(std::numeric_limits<int>::max());
            }))
            return invalid();
        *attributes = snapshot;
        return CUDA_SUCCESS;
    }
    const auto result = std::bit_cast<int>(static_cast<std::uint32_t>(reply->id));
    return !is_query || publish(output, result) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult function_control(ExecutionFunctionControl operation, CUfunction handle,
                          std::uint32_t argument, int value, int* output = nullptr) noexcept {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return function_control_locked(shared, exclusion, operation, handle, argument, value,
                                       output);
    });
}
CUresult get_global(State& shared, Exclusion& exclusion, Entry*& output, Entry* module,
                    std::string_view name) {
    auto* context = shared.context();
    if (!context || !module || !module->is_alive || module->context != context ||
        module->is_retiring)
        return CUDA_ERROR_INVALID_HANDLE;
    if (module->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin module_use(module->active_calls, exclusion);
    ExecutionPayloadWriter input;
    if (!input.write_u64(module->remote) || !input.write_string(name))
        return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply =
        shared.call(exclusion, *context, ExecutionOpcode::kGetGlobal, std::move(input));
    if (status(reply) != CUDA_SUCCESS) return status(reply);
    if (!reply->id || (reply->id & kExecutionProcessAllocationTag) || !reply->address ||
        !reply->parameter_bytes || reply->parameter_bytes - 1 > UINT64_MAX - reply->address ||
        !module->is_alive) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    for (const auto& entry : shared.entries) {
        if (entry->kind == Kind::kModuleGlobal && entry->is_alive && entry->context == context &&
            entry->remote == reply->id) {
            if (entry->parent != module->remote || entry->address != reply->address ||
                entry->bytes != reply->parameter_bytes) {
                context->is_uncertain = true;
                return CUDA_ERROR_UNKNOWN;
            }
            output = entry.get();
            return CUDA_SUCCESS;
        }
    }
    CounterPin publishing(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    const auto epoch = context->epoch;
    const auto overlaps = [&](const auto& entry) {
        return (entry->kind == Kind::kAllocation || entry->kind == Kind::kModuleGlobal ||
                entry->kind == Kind::kProcessAllocation) &&
               entry->is_alive &&
               (entry->context == context || entry->kind == Kind::kProcessAllocation) &&
               (reply->address >= entry->address
                    ? reply->address - entry->address < entry->bytes
                    : entry->address - reply->address < reply->parameter_bytes);
    };
    while (std::ranges::any_of(shared.entries, [&](const auto& entry) {
        return overlaps(entry) && shared.range_is_retiring(*entry);
    })) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    if (std::ranges::any_of(shared.entries, overlaps)) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    auto* entry = shared.allocate_entry(Kind::kModuleGlobal, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    entry->parent = module->remote;
    entry->remote = reply->id;
    entry->address = reply->address;
    entry->bytes = reply->parameter_bytes;
    entry->is_alive = true;
    output = entry;
    return CUDA_SUCCESS;
}
CUresult destroy_resource(State& shared, Exclusion& exclusion, const void* handle, Kind kind,
                          ExecutionOpcode operation) {
    auto* context = shared.context();
    auto* entry = shared.find(handle, kind);
    if (!context || !entry || entry->context != context || entry->is_retiring)
        return CUDA_ERROR_INVALID_HANDLE;
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry->remote)) return CUDA_ERROR_OUT_OF_MEMORY;
    FlagPin retiring(entry->is_retiring, exclusion);
    while ((kind == Kind::kModule || kind == Kind::kAllocation || kind == Kind::kStream) &&
           entry->active_calls) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!entry->is_alive || !context->is_alive) return CUDA_ERROR_INVALID_HANDLE;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    const auto result = shared.call(exclusion, *context, operation, std::move(input));
    if (status(result) == CUDA_SUCCESS) {
        entry->is_alive = false;
        if (kind == Kind::kModule)
            for (auto& child : shared.entries)
                if ((child->kind == Kind::kFunction || child->kind == Kind::kModuleGlobal) &&
                    child->context == context && child->parent == entry->remote)
                    child->is_alive = false;
    }
    return status(result);
}
CUresult memory_allocate(State& shared, Exclusion& exclusion, CUdeviceptr* output,
                         std::size_t bytes) {
    if (!writable(output) || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    auto* entry = shared.allocate_entry(Kind::kAllocation, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter input;
    if (!input.write_u64(bytes)) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result =
        shared.call(exclusion, *context, ExecutionOpcode::kAllocate, std::move(input));
    if (status(result) != CUDA_SUCCESS) return status(result);
    if (!result->id || (result->id & kExecutionProcessAllocationTag) || !result->address ||
        bytes > UINT64_MAX - result->address) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    CounterPin publishing(context->active_calls, exclusion);
    CounterPin publishing_work(context->active_mutations, exclusion);
    const auto epoch = context->epoch;
    const auto overlaps = [&](const auto& other) {
        return (other->kind == Kind::kAllocation || other->kind == Kind::kModuleGlobal ||
                other->kind == Kind::kProcessAllocation) &&
               other->is_alive &&
               (other->context == context || other->kind == Kind::kProcessAllocation) &&
               (result->address >= other->address ? result->address - other->address < other->bytes
                                                  : other->address - result->address < bytes);
    };
    while (std::ranges::any_of(shared.entries, [&](const auto& other) {
        return overlaps(other) && shared.range_is_retiring(*other);
    })) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    if (std::ranges::any_of(shared.entries, overlaps)) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    entry->remote = result->id;
    entry->address = result->address;
    entry->bytes = bytes;
    entry->is_alive = true;
    return publish(output, static_cast<CUdeviceptr>(entry->address)) ? CUDA_SUCCESS
                                                                     : CUDA_ERROR_UNKNOWN;
}
struct MemoryOperation {
    ExecutionMemoryKind kind{ExecutionMemoryKind::kCopy};
    CUdeviceptr destination{}, source{};
    std::size_t count{};
    std::uint32_t value{};
    CUstream stream{};
    bool is_asynchronous{};
};
CUresult resolve_owned_stream(State& shared, Exclusion& exclusion, CUstream handle, Entry*& output);
struct HostFunctionAdmission {
    State& shared;
    HostFunction& function;
    Exclusion& exclusion;
    bool was_submitted{}, is_finished{};
    ~HostFunctionAdmission() {
        if (is_finished || ::getpid() != kProcess || !exclusion.owns_lock()) return;
        if (was_submitted)
            shared.quarantine_host_function(function);
        else
            (void)shared.release_host_function(function);
    }
};
CUresult enqueue_host_function(State& shared, Exclusion& exclusion, CUstream handle,
                               cudaHostFn_t function, void* data) {
    if (!function) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (context->host_function_error != CUDA_SUCCESS) return context->host_function_error;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (shared.host_functions.size() >= kMaxHostFunctions ||
        context->active_calls >= kMaxHandles - 1 || context->active_mutations >= kMaxHandles - 1 ||
        context->host_function_pins >= kMaxHostFunctions ||
        (stream && stream->active_calls >= kMaxHandles) || context->submissions == UINT64_MAX)
        return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter enqueue;
    if (!enqueue.write_u64(stream ? stream->remote : 0)) return CUDA_ERROR_OUT_OF_MEMORY;
    auto pending = std::make_unique<HostFunction>();
    pending->origin = context;
    pending->stream = stream;
    pending->epoch = context->epoch;
    pending->function = function;
    pending->data = data;
    shared.ensure_host_function_progress();  // No admission if thread creation fails.
    auto& record = *pending;
    shared.host_functions.push_back(std::move(pending));
    ++context->active_mutations;
    ++context->host_function_pins;
    if (stream) ++stream->active_calls;
    HostFunctionAdmission admission{shared, record, exclusion};
    ++context->submissions;
    const auto reply = shared.rpc(exclusion, *context, ExecutionOpcode::kEnqueueHostFunction,
                                  enqueue.take(), -1, &admission.was_submitted, std::nullopt,
                                  &record.continuation, &record.enqueue_sequence);
    if (reply && reply->status > 0 && reply->status <= 999 && !reply->id && !reply->address &&
        !reply->parameter_bytes && !reply->original_status && reply->arguments.empty() &&
        reply->data.empty() &&
        (reply->disposition == ExecutionWireDisposition::kComplete ||
         reply->disposition == ExecutionWireDisposition::kRejected)) {
        const auto result = static_cast<CUresult>(reply->status);
        admission.is_finished = shared.release_host_function(record);
        return result;
    }
    if (!reply || reply->status || !reply->id || !host_function_receipt(*reply, reply->id) ||
        context->is_uncertain || !context->is_alive || context->epoch != record.epoch ||
        std::ranges::any_of(shared.host_functions, [&](const auto& previous) {
            return previous.get() != &record && previous->origin == context &&
                   previous->epoch == record.epoch && previous->remote == reply->id;
        }))
        return CUDA_ERROR_UNKNOWN;
    record.remote = reply->id;
    ExecutionPayloadWriter query, finish, exception;
    if (!query.write_u64(record.remote) || !finish.write_u64(record.remote) ||
        !finish.write_u32(1) || !exception.write_u64(record.remote) || !exception.write_u32(0))
        return CUDA_ERROR_OUT_OF_MEMORY;
    record.query_payload = query.take();
    record.finish_payload = finish.take();
    record.exception_payload = exception.take();
    record.phase = HostFunctionPhase::kWaiting;
    admission.is_finished = true;
    shared.host_function_wake.notify_one();
    return CUDA_SUCCESS;
}
CUresult execute_memory(State& shared, Exclusion& exclusion, const MemoryOperation& memory) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, memory.stream, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    std::size_t width{1};
    if (memory.kind == ExecutionMemoryKind::kFill16) width = 2;
    if (memory.kind == ExecutionMemoryKind::kFill32) width = 4;
    if (memory.count > std::numeric_limits<std::size_t>::max() / width ||
        memory.destination % width)
        return CUDA_ERROR_INVALID_VALUE;
    const auto bytes = memory.count * width;
    auto* destination = shared.range(memory.destination, bytes);
    auto* source =
        memory.kind == ExecutionMemoryKind::kCopy ? shared.range(memory.source, bytes) : nullptr;
    const bool has_null_destination = !bytes && !memory.destination;
    const bool has_null_source = !bytes && !memory.source;
    if ((!destination && !has_null_destination) || (destination && destination->is_retiring) ||
        (memory.kind == ExecutionMemoryKind::kCopy &&
         ((!source && !has_null_source) || (source && source->is_retiring))))
        return CUDA_ERROR_INVALID_VALUE;
    if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter input;
    if (!input.write_u32(static_cast<std::uint32_t>(memory.kind)) ||
        !input.write_u32(memory.is_asynchronous ? 1 : 0) ||
        !input.write_u64(destination ? destination->remote : 0) ||
        !input.write_u64(destination ? memory.destination - destination->address : 0) ||
        !input.write_u64(source ? source->remote : 0) ||
        !input.write_u64(source ? memory.source - source->address : 0) ||
        !input.write_u64(memory.count) || !input.write_u32(memory.value) ||
        !input.write_u64(stream ? stream->remote : 0))
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (shared.graph_capture) {
        if (!memory.is_asynchronous) return reject_graph_capture(shared, exclusion);
        const auto payload = input.take();
        return capture_graph_command(shared, exclusion, *context, stream,
                                     ExecutionGraphAction::kCaptureMemory, payload);
    }
    ++context->submissions;
    return status(shared.call(exclusion, *context, ExecutionOpcode::kMemory, std::move(input)));
}
struct Launch {
    CUfunction function{};
    std::array<unsigned int, 3> grid{}, block{};
    unsigned int shared{};
    CUstream stream{};
    void** arguments{};
    void** extra{};
};
CUresult copy_host_async(State& shared, Exclusion& exclusion, CUdeviceptr device, void* host,
                         std::size_t bytes, CUstream stream_handle, bool is_upload) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, stream_handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    auto* allocation = bytes ? shared.range(device, bytes) : nullptr;
    auto* host_allocation = bytes ? shared.host_range(host, bytes) : nullptr;
    if (bytes && !allocation) return CUDA_ERROR_INVALID_VALUE;
    if (bytes && !host_allocation)
        return CUDA_ERROR_NOT_SUPPORTED;  // Pageable/unowned registration is not pinned support.
    if (host_allocation && shared.host_context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (host_allocation && host_allocation->active_calls >= kMaxHandles)
        return CUDA_ERROR_OUT_OF_MEMORY;
    std::optional<CounterPin> host_use;
    if (host_allocation) host_use.emplace(host_allocation->active_calls, exclusion);
    ExecutionPayloadWriter input;
    if (!input.write_u64(host_allocation ? shared.host_context->connection->session() : 0) ||
        !input.write_u64(host_allocation ? host_allocation->remote : 0) ||
        !input.write_u64(host_allocation
                             ? reinterpret_cast<std::uintptr_t>(host) - host_allocation->address
                             : 0) ||
        !input.write_u64(allocation ? allocation->remote : 0) ||
        !input.write_u64(allocation ? device - allocation->address : 0) ||
        !input.write_u64(bytes) || !input.write_u64(stream ? stream->remote : 0) ||
        !input.write_u32(is_upload ? 1U : 0U))
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    ++context->submissions;
    return status(
        shared.call(exclusion, *context, ExecutionOpcode::kHostTransfer, std::move(input)));
}
enum class FrontierErrorPolicy { kIncludeHostFunctionError, kPhysicalOnly };
enum class AsyncNotificationDomain { kDriver, kRuntime };
CUresult native_frontier(
    State& shared, std::unique_lock<std::mutex>& exclusion, std::optional<CUstream> stream,
    bool should_wait, Entry* selected_context = nullptr,
    bool includes_submission_dependencies = false,
    FrontierErrorPolicy error_policy = FrontierErrorPolicy::kIncludeHostFunctionError,
    AsyncNotificationDomain notification_domain = AsyncNotificationDomain::kDriver) {
    auto* context = selected_context ? selected_context : shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    std::uint64_t stream_id{};
    Entry* captured_stream{};
    if (stream) {
        const auto resolved = resolve_owned_stream(shared, exclusion, *stream, captured_stream);
        if (resolved != CUDA_SUCCESS) return resolved;
        if (captured_stream && captured_stream->context != context)
            return CUDA_ERROR_INVALID_HANDLE;
        stream_id = captured_stream ? captured_stream->remote : 0;
    }
    if (should_wait) {
        const auto flushed = shared.flush_retired_async_launches(exclusion, *context);
        if (flushed != CUDA_SUCCESS) return flushed;
    }
    ExecutionPayloadWriter capture;
    if (!capture.write_u32(stream ? (includes_submission_dependencies ? 2 : 1) : 0) ||
        !capture.write_u64(stream_id))
        return CUDA_ERROR_OUT_OF_MEMORY;
    // Reserve the bounded local snapshot before creating a remote frontier.
    // Another caller may admit while capture yields, so reserve the full bound.
    std::vector<std::shared_ptr<AsyncLaunch>> asynchronous;
    asynchronous.reserve(kMaxAsyncLaunches);
    const auto captured =
        shared.call(exclusion, *context, ExecutionOpcode::kCaptureFrontier, std::move(capture));
    if (status(captured) != CUDA_SUCCESS) return status(captured);
    if (!captured->id || captured->original_status) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    const auto id = captured->id;
    const auto epoch = context->epoch;
    for (const auto& launch : shared.async_launches) {
        const bool in_stream =
            !stream || launch->stream == stream_id ||
            (includes_submission_dependencies &&
             ((stream_id == 0 && !launch->is_nonblocking) ||
              (launch->stream == 0 &&
               (!captured_stream || !(captured_stream->stream_flags & CU_STREAM_NON_BLOCKING)))));
        if (launch->context == context && launch->epoch == epoch && !launch->is_preparing &&
            launch->sequence && launch->sequence <= captured->address && in_stream)
            asynchronous.push_back(launch);
    }
    // Allocate both reusable request buffers before waiting. A query/release
    // never retries a native original, and CPU sleep conveys no GPU authority.
    ExecutionPayloadWriter query, release;
    if (!query.write_u64(id) || !release.write_u64(id)) {
        context->is_uncertain = true;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const auto query_bytes = query.take();
    auto release_bytes = release.take();
    for (;;) {
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (captured_stream && !captured_stream->is_alive) {
            const auto abandoned =
                shared.rpc(exclusion, *context, ExecutionOpcode::kReleaseFrontier, release_bytes);
            if (!abandoned || abandoned->status != 0) context->is_uncertain = true;
            return CUDA_ERROR_INVALID_HANDLE;
        }
        auto observed =
            shared.rpc(exclusion, *context, ExecutionOpcode::kQueryFrontier, query_bytes);
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (!observed || observed->disposition != ExecutionWireDisposition::kComplete ||
            observed->original_status || observed->parameter_bytes > 1) {
            context->is_uncertain = true;
            return observed && observed->status ? status(observed) : CUDA_ERROR_UNKNOWN;
        }
        auto completion = status(observed);
        shared.observe_async_launches(exclusion, asynchronous);
        for (const auto& launch : asynchronous) {
            if ((completion == CUDA_SUCCESS || completion == CUDA_ERROR_NOT_READY) &&
                launch->failure != CUDA_SUCCESS)
                completion = launch->failure;
        }
        const bool awaits_observer = std::ranges::any_of(asynchronous, [](const auto& launch) {
            return launch->is_polling && launch->failure == CUDA_SUCCESS;
        });
        if (should_wait && awaits_observer &&
            (completion == CUDA_SUCCESS || error_policy == FrontierErrorPolicy::kPhysicalOnly)) {
            // Success/physical cleanup must reconcile an outstanding metadata
            // observation. An independently proven error still returns promptly.
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
            continue;
        }
        const bool is_async_failure =
            completion != CUDA_SUCCESS && completion != CUDA_ERROR_NOT_READY &&
            std::ranges::any_of(asynchronous,
                                [&](const auto& launch) { return launch->failure == completion; });
        const bool waits_error_retirement = should_wait &&
                                            error_policy == FrontierErrorPolicy::kPhysicalOnly &&
                                            is_async_failure && !observed->parameter_bytes;
        if (!waits_error_retirement && (!should_wait || completion != CUDA_ERROR_NOT_READY)) {
            const auto released =
                shared.rpc(exclusion, *context, ExecutionOpcode::kReleaseFrontier, release_bytes);
            if (!released || released->disposition != ExecutionWireDisposition::kComplete ||
                released->original_status || released->status != 0) {
                context->is_uncertain = true;
                return completion == CUDA_SUCCESS || completion == CUDA_ERROR_NOT_READY
                           ? (released && released->status ? status(released) : CUDA_ERROR_UNKNOWN)
                           : completion;
            }
            if (error_policy == FrontierErrorPolicy::kPhysicalOnly && is_async_failure &&
                observed->parameter_bytes)
                return CUDA_SUCCESS;
            if (completion != CUDA_SUCCESS && completion != CUDA_ERROR_NOT_READY) {
                for (const auto& launch : asynchronous) {
                    // Driver observation must not consume a Runtime-origin
                    // notification. A Runtime wrapper writes only its caller's
                    // TLS after this return, never another thread's error state.
                    if (!launch->is_runtime) {
                        if (launch->failure == completion) launch->is_notified = true;
                    } else if (notification_domain == AsyncNotificationDomain::kRuntime &&
                               async_caller && launch->caller.lock() == async_caller) {
                        launch->synchronized_failure = completion;
                        if (launch->failure == completion) launch->is_notified = true;
                    }
                }
            }
            // Cleanup may omit the already reported CPU callback notification,
            // never an actual source/frontier failure or an uncertain release.
            return completion == CUDA_SUCCESS &&
                           error_policy == FrontierErrorPolicy::kIncludeHostFunctionError
                       ? context->host_function_error
                       : completion;
        }
        exclusion.unlock();
        std::this_thread::sleep_for(std::chrono::microseconds{50});
        if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
        exclusion.lock();
    }
}
CUresult copy_host_sync(State& shared, Exclusion& exclusion, CUdeviceptr device, void* host,
                        std::size_t bytes, bool is_upload,
                        AsyncNotificationDomain notification_domain) {
    const auto issued = copy_host_async(shared, exclusion, device, host, bytes, nullptr, is_upload);
    if (issued != CUDA_SUCCESS) return issued;
    return native_frontier(shared, exclusion, std::optional<CUstream>{nullptr}, true, nullptr,
                           false, FrontierErrorPolicy::kIncludeHostFunctionError,
                           notification_domain);
}
struct StagingRequest {
    void* host{};
    std::size_t bytes{};
    bool is_upload{};
};
struct StagingBatch {
    State& shared;
    Exclusion& exclusion;
    std::array<StagingAllocation*, 2> allocations{};
    bool has_transfer_started{};
    ~StagingBatch() {
        if (::getpid() != kProcess || !exclusion.owns_lock()) return;
        for (auto* allocation : allocations) {
            if (!allocation) continue;
            if (has_transfer_started && allocation->is_preparing)
                allocation->is_uncertain = true;  // No receipt survived stack unwinding.
            if (allocation->is_uncertain) {
                allocation->owner->is_uncertain = true;
            } else if (allocation->remote) {
                // Preparation may fail on the second endpoint before any
                // transfer is submitted. The already imported sibling has no
                // borrower and can be retired or cached normally.
                allocation->is_preparing = false;
                allocation->is_idle = true;
                allocation->should_release = !allocation->can_reuse;
            } else {
                std::erase_if(shared.staging_allocations,
                              [&](const auto& candidate) { return candidate.get() == allocation; });
            }
        }
        shared.staging_wake.notify_one();
    }
    void finish() {
        for (auto*& allocation : allocations) {
            if (allocation) allocation->is_preparing = false;
            allocation = nullptr;
        }
        shared.staging_wake.notify_one();
        // Private backing survives the SDK return until its independent owner
        // proves that every accepted physical lease has actually retired.
        shared.poll_staging();
    }
};
CUresult prepare_staging_batch(State& shared, Exclusion& exclusion, Entry& origin, Kind owner_kind,
                               std::span<const StagingRequest> requests, StagingBatch& batch) {
    if (requests.empty() || requests.size() > batch.allocations.size() ||
        (owner_kind != Kind::kStagingContext && owner_kind != Kind::kPageableContext))
        return CUDA_ERROR_INVALID_VALUE;
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto page_bytes = static_cast<std::size_t>(page);
    std::array<std::size_t, 2> extents{};
    std::size_t aggregate{};
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        if (!request.host) return CUDA_ERROR_INVALID_VALUE;
        if (request.bytes > kMaxStagingBytes) return CUDA_ERROR_OUT_OF_MEMORY;
        if (request.bytes > SIZE_MAX - page_bytes + 1) return CUDA_ERROR_INVALID_VALUE;
        const auto capacity = std::max(request.bytes, std::size_t{1});
        extents[index] = ((capacity - 1) / page_bytes + 1) * page_bytes;
        if (extents[index] > kMaxStagingBytes - aggregate ||
            extents[index] > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()))
            return CUDA_ERROR_OUT_OF_MEMORY;
        aggregate += extents[index];
    }
    auto*& owner =
        owner_kind == Kind::kPageableContext ? shared.pageable_context : shared.staging_context;
    if (!owner) {
        owner = shared.allocate_entry(owner_kind, nullptr);
        if (!owner) return CUDA_ERROR_OUT_OF_MEMORY;
        owner->context = owner;
        owner->is_alive = true;
    }
    const auto epoch = origin.epoch;
    bool can_reuse = shared.should_cache_staging;
#ifdef GLIMMER_HAS_NCCL
    can_reuse = can_reuse && (!nccl_caller || !nccl_caller->depth);
#endif
    std::array<StagingAllocation*, 2> selected{};
    for (;;) {
        shared.poll_staging();
        if (owner->is_uncertain || origin.is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (!origin.is_alive || origin.epoch != epoch || origin.is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        selected.fill(nullptr);
        std::size_t additional_bytes{}, additional_count{};
        for (std::size_t index = 0; index < requests.size(); ++index) {
            if (can_reuse) {
                const auto reusable =
                    std::ranges::find_if(shared.staging_allocations, [&](const auto& candidate) {
                        return candidate->owner == owner && candidate->origin == &origin &&
                               candidate->epoch == epoch && candidate->can_reuse &&
                               candidate->is_idle && !candidate->is_preparing &&
                               !candidate->is_uncertain && !candidate->should_release &&
                               !candidate->release_sequence &&
                               candidate->capacity_bytes >= requests[index].bytes &&
                               std::ranges::find(selected, candidate.get()) == selected.end();
                    });
                if (reusable != shared.staging_allocations.end()) selected[index] = reusable->get();
            }
            if (!selected[index]) {
                additional_bytes += extents[index];
                ++additional_count;
            }
        }
        std::size_t retained{};
        for (const auto& stage : shared.staging_allocations) retained += stage->mapping.bytes;
        if (additional_count <= kMaxHostAllocations - shared.staging_allocations.size() &&
            retained <= kMaxStagingBytes - additional_bytes)
            break;
        // Reserve both endpoints together. Never retain one preparing half
        // while waiting for another caller to free the other half of the quota.
        for (auto& candidate : shared.staging_allocations) {
            if (candidate->is_uncertain || candidate->owner->is_uncertain)
                return CUDA_ERROR_UNKNOWN;
            if (!candidate->is_preparing) candidate->should_release = true;
        }
        shared.staging_wake.notify_one();
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
    std::array<std::unique_ptr<StagingAllocation>, 2> local;
    std::array<ExecutionOwnedDescriptor, 2> descriptors;
    shared.staging_allocations.reserve(shared.staging_allocations.size() + requests.size());
    for (std::size_t index = 0; index < requests.size(); ++index) {
        if (selected[index]) continue;
        auto allocation = std::make_unique<StagingAllocation>();
        allocation->origin = &origin;
        allocation->owner = owner;
        allocation->epoch = epoch;
        allocation->mapping.bytes = extents[index];
        allocation->capacity_bytes = std::max(requests[index].bytes, std::size_t{1});
        allocation->can_reuse = can_reuse;
        descriptors[index] = ExecutionOwnedDescriptor(
            ::memfd_create("glimmer-staging", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (descriptors[index].get() < 0 ||
            ::ftruncate(descriptors[index].get(), static_cast<off_t>(extents[index])) != 0 ||
            ::fcntl(descriptors[index].get(), F_ADD_SEALS,
                    F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0)
            return CUDA_ERROR_OUT_OF_MEMORY;
        allocation->mapping.pointer = ::mmap(nullptr, extents[index], PROT_READ | PROT_WRITE,
                                             MAP_SHARED, descriptors[index].get(), 0);
        if (allocation->mapping.pointer == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
        selected[index] = allocation.get();
        local[index] = std::move(allocation);
    }
    // All storage and vector capacity exist before publishing any reservation.
    // No unlock occurs until both endpoints are visible and marked preparing.
    for (std::size_t index = 0; index < requests.size(); ++index) {
        selected[index]->is_preparing = true;
        selected[index]->is_idle = false;
        batch.allocations[index] = selected[index];
        if (local[index]) shared.staging_allocations.push_back(std::move(local[index]));
    }
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        if (request.is_upload
                ? !self_copy(selected[index]->mapping.pointer, request.host, request.bytes, false)
                : !writable_host_range(request.host, request.bytes))
            return CUDA_ERROR_INVALID_VALUE;
    }
    const auto opened = shared.open_context(exclusion, *owner, CU_CTX_MAP_HOST);
    if (opened != CUDA_SUCCESS) return opened;
    shared.ensure_staging_progress();
    for (std::size_t index = 0; index < requests.size(); ++index) {
        auto& stage = *selected[index];
        if (stage.remote) continue;
        ExecutionPayloadWriter input;
        if (!input.write_u64(stage.capacity_bytes) || !input.write_u64(extents[index]) ||
            !input.write_u32(0))
            return CUDA_ERROR_OUT_OF_MEMORY;
        stage.is_uncertain = true;
        ExecutionSourceContinuation imported_origin;
        const auto imported =
            shared.call(exclusion, *owner, ExecutionOpcode::kImportHostMemory, std::move(input),
                        descriptors[index].get(), std::nullopt, &imported_origin);
        if (status(imported) != CUDA_SUCCESS) {
            ExecutionPayloadReader ownership(imported ? std::span<const std::byte>{imported->data}
                                                      : std::span<const std::byte>{});
            std::uint32_t quarantine{};
            stage.is_uncertain =
                !imported || !ownership.read_u32(quarantine) || !ownership.empty() || quarantine;
            if (owner_kind == Kind::kPageableContext && imported &&
                (imported->status <= 0 || imported->status > 999 || imported->original_status ||
                 imported->id || imported->address || imported->parameter_bytes ||
                 !imported->arguments.empty() ||
                 imported->disposition != ExecutionWireDisposition::kRejected))
                stage.is_uncertain = true;
            if (stage.is_uncertain) owner->is_uncertain = true;
            return stage.is_uncertain && owner_kind == Kind::kPageableContext ? CUDA_ERROR_UNKNOWN
                                                                              : status(imported);
        }
        stage.remote = imported->id;
        const bool is_pageable = owner_kind == Kind::kPageableContext;
        ExecutionPayloadReader ownership(imported->data);
        std::uint32_t quarantine{};
        if (!stage.remote || (is_pageable ? imported->address != 0 : imported->address == 0) ||
            (is_pageable && (imported->disposition != ExecutionWireDisposition::kComplete ||
                             imported->original_status || imported->parameter_bytes ||
                             !imported->arguments.empty() || !ownership.read_u32(quarantine) ||
                             !ownership.empty() || quarantine))) {
            stage.is_uncertain = owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
#ifdef GLIMMER_HAS_NCCL
        if (imported_origin.group)
            stage.nccl_cleanup =
                SourceCleanupBinding{imported_origin, owner->connection->session(), stage.remote};
        if (stage.nccl_cleanup) stage.can_reuse = false;
#endif
        stage.is_uncertain = false;
    }
    return CUDA_SUCCESS;
}
CUresult copy_pageable_chunk(State& shared, Exclusion& exclusion, CUdeviceptr device, void* host,
                             std::size_t bytes, CUstream stream_handle, bool is_upload,
                             bool is_asynchronous, AsyncNotificationDomain notification_domain) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (!bytes)
        return copy_host_async(shared, exclusion, device, host, bytes, stream_handle, is_upload);
    if (!host) return CUDA_ERROR_INVALID_VALUE;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, stream_handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    auto* allocation = shared.range(device, bytes);
    if (!allocation || allocation->is_retiring) return CUDA_ERROR_INVALID_VALUE;
    if (context->active_mutations >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin publishing(context->active_mutations, exclusion);
    if (context->is_primary && !context->is_active) {
        const auto opened =
            shared.open_context(exclusion, *context, context->context_flags | CU_CTX_MAP_HOST);
        if (opened != CUDA_SUCCESS) return opened;
    }
    const auto epoch = context->epoch;
    if (!is_asynchronous && is_upload) {
        // The ordinary pageable upload stages only after its default-stream
        // prefix. Async uploads may stage immediately, as the native small-copy
        // control does; never force them to wait for their GPU predecessor.
        const auto stopped = native_frontier(
            shared, exclusion, std::optional<CUstream>{nullptr}, true, nullptr, true,
            FrontierErrorPolicy::kIncludeHostFunctionError, notification_domain);
        if (stopped != CUDA_SUCCESS) return stopped;
    }
    StagingBatch batch{shared, exclusion};
    const std::array requests{StagingRequest{host, bytes, is_upload}};
    const auto prepared =
        prepare_staging_batch(shared, exclusion, *context, Kind::kStagingContext, requests, batch);
    if (prepared != CUDA_SUCCESS) return prepared;
    auto* stage = batch.allocations.front();
    CUresult result = CUDA_ERROR_INVALID_VALUE;
    if (allocation->is_alive && !allocation->is_retiring &&
        (!stream || (stream->is_alive && !stream->is_retiring)) && context->is_alive &&
        context->epoch == epoch && !context->is_uncertain) {
        ExecutionPayloadWriter transfer;
        if (context->submissions == UINT64_MAX ||
            !transfer.write_u64(stage->owner->connection->session()) ||
            !transfer.write_u64(stage->remote) || !transfer.write_u64(0) ||
            !transfer.write_u64(allocation->remote) ||
            !transfer.write_u64(device - allocation->address) || !transfer.write_u64(bytes) ||
            !transfer.write_u64(stream ? stream->remote : 0) ||
            !transfer.write_u32(is_upload ? 1U : 0U))
            result = CUDA_ERROR_OUT_OF_MEMORY;
        else {
            ++context->submissions;
            batch.has_transfer_started = true;
            result = status(shared.call(exclusion, *context, ExecutionOpcode::kStagedHostTransfer,
                                        std::move(transfer)));
            if (result == CUDA_SUCCESS && !is_upload &&
                !self_copy(host, stage->mapping.pointer, bytes, true))
                result = CUDA_ERROR_INVALID_VALUE;
        }
    }
    batch.finish();
    return result;
}
CUresult copy_pageable(State& shared, Exclusion& exclusion, CUdeviceptr device, void* host,
                       std::size_t bytes, CUstream stream, bool is_upload, bool is_asynchronous,
                       AsyncNotificationDomain notification_domain) {
    if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
    const auto address = reinterpret_cast<std::uintptr_t>(host);
    if (!shared.range(device, bytes) || !host || bytes > UINTPTR_MAX - address ||
        bytes > std::numeric_limits<CUdeviceptr>::max() - device)
        return CUDA_ERROR_INVALID_VALUE;
    for (std::size_t offset = 0; offset < bytes;) {
        const auto count = std::min(bytes - offset, kMaxStagingBytes);
        // Checked numeric range arithmetic, never an unchecked application dereference.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* chunk = reinterpret_cast<void*>(address + offset);
        const auto result =
            copy_pageable_chunk(shared, exclusion, device + offset, chunk, count, stream, is_upload,
                                is_asynchronous || offset != 0, notification_domain);
        if (result != CUDA_SUCCESS) return result;
        offset += count;
    }
    return CUDA_SUCCESS;
}
CUresult transfer_host(
    State& shared, Exclusion& exclusion, CUdeviceptr device, void* host, std::size_t bytes,
    CUstream stream, bool is_upload, bool is_asynchronous,
    AsyncNotificationDomain notification_domain = AsyncNotificationDomain::kDriver) {
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    if (!bytes || shared.host_range(host, bytes))
        return is_asynchronous
                   ? copy_host_async(shared, exclusion, device, host, bytes, stream, is_upload)
                   : copy_host_sync(shared, exclusion, device, host, bytes, is_upload,
                                    notification_domain);
    const auto address = reinterpret_cast<std::uintptr_t>(host);
    for (const auto* allocation : shared.host_allocations)
        if (address >= allocation->address && address - allocation->address <= allocation->bytes)
            return allocation->is_uncertain ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_VALUE;
    return copy_pageable(shared, exclusion, device, host, bytes, stream, is_upload, is_asynchronous,
                         notification_domain);
}
CUresult unified_copy(State& shared, Exclusion& exclusion, CUdeviceptr destination,
                      CUdeviceptr source, std::size_t bytes, CUstream stream_handle,
                      bool is_asynchronous) {
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (!context->is_active || !context->connection) return CUDA_ERROR_INVALID_CONTEXT;
    // The legacy Async ABI accepts the per-thread special handle. Resolve it
    // through the same TLS/context proxy as other owned stream operations;
    // this does not select the separate _ptsz symbol's implicit-default ABI.
    if (destination > UINTPTR_MAX || source > UINTPTR_MAX) return CUDA_ERROR_INVALID_VALUE;
    if (context->active_calls >= kMaxHandles - 1 || context->active_mutations >= kMaxHandles - 1)
        return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin origin(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, stream_handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    const auto epoch = context->epoch;
    UnifiedEndpoint target, input;
    auto classified =
        shared.classify_unified_endpoint(static_cast<std::uintptr_t>(destination), bytes, target);
    if (classified != CUDA_SUCCESS) return classified;
    classified =
        shared.classify_unified_endpoint(static_cast<std::uintptr_t>(source), bytes, input);
    if (classified != CUDA_SUCCESS) return classified;
    if (!bytes && (target.wire.kind == ExecutionCopyEndpointKind::kPageable ||
                   input.wire.kind == ExecutionCopyEndpointKind::kPageable))
        return CUDA_ERROR_NOT_SUPPORTED;  // Arbitrary nonnull empty host spans are not qualified.
    const bool is_host_only = target.wire.kind != ExecutionCopyEndpointKind::kDevice &&
                              input.wire.kind != ExecutionCopyEndpointKind::kDevice;
    if (is_host_only && (!bytes || (destination < source + bytes && source < destination + bytes)))
        return CUDA_ERROR_NOT_SUPPORTED;  // Empty and overlapping H2H remain unqualified.

    const std::array resources{target.allocation, target.parent, input.allocation, input.parent,
                               stream};
    std::array<std::optional<CounterPin>, resources.size()> pins;
    for (std::size_t index = 0; index < resources.size(); ++index) {
        auto* resource = resources[index];
        if (!resource) continue;
        if (resource->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        pins[index].emplace(resource->active_calls, exclusion);
    }
    if (input.wire.kind == ExecutionCopyEndpointKind::kPageable &&
        (!is_asynchronous || is_host_only)) {
        // Native H2H consumes a pageable source after its stream predecessors,
        // including CPU callbacks, even through the Async entry. Preserve that
        // snapshot point without delaying asynchronous host-to-device staging.
        const auto source_stream = is_asynchronous ? stream_handle : nullptr;
        const auto stopped = native_frontier(
            shared, exclusion, std::optional<CUstream>{source_stream}, true, nullptr, true,
            FrontierErrorPolicy::kIncludeHostFunctionError, AsyncNotificationDomain::kDriver);
        if (stopped != CUDA_SUCCESS) return stopped;
    }
    std::array<StagingRequest, 2> requests;
    std::array<UnifiedEndpoint*, 2> staged_endpoints{};
    std::size_t request_count{};
    for (auto* endpoint : {&target, &input}) {
        if (endpoint->wire.kind != ExecutionCopyEndpointKind::kPageable) continue;
        // Classification checked the complete numeric span before any host
        // access. Conversion is only for bounded self-process reads/writes.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* pointer = reinterpret_cast<void*>(endpoint->address);
        requests[request_count] = {pointer, bytes, endpoint == &input};
        staged_endpoints[request_count++] = endpoint;
    }
    StagingBatch batch{shared, exclusion};
    if (request_count) {
        const auto prepared = prepare_staging_batch(
            shared, exclusion, *context, Kind::kPageableContext,
            std::span<const StagingRequest>{requests.data(), request_count}, batch);
        if (prepared != CUDA_SUCCESS) return prepared;
        for (std::size_t index = 0; index < request_count; ++index) {
            const auto* stage = batch.allocations[index];
            staged_endpoints[index]->wire = {ExecutionCopyEndpointKind::kPageable,
                                             stage->owner->connection->session(), stage->remote, 0};
        }
    }
    if (!context->is_alive || context->epoch != epoch || context->is_closing)
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    for (const auto* resource : resources) {
        if (!resource) continue;
        if (!resource->is_alive) return CUDA_ERROR_INVALID_VALUE;
        if (resource->is_uncertain) return CUDA_ERROR_UNKNOWN;
        // A concurrent destroy may already be waiting for our pin. Its pending
        // retirement cannot revoke this previously acquired operation lease.
    }
    ExecutionPayloadWriter payload;
    if (context->submissions == UINT64_MAX ||
        !write_execution_unified_copy(payload, {target.wire, input.wire, bytes,
                                                stream ? stream->remote : 0, is_asynchronous}))
        return CUDA_ERROR_OUT_OF_MEMORY;
    ++context->submissions;
    ExecutionSourceContinuation continuation;
    struct ReceiptGuard {
        State& shared;
        Exclusion& exclusion;
        Entry& context;
        StagingBatch& batch;
        ExecutionSourceContinuation& continuation;
        bool was_submitted{}, is_resolved{};
        ~ReceiptGuard() {
            if (::getpid() != kProcess || !exclusion.owns_lock() || !was_submitted || is_resolved)
                return;
            context.is_uncertain = true;
            batch.has_transfer_started = true;
            if (continuation.group) shared.quarantine_source_continuation(continuation);
        }
    } receipt{shared, exclusion, *context, batch, continuation};
    const auto reply =
        shared.rpc(exclusion, *context, ExecutionOpcode::kUnifiedCopy, payload.take(), -1,
                   &receipt.was_submitted, std::nullopt, &continuation);
    const bool has_envelope = reply && reply->status >= 0 && reply->status <= 999 &&
                              !reply->address && !reply->parameter_bytes &&
                              reply->arguments.empty() && reply->data.empty();
    const bool has_native_result =
        has_envelope && reply->disposition == ExecutionWireDisposition::kComplete && reply->id &&
        reply->original_status && *reply->original_status >= 0 && *reply->original_status <= 999 &&
        (*reply->original_status == reply->status ||
         (*reply->original_status == CUDA_SUCCESS && reply->status != CUDA_SUCCESS));
    const bool has_rejection = has_envelope &&
                               reply->disposition == ExecutionWireDisposition::kRejected &&
                               reply->status != CUDA_SUCCESS && !reply->original_status;
    if (!has_native_result && !has_rejection) {
        context->is_uncertain = true;
        if (continuation.group) shared.quarantine_source_continuation(continuation);
        // The batch destructor retains and poisons both private endpoints.
        // No fallback, replay, copyback or manufactured retirement is allowed.
        return CUDA_ERROR_UNKNOWN;
    }
    receipt.is_resolved = true;
    auto result = static_cast<CUresult>(reply->status);
    if (result == CUDA_SUCCESS) {
        for (std::size_t index = 0; index < request_count; ++index)
            if (!requests[index].is_upload &&
                !self_copy(requests[index].host, batch.allocations[index]->mapping.pointer, bytes,
                           true))
                result = CUDA_ERROR_INVALID_VALUE;
    }
    batch.finish();
    return result;
}
CUresult retire_primary(State& shared, std::unique_lock<std::mutex>& exclusion, CUdevice device,
                        bool is_release) {
    if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
    auto* primary = shared.primary;
    if (!primary) return is_release ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
    const auto opened = shared.await_context_transition(exclusion, *primary);
    if (opened != CUDA_SUCCESS) return opened;
    if (primary->is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (is_release && !primary->primary_references) return CUDA_ERROR_INVALID_CONTEXT;
    if (is_release && primary->primary_references > 1) {
        --primary->primary_references;
        return CUDA_SUCCESS;
    }
    if (primary->retirement_requests == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin retiring(primary->retirement_requests, exclusion);
    while (primary->is_active) {
        const auto submitted = primary->submissions;
        const auto stopped = native_frontier(shared, exclusion, std::nullopt, true, primary, false,
                                             FrontierErrorPolicy::kPhysicalOnly);
        if (stopped != CUDA_SUCCESS) return stopped;
        if (is_release && !primary->primary_references) return CUDA_ERROR_INVALID_CONTEXT;
        if (is_release && primary->primary_references > 1) {
            --primary->primary_references;
            return CUDA_SUCCESS;
        }
        if (submitted != primary->submissions) continue;
        if (primary->active_mutations) {
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
            continue;  // Recapture and recheck references before sealing close.
        }
        // Exclusion is held again: no accepted suffix can slip between the
        // stable logical frontier and the actual quiescent owner teardown.
        const auto result =
            shared.call(exclusion, *primary, ExecutionOpcode::kClose, ExecutionPayloadWriter{});
        if (status(result) != CUDA_SUCCESS) {
            primary->is_uncertain = true;
            return status(result);
        }
        for (auto& entry : shared.entries)
            if (entry.get() != primary && entry->context == primary) entry->is_alive = false;
        primary->connection.reset();
        primary->is_active = false;
        primary->is_closing = false;
        primary->submissions = 0;
        primary->runtime_symbols.clear();
        primary->runtime_kernels.clear();
        primary->has_runtime_bindings = false;
        if (primary->epoch == UINT64_MAX) {
            primary->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        ++primary->epoch;
    }
    primary->context_flags = 0;
    if (is_release) --primary->primary_references;
    return CUDA_SUCCESS;
}

CUresult set_context_flags(State& shared, Exclusion& exclusion, Entry& context,
                           unsigned int flags) {
    const auto checked = context_flags_status(flags);
    if (checked != CUDA_SUCCESS) return checked;
    const auto opened = shared.await_context_transition(exclusion, context);
    if (opened != CUDA_SUCCESS) return opened;
    if (context.is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (context.is_active) {
        ExecutionPayloadWriter input;
        if (!input.write_u32(
                flags | (context.is_primary ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U)))
            return CUDA_ERROR_OUT_OF_MEMORY;
        const auto result =
            shared.call(exclusion, context, ExecutionOpcode::kSetContextFlags, std::move(input));
        if (status(result) != CUDA_SUCCESS) return status(result);
    }
    context.context_flags = flags;
    return CUDA_SUCCESS;
}
CUresult launch_kernel(State& shared, Exclusion& exclusion, const Launch& launch,
                       bool is_runtime = false) {
    auto* context = shared.context();
    auto* function = shared.find(launch.function, Kind::kFunction);
    if (!context || !function || function->context != context) return CUDA_ERROR_INVALID_HANDLE;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, launch.stream, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (launch.arguments && launch.extra) return CUDA_ERROR_INVALID_VALUE;
    const auto parent = std::ranges::find_if(shared.entries, [&](const auto& entry) {
        return entry->kind == Kind::kModule && entry->is_alive && entry->context == context &&
               entry->remote == function->parent;
    });
    if (parent == shared.entries.end() || (*parent)->is_retiring || function->is_retiring ||
        (stream && stream->is_retiring))
        return CUDA_ERROR_INVALID_HANDLE;
    auto* module = parent->get();
    if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    std::vector<std::byte> parameters(static_cast<std::size_t>(function->bytes));
    if (launch.extra) {
        const void* buffer{};
        std::size_t bytes{};
        bool has_buffer{}, has_size{}, has_end{};
        for (std::size_t index = 0; index < 5;) {
            void* key{};
            if (!self_copy(&key, pointer_slot(launch.extra, index++), sizeof(key), false))
                return CUDA_ERROR_INVALID_VALUE;
            if (key == CU_LAUNCH_PARAM_END) {
                has_end = true;
                break;
            }
            void* value{};
            if (index == 5 ||
                !self_copy(&value, pointer_slot(launch.extra, index++), sizeof(value), false))
                return CUDA_ERROR_INVALID_VALUE;
            if (key == CU_LAUNCH_PARAM_BUFFER_POINTER && !has_buffer) {
                buffer = value;
                has_buffer = true;
            } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE && !has_size) {
                has_size = self_copy(&bytes, value, sizeof(bytes), false);
                if (!has_size) return CUDA_ERROR_INVALID_VALUE;
            } else
                return CUDA_ERROR_INVALID_VALUE;
        }
        if (!has_end || !has_buffer || !has_size || bytes != parameters.size() ||
            !self_copy(parameters.data(), buffer, bytes, false))
            return CUDA_ERROR_INVALID_VALUE;
    } else if (!copy_launch_arguments(function->arguments, launch.arguments, parameters))
        return CUDA_ERROR_INVALID_VALUE;
    ExecutionPayloadWriter input;
    if (!input.write_u64(function->remote) || !input.write_u64(stream ? stream->remote : 0))
        return CUDA_ERROR_OUT_OF_MEMORY;
    for (const auto dimension : launch.grid)
        if (!input.write_u32(dimension)) return CUDA_ERROR_OUT_OF_MEMORY;
    for (const auto dimension : launch.block)
        if (!input.write_u32(dimension)) return CUDA_ERROR_OUT_OF_MEMORY;
    if (!input.write_u32(launch.shared) || !input.write_blob(parameters))
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (shared.graph_capture) {
        const auto payload = input.take();
        return capture_graph_command(shared, exclusion, *context, stream,
                                     ExecutionGraphAction::kCaptureKernel, payload);
    }
    HostUsePins host_uses(shared, exclusion);
    if (context->active_calls >= kMaxHandles - 1 || context->active_mutations >= kMaxHandles - 1 ||
        module->active_calls == kMaxHandles || function->active_calls == kMaxHandles ||
        (stream && stream->active_calls == kMaxHandles))
        return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin active(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    CounterPin module_use(module->active_calls, exclusion);
    CounterPin function_use(function->active_calls, exclusion);
    std::optional<CounterPin> stream_use;
    if (stream) stream_use.emplace(stream->active_calls, exclusion);
    const auto epoch = context->epoch;
    const bool is_async = !shared.source_continuation(*context);
    bool has_admission_pressure{};
    const auto validate_owner = [&]() -> CUresult {
        if (!context->is_alive || context->epoch != epoch || context->is_closing ||
            (has_admission_pressure && context->retirement_requests))
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain || module->is_uncertain || function->is_uncertain ||
            (stream && stream->is_uncertain))
            return CUDA_ERROR_UNKNOWN;
        if (!module->is_alive || !function->is_alive || module->is_retiring ||
            function->is_retiring || (stream && (!stream->is_alive || stream->is_retiring)) ||
            (has_admission_pressure && module->retirement_requests))
            return CUDA_ERROR_INVALID_HANDLE;
        return CUDA_SUCCESS;
    };
    unsigned int backoff_us{50};
    std::shared_ptr<AsyncLaunch> admitted;
    if (is_async) {
        shared.retire_async_notifications();
        while (shared.async_launches.size() == kMaxAsyncLaunches) {
            has_admission_pressure = true;
            if (const auto checked = validate_owner(); checked != CUDA_SUCCESS) return checked;
            shared.reap_async_launches(exclusion, true);
            if (const auto status = validate_owner(); status != CUDA_SUCCESS) return status;
            if (shared.async_launches.size() != kMaxAsyncLaunches) break;
            const bool can_recover =
                std::ranges::any_of(shared.async_launches, [](const auto& record) {
                    return record->is_preparing || record->is_polling ||
                           (record->frontier && record->context->is_alive &&
                            !record->context->is_uncertain &&
                            record->context->epoch == record->epoch);
                });
            if (!can_recover) return CUDA_ERROR_OUT_OF_MEMORY;
            if (!yield_admission(exclusion, backoff_us)) return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (const auto status = validate_owner(); status != CUDA_SUCCESS) return status;
        shared.ensure_async_status_progress();
        if (!input.write_u32(kExecutionAsyncLaunchMagic)) return CUDA_ERROR_OUT_OF_MEMORY;
        if (!async_caller) async_caller = std::make_shared<const int>(0);
        admitted = std::make_shared<AsyncLaunch>();
        admitted->context = context;
        admitted->caller = async_caller;
        admitted->epoch = context->epoch;
        admitted->stream = stream ? stream->remote : 0;
        admitted->is_runtime = is_runtime;
        admitted->is_nonblocking = stream && (stream->stream_flags & CU_STREAM_NON_BLOCKING);
        shared.async_launches.push_back(admitted);
    }
    struct AdmissionGuard {
        State& state;
        Exclusion& lock;
        std::shared_ptr<AsyncLaunch> record;
        int exceptions{std::uncaught_exceptions()};
        AdmissionGuard(State& owner, Exclusion& exclusion, std::shared_ptr<AsyncLaunch> launch)
            : state(owner), lock(exclusion), record(std::move(launch)) {}
        AdmissionGuard(const AdmissionGuard&) = delete;
        AdmissionGuard& operator=(const AdmissionGuard&) = delete;
        ~AdmissionGuard() {
            if (!record || ::getpid() != kProcess || !lock.owns_lock()) return;
            if (std::uncaught_exceptions() != exceptions) {
                record->context->is_uncertain = true;
                record->failure = CUDA_ERROR_UNKNOWN;
            }
            record->is_preparing = false;
            if (!record->frontier && record->failure == CUDA_SUCCESS)
                std::erase(state.async_launches, record);
        }
    } guard{shared, exclusion, admitted};
    if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    ++context->submissions;
    const auto copied_payload = input.take();
    std::optional<ExecutionWireReply> result;
    for (;;) {
        if (const auto checked = validate_owner(); checked != CUDA_SUCCESS) return checked;
        result = shared.call_owned(exclusion, *context, ExecutionOpcode::kLaunch, copied_payload);
        if (!result || result->disposition != ExecutionWireDisposition::kAdmissionBusy) break;
        // Only a canonical authenticated pre-admission response permits retry.
        // The copied payload and preparing record remain owned; no GPU receipt
        // or public success exists yet. A real accepted command never retries.
        if (!admitted || result->status || result->original_status || result->id ||
            result->address || result->parameter_bytes || !result->arguments.empty() ||
            !result->data.empty()) {
            context->is_uncertain = true;
            if (admitted) admitted->failure = CUDA_ERROR_UNKNOWN;
            return CUDA_ERROR_UNKNOWN;
        }
        has_admission_pressure = true;
        if (const auto checked = validate_owner(); checked != CUDA_SUCCESS) return checked;
        // Service pressure can occur before the local tracker fills. Flush
        // proven retirements even below the ordinary metadata batch threshold,
        // otherwise a small service frontier budget can wait on this caller.
        shared.reap_async_launches(exclusion, true);
        if (const auto checked = validate_owner(); checked != CUDA_SUCCESS) return checked;
        if (!yield_admission(exclusion, backoff_us)) return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (admitted && result && result->disposition != ExecutionWireDisposition::kAdmissionAccepted &&
        (result->disposition != ExecutionWireDisposition::kRejected ||
         result->status == CUDA_SUCCESS || result->original_status || result->id ||
         result->address || result->parameter_bytes || !result->arguments.empty() ||
         !result->data.empty())) {
        // Only a clean pre-admission rejection may omit the exact frontier.
        // Legacy host/library results or malformed error carriers leave possible
        // accepted ownership unproven, even when their CUDA status is nonzero.
        context->is_uncertain = true;
        admitted->failure = CUDA_ERROR_UNKNOWN;
        admitted->is_notified = admitted->is_runtime;
        return CUDA_ERROR_UNKNOWN;
    }
    if (result && result->disposition == ExecutionWireDisposition::kAdmissionAccepted) {
        const bool duplicate = std::ranges::any_of(shared.async_launches, [&](const auto& other) {
            return other != admitted && other->context == context &&
                   other->epoch == context->epoch &&
                   ((other->sequence && other->sequence == result->id) ||
                    (other->frontier && other->frontier == result->address));
        });
        if (!admitted || result->status || result->original_status || !result->id ||
            !result->address || result->parameter_bytes || !result->arguments.empty() ||
            !result->data.empty() || duplicate) {
            context->is_uncertain = true;
            if (admitted) {
                admitted->failure = CUDA_ERROR_UNKNOWN;
                admitted->is_notified = admitted->is_runtime;
            }
            return CUDA_ERROR_UNKNOWN;
        }
        admitted->sequence = result->id;
        admitted->frontier = result->address;
        shared.request_async_status_progress();
    } else if (!result && admitted) {
        admitted->failure = CUDA_ERROR_UNKNOWN;
        // The Runtime wrapper reports this same error into its caller's TLS
        // immediately; retain ownership, not a second GetLastError notification.
        admitted->is_notified = admitted->is_runtime;
    }
    return status(result);
}

// Runtime metadata and virtual resources share State::mutex. No second lock or
// recursive public Driver wrapper participates in this execution domain.
constexpr std::size_t kMaxRuntimeBytes = std::size_t{2} * 1024 * 1024 * 1024;
constexpr std::size_t kMaxRuntimeMetadataBytes = std::size_t{64} * 1024 * 1024;
constexpr std::size_t kMaxRuntimeRegistrations = kMaxRuntimeMetadataBytes / sizeof(RuntimeKernel);
RuntimeImage* runtime_image(State& shared, void** handle) noexcept {
    const auto found = std::ranges::find_if(shared.runtime_images, [&](const auto& image) {
        return image.get() == reinterpret_cast<RuntimeImage*>(handle) && image->is_alive;
    });
    return found == shared.runtime_images.end() ? nullptr : found->get();
}
CUresult register_runtime_image(State& shared, void* wrapper, void*** output) {
    __fatBinC_Wrapper_t copied{};
    if (!self_copy(&copied, wrapper, sizeof(copied), false) || copied.magic != FATBINC_MAGIC ||
        (copied.version != FATBINC_VERSION && copied.version != FATBINC_LINK_VERSION) ||
        !copied.data)
        return CUDA_ERROR_INVALID_IMAGE;
    // v2 data is the final linked image. The separate host-only prelink array
    // is neither an executable input nor a pointer that can cross the wire.
    if (shared.runtime_images.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    std::array<std::byte, 16> prefix{};
    if (!self_copy(prefix.data(), copied.data, prefix.size(), false))
        return CUDA_ERROR_INVALID_IMAGE;
    constexpr std::array<std::byte, 4> signature{std::byte{0x50}, std::byte{0xed}, std::byte{0x55},
                                                 std::byte{0xba}};
    if (!std::equal(signature.begin(), signature.end(), prefix.begin()))
        return CUDA_ERROR_INVALID_IMAGE;
    const auto extent = glimmer::transform::cuda_module_image_extent(prefix);
    if (!extent || *extent > kMaxImageBytes) return CUDA_ERROR_INVALID_IMAGE;
    if (*extent > kMaxRuntimeBytes - shared.runtime_image_bytes) return CUDA_ERROR_OUT_OF_MEMORY;
    auto image = std::make_unique<RuntimeImage>();
    image->bytes.resize(*extent);
    if (!self_copy(image->bytes.data(), copied.data, *extent, false) ||
        glimmer::transform::cuda_module_image_extent(image->bytes) != extent)
        return CUDA_ERROR_INVALID_IMAGE;
    image->retained_bytes = *extent;
    auto* identity = image.get();
    shared.runtime_images.push_back(std::move(image));
    shared.runtime_image_bytes += *extent;
    *output = reinterpret_cast<void**>(identity);
    return CUDA_SUCCESS;
}
CUresult register_runtime_function(State& shared, void** handle, const void* host,
                                   const char* name) {
    auto* image = runtime_image(shared, handle);
    if (!image || image->is_closing || image->is_finalized) return CUDA_ERROR_INVALID_HANDLE;
    if (image->error != CUDA_SUCCESS) return image->error;
    const auto symbol = owned_string(name, 4096);
    if (!host || !symbol || symbol->empty()) {
        image->error = CUDA_ERROR_INVALID_VALUE;
        return image->error;
    }
    if (std::ranges::find(image->kernels, host, &RuntimeKernel::host) != image->kernels.end())
        return CUDA_SUCCESS;
    const auto bytes = symbol->size() + 1 + sizeof(RuntimeKernel);
    if (image->kernels.size() == kMaxRuntimeRegistrations ||
        bytes > kMaxRuntimeMetadataBytes - shared.runtime_metadata_bytes)
        return image->error = CUDA_ERROR_OUT_OF_MEMORY;
    image->kernels.push_back({host, *symbol});
    image->metadata_bytes += bytes;
    shared.runtime_metadata_bytes += bytes;
    try {
        for (const auto& context : shared.entries) {
            if (context->kind != Kind::kContext || !context->is_alive ||
                !context->has_runtime_bindings || context->runtime_kernels.contains(host))
                continue;
            if (context->runtime_kernels.size() == kMaxRuntimeRegistrations)
                return image->error = CUDA_ERROR_OUT_OF_MEMORY;
            context->runtime_kernels.emplace(
                host, RuntimeKernelBinding{image, image->kernels.size() - 1});
        }
    } catch (...) {
        return image->error = CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}
CUresult snapshot_runtime_bindings(State& shared, Entry& context) {
    if (context.has_runtime_bindings) return CUDA_SUCCESS;
    std::unordered_map<const void*, RuntimeSymbolBinding> symbols;
    std::unordered_map<const void*, RuntimeKernelBinding> kernels;
    for (const auto& image : shared.runtime_images) {
        if (!image->is_alive || image->is_closing) continue;
        for (std::size_t index = 0; index < image->kernels.size(); ++index) {
            const auto host = image->kernels[index].host;
            if (kernels.contains(host)) continue;
            if (kernels.size() == kMaxRuntimeRegistrations) return CUDA_ERROR_OUT_OF_MEMORY;
            kernels.emplace(host, RuntimeKernelBinding{image.get(), index});
        }
        for (std::size_t index = 0; index < image->variables.size(); ++index) {
            const auto host = image->variables[index].host;
            if (symbols.contains(host)) continue;
            if (symbols.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
            symbols.emplace(host, RuntimeSymbolBinding{image.get(), index});
        }
    }
    context.runtime_symbols = std::move(symbols);
    context.runtime_kernels = std::move(kernels);
    context.has_runtime_bindings = true;
    return CUDA_SUCCESS;
}
CUresult runtime_context(State& shared, Exclusion& exclusion, bool select_primary = false) {
    auto* current = shared.context();
    if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (current && !current->is_primary && !select_primary)
        return snapshot_runtime_bindings(shared, *current);
    if (!current && !context_stack.empty()) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (!shared.config) return CUDA_ERROR_NOT_INITIALIZED;
    auto* primary = shared.primary_entry();
    if (!primary) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto waited = shared.await_context_transition(exclusion, *primary);
    if (waited != CUDA_SUCCESS) return waited;
    if (primary->is_uncertain ||
        (!shared.runtime_owns_primary && primary->primary_references == UINT64_MAX))
        return CUDA_ERROR_UNKNOWN;
    if (context_stack.empty()) context_stack.reserve(1);
    if (!primary->is_active) {
        const auto result =
            shared.open_context(exclusion, *primary, primary->context_flags | CU_CTX_MAP_HOST,
                                ExecutionContextPurpose::kRuntimePrimary);
        if (result != CUDA_SUCCESS) return result;
    }
    if (!shared.runtime_owns_primary) {
        if (primary->primary_references == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
        ++primary->primary_references;
        shared.runtime_owns_primary = true;
    }
    if (context_stack.empty())
        context_stack.push_back(primary);
    else
        context_stack.back() = primary;
    return snapshot_runtime_bindings(shared, *primary);
}
CUresult runtime_module(State& shared, Exclusion& exclusion, RuntimeImage& image,
                        RuntimeBinding*& output) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    const auto epoch = context->epoch;
    const auto found = std::ranges::find_if(image.bindings, [&](const auto& entry) {
        return entry->context == context && entry->epoch == epoch &&
               (entry->is_loading || (entry->module && entry->module->is_alive));
    });
    RuntimeBinding* binding{};
    if (found == image.bindings.end()) {
        if (image.bindings.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        auto owned = std::make_unique<RuntimeBinding>();
        owned->context = context;
        owned->epoch = epoch;
        binding = owned.get();
        image.bindings.push_back(std::move(owned));
        FlagPin loading(binding->is_loading, exclusion);
        CUmodule module{};
        const auto loaded = load_module(shared, exclusion, &module, image.bytes.data());
        if (loaded != CUDA_SUCCESS) return loaded;
        binding->module = shared.find(module, Kind::kModule);
    } else {
        binding = found->get();
    }
    for (;;) {
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (image.is_closing || !image.is_alive) return CUDA_ERROR_INVALID_HANDLE;
        if (!binding->is_loading &&
            (!binding->module || !binding->module->is_alive || binding->module->is_retiring))
            return CUDA_ERROR_INVALID_HANDLE;
        if (!binding->is_loading) break;
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
    output = binding;
    return CUDA_SUCCESS;
}
CUresult select_runtime_kernel(State& shared, const void* host, RuntimeImage*& image,
                               const RuntimeKernel*& kernel) {
    auto* context = shared.context();
    if (!context || !context->has_runtime_bindings) return CUDA_ERROR_NOT_INITIALIZED;
    const auto selected = context->runtime_kernels.find(host);
    if (selected == context->runtime_kernels.end()) return CUDA_ERROR_NOT_FOUND;
    image = selected->second.image;
    if (!image || !image->is_alive || image->is_closing) return CUDA_ERROR_NOT_FOUND;
    if (image->error != CUDA_SUCCESS) return image->error;
    if (!image->is_finalized) return CUDA_ERROR_NOT_INITIALIZED;
    if (selected->second.kernel >= image->kernels.size()) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    kernel = &image->kernels[selected->second.kernel];
    return CUDA_SUCCESS;
}
Entry* cached_runtime_function(State& shared, const RuntimeImage& image, const void* host) {
    auto* context = shared.context();
    if (!context) return nullptr;
    for (const auto& binding : image.bindings) {
        if (binding->context != context || binding->epoch != context->epoch || !binding->module ||
            !binding->module->is_alive || binding->module->is_retiring)
            continue;
        const auto found = std::ranges::find_if(binding->functions, [&](const auto& function) {
            return function.host == host && function.function->is_alive;
        });
        if (found != binding->functions.end()) return found->function;
    }
    return nullptr;
}
CUresult runtime_function(State& shared, Exclusion& exclusion, const void* host,
                          CUfunction* output) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    RuntimeImage* image{};
    const RuntimeKernel* kernel{};
    const auto selected = select_runtime_kernel(shared, host, image, kernel);
    if (selected != CUDA_SUCCESS) return selected;
    const auto name = kernel->name;
    if (image->pending_lookups == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin lookup(image->pending_lookups, exclusion);
    const auto epoch = context->epoch;
    RuntimeBinding* binding{};
    const auto loaded = runtime_module(shared, exclusion, *image, binding);
    if (loaded != CUDA_SUCCESS) return loaded;
    for (;;) {
        if (!context->is_alive || context->epoch != epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (image->is_closing || !image->is_alive) return CUDA_ERROR_INVALID_HANDLE;
        if (!binding->is_loading && (!binding->module || !binding->module->is_alive))
            return CUDA_ERROR_INVALID_HANDLE;
        const auto found = std::ranges::find_if(binding->functions, [&](const auto& entry) {
            return entry.host == host && entry.function->is_alive;
        });
        if (found != binding->functions.end()) {
            *output = reinterpret_cast<CUfunction>(found->function);
            return CUDA_SUCCESS;
        }
        if (!binding->is_loading && !binding->is_preparing) break;
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (binding->functions.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    binding->functions.reserve(binding->functions.size() + 1);
    FlagPin preparing(binding->is_preparing, exclusion);
    const auto result = get_function(shared, exclusion, output,
                                     reinterpret_cast<CUmodule>(binding->module), name.c_str());
    if (result == CUDA_SUCCESS)
        binding->functions.push_back({host, shared.find(*output, Kind::kFunction)});
    if (image->is_closing) return CUDA_ERROR_INVALID_HANDLE;
    return result;
}
CUresult register_runtime_variable(State& shared, void** handle, const void* host, const char* name,
                                   int external, std::size_t bytes, int constant, int global) {
    auto* image = runtime_image(shared, handle);
    if (!image || image->is_closing || image->is_finalized) return CUDA_ERROR_INVALID_HANDLE;
    if (image->error != CUDA_SUCCESS) return image->error;
    const auto symbol = owned_string(name, 4096);
    if (!host || !symbol || symbol->empty() || external < 0 || external > 1 || constant < 0 ||
        constant > 1 || global < 0 || global > 1)
        return image->error = CUDA_ERROR_INVALID_VALUE;
    // Original CUDA 12.6 keeps the first association within one module. The
    // same coalesced host identity in another module is legitimate metadata.
    if (std::ranges::find(image->variables, host, &RuntimeVariable::host) != image->variables.end())
        return CUDA_SUCCESS;
    const auto retained = sizeof(RuntimeVariable) + symbol->size() + 1;
    if (image->variables.size() == kMaxHandles ||
        retained > kMaxRuntimeMetadataBytes - shared.runtime_metadata_bytes)
        return image->error = CUDA_ERROR_OUT_OF_MEMORY;
    image->variables.push_back({host, *symbol, bytes, external != 0, constant != 0, global != 0});
    image->metadata_bytes += retained;
    shared.runtime_metadata_bytes += retained;
    try {
        for (const auto& context : shared.entries) {
            if (context->kind != Kind::kContext || !context->is_alive ||
                !context->has_runtime_bindings || context->runtime_symbols.contains(host))
                continue;
            if (context->runtime_symbols.size() == kMaxHandles)
                return image->error = CUDA_ERROR_OUT_OF_MEMORY;
            context->runtime_symbols.emplace(
                host, RuntimeSymbolBinding{image, image->variables.size() - 1});
        }
    } catch (...) {
        return image->error = CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}
CUresult runtime_symbol(State& shared, Exclusion& exclusion, const void* host, Entry*& output,
                        Entry** module = nullptr) {
    auto* context = shared.context();
    if (!context || !context->has_runtime_bindings) return CUDA_ERROR_NOT_INITIALIZED;
    const auto selected = context->runtime_symbols.find(host);
    if (selected == context->runtime_symbols.end()) return CUDA_ERROR_NOT_FOUND;
    auto* image = selected->second.image;
    if (!image || !image->is_alive || image->is_closing) return CUDA_ERROR_NOT_FOUND;
    if (image->error != CUDA_SUCCESS) return image->error;
    if (!image->is_finalized) return CUDA_ERROR_NOT_INITIALIZED;
    if (selected->second.variable >= image->variables.size()) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    const auto variable = image->variables[selected->second.variable];
    if (image->pending_lookups >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin lookup(image->pending_lookups, exclusion);
    RuntimeBinding* binding{};
    const auto loaded = runtime_module(shared, exclusion, *image, binding);
    if (loaded != CUDA_SUCCESS) return loaded;
    for (;;) {
        if (!image->is_alive || image->is_closing) return CUDA_ERROR_INVALID_HANDLE;
        if (!context->is_alive || context->epoch != binding->epoch || context->is_closing)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        const auto cached = std::ranges::find(binding->symbols, host, &RuntimeResolvedSymbol::host);
        if (cached != binding->symbols.end() && cached->global->is_alive) {
            if (cached->global->is_retiring || !binding->module->is_alive ||
                binding->module->is_retiring)
                return CUDA_ERROR_INVALID_HANDLE;
            output = cached->global;
            if (module) *module = binding->module;
            return CUDA_SUCCESS;
        }
        if (!binding->is_preparing) break;
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (binding->symbols.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    binding->symbols.reserve(binding->symbols.size() + 1);
    FlagPin preparing(binding->is_preparing, exclusion);
    const auto result = get_global(shared, exclusion, output, binding->module, variable.name);
    if (result != CUDA_SUCCESS) return result;
    if (image->is_closing || !image->is_alive) return CUDA_ERROR_INVALID_HANDLE;
    if (variable.is_external ? variable.bytes > output->bytes : variable.bytes != output->bytes)
        return CUDA_ERROR_INVALID_VALUE;
    binding->symbols.push_back({host, output});
    if (module) *module = binding->module;
    return CUDA_SUCCESS;
}
CUresult unregister_runtime_image(State& shared, std::unique_lock<std::mutex>& exclusion,
                                  void** handle) {
    auto* image = runtime_image(shared, handle);
    if (!image || image->is_closing) return CUDA_ERROR_INVALID_HANDLE;
    image->is_closing = true;
    while (image->pending_lookups)
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
    for (const auto& binding : image->bindings) {
        if (!binding->module || !binding->module->is_alive) continue;
        if (binding->module->is_retiring) return CUDA_ERROR_INVALID_HANDLE;
        if (binding->module->retirement_requests == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin retiring(binding->module->retirement_requests, exclusion);
        auto* context = binding->context;
        while (binding->module->active_calls) {
            if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
            if (!context->is_alive || context->epoch != binding->epoch) break;
            if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        }
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
        while (binding->module->is_alive) {
            const auto submitted = context->submissions;
            const auto stopped = native_frontier(
                shared, exclusion, std::nullopt, true, context, false,
                FrontierErrorPolicy::kIncludeHostFunctionError, AsyncNotificationDomain::kRuntime);
            if (stopped != CUDA_SUCCESS) return stopped;
            if (submitted != context->submissions) continue;
            if (context->active_mutations) {
                if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
                continue;  // A locally pending launch is not yet in the remote frontier.
            }
            ExecutionPayloadWriter input;
            if (!input.write_u64(binding->module->remote)) return CUDA_ERROR_OUT_OF_MEMORY;
            const auto result =
                shared.call(exclusion, *context, ExecutionOpcode::kUnloadModule, std::move(input));
            if (status(result) != CUDA_SUCCESS) return status(result);
            binding->module->is_alive = false;
            for (auto& entry : shared.entries)
                if ((entry->kind == Kind::kFunction || entry->kind == Kind::kModuleGlobal) &&
                    entry->context == context && entry->parent == binding->module->remote)
                    entry->is_alive = false;
        }
    }
    shared.runtime_image_bytes -= image->retained_bytes;
    shared.runtime_metadata_bytes -= image->metadata_bytes;
    image->retained_bytes = 0;
    image->metadata_bytes = 0;
    std::vector<std::byte>{}.swap(image->bytes);
    std::vector<RuntimeKernel>{}.swap(image->kernels);
    std::vector<RuntimeVariable>{}.swap(image->variables);
    std::vector<std::unique_ptr<RuntimeBinding>>{}.swap(image->bindings);
    image->is_alive = false;
    return CUDA_SUCCESS;
}
CUresult runtime_quiescent(
    State& shared, std::unique_lock<std::mutex>& exclusion,
    FrontierErrorPolicy error_policy = FrontierErrorPolicy::kIncludeHostFunctionError) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    for (;;) {
        const auto submitted = context->submissions;
        const auto stopped = native_frontier(shared, exclusion, std::nullopt, true, context, false,
                                             error_policy, AsyncNotificationDomain::kRuntime);
        if (stopped != CUDA_SUCCESS) return stopped;
        if (submitted == context->submissions) return CUDA_SUCCESS;
    }
}
std::optional<CUfunction_attribute> runtime_function_attribute(
    cudaFuncAttribute attribute) noexcept {
    switch (attribute) {
        case cudaFuncAttributeMaxDynamicSharedMemorySize:
            return CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        case cudaFuncAttributePreferredSharedMemoryCarveout:
            return CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
        case cudaFuncAttributeClusterDimMustBeSet:
            return CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET;
        case cudaFuncAttributeRequiredClusterWidth:
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH;
        case cudaFuncAttributeRequiredClusterHeight:
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT;
        case cudaFuncAttributeRequiredClusterDepth:
            return CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH;
        case cudaFuncAttributeNonPortableClusterSizeAllowed:
            return CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED;
        case cudaFuncAttributeClusterSchedulingPolicyPreference:
            return CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        default:
            return std::nullopt;
    }
}
CUresult publish_runtime_function_attributes(cudaFuncAttributes* output,
                                             const ExecutionFunctionAttributes& source) noexcept {
    cudaFuncAttributes value{};
    value.sharedSizeBytes = static_cast<std::size_t>(source.sizes[0]);
    value.constSizeBytes = static_cast<std::size_t>(source.sizes[1]);
    value.localSizeBytes = static_cast<std::size_t>(source.sizes[2]);
    value.maxThreadsPerBlock = source.values[0];
    value.numRegs = source.values[1];
    value.ptxVersion = source.values[2];
    value.binaryVersion = source.values[3];
    value.cacheModeCA = source.values[4];
    value.maxDynamicSharedSizeBytes = source.values[5];
    value.preferredShmemCarveout = source.values[6];
    value.clusterDimMustBeSet = source.values[7];
    value.requiredClusterWidth = source.values[8];
    value.requiredClusterHeight = source.values[9];
    value.requiredClusterDepth = source.values[10];
    value.clusterSchedulingPolicyPreference = source.values[11];
    value.nonPortableClusterSizeAllowed = source.values[12];
    // The public ABI documents only this prefix. Reserved slots and trailing
    // padding belong to the caller and are neither transported nor overwritten.
    constexpr auto bytes = offsetof(cudaFuncAttributes, reserved);
    static_assert(bytes == 3 * sizeof(std::size_t) + 13 * sizeof(int));
    if (!writable_host_range(output, sizeof(*output))) return CUDA_ERROR_INVALID_VALUE;
    return self_copy(output, &value, bytes, true) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult runtime_function_control(State& shared, Exclusion& exclusion, const void* host,
                                  ExecutionFunctionControl operation, std::uint32_t argument,
                                  int value, cudaFuncAttributes* output = nullptr) {
    const bool is_query = operation == ExecutionFunctionControl::kGetAttributes;
    if (is_query && !writable_host_range(output, sizeof(*output))) return CUDA_ERROR_INVALID_VALUE;
    if (!host) return CUDA_ERROR_NOT_FOUND;
    const auto initialized = runtime_context(shared, exclusion);
    if (initialized != CUDA_SUCCESS) return initialized;
    auto* context = shared.context();
    if (!context || !context->has_runtime_bindings) return CUDA_ERROR_NOT_FOUND;
    RuntimeImage* image{};
    const RuntimeKernel* kernel{};
    const auto selected = select_runtime_kernel(shared, host, image, kernel);
    if (selected != CUDA_SUCCESS) return selected;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (image->pending_lookups >= kMaxHandles - 1 || context->active_calls >= kMaxHandles - 2 ||
        context->active_mutations >= kMaxHandles - 2)
        return CUDA_ERROR_OUT_OF_MEMORY;
    // Retain the selected registration and context generation across every
    // unlocked cold factory and metadata RPC. Reset/unregister must wait; no
    // caller-native function pointer or second frontend lock is involved.
    CounterPin lookup(image->pending_lookups, exclusion);
    CounterPin active(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    const auto epoch = context->epoch;
    if (!cached_runtime_function(shared, *image, host)) {
        const auto stopped = runtime_quiescent(shared, exclusion);
        if (stopped != CUDA_SUCCESS) return stopped;
    }
    if (!image->is_alive || image->is_closing) return CUDA_ERROR_NOT_FOUND;
    if (shared.context() != context || !context->is_alive || context->epoch != epoch ||
        context->is_closing)
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    CUfunction function{};
    const auto prepared = runtime_function(shared, exclusion, host, &function);
    if (prepared != CUDA_SUCCESS)
        return !image->is_alive || image->is_closing ? CUDA_ERROR_NOT_FOUND : prepared;
    ExecutionFunctionAttributes attributes;
    const auto result = function_control_locked(shared, exclusion, operation, function, argument,
                                                value, nullptr, is_query ? &attributes : nullptr);
    if (result != CUDA_SUCCESS) return result;
    // A concurrent unregister may have announced its intent, but cannot retire
    // this image while lookup is pinned. An already issued query may finish.
    if (!image->is_alive || shared.context() != context || !context->is_alive ||
        context->epoch != epoch || context->is_closing)
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    return is_query ? publish_runtime_function_attributes(output, attributes) : CUDA_SUCCESS;
}
cudaError_t runtime_result(CUresult result) noexcept {
    cudaError_t value{cudaErrorUnknown};
    switch (result) {
        case CUDA_SUCCESS:
            value = cudaSuccess;
            break;
        case CUDA_ERROR_INVALID_VALUE:
            value = cudaErrorInvalidValue;
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            value = cudaErrorMemoryAllocation;
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            value = cudaErrorInitializationError;
            break;
        case CUDA_ERROR_DEINITIALIZED:
            value = cudaErrorCudartUnloading;
            break;
        case CUDA_ERROR_NO_DEVICE:
            value = cudaErrorNoDevice;
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            value = cudaErrorInvalidDevice;
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            value = cudaErrorInvalidKernelImage;
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            value = cudaErrorIncompatibleDriverContext;
            break;
        case CUDA_ERROR_INVALID_HANDLE:
            value = cudaErrorInvalidResourceHandle;
            break;
        case CUDA_ERROR_NOT_FOUND:
            value = cudaErrorInvalidDeviceFunction;
            break;
        case CUDA_ERROR_NOT_READY:
            value = cudaErrorNotReady;
            break;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            value = cudaErrorIllegalAddress;
            break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            value = cudaErrorLaunchOutOfResources;
            break;
        case CUDA_ERROR_LAUNCH_FAILED:
            value = cudaErrorLaunchFailure;
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            value = cudaErrorLaunchTimeout;
            break;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            value = cudaErrorContextIsDestroyed;
            break;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            value = cudaErrorHostMemoryAlreadyRegistered;
            break;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            value = cudaErrorHostMemoryNotRegistered;
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            value = cudaErrorNotSupported;
            break;
        case CUDA_ERROR_NOT_PERMITTED:
            value = cudaErrorNotPermitted;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            value = cudaErrorStreamCaptureUnsupported;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            value = cudaErrorStreamCaptureInvalidated;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            value = cudaErrorStreamCaptureMerge;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            value = cudaErrorStreamCaptureUnmatched;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            value = cudaErrorStreamCaptureUnjoined;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            value = cudaErrorStreamCaptureIsolation;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            value = cudaErrorStreamCaptureImplicit;
            break;
        case CUDA_ERROR_CAPTURED_EVENT:
            value = cudaErrorCapturedEvent;
            break;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            value = cudaErrorStreamCaptureWrongThread;
            break;
        default:
            break;
    }
    if (value != cudaSuccess) runtime_last_error = value;
    return value;
}
void observe_async_runtime_error() noexcept {
    if (!async_caller) return;  // Cold error queries never create a context or caller record.
    const auto observed = locked_boundary(
        [&](State& shared, Exclusion&) {
            std::array<std::shared_ptr<AsyncLaunch>, kMaxAsyncLaunches> snapshot;
            std::size_t count{};
            for (const auto& launch : shared.async_launches)
                if (launch->is_runtime && launch->caller.lock() == async_caller) {
                    if (launch->frontier && launch->context->is_uncertain)
                        cache_async_failure(*launch, CUDA_ERROR_UNKNOWN);
                    snapshot[count++] = launch;
                }
            const auto selected = std::span{snapshot}.first(count);
            if (count) shared.request_async_status_progress();
            // Preserve an already pending local error. A successful admission
            // or a still-pending frontier cannot clear or overwrite it.
            if (runtime_last_error != cudaSuccess) return CUDA_SUCCESS;
            CUresult notification = CUDA_SUCCESS;
            for (const auto& launch : selected)
                if (launch->is_runtime && launch->caller.lock() == async_caller &&
                    !launch->is_notified && launch->failure != CUDA_SUCCESS) {
                    launch->is_notified = true;
                    notification = launch->failure;
                    break;
                }
            shared.retire_async_notifications();
            return notification;
        },
        false);
    if (observed != CUDA_SUCCESS && runtime_last_error == cudaSuccess)
        (void)runtime_result(observed);
}

#include "process_memory_frontend.inc"
#include "graph_frontend.inc"

struct PointerMetadataSource {
    Entry* memory{};
    Entry* owner{};
    Entry* origin{};
    Entry* module{};
};
bool pointer_context(const Entry& entry) noexcept {
    return entry.kind == Kind::kContext || entry.kind == Kind::kHostContext ||
           entry.kind == Kind::kPointerMetadataContext || entry.kind == Kind::kProcessMemoryContext;
}
// All participants are selected under the same exclusion. Never hold a partial
// pin set while waiting for a retiring resource: host entries can be erased.
class PointerMetadataPins {
   public:
    explicit PointerMetadataPins(Exclusion& exclusion) : exclusion_(exclusion) {}
    PointerMetadataPins(const PointerMetadataPins&) = delete;
    PointerMetadataPins& operator=(const PointerMetadataPins&) = delete;
    ~PointerMetadataPins() {
        if (::getpid() != kProcess || !exclusion_.owns_lock()) return;
        for (std::size_t index = 0; index < count_; ++index) {
            auto& entry = *entries_[index];
            --entry.active_calls;
            if (pointer_context(entry)) --entry.active_mutations;
        }
    }
    bool acquire(std::array<Entry*, 7> candidates) noexcept {
        std::array<Entry*, 7> unique{};
        std::size_t count{};
        for (auto* entry : candidates) {
            if (!entry ||
                std::find(unique.begin(), unique.begin() + count, entry) != unique.begin() + count)
                continue;
            if (!entry->is_alive || entry->is_uncertain || entry->is_retiring ||
                entry->active_calls >= kMaxHandles - 1 ||
                (pointer_context(*entry) &&
                 (!entry->is_active || entry->is_opening || entry->is_closing ||
                  !entry->connection || entry->active_mutations >= kMaxHandles - 1)))
                return false;
            unique[count++] = entry;
        }
        entries_ = unique;
        count_ = count;
        for (std::size_t index = 0; index < count_; ++index) {
            auto& entry = *entries_[index];
            epochs_[index] = entry.epoch;
            ++entry.active_calls;
            if (pointer_context(entry)) ++entry.active_mutations;
        }
        return true;
    }
    bool healthy() const noexcept {
        for (std::size_t index = 0; index < count_; ++index) {
            const auto& entry = *entries_[index];
            // Announced retirement may wait for this query. The actual entry
            // and owning context must remain the same until publication.
            if (!entry.is_alive || entry.is_uncertain || entry.epoch != epochs_[index] ||
                (pointer_context(entry) && (!entry.is_active || entry.is_closing)))
                return false;
        }
        return true;
    }
    void quarantine() noexcept {
        for (std::size_t index = 0; index < count_; ++index) entries_[index]->is_uncertain = true;
    }

   private:
    Exclusion& exclusion_;
    std::array<Entry*, 7> entries_{};
    std::array<std::uint64_t, 7> epochs_{};
    std::size_t count_{};
};

CUresult classify_pointer(State& shared, std::uint64_t address, PointerMetadataSource& source,
                          bool& should_wait) noexcept {
    source = {};
    should_wait = false;
    if (!address) return CUDA_SUCCESS;
    bool uncertain{};
    std::size_t live_matches{};
    for (const auto& candidate : shared.entries) {
        auto& entry = *candidate;
        if ((entry.kind != Kind::kAllocation && entry.kind != Kind::kProcessAllocation &&
             entry.kind != Kind::kModuleGlobal && entry.kind != Kind::kHostAllocation) ||
            address < entry.address || address - entry.address >= entry.bytes)
            continue;
        if (!entry.is_alive) {
            uncertain = uncertain || entry.is_uncertain;
            // Host admission publishes its numeric address before the local
            // mapping is committed. Such an entry is neither a dead resource
            // nor an unregistered caller address yet.
            should_wait =
                should_wait || (entry.kind == Kind::kHostAllocation && !entry.is_uncertain);
            continue;
        }
        // Positive live containment wins over settled old addresses and
        // endpoints. No page-rounded host extent grants resource authority.
        ++live_matches;
        source = {};
        source.memory = &entry;
        source.origin = entry.context;
        source.owner = entry.kind == Kind::kHostAllocation ? shared.host_context : entry.context;
        if (entry.kind == Kind::kModuleGlobal) {
            for (const auto& parent : shared.entries)
                if (parent->kind == Kind::kModule && parent->is_alive &&
                    parent->context == entry.context && parent->remote == entry.parent) {
                    source.module = parent.get();
                    break;
                }
            if (!source.module) return CUDA_ERROR_UNKNOWN;
        }
        for (auto* participant : {source.memory, source.owner, source.origin, source.module}) {
            if (!participant) continue;
            if (participant->is_uncertain || !participant->is_alive) return CUDA_ERROR_UNKNOWN;
            should_wait = should_wait || participant->is_retiring || participant->is_opening ||
                          participant->is_closing;
            if (pointer_context(*participant) && !should_wait &&
                (!participant->is_active || !participant->connection))
                return CUDA_ERROR_UNKNOWN;
        }
        if (!source.owner || !source.origin || !source.memory->remote ||
            (source.memory->kind == Kind::kHostAllocation &&
             source.memory->epoch != source.origin->epoch))
            return CUDA_ERROR_UNKNOWN;
    }
    if (!source.memory) return uncertain ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
    // A successful free can permit another context to reuse the address
    // before its frontend ACK retires the old entry. Wait without pins and
    // rescan all matches; only simultaneous stable live owners are ambiguous.
    if (live_matches > 1 && !should_wait) return CUDA_ERROR_UNKNOWN;
    return CUDA_SUCCESS;
}

bool valid_pointer_effects(const ExecutionPointerQuery& query,
                           const ExecutionPointerObservation& value,
                           std::int32_t status_code) noexcept {
    const bool is_owned = query.kind == ExecutionPointerKind::kDevice ||
                          query.kind == ExecutionPointerKind::kHost ||
                          query.kind == ExecutionPointerKind::kProcessDevice;
    const auto input = ExecutionPointerAlias::kInput;
    if (value.device_alias == input && !is_owned) return false;
    if (value.host_alias == input && query.kind != ExecutionPointerKind::kHost &&
        query.kind != ExecutionPointerKind::kUnregistered)
        return false;
    if (query.domain == ExecutionPointerDomain::kDriver) {
        if (query.selector == 1)
            return query.kind == ExecutionPointerKind::kProcessDevice && !(value.effects & ~4U) &&
                   value.device_alias == ExecutionPointerAlias::kNull &&
                   (status_code != 0 || value.effects == 4U);
        const auto mask = query.selector == 3 ? 4U : 2U;
        if (value.effects & ~mask) return false;
        return status_code != 0 ||
               (is_owned && value.effects == mask &&
                (query.selector == 3 ? value.device_alias == input : value.device >= 0));
    }
    if (status_code != 0) return true;  // Independently observed native error effects.
    if (value.effects != 15U) return false;
    if (!is_owned)
        return value.type == 0 && value.device == -2 &&
               value.device_alias == ExecutionPointerAlias::kNull &&
               value.host_alias == (query.kind == ExecutionPointerKind::kNull
                                        ? ExecutionPointerAlias::kNull
                                        : input);
    return value.device >= 0 &&
           (query.kind == ExecutionPointerKind::kHost ? value.type == 1 : value.type == 2);
}

bool publish_pointer_effects(void* output, std::uintptr_t address,
                             const ExecutionPointerQuery& query,
                             const ExecutionPointerObservation& value) noexcept {
    static_assert(sizeof(CUdeviceptr) == 8 && sizeof(int) == 4 && sizeof(void*) == 8);
    static_assert(sizeof(cudaPointerAttributes) == 24 &&
                  offsetof(cudaPointerAttributes, type) == 0 &&
                  offsetof(cudaPointerAttributes, device) == 4 &&
                  offsetof(cudaPointerAttributes, devicePointer) == 8 &&
                  offsetof(cudaPointerAttributes, hostPointer) == 16);
    const auto bytes = query.domain == ExecutionPointerDomain::kRuntime
                           ? sizeof(cudaPointerAttributes)
                           : (query.selector == 3 || query.selector == 1 ? 8U : 4U);
    if (!writable_host_range(output, bytes)) return false;
    if (!value.effects) return true;
    const std::uint64_t device = value.device_alias == ExecutionPointerAlias::kInput ? address : 0;
    const std::uint64_t host = value.host_alias == ExecutionPointerAlias::kInput ? address : 0;
    if (query.domain == ExecutionPointerDomain::kDriver)
        return query.selector == 3 || query.selector == 1
                   ? self_copy(output, &device, sizeof(device), true)
                   : self_copy(output, &value.device, sizeof(value.device), true);
    // Do not read caller storage to preserve untouched fields: it may be
    // write-only. Submit all observed effects in one self-process write.
    std::array<iovec, 4> local{}, remote{};
    std::size_t count{}, total{};
    const auto append = [&](const void* data, std::size_t offset, std::size_t size) {
        local[count] = {const_cast<void*>(data), size};
        remote[count] = {static_cast<std::byte*>(output) + offset, size};
        ++count;
        total += size;
    };
    if (value.effects & 1U) append(&value.type, 0, sizeof(value.type));
    if (value.effects & 2U) append(&value.device, 4, sizeof(value.device));
    if (value.effects & 4U) append(&device, 8, sizeof(device));
    if (value.effects & 8U) append(&host, 16, sizeof(host));
    const auto written =
        ::process_vm_writev(kProcess, local.data(), count, remote.data(), count, 0);
    return written >= 0 && static_cast<std::size_t>(written) == total;
}

CUresult pointer_metadata(State& shared, Exclusion& exclusion, void* output, std::uintptr_t address,
                          ExecutionPointerQuery query,
                          std::optional<std::int32_t>& original_status) {
    const bool is_runtime = query.domain == ExecutionPointerDomain::kRuntime;
    if (!is_runtime && query.selector != 1 && query.selector != 3 && query.selector != 9)
        return CUDA_ERROR_NOT_SUPPORTED;
    const auto bytes = is_runtime ? sizeof(cudaPointerAttributes)
                                  : (query.selector == 3 || query.selector == 1 ? 8U : 4U);
    if (!writable_host_range(output, bytes)) return CUDA_ERROR_INVALID_VALUE;
    if (!is_runtime) {
        const auto initialized = ensure_driver_initialization(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
    }
    if (!shared.config) return CUDA_ERROR_NOT_INITIALIZED;
    if (!shared.pointer_metadata_context) {
        auto* entry = shared.allocate_entry(Kind::kPointerMetadataContext, nullptr);
        if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
        entry->context = entry;
        entry->is_alive = true;
        shared.pointer_metadata_context = entry;
    }
    auto& carrier = *shared.pointer_metadata_context;
    const auto opened = shared.open_context(exclusion, carrier, 0);
    if (opened != CUDA_SUCCESS) return opened;
    Entry* current{};
    PointerMetadataSource source;
    for (;;) {
        bool should_wait{};
        current = context_stack.empty() ? nullptr : context_stack.back();
        if (current) {
            if (current->is_uncertain) return CUDA_ERROR_UNKNOWN;
            if (!current->is_alive) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
            should_wait = current->is_opening || current->is_closing;
            // A retained, reset primary token is not a current native context.
            // This observation must never lazily create an application context.
            if (!should_wait && !current->is_active) current = nullptr;
        }
        bool source_wait{};
        const auto classified = classify_pointer(shared, address, source, source_wait);
        if (classified != CUDA_SUCCESS) return classified;
        if (!should_wait && !source_wait) break;
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (carrier.is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    PointerMetadataPins pins(exclusion);
    if (!pins.acquire({&carrier, current, source.owner, source.origin, source.memory, source.module,
                       nullptr}))
        return CUDA_ERROR_NOT_READY;
    query.query_session = current ? current->connection->session() : 0;
    query.kind = address ? ExecutionPointerKind::kUnregistered : ExecutionPointerKind::kNull;
    if (source.memory) {
        query.kind = source.memory->kind == Kind::kHostAllocation ? ExecutionPointerKind::kHost
                     : source.memory->kind == Kind::kProcessAllocation
                         ? ExecutionPointerKind::kProcessDevice
                         : ExecutionPointerKind::kDevice;
        query.source_session = source.owner->connection->session();
        query.allocation = source.memory->remote;
        query.offset = address - source.memory->address;
    }
    if (!is_runtime && query.selector == 1 && query.kind != ExecutionPointerKind::kProcessDevice)
        return CUDA_ERROR_NOT_SUPPORTED;
    ExecutionPayloadWriter input;
    if (!write_execution_pointer_query(input, query)) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply =
        shared.call(exclusion, carrier, ExecutionOpcode::kPointerMetadata, std::move(input));
    const auto malformed = [&] {
        pins.quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    if (!reply || !pins.healthy() || reply->id || reply->address || !reply->arguments.empty() ||
        reply->parameter_bytes > 1)
        return malformed();
    const auto native_status_valid = [&](std::int32_t value) {
        return value >= 0 && value <= (is_runtime ? 10000 : 999);
    };
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (reply->status <= 0 || reply->status > 999 || !reply->data.empty() ||
            (reply->original_status &&
             (!reply->parameter_bytes || !native_status_valid(*reply->original_status))))
            return malformed();
        if (reply->parameter_bytes) pins.quarantine();
        return static_cast<CUresult>(reply->status);
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || reply->parameter_bytes ||
        !reply->original_status || *reply->original_status != reply->status ||
        !native_status_valid(reply->status) ||
        reply->data.size() != kExecutionPointerObservationBytes)
        return malformed();
    ExecutionPayloadReader receipt(reply->data);
    ExecutionPointerObservation observation;
    if (!read_execution_pointer_observation(receipt, observation) || !receipt.empty() ||
        !valid_pointer_effects(query, observation, reply->status))
        return malformed();
    if (is_runtime) shared.has_driver_initialization = true;
    if (!publish_pointer_effects(output, address, query, observation))
        return CUDA_ERROR_INVALID_VALUE;
    original_status = reply->status;
    return CUDA_SUCCESS;
}
template <typename Query>
CUresult device_metadata_query(Query&& query) noexcept {
    const auto first = query();
    if (first != CUDA_ERROR_NOT_INITIALIZED) return first;
    // Remote Runtime success establishes logical Driver initialization without
    // invoking the caller Runtime or creating a caller-side CUDA context. Only
    // resource-free metadata may bridge that state to the original Driver.
    bool has_initialization{};
    const auto observed = boundary([&](State& shared) {
        has_initialization = shared.has_driver_initialization;
        return CUDA_SUCCESS;
    });
    if (observed != CUDA_SUCCESS || !has_initialization) return first;
    const auto initialize = original<decltype(&cuInit)>("cuInit");
    if (!initialize) return first;
    // No frontend exclusion is held across either SDK call. Never retry a
    // resource command, mask native initialization errors or import a context.
    const auto initialized = initialize(0);
    if (initialized != CUDA_SUCCESS) return initialized;
    return query();
}
}  // namespace

#define GLIMMER_QUERY_BODY(NAME, PARAMETERS, BODY)                     \
    extern "C" CUresult CUDAAPI NAME PARAMETERS {                      \
        if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT; \
        if (is_host_function_active) return CUDA_ERROR_NOT_PERMITTED;  \
        static const auto native = original<decltype(&(NAME))>(#NAME); \
        if (!native) return CUDA_ERROR_NOT_INITIALIZED;                \
        return BODY;                                                   \
    }
#define GLIMMER_QUERY(NAME, PARAMETERS, ARGUMENTS) \
    GLIMMER_QUERY_BODY(NAME, PARAMETERS, native ARGUMENTS)
#define GLIMMER_DEVICE_QUERY(NAME, PARAMETERS, ARGUMENTS) \
    GLIMMER_QUERY_BODY(NAME, PARAMETERS, device_metadata_query([&] { return native ARGUMENTS; }))
GLIMMER_QUERY(cuInit, (unsigned int flags), (flags))
GLIMMER_DEVICE_QUERY(cuDeviceGet, (CUdevice * device, int ordinal), (device, ordinal))
GLIMMER_DEVICE_QUERY(cuDeviceGetCount, (int* count), (count))
GLIMMER_DEVICE_QUERY(cuDeviceGetAttribute,
                     (int* value, CUdevice_attribute attribute, CUdevice device),
                     (value, attribute, device))
GLIMMER_QUERY(cuDriverGetVersion, (int* version), (version))
GLIMMER_QUERY(cuGetErrorName, (CUresult error, const char** name), (error, name))
GLIMMER_QUERY(cuGetErrorString, (CUresult error, const char** text), (error, text))
#undef GLIMMER_QUERY
#undef GLIMMER_DEVICE_QUERY
#undef GLIMMER_QUERY_BODY

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext* output, unsigned int flags, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_context(shared, exclusion, output, {flags, device});
    });
}
// Exact native context creation ABI; count/flags cannot be reordered.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuCtxCreate_v3(CUcontext* output, CUexecAffinityParam* parameters,
                                           int count, unsigned int flags, CUdevice device) {
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (count < 0) return CUDA_ERROR_INVALID_VALUE;
    if (count != 0) return CUDA_ERROR_NOT_SUPPORTED;
    (void)parameters;  // A zero count does not consume an affinity array.
    return cuCtxCreate_v2(output, flags, device);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuCtxCreate_v4(CUcontext* output, CUctxCreateParams* parameters,
                                           unsigned int flags, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        CUctxCreateParams copied{};
        if (!self_copy(&copied, parameters, sizeof(copied), false)) return CUDA_ERROR_INVALID_VALUE;
        // A non-null, empty structure is valid in the installed Driver. NULL
        // itself is not a structure, and a positive count needs a real family.
        if ((copied.execAffinityParams && copied.cigParams) || copied.numExecAffinityParams < 0 ||
            (copied.numExecAffinityParams > 0 && !copied.execAffinityParams))
            return CUDA_ERROR_INVALID_VALUE;
        if (copied.cigParams || copied.numExecAffinityParams != 0) return CUDA_ERROR_NOT_SUPPORTED;
        return create_context(shared, exclusion, output, {flags, device});
    });
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
        const auto initialized = ensure_driver_initialization(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
        return publish(output, reinterpret_cast<CUcontext>(shared.context()))
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void* output, CUpointer_attribute attribute,
                                                  CUdeviceptr pointer) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            ExecutionPointerQuery query;
            query.selector = static_cast<std::uint32_t>(attribute);
            return pointer_metadata(shared, exclusion, output, static_cast<std::uintptr_t>(pointer),
                                    query, original_status);
        },
        false);
    return result == CUDA_SUCCESS && original_status ? static_cast<CUresult>(*original_status)
                                                     : result;
}
extern "C" cudaError_t CUDARTAPI cudaPointerGetAttributes(cudaPointerAttributes* output,
                                                          const void* pointer) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            ExecutionPointerQuery query;
            query.domain = ExecutionPointerDomain::kRuntime;
            query.selector = 0;
            return pointer_metadata(shared, exclusion, output,
                                    reinterpret_cast<std::uintptr_t>(pointer), query,
                                    original_status);
        },
        false);
    if (result != CUDA_SUCCESS || !original_status) return runtime_result(result);
    const auto observed = static_cast<cudaError_t>(*original_status);
    if (observed != cudaSuccess) runtime_last_error = observed;
    return observed;
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        auto* primary = shared.primary_entry();
        if (!primary) return CUDA_ERROR_OUT_OF_MEMORY;
        const auto waited = shared.await_context_transition(exclusion, *primary);
        if (waited != CUDA_SUCCESS) return waited;
        if (primary->is_uncertain || primary->primary_references == UINT64_MAX)
            return CUDA_ERROR_UNKNOWN;
        if (!primary->is_active) {
            const auto opened =
                shared.open_context(exclusion, *primary, primary->context_flags | CU_CTX_MAP_HOST);
            if (opened != CUDA_SUCCESS) return opened;
        }
        if (primary->primary_references == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
        ++primary->primary_references;
        if (!publish(output, reinterpret_cast<CUcontext>(primary))) {
            primary->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& exclusion) {
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, exclusion, device, true);
        return result == CUDA_SUCCESS && primary && !primary->is_active
                   ? shared.retired_driver_error(*primary, epoch)
                   : result;
    });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& exclusion) {
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, exclusion, device, false);
        return result == CUDA_SUCCESS && primary && !primary->is_active
                   ? shared.retired_driver_error(*primary, epoch)
                   : result;
    });
}
// Historical primary ABIs differ in reference/reset semantics. Explicitly
// reject them rather than expose a client-native context behind virtual handles.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice device) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice device) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice device, unsigned int) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed native CUDA ABI.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice device, unsigned int* flags,
                                                       int* active) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(flags) || !writable(active)) return CUDA_ERROR_INVALID_VALUE;
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        if (shared.primary) {
            const auto waited = shared.await_context_transition(exclusion, *shared.primary);
            if (waited != CUDA_SUCCESS) return waited;
        }
        if (shared.primary && shared.primary->is_uncertain) return CUDA_ERROR_UNKNOWN;
        return publish(flags, shared.primary ? shared.primary->context_flags : 0U) &&
                       publish(active, shared.primary && shared.primary->is_active ? 1 : 0)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed native CUDA ABI.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice device, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        const auto checked = context_flags_status(flags);
        if (checked != CUDA_SUCCESS) return checked;
        // The primary already maps host memory. Unlike cuCtxSetFlags, this
        // original Driver entry rejects explicitly requesting that capability.
        if (flags & CU_CTX_MAP_HOST) return CUDA_ERROR_INVALID_VALUE;
        auto* primary = shared.primary_entry();
        return primary ? set_context_flags(shared, exclusion, *primary, flags)
                       : CUDA_ERROR_OUT_OF_MEMORY;
    });
}
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* output) {
    return boundary([&](State& shared) {
        if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
        return publish(output, CUdevice{0}) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuCtxGetFlags(unsigned int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        auto* context = shared.context();
        if (!context) return CUDA_ERROR_INVALID_CONTEXT;
        if (!context->is_active && context->is_primary)
            return publish(output, context->context_flags | CU_CTX_MAP_HOST)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        const auto result = shared.call(exclusion, *context, ExecutionOpcode::kGetContextFlags,
                                        ExecutionPayloadWriter{});
        if (status(result) != CUDA_SUCCESS) return status(result);
        if (result->id > UINT32_MAX) {
            context->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        auto flags = static_cast<unsigned int>(result->id);
        if (!context->is_primary)
            flags = (flags & ~static_cast<unsigned int>(CU_CTX_MAP_HOST)) |
                    (context->context_flags & CU_CTX_MAP_HOST);
        return publish(output, flags) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuCtxSetFlags(unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* context = shared.context();
        return context ? set_context_flags(shared, exclusion, *context, flags)
                       : CUDA_ERROR_INVALID_CONTEXT;
    });
}
namespace {
std::int32_t context_control(bool is_runtime, ExecutionContextControl operation,
                             std::uint32_t argument, std::uint64_t value, void* output = nullptr,
                             void* second = nullptr,
                             CUcontext explicit_context = nullptr) noexcept {
    using Operation = ExecutionContextControl;
    std::optional<std::int32_t> native_status;
    const auto prepared = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto bytes = operation == Operation::kGetLimit      ? sizeof(std::size_t)
                           : operation == Operation::kGetAffinity ? sizeof(CUexecAffinityParam)
                           : operation == Operation::kGetCache ||
                                   operation == Operation::kGetSharedMemory ||
                                   operation == Operation::kGetApiVersion
                               ? sizeof(std::uint32_t)
                               : 0;
        if ((bytes && !writable_host_range(output, bytes)) ||
            (operation == Operation::kGetPriorityRange &&
             ((output && !writable_host_range(output, sizeof(int))) ||
              (second && !writable_host_range(second, sizeof(int))))))
            return CUDA_ERROR_INVALID_VALUE;
        if (is_runtime) {
            const auto opened = runtime_context(shared, exclusion);
            if (opened != CUDA_SUCCESS) return opened;
        }
        auto* owner =
            explicit_context ? shared.find(explicit_context, Kind::kContext) : shared.context();
        if (!owner) return CUDA_ERROR_INVALID_CONTEXT;
        if (owner->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (owner->active_calls >= kMaxHandles - 1 || owner->active_mutations >= kMaxHandles - 1)
            return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin active(owner->active_calls, exclusion);
        CounterPin mutation(owner->active_mutations, exclusion);
        ExecutionPayloadWriter input;
        if (!input.write_u32(static_cast<std::uint32_t>(operation)) || !input.write_u32(argument) ||
            !input.write_u64(value))
            return CUDA_ERROR_OUT_OF_MEMORY;
        const auto reply = shared.call(exclusion, *owner,
                                       is_runtime ? ExecutionOpcode::kRuntimeContextControl
                                                  : ExecutionOpcode::kDriverContextControl,
                                       std::move(input));
        const auto invalid = [&] {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        };
        if (!reply || reply->parameter_bytes || !reply->data.empty() || !reply->arguments.empty())
            return invalid();
        if (reply->disposition != ExecutionWireDisposition::kComplete) {
            if (reply->original_status || reply->status <= 0 || reply->status > 999 || reply->id ||
                reply->address)
                return invalid();
            return status(reply);
        }
        if (!reply->original_status || reply->status != *reply->original_status ||
            reply->status < 0 || reply->status > (is_runtime ? 10000 : 999))
            return invalid();
        if (reply->status != 0) {
            if (reply->id || reply->address) return invalid();
            native_status = reply->status;
            return CUDA_SUCCESS;
        }
        if (operation != Operation::kGetPriorityRange && reply->address) return invalid();
        if ((operation == Operation::kGetCache && reply->id > 3) ||
            (operation == Operation::kGetSharedMemory && reply->id > 2) ||
            ((operation == Operation::kGetApiVersion || operation == Operation::kGetAffinity ||
              operation == Operation::kGetPriorityRange) &&
             reply->id > UINT32_MAX) ||
            (operation == Operation::kGetPriorityRange && reply->address > UINT32_MAX) ||
            (operation == Operation::kGetLimit && reply->id > SIZE_MAX) ||
            (!bytes && operation != Operation::kGetPriorityRange && reply->id))
            return invalid();
        native_status = 0;
        if (operation == Operation::kGetLimit)
            return publish(static_cast<std::size_t*>(output), static_cast<std::size_t>(reply->id))
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        if (operation == Operation::kGetAffinity) {
            if (argument != 0) return invalid();
            CUexecAffinityParam parameter{};
            parameter.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
            parameter.param.smCount.val = static_cast<unsigned int>(reply->id);
            return publish(static_cast<CUexecAffinityParam*>(output), parameter)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        if (operation == Operation::kGetPriorityRange) {
            const auto least = std::bit_cast<int>(static_cast<std::uint32_t>(reply->id));
            const auto greatest = std::bit_cast<int>(static_cast<std::uint32_t>(reply->address));
            return ((!output || publish(static_cast<int*>(output), least)) &&
                    (!second || publish(static_cast<int*>(second), greatest)))
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        const auto scalar = static_cast<std::uint32_t>(reply->id);
        return !bytes || self_copy(output, &scalar, sizeof(scalar), true)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
    if (prepared != CUDA_SUCCESS)
        return is_runtime ? static_cast<std::int32_t>(runtime_result(prepared))
                          : static_cast<std::int32_t>(prepared);
    if (!native_status)
        return is_runtime ? static_cast<std::int32_t>(runtime_result(CUDA_ERROR_UNKNOWN))
                          : static_cast<std::int32_t>(CUDA_ERROR_UNKNOWN);
    if (is_runtime && *native_status) runtime_last_error = static_cast<cudaError_t>(*native_status);
    return *native_status;
}
}  // namespace
extern "C" CUresult CUDAAPI cuCtxGetLimit(std::size_t* output, CUlimit limit) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetLimit,
                                                 static_cast<std::uint32_t>(limit), 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, std::size_t value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetLimit,
                                                 static_cast<std::uint32_t>(limit), value));
}
extern "C" CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache* output) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetCache, 0, 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetCache,
                                                 static_cast<std::uint32_t>(value), 0));
}
extern "C" CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig* output) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetSharedMemory, 0, 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetSharedMemory,
                                                 static_cast<std::uint32_t>(value), 0));
}
extern "C" CUresult CUDAAPI cuCtxResetPersistingL2Cache() {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kResetPersistingCache, 0, 0));
}
extern "C" CUresult CUDAAPI cuCtxGetApiVersion(CUcontext context, unsigned int* output) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetApiVersion, 0,
                                                 0, output, nullptr, context));
}
extern "C" CUresult CUDAAPI cuCtxGetExecAffinity(CUexecAffinityParam* output,
                                                 CUexecAffinityType type) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetAffinity,
                                                 static_cast<std::uint32_t>(type), 0, output));
}
extern "C" CUresult CUDAAPI cuCtxGetStreamPriorityRange(int* least, int* greatest) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetPriorityRange, 0, 0, least, greatest));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetLimit(std::size_t* output, cudaLimit limit) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kGetLimit,
                                                    static_cast<std::uint32_t>(limit), 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetLimit(cudaLimit limit, std::size_t value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetLimit,
                                                    static_cast<std::uint32_t>(limit), value));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(cudaFuncCache* output) {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kGetCache, 0, 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(cudaFuncCache value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetCache,
                                                    static_cast<std::uint32_t>(value), 0));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* output) {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kGetSharedMemory, 0, 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(cudaSharedMemConfig value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetSharedMemory,
                                                    static_cast<std::uint32_t>(value), 0));
}
extern "C" cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache() {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kResetPersistingCache, 0, 0));
}
extern "C" cudaError_t CUDARTAPI cudaThreadGetLimit(std::size_t* output, cudaLimit limit) {
    return cudaDeviceGetLimit(output, limit);
}
extern "C" cudaError_t CUDARTAPI cudaThreadSetLimit(cudaLimit limit, std::size_t value) {
    return cudaDeviceSetLimit(limit, value);
}
extern "C" cudaError_t CUDARTAPI cudaThreadGetCacheConfig(cudaFuncCache* output) {
    return cudaDeviceGetCacheConfig(output);
}
extern "C" cudaError_t CUDARTAPI cudaThreadSetCacheConfig(cudaFuncCache value) {
    return cudaDeviceSetCacheConfig(value);
}
extern "C" CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
    return boundary([&](State& shared) {
        auto* selected = context ? shared.find(context, Kind::kContext) : nullptr;
        if (context && (!selected || selected->is_uncertain)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!selected) {
            if (!context_stack.empty()) context_stack.pop_back();
        } else if (context_stack.empty())
            context_stack.push_back(selected);
        else
            context_stack.back() = selected;
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* selected = shared.find(context, Kind::kContext);
        if (!selected) return CUDA_ERROR_INVALID_CONTEXT;
        if (selected->is_primary) return CUDA_ERROR_INVALID_CONTEXT;
        const auto epoch = selected->epoch;
        const auto result =
            shared.call(exclusion, *selected, ExecutionOpcode::kClose, ExecutionPayloadWriter{});
        if (status(result) == CUDA_SUCCESS) {
            for (auto& entry : shared.entries)
                if (entry->context == selected) entry->is_alive = false;
            std::erase(context_stack, selected);
            return shared.retired_driver_error(*selected, epoch);
        }
        return status(result);
    });
}
extern "C" CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext context) {
    return boundary([&](State& shared) {
        auto* selected = shared.find(context, Kind::kContext);
        if (!selected || selected->is_uncertain) return CUDA_ERROR_INVALID_CONTEXT;
        if (context_stack.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        context_stack.push_back(selected);
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* output) {
    return boundary([&](State& shared) {
        auto* current = shared.context();
        if (!current) return CUDA_ERROR_INVALID_CONTEXT;
        if (!publish(output, reinterpret_cast<CUcontext>(current))) return CUDA_ERROR_INVALID_VALUE;
        context_stack.pop_back();
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* output, const void* image) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return load_module(shared, exclusion, output, image);
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* output, const void* image) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return load_module(shared, exclusion, output, image);
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* output, const void* image,
                                               unsigned int option_count, CUjit_option* options,
                                               void** values) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!option_count) return load_module(shared, exclusion, output, image);
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
        JitSubmission submission;
        const auto copied = snapshot_jit(submission, output, option_count, options, values);
        return copied == CUDA_SUCCESS ? load_module(shared, exclusion, output, image, &submission)
                                      : copied;
    });
}
extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* output, CUmodule module,
                                                const char* name) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return get_function(shared, exclusion, output, module, name);
    });
}
extern "C" CUresult CUDAAPI cuFuncGetAttribute(int* output, CUfunction_attribute attribute,
                                               CUfunction function) {
    return function_control(ExecutionFunctionControl::kGetAttribute, function,
                            std::bit_cast<std::uint32_t>(attribute), 0, output);
}
extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction function, CUfunction_attribute attribute,
                                               int value) {
    return function_control(ExecutionFunctionControl::kSetAttribute, function,
                            std::bit_cast<std::uint32_t>(attribute), value);
}
extern "C" CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction function, CUfunc_cache value) {
    return function_control(ExecutionFunctionControl::kSetCache, function,
                            std::bit_cast<std::uint32_t>(value), 0);
}
// These exact-ABI leaves contain constructor-only lookups of reference stubs.
// They grant no resource authority and deliberately never inspect caller inputs.
extern "C" CUresult CUDAAPI cuLinkCreate_v2(unsigned int, CUjit_option*, void**, CUlinkState*) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkCreate_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkAddData_v2(CUlinkState, CUjitInputType, void*, std::size_t,
                                             const char*, unsigned int, CUjit_option*, void**) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkAddData_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkAddFile_v2(CUlinkState, CUjitInputType, const char*, unsigned int,
                                             CUjit_option*, void**) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkAddFile_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkComplete(CUlinkState, void**, std::size_t*) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkComplete\n");
}
extern "C" CUresult CUDAAPI cuLinkDestroy(CUlinkState) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkDestroy\n");
}
extern "C" CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, CUstream,
                                                      void**) {
    return reject_constructor_only(
        "Glimmer: unsupported owned CUDA operation: cuLaunchCooperativeKernel\n");
}
extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return destroy_resource(shared, exclusion, module, Kind::kModule,
                                ExecutionOpcode::kUnloadModule);
    });
}
extern "C" CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr* address, std::size_t* bytes,
                                                 CUmodule module, const char* name) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if ((address && !writable(address)) || (bytes && !writable(bytes)))
            return CUDA_ERROR_INVALID_VALUE;
        const auto symbol = owned_string(name, 4096);
        if (!symbol || symbol->empty()) return CUDA_ERROR_INVALID_VALUE;
        Entry* global{};
        const auto result =
            get_global(shared, exclusion, global, shared.find(module, Kind::kModule), *symbol);
        if (result != CUDA_SUCCESS) return result;
        return (!address || publish(address, static_cast<CUdeviceptr>(global->address))) &&
                       (!bytes || publish(bytes, static_cast<std::size_t>(global->bytes)))
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* output, std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return memory_allocate(shared, exclusion, output, bytes);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* output, std::size_t bytes,
                                            CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return allocate_process_memory(shared, exclusion, output, bytes, nullptr, stream, false);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* output, std::size_t bytes,
                                                    CUmemoryPool pool, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return allocate_process_memory(shared, exclusion, output, bytes, pool, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return query_process_pool(shared, exclusion, output, device,
                                  ExecutionProcessMemoryOperation::kGetDefaultPool);
    });
}
extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return query_process_pool(shared, exclusion, output, device,
                                  ExecutionProcessMemoryOperation::kGetCurrentPool);
    });
}
extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return set_process_pool(shared, exclusion, device, pool);
    });
}
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr pointer) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.range(pointer, 0, true);
        if (entry && entry->kind == Kind::kProcessAllocation)
            return release_process_memory(shared, exclusion, *entry);
        if (!entry || entry->kind != Kind::kAllocation) return CUDA_ERROR_INVALID_VALUE;
        return destroy_resource(shared, exclusion, entry, Kind::kAllocation,
                                ExecutionOpcode::kRelease);
    });
}
extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr destination, CUdeviceptr source,
                                     std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return unified_copy(shared, exclusion, destination, source, bytes, nullptr, false);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return unified_copy(shared, exclusion, destination, source, bytes, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr, CUdeviceptr, std::size_t) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination, const void* source,
                                            std::size_t bytes) {
    CopyAudit audit(CopyAudit::Api::kDriverUpload, bytes);
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, destination, const_cast<void*>(source), bytes,
                             nullptr, true, false);
    });
    audit.finish(result);
    return result;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void** output, std::size_t bytes, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return host_allocate(shared, exclusion, output, bytes, flags);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void** output, std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return host_allocate(shared, exclusion, output, bytes, 0);
    });
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!pointer) return CUDA_SUCCESS;
        auto* entry = shared.host_range(pointer, 0, true);
        return entry ? shared.release_host(exclusion, *entry) : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuMemHostGetFlags(unsigned int* output, void* pointer) {
    return boundary([&](State& shared) { return host_flags(shared, output, pointer); });
}
extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* output, void* pointer,
                                                         unsigned int flags) {
    return boundary([&](State& shared) { return host_pointer(shared, output, pointer, flags); });
}
extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void*, std::size_t, unsigned int) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemHostUnregister(void*) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void* source,
                                                 std::size_t bytes, CUstream stream) {
    CopyAudit audit(CopyAudit::Api::kDriverAsyncUpload, bytes);
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, destination, const_cast<void*>(source), bytes,
                             stream, true, true);
    });
    audit.finish(result);
    return result;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* destination, CUdeviceptr source,
                                                 std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, source, destination, bytes, stream, false, true);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void* destination, CUdeviceptr source,
                                            std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, source, destination, bytes, nullptr, false, false);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source,
                                            std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.destination = destination, .source = source, .count = bytes});
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.destination = destination,
                               .source = source,
                               .count = bytes,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr destination, unsigned char value,
                                          std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr destination, unsigned char value,
                                            std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr destination, unsigned short value,
                                           std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill16,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr destination, unsigned short value,
                                             std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill16,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr destination, unsigned int value,
                                           std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill32,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr destination, unsigned int value,
                                             std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill32,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
namespace {
CUresult create_owned_event(State& shared, Exclusion& exclusion, CUevent* output,
                            unsigned int flags) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    auto* entry = shared.allocate_entry(Kind::kEvent, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter input;
    if (!input.write_u32(flags)) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result =
        shared.call(exclusion, *context, ExecutionOpcode::kCreateEvent, std::move(input));
    if (status(result) != CUDA_SUCCESS) return status(result);
    if (!result->id) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    entry->remote = result->id;
    entry->is_alive = true;
    return publish(output, reinterpret_cast<CUevent>(entry)) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult owned_event_operation(State& shared, Exclusion& exclusion, CUevent handle,
                               ExecutionOpcode operation, CUstream stream_handle = nullptr,
                               unsigned int flags = 0) {
    auto* context = shared.context();
    auto* event = shared.find(handle, Kind::kEvent);
    if (!context || !event) return CUDA_ERROR_INVALID_HANDLE;
    const bool is_foreign_wait =
        operation == ExecutionOpcode::kStreamWaitEvent && event->context != context;
    if (event->context != context && !is_foreign_wait) return CUDA_ERROR_INVALID_HANDLE;
    ExecutionPayloadWriter input;
    std::optional<CounterPin> source_use;
    if (is_foreign_wait) {
        auto* source = event->context;
        if (!source->is_alive || source->is_uncertain || source->is_closing || !source->connection)
            return CUDA_ERROR_INVALID_HANDLE;
        if (source->active_calls == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        source_use.emplace(source->active_calls, exclusion);
        if (!input.write_u64(source->connection->session())) return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (!input.write_u64(event->remote)) return CUDA_ERROR_OUT_OF_MEMORY;
    if (operation == ExecutionOpcode::kRecordEvent ||
        operation == ExecutionOpcode::kStreamWaitEvent) {
        Entry* stream{};
        const auto resolved = resolve_owned_stream(shared, exclusion, stream_handle, stream);
        if (resolved != CUDA_SUCCESS) return resolved;
        if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
        if (!input.write_u64(stream ? stream->remote : 0) || !input.write_u32(flags))
            return CUDA_ERROR_OUT_OF_MEMORY;
        ++context->submissions;
    }
    const auto result = shared.call(
        exclusion, *context, is_foreign_wait ? ExecutionOpcode::kWaitForeignEvent : operation,
        std::move(input));
    if (status(result) == CUDA_SUCCESS && operation == ExecutionOpcode::kDestroyEvent)
        event->is_alive = false;
    return status(result);
}
CUresult owned_event_elapsed(State& shared, Exclusion& exclusion, float* output, CUevent first,
                             CUevent second) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    const auto* start = shared.find(first, Kind::kEvent);
    const auto* end = shared.find(second, Kind::kEvent);
    if (!context || !start || !end || start->context != context || end->context != context)
        return CUDA_ERROR_INVALID_HANDLE;
    ExecutionPayloadWriter input;
    if (!input.write_u64(start->remote) || !input.write_u64(end->remote))
        return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply =
        shared.call(exclusion, *context, ExecutionOpcode::kEventElapsedTime, std::move(input));
    if (status(reply) != CUDA_SUCCESS) return status(reply);
    if (reply->address > UINT32_MAX) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    const auto value = std::bit_cast<float>(static_cast<std::uint32_t>(reply->address));
    return publish(output, value) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult create_owned_stream(State& shared, Exclusion& exclusion, CUstream* output,
                             unsigned int flags, std::optional<int> priority = std::nullopt) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    if (flags & ~static_cast<unsigned int>(CU_STREAM_NON_BLOCKING)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    auto* entry = shared.allocate_entry(Kind::kStream, context);
    if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
    ExecutionPayloadWriter input;
    if (!input.write_u32(flags) ||
        (priority && !input.write_u32(std::bit_cast<std::uint32_t>(*priority))))
        return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result = shared.call(
        exclusion, *context,
        priority ? ExecutionOpcode::kCreatePriorityStream : ExecutionOpcode::kCreateStream,
        std::move(input));
    if (status(result) != CUDA_SUCCESS) return status(result);
    if (!result->id || (priority && result->address > UINT32_MAX)) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    entry->remote = result->id;
    entry->stream_flags = flags;
    entry->stream_priority =
        priority ? std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(result->address)) : 0;
    entry->is_alive = true;
    return publish(output, reinterpret_cast<CUstream>(entry)) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult resolve_owned_stream(State& shared, Exclusion& exclusion, CUstream handle,
                              Entry*& output) {
    output = nullptr;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (!handle || handle == CU_STREAM_LEGACY) return CUDA_SUCCESS;
    if (handle == CU_STREAM_PER_THREAD) {
        std::erase_if(default_stream_bindings, [](const auto& item) {
            return !item.context->is_alive || item.context->epoch != item.epoch ||
                   !item.stream->is_alive;
        });
        const auto found = std::ranges::find_if(default_stream_bindings, [&](const auto& item) {
            return item.context == context && item.epoch == context->epoch;
        });
        if (found != default_stream_bindings.end()) {
            output = found->stream;
        } else {
            if (default_stream_bindings.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
            default_stream_bindings.reserve(default_stream_bindings.size() + 1);
            CUstream proxy{};
            const auto created = create_owned_stream(shared, exclusion, &proxy, CU_STREAM_DEFAULT);
            if (created != CUDA_SUCCESS) return created;
            output = shared.find(proxy, Kind::kStream);
            if (!output) {
                context->is_uncertain = true;
                return CUDA_ERROR_UNKNOWN;
            }
            default_stream_bindings.push_back({context, context->epoch, output});
        }
    } else {
        output = shared.find(handle, Kind::kStream);
    }
    return output && output->context == context && !output->is_retiring ? CUDA_SUCCESS
                                                                        : CUDA_ERROR_INVALID_HANDLE;
}
CUresult owned_stream_info(State& shared, Exclusion& exclusion, CUstream stream, void* output,
                           bool is_priority) {
    if (!writable_host_range(output, sizeof(unsigned int))) return CUDA_ERROR_INVALID_VALUE;
    Entry* entry{};
    const auto resolved = resolve_owned_stream(shared, exclusion, stream, entry);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (is_priority)
        return publish(static_cast<int*>(output), entry ? entry->stream_priority : 0)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    return publish(static_cast<unsigned int*>(output), entry ? entry->stream_flags : 0U)
               ? CUDA_SUCCESS
               : CUDA_ERROR_INVALID_VALUE;
}
}  // namespace
extern "C" CUresult CUDAAPI cuEventCreate(CUevent* output, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_event(shared, exclusion, output, flags);
    });
}
extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kRecordEvent,
                                     stream);
    });
}
extern "C" CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream,
                                                   unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kRecordEvent,
                                     stream, flags);
    });
}
extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kStreamWaitEvent,
                                     stream, flags);
    });
}
extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kQueryEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kSynchronizeEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kDestroyEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventElapsedTime(float* output, CUevent start, CUevent end) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_elapsed(shared, exclusion, output, start, end);
    });
}
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* output, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_stream(shared, exclusion, output, flags);
    });
}
extern "C" CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* output, unsigned int flags,
                                                       int priority) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_stream(shared, exclusion, output, flags, priority);
    });
}
extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_stream_info(shared, exclusion, stream, output, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_stream_info(shared, exclusion, stream, output, false);
    });
}
extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return destroy_resource(shared, exclusion, stream, Kind::kStream,
                                ExecutionOpcode::kDestroyStream);
    });
}
extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return locked_boundary([](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, std::nullopt, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, stream, false);
    });
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_x,
                                           unsigned int grid_y, unsigned int grid_z,
                                           unsigned int block_x, unsigned int block_y,
                                           unsigned int block_z, unsigned int shared_bytes,
                                           CUstream stream, void** arguments, void** extra) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return launch_kernel(shared, exclusion,
                             {function,
                              {grid_x, grid_y, grid_z},
                              {block_x, block_y, block_z},
                              shared_bytes,
                              stream,
                              arguments,
                              extra});
    });
}
namespace {
CUresult cached_entry(const char* name, void** output, int version, cuuint64_t flags,
                      CUdriverProcAddressQueryResult* symbol_status, bool is_v2) noexcept {
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (is_host_function_active) return CUDA_ERROR_NOT_PERMITTED;
    if (is_entered) return CUDA_ERROR_NOT_SUPPORTED;
    Entered entered;
    try {
        if (!writable(output) || (symbol_status && !writable(symbol_status)))
            return CUDA_ERROR_INVALID_VALUE;
        const auto symbol = owned_string(name, 256);
        if (!symbol) return CUDA_ERROR_INVALID_VALUE;
        glimmer::interceptor::execution::DriverEntryQuery query{*symbol, version, flags, is_v2,
                                                                symbol_status != nullptr};
        if (symbol_status &&
            !self_copy(&query.symbol_status, symbol_status, sizeof(query.symbol_status), false))
            return CUDA_ERROR_INVALID_VALUE;
        const auto result = glimmer::interceptor::execution::driver_entry(query);
        if (!publish(output, result.address) ||
            (symbol_status &&
             !self_copy(symbol_status, &result.symbol_status, sizeof(result.symbol_status), true)))
            return CUDA_ERROR_INVALID_VALUE;
        return result.status;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
}  // namespace
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cached_entry(name, output, version, flags, nullptr, false);
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    return cached_entry(name, output, version, flags, symbol_status, true);
}

// These compiler callbacks never invoke native registration with virtual
// handles. They snapshot CPU metadata; first actual use binds server resources.
// Exact NVIDIA compiler callback names and parameter ordering are fixed ABI.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters)
extern "C" void** CUDARTAPI __cudaRegisterFatBinary(void* wrapper) {
    void** result{};
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) { return register_runtime_image(shared, wrapper, &result); },
        false));
    return result;
}
extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host,
                                                 char* device_function, const char* device_name,
                                                 int thread_limit, uint3* thread_index,
                                                 uint3* block_index, dim3* block_dimension,
                                                 dim3* grid_dimension, int* warp_size) {
    (void)device_name;
    (void)thread_limit;
    (void)thread_index;
    (void)block_index;
    (void)block_dimension;
    (void)grid_dimension;
    (void)warp_size;
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            return register_runtime_function(shared, handle, host, device_function);
        },
        false));
}
extern "C" void CUDARTAPI __cudaRegisterFatBinaryEnd(void** handle) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            auto* image = runtime_image(shared, handle);
            if (!image || image->is_closing) return CUDA_ERROR_INVALID_HANDLE;
            image->is_finalized = true;
            return image->error;
        },
        false));
}
extern "C" void CUDARTAPI __cudaUnregisterFatBinary(void** handle) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto& lock) { return unregister_runtime_image(shared, lock, handle); },
        false));
}
namespace {
void reject_runtime_registration(void** handle) noexcept {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            auto* image = runtime_image(shared, handle);
            if (!image) return CUDA_ERROR_INVALID_HANDLE;
            image->error = CUDA_ERROR_NOT_SUPPORTED;
            return image->error;
        },
        false));
}
}  // namespace
extern "C" void CUDARTAPI __cudaRegisterVar(void** handle, char* host, char* device_symbol,
                                            const char* name, int external, std::size_t bytes,
                                            int constant, int global) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, Exclusion&) {
            return register_runtime_variable(shared, handle, host,
                                             device_symbol ? device_symbol : name, external, bytes,
                                             constant, global);
        },
        false));
}
extern "C" void CUDARTAPI __cudaRegisterManagedVar(void** handle, void**, char*, const char*, int,
                                                   std::size_t, int, int) {
    reject_runtime_registration(handle);
}
extern "C" void CUDARTAPI __cudaRegisterTexture(void** handle, const struct textureReference*,
                                                const void**, const char*, int, int, int) {
    reject_runtime_registration(handle);
}
extern "C" void CUDARTAPI __cudaRegisterSurface(void** handle, const struct surfaceReference*,
                                                const void**, const char*, int, int) {
    reject_runtime_registration(handle);
}
extern "C" char CUDARTAPI __cudaInitModule(void** handle) {
    reject_runtime_registration(handle);
    return 0;
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters)
extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    observe_async_runtime_error();
    const auto value = runtime_last_error;
    runtime_last_error = cudaSuccess;
    return value;
}
extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError() {
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (::getpid() == kProcess) observe_async_runtime_error();
    return ::getpid() == kProcess ? runtime_last_error : cudaErrorIncompatibleDriverContext;
}
namespace {
cudaError_t query_runtime_metadata(void* output, std::size_t bytes, std::uint32_t kind, int device,
                                   int attribute = 0, int peer = 0) noexcept {
    cudaError_t original_status{cudaErrorUnknown};
    const auto prepared = locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, bytes)) return CUDA_ERROR_INVALID_VALUE;
        if (CUDART_VERSION != 12060 || sizeof(void*) != 8) return CUDA_ERROR_NOT_SUPPORTED;
        const auto opened =
            kind == 6 ? runtime_context(shared, exclusion) : shared.open_host_context(exclusion);
        if (opened != CUDA_SUCCESS) return opened;
        auto* owner = kind == 6 ? shared.context() : shared.host_context;
        if (owner->active_calls >= kMaxHandles - 1) return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin origin(owner->active_calls, exclusion);
        ExecutionPayloadWriter input;
        if (!input.write_u32(kind) || !input.write_u32(std::bit_cast<std::uint32_t>(device)) ||
            !input.write_u32(std::bit_cast<std::uint32_t>(attribute)) ||
            !input.write_u32(std::bit_cast<std::uint32_t>(peer)))
            return CUDA_ERROR_OUT_OF_MEMORY;
        const auto reply =
            shared.call(exclusion, *owner, ExecutionOpcode::kRuntimeMetadata, std::move(input));
        if (!reply || reply->parameter_bytes) {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        if (reply->disposition != ExecutionWireDisposition::kComplete || !reply->original_status)
            return status(reply);
        if (reply->status != *reply->original_status || reply->status < 0 ||
            reply->status > 10000 || reply->address != 12060 ||
            (reply->status == 0 ? reply->data.size() != bytes : !reply->data.empty())) {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        original_status = static_cast<cudaError_t>(*reply->original_status);
        if (original_status == cudaSuccess && !self_copy(output, reply->data.data(), bytes, true))
            return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    });
    if (prepared != CUDA_SUCCESS) return runtime_result(prepared);
    if (original_status != cudaSuccess) runtime_last_error = original_status;
    return original_status;
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int* output) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    if (!writable_host_range(output, sizeof(*output)))
        return runtime_result(CUDA_ERROR_INVALID_VALUE);
    const auto initialized = glimmer::interceptor::execution::initialize_driver_metadata();
    if (initialized != CUDA_SUCCESS) return runtime_result(initialized);
    const auto query = original<decltype(&cuDeviceGetCount)>("cuDeviceGetCount");
    if (!query) return runtime_result(CUDA_ERROR_NOT_INITIALIZED);
    int value{};
    const auto result = query(&value);
    if (result != CUDA_SUCCESS) return runtime_result(result);
    if (value < 0) return runtime_result(CUDA_ERROR_UNKNOWN);
    return publish(output, value) ? cudaSuccess : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int* output) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    if (!writable(output)) return runtime_result(CUDA_ERROR_INVALID_VALUE);
    const auto initialized = glimmer::interceptor::execution::initialize_driver_metadata();
    if (initialized != CUDA_SUCCESS) return runtime_result(initialized);
    const auto query = original<decltype(&cuDriverGetVersion)>("cuDriverGetVersion");
    if (!query) return runtime_result(CUDA_ERROR_NOT_INITIALIZED);
    int value{};
    const auto result = query(&value);
    return runtime_result(
        result == CUDA_SUCCESS && !publish(output, value) ? CUDA_ERROR_INVALID_VALUE : result);
}
// The installed SDK expands this entry to the v2 property ABI. The legacy
// property struct has a different layout and is not silently aliased here.
extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* output, int device) {
    return query_runtime_metadata(output, sizeof(*output), 1, device);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* output, cudaDeviceAttr attribute,
                                                        int device) {
    return query_runtime_metadata(output, sizeof(*output), 2, device, static_cast<int>(attribute));
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Runtime SDK ABI.
extern "C" cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int* output, cudaDeviceP2PAttr attribute,
                                                           int source, int destination) {
    return query_runtime_metadata(output, sizeof(*output), 3, source, static_cast<int>(attribute),
                                  destination);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int* output, int device, int peer) {
    return query_runtime_metadata(output, sizeof(*output), 5, device, 0, peer);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if ((least && !writable(least)) || (greatest && !writable(greatest)))
        return runtime_result(CUDA_ERROR_INVALID_VALUE);
    std::array<int, 2> values{};
    const auto result = query_runtime_metadata(values.data(), sizeof(values), 4, 0);
    if (result != cudaSuccess) return result;
    return (!least || publish(least, values[0])) && (!greatest || publish(greatest, values[1]))
               ? cudaSuccess
               : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* available, std::size_t* total) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (!writable(available) || !writable(total)) return runtime_result(CUDA_ERROR_INVALID_VALUE);
    std::array<std::size_t, 2> values{};
    const auto result = query_runtime_metadata(values.data(), sizeof(values), 6, 0);
    if (result != cudaSuccess) return result;
    return publish(available, values[0]) && publish(total, values[1])
               ? cudaSuccess
               : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
    return runtime_result(boundary(
        [&](State&) { return publish(device, 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE; }));
}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& lock) {
        return device == 0 ? runtime_context(shared, lock, true) : CUDA_ERROR_INVALID_DEVICE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void** output, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!bytes)
            return publish(output, static_cast<void*>(nullptr)) ? CUDA_SUCCESS
                                                                : CUDA_ERROR_INVALID_VALUE;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUdeviceptr pointer{};
        const auto result = memory_allocate(shared, lock, &pointer, bytes);
        // GPU UVA is a validated numeric server address, not a host object.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* address = reinterpret_cast<void*>(pointer);
        return result == CUDA_SUCCESS && !publish(output, address) ? CUDA_ERROR_INVALID_VALUE
                                                                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS || !pointer) return initialized;
        auto* entry = shared.range(reinterpret_cast<CUdeviceptr>(pointer), 0, true);
        if (!entry || (entry->kind != Kind::kAllocation && entry->kind != Kind::kProcessAllocation))
            return CUDA_ERROR_INVALID_VALUE;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        if (entry->kind == Kind::kProcessAllocation)
            return release_process_memory(shared, lock, *entry);
        return destroy_resource(shared, lock, entry, Kind::kAllocation, ExecutionOpcode::kRelease);
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool, std::size_t bytes) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto opened = open_process_memory(shared, exclusion);
        if (opened != CUDA_SUCCESS) return opened;
        ExecutionProcessMemoryRequest request;
        request.operation = ExecutionProcessMemoryOperation::kTrimPool;
        request.bytes = bytes;
        const auto selected =
            process_pool_id(shared, reinterpret_cast<CUmemoryPool>(pool), request.pool);
        if (selected != CUDA_SUCCESS) return selected;
        const auto reply =
            call_process_memory(shared, exclusion, *shared.process_memory_context, request);
        original_status = reply.original_status;
        return reply.status;
    });
    if (result != CUDA_SUCCESS || !original_status) return runtime_result(result);
    const auto observed = static_cast<cudaError_t>(*original_status);
    if (observed != cudaSuccess) runtime_last_error = observed;
    return observed;
}
namespace {
cudaError_t runtime_symbol_result(CUresult result) noexcept {
    if (result != CUDA_ERROR_NOT_FOUND) return runtime_result(result);
    runtime_last_error = cudaErrorInvalidSymbol;
    return cudaErrorInvalidSymbol;
}
cudaError_t copy_runtime_symbol(const void* symbol, void* other, std::size_t bytes,
                                std::size_t offset, cudaMemcpyKind kind, cudaStream_t stream,
                                bool is_to_symbol, bool is_asynchronous) noexcept {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (kind != cudaMemcpyDefault && kind != cudaMemcpyDeviceToDevice &&
        kind != (is_to_symbol ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost)) {
        runtime_last_error = cudaErrorInvalidMemcpyDirection;
        return cudaErrorInvalidMemcpyDirection;
    }
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        Entry* module{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global, &module);
        if (queried != CUDA_SUCCESS) return queried;
        if (offset > global->bytes || bytes > global->bytes - offset)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        if (module->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin module_use(module->active_calls, exclusion);
        const auto device = static_cast<CUdeviceptr>(global->address + offset);
        const auto other_device = reinterpret_cast<CUdeviceptr>(other);
        const bool is_device = kind == cudaMemcpyDeviceToDevice ||
                               (kind == cudaMemcpyDefault && shared.range(other_device, bytes));
        if (is_device) {
            return execute_memory(
                shared, exclusion,
                {.destination = is_to_symbol ? device : other_device,
                 .source = is_to_symbol ? other_device : device,
                 .count = bytes,
                 .stream = is_asynchronous ? reinterpret_cast<CUstream>(stream) : nullptr,
                 .is_asynchronous = is_asynchronous});
        }
        return transfer_host(shared, exclusion, device, other, bytes,
                             is_asynchronous ? reinterpret_cast<CUstream>(stream) : nullptr,
                             is_to_symbol, is_asynchronous, AsyncNotificationDomain::kRuntime);
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGetSymbolAddress(void** output, const void* symbol) {
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global);
        if (queried != CUDA_SUCCESS) return queried;
        return publish(output, reinterpret_cast<void*>(global->address)) ? CUDA_SUCCESS
                                                                         : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaGetSymbolSize(std::size_t* output, const void* symbol) {
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global);
        if (queried != CUDA_SUCCESS) return queried;
        return publish(output, static_cast<std::size_t>(global->bytes)) ? CUDA_SUCCESS
                                                                        : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void* symbol, const void* source,
                                                    std::size_t bytes, std::size_t offset,
                                                    cudaMemcpyKind kind) {
    return copy_runtime_symbol(symbol, const_cast<void*>(source), bytes, offset, kind, nullptr,
                               true, false);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void* destination, const void* symbol,
                                                      std::size_t bytes, std::size_t offset,
                                                      cudaMemcpyKind kind) {
    return copy_runtime_symbol(symbol, destination, bytes, offset, kind, nullptr, false, false);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void* symbol, const void* source,
                                                         std::size_t bytes, std::size_t offset,
                                                         cudaMemcpyKind kind, cudaStream_t stream) {
    return copy_runtime_symbol(symbol, const_cast<void*>(source), bytes, offset, kind, stream, true,
                               true);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void* destination, const void* symbol,
                                                           std::size_t bytes, std::size_t offset,
                                                           cudaMemcpyKind kind,
                                                           cudaStream_t stream) {
    return copy_runtime_symbol(symbol, destination, bytes, offset, kind, stream, false, true);
}
extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** output, std::size_t bytes,
                                               unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_allocate(shared, exclusion, output, bytes, flags)
                                           : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** output, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_allocate(shared, exclusion, output, bytes, 0)
                                           : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!pointer) return CUDA_SUCCESS;
        auto* entry = shared.host_range(pointer, 0, true);
        return entry ? shared.release_host(exclusion, *entry) : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int* output, void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_flags(shared, output, pointer) : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** output, void* pointer,
                                                          unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        CUdeviceptr mapped{};
        const auto queried = host_pointer(shared, &mapped, pointer, flags);
        return queried == CUDA_SUCCESS && !publish(output, pointer) ? CUDA_ERROR_INVALID_VALUE
                                                                    : queried;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostRegister(void*, std::size_t, unsigned int) {
    return runtime_result(boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; }));
}
extern "C" cudaError_t CUDARTAPI cudaHostUnregister(void*) {
    return runtime_result(boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; }));
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t bytes, cudaMemcpyKind kind) {
    CopyAudit audit(CopyAudit::Api::kRuntime, bytes);
    const auto result = runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto target = reinterpret_cast<CUdeviceptr>(destination);
        const auto origin = reinterpret_cast<CUdeviceptr>(source);
        if (kind == cudaMemcpyDefault) {
            const bool to_device = shared.range(target, bytes) != nullptr;
            const bool from_device = shared.range(origin, bytes) != nullptr;
            kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                             : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
        }
        if (kind == cudaMemcpyHostToHost) {
            if (shared.graph_capture) return reject_graph_capture(shared, lock);
            if (bytes > kExecutionMaxPayloadBytes) return CUDA_ERROR_NOT_SUPPORTED;
            std::vector<std::byte> owned(bytes);
            return self_copy(owned.data(), source, bytes, false) &&
                           self_copy(destination, owned.data(), bytes, true)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        if (kind == cudaMemcpyDeviceToDevice)
            return execute_memory(shared, lock,
                                  {.destination = target, .source = origin, .count = bytes});
        return transfer_host(
            shared, lock, kind == cudaMemcpyHostToDevice ? target : origin,
            kind == cudaMemcpyHostToDevice ? const_cast<void*>(source) : destination, bytes,
            nullptr, kind == cudaMemcpyHostToDevice, false, AsyncNotificationDomain::kRuntime);
    }));
    audit.finish(result);
    return result;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact public Runtime ABI.
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t bytes, cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
    CopyAudit audit(CopyAudit::Api::kRuntimeAsync, bytes);
    const auto result = runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto target = reinterpret_cast<CUdeviceptr>(destination);
        const auto origin = reinterpret_cast<CUdeviceptr>(source);
        if (kind == cudaMemcpyDefault) {
            const bool to_device = shared.range(target, bytes) != nullptr;
            const bool from_device = shared.range(origin, bytes) != nullptr;
            kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                             : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
        }
        if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost)
            return transfer_host(
                shared, exclusion, kind == cudaMemcpyHostToDevice ? target : origin,
                kind == cudaMemcpyHostToDevice ? const_cast<void*>(source) : destination, bytes,
                reinterpret_cast<CUstream>(stream), kind == cudaMemcpyHostToDevice, true,
                AsyncNotificationDomain::kRuntime);
        if (kind != cudaMemcpyDeviceToDevice) return CUDA_ERROR_NOT_SUPPORTED;
        return execute_memory(shared, exclusion,
                              {.destination = target,
                               .source = origin,
                               .count = bytes,
                               .stream = reinterpret_cast<CUstream>(stream),
                               .is_asynchronous = true});
    }));
    audit.finish(result);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaMemset(void* destination, int value, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = reinterpret_cast<CUdeviceptr>(destination),
                               .count = bytes,
                               .value = static_cast<unsigned char>(value)});
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* destination, int value, std::size_t bytes,
                                                 cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = reinterpret_cast<CUdeviceptr>(destination),
                               .count = bytes,
                               .value = static_cast<unsigned char>(value),
                               .stream = reinterpret_cast<CUstream>(stream),
                               .is_asynchronous = true});
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* output,
                                                           unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (flags & ~static_cast<unsigned int>(cudaStreamNonBlocking))
            return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUstream stream{};
        const auto result = create_owned_stream(shared, lock, &stream, flags);
        return result == CUDA_SUCCESS && !publish(output, reinterpret_cast<cudaStream_t>(stream))
                   ? CUDA_ERROR_INVALID_VALUE
                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* output,
                                                              unsigned int flags, int priority) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto stopped = runtime_quiescent(shared, exclusion);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUstream stream{};
        const auto result = create_owned_stream(shared, exclusion, &stream, flags, priority);
        return result == CUDA_SUCCESS && !publish(output, reinterpret_cast<cudaStream_t>(stream))
                   ? CUDA_ERROR_INVALID_VALUE
                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int* output) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? owned_stream_info(shared, exclusion, reinterpret_cast<CUstream>(stream),
                                       output, true)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int* output) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? owned_stream_info(shared, exclusion, reinterpret_cast<CUstream>(stream),
                                       output, false)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                                       cudaStreamCaptureStatus* output) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return graph_capture_status(shared, exclusion, stream, output, original);
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,
                                                        cudaStreamCaptureMode mode) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_begin(shared, exclusion, stream, mode, original)
                                           : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* output) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_end(shared, exclusion, stream, output, original)
                                           : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* output,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_instantiate(shared, exclusion, output, graph, flags, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_destroy(shared, exclusion, graph, Kind::kGraph, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_destroy(shared, exclusion, executable, Kind::kGraphExecutable, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_launch(shared, exclusion, executable, stream)
                                           : initialized;
    }));
}
extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec executable, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return graph_launch(shared, exclusion, executable, stream);
    });
}
extern "C" cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    return runtime_result(locked_boundary(
        [&](State&, Exclusion&) {
            cudaStreamCaptureMode requested{};
            if (!writable(mode) || !self_copy(&requested, mode, sizeof(requested), false) ||
                requested < cudaStreamCaptureModeGlobal || requested > cudaStreamCaptureModeRelaxed)
                return CUDA_ERROR_INVALID_VALUE;
            if (!publish(mode, runtime_capture_mode)) return CUDA_ERROR_INVALID_VALUE;
            runtime_capture_mode = requested;
            return CUDA_SUCCESS;
        },
        false));
}
// The SDK expands the modern six-argument entry to cudaStreamGetCaptureInfo_v2.
// Only capture status is defined outside capture; leave all other outputs alone.
extern "C" cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id,
    cudaGraph_t* graph, const cudaGraphNode_t** dependencies, std::size_t* count) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(status)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        cudaStreamCaptureStatus observed{};
        const auto queried = graph_capture_status(shared, exclusion, stream, &observed, original);
        if (queried != CUDA_SUCCESS || (original && *original)) return queried;
        if (observed != cudaStreamCaptureStatusNone && (id || graph || dependencies || count))
            return reject_graph_capture(shared, exclusion);
        return publish(status, observed) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* output, unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? create_owned_event(shared, lock, reinterpret_cast<CUevent*>(output), flags)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* output) {
    return cudaEventCreateWithFlags(output, cudaEventDefault);
}
namespace {
cudaError_t runtime_event_operation(cudaEvent_t event, ExecutionOpcode operation,
                                    cudaStream_t stream = nullptr, unsigned int flags = 0) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? owned_event_operation(shared, lock, reinterpret_cast<CUevent>(event),
                                           operation, reinterpret_cast<CUstream>(stream), flags)
                   : initialized;
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return runtime_event_operation(event, ExecutionOpcode::kRecordEvent, stream);
}
extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                          unsigned int flags) {
    return runtime_event_operation(event, ExecutionOpcode::kRecordEvent, stream, flags);
}
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                     unsigned int flags) {
    return runtime_event_operation(event, ExecutionOpcode::kStreamWaitEvent, stream, flags);
}
extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kQueryEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kSynchronizeEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kDestroyEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* output, cudaEvent_t start,
                                                      cudaEvent_t end) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? owned_event_elapsed(shared, lock, output, reinterpret_cast<CUevent>(start),
                                         reinterpret_cast<CUevent>(end))
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* output) {
    return cudaStreamCreateWithFlags(output, cudaStreamDefault);
}
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        auto* entry = shared.find(stream, Kind::kStream);
        if (!entry || entry->context != shared.context()) return CUDA_ERROR_INVALID_HANDLE;
        const auto stopped = runtime_quiescent(shared, lock, FrontierErrorPolicy::kPhysicalOnly);
        if (stopped != CUDA_SUCCESS) return stopped;
        return destroy_resource(shared, lock, stream, Kind::kStream,
                                ExecutionOpcode::kDestroyStream);
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, stream, true, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, stream, false, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t function,
                                                    void* data) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!function) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? enqueue_host_function(shared, exclusion, stream, function, data)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t function,
                                                         void* data) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!function) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        const auto selected = stream ? stream : cudaStreamPerThread;
        return initialized == CUDA_SUCCESS
                   ? enqueue_host_function(shared, exclusion, selected, function, data)
                   : initialized;
    }));
}
namespace {
cudaError_t synchronize_runtime_device() {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, std::nullopt, true, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
cudaError_t reset_runtime_device() {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        auto* context = shared.context();
        // Runtime reset releases its primary context, not a user-created context.
        // Its Runtime-owned subresources need a separate precise reset path.
        if (context && !context->is_primary) return CUDA_ERROR_NOT_SUPPORTED;
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, lock, 0, false);
        if (result != CUDA_SUCCESS) return result;
        if (shared.runtime_owns_primary) {
            if (!shared.primary || !shared.primary->primary_references) return CUDA_ERROR_UNKNOWN;
            --shared.primary->primary_references;
            shared.runtime_owns_primary = false;
        }
        return primary ? shared.retired_driver_error(*primary, epoch) : CUDA_SUCCESS;
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    return synchronize_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaThreadSynchronize() {
    return synchronize_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaDeviceReset() {
    return reset_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaThreadExit() {
    return reset_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaFuncGetAttributes(cudaFuncAttributes* output,
                                                       const void* function) {
    return runtime_result(locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            return runtime_function_control(shared, exclusion, function,
                                            ExecutionFunctionControl::kGetAttributes, 0, 0, output);
        },
        false));
}
extern "C" cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* function,
                                                      cudaFuncAttribute attribute, int value) {
    return runtime_result(locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            const auto driver_attribute = runtime_function_attribute(attribute);
            if (!driver_attribute) return CUDA_ERROR_INVALID_VALUE;
            return runtime_function_control(shared, exclusion, function,
                                            ExecutionFunctionControl::kSetAttribute,
                                            static_cast<std::uint32_t>(*driver_attribute), value);
        },
        false));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* host, dim3 grid, dim3 block,
                                                  void** arguments, std::size_t shared_bytes,
                                                  cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (shared_bytes > UINT32_MAX) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        // Cold module/function factories require logical quiescence. Release
        // exclusion while waiting, exactly like explicit synchronization.
        RuntimeImage* image{};
        const RuntimeKernel* kernel{};
        const auto selected = select_runtime_kernel(shared, host, image, kernel);
        if (selected != CUDA_SUCCESS) return selected;
        const bool is_prepared = cached_runtime_function(shared, *image, host) != nullptr;
        bool is_capturing{};
        if (shared.graph_capture) {
            Entry* capture_stream{};
            const auto resolved = resolve_owned_stream(shared, lock, stream, capture_stream);
            if (resolved != CUDA_SUCCESS) return resolved;
            if (!capture_stream || !graph_capture_owner(shared, shared.context(), capture_stream))
                return reject_graph_capture(shared, lock);
            is_capturing = true;
        }
        if (!is_prepared && !is_capturing) {
            const auto stopped = runtime_quiescent(shared, lock);
            if (stopped != CUDA_SUCCESS) return stopped;
        }
        CUfunction function{};
        const GraphPreparationScope capture_preparation(is_capturing);
        const auto prepared = runtime_function(shared, lock, host, &function);
        if (prepared != CUDA_SUCCESS) return prepared;
        return launch_kernel(shared, lock,
                             {function,
                              {grid.x, grid.y, grid.z},
                              {block.x, block.y, block.z},
                              static_cast<unsigned int>(shared_bytes),
                              stream,
                              arguments,
                              nullptr},
                             true);
    }));
}
// Metadata-only Runtime queries never consume a virtual resource or initialize
// a caller-native context. Preserve the installed library's actual strings.
#define GLIMMER_RUNTIME_ERROR_QUERY(NAME)                                                          \
    extern "C" const char* CUDARTAPI NAME(cudaError_t error) {                                     \
        if (::getpid() != kProcess) return "cudaErrorIncompatibleDriverContext";                   \
        if (is_host_function_active) return "cudaErrorNotPermitted";                               \
        static const auto native = reinterpret_cast<decltype(&(NAME))>(::dlsym(RTLD_NEXT, #NAME)); \
        return native ? native(error) : "CUDA Runtime error metadata unavailable";                 \
    }
GLIMMER_RUNTIME_ERROR_QUERY(cudaGetErrorName)
GLIMMER_RUNTIME_ERROR_QUERY(cudaGetErrorString)
#undef GLIMMER_RUNTIME_ERROR_QUERY

namespace {
struct RuntimeEntryStyle {
    bool is_versioned{}, is_ptds{};
};
cudaError_t runtime_cached_entry(const char* name, void** output, unsigned int version,
                                 unsigned long long flags,
                                 cudaDriverEntryPointQueryResult* symbol_status,
                                 RuntimeEntryStyle style) noexcept {
    cudaError_t status{cudaErrorInvalidValue};
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    try {
        if (writable(output) && (!symbol_status || writable(symbol_status))) {
            if (const auto symbol = owned_string(name, 256)) {
                glimmer::interceptor::execution::RuntimeEntryQuery query{
                    *symbol,       version,
                    flags,         style.is_versioned,
                    style.is_ptds, symbol_status != nullptr};
                if (!symbol_status || self_copy(&query.symbol_status, symbol_status,
                                                sizeof(query.symbol_status), false)) {
                    const auto result = glimmer::interceptor::execution::runtime_entry(query);
                    status =
                        publish(output, result.address) &&
                                (!symbol_status || self_copy(symbol_status, &result.symbol_status,
                                                             sizeof(result.symbol_status), true))
                            ? result.status
                            : cudaErrorInvalidValue;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        status = cudaErrorMemoryAllocation;
    } catch (...) {
        status = cudaErrorUnknown;
    }
    if (status != cudaSuccess) runtime_last_error = status;
    return status;
}
}  // namespace
extern "C" cudaError_t CUDARTAPI
cudaGetDriverEntryPoint(const char* name, void** output, unsigned long long flags,
                        cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, 0, flags, symbol_status, {false, false});
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char* name, void** output, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, version, flags, symbol_status, {true, false});
}
extern "C" cudaError_t CUDARTAPI
cudaGetDriverEntryPoint_ptsz(const char* name, void** output, unsigned long long flags,
                             cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, 0, flags, symbol_status, {false, true});
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char* name, void** output, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, version, flags, symbol_status, {true, true});
}
extern "C" cudaError_t CUDARTAPI cudaGetExportTable(const void** output, const cudaUUID_t* id) {
    return runtime_result(locked_boundary(
        [&](State&, auto&) {
            cudaUUID_t copied{};
            if (!writable(output) || !self_copy(&copied, id, sizeof(copied), false) ||
                !publish(output, static_cast<const void*>(nullptr)))
                return CUDA_ERROR_INVALID_VALUE;
            return CUDA_ERROR_NOT_SUPPORTED;
        },
        false));
}

namespace {
cublasStatus_t blas_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_SUCCESS:
            return CUBLAS_STATUS_SUCCESS;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return CUBLAS_STATUS_ALLOC_FAILED;
        case CUDA_ERROR_INVALID_VALUE:
            return CUBLAS_STATUS_INVALID_VALUE;
        case CUDA_ERROR_NOT_SUPPORTED:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        default:
            return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
cublasStatus_t blas_status(const std::optional<ExecutionWireReply>& reply,
                           bool allows_replacement = false) noexcept {
    if (!reply) return CUBLAS_STATUS_INTERNAL_ERROR;
    if (reply->disposition == ExecutionWireDisposition::kComputeReplacement) {
        if (!allows_replacement || reply->original_status || !reply->id)
            return CUBLAS_STATUS_INTERNAL_ERROR;
        if (reply->status == CUDA_ERROR_LAUNCH_FAILED ||
            reply->status == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ||
            reply->status == CUDA_ERROR_LAUNCH_TIMEOUT)
            return CUBLAS_STATUS_EXECUTION_FAILED;
        return blas_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition == ExecutionWireDisposition::kComplete && reply->original_status &&
        reply->status == *reply->original_status) {
        const auto value = *reply->original_status;
        return value >= 0 && value <= 31 ? static_cast<cublasStatus_t>(value)
                                         : CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return blas_error(static_cast<CUresult>(reply->status));
}
template <typename Function>
cublasStatus_t blas_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (is_entered) return CUBLAS_STATUS_NOT_SUPPORTED;
    Entered entered;
    try {
        auto& shared = state();
        const auto entered_at = shared.rpc_metrics
                                    ? shared.rpc_metrics->entry()
                                    : glimmer::interceptor::execution::RpcMetrics::Stamp{};
        Exclusion exclusion(shared.mutex);
        glimmer::interceptor::execution::RpcMetrics::Scope boundary_observation(
            shared.rpc_metrics.get(), exclusion, entered_at, true);
        if (!shared.config) return CUBLAS_STATUS_NOT_INITIALIZED;
        return function(shared, exclusion);
    } catch (const std::bad_alloc&) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
template <typename Function>
cublasStatus_t blas_handle(cublasHandle_t handle, bool is_mutating, Function&& function,
                           bool is_command = false, Kind kind = Kind::kBlas,
                           bool allows_blas_alias = false) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.find(handle, kind);
        if (!entry && allows_blas_alias) entry = shared.find(handle, Kind::kBlas);
        auto* context = shared.context();
        if (!entry || !context || entry->context != context || context->is_closing ||
            context->is_uncertain || entry->is_uncertain)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        if (context->active_calls >= kMaxHandles - 1 ||
            ((is_mutating || is_command) && context->active_mutations == kMaxHandles) ||
            entry->active_calls == kMaxHandles)
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto epoch = context->epoch;
        // Read-only queries must drain after context sealing, not prevent
        // sealing indefinitely when another thread keeps polling metadata.
        std::optional<CounterPin> work;
        if (is_mutating || is_command) work.emplace(context->active_mutations, exclusion);
        CounterPin origin(context->active_calls, exclusion);
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && context->epoch == epoch;
        };
        while (entry->is_retiring) {
            if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        std::optional<FlagPin> mutation;
        if (is_mutating) {
            mutation.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls) {
                if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
            }
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
std::optional<ExecutionWireReply> query_blas(State& shared, Exclusion& exclusion, Entry& entry,
                                             ExecutionBlasAttribute attribute) {
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry.remote) || !input.write_u32(static_cast<std::uint32_t>(attribute)))
        return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
    return shared.call(exclusion, *entry.context, ExecutionOpcode::kGetBlasAttribute,
                       std::move(input));
}
cublasStatus_t set_blas_value(cublasHandle_t handle, ExecutionBlasAttribute attribute,
                              std::uint64_t value) noexcept {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) ||
            !input.write_u32(static_cast<std::uint32_t>(attribute)) || !input.write_u64(value))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasAttribute, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS && attribute == ExecutionBlasAttribute::kPointerMode)
            entry.blas_pointer_mode = static_cast<unsigned int>(value);
        return result;
    });
}
template <typename Value>
cublasStatus_t get_blas_value(cublasHandle_t handle, ExecutionBlasAttribute attribute,
                              Value* output) noexcept {
    static thread_local off_t hint{-1};
    return blas_handle(handle, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!RpcMetrics::Scope::host_range(
                shared.rpc_metrics.get(), exclusion, RpcMetrics::HostRangeCaller::kBlasValue,
                [&](RpcMetrics::HostTrace* trace) noexcept {
                    return writable_host_range(output, sizeof(*output), trace,
                                               shared.should_seek_host_range ? &hint : nullptr);
                }))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto reply = query_blas(shared, exclusion, entry, attribute);
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) return result;
        const auto maximum = (attribute == ExecutionBlasAttribute::kPointerMode ||
                              attribute == ExecutionBlasAttribute::kAtomicsMode)
                                 ? std::uint64_t{1}
                             : attribute == ExecutionBlasAttribute::kMathMode
                                 ? std::uint64_t{31}
                                 : static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (reply->address > maximum || (attribute == ExecutionBlasAttribute::kPointerMode &&
                                         reply->address != entry.blas_pointer_mode))
            return CUBLAS_STATUS_INTERNAL_ERROR;
        const auto value = static_cast<Value>(reply->address);
        return publish(output, value) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    });
}
}  // namespace
namespace {
CUresult open_lt_metadata(State& shared, Exclusion& exclusion) {
    if (!shared.lt_metadata_context) {
        auto* context = shared.allocate_entry(Kind::kLtMetadataContext, nullptr);
        if (!context) return CUDA_ERROR_OUT_OF_MEMORY;
        context->context = context;
        context->is_alive = true;
        shared.lt_metadata_context = context;
    }
    return shared.open_context(exclusion, *shared.lt_metadata_context, 0);
}
cublasStatus_t lt_status(const std::optional<ExecutionWireReply>& reply, Entry& context,
                         LtEntry* entry = nullptr) noexcept {
    if (!reply || reply->parameter_bytes > 1 || reply->parameter_bytes == 1) {
        context.is_uncertain = true;
        if (entry) entry->is_uncertain = true;
        if (!reply || reply->parameter_bytes > 1) return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return blas_status(reply);
}
struct LtMetadataCreate {
    ExecutionLtObjectKind kind{};
    std::uint32_t type{CUDA_R_32F}, compute{CUBLAS_COMPUTE_32F}, scale{CUDA_R_32F};
    std::uint64_t rows{}, columns{};
    std::int64_t leading{};
};
template <typename Handle>
cublasStatus_t create_lt_metadata(Handle* output, const LtMetadataCreate& specification) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(std::uintptr_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        auto* context = shared.lt_metadata_context;
        if (!shared.has_lt_tokens || shared.lt_entries.size() == kMaxHandles ||
            context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CUBLAS_STATUS_ALLOC_FAILED;
        auto entry = std::make_shared<LtEntry>();
        entry->context = context;
        entry->kind = specification.kind;
        entry->epoch = context->epoch;
        entry->token = shared.next_lt_token;
        entry->scale_type = specification.scale;
        if (shared.next_lt_token == UINTPTR_MAX)
            shared.has_lt_tokens = false;
        else
            ++shared.next_lt_token;
        ExecutionPayloadWriter input;
        if (!input.write_u32(static_cast<std::uint32_t>(specification.kind)) ||
            !input.write_u32(specification.type) || !input.write_u32(specification.compute) ||
            !input.write_u32(specification.scale) || !input.write_u64(specification.rows) ||
            !input.write_u64(specification.columns) ||
            !input.write_u64(std::bit_cast<std::uint64_t>(specification.leading)))
            return CUBLAS_STATUS_ALLOC_FAILED;
        // Occupy bounded capacity before releasing exclusion. Unpublished or
        // uncertain native ownership is never replayed or silently reclaimed.
        shared.lt_entries.push_back(entry);
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateLtMetadata, std::move(input));
        const auto result = lt_status(reply, *context, entry.get());
        if (result != CUBLAS_STATUS_SUCCESS) {
            if (!context->is_uncertain && reply && !reply->id &&
                ((reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->status == *reply->original_status) ||
                 (!reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kRejected &&
                  reply->status != CUDA_SUCCESS)))
                std::erase(shared.lt_entries, entry);
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        // Public opaque identity only: no pointer arithmetic or dereference.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto handle = reinterpret_cast<Handle>(static_cast<std::uintptr_t>(entry->token));
        if (!publish(output, handle)) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
template <typename Handle, typename Function>
cublasStatus_t lt_metadata(Handle handle, ExecutionLtObjectKind kind, bool is_mutating,
                           Function&& function) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        const auto token = reinterpret_cast<std::uintptr_t>(handle);
        const auto found = std::ranges::find_if(shared.lt_entries, [=](const auto& value) {
            return value->token == token && value->kind == kind && value->is_alive;
        });
        if (found == shared.lt_entries.end()) return CUBLAS_STATUS_NOT_INITIALIZED;
        // The strong record keeps counters/retirement pins valid even after
        // successful destruction removes the record from the public registry.
        const auto entry = *found;
        auto* context = entry->context;
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && !context->is_closing && context->epoch == entry->epoch;
        };
        if (!valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        if (entry->active_calls == kMaxHandles || context->active_calls >= kMaxHandles - 1 ||
            (is_mutating && context->active_mutations >= kMaxHandles - 1))
            return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin origin(context->active_calls, exclusion);
        std::optional<CounterPin> work;
        if (is_mutating) work.emplace(context->active_mutations, exclusion);
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        std::optional<FlagPin> retiring;
        if (is_mutating) {
            retiring.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls)
                if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
template <typename Handle>
cublasStatus_t destroy_lt_metadata(Handle handle, ExecutionLtObjectKind kind) noexcept {
    return lt_metadata(handle, kind, true, [](State& shared, Exclusion& exclusion, LtEntry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kDestroyLtMetadata, std::move(input));
        const auto result = lt_status(reply, *entry.context, &entry);
        if (result == CUBLAS_STATUS_SUCCESS) {
            entry.is_alive = false;
            std::erase_if(shared.lt_entries,
                          [&](const auto& value) { return value.get() == &entry; });
        }
        return result;
    });
}
std::optional<std::size_t> lt_attribute_bytes(ExecutionLtObjectKind kind,
                                              std::uint32_t attribute) noexcept {
    if (kind == ExecutionLtObjectKind::kLayout && attribute <= CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET)
        return attribute == CUBLASLT_MATRIX_LAYOUT_TYPE ||
                       attribute == CUBLASLT_MATRIX_LAYOUT_ORDER ||
                       attribute == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
                   ? 4
                   : 8;
    if (kind == ExecutionLtObjectKind::kOperation) {
        if (attribute == CUBLASLT_MATMUL_DESC_FAST_ACCUM) return 1;
        if (attribute <= CUBLASLT_MATMUL_DESC_EPILOGUE ||
            attribute == CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET ||
            attribute == CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)
            return 4;
        if (attribute == CUBLASLT_MATMUL_DESC_BIAS_POINTER ||
            attribute == CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE ||
            attribute == CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE)
            return 8;
    }
    if (kind == ExecutionLtObjectKind::kPreference) {
        if (attribute == CUBLASLT_MATMUL_PREF_SEARCH_MODE ||
            attribute == CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK ||
            (attribute >= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES &&
             attribute <= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES) ||
            attribute == CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT)
            return 4;
        if (attribute == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES ||
            attribute == CUBLASLT_MATMUL_PREF_IMPL_MASK)
            return 8;
    }
    return std::nullopt;
}
template <typename Handle>
cublasStatus_t set_lt_attribute(Handle handle, ExecutionLtObjectKind kind, std::uint32_t attribute,
                                const void* buffer, std::size_t bytes) noexcept {
    return lt_metadata(
        handle, kind, true, [&](State& shared, Exclusion& exclusion, LtEntry& entry) {
            const auto width = lt_attribute_bytes(kind, attribute);
            if (!width) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (bytes != *width) return CUBLAS_STATUS_INVALID_VALUE;
            std::vector<std::byte> data(bytes);
            if (!self_copy(data.data(), buffer, bytes, false)) return CUBLAS_STATUS_INVALID_VALUE;
            Entry* allocation{};
            std::uint64_t pointer{};
            std::optional<CounterPin> allocation_origin;
            const bool is_pointer = kind == ExecutionLtObjectKind::kOperation &&
                                    attribute == CUBLASLT_MATMUL_DESC_BIAS_POINTER;
            if (is_pointer) {
                if (bytes != sizeof(pointer)) return CUBLAS_STATUS_INVALID_VALUE;
                std::memcpy(&pointer, data.data(), sizeof(pointer));
                if (pointer) {
                    allocation = shared.range(pointer, 1);
                    if (!allocation || allocation->is_retiring || allocation->context->is_closing ||
                        allocation->context->is_uncertain)
                        return CUBLAS_STATUS_INVALID_VALUE;
                    if (allocation->context->active_calls == kMaxHandles)
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    allocation_origin.emplace(allocation->context->active_calls, exclusion);
                }
                std::ranges::fill(data, std::byte{0});
            }
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(attribute) ||
                !input.write_blob(data) || !input.write_u32(allocation ? 1 : 0))
                return CUBLAS_STATUS_ALLOC_FAILED;
            if (allocation && (!input.write_u64(allocation->context->connection->session()) ||
                               !input.write_u64(allocation->remote) ||
                               !input.write_u64(pointer - allocation->address)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kSetLtAttribute, std::move(input));
            const auto result = lt_status(reply, *entry.context, &entry);
            if (result == CUBLAS_STATUS_SUCCESS && kind == ExecutionLtObjectKind::kOperation &&
                bytes == 4) {
                if (attribute == CUBLASLT_MATMUL_DESC_SCALE_TYPE)
                    std::memcpy(&entry.scale_type, data.data(), 4);
                if (attribute == CUBLASLT_MATMUL_DESC_POINTER_MODE)
                    std::memcpy(&entry.pointer_mode, data.data(), 4);
            }
            return result;
        });
}
template <typename Handle>
cublasStatus_t get_lt_attribute(Handle handle, ExecutionLtObjectKind kind, std::uint32_t attribute,
                                void* buffer, std::size_t bytes, std::size_t* written) noexcept {
    return lt_metadata(
        handle, kind, false, [&](State& shared, Exclusion& exclusion, LtEntry& entry) {
            if (bytes > 4096 || (!bytes && !written) ||
                (bytes && !writable_host_range(buffer, bytes)) ||
                (written && !writable_host_range(written, sizeof(*written))))
                return CUBLAS_STATUS_INVALID_VALUE;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(attribute) ||
                !input.write_u64(bytes))
                return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetLtAttribute, std::move(input));
            const auto result = lt_status(reply, *entry.context, &entry);
            if (result != CUBLAS_STATUS_SUCCESS) return result;
            if (!reply || reply->address > 4096 ||
                (bytes ? reply->data.size() != reply->address || reply->address > bytes
                       : !reply->data.empty())) {
                entry.context->is_uncertain = entry.is_uncertain = true;
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            const auto count = static_cast<std::size_t>(reply->address);
            if ((bytes && !self_copy(buffer, reply->data.data(), count, true)) ||
                (written && !publish(written, count)))
                return CUBLAS_STATUS_INVALID_VALUE;
            return CUBLAS_STATUS_SUCCESS;
        });
}
std::size_t lt_version(bool runtime) noexcept {
    std::size_t value{};
    const auto result = blas_boundary([&](State& shared, Exclusion& exclusion) {
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        ExecutionPayloadWriter input;
        if (!input.write_u32(runtime ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *shared.lt_metadata_context,
                                       ExecutionOpcode::kLtVersion, std::move(input));
        const auto status = lt_status(reply, *shared.lt_metadata_context);
        if (status == CUBLAS_STATUS_SUCCESS) {
            if (!reply || reply->original_status || reply->address > SIZE_MAX)
                return CUBLAS_STATUS_INTERNAL_ERROR;
            value = static_cast<std::size_t>(reply->address);
        }
        return status;
    });
    return result == CUBLAS_STATUS_SUCCESS ? value : 0;
}
struct LtDescriptorArguments {
    void* operation{};
    std::array<void*, 4> matrices{};
    void* preference{};
};
template <typename Function>
cublasStatus_t with_lt_descriptors(State& shared, Exclusion& exclusion, Entry& handle,
                                   const LtDescriptorArguments& arguments, Function&& function) {
    std::array<std::shared_ptr<LtEntry>, 6> entries;
    const std::array<void*, 6> handles{arguments.operation,   arguments.matrices[0],
                                       arguments.matrices[1], arguments.matrices[2],
                                       arguments.matrices[3], arguments.preference};
    auto* metadata = shared.lt_metadata_context;
    if (!metadata || !metadata->connection || metadata->is_uncertain || metadata->is_closing)
        return CUBLAS_STATUS_NOT_INITIALIZED;
    for (;;) {
        bool waiting{};
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if ((index == 3 || index == 5) && !handles[index]) continue;
            const auto kind = index == 0   ? ExecutionLtObjectKind::kOperation
                              : index == 5 ? ExecutionLtObjectKind::kPreference
                                           : ExecutionLtObjectKind::kLayout;
            const auto token = reinterpret_cast<std::uintptr_t>(handles[index]);
            const auto found = std::ranges::find_if(shared.lt_entries, [&](const auto& entry) {
                return entry->token == token && entry->kind == kind && entry->is_alive;
            });
            if (found == shared.lt_entries.end()) return CUBLAS_STATUS_NOT_INITIALIZED;
            entries[index] = *found;
            const auto& entry = **found;
            if (entry.context != metadata || entry.epoch != metadata->epoch || entry.is_uncertain)
                return CUBLAS_STATUS_NOT_INITIALIZED;
            if (entry.active_calls > kMaxHandles - entries.size())
                return CUBLAS_STATUS_ALLOC_FAILED;
            waiting |= entry.is_retiring;
        }
        if (!waiting) break;
        if (!yield_frontend(exclusion) || handle.context->is_uncertain || metadata->is_uncertain ||
            metadata->is_closing)
            return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (metadata->active_calls >= kMaxHandles - 1) return CUBLAS_STATUS_ALLOC_FAILED;
    CounterPin origin(metadata->active_calls, exclusion);
    std::array<std::optional<CounterPin>, 6> pins;
    for (std::size_t index = 0; index < entries.size(); ++index)
        if (entries[index]) pins[index].emplace(entries[index]->active_calls, exclusion);
    ExecutionPayloadWriter input;
    if (!input.write_u64(handle.remote) || !input.write_u64(metadata->connection->session()))
        return CUBLAS_STATUS_ALLOC_FAILED;
    for (std::size_t index = 0; index < 5; ++index)
        if (!input.write_u64(entries[index] ? entries[index]->remote : 0))
            return CUBLAS_STATUS_ALLOC_FAILED;
    return function(entries, input);
}
cublasStatus_t query_lt_algorithms(cublasLtHandle_t handle, const LtDescriptorArguments& arguments,
                                   int requested, const cublasLtMatmulAlgo_t* algorithm,
                                   cublasLtMatmulHeuristicResult_t* output,
                                   int* returned) noexcept {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const bool is_check = algorithm != nullptr;
            if (requested <= 0 || !output ||
                (!is_check && !writable_host_range(returned, sizeof(*returned))))
                return CUBLAS_STATUS_INVALID_VALUE;
            if (requested > 64) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (!writable_host_range(output, sizeof(*output) * static_cast<std::size_t>(requested)))
                return CUBLAS_STATUS_INVALID_VALUE;
            cublasLtMatmulAlgo_t copied{};
            if (is_check && !self_copy(&copied, algorithm, sizeof(copied), false))
                return CUBLAS_STATUS_INVALID_VALUE;
            return with_lt_descriptors(
                shared, exclusion, entry, arguments,
                [&](const auto& descriptors, ExecutionPayloadWriter& input) {
                    if (is_check) {
                        if (!input.write_blob(std::as_bytes(std::span{&copied, 1})))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    } else if (!descriptors[5] || !input.write_u64(descriptors[5]->remote) ||
                               !input.write_u32(static_cast<std::uint32_t>(requested)))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    const auto reply = shared.call(exclusion, *entry.context,
                                                   is_check ? ExecutionOpcode::kLtAlgorithmCheck
                                                            : ExecutionOpcode::kLtHeuristic,
                                                   std::move(input));
                    const auto status = lt_status(reply, *entry.context);
                    if (status != CUBLAS_STATUS_SUCCESS) return status;
                    const auto invalid = [&] {
                        entry.is_uncertain = entry.context->is_uncertain = true;
                        return CUBLAS_STATUS_INTERNAL_ERROR;
                    };
                    if (!reply || !reply->address) return invalid();
                    ExecutionPayloadReader result(reply->data);
                    std::uint32_t count{};
                    if (!result.read_u32(count) || count > static_cast<std::uint32_t>(requested) ||
                        (is_check && count != 1))
                        return invalid();
                    std::vector<cublasLtMatmulHeuristicResult_t> values(count);
                    for (auto& value : values) {
                        std::span<const std::byte> bytes;
                        std::uint64_t workspace{};
                        std::uint32_t state{}, waves{};
                        if (!result.read_blob(bytes) || bytes.size() != sizeof(value.algo) ||
                            !result.read_u64(workspace) || workspace > SIZE_MAX ||
                            !result.read_u32(state) || !result.read_u32(waves))
                            return invalid();
                        std::memcpy(&value.algo, bytes.data(), bytes.size());
                        value.workspaceSize = static_cast<std::size_t>(workspace);
                        value.state = static_cast<cublasStatus_t>(state);
                        value.wavesCount = std::bit_cast<float>(waves);
                        for (auto& reserved : value.reserved) {
                            std::uint32_t bits{};
                            if (!result.read_u32(bits)) return invalid();
                            reserved = std::bit_cast<std::int32_t>(bits);
                        }
                    }
                    if (!result.empty()) return invalid();
                    if (is_check) {
                        // AlgoCheck never updates result->algo, including when
                        // its input algorithm is a different nonaliasing object.
                        constexpr auto offset =
                            offsetof(cublasLtMatmulHeuristicResult_t, workspaceSize);
                        if (!self_copy(reinterpret_cast<std::byte*>(output) + offset,
                                       reinterpret_cast<const std::byte*>(values.data()) + offset,
                                       sizeof(*output) - offset, true))
                            return CUBLAS_STATUS_INVALID_VALUE;
                    } else if (!self_copy(output, values.data(), values.size() * sizeof(*output),
                                          true) ||
                               !publish(returned, static_cast<int>(count)))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    return CUBLAS_STATUS_SUCCESS;
                });
        },
        false, Kind::kLt, true);
}
}  // namespace
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact installed SDK ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* output,
                                                                  cudaDataType type,
                                                                  std::uint64_t rows,
                                                                  std::uint64_t columns,
                                                                  std::int64_t leading) {
    return create_lt_metadata(output,
                              {ExecutionLtObjectKind::kLayout, static_cast<std::uint32_t>(type),
                               CUBLAS_COMPUTE_32F, CUDA_R_32F, rows, columns, leading});
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* output,
                                                                cublasComputeType_t compute,
                                                                cudaDataType scale) {
    return create_lt_metadata(
        output, {ExecutionLtObjectKind::kOperation, CUDA_R_32F, static_cast<std::uint32_t>(compute),
                 static_cast<std::uint32_t>(scale), 0, 0, 0});
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* output) {
    return create_lt_metadata(output, {ExecutionLtObjectKind::kPreference});
}
#define LT_ATTRIBUTES(PREFIX, HANDLE, ATTRIBUTE, KIND)                                          \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##Destroy(HANDLE handle) {                     \
        return destroy_lt_metadata(handle, ExecutionLtObjectKind::KIND);                        \
    }                                                                                           \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##SetAttribute(                                \
        HANDLE handle, ATTRIBUTE attribute, const void* buffer, std::size_t bytes) {            \
        return set_lt_attribute(handle, ExecutionLtObjectKind::KIND,                            \
                                static_cast<std::uint32_t>(attribute), buffer, bytes);          \
    }                                                                                           \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##GetAttribute(                                \
        HANDLE handle, ATTRIBUTE attribute, void* buffer, std::size_t bytes,                    \
        std::size_t* written) {                                                                 \
        return get_lt_attribute(handle, ExecutionLtObjectKind::KIND,                            \
                                static_cast<std::uint32_t>(attribute), buffer, bytes, written); \
    }
LT_ATTRIBUTES(cublasLtMatrixLayout, cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t,
              kLayout)
LT_ATTRIBUTES(cublasLtMatmulDesc, cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, kOperation)
LT_ATTRIBUTES(cublasLtMatmulPreference, cublasLtMatmulPreference_t,
              cublasLtMatmulPreferenceAttributes_t, kPreference)
#undef LT_ATTRIBUTES
extern "C" std::size_t CUBLASWINAPI cublasLtGetVersion() {
    return lt_version(false);
}
extern "C" std::size_t CUBLASWINAPI cublasLtGetCudartVersion() {
    return lt_version(true);
}
// These entry points must not reach the caller's SDK with owned opaque
// descriptors. GPU execution and caller-allocated descriptor storage need
// their own authenticated contracts, rather than a native fallback.
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed Lt ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* output) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(cublasLtHandle_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return blas_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        auto* entry = shared.allocate_entry(Kind::kLt, context);
        if (!entry) return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateLt, ExecutionPayloadWriter{});
        const auto result = lt_status(reply, *context);
        if (result != CUBLAS_STATUS_SUCCESS || context->is_uncertain) {
            if (!context->is_uncertain && reply && !reply->id &&
                ((reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->status == *reply->original_status) ||
                 (!reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kRejected && reply->status)))
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_INTERNAL_ERROR : result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<cublasLtHandle_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t handle) {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), true,
        [](State& shared, Exclusion& exclusion, Entry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyLt,
                                           std::move(input));
            const auto result = lt_status(reply, *entry.context);
            if (result == CUBLAS_STATUS_SUCCESS) entry.is_alive = false;
            if (entry.context->is_uncertain) entry.is_uncertain = true;
            return result;
        },
        false, Kind::kLt);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    cublasLtMatmulPreference_t preference, int requested, cublasLtMatmulHeuristicResult_t* output,
    int* returned) {
    return query_lt_algorithms(handle, {operation, {a, b, c, d}, preference}, requested, nullptr,
                               output, returned);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    const cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulHeuristicResult_t* output) {
    if (!algorithm) return CUBLAS_STATUS_INVALID_VALUE;
    return query_lt_algorithms(handle, {operation, {a, b, c, d}, nullptr}, 1, algorithm, output,
                               nullptr);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulAlgoCapAttributes_t attribute,
    void* buffer, std::size_t bytes, std::size_t* written) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (bytes > 4096) return CUBLAS_STATUS_NOT_SUPPORTED;
        cublasLtMatmulAlgo_t copied{};
        std::size_t initial_written{};
        if (!self_copy(&copied, algorithm, sizeof(copied), false) ||
            (bytes && buffer && !writable_host_range(buffer, bytes)) ||
            (written && (!writable(written) ||
                         !self_copy(&initial_written, written, sizeof(initial_written), false))))
            return CUBLAS_STATUS_INVALID_VALUE;
        if (bytes && buffer && written) {
            const auto left = reinterpret_cast<std::uintptr_t>(buffer);
            const auto right = reinterpret_cast<std::uintptr_t>(written);
            if (left <= right ? right - left < bytes : left - right < sizeof(*written))
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        std::vector<std::byte> initial(buffer ? bytes : 0);
        if (!initial.empty() && !self_copy(initial.data(), buffer, bytes, false))
            return CUBLAS_STATUS_INVALID_VALUE;
        if (shared.lt_metadata_context &&
            (shared.lt_metadata_context->is_uncertain || shared.lt_metadata_context->is_closing))
            return CUBLAS_STATUS_NOT_INITIALIZED;
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        auto& context = *shared.lt_metadata_context;
        if (context.is_uncertain || context.is_closing) return CUBLAS_STATUS_NOT_INITIALIZED;
        if (context.active_calls >= kMaxHandles - 1) return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin use(context.active_calls, exclusion);
        const auto presence = (buffer ? 1U : 0U) | (written ? 2U : 0U);
        ExecutionPayloadWriter input;
        if (!input.write_blob(std::as_bytes(std::span{&copied, 1})) ||
            !input.write_u32(static_cast<std::uint32_t>(attribute)) || !input.write_u64(bytes) ||
            !input.write_u32(presence) || !input.write_blob(initial) ||
            !input.write_u64(initial_written))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, context, ExecutionOpcode::kLtAlgorithmCapability,
                                       std::move(input));
        // Validate the status domain before conversion to an SDK enum and
        // before publishing output, including native error-side effects.
        if (reply && ((reply->original_status &&
                       (*reply->original_status < 0 || *reply->original_status > 31)) ||
                      (!reply->original_status && (reply->status < 0 || reply->status > 999)))) {
            context.is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        const auto result = lt_status(reply, context);
        if (!reply || context.is_uncertain)
            return result == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_INTERNAL_ERROR : result;
        const auto invalid = [&] {
            context.is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        };
        if (reply->disposition != ExecutionWireDisposition::kComplete) {
            if (reply->id || reply->address || !reply->data.empty() || reply->original_status ||
                result == CUBLAS_STATUS_SUCCESS)
                return invalid();
            return result;
        }
        if (!reply->original_status || reply->status != *reply->original_status || reply->id ||
            !reply->arguments.empty() || reply->address > SIZE_MAX ||
            reply->data.size() != initial.size() ||
            (!written && reply->address != initial_written) ||
            (result == CUBLAS_STATUS_SUCCESS && written && bytes && reply->address > bytes))
            return invalid();
        const auto count = static_cast<std::size_t>(reply->address);
        if ((!initial.empty() && !self_copy(buffer, reply->data.data(), initial.size(), true)) ||
            (written && !publish(written, count)))
            return CUBLAS_STATUS_INVALID_VALUE;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatrixTransform(cublasLtHandle_t, cublasLtMatrixTransformDesc_t, const void*, const void*,
                        cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
                        void*, cublasLtMatrixLayout_t, cudaStream_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(
    cublasLtMatrixLayout_t, std::size_t, cudaDataType, std::uint64_t, std::uint64_t, std::int64_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t,
                                                                       std::size_t,
                                                                       cublasComputeType_t,
                                                                       cudaDataType) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t, std::size_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t a_layout, const void* b, cublasLtMatrixLayout_t b_layout,
    const void* beta, const void* c, cublasLtMatrixLayout_t c_layout, void* d,
    cublasLtMatrixLayout_t d_layout, const cublasLtMatmulAlgo_t* algorithm, void* workspace,
    std::size_t workspace_bytes, cudaStream_t stream) {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            if (entry.context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            if (stream == cudaStreamPerThread) return CUBLAS_STATUS_NOT_SUPPORTED;
            auto* binding =
                stream && stream != cudaStreamLegacy ? shared.find(stream, Kind::kStream) : nullptr;
            if (stream && stream != cudaStreamLegacy &&
                (!binding || binding->context != entry.context || binding->is_retiring))
                return CUBLAS_STATUS_INVALID_VALUE;
            return with_lt_descriptors(
                shared, exclusion, entry,
                {operation, {a_layout, b_layout, c_layout, d_layout}, nullptr},
                [&](const auto& descriptors, ExecutionPayloadWriter& input) {
                    if (!input.write_u32(algorithm ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
                    if (algorithm) {
                        cublasLtMatmulAlgo_t copy{};
                        if (!self_copy(&copy, algorithm, sizeof(copy), false))
                            return CUBLAS_STATUS_INVALID_VALUE;
                        if (!input.write_blob(std::as_bytes(std::span{&copy, 1})))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    }
                    if (!input.write_u64(binding ? binding->remote : 0) ||
                        !input.write_u64(workspace_bytes))
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    const auto range = [&](const void* pointer, std::size_t bytes,
                                           bool optional = false) {
                        if (!pointer && optional && !bytes)
                            return input.write_u64(0) && input.write_u64(0);
                        const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                        auto* allocation = shared.range(address, bytes);
                        return allocation && !allocation->is_retiring &&
                               allocation->context == entry.context &&
                               input.write_u64(allocation->remote) &&
                               input.write_u64(address - allocation->address);
                    };
                    if (!range(workspace, workspace_bytes, true))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    for (const auto* matrix : {a, b, c, static_cast<const void*>(d)})
                        if (!range(matrix, 1)) return CUBLAS_STATUS_INVALID_VALUE;
                    const auto type = static_cast<cudaDataType>(descriptors[0]->scale_type);
                    std::size_t width{};
                    switch (type) {
                        case CUDA_R_16F:
                            width = 2;
                            break;
                        case CUDA_R_32F:
                        case CUDA_R_32I:
                            width = 4;
                            break;
                        case CUDA_R_64F:
                        case CUDA_C_32F:
                            width = 8;
                            break;
                        case CUDA_C_64F:
                            width = 16;
                            break;
                        default:
                            return CUBLAS_STATUS_NOT_SUPPORTED;
                    }
                    const auto mode = descriptors[0]->pointer_mode;
                    if (mode > 1) return CUBLAS_STATUS_NOT_SUPPORTED;
                    if (!input.write_u32(mode) ||
                        !input.write_u32(static_cast<std::uint32_t>(width)))
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    if (mode) {
                        if (!range(alpha, width) || !range(beta, width))
                            return CUBLAS_STATUS_INVALID_VALUE;
                    } else {
                        std::array<std::byte, 16> av{}, bv{};
                        if (!self_copy(av.data(), alpha, width, false) ||
                            !self_copy(bv.data(), beta, width, false))
                            return CUBLAS_STATUS_INVALID_VALUE;
                        if (!input.write_blob(std::span{av}.first(width)) ||
                            !input.write_blob(std::span{bv}.first(width)))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    }
                    ++entry.context->submissions;
                    return blas_status(shared.call(exclusion, *entry.context,
                                                   ExecutionOpcode::kLtMatmul, std::move(input)),
                                       true);
                });
        },
        true, Kind::kLt, true);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
namespace {
curandStatus_t random_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_ERROR_OUT_OF_MEMORY:
            return CURAND_STATUS_ALLOCATION_FAILED;
        case CUDA_ERROR_INVALID_VALUE:
            return CURAND_STATUS_OUT_OF_RANGE;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return CURAND_STATUS_NOT_INITIALIZED;
        case CUDA_ERROR_NOT_SUPPORTED:
            return CURAND_STATUS_TYPE_ERROR;
        default:
            return CURAND_STATUS_INTERNAL_ERROR;
    }
}
curandStatus_t random_status(const std::optional<ExecutionWireReply>& reply) noexcept {
    if (!reply) return CURAND_STATUS_INTERNAL_ERROR;
    if (reply->disposition == ExecutionWireDisposition::kComplete && reply->original_status &&
        reply->status == *reply->original_status) {
        const auto value = *reply->original_status;
        if (value == 0 || (value >= 100 && value <= 106) || (value >= 201 && value <= 204) ||
            value == 999)
            return static_cast<curandStatus_t>(value);
        return CURAND_STATUS_INTERNAL_ERROR;
    }
    return random_error(static_cast<CUresult>(reply->status));
}
template <typename Function>
curandStatus_t random_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return CURAND_STATUS_NOT_INITIALIZED;
    if (is_entered) return CURAND_STATUS_TYPE_ERROR;
    Entered entered;
    try {
        auto& shared = state();
        const auto entered_at = shared.rpc_metrics
                                    ? shared.rpc_metrics->entry()
                                    : glimmer::interceptor::execution::RpcMetrics::Stamp{};
        Exclusion exclusion(shared.mutex);
        glimmer::interceptor::execution::RpcMetrics::Scope boundary_observation(
            shared.rpc_metrics.get(), exclusion, entered_at, true);
        if (!shared.config) return CURAND_STATUS_NOT_INITIALIZED;
        return function(shared, exclusion);
    } catch (const std::bad_alloc&) {
        return CURAND_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return CURAND_STATUS_INTERNAL_ERROR;
    }
}
template <typename Function>
curandStatus_t random_handle(curandGenerator_t handle, Function&& function) noexcept {
    return random_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.find(handle, Kind::kRandom);
        auto* context = shared.context();
        if (!entry || !context || entry->context != context || context->is_closing ||
            context->is_uncertain || entry->is_uncertain)
            return CURAND_STATUS_NOT_INITIALIZED;
        if (context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1 || entry->active_calls == kMaxHandles)
            return CURAND_STATUS_ALLOCATION_FAILED;
        const auto epoch = context->epoch;
        CounterPin work(context->active_mutations, exclusion);
        CounterPin origin(context->active_calls, exclusion);
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && context->epoch == epoch;
        };
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CURAND_STATUS_NOT_INITIALIZED;
        FlagPin mutation(entry->is_retiring, exclusion);
        while (entry->active_calls)
            if (!yield_frontend(exclusion) || !valid()) return CURAND_STATUS_NOT_INITIALIZED;
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
}  // namespace
extern "C" curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t* output,
                                                          curandRngType_t type) {
    return random_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(*output))) return CURAND_STATUS_OUT_OF_RANGE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return random_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CURAND_STATUS_NOT_INITIALIZED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        ExecutionPayloadWriter input;
        std::uint32_t type_bits{};
        static_assert(sizeof(type) == sizeof(type_bits));
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (!input.write_u32(type_bits)) return CURAND_STATUS_ALLOCATION_FAILED;
        auto* entry = shared.allocate_entry(Kind::kRandom, context);
        if (!entry) return CURAND_STATUS_ALLOCATION_FAILED;
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateRand, std::move(input));
        const auto result = random_status(reply);
        if (result != CURAND_STATUS_SUCCESS) {
            if (reply && !reply->id &&
                ((reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->original_status && reply->status == *reply->original_status) ||
                 (reply->disposition == ExecutionWireDisposition::kRejected &&
                  !reply->original_status &&
                  (reply->status == CUDA_ERROR_INVALID_VALUE ||
                   reply->status == CUDA_ERROR_OUT_OF_MEMORY ||
                   reply->status == CUDA_ERROR_NOT_SUPPORTED))))
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CURAND_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<curandGenerator_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CURAND_STATUS_INTERNAL_ERROR;
        }
        return CURAND_STATUS_SUCCESS;
    });
}
extern "C" curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t handle,
                                                                       unsigned long long seed) {
    return random_handle(handle, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) || !input.write_u64(seed))
            return CURAND_STATUS_ALLOCATION_FAILED;
        return random_status(shared.call(exclusion, *entry.context, ExecutionOpcode::kSetRandSeed,
                                         std::move(input)));
    });
}
extern "C" curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t handle) {
    return random_handle(handle, [](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CURAND_STATUS_ALLOCATION_FAILED;
        const auto reply =
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyRand, std::move(input));
        const auto result = random_status(reply);
        if (result == CURAND_STATUS_SUCCESS)
            entry.is_alive = false;
        else if (!reply || reply->original_status)
            entry.is_uncertain = true;
        return result;
    });
}

// No client-native fallback may consume an opaque server generator. These are
// explicit policy rejections, not reports of an original SDK invocation.
extern "C" {
curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t*, curandRngType_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t, unsigned long long) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t, curandOrdering_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(curandGenerator_t, unsigned int) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetStream(curandGenerator_t, cudaStream_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerate(curandGenerator_t, unsigned int*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t, unsigned long long*,
                                                std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t, float*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateUniformDouble(curandGenerator_t, double*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t, float*, std::size_t, float,
                                              float) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t, double*, std::size_t, double,
                                                    double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t, float*, std::size_t, float,
                                                 float) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLogNormalDouble(curandGenerator_t, double*, std::size_t,
                                                       double, double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t, unsigned int*, std::size_t,
                                               double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGeneratePoissonMethod(curandGenerator_t, unsigned int*, std::size_t,
                                                     double, curandMethod_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateBinomial(curandGenerator_t, unsigned int*, std::size_t,
                                                unsigned int, double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateBinomialMethod(curandGenerator_t, unsigned int*, std::size_t,
                                                      unsigned int, double, curandMethod_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
}

extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* output) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(cublasHandle_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return blas_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations == kMaxHandles)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        auto* entry = shared.allocate_entry(Kind::kBlas, context);
        if (!entry) return CUBLAS_STATUS_ALLOC_FAILED;
        ExecutionPayloadWriter input;
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateBlas, std::move(input));
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) {
            if (reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                reply->original_status && reply->status == *reply->original_status && !reply->id) {
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            } else {
                context->is_uncertain = entry->is_uncertain = true;
            }
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<cublasHandle_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    return blas_handle(handle, true, [](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply =
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyBlas, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS)
            entry.is_alive = false;
        else if (!reply || reply->original_status)
            entry.is_uncertain = true;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                                             std::size_t bytes) {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        Entry* allocation{};
        std::uint64_t offset{};
        std::optional<CounterPin> workspace_use;
        if (workspace) {
            const auto address = reinterpret_cast<CUdeviceptr>(workspace);
            allocation = shared.range(address, bytes ? bytes : 1);
            if (!allocation || allocation->context != entry.context ||
                shared.range_is_retiring(*allocation))
                return CUBLAS_STATUS_INVALID_VALUE;
            if (allocation->active_calls >= kMaxHandles) return CUBLAS_STATUS_ALLOC_FAILED;
            workspace_use.emplace(allocation->active_calls, exclusion);
            offset = address - allocation->address;
        } else if (bytes) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) ||
            !input.write_u64(allocation ? allocation->remote : 0) || !input.write_u64(offset) ||
            !input.write_u64(bytes))
            return CUBLAS_STATUS_ALLOC_FAILED;
        return blas_status(shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasWorkspace, std::move(input)));
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle,
                                                            cublasAtomicsMode_t mode) {
    std::uint32_t bits{};
    static_assert(sizeof(mode) == sizeof(bits));
    std::memcpy(&bits, &mode, sizeof(bits));
    return set_blas_value(handle, ExecutionBlasAttribute::kAtomicsMode, bits);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle,
                                                            cublasAtomicsMode_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kAtomicsMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetSmCountTarget(cublasHandle_t handle, int count) {
    return set_blas_value(handle, ExecutionBlasAttribute::kSmCountTarget,
                          static_cast<std::uint64_t>(count));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetSmCountTarget(cublasHandle_t handle, int* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kSmCountTarget, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle,
                                                          cudaStream_t stream) {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (stream == cudaStreamLegacy || stream == cudaStreamPerThread)
            return CUBLAS_STATUS_NOT_SUPPORTED;
        auto* binding = stream ? shared.find(stream, Kind::kStream) : nullptr;
        if (stream && (!binding || binding->context != entry.context || binding->is_retiring))
            return CUBLAS_STATUS_INVALID_VALUE;
        ExecutionPayloadWriter input;
        const auto remote = binding ? binding->remote : 0;
        if (!input.write_u64(entry.remote) ||
            !input.write_u32(static_cast<std::uint32_t>(ExecutionBlasAttribute::kStream)) ||
            !input.write_u64(remote))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasAttribute, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS) entry.blas_stream = remote;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle,
                                                          cudaStream_t* output) {
    return blas_handle(handle, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!RpcMetrics::Scope::host_range(
                shared.rpc_metrics.get(), exclusion, RpcMetrics::HostRangeCaller::kBlasStream,
                [&](RpcMetrics::HostTrace* trace) noexcept {
                    return writable_host_range(output, sizeof(cudaStream_t), trace);
                }))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto reply = query_blas(shared, exclusion, entry, ExecutionBlasAttribute::kStream);
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) return result;
        if (reply->address != entry.blas_stream) return CUBLAS_STATUS_INTERNAL_ERROR;
        cudaStream_t stream{};
        if (reply->address) {
            const auto found = std::ranges::find_if(shared.entries, [&](const auto& value) {
                return value->kind == Kind::kStream && value->context == entry.context &&
                       value->remote == reply->address && value->is_alive;
            });
            if (found == shared.entries.end()) return CUBLAS_STATUS_INTERNAL_ERROR;
            stream = reinterpret_cast<cudaStream_t>(found->get());
        }
        return publish(output, stream) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                               cublasPointerMode_t mode) {
    return set_blas_value(handle, ExecutionBlasAttribute::kPointerMode,
                          static_cast<std::uint64_t>(mode));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle,
                                                               cublasPointerMode_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kPointerMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    return set_blas_value(handle, ExecutionBlasAttribute::kMathMode,
                          static_cast<std::uint64_t>(mode));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle,
                                                         cublasMath_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kMathMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kVersion, output);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuBLAS ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t ta,
                                                      cublasOperation_t tb, int rows, int columns,
                                                      int depth, const float* alpha, const float* a,
                                                      int leading_a, const float* b, int leading_b,
                                                      const float* beta, float* c, int leading_c) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            // Positive-size matrix starts must not bind to the preceding
            // allocation's one-past-end address when allocations are adjacent.
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, sizeof(float)) || !range(beta, sizeof(float)))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::uint32_t av{}, bv{};
                if (!self_copy(&av, alpha, sizeof(av), false) ||
                    !self_copy(&bv, beta, sizeof(bv), false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_u32(av) || !input.write_u32(bv)) return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(
                shared.call(exclusion, *context, ExecutionOpcode::kSgemm, std::move(input)), true);
        },
        true);
}

namespace {
std::size_t blas_ex_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
    std::size_t width{};
    switch (compute) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            width = 2;
            break;
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_PEDANTIC:
        case CUBLAS_COMPUTE_32F_FAST_16F:
        case CUBLAS_COMPUTE_32F_FAST_16BF:
        case CUBLAS_COMPUTE_32F_FAST_TF32:
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            width = 4;
            break;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            width = 8;
            break;
        default:
            return 0;
    }
    const bool is_complex = output == CUDA_C_16F || output == CUDA_C_16BF || output == CUDA_C_32F ||
                            output == CUDA_C_64F || output == CUDA_C_8I || output == CUDA_C_8U ||
                            output == CUDA_C_16I || output == CUDA_C_16U || output == CUDA_C_32I ||
                            output == CUDA_C_32U || output == CUDA_C_64I || output == CUDA_C_64U;
    return width * (is_complex ? 2U : 1U);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): original installed C ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const void* alpha, const void* a, cudaDataType a_type, int leading_a, const void* b,
    cudaDataType b_type, int leading_b, const void* beta, void* c, cudaDataType c_type,
    int leading_c, cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const auto scalar_bytes = blas_ex_scalar_bytes(compute, c_type);
            if (!scalar_bytes) return CUBLAS_STATUS_NOT_SUPPORTED;
            for (const auto type : {a_type, b_type, c_type}) {
                const auto bits = std::bit_cast<std::uint32_t>(type);
                if (bits > 29) return CUBLAS_STATUS_INVALID_VALUE;
                if (bits >= 16 && bits <= 19) return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            const auto algorithm_bits = std::bit_cast<std::int32_t>(algorithm);
            if (algorithm_bits < -1 || algorithm_bits > 127) return CUBLAS_STATUS_INVALID_VALUE;
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            for (const auto value :
                 {static_cast<std::uint32_t>(a_type), static_cast<std::uint32_t>(b_type),
                  static_cast<std::uint32_t>(c_type), static_cast<std::uint32_t>(compute),
                  std::bit_cast<std::uint32_t>(algorithm_bits)})
                if (!input.write_u32(value)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1U : 0U)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, scalar_bytes) || !range(beta, scalar_bytes))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::array<std::byte, 16> av{}, bv{};
                if (!self_copy(av.data(), alpha, scalar_bytes, false) ||
                    !self_copy(bv.data(), beta, scalar_bytes, false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_blob(std::span{av}.first(scalar_bytes)) ||
                    !input.write_blob(std::span{bv}.first(scalar_bytes)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(
                shared.call(exclusion, *context, ExecutionOpcode::kGemmEx, std::move(input)), true);
        },
        true);
}

namespace {
// NOLINTBEGIN(bugprone-easily-swappable-parameters): shared exact SDK argument order.
cublasStatus_t submit_strided_blas(cublasHandle_t handle, cublasOperation_t ta,
                                   cublasOperation_t tb, int rows, int columns, int depth,
                                   const void* alpha, const void* a, cudaDataType a_type,
                                   int leading_a, long long stride_a, const void* b,
                                   cudaDataType b_type, int leading_b, long long stride_b,
                                   const void* beta, void* c, cudaDataType c_type, int leading_c,
                                   long long stride_c, int batches, cublasComputeType_t compute,
                                   cublasGemmAlgo_t algorithm, ExecutionOpcode operation) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const auto scalar_bytes = blas_ex_scalar_bytes(compute, c_type);
            if (!scalar_bytes) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (batches <= 0) return CUBLAS_STATUS_INVALID_VALUE;
            // Inspect raw external enum bits before evaluating either enum.
            std::uint32_t transpose_a_bits{}, transpose_b_bits{};
            static_assert(sizeof(ta) == sizeof(transpose_a_bits) &&
                          sizeof(tb) == sizeof(transpose_b_bits));
            std::memcpy(&transpose_a_bits, &ta, sizeof(transpose_a_bits));
            std::memcpy(&transpose_b_bits, &tb, sizeof(transpose_b_bits));
            if (transpose_a_bits > 2 || transpose_b_bits > 2) return CUBLAS_STATUS_INVALID_VALUE;
            // Installed cuBLAS faults asynchronously on a valid reverse hull.
            // Do not expose that known-unsafe native path in the shared owner.
            if (batches > 1 && (stride_a < 0 || stride_b < 0 || stride_c < 0))
                return CUBLAS_STATUS_NOT_SUPPORTED;
            for (const auto type : {a_type, b_type, c_type}) {
                const auto bits = std::bit_cast<std::uint32_t>(type);
                if (bits > 29) return CUBLAS_STATUS_INVALID_VALUE;
                if (bits >= 16 && bits <= 19) return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            const auto algorithm_bits = std::bit_cast<std::int32_t>(algorithm);
            if (algorithm_bits < -1 || algorithm_bits > 127) return CUBLAS_STATUS_INVALID_VALUE;
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            for (const auto stride : {stride_a, stride_b, stride_c})
                if (!input.write_u64(std::bit_cast<std::uint64_t>(stride)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u32(static_cast<std::uint32_t>(batches)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value :
                 {static_cast<std::uint32_t>(a_type), static_cast<std::uint32_t>(b_type),
                  static_cast<std::uint32_t>(c_type), static_cast<std::uint32_t>(compute),
                  std::bit_cast<std::uint32_t>(algorithm_bits)})
                if (!input.write_u32(value)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1U : 0U)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, scalar_bytes) || !range(beta, scalar_bytes))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::array<std::byte, 16> av{}, bv{};
                if (!self_copy(av.data(), alpha, scalar_bytes, false) ||
                    !self_copy(bv.data(), beta, scalar_bytes, false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_blob(std::span{av}.first(scalar_bytes)) ||
                    !input.write_blob(std::span{bv}.first(scalar_bytes)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(shared.call(exclusion, *context, operation, std::move(input)), true);
        },
        true);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const void* alpha, const void* a, cudaDataType a_type, int leading_a,
    long long stride_a, const void* b, cudaDataType b_type, int leading_b, long long stride_b,
    const void* beta, void* c, cudaDataType c_type, int leading_c, long long stride_c, int batches,
    cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
    return submit_strided_blas(handle, ta, tb, rows, columns, depth, alpha, a, a_type, leading_a,
                               stride_a, b, b_type, leading_b, stride_b, beta, c, c_type, leading_c,
                               stride_c, batches, compute, algorithm,
                               ExecutionOpcode::kStridedGemmEx);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const float* alpha, const float* a, int leading_a, long long stride_a,
    const float* b, int leading_b, long long stride_b, const float* beta, float* c, int leading_c,
    long long stride_c, int batches) {
    return submit_strided_blas(handle, ta, tb, rows, columns, depth, alpha, a, CUDA_R_32F,
                               leading_a, stride_a, b, CUDA_R_32F, leading_b, stride_b, beta, c,
                               CUDA_R_32F, leading_c, stride_c, batches, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT, ExecutionOpcode::kStridedSgemm);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

#include "geam_frontend.inc"
#include "blas_extension_frontend.inc"

#ifdef GLIMMER_HAS_NCCL
#include "nccl_frontend.inc"
#endif

#ifdef GLIMMER_HAS_CUDNN
#include "dnn_frontend.inc"
#include "library_info_frontend.inc"
#include "convolution_frontend.inc"
#include "backward_convolution_frontend.inc"
#include "folded_convolution_frontend.inc"
#include "pooling_frontend.inc"
#include "lrn_frontend.inc"
#include "reduction_frontend.inc"
#include "batch_norm_frontend.inc"
#include "dropout_frontend.inc"
#include "op_tensor_frontend.inc"
#include "reorder_filter_frontend.inc"
#include "rnn_data_frontend.inc"
#include "rnn_attention_frontend.inc"
#include "rnn_query_frontend.inc"
#include "rnn_compute_frontend.inc"
#include "rnn_weight_params_frontend.inc"
#if CUDNN_MAJOR == 8
#include "legacy_frontend.inc"
#endif
#include "attn_weights_frontend.inc"
#include "attention_frontend.inc"
#include "backend_graph_frontend.inc"
#endif
