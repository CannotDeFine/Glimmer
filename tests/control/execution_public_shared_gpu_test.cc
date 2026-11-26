#include "runtime.h"
#include "profile_archive.h"
#include "internal/process_identity.h"
#include "internal/cuda_gemm_signal_layout.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::backend::cuda;
using namespace glimmer::control;
std::atomic<unsigned int> creates{}, destroys{}, common_destroys{};
std::atomic<CUcontext> common_context{};
std::vector<unsigned int> workers;
// Owner-thread-only bounded observations. They never supply execution authority.
enum class KernelKind : std::uint8_t { kOther, kLow, kHigh, kOriginalLow };
struct LowArguments {
    CUdeviceptr output{};
    std::uint32_t count{}, padding{}, add{}, marker{}, unused[2]{};
    std::uint32_t x{}, y{}, z{}, grid_padding{};
    CUdeviceptr counter{}, retreat{}, completion{};
    bool operator==(const LowArguments&) const = default;
};
static_assert(sizeof(LowArguments) == 72 && offsetof(LowArguments, x) == 32 &&
              offsetof(LowArguments, counter) == 48);
struct PriorityTrace {
    struct Function {
        CUfunction native{};
        CUmodule module{};
        KernelKind kind{};
    };
    struct Launch {
        Function function;
        CUcontext context{};
        CUstream stream{};
        unsigned int workers{};
        LowArguments arguments;
        std::uint64_t serial{};
    };
    struct Upload {
        CUdeviceptr address{};
        std::uint64_t value{}, serial{};
    };
    struct Free {
        CUdeviceptr address{};
        std::uint64_t serial{};
    };
    std::array<Function, 64> functions{};
    std::array<Launch, 32> launches{};
    std::array<Upload, 128> uploads{};
    std::array<Free, 64> frees{};
    std::size_t function_count{}, launch_count{}, upload_count{}, free_count{};
    std::uint64_t serial{};
    bool has_mismatch{}, observes_completion{};
    void begin_phase() noexcept {
        launch_count = upload_count = free_count = 0;
        serial = 0;
    }
};
PriorityTrace* priority_trace{};
struct GemmTrace {
    struct Mapping {
        std::byte* host{};
        std::size_t bytes{};
        CUdeviceptr device{};
    };
    std::array<Mapping, 16> mappings{};
    std::size_t mapping_count{}, allocations{}, releases{};
    CUfunction function{};
    CUstream stream{};
    CUcontext context{};
    CUdeviceptr signal{};
    std::array<std::byte, 104> arguments{};
    unsigned int launches{}, high_gemms{};
    std::uint64_t second_launch_return_ns{};
    bool mismatch{}, measure_timing{};
};
GemmTrace* gemm_trace{};
std::uint64_t monotonic_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}
bool valid_socket_address(std::string_view path) noexcept {
    return !path.empty() && path.size() < sizeof(sockaddr_un::sun_path);
}
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
CUresult CUDAAPI create_context(CUcontext* output, unsigned int flags, CUdevice device) {
    const auto status = cuCtxCreate(output, flags, device);
    if (status == CUDA_SUCCESS) {
        ++creates;
        CUcontext empty{};
        (void)common_context.compare_exchange_strong(empty, *output);
    }
    return status;
}
CUresult CUDAAPI destroy_context(CUcontext context) {
    const auto status = cuCtxDestroy(context);
    if (status == CUDA_SUCCESS) {
        ++destroys;
        if (context == common_context.load()) ++common_destroys;
    }
    return status;
}
CUresult CUDAAPI get_function(CUfunction* output, CUmodule module, const char* name) {
    const auto status = cuModuleGetFunction(output, module, name);
    if (status == CUDA_SUCCESS && gemm_trace &&
        std::string_view{name} == "glimmer_owned_sgemm_nn_observed")
        gemm_trace->function = *output;
    if (status == CUDA_SUCCESS && priority_trace) {
        auto& trace = *priority_trace;
        const auto kind = std::string_view{name} == "binary_add__glimmer_ptb_preemptive"
                              ? KernelKind::kLow
                          : std::string_view{name} == "binary_high" ? KernelKind::kHigh
                          : std::string_view{name} == "binary_add"  ? KernelKind::kOriginalLow
                                                                    : KernelKind::kOther;
        if (trace.function_count < trace.functions.size())
            trace.functions[trace.function_count++] = {*output, module, kind};
        else
            trace.has_mismatch = true;
    }
    return status;
}
CUresult CUDAAPI host_allocate(void** output, std::size_t bytes, unsigned int flags) {
    const auto status = cuMemHostAlloc(output, bytes, flags);
    if (status == CUDA_SUCCESS && gemm_trace) {
        auto& trace = *gemm_trace;
        ++trace.allocations;
        const auto mappings = std::span{trace.mappings}.first(trace.mapping_count);
        const auto free = std::ranges::find(mappings, nullptr, &GemmTrace::Mapping::host);
        if (free != mappings.end())
            *free = {static_cast<std::byte*>(*output), bytes, 0};
        else if (trace.mapping_count < trace.mappings.size())
            trace.mappings[trace.mapping_count++] = {static_cast<std::byte*>(*output), bytes, 0};
        else
            trace.mismatch = true;
    }
    return status;
}
CUresult CUDAAPI host_map(CUdeviceptr* output, void* host, unsigned int flags) {
    const auto status = cuMemHostGetDevicePointer(output, host, flags);
    if (status == CUDA_SUCCESS && gemm_trace) {
        auto& trace = *gemm_trace;
        const auto mappings = std::span{trace.mappings}.first(trace.mapping_count);
        const auto found = std::ranges::find(mappings, host, &GemmTrace::Mapping::host);
        if (found != mappings.end()) found->device = *output;
    }
    return status;
}
CUresult CUDAAPI host_release(void* host) {
    const auto status = cuMemFreeHost(host);
    if (status == CUDA_SUCCESS && gemm_trace) {
        auto& trace = *gemm_trace;
        const auto mappings = std::span{trace.mappings}.first(trace.mapping_count);
        const auto found = std::ranges::find(mappings, host, &GemmTrace::Mapping::host);
        if (found == mappings.end())
            trace.mismatch = true;
        else {
            *found = {};
            ++trace.releases;
        }
    }
    return status;
}
CUresult CUDAAPI upload_async(CUdeviceptr destination, const void* source, std::size_t bytes,
                              CUstream stream) {
    const auto status = cuMemcpyHtoDAsync(destination, source, bytes, stream);
    if (status == CUDA_SUCCESS && priority_trace && bytes == sizeof(std::uint64_t)) {
        auto& trace = *priority_trace;
        if (trace.upload_count < trace.uploads.size()) {
            std::uint64_t value{};
            std::memcpy(&value, source, sizeof(value));
            trace.uploads[trace.upload_count++] = {destination, value, ++trace.serial};
        } else
            trace.has_mismatch = true;
    }
    return status;
}
CUresult CUDAAPI release(CUdeviceptr address) {
    const auto status = cuMemFree(address);
    if (status == CUDA_SUCCESS && priority_trace) {
        auto& trace = *priority_trace;
        if (trace.free_count < trace.frees.size())
            trace.frees[trace.free_count++] = {address, ++trace.serial};
        else
            trace.has_mismatch = true;
    }
    return status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original CUDA ABI.
CUresult CUDAAPI launch(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                        CUstream stream, void** parameters, void** extra) {
    const auto status =
        cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
    if (status == CUDA_SUCCESS) workers.push_back(gx);
    if (status == CUDA_SUCCESS && gemm_trace && function == gemm_trace->function) {
        auto& trace = *gemm_trace;
        CUcontext context{};
        const bool packed = !parameters && extra && extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER &&
                            extra[1] && extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[3] &&
                            extra[4] == CU_LAUNCH_PARAM_END &&
                            *static_cast<const std::size_t*>(extra[3]) == trace.arguments.size();
        if (!packed || cuCtxGetCurrent(&context) != CUDA_SUCCESS || gx != 320 || gy != 1 ||
            gz != 1 || bx != 128 || by != 1 || bz != 1 || shared)
            trace.mismatch = true;
        else if (!trace.launches) {
            trace.stream = stream;
            trace.context = context;
            std::memcpy(trace.arguments.data(), extra[1], trace.arguments.size());
            std::memcpy(&trace.signal, trace.arguments.data() + 88, sizeof(trace.signal));
        } else
            trace.mismatch |=
                !trace.high_gemms || trace.stream != stream || trace.context != context ||
                std::memcmp(trace.arguments.data(), extra[1], trace.arguments.size()) != 0;
        if (trace.measure_timing && trace.launches == 1)
            trace.second_launch_return_ns = monotonic_ns();
        ++trace.launches;
    }
    if (status == CUDA_SUCCESS && priority_trace) {
        auto& trace = *priority_trace;
        const auto known = std::span{trace.functions}.first(trace.function_count);
        const auto found = std::ranges::find(known, function, &PriorityTrace::Function::native);
        CUcontext context{};
        const auto current = cuCtxGetCurrent(&context);
        if (found == known.end() || current != CUDA_SUCCESS || !context ||
            trace.launch_count == trace.launches.size()) {
            trace.has_mismatch = true;
            return status;
        }
        LowArguments arguments;
        if (found->kind == KernelKind::kLow || found->kind == KernelKind::kOriginalLow) {
            const auto bytes = found->kind == KernelKind::kOriginalLow
                                   ? std::size_t{32}
                                   : sizeof(arguments) - (trace.observes_completion ? 0 : 8);
            const bool packed = !parameters && extra &&
                                extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER && extra[1] &&
                                extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[3] &&
                                extra[4] == CU_LAUNCH_PARAM_END &&
                                *static_cast<const std::size_t*>(extra[3]) == bytes;
            if (packed)
                std::memcpy(&arguments, extra[1], bytes);
            else
                trace.has_mismatch = true;
        }
        trace.has_mismatch |= gy != 1 || gz != 1 || bx != 32 || by != 1 || bz != 1 || shared;
        trace.launches[trace.launch_count++] = {*found, context,   stream,
                                                gx,     arguments, ++trace.serial};
    }
    return status;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
template <class Function>
Function original(const char* name) {
    void* address{};
    CUdriverProcAddressQueryResult queried{};
    checked(cuGetProcAddress(name, &address, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &queried),
            "resolve exact legacy ABI");
    expect(address && queried == CU_GET_PROC_ADDRESS_SUCCESS, "native legacy SDK symbol exists");
    return reinterpret_cast<Function>(address);
}
CudaExecutionDomainDispatch domain_dispatch() {
    CudaExecutionDomainDispatch result{cuInit,          cuDeviceGet,     create_context,
                                       cuCtxSetCurrent, cuCtxGetCurrent, destroy_context};
    auto& configuration = result.configuration;
    configuration = {cuCtxGetFlags,
                     cuCtxSetFlags,
                     cuCtxGetLimit,
                     cuCtxSetLimit,
                     cuCtxGetCacheConfig,
                     cuCtxSetCacheConfig,
                     original<decltype(configuration.get_shared)>("cuCtxGetSharedMemConfig"),
                     original<decltype(configuration.set_shared)>("cuCtxSetSharedMemConfig")};
    return result;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {cuCtxGetCurrent, launch, cuEventCreate, cuEventRecord, cuEventQuery, cuEventDestroy,
         cuStreamSynchronize, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemsetD8, cuMemsetD8Async,
         cuMemsetD16, cuMemsetD16Async, cuMemsetD32, cuMemsetD32Async},
        cuInit,
        cuDeviceGet,
        create_context,
        cuCtxSetCurrent,
        destroy_context,
        cuModuleLoadData,
        cuModuleUnload,
        get_function,
        cuFuncGetParamInfo,
        cuMemAlloc,
        release,
        cuMemcpyHtoD,
        cuMemcpyDtoH,
        cuStreamCreate,
        cuStreamDestroy,
        host_allocate,
        host_release,
        host_map,
        upload_async,
        cuMemcpyDtoHAsync,
        cuOccupancyMaxActiveBlocksPerMultiprocessor};
    result.device_attribute = cuDeviceGetAttribute;
    result.device_uuid = cuDeviceGetUuid;
    result.driver_version = cuDriverGetVersion;
    result.context_get_flags = cuCtxGetFlags;
    result.context_set_flags = cuCtxSetFlags;
    result.function_get_attribute = cuFuncGetAttribute;
    result.function_set_attribute = cuFuncSetAttribute;
    result.function_set_cache = cuFuncSetCacheConfig;
    result.context_synchronize = cuCtxSynchronize;
    result.host_register = cuMemHostRegister;
    result.host_unregister = cuMemHostUnregister;
    result.stream_wait_event = cuStreamWaitEvent;
    result.stream_create_priority = cuStreamCreateWithPriority;
    result.stream_get_priority = cuStreamGetPriority;
    result.module_global = cuModuleGetGlobal;
    result.pointer_get_attribute = cuPointerGetAttribute;
    result.runtime.version = cudaRuntimeGetVersion;
    result.runtime.attribute = cudaDeviceGetAttribute;
    result.runtime.pointer_attributes = cudaPointerGetAttributes;
    result.context.get_limit = cuCtxGetLimit;
    result.context.set_limit = cuCtxSetLimit;
    result.context.get_cache = cuCtxGetCacheConfig;
    result.context.set_cache = cuCtxSetCacheConfig;
    result.context.get_shared =
        original<decltype(result.context.get_shared)>("cuCtxGetSharedMemConfig");
    result.context.set_shared =
        original<decltype(result.context.set_shared)>("cuCtxSetSharedMemConfig");
    result.context.runtime.get_cache = cudaDeviceGetCacheConfig;
    result.context.runtime.set_cache = cudaDeviceSetCacheConfig;
    result.blas = {cublasCreate_v2,         cublasDestroy_v2,   cublasSetStream_v2,
                   cublasSgemm_v2,          cublasGetStream_v2, cublasSetPointerMode_v2,
                   cublasGetPointerMode_v2, cublasSetMathMode,  cublasGetMathMode,
                   cublasGetVersion_v2};
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected cuBLAS ABI.
    result.blas.sgemm = [](cublasHandle_t handle, cublasOperation_t trans_a,
                           cublasOperation_t trans_b, int m, int n, int k, const float* alpha,
                           const float* a, int lda, const float* b, int ldb, const float* beta,
                           float* c, int ldc) {
        const auto status =
            cublasSgemm_v2(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        if (status == CUBLAS_STATUS_SUCCESS && gemm_trace) {
            CUcontext context{};
            if (gemm_trace->launches != 1 || cuCtxGetCurrent(&context) != CUDA_SUCCESS ||
                context != common_context.load())
                gemm_trace->mismatch = true;
            ++gemm_trace->high_gemms;
        }
        return status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return result;
}

struct ChildConfig {
    std::string executable, preload, socket, directory;
    TaskPeerIdentity server;
};
struct Child {
    pid_t pid{-1};
    ExecutionOwnedDescriptor input, output, log;
    bool exited{}, is_waiting_start{};
    std::size_t observed_bytes{};
    std::string buffered;
    std::vector<std::string> lines;
    Child(const ChildConfig& config, const std::string& role, bool is_cached = false,
          bool should_gate_start = false, std::string_view application_role = {})
        : log(::open((config.directory + "/" + role + ".log").c_str(),
                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600)),
          is_waiting_start(should_gate_start) {
        expect(log.get() >= 0, "open independent ordinary application log");
        std::array<int, 2> incoming{}, outgoing{};
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, incoming.data()) == 0,
               "bounded application phase socket");
        std::array<ExecutionOwnedDescriptor, 2> incoming_owner{
            ExecutionOwnedDescriptor{incoming[0]}, ExecutionOwnedDescriptor{incoming[1]}};
        expect(::pipe2(outgoing.data(), O_CLOEXEC) == 0, "bounded application output pipe");
        std::array<ExecutionOwnedDescriptor, 2> outgoing_owner{
            ExecutionOwnedDescriptor{outgoing[0]}, ExecutionOwnedDescriptor{outgoing[1]}};
        const auto flags = ::fcntl(outgoing[0], F_GETFL);
        expect(flags >= 0 && ::fcntl(outgoing[0], F_SETFL, flags | O_NONBLOCK) == 0,
               "nonblocking bounded ordinary output");
        const auto server_pid = std::to_string(config.server.pid);
        const auto server_start = std::to_string(config.server.start_time_ticks);
        const auto argument =
            "--" + std::string(application_role.empty() ? role : application_role);
        pid = ::fork();
        expect(pid >= 0, "fork actual ordinary SDK application before CUDA initialization");
        if (!pid) {
            // Close opposite pipe/socket ends before a blocking cold gate.
            // Otherwise the child retains its own writer and parent cancellation
            // cannot deliver EOF during exception cleanup.
            (void)::close(incoming[1]);
            (void)::close(outgoing[0]);
            if (should_gate_start) {
                char start{};
                ssize_t received{};
                do {
                    received = ::read(incoming[0], &start, 1);
                } while (received < 0 && errno == EINTR);
                if (received != 1 || start != 's') _exit(126);
            }
            if (::dup2(incoming[0], STDIN_FILENO) < 0 || ::dup2(outgoing[1], STDOUT_FILENO) < 0 ||
                ::dup2(outgoing[1], STDERR_FILENO) < 0)
                _exit(126);
            for (const auto descriptor : {incoming[0], outgoing[1]}) (void)::close(descriptor);
            if (::setenv("LD_PRELOAD", config.preload.c_str(), 1) ||
                ::setenv("GLIMMER_EXECUTION_SOCKET", config.socket.c_str(), 1) ||
                ::setenv("GLIMMER_EXECUTION_SERVER_PID", server_pid.c_str(), 1) ||
                ::setenv("GLIMMER_EXECUTION_SERVER_START", server_start.c_str(), 1))
                _exit(126);
            if (is_cached)
                ::execl(config.executable.c_str(), config.executable.c_str(), argument.c_str(),
                        "--cached-lookups", static_cast<char*>(nullptr));
            else
                ::execl(config.executable.c_str(), config.executable.c_str(), argument.c_str(),
                        static_cast<char*>(nullptr));
            _exit(127);
        }
        input = std::move(incoming_owner[1]);
        output = std::move(outgoing_owner[0]);
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() {
        input = ExecutionOwnedDescriptor{};
        output = ExecutionOwnedDescriptor{};
        if (pid > 0 && !exited) {
            int status{};
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
    TaskPeerIdentity identity() const {
        std::uint64_t start{};
        expect(read_process_start_time(static_cast<std::uint32_t>(pid), &start),
               "exact actual child generation");
        return {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start};
    }
    void start() {
        expect(is_waiting_start, "only a cold gated child can start once");
        constexpr char start = 's';
        expect(::send(input.get(), &start, 1, MSG_NOSIGNAL) == 1,
               "exact pre-exec child start after service readiness");
        is_waiting_start = false;
    }
    void run() const {
        expect(!is_waiting_start, "application input cannot release a cold pre-exec start gate");
        constexpr std::string_view command = "run\n";
        expect(::send(input.get(), command.data(), command.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(command.size()),
               "exact original application input phase");
    }
    void drain_output() {
        std::array<char, 4096> data{};
        while (true) {
            const auto count = ::read(output.get(), data.data(), data.size());
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            expect(count >= 0, "read exact owned application output");
            if (!count) break;
            const auto bytes = static_cast<std::size_t>(count);
            expect(bytes <= std::size_t{1024} * 1024 - observed_bytes,
                   "bounded complete application output");
            observed_bytes += bytes;
            std::size_t written{};
            while (written < bytes) {
                const auto result = ::write(log.get(), data.data() + written, bytes - written);
                if (result < 0 && errno == EINTR) continue;
                expect(result > 0, "write complete original application evidence");
                written += static_cast<std::size_t>(result);
            }
            buffered.append(data.data(), bytes);
            while (true) {
                const auto newline = buffered.find('\n');
                if (newline == std::string::npos) break;
                expect(lines.size() < 128, "bounded ordinary application phase records");
                lines.push_back(buffered.substr(0, newline));
                buffered.erase(0, newline + 1);
            }
        }
    }
    void observe() {
        drain_output();
        if (!exited) {
            int status{};
            const auto observed = ::waitpid(pid, &status, WNOHANG);
            expect(observed == 0 || observed == pid, "observe exact child waitpid");
            if (observed == pid) {
                exited = true;
                expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                       "ordinary application full numerical/API checks");
                // Exit can race the preceding EAGAIN. Once waitpid proves
                // exit, collect the final buffered receipt before publication.
                drain_output();
            }
        }
    }
    bool has(std::string_view line) const {
        return std::ranges::find(lines, line) != lines.end();
    }
};

void phase_protocol(const char* executable) {
    char pattern[] = "/tmp/glimmer-public-phase-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    expect(directory, "private CPU-only phase protocol directory");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "CPU-only original owner identity");
    {
        Child child({executable, "", "/unused", directory, *identity}, "phase-child", false, true);
        bool input_rejected{};
        try {
            child.run();
        } catch (const std::runtime_error&) {
            input_rejected = true;
        }
        expect(input_rejected, "ordinary input cannot substitute a cold child start");
        child.start();
        bool duplicate_rejected{};
        try {
            child.start();
        } catch (const std::runtime_error&) {
            duplicate_rejected = true;
        }
        expect(duplicate_rejected, "cold start gate cannot consume application phase bytes twice");
        for (unsigned int phase = 0; phase < 4; ++phase) child.run();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!child.exited) {
            child.observe();
            expect(std::chrono::steady_clock::now() < deadline, "bounded CPU-only child exit");
            std::this_thread::yield();
        }
        expect(
            child.lines == std::vector<std::string>{"PHASE 0", "PHASE 1", "PHASE 2", "PHASE 3"} &&
                child.buffered.empty(),
            "terminal output is fully drained after actual waitpid");
        bool was_rejected{};
        try {
            child.run();
        } catch (const std::runtime_error&) {
            was_rejected = true;
        }
        expect(was_rejected, "exited phase receiver rejects without SIGPIPE or new work");
    }
    {
        Child cancelled({executable, "", "/unused", directory, *identity}, "phase-cancel", false,
                        true);
        cancelled.input = ExecutionOwnedDescriptor{};
        bool was_rejected{};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        try {
            while (!cancelled.exited) {
                cancelled.observe();
                expect(std::chrono::steady_clock::now() < deadline, "bounded cold cancellation");
                std::this_thread::yield();
            }
        } catch (const std::runtime_error&) {
            was_rejected = true;
        }
        expect(was_rejected && cancelled.exited && cancelled.lines.empty(),
               "parent EOF cancels a cold child before application/SDK startup");
    }
    expect(::unlink((std::string(directory) + "/phase-child.log").c_str()) == 0 &&
               ::unlink((std::string(directory) + "/phase-cancel.log").c_str()) == 0 &&
               ::rmdir(directory) == 0,
           "retire only known idle reproducible CPU fixture inputs");
}

CudaProfileConfig public_profile_config() {
    CudaProfileConfig profile;
    profile.workers = {320};
    profile.warmup_calls = 0;
    profile.samples_per_candidate = 1;
    return profile;
}
ExecutionServiceConfig public_service_config(const ChildConfig& child_config) {
    ExecutionServiceConfig config;
    config.transport.socket_path = child_config.socket;
    config.transport.max_peers = 16;
    config.transport.max_pending_requests = 16;
    config.transport.io_timeout_ms = 10000;
    config.session_limits.max_device_bytes = std::size_t{64} * 1024 * 1024;
    config.persistent.workers = 320;
    config.profile = public_profile_config();
    config.exit_when_idle = true;
    config.should_audit_native = true;
    return config;
}
void train_public_singles(const ChildConfig& child_config, const std::string& output) {
    expect(valid_socket_address(child_config.socket), "bounded public profile address");
    Child first(child_config, "sample-a", false, true, "low");
    Child second(child_config, "sample-b", false, true, "low");
    auto config = public_service_config(child_config);
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
    config.profile_output = output;
    // Both clients are forked before SDK startup. Keep one learning cache across
    // their independent original/private inputs; incomplete cases are not saved.
    config.exit_when_idle = false;
    ExecutionService service(config, dispatch());
    expect(service.start(), "actual ordinary single-profile service");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{90};
    const auto advance = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded public single sampling");
        expect(service.step(1), "actual single-profile owner progress");
        first.observe();
        second.observe();
    };
    for (auto* child : {&first, &second}) {
        child->start();
        while (!child->has("Ordinary Runtime application ready: role=low")) advance();
        child->run();
        while (!child->exited) advance();
    }
    expect(service.request_stop() && service.save_profile(), "publish complete actual singles");
    const auto measured = service.profile_snapshot();
    expect(measured.cases == 2 && measured.trained_cases == 2 && measured.samples == 4 &&
               !measured.rejected_samples && !measured.cancelled && creates.load() &&
               creates.load() == destroys.load(),
           "both original/private warmup and phase cases train with normal SDK cleanup");
    std::cout
        << "{\"status\":\"passed\",\"stage\":\"single-profile\",\"ordinary_public_children\":2,"
           "\"cases\":2,\"trained_cases\":2,\"samples\":4,\"native_contexts\":"
        << creates.load() << ",\"native_destroys\":" << destroys.load()
        << ",\"cleanup\":true,\"full_device_qualified\":false,\"performance_qualified\":false,"
           "\"full_goal_qualified\":false}\n";
}
void write_arrival_trace(const PriorityTrace& trace) {
    const auto launches = std::span{trace.launches}.first(trace.launch_count);
    const auto is_counter = [&](CUdeviceptr address) {
        return std::ranges::any_of(launches, [&](const auto& entry) {
            return entry.function.kind == KernelKind::kLow && entry.arguments.counter == address;
        });
    };
    for (std::uint64_t serial = 1; serial <= trace.serial; ++serial) {
        for (const auto& entry : std::span{trace.uploads}.first(trace.upload_count))
            if (entry.serial == serial && is_counter(entry.address))
                std::cout << "PUBLIC_ARRIVAL_NATIVE {\"event\":\"counter_initialize\",\"serial\":"
                          << serial << ",\"counter\":" << entry.address
                          << ",\"value\":" << entry.value << "}\n";
        for (const auto& entry : launches) {
            if (entry.serial != serial) continue;
            const auto& input = entry.arguments;
            std::cout << "PUBLIC_ARRIVAL_NATIVE {\"event\":\"launch\",\"serial\":" << serial
                      << ",\"kind\":\""
                      << (entry.function.kind == KernelKind::kLow ? "persistent" : "original")
                      << "\",\"context\":" << reinterpret_cast<std::uintptr_t>(entry.context)
                      << ",\"module\":" << reinterpret_cast<std::uintptr_t>(entry.function.module)
                      << ",\"function\":" << reinterpret_cast<std::uintptr_t>(entry.function.native)
                      << ",\"stream\":" << reinterpret_cast<std::uintptr_t>(entry.stream)
                      << ",\"workers\":" << entry.workers << ",\"arguments\":[" << input.output
                      << ',' << input.count << ',' << input.padding << ',' << input.add << ','
                      << input.marker << ',' << input.unused[0] << ',' << input.unused[1] << ','
                      << input.x << ',' << input.y << ',' << input.z << ',' << input.grid_padding
                      << ',' << input.counter << ',' << input.retreat << "]}\n";
        }
        for (const auto& entry : std::span{trace.frees}.first(trace.free_count))
            if (entry.serial == serial && is_counter(entry.address))
                std::cout << "PUBLIC_ARRIVAL_NATIVE {\"event\":\"counter_free\",\"serial\":"
                          << serial << ",\"counter\":" << entry.address << "}\n";
    }
}
struct PublicArrivalOptions {
    std::string singles, pairs;
    bool is_cached{}, is_learning{};
};
void run_public_arrivals(const ChildConfig& child_config, const PublicArrivalOptions& options) {
    const auto& [singles, pairs, is_cached, is_learning] = options;
    expect(valid_socket_address(child_config.socket), "bounded public arrival address");
    const std::size_t count = is_learning ? 2 : 3;
    std::vector<std::unique_ptr<Child>> children;
    for (std::size_t index = 0; index < count; ++index)
        children.push_back(std::make_unique<Child>(child_config, "client-" + std::to_string(index),
                                                   is_cached, true, "low"));
    // No explicit CUDA entry or framework worker precedes any of these forks.
    PriorityTrace trace;
    priority_trace = &trace;
    struct ResetTrace {
        ~ResetTrace() {
            priority_trace = nullptr;
        }
    } reset_trace;
    workers.reserve(32);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_dispatch(),
        CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 8, std::size_t{64} * 1024 * 1024});
    checked(domain->initialize(), "actual public arrival common context");
    auto config = public_service_config(child_config);
    config.profile_input = singles;
    for (const auto& child : children)
        config.client_priorities.push_back(
            {child->identity(), 0, ExecutionClientKernelMode::kConfigured});
    config.paired = ExecutionServicePairConfig{
        domain,
        {.max_candidates = 1, .warmup_calls = 0, .samples_per_candidate = 1},
        is_learning ? std::string{} : pairs,
        {160},
        is_learning,
        is_learning ? pairs : std::string{},
        !is_learning};
    config.should_audit_priority = true;
    ExecutionService service(config, dispatch());
    expect(service.start(), "actual public arrival service");
    for (auto& child : children) child->start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{90};
    const auto advance = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual Runtime arrivals");
        if (!service.finished()) expect(service.step(1), "actual arrival owner progress");
        for (auto& child : children) child->observe();
    };
    while (!std::ranges::all_of(children, [](const auto& child) {
        return child->has("Ordinary Runtime application ready: role=low");
    }))
        advance();
    expect(domain->snapshot().attached_members == count && !common_destroys.load() &&
               !trace.has_mismatch,
           "all ordinary warmups retain independent actual common-domain members");
    trace.begin_phase();
    std::cout << "PUBLIC_ARRIVAL_PHASE\n" << std::flush;
    children[0]->run();
    if (!is_learning) {
        while (!trace.launch_count) advance();
        expect(trace.launch_count == 1 && trace.launches[0].function.kind == KernelKind::kLow &&
                   trace.launches[0].workers == 320,
               "new equal-priority input follows actual solo persistent native issue");
        const auto arrival = std::chrono::steady_clock::now() + std::chrono::milliseconds{5};
        while (std::chrono::steady_clock::now() < arrival) advance();
    }
    std::cout << "PUBLIC_ARRIVAL_NEW_INPUT\n" << std::flush;
    for (std::size_t index = 1; index < count; ++index) children[index]->run();
    while (!service.finished() ||
           !std::ranges::all_of(children, [](const auto& child) { return child->exited; }))
        advance();
    expect(service.request_stop(), "actual arrival audit/transport shutdown");
    const auto measured = service.pair_profile_snapshot();
    expect(service.pair_profile_status() == CUDA_SUCCESS && measured.pairs == 2 &&
               measured.trained_pairs == 2 && !measured.active && !measured.rejected_samples &&
               !measured.cancelled && (is_learning ? service.save_pair_profile() : true),
           "complete actual warmup/phase pair measurements restore without replay");
    const auto launches = std::span{trace.launches}.first(trace.launch_count);
    expect(!trace.has_mismatch && !launches.empty() &&
               std::ranges::all_of(
                   launches,
                   [&](const auto& entry) {
                       const auto& input = entry.arguments;
                       return entry.context == common_context.load() && input.output &&
                              input.count == 65536 * 32 && input.add == 1 && input.marker == 8192 &&
                              !input.unused[0] && !input.unused[1] &&
                              ((entry.function.kind == KernelKind::kLow && input.counter &&
                                input.retreat && (entry.workers == 160 || entry.workers == 320)) ||
                               (entry.function.kind == KernelKind::kOriginalLow &&
                                entry.workers == 65536));
                   }),
           "all actual ordinary phase arguments, native compute owner and explicit geometries");
    if (is_learning)
        expect(launches.size() == 2 && launches[0].workers == 160 && launches[1].workers == 160,
               "the exact isolated pair candidate executes, not a fabricated measurement");
    else {
        const auto& initial = launches.front();
        std::size_t continuations{};
        for (const auto& entry : launches) {
            if (entry.arguments.output != initial.arguments.output) continue;
            expect(entry.function.native == initial.function.native &&
                       entry.function.module == initial.function.module &&
                       entry.stream == initial.stream && entry.arguments == initial.arguments,
                   "same actual A command retains its original private ABI and counter on every "
                   "continuation");
            ++continuations;
        }
        expect(continuations >= 2, "actual A stop must have a retained native continuation");
    }
    expect(domain->snapshot().is_shutdown && !domain->snapshot().is_quarantined &&
               common_destroys.load() == 1 && creates.load() == count + 1 &&
               creates.load() == destroys.load(),
           "common compute and every isolated host helper retire normally");
    write_arrival_trace(trace);
    const auto shared = std::ranges::count(launches, 160U, &PriorityTrace::Launch::workers);
    std::cout << "{\"status\":\"passed\",\"stage\":\"" << (is_learning ? "pair-profile" : "arrival")
              << "\",\"cached\":" << (is_cached ? "true" : "false")
              << ",\"ordinary_public_children\":" << count
              << ",\"common_contexts\":1,"
                 "\"common_destroys\":1,\"native_contexts\":"
              << creates.load() << ",\"native_destroys\":" << destroys.load()
              << ",\"phase_native_launches\":" << trace.launch_count
              << ",\"shared_worker_launches\":" << shared
              << ",\"trained_pairs\":2,\"actual_measurements\":true,\"cleanup\":true,"
                 "\"hardware_co_residency_qualified\":false,\"full_device_qualified\":false,"
                 "\"priority_qualified\":false,\"performance_qualified\":false,\"full_goal_"
                 "qualified\":false}\n";
}

void run_priority(const ChildConfig& child_config, bool is_cached, bool observes_completion) {
    expect(valid_socket_address(child_config.socket),
           "private priority socket path exceeds the Unix address bound");
    Child low(child_config, "low", is_cached, true);
    Child high(child_config, "high", is_cached, true);
    // No explicit SDK or service/host-worker entry precedes these two forks.
    PriorityTrace trace;
    trace.observes_completion = observes_completion;
    priority_trace = &trace;
    struct ResetTrace {
        ~ResetTrace() {
            priority_trace = nullptr;
        }
    } reset_trace;
    workers.reserve(32);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_dispatch(),
        CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 8, std::size_t{64} * 1024 * 1024});
    checked(domain->initialize(), "create actual common priority compute context");
    CudaProfileConfig profile;
    profile.workers = {320};
    profile.warmup_calls = 0;
    profile.samples_per_candidate = 1;
    CudaProfileCache empty(profile, nullptr, false);
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "actual public priority profile program identity");
    const auto archive = empty.encode(*identity);
    const auto single = child_config.directory + "/single";
    expect(archive && write_profile_archive(single, *archive),
           "protected readonly cold priority input, no replay");
    ExecutionServiceConfig config;
    config.transport.socket_path = child_config.socket;
    config.transport.max_peers = 16;
    config.transport.max_pending_requests = 16;
    config.transport.io_timeout_ms = 10000;
    config.session_limits.max_device_bytes = std::size_t{64} * 1024 * 1024;
    config.persistent.workers = 320;
    config.persistent.observes_block_completion = observes_completion;
    config.client_priorities = {{low.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {high.identity(), 100, ExecutionClientKernelMode::kOriginal}};
    config.profile = profile;
    config.profile_input = single;
    config.paired = ExecutionServicePairConfig{domain, {}, {}, {80, 160}};
    config.should_audit_native = config.should_audit_priority = true;
    config.exit_when_idle = true;
    ExecutionService service(config, dispatch());
    expect(service.start(), "actual public priority service composition");
    low.start();
    high.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{90};
    const auto advance = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual public priority run");
        if (!service.finished()) expect(service.step(1), "actual priority owner progress");
        low.observe();
        high.observe();
    };
    while (!low.has("Ordinary Runtime application ready: role=low") ||
           !high.has("Ordinary Runtime application ready: role=high"))
        advance();
    expect(
        domain->snapshot().attached_members == 2 && !common_destroys.load() && !trace.has_mismatch,
        "two ordinary warmups retain the actual common domain");
    trace.begin_phase();
    std::cout << "PUBLIC_PRIORITY_PHASE\n" << std::flush;
    low.run();
    while (!trace.launch_count) advance();
    expect(trace.launch_count == 1 && trace.launches[0].function.kind == KernelKind::kLow,
           "high arrival follows actual low native issue, not warmup");
    // Arrival pacing only, never a live-counter or useful-work/stop proof.
    const auto arrival = std::chrono::steady_clock::now() + std::chrono::milliseconds{5};
    while (std::chrono::steady_clock::now() < arrival) advance();
    if (observes_completion) {
        const auto low_peer = low.identity();
        std::optional<ExecutionBlockCompletionObservation> publication;
        while (!(publication = service.block_completion_observation(low_peer))) advance();
        const auto& progress = publication->progress;
        expect(trace.launch_count == 1 && publication->session && publication->sequence &&
                   progress.total_blocks == 65536 && progress.physical_launches == 1 &&
                   !progress.physical_stops && !progress.completed_blocks &&
                   progress.observes_block_completion && progress.has_completed_block,
               "actual mapped GPU block-exit publication precedes high input, independent of stop");
        auto stale_peer = low_peer;
        ++stale_peer.start_time_ticks;
        expect(!service.block_completion_observation(high.identity()) &&
                   !service.block_completion_observation(stale_peer),
               "high peer and stale low generation cannot borrow another command's publication");
        std::cout << "PUBLIC_PRIORITY_BLOCK_COMPLETION {\"session\":" << publication->session
                  << ",\"sequence\":" << publication->sequence
                  << ",\"total_blocks\":65536,\"physical_launches\":1,\"physical_stops\":0,"
                     "\"stopped_completed_blocks\":0,\"has_completed_block\":true}\n"
                  << std::flush;
    }
    std::cout << "PUBLIC_PRIORITY_HIGH_ARRIVAL\n" << std::flush;
    high.run();
    while (!low.exited || !high.exited || !service.finished()) advance();
    expect(service.request_stop(),
           "complete actual service audit/transport closure before receipt");

    expect(!trace.has_mismatch && trace.launch_count >= 4,
           "low initial, two original high calls and retained low continuations");
    const auto launches = std::span{trace.launches}.first(trace.launch_count);
    const auto& initial = launches.front();
    const auto& resumed = launches.back();
    const auto continuations = launches.subspan(3);
    expect(initial.function.kind == KernelKind::kLow &&
               launches[1].function.kind == KernelKind::kHigh &&
               launches[2].function.kind == KernelKind::kHigh &&
               std::ranges::all_of(continuations,
                                   [&](const auto& entry) {
                                       return entry.function.kind == KernelKind::kLow &&
                                              entry.function.native == initial.function.native &&
                                              entry.function.module == initial.function.module &&
                                              entry.stream == initial.stream &&
                                              entry.arguments == initial.arguments;
                                   }) &&
               launches[1].function.native == launches[2].function.native &&
               launches[1].stream == launches[2].stream &&
               initial.function.module != launches[1].function.module &&
               std::ranges::all_of(launches,
                                   [](const auto& entry) {
                                       return entry.context == common_context.load() &&
                                              entry.workers == 320;
                                   }),
           "actual common context, independent modules and exact retained low launch bytes");
    const auto& arguments = initial.arguments;
    expect(arguments.count == 65536 * 32 && arguments.add == 1 && arguments.marker == 8192 &&
               !arguments.unused[0] && !arguments.unused[1] && arguments.x == 65536 &&
               arguments.y == 1 && arguments.z == 1 && arguments.output && arguments.counter &&
               arguments.retreat && arguments.counter != arguments.output,
           "original non-idempotent arguments, logical geometry and actual private progress");
    expect(
        observes_completion ? arguments.completion == arguments.retreat + 4 : !arguments.completion,
        "default ABI stays unchanged; observed ABI explicitly binds its own aligned word");
    const auto uploads = std::span{trace.uploads}.first(trace.upload_count);
    const auto zeros = std::ranges::count_if(
        uploads, [&](const auto& entry) { return entry.address == arguments.counter; });
    expect(zeros == 1 && std::ranges::any_of(uploads,
                                             [&](const auto& entry) {
                                                 return entry.address == arguments.counter &&
                                                        !entry.value &&
                                                        entry.serial < initial.serial;
                                             }),
           "actual retained counter is initialized once before initial issue, never on resume");
    const auto frees = std::span{trace.frees}.first(trace.free_count);
    expect(std::ranges::count_if(
               frees, [&](const auto& entry) { return entry.address == arguments.counter; }) == 1 &&
               std::ranges::any_of(frees,
                                   [&](const auto& entry) {
                                       return entry.address == arguments.counter &&
                                              entry.serial > resumed.serial;
                                   }),
           "actual retained progress allocation survives until final continuation");
    expect(domain->snapshot().is_shutdown && !domain->snapshot().is_quarantined &&
               common_destroys.load() == 1 && creates.load() == 3 && destroys.load() == 3,
           "one common compute and two isolated host owners retire exactly once");
    for (const auto& entry : uploads)
        if (entry.address == arguments.counter)
            std::cout << "PUBLIC_PRIORITY_NATIVE {\"event\":\"counter_initialize\",\"serial\":"
                      << entry.serial << ",\"counter\":" << entry.address
                      << ",\"value\":" << entry.value << "}\n";
    for (const auto& entry : launches) {
        std::cout << "PUBLIC_PRIORITY_NATIVE {\"event\":\"launch\",\"serial\":" << entry.serial
                  << ",\"kind\":\"" << (entry.function.kind == KernelKind::kLow ? "low" : "high")
                  << "\",\"context\":" << reinterpret_cast<std::uintptr_t>(entry.context)
                  << ",\"module\":" << reinterpret_cast<std::uintptr_t>(entry.function.module)
                  << ",\"function\":" << reinterpret_cast<std::uintptr_t>(entry.function.native)
                  << ",\"stream\":" << reinterpret_cast<std::uintptr_t>(entry.stream)
                  << ",\"workers\":" << entry.workers << ",\"arguments\":[";
        if (entry.function.kind == KernelKind::kLow) {
            const auto& input = entry.arguments;
            std::cout << input.output << ',' << input.count << ',' << input.padding << ','
                      << input.add << ',' << input.marker << ',' << input.unused[0] << ','
                      << input.unused[1] << ',' << input.x << ',' << input.y << ',' << input.z
                      << ',' << input.grid_padding << ',' << input.counter << ',' << input.retreat;
            if (observes_completion) std::cout << ',' << input.completion;
        }
        std::cout << "]}\n";
    }
    for (const auto& entry : frees)
        if (entry.address == arguments.counter)
            std::cout << "PUBLIC_PRIORITY_NATIVE {\"event\":\"counter_free\",\"serial\":"
                      << entry.serial << ",\"counter\":" << entry.address << "}\n";
    std::cout << "{\"status\":\"passed\",\"cached\":" << (is_cached ? "true" : "false")
              << ",\"ordinary_public_children\":2,\"common_contexts\":1,\"common_destroys\":1,"
                 "\"native_contexts\":3,\"native_destroys\":3,\"phase_native_launches\":"
              << trace.launch_count << ",\"low_physical_launches\":" << trace.launch_count - 2
              << ",\"actual_workers\":320,\"same_counter\":true,\"same_arguments\":true,"
                 "\"counter_initializations\":1,\"counter_frees\":1,\"cleanup\":true,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,"
                 "\"performance_qualified\":false,\"full_goal_qualified\":false";
    if (observes_completion)
        std::cout << ",\"observes_block_completion\":true,\"same_completion_word\":true,"
                     "\"independent_pre_demand_useful_compute\":true";
    std::cout << "}\n";
}

void run_gemm_priority(const ChildConfig& child_config, bool sharded = true,
                       bool measure_timing = false) {
    expect(valid_socket_address(child_config.socket), "bounded GEMM service address");
    Child low(child_config, "gemm-low", false, true, "low");
    Child high(child_config, "gemm-high", false, true, "high");
    GemmTrace trace;
    trace.measure_timing = measure_timing;
    gemm_trace = &trace;
    struct ResetTrace {
        ~ResetTrace() {
            gemm_trace = nullptr;
        }
    } reset_trace;
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_dispatch(),
        CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 8, std::size_t{512} * 1024 * 1024});
    checked(domain->initialize(), "create actual common GEMM compute context");
    auto config = public_service_config(child_config);
    config.session_limits.max_device_bytes = std::size_t{256} * 1024 * 1024;
    config.persistent.replace_sgemm = true;
    config.persistent.gemm_tile_side = 32;
    config.persistent.uses_sharded_host_retreat = sharded;
    config.persistent.observes_block_completion = true;
    config.should_audit_priority = true;
    config.client_priorities = {{low.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {high.identity(), 100, ExecutionClientKernelMode::kOriginal}};
    config.paired = ExecutionServicePairConfig{domain, {}, {}, {80, 160}};
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "actual GEMM diagnostic profile identity");
    CudaProfileCache empty(config.profile, nullptr, false);
    const auto archive = empty.encode(*identity);
    config.profile_input = child_config.directory + "/gemm-single";
    expect(archive && write_profile_archive(config.profile_input, *archive),
           "empty readonly GEMM diagnostic profile");
    ExecutionService service(config, dispatch());
    expect(service.start(), "start authenticated shared GEMM service");
    low.start();
    high.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{90};
    const auto advance = [&] {
        expect(std::chrono::steady_clock::now() < deadline, "bounded shared GEMM run");
        if (!service.finished()) expect(service.step(1), "shared GEMM service progress");
        low.observe();
        high.observe();
    };
    while (!low.has("READY low") || !high.has("READY high")) advance();
    expect(domain->snapshot().attached_members == 2 && !common_destroys.load(),
           "both applications retain their own members in one physical context");
    low.run();
    std::optional<ExecutionBlockCompletionObservation> before;
    while (!(before = service.block_completion_observation(low.identity()))) advance();
    expect(trace.launches == 1 && !trace.mismatch && trace.context == common_context.load() &&
               before->progress.total_blocks == 16384 && before->progress.has_completed_block &&
               !before->progress.physical_stops,
           "actual partially completed sharded GEMM precedes high-priority arrival");
    const auto demand_begin_ns = measure_timing ? monotonic_ns() : 0;
    high.run();
    const auto demand_end_ns = measure_timing ? monotonic_ns() : 0;
    bool saw_request{};
    std::uint64_t scans{}, last_failed_scan_end_ns{}, first_success_scan_begin_ns{},
        first_success_scan_end_ns{}, high_verified_observed_ns{}, low_verified_observed_ns{};
    while (!low.exited || !high.exited || !service.finished()) {
        advance();
        if (measure_timing && !high_verified_observed_ns && high.has("VERIFIED high 65536"))
            high_verified_observed_ns = monotonic_ns();
        if (measure_timing && !low_verified_observed_ns && low.has("VERIFIED low 16777216"))
            low_verified_observed_ns = monotonic_ns();
        for (const auto& mapping : std::span{trace.mappings}.first(trace.mapping_count)) {
            const auto signal_bytes = sharded ? kGemmSignalSlotBytes : sizeof(std::uint32_t);
            if (!mapping.host || !mapping.device || mapping.bytes < signal_bytes ||
                trace.signal < mapping.device ||
                trace.signal - mapping.device > mapping.bytes - signal_bytes)
                continue;
            const auto offset = trace.signal - mapping.device;
            bool requested = true;
            const auto scan_begin = measure_timing ? monotonic_ns() : 0;
            for (std::size_t index = 0; index < (sharded ? kGemmSignalShards : 1); ++index) {
                auto& word = *reinterpret_cast<std::uint32_t*>(mapping.host + offset +
                                                               index * kGemmSignalShardBytes);
                requested &= std::atomic_ref(word).load(std::memory_order_acquire) == 1;
            }
            if (measure_timing) {
                const auto scan_end = monotonic_ns();
                ++scans;
                if (requested && !first_success_scan_end_ns) {
                    first_success_scan_begin_ns = scan_begin;
                    first_success_scan_end_ns = scan_end;
                } else if (!requested && !first_success_scan_end_ns)
                    last_failed_scan_end_ns = scan_end;
            }
            saw_request |= requested;
        }
    }
    expect(service.request_stop() && service.audit_healthy(), "close the complete service audit");
    const auto final_domain = domain->snapshot();
    expect(final_domain.is_shutdown && !final_domain.is_quarantined &&
               !final_domain.retained_members && !final_domain.attached_members &&
               !final_domain.reserved_device_bytes && common_destroys.load() == 1 &&
               creates.load() == destroys.load() && trace.allocations &&
               trace.allocations == trace.releases &&
               std::ranges::all_of(
                   trace.mappings,
                   [](const auto& mapping) { return !mapping.host && !mapping.device; }),
           "both application members and the physical domain retire exactly once");
    expect(saw_request && trace.launches >= 2 && trace.high_gemms == 1 && !trace.mismatch &&
               low.has("VERIFIED low 16777216") && high.has("VERIFIED high 65536"),
           "full outputs and retained launch bytes prove request, continuation and correctness");
    expect(::unlink(config.profile_input.c_str()) == 0, "remove owned temporary GEMM profile");
    if (measure_timing)
        expect(demand_begin_ns && demand_end_ns >= demand_begin_ns && scans &&
                   first_success_scan_begin_ns >= demand_end_ns &&
                   first_success_scan_end_ns >= first_success_scan_begin_ns &&
                   trace.second_launch_return_ns >= demand_end_ns &&
                   high_verified_observed_ns >= demand_end_ns &&
                   low_verified_observed_ns >= demand_end_ns,
               "complete bounded host-observed GEMM event timeline");
    std::cout << "{\"status\":\"passed\",\"shared_contexts\":1,\"applications\":2,"
                 "\"sharded_request\":"
              << (sharded ? "true" : "false") << ",\"gemm_launches\":" << trace.launches;
    if (measure_timing)
        std::cout << ",\"diagnostic_only\":true,\"demand_send_begin_ns\":" << demand_begin_ns
                  << ",\"demand_send_end_ns\":" << demand_end_ns << ",\"signal_scans\":" << scans
                  << ",\"last_failed_signal_scan_end_ns\":" << last_failed_scan_end_ns
                  << ",\"first_success_signal_scan_begin_ns\":" << first_success_scan_begin_ns
                  << ",\"first_success_signal_scan_end_ns\":" << first_success_scan_end_ns
                  << ",\"low_second_launch_return_ns\":" << trace.second_launch_return_ns
                  << ",\"high_verified_observed_ns\":" << high_verified_observed_ns
                  << ",\"low_verified_observed_ns\":" << low_verified_observed_ns;
    std::cout << ",\"full_device_qualified\":false,\"performance_qualified\":false}\n";
}

void run(const std::string& application, const std::string& preload, const std::string& directory) {
    const auto server = current_process_identity();
    expect(server.has_value(), "actual authenticated embedded service owner");
    const auto socket = directory + "/service.sock";
    const ChildConfig child_config{application, preload, socket, directory, *server};
    Child cycler(child_config, "cycler");
    Child sibling(child_config, "sibling");
    // All explicit SDK entry and service/host-worker startup follow both forks.
    workers.reserve(32);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_dispatch(),
        CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 8, std::size_t{64} * 1024 * 1024});
    checked(domain->initialize(), "create the actual common native compute context");
    CudaProfileConfig profile;
    profile.workers = {320};
    profile.warmup_calls = 0;
    profile.samples_per_candidate = 1;
    CudaProfileCache empty(profile, nullptr, false);
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "exact embedded program profile identity");
    const auto archive = empty.encode(*identity);
    const auto single = directory + "/single";
    expect(archive && write_profile_archive(single, *archive),
           "protected explicit readonly cold profile");
    ExecutionServiceConfig config;
    config.transport.socket_path = socket;
    config.transport.max_peers = 16;
    config.transport.max_pending_requests = 16;
    config.transport.io_timeout_ms = 10000;
    config.session_limits.max_device_bytes = std::size_t{64} * 1024 * 1024;
    config.persistent.workers = 320;
    config.client_priorities = {{cycler.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {sibling.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.profile = profile;
    config.profile_input = single;
    config.paired = ExecutionServicePairConfig{domain, {}, {}, {80, 160}};
    config.exit_when_idle = true;
    ExecutionService service(config, dispatch());
    expect(service.start(), "start trusted private ordinary public-path assembly");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{90};
    const auto advance = [&] {
        expect(std::chrono::steady_clock::now() < deadline,
               "bounded ordinary public application execution");
        if (!service.finished()) expect(service.step(1), "actual service owner progress");
        cycler.observe();
        sibling.observe();
    };
    sibling.run();
    while (!sibling.has("READY")) advance();
    expect(domain->snapshot().attached_members == 1 && !common_destroys.load(),
           "sibling owns the first shared logical member");
    unsigned int step{};
    for (unsigned int generation = 0; generation < 3; ++generation) {
        cycler.run();
        const auto marker = "GEN " + std::to_string(generation);
        while (!cycler.has(marker)) advance();
        expect(domain->snapshot().attached_members == 1 && !common_destroys.load(),
               "Runtime reset retires only its actual shared logical generation");
        sibling.run();
        const auto completed = "STEP " + std::to_string(step++);
        while (!sibling.has(completed)) advance();
    }
    cycler.run();
    while (!cycler.exited) advance();
    expect(domain->snapshot().attached_members == 1 && !common_destroys.load(),
           "final Driver release preserves the retained shared sibling");
    while (step < 6) {
        sibling.run();
        const auto completed = "STEP " + std::to_string(step++);
        while (!sibling.has(completed)) advance();
    }
    sibling.run();
    while (!sibling.exited || !service.finished()) advance();
    expect(domain->snapshot().is_shutdown && !domain->snapshot().is_quarantined &&
               common_destroys.load() == 1 && creates.load() == 4 && destroys.load() == 4 &&
               workers == std::vector<unsigned int>(12, 320),
           "exact original logical launches, common owner and native helpers retire normally");
    std::cout << "{\"status\":\"passed\",\"ordinary_public_children\":2,\"runtime_generations\":4,"
                 "\"runtime_resets\":4,\"driver_last_release\":true,\"common_contexts\":1,\"common_"
                 "destroys\":1,"
                 "\"native_contexts\":"
              << creates.load() << ",\"native_destroys\":" << destroys.load()
              << ",\"kernel_calls\":12,\"actual_workers\":320,\"sibling_preserved\":true,"
                 "\"cleanup\":true,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,\"performance_"
                 "qualified\":false,\"full_goal_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--phase-child") {
        for (unsigned int phase = 0; phase < 4; ++phase) {
            std::string line;
            if (!std::getline(std::cin, line) || line != "run") return EXIT_FAILURE;
            std::cout << "PHASE " << phase << '\n' << std::flush;
        }
        return EXIT_SUCCESS;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout
            << "Usage: public-shared-test SDK_APPLICATION PRELOAD PRIVATE_DIRECTORY\n"
               "       public-shared-test --gemm-priority SDK_APPLICATION PRELOAD "
               "PRIVATE_DIRECTORY\n"
               "       public-shared-test --gemm-priority-timing scalar|sharded SDK_APPLICATION "
               "PRELOAD PRIVATE_DIRECTORY\n"
               "       public-shared-test --priority SDK_APPLICATION PRELOAD "
               "PRIVATE_DIRECTORY [--cached-lookups] [--observe-block-completion]\n"
               "       public-shared-test --single-profile SDK_APPLICATION PRELOAD "
               "PRIVATE_DIRECTORY SINGLE_OUTPUT\n"
               "       public-shared-test --pair-profile SDK_APPLICATION PRELOAD "
               "PRIVATE_DIRECTORY SINGLE_INPUT PAIR_OUTPUT\n"
               "       public-shared-test --arrival SDK_APPLICATION PRELOAD "
               "PRIVATE_DIRECTORY SINGLE_INPUT PAIR_INPUT [--cached-lookups]\n";
        return EXIT_SUCCESS;
    }
    try {
        if (argc == 6 && std::string_view(argv[1]) == "--gemm-priority-timing") {
            const auto variant = std::string_view(argv[2]);
            expect(variant == "scalar" || variant == "sharded", "explicit GEMM signal variant");
            const auto server = current_process_identity();
            expect(server.has_value(), "authenticated shared GEMM timing owner");
            run_gemm_priority(
                {argv[3], argv[4], std::string(argv[5]) + "/service.sock", argv[5], *server},
                variant == "sharded", true);
            return EXIT_SUCCESS;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--gemm-priority") {
            const auto server = current_process_identity();
            expect(server.has_value(), "authenticated shared GEMM owner");
            run_gemm_priority(
                {argv[2], argv[3], std::string(argv[4]) + "/service.sock", argv[4], *server});
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--priority-path-protocol") {
            expect(!valid_socket_address("") && valid_socket_address(std::string(107, 'x')) &&
                       !valid_socket_address(std::string(108, 'x')) &&
                       !valid_socket_address(std::string(4096, 'x')),
                   "exact private Unix address boundaries without any SDK/file/process work");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--phase-protocol") {
            phase_protocol(argv[0]);
            return EXIT_SUCCESS;
        }
        if ((argc == 6 && std::string_view(argv[1]) == "--single-profile") ||
            (argc == 7 && std::string_view(argv[1]) == "--pair-profile") ||
            ((argc == 7 || argc == 8) && std::string_view(argv[1]) == "--arrival")) {
            const bool is_cached = argc == 8;
            expect(!is_cached || std::string_view(argv[7]) == "--cached-lookups",
                   "exact nonrepeated arrival cached option required");
            const auto server = current_process_identity();
            expect(server.has_value(), "authenticated public profiling/arrival owner");
            const ChildConfig child_config{argv[2], argv[3], std::string(argv[4]) + "/service.sock",
                                           argv[4], *server};
            if (std::string_view(argv[1]) == "--single-profile")
                train_public_singles(child_config, argv[5]);
            else
                run_public_arrivals(child_config,
                                    {.singles = argv[5],
                                     .pairs = argv[6],
                                     .is_cached = is_cached,
                                     .is_learning = std::string_view(argv[1]) == "--pair-profile"});
            return EXIT_SUCCESS;
        }
        if (argc >= 5 && argc <= 7 && std::string_view(argv[1]) == "--priority") {
            bool is_cached{}, observes_completion{};
            for (int index = 5; index < argc; ++index) {
                const auto flag = std::string_view(argv[index]);
                if (flag == "--cached-lookups" && !is_cached)
                    is_cached = true;
                else if (flag == "--observe-block-completion" && !observes_completion)
                    observes_completion = true;
                else
                    expect(false, "exact nonrepeated ordinary priority options required");
            }
            const auto server = current_process_identity();
            expect(server.has_value(), "authenticated public priority owner");
            run_priority(
                {argv[2], argv[3], std::string(argv[4]) + "/service.sock", argv[4], *server},
                is_cached, observes_completion);
            return EXIT_SUCCESS;
        }
        expect(argc == 4, "exact ordinary application, preload and evidence directory required");
        run(argv[1], argv[2], argv[3]);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Public shared lifecycle: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
