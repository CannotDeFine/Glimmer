#include "runtime.h"
#include "profile_archive.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
constexpr unsigned int kBlocks = 4096, kThreads = 32, kWorkers = 320, kCommands = 4;
constexpr std::size_t kElements = std::size_t{kBlocks} * kThreads, kGuards = 64;
constexpr std::uint32_t kGuard = 0x173a92c1;
constexpr std::string_view kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry add_once(.param .u64 output, .param .u32 value) {
 .reg .pred %p;
 .reg .b32 %r<8>;
 .reg .b64 %rd<9>;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [value];
 mov.u64 %rd2, %globaltimer;
WAIT:
 mov.u64 %rd3, %globaltimer; sub.u64 %rd4, %rd3, %rd2;
 setp.lt.u64 %p, %rd4, 1000000; @%p bra WAIT;
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 mul.wide.u32 %rd5, %r5, 4; add.u64 %rd6, %rd1, %rd5;
 atom.global.add.u32 %r6, [%rd6], %r1;
 ret;
}
)ptx";

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": " + std::to_string(status));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete, "complete SDK operation");
    check(result.status, "actual session SDK result");
}
struct NativeTrace {
    unsigned int creates{}, destroys{};
    std::vector<unsigned int> workers;
    unsigned int function_setters{};
    unsigned int pointer_getters{}, query_contexts{};
};
NativeTrace trace;
CUresult CUDAAPI create_context(CUcontext* context, unsigned int flags, CUdevice device) {
    const auto status = cuCtxCreate(context, flags, device);
    if (status == CUDA_SUCCESS) ++trace.creates;
    return status;
}
CUresult CUDAAPI destroy_context(CUcontext context) {
    const auto status = cuCtxDestroy(context);
    if (status == CUDA_SUCCESS) ++trace.destroys;
    return status;
}
CUresult CUDAAPI launch(CUfunction function, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                        CUstream stream, void** parameters, void** extra) {
    const auto status =
        cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
    if (status == CUDA_SUCCESS) trace.workers.push_back(gx);
    return status;
}
template <class Function>
Function configuration_symbol(const char* name) {
    void* address{};
    CUdriverProcAddressQueryResult queried{};
    check(cuGetProcAddress(name, &address, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &queried),
          "resolve exact original legacy context ABI");
    expect(address && queried == CU_GET_PROC_ADDRESS_SUCCESS, "original context SDK symbol exists");
    return reinterpret_cast<Function>(address);
}
CudaExecutionDomainDispatch configured_domain_dispatch() {
    CudaExecutionDomainDispatch result{cuInit,          cuDeviceGet,     create_context,
                                       cuCtxSetCurrent, cuCtxGetCurrent, destroy_context};
    auto& configuration = result.configuration;
    configuration = {
        cuCtxGetFlags,
        cuCtxSetFlags,
        cuCtxGetLimit,
        cuCtxSetLimit,
        cuCtxGetCacheConfig,
        cuCtxSetCacheConfig,
        configuration_symbol<decltype(configuration.get_shared)>("cuCtxGetSharedMemConfig"),
        configuration_symbol<decltype(configuration.set_shared)>("cuCtxSetSharedMemConfig")};
    return result;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{{cuCtxGetCurrent, launch, cuEventCreate, cuEventRecord, cuEventQuery,
                                cuEventDestroy, cuStreamSynchronize},
                               cuInit,
                               cuDeviceGet,
                               create_context,
                               cuCtxSetCurrent,
                               destroy_context,
                               cuModuleLoadData,
                               cuModuleUnload,
                               cuModuleGetFunction,
                               cuFuncGetParamInfo,
                               cuMemAlloc,
                               cuMemFree,
                               cuMemcpyHtoD,
                               cuMemcpyDtoH,
                               cuStreamCreate,
                               cuStreamDestroy,
                               cuMemHostAlloc,
                               cuMemFreeHost,
                               cuMemHostGetDevicePointer,
                               cuMemcpyHtoDAsync,
                               cuMemcpyDtoHAsync,
                               cuOccupancyMaxActiveBlocksPerMultiprocessor};
    result.device_uuid = cuDeviceGetUuid;
    result.driver_version = cuDriverGetVersion;
    result.device_attribute = cuDeviceGetAttribute;
    result.function_get_attribute = cuFuncGetAttribute;
    result.function_set_attribute = cuFuncSetAttribute;
    result.pointer_get_attribute = [](void* output, CUpointer_attribute attribute,
                                      CUdeviceptr input) {
        ++trace.pointer_getters;
        return cuPointerGetAttribute(output, attribute, input);
    };
    result.function_set_cache = [](CUfunction function, CUfunc_cache cache) {
        const auto status = cuFuncSetCacheConfig(function, cache);
        if (status == CUDA_SUCCESS) ++trace.function_setters;
        return status;
    };
    result.stream_wait_event = cuStreamWaitEvent;
    result.context_get_flags = cuCtxGetFlags;
    result.context_set_flags = cuCtxSetFlags;
    result.context.get_cache = cuCtxGetCacheConfig;
    result.context.set_cache = cuCtxSetCacheConfig;
    return result;
}
std::vector<std::uint32_t> original_output() {
    std::vector<std::uint32_t> result(kElements + 2 * kGuards, kGuard);
    std::fill_n(result.begin() + kGuards, kElements, 0);
    return result;
}
void verify_output(std::span<const std::uint32_t> output, std::uint32_t expected) {
    expect(output.size() == kElements + 2 * kGuards, "complete guarded output size");
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(
            output[index] == (index < kGuards || index >= kGuards + kElements ? kGuard : expected),
            "every non-idempotent value and guard must agree");
}
std::array<std::byte, 12> arguments(CUdeviceptr address) {
    std::array<std::byte, 12> result{};
    const auto output = address + kGuards * sizeof(std::uint32_t);
    constexpr std::uint32_t value = 7;
    std::memcpy(result.data(), &output, 8);
    std::memcpy(result.data() + 8, &value, 4);
    return result;
}
CudaProfileConfig single_config() {
    CudaProfileConfig result;
    result.workers = {kWorkers};
    result.warmup_calls = 0;
    result.samples_per_candidate = 1;
    return result;
}
void retire_prefix(CudaExecutionSession& session) {
    const auto captured = session.capture_frontier();
    check(captured.status, "capture the actual original/private prefix");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (true) {
        const auto status = session.query_frontier(captured.frontier);
        if (status == CUDA_SUCCESS) break;
        expect(status == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline,
               "bounded actual private-prefix retirement, never inferred from a host return");
        check(session.poll().status, "observe actual private-prefix events");
    }
    success(session.deactivate());
    success(session.activate());
}
void train_originals(const std::string& path) {
    auto learning = std::make_shared<CudaProfileCache>(single_config());
    check(cuInit(0), "initialize Driver only after both clients were forked");
    CudaExecutionDomainOwner owner(configured_domain_dispatch(),
                                   {98, 0, CU_CTX_MAP_HOST, 1, std::size_t{64} * 1024 * 1024});
    check(owner.initialize(), "configured single-profile native domain");
    {
        auto member = owner.create_member({{1, 1}, 1});
        check(member.status, "configured single-profile logical member");
        CudaExecutionSession session({1, 1}, 1, dispatch(), {}, {kWorkers}, {}, learning, false, {},
                                     std::move(member.member));
        success(session.initialize(0));
        CUdevice device{};
        int sms{}, major{}, minor{};
        check(cuDeviceGet(&device, 0), "exact device");
        check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
              "actual device resources");
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
              "actual device architecture");
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
              "actual device architecture minor");
        expect(sms == 80 && major == 7 && minor == 0, "this exact input requires the 80-SM V100");
        const auto configured =
            session.context_control(CudaContextControl::kSetCache, CU_FUNC_CACHE_PREFER_SHARED);
        expect(configured.status == CUDA_SUCCESS && configured.original_status == CUDA_SUCCESS,
               "single measurements use an actually normalized requested context configuration");
        const auto module = session.load_ptx(kSource);
        success(module);
        const auto function = session.get_function(module.id, "add_once");
        success(function);
        const auto configured_function = session.function_control(
            CudaFunctionControl::kSetCache, function.id, CU_FUNC_CACHE_PREFER_SHARED);
        expect(configured_function.status == CUDA_SUCCESS &&
                   configured_function.original_status == CUDA_SUCCESS,
               "single measurements bind successful original/private function cache configuration");
        expect(function.parameter_bytes == 12 && function.arguments.size() == 2,
               "original Driver ABI is unchanged");
        auto output = original_output();
        const auto memory = session.allocate(output.size() * sizeof(output[0]));
        success(memory);
        success(session.upload({memory.id, 0}, std::as_bytes(std::span{output})));
        retire_prefix(session);
        const auto packed = arguments(memory.address);
        for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
            const auto accepted = session.accept_kernel(sequence, {.function = function.id,
                                                                   .grid = {kBlocks, 1, 1},
                                                                   .block = {kThreads, 1, 1},
                                                                   .parameters = packed});
            expect(accepted.accepted(), "distinct original/private single-profile logical input");
            const auto pending = session.pending_command();
            const auto issued = session.issue(pending.command);
            if (issued.state != CudaExecutionIssueState::kIssued)
                throw std::runtime_error(
                    "actual single-profile issue: sequence=" + std::to_string(sequence) +
                    ", pending_status=" + std::to_string(pending.status) +
                    ", token=" + std::to_string(bool(pending.command)) +
                    ", issue_state=" + std::to_string(static_cast<unsigned int>(issued.state)) +
                    ", native_status=" + std::to_string(issued.native_status) +
                    ", completion_status=" + std::to_string(issued.completion_status));
            check(issued.native_status, "single-profile launch");
            check(issued.completion_status, "single-profile completion boundary");
            expect(session.acknowledge_result(accepted.receipt), "one original host result ACK");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
            while (session.snapshot().submissions.outstanding) {
                check(session.poll().status, "actual single-profile final retirement");
                expect(std::chrono::steady_clock::now() < deadline, "bounded single-profile work");
            }
            retire_prefix(session);
        }
        success(session.download({memory.id, 0}, std::as_writable_bytes(std::span{output})));
        verify_output(output, 14);
        retire_prefix(session);
        success(session.deactivate());
        {
            CudaExecutionSession querying({7, 7}, 7, dispatch());
            CudaExecutionSession carrier({8, 8}, 8, dispatch());
            success(querying.initialize(0));
            ++trace.query_contexts;
            const CudaPointerQuery query{
                CudaPointerDomain::kDriver, 9, CudaPointerKind::kDevice, {memory.id, 0}};
            for (auto* selected : {&querying, static_cast<CudaExecutionSession*>(nullptr)}) {
                const auto observed = carrier.pointer_metadata(query, selected, &session);
                expect(
                    observed.disposition == CudaSessionDisposition::kComplete &&
                        observed.status == CUDA_SUCCESS &&
                        observed.original_status == CUDA_SUCCESS &&
                        observed.context_selection == CudaPointerContextSelection::kQuery &&
                        observed.observation && observed.observation->effects == 2 &&
                        observed.observation->device == 0,
                    "actual original pointer status and independent proven query/NULL selection");
                CUcontext current{};
                check(cuCtxGetCurrent(&current), "actual post-query TLS");
                expect(bool(current) == bool(selected), "actual query selection remains current");
            }
            auto invalid = query;
            invalid.range.offset = SIZE_MAX;
            const auto rejected = carrier.pointer_metadata(invalid, &querying, &session);
            expect(rejected.disposition == CudaSessionDisposition::kRejected &&
                       !rejected.original_status && !rejected.observation &&
                       rejected.context_selection == CudaPointerContextSelection::kUnchanged &&
                       trace.pointer_getters == 2,
                   "invalid ownership cannot select a context or call the requested getter");
            success(querying.activate());
            success(querying.shutdown());
            success(carrier.shutdown());
        }
        success(session.activate());
        success(session.shutdown());
    }
    check(owner.close(), "close retired single-profile domain");
    check(owner.shutdown(), "destroy actual single-profile owner");
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "exact application binary profile identity");
    const auto archive = learning->encode(*identity);
    const auto measured = learning->snapshot();
    expect(archive && measured.trained_cases == 1 && measured.samples == 2 &&
               !measured.rejected_samples && !measured.cancelled &&
               write_profile_archive(path, *archive) &&
               trace.workers == std::vector<unsigned int>{kBlocks, kWorkers},
           "qualified singles come from real distinct native/private commands, never fake clocks");
}
void signal_byte(int socket, char value) {
    expect(::send(socket, &value, 1, MSG_NOSIGNAL) == 1, "fixture barrier send");
}
void wait_byte(int socket, char expected) {
    pollfd descriptor{socket, POLLIN, 0};
    expect(::poll(&descriptor, 1, 30000) == 1, "bounded client barrier");
    char value{};
    expect(::recv(socket, &value, 1, 0) == 1 && value == expected, "exact client barrier");
}
ExecutionWireReply reply(std::optional<ExecutionWireReply> result, bool is_native = true,
                         const char* operation = "remote operation") {
    if (!result || result->status || result->disposition != ExecutionWireDisposition::kComplete ||
        (is_native && result->original_status != CUDA_SUCCESS))
        throw std::runtime_error(
            std::string(operation) +
            ": status=" + (result ? std::to_string(result->status) : "none") + ", disposition=" +
            (result ? std::to_string(static_cast<unsigned int>(result->disposition)) : "none") +
            ", original_status=" +
            (result && result->original_status ? std::to_string(*result->original_status)
                                               : "none"));
    return std::move(*result);
}
void client_input(const std::string& socket, TaskPeerIdentity server, int barrier) {
    wait_byte(barrier, 'C');
    ExecutionSocketClient client({socket, server, 30000});
    const auto bootstrap =
        encode_execution_context_bootstrap({ExecutionContextPurpose::kRuntimePrimary, 0, 1});
    expect(bootstrap.has_value(), "exact cold Runtime-primary descriptor");
    reply(client.open(*bootstrap), true, "open Runtime-primary");
    ExecutionPayloadWriter configuration;
    expect(configuration.write_u32(static_cast<std::uint32_t>(CudaContextControl::kSetCache)) &&
               configuration.write_u32(CU_FUNC_CACHE_PREFER_SHARED) && configuration.write_u64(0),
           "original requested context configuration");
    reply(client.exchange(ExecutionOpcode::kDriverContextControl, configuration.take()), true,
          "set original context cache preference");
    ExecutionPayloadWriter source;
    expect(source.write_string(kSource), "original source");
    const auto module =
        reply(client.exchange(ExecutionOpcode::kLoadPtx, source.take()), true, "load original PTX");
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("add_once"), "original lookup");
    const auto function = reply(client.exchange(ExecutionOpcode::kGetFunction, lookup.take()), true,
                                "get original function");
    expect(function.parameter_bytes == 12 && function.arguments.size() == 2, "original public ABI");
    ExecutionPayloadWriter function_configuration;
    expect(function_configuration.write_u32(
               static_cast<std::uint32_t>(CudaFunctionControl::kSetCache)) &&
               function_configuration.write_u64(function.id) &&
               function_configuration.write_u32(CU_FUNC_CACHE_PREFER_SHARED) &&
               function_configuration.write_u32(0),
           "original function cache configuration");
    reply(client.exchange(ExecutionOpcode::kFunctionControl, function_configuration.take()), true,
          "set original and private function cache preference");
    auto values = original_output();
    ExecutionPayloadWriter extent;
    expect(extent.write_u64(values.size() * sizeof(values[0])), "original full allocation");
    const auto memory = reply(client.exchange(ExecutionOpcode::kAllocate, extent.take()), true,
                              "allocate original output");
    ExecutionPayloadWriter upload;
    expect(upload.write_u64(memory.id) && upload.write_u64(0) &&
               upload.write_blob(std::as_bytes(std::span{values})),
           "all original values and guards");
    reply(client.exchange(ExecutionOpcode::kUpload, upload.take()), true, "upload original input");
    signal_byte(barrier, 'R');
    wait_byte(barrier, 'G');
    const auto packed = arguments(memory.address);
    std::array<std::uint64_t, kCommands> sequences{};
    for (auto& sequence : sequences) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(function.id) && input.write_u64(0), "original handles");
        for (const auto dimension : {kBlocks, 1U, 1U, kThreads, 1U, 1U})
            expect(input.write_u32(dimension), "original logical geometry");
        expect(input.write_u32(0) && input.write_blob(packed), "original packed arguments");
        const auto submitted = client.submit(ExecutionOpcode::kLaunch, input.take());
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
               "submit distinct original input before waiting for results");
        sequence = submitted.sequence;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    for (std::size_t index = 0; index < sequences.size(); ++index) {
        std::optional<ExecutionWireReply> observed;
        while (!(observed = client.take_reply(sequences[index])))
            expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded original host result");
        expect(reply(std::move(observed), true, "actual kernel host result").id == index + 1,
               "one result per logical input");
    }
    ExecutionPayloadWriter scope;
    expect(scope.write_u32(0) && scope.write_u64(0), "full logical frontier");
    const auto frontier =
        reply(client.exchange(ExecutionOpcode::kCaptureFrontier, scope.take()), false);
    ExecutionPayloadWriter frontier_id;
    expect(frontier_id.write_u64(frontier.id), "exact frontier identity");
    const auto frontier_bytes = frontier_id.take();
    while (true) {
        const auto observed = client.exchange(ExecutionOpcode::kQueryFrontier, frontier_bytes);
        expect(observed && !observed->original_status, "CPU-only logical retirement observation");
        if (!observed->status) break;
        expect(
            observed->status == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline,
            "bounded actual final logical retirement");
    }
    reply(client.exchange(ExecutionOpcode::kReleaseFrontier, frontier_bytes), false);
    ExecutionPayloadWriter download;
    expect(download.write_u64(memory.id) && download.write_u64(0) &&
               download.write_u64(values.size() * sizeof(values[0])),
           "full original output readback");
    const auto downloaded = reply(client.exchange(ExecutionOpcode::kDownload, download.take()));
    expect(downloaded.data.size() == values.size() * sizeof(values[0]), "all bytes returned");
    std::memcpy(values.data(), downloaded.data.data(), downloaded.data.size());
    verify_output(values, 7 * kCommands);
    const auto closed = reply(client.exchange(ExecutionOpcode::kClose), false);
    expect(!closed.original_status,
           "member-only close does not fabricate a native context destroy");
}
struct Child {
    pid_t pid{-1};
    int socket{-1};
    bool ready{};
    Child(const std::string& path, TaskPeerIdentity server) {
        std::array<int, 2> descriptors{};
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors.data()) == 0,
               "original client fixture socketpair");
        pid = ::fork();
        expect(pid >= 0, "fork original clients BEFORE any Driver initialization");
        if (!pid) {
            (void)::close(descriptors[0]);
            int status = EXIT_SUCCESS;
            try {
                client_input(path, server, descriptors[1]);
            } catch (const std::exception& error) {
                std::cerr << "Pair GPU client: " << error.what() << '\n';
                status = EXIT_FAILURE;
            }
            (void)::close(descriptors[1]);
            std::exit(status);
        }
        (void)::close(descriptors[1]);
        socket = descriptors[0];
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() {
        if (socket >= 0) (void)::close(socket);
        if (pid > 0) {
            int status{};
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
    TaskPeerIdentity identity() const {
        std::uint64_t start{};
        expect(read_process_start_time(static_cast<std::uint32_t>(pid), &start),
               "actual pinned child generation");
        return {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start};
    }
    void observe_ready() {
        if (ready) return;
        char value{};
        const auto received = ::recv(socket, &value, 1, MSG_DONTWAIT);
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
        expect(received == 1 && value == 'R', "original GPU client ready");
        ready = true;
    }
    void join() {
        int status{};
        expect(::waitpid(pid, &status, 0) == pid, "join exact owned input");
        pid = -1;
        expect(WIFEXITED(status) && WEXITSTATUS(status) == 0, "all client numerical/status checks");
    }
};
void run(const std::string& root) {
    trace.workers.reserve(16);
    const auto server = current_process_identity();
    expect(server.has_value(), "authenticated service owner");
    const auto path = root + "/service.sock", singles = root + "/single", pairs = root + "/pair";
    Child left(path, *server), right(path, *server);
    train_originals(singles);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configured_domain_dispatch(),
        CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 2, std::size_t{64} * 1024 * 1024});
    check(domain->initialize(), "actual shared native domain");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 2;
    config.transport.max_pending_requests = 8;
    config.transport.io_timeout_ms = 30000;
    config.session_limits.submissions = {4, 4096};
    config.persistent.workers = kWorkers;
    config.profile = single_config();
    config.profile_input = singles;
    config.client_priorities = {{left.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {right.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.paired = ExecutionServicePairConfig{
        domain, {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        {},     {80, 160},
        true,   pairs};
    config.exit_when_idle = true;
    ExecutionService service(config, dispatch());
    expect(service.start() && !service.save_pair_profile(), "start actual private sampler");
    signal_byte(left.socket, 'C');
    signal_byte(right.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
    while (!left.ready || !right.ready) {
        expect(service.step(1), "prepare actual independent shared members");
        left.observe_ready();
        right.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual member setup");
    }
    signal_byte(left.socket, 'G');
    signal_byte(right.socket, 'G');
    while (!service.finished()) {
        expect(service.step(1), "actual paired profile execution, retirement and cleanup");
        expect(std::chrono::steady_clock::now() < deadline, "bounded actual paired execution");
    }
    left.join();
    right.join();
    expect(service.request_stop() && service.save_pair_profile(),
           "publish complete actual samples");
    const auto measured = service.pair_profile_snapshot();
    const auto identity = profile_runtime_identity();
    const auto single_archive = read_profile_archive(singles);
    const auto archive = read_execution_archive(pairs, kCudaPairProfileArchiveMaxBytes);
    auto originals = std::make_shared<CudaProfileCache>(single_config(), nullptr, false);
    expect(identity && single_archive && originals->restore(*single_archive, *identity),
           "actual original samples restore without CUDA or replay");
    CudaPairProfileCache restored(originals, config.paired->profile, nullptr, false);
    expect(archive &&
               (*archive)[4 + std::string_view{"Glimmer pair profile archive"}.size()] ==
                   std::byte{3} &&
               (*single_archive)[4 + std::string_view{"Glimmer profile archive"}.size()] ==
                   std::byte{3} &&
               restored.restore(*archive, *identity) && restored.snapshot().loaded_pairs == 1 &&
               restored.snapshot().trained_pairs == 1 && measured.samples == kCommands &&
               measured.trained_pairs == 1 && !measured.active && !measured.rejected_samples &&
               !measured.cancelled && !service.save_pair_profile() &&
               trace.workers == std::vector<unsigned int>{kBlocks, kWorkers, 80, 80, 80, 160, 160,
                                                          80, 160, 160} &&
               trace.creates == 3 && trace.destroys == 3 && trace.query_contexts == 1 &&
               trace.pointer_getters == 2 && domain->snapshot().is_shutdown,
           "all actual native epochs, independent once-only samples and protected restore agree");
    expect(trace.function_setters == 6, "six actual successful original/private cache setters");
    CUcontext current{};
    check(cuCtxGetCurrent(&current), "final native owner observation");
    expect(!current && ::access(path.c_str(), F_OK) < 0 && errno == ENOENT,
           "native TLS and private service socket clean normally");
    std::cout << "{\"status\":\"passed\",\"private_service_pair_profile\":true,"
                 "\"samples\":4,\"native_launches\":10,\"contexts\":3,\"destroys\":3,"
                 "\"compute_domains\":2,\"metadata_query_contexts\":1,\"pointer_getters\":2,"
                 "\"query_selection_verified\":true,"
                 "\"configuration_bound_context\":true,\"configuration_bound_function\":true,"
                 "\"function_setters\":6,"
                 "\"single_archive_version\":3,\"pair_archive_version\":3,"
                 "\"client_commands\":8,\"worker_catalog\":[80,160],\"output_values\":"
              << 3 * (kElements + 2 * kGuards)
              << ",\"guards\":384,\"archive_readonly_restore\":true,\"overwrite_rejected\":true,"
                 "\"cleanup\":true,\"hardware_co_residency_qualified\":false,"
                 "\"full_device_qualified\":false,\"priority_qualified\":false,"
                 "\"performance_qualified\":false,\"full_goal_qualified\":false}\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout
            << "Usage: glimmer_execution_pair_profile_gpu_test [existing-evidence-directory]\n";
        return EXIT_SUCCESS;
    }
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]).empty())) return EXIT_FAILURE;
    try {
        char pattern[] = "/tmp/glimmer-pair-gpu-XXXXXX";
        const auto* root = argc == 2 ? argv[1] : ::mkdtemp(pattern);
        expect(root != nullptr, "owned temporary GPU input directory");
        run(root);
        if (argc == 1) {
            expect(::unlink((std::string(root) + "/single").c_str()) == 0 &&
                       ::unlink((std::string(root) + "/pair").c_str()) == 0 && ::rmdir(root) == 0,
                   "remove only the known idle self-test inputs");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Private paired profile GPU: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
