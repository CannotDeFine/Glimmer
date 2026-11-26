#include "glimmer/control/execution_process_memory_protocol.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_process_memory_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kNativePool = 0xd0000;
constexpr std::uint32_t kGuard = 0xdeadbeef;
enum class Scenario { kNormal, kMalformed, kNativeOutputError, kPoolOutputError };
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "public owned process-memory operation succeeds");
}
template <typename Pointer>
Pointer address(std::uintptr_t bits) {
    // Synthetic SDK/output storage bits are never dereferenced by the fixture.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<Pointer>(bits);
}
template <typename Function>
Function cached(void* sdk, const char* name) {
    const auto query =
        reinterpret_cast<decltype(&cuGetProcAddress_v2)>(::dlsym(sdk, "cuGetProcAddress_v2"));
    expect(query, "owned query entry is exported");
    void* function{};
    CUdriverProcAddressQueryResult status{};
    check(query(name, &function, 12060, 0, &status));
    expect(function && status == CU_GET_PROC_ADDRESS_SUCCESS, "cached owned API selected");
    return reinterpret_cast<Function>(function);
}
void application(const char* driver, Scenario scenario) {
    auto* sdk = ::dlopen(driver, RTLD_NOW | RTLD_LOCAL);
    expect(sdk, "open isolated metadata-only provider");
    const auto default_pool =
        cached<decltype(&cuDeviceGetDefaultMemPool)>(sdk, "cuDeviceGetDefaultMemPool");
    const auto current_pool = cached<decltype(&cuDeviceGetMemPool)>(sdk, "cuDeviceGetMemPool");
    const auto set_pool = cached<decltype(&cuDeviceSetMemPool)>(sdk, "cuDeviceSetMemPool");
    const auto allocate = cached<decltype(&cuMemAllocAsync)>(sdk, "cuMemAllocAsync");
    const auto allocate_pool =
        cached<decltype(&cuMemAllocFromPoolAsync)>(sdk, "cuMemAllocFromPoolAsync");
    const auto release = cached<decltype(&cuMemFree)>(sdk, "cuMemFree");
    const auto explicit_current =
        reinterpret_cast<decltype(&cuDeviceGetMemPool)>(::dlsym(sdk, "cuDeviceGetMemPool"));
    expect(explicit_current == &cuDeviceGetMemPool,
           "explicit provider handle cannot bypass ownership");
    auto pool = address<CUmemoryPool>(0x778899);
    check(cuDeviceGetDefaultMemPool(&pool, 0));
    expect(pool && reinterpret_cast<std::uintptr_t>(pool) != kNativePool,
           "context-free query returns an opaque local pool token");
    CUmemoryPool again{};
    check(default_pool(&again, 0));
    expect(again == pool, "repeated pool query preserves the public token");
    check(current_pool(&again, 0));
    const auto original_current = again;
    expect(original_current && original_current != pool &&
               reinterpret_cast<std::uintptr_t>(original_current) != kNativePool + 0x10000,
           "a distinct current native pool has its own opaque public token");
    check(explicit_current(&again, 0));
    expect(again == original_current, "repeat current query preserves its distinct token");
    check(cuDeviceSetMemPool(0, pool));
    check(current_pool(&again, 0));
    expect(again == pool, "direct setter changes the observed current pool to the default");
    check(set_pool(0, original_current));
    check(explicit_current(&again, 0));
    expect(again == original_current, "cached setter restores the distinct current pool");
    expect(cuDeviceSetMemPool(0, address<CUmemoryPool>(kNativePool)) == CUDA_ERROR_INVALID_HANDLE,
           "raw server pool bits do not confer public authority");
    expect(cuDeviceGetDefaultMemPool(nullptr, 0) == CUDA_ERROR_INVALID_VALUE,
           "null output preserves the original SDK error");
    auto sentinel = address<CUmemoryPool>(0x123456);
    expect(cuDeviceGetMemPool(&sentinel, -1) == CUDA_ERROR_INVALID_DEVICE &&
               sentinel == address<CUmemoryPool>(0x123456),
           "untouched native error output remains unchanged");
    expect(
        cudaMemPoolTrimTo(reinterpret_cast<cudaMemPool_t>(pool), 0) == cudaSuccess &&
            cudaMemPoolTrimTo(reinterpret_cast<cudaMemPool_t>(pool), 13) == cudaErrorInvalidValue &&
            cudaPeekAtLastError() == cudaErrorInvalidValue &&
            cudaGetLastError() == cudaErrorInvalidValue,
        "context-free Runtime trim preserves its independent native status and last error");
    expect(static_cast<int>(cudaMemPoolTrimTo(reinterpret_cast<cudaMemPool_t>(pool), 10719)) ==
                   10719 &&
               static_cast<int>(cudaPeekAtLastError()) == 10719 &&
               static_cast<int>(cudaGetLastError()) == 10719,
           "Runtime API-failure-base errors above 10000 preserve original status bits");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited frontend ownership control");
    if (!child) {
        CUmemoryPool inherited{};
        const bool rejected = cuDeviceGetMemPool(&inherited, 0) == CUDA_ERROR_INVALID_CONTEXT &&
                              glimmer_process_memory_native_calls() == 0;
        ::_exit(rejected ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == EXIT_SUCCESS,
           "inherited process tokens reject before frontend locks or caller-native work");
    if (scenario == Scenario::kPoolOutputError) {
        auto failed = address<CUmemoryPool>(0x1122334455667788ULL);
        expect(cuDeviceGetMemPool(&failed, -2) == CUDA_ERROR_UNKNOWN &&
                   reinterpret_cast<std::uintptr_t>(failed) == 0x11223344aabbccddULL,
               "quarantined error-side pool output preserves captured partial-write bits");
        expect(cuDeviceSetMemPool(0, failed) == CUDA_ERROR_UNKNOWN,
               "failed output cannot mint a pool capability or trigger replay");
    } else {
        CUcontext first{};
        check(cuCtxCreate(&first, 0, 0));
        CUstream stream{};
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CUdeviceptr a = 0x1122334455667788ULL;
        if (scenario != Scenario::kNormal) {
            const auto result = allocate(&a, 64, stream);
            if (scenario == Scenario::kMalformed)
                expect(result == CUDA_ERROR_UNKNOWN && a == 0x1122334455667788ULL,
                       "malformed successful receipt cannot publish an allocation");
            else
                expect(result == CUDA_ERROR_UNKNOWN && a == 0x11223344aabbccddULL,
                       "allocation error output is copied without granting range authority");
            expect(cuMemAllocAsync(&a, 64, stream) == CUDA_ERROR_UNKNOWN,
                   "unknown process ownership rejects subsequent allocation without replay");
        } else {
            check(cuMemAllocAsync(&a, 64, nullptr));
            CUdeviceptr b{};
            check(allocate_pool(&b, 64, pool, stream));
            CUdeviceptr empty = 0x76543210;
            check(allocate(&empty, 0, CU_STREAM_PER_THREAD));
            expect(empty == 0, "zero-size original allocation publishes a written null");
            empty = 0x76543210;
            check(cuMemAllocFromPoolAsync(&empty, 0, original_current, stream));
            expect(empty == 0, "direct explicit-pool zero-size route preserves written null");
            expect(cuMemAllocAsync(nullptr, 64, stream) == CUDA_ERROR_INVALID_VALUE,
                   "null allocation output is passed to the SDK");
            expect(cuMemAllocFromPoolAsync(&empty, 64, address<CUmemoryPool>(kNativePool),
                                           stream) == CUDA_ERROR_INVALID_HANDLE &&
                       cuMemAllocAsync(&empty, 64, reinterpret_cast<CUstream>(first)) ==
                           CUDA_ERROR_INVALID_HANDLE,
                   "foreign pool and wrong-kind stream reject before native admission");
            const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
            void* inaccessible =
                ::mmap(nullptr, page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            expect(inaccessible != MAP_FAILED, "create inaccessible output storage");
            expect(cuMemAllocAsync(static_cast<CUdeviceptr*>(inaccessible), 64, stream) ==
                           CUDA_ERROR_INVALID_VALUE &&
                       ::munmap(inaccessible, page) == 0,
                   "unwritable output rejects safely before RPC");
            check(cuCtxDestroy(first));
            CUcontext associated = address<CUcontext>(0x1234);
            check(cuPointerGetAttribute(&associated, CU_POINTER_ATTRIBUTE_CONTEXT, a));
            expect(!associated, "process allocation metadata survives source context Close");
            CUdeviceptr alias{};
            check(cuPointerGetAttribute(&alias, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, a + 4));
            expect(alias == a + 4, "interior process-pointer metadata preserves offset");
            cudaPointerAttributes attributes{};
            expect(cudaPointerGetAttributes(&attributes, address<void*>(b)) == cudaSuccess &&
                       attributes.type == cudaMemoryTypeDevice && attributes.device == 0 &&
                       attributes.devicePointer == address<void*>(b) && !attributes.hostPointer,
                   "Runtime metadata uses the process carrier without recreating source context");
            CUcontext second{};
            check(cuCtxCreate(&second, 0, 0));
            check(cuMemsetD32(a + 4, 7, 14));
            check(cuMemcpyDtoD(b + 4, a + 4, 56));
            expect(cuMemsetD32(a + 60, 9, 2) == CUDA_ERROR_INVALID_VALUE,
                   "cross-context access still enforces the exact allocation extent");
            expect(cudaMemPoolTrimTo(reinterpret_cast<cudaMemPool_t>(pool), 0) == cudaSuccess,
                   "live process memory and pool token survive source Close");
            check(release(a));
            expect(cuMemFree(a) == CUDA_ERROR_INVALID_VALUE,
                   "successful free retires exactly one public allocation");
            expect(cudaFree(address<void*>(b)) == cudaSuccess,
                   "Runtime synchronous free accepts process allocation in a new context");
            check(cuCtxDestroy(second));
            check(current_pool(&again, 0));
            expect(again == original_current,
                   "selected process pool identity survives both contexts retiring");
        }
    }
    expect(glimmer_process_memory_native_calls() == 0,
           "public operations never enter the caller-native resource provider");
    expect(::dlclose(sdk) == 0, "balance provider reference");
}

struct StopServer {
    ExecutionSocketServer& server;
    ~StopServer() {
        server.stop();
    }
};
void run(const std::string& path, const char* driver, Scenario scenario) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated process-memory endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "endpoint identity exists");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind exact authenticated endpoint");
    auto client = std::async(std::launch::async, [&] { application(driver, scenario); });
    const StopServer stop{server};
    ExecutionAsyncFixture asynchronous;
    std::uint64_t carrier{}, first{}, second{}, next_allocation{}, next_stream{};
    std::uint64_t selected_pool{2};
    unsigned int allocations{}, releases{}, queries{}, memory_calls{}, pool_selections{};
    bool closed_first{}, saw_fault{};
    std::map<std::uint64_t, std::array<std::uint32_t, 16>> images;
    const auto deadline = Clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline,
               "bounded authenticated endpoint progress");
        auto request = server.take_request();
        if (!request) continue;
        expect(!saw_fault, "uncertain ownership cannot submit later remote work");
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        const auto operation = request->message.header.operation;
        ExecutionPayloadReader input(request->message.payload);
        if (operation == ExecutionOpcode::kHello) {
            std::uint32_t role{}, flags{};
            auto probe = input;
            if (probe.read_u32(role) && probe.read_u32(flags) && probe.empty()) {
                expect(!flags && (role == kExecutionProcessMemoryRole ||
                                  role == kExecutionPointerMetadataRole),
                       "exact helper role");
                if (role == kExecutionProcessMemoryRole) {
                    expect(!carrier, "one stable process owner carrier");
                    carrier = request->connection;
                }
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode helper ownership proof");
                reply.data.assign(ownership->begin(), ownership->end());
            } else {
                expect(decode_execution_context_bootstrap(request->message.payload).has_value(),
                       "ordinary context creation retains its existing bootstrap");
                if (!first)
                    first = request->connection;
                else {
                    expect(closed_first && !second, "second context follows first destruction");
                    second = request->connection;
                }
            }
        } else if (operation == ExecutionOpcode::kProcessMemory) {
            ExecutionProcessMemoryRequest call;
            expect(read_execution_process_memory_request(input, call) && input.empty(),
                   "canonical fixed-width process-memory request");
            const bool is_allocation =
                call.operation == ExecutionProcessMemoryOperation::kAllocateAsync ||
                call.operation == ExecutionProcessMemoryOperation::kAllocateFromPoolAsync;
            expect(request->connection == (is_allocation ? first : carrier),
                   "allocation retains source ordering; pool calls retain context-free ownership");
            ExecutionProcessMemoryObservation effects;
            if (is_allocation) {
                ++allocations;
                expect(!closed_first && (call.bytes == 64 || call.bytes == 0) && call.pool <= 2,
                       "exact source allocation capability");
                if (scenario != Scenario::kNormal) {
                    saw_fault = true;
                    effects.output_written = true;
                    if (scenario == Scenario::kMalformed) {
                        reply.id = kExecutionProcessAllocationTag;
                        reply.address = 0x100000;
                    } else {
                        reply.status = CUDA_ERROR_UNKNOWN;
                        reply.original_status = CUDA_ERROR_INVALID_VALUE;
                        reply.parameter_bytes = 1;
                        reply.address = (call.output_seed & 0xffffffff00000000ULL) | 0xaabbccdd;
                    }
                } else if (!call.has_output) {
                    reply.status = CUDA_ERROR_INVALID_VALUE;
                    reply.original_status = reply.status;
                } else {
                    effects.output_written = true;
                    if (call.bytes) {
                        reply.id = kExecutionProcessAllocationTag | ++next_allocation;
                        reply.address = 0x100000 * next_allocation;
                        auto& image = images[reply.id];
                        image.fill(0);
                        image.front() = image.back() = kGuard;
                    }
                }
            } else if (call.operation == ExecutionProcessMemoryOperation::kGetDefaultPool ||
                       call.operation == ExecutionProcessMemoryOperation::kGetCurrentPool) {
                ++queries;
                if (!call.has_output || call.device < 0) {
                    reply.status =
                        call.has_output ? CUDA_ERROR_INVALID_DEVICE : CUDA_ERROR_INVALID_VALUE;
                    reply.original_status = reply.status;
                    reply.address = call.output_seed;
                    if (call.device == -2) {
                        saw_fault = true;
                        reply.status = CUDA_ERROR_UNKNOWN;
                        reply.parameter_bytes = 1;
                        effects.output_written = true;
                        reply.address = (call.output_seed & 0xffffffff00000000ULL) | 0xaabbccdd;
                    }
                } else {
                    effects.output_written = true;
                    reply.id = call.operation == ExecutionProcessMemoryOperation::kGetDefaultPool
                                   ? 1
                                   : selected_pool;
                    reply.address = kNativePool + (reply.id - 1) * 0x10000;
                }
            } else {
                expect(call.pool >= 1 && call.pool <= 2 && !call.has_output,
                       "pool mutation uses authenticated identity");
                if (call.operation == ExecutionProcessMemoryOperation::kTrimPool)
                    reply.original_status = call.bytes == 10719 ? 10719
                                            : call.bytes == 13  ? cudaErrorInvalidValue
                                                                : cudaSuccess;
                else {
                    expect(call.operation == ExecutionProcessMemoryOperation::kSetCurrentPool,
                           "only covered pool mutation reaches endpoint");
                    expect(call.pool == (pool_selections ? 2U : 1U) && pool_selections < 2,
                           "both distinct pool selections reach the original endpoint once");
                    selected_pool = call.pool;
                    ++pool_selections;
                }
            }
            ExecutionPayloadWriter output;
            expect(write_execution_process_memory_observation(output, effects),
                   "encode observed output effect");
            reply.data = output.take();
        } else if (operation == ExecutionOpcode::kCreateStream) {
            reply.id = ++next_stream;
        } else if (operation == ExecutionOpcode::kPointerMetadata) {
            ExecutionPointerQuery query;
            expect(read_execution_pointer_query(input, query) && input.empty() && closed_first &&
                       query.kind == ExecutionPointerKind::kProcessDevice &&
                       query.source_session == carrier && !query.query_session &&
                       images.contains(query.allocation),
                   "pointer query uses surviving process capability");
            ExecutionPointerObservation observation;
            if (query.domain == ExecutionPointerDomain::kRuntime) {
                observation.effects = 15;
                observation.type = 2;
                observation.device_alias = ExecutionPointerAlias::kInput;
            } else {
                observation.effects = 4;
                observation.device_alias = query.selector == 1 ? ExecutionPointerAlias::kNull
                                                               : ExecutionPointerAlias::kInput;
            }
            ExecutionPayloadWriter output;
            expect(write_execution_pointer_observation(output, observation),
                   "encode actual pointer observation");
            reply.data = output.take();
        } else if (operation == ExecutionOpcode::kMemory) {
            expect(request->connection == second,
                   "new context consumes original process allocation");
            std::uint32_t kind{}, async{}, value{};
            std::uint64_t target{}, target_offset{}, source{}, source_offset{}, count{}, stream{};
            expect(input.read_u32(kind) && input.read_u32(async) && input.read_u64(target) &&
                       input.read_u64(target_offset) && input.read_u64(source) &&
                       input.read_u64(source_offset) && input.read_u64(count) &&
                       input.read_u32(value) && input.read_u64(stream) && input.empty() && !async &&
                       !stream && target_offset == 4 && images.contains(target),
                   "cross-context compute preserves original range token and offset");
            auto& destination = images.at(target);
            if (kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32)) {
                expect(count == 14 && value == 7 && !source && !source_offset,
                       "exact integer fill");
                std::fill(destination.begin() + 1, destination.end() - 1, value);
            } else {
                expect(kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kCopy) &&
                           source_offset == 4 && count == 56 && images.contains(source),
                       "exact D2D copy");
                const auto& origin = images.at(source);
                std::copy(origin.begin() + 1, origin.end() - 1, destination.begin() + 1);
            }
            ++memory_calls;
        } else if (operation == ExecutionOpcode::kRelease) {
            std::uint64_t id{};
            expect(input.read_u64(id) && input.empty() && request->connection == second &&
                       images.contains(id),
                   "cross-context synchronous free uses exact tagged identity once");
            const auto& image = images.at(id);
            expect(
                image.front() == kGuard && image.back() == kGuard &&
                    std::all_of(image.begin() + 1, image.end() - 1, [](auto v) { return v == 7; }),
                "all IPC numerical values and guards survive source context Close and trim");
            images.erase(id);
            ++releases;
        } else if (operation == ExecutionOpcode::kCaptureFrontier) {
            std::uint32_t kind{};
            std::uint64_t stream{};
            expect(input.read_u32(kind) && kind <= 2 && input.read_u64(stream) && input.empty(),
                   "Runtime free preserves existing source synchronization");
            reply = asynchronous.capture(request->connection,
                                         kind ? std::optional{stream} : std::nullopt);
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier) {
            reply = asynchronous.observe(*request);
        } else {
            expect(operation == ExecutionOpcode::kClose, "only covered operations reach endpoint");
            if (request->connection == first) closed_first = true;
            asynchronous.close(request->connection);
        }
        auto header = request->message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request->connection;
        header.incarnation = {83, 89};
        const auto encoded = encode_execution_reply(reply);
        expect(encoded && server.reply(request->connection, {header, *encoded}),
               "publish one exact receipt");
    }
    client.get();
    if (scenario == Scenario::kNormal)
        expect(allocations == 5 && releases == 2 && queries == 9 && memory_calls == 2 &&
                   pool_selections == 2 && selected_pool == 2 && images.empty(),
               "all operations complete once with normal resource cleanup");
    else
        expect(saw_fault && releases == 0, "failure never fabricates resource release");
    std::cout << "Owned process-memory public IPC verified; GPU qualification remains separate\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return EXIT_FAILURE;
    Scenario scenario = Scenario::kNormal;
    if (argc == 3) {
        const std::string_view mode{argv[2]};
        if (mode == "--malformed")
            scenario = Scenario::kMalformed;
        else if (mode == "--native-output-error")
            scenario = Scenario::kNativeOutputError;
        else if (mode == "--pool-output-error")
            scenario = Scenario::kPoolOutputError;
        else
            return EXIT_FAILURE;
    }
    char pattern[] = "/tmp/glimmer-process-memory-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock", argv[1], scenario);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
