#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>

extern "C" unsigned int glimmer_memory_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
struct StopServerOnExit {
    ExecutionSocketServer& server;
    ~StopServerOnExit() {
        server.stop();
    }
};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void* runtime_pointer(CUdeviceptr address) {
    // Numeric server UVA is an opaque public Runtime device pointer.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(address);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "controlled memory API succeeds");
}
template <typename Function>
Function runtime_symbol(const char* name) {
    auto* value = ::dlsym(RTLD_DEFAULT, name);
    expect(value, "owned Runtime memory entry is exported");
    return reinterpret_cast<Function>(value);
}
template <typename Function>
Function cached(void* sdk, const char* name) {
    const auto query =
        reinterpret_cast<decltype(&cuGetProcAddress_v2)>(::dlsym(sdk, "cuGetProcAddress_v2"));
    expect(query, "Driver metadata query wrapper is exported");
    void* function{};
    CUdriverProcAddressQueryResult status{};
    expect(query(name, &function, 12000, 0, &status) == CUDA_SUCCESS && function &&
               status == CU_GET_PROC_ADDRESS_SUCCESS,
           "cached owned memory entry is available");
    return reinterpret_cast<Function>(function);
}
void application(const char* driver_path) {
    auto* sdk = ::dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    expect(sdk, "load the separate metadata-only memory Driver fixture");
    const auto copy = cached<decltype(&cuMemcpyDtoD)>(sdk, "cuMemcpyDtoD");
    const auto copy_async = cached<decltype(&cuMemcpyDtoDAsync)>(sdk, "cuMemcpyDtoDAsync");
    const auto fill8 = cached<decltype(&cuMemsetD8)>(sdk, "cuMemsetD8");
    const auto fill8_async = cached<decltype(&cuMemsetD8Async)>(sdk, "cuMemsetD8Async");
    const auto fill16 = cached<decltype(&cuMemsetD16)>(sdk, "cuMemsetD16");
    const auto fill16_async = cached<decltype(&cuMemsetD16Async)>(sdk, "cuMemsetD16Async");
    const auto fill32 = cached<decltype(&cuMemsetD32)>(sdk, "cuMemsetD32");
    const auto fill32_async = cached<decltype(&cuMemsetD32Async)>(sdk, "cuMemsetD32Async");
    CUcontext context{};
    CUdeviceptr source{}, destination{};
    CUstream stream{};
    check(cuCtxCreate(&context, 0, 0));
    check(cuMemAlloc(&source, 256));
    check(cuMemAlloc(&destination, 256));
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    for (const bool is_cached : {false, true}) {
        check((is_cached ? copy : &cuMemcpyDtoD)(destination + 32, source + 16, 16));
        check((is_cached ? copy_async : &cuMemcpyDtoDAsync)(destination + 32, source + 16, 16,
                                                            stream));
        check((is_cached ? fill8 : &cuMemsetD8)(destination, 0xAB, 16));
        check((is_cached ? fill8_async : &cuMemsetD8Async)(destination, 0xCD, 16, stream));
        check((is_cached ? fill16 : &cuMemsetD16)(destination + 64, 0x1234, 4));
        check((is_cached ? fill16_async : &cuMemsetD16Async)(destination + 64, 0x5678, 4, stream));
        check((is_cached ? fill32 : &cuMemsetD32)(destination + 96, 0x12345678, 4));
        check((is_cached ? fill32_async : &cuMemsetD32Async)(destination + 96, 0xABCDEF01, 4,
                                                             stream));
    }
    const auto runtime_copy = runtime_symbol<decltype(&cudaMemcpy)>("cudaMemcpy");
    const auto runtime_copy_async = runtime_symbol<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    const auto runtime_fill = runtime_symbol<decltype(&cudaMemset)>("cudaMemset");
    const auto runtime_fill_async = runtime_symbol<decltype(&cudaMemsetAsync)>("cudaMemsetAsync");
    expect(runtime_copy(runtime_pointer(destination + 32), runtime_pointer(source + 16), 16,
                        cudaMemcpyDeviceToDevice) == cudaSuccess &&
               runtime_copy_async(runtime_pointer(destination + 32), runtime_pointer(source + 16),
                                  16, cudaMemcpyDefault,
                                  reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess &&
               runtime_fill(runtime_pointer(destination), -1, 16) == cudaSuccess &&
               runtime_fill_async(runtime_pointer(destination), 0x1EF, 16,
                                  reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess,
           "Runtime device copies and byte fills use the same owned Driver context without native "
           "caches");
    expect(cuMemsetD32(destination + 1, 1, 1) == CUDA_ERROR_INVALID_VALUE &&
               cuMemsetD32(destination, 1, SIZE_MAX) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpyDtoD(destination + 250, source, 16) == CUDA_ERROR_INVALID_VALUE,
           "alignment, overflow and range reject before IPC/native work");
    check(cuMemcpyDtoDAsync(destination, source, 16, CU_STREAM_PER_THREAD));
    expect(cuMemsetD8(destination, 0xEE, 13) == CUDA_ERROR_INVALID_VALUE,
           "an actual remote SDK failure remains unchanged without replay");
    check(cuMemFree(source));
    check(cuMemFree(destination));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(glimmer_memory_native_calls() == 0,
           "no memory operation falls back into the caller's native Driver");
    expect(::dlclose(sdk) == 0, "balance explicit Driver fixture reference");
}
void run(const std::string& path, const char* driver, bool inject_failure) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start actual authenticated memory endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "memory endpoint identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select exact authenticated owned endpoint");
    auto client = std::async(std::launch::async, [&] { application(driver); });
    // Destroy this before the future: an endpoint assertion must disconnect
    // the waiting client before async future destruction joins its thread.
    const StopServerOnExit stop{server};
    unsigned int allocations{}, streams{}, ptds_calls{}, memory_calls{}, errors{};
    std::map<std::uint64_t, std::size_t> extents;
    const auto deadline = Clock::now() + std::chrono::seconds{6};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline,
               "bounded actual memory endpoint progress");
        auto request = server.take_request();
        if (!request) continue;
        if (inject_failure) throw std::runtime_error("injected endpoint assertion");
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        ExecutionPayloadReader input(request->message.payload);
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && bytes == 256 && input.empty(),
                   "exact allocation request");
            reply.id = ++allocations;
            reply.address = 0x100000 + std::uint64_t{allocations - 1} * 0x10000;
            extents.emplace(reply.id, bytes);
        } else if (operation == ExecutionOpcode::kCreateStream) {
            expect(++streams <= 2, "one explicit stream and one owned PTDS stream");
            reply.id = 2 + streams;
        } else if (operation == ExecutionOpcode::kMemory) {
            ++memory_calls;
            std::uint32_t kind{}, asynchronous{}, value{};
            std::uint64_t destination{}, destination_offset{}, source{}, source_offset{}, count{},
                stream{};
            expect(input.read_u32(kind) && input.read_u32(asynchronous) &&
                       input.read_u64(destination) && input.read_u64(destination_offset) &&
                       input.read_u64(source) && input.read_u64(source_offset) &&
                       input.read_u64(count) && input.read_u32(value) && input.read_u64(stream) &&
                       input.empty(),
                   "fixed-width memory wire fields contain no host struct or source pointer");
            expect(destination == 2 && extents.contains(destination) && asynchronous <= 1 &&
                       (asynchronous ? stream == 3 || stream == 4 : stream == 0),
                   "exact virtual allocation and stream identities");
            if (kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kCopy)) {
                const bool is_ptds = stream == 4;
                ptds_calls += is_ptds;
                expect(source == 1 && source_offset == (is_ptds ? 0U : 16U) &&
                           destination_offset == (is_ptds ? 0U : 32U) && count == 16 && !value,
                       "D2D source/destination offsets and byte count survive IPC");
            } else {
                expect(!source && !source_offset,
                       "fill carries scalar bits, not a source process pointer");
                if (kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kFill16))
                    expect(destination_offset == 64 && count == 4 &&
                               (value == 0x1234 || value == 0x5678),
                           "16-bit fill uses element count and complete scalar");
                else if (kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32))
                    expect(destination_offset == 96 && count == 4 &&
                               (value == 0x12345678 || value == 0xABCDEF01),
                           "32-bit fill preserves full pattern bits");
                else
                    expect(kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kFill8) &&
                               destination_offset == 0 && (count == 16 || count == 13) &&
                               value <= UINT8_MAX,
                           "byte fill preserves truncation and original byte count");
            }
            if (count == 13) {
                reply.status = CUDA_ERROR_INVALID_VALUE;
                reply.original_status = reply.status;
                ++errors;
            }
            reply.id = memory_calls;
        }
        auto header = request->message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request->connection;
        header.incarnation = {83, 89};
        const auto bytes = encode_execution_reply(reply);
        expect(bytes && server.reply(request->connection, {header, *bytes}),
               "deliver actual original memory result");
    }
    client.get();
    expect(memory_calls == 22 && errors == 1 && allocations == 2 && streams == 2 && ptds_calls == 1,
           "all direct/cached Driver and Runtime memory calls execute once, rejected inputs "
           "execute none");
    server.stop();
    std::cout << "Owned memory ABI verified: driver_entries=8; runtime_entries=4; cached=true; "
                 "native_caller_calls=0; gpu_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return EXIT_FAILURE;
    const bool inject_failure = argc == 3 && std::string_view{argv[2]} == "--endpoint-failure";
    if (argc == 3 && !inject_failure) return EXIT_FAILURE;
    char pattern[] = "/tmp/glimmer-memory-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock", argv[1], inject_failure);
        if (inject_failure) result = EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = inject_failure && std::string_view{error.what()} == "injected endpoint assertion"
                     ? EXIT_SUCCESS
                     : EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
