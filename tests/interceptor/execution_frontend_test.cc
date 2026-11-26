#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"
#include "execution_staging_fixture.h"
#include "execution_async_fixture.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

namespace argument_read_fixture {
thread_local bool captures{};
thread_local std::size_t calls{};
std::atomic<std::size_t> ordinary_calls{}, wide_calls{};
}  // namespace argument_read_fixture

extern "C" ssize_t process_vm_readv(pid_t pid, const iovec* local, unsigned long local_count,
                                    const iovec* remote, unsigned long remote_count,
                                    unsigned long flags) noexcept {
    if (argument_read_fixture::captures) ++argument_read_fixture::calls;
    return static_cast<ssize_t>(
        ::syscall(SYS_process_vm_readv, pid, local, local_count, remote, remote_count, flags));
}

namespace {
using namespace glimmer::control;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "public remote Driver call succeeds");
}
void exercise() {
    CUcontext context{};
    expect(cuCtxCreate(reinterpret_cast<CUcontext*>(1), 0, 0) == CUDA_ERROR_INVALID_VALUE,
           "invalid output fails before remote creation");
    expect(cuCtxGetCurrent(&context) == CUDA_ERROR_NOT_INITIALIZED && !context,
           "without an initialized SDK or owned context, the original initialization error wins");
    check(cuCtxCreate(&context, 0, 0));
    const auto original_context = context;
    auto other_thread = std::async(std::launch::async, [] {
        CUcontext observed = reinterpret_cast<CUcontext>(1);
        check(cuCtxGetCurrent(&observed));
        expect(!observed, "current context belongs to the calling thread");
    });
    other_thread.get();
    check(cuCtxCreate(&context, 0, 0));
    check(cuCtxDestroy(context));
    check(cuCtxGetCurrent(&context));
    expect(context == original_context, "destroying current context restores its predecessor");
    check(cuCtxPushCurrent(original_context));
    check(cuCtxPopCurrent(&context));
    expect(context == original_context, "public push/pop preserve the exact context stack");
    check(cuCtxSetCurrent(nullptr));
    check(cuCtxGetCurrent(&context));
    expect(!context, "setting NULL pops the current stack entry");
    check(cuCtxSetCurrent(original_context));
    CUmodule module{};
    expect(cuModuleLoadData(&module, reinterpret_cast<const void*>(1)) == CUDA_ERROR_INVALID_IMAGE,
           "invalid PTX host address is not dereferenced or submitted");
    std::array<std::byte, 64> malformed{};
    std::memcpy(malformed.data(), "\177ELF", 4);
    expect(cuModuleLoadData(&module, malformed.data()) == CUDA_ERROR_INVALID_IMAGE,
           "malformed binary tables cannot submit a remote module creation");
    const auto page_bytes = ::sysconf(_SC_PAGESIZE);
    expect(page_bytes >= 64, "obtain host page geometry");
    const auto mapped_bytes = static_cast<std::size_t>(page_bytes) * 2;
    void* mapping =
        ::mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(mapping != MAP_FAILED, "create bounded binary read fixture");
    auto* boundary = static_cast<std::byte*>(mapping) + page_bytes;
    std::array<std::uint64_t, 2> header{0x00100001ba55ed50, 64};
    std::memcpy(boundary - sizeof(header), header.data(), sizeof(header));
    const auto protection = ::mprotect(boundary, static_cast<std::size_t>(page_bytes), PROT_NONE);
    const auto unreadable =
        protection == 0 ? cuModuleLoadData(&module, boundary - sizeof(header)) : CUDA_ERROR_UNKNOWN;
    const auto unmapped = ::munmap(mapping, mapped_bytes);
    expect(protection == 0 && unmapped == 0 && unreadable == CUDA_ERROR_INVALID_IMAGE,
           "binary prefix cannot authorize a short snapshot across an unreadable host page");
    check(cuModuleLoadData(&module, "owned PTX fixture"));
    CUfunction function{};
    check(cuModuleGetFunction(&function, module, "kernel"));
    CUfunction repeated_function{};
    check(cuModuleGetFunction(&repeated_function, module, "kernel"));
    expect(repeated_function == function,
           "repeated SDK function identity keeps one owned opaque handle");
    CUdeviceptr global{};
    std::size_t global_bytes{};
    check(cuModuleGetGlobal(&global, &global_bytes, module, "global"));
    expect(global == 0x300000 && global_bytes == 8 && cuMemFree(global) == CUDA_ERROR_INVALID_VALUE,
           "public Driver global metadata is module-owned, not an allocation-free target");
    global_bytes = 99;
    expect(cuModuleGetGlobal(nullptr, &global_bytes, module, "missing") == CUDA_ERROR_NOT_FOUND &&
               global_bytes == 99,
           "original Driver symbol error leaves optional outputs unchanged");
    CUdeviceptr pointer{};
    check(cuMemAlloc(&pointer, 16));
    expect(pointer == 0x100000, "device address is the validated server allocation");
    CUdeviceptr adjacent{};
    check(cuMemAlloc(&adjacent, 16));
    expect(adjacent == pointer + 16, "adjacent nonoverlapping server allocations are permitted");
    check(cuMemFree(adjacent));
    std::uint32_t input = 7, output{};
    check(cuMemcpyHtoD(pointer + 4, &input, sizeof(input)));
    check(cuMemcpyDtoH(&output, pointer + 4, sizeof(output)));
    expect(output == input, "checked range and copied host bytes round trip");
    expect(cuMemcpyHtoD(pointer + 15, &input, sizeof(input)) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpyHtoD(pointer, reinterpret_cast<const void*>(1), sizeof(input)) ==
                   CUDA_ERROR_INVALID_VALUE,
           "out-of-range and unreadable copy inputs never reach the server");
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cuStreamQuery(stream));
    check(cuStreamSynchronize(stream));
    check(cuStreamQuery(CU_STREAM_LEGACY));
    check(cuStreamQuery(CU_STREAM_PER_THREAD));
    check(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    unsigned int default_flags = 99;
    check(cuStreamGetFlags(CU_STREAM_PER_THREAD, &default_flags));
    expect(!default_flags &&
               cuStreamSynchronize(reinterpret_cast<CUstream>(3)) == CUDA_ERROR_INVALID_HANDLE,
           "owned per-thread proxy preserves its stream while unknown handles reject");
    void* arguments[]{&pointer, &input};
    argument_read_fixture::calls = 0;
    argument_read_fixture::captures = true;
    const auto copied =
        cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream, arguments, nullptr);
    argument_read_fixture::captures = false;
    argument_read_fixture::ordinary_calls.store(argument_read_fixture::calls);
    check(copied);
    input = 99;
    void* argument_pages =
        ::mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(argument_pages != MAP_FAILED, "map real partially readable argument storage");
    auto* argument_boundary = static_cast<std::byte*>(argument_pages) + page_bytes;
    void* first_argument = &pointer;
    std::memcpy(argument_boundary - sizeof(first_argument), &first_argument,
                sizeof(first_argument));
    const auto protected_arguments =
        ::mprotect(argument_boundary, static_cast<std::size_t>(page_bytes), PROT_NONE);
    void* partial_values[]{&pointer, argument_boundary};
    const auto partial_value =
        protected_arguments == 0
            ? cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream, partial_values, nullptr)
            : CUDA_ERROR_UNKNOWN;
    const auto partial_slots =
        protected_arguments == 0
            ? cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream,
                             reinterpret_cast<void**>(argument_boundary - sizeof(void*)), nullptr)
            : CUDA_ERROR_UNKNOWN;
    const auto unmapped_arguments = ::munmap(argument_pages, mapped_bytes);
    expect(protected_arguments == 0 && unmapped_arguments == 0 &&
               partial_value == CUDA_ERROR_INVALID_VALUE &&
               partial_slots == CUDA_ERROR_INVALID_VALUE,
           "short pointer/value snapshots reject before admission and preserve a healthy context");
    CUfunction wide{};
    check(cuModuleGetFunction(&wide, module, "many"));
    std::array<std::uint32_t, 65> values{};
    std::array<void*, 65> addresses{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<std::uint32_t>(index + 100);
        addresses[index] = &values[index];
    }
    argument_read_fixture::calls = 0;
    argument_read_fixture::captures = true;
    const auto gathered =
        cuLaunchKernel(wide, 320, 1, 1, 32, 1, 1, 0, stream, addresses.data(), nullptr);
    argument_read_fixture::captures = false;
    argument_read_fixture::wide_calls.store(argument_read_fixture::calls);
    check(gathered);
    values.fill(0);
    // Deliberate malformed application pointer for the overflow rejection test.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* overflowing_arguments = reinterpret_cast<void**>(UINTPTR_MAX - 1);
    expect(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, overflowing_arguments, nullptr) ==
               CUDA_ERROR_INVALID_VALUE,
           "overflowing application argument pointer rejects without undefined arithmetic");
    std::array<std::byte, 12> packed{};
    std::memcpy(packed.data(), &pointer, sizeof(pointer));
    std::memcpy(packed.data() + 8, &input, sizeof(input));
    std::size_t packed_size = packed.size();
    void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(), CU_LAUNCH_PARAM_BUFFER_SIZE,
                  &packed_size, CU_LAUNCH_PARAM_END};
    check(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream, nullptr, extra));
    expect(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream, nullptr, extra) ==
               CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
           "pre-admission resource rejection preserves its Driver status without retry");
    std::fill(packed.begin(), packed.end(), std::byte{0});
    const auto child = ::fork();
    expect(child >= 0, "fork inherited frontend fixture");
    if (child == 0) _exit(cuCtxGetCurrent(&context) == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork child rejects before inherited frontend locks");
    check(cuCtxSynchronize());
    check(cuMemFree(pointer));
    expect(cuMemFree(pointer) == CUDA_ERROR_INVALID_VALUE, "stale allocation cannot free twice");
    check(cuStreamDestroy(stream));
    expect(cuStreamQuery(stream) == CUDA_ERROR_INVALID_HANDLE,
           "stale stream cannot mint a new frontier");
    check(cuModuleUnload(module));
    expect(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) ==
               CUDA_ERROR_INVALID_HANDLE,
           "module release invalidates its function handles");
    check(cuCtxDestroy(original_context));
    expect(cuCtxSetCurrent(original_context) == CUDA_ERROR_INVALID_CONTEXT,
           "dead context tokens cannot be recycled");
    expect(cuCtxCreate(&context, 0, 0) == CUDA_ERROR_UNKNOWN,
           "ambiguous native creation returns uncertainty, not fabricated SDK success");
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start injectable authenticated endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted service process identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "set explicit process-local frontend selection before first use");
    auto client = std::async(std::launch::async, exercise);
    std::array<std::byte, 16> memory{};
    ExecutionStagingFixture staging(memory, 3);
    ExecutionAsyncFixture asynchronous;
    unsigned int launches{}, hellos{}, closes{}, allocations{}, streams{}, wide_launches{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress actual authenticated frontend traffic");
        if (auto request = server.take_request()) {
            auto header = request->message.header;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = 0;
            std::uint64_t id{}, offset{}, count{};
            std::span<const std::byte> data;
            const auto staged = staging.handle(*request);
            if (staged)
                reply = *staged;
            else
                switch (header.operation) {
                    case ExecutionOpcode::kHello:
                        ++hellos;
                        if (hellos == 3) {
                            server.close_peer(request->connection);
                            continue;
                        }
                        break;
                    case ExecutionOpcode::kClose:
                        asynchronous.close(request->connection);
                        ++closes;
                        break;
                    case ExecutionOpcode::kLoadPtx:
                        reply.id = 1;
                        break;
                    case ExecutionOpcode::kGetFunction: {
                        std::string_view name;
                        expect(input.read_u64(id) && id == 1 && input.read_string(name) &&
                                   input.empty(),
                               "exact original function lookup");
                        if (name == "many") {
                            reply.id = 9;
                            reply.parameter_bytes = std::uint64_t{65} * 4;
                            for (std::uint64_t index = 0; index < 65; ++index)
                                reply.arguments.push_back({index * 4, 4});
                        } else {
                            expect(name == "kernel", "original two-argument kernel");
                            reply.id = 2;
                            reply.parameter_bytes = 12;
                            reply.arguments = {{0, 8}, {8, 4}};
                        }
                        break;
                    }
                    case ExecutionOpcode::kGetGlobal: {
                        std::string_view name;
                        expect(input.read_u64(id) && id == 1 && input.read_string(name) &&
                                   input.empty(),
                               "owned Driver global lookup module and copied name");
                        if (name == "missing") {
                            reply.status = CUDA_ERROR_NOT_FOUND;
                            reply.original_status = reply.status;
                        } else {
                            expect(name == "global", "Driver global symbol name");
                            reply.id = 8;
                            reply.address = 0x300000;
                            reply.parameter_bytes = 8;
                        }
                        break;
                    }
                    case ExecutionOpcode::kAllocate:
                        reply.id = 3 + allocations * 2;
                        reply.address = 0x100000 + allocations * 16;
                        ++allocations;
                        break;
                    case ExecutionOpcode::kCreateStream: {
                        std::uint32_t flags{};
                        expect(
                            input.read_u32(flags) && flags == (streams ? 0U : 1U) && input.empty(),
                            "explicit stream and distinct blocking per-thread proxy factories");
                        reply.id = 4 + streams++ * 2;
                        break;
                    }
                    case ExecutionOpcode::kUpload:
                        expect(input.read_u64(id) && id == 3 && input.read_u64(offset) &&
                                   offset == 4 && input.read_blob(data) && data.size() == 4 &&
                                   input.empty(),
                               "exact owned upload");
                        std::copy(data.begin(), data.end(), memory.begin() + offset);
                        break;
                    case ExecutionOpcode::kDownload:
                        expect(input.read_u64(id) && id == 3 && input.read_u64(offset) &&
                                   offset == 4 && input.read_u64(count) && count == 4 &&
                                   input.empty(),
                               "exact download range");
                        reply.data.assign(memory.begin() + offset, memory.begin() + offset + count);
                        break;
                    case ExecutionOpcode::kLaunch: {
                        std::uint64_t kernel{};
                        expect(input.read_u64(kernel) && (kernel == 2 || kernel == 9) &&
                                   input.read_u64(id) && id == 4,
                               "virtual handles map to the correct session resources");
                        std::uint32_t dimension{};
                        for (unsigned int index = 0; index < 7; ++index)
                            expect(input.read_u32(dimension), "owned geometry");
                        expect(input.read_blob(data) && data.size() == (kernel == 2 ? 12U : 260U),
                               "owned ABI bytes");
                        ExecutionAsyncFixture::tail(input);
                        if (kernel == 9) {
                            for (std::size_t index = 0; index < 65; ++index) {
                                std::uint32_t value{};
                                std::memcpy(&value, data.data() + index * 4, 4);
                                expect(value == index + 100,
                                       "owned values cross the bounded gather chunk");
                            }
                            reply = asynchronous.admit(request->connection, 4);
                            asynchronous.complete(request->connection, reply.id, CUDA_SUCCESS);
                            ++wide_launches;
                            break;
                        }
                        std::uint32_t value{};
                        std::memcpy(&value, data.data() + 8, sizeof(value));
                        expect(value == (launches == 0 ? 7U : 99U),
                               "pointer-array and packed bytes copied exactly");
                        if (launches == 2) {
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                            reply.original_status.reset();
                        } else {
                            reply = asynchronous.admit(request->connection, 4);
                            asynchronous.complete(request->connection, reply.id, CUDA_SUCCESS);
                        }
                        ++launches;
                        break;
                    }
                    case ExecutionOpcode::kCaptureFrontier: {
                        std::uint32_t kind{};
                        expect(input.read_u32(kind) && kind <= 2 && input.read_u64(id) &&
                                   input.empty() && (id == 0 || id == 4 || id == 6),
                               "completion captures the exact legacy/explicit/per-thread stream "
                               "identity");
                        reply = asynchronous.capture(request->connection,
                                                     kind ? std::optional{id} : std::nullopt);
                        break;
                    }
                    case ExecutionOpcode::kQueryFrontier:
                    case ExecutionOpcode::kReleaseFrontier:
                    case ExecutionOpcode::kQueryAsyncFrontiers:
                    case ExecutionOpcode::kReleaseAsyncFrontiers:
                        reply = asynchronous.observe(*request);
                        break;
                    default:
                        break;
                }
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            auto encoded = encode_execution_reply(reply);
            expect(encoded && server.reply(request->connection, {header, std::move(*encoded)}),
                   "publish actual test SDK reply");
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded public frontend fixture");
    }
    client.get();
    expect(staging.empty(), "all private staged copies retire before origin teardown");
    expect(asynchronous.empty(), "all admitted command frontiers retire before origin teardown");
    expect(hellos == 3 && closes == 2 && allocations == 2 && launches == 3 && wide_launches == 1,
           "invalid, stale, forked and overflow requests issue no duplicate remote work");
    server.stop();
    std::cout << "Checked launch reads: two=" << argument_read_fixture::ordinary_calls.load()
              << ", wide=" << argument_read_fixture::wide_calls.load() << '\n';
    expect(argument_read_fixture::ordinary_calls.load() == 2 &&
               argument_read_fixture::wide_calls.load() >= 2 &&
               argument_read_fixture::wide_calls.load() <= 4,
           "checked argument snapshots use bounded vector reads after complete original cleanup");
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-exec-front-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
