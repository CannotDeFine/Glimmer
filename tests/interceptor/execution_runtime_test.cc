#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"
#include "execution_staging_fixture.h"
#include "execution_async_fixture.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fatbinary_section.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// Exact NVIDIA compiler callback ABI declarations.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** __cudaRegisterFatBinary(void*);
extern "C" void __cudaRegisterFatBinaryEnd(void**);
extern "C" void __cudaUnregisterFatBinary(void**);
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                       dim3*, dim3*, int*);
extern "C" void __cudaRegisterVar(void**, char*, char*, const char*, int, std::size_t, int, int);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudaError_t value) {
    expect(value == cudaSuccess, "public owned Runtime operation succeeds");
}
void exercise() {
    CUcontext driver_primary{};
    expect(cuDevicePrimaryCtxRetain(&driver_primary, 0) == CUDA_SUCCESS,
           "Driver-first primary activates its own cold purpose");
    check(cudaSetDevice(0));
    check(cudaFree(nullptr));
    check(cudaDeviceReset());
    expect(cuDevicePrimaryCtxRelease(0) == CUDA_SUCCESS,
           "Runtime use/reset does not relabel the active Driver-first generation");
    expect(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS,
           "explicitly clear the preserved inactive primary stack before lazy registration");
    std::vector<std::byte> image(std::size_t{17} * 1024 * 1024);
    const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, image.size() - 16};
    std::memcpy(image.data(), prefix.data(), sizeof(prefix));
    image[16] = std::byte{7};
    __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                reinterpret_cast<const unsigned long long*>(image.data()), nullptr};
    auto** registration = __cudaRegisterFatBinary(&wrapper);
    expect(registration, "compiler wrapper receives a stable CPU-only registration identity");
    const int host_identity{}, untouched_kernel_identity{};
    const auto* host = reinterpret_cast<const char*>(&host_identity);
    const auto* untouched_kernel = reinterpret_cast<const char*>(&untouched_kernel_identity);
    char name[] = "binary_add";
    const int global_identity{}, alias_identity{}, missing_identity{}, untouched_identity{};
    auto* global_host = reinterpret_cast<char*>(const_cast<int*>(&global_identity));
    auto* alias_host = reinterpret_cast<char*>(const_cast<int*>(&alias_identity));
    auto* missing_host = reinterpret_cast<char*>(const_cast<int*>(&missing_identity));
    auto* untouched_host = reinterpret_cast<char*>(const_cast<int*>(&untouched_identity));
    alignas(8) std::array<std::byte, 24> small_image{};
    const std::array<std::uint64_t, 2> small_prefix{0x00100001ba55ed50, 8};
    std::memcpy(small_image.data(), small_prefix.data(), sizeof(small_prefix));
    small_image[16] = std::byte{7};
    __fatBinC_Wrapper_t small_wrapper{
        FATBINC_MAGIC, FATBINC_VERSION,
        reinterpret_cast<const unsigned long long*>(small_image.data()), nullptr};
    auto** shadow = __cudaRegisterFatBinary(&small_wrapper);
    expect(shadow, "second compiler module owns its own immutable image");
    std::array<void**, 3> additional{};
    for (auto& handle : additional) {
        handle = __cudaRegisterFatBinary(&wrapper);
        expect(handle, "aggregate immutable compiler snapshots can exceed the old 64-MiB cap");
    }
    for (auto* handle : additional) __cudaUnregisterFatBinary(handle);
    check(cudaGetLastError());
    char shadow_name[] = "shadow_global";
    char shadow_kernel[] = "shadow_add";
    // Callback order is deliberately opposite to image creation order.
    __cudaRegisterVar(shadow, global_host, shadow_name, shadow_name, 0, 16, 0, 1);
    __cudaRegisterVar(shadow, untouched_host, shadow_name, shadow_name, 0, 16, 0, 1);
    __cudaRegisterFunction(shadow, host, shadow_kernel, shadow_kernel, -1, nullptr, nullptr,
                           nullptr, nullptr, nullptr);
    __cudaRegisterFunction(shadow, untouched_kernel, shadow_kernel, shadow_kernel, -1, nullptr,
                           nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFatBinaryEnd(shadow);
    __cudaRegisterFunction(registration, host, name, name, -1, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
    __cudaRegisterFunction(registration, host, shadow_kernel, shadow_kernel, -1, nullptr, nullptr,
                           nullptr, nullptr, nullptr);
    __cudaRegisterFunction(registration, untouched_kernel, name, name, -1, nullptr, nullptr,
                           nullptr, nullptr, nullptr);
    char global_name[] = "global";
    __cudaRegisterVar(registration, global_host, global_name, "display_only_global", 0, 16, 0, 1);
    __cudaRegisterVar(registration, alias_host, global_name, global_name, 1, 0, 0, 1);
    __cudaRegisterVar(registration, missing_host, nullptr, "missing", 0, 16, 0, 1);
    __cudaRegisterVar(registration, untouched_host, global_name, global_name, 0, 16, 0, 1);
    __cudaRegisterVar(registration, global_host, shadow_name, shadow_name, 0, 8, 0, 1);
    expect(cudaGetLastError() == cudaSuccess,
           "coalesced host variables do not poison either compiler image");
    global_name[0] = '!';
    __cudaRegisterFatBinaryEnd(registration);
    image[16] = std::byte{99};
    name[0] = '!';
    CUcontext current{};
    expect(cuCtxGetCurrent(&current) == CUDA_SUCCESS && !current,
           "registration does not create or push any context");
    int device{-1}, active{-1};
    unsigned int flags{};
    check(cudaGetDevice(&device));
    expect(device == 0 && cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS && !active,
           "device metadata is lazy and CPU-only");
    expect(cudaMalloc(reinterpret_cast<void**>(1), 16) == cudaErrorInvalidValue &&
               cudaPeekAtLastError() == cudaErrorInvalidValue &&
               cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess,
           "invalid output and sticky/reset Runtime error semantics");
    for (unsigned int generation = 0; generation < 3; ++generation) {
        CUcontext user_context{};
        if (generation < 2)
            check(cudaSetDevice(0));
        else
            expect(cuCtxCreate(&user_context, 0, 0) == CUDA_SUCCESS,
                   "create independent user context");
        check(cudaFree(nullptr));
        void* pointer{};
        check(cudaMalloc(&pointer, 16));
        if (generation == 2) {
            expect(cudaDeviceReset() == cudaErrorNotSupported &&
                       cudaGetLastError() == cudaErrorNotSupported &&
                       cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == user_context,
                   "unimplemented user Runtime reset cannot destroy Driver-owned context/state");
        }
        std::array<std::uint32_t, 4> data{7, 11, 13, 17};
        check(cudaMemcpy(pointer, data.data(), sizeof(data), cudaMemcpyHostToDevice));
        cudaStream_t stream{};
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::size_t global_bytes{};
        void* global_address{};
        check(cudaGetSymbolSize(&global_bytes, global_host));
        check(cudaGetSymbolAddress(&global_address, alias_host));
        expect(global_bytes == 16 && global_address == reinterpret_cast<void*>(0x200000),
               "registered static symbol and extern alias share owned range metadata");
        expect(cudaFree(global_address) == cudaErrorInvalidValue &&
                   cudaGetLastError() == cudaErrorInvalidValue,
               "module-owned symbols cannot be released as malloc allocations");
        check(
            cudaMemcpyToSymbol(global_host, data.data(), sizeof(data), 0, cudaMemcpyHostToDevice));
        std::array<std::uint32_t, 4> global_values{};
        check(cudaMemcpyFromSymbol(global_values.data(), alias_host, sizeof(global_values), 0,
                                   cudaMemcpyDeviceToHost));
        expect(global_values == data, "synchronous whole-symbol round trip");
        const std::uint32_t replacement = 29;
        check(cudaMemcpyToSymbolAsync(global_host, &replacement, sizeof(replacement), 4,
                                      cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpyFromSymbolAsync(global_values.data(), global_host, sizeof(global_values), 0,
                                        cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        expect(global_values == std::array<std::uint32_t, 4>{7, 29, 13, 17},
               "async symbol copies preserve offset, guard values and selected stream");
        check(cudaMemcpyToSymbol(global_host, pointer, sizeof(data), 0, cudaMemcpyDeviceToDevice));
        check(cudaMemcpyFromSymbolAsync(pointer, global_host, sizeof(data), 0,
                                        cudaMemcpyDeviceToDevice, stream));
        check(cudaStreamSynchronize(stream));
        global_bytes = 99;
        expect(cudaGetSymbolSize(&global_bytes, missing_host) == cudaErrorInvalidSymbol &&
                   global_bytes == 99 && cudaGetLastError() == cudaErrorInvalidSymbol,
               "actual missing native symbol leaves output unchanged with Runtime symbol error");
        expect(cudaMemcpyToSymbol(global_host, data.data(), sizeof(data), SIZE_MAX,
                                  cudaMemcpyHostToDevice) == cudaErrorInvalidValue &&
                   cudaGetLastError() == cudaErrorInvalidValue,
               "overflowing symbol offsets reject without a transfer");
        expect(cudaMemcpyToSymbol(global_host, data.data(), 4, 0, cudaMemcpyDeviceToHost) ==
                       cudaErrorInvalidMemcpyDirection &&
                   cudaGetLastError() == cudaErrorInvalidMemcpyDirection,
               "symbol copy direction has its original Runtime error domain");
        struct alignas(16) Values {
            std::uint32_t add{3}, marker{}, unused[2]{};
        } values;
        std::uint32_t count{4};
        void* arguments[]{&pointer, &count, &values};
        const auto first = cudaLaunchKernel(host, dim3{320}, dim3{32}, arguments, 0, stream);
        if (generation == 0) {
            expect(first == cudaErrorLaunchOutOfResources && static_cast<int>(first) == 701,
                   "pre-admission rejection uses the installed Runtime ABI, not legacy values");
            auto independent = std::async(std::launch::async, [] { return cudaGetLastError(); });
            expect(independent.get() == cudaSuccess && cudaGetLastError() == first &&
                       cudaGetLastError() == cudaSuccess,
                   "last errors are thread-local and consumed exactly once");
            check(cudaLaunchKernel(host, dim3{320}, dim3{32}, arguments, 0, stream));
        } else
            check(first);
        values.add = 99;
        check(cudaStreamQuery(stream));
        check(cudaStreamSynchronize(stream));
        data.fill(0);
        check(cudaMemcpy(data.data(), pointer, sizeof(data), cudaMemcpyDefault));
        expect(data == std::array<std::uint32_t, 4>{7, 11, 13, 17},
               "owned checked copy round trip");
        check(cudaDeviceSynchronize());
        check(cudaFree(pointer));
        check(cudaStreamDestroy(stream));
        if (generation == 2) {
            __cudaUnregisterFatBinary(registration);
            check(cudaGetLastError());
            expect(cudaGetSymbolSize(&global_bytes, global_host) == cudaErrorInvalidSymbol &&
                       cudaGetLastError() == cudaErrorInvalidSymbol,
                   "unregistered symbol identity cannot reuse its retired module binding");
            expect(cudaGetSymbolSize(&global_bytes, untouched_host) == cudaErrorInvalidSymbol &&
                       cudaGetLastError() == cudaErrorInvalidSymbol,
                   "unqueried symbol association also stays invalid until context reset");
            expect(cudaLaunchKernel(host, dim3{1}, dim3{1}, nullptr, 0, nullptr) ==
                       cudaErrorInvalidDeviceFunction,
                   "unregistered host function cannot escape to caller-native launch");
            expect(cudaGetLastError() == cudaErrorInvalidDeviceFunction,
                   "unknown host function Runtime error");
            expect(cudaLaunchKernel(untouched_kernel, dim3{1}, dim3{1}, nullptr, 0, nullptr) ==
                           cudaErrorInvalidDeviceFunction &&
                       cudaGetLastError() == cudaErrorInvalidDeviceFunction,
                   "unqueried coalesced kernel cannot silently retarget a surviving module before "
                   "reset");
        }
        if (generation < 2)
            check(cudaDeviceReset());
        else
            expect(cuCtxDestroy(user_context) == CUDA_SUCCESS, "explicit user context teardown");
    }
    check(cudaSetDevice(0));
    std::size_t surviving_bytes{};
    check(cudaGetSymbolSize(&surviving_bytes, global_host));
    expect(surviving_bytes == 16, "fresh context selects the surviving registered module");
    void* fresh_pointer{};
    cudaStream_t fresh_stream{};
    check(cudaMalloc(&fresh_pointer, 16));
    check(cudaStreamCreateWithFlags(&fresh_stream, cudaStreamNonBlocking));
    struct alignas(16) FreshValue {
        std::uint32_t add{3}, padding[3]{};
    } fresh_value;
    std::uint32_t fresh_count{4};
    void* fresh_arguments[]{&fresh_pointer, &fresh_count, &fresh_value};
    check(cudaLaunchKernel(host, dim3{320}, dim3{32}, fresh_arguments, 0, fresh_stream));
    check(cudaStreamSynchronize(fresh_stream));
    check(cudaFree(fresh_pointer));
    check(cudaStreamDestroy(fresh_stream));
    __cudaUnregisterFatBinary(shadow);
    check(cudaGetLastError());
    check(cudaDeviceReset());
    expect(__cudaRegisterFatBinary(reinterpret_cast<void*>(1)) == nullptr &&
               cudaGetLastError() == cudaErrorInvalidKernelImage,
           "unreadable compiler wrapper rejects without dereference or IPC");
    wrapper.version = FATBINC_LINK_VERSION;
    const void* prelinked[] = {image.data(), nullptr};
    wrapper.filename_or_fatbins = prelinked;
    auto** linked = __cudaRegisterFatBinary(&wrapper);
    expect(linked && cudaGetLastError() == cudaSuccess,
           "compiler v2 wrapper owns its final linked image without native registration");
    __cudaRegisterFatBinaryEnd(linked);
    __cudaUnregisterFatBinary(linked);
    check(cudaGetLastError());
    const auto* final_image = wrapper.data;
    wrapper.data = nullptr;
    expect(!__cudaRegisterFatBinary(&wrapper) && cudaGetLastError() == cudaErrorInvalidKernelImage,
           "prelink array alone is not an executable final image or JIT-link implementation");
    wrapper.data = final_image;
    wrapper.version = FATBINC_LINK_VERSION + 1;
    expect(!__cudaRegisterFatBinary(&wrapper) && cudaGetLastError() == cudaErrorInvalidKernelImage,
           "unsupported compiler wrapper version never reaches native registration");
    wrapper.version = FATBINC_VERSION;
    wrapper.filename_or_fatbins = nullptr;
    const auto page_bytes = ::sysconf(_SC_PAGESIZE);
    expect(page_bytes >= 32, "known page geometry for checked Runtime snapshots");
    const auto mapped_bytes = static_cast<std::size_t>(page_bytes) * 2;
    auto* mapping = static_cast<std::byte*>(
        ::mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    expect(mapping != MAP_FAILED, "private read-protected wrapper fixture");
    auto* boundary = mapping + page_bytes;
    std::memcpy(boundary - 16, prefix.data(), 16);
    wrapper.data = reinterpret_cast<const unsigned long long*>(boundary - 16);
    const auto protected_status =
        ::mprotect(boundary, static_cast<std::size_t>(page_bytes), PROT_NONE);
    auto** unreadable = protected_status == 0 ? __cudaRegisterFatBinary(&wrapper) : registration;
    const auto unmapped = ::munmap(mapping, mapped_bytes);
    expect(protected_status == 0 && unmapped == 0 && !unreadable &&
               cudaGetLastError() == cudaErrorInvalidKernelImage,
           "header cannot authorize a snapshot extending into an unreadable page");
    wrapper.data = reinterpret_cast<const unsigned long long*>(image.data());
    auto** conflict = __cudaRegisterFatBinary(&wrapper);
    auto** other = __cudaRegisterFatBinary(&wrapper);
    expect(conflict && other && conflict != other, "independent bounded registration owners");
    char first_name[] = "first", second_name[] = "second";
    __cudaRegisterFunction(conflict, host, first_name, first_name, -1, nullptr, nullptr, nullptr,
                           nullptr, nullptr);
    __cudaRegisterFunction(other, host, second_name, second_name, -1, nullptr, nullptr, nullptr,
                           nullptr, nullptr);
    expect(cudaGetLastError() == cudaSuccess,
           "coalesced host functions retain independent module metadata instead of poisoning both");
    __cudaUnregisterFatBinary(conflict);
    __cudaUnregisterFatBinary(other);
    check(cudaGetLastError());
    auto** unsupported = __cudaRegisterFatBinary(&wrapper);
    expect(unsupported && unsupported != registration, "registration identities never recycle");
    __cudaRegisterVar(unsupported, nullptr, nullptr, "global", 0, 4, 0, 1);
    expect(cudaGetLastError() == cudaErrorInvalidValue,
           "invalid ordinary variable registration is rejected before a host-pointer dereference");
    __cudaUnregisterFatBinary(unsupported);
    check(cudaGetLastError());
    const auto child = ::fork();
    expect(child >= 0, "fork Runtime ownership fixture");
    if (child == 0) {
        void* pointer{};
        _exit(cudaMalloc(&pointer, 4) == cudaErrorIncompatibleDriverContext ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               glimmer_execution_lookup_count(2) == 0,
           "fork and Runtime routes never create caller-native resources");
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated Runtime endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual Runtime owner identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure before public registration");
    auto client = std::async(std::launch::async, exercise);
    std::array<std::byte, 16> memory{};
    std::array<std::byte, 16> globals{};
    ExecutionStagingFixture staging(memory, 31);
    ExecutionAsyncFixture asynchronous;
    staging.add_device(32, globals);
    staging.allow_stream(21);
    unsigned int hellos{}, closes{}, loads{}, functions{}, launches{}, unloads{};
    const std::array expected_bootstraps{
        ExecutionContextBootstrap{ExecutionContextPurpose::kProcessPrimary, CU_CTX_MAP_HOST, 1},
        ExecutionContextBootstrap{ExecutionContextPurpose::kRuntimePrimary, CU_CTX_MAP_HOST, 3},
        ExecutionContextBootstrap{ExecutionContextPurpose::kRuntimePrimary, CU_CTX_MAP_HOST, 5},
        ExecutionContextBootstrap{ExecutionContextPurpose::kExplicitUser, 0, 1},
        ExecutionContextBootstrap{ExecutionContextPurpose::kRuntimePrimary, CU_CTX_MAP_HOST, 7}};
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress actual Runtime transport");
        if (const auto request = server.take_request()) {
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
                        expect(hellos < expected_bootstraps.size() &&
                                   decode_execution_context_bootstrap(request->message.payload) ==
                                       expected_bootstraps[hellos],
                               "Runtime cold activation alone declares Runtime-primary purpose; "
                               "Driver-first and explicit-user contexts never relabel");
                        ++hellos;
                        break;
                    case ExecutionOpcode::kClose:
                        asynchronous.close(request->connection);
                        ++closes;
                        break;
                    case ExecutionOpcode::kLoadBinary:
                        globals.fill(std::byte{});
                        expect(input.read_blob(data) && data.size() == 24 &&
                                   data[16] == std::byte{7} && input.empty(),
                               "registration owns the pre-mutation raw fatbinary bytes");
                        reply.id = 11;
                        ++loads;
                        break;
                    case ExecutionOpcode::kLoadModuleFile: {
                        std::uint32_t kind{};
                        struct stat info {};
                        const auto descriptor = request->descriptor.get();
                        constexpr auto required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK;
                        expect(input.read_u32(kind) &&
                                   kind == static_cast<std::uint32_t>(
                                               ExecutionModuleImageKind::kBinary) &&
                                   input.read_u64(count) &&
                                   count == std::size_t{17} * 1024 * 1024 && input.empty() &&
                                   descriptor >= 0 && ::fstat(descriptor, &info) == 0 &&
                                   static_cast<std::uint64_t>(info.st_size) == count &&
                                   (::fcntl(descriptor, F_GET_SEALS) & required_seals) ==
                                       required_seals,
                               "large registered image crosses a content-sealed exact-size "
                               "capability");
                        const std::byte changed{99};
                        expect(::pwrite(descriptor, &changed, 1, 16) < 0 && errno == EPERM &&
                                   ::ftruncate(descriptor, 0) < 0 && errno == EPERM,
                               "neither source bytes nor extent can mutate after transport");
                        auto* view = static_cast<const std::byte*>(
                            ::mmap(nullptr, count, PROT_READ, MAP_PRIVATE, descriptor, 0));
                        expect(view != MAP_FAILED && view[16] == std::byte{7},
                               "large intake owns the original pre-mutation compiler snapshot");
                        expect(
                            ::munmap(const_cast<std::byte*>(view), count) == 0,
                            "temporary owner view retires independently of the request capability");
                        globals.fill(std::byte{});
                        reply.id = 11;
                        ++loads;
                        break;
                    }
                    case ExecutionOpcode::kGetGlobal: {
                        std::string_view symbol;
                        expect(input.read_u64(id) && id == 11 && input.read_string(symbol) &&
                                   input.empty(),
                               "global query carries owned module ID and copied symbol name");
                        if (symbol == "missing") {
                            reply.status = CUDA_ERROR_NOT_FOUND;
                            reply.original_status = reply.status;
                        } else {
                            expect(symbol == "global" || symbol == "shadow_global",
                                   "registration snapshots original symbol name");
                            reply.id = 32;
                            reply.address = 0x200000;
                            reply.parameter_bytes = globals.size();
                        }
                        break;
                    }
                    case ExecutionOpcode::kMemory: {
                        std::uint32_t kind{}, asynchronous{}, fill{};
                        std::uint64_t to{}, to_offset{}, from{}, from_offset{}, stream{};
                        expect(input.read_u32(kind) && !kind && input.read_u32(asynchronous) &&
                                   asynchronous <= 1 && input.read_u64(to) &&
                                   input.read_u64(to_offset) && input.read_u64(from) &&
                                   input.read_u64(from_offset) && input.read_u64(count) &&
                                   input.read_u32(fill) && !fill && input.read_u64(stream) &&
                                   (!stream || stream == 21) && input.empty() &&
                                   (to == 31 || to == 32) && (from == 31 || from == 32) &&
                                   to_offset <= 16 && count <= 16 - to_offset &&
                                   from_offset <= 16 && count <= 16 - from_offset,
                               "device symbol copy retains both owned ranges and stream");
                        std::memmove((to == 31 ? memory.data() : globals.data()) + to_offset,
                                     (from == 31 ? memory.data() : globals.data()) + from_offset,
                                     count);
                        reply.id = 1;
                        break;
                    }
                    case ExecutionOpcode::kGetFunction: {
                        std::string_view symbol;
                        expect(input.read_u64(id) && id == 11 && input.read_string(symbol) &&
                                   symbol == (functions < 3 ? "binary_add" : "shadow_add") &&
                                   input.empty(),
                               "copied device name and correct module");
                        reply.id = 12;
                        reply.parameter_bytes = 32;
                        reply.arguments = {{0, 8}, {8, 4}, {16, 16}};
                        ++functions;
                        break;
                    }
                    case ExecutionOpcode::kAllocate:
                        reply.id = 31;
                        reply.address = 0x100000;
                        break;
                    case ExecutionOpcode::kCreateStream:
                        reply.id = 21;
                        break;
                    case ExecutionOpcode::kUpload:
                        expect(input.read_u64(id) && id == 31 && input.read_u64(offset) &&
                                   !offset && input.read_blob(data) &&
                                   data.size() == memory.size() && input.empty(),
                               "owned Runtime upload");
                        std::copy(data.begin(), data.end(), memory.begin());
                        break;
                    case ExecutionOpcode::kDownload:
                        expect(input.read_u64(id) && id == 31 && input.read_u64(offset) &&
                                   !offset && input.read_u64(count) && count == memory.size() &&
                                   input.empty(),
                               "owned Runtime download");
                        reply.data.assign(memory.begin(), memory.end());
                        break;
                    case ExecutionOpcode::kLaunch: {
                        expect(input.read_u64(id) && id == 12 && input.read_u64(id) && id == 21,
                               "owned Runtime kernel and stream");
                        std::uint32_t dimension{};
                        for (unsigned int index = 0; index < 7; ++index)
                            expect(input.read_u32(dimension), "Runtime launch geometry");
                        expect(input.read_blob(data) && data.size() == 32,
                               "original Runtime packed ABI");
                        ExecutionAsyncFixture::tail(input);
                        std::uint32_t value{};
                        std::memcpy(&value, data.data() + 16, sizeof(value));
                        expect(value == 3, "by-value arguments are snapshotted before return");
                        if (!launches++) {
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                            reply.original_status.reset();
                        } else {
                            reply = asynchronous.admit(request->connection, 21);
                            asynchronous.complete(request->connection, reply.id, CUDA_SUCCESS);
                        }
                        break;
                    }
                    case ExecutionOpcode::kUnloadModule:
                        ++unloads;
                        break;
                    case ExecutionOpcode::kCaptureFrontier: {
                        std::uint32_t scope{};
                        expect(input.read_u32(scope) && scope <= 2 && input.read_u64(id) &&
                                   input.empty(),
                               "exact Runtime context/stream capture");
                        reply = asynchronous.capture(request->connection,
                                                     scope ? std::optional{id} : std::nullopt);
                        break;
                    }
                    case ExecutionOpcode::kQueryFrontier:
                    case ExecutionOpcode::kReleaseFrontier:
                    case ExecutionOpcode::kQueryAsyncFrontiers:
                    case ExecutionOpcode::kReleaseAsyncFrontiers:
                        reply = asynchronous.observe(*request);
                        break;
                    case ExecutionOpcode::kRelease:
                    case ExecutionOpcode::kDestroyStream:
                        break;
                    default:
                        throw std::runtime_error("unexpected Runtime target operation");
                }
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {59, 61};
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && server.reply(request->connection, {header, *encoded}),
                   "publish owned Runtime result");
        }
        expect(Clock::now() < deadline, "bounded Runtime ownership test");
    }
    client.get();
    expect(staging.empty(), "Runtime reset settles exact staging registrations");
    expect(asynchronous.empty(), "Runtime reset settles exact admitted command frontiers");
    expect(
        hellos == 5 && closes == 5 && loads == 4 && functions == 4 && launches == 5 && unloads == 2,
        "exact per-generation resources, cached function identity and once-only unregister");
    server.stop();
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-runtime-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    std::jthread watchdog([](const std::stop_token& stop) {
        const auto deadline = Clock::now() + std::chrono::seconds{12};
        while (!stop.stop_requested()) {
            if (Clock::now() >= deadline) _exit(98);
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    });
    int result{};
    try {
        run(std::string(directory) + "/service.sock");
        std::cout << "Public Runtime registration/ownership verified: generations=3; "
                     "native_resources=0\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    watchdog.request_stop();
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
