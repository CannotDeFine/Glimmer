#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_memory_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned public host operation succeeds");
}
void* address(std::uint64_t value) {
    // Explicit synthetic SDK UVA, claimed only using MAP_FIXED_NOREPLACE.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value));
}
struct Pipe {
    ExecutionOwnedDescriptor read, write;
    Pipe() {
        std::array<int, 2> descriptors{};
        expect(::pipe2(descriptors.data(), O_CLOEXEC) == 0, "create bounded fixture control pipe");
        read = ExecutionOwnedDescriptor(descriptors[0]);
        write = ExecutionOwnedDescriptor(descriptors[1]);
    }
};
void send(int descriptor, char value) {
    expect(::write(descriptor, &value, 1) == 1, "send fixture latch");
}
char receive(int descriptor) {
    char value{};
    expect(::read(descriptor, &value, 1) == 1, "receive fixture latch");
    return value;
}
void reply(ExecutionSocketServer& endpoint, const ExecutionReceivedRequest& request,
           const ExecutionWireReply& response) {
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = request.connection;
    header.incarnation = {101, 103};
    auto payload = encode_execution_reply(response);
    expect(payload && endpoint.reply(request.connection, {header, std::move(*payload)}),
           "exact authenticated reply identity");
}
struct Region {
    void* pointer{};
    std::size_t extent{}, bytes{};
};
struct Latches {
    int events{}, commands{};
};
enum class Scenario { kNormal, kQuarantine, kRecycling };
void server(const std::string& path, std::uint64_t collision, Latches latches, Scenario scenario) {
    const bool is_quarantine = scenario == Scenario::kQuarantine;
    const bool is_recycling = scenario == Scenario::kRecycling;
    const auto events = latches.events, commands = latches.commands;
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer endpoint(config);
    expect(endpoint.start(), "start actual cross-process CPU host endpoint");
    send(events, 'R');
    std::map<std::uint64_t, Region> regions;
    std::map<std::uint64_t, unsigned int> contexts;
    std::map<std::uint64_t, bool> host_roles;
    ExecutionAsyncFixture asynchronous;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> pending_kernel;
    std::optional<ExecutionReceivedRequest> pending_kernel_release;
    std::optional<ExecutionReceivedRequest> pending_import;
    std::optional<ExecutionWireReply> pending_import_reply;
    std::optional<ExecutionReceivedRequest> pending_release;
    std::optional<ExecutionWireReply> pending_release_reply;
    std::optional<std::uint64_t> reused_address;
    std::vector<std::byte> device(256);
    unsigned int imports{}, releases{}, launches{}, ordinary_contexts{};
    const auto retire_host = [&](ExecutionReceivedRequest request) {
        ExecutionPayloadReader input(request.message.payload);
        std::uint64_t id{};
        expect(!pending_kernel && host_roles.at(request.connection) && input.read_u64(id) &&
                   input.empty(),
               "actual host unregistration waits the accepted physical command, not its RPC");
        const auto found = regions.find(id);
        expect(found != regions.end(), "native release names a live imported range");
        if (id == 260 && scenario == Scenario::kNormal)
            reused_address = reinterpret_cast<std::uintptr_t>(found->second.pointer);
        expect(::munmap(found->second.pointer, found->second.extent) == 0,
               "release exact shared range once");
        regions.erase(found);
        ++releases;
        ExecutionWireReply response;
        response.disposition = ExecutionWireDisposition::kComplete;
        response.original_status = 0;
        if (id == 260 && scenario == Scenario::kNormal) {
            pending_release = std::move(request);
            pending_release_reply = std::move(response);
            send(events, 'F');
        } else {
            reply(endpoint, request, response);
        }
    };
    const auto deadline = Clock::now() + std::chrono::seconds(is_recycling ? 120 : 12);
    for (;;) {
        expect(Clock::now() < deadline && endpoint.poll_for(1), "bounded fake endpoint progress");
        pollfd control{commands, POLLIN, 0};
        const auto ready = ::poll(&control, 1, 0);
        expect(ready >= 0, "poll exact fixture latch");
        if (ready && (control.revents & POLLIN)) {
            const auto command = receive(commands);
            if (command == 'Q') {
                expect(
                    !pending_kernel && !pending_kernel_release && !pending_import &&
                        !pending_release && asynchronous.empty() &&
                        (is_quarantine
                             ? imports == 1 && releases == 0 && regions.size() == 1
                             : regions.empty() && imports >= (is_recycling ? 17002U : 260U) &&
                                   releases == imports - 1 && launches == (is_recycling ? 0U : 1U)),
                    "every successful import and unpublished rollback releases exactly once");
                return;
            }
            if (command == 'D') {
                expect(pending_import && pending_import_reply,
                       "release exactly one delayed host factory result");
                reply(endpoint, *pending_import, *pending_import_reply);
                pending_import.reset();
                pending_import_reply.reset();
                continue;
            }
            if (command == 'F') {
                expect(pending_release && pending_release_reply,
                       "release exactly one acknowledged native free");
                reply(endpoint, *pending_release, *pending_release_reply);
                pending_release.reset();
                pending_release_reply.reset();
                continue;
            }
            expect(command == 'C' && pending_kernel.has_value(),
                   "release only the accepted command's independently held physical tail");
            asynchronous.complete(pending_kernel->first, pending_kernel->second, CUDA_SUCCESS);
            pending_kernel.reset();
            expect(pending_kernel_release.has_value(),
                   "the actual host-release request arrived while its physical owner was held");
            retire_host(std::move(*pending_kernel_release));
            pending_kernel_release.reset();
        }
        auto request = endpoint.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply response;
        response.disposition = ExecutionWireDisposition::kComplete;
        response.original_status = 0;
        std::uint64_t id{}, bytes{}, extent{};
        std::uint32_t flags{};
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kHello) {
            const bool is_host = request->message.payload.size() == 8;
            if (is_host) {
                std::uint32_t role{};
                expect(input.read_u32(role) && role == kExecutionHostOwnerRole &&
                           input.read_u32(flags) && input.empty(),
                       "explicit private role");
            } else {
                const auto descriptor =
                    decode_execution_context_bootstrap(request->message.payload);
                const bool is_runtime = ++ordinary_contexts >= 3;
                expect(descriptor &&
                           descriptor->purpose == (is_runtime
                                                       ? ExecutionContextPurpose::kRuntimePrimary
                                                       : ExecutionContextPurpose::kExplicitUser) &&
                           descriptor->source_generation ==
                               (is_runtime ? 2 * std::uint64_t{ordinary_contexts - 3} + 1 : 1) &&
                           descriptor->flags == (is_runtime ? CU_CTX_MAP_HOST : 0),
                       "ordinary context flags come from the typed factory, never the role word");
                flags = descriptor->flags;
            }
            contexts[request->connection] = flags;
            host_roles[request->connection] = is_host;
            const auto ownership =
                encode_execution_bootstrap_ownership(ExecutionBootstrapOwnership::kRetained);
            response.data.assign(ownership->begin(), ownership->end());
        } else if (operation == ExecutionOpcode::kImportHostMemory) {
            expect(!pending_import, "pending host admission occupies the last local capacity slot");
            expect(host_roles.at(request->connection) && input.read_u64(bytes) &&
                       input.read_u64(extent) && input.read_u32(flags) && input.empty() &&
                       bytes > 0 && bytes <= 4096 && extent == 4096 &&
                       request->descriptor.get() >= 0,
                   "fixed-width exact shared host import ABI");
            ExecutionPayloadWriter ownership;
            expect(ownership.write_u32(is_quarantine ? 1U : 0U),
                   "exact native uncertainty outcome");
            response.data = ownership.take();
            ++imports;
            if (imports == 1 && !is_quarantine) {
                response.status = CUDA_ERROR_OUT_OF_MEMORY;
                response.original_status = CUDA_ERROR_OUT_OF_MEMORY;
            } else {
                const auto native_address =
                    imports == 261 && scenario == Scenario::kNormal
                        ? reused_address.value()
                        : std::uint64_t{0x380000000000} + std::uint64_t{imports} * 0x20000;
                void* mapped =
                    ::mmap(address(native_address), 4096, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_FIXED_NOREPLACE, request->descriptor.get(), 0);
                expect(mapped != MAP_FAILED && mapped == address(native_address),
                       "fake SDK registers actual shared backing");
                regions.emplace(imports, Region{mapped, 4096, static_cast<std::size_t>(bytes)});
                response.id = imports;
                response.address = imports == 2 ? collision : native_address;
                if (is_quarantine) {
                    response.status = CUDA_ERROR_UNKNOWN;
                    response.original_status = CUDA_ERROR_UNKNOWN;
                    response.id = response.address = 0;
                }
                if (imports == 259 && scenario == Scenario::kNormal) {
                    pending_import = std::move(*request);
                    pending_import_reply = std::move(response);
                    send(events, 'I');
                    continue;
                }
            }
        } else if (operation == ExecutionOpcode::kReleaseHostMemory) {
            if (pending_kernel) {
                expect(!pending_kernel_release, "one pending host retirement retains its mapping");
                pending_kernel_release = std::move(*request);
                send(events, 'L');
            } else {
                retire_host(std::move(*request));
            }
            continue;
        } else if (operation == ExecutionOpcode::kHostTransfer) {
            std::uint64_t owner{}, offset{}, allocation{}, device_offset{}, count{}, stream{};
            std::uint32_t upload{};
            expect(input.read_u64(owner) && input.read_u64(id) && input.read_u64(offset) &&
                       input.read_u64(allocation) && input.read_u64(device_offset) &&
                       input.read_u64(count) && input.read_u64(stream) && input.read_u32(upload) &&
                       upload <= 1 && input.empty() && !stream && count <= 256,
                   "typed bidirectional shared transfer ABI");
            if (count) {
                expect(host_roles.at(owner) && allocation == 7 && !device_offset &&
                           regions.contains(id) && offset + count <= regions.at(id).bytes,
                       "authenticated host/device identity and range");
                auto* source = static_cast<std::byte*>(regions.at(id).pointer) + offset;
                if (upload)
                    std::memcpy(device.data(), source, static_cast<std::size_t>(count));
                else
                    std::memcpy(source, device.data(), static_cast<std::size_t>(count));
            }
            response.id = 1;
        } else if (operation == ExecutionOpcode::kAllocate) {
            expect(input.read_u64(bytes) && bytes == 256 && input.empty(), "typed device range");
            response.id = 7;
            response.address = 0x100000;
        } else if (operation == ExecutionOpcode::kLoadPtx)
            response.id = 9;
        else if (operation == ExecutionOpcode::kGetFunction)
            response.id = 11;
        else if (operation == ExecutionOpcode::kLaunch) {
            expect(!pending_kernel, "one independently held fake physical command");
            ExecutionAsyncFixture::zero_argument_launch(input, 11, 1, 1);
            ++launches;
            response = asynchronous.admit(request->connection, 0);
            pending_kernel = std::pair{request->connection, response.id};
            reply(endpoint, *request, response);
            send(events, 'K');
            continue;
        } else if (operation == ExecutionOpcode::kGetContextFlags)
            response.id = contexts.at(request->connection);
        else if (operation == ExecutionOpcode::kCaptureFrontier) {
            std::uint32_t scope{};
            expect(input.read_u32(scope) && scope <= 2 && input.read_u64(id) && id == 0 &&
                       input.empty(),
                   "host lifetime captures the exact context/default-stream prefix");
            response =
                asynchronous.capture(request->connection, scope ? std::optional{id} : std::nullopt);
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier ||
                   operation == ExecutionOpcode::kQueryAsyncFrontiers ||
                   operation == ExecutionOpcode::kReleaseAsyncFrontiers)
            response = asynchronous.observe(*request);
        else {
            expect(!is_quarantine || operation != ExecutionOpcode::kClose,
                   "unknown host factory ownership must not permit origin context destruction");
            expect(operation == ExecutionOpcode::kClose || operation == ExecutionOpcode::kRelease ||
                       operation == ExecutionOpcode::kUnloadModule,
                   "only covered public operations execute");
            if (operation == ExecutionOpcode::kClose) asynchronous.close(request->connection);
        }
        reply(endpoint, *request, response);
        if (operation == ExecutionOpcode::kImportHostMemory && imports == 261 &&
            scenario == Scenario::kNormal)
            send(events, 'A');
    }
}
template <class Function>
Function cached(void* sdk, const char* name) {
    auto query =
        reinterpret_cast<decltype(&cuGetProcAddress_v2)>(::dlsym(sdk, "cuGetProcAddress_v2"));
    expect(query, "owned Driver query entry exists");
    void* value{};
    CUdriverProcAddressQueryResult status{};
    check(query(name, &value, 12000, 0, &status));
    expect(value && status == CU_GET_PROC_ADDRESS_SUCCESS, "cached exact public host ABI");
    return reinterpret_cast<Function>(value);
}
template <class Function>
Function runtime(const char* name) {
    auto* value = ::dlsym(RTLD_DEFAULT, name);
    expect(value, "Runtime host symbol is exported");
    return reinterpret_cast<Function>(value);
}
void application(const char* sdk_path, void* collision, Latches latches, Scenario scenario) {
    const bool is_quarantine = scenario == Scenario::kQuarantine;
    const auto events = latches.events, commands = latches.commands;
    auto* sdk = ::dlopen(sdk_path, RTLD_NOW | RTLD_LOCAL);
    expect(sdk, "open fake metadata-only Driver scope");
    auto allocate = cached<decltype(&cuMemHostAlloc)>(sdk, "cuMemHostAlloc");
    auto release = cached<decltype(&cuMemFreeHost)>(sdk, "cuMemFreeHost");
    auto flags = cached<decltype(&cuMemHostGetFlags)>(sdk, "cuMemHostGetFlags");
    auto pointer = cached<decltype(&cuMemHostGetDevicePointer)>(sdk, "cuMemHostGetDevicePointer");
    auto upload = cached<decltype(&cuMemcpyHtoDAsync)>(sdk, "cuMemcpyHtoDAsync");
    auto download = cached<decltype(&cuMemcpyDtoHAsync)>(sdk, "cuMemcpyDtoHAsync");
    auto default_allocate = cached<decltype(&cuMemAllocHost)>(sdk, "cuMemAllocHost");
    auto register_existing = cached<decltype(&cuMemHostRegister)>(sdk, "cuMemHostRegister");
    auto unregister_existing = cached<decltype(&cuMemHostUnregister)>(sdk, "cuMemHostUnregister");
    expect(register_existing(collision, 4096, 0) == CUDA_ERROR_NOT_SUPPORTED &&
               unregister_existing(collision) == CUDA_ERROR_NOT_SUPPORTED,
           "arbitrary existing registration rejects explicitly, never caller-native fallback");
    check(release(nullptr));
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    void* value = collision;
    if (is_quarantine) {
        expect(allocate(&value, 256, 0) == CUDA_ERROR_UNKNOWN && value == collision &&
                   allocate(&value, 256, 0) == CUDA_ERROR_UNKNOWN &&
                   cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
               "unknown native host registration retains ownership and forbids replay/context "
               "destruction");
        expect(glimmer_memory_native_calls() == 0,
               "unknown ownership never falls back to caller CUDA");
        send(commands, 'Q');
        expect(::dlclose(sdk) == 0, "release fixture SDK loader reference");
        return;
    }
    expect(allocate(&value, 256, 0) == CUDA_ERROR_OUT_OF_MEMORY && value == collision,
           "known empty registration error preserves output and allows a new attempt");
    expect(allocate(&value, 256, 0) == CUDA_ERROR_OUT_OF_MEMORY && value == collision &&
               *static_cast<unsigned char*>(collision) == 77,
           "UVA placement collision releases the unpublished native resource without overwriting "
           "application memory");
    if (scenario == Scenario::kRecycling) {
        for (unsigned int iteration = 0; iteration < 17000; ++iteration) {
            void* buffer{};
            check(allocate(&buffer, 256, 0));
            check(release(buffer));
        }
        check(cuCtxDestroy(context));
        expect(glimmer_memory_native_calls() == 0,
               "long-lived host recycling never falls back to caller CUDA");
        expect(::dlclose(sdk) == 0, "release recycling SDK reference");
        send(commands, 'Q');
        std::cout << "Host metadata recycling verified: allocations=17000; native_calls=0\n";
        return;
    }
    check(allocate(&value, 256, 0));
    CUcontext current{};
    check(cuCtxGetCurrent(&current));
    expect(current == context, "private host context never changes the user stack");
    unsigned int observed{};
    CUdeviceptr mapped{};
    check(flags(&observed, value));
    check(pointer(&mapped, static_cast<char*>(value) + 1, 0));
    expect(
        observed == CU_MEMHOSTALLOC_DEVICEMAP && mapped == reinterpret_cast<CUdeviceptr>(value) + 1,
        "native-normalized flags and exact interior mapped-pointer bytes");
    expect(flags(&observed, static_cast<char*>(value) + 256) == CUDA_ERROR_INVALID_VALUE &&
               pointer(&mapped, value, 1) == CUDA_ERROR_INVALID_VALUE,
           "bounds and query flags reject locally");
    CUdeviceptr device{};
    check(cuMemAlloc(&device, 256));
    expect(
        upload(device, static_cast<char*>(value) + 128, 129, nullptr) == CUDA_ERROR_INVALID_VALUE,
        "an overlong owned pinned range cannot silently become a pageable staging allocation");
    std::memset(value, 43, 256);
    check(upload(device, value, 256, nullptr));
    std::memset(value, 0, 256);
    check(download(value, device, 256, nullptr));
    for (std::size_t index = 0; index < 256; ++index)
        expect(static_cast<unsigned char*>(value)[index] == 43,
               "actual shared host bytes, not retained caller pointers");
    check(cuMemcpyHtoD(device, value, 256));
    check(cuMemcpyDtoH(value, device, 256));
    CUmodule module{};
    CUfunction function{};
    check(cuModuleLoadData(
        &module, ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry kernel() {ret;}"));
    check(cuModuleGetFunction(&function, module, "kernel"));
    auto launch = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr);
    });
    expect(receive(events) == 'K', "observe actual admission with an independently pending tail");
    check(launch.get());
    auto freeing = std::async(std::launch::async, [&] { return release(value); });
    expect(receive(events) == 'L', "native host owner receives and retains the exact free request");
    check(cuCtxGetFlags(&observed));
    expect(observed == 0, "unrelated metadata progresses while kernel/free wait");
    expect(freeing.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready,
           "free waits accepted host-kernel physical retirement without holding global exclusion");
    send(commands, 'C');
    check(freeing.get());
    expect(release(value) == CUDA_ERROR_INVALID_VALUE, "duplicate free cannot issue again");
    void* zero = value;
    check(default_allocate(&zero, 0));
    expect(!zero, "zero allocation has no native host backing");
    std::vector<void*> bulk(255);
    for (auto& buffer : bulk) check(allocate(&buffer, 256, 0));
    auto last = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        void* buffer{};
        check(allocate(&buffer, 256, 0));
        return buffer;
    });
    expect(receive(events) == 'I', "last capacity slot waits for its actual native result");
    void* full = collision;
    expect(allocate(&full, 256, 0) == CUDA_ERROR_OUT_OF_MEMORY && full == collision,
           "parallel unfinished allocation counts against bounded host capacity before native "
           "ownership");
    send(commands, 'D');
    bulk.push_back(last.get());
    for (auto* buffer : bulk) check(release(buffer));
    check(default_allocate(&value, 256));
    check(cuCtxSetCurrent(nullptr));
    expect(flags(&observed, value) == CUDA_ERROR_INVALID_CONTEXT,
           "Driver query requires current context");
    auto* previous_address = value;
    auto old_free = std::async(std::launch::async, [&] { return release(previous_address); });
    expect(receive(events) == 'F',
           "native unregister finished before its old client reply is delivered");
    auto reused = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        void* buffer{};
        check(allocate(&buffer, 256, 0));
        return buffer;
    });
    expect(receive(events) == 'A', "new native import reuses the old local mapping address");
    expect(reused.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready,
           "new publication waits for finite retiring mapping cleanup instead of reporting false "
           "collision/OOM");
    send(commands, 'F');
    check(old_free.get());
    value = reused.get();
    expect(value == previous_address, "native address reuse preserves the real public pointer ABI");
    check(release(value));
    check(cuCtxSetCurrent(context));
    auto runtime_allocate = runtime<decltype(&cudaHostAlloc)>("cudaHostAlloc");
    auto runtime_free = runtime<decltype(&cudaFreeHost)>("cudaFreeHost");
    auto runtime_flags = runtime<decltype(&cudaHostGetFlags)>("cudaHostGetFlags");
    auto runtime_pointer = runtime<decltype(&cudaHostGetDevicePointer)>("cudaHostGetDevicePointer");
    auto runtime_copy = runtime<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    expect(runtime_allocate(&value, 256, 1) == cudaSuccess,
           "Runtime host allocation uses same native owner");
    expect(runtime_flags(&observed, value) == cudaSuccess && observed == 3,
           "Runtime flags normalize mapped portable memory");
    void* alias{};
    expect(runtime_pointer(&alias, value, 0) == cudaSuccess && alias == value,
           "Runtime mapped pointer shares CPU address");
    std::memset(value, 59, 256);
    expect(runtime_copy(address(device), value, 256, cudaMemcpyDefault, nullptr) == cudaSuccess &&
               runtime_copy(value, address(device), 256, cudaMemcpyDefault, nullptr) == cudaSuccess,
           "Runtime default-direction pinned async copies keep typed shared dataflow");
    for (std::size_t index = 0; index < 256; ++index)
        expect(static_cast<unsigned char*>(value)[index] == 59, "complete Runtime copied data");
    expect(runtime_free(value) == cudaSuccess, "Runtime free exactly once");
    check(allocate(&value, 256, CU_MEMHOSTALLOC_PORTABLE));
    check(cuCtxDestroy(context));
    check(cuCtxCreate(&context, 0, 0));
    expect(flags(&observed, value) == CUDA_ERROR_INVALID_VALUE &&
               release(value) == CUDA_ERROR_INVALID_VALUE,
           "origin close frees even portable host memory before invalidating the application "
           "generation");
    check(cuCtxDestroy(context));
    expect(runtime_allocate(&value, 256, 0) == cudaSuccess,
           "first Runtime host use lazily initializes a primary context");
    auto reset = runtime<decltype(&cudaDeviceReset)>("cudaDeviceReset");
    expect(reset() == cudaSuccess && runtime_flags(&observed, value) == cudaErrorInvalidValue &&
               runtime_free(value) == cudaErrorInvalidValue && reset() == cudaSuccess,
           "primary reset releases host resources and never rebinds them into a later generation");
    expect(glimmer_memory_native_calls() == 0,
           "no caller-native host registration/allocation/copy/free fallback");
    expect(::dlclose(sdk) == 0, "release fixture SDK loader reference");
    send(commands, 'Q');
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 3,
               "fake Driver path and optional isolated quarantine control required");
        expect(argc == 2 || std::string(argv[2]) == "--quarantine" ||
                   std::string(argv[2]) == "--recycle",
               "unknown host fixture mode");
        const auto scenario = argc == 2                                ? Scenario::kNormal
                              : std::string(argv[2]) == "--quarantine" ? Scenario::kQuarantine
                                                                       : Scenario::kRecycling;
        void* collision = ::mmap(address(0x370000000000), 4096, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        expect(collision != MAP_FAILED, "reserve an application-owned collision witness");
        std::memset(collision, 77, 4096);
        Pipe events, commands;
        const auto path = "/tmp/glimmer-public-host-" + std::to_string(::getpid());
        const auto child = ::fork();
        expect(child >= 0, "fork actual CPU native-owner endpoint before frontend state exists");
        if (child == 0) {
            events.read = ExecutionOwnedDescriptor{};
            commands.write = ExecutionOwnedDescriptor{};
            try {
                server(path, reinterpret_cast<std::uintptr_t>(collision),
                       {events.write.get(), commands.read.get()}, scenario);
                ::_exit(EXIT_SUCCESS);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                ::_exit(EXIT_FAILURE);
            }
        }
        events.write = ExecutionOwnedDescriptor{};
        commands.read = ExecutionOwnedDescriptor{};
        expect(receive(events.read.get()) == 'R', "actual child endpoint ready");
        std::uint64_t start{};
        expect(read_process_start_time(static_cast<std::uint32_t>(child), &start),
               "exact child PID/start authentication");
        const auto pid_text = std::to_string(child), start_text = std::to_string(start);
        expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid_text.c_str(), 1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_START", start_text.c_str(), 1) == 0,
               "configure authenticated public frontend");
        application(argv[1], collision, {events.read.get(), commands.write.get()}, scenario);
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "child native-owner checks passed");
        expect(::munmap(collision, 4096) == 0,
               "original application mapping survives collision rollback unchanged");
        if (scenario == Scenario::kNormal) {
            const auto isolated = ::fork();
            expect(isolated >= 0, "fork fresh executable for unknown host ownership");
            if (isolated == 0) {
                std::array<char*, 4> arguments{argv[0], argv[1], const_cast<char*>("--quarantine"),
                                               nullptr};
                ::execv(argv[0], arguments.data());
                ::_exit(EXIT_FAILURE);
            }
            expect(::waitpid(isolated, &status, 0) == isolated && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "unknown native host ownership is contained in a fresh isolated process");
        }
        std::cout << "Public host ABI, shared dataflow, admission, rollback and origin lifetime "
                     "verified\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
