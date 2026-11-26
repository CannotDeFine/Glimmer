#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern "C" unsigned int glimmer_unified_copy_native_calls();
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr, CUdeviceptr, std::size_t);
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr, CUdeviceptr, std::size_t, CUstream);

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr std::size_t kBytes = 256;
constexpr CUdeviceptr kFirst = 0x100000;
constexpr CUdeviceptr kSecond = 0x110000;
constexpr CUdeviceptr kForeign = 0x120000;
constexpr std::uint64_t kHostAddress = 0x380000000000;
constexpr unsigned int kHostileCases = 14;
thread_local std::atomic<bool>* wait_observation{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    expect(status == CUDA_SUCCESS, "public owned Driver operation succeeds");
}
template <class Predicate>
void await(Predicate&& predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        expect(Clock::now() < deadline, "bounded independent public-call latch");
        std::this_thread::yield();
    }
}
template <class Pointer>
CUdeviceptr address(Pointer* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}
struct Libraries {
    void* provider{};
    decltype(&cuMemcpy) copy{};
    decltype(&cuMemcpyAsync) asynchronous{};
    decltype(&cuMemcpy_ptds) per_thread{};
    decltype(&cuMemcpyAsync_ptsz) per_thread_async{};
    explicit Libraries(const char* path) {
        provider = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
        expect(provider != nullptr, "load the genuine native-trap provider scope");
        auto query = reinterpret_cast<decltype(&cuGetProcAddress_v2)>(
            ::dlsym(provider, "cuGetProcAddress_v2"));
        expect(query != nullptr, "qualified SDK Driver query lookup");
        const auto resolve = [&](const char* name, cuuint64_t flags) {
            void* result{};
            CUdriverProcAddressQueryResult status{};
            check(query(name, &result, 4000, flags, &status));
            expect(result && status == CU_GET_PROC_ADDRESS_SUCCESS,
                   "cached lookup retains the actual historical UVA ABI");
            return result;
        };
        copy = reinterpret_cast<decltype(copy)>(resolve("cuMemcpy", 0));
        asynchronous = reinterpret_cast<decltype(asynchronous)>(resolve("cuMemcpyAsync", 1));
        per_thread = reinterpret_cast<decltype(per_thread)>(resolve("cuMemcpy", 2));
        per_thread_async =
            reinterpret_cast<decltype(per_thread_async)>(resolve("cuMemcpyAsync", 2));
        expect(copy == &cuMemcpy && asynchronous == &cuMemcpyAsync &&
                   per_thread == &cuMemcpy_ptds && per_thread_async == &cuMemcpyAsync_ptsz &&
                   ::dlsym(provider, "cuMemcpy") == reinterpret_cast<void*>(copy) &&
                   ::dlsym(RTLD_DEFAULT, "cuMemcpyAsync") == reinterpret_cast<void*>(asynchronous),
               "direct, global, qualified and cached entry points select owned exact ABIs");
        void* obsolete = reinterpret_cast<void*>(copy);
        CUdriverProcAddressQueryResult status{};
        check(query("cuMemcpy", &obsolete, 3999, 0, &status));
        expect(!obsolete && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT,
               "native historical version rejection is preserved");
    }
    ~Libraries() {
        if (provider && ::dlclose(provider) != 0) std::terminate();
    }
    Libraries(const Libraries&) = delete;
    Libraries& operator=(const Libraries&) = delete;
};
struct Latches {
    std::atomic<bool> held{}, may_reply{}, free_started{}, stream_destroy_started{};
    std::atomic<bool> free_waited{}, stream_destroy_waited{};
    std::atomic<std::byte*> source_after_prefix{};
};
void application(const char* provider, Latches& latches, unsigned int hostile,
                 bool per_thread_hostile) {
    Libraries sdk(provider);
    expect(cuMemcpy(0, 0, 0) == CUDA_ERROR_INVALID_CONTEXT,
           "no current Driver context cannot activate a Runtime primary");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    CUdeviceptr first{};
    check(cuMemAlloc(&first, kBytes));
    expect(first == kFirst, "first synthetic device identity");
    std::array<std::byte, kBytes> input{}, output{};
    input.fill(std::byte{43});
    output.fill(std::byte{0xa7});
    if (hostile) {
        expect(sdk.asynchronous(address(output.data()), first, 128,
                                per_thread_hostile ? CU_STREAM_PER_THREAD : nullptr) ==
                       CUDA_ERROR_UNKNOWN &&
                   std::ranges::all_of(output, [](auto value) { return value == std::byte{0xa7}; }),
               "malformed or lost reply cannot publish private output");
        expect(cuMemcpy(first, first, 1) != CUDA_SUCCESS &&
                   cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN &&
                   glimmer_unified_copy_native_calls() == 0,
               "uncertain command is retained without replay, close or native fallback");
        return;
    }
    CUdeviceptr second{};
    CUstream stream{};
    check(cuMemAlloc(&second, kBytes));
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    expect(second == kSecond, "second synthetic device identity");
    check(cuMemcpyAsync(first + 8, address(input.data()), 128, stream));
    input.fill(std::byte{17});
    check(sdk.copy(second + 16, first + 8, 128));
    check(sdk.copy(address(output.data() + 4), second + 16, 128));
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(output[index] == (index >= 4 && index < 132 ? std::byte{43} : std::byte{0xa7}),
               "full pageable output and guards preserve the immutable upload snapshot");

    void* pinned{};
    check(cuMemHostAlloc(&pinned, 300, CU_MEMHOSTALLOC_PORTABLE));
    auto host = std::span{static_cast<std::byte*>(pinned), std::size_t{300}};
    std::ranges::fill(host, std::byte{0xa7});
    std::ranges::fill(host.subspan(4, 128), std::byte{0x5d});
    check(cuMemcpy(first + 8, address(host.data() + 4), 128));
    check(sdk.asynchronous(address(host.data() + 140), first + 8, 128, stream));
    for (std::size_t index = 0; index < host.size(); ++index)
        expect(host[index] == ((index >= 4 && index < 132) || (index >= 140 && index < 268)
                                   ? std::byte{0x5d}
                                   : std::byte{0xa7}),
               "public HST1 spans use the registered owner, preserving all guards");
    check(sdk.copy(first + 8, address(input.data()), 128));

    for (const bool asynchronous : {false, true})
        for (unsigned int route = 0; route != 4; ++route) {
            const bool from_pinned = (route & 1U) != 0;
            const bool to_pinned = (route & 2U) != 0;
            std::ranges::fill(host, std::byte{0xa7});
            std::ranges::fill(host.subspan(4, 67), std::byte{0x5d});
            output.fill(std::byte{0xa7});
            const auto source = from_pinned ? address(host.data() + 4) : address(input.data() + 5);
            const auto destination =
                to_pinned ? address(host.data() + 140) : address(output.data() + 7);
            check(asynchronous ? sdk.asynchronous(destination, source, 67, stream)
                               : sdk.copy(destination, source, 67));
            const auto copied = from_pinned ? std::byte{0x5d} : std::byte{17};
            for (std::size_t index = 0; index != output.size(); ++index)
                expect(output[index] ==
                           (!to_pinned && index >= 7 && index < 74 ? copied : std::byte{0xa7}),
                       "complete H2H pageable destination and guards");
            for (std::size_t index = 0; index != host.size(); ++index)
                expect(
                    host[index] == (index >= 4 && index < 71                   ? std::byte{0x5d}
                                    : to_pinned && index >= 140 && index < 207 ? copied
                                                                               : std::byte{0xa7}),
                    "complete H2H registered source/destination and guards");
            expect(std::ranges::all_of(input, [](auto value) { return value == std::byte{17}; }),
                   "H2H staging never writes the pageable source");
        }
    output.fill(std::byte{0xa7});
    latches.source_after_prefix = input.data();
    check(sdk.asynchronous(address(output.data() + 7), address(input.data() + 5), 67, stream));
    expect(!latches.source_after_prefix,
           "asynchronous H2H snapshots after its actual upcoming source-stream prefix");
    for (std::size_t index = 0; index != output.size(); ++index)
        expect(output[index] == (index >= 7 && index < 74 ? std::byte{0x6b} : std::byte{0xa7}),
               "asynchronous pageable H2H consumes every producer-updated byte and no guard");
    expect(std::ranges::all_of(input, [](auto value) { return value == std::byte{0x6b}; }),
           "the completed prefix publishes its source mutation independently of copy issue");
    input.fill(std::byte{17});
    output.fill(std::byte{0xa7});
    expect(sdk.asynchronous(address(output.data()), address(host.data()), 13, stream) ==
                   CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES &&
               std::ranges::all_of(output, [](auto value) { return value == std::byte{0xa7}; }),
           "H2H original error keeps provenance and never publishes private destination");
    expect(sdk.copy(address(input.data() + 1), address(input.data()), 16) ==
                   CUDA_ERROR_NOT_SUPPORTED &&
               sdk.copy(address(host.data() + 5), address(host.data() + 4), 16) ==
                   CUDA_ERROR_NOT_SUPPORTED,
           "overlapping H2H spans reject before staging or original issue");
    constexpr std::size_t oversized = std::size_t{33} * 1024 * 1024;
    void* large_source = ::mmap(nullptr, oversized, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void* large_destination =
        ::mmap(nullptr, oversized, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(large_source != MAP_FAILED && large_destination != MAP_FAILED,
           "create distinct protected spans for an aggregate staging quota control");
    expect(sdk.asynchronous(address(large_destination), address(large_source), oversized, stream) ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "both pageable reservations reject atomically before reading protected source");
    expect(::munmap(large_source, oversized) == 0 && ::munmap(large_destination, oversized) == 0,
           "release only the two exact quota-control mappings");

    for (const auto& [count, expected] :
         std::array{std::pair{std::size_t{13}, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
                    std::pair{std::size_t{17}, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES},
                    std::pair{std::size_t{19}, CUDA_ERROR_INVALID_VALUE},
                    std::pair{std::size_t{23}, CUDA_ERROR_OUT_OF_MEMORY},
                    std::pair{std::size_t{29}, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES}}) {
        output.fill(std::byte{0xa7});
        expect(sdk.asynchronous(address(output.data()), first + 8, count, stream) == expected &&
                   std::ranges::all_of(output, [](auto value) { return value == std::byte{0xa7}; }),
               "native, post-SDK frontier and pre-entry errors preserve public output");
    }
    check(cuMemcpy(0, first, 0));
    void* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "protected zero-byte source mapping");
    expect(sdk.asynchronous(first, address(inaccessible), 0, stream) == CUDA_ERROR_NOT_SUPPORTED,
           "unknown nonnull zero-byte host address does not gain native validity from staging");
    expect(sdk.asynchronous(first, address(inaccessible), 1, stream) == CUDA_ERROR_INVALID_VALUE,
           "positive protected input rejects without import or issue");
    expect(::munmap(inaccessible, 4096) == 0, "release protected test mapping");

    expect(cuMemcpy(first + kBytes, first, 1) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpy(first, first - 1, 2) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpy(first, UINT64_MAX, 2) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpy(first, address(host.data() + 299), 2) == CUDA_ERROR_INVALID_VALUE &&
               cuMemcpy(first, address(host.data() + 300), 1) == CUDA_ERROR_INVALID_VALUE,
           "endpoint, crossing, overflow and host-padding spans cannot fall back to pageable");
    expect(cuMemcpy(0, 0, 0) == CUDA_ERROR_NOT_SUPPORTED &&
               cuMemcpy_ptds(first, second, 1) == CUDA_ERROR_NOT_SUPPORTED &&
               sdk.per_thread(first, second, 1) == CUDA_ERROR_NOT_SUPPORTED &&
               cuMemcpyAsync_ptsz(first, second, 1, stream) == CUDA_ERROR_NOT_SUPPORTED &&
               sdk.per_thread_async(first, second, 1, stream) == CUDA_ERROR_NOT_SUPPORTED,
           "unqualified empty and exact per-thread variants fail closed without native resources");

    const auto per_thread_copies = [&](std::byte value) {
        std::array<std::byte, kBytes> source{}, destination{};
        source.fill(value);
        destination.fill(std::byte{0xa7});
        const auto upload =
            cuMemcpyAsync(first + 8, address(source.data()), 128, CU_STREAM_PER_THREAD);
        expect(upload == CUDA_SUCCESS, "generic Async special handle uses its owned TLS stream");
        check(sdk.asynchronous(second + 16, first + 8, 128, CU_STREAM_PER_THREAD));
        check(sdk.asynchronous(address(destination.data() + 4), second + 16, 128,
                               CU_STREAM_PER_THREAD));
        for (std::size_t index = 0; index < kBytes; ++index)
            expect(source[index] == value &&
                       destination[index] == (index >= 4 && index < 132 ? value : std::byte{0xa7}),
                   "per-thread public direct/cached copies preserve full source/output guards");
    };
    per_thread_copies(std::byte{17});
    auto other_caller = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        per_thread_copies(std::byte{49});
        check(cuCtxSetCurrent(nullptr));
    });
    other_caller.get();
    per_thread_copies(std::byte{17});
    for (const bool cached : {false, true}) {
        input.fill(std::byte{17});
        output.fill(std::byte{0xa7});
        latches.source_after_prefix = input.data();
        const auto copy = cached ? sdk.asynchronous : &cuMemcpyAsync;
        check(
            copy(address(output.data() + 7), address(input.data() + 5), 67, CU_STREAM_PER_THREAD));
        expect(!latches.source_after_prefix,
               "per-thread H2H input waits its exact mapped source-stream prefix");
        for (std::size_t index = 0; index < kBytes; ++index)
            expect(
                input[index] == std::byte{0x6b} &&
                    output[index] == (index >= 7 && index < 74 ? std::byte{0x6b} : std::byte{0xa7}),
                "direct/cached per-thread H2H consumes producer bytes with complete guards");
    }
    CUcontext foreign{};
    CUdeviceptr foreign_device{};
    check(cuCtxCreate(&foreign, 0, 0));
    check(cuMemAlloc(&foreign_device, kBytes));
    expect(foreign_device == kForeign, "foreign device identity");
    check(cuCtxSetCurrent(context));
    expect(cuMemcpy(first, foreign_device, 1) == CUDA_ERROR_INVALID_VALUE,
           "foreign device allocation cannot become a host pointer");
    check(cuCtxSetCurrent(foreign));
    check(cuMemFree(foreign_device));
    check(cuCtxDestroy(foreign));
    check(cuCtxSetCurrent(context));
    expect(cuMemcpy(first, foreign_device, 1) == CUDA_ERROR_INVALID_VALUE,
           "retired device interval remains fail-closed");

    output.fill(std::byte{0xa7});
    auto download = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return sdk.asynchronous(address(output.data()), first + 8, 31, stream);
    });
    await([&] { return latches.held.load(); });
    auto freeing = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        latches.free_started = true;
        wait_observation = &latches.free_waited;
        const auto result = cuMemFree(first);
        wait_observation = nullptr;
        return result;
    });
    auto destroying = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        latches.stream_destroy_started = true;
        wait_observation = &latches.stream_destroy_waited;
        const auto result = cuStreamDestroy(stream);
        wait_observation = nullptr;
        return result;
    });
    await([&] { return latches.free_waited && latches.stream_destroy_waited; });
    unsigned int flags{99};
    check(cuCtxGetFlags(&flags));
    expect(!flags && download.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
               freeing.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
               destroying.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
               std::ranges::all_of(output, [](auto value) { return value == std::byte{0xa7}; }),
           "held exact reply pins device and stream while independent metadata progresses");
    latches.may_reply = true;
    check(download.get());
    check(freeing.get());
    check(destroying.get());
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(output[index] == (index < 31 ? std::byte{17} : std::byte{0xa7}),
               "held successful copyback changes every requested byte and no guard");
    check(cuMemFreeHost(pinned));
    check(cuMemFree(second));
    check(cuCtxDestroy(context));
    expect(glimmer_unified_copy_native_calls() == 0,
           "all direct and cached copies avoid the caller-native trap");
}

struct Endpoint {
    std::uint32_t kind{};
    std::uint64_t connection{}, allocation{}, offset{};
};
struct Copy {
    Endpoint destination, source;
    std::uint64_t bytes{}, stream{};
    bool asynchronous{};
};
// Independent fixed-offset wire oracle; never call the production UVA codec.
Copy decode(std::span<const std::byte> bytes) {
    expect(bytes.size() == 76, "exact append-only UVA packet width");
    const auto integer = [&](std::size_t offset, std::size_t count) {
        std::uint64_t result{};
        for (std::size_t index = 0; index < count; ++index)
            result |= std::uint64_t{std::to_integer<unsigned char>(bytes[offset + index])}
                      << (index * 8);
        return result;
    };
    const auto endpoint = [&](std::size_t offset) {
        Endpoint value{static_cast<std::uint32_t>(integer(offset, 4)), integer(offset + 4, 8),
                       integer(offset + 12, 8), integer(offset + 20, 8)};
        expect(value.kind <= 3 &&
                   (value.kind ? value.allocation != 0
                               : !value.connection && !value.allocation && !value.offset) &&
                   (value.kind >= 2 ? value.connection != 0 : value.connection == 0),
               "typed opaque endpoint capability, never a public host pointer");
        return value;
    };
    Copy result{endpoint(0), endpoint(28), integer(56, 8), integer(64, 8), integer(72, 4) != 0};
    expect(integer(72, 4) <= 1 && (result.asynchronous || !result.stream),
           "exact async flag and synchronous legacy stream");
    return result;
}
struct Mapping {
    void* pointer{MAP_FAILED};
    std::size_t extent{}, bytes{};
    std::uint64_t owner{};
    std::uint32_t role{};
};
struct Device {
    std::array<std::byte, kBytes> data{};
    std::uint64_t owner{};
};
void run(const char* provider, unsigned int hostile) {
    const bool per_thread_hostile = hostile > kHostileCases / 2;
    if (per_thread_hostile) hostile -= kHostileCases / 2;
    const auto path = "/tmp/glimmer-unified-copy-" + std::to_string(::getpid());
    ExecutionSocketConfig config;
    config.socket_path = path;
    // Keep one admission available for the independent metadata call while
    // the exact copy reply is held to exercise resource-lifetime pins.
    config.max_pending_requests = 2;
    ExecutionSocketServer server(config);
    expect(server.start(), "start the authenticated CPU endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted endpoint process identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0 &&
               ::unsetenv("GLIMMER_EXECUTION_STAGING_CACHE") == 0,
           "select actual authenticated transport and default bounded cache");
    std::map<std::uint64_t, std::uint32_t> roles;
    std::map<std::uint64_t, Device> devices;
    std::map<std::uint64_t, Mapping> mappings;
    std::map<std::uint64_t, std::uint64_t> streams;
    std::vector<std::uint64_t> per_thread_issues;
    std::uint64_t next_device{10}, next_host{100}, next_command{};
    unsigned int copies{}, host_copies{}, pageable_copies{}, imports{}, releases{}, closes{};
    unsigned int streams_created{};
    unsigned int prefix_queries{}, prefix_releases{};
    std::uint64_t captured_stream{};
    std::optional<ExecutionReceivedRequest> held;
    ExecutionWireReply held_reply;
    Latches latches;
    auto client = std::async(std::launch::async,
                             [&] { application(provider, latches, hostile, per_thread_hostile); });
    struct StopBeforeJoin {
        ExecutionSocketServer& server;
        ~StopBeforeJoin() {
            server.stop();
        }
    } stop_before_join{server};
    const auto respond = [&](const ExecutionReceivedRequest& request,
                             const ExecutionWireReply& result) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {131, 137};
        const auto bytes = encode_execution_reply(result);
        expect(bytes && server.reply(request.connection, {header, *bytes}),
               "exact authenticated reply");
    };
    const auto view = [&](const Endpoint& endpoint, const Copy& copy, std::uint64_t origin) {
        const auto bytes = copy.bytes;
        std::span<std::byte> result;
        if (!endpoint.kind) {
            expect(!bytes, "NULL is only an empty endpoint");
            return result;
        }
        if (endpoint.kind == 1) {
            expect(devices.contains(endpoint.allocation) &&
                       devices.at(endpoint.allocation).owner == origin,
                   "device endpoint belongs to the exact ordinary source");
            result = devices.at(endpoint.allocation).data;
        } else {
            expect(mappings.contains(endpoint.allocation), "live helper allocation identity");
            auto& mapping = mappings.at(endpoint.allocation);
            expect(mapping.owner == endpoint.connection &&
                       mapping.role == (endpoint.kind == 2 ? 0x31545348U : 0x31474150U),
                   "public registered HST1 and private unregistered PAG1 cannot cross roles");
            result = {static_cast<std::byte*>(mapping.pointer), mapping.bytes};
        }
        expect(endpoint.offset <= result.size() && bytes <= result.size() - endpoint.offset,
               "independent complete endpoint span bounds");
        return result.subspan(static_cast<std::size_t>(endpoint.offset),
                              static_cast<std::size_t>(bytes));
    };
    const auto deadline = Clock::now() + std::chrono::seconds(15);
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline,
               "bounded public UVA fixture progress");
        if (held && latches.may_reply) {
            respond(*held, held_reply);
            held.reset();
        }
        auto request = server.take_request();
        if (!request) continue;
        const auto operation = request->message.header.operation;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply result;
        result.disposition = ExecutionWireDisposition::kComplete;
        if (operation == ExecutionOpcode::kHello) {
            std::uint32_t role{}, flags{};
            if (request->message.payload.size() == 8) {
                expect(input.read_u32(role) && (role == 0x31545348U || role == 0x31474150U) &&
                           input.read_u32(flags) && flags == 8 && input.empty(),
                       "only exact public HST1 or unregistered PAG1 helper roles");
            } else {
                expect(decode_execution_context_bootstrap(request->message.payload).has_value(),
                       "ordinary explicit context bootstrap");
            }
            roles.emplace(request->connection, role);
            const auto ownership =
                encode_execution_bootstrap_ownership(ExecutionBootstrapOwnership::kRetained);
            result.data.assign(ownership->begin(), ownership->end());
        } else if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && bytes == kBytes && input.empty(), "fixed device size");
            result.id = ++next_device;
            result.address = kFirst + (result.id - 11) * 0x10000;
            auto& device = devices[result.id];
            device.owner = request->connection;
            device.data.fill(std::byte{0xcc});
        } else if (operation == ExecutionOpcode::kCreateStream) {
            std::uint32_t flags{};
            expect(input.read_u32(flags) && input.empty() &&
                       flags == (streams_created || per_thread_hostile ? CU_STREAM_DEFAULT
                                                                       : CU_STREAM_NON_BLOCKING),
                   "explicit nonblocking stream and regular blocking per-thread proxies");
            ++streams_created;
            result.id = 50 + streams_created;
            streams.emplace(result.id, request->connection);
        } else if (operation == ExecutionOpcode::kImportHostMemory) {
            std::uint64_t bytes{}, extent{};
            std::uint32_t flags{};
            const auto role = roles.at(request->connection);
            expect(role && input.read_u64(bytes) && bytes && input.read_u64(extent) &&
                       extent == 4096 && bytes <= extent && input.read_u32(flags) &&
                       input.empty() && flags == (role == 0x31545348U ? 1U : 0U) &&
                       request->descriptor.get() >= 0,
                   "sealed mapping capability has exact memory class, logical size and extent");
            struct stat information {};
            const auto seals = ::fcntl(request->descriptor.get(), F_GET_SEALS);
            constexpr auto required_seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
            expect(::fstat(request->descriptor.get(), &information) == 0 &&
                       S_ISREG(information.st_mode) && information.st_size == 4096 && seals >= 0 &&
                       (seals & required_seals) == required_seals,
                   "actual shared backing has immutable extent and seal set");
            auto* pointer = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                                   request->descriptor.get(), 0);
            expect(pointer != MAP_FAILED, "map actual shared backing in fake owner");
            result.id = ++next_host;
            mappings.emplace(result.id, Mapping{pointer, 4096, static_cast<std::size_t>(bytes),
                                                request->connection, role});
            if (role == 0x31545348U) {
                result.address = kHostAddress;
                result.original_status = 0;
            }
            ExecutionPayloadWriter ownership;
            expect(ownership.write_u32(0), "canonical known ownership import marker");
            result.data = ownership.take();
            ++imports;
        } else if (static_cast<std::uint32_t>(operation) == 141) {
            const auto copy = decode(request->message.payload);
            expect(copy.stream == 0 || (streams.contains(copy.stream) &&
                                        streams.at(copy.stream) == request->connection),
                   "only the exact context's owned stream IDs cross wire");
            if (copy.stream > 51) {
                expect(copy.asynchronous, "per-thread proxies cannot alter synchronous SDK ABI");
                per_thread_issues.push_back(copy.stream);
            }
            auto destination = view(copy.destination, copy, request->connection);
            auto source = view(copy.source, copy, request->connection);
            expect(copy.bytes || copy.destination.kind == 1 || copy.source.kind == 1,
                   "host-only routes require a positive complete span");
            ++copies;
            host_copies += copy.destination.kind == 2 || copy.source.kind == 2 ? 1U : 0U;
            pageable_copies += copy.destination.kind == 3 || copy.source.kind == 3 ? 1U : 0U;
            if (!copy.asynchronous && copy.source.kind == 3)
                expect(prefix_queries && prefix_queries == prefix_releases,
                       "ordinary pageable input snapshots after the exact dependency prefix");
            result.id = ++next_command;
            result.original_status = 0;
            if (copy.bytes == 13) {
                result.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                result.original_status = result.status;
            } else if (copy.bytes == 17) {
                result.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                std::ranges::fill(destination, std::byte{0x11});
            } else if (copy.bytes == 19) {
                result.status = CUDA_ERROR_INVALID_VALUE;
                result.original_status = result.status;
            } else if (copy.bytes == 23 || copy.bytes == 29) {
                result.disposition = ExecutionWireDisposition::kRejected;
                result.status = copy.bytes == 23 ? CUDA_ERROR_OUT_OF_MEMORY
                                                 : CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                result.original_status.reset();
                if (copy.bytes == 23) result.id = 0;
            } else if (copy.bytes) {
                std::ranges::copy(source, destination.begin());
            }
            if (hostile) {
                expect(copies == 1, "uncertain copies never replay");
                if (hostile == 1) result.original_status.reset();
                if (hostile == 2) result.id = 0;
                if (hostile == 3) result.address = 1;
                if (hostile == 4) result.parameter_bytes = 1;
                if (hostile == 5) result.data.push_back(std::byte{1});
                if (hostile == 6) result.original_status = CUDA_ERROR_INVALID_VALUE;
                if (hostile == 7) {
                    server.close_peer(request->connection);
                    continue;
                }
            } else if (copy.bytes == 31) {
                held = std::move(*request);
                held_reply = result;
                latches.held = true;
                continue;
            }
        } else if (operation == ExecutionOpcode::kQueryStagingIdle ||
                   operation == ExecutionOpcode::kReleaseHostMemory) {
            std::uint64_t id{};
            expect(!hostile && input.read_u64(id) && input.empty() && mappings.contains(id) &&
                       mappings.at(id).owner == request->connection,
                   "unknown backing never becomes eligible for idle or release");
            if (operation == ExecutionOpcode::kQueryStagingIdle) {
                expect(mappings.at(id).role == 0x31474150U, "public HST1 is not a private cache");
                result.id = id;
            } else {
                auto& mapping = mappings.at(id);
                if (mapping.role == 0x31545348U) result.original_status = 0;
                expect(::munmap(mapping.pointer, mapping.extent) == 0,
                       "release exact fake-owner mapping");
                mappings.erase(id);
                ++releases;
            }
        } else if (operation == ExecutionOpcode::kCaptureFrontier) {
            std::uint32_t kind{};
            std::uint64_t stream{};
            expect(input.read_u32(kind) && kind == 2 && input.read_u64(stream) &&
                       (stream == 0 ||
                        (streams.contains(stream) && streams.at(stream) == request->connection)) &&
                       input.empty(),
                   "pageable snapshots capture only their upcoming source-stream dependencies");
            captured_stream = stream;
            result.id = 71;
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier) {
            std::uint64_t id{};
            expect(input.read_u64(id) && id == 71 && input.empty(),
                   "exact prefix observation identity");
            if (operation == ExecutionOpcode::kQueryFrontier) {
                if (captured_stream)
                    if (auto* source = latches.source_after_prefix.exchange(nullptr))
                        std::fill_n(source, kBytes, std::byte{0x6b});
                ++prefix_queries;
            } else
                ++prefix_releases;
        } else if (operation == ExecutionOpcode::kRelease) {
            std::uint64_t id{};
            expect(!held && !hostile && input.read_u64(id) && input.empty() && devices.contains(id),
                   "device cannot release before its public-call pin and reply");
            for (std::size_t index = 0; index < kBytes; ++index)
                if ((id == 11 && (index < 8 || index >= 136)) ||
                    (id == 12 && (index < 16 || index >= 144)) || id == 13)
                    expect(devices.at(id).data[index] == std::byte{0xcc},
                           "all device padding guards survive");
            devices.erase(id);
        } else if (operation == ExecutionOpcode::kDestroyStream) {
            std::uint64_t id{};
            expect(!held && !hostile && input.read_u64(id) && id == 51 && input.empty(),
                   "stream release waits for exact reply");
            expect(streams.erase(id) == 1, "explicit stream releases its exact remote owner");
        } else if (operation == ExecutionOpcode::kClose) {
            expect(!hostile && input.empty(), "uncertainty cannot manufacture normal close");
            std::erase_if(streams,
                          [&](const auto& entry) { return entry.second == request->connection; });
            ++closes;
        } else {
            expect(operation == ExecutionOpcode::kGetContextFlags && input.empty(),
                   "no directional copy substitution or unexpected command");
        }
        respond(*request, result);
    }
    client.get();
    expect(!held && glimmer_unified_copy_native_calls() == 0, "no held reply or native fallback");
    if (hostile) {
        expect(copies == 1 && imports == 1 && releases == 0 && closes == 0 &&
                   mappings.size() == 1 && streams.size() == (per_thread_hostile ? 1U : 0U),
               "isolated uncertainty retains actual mapping and prevents fake cleanup");
        server.stop();
        // This child intentionally retains uncertain frontend ownership. It is
        // containment evidence, not normal cleanup or a leak-free recovery claim.
        std::_Exit(EXIT_SUCCESS);
    }
    expect(devices.empty() && mappings.empty() && streams.empty() && imports == releases &&
               closes == 2 && copies == 34 && host_copies == 9 && pageable_copies == 25 &&
               prefix_queries == 9 && prefix_releases == 9 && streams_created == 3 &&
               per_thread_issues ==
                   std::vector<std::uint64_t>{52, 52, 52, 53, 53, 53, 52, 52, 52, 52, 52},
           "complete normal cleanup precedes exact command/classification ledger acceptance");
    server.stop();
}
}  // namespace

// A thread-local observation only: use the real sleep syscall unchanged. The
// two resource-destruction callers must actually enter a finite frontend wait;
// merely starting a future is not proof that its operation reached the pin.
extern "C" int nanosleep(const timespec* requested, timespec* remaining) {
    const auto saved_errno = errno;
    if (wait_observation && requested && requested->tv_sec == 0 && requested->tv_nsec == 50000)
        wait_observation->store(true);
    errno = saved_errno;
    return static_cast<int>(::syscall(SYS_nanosleep, requested, remaining));
}

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 3, "usage: unified-copy-test provider [hostile-case-1..14]");
        unsigned int hostile{};
        if (argc == 3) {
            const std::string_view text{argv[2]};
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), hostile);
            expect(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && hostile &&
                       hostile <= kHostileCases,
                   "bounded hostile receipt case");
        }
        run(argv[1], hostile);
        std::cout << "Public exact UVA copies, ownership and normal cleanup passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
