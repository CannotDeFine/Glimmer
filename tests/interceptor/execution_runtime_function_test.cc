#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>

#ifdef GLIMMER_RUNTIME_FUNCTION_PROVIDER
namespace {
std::atomic<unsigned int> native_consumers{};
}
extern "C" unsigned int glimmer_runtime_function_native_calls() {
    return native_consumers.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int) {
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
    if (!version) return cudaErrorInvalidValue;
    *version = CUDART_VERSION;
    return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaFuncGetAttributes(cudaFuncAttributes*, const void*) {
    ++native_consumers;
    return cudaErrorUnknown;
}
extern "C" cudaError_t CUDARTAPI cudaFuncSetAttribute(const void*, cudaFuncAttribute, int) {
    ++native_consumers;
    return cudaErrorUnknown;
}
extern "C" CUresult CUDAAPI cuFuncGetAttribute(int*, CUfunction_attribute, CUfunction) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction, CUfunction_attribute, int) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
#else
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <fatbinary_section.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// These declarations are the exact NVIDIA compiler registration ABI.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** __cudaRegisterFatBinary(void*);
extern "C" void __cudaRegisterFatBinaryEnd(void**);
extern "C" void __cudaUnregisterFatBinary(void**);
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                       dim3*, dim3*, int*);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" unsigned int glimmer_runtime_function_native_calls();

namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
constexpr unsigned int kLastCase = 33;
constexpr std::array<cudaFuncAttribute, 8> kSetters{
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaFuncAttributeClusterDimMustBeSet,
    cudaFuncAttributeRequiredClusterWidth,
    cudaFuncAttributeRequiredClusterHeight,
    cudaFuncAttributeRequiredClusterDepth,
    cudaFuncAttributeNonPortableClusterSizeAllowed,
    cudaFuncAttributeClusterSchedulingPolicyPreference};
// This mapping is independent of the frontend helper's enum switch.
constexpr std::array<std::uint32_t, 8> kDriverSelectors{8, 9, 10, 11, 12, 13, 14, 15};
constexpr std::array<int, 8> kSetterValues{INT32_MIN, -1, 0, 3, -5, INT32_MAX, 17, -19};

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
    expect(status == cudaSuccess, "owned Runtime metadata operation succeeds");
}
bool owned(void* address) {
    Dl_info information{};
    return address && ::dladdr(address, &information) && information.dli_fname &&
           std::string_view(information.dli_fname).ends_with("libglimmer_execution_interceptor.so");
}
struct Functions {
    decltype(&cudaFuncGetAttributes) get{};
    decltype(&cudaFuncSetAttribute) set{};
};
std::vector<Functions> routes(const char* provider_path) {
    void* provider = ::dlopen(provider_path, RTLD_NOW | RTLD_LOCAL);
    expect(provider, "load independent Runtime resource-consumer provider");
    std::vector<Functions> result{{&cudaFuncGetAttributes, &cudaFuncSetAttribute}};
    expect(owned(reinterpret_cast<void*>(result.front().get)) &&
               owned(reinterpret_cast<void*>(result.front().set)),
           "direct PLT exports resolve to controlled Runtime routes");
    for (void* handle : std::array{RTLD_DEFAULT, provider}) {
        void* get = ::dlsym(handle, "cudaFuncGetAttributes");
        void* set = ::dlsym(handle, "cudaFuncSetAttribute");
        expect(owned(get) && owned(set), "global and qualified Runtime lookups cannot escape");
        result.push_back({reinterpret_cast<decltype(&cudaFuncGetAttributes)>(get),
                          reinterpret_cast<decltype(&cudaFuncSetAttribute)>(set)});
    }
    // Save a separate explicit lookup and close its loader reference before use.
    result.push_back(result.back());
    expect(::dlclose(provider) == 0, "cached controlled entries outlive explicit provider close");
    return result;
}
struct Registration {
    int host{}, alias{};
    void** handle{};
    Registration() {
        alignas(8) std::array<std::byte, 24> image{};
        const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, 8};
        std::memcpy(image.data(), prefix.data(), sizeof(prefix));
        image[16] = std::byte{7};
        __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                    reinterpret_cast<const unsigned long long*>(image.data()),
                                    nullptr};
        handle = __cudaRegisterFatBinary(&wrapper);
        expect(handle, "manual compiler registration is CPU-only and returns a stable identity");
        char name[] = "runtime_attributes";
        for (const int* identity : std::array{&host, &alias})
            __cudaRegisterFunction(handle, reinterpret_cast<const char*>(identity), name, name, -1,
                                   nullptr, nullptr, nullptr, nullptr, nullptr);
        __cudaRegisterFatBinaryEnd(handle);
        check(cudaGetLastError());
        // The subsequent cold RPC must use the captured source and symbol name.
        image[16] = std::byte{99};
        name[0] = '!';
    }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;
    void unregister() {
        auto** value = handle;
        handle = nullptr;
        if (value) __cudaUnregisterFatBinary(value);
    }
    ~Registration() {
        if (handle) __cudaUnregisterFatBinary(handle);
    }
};
struct Control {
    std::atomic<unsigned int> calls{}, loads{}, functions{}, frontiers{}, unloads{}, closes{};
    std::atomic<bool> hold{}, entered{}, release{};
    std::atomic<void*> protected_output{};
};
void fill_sentinel(cudaFuncAttributes& output) {
    std::memset(&output, 0xa5, sizeof(output));
}
bool sentinel_bytes(std::span<const std::byte> bytes) {
    for (const auto byte : bytes)
        if (byte != std::byte{0xa5}) return false;
    return true;
}
bool unchanged(const cudaFuncAttributes& value) {
    return sentinel_bytes(std::as_bytes(std::span{&value, std::size_t{1}}));
}
bool complete_attributes(const cudaFuncAttributes& value) {
    const auto bytes = std::as_bytes(std::span{&value, std::size_t{1}});
    return value.sharedSizeBytes == 17 && value.constSizeBytes == INT32_MAX &&
           value.localSizeBytes == 0 && value.maxThreadsPerBlock == 1024 && value.numRegs == 37 &&
           value.ptxVersion == 86 && value.binaryVersion == 70 && value.cacheModeCA == -3 &&
           value.maxDynamicSharedSizeBytes == INT32_MIN && value.preferredShmemCarveout == -1 &&
           value.clusterDimMustBeSet == 1 && value.requiredClusterWidth == 2 &&
           value.requiredClusterHeight == -7 && value.requiredClusterDepth == 5 &&
           value.clusterSchedulingPolicyPreference == INT32_MAX &&
           value.nonPortableClusterSizeAllowed == -11 &&
           sentinel_bytes(bytes.subspan(offsetof(cudaFuncAttributes, reserved)));
}
void query(const Functions& api, const void* host) {
    cudaFuncAttributes output;
    fill_sentinel(output);
    check(api.get(&output, host));
    expect(
        complete_attributes(output),
        "all sixteen original fields survive while every reserved and padding byte is untouched");
}
void wait_entered(Control& control) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!control.entered) {
        if (std::chrono::steady_clock::now() >= deadline) {
            control.release = true;
            throw std::runtime_error("held metadata call did not reach owner");
        }
        std::this_thread::yield();
    }
}

void normal(const std::vector<Functions>& entries, Registration& registration, Control& control) {
    expect(cuInit(0) == CUDA_SUCCESS, "initialize Driver metadata without creating a context");
    expect(control.loads == 0 && control.calls == 0,
           "registration does not create remote resources");
    CUcontext current{};
    expect(cuCtxGetCurrent(&current) == CUDA_SUCCESS && !current,
           "compiler registration leaves the native context uninitialized");
    for (const auto& api : entries) {
        query(api, &registration.host);
        for (std::size_t index = 0; index < kSetters.size(); ++index)
            check(api.set(&registration.host, kSetters[index], kSetterValues[index]));
    }
    expect(control.loads == 1 && control.functions == 1 && control.frontiers == 1,
           "one cold materialization and no warm getter or setter drain");
    query(entries.front(), &registration.alias);
    expect(control.loads == 1 && control.functions == 2,
           "registered aliases reuse their exact module and coalesce the original function ID");
    cudaFuncAttributes output;
    fill_sentinel(output);
    const auto before = control.calls.load();
    int unregistered{};
    expect(entries.front().get(nullptr, &registration.host) == cudaErrorInvalidValue &&
               entries.front().get(reinterpret_cast<cudaFuncAttributes*>(1), &registration.host) ==
                   cudaErrorInvalidValue &&
               entries.front().get(&output, nullptr) == cudaErrorInvalidDeviceFunction &&
               entries.front().get(&output, &unregistered) == cudaErrorInvalidDeviceFunction &&
               entries.front().get(&output, reinterpret_cast<void*>(73)) ==
                   cudaErrorInvalidDeviceFunction &&
               entries.front().set(&registration.host, static_cast<cudaFuncAttribute>(7), -1) ==
                   cudaErrorInvalidValue &&
               entries.front().set(&registration.host, cudaFuncAttributeMax, -1) ==
                   cudaErrorInvalidValue &&
               entries.front().set(&unregistered, kSetters.front(), -1) ==
                   cudaErrorInvalidDeviceFunction,
           "invalid spans, unknown host symbols, raw remote IDs and unsupported selectors reject "
           "locally");
    expect(unchanged(output) && control.calls == before &&
               cudaPeekAtLastError() == cudaErrorInvalidDeviceFunction &&
               cudaGetLastError() == cudaErrorInvalidDeviceFunction &&
               cudaGetLastError() == cudaSuccess,
           "local errors preserve output, make no RPC and retain Runtime last-error semantics");
    check(cudaDeviceReset());
    expect(control.closes == 1, "reset closes the original owned context once");
    query(entries.front(), &registration.host);
    expect(control.loads == 2 && control.functions == 3,
           "surviving registration rematerializes only in the new owned context generation");
    registration.unregister();
    const auto retired = control.calls.load();
    expect(
        entries.front().get(&output, &registration.host) == cudaErrorInvalidDeviceFunction &&
            entries.front().set(&registration.alias, kSetters.front(), -1) ==
                cudaErrorInvalidDeviceFunction &&
            control.calls == retired,
        "unregister invalidates both host bindings without native fallback or another control RPC");
    check(cudaDeviceReset());
    expect(control.calls == 38 && control.unloads == 1 && control.closes == 2,
           "four lookup routes, alias and reset issue exactly once with known cleanup");
}

void publication(const Functions& api, Registration& registration, Control& control) {
    const auto page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    void* memory =
        ::mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(memory != MAP_FAILED, "create guarded aggregate output page");
    auto* output = static_cast<cudaFuncAttributes*>(memory);
    fill_sentinel(*output);
    control.protected_output = memory;
    expect(api.get(output, &registration.host) == cudaErrorInvalidValue,
           "output protection change after RPC forbids aggregate publication");
    expect(::mprotect(memory, page_size, PROT_READ | PROT_WRITE) == 0 && unchanged(*output),
           "failed late publication leaves the entire output unchanged");
    expect(::munmap(memory, page_size) == 0, "release guarded aggregate output page");
    query(api, &registration.host);
    expect(control.calls == 2, "local output failure does not poison proven remote ownership");
    registration.unregister();
    check(cudaDeviceReset());
}

void lifecycle(const Functions& api, Registration& registration, Control& control, bool setter) {
    query(api, &registration.host);
    control.hold = true;
    auto active = std::async(std::launch::async, [&] {
        if (setter)
            check(api.set(&registration.host, kSetters.front(), kSetterValues.front()));
        else
            query(api, &registration.host);
    });
    wait_entered(control);
    std::atomic<bool> unregister_started{};
    auto unload = std::async(std::launch::async, [&] {
        unregister_started = true;
        registration.unregister();
    });
    while (!unregister_started) std::this_thread::yield();
    const bool blocked =
        unload.wait_for(10ms) == std::future_status::timeout && control.unloads == 0;
    control.release = true;
    active.get();
    unload.get();
    expect(
        blocked && control.unloads == 1,
        "registration and module pins survive an issued getter/setter until its reply completes");
    cudaFuncAttributes output;
    fill_sentinel(output);
    expect(api.get(&output, &registration.alias) == cudaErrorInvalidDeviceFunction &&
               unchanged(output) && control.calls == 2,
           "unregister retires aliases only after the in-flight operation completes");
    check(cudaDeviceReset());
}

bool is_setter_case(unsigned int scenario) {
    return (scenario >= 20 && scenario <= 23) || scenario == 33;
}
bool is_clean_case(unsigned int scenario) {
    return scenario == 1 || scenario == 2 || scenario == 22 || scenario == 23 || scenario == 33;
}
void exercise(const char* provider, unsigned int scenario, Control& control) {
    const auto entries = routes(provider);
    if (scenario == 32) {
        int host{};
        cudaFuncAttributes output;
        fill_sentinel(output);
        check(cudaFree(nullptr));
        expect(
            entries.front().get(&output, &host) == cudaErrorInvalidDeviceFunction &&
                entries.front().set(&host, kSetters.front(), -1) ==
                    cudaErrorInvalidDeviceFunction &&
                unchanged(output) && control.calls == 0 && control.loads == 0 &&
                control.functions == 0,
            "initialized Runtime without any registration reports invalid device function locally");
        check(cudaDeviceReset());
        expect(control.closes == 1,
               "empty registration control still closes its real owned context");
        return;
    }
    Registration registration;
    if (!scenario) {
        normal(entries, registration, control);
        return;
    }
    if (scenario == 24) {
        publication(entries.front(), registration, control);
        return;
    }
    if (scenario == 25 || scenario == 26) {
        lifecycle(entries.front(), registration, control, scenario == 26);
        return;
    }
    cudaFuncAttributes output;
    fill_sentinel(output);
    const auto result =
        is_setter_case(scenario)
            ? entries.front().set(&registration.host, kSetters.front(), kSetterValues.front())
            : entries.front().get(&output, &registration.host);
    const auto expected = scenario == 1 || scenario == 22 ? cudaErrorInvalidValue
                          : scenario == 2                 ? cudaErrorNotSupported
                          : scenario == 3                 ? cudaErrorIncompatibleDriverContext
                          : scenario == 23                ? cudaErrorNotReady
                          : scenario == 27                ? cudaErrorInvalidDeviceFunction
                          : scenario == 28                ? cudaErrorInvalidKernelImage
                          : scenario == 33                ? cudaErrorNotPermitted
                                                          : cudaErrorUnknown;
    expect(result == expected && unchanged(output),
           "requested native/policy errors are exact and hostile receipts never publish partial "
           "fields");
    expect(cudaPeekAtLastError() == expected && cudaGetLastError() == expected &&
               cudaGetLastError() == cudaSuccess,
           "remote metadata failures retain Runtime last-error semantics");
    if (scenario == 27 || scenario == 28) {
        expect(control.calls == 0, "failed cold module/function lookup never requests attributes");
        registration.unregister();
        check(cudaDeviceReset());
    } else if (is_clean_case(scenario)) {
        query(entries.front(), &registration.host);
        expect(control.calls == 2, "clean SDK or policy error permits healthy reuse");
        registration.unregister();
        check(cudaDeviceReset());
    } else {
        const auto before = control.calls.load();
        expect(entries.front().get(&output, &registration.host) != cudaSuccess &&
                   unchanged(output) && control.calls == before,
               "malformed or quarantined reply makes the owner inaccessible without another RPC");
        // Unregister must not invent cleanup authority for the quarantined owner.
        registration.unregister();
        expect(control.unloads == 0, "unknown ownership cannot release the remote module");
    }
}

std::vector<std::byte> attribute_bytes() {
    // Independent fake-native values; intentionally not encoded through the new
    // aggregate codec so a shared field-order mistake cannot satisfy this oracle.
    ExecutionPayloadWriter output;
    expect(output.write_u64(17) && output.write_u64(INT32_MAX) && output.write_u64(0),
           "serialize independent size observations");
    for (const int value :
         std::array{1024, 37, 86, 70, -3, INT32_MIN, -1, 1, 2, -7, 5, INT32_MAX, -11})
        expect(output.write_u32(std::bit_cast<std::uint32_t>(value)),
               "serialize independent signed native observation");
    return output.take();
}
ExecutionWireReply response(unsigned int scenario, bool setter) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = 0;
    if (!setter) reply.data = attribute_bytes();
    switch (scenario) {
        case 1:
        case 22:
        case 33:
            reply.status = scenario == 33 ? CUDA_ERROR_NOT_PERMITTED : CUDA_ERROR_INVALID_VALUE;
            reply.original_status = reply.status;
            reply.data.clear();
            break;
        case 2:
        case 23:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = scenario == 2 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_NOT_READY;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 3:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.original_status.reset();
            reply.parameter_bytes = 1;
            reply.data.clear();
            break;
        case 4:
            reply.parameter_bytes = 1;
            break;
        case 5:
        case 20:
            reply.id = 1;
            break;
        case 6:
            reply.address = 1;
            break;
        case 7:
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
            break;
        case 8:
            reply.original_status.reset();
            break;
        case 9:
            reply.original_status = CUDA_ERROR_INVALID_VALUE;
            break;
        case 10:
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 11:
            reply.status = 1000;
            reply.original_status = 1000;
            break;
        case 12:
            reply.data.pop_back();
            break;
        case 13:
            reply.data.push_back(std::byte{0});
            break;
        case 14:
        case 29:
        case 30: {
            const std::size_t offset = scenario == 14 ? 0 : scenario == 29 ? 8 : 16;
            // 2^31 is representable in the wire size but not the original Driver int.
            for (std::size_t index = 0; index < 8; ++index)
                reply.data[offset + index] = std::byte{};
            reply.data[offset + 3] = std::byte{0x80};
            break;
        }
        case 15:
            reply.status = CUDA_ERROR_INVALID_VALUE;
            reply.original_status = reply.status;
            break;
        case 16:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.data.clear();
            break;
        case 17:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            reply.id = 1;
            reply.data.clear();
            break;
        case 18:
            reply.parameter_bytes = 2;
            break;
        case 21:
            reply.data = {std::byte{0}};
            break;
        case 31:
            reply.data.clear();
            break;
        default:
            break;
    }
    return reply;
}

void run(const char* provider, unsigned int scenario, const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated fake Runtime owner");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "observe real endpoint process identity");
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(), 1) ==
                   0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START",
                        std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
           "configure authenticated endpoint before any Runtime resource operation");
    Control control;
    auto client = std::async(std::launch::async, [&] { exercise(provider, scenario, control); });
    std::optional<ExecutionReceivedRequest> pending;
    ExecutionWireReply pending_reply;
    auto publish = [&](ExecutionReceivedRequest& request, const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {67, 71};
        const auto bytes = encode_execution_reply(reply);
        expect(bytes.has_value() && decode_execution_reply(*bytes).has_value(),
               "hostile semantic envelopes remain structurally valid transport messages");
        expect(server.reply(request.connection, {header, *bytes}),
               "publish exact fake-native receipt");
    };
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    try {
        while (client.wait_for(0ms) != std::future_status::ready) {
            expect(server.poll_for(1), "progress real authenticated Runtime IPC");
            if (pending && control.release) {
                publish(*pending, pending_reply);
                pending.reset();
            }
            if (auto request = server.take_request()) {
                ExecutionPayloadReader input(request->message.payload);
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.original_status = 0;
                switch (request->message.header.operation) {
                    case ExecutionOpcode::kHello:
                    case ExecutionOpcode::kClose:
                        if (request->message.header.operation == ExecutionOpcode::kClose)
                            ++control.closes;
                        break;
                    case ExecutionOpcode::kCaptureFrontier:
                        ++control.frontiers;
                        reply.id = 91;
                        reply.original_status.reset();
                        break;
                    case ExecutionOpcode::kQueryFrontier:
                    case ExecutionOpcode::kReleaseFrontier:
                        reply.original_status.reset();
                        break;
                    case ExecutionOpcode::kLoadBinary: {
                        std::span<const std::byte> bytes;
                        expect(input.read_blob(bytes) && input.empty() && bytes.size() == 24 &&
                                   bytes[16] == std::byte{7},
                               "cold lookup uses immutable registered image rather than caller "
                               "memory");
                        ++control.loads;
                        reply.id = 41;
                        if (scenario == 28) {
                            reply.id = 0;
                            reply.status = CUDA_ERROR_INVALID_IMAGE;
                            reply.original_status = reply.status;
                        }
                        break;
                    }
                    case ExecutionOpcode::kGetFunction: {
                        std::uint64_t module{};
                        std::string_view name;
                        expect(input.read_u64(module) && module == 41 && input.read_string(name) &&
                                   name == "runtime_attributes" && input.empty(),
                               "cold function resolves only its owned copied registration name");
                        ++control.functions;
                        reply.id = 73;
                        if (scenario == 27) {
                            reply.id = 0;
                            reply.status = CUDA_ERROR_NOT_FOUND;
                            reply.original_status = reply.status;
                        }
                        break;
                    }
                    case ExecutionOpcode::kFunctionControl: {
                        std::uint32_t operation{}, argument{}, value{};
                        std::uint64_t function{};
                        expect(request->message.payload.size() == 20 && input.read_u32(operation) &&
                                   input.read_u64(function) && input.read_u32(argument) &&
                                   input.read_u32(value) && input.empty() && function == 73,
                               "Runtime control uses the exact twenty-byte owned-function message");
                        const bool setter = operation == 2;
                        if (setter) {
                            bool matches{};
                            for (std::size_t index = 0; index < kDriverSelectors.size(); ++index)
                                matches =
                                    matches ||
                                    (argument == kDriverSelectors[index] &&
                                     value == std::bit_cast<std::uint32_t>(kSetterValues[index]));
                            expect(matches,
                                   "all eight selectors preserve signed values and exact Driver "
                                   "mapping");
                        } else {
                            expect(
                                operation == 4 && argument == 0 && value == 0,
                                "aggregate query has canonical operation4 and zero unused fields");
                        }
                        const auto number = ++control.calls;
                        if (scenario == 19 && number == 1) {
                            server.close_peer(request->connection);
                            continue;
                        }
                        reply = response(number == 1 ? scenario : 0, setter);
                        if (scenario == 24 && number == 1) {
                            const auto bytes = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
                            expect(
                                ::mprotect(control.protected_output.load(), bytes, PROT_NONE) == 0,
                                "change output protection only after exact request arrives");
                        }
                        if (control.hold) {
                            control.hold = false;
                            control.entered = true;
                            pending = std::move(request);
                            pending_reply = std::move(reply);
                            continue;
                        }
                        break;
                    }
                    case ExecutionOpcode::kUnloadModule: {
                        std::uint64_t module{};
                        expect(input.read_u64(module) && module == 41 && input.empty(),
                               "cleanup uses the exact remotely owned module");
                        ++control.unloads;
                        break;
                    }
                    default:
                        throw std::runtime_error("unexpected native fallback or Runtime request");
                }
                publish(*request, reply);
            }
            expect(std::chrono::steady_clock::now() < deadline, "bounded Runtime function fixture");
        }
        client.get();
    } catch (...) {
        control.release = true;
        server.stop();
        throw;
    }
    server.stop();
    expect(glimmer_runtime_function_native_calls() == 0,
           "no caller-native Runtime or Driver consumer receives a registered or remote token");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 4, "provider path and optional isolated case required");
        if (argc == 2) {
            for (unsigned int scenario = 0; scenario <= kLastCase; ++scenario) {
                const auto text = std::to_string(scenario);
                const auto child = ::fork();
                expect(child >= 0, "create isolated Runtime receipt control");
                if (!child) {
                    ::execl(argv[0], argv[0], argv[1], "--case", text.c_str(), nullptr);
                    _exit(127);
                }
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           WEXITSTATUS(status) == 0,
                       "isolated Runtime attribute control failed");
            }
            return EXIT_SUCCESS;
        }
        expect(std::string_view(argv[2]) == "--case", "explicit isolated case selector");
        const auto scenario = static_cast<unsigned int>(std::stoul(argv[3]));
        expect(scenario <= kLastCase, "bounded Runtime attribute scenario number");
        char pattern[] = "/tmp/glimmer-runtime-function-XXXXXX";
        const auto* directory = ::mkdtemp(pattern);
        expect(directory, "create private Runtime endpoint directory");
        try {
            run(argv[1], scenario, std::string(directory) + "/service.sock");
        } catch (...) {
            (void)::rmdir(directory);
            throw;
        }
        expect(::rmdir(directory) == 0, "remove completed Runtime endpoint directory");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
