#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);

#ifdef GLIMMER_POINTER_METADATA_PROVIDER
namespace {
std::atomic<unsigned int> consumers{};
std::atomic<bool> initialized{};
std::atomic<unsigned int> initialization_calls{};
std::atomic<CUresult> initialization_status{CUDA_SUCCESS};
}  // namespace
extern "C" unsigned int glimmer_pointer_native_calls() {
    return consumers.load();
}
extern "C" unsigned int glimmer_pointer_initialization_calls() {
    return initialization_calls.load();
}
extern "C" void glimmer_pointer_initialization_status(CUresult value) {
    initialization_status = value;
}
extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    ++initialization_calls;
    const auto status = initialization_status.load();
    if (status != CUDA_SUCCESS) return status;
    initialized = true;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGet(CUdevice* output, int ordinal) {
    if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal) return CUDA_ERROR_INVALID_DEVICE;
    *output = 0;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGetCount(int* output) {
    if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    *output = 1;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGetAttribute(int* output, CUdevice_attribute attribute,
                                                 CUdevice device) {
    if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!output || attribute != CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        return CUDA_ERROR_INVALID_VALUE;
    if (device) return CUDA_ERROR_INVALID_DEVICE;
    *output = 80;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
    if (!version) return cudaErrorInvalidValue;
    *version = 12060;
    return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void*, CUpointer_attribute, CUdeviceptr) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" cudaError_t CUDARTAPI cudaPointerGetAttributes(cudaPointerAttributes*, const void*) {
    ++consumers;
    return cudaErrorUnknown;
}
extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr*, std::size_t) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void**, std::size_t, unsigned int) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void*) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule*, const void*) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr*, std::size_t*, CUmodule,
                                                 const char*) {
    ++consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void**, std::size_t) {
    ++consumers;
    return cudaErrorUnknown;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void*) {
    ++consumers;
    return cudaErrorUnknown;
}
extern "C" cudaError_t CUDARTAPI cudaDeviceReset() {
    ++consumers;
    return cudaErrorUnknown;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA lookup ABI.
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    if (!name || !output || flags) return CUDA_ERROR_INVALID_VALUE;
    *output = std::string_view(name) == "cuPointerGetAttribute"
                  ? reinterpret_cast<void*>(&cuPointerGetAttribute)
                  : nullptr;
    if (status)
        *status = *output ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cuGetProcAddress_v2(name, output, version, flags, nullptr);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#else
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
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

extern "C" unsigned int glimmer_pointer_native_calls();
extern "C" unsigned int glimmer_pointer_initialization_calls();
extern "C" void glimmer_pointer_initialization_status(CUresult);
namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
constexpr unsigned int kLastCase = 57;
constexpr std::uint64_t kDevice = 0x310000000000ULL;
constexpr std::uint64_t kGlobal = 0x320000000000ULL;
constexpr std::uint64_t kHost = 0x330000000000ULL;
constexpr std::uint64_t kGuard = 0x538be91745c2a06fULL;
constexpr std::uint64_t kDeviceSentinel = 0x1234abcddcba4321ULL;
constexpr int kOrdinalSentinel = 0x12345678;
constexpr const char* kSource =
    ".version 7.0\n.target sm_70\n.address_size 64\n.visible .global .align 4 .u32 global[8];\n";

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult result) {
    expect(result == CUDA_SUCCESS, "owned Driver operation succeeds");
}
void check(cudaError_t result) {
    expect(result == cudaSuccess, "owned Runtime operation succeeds");
}
void* pointer(std::uint64_t value) {
    // Test numeric identities are never dereferenced except an actual mapped host allocation.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value));
}
bool owned(void* value) {
    Dl_info information{};
    return value && ::dladdr(value, &information) && information.dli_fname &&
           std::string_view(information.dli_fname).ends_with("libglimmer_execution_interceptor.so");
}
struct Routes {
    std::vector<decltype(&cuPointerGetAttribute)> driver;
    std::vector<decltype(&cudaPointerGetAttributes)> runtime;
    std::array<unsigned int (*)(), 2> native_counters{};
    void check_native() const {
        for (const auto counter : native_counters)
            expect(counter && counter() == 0,
                   "neither Driver nor Runtime native consumer saw caller resources");
    }
};
struct Providers {
    const char* runtime;
    const char* driver;
};
Routes routes(Providers paths) {
    void* provider = ::dlopen(paths.runtime, RTLD_NOW | RTLD_LOCAL);
    void* driver_provider = ::dlopen(paths.driver, RTLD_NOW | RTLD_LOCAL);
    expect(provider && driver_provider, "load independent matching Runtime and Driver providers");
    Routes result{{&cuPointerGetAttribute}, {&cudaPointerGetAttributes}};
    result.native_counters = {
        reinterpret_cast<unsigned int (*)()>(::dlsym(provider, "glimmer_pointer_native_calls")),
        reinterpret_cast<unsigned int (*)()>(
            ::dlsym(driver_provider, "glimmer_pointer_native_calls"))};
    expect(owned(reinterpret_cast<void*>(result.driver.front())) &&
               owned(reinterpret_cast<void*>(result.runtime.front())),
           "direct PLT routes owned");
    for (const bool explicit_handle : std::array{false, true}) {
        auto* driver =
            ::dlsym(explicit_handle ? driver_provider : RTLD_DEFAULT, "cuPointerGetAttribute");
        auto* runtime =
            ::dlsym(explicit_handle ? provider : RTLD_DEFAULT, "cudaPointerGetAttributes");
        expect(owned(driver) && owned(runtime),
               "global and qualified pointer routes are intercepted");
        result.driver.push_back(reinterpret_cast<decltype(&cuPointerGetAttribute)>(driver));
        result.runtime.push_back(reinterpret_cast<decltype(&cudaPointerGetAttributes)>(runtime));
    }
    for (unsigned int kind = 0; kind != 3; ++kind) {
        void* entry{};
        if (kind == 0)
            check(cuGetProcAddress("cuPointerGetAttribute", &entry, 12060, 0));
        else if (kind == 1) {
            CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            check(cuGetProcAddress_v2("cuPointerGetAttribute", &entry, 12060, 0, &status));
            expect(status == CU_GET_PROC_ADDRESS_SUCCESS, "Driver cached lookup status");
        } else {
            cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
            check(cudaGetDriverEntryPointByVersion("cuPointerGetAttribute", &entry, 12060, 0,
                                                   &status));
            expect(status == cudaDriverEntryPointSuccess, "Runtime-to-Driver cached lookup status");
        }
        expect(owned(entry), "proc-address metadata resource consumer cannot escape");
        result.driver.push_back(reinterpret_cast<decltype(&cuPointerGetAttribute)>(entry));
    }
    result.runtime.push_back(result.runtime.back());
    expect(::dlclose(provider) == 0 && ::dlclose(driver_provider) == 0,
           "cached pointers survive both explicit provider-reference closes");
    return result;
}

// This oracle deliberately uses numeric fields and independent wire construction,
// not the new pointer codec or frontend classification helpers.
struct Expected {
    std::uint32_t domain{}, selector{}, kind{};
    std::uint64_t query{}, source{}, allocation{}, offset{};
};
struct Control {
    std::mutex mutex;
    std::deque<Expected> expected;
    std::atomic<std::uint64_t> latest_application{}, host_owner{};
    std::atomic<unsigned int> queries{}, application_hellos{}, metadata_hellos{}, releases{},
        unloads{}, closes{}, host_releases{}, allocations{};
    std::atomic<bool> hold{}, hold_cleanup{}, entered{}, release{};
    std::atomic<void*> protected_output{};
    void next(Expected value) {
        std::lock_guard lock(mutex);
        expected.push_back(value);
    }
    Expected take() {
        std::lock_guard lock(mutex);
        expect(!expected.empty(), "no unplanned or replayed pointer metadata request");
        const auto value = expected.front();
        expected.pop_front();
        return value;
    }
    bool empty() {
        std::lock_guard lock(mutex);
        return expected.empty();
    }
};
struct Location {
    std::uint64_t bits{}, query{}, source{}, allocation{}, offset{};
    std::uint32_t kind{};
    bool unavailable{};
};
struct GuardedRuntime {
    std::uint64_t before{kGuard};
    cudaPointerAttributes value;
    std::uint64_t after{kGuard};
    GuardedRuntime() {
        std::memset(&value, 0xa5, sizeof(value));
    }
};
void expect_runtime(const GuardedRuntime& actual, const cudaPointerAttributes& expected) {
    expect(actual.before == kGuard && actual.after == kGuard &&
               std::memcmp(&actual.value, &expected, sizeof(expected)) == 0,
           "exact24-byte Runtime effects preserve all inactive bytes and both guards");
}
void runtime_query(Control& control, decltype(&cudaPointerGetAttributes) api, Location location) {
    control.next({2, 0, location.kind, location.query, location.source, location.allocation,
                  location.offset});
    GuardedRuntime output;
    check(api(&output.value, pointer(location.bits)));
    cudaPointerAttributes expected{};
    expected.type = location.kind < 2    ? cudaMemoryTypeUnregistered
                    : location.kind == 2 ? cudaMemoryTypeDevice
                                         : cudaMemoryTypeHost;
    expected.device = location.kind < 2 ? -2 : 0;
    expected.devicePointer =
        location.kind >= 2 && !location.unavailable ? pointer(location.bits) : nullptr;
    expected.hostPointer = location.kind == 2 ? nullptr : pointer(location.bits);
    expect_runtime(output, expected);
}
void driver_query(Control& control, decltype(&cuPointerGetAttribute) api, Location location) {
    for (const std::uint32_t selector : std::array{3U, 9U}) {
        control.next({1, selector, location.kind, location.query, location.source,
                      location.allocation, location.offset});
        struct Output {
            std::uint64_t before{kGuard}, value{kDeviceSentinel}, after{kGuard};
        } output;
        const auto status =
            api(&output.value, static_cast<CUpointer_attribute>(selector), location.bits);
        auto expected = kDeviceSentinel;
        CUresult expected_status = CUDA_SUCCESS;
        if (location.kind < 2)
            expected_status = CUDA_ERROR_INVALID_VALUE;
        else if (selector == 9) {
            const int ordinal = 0;
            std::memcpy(&expected, &ordinal, sizeof(ordinal));
        } else if (location.unavailable) {
            expected_status =
                location.query ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_CONTEXT;
            expected = 0;
        } else
            expected = location.bits;
        expect(status == expected_status && output.value == expected && output.before == kGuard &&
                   output.after == kGuard,
               "Driver exact4/8-byte effects distinguish unchanged error and error-writing-NULL");
    }
}
void all_queries(Control& control, const Routes& entries, Location value) {
    for (const auto api : entries.driver) driver_query(control, api, value);
    for (const auto api : entries.runtime) runtime_query(control, api, value);
}
Location unknown(std::uint64_t bits, std::uint64_t query = 0) {
    return {bits, query, 0, 0, 0, bits ? 1U : 0U, false};
}
void normal(Control& control, Providers paths) {
    CUdevice device_query = 17;
    int count_query = 19, attribute_query = 23;
    expect(cuDeviceGet(&device_query, 0) == CUDA_ERROR_NOT_INITIALIZED && device_query == 17 &&
               cuDeviceGetCount(&count_query) == CUDA_ERROR_NOT_INITIALIZED && count_query == 19 &&
               cuDeviceGetAttribute(&attribute_query, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    0) == CUDA_ERROR_NOT_INITIALIZED &&
               attribute_query == 23 && glimmer_pointer_initialization_calls() == 0,
           "device queries cannot initialize a process before successful Runtime/Driver ownership");
    std::uint64_t output = kDeviceSentinel;
    expect(cuPointerGetAttribute(&output, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, 0) ==
                   CUDA_ERROR_NOT_INITIALIZED &&
               output == kDeviceSentinel && control.queries == 0,
           "uninitialized Driver status is observed, not guessed or replaced with INVALID_CONTEXT");
    runtime_query(control, &cudaPointerGetAttributes, unknown(0));
    expect(control.application_hellos == 0 && control.metadata_hellos == 1,
           "Runtime unknown query opens CPU PMD1 without forcing a primary context");
    // Successful remote Runtime metadata initialized the logical Driver only.
    // Bridge caller-side metadata without creating any caller CUDA context.
    glimmer_pointer_initialization_status(CUDA_ERROR_NO_DEVICE);
    expect(cuDeviceGet(&device_query, 0) == CUDA_ERROR_NO_DEVICE && device_query == 17 &&
               glimmer_pointer_initialization_calls() == 1,
           "exact caller metadata initialization failure is returned with unchanged output");
    glimmer_pointer_initialization_status(CUDA_SUCCESS);
    check(cuDeviceGet(&device_query, 0));
    check(cuDeviceGetCount(&count_query));
    check(cuDeviceGetAttribute(&attribute_query, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0));
    expect(device_query == 0 && count_query == 1 && attribute_query == 80 &&
               glimmer_pointer_initialization_calls() == 2 &&
               cuDeviceGet(&device_query, 1) == CUDA_ERROR_INVALID_DEVICE && device_query == 0 &&
               cuDeviceGetCount(nullptr) == CUDA_ERROR_INVALID_VALUE,
           "one successful metadata initialization preserves all native query results/errors");
    const auto cached_device =
        reinterpret_cast<decltype(&cuDeviceGet)>(::dlsym(RTLD_DEFAULT, "cuDeviceGet"));
    expect(cached_device == &cuDeviceGet, "qualified cached device query remains controlled");
    check(cached_device(&device_query, 0));
    CUcontext no_context = reinterpret_cast<CUcontext>(1);
    check(cuCtxGetCurrent(&no_context));
    expect(!no_context && control.application_hellos == 0 &&
               glimmer_pointer_initialization_calls() == 2,
           "metadata bridge neither activates a context nor repeats native initialization");
    driver_query(control, &cuPointerGetAttribute, unknown(0));
    check(cuInit(0));
    const auto entries = routes(paths);
    std::array<std::uint32_t, 4> host_values{19, 23, 29, 31};
    void* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "protected unknown pointee is never read");
    for (const auto bits : std::array<std::uint64_t, 5>{
             0, 1, 0x0200000000001234ULL, reinterpret_cast<std::uintptr_t>(host_values.data()),
             reinterpret_cast<std::uintptr_t>(inaccessible)})
        all_queries(control, entries, unknown(bits));
    expect(host_values == std::array<std::uint32_t, 4>{19, 23, 29, 31} &&
               ::munmap(inaccessible, 4096) == 0 && control.application_hellos == 0,
           "unknown pointer bits never become pointee reads or application ownership");

    CUcontext first{};
    check(cuCtxCreate(&first, 0, 0));
    const auto source = control.latest_application.load();
    CUdeviceptr device{};
    check(cuMemAlloc(&device, 256));
    expect(device == kDevice, "independent device allocation identity");
    for (const auto offset : std::array<std::uint64_t, 3>{0, 17, 255})
        all_queries(control, entries, {device + offset, source, source, 7, offset, 2, false});
    all_queries(control, entries, unknown(device + 256, source));
    all_queries(control, entries, unknown(device + 257, source));
    CUmodule module{};
    check(cuModuleLoadData(&module, kSource));
    CUdeviceptr global{}, alias{};
    std::size_t size{}, alias_size{};
    check(cuModuleGetGlobal(&global, &size, module, "global"));
    check(cuModuleGetGlobal(&alias, &alias_size, module, "alias"));
    expect(global == kGlobal && alias == global && size == 32 && alias_size == 32,
           "module aliases coalesce exact original global storage");
    all_queries(control, entries, {global + 4, source, source, 31, 4, 2, false});
    all_queries(control, entries, unknown(global + 32, source));
    void* host{};
    check(cuMemHostAlloc(&host, 256, CU_MEMHOSTALLOC_DEVICEMAP));
    expect(host == pointer(kHost), "mapped host is real shared backing at synthetic public UVA");
    std::memset(host, 0x37, 256);
    const auto host_source = control.host_owner.load();
    all_queries(control, entries, {kHost + 17, source, host_source, 61, 17, 3, false});
    all_queries(control, entries, unknown(kHost + 256, source));
    all_queries(control, entries, unknown(kHost + 1024, source));
    CUcontext second{};
    check(cuCtxCreate(&second, 0, 0));
    const auto other = control.latest_application.load();
    expect(source != other, "two real authenticated application sessions");
    all_queries(control, entries, {device, other, source, 7, 0, 2, false});
    all_queries(control, entries, {global, other, source, 31, 0, 2, true});
    all_queries(control, entries, {kHost, other, host_source, 61, 0, 3, false});
    // Each successful context creation pushes one entry. Detaching B exposes A;
    // detach both before claiming that the query has no current context.
    check(cuCtxSetCurrent(nullptr));
    check(cuCtxSetCurrent(nullptr));
    CUcontext current = first;
    check(cuCtxGetCurrent(&current));
    expect(!current, "both created contexts are detached before no-current queries");
    all_queries(control, entries, {device, 0, source, 7, 0, 2, true});
    all_queries(control, entries, {global, 0, source, 31, 0, 2, true});
    all_queries(control, entries, {kHost, 0, host_source, 61, 0, 3, true});
    current = first;
    check(cuCtxGetCurrent(&current));
    expect(!current && control.application_hellos == 2,
           "no-current queries neither activate source nor create primary context");
    check(cuCtxSetCurrent(first));
    check(cuMemFree(device));
    all_queries(control, entries, unknown(device, source));
    check(cuMemAlloc(&device, 256));
    all_queries(control, entries, {device, source, source, 8, 0, 2, false});
    check(cuMemFree(device));
    for (std::size_t index = 0; index != 256; ++index)
        expect(static_cast<unsigned char*>(host)[index] == 0x37,
               "metadata leaves host payload intact");
    check(cuMemFreeHost(host));
    all_queries(control, entries, unknown(kHost, source));
    check(cuModuleUnload(module));
    all_queries(control, entries, unknown(global, source));
    check(cuCtxDestroy(second));
    check(cuCtxDestroy(first));
    expect(control.releases == 2 && control.unloads == 1 && control.host_releases == 1,
           "all known native resource identities release exactly once");
    entries.check_native();
}

std::vector<std::byte> effect_bytes(std::array<std::uint32_t, 5> fields) {
    ExecutionPayloadWriter output;
    for (const auto field : fields)
        expect(output.write_u32(field), "independent effect wire bytes");
    return output.take();
}
ExecutionWireReply observed(const Expected& request) {
    ExecutionWireReply result;
    result.disposition = ExecutionWireDisposition::kComplete;
    result.original_status = 0;
    const bool unknown_pointer = request.kind < 2;
    const bool unavailable =
        !request.query || (request.allocation == 31 && request.query != request.source);
    std::array<std::uint32_t, 5> fields{};
    if (request.domain == 2) {
        fields = {15,
                  unknown_pointer     ? 0U
                  : request.kind == 2 ? 2U
                                      : 1U,
                  unknown_pointer ? 0xfffffffeU : 0U, !unknown_pointer && !unavailable ? 1U : 0U,
                  request.kind == 0 || request.kind == 2 ? 0U : 1U};
    } else if (unknown_pointer) {
        result.status = 1;
        result.original_status = 1;
    } else if (request.selector == 9) {
        fields[0] = 2;
    } else {
        fields[0] = 4;
        fields[3] = unavailable ? 0U : 1U;
        if (unavailable) {
            result.status = request.query ? 1 : 201;
            result.original_status = result.status;
        }
    }
    result.data = effect_bytes(fields);
    return result;
}

ExecutionWireReply altered(ExecutionWireReply reply, unsigned int scenario) {
    switch (scenario) {
        case 20:
            reply.id = 1;
            break;
        case 21:
            reply.address = 1;
            break;
        case 22:
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
            break;
        case 23:
            reply.parameter_bytes = 2;
            break;
        case 24:
            reply.parameter_bytes = 1;
            break;
        case 25:
            reply.original_status.reset();
            break;
        case 26:
            reply.original_status = 1;
            break;
        case 27:
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 28:
            reply.status = 10001;
            reply.original_status = 10001;
            break;
        case 29:
            reply.data.pop_back();
            break;
        case 30:
            reply.data.push_back(std::byte{});
            break;
        case 31:
            reply.data[0] = std::byte{16};
            break;
        case 32:
            reply.data = effect_bytes({14, 2, 0, 1, 0});
            break;
        case 33:
            reply.data[12] = std::byte{2};
            break;
        case 34:
            reply.data[4] = std::byte{4};
            break;
        case 35:
            reply.data = effect_bytes({15, 2, UINT32_MAX, 1, 0});
            break;
        case 36:
            reply.data[4] = std::byte{};
            break;
        case 37:
            reply.data[16] = std::byte{1};
            break;
        case 38:
            reply.data = effect_bytes({14, 0, 0, 1, 0});
            break;
        case 39:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 1;
            reply.original_status.reset();
            break;
        case 40:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 41:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 1;
            reply.original_status = 1;
            reply.data.clear();
            break;
        case 42:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.id = 1;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 44:
            reply.data = effect_bytes({2, 0, 0, 0, 0});
            break;
        case 45:
            reply.data = effect_bytes({4, 0, 0, 1, 0});
            break;
        case 46:
            reply.data = effect_bytes({15, 2, 0, 0x12345678, 0});
            break;
        case 47:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 999;
            reply.parameter_bytes = 1;
            reply.original_status = 98;
            reply.data.clear();
            break;
        case 48:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 1;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 49:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 600;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 50:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = -1;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 51:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 600;
            reply.data.clear();
            break;
        case 52:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 600;
            reply.original_status.reset();
            reply.parameter_bytes = 2;
            reply.data.clear();
            break;
        case 53:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 600;
            reply.original_status.reset();
            break;
        case 54:
            reply.status = 1000;
            reply.original_status = 1000;
            break;
        case 55:
            reply.status = 10000;
            reply.original_status = 10000;
            reply.data = effect_bytes({0, 0, 0, 0, 0});
            break;
        default:
            break;
    }
    return reply;
}

struct Allocation {
    CUcontext context{};
    CUdeviceptr device{};
    std::uint64_t session{};
    explicit Allocation(Control& control, bool primary = false) {
        if (primary) {
            void* memory{};
            check(cudaMalloc(&memory, 256));
            device = reinterpret_cast<CUdeviceptr>(memory);
            check(cuCtxGetCurrent(&context));
        } else {
            check(cuCtxCreate(&context, 0, 0));
            check(cuMemAlloc(&device, 256));
        }
        session = control.latest_application;
    }
    Location location() const {
        return {device, session, session, 7, 0, 2, false};
    }
    void close() {
        check(cuMemFree(device));
        check(cuCtxDestroy(context));
    }
};
void failure_effects(Control& control, const Routes& entries, bool driver) {
    Allocation allocation(control);
    if (driver) {
        control.next({1, 9, 2, allocation.session, allocation.session, 7, 0});
        struct Output {
            int value{kOrdinalSentinel};
            std::uint32_t tail{0x795abc41};
        } output;
        expect(
            entries.driver.front()(&output.value, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                   allocation.device) == CUDA_ERROR_INVALID_VALUE &&
                output.value == -17 && output.tail == 0x795abc41,
            "native Driver error can publish a raw negative ordinal without an eight-byte write");
    } else {
        for (std::uint32_t mask = 0; mask != 16; ++mask) {
            control.next({2, 0, 2, allocation.session, allocation.session, 7, 0});
            GuardedRuntime output;
            cudaPointerAttributes expected;
            std::memcpy(&expected, &output.value, sizeof(expected));
            const std::int32_t raw_type = mask & 1U ? INT32_MIN : 0;
            const std::int32_t raw_device = mask & 2U ? -17 : 0;
            if (mask & 1U) std::memcpy(&expected.type, &raw_type, sizeof(raw_type));
            if (mask & 2U) std::memcpy(&expected.device, &raw_device, sizeof(raw_device));
            if (mask & 4U) expected.devicePointer = pointer(allocation.device);
            if (mask & 8U) expected.hostPointer = nullptr;
            expect(entries.runtime.front()(&output.value, pointer(allocation.device)) ==
                       cudaErrorInvalidDeviceFunction,
                   "Runtime original98 is not translated through a Driver status mapping");
            expect_runtime(output, expected);
            expect(cudaPeekAtLastError() == cudaErrorInvalidDeviceFunction &&
                       cudaGetLastError() == cudaErrorInvalidDeviceFunction &&
                       cudaGetLastError() == cudaSuccess,
                   "native Runtime error remains last-error");
        }
    }
    runtime_query(control, entries.runtime.front(), allocation.location());
    allocation.close();
}
void wait_entered(Control& control) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!control.entered) {
        if (std::chrono::steady_clock::now() >= deadline) {
            control.release = true;
            throw std::runtime_error("held pointer request did not reach owner");
        }
        std::this_thread::yield();
    }
}
void lifecycle(Control& control, const Routes& entries, unsigned int scenario) {
    Allocation allocation(control, scenario == 7);
    Location location = allocation.location();
    CUcontext query_context = allocation.context;
    if (scenario == 12 || scenario == 13) {
        check(cuCtxCreate(&query_context, 0, 0));
        location.query = control.latest_application;
        expect(location.query != location.source, "cross-owner query and source sessions differ");
    }
    CUmodule module{};
    void* host{};
    if (scenario == 4) {
        check(cuModuleLoadData(&module, kSource));
        CUdeviceptr global{};
        std::size_t bytes{};
        check(cuModuleGetGlobal(&global, &bytes, module, "global"));
        location = {global, allocation.session, allocation.session, 31, 0, 2, false};
    }
    if (scenario == 6) {
        check(cuMemHostAlloc(&host, 256, CU_MEMHOSTALLOC_DEVICEMAP));
        location = {kHost, allocation.session, control.host_owner, 61, 0, 3, false};
    }
    control.hold = true;
    auto querying = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(query_context));
        runtime_query(control, entries.runtime.front(), location);
    });
    wait_entered(control);
    const auto frees_before = control.releases.load();
    const auto unloads_before = control.unloads.load();
    const auto closes_before = control.closes.load();
    const auto hosts_before = control.host_releases.load();
    std::atomic<bool> started{};
    auto retiring = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(allocation.context));
        started = true;
        if (scenario == 3)
            check(cuMemFree(allocation.device));
        else if (scenario == 4)
            check(cuModuleUnload(module));
        else if (scenario == 5 || scenario == 12)
            check(cuCtxDestroy(allocation.context));
        else if (scenario == 13)
            check(cuCtxDestroy(query_context));
        else if (scenario == 6)
            check(cuMemFreeHost(host));
        else
            check(cudaDeviceReset());
    });
    while (!started) std::this_thread::yield();
    const bool blocked = retiring.wait_for(20ms) == std::future_status::timeout &&
                         control.releases == frees_before && control.unloads == unloads_before &&
                         control.closes == closes_before && control.host_releases == hosts_before;
    control.release = true;
    querying.get();
    retiring.get();
    expect(blocked,
           "free/unload/context-close/host-free/reset cannot pass issued query publication");
    if (scenario == 3)
        check(cuCtxDestroy(allocation.context));
    else if (scenario == 4 || scenario == 6)
        allocation.close();
    else if (scenario == 12)
        check(cuCtxDestroy(query_context));
    else if (scenario == 13) {
        check(cuCtxSetCurrent(allocation.context));
        allocation.close();
    }
    if (scenario != 5 && scenario != 7) {
        check(cuCtxSetCurrent(nullptr));
        runtime_query(control, entries.runtime.front(), unknown(location.bits));
    }
}
void publication(Control& control, const Routes& entries, unsigned int scenario) {
    Allocation allocation(control);
    const std::size_t width = scenario == 8 ? 24 : scenario == 9 ? 8 : 4;
    auto* page = static_cast<std::byte*>(
        ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    expect(page != MAP_FAILED, "create late-protected exact output region");
    std::memset(page, 0xa5, 4096);
    control.protected_output = page;
    control.next({scenario == 8 ? 2U : 1U,
                  scenario == 8   ? 0U
                  : scenario == 9 ? 3U
                                  : 9U,
                  2, allocation.session, allocation.session, 7, 0});
    const int result =
        scenario == 8
            ? static_cast<int>(entries.runtime.front()(
                  reinterpret_cast<cudaPointerAttributes*>(page), pointer(allocation.device)))
            : static_cast<int>(entries.driver.front()(page,
                                                      scenario == 9
                                                          ? CU_POINTER_ATTRIBUTE_DEVICE_POINTER
                                                          : CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                                      allocation.device));
    expect(result == 1, "late invalid output span returns local invalid-value without publication");
    expect(::mprotect(page, 4096, PROT_READ | PROT_WRITE) == 0,
           "restore page only after metadata call returns");
    for (std::size_t index = 0; index != width; ++index)
        expect(page[index] == std::byte{0xa5}, "late failure preserves complete output bytes");
    expect(::munmap(page, 4096) == 0, "release late-publication test page");
    runtime_query(control, entries.runtime.front(), allocation.location());
    allocation.close();
}
void local_and_exact_spans(Control& control, const Routes& entries) {
    Allocation allocation(control);
    auto* page = static_cast<std::byte*>(
        ::mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    expect(page != MAP_FAILED && ::mprotect(page + 4096, 4096, PROT_NONE) == 0,
           "exact end-of-page output guard");
    std::memset(page, 0xa5, 4096);
    const auto before = control.queries.load();
    expect(entries.runtime.front()(nullptr, pointer(allocation.device)) == cudaErrorInvalidValue &&
               entries.runtime.front()(reinterpret_cast<cudaPointerAttributes*>(page + 4096 - 23),
                                       pointer(allocation.device)) == cudaErrorInvalidValue &&
               entries.driver.front()(nullptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                      allocation.device) == CUDA_ERROR_INVALID_VALUE &&
               entries.driver.front()(page + 4096 - 7, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                      allocation.device) == CUDA_ERROR_INVALID_VALUE &&
               entries.driver.front()(page + 4096 - 3, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                      allocation.device) == CUDA_ERROR_INVALID_VALUE &&
               control.queries == before,
           "all null and partially writable output spans reject locally with no request");
    for (const unsigned int selector : std::array{0U, 1U, 2U, 4U, 8U, 10U})
        expect(entries.driver.front()(page, static_cast<CUpointer_attribute>(selector),
                                      allocation.device) == CUDA_ERROR_NOT_SUPPORTED &&
                   control.queries == before,
               "out-of-scope selectors reject explicitly without requested native provenance");
    for (const unsigned int width : std::array{24U, 8U, 4U}) {
        const bool runtime = width == 24;
        control.next({runtime ? 2U : 1U,
                      runtime      ? 0U
                      : width == 8 ? 3U
                                   : 9U,
                      2, allocation.session, allocation.session, 7, 0});
        if (runtime)
            check(entries.runtime.front()(
                reinterpret_cast<cudaPointerAttributes*>(page + 4096 - width),
                pointer(allocation.device)));
        else
            check(entries.driver.front()(page + 4096 - width,
                                         width == 8 ? CU_POINTER_ATTRIBUTE_DEVICE_POINTER
                                                    : CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                         allocation.device));
        expect(page[4096 - width - 1] == std::byte{0xa5}, "prefix guard remains unchanged");
        std::memset(page, 0xa5, 4096);
    }
    expect(::munmap(page, 8192) == 0, "release exact-span guard pages");
    allocation.close();
}
void retirement_outcome(Control& control, const Routes& entries, unsigned int scenario) {
    Allocation allocation(control);
    if (scenario == 14) {
        expect(cuMemFree(allocation.device) == CUDA_ERROR_INVALID_VALUE,
               "known original release error preserves the live resource identity");
        runtime_query(control, entries.runtime.front(), allocation.location());
        allocation.close();
    } else if (scenario == 15) {
        expect(cuMemFree(allocation.device) == CUDA_ERROR_UNKNOWN,
               "lost release response cannot establish physical retirement");
        GuardedRuntime output;
        cudaPointerAttributes before;
        std::memcpy(&before, &output.value, sizeof(before));
        expect(
            entries.runtime.front()(&output.value, pointer(allocation.device)) ==
                    cudaErrorUnknown &&
                control.queries == 0,
            "uncertain free must not turn an owned pointer into successful unregistered metadata");
        expect_runtime(output, before);
    } else {
        control.hold_cleanup = true;
        auto freeing = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(allocation.context));
            check(cuMemFree(allocation.device));
        });
        wait_entered(control);
        std::atomic<bool> started{};
        auto querying = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(allocation.context));
            started = true;
            runtime_query(control, entries.runtime.front(),
                          unknown(allocation.device, allocation.session));
        });
        while (!started) std::this_thread::yield();
        const bool waited =
            querying.wait_for(20ms) == std::future_status::timeout && control.queries == 0;
        control.release = true;
        freeing.get();
        querying.get();
        expect(waited,
               "retiring classification waits without self-pins, then rescans after known free");
        check(cuCtxDestroy(allocation.context));
    }
}
void adjacent_live_range(Control& control, const Routes& entries) {
    Allocation allocation(control);
    CUdeviceptr next{};
    check(cuMemAlloc(&next, 256));
    expect(next == allocation.device + 256, "second allocation starts at first logical endpoint");
    all_queries(control, entries, {next, allocation.session, allocation.session, 8, 0, 2, false});
    all_queries(control, entries,
                {next + 255, allocation.session, allocation.session, 8, 255, 2, false});
    all_queries(control, entries, unknown(next + 256, allocation.session));
    check(cuMemFree(next));
    allocation.close();
}
void cross_context_retiring_alias(Control& control, const Routes& entries) {
    Allocation old_owner(control);
    control.hold_cleanup = true;
    auto freeing = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(old_owner.context));
        check(cuMemFree(old_owner.device));
    });
    wait_entered(control);

    // A different context can publish a new allocation with the same numeric
    // address while the old context's successful free receipt is still pending.
    Allocation fresh_owner(control);
    expect(fresh_owner.context != old_owner.context && fresh_owner.session != old_owner.session &&
               fresh_owner.device == old_owner.device && control.allocations == 2,
           "distinct context publishes the same address before old retirement acknowledgement");
    std::atomic<bool> started{};
    auto querying = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(fresh_owner.context));
        started = true;
        runtime_query(
            control, entries.runtime.front(),
            {fresh_owner.device, fresh_owner.session, fresh_owner.session, 8, 0, 2, false});
    });
    while (!started) std::this_thread::yield();
    const bool waited = querying.wait_for(20ms) == std::future_status::timeout &&
                        control.queries == 0 && control.releases == 1;
    control.release = true;
    freeing.get();
    querying.get();
    expect(waited && control.queries == 1,
           "cross-context overlap waits without pins, then queries only the fresh owner once");
    check(cuCtxSetCurrent(fresh_owner.context));
    fresh_owner.close();
    check(cuCtxSetCurrent(old_owner.context));
    check(cuCtxDestroy(old_owner.context));
    expect(control.releases == 2 && control.closes == 2,
           "both independent allocation lifetimes finish exactly once after alias disambiguation");
}
void adversarial(Control& control, const Routes& entries, unsigned int scenario) {
    Allocation allocation(control);
    const bool driver = scenario == 44 || scenario == 45 || scenario == 54;
    const auto selector = scenario == 45 ? 9U : 3U;
    control.next({driver ? 1U : 2U, driver ? selector : 0U, 2, allocation.session,
                  allocation.session, 7, 0});
    GuardedRuntime output;
    cudaPointerAttributes before;
    std::memcpy(&before, &output.value, sizeof(before));
    std::uint64_t scalar = kDeviceSentinel;
    const auto result =
        driver
            ? static_cast<int>(entries.driver.front()(
                  &scalar, static_cast<CUpointer_attribute>(selector), allocation.device))
            : static_cast<int>(entries.runtime.front()(&output.value, pointer(allocation.device)));
    const int expected_status = scenario == 48   ? 1
                                : scenario == 49 ? 600
                                : scenario == 55 ? 10000
                                                 : 999;
    expect(result == expected_status && scalar == kDeviceSentinel,
           "hostile receipt rejects without exposing raw pointer or invented original status");
    expect_runtime(output, before);
    if (scenario == 48 || scenario == 49 || scenario == 55) {
        runtime_query(control, entries.runtime.front(), allocation.location());
        allocation.close();
    } else {
        const auto calls = control.queries.load();
        expect(entries.runtime.front()(&output.value, pointer(allocation.device)) != cudaSuccess &&
                   control.queries == calls,
               "malformed/quarantined ownership is inaccessible without native fallback or replay");
        expect_runtime(output, before);
        expect(control.releases == 0 && control.closes == 0,
               "unproven owner cannot send teardown based only on the failed host receipt");
    }
}
void exercise(Providers paths, unsigned int scenario, Control& control) {
    if (scenario == 0) {
        normal(control, paths);
        return;
    }
    check(cuInit(0));
    const auto entries = routes(paths);
    if (scenario == 1 || scenario == 2)
        failure_effects(control, entries, scenario == 2);
    else if ((scenario >= 3 && scenario <= 7) || scenario == 12 || scenario == 13)
        lifecycle(control, entries, scenario);
    else if (scenario >= 8 && scenario <= 10)
        publication(control, entries, scenario);
    else if (scenario == 11)
        local_and_exact_spans(control, entries);
    else if (scenario >= 14 && scenario <= 16)
        retirement_outcome(control, entries, scenario);
    else if (scenario == 17)
        adjacent_live_range(control, entries);
    else if (scenario == 57)
        cross_context_retiring_alias(control, entries);
    else
        adversarial(control, entries, scenario);
    entries.check_native();
}

struct Mapping {
    void* address{MAP_FAILED};
    void release() {
        if (address != MAP_FAILED) {
            expect(::munmap(address, 4096) == 0, "release fake service host alias once");
            address = MAP_FAILED;
        }
    }
    ~Mapping() {
        if (address != MAP_FAILED) (void)::munmap(address, 4096);
    }
};
void run(Providers providers, unsigned int scenario, const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated independent pointer metadata endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "observe real process identity");
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(), 1) ==
                   0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START",
                        std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
           "authenticate actual endpoint before first intercepted call");
    Control control;
    Mapping host_mapping;
    std::map<std::uint64_t, std::uint32_t> roles;
    std::map<std::uint64_t, std::pair<std::uint64_t, bool>> frontiers;
    std::uint64_t next_frontier = 90;
    std::optional<ExecutionReceivedRequest> pending;
    ExecutionWireReply pending_reply;
    auto client = std::async(std::launch::async, [&] { exercise(providers, scenario, control); });
    const auto publish = [&](const ExecutionReceivedRequest& request,
                             const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {173, 179};
        auto bytes = encode_execution_reply(reply);
        expect(bytes && decode_execution_reply(*bytes),
               "semantic hostile capsules are structurally valid");
        if (scenario == 56 &&
            request.message.header.operation == static_cast<ExecutionOpcode>(130)) {
            (*bytes)[0] = std::byte{4};
            expect(!decode_execution_reply(*bytes),
                   "disposition4 is rejected by existing framing, not added as Deferred");
        }
        expect(server.reply(request.connection, {header, *bytes}),
               "publish exact authenticated receipt");
    };
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    try {
        while (client.wait_for(0ms) != std::future_status::ready) {
            expect(server.poll_for(1), "progress actual pointer IPC");
            if (pending && control.release) {
                publish(*pending, pending_reply);
                pending.reset();
            }
            if (auto request = server.take_request()) {
                ExecutionPayloadReader input(request->message.payload);
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.original_status = 0;
                const auto operation = request->message.header.operation;
                if (operation == ExecutionOpcode::kHello) {
                    std::uint32_t role{}, flags{};
                    if (request->message.payload.size() == 8) {
                        expect(input.read_u32(role) && input.read_u32(flags) && input.empty(),
                               "exact helper role/flags bootstrap");
                    } else {
                        const auto descriptor =
                            decode_execution_context_bootstrap(request->message.payload);
                        expect(descriptor && descriptor->source_generation,
                               "ordinary application uses a versioned cold context descriptor");
                        flags = descriptor->flags;
                    }
                    roles[request->connection] = role;
                    if (role == 0x31444d50U) {
                        expect(!flags, "PMD1 is CPU metadata only with flags zero");
                        ++control.metadata_hellos;
                        reply.original_status.reset();
                    } else if (role == kExecutionHostOwnerRole)
                        control.host_owner = request->connection;
                    else {
                        expect(!role, "no staging or unrelated metadata owner is created");
                        ++control.application_hellos;
                        control.latest_application = request->connection;
                    }
                    const auto ownership = encode_execution_bootstrap_ownership(
                        role == 0x31444d50U ? ExecutionBootstrapOwnership::kMetadataRetained
                                            : ExecutionBootstrapOwnership::kRetained);
                    expect(ownership.has_value(), "encode known bootstrap ownership");
                    reply.data.assign(ownership->begin(), ownership->end());
                } else if (scenario == 7 && operation == ExecutionOpcode::kCaptureFrontier) {
                    std::uint32_t scope{};
                    std::uint64_t stream{};
                    expect(roles.at(request->connection) == 0 && input.read_u32(scope) &&
                               input.read_u64(stream) && input.empty() && !scope && !stream,
                           "only ordinary Runtime setup/reset captures its context frontier");
                    reply.id = ++next_frontier;
                    expect(
                        frontiers.emplace(reply.id, std::pair{request->connection, false}).second,
                        "each ordinary frontier has one distinct live identity");
                    reply.original_status.reset();
                } else if (scenario == 7 && (operation == ExecutionOpcode::kQueryFrontier ||
                                             operation == ExecutionOpcode::kReleaseFrontier)) {
                    std::uint64_t frontier{};
                    expect(input.read_u64(frontier) && input.empty() &&
                               frontiers.contains(frontier) &&
                               frontiers.at(frontier).first == request->connection,
                           "ordinary frontier observations retain their authenticated owner");
                    if (operation == ExecutionOpcode::kQueryFrontier) {
                        expect(!frontiers.at(frontier).second, "frontier is queried once");
                        frontiers.at(frontier).second = true;
                    } else {
                        expect(frontiers.at(frontier).second, "frontier release follows its query");
                        frontiers.erase(frontier);
                    }
                    reply.original_status.reset();
                } else if (operation == ExecutionOpcode::kAllocate) {
                    std::uint64_t bytes{};
                    expect(input.read_u64(bytes) && bytes == 256 && input.empty(),
                           "logical256-byte allocation");
                    const auto index = ++control.allocations;
                    expect(index <= 2, "bounded allocation/reuse inventory");
                    reply.id = index == 1 ? 7 : 8;
                    reply.address = kDevice + (scenario == 17 && index == 2 ? 256 : 0);
                } else if (operation == ExecutionOpcode::kRelease) {
                    std::uint64_t allocation{};
                    expect(input.read_u64(allocation) && (allocation == 7 || allocation == 8) &&
                               input.empty(),
                           "free exact logical resource identity, not caller pointer");
                    ++control.releases;
                    if (scenario == 14 && control.releases == 1) {
                        reply.status = 1;
                        reply.original_status = 1;
                    }
                    if (scenario == 15) {
                        server.close_peer(request->connection);
                        continue;
                    }
                    if (control.hold_cleanup.exchange(false)) {
                        pending = std::move(request);
                        pending_reply = reply;
                        control.entered = true;
                        continue;
                    }
                } else if (operation == ExecutionOpcode::kLoadPtx) {
                    std::string_view source;
                    expect(input.read_string(source) && source == kSource && input.empty(),
                           "unchanged source image");
                    reply.id = 29;
                } else if (operation == ExecutionOpcode::kGetGlobal) {
                    std::uint64_t module{};
                    std::string_view name;
                    expect(input.read_u64(module) && module == 29 && input.read_string(name) &&
                               (name == "global" || name == "alias") && input.empty(),
                           "typed module symbol lookup");
                    reply.id = 31;
                    reply.address = kGlobal;
                    reply.parameter_bytes = 32;
                } else if (operation == ExecutionOpcode::kUnloadModule) {
                    std::uint64_t module{};
                    expect(input.read_u64(module) && module == 29 && input.empty(),
                           "exact module cleanup");
                    ++control.unloads;
                } else if (operation == ExecutionOpcode::kImportHostMemory) {
                    std::uint64_t bytes{}, extent{};
                    std::uint32_t flags{};
                    expect(roles.at(request->connection) == kExecutionHostOwnerRole &&
                               input.read_u64(bytes) && bytes == 256 && input.read_u64(extent) &&
                               extent == 4096 && input.read_u32(flags) && input.empty() &&
                               request->descriptor.get() >= 0,
                           "host queries retain logical256 bytes, not mapped4096-byte padding");
                    host_mapping.address = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                                                  request->descriptor.get(), 0);
                    expect(host_mapping.address != MAP_FAILED &&
                               host_mapping.address != pointer(kHost),
                           "service host mapping differs from public device/host alias");
                    reply.id = 61;
                    reply.address = kHost;
                    ExecutionPayloadWriter ownership;
                    expect(ownership.write_u32(0), "known native host ownership");
                    reply.data = ownership.take();
                } else if (operation == ExecutionOpcode::kReleaseHostMemory) {
                    std::uint64_t id{};
                    expect(input.read_u64(id) && id == 61 && input.empty(),
                           "exact host cleanup identity");
                    ++control.host_releases;
                    host_mapping.release();
                } else if (operation == ExecutionOpcode::kClose) {
                    if (roles.at(request->connection) == 0)
                        ++control.closes;
                    else
                        reply.original_status.reset();
                } else if (operation == static_cast<ExecutionOpcode>(130)) {
                    const auto expected = control.take();
                    std::array<std::uint32_t, 4> fields{};
                    std::array<std::uint64_t, 4> ids{};
                    for (auto& field : fields)
                        expect(input.read_u32(field), "fixed pointer discriminator");
                    for (auto& id : ids)
                        expect(input.read_u64(id), "fixed symbolic pointer identity");
                    expect(request->message.payload.size() == 48 && input.empty() &&
                               roles.at(request->connection) == 0x31444d50U,
                           "exact48-byte PMD1 request uses only the metadata channel");
                    const std::array<std::uint32_t, 4> wanted_fields{
                        expected.domain, expected.selector, expected.kind, 0};
                    const std::array<std::uint64_t, 4> wanted_ids{
                        expected.query, expected.source, expected.allocation, expected.offset};
                    for (std::size_t index = 0; index != fields.size(); ++index) {
                        if (fields[index] != wanted_fields[index] ||
                            ids[index] != wanted_ids[index])
                            throw std::runtime_error(
                                "PMD1 query " + std::to_string(control.queries.load() + 1) +
                                " field " + std::to_string(index) + ": discriminator " +
                                std::to_string(fields[index]) + "/" +
                                std::to_string(wanted_fields[index]) + ", identity " +
                                std::to_string(ids[index]) + "/" +
                                std::to_string(wanted_ids[index]));
                    }
                    const auto number = ++control.queries;
                    reply = observed(expected);
                    if (scenario == 1 && number <= 16) {
                        const auto mask = number - 1;
                        reply.status = 98;
                        reply.original_status = 98;
                        reply.data =
                            effect_bytes({mask, mask & 1U ? 0x80000000U : 0,
                                          mask & 2U ? std::bit_cast<std::uint32_t>(-17) : 0,
                                          mask & 4U ? 1U : 0U, 0});
                    } else if (scenario == 2 && number == 1) {
                        reply.status = 1;
                        reply.original_status = 1;
                        reply.data = effect_bytes({2, 0, std::bit_cast<std::uint32_t>(-17), 0, 0});
                    } else if (number == 1 && scenario >= 20)
                        reply = altered(std::move(reply), scenario);
                    if (number == 1 && scenario == 43) {
                        server.close_peer(request->connection);
                        continue;
                    }
                    if (number == 1 && scenario >= 8 && scenario <= 10)
                        expect(::mprotect(control.protected_output.load(), 4096, PROT_NONE) == 0,
                               "protect caller output only after actual metadata request");
                    if (control.hold.exchange(false)) {
                        pending = std::move(request);
                        pending_reply = std::move(reply);
                        control.entered = true;
                        continue;
                    }
                } else
                    throw std::runtime_error("unexpected operation " +
                                             std::to_string(static_cast<unsigned int>(operation)) +
                                             " in pointer metadata case " +
                                             std::to_string(scenario));
                publish(*request, reply);
            }
            expect(std::chrono::steady_clock::now() < deadline, "bounded pointer metadata fixture");
        }
        client.get();
        expect(control.empty() && !pending && frontiers.empty(),
               "every expected call occurred exactly once and no frontier survives");
    } catch (...) {
        control.release = true;
        server.stop();
        throw;
    }
    server.stop();
    expect(glimmer_pointer_native_calls() == 0,
           "no native caller-side resource consumer receives remote handles or public tokens");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 3 || argc == 5,
               "Runtime/Driver provider paths and optional --case selector required");
        if (argc == 3) {
            for (unsigned int scenario = 0; scenario <= kLastCase; ++scenario) {
                if (scenario >= 18 && scenario < 20) continue;
                const auto number = std::to_string(scenario);
                const auto child = ::fork();
                expect(child >= 0, "fork isolated metadata test before any frontend activity");
                if (!child) {
                    ::execl(argv[0], argv[0], argv[1], argv[2], "--case", number.c_str(), nullptr);
                    _exit(127);
                }
                int status{};
                if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
                    WEXITSTATUS(status) != 0)
                    throw std::runtime_error("isolated pointer metadata case " + number +
                                             " failed");
            }
            std::cout << "56 pointer metadata scenarios passed\n";
            return EXIT_SUCCESS;
        }
        expect(std::string_view(argv[3]) == "--case", "explicit fresh-process case selector");
        const auto scenario = static_cast<unsigned int>(std::stoul(argv[4]));
        expect(scenario <= kLastCase && (scenario < 18 || scenario >= 20),
               "bounded case inventory");
        char pattern[] = "/tmp/glimmer-pointer-metadata-XXXXXX";
        const auto* directory = ::mkdtemp(pattern);
        expect(directory, "create private authenticated endpoint directory");
        try {
            run({argv[1], argv[2]}, scenario, std::string(directory) + "/service.sock");
        } catch (...) {
            (void)::rmdir(directory);
            throw;
        }
        expect(::rmdir(directory) == 0, "remove completed endpoint directory");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
