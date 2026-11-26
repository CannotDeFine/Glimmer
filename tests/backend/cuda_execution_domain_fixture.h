#pragma once

#include "internal/cuda_execution_domain.h"
#include "internal/cuda_execution_session.h"
#include "internal/cuda_ptx_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace glimmer::test::cuda_domain {
using namespace glimmer::backend::cuda;
constexpr std::string_view kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry kernel(.param .u64 pointer, .param .u32 value) {
 .reg .pred %p;
 .reg .b32 %r<5>;
 .reg .b64 %rd<4>;
 mov.u32 %r1, %tid.x;
 setp.ne.u32 %p, %r1, 0;
 @%p bra EXIT;
 ld.param.u64 %rd1, [pointer];
 ld.param.u32 %r2, [value];
 mov.u32 %r3, %ctaid.x;
 mul.wide.u32 %rd2, %r3, 4;
 add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r4, [%rd3];
 add.u32 %r4, %r4, %r2;
 st.global.u32 [%rd3], %r4;
EXIT: ret;
}
)ptx";

inline void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline void success(const CudaSessionResult& result) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("Session operation failed: disposition=" +
                                 std::to_string(static_cast<int>(result.disposition)) +
                                 ", status=" + std::to_string(static_cast<int>(result.status)));
}
struct FakeContext {
    bool is_live{};
};
struct FakeFunction {
    bool is_private{};
    std::array<int, 16> attributes{1024, 0, 0, 0, 17, 70, 70, 0, 49152, -1};
    CUfunc_cache cache{CU_FUNC_CACHE_PREFER_NONE};
};
struct FakeModule {
    bool is_live{};
    CUcontext context{};
    FakeFunction original{}, prepared{true};
};
struct FakeAllocation {
    bool is_live{};
    CUcontext context{};
    CUdeviceptr address{};
    std::size_t bytes{};
    std::array<std::byte, 256> data{};
};
struct FakeStream {
    bool is_live{};
    CUcontext context{};
    unsigned int flags{};
};
struct FakeEvent {
    bool is_live{}, is_recorded{};
    CUcontext context{};
    unsigned int launch{};
};
struct alignas(64) FakeHost {
    bool is_live{};
    CUcontext context{};
    alignas(64) std::array<std::byte, 4224> data{};
};
struct FakeDriver {
    std::array<FakeContext, 4> contexts{};
    std::array<FakeModule, 8> modules{};
    std::array<FakeAllocation, 16> allocations{};
    std::array<FakeStream, 16> streams{};
    std::array<FakeEvent, 32> events{};
    std::array<FakeHost, 8> hosts{};
    CUcontext current{};
    unsigned int context_creates{}, context_destroys{}, allocation_calls{}, release_calls{};
    unsigned int module_unloads{}, stream_destroys{}, event_destroys{}, host_frees{}, launches{};
    unsigned int counter_initializations{};
    unsigned int module_load_calls{}, parameter_calls{}, configuration_calls{};
    bool has_profile_function_metadata{};
    CUresult function_configuration_status{CUDA_SUCCESS};
    unsigned int context_control_calls{}, context_flags{CU_CTX_MAP_HOST};
    CUfunc_cache cache{CU_FUNC_CACHE_PREFER_NONE};
    CUsharedconfig shared{CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE};
    std::array<std::size_t, 3> context_limits{1024, 4096, 8192};
    CUresult next_context_control_status{CUDA_SUCCESS};
    std::size_t next_allocation{}, next_module{}, next_stream{}, next_event{}, next_host{};
    CUresult query_status{CUDA_SUCCESS};
    CUresult next_allocation_status{CUDA_SUCCESS}, next_host_status{CUDA_SUCCESS};
    CUresult next_host_map_status{CUDA_SUCCESS};
    CUresult next_context_query_status{CUDA_SUCCESS};
    CUresult next_module_status{CUDA_SUCCESS};
    CUresult next_launch_status{CUDA_SUCCESS};
    unsigned int ready_launch{};
    CUdeviceptr launch_output{};
    std::uint64_t next_claimed{3};
    bool requires_explicit_cleanup{};
    bool defers_pair_metadata{};
    std::function<void()> on_allocate;
    std::function<void()> on_module_load;
    std::function<void(bool)> on_function_get;
    std::function<void()> on_launch;
    std::vector<std::uint64_t> previous_counters;
    std::vector<unsigned int> physical_workers;

    [[nodiscard]] std::size_t live_resources() const {
        const auto live = [](const auto& resources) {
            return std::ranges::count_if(resources,
                                         [](const auto& value) { return value.is_live; });
        };
        return static_cast<std::size_t>(live(modules) + live(allocations) + live(streams) +
                                        live(events) + live(hosts));
    }
    [[nodiscard]] unsigned int cleanup_calls() const {
        return context_destroys + release_calls + module_unloads + stream_destroys +
               event_destroys + host_frees;
    }
};
inline FakeDriver driver;

inline FakeAllocation& allocation(CUdeviceptr address, std::size_t bytes) {
    const auto found = std::ranges::find_if(driver.allocations, [&](const auto& value) {
        return value.is_live && value.context == driver.current && address >= value.address &&
               address - value.address <= value.bytes &&
               bytes <= value.bytes - (address - value.address);
    });
    expect(found != driver.allocations.end(), "fake Driver validates exact owned device span");
    return *found;
}
inline std::byte* device_bytes(CUdeviceptr address, std::size_t bytes) {
    auto& value = allocation(address, bytes);
    return value.data.data() + static_cast<std::size_t>(address - value.address);
}
inline CUresult CUDAAPI initialize(unsigned int flags) {
    expect(flags == 0, "only the configured fake Driver initialization is requested");
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI device_get(CUdevice* device, int ordinal) {
    expect(ordinal == 0, "fixture has exactly one configured fake device");
    *device = 0;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI context_create(CUcontext* output, unsigned int flags, CUdevice) {
    expect(driver.context_creates < driver.contexts.size(), "bounded native context arena");
    auto& context = driver.contexts[driver.context_creates++];
    expect(!context.is_live && !driver.current, "context creation starts with NULL owner TLS");
    context.is_live = true;
    driver.context_flags = flags;
    *output = reinterpret_cast<CUcontext>(&context);
    driver.current = *output;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI context_get_current(CUcontext* output) {
    const auto status = std::exchange(driver.next_context_query_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    *output = driver.current;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI context_set_current(CUcontext context) {
    driver.current = context;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI context_destroy(CUcontext context) {
    expect(context && (!driver.current || context == driver.current),
           "destroy only the exact owner context from its selected or NULL TLS state");
    if (driver.requires_explicit_cleanup)
        expect(driver.live_resources() == 0, "domain destroy cannot substitute for member cleanup");
    ++driver.context_destroys;
    auto* owner = reinterpret_cast<FakeContext*>(context);
    expect(owner->is_live, "native context destroy is once-only");
    owner->is_live = false;
    // Isolated CUDA context destruction implicitly releases its own native
    // allocations/modules/streams/events, not another context's resources.
    const auto retire = [context](auto& resources) {
        for (auto& resource : resources)
            if (resource.context == context) resource.is_live = false;
    };
    retire(driver.allocations);
    retire(driver.modules);
    retire(driver.streams);
    retire(driver.events);
    driver.current = nullptr;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI module_load(CUmodule* output, const void* image) {
    ++driver.module_load_calls;
    if (auto callback = std::exchange(driver.on_module_load, {})) callback();
    const auto status = std::exchange(driver.next_module_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    expect(driver.current && image && driver.next_module < driver.modules.size(),
           "module intake uses a bounded selected-context owner");
    auto& module = driver.modules[driver.next_module++];
    module.is_live = true;
    module.context = driver.current;
    *output = reinterpret_cast<CUmodule>(&module);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI module_unload(CUmodule native) {
    auto* module = reinterpret_cast<FakeModule*>(native);
    expect(module && module->is_live && module->context == driver.current,
           "unload only a live member-owned module");
    module->is_live = false;
    ++driver.module_unloads;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI function_get(CUfunction* output, CUmodule native, const char* name) {
    auto* module = reinterpret_cast<FakeModule*>(native);
    expect(module && module->is_live && module->context == driver.current && name,
           "function lookup uses the exact live module owner");
    const bool is_private = std::string_view{name}.find("__glimmer_ptb") != std::string_view::npos;
    if (driver.on_function_get) driver.on_function_get(is_private);
    *output = reinterpret_cast<CUfunction>(is_private ? &module->prepared : &module->original);
    return CUDA_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA callback ABI.
inline CUresult CUDAAPI parameter_info(CUfunction native, std::size_t index, std::size_t* offset,
                                       std::size_t* bytes) {
    ++driver.parameter_calls;
    const auto* function = reinterpret_cast<FakeFunction*>(native);
    constexpr std::array<std::size_t, 7> offsets{0, 8, 12, 16, 20, 24, 32};
    constexpr std::array<std::size_t, 7> widths{8, 4, 4, 4, 4, 8, 8};
    if (index >= (function->is_private ? 7U : 2U)) return CUDA_ERROR_INVALID_VALUE;
    *offset = offsets[index];
    *bytes = widths[index];
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI allocate(CUdeviceptr* output, std::size_t bytes) {
    ++driver.allocation_calls;
    if (auto callback = std::exchange(driver.on_allocate, {})) callback();
    const auto status = std::exchange(driver.next_allocation_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    expect(driver.current && bytes && bytes <= 256 &&
               driver.next_allocation < driver.allocations.size(),
           "native allocation stays within the fixture's bounded owned arena");
    auto& value = driver.allocations[driver.next_allocation];
    value.is_live = true;
    value.context = driver.current;
    value.address = (static_cast<CUdeviceptr>(driver.next_allocation) + 1) * 4096;
    value.bytes = bytes;
    value.data.fill(std::byte{});
    ++driver.next_allocation;
    *output = value.address;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI release(CUdeviceptr address) {
    auto& value = allocation(address, 0);
    expect(address == value.address, "native free uses the exact allocation base");
    value.is_live = false;
    ++driver.release_calls;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI upload(CUdeviceptr address, const void* source, std::size_t bytes) {
    std::memcpy(device_bytes(address, bytes), source, bytes);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI download(void* destination, CUdeviceptr address, std::size_t bytes) {
    std::memcpy(destination, device_bytes(address, bytes), bytes);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI upload_async(CUdeviceptr address, const void* source, std::size_t bytes,
                                     CUstream) {
    if (bytes == sizeof(std::uint64_t)) ++driver.counter_initializations;
    return upload(address, source, bytes);
}
inline CUresult CUDAAPI download_async(void* destination, CUdeviceptr address, std::size_t bytes,
                                       CUstream) {
    return download(destination, address, bytes);
}
inline CUresult CUDAAPI stream_create(CUstream* output, unsigned int flags) {
    expect(driver.current && driver.next_stream < driver.streams.size(), "bounded stream arena");
    auto& stream = driver.streams[driver.next_stream++];
    stream = {true, driver.current, flags};
    *output = reinterpret_cast<CUstream>(&stream);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_destroy(CUstream native) {
    auto* stream = reinterpret_cast<FakeStream*>(native);
    expect(stream && stream->is_live && stream->context == driver.current,
           "destroy only the member's live stream");
    stream->is_live = false;
    ++driver.stream_destroys;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI event_create(CUevent* output, unsigned int) {
    expect(driver.current && driver.next_event < driver.events.size(), "bounded event arena");
    auto& event = driver.events[driver.next_event++];
    event = {true, false, driver.current};
    *output = reinterpret_cast<CUevent>(&event);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI event_record(CUevent native, CUstream) {
    auto* event = reinterpret_cast<FakeEvent*>(native);
    expect(event && event->is_live && event->context == driver.current,
           "record only a live selected-context event");
    event->is_recorded = true;
    event->launch = driver.launches;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI event_query(CUevent native) {
    const auto* event = reinterpret_cast<FakeEvent*>(native);
    expect(event && event->is_live && event->context == driver.current,
           "query only the exact live owned event");
    return !event->is_recorded || (event->launch && event->launch == driver.ready_launch)
               ? CUDA_SUCCESS
               : driver.query_status;
}
inline CUresult CUDAAPI event_destroy(CUevent native) {
    auto* event = reinterpret_cast<FakeEvent*>(native);
    expect(event && event->is_live && event->context == driver.current,
           "event destruction is exact and once-only");
    event->is_live = false;
    ++driver.event_destroys;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_wait_event(CUstream, CUevent event, unsigned int flags) {
    expect(flags == 0 && event && reinterpret_cast<FakeEvent*>(event)->is_live,
           "native ordering waits retain their actual event");
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_synchronize(CUstream) {
    return driver.query_status;
}
inline CUresult CUDAAPI launch(CUfunction native, unsigned int workers, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int, unsigned int, CUstream,
                               void**, void** extra) {
    const auto* function = reinterpret_cast<FakeFunction*>(native);
    expect(function && extra &&
               *static_cast<const std::size_t*>(extra[3]) == (function->is_private ? 40U : 12U),
           "fixture independently checks the exact selected public/private ABI");
    const auto* parameters = static_cast<const std::byte*>(extra[1]);
    std::memcpy(&driver.launch_output, parameters, 8);
    if (driver.on_launch) driver.on_launch();
    ++driver.launches;
    driver.physical_workers.push_back(workers);
    const auto status = std::exchange(driver.next_launch_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    CUdeviceptr output{}, counter{};
    std::uint32_t value{}, total{};
    std::memcpy(&output, parameters, 8);
    std::memcpy(&value, parameters + 8, 4);
    std::uint64_t previous{};
    if (function->is_private) {
        std::memcpy(&total, parameters + 12, 4);
        std::memcpy(&counter, parameters + 24, 8);
        std::memcpy(&previous, device_bytes(counter, 8), 8);
    } else {
        total = workers;
    }
    const auto claimed = function->is_private ? driver.next_claimed : total;
    expect(previous <= claimed && claimed <= total,
           "fake persistent execution preserves monotone exact logical claims");
    for (auto block = previous; block < claimed; ++block) {
        auto* target = device_bytes(output + block * sizeof(value), sizeof(value));
        std::uint32_t old{};
        std::memcpy(&old, target, sizeof(old));
        old += value;
        std::memcpy(target, &old, sizeof(old));
    }
    if (function->is_private) {
        std::memcpy(device_bytes(counter, 8), &claimed, 8);
        driver.previous_counters.push_back(previous);
    }
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI host_allocate(void** output, std::size_t bytes, unsigned int) {
    const auto status = std::exchange(driver.next_host_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    expect((bytes == 128 || bytes == 4224) && driver.next_host < driver.hosts.size(),
           "two bounded aligned progress slots");
    auto& host = driver.hosts[driver.next_host++];
    host.is_live = true;
    host.context = driver.current;
    *output = host.data.data();
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI host_release(void* pointer) {
    const auto found = std::ranges::find_if(driver.hosts, [&](const auto& host) {
        return host.is_live && host.context == driver.current && host.data.data() == pointer;
    });
    expect(found != driver.hosts.end(), "release the exact private host arena once");
    found->is_live = false;
    ++driver.host_frees;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI host_map(CUdeviceptr* address, void* pointer, unsigned int) {
    const auto status = std::exchange(driver.next_host_map_status, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS) return status;
    *address = reinterpret_cast<CUdeviceptr>(pointer);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI occupancy(int* blocks, CUfunction, int, std::size_t) {
    *blocks = 4;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI function_set_attribute(CUfunction native, CUfunction_attribute attribute,
                                               int value) {
    ++driver.configuration_calls;
    if (driver.has_profile_function_metadata) {
        const auto status = std::exchange(driver.function_configuration_status, CUDA_SUCCESS);
        if (status != CUDA_SUCCESS) return status;
        const auto index = static_cast<std::size_t>(attribute);
        if (index >= 16) return CUDA_ERROR_INVALID_VALUE;
        reinterpret_cast<FakeFunction*>(native)->attributes[index] = value;
    }
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
inline CUresult CUDAAPI get_context_flags(unsigned int* output) {
    *output = driver.context_flags;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI set_context_flags(unsigned int flags) {
    ++driver.context_control_calls;
    driver.context_flags = flags;
    return std::exchange(driver.next_context_control_status, CUDA_SUCCESS);
}
inline CUresult CUDAAPI get_context_cache(CUfunc_cache* output) {
    *output = driver.cache;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI set_context_cache(CUfunc_cache cache) {
    ++driver.context_control_calls;
    driver.cache = cache;
    return std::exchange(driver.next_context_control_status, CUDA_SUCCESS);
}
inline CUresult CUDAAPI get_context_shared(CUsharedconfig* output) {
    *output = driver.shared;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI set_context_shared(CUsharedconfig shared) {
    ++driver.context_control_calls;
    driver.shared = shared;
    return std::exchange(driver.next_context_control_status, CUDA_SUCCESS);
}
inline CUresult CUDAAPI get_context_limit(std::size_t* output, CUlimit limit) {
    const auto selector = static_cast<unsigned int>(limit);
    if (selector >= driver.context_limits.size()) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    *output = driver.context_limits[selector];
    return CUDA_SUCCESS;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact native CUDA ABI.
inline CUresult CUDAAPI set_context_limit(CUlimit limit, std::size_t bytes) {
    ++driver.context_control_calls;
    const auto selector = static_cast<unsigned int>(limit);
    if (selector >= driver.context_limits.size()) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    driver.context_limits[selector] = ((bytes + 255) / 256) * 256;
    return std::exchange(driver.next_context_control_status, CUDA_SUCCESS);
}
inline CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch result;
    result.execution = {context_get_current, launch,        event_create,      event_record,
                        event_query,         event_destroy, stream_synchronize};
    result.initialize = initialize;
    result.device_get = device_get;
    result.context_create = context_create;
    result.context_set_current = context_set_current;
    result.context_destroy = context_destroy;
    result.module_load = module_load;
    result.module_unload = module_unload;
    result.function_get = function_get;
    result.parameter_info = parameter_info;
    result.allocate = allocate;
    result.release = release;
    result.upload = upload;
    result.download = download;
    result.stream_create = stream_create;
    result.stream_destroy = stream_destroy;
    result.host_allocate = host_allocate;
    result.host_release = host_release;
    result.host_map = host_map;
    result.upload_async = upload_async;
    result.download_async = download_async;
    result.occupancy = occupancy;
    result.function_set_attribute = function_set_attribute;
    result.stream_wait_event = stream_wait_event;
    result.context_get_flags = get_context_flags;
    result.context_set_flags = set_context_flags;
    result.context.get_limit = get_context_limit;
    result.context.set_limit = set_context_limit;
    result.context.get_cache = get_context_cache;
    result.context.set_cache = set_context_cache;
    result.context.get_shared = get_context_shared;
    result.context.set_shared = set_context_shared;
    result.context.runtime.get_cache = [](cudaFuncCache* output) {
        *output = static_cast<cudaFuncCache>(driver.cache);
        return cudaSuccess;
    };
    result.context.runtime.set_cache = [](cudaFuncCache cache) {
        (void)set_context_cache(static_cast<CUfunc_cache>(cache));
        return cudaErrorInvalidDeviceFunction;
    };
    return result;
}
inline CudaExecutionDomainDispatch domain_dispatch() {
    return {initialize,          device_get,          context_create,
            context_set_current, context_get_current, context_destroy};
}
inline CudaExecutionDomainDispatch configuration_domain_dispatch() {
    auto result = domain_dispatch();
    result.configuration = {get_context_flags,  set_context_flags, get_context_limit,
                            set_context_limit,  get_context_cache, set_context_cache,
                            get_context_shared, set_context_shared};
    return result;
}
inline CudaSessionLimits limits(std::uint64_t first_resource_id) {
    CudaSessionLimits result;
    result.first_resource_id = first_resource_id;
    result.submissions = {2, 4096};
    return result;
}
struct KernelInput {
    struct Binding {
        std::uint64_t function{}, stream{};
    };
    std::array<std::byte, 12> parameters{};
    CudaExecutionKernel kernel;
    KernelInput(Binding binding, CUdeviceptr output)
        : kernel{.function = binding.function,
                 .stream = binding.stream,
                 .grid = {8, 1, 1},
                 .block = {32, 1, 1},
                 .parameters = parameters} {
        constexpr std::uint32_t value = 7;
        std::memcpy(parameters.data(), &output, 8);
        std::memcpy(parameters.data() + 8, &value, 4);
    }
};

inline std::uint64_t pair_now_ns{1};
inline std::uint64_t pair_clock() noexcept {
    return pair_now_ns;
}

inline CudaSessionDispatch pair_dispatch() {
    auto result = session_dispatch();
    result.device_uuid = [](CUuuid* uuid, CUdevice) {
        *uuid = {};
        uuid->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    result.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    result.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice) {
        switch (attribute) {
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
                *output = 7;
                break;
            case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
                *output = 0;
                break;
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
                *output = 80;
                break;
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
                *output = 32;
                break;
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
                *output = 2048;
                break;
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
                *output = 65536;
                break;
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
                *output = 98304;
                break;
            default:
                return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    };
    result.function_get_attribute = [](int* output, CUfunction_attribute attribute,
                                       CUfunction native) {
        if (driver.defers_pair_metadata) return CUDA_ERROR_NOT_READY;
        if (driver.has_profile_function_metadata) {
            const auto index = static_cast<std::size_t>(attribute);
            if (index >= 16) return CUDA_ERROR_INVALID_VALUE;
            *output = reinterpret_cast<FakeFunction*>(native)->attributes[index];
            if (attribute == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES &&
                reinterpret_cast<FakeFunction*>(native)->is_private)
                *output = 128;
            return CUDA_SUCCESS;
        }
        if (attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
            *output = -1;
            return CUDA_SUCCESS;
        }
        expect(reinterpret_cast<FakeFunction*>(native)->is_private,
               "pair metadata queries the actual retained private function");
        if (attribute == CU_FUNC_ATTRIBUTE_NUM_REGS)
            *output = 17;
        else if (attribute == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            *output = 128;
        else
            return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    };
    result.function_set_cache = [](CUfunction native, CUfunc_cache cache) {
        ++driver.configuration_calls;
        const auto status = std::exchange(driver.function_configuration_status, CUDA_SUCCESS);
        if (status != CUDA_SUCCESS) return status;
        reinterpret_cast<FakeFunction*>(native)->cache = cache;
        return CUDA_SUCCESS;
    };
    return result;
}

inline std::shared_ptr<CudaProfileCache> pair_singles() {
    CudaProfileConfig config;
    config.workers = {320};
    config.warmup_calls = 0;
    config.samples_per_candidate = 1;
    CudaProfileCache learning(config, pair_clock);
    const auto function = learning.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                        kSource, {}, "kernel");
    for (const auto duration : {100U, 80U}) {
        auto choice = learning.choose(function, {{8, 1, 1}, {32, 1, 1}, 0}, 320, true);
        expect(choice.observation != nullptr,
               "train actual original/private single-cache observations");
        pair_now_ns = 100;
        choice.observation->begin();
        pair_now_ns += duration;
        choice.observation->finish(true);
    }
    const auto archive = learning.encode("actual-pair-cpu-fixture");
    auto result = std::make_shared<CudaProfileCache>(config, pair_clock, false);
    expect(archive && result->restore(*archive, "actual-pair-cpu-fixture"),
           "qualify exact original/code/platform identities through readonly restore");
    return result;
}

inline std::unique_ptr<CudaPairProfileCache> pair_measurements(
    const std::shared_ptr<CudaProfileCache>& singles,
    const std::array<CudaSessionPairKernelDescription, 2>& kernels, bool is_cold) {
    const CudaPairProfileConfig config{
        .max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1};
    auto result = std::make_unique<CudaPairProfileCache>(singles, config, pair_clock, false);
    if (is_cold) return result;
    CudaPairProfileCache learning(singles, config, pair_clock);
    const std::array bound{learning.bind(kernels[0].profile, kernels[0].geometry),
                           learning.bind(kernels[1].profile, kernels[1].geometry)};
    expect(bound[0] && bound[1], "pair training resolves both qualified originals");
    std::array<CudaPairProfileCandidate, 4> candidates{};
    std::size_t count{};
    for (const auto left : {80U, 160U}) {
        for (const auto right : {80U, 160U}) {
            auto& candidate = candidates[count++];
            candidate.legs = {kernels[0].variant, kernels[1].variant};
            candidate.legs[0].workers = left;
            candidate.legs[1].workers = right;
        }
    }
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        auto reservation =
            learning.reserve(bound, {CudaPairProfileDomainKind::kSameContext, {99, 99}},
                             kernels[0].limits, candidates);
        expect(reservation && reservation.candidate == index,
               "explicit CPU fixture reserves each exact actual-resource candidate");
        const auto duration = 400 - 100 * index;
        pair_now_ns = 1000;
        reservation.observations[0]->begin();
        pair_now_ns = 1005;
        reservation.observations[1]->begin();
        pair_now_ns = 1000 + duration;
        reservation.observations[0]->finish(true);
        pair_now_ns = 1005 + duration;
        reservation.observations[1]->finish(true);
    }
    const auto archive = learning.encode("actual-pair-cpu-fixture");
    expect(archive && result->restore(*archive, "actual-pair-cpu-fixture"),
           "readonly exact catalog restore; injected CPU durations are not GPU evidence");
    return result;
}

inline glimmer::core::ExecutionPairPlan pair_authorization(
    const CudaSessionPairSelection& selection) {
    const auto proposal = selection.proposal({101, 202});
    const auto contexts = selection.contexts();
    std::array<glimmer::core::ExecutionPriorityPeer, 2> peers{};
    for (std::size_t leg = 0; leg < peers.size(); ++leg) {
        peers[leg] = {.session = proposal.legs[leg].session,
                      .has_ready_work = !proposal.legs[leg].command.is_resume,
                      .has_paused_work = proposal.legs[leg].command.is_resume,
                      .command = proposal.legs[leg].command,
                      .pair_context = contexts[leg]};
    }
    return glimmer::core::plan_execution_pair(peers, proposal);
}

}  // namespace glimmer::test::cuda_domain
