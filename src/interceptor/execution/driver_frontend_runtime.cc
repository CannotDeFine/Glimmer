#include "driver_frontend_state.h"

// These compiler callbacks never invoke native registration with virtual
// handles. They snapshot CPU metadata; first actual use binds server resources.
// Exact NVIDIA compiler callback names and parameter ordering are fixed ABI.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters)
extern "C" void** CUDARTAPI __cudaRegisterFatBinary(void* wrapper) {
    void** result{};
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) { return register_runtime_image(shared, wrapper, &result); },
        false));
    return result;
}
extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host,
                                                 char* device_function, const char* device_name,
                                                 int thread_limit, uint3* thread_index,
                                                 uint3* block_index, dim3* block_dimension,
                                                 dim3* grid_dimension, int* warp_size) {
    (void)device_name;
    (void)thread_limit;
    (void)thread_index;
    (void)block_index;
    (void)block_dimension;
    (void)grid_dimension;
    (void)warp_size;
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            return register_runtime_function(shared, handle, host, device_function);
        },
        false));
}
extern "C" void CUDARTAPI __cudaRegisterFatBinaryEnd(void** handle) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            auto* image = runtime_image(shared, handle);
            if (!image || image->is_closing) return CUDA_ERROR_INVALID_HANDLE;
            image->is_finalized = true;
            return image->error;
        },
        false));
}
extern "C" void CUDARTAPI __cudaUnregisterFatBinary(void** handle) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto& lock) { return unregister_runtime_image(shared, lock, handle); },
        false));
}
namespace {
void reject_runtime_registration(void** handle) noexcept {
    (void)runtime_result(locked_boundary(
        [&](State& shared, auto&) {
            auto* image = runtime_image(shared, handle);
            if (!image) return CUDA_ERROR_INVALID_HANDLE;
            image->error = CUDA_ERROR_NOT_SUPPORTED;
            return image->error;
        },
        false));
}
}  // namespace
extern "C" void CUDARTAPI __cudaRegisterVar(void** handle, char* host, char* device_symbol,
                                            const char* name, int external, std::size_t bytes,
                                            int constant, int global) {
    (void)runtime_result(locked_boundary(
        [&](State& shared, Exclusion&) {
            return register_runtime_variable(shared, handle, host,
                                             device_symbol ? device_symbol : name, external, bytes,
                                             constant, global);
        },
        false));
}
extern "C" void CUDARTAPI __cudaRegisterManagedVar(void** handle, void**, char*, const char*, int,
                                                   std::size_t, int, int) {
    reject_runtime_registration(handle);
}
extern "C" void CUDARTAPI __cudaRegisterTexture(void** handle, const struct textureReference*,
                                                const void**, const char*, int, int, int) {
    reject_runtime_registration(handle);
}
extern "C" void CUDARTAPI __cudaRegisterSurface(void** handle, const struct surfaceReference*,
                                                const void**, const char*, int, int) {
    reject_runtime_registration(handle);
}
extern "C" char CUDARTAPI __cudaInitModule(void** handle) {
    reject_runtime_registration(handle);
    return 0;
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters)
extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    observe_async_runtime_error();
    const auto value = runtime_last_error;
    runtime_last_error = cudaSuccess;
    return value;
}
extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError() {
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (::getpid() == kProcess) observe_async_runtime_error();
    return ::getpid() == kProcess ? runtime_last_error : cudaErrorIncompatibleDriverContext;
}
namespace {
cudaError_t query_runtime_metadata(void* output, std::size_t bytes, std::uint32_t kind, int device,
                                   int attribute = 0, int peer = 0) noexcept {
    cudaError_t original_status{cudaErrorUnknown};
    const auto prepared = locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, bytes)) return CUDA_ERROR_INVALID_VALUE;
        if (CUDART_VERSION != 12060 || sizeof(void*) != 8) return CUDA_ERROR_NOT_SUPPORTED;
        const auto opened =
            kind == 6 ? runtime_context(shared, exclusion) : shared.open_host_context(exclusion);
        if (opened != CUDA_SUCCESS) return opened;
        auto* owner = kind == 6 ? shared.context() : shared.host_context;
        if (owner->active_calls >= kMaxHandles - 1) return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin origin(owner->active_calls, exclusion);
        ExecutionPayloadWriter input;
        if (!input.write_u32(kind) || !input.write_u32(std::bit_cast<std::uint32_t>(device)) ||
            !input.write_u32(std::bit_cast<std::uint32_t>(attribute)) ||
            !input.write_u32(std::bit_cast<std::uint32_t>(peer)))
            return CUDA_ERROR_OUT_OF_MEMORY;
        const auto reply =
            shared.call(exclusion, *owner, ExecutionOpcode::kRuntimeMetadata, std::move(input));
        if (!reply || reply->parameter_bytes) {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        if (reply->disposition != ExecutionWireDisposition::kComplete || !reply->original_status)
            return status(reply);
        if (reply->status != *reply->original_status || reply->status < 0 ||
            reply->status > 10000 || reply->address != 12060 ||
            (reply->status == 0 ? reply->data.size() != bytes : !reply->data.empty())) {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        original_status = static_cast<cudaError_t>(*reply->original_status);
        if (original_status == cudaSuccess && !self_copy(output, reply->data.data(), bytes, true))
            return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    });
    if (prepared != CUDA_SUCCESS) return runtime_result(prepared);
    if (original_status != cudaSuccess) runtime_last_error = original_status;
    return original_status;
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int* output) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    if (!writable_host_range(output, sizeof(*output)))
        return runtime_result(CUDA_ERROR_INVALID_VALUE);
    const auto initialized = glimmer::interceptor::execution::initialize_driver_metadata();
    if (initialized != CUDA_SUCCESS) return runtime_result(initialized);
    const auto query = original<decltype(&cuDeviceGetCount)>("cuDeviceGetCount");
    if (!query) return runtime_result(CUDA_ERROR_NOT_INITIALIZED);
    int value{};
    const auto result = query(&value);
    if (result != CUDA_SUCCESS) return runtime_result(result);
    if (value < 0) return runtime_result(CUDA_ERROR_UNKNOWN);
    return publish(output, value) ? cudaSuccess : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int* output) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    if (!writable(output)) return runtime_result(CUDA_ERROR_INVALID_VALUE);
    const auto initialized = glimmer::interceptor::execution::initialize_driver_metadata();
    if (initialized != CUDA_SUCCESS) return runtime_result(initialized);
    const auto query = original<decltype(&cuDriverGetVersion)>("cuDriverGetVersion");
    if (!query) return runtime_result(CUDA_ERROR_NOT_INITIALIZED);
    int value{};
    const auto result = query(&value);
    return runtime_result(
        result == CUDA_SUCCESS && !publish(output, value) ? CUDA_ERROR_INVALID_VALUE : result);
}
// The installed SDK expands this entry to the v2 property ABI. The legacy
// property struct has a different layout and is not silently aliased here.
extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp* output, int device) {
    return query_runtime_metadata(output, sizeof(*output), 1, device);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int* output, cudaDeviceAttr attribute,
                                                        int device) {
    return query_runtime_metadata(output, sizeof(*output), 2, device, static_cast<int>(attribute));
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Runtime SDK ABI.
extern "C" cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int* output, cudaDeviceP2PAttr attribute,
                                                           int source, int destination) {
    return query_runtime_metadata(output, sizeof(*output), 3, source, static_cast<int>(attribute),
                                  destination);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int* output, int device, int peer) {
    return query_runtime_metadata(output, sizeof(*output), 5, device, 0, peer);
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if ((least && !writable(least)) || (greatest && !writable(greatest)))
        return runtime_result(CUDA_ERROR_INVALID_VALUE);
    std::array<int, 2> values{};
    const auto result = query_runtime_metadata(values.data(), sizeof(values), 4, 0);
    if (result != cudaSuccess) return result;
    return (!least || publish(least, values[0])) && (!greatest || publish(greatest, values[1]))
               ? cudaSuccess
               : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* available, std::size_t* total) {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (!writable(available) || !writable(total)) return runtime_result(CUDA_ERROR_INVALID_VALUE);
    std::array<std::size_t, 2> values{};
    const auto result = query_runtime_metadata(values.data(), sizeof(values), 6, 0);
    if (result != cudaSuccess) return result;
    return publish(available, values[0]) && publish(total, values[1])
               ? cudaSuccess
               : runtime_result(CUDA_ERROR_INVALID_VALUE);
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
    return runtime_result(boundary(
        [&](State&) { return publish(device, 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE; }));
}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& lock) {
        return device == 0 ? runtime_context(shared, lock, true) : CUDA_ERROR_INVALID_DEVICE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void** output, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!bytes)
            return publish(output, static_cast<void*>(nullptr)) ? CUDA_SUCCESS
                                                                : CUDA_ERROR_INVALID_VALUE;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUdeviceptr pointer{};
        const auto result = memory_allocate(shared, lock, &pointer, bytes);
        // GPU UVA is a validated numeric server address, not a host object.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* address = reinterpret_cast<void*>(pointer);
        return result == CUDA_SUCCESS && !publish(output, address) ? CUDA_ERROR_INVALID_VALUE
                                                                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS || !pointer) return initialized;
        auto* entry = shared.range(reinterpret_cast<CUdeviceptr>(pointer), 0, true);
        if (!entry || (entry->kind != Kind::kAllocation && entry->kind != Kind::kProcessAllocation))
            return CUDA_ERROR_INVALID_VALUE;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        if (entry->kind == Kind::kProcessAllocation)
            return release_process_memory(shared, lock, *entry);
        return destroy_resource(shared, lock, entry, Kind::kAllocation, ExecutionOpcode::kRelease);
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool, std::size_t bytes) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto opened = open_process_memory(shared, exclusion);
        if (opened != CUDA_SUCCESS) return opened;
        ExecutionProcessMemoryRequest request;
        request.operation = ExecutionProcessMemoryOperation::kTrimPool;
        request.bytes = bytes;
        const auto selected =
            process_pool_id(shared, reinterpret_cast<CUmemoryPool>(pool), request.pool);
        if (selected != CUDA_SUCCESS) return selected;
        const auto reply =
            call_process_memory(shared, exclusion, *shared.process_memory_context, request);
        original_status = reply.original_status;
        return reply.status;
    });
    if (result != CUDA_SUCCESS || !original_status) return runtime_result(result);
    const auto observed = static_cast<cudaError_t>(*original_status);
    if (observed != cudaSuccess) runtime_last_error = observed;
    return observed;
}
namespace {
cudaError_t runtime_symbol_result(CUresult result) noexcept {
    if (result != CUDA_ERROR_NOT_FOUND) return runtime_result(result);
    runtime_last_error = cudaErrorInvalidSymbol;
    return cudaErrorInvalidSymbol;
}
cudaError_t copy_runtime_symbol(const void* symbol, void* other, std::size_t bytes,
                                std::size_t offset, cudaMemcpyKind kind, cudaStream_t stream,
                                bool is_to_symbol, bool is_asynchronous) noexcept {
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (kind != cudaMemcpyDefault && kind != cudaMemcpyDeviceToDevice &&
        kind != (is_to_symbol ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost)) {
        runtime_last_error = cudaErrorInvalidMemcpyDirection;
        return cudaErrorInvalidMemcpyDirection;
    }
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        Entry* module{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global, &module);
        if (queried != CUDA_SUCCESS) return queried;
        if (offset > global->bytes || bytes > global->bytes - offset)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        if (module->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin module_use(module->active_calls, exclusion);
        const auto device = static_cast<CUdeviceptr>(global->address + offset);
        const auto other_device = reinterpret_cast<CUdeviceptr>(other);
        const bool is_device = kind == cudaMemcpyDeviceToDevice ||
                               (kind == cudaMemcpyDefault && shared.range(other_device, bytes));
        if (is_device) {
            return execute_memory(
                shared, exclusion,
                {.destination = is_to_symbol ? device : other_device,
                 .source = is_to_symbol ? other_device : device,
                 .count = bytes,
                 .stream = is_asynchronous ? reinterpret_cast<CUstream>(stream) : nullptr,
                 .is_asynchronous = is_asynchronous});
        }
        return transfer_host(shared, exclusion, device, other, bytes,
                             is_asynchronous ? reinterpret_cast<CUstream>(stream) : nullptr,
                             is_to_symbol, is_asynchronous, AsyncNotificationDomain::kRuntime);
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaGetSymbolAddress(void** output, const void* symbol) {
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global);
        if (queried != CUDA_SUCCESS) return queried;
        return publish(output, reinterpret_cast<void*>(global->address)) ? CUDA_SUCCESS
                                                                         : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaGetSymbolSize(std::size_t* output, const void* symbol) {
    return runtime_symbol_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        Entry* global{};
        const auto queried = runtime_symbol(shared, exclusion, symbol, global);
        if (queried != CUDA_SUCCESS) return queried;
        return publish(output, static_cast<std::size_t>(global->bytes)) ? CUDA_SUCCESS
                                                                        : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void* symbol, const void* source,
                                                    std::size_t bytes, std::size_t offset,
                                                    cudaMemcpyKind kind) {
    return copy_runtime_symbol(symbol, const_cast<void*>(source), bytes, offset, kind, nullptr,
                               true, false);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void* destination, const void* symbol,
                                                      std::size_t bytes, std::size_t offset,
                                                      cudaMemcpyKind kind) {
    return copy_runtime_symbol(symbol, destination, bytes, offset, kind, nullptr, false, false);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void* symbol, const void* source,
                                                         std::size_t bytes, std::size_t offset,
                                                         cudaMemcpyKind kind, cudaStream_t stream) {
    return copy_runtime_symbol(symbol, const_cast<void*>(source), bytes, offset, kind, stream, true,
                               true);
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void* destination, const void* symbol,
                                                           std::size_t bytes, std::size_t offset,
                                                           cudaMemcpyKind kind,
                                                           cudaStream_t stream) {
    return copy_runtime_symbol(symbol, destination, bytes, offset, kind, stream, false, true);
}
extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** output, std::size_t bytes,
                                               unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_allocate(shared, exclusion, output, bytes, flags)
                                           : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** output, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_allocate(shared, exclusion, output, bytes, 0)
                                           : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!pointer) return CUDA_SUCCESS;
        auto* entry = shared.host_range(pointer, 0, true);
        return entry ? shared.release_host(exclusion, *entry) : CUDA_ERROR_INVALID_VALUE;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int* output, void* pointer) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? host_flags(shared, output, pointer) : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** output, void* pointer,
                                                          unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        CUdeviceptr mapped{};
        const auto queried = host_pointer(shared, &mapped, pointer, flags);
        return queried == CUDA_SUCCESS && !publish(output, pointer) ? CUDA_ERROR_INVALID_VALUE
                                                                    : queried;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaHostRegister(void*, std::size_t, unsigned int) {
    return runtime_result(boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; }));
}
extern "C" cudaError_t CUDARTAPI cudaHostUnregister(void*) {
    return runtime_result(boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; }));
}
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t bytes, cudaMemcpyKind kind) {
    CopyAudit audit(CopyAudit::Api::kRuntime, bytes);
    const auto result = runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto target = reinterpret_cast<CUdeviceptr>(destination);
        const auto origin = reinterpret_cast<CUdeviceptr>(source);
        if (kind == cudaMemcpyDefault) {
            const bool to_device = shared.range(target, bytes) != nullptr;
            const bool from_device = shared.range(origin, bytes) != nullptr;
            kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                             : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
        }
        if (kind == cudaMemcpyHostToHost) {
            if (shared.graph_capture) return reject_graph_capture(shared, lock);
            if (bytes > kExecutionMaxPayloadBytes) return CUDA_ERROR_NOT_SUPPORTED;
            std::vector<std::byte> owned(bytes);
            return self_copy(owned.data(), source, bytes, false) &&
                           self_copy(destination, owned.data(), bytes, true)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        if (kind == cudaMemcpyDeviceToDevice)
            return execute_memory(shared, lock,
                                  {.destination = target, .source = origin, .count = bytes});
        return transfer_host(
            shared, lock, kind == cudaMemcpyHostToDevice ? target : origin,
            kind == cudaMemcpyHostToDevice ? const_cast<void*>(source) : destination, bytes,
            nullptr, kind == cudaMemcpyHostToDevice, false, AsyncNotificationDomain::kRuntime);
    }));
    audit.finish(result);
    return result;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact public Runtime ABI.
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t bytes, cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
    CopyAudit audit(CopyAudit::Api::kRuntimeAsync, bytes);
    const auto result = runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
            return CUDA_ERROR_INVALID_VALUE;
        if (!bytes) return CUDA_SUCCESS;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto target = reinterpret_cast<CUdeviceptr>(destination);
        const auto origin = reinterpret_cast<CUdeviceptr>(source);
        if (kind == cudaMemcpyDefault) {
            const bool to_device = shared.range(target, bytes) != nullptr;
            const bool from_device = shared.range(origin, bytes) != nullptr;
            kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                             : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
        }
        if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost)
            return transfer_host(
                shared, exclusion, kind == cudaMemcpyHostToDevice ? target : origin,
                kind == cudaMemcpyHostToDevice ? const_cast<void*>(source) : destination, bytes,
                reinterpret_cast<CUstream>(stream), kind == cudaMemcpyHostToDevice, true,
                AsyncNotificationDomain::kRuntime);
        if (kind != cudaMemcpyDeviceToDevice) return CUDA_ERROR_NOT_SUPPORTED;
        return execute_memory(shared, exclusion,
                              {.destination = target,
                               .source = origin,
                               .count = bytes,
                               .stream = reinterpret_cast<CUstream>(stream),
                               .is_asynchronous = true});
    }));
    audit.finish(result);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaMemset(void* destination, int value, std::size_t bytes) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = reinterpret_cast<CUdeviceptr>(destination),
                               .count = bytes,
                               .value = static_cast<unsigned char>(value)});
    }));
}
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* destination, int value, std::size_t bytes,
                                                 cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = reinterpret_cast<CUdeviceptr>(destination),
                               .count = bytes,
                               .value = static_cast<unsigned char>(value),
                               .stream = reinterpret_cast<CUstream>(stream),
                               .is_asynchronous = true});
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* output,
                                                           unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (flags & ~static_cast<unsigned int>(cudaStreamNonBlocking))
            return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto stopped = runtime_quiescent(shared, lock);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUstream stream{};
        const auto result = create_owned_stream(shared, lock, &stream, flags);
        return result == CUDA_SUCCESS && !publish(output, reinterpret_cast<cudaStream_t>(stream))
                   ? CUDA_ERROR_INVALID_VALUE
                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* output,
                                                              unsigned int flags, int priority) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        const auto stopped = runtime_quiescent(shared, exclusion);
        if (stopped != CUDA_SUCCESS) return stopped;
        CUstream stream{};
        const auto result = create_owned_stream(shared, exclusion, &stream, flags, priority);
        return result == CUDA_SUCCESS && !publish(output, reinterpret_cast<cudaStream_t>(stream))
                   ? CUDA_ERROR_INVALID_VALUE
                   : result;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int* output) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? owned_stream_info(shared, exclusion, reinterpret_cast<CUstream>(stream),
                                       output, true)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int* output) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? owned_stream_info(shared, exclusion, reinterpret_cast<CUstream>(stream),
                                       output, false)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                                       cudaStreamCaptureStatus* output) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        return graph_capture_status(shared, exclusion, stream, output, original);
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,
                                                        cudaStreamCaptureMode mode) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_begin(shared, exclusion, stream, mode, original)
                                           : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* output) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_end(shared, exclusion, stream, output, original)
                                           : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* output,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_instantiate(shared, exclusion, output, graph, flags, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_destroy(shared, exclusion, graph, Kind::kGraph, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? graph_destroy(shared, exclusion, executable, Kind::kGraphExecutable, original)
                   : initialized;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS ? graph_launch(shared, exclusion, executable, stream)
                                           : initialized;
    }));
}
extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec executable, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return graph_launch(shared, exclusion, executable, stream);
    });
}
extern "C" cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    return runtime_result(locked_boundary(
        [&](State&, Exclusion&) {
            cudaStreamCaptureMode requested{};
            if (!writable(mode) || !self_copy(&requested, mode, sizeof(requested), false) ||
                requested < cudaStreamCaptureModeGlobal || requested > cudaStreamCaptureModeRelaxed)
                return CUDA_ERROR_INVALID_VALUE;
            if (!publish(mode, runtime_capture_mode)) return CUDA_ERROR_INVALID_VALUE;
            runtime_capture_mode = requested;
            return CUDA_SUCCESS;
        },
        false));
}
// The SDK expands the modern six-argument entry to cudaStreamGetCaptureInfo_v2.
// Only capture status is defined outside capture; leave all other outputs alone.
extern "C" cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id,
    cudaGraph_t* graph, const cudaGraphNode_t** dependencies, std::size_t* count) {
    std::optional<std::int32_t> original;
    const auto proof = locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(status)) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        cudaStreamCaptureStatus observed{};
        const auto queried = graph_capture_status(shared, exclusion, stream, &observed, original);
        if (queried != CUDA_SUCCESS || (original && *original)) return queried;
        if (observed != cudaStreamCaptureStatusNone && (id || graph || dependencies || count))
            return reject_graph_capture(shared, exclusion);
        return publish(status, observed) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
    return graph_runtime_result(proof, original);
}
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* output, unsigned int flags) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? create_owned_event(shared, lock, reinterpret_cast<CUevent*>(output), flags)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* output) {
    return cudaEventCreateWithFlags(output, cudaEventDefault);
}
namespace {
cudaError_t runtime_event_operation(cudaEvent_t event, ExecutionOpcode operation,
                                    cudaStream_t stream = nullptr, unsigned int flags = 0) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? owned_event_operation(shared, lock, reinterpret_cast<CUevent>(event),
                                           operation, reinterpret_cast<CUstream>(stream), flags)
                   : initialized;
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return runtime_event_operation(event, ExecutionOpcode::kRecordEvent, stream);
}
extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                          unsigned int flags) {
    return runtime_event_operation(event, ExecutionOpcode::kRecordEvent, stream, flags);
}
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                     unsigned int flags) {
    return runtime_event_operation(event, ExecutionOpcode::kStreamWaitEvent, stream, flags);
}
extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kQueryEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kSynchronizeEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    return runtime_event_operation(event, ExecutionOpcode::kDestroyEvent);
}
extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* output, cudaEvent_t start,
                                                      cudaEvent_t end) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? owned_event_elapsed(shared, lock, output, reinterpret_cast<CUevent>(start),
                                         reinterpret_cast<CUevent>(end))
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* output) {
    return cudaStreamCreateWithFlags(output, cudaStreamDefault);
}
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        auto* entry = shared.find(stream, Kind::kStream);
        if (!entry || entry->context != shared.context()) return CUDA_ERROR_INVALID_HANDLE;
        const auto stopped = runtime_quiescent(shared, lock, FrontierErrorPolicy::kPhysicalOnly);
        if (stopped != CUDA_SUCCESS) return stopped;
        return destroy_resource(shared, lock, stream, Kind::kStream,
                                ExecutionOpcode::kDestroyStream);
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, stream, true, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, stream, false, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t function,
                                                    void* data) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!function) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        return initialized == CUDA_SUCCESS
                   ? enqueue_host_function(shared, exclusion, stream, function, data)
                   : initialized;
    }));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t function,
                                                         void* data) {
    return runtime_result(locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!function) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        const auto selected = stream ? stream : cudaStreamPerThread;
        return initialized == CUDA_SUCCESS
                   ? enqueue_host_function(shared, exclusion, selected, function, data)
                   : initialized;
    }));
}
namespace {
cudaError_t synchronize_runtime_device() {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        const auto initialized = runtime_context(shared, lock);
        return initialized == CUDA_SUCCESS
                   ? native_frontier(shared, lock, std::nullopt, true, nullptr, false,
                                     FrontierErrorPolicy::kIncludeHostFunctionError,
                                     AsyncNotificationDomain::kRuntime)
                   : initialized;
    }));
}
cudaError_t reset_runtime_device() {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        auto* context = shared.context();
        // Runtime reset releases its primary context, not a user-created context.
        // Its Runtime-owned subresources need a separate precise reset path.
        if (context && !context->is_primary) return CUDA_ERROR_NOT_SUPPORTED;
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, lock, 0, false);
        if (result != CUDA_SUCCESS) return result;
        if (shared.runtime_owns_primary) {
            if (!shared.primary || !shared.primary->primary_references) return CUDA_ERROR_UNKNOWN;
            --shared.primary->primary_references;
            shared.runtime_owns_primary = false;
        }
        return primary ? shared.retired_driver_error(*primary, epoch) : CUDA_SUCCESS;
    }));
}
}  // namespace
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    return synchronize_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaThreadSynchronize() {
    return synchronize_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaDeviceReset() {
    return reset_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaThreadExit() {
    return reset_runtime_device();
}
extern "C" cudaError_t CUDARTAPI cudaFuncGetAttributes(cudaFuncAttributes* output,
                                                       const void* function) {
    return runtime_result(locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            return runtime_function_control(shared, exclusion, function,
                                            ExecutionFunctionControl::kGetAttributes, 0, 0, output);
        },
        false));
}
extern "C" cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* function,
                                                      cudaFuncAttribute attribute, int value) {
    return runtime_result(locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            const auto driver_attribute = runtime_function_attribute(attribute);
            if (!driver_attribute) return CUDA_ERROR_INVALID_VALUE;
            return runtime_function_control(shared, exclusion, function,
                                            ExecutionFunctionControl::kSetAttribute,
                                            static_cast<std::uint32_t>(*driver_attribute), value);
        },
        false));
}
extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* host, dim3 grid, dim3 block,
                                                  void** arguments, std::size_t shared_bytes,
                                                  cudaStream_t stream) {
    return runtime_result(locked_boundary([&](State& shared, auto& lock) {
        if (shared_bytes > UINT32_MAX) return CUDA_ERROR_INVALID_VALUE;
        const auto initialized = runtime_context(shared, lock);
        if (initialized != CUDA_SUCCESS) return initialized;
        // Cold module/function factories require logical quiescence. Release
        // exclusion while waiting, exactly like explicit synchronization.
        RuntimeImage* image{};
        const RuntimeKernel* kernel{};
        const auto selected = select_runtime_kernel(shared, host, image, kernel);
        if (selected != CUDA_SUCCESS) return selected;
        const bool is_prepared = cached_runtime_function(shared, *image, host) != nullptr;
        bool is_capturing{};
        if (shared.graph_capture) {
            Entry* capture_stream{};
            const auto resolved = resolve_owned_stream(shared, lock, stream, capture_stream);
            if (resolved != CUDA_SUCCESS) return resolved;
            if (!capture_stream || !graph_capture_owner(shared, shared.context(), capture_stream))
                return reject_graph_capture(shared, lock);
            is_capturing = true;
        }
        if (!is_prepared && !is_capturing) {
            const auto stopped = runtime_quiescent(shared, lock);
            if (stopped != CUDA_SUCCESS) return stopped;
        }
        CUfunction function{};
        const GraphPreparationScope capture_preparation(is_capturing);
        const auto prepared = runtime_function(shared, lock, host, &function);
        if (prepared != CUDA_SUCCESS) return prepared;
        return launch_kernel(shared, lock,
                             {function,
                              {grid.x, grid.y, grid.z},
                              {block.x, block.y, block.z},
                              static_cast<unsigned int>(shared_bytes),
                              stream,
                              arguments,
                              nullptr},
                             true);
    }));
}
// Metadata-only Runtime queries never consume a virtual resource or initialize
// a caller-native context. Preserve the installed library's actual strings.
#define GLIMMER_RUNTIME_ERROR_QUERY(NAME)                                                          \
    extern "C" const char* CUDARTAPI NAME(cudaError_t error) {                                     \
        if (::getpid() != kProcess) return "cudaErrorIncompatibleDriverContext";                   \
        if (is_host_function_active) return "cudaErrorNotPermitted";                               \
        static const auto native = reinterpret_cast<decltype(&(NAME))>(::dlsym(RTLD_NEXT, #NAME)); \
        return native ? native(error) : "CUDA Runtime error metadata unavailable";                 \
    }
GLIMMER_RUNTIME_ERROR_QUERY(cudaGetErrorName)
GLIMMER_RUNTIME_ERROR_QUERY(cudaGetErrorString)
#undef GLIMMER_RUNTIME_ERROR_QUERY

namespace {
struct RuntimeEntryStyle {
    bool is_versioned{}, is_ptds{};
};
cudaError_t runtime_cached_entry(const char* name, void** output, unsigned int version,
                                 unsigned long long flags,
                                 cudaDriverEntryPointQueryResult* symbol_status,
                                 RuntimeEntryStyle style) noexcept {
    cudaError_t status{cudaErrorInvalidValue};
    if (::getpid() != kProcess) return cudaErrorIncompatibleDriverContext;
    if (is_host_function_active) return cudaErrorNotPermitted;
    if (is_entered) return cudaErrorNotSupported;
    try {
        if (writable(output) && (!symbol_status || writable(symbol_status))) {
            if (const auto symbol = owned_string(name, 256)) {
                glimmer::interceptor::execution::RuntimeEntryQuery query{
                    *symbol,       version,
                    flags,         style.is_versioned,
                    style.is_ptds, symbol_status != nullptr};
                if (!symbol_status || self_copy(&query.symbol_status, symbol_status,
                                                sizeof(query.symbol_status), false)) {
                    const auto result = glimmer::interceptor::execution::runtime_entry(query);
                    status =
                        publish(output, result.address) &&
                                (!symbol_status || self_copy(symbol_status, &result.symbol_status,
                                                             sizeof(result.symbol_status), true))
                            ? result.status
                            : cudaErrorInvalidValue;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        status = cudaErrorMemoryAllocation;
    } catch (...) {
        status = cudaErrorUnknown;
    }
    if (status != cudaSuccess) runtime_last_error = status;
    return status;
}
}  // namespace
extern "C" cudaError_t CUDARTAPI
cudaGetDriverEntryPoint(const char* name, void** output, unsigned long long flags,
                        cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, 0, flags, symbol_status, {false, false});
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char* name, void** output, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, version, flags, symbol_status, {true, false});
}
extern "C" cudaError_t CUDARTAPI
cudaGetDriverEntryPoint_ptsz(const char* name, void** output, unsigned long long flags,
                             cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, 0, flags, symbol_status, {false, true});
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char* name, void** output, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* symbol_status) {
    return runtime_cached_entry(name, output, version, flags, symbol_status, {true, true});
}
extern "C" cudaError_t CUDARTAPI cudaGetExportTable(const void** output, const cudaUUID_t* id) {
    return runtime_result(locked_boundary(
        [&](State&, auto&) {
            cudaUUID_t copied{};
            if (!writable(output) || !self_copy(&copied, id, sizeof(copied), false) ||
                !publish(output, static_cast<const void*>(nullptr)))
                return CUDA_ERROR_INVALID_VALUE;
            return CUDA_ERROR_NOT_SUPPORTED;
        },
        false));
}
