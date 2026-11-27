#include "driver_frontend_state.h"

#define GLIMMER_QUERY_BODY(NAME, PARAMETERS, BODY)                     \
    extern "C" CUresult CUDAAPI NAME PARAMETERS {                      \
        if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT; \
        if (is_host_function_active) return CUDA_ERROR_NOT_PERMITTED;  \
        static const auto native = original<decltype(&(NAME))>(#NAME); \
        if (!native) return CUDA_ERROR_NOT_INITIALIZED;                \
        return BODY;                                                   \
    }
#define GLIMMER_QUERY(NAME, PARAMETERS, ARGUMENTS) \
    GLIMMER_QUERY_BODY(NAME, PARAMETERS, native ARGUMENTS)
#define GLIMMER_DEVICE_QUERY(NAME, PARAMETERS, ARGUMENTS) \
    GLIMMER_QUERY_BODY(NAME, PARAMETERS, device_metadata_query([&] { return native ARGUMENTS; }))
GLIMMER_QUERY(cuInit, (unsigned int flags), (flags))
GLIMMER_DEVICE_QUERY(cuDeviceGet, (CUdevice * device, int ordinal), (device, ordinal))
GLIMMER_DEVICE_QUERY(cuDeviceGetCount, (int* count), (count))
GLIMMER_DEVICE_QUERY(cuDeviceGetAttribute,
                     (int* value, CUdevice_attribute attribute, CUdevice device),
                     (value, attribute, device))
GLIMMER_QUERY(cuDriverGetVersion, (int* version), (version))
GLIMMER_QUERY(cuGetErrorName, (CUresult error, const char** name), (error, name))
GLIMMER_QUERY(cuGetErrorString, (CUresult error, const char** text), (error, text))
#undef GLIMMER_QUERY
#undef GLIMMER_DEVICE_QUERY
#undef GLIMMER_QUERY_BODY

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext* output, unsigned int flags, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_context(shared, exclusion, output, {flags, device});
    });
}
// Exact native context creation ABI; count/flags cannot be reordered.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuCtxCreate_v3(CUcontext* output, CUexecAffinityParam* parameters,
                                           int count, unsigned int flags, CUdevice device) {
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (count < 0) return CUDA_ERROR_INVALID_VALUE;
    if (count != 0) return CUDA_ERROR_NOT_SUPPORTED;
    (void)parameters;  // A zero count does not consume an affinity array.
    return cuCtxCreate_v2(output, flags, device);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuCtxCreate_v4(CUcontext* output, CUctxCreateParams* parameters,
                                           unsigned int flags, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        CUctxCreateParams copied{};
        if (!self_copy(&copied, parameters, sizeof(copied), false)) return CUDA_ERROR_INVALID_VALUE;
        // A non-null, empty structure is valid in the installed Driver. NULL
        // itself is not a structure, and a positive count needs a real family.
        if ((copied.execAffinityParams && copied.cigParams) || copied.numExecAffinityParams < 0 ||
            (copied.numExecAffinityParams > 0 && !copied.execAffinityParams))
            return CUDA_ERROR_INVALID_VALUE;
        if (copied.cigParams || copied.numExecAffinityParams != 0) return CUDA_ERROR_NOT_SUPPORTED;
        return create_context(shared, exclusion, output, {flags, device});
    });
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
        const auto initialized = ensure_driver_initialization(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return initialized;
        if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
        return publish(output, reinterpret_cast<CUcontext>(shared.context()))
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void* output, CUpointer_attribute attribute,
                                                  CUdeviceptr pointer) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            ExecutionPointerQuery query;
            query.selector = static_cast<std::uint32_t>(attribute);
            return pointer_metadata(shared, exclusion, output, static_cast<std::uintptr_t>(pointer),
                                    query, original_status);
        },
        false);
    return result == CUDA_SUCCESS && original_status ? static_cast<CUresult>(*original_status)
                                                     : result;
}
extern "C" cudaError_t CUDARTAPI cudaPointerGetAttributes(cudaPointerAttributes* output,
                                                          const void* pointer) {
    std::optional<std::int32_t> original_status;
    const auto result = locked_boundary(
        [&](State& shared, Exclusion& exclusion) {
            ExecutionPointerQuery query;
            query.domain = ExecutionPointerDomain::kRuntime;
            query.selector = 0;
            return pointer_metadata(shared, exclusion, output,
                                    reinterpret_cast<std::uintptr_t>(pointer), query,
                                    original_status);
        },
        false);
    if (result != CUDA_SUCCESS || !original_status) return runtime_result(result);
    const auto observed = static_cast<cudaError_t>(*original_status);
    if (observed != cudaSuccess) runtime_last_error = observed;
    return observed;
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        auto* primary = shared.primary_entry();
        if (!primary) return CUDA_ERROR_OUT_OF_MEMORY;
        const auto waited = shared.await_context_transition(exclusion, *primary);
        if (waited != CUDA_SUCCESS) return waited;
        if (primary->is_uncertain || primary->primary_references == UINT64_MAX)
            return CUDA_ERROR_UNKNOWN;
        if (!primary->is_active) {
            const auto opened =
                shared.open_context(exclusion, *primary, primary->context_flags | CU_CTX_MAP_HOST);
            if (opened != CUDA_SUCCESS) return opened;
        }
        if (primary->primary_references == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
        ++primary->primary_references;
        if (!publish(output, reinterpret_cast<CUcontext>(primary))) {
            primary->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& exclusion) {
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, exclusion, device, true);
        return result == CUDA_SUCCESS && primary && !primary->is_active
                   ? shared.retired_driver_error(*primary, epoch)
                   : result;
    });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& exclusion) {
        auto* primary = shared.primary;
        const auto epoch = primary ? primary->epoch : 0;
        const auto result = retire_primary(shared, exclusion, device, false);
        return result == CUDA_SUCCESS && primary && !primary->is_active
                   ? shared.retired_driver_error(*primary, epoch)
                   : result;
    });
}
// Historical primary ABIs differ in reference/reset semantics. Explicitly
// reject them rather than expose a client-native context behind virtual handles.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice device) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice device) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice device, unsigned int) {
    return boundary(
        [&](State&) { return device == 0 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_DEVICE; });
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed native CUDA ABI.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice device, unsigned int* flags,
                                                       int* active) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(flags) || !writable(active)) return CUDA_ERROR_INVALID_VALUE;
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        if (shared.primary) {
            const auto waited = shared.await_context_transition(exclusion, *shared.primary);
            if (waited != CUDA_SUCCESS) return waited;
        }
        if (shared.primary && shared.primary->is_uncertain) return CUDA_ERROR_UNKNOWN;
        return publish(flags, shared.primary ? shared.primary->context_flags : 0U) &&
                       publish(active, shared.primary && shared.primary->is_active ? 1 : 0)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed native CUDA ABI.
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice device, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (device != 0) return CUDA_ERROR_INVALID_DEVICE;
        const auto checked = context_flags_status(flags);
        if (checked != CUDA_SUCCESS) return checked;
        // The primary already maps host memory. Unlike cuCtxSetFlags, this
        // original Driver entry rejects explicitly requesting that capability.
        if (flags & CU_CTX_MAP_HOST) return CUDA_ERROR_INVALID_VALUE;
        auto* primary = shared.primary_entry();
        return primary ? set_context_flags(shared, exclusion, *primary, flags)
                       : CUDA_ERROR_OUT_OF_MEMORY;
    });
}
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* output) {
    return boundary([&](State& shared) {
        if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
        return publish(output, CUdevice{0}) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuCtxGetFlags(unsigned int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        auto* context = shared.context();
        if (!context) return CUDA_ERROR_INVALID_CONTEXT;
        if (!context->is_active && context->is_primary)
            return publish(output, context->context_flags | CU_CTX_MAP_HOST)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        const auto result = shared.call(exclusion, *context, ExecutionOpcode::kGetContextFlags,
                                        ExecutionPayloadWriter{});
        if (status(result) != CUDA_SUCCESS) return status(result);
        if (result->id > UINT32_MAX) {
            context->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        auto flags = static_cast<unsigned int>(result->id);
        if (!context->is_primary)
            flags = (flags & ~static_cast<unsigned int>(CU_CTX_MAP_HOST)) |
                    (context->context_flags & CU_CTX_MAP_HOST);
        return publish(output, flags) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuCtxSetFlags(unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* context = shared.context();
        return context ? set_context_flags(shared, exclusion, *context, flags)
                       : CUDA_ERROR_INVALID_CONTEXT;
    });
}
namespace {
std::int32_t context_control(bool is_runtime, ExecutionContextControl operation,
                             std::uint32_t argument, std::uint64_t value, void* output = nullptr,
                             void* second = nullptr,
                             CUcontext explicit_context = nullptr) noexcept {
    using Operation = ExecutionContextControl;
    std::optional<std::int32_t> native_status;
    const auto prepared = locked_boundary([&](State& shared, Exclusion& exclusion) {
        const auto bytes = operation == Operation::kGetLimit      ? sizeof(std::size_t)
                           : operation == Operation::kGetAffinity ? sizeof(CUexecAffinityParam)
                           : operation == Operation::kGetCache ||
                                   operation == Operation::kGetSharedMemory ||
                                   operation == Operation::kGetApiVersion
                               ? sizeof(std::uint32_t)
                               : 0;
        if ((bytes && !writable_host_range(output, bytes)) ||
            (operation == Operation::kGetPriorityRange &&
             ((output && !writable_host_range(output, sizeof(int))) ||
              (second && !writable_host_range(second, sizeof(int))))))
            return CUDA_ERROR_INVALID_VALUE;
        if (is_runtime) {
            const auto opened = runtime_context(shared, exclusion);
            if (opened != CUDA_SUCCESS) return opened;
        }
        auto* owner =
            explicit_context ? shared.find(explicit_context, Kind::kContext) : shared.context();
        if (!owner) return CUDA_ERROR_INVALID_CONTEXT;
        if (owner->is_uncertain) return CUDA_ERROR_UNKNOWN;
        if (owner->active_calls >= kMaxHandles - 1 || owner->active_mutations >= kMaxHandles - 1)
            return CUDA_ERROR_OUT_OF_MEMORY;
        CounterPin active(owner->active_calls, exclusion);
        CounterPin mutation(owner->active_mutations, exclusion);
        ExecutionPayloadWriter input;
        if (!input.write_u32(static_cast<std::uint32_t>(operation)) || !input.write_u32(argument) ||
            !input.write_u64(value))
            return CUDA_ERROR_OUT_OF_MEMORY;
        const auto reply = shared.call(exclusion, *owner,
                                       is_runtime ? ExecutionOpcode::kRuntimeContextControl
                                                  : ExecutionOpcode::kDriverContextControl,
                                       std::move(input));
        const auto invalid = [&] {
            owner->is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        };
        if (!reply || reply->parameter_bytes || !reply->data.empty() || !reply->arguments.empty())
            return invalid();
        if (reply->disposition != ExecutionWireDisposition::kComplete) {
            if (reply->original_status || reply->status <= 0 || reply->status > 999 || reply->id ||
                reply->address)
                return invalid();
            return status(reply);
        }
        if (!reply->original_status || reply->status != *reply->original_status ||
            reply->status < 0 || reply->status > (is_runtime ? 10000 : 999))
            return invalid();
        if (reply->status != 0) {
            if (reply->id || reply->address) return invalid();
            native_status = reply->status;
            return CUDA_SUCCESS;
        }
        if (operation != Operation::kGetPriorityRange && reply->address) return invalid();
        if ((operation == Operation::kGetCache && reply->id > 3) ||
            (operation == Operation::kGetSharedMemory && reply->id > 2) ||
            ((operation == Operation::kGetApiVersion || operation == Operation::kGetAffinity ||
              operation == Operation::kGetPriorityRange) &&
             reply->id > UINT32_MAX) ||
            (operation == Operation::kGetPriorityRange && reply->address > UINT32_MAX) ||
            (operation == Operation::kGetLimit && reply->id > SIZE_MAX) ||
            (!bytes && operation != Operation::kGetPriorityRange && reply->id))
            return invalid();
        native_status = 0;
        if (operation == Operation::kGetLimit)
            return publish(static_cast<std::size_t*>(output), static_cast<std::size_t>(reply->id))
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        if (operation == Operation::kGetAffinity) {
            if (argument != 0) return invalid();
            CUexecAffinityParam parameter{};
            parameter.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
            parameter.param.smCount.val = static_cast<unsigned int>(reply->id);
            return publish(static_cast<CUexecAffinityParam*>(output), parameter)
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        if (operation == Operation::kGetPriorityRange) {
            const auto least = std::bit_cast<int>(static_cast<std::uint32_t>(reply->id));
            const auto greatest = std::bit_cast<int>(static_cast<std::uint32_t>(reply->address));
            return ((!output || publish(static_cast<int*>(output), least)) &&
                    (!second || publish(static_cast<int*>(second), greatest)))
                       ? CUDA_SUCCESS
                       : CUDA_ERROR_INVALID_VALUE;
        }
        const auto scalar = static_cast<std::uint32_t>(reply->id);
        return !bytes || self_copy(output, &scalar, sizeof(scalar), true)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
    if (prepared != CUDA_SUCCESS)
        return is_runtime ? static_cast<std::int32_t>(runtime_result(prepared))
                          : static_cast<std::int32_t>(prepared);
    if (!native_status)
        return is_runtime ? static_cast<std::int32_t>(runtime_result(CUDA_ERROR_UNKNOWN))
                          : static_cast<std::int32_t>(CUDA_ERROR_UNKNOWN);
    if (is_runtime && *native_status) runtime_last_error = static_cast<cudaError_t>(*native_status);
    return *native_status;
}
}  // namespace
extern "C" CUresult CUDAAPI cuCtxGetLimit(std::size_t* output, CUlimit limit) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetLimit,
                                                 static_cast<std::uint32_t>(limit), 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, std::size_t value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetLimit,
                                                 static_cast<std::uint32_t>(limit), value));
}
extern "C" CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache* output) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetCache, 0, 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetCache,
                                                 static_cast<std::uint32_t>(value), 0));
}
extern "C" CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig* output) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetSharedMemory, 0, 0, output));
}
extern "C" CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig value) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kSetSharedMemory,
                                                 static_cast<std::uint32_t>(value), 0));
}
extern "C" CUresult CUDAAPI cuCtxResetPersistingL2Cache() {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kResetPersistingCache, 0, 0));
}
extern "C" CUresult CUDAAPI cuCtxGetApiVersion(CUcontext context, unsigned int* output) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetApiVersion, 0,
                                                 0, output, nullptr, context));
}
extern "C" CUresult CUDAAPI cuCtxGetExecAffinity(CUexecAffinityParam* output,
                                                 CUexecAffinityType type) {
    return static_cast<CUresult>(context_control(false, ExecutionContextControl::kGetAffinity,
                                                 static_cast<std::uint32_t>(type), 0, output));
}
extern "C" CUresult CUDAAPI cuCtxGetStreamPriorityRange(int* least, int* greatest) {
    return static_cast<CUresult>(
        context_control(false, ExecutionContextControl::kGetPriorityRange, 0, 0, least, greatest));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetLimit(std::size_t* output, cudaLimit limit) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kGetLimit,
                                                    static_cast<std::uint32_t>(limit), 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetLimit(cudaLimit limit, std::size_t value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetLimit,
                                                    static_cast<std::uint32_t>(limit), value));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(cudaFuncCache* output) {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kGetCache, 0, 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(cudaFuncCache value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetCache,
                                                    static_cast<std::uint32_t>(value), 0));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* output) {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kGetSharedMemory, 0, 0, output));
}
extern "C" cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(cudaSharedMemConfig value) {
    return static_cast<cudaError_t>(context_control(true, ExecutionContextControl::kSetSharedMemory,
                                                    static_cast<std::uint32_t>(value), 0));
}
extern "C" cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache() {
    return static_cast<cudaError_t>(
        context_control(true, ExecutionContextControl::kResetPersistingCache, 0, 0));
}
extern "C" cudaError_t CUDARTAPI cudaThreadGetLimit(std::size_t* output, cudaLimit limit) {
    return cudaDeviceGetLimit(output, limit);
}
extern "C" cudaError_t CUDARTAPI cudaThreadSetLimit(cudaLimit limit, std::size_t value) {
    return cudaDeviceSetLimit(limit, value);
}
extern "C" cudaError_t CUDARTAPI cudaThreadGetCacheConfig(cudaFuncCache* output) {
    return cudaDeviceGetCacheConfig(output);
}
extern "C" cudaError_t CUDARTAPI cudaThreadSetCacheConfig(cudaFuncCache value) {
    return cudaDeviceSetCacheConfig(value);
}
extern "C" CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
    return boundary([&](State& shared) {
        auto* selected = context ? shared.find(context, Kind::kContext) : nullptr;
        if (context && (!selected || selected->is_uncertain)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!selected) {
            if (!context_stack.empty()) context_stack.pop_back();
        } else if (context_stack.empty())
            context_stack.push_back(selected);
        else
            context_stack.back() = selected;
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* selected = shared.find(context, Kind::kContext);
        if (!selected) return CUDA_ERROR_INVALID_CONTEXT;
        if (selected->is_primary) return CUDA_ERROR_INVALID_CONTEXT;
        const auto epoch = selected->epoch;
        const auto result =
            shared.call(exclusion, *selected, ExecutionOpcode::kClose, ExecutionPayloadWriter{});
        if (status(result) == CUDA_SUCCESS) {
            for (auto& entry : shared.entries)
                if (entry->context == selected) entry->is_alive = false;
            std::erase(context_stack, selected);
            return shared.retired_driver_error(*selected, epoch);
        }
        return status(result);
    });
}
extern "C" CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext context) {
    return boundary([&](State& shared) {
        auto* selected = shared.find(context, Kind::kContext);
        if (!selected || selected->is_uncertain) return CUDA_ERROR_INVALID_CONTEXT;
        if (context_stack.size() == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        context_stack.push_back(selected);
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* output) {
    return boundary([&](State& shared) {
        auto* current = shared.context();
        if (!current) return CUDA_ERROR_INVALID_CONTEXT;
        if (!publish(output, reinterpret_cast<CUcontext>(current))) return CUDA_ERROR_INVALID_VALUE;
        context_stack.pop_back();
        return CUDA_SUCCESS;
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* output, const void* image) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return load_module(shared, exclusion, output, image);
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* output, const void* image) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return load_module(shared, exclusion, output, image);
    });
}
extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* output, const void* image,
                                               unsigned int option_count, CUjit_option* options,
                                               void** values) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!option_count) return load_module(shared, exclusion, output, image);
        if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
        if (!shared.context()) return CUDA_ERROR_INVALID_CONTEXT;
        JitSubmission submission;
        const auto copied = snapshot_jit(submission, output, option_count, options, values);
        return copied == CUDA_SUCCESS ? load_module(shared, exclusion, output, image, &submission)
                                      : copied;
    });
}
extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* output, CUmodule module,
                                                const char* name) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return get_function(shared, exclusion, output, module, name);
    });
}
extern "C" CUresult CUDAAPI cuFuncGetAttribute(int* output, CUfunction_attribute attribute,
                                               CUfunction function) {
    return function_control(ExecutionFunctionControl::kGetAttribute, function,
                            std::bit_cast<std::uint32_t>(attribute), 0, output);
}
extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction function, CUfunction_attribute attribute,
                                               int value) {
    return function_control(ExecutionFunctionControl::kSetAttribute, function,
                            std::bit_cast<std::uint32_t>(attribute), value);
}
extern "C" CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction function, CUfunc_cache value) {
    return function_control(ExecutionFunctionControl::kSetCache, function,
                            std::bit_cast<std::uint32_t>(value), 0);
}
// These exact-ABI leaves contain constructor-only lookups of reference stubs.
// They grant no resource authority and deliberately never inspect caller inputs.
extern "C" CUresult CUDAAPI cuLinkCreate_v2(unsigned int, CUjit_option*, void**, CUlinkState*) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkCreate_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkAddData_v2(CUlinkState, CUjitInputType, void*, std::size_t,
                                             const char*, unsigned int, CUjit_option*, void**) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkAddData_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkAddFile_v2(CUlinkState, CUjitInputType, const char*, unsigned int,
                                             CUjit_option*, void**) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkAddFile_v2\n");
}
extern "C" CUresult CUDAAPI cuLinkComplete(CUlinkState, void**, std::size_t*) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkComplete\n");
}
extern "C" CUresult CUDAAPI cuLinkDestroy(CUlinkState) {
    return reject_constructor_only("Glimmer: unsupported owned CUDA operation: cuLinkDestroy\n");
}
extern "C" CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, CUstream,
                                                      void**) {
    return reject_constructor_only(
        "Glimmer: unsupported owned CUDA operation: cuLaunchCooperativeKernel\n");
}
extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return destroy_resource(shared, exclusion, module, Kind::kModule,
                                ExecutionOpcode::kUnloadModule);
    });
}
extern "C" CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr* address, std::size_t* bytes,
                                                 CUmodule module, const char* name) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if ((address && !writable(address)) || (bytes && !writable(bytes)))
            return CUDA_ERROR_INVALID_VALUE;
        const auto symbol = owned_string(name, 4096);
        if (!symbol || symbol->empty()) return CUDA_ERROR_INVALID_VALUE;
        Entry* global{};
        const auto result =
            get_global(shared, exclusion, global, shared.find(module, Kind::kModule), *symbol);
        if (result != CUDA_SUCCESS) return result;
        return (!address || publish(address, static_cast<CUdeviceptr>(global->address))) &&
                       (!bytes || publish(bytes, static_cast<std::size_t>(global->bytes)))
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* output, std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return memory_allocate(shared, exclusion, output, bytes);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* output, std::size_t bytes,
                                            CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return allocate_process_memory(shared, exclusion, output, bytes, nullptr, stream, false);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* output, std::size_t bytes,
                                                    CUmemoryPool pool, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return allocate_process_memory(shared, exclusion, output, bytes, pool, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return query_process_pool(shared, exclusion, output, device,
                                  ExecutionProcessMemoryOperation::kGetDefaultPool);
    });
}
extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* output, CUdevice device) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return query_process_pool(shared, exclusion, output, device,
                                  ExecutionProcessMemoryOperation::kGetCurrentPool);
    });
}
extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return set_process_pool(shared, exclusion, device, pool);
    });
}
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr pointer) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.range(pointer, 0, true);
        if (entry && entry->kind == Kind::kProcessAllocation)
            return release_process_memory(shared, exclusion, *entry);
        if (!entry || entry->kind != Kind::kAllocation) return CUDA_ERROR_INVALID_VALUE;
        return destroy_resource(shared, exclusion, entry, Kind::kAllocation,
                                ExecutionOpcode::kRelease);
    });
}
extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr destination, CUdeviceptr source,
                                     std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return unified_copy(shared, exclusion, destination, source, bytes, nullptr, false);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return unified_copy(shared, exclusion, destination, source, bytes, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr, CUdeviceptr, std::size_t) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination, const void* source,
                                            std::size_t bytes) {
    CopyAudit audit(CopyAudit::Api::kDriverUpload, bytes);
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, destination, const_cast<void*>(source), bytes,
                             nullptr, true, false);
    });
    audit.finish(result);
    return result;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void** output, std::size_t bytes, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return host_allocate(shared, exclusion, output, bytes, flags);
    });
}
extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void** output, std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return host_allocate(shared, exclusion, output, bytes, 0);
    });
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        if (!pointer) return CUDA_SUCCESS;
        auto* entry = shared.host_range(pointer, 0, true);
        return entry ? shared.release_host(exclusion, *entry) : CUDA_ERROR_INVALID_VALUE;
    });
}
extern "C" CUresult CUDAAPI cuMemHostGetFlags(unsigned int* output, void* pointer) {
    return boundary([&](State& shared) { return host_flags(shared, output, pointer); });
}
extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* output, void* pointer,
                                                         unsigned int flags) {
    return boundary([&](State& shared) { return host_pointer(shared, output, pointer, flags); });
}
extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void*, std::size_t, unsigned int) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemHostUnregister(void*) {
    return boundary([](State&) { return CUDA_ERROR_NOT_SUPPORTED; });
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void* source,
                                                 std::size_t bytes, CUstream stream) {
    CopyAudit audit(CopyAudit::Api::kDriverAsyncUpload, bytes);
    const auto result = locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, destination, const_cast<void*>(source), bytes,
                             stream, true, true);
    });
    audit.finish(result);
    return result;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* destination, CUdeviceptr source,
                                                 std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, source, destination, bytes, stream, false, true);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void* destination, CUdeviceptr source,
                                            std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return transfer_host(shared, exclusion, source, destination, bytes, nullptr, false, false);
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source,
                                            std::size_t bytes) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.destination = destination, .source = source, .count = bytes});
    });
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t bytes, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.destination = destination,
                               .source = source,
                               .count = bytes,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr destination, unsigned char value,
                                          std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr destination, unsigned char value,
                                            std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill8,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr destination, unsigned short value,
                                           std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill16,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr destination, unsigned short value,
                                             std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill16,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr destination, unsigned int value,
                                           std::size_t count) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill32,
                               .destination = destination,
                               .count = count,
                               .value = value});
    });
}
extern "C" CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr destination, unsigned int value,
                                             std::size_t count, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return execute_memory(shared, exclusion,
                              {.kind = ExecutionMemoryKind::kFill32,
                               .destination = destination,
                               .count = count,
                               .value = value,
                               .stream = stream,
                               .is_asynchronous = true});
    });
}
extern "C" CUresult CUDAAPI cuEventCreate(CUevent* output, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_event(shared, exclusion, output, flags);
    });
}
extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kRecordEvent,
                                     stream);
    });
}
extern "C" CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream,
                                                   unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kRecordEvent,
                                     stream, flags);
    });
}
extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kStreamWaitEvent,
                                     stream, flags);
    });
}
extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kQueryEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kSynchronizeEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_operation(shared, exclusion, event, ExecutionOpcode::kDestroyEvent);
    });
}
extern "C" CUresult CUDAAPI cuEventElapsedTime(float* output, CUevent start, CUevent end) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_event_elapsed(shared, exclusion, output, start, end);
    });
}
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* output, unsigned int flags) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_stream(shared, exclusion, output, flags);
    });
}
extern "C" CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* output, unsigned int flags,
                                                       int priority) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return create_owned_stream(shared, exclusion, output, flags, priority);
    });
}
extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_stream_info(shared, exclusion, stream, output, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int* output) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return owned_stream_info(shared, exclusion, stream, output, false);
    });
}
extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return destroy_resource(shared, exclusion, stream, Kind::kStream,
                                ExecutionOpcode::kDestroyStream);
    });
}
extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return locked_boundary([](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, std::nullopt, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, stream, true);
    });
}
extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    return locked_boundary([&](State& shared, std::unique_lock<std::mutex>& lock) {
        return native_frontier(shared, lock, stream, false);
    });
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_x,
                                           unsigned int grid_y, unsigned int grid_z,
                                           unsigned int block_x, unsigned int block_y,
                                           unsigned int block_z, unsigned int shared_bytes,
                                           CUstream stream, void** arguments, void** extra) {
    return locked_boundary([&](State& shared, Exclusion& exclusion) {
        return launch_kernel(shared, exclusion,
                             {function,
                              {grid_x, grid_y, grid_z},
                              {block_x, block_y, block_z},
                              shared_bytes,
                              stream,
                              arguments,
                              extra});
    });
}
namespace {
CUresult cached_entry(const char* name, void** output, int version, cuuint64_t flags,
                      CUdriverProcAddressQueryResult* symbol_status, bool is_v2) noexcept {
    if (::getpid() != kProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (is_host_function_active) return CUDA_ERROR_NOT_PERMITTED;
    if (is_entered) return CUDA_ERROR_NOT_SUPPORTED;
    Entered entered;
    try {
        if (!writable(output) || (symbol_status && !writable(symbol_status)))
            return CUDA_ERROR_INVALID_VALUE;
        const auto symbol = owned_string(name, 256);
        if (!symbol) return CUDA_ERROR_INVALID_VALUE;
        glimmer::interceptor::execution::DriverEntryQuery query{*symbol, version, flags, is_v2,
                                                                symbol_status != nullptr};
        if (symbol_status &&
            !self_copy(&query.symbol_status, symbol_status, sizeof(query.symbol_status), false))
            return CUDA_ERROR_INVALID_VALUE;
        const auto result = glimmer::interceptor::execution::driver_entry(query);
        if (!publish(output, result.address) ||
            (symbol_status &&
             !self_copy(symbol_status, &result.symbol_status, sizeof(result.symbol_status), true)))
            return CUDA_ERROR_INVALID_VALUE;
        return result.status;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
}  // namespace
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cached_entry(name, output, version, flags, nullptr, false);
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    return cached_entry(name, output, version, flags, symbol_status, true);
}
