#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
void expect(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned setup/cleanup succeeds");
}
enum class Phase { kProof, kInitialize, kSelect, kVersion, kDriver, kRuntime };
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
struct State {
    std::array<char, 4> contexts{};
    char module{}, function{};
    std::array<std::byte, 32> global{}, host{};
    CUcontext current{}, module_context{};
    std::map<CUdeviceptr, Memory> allocations;
    std::array<bool, 16> events{};
    unsigned int creates{}, destroys{}, unloads{}, releases{}, launches{}, event_queries{}, syncs{};
    unsigned int driver_calls{}, runtime_calls{}, version_calls{};
    bool initialized{}, can_stop{}, module_live{}, host_live{};
    int version{12060};
    CUresult initialize_status{CUDA_SUCCESS}, select_status{CUDA_SUCCESS};
    cudaError_t version_status{cudaSuccess};
    std::size_t fail_proof{};
    CUresult proof_status{CUDA_ERROR_OUT_OF_MEMORY};
    std::vector<Phase> trace;
    std::vector<CUcontext> query_contexts;
    CUdeviceptr last_input{};
    std::function<void(Phase)> probe;
    std::function<CUresult(void*, CUpointer_attribute, CUdeviceptr)> driver;
    std::function<cudaError_t(cudaPointerAttributes*, const void*)> runtime;
} state;
void touch(Phase phase) {
    state.trace.push_back(phase);
    if (state.probe) state.probe(phase);
}
CUdeviceptr host_address() {
    return reinterpret_cast<CUdeviceptr>(state.host.data()) + 4096;
}
struct Classification {
    int type{}, device{-2};
    CUdeviceptr device_pointer{}, host_pointer{};
};
Classification classify(CUdeviceptr pointer) {
    for (const auto& [base, allocation] : state.allocations)
        if (pointer >= base && pointer - base < allocation.bytes.size())
            return {2, 7, state.current ? pointer : 0, 0};
    const auto global = reinterpret_cast<CUdeviceptr>(state.global.data());
    if (state.module_live && pointer >= global && pointer - global < state.global.size())
        return {2, 7, state.current == state.module_context ? pointer : 0, 0};
    const auto host = reinterpret_cast<CUdeviceptr>(state.host.data());
    if (state.host_live && pointer >= host && pointer - host < state.host.size())
        return {1, 5, state.current ? host_address() + (pointer - host) : 0, pointer};
    return {0, -2, 0, pointer};
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): injected CUDA ABI signatures.
CudaSessionDispatch dispatch(unsigned int missing = 0) {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int flags) {
        expect(flags == 0, "metadata initialization uses exact flags zero");
        touch(Phase::kInitialize);
        if (state.initialize_status == CUDA_SUCCESS) state.initialized = true;
        return state.initialize_status;
    };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(state.creates < state.contexts.size(), "bounded fake contexts");
        *output = reinterpret_cast<CUcontext>(&state.contexts[state.creates++]);
        state.current = *output;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        touch(Phase::kSelect);
        if (state.select_status == CUDA_SUCCESS) state.current = context;
        return state.select_status;
    };
    sdk.context_destroy = [](CUcontext context) {
        expect(context == state.current, "cleanup runs on the exact current context");
        ++state.destroys;
        state.current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        touch(Phase::kProof);
        *output = state.current;
        if (state.fail_proof && static_cast<std::size_t>(std::ranges::count(
                                    state.trace, Phase::kProof)) == state.fail_proof)
            return state.proof_status;
        return state.initialized ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
    };
    sdk.module_load = [](CUmodule* output, const void*) {
        *output = reinterpret_cast<CUmodule>(&state.module);
        state.module_live = true;
        state.module_context = state.current;
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule module) {
        expect(module == reinterpret_cast<CUmodule>(&state.module) && state.module_live,
               "release live original module once");
        ++state.unloads;
        state.module_live = false;
        return CUDA_SUCCESS;
    };
    sdk.module_global = [](CUdeviceptr* output, std::size_t* bytes, CUmodule, const char*) {
        *output = reinterpret_cast<CUdeviceptr>(state.global.data());
        *bytes = state.global.size();
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule, const char*) {
        *output = reinterpret_cast<CUfunction>(&state.function);
        return CUDA_SUCCESS;
    };
    sdk.parameter_info = [](CUfunction, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        expect(index == 0, "original kernel has one pointer parameter");
        *offset = 0;
        *bytes = 8;
        return CUDA_SUCCESS;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        Memory memory{state.current, std::vector<std::byte>(bytes)};
        *output = reinterpret_cast<CUdeviceptr>(memory.bytes.data());
        expect(state.allocations.emplace(*output, std::move(memory)).second,
               "fake native allocation identities are distinct");
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr pointer) {
        expect(state.allocations.erase(pointer) == 1, "release owned allocation once");
        ++state.releases;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr, const void*, std::size_t) { return CUDA_SUCCESS; };
    sdk.download = [](void*, CUdeviceptr, std::size_t) { return CUDA_SUCCESS; };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        *output = reinterpret_cast<CUstream>(&state.function);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream) { return CUDA_SUCCESS; };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) {
        ++state.launches;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        const auto found = std::ranges::find(state.events, false);
        expect(found != state.events.end(), "bounded physical event inventory");
        *found = true;
        *output = reinterpret_cast<CUevent>(&*found);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent, CUstream) { return CUDA_SUCCESS; };
    sdk.execution.event_query = [](CUevent) {
        ++state.event_queries;
        return state.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        auto* value = reinterpret_cast<bool*>(event);
        expect(*value, "physical event destroyed once");
        *value = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++state.syncs;
        return state.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.runtime.version = [](int* output) {
        ++state.version_calls;
        touch(Phase::kVersion);
        *output = state.version;
        return state.version_status;
    };
    sdk.pointer_get_attribute = [](void* output, CUpointer_attribute attribute,
                                   CUdeviceptr pointer) {
        ++state.driver_calls;
        state.last_input = pointer;
        state.query_contexts.push_back(state.current);
        touch(Phase::kDriver);
        if (state.driver) return state.driver(output, attribute, pointer);
        const auto value = classify(pointer);
        if (!value.type) return CUDA_ERROR_INVALID_VALUE;
        if (attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL) {
            *static_cast<int*>(output) = value.device;
            return CUDA_SUCCESS;
        }
        expect(attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER, "only actual reference selectors");
        *static_cast<CUdeviceptr*>(output) = value.device_pointer;
        return value.device_pointer ? CUDA_SUCCESS
               : state.current      ? CUDA_ERROR_INVALID_VALUE
                                    : CUDA_ERROR_INVALID_CONTEXT;
    };
    sdk.runtime.pointer_attributes = [](cudaPointerAttributes* output, const void* pointer) {
        ++state.runtime_calls;
        state.last_input = reinterpret_cast<CUdeviceptr>(pointer);
        state.query_contexts.push_back(state.current);
        touch(Phase::kRuntime);
        if (state.runtime) return state.runtime(output, pointer);
        const auto value = classify(reinterpret_cast<CUdeviceptr>(pointer));
        output->type = static_cast<cudaMemoryType>(value.type);
        output->device = value.device;
        // Synthetic SDK aliases are only compared as bits, never dereferenced.
        // NOLINTBEGIN(performance-no-int-to-ptr)
        output->devicePointer = reinterpret_cast<void*>(value.device_pointer);
        output->hostPointer = reinterpret_cast<void*>(value.host_pointer);
        // NOLINTEND(performance-no-int-to-ptr)
        return cudaSuccess;
    };
    if (missing == 1) sdk.pointer_get_attribute = nullptr;
    if (missing == 2) sdk.runtime.pointer_attributes = nullptr;
    if (missing == 3) sdk.runtime.version = nullptr;
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
constexpr std::string_view kSource =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .global .align 4 .u32 variable[8];\n"
    ".visible .entry kernel(.param .u64 pointer) { ret; }\n";
struct Fixture {
    std::unique_ptr<CudaExecutionSession> carrier, source, querying;
    std::uint64_t allocation{}, module{}, global{}, function{};
    CUdeviceptr address{};
    CUcontext source_context{}, query_context{};
    std::shared_ptr<int> host_lease = std::make_shared<int>(1);
    explicit Fixture(unsigned int missing = 0) {
        state = {};
        carrier = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{1, 2}, 3,
                                                         dispatch(missing));
        source =
            std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{1, 2}, 4, dispatch());
        querying =
            std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{1, 2}, 5, dispatch());
        success(source->initialize(0));
        source_context = state.current;
        const auto memory = source->allocate(32);
        success(memory);
        allocation = memory.id;
        address = memory.address;
        const auto loaded = source->load_ptx(kSource);
        success(loaded);
        module = loaded.id;
        const auto symbol = source->get_global(module, "variable");
        success(symbol);
        global = symbol.id;
        const auto kernel = source->get_function(module, "kernel");
        success(kernel);
        function = kernel.id;
        success(source->deactivate());
        success(querying->initialize(0));
        query_context = state.current;
        state.trace.clear();
    }
    CudaPointerQuery device(CudaPointerDomain domain = CudaPointerDomain::kRuntime,
                            std::uint32_t selector = 0, bool is_global = false) const {
        return {domain, selector, CudaPointerKind::kDevice, {is_global ? global : allocation, 3}};
    }
    CudaHostMemoryView host() {
        state.host_live = true;
        return {state.host.data() + 3, host_address() + 3, 29, host_lease};
    }
    CudaPointerMetadataResult query(const CudaPointerQuery& value, bool has_context = true) {
        return carrier->pointer_metadata(
            value, has_context ? querying.get() : nullptr,
            value.kind == CudaPointerKind::kDevice ? source.get() : nullptr,
            value.kind == CudaPointerKind::kHost ? std::optional{host()} : std::nullopt);
    }
    CudaExecutionKernel kernel(std::array<std::byte, 8>& parameters) const {
        std::memcpy(parameters.data(), &address, sizeof(address));
        return {
            .function = function, .grid = {1, 1, 1}, .block = {32, 1, 1}, .parameters = parameters};
    }
    void close(unsigned int expected_releases = 1) {
        state.probe = {};
        state.driver = {};
        state.runtime = {};
        success(source->activate());
        success(source->release(allocation));
        success(source->unload_module(module));
        success(source->shutdown());
        success(querying->activate());
        success(querying->shutdown());
        success(carrier->shutdown());
        expect(state.creates == 2 && state.destroys == 2 && state.unloads == 1 &&
                   state.releases == expected_releases && state.allocations.empty(),
               "carrier creates no context and normal source cleanup remains exact");
    }
};
const CudaPointerObservation& complete(const CudaPointerMetadataResult& value, int native = 0) {
    expect(value.disposition == CudaSessionDisposition::kComplete && value.status == CUDA_SUCCESS &&
               value.context_selection == CudaPointerContextSelection::kQuery &&
               value.original_status == native && value.observation.has_value(),
           "healthy original result and independent output effects");
    return *value.observation;
}
void rejected(const CudaPointerMetadataResult& value, CUresult status) {
    expect(value.disposition == CudaSessionDisposition::kRejected && value.status == status &&
               !value.original_status && !value.observation,
           "preparation rejection has no requested native provenance or output");
}
void classes_and_contexts() {
    Fixture fixture;
    for (bool has_context : {false, true}) {
        for (bool is_global : {false, true}) {
            auto value = fixture.query(fixture.device(CudaPointerDomain::kRuntime, 0, is_global),
                                       has_context);
            const auto observation = complete(value);
            const bool accessible = has_context && !is_global;
            expect(observation.effects == 15 && observation.type == 2 && observation.device == 7 &&
                       observation.host_alias == CudaPointerAlias::kNull &&
                       observation.device_alias ==
                           (accessible ? CudaPointerAlias::kInput : CudaPointerAlias::kNull),
                   "runtime respects current context, source ordinal and unavailable aliases");
            value = fixture.query(fixture.device(CudaPointerDomain::kDriver, 3, is_global),
                                  has_context);
            const auto& pointer = complete(value, accessible ? 0 : has_context ? 1 : 201);
            expect(pointer.effects == 4 && pointer.device_alias == observation.device_alias,
                   "Driver errors can write a verified null alias");
            value = fixture.query(fixture.device(CudaPointerDomain::kDriver, 9, is_global),
                                  has_context);
            const auto& ordinal = complete(value);
            expect(ordinal.effects == 2 && ordinal.device == 7,
                   "ordinal is the source device, not current device");
            expect(state.current == (has_context ? fixture.query_context : nullptr),
                   "query leaves its selected context current without restoration");
            expect(state.query_contexts.back() == (has_context ? fixture.query_context : nullptr) &&
                       state.last_input == (is_global
                                                ? reinterpret_cast<CUdeviceptr>(state.global.data())
                                                : fixture.address) +
                                               3,
                   "native callback receives the exact source address under the explicit query "
                   "context");
        }
        auto value = fixture.query({CudaPointerDomain::kRuntime, 0, CudaPointerKind::kHost, {}},
                                   has_context);
        const auto& host = complete(value);
        expect(host.effects == 15 && host.type == 1 && host.device == 5 &&
                   host.host_alias == CudaPointerAlias::kInput &&
                   host.device_alias ==
                       (has_context ? CudaPointerAlias::kInput : CudaPointerAlias::kNull),
               "different private host/device addresses project independently");
        value =
            fixture.query({CudaPointerDomain::kDriver, 3, CudaPointerKind::kHost, {}}, has_context);
        const auto& device_alias = complete(value, has_context ? 0 : 201);
        expect(device_alias.effects == 4 &&
                   device_alias.device_alias ==
                       (has_context ? CudaPointerAlias::kInput : CudaPointerAlias::kNull) &&
                   state.last_input == reinterpret_cast<CUdeviceptr>(state.host.data() + 3),
               "Driver host query validates a distinct private GPU alias, not the CPU address");
        value =
            fixture.query({CudaPointerDomain::kDriver, 9, CudaPointerKind::kHost, {}}, has_context);
        expect(complete(value).device == 5, "mapped-host ordinal is independently native observed");
    }
    auto value =
        fixture.carrier->pointer_metadata(fixture.device(CudaPointerDomain::kDriver, 3, true),
                                          fixture.source.get(), fixture.source.get());
    expect(complete(value).device_alias == CudaPointerAlias::kInput,
           "same query/source owner scopes are deduplicated");
    value = fixture.source->pointer_metadata(fixture.device(), fixture.source.get(),
                                             fixture.source.get());
    expect(complete(value).device_alias == CudaPointerAlias::kInput,
           "carrier/query/source may all be the same scoped owner");
    for (const auto kind : {CudaPointerKind::kNull, CudaPointerKind::kUnregistered}) {
        value = fixture.query({CudaPointerDomain::kRuntime, 0, kind, {}}, false);
        const auto& unknown = complete(value);
        expect(
            unknown.effects == 15 && unknown.type == 0 && unknown.device == -2 &&
                unknown.device_alias == CudaPointerAlias::kNull &&
                unknown.host_alias == (kind == CudaPointerKind::kNull ? CudaPointerAlias::kNull
                                                                      : CudaPointerAlias::kInput),
            "address-only sentinel preserves null/unregistered semantics");
        const auto sentinel = state.last_input;
        value = fixture.query({CudaPointerDomain::kDriver, 3, kind, {}}, false);
        expect(complete(value, 1).effects == 0 && state.last_input == sentinel,
               "one stable carrier sentinel; native untouched error stays untouched");
        value = fixture.query({CudaPointerDomain::kDriver, 9, kind, {}}, false);
        expect(complete(value, 1).effects == 0,
               "unknown ordinal error also preserves untouched output");
    }
    expect(state.launches == 0 && state.event_queries == 0 && state.syncs == 0,
           "metadata issues no application work, event query or synchronization");
    fixture.close();
}
void output_effects() {
    Fixture fixture;
    state.driver = [](void* output, CUpointer_attribute attribute, CUdeviceptr pointer) {
        if (attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
            *static_cast<CUdeviceptr*>(output) = pointer;
        else
            *static_cast<int*>(output) = -7;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    auto value = fixture.query(fixture.device(CudaPointerDomain::kDriver, 3));
    expect(complete(value, 801).effects == 4 &&
               value.observation->device_alias == CudaPointerAlias::kInput,
           "verified alias mutation survives a native Driver error");
    value = fixture.query(fixture.device(CudaPointerDomain::kDriver, 9));
    expect(complete(value, 801).effects == 2 && value.observation->device == -7,
           "Driver error scalar is not reinterpreted as pointer authority");
    for (std::uint32_t mask = 0; mask != 16; ++mask) {
        state.runtime = [mask](cudaPointerAttributes* output, const void* pointer) {
            const std::int32_t raw_type = -91;
            if (mask & 1) std::memcpy(&output->type, &raw_type, sizeof(raw_type));
            if (mask & 2) output->device = -17;
            // Model a distinct native host/device alias without accessing it.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            if (mask & 4) output->devicePointer = reinterpret_cast<void*>(host_address() + 3);
            if (mask & 8) output->hostPointer = const_cast<void*>(pointer);
            return cudaErrorInvalidDeviceFunction;
        };
        value = fixture.query({CudaPointerDomain::kRuntime, 0, CudaPointerKind::kHost, {}});
        const auto& observed = complete(value, static_cast<int>(cudaErrorInvalidDeviceFunction));
        expect(
            observed.effects == mask && observed.type == ((mask & 1) ? -91 : 0) &&
                observed.device == ((mask & 2) ? -17 : 0) &&
                observed.device_alias ==
                    ((mask & 4) ? CudaPointerAlias::kInput : CudaPointerAlias::kNull) &&
                observed.host_alias ==
                    ((mask & 8) ? CudaPointerAlias::kInput : CudaPointerAlias::kNull),
            "all sixteen Runtime error masks retain raw signed bits and canonical inactive fields");
    }
    fixture.close();
}
void invalid_and_private_errors() {
    for (unsigned int missing = 1; missing != 4; ++missing) {
        Fixture fixture(missing);
        rejected(fixture.query(fixture.device(
                     missing == 1 ? CudaPointerDomain::kDriver : CudaPointerDomain::kRuntime,
                     missing == 1 ? 3 : 0)),
                 CUDA_ERROR_NOT_SUPPORTED);
        expect(state.trace.empty(), "missing callback rejects before any SDK observation");
        fixture.close();
    }
    Fixture fixture;
    auto invalid = fixture.device();
    for (const auto offset : {std::size_t{32}, std::numeric_limits<std::size_t>::max()}) {
        invalid.range.offset = offset;
        rejected(fixture.query(invalid), CUDA_ERROR_INVALID_VALUE);
    }
    invalid = fixture.device();
    invalid.range.allocation = fixture.module;
    rejected(fixture.query(invalid), CUDA_ERROR_INVALID_VALUE);
    invalid = fixture.device(CudaPointerDomain::kDriver, 4);
    rejected(fixture.query(invalid), CUDA_ERROR_INVALID_VALUE);
    invalid.domain = static_cast<CudaPointerDomain>(3);
    rejected(fixture.query(invalid), CUDA_ERROR_INVALID_VALUE);
    invalid = {CudaPointerDomain::kRuntime, 1, CudaPointerKind::kNull, {}};
    rejected(fixture.query(invalid), CUDA_ERROR_INVALID_VALUE);
    rejected(fixture.carrier->pointer_metadata(fixture.device()), CUDA_ERROR_INVALID_VALUE);
    rejected(fixture.carrier->pointer_metadata(
                 {CudaPointerDomain::kRuntime, 0, CudaPointerKind::kNull, {}},
                 fixture.querying.get(), fixture.source.get()),
             CUDA_ERROR_INVALID_VALUE);
    auto host = fixture.host();
    host.lifetime.reset();
    rejected(fixture.carrier->pointer_metadata(
                 {CudaPointerDomain::kRuntime, 0, CudaPointerKind::kHost, {}},
                 fixture.querying.get(), nullptr, host),
             CUDA_ERROR_INVALID_VALUE);
    expect(state.trace.empty(), "invalid authority never calls a native callback");
    for (int version : {0, 11080, 12059, 12061}) {
        state.version = version;
        rejected(fixture.query(fixture.device()), CUDA_ERROR_NOT_SUPPORTED);
    }
    state.version = 12060;
    state.version_status = cudaErrorUnknown;
    rejected(fixture.query(fixture.device()), CUDA_ERROR_NOT_SUPPORTED);
    state.version_status = cudaSuccess;
    state.initialize_status = CUDA_ERROR_OUT_OF_MEMORY;
    rejected(fixture.query(fixture.device()), CUDA_ERROR_OUT_OF_MEMORY);
    state.initialize_status = CUDA_SUCCESS;
    state.select_status = CUDA_ERROR_INVALID_CONTEXT;
    rejected(fixture.query(fixture.device()), CUDA_ERROR_INVALID_CONTEXT);
    state.select_status = CUDA_SUCCESS;
    expect(state.driver_calls == 0 && state.runtime_calls == 0 &&
               !fixture.carrier->snapshot().is_quarantined,
           "private failures cannot acquire requested-getter provenance or dirty known owners");
    fixture.close();
}
void cold_carrier() {
    state = {};
    CudaExecutionSession carrier({1, 2}, 3, dispatch());
    const CudaPointerQuery unknown{
        CudaPointerDomain::kRuntime, 0, CudaPointerKind::kUnregistered, {}};
    state.initialize_status = CUDA_ERROR_OUT_OF_MEMORY;
    rejected(carrier.pointer_metadata(unknown), CUDA_ERROR_OUT_OF_MEMORY);
    expect(!state.initialized && !carrier.snapshot().is_quarantined && state.runtime_calls == 0,
           "failed first initialization remains a clean preparation failure");
    state.initialize_status = CUDA_SUCCESS;
    const auto result = carrier.pointer_metadata(unknown);
    expect(complete(result).type == 0 && state.initialized && state.creates == 0 && !state.current,
           "first metadata query initializes Driver metadata without a CUDA context");
    success(carrier.shutdown());
    expect(state.destroys == 0, "uninitialized carrier shutdown never destroys a context");
}
void pending_and_reentry() {
    const std::array phases{Phase::kProof,   Phase::kInitialize, Phase::kSelect,
                            Phase::kVersion, Phase::kDriver,     Phase::kRuntime};
    for (unsigned int scenario = 0; scenario != 13; ++scenario) {
        const auto armed = scenario < phases.size() ? phases[scenario] : Phase::kProof;
        const std::size_t occurrence = scenario < phases.size() ? 1 : scenario - 4;
        Fixture fixture;
        success(fixture.source->activate());
        std::array<std::byte, 8> parameters{};
        const auto accepted = fixture.source->accept_kernel(1, fixture.kernel(parameters));
        expect(accepted.accepted(), "preaccepted real dispatcher command");
        bool touched{};
        std::size_t observed{};
        state.probe = [&](Phase phase) {
            if (touched || phase != armed || ++observed != occurrence) return;
            touched = true;
            const auto callbacks = state.trace.size();
            for (auto* owner :
                 {fixture.carrier.get(), fixture.source.get(), fixture.querying.get()}) {
                expect(owner->pointer_metadata(
                                {CudaPointerDomain::kDriver, 3, CudaPointerKind::kNull, {}})
                               .disposition == CudaSessionDisposition::kDeferred,
                       "nested pointer query is deferred before SDK calls");
                expect(owner->shutdown().disposition == CudaSessionDisposition::kDeferred &&
                           owner->initialize(0).disposition == CudaSessionDisposition::kDeferred,
                       "all participating owners exclude initialization and destruction");
                owner->close();
                expect(owner->activate().status != CUDA_SUCCESS &&
                           owner->deactivate().status != CUDA_SUCCESS,
                       "activation cannot change the query's current context");
            }
            expect(fixture.source->release(fixture.allocation).disposition ==
                           CudaSessionDisposition::kDeferred &&
                       fixture.source->unload_module(fixture.module).disposition ==
                           CudaSessionDisposition::kDeferred,
                   "source exclusion protects explicit free and module unload");
            (void)fixture.source->issue_next();
            (void)fixture.source->poll();
            (void)fixture.source->drain_issued();
            expect(state.trace.size() == callbacks && state.launches == 0 &&
                       state.event_queries == 0 && state.syncs == 0,
                   "reentry neither consumes the queued command nor invokes another SDK callback");
        };
        state.trace.clear();
        const auto domain =
            armed == Phase::kDriver ? CudaPointerDomain::kDriver : CudaPointerDomain::kRuntime;
        auto result =
            fixture.query(fixture.device(domain, domain == CudaPointerDomain::kDriver ? 3 : 0));
        complete(result);
        expect(touched && state.launches == 0, "query does not drain queued work");
        state.probe = {};
        success(fixture.source->activate());
        expect(fixture.source->issue_next().state == CudaExecutionIssueState::kIssued &&
                   fixture.source->acknowledge_result(accepted.receipt),
               "queued command issues once after query");
        expect(fixture.source->poll().retained == 1, "physical event remains pending after ACK");
        const auto polls = state.event_queries;
        const auto syncs = state.syncs;
        result = fixture.query(fixture.device());
        complete(result);
        expect(state.event_queries == polls && state.syncs == syncs && state.launches == 1,
               "metadata also does not poll, drain or replay physically pending work");
        state.can_stop = true;
        success(fixture.source->activate());
        expect(fixture.source->poll().retired == 1, "independent actual stop retires command once");
        fixture.close();
    }
}
void stale_ranges_and_host_lease() {
    Fixture fixture;
    success(fixture.source->activate());
    success(fixture.source->release(fixture.allocation));
    state.trace.clear();
    rejected(fixture.query(fixture.device()), CUDA_ERROR_INVALID_VALUE);
    expect(state.trace.empty(), "released device identity is never reclassified by the backend");
    const auto replacement = fixture.source->allocate(32);
    success(replacement);
    expect(replacement.id != fixture.allocation,
           "reused native address cannot reuse logical identity");
    fixture.allocation = replacement.id;
    fixture.address = replacement.address;
    auto view = fixture.host();
    const std::weak_ptr<const void> retained = view.lifetime;
    fixture.host_lease.reset();
    bool saw_lease{};
    state.probe = [&](Phase) {
        expect(!retained.expired(),
               "host mapping lease remains alive through every native callback");
        saw_lease = true;
    };
    const auto value = fixture.carrier->pointer_metadata(
        {CudaPointerDomain::kRuntime, 0, CudaPointerKind::kHost, {}}, fixture.querying.get(),
        nullptr, std::move(view));
    complete(value);
    expect(saw_lease && retained.expired(),
           "CPU host lease is released only after observation finishes");
    state.probe = {};
    fixture.close(2);
}
void isolated(const std::function<void()>& body) {
    const auto pid = ::fork();
    expect(pid >= 0, "isolate deliberately faulted ownership");
    if (!pid) {
        try {
            body();
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated ownership/projection case passed");
}
void proof_and_projection_failures() {
    for (const auto domain : {CudaPointerDomain::kDriver, CudaPointerDomain::kRuntime}) {
        const std::size_t proofs = domain == CudaPointerDomain::kDriver ? 6 : 8;
        for (std::size_t position = 1; position <= proofs; ++position) {
            isolated([=] {
                // Deliberate uncertain owners survive until this child's exit.
                auto* fixture = new Fixture;
                state.fail_proof = position;
                const auto value = fixture->query(
                    fixture->device(domain, domain == CudaPointerDomain::kDriver ? 3 : 0));
                expect(
                    value.disposition == CudaSessionDisposition::kRejected &&
                        value.context_selection == CudaPointerContextSelection::kUnproven &&
                        value.status == CUDA_ERROR_OUT_OF_MEMORY && !value.observation &&
                        value.original_status.has_value() == (position == proofs),
                    "every proof failure preserves exact getter provenance only if already issued");
                expect(fixture->carrier->snapshot().is_quarantined &&
                           fixture->source->snapshot().is_quarantined &&
                           fixture->querying->snapshot().is_quarantined && state.releases == 0 &&
                           state.unloads == 0 && state.destroys == 0 &&
                           static_cast<std::size_t>(
                               std::ranges::count(state.trace, Phase::kProof)) == position,
                       "all participating owners fault with no callback continuation or cleanup "
                       "guess");
            });
        }
    }
    for (unsigned int corruption = 0; corruption != 8; ++corruption) {
        isolated([=] {
            auto* fixture = new Fixture;
            state.runtime = [corruption](cudaPointerAttributes* output, const void* input) {
                output->type = cudaMemoryTypeDevice;
                output->device = 7;
                output->devicePointer = const_cast<void*>(input);
                output->hostPointer = nullptr;
                switch (corruption) {
                    case 0:
                        output->devicePointer = reinterpret_cast<void*>(123);
                        break;
                    case 1:
                        output->hostPointer = const_cast<void*>(input);
                        break;
                    case 2:
                        output->type = cudaMemoryTypeHost;
                        break;
                    case 3:
                        output->device = -2;
                        break;
                    case 4: {
                        const int invalid = 99;
                        std::memcpy(&output->type, &invalid, sizeof(invalid));
                        break;
                    }
                    case 5:
                        output->device = -3;
                        break;
                    case 6:
                        output->hostPointer = reinterpret_cast<void*>(123);
                        break;
                    default:
                        output->devicePointer = reinterpret_cast<void*>(123);
                        break;
                }
                return corruption >= 6 ? cudaErrorInvalidValue : cudaSuccess;
            };
            const auto value = fixture->query(fixture->device());
            expect(value.disposition == CudaSessionDisposition::kRejected &&
                       value.status == CUDA_ERROR_UNKNOWN &&
                       value.original_status ==
                           (corruption >= 6 ? static_cast<int>(cudaErrorInvalidValue) : 0) &&
                       !value.observation && fixture->source->snapshot().is_quarantined,
                   "malformed successes and native-error fields cannot leak unowned pointers");
        });
    }
    isolated([] {
        auto* fixture = new Fixture;
        state.runtime = [](cudaPointerAttributes*, const void*) { return cudaSuccess; };
        const auto value = fixture->query(fixture->device());
        expect(
            value.status == CUDA_ERROR_UNKNOWN && value.original_status == 0 && !value.observation,
            "successful untouched poisoned fields cannot masquerade as a complete observation");
    });
    isolated([] {
        auto* fixture = new Fixture;
        state.runtime = [](cudaPointerAttributes* output, const void* input) {
            expect(input != nullptr, "unregistered input uses the nonnull private sentinel");
            output->type = cudaMemoryTypeUnregistered;
            output->device = -2;
            output->devicePointer = nullptr;
            output->hostPointer = nullptr;
            return cudaSuccess;
        };
        const auto value = fixture->query(
            {CudaPointerDomain::kRuntime, 0, CudaPointerKind::kUnregistered, {}}, false);
        expect(value.disposition == CudaSessionDisposition::kRejected &&
                   value.status == CUDA_ERROR_UNKNOWN && value.original_status == 0 &&
                   !value.observation && fixture->carrier->snapshot().is_quarantined &&
                   !fixture->source->snapshot().is_quarantined &&
                   !fixture->querying->snapshot().is_quarantined && state.runtime_calls == 1,
               "unregistered success must return the exact sentinel host alias, not null");
    });
    for (const auto domain : {CudaPointerDomain::kDriver, CudaPointerDomain::kRuntime}) {
        isolated([=] {
            auto* fixture = new Fixture;
            state.driver = [](void*, CUpointer_attribute, CUdeviceptr) {
                return CUDA_ERROR_NOT_SUPPORTED;
            };
            state.runtime = [](cudaPointerAttributes*, const void*) {
                return cudaErrorInvalidValue;
            };
            state.fail_proof = domain == CudaPointerDomain::kDriver ? 6 : 8;
            const auto value = fixture->query(
                fixture->device(domain, domain == CudaPointerDomain::kDriver ? 9 : 0));
            expect(value.status == CUDA_ERROR_OUT_OF_MEMORY && !value.observation &&
                       value.original_status == (domain == CudaPointerDomain::kDriver
                                                     ? static_cast<int>(CUDA_ERROR_NOT_SUPPORTED)
                                                     : static_cast<int>(cudaErrorInvalidValue)),
                   "failed post-proof retains the exact already observed native error domain");
        });
    }
    for (unsigned int corruption = 0; corruption != 4; ++corruption) {
        isolated([=] {
            auto* fixture = new Fixture;
            state.driver = [corruption](void* output, CUpointer_attribute, CUdeviceptr) {
                if (corruption < 2)
                    *static_cast<CUdeviceptr*>(output) = corruption ? 0 : 123;
                else if (corruption == 2)
                    *static_cast<int*>(output) = -2;
                return corruption == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
            };
            const auto value =
                fixture->query(fixture->device(CudaPointerDomain::kDriver, corruption < 2 ? 3 : 9));
            expect(value.status == CUDA_ERROR_UNKNOWN && !value.observation &&
                       value.original_status == (corruption == 0 ? 1 : 0),
                   "Driver raw aliases and contradictory successful scalars never gain authority");
        });
    }
}
void context_mutations() {
    // Mutating callbacks still return their requested status. Only the following
    // independent context observation detects the lost proof.
    for (const auto armed :
         {Phase::kInitialize, Phase::kVersion, Phase::kDriver, Phase::kRuntime}) {
        isolated([=] {
            auto* fixture = new Fixture;
            bool touched{};
            state.probe = [&](Phase phase) {
                if (!touched && phase == armed) {
                    touched = true;
                    state.current = reinterpret_cast<CUcontext>(&state.contexts[3]);
                }
            };
            const bool driver = armed == Phase::kDriver;
            const auto value = fixture->query(fixture->device(
                driver ? CudaPointerDomain::kDriver : CudaPointerDomain::kRuntime, driver ? 9 : 0));
            expect(touched && value.disposition == CudaSessionDisposition::kRejected &&
                       value.context_selection == CudaPointerContextSelection::kUnproven &&
                       value.status == CUDA_ERROR_INVALID_CONTEXT && !value.observation &&
                       value.original_status.has_value() == (driver || armed == Phase::kRuntime) &&
                       fixture->carrier->snapshot().is_quarantined &&
                       fixture->source->snapshot().is_quarantined &&
                       fixture->querying->snapshot().is_quarantined,
                   "context-changing callbacks fault every owner and cannot publish observations");
        });
    }
}
void wrong_thread() {
    Fixture fixture;
    CudaPointerMetadataResult result;
    std::thread other([&] { result = fixture.query(fixture.device()); });
    other.join();
    rejected(result, CUDA_ERROR_INVALID_CONTEXT);
    expect(result.context_selection == CudaPointerContextSelection::kUnchanged,
           "rejected foreign entry has no native selection effect");
    expect(state.trace.empty(), "foreign thread cannot invoke any SDK callback");
    fixture.close();
}
void selection_effects_are_not_original_statuses() {
    Fixture fixture;
    auto query = fixture.device();
    query.range.offset = std::numeric_limits<std::size_t>::max();
    auto value = fixture.query(query);
    rejected(value, CUDA_ERROR_INVALID_VALUE);
    expect(
        value.context_selection == CudaPointerContextSelection::kUnchanged && state.trace.empty(),
        "pre-SDK invalid input cannot claim a native selection");
    state.version = 12000;
    value = fixture.query(fixture.device());
    rejected(value, CUDA_ERROR_NOT_SUPPORTED);
    expect(value.context_selection == CudaPointerContextSelection::kQuery &&
               state.current == fixture.query_context && !state.runtime_calls,
           "version rejection preserves proven query selection without getter provenance");
    state.version = 12060;
    state.select_status = CUDA_ERROR_INVALID_CONTEXT;
    value = fixture.query(fixture.device());
    rejected(value, CUDA_ERROR_INVALID_CONTEXT);
    expect(value.context_selection == CudaPointerContextSelection::kUnchanged,
           "failed selection with proven unchanged TLS preserves the previous owner");
    state.select_status = CUDA_SUCCESS;
    value = fixture.query(fixture.device(), false);
    complete(value);
    expect(value.context_selection == CudaPointerContextSelection::kQuery && !state.current,
           "explicit no-current query selection is not an unchanged-owner receipt");
    fixture.close();
}
}  // namespace
int main() {
    try {
        classes_and_contexts();
        output_effects();
        invalid_and_private_errors();
        cold_carrier();
        pending_and_reentry();
        stale_ranges_and_host_lease();
        proof_and_projection_failures();
        context_mutations();
        wrong_thread();
        selection_effects_are_not_original_statuses();
    } catch (const std::exception& error) {
        std::cerr << "pointer metadata: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
