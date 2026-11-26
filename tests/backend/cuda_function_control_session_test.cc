#include "internal/cuda_execution_session.h"
#include "internal/cuda_profile_cache.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
constexpr std::string_view kSource =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry kernel(.param .u64 pointer, .param .u32 value) { ret; }\n";
constexpr std::array kNames{"kernel",
                            "kernel__glimmer_ptb",
                            "kernel__glimmer_ptb_block_static",
                            "kernel__glimmer_ptb_dynamic",
                            "kernel__glimmer_ptb_preemptive",
                            "kernel__glimmer_ptb_sliced"};
constexpr std::array kKinds{Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive,
                            Kind::kSliced};
void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "successful owned session operation");
}
enum class Phase { kNone, kContext, kOriginal, kPrivate, kObserver, kHint };
enum class PublicAliases { kNone, kDistinct, kShared };
struct Invocation {
    CudaFunctionControl operation;
    CUfunction function;
    std::uint32_t argument;
    int value;
};
struct State {
    std::array<char, 4> contexts{};
    std::array<char, 11> functions{};
    std::array<char, 8> streams{};
    std::array<bool, 16> events{};
    char module{};
    CUcontext current{};
    unsigned int creates{}, destroys{}, loads{}, unloads{}, stream_creates{}, launches{};
    unsigned int context_calls{}, probes{};
    unsigned int observation_calls{}, function_gets{};
    unsigned int attribute_calls{}, event_queries{}, stream_syncs{};
    std::size_t lookup_kind{};
    unsigned int aliases{};
    PublicAliases public_aliases{PublicAliases::kNone};
    bool records_controls{}, can_stop{}, module_live{}, fails_parameter{}, arms_preparation_proof{};
    CUresult native_status{CUDA_SUCCESS}, private_status{CUDA_SUCCESS};
    CUresult observation_status{CUDA_SUCCESS}, preparation_status{CUDA_SUCCESS};
    CUresult context_status{CUDA_SUCCESS};
    int query_value{std::numeric_limits<int>::min()};
    int original_carveout{37};
    int normalized_carveout{63};
    std::size_t private_failure_at{1}, private_calls{};
    std::size_t preparation_failure_at{};
    std::size_t attribute_failure_at{};
    bool uses_attribute_values{};
    std::array<int, 16> attribute_values{};
    std::vector<Invocation> controls;
    std::vector<Invocation> preparation;
    std::vector<CUfunction> launched;
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::map<void*, std::size_t> host;
    Phase armed{Phase::kNone};
    std::function<void()> probe;
    std::function<void(unsigned int)> attribute_probe, context_probe;
} state;
CUfunction native_function(std::size_t index) {
    return reinterpret_cast<CUfunction>(&state.functions[index]);
}
void touch(Phase phase) {
    if (state.armed == phase && phase != Phase::kNone) {
        state.armed = Phase::kNone;
        ++state.probes;
        state.probe();
    }
}
std::byte* memory_at(CUdeviceptr address, std::size_t bytes) {
    for (auto& [base, storage] : state.memory)
        if (address >= base && address - base <= storage.size() &&
            bytes <= storage.size() - (address - base))
            return storage.data() + (address - base);
    for (const auto& [pointer, size] : state.host) {
        const auto base = reinterpret_cast<CUdeviceptr>(pointer);
        if (address >= base && address - base <= size && bytes <= size - (address - base))
            return static_cast<std::byte*>(pointer) + (address - base);
    }
    throw std::runtime_error("fake native transfer must remain in an owned allocation");
}
CUresult configure(CudaFunctionControl operation, CUfunction function, std::uint32_t argument,
                   int value) {
    expect(state.current && state.module_live, "configuration retains native module/context");
    if (!state.records_controls) {
        state.preparation.push_back({operation, function, argument, value});
        touch(Phase::kHint);
        const auto status = state.preparation_failure_at == 0 ||
                                    state.preparation.size() == state.preparation_failure_at
                                ? state.preparation_status
                                : CUDA_SUCCESS;
        if (status == CUDA_SUCCESS && function == native_function(0) &&
            argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)
            state.original_carveout = value;
        return status;
    }
    state.controls.push_back({operation, function, argument, value});
    const bool original = function == native_function(0);
    touch(original ? Phase::kOriginal : Phase::kPrivate);
    if (original) {
        if (state.native_status == CUDA_SUCCESS &&
            operation == CudaFunctionControl::kSetAttribute &&
            argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)
            state.original_carveout = state.normalized_carveout;
        return state.native_status;
    }
    return ++state.private_calls == state.private_failure_at ? state.private_status : CUDA_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected CUDA callback ABIs.
CudaSessionDispatch dispatch(unsigned int missing = 0) {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* value, int ordinal) {
        *value = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(state.creates < state.contexts.size(), "bounded independent context identity");
        *output = reinterpret_cast<CUcontext>(&state.contexts[state.creates++]);
        state.current = *output;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext value) {
        state.current = value;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext value) {
        expect(value == state.current, "never destroy an unproven current owner");
        ++state.destroys;
        state.current = nullptr;
        state.memory.clear();
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        ++state.context_calls;
        touch(Phase::kContext);
        if (state.context_probe) state.context_probe(state.context_calls);
        *output = state.current;
        return state.context_status;
    };
    sdk.module_load = [](CUmodule* output, const void* image) {
        expect(std::string_view(static_cast<const char*>(image)).find(".entry kernel(") !=
                   std::string_view::npos,
               "original entry remains in owned prepared image");
        ++state.loads;
        state.module_live = true;
        *output = reinterpret_cast<CUmodule>(&state.module);
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule module) {
        expect(module == reinterpret_cast<CUmodule>(&state.module) && state.module_live,
               "unload live original-plus-private module once");
        state.module_live = false;
        ++state.unloads;
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule, const char* name) {
        ++state.function_gets;
        const bool is_public_alias = state.public_aliases != PublicAliases::kNone &&
                                     std::string_view(name).starts_with("alias");
        const auto canonical =
            is_public_alias ? std::string("kernel") + (name + 5) : std::string(name);
        const auto found = std::ranges::find(kNames, canonical);
        expect(found != kNames.end(), "exact original and five variant names");
        state.lookup_kind = static_cast<std::size_t>(found - kNames.begin());
        auto index = state.aliases == 1   ? 0
                     : state.aliases == 2 ? std::min(state.lookup_kind, std::size_t{1})
                                          : state.lookup_kind;
        if (is_public_alias && state.public_aliases == PublicAliases::kDistinct && index != 0)
            index += 5;
        *output = native_function(index);
        if (state.arms_preparation_proof) {
            state.arms_preparation_proof = false;
            state.armed = Phase::kContext;
        }
        return CUDA_SUCCESS;
    };
    sdk.parameter_info = [](CUfunction, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        if (state.fails_parameter) return CUDA_ERROR_INVALID_VALUE;
        // In the alias fault model the fake SDK reports each requested entry's
        // ABI during preparation. This stresses deduplication, not vendor ABI parity.
        constexpr std::array<std::size_t, 7> offsets{0, 8, 12, 16, 20, 24, 32};
        constexpr std::array<std::size_t, 7> widths{8, 4, 4, 4, 4, 8, 8};
        constexpr std::array<std::size_t, 6> counts{2, 3, 5, 6, 7, 7};
        expect(index < counts[state.lookup_kind], "no out-of-range native ABI probing");
        *offset = state.lookup_kind == 5 && index == 6 ? 28 : offsets[index];
        *bytes = state.lookup_kind == 5 && index >= 5 ? 4 : widths[index];
        return CUDA_SUCCESS;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        std::vector<std::byte> memory(bytes);
        *output = reinterpret_cast<CUdeviceptr>(memory.data());
        expect(*output && state.memory.emplace(*output, std::move(memory)).second,
               "distinct owned fake device allocation");
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr pointer) {
        expect(state.memory.erase(pointer) == 1, "release exact owned device allocation once");
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        std::memcpy(memory_at(destination, bytes), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        std::memcpy(destination, memory_at(source, bytes), bytes);
        return CUDA_SUCCESS;
    };
    sdk.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                          CUstream) {
        std::memcpy(memory_at(destination, bytes), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download_async = [](void* destination, CUdeviceptr source, std::size_t bytes, CUstream) {
        std::memcpy(destination, memory_at(source, bytes), bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_allocate = [](void** output, std::size_t bytes, unsigned int) {
        *output = ::operator new(bytes, std::align_val_t{64});
        state.host.emplace(*output, bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_release = [](void* pointer) {
        expect(state.host.erase(pointer) == 1, "release private progress host owner once");
        ::operator delete(pointer, std::align_val_t{64});
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* output, void* pointer, unsigned int) {
        expect(state.host.contains(pointer), "map only private allocated host progress");
        *output = reinterpret_cast<CUdeviceptr>(pointer);
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        expect(state.stream_creates < state.streams.size(), "bounded private streams");
        *output = reinterpret_cast<CUstream>(&state.streams[state.stream_creates++]);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream) { return CUDA_SUCCESS; };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        const auto found = std::ranges::find(state.events, false);
        expect(found != state.events.end(), "bounded event lease");
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
        auto* live = reinterpret_cast<bool*>(event);
        expect(*live, "one event destruction after proof");
        *live = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++state.stream_syncs;
        return state.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.launch_kernel = [](CUfunction function, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void** extra) {
        ++state.launches;
        state.launched.push_back(function);
        expect(extra && extra[1] && extra[3], "native packed argument launch");
        const auto* parameters = static_cast<const std::byte*>(extra[1]);
        CUdeviceptr output{};
        std::memcpy(&output, parameters, sizeof(output));
        std::memcpy(memory_at(output, 4), parameters + 8, 4);
        if (function == native_function(4) || function == native_function(9)) {
            CUdeviceptr counter{};
            std::memcpy(&counter, parameters + 24, sizeof(counter));
            constexpr std::uint64_t complete{8};
            std::memcpy(memory_at(counter, sizeof(complete)), &complete, sizeof(complete));
        }
        return CUDA_SUCCESS;
    };
    sdk.occupancy = [](int* output, CUfunction, int, std::size_t) {
        *output = 4;
        return CUDA_SUCCESS;
    };
    sdk.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice) {
        *output = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 7 : 0;
        return CUDA_SUCCESS;
    };
    sdk.context_get_flags = [](unsigned int* output) {
        *output = CU_CTX_MAP_HOST;
        return CUDA_SUCCESS;
    };
    sdk.context_set_flags = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_uuid = [](CUuuid* output, CUdevice) {
        *output = {};
        output->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        *output = 12060;
        return CUDA_SUCCESS;
    };
    sdk.function_get_attribute = [](int* output, CUfunction_attribute attribute,
                                    CUfunction function) {
        expect(function == native_function(0), "getter queries only requested original function");
        if (!state.records_controls) {
            expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                   "private observer reads only the original attribute being overridden");
            ++state.observation_calls;
            *output = state.original_carveout;
            touch(Phase::kObserver);
            return state.observation_status;
        }
        state.controls.push_back({CudaFunctionControl::kGetAttribute, function,
                                  static_cast<std::uint32_t>(attribute), 0});
        ++state.attribute_calls;
        // Error also touches private scratch, never published.
        *output = attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
                      ? state.original_carveout
                  : state.uses_attribute_values
                      ? state.attribute_values.at(static_cast<std::size_t>(attribute))
                      : state.query_value;
        touch(Phase::kOriginal);
        if (state.attribute_probe) state.attribute_probe(state.attribute_calls);
        return state.attribute_failure_at && state.attribute_calls != state.attribute_failure_at
                   ? CUDA_SUCCESS
                   : state.native_status;
    };
    sdk.function_set_attribute = [](CUfunction function, CUfunction_attribute attribute,
                                    int value) {
        return configure(CudaFunctionControl::kSetAttribute, function,
                         static_cast<std::uint32_t>(attribute), value);
    };
    sdk.function_set_cache = [](CUfunction function, CUfunc_cache cache) {
        return configure(CudaFunctionControl::kSetCache, function,
                         static_cast<std::uint32_t>(cache), 0);
    };
    if (missing == 1) sdk.function_get_attribute = nullptr;
    if (missing == 2) sdk.function_set_attribute = nullptr;
    if (missing == 3) sdk.function_set_cache = nullptr;
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

struct Owner {
    std::shared_ptr<CudaProfileCache> profiles;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t module{}, function{}, allocation{}, next{1};
    std::array<std::byte, 12> parameters{};
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): independent bounded fixture modes.
    explicit Owner(unsigned int missing = 0, unsigned int aliases = 0, bool native_only = false,
                   PublicAliases public_aliases = PublicAliases::kNone) {
        state = {};
        state.aliases = aliases;
        state.public_aliases = public_aliases;
        CudaProfileConfig config{32,  std::size_t{1024} * 1024,      0, 1,
                                 {2}, {kKinds.begin(), kKinds.end()}};
        profiles = std::make_shared<CudaProfileCache>(config);
        const bool prepares_variants = missing == 0 && !native_only;
        session = std::make_unique<CudaExecutionSession>(
            ExecutionSessionIdentity{17, 29}, 31, dispatch(missing), CudaSessionLimits{},
            CudaSessionPersistentConfig{prepares_variants ? 4U : 0U}, nullptr,
            prepares_variants ? profiles : nullptr);
        success(session->initialize(0));
        auto source = std::string(kSource);
        if (public_aliases != PublicAliases::kNone)
            source += ".visible .entry alias(.param .u64 pointer, .param .u32 value) { ret; }\n";
        const auto loaded = session->load_ptx(source);
        success(loaded);
        module = loaded.id;
        const auto found = session->get_function(module, "kernel");
        success(found);
        function = found.id;
        const auto memory = session->allocate(4);
        success(memory);
        allocation = memory.id;
        std::memcpy(parameters.data(), &memory.address, 8);
        constexpr std::uint32_t value{0x12345678};
        std::memcpy(parameters.data() + 8, &value, 4);
        state.records_controls = true;
        expect(state.loads == 1 && session->snapshot().functions == 1,
               "public original function retains one loaded module");
    }
    CudaExecutionKernel kernel() const {
        return {
            .function = function, .grid = {8, 1, 1}, .block = {32, 1, 1}, .parameters = parameters};
    }
    CudaFunctionControlResult control(CudaFunctionControl operation, std::uint32_t argument = 0,
                                      int value = 0) {
        return session->function_control(operation, function, argument, value);
    }
    void launch(bool persistent, std::optional<std::size_t> expected_function = {}) {
        const auto accepted = session->accept_kernel(next++, kernel());
        expect(accepted.accepted(), "accept original application launch exactly once");
        expect(session->persistent_progress(next - 1).has_value() == persistent,
               "selection is genuinely persistent or original as required");
        expect(session->issue_next().state == CudaExecutionIssueState::kIssued,
               "issue accepted original or prepared native function");
        expect(state.launched.back() ==
                   native_function(expected_function.value_or(persistent ? 4 : 0)),
               "actual selected native function, not an acceptance-counter claim");
        expect(session->acknowledge_result(accepted.receipt), "host result acknowledgment");
        expect(session->poll().retained == 1, "ACK cannot release pending physical owners");
        state.can_stop = true;
        expect(session->poll().retired == 1, "retire exactly once after physical proof");
        state.can_stop = false;
    }
    void close() {
        success(session->release(allocation));
        success(session->unload_module(module));
        success(session->shutdown());
        expect(state.unloads == 1 && state.destroys == 1 && state.memory.empty() &&
                   state.host.empty() &&
                   std::ranges::none_of(state.events, [](bool value) { return value; }),
               "all known original/private owners retire once");
    }
};
struct PreparationConfig {
    int initial{37};
    bool has_observer{true};
};
struct PreparationOwner {
    std::shared_ptr<CudaProfileCache> profiles;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t module{};
    explicit PreparationOwner(PreparationConfig configuration = {}) {
        state = {};
        state.original_carveout = configuration.initial;
        CudaProfileConfig config{32,  std::size_t{1024} * 1024,      0, 1,
                                 {2}, {kKinds.begin(), kKinds.end()}};
        profiles = std::make_shared<CudaProfileCache>(config);
        session = std::make_unique<CudaExecutionSession>(
            ExecutionSessionIdentity{17, 29}, 31, dispatch(configuration.has_observer ? 0 : 1),
            CudaSessionLimits{}, CudaSessionPersistentConfig{4}, nullptr, profiles);
        success(session->initialize(0));
        const auto loaded = session->load_ptx(kSource);
        success(loaded);
        module = loaded.id;
    }
    CudaSessionResult prepare() {
        return session->get_function(module, "kernel");
    }
    void close() {
        success(session->unload_module(module));
        success(session->shutdown());
        expect(state.unloads == state.loads && state.destroys == 1 && state.memory.empty() &&
                   state.host.empty(),
               "preparation ownership settles once through the public lifecycle");
    }
};
void native_result(const CudaFunctionControlResult& result, CUresult original) {
    expect(result.disposition == CudaSessionDisposition::kComplete &&
               result.status == CUDA_SUCCESS && result.original_status == original,
           "requested original status is distinct from policy/private status");
}
void original_default() {
    Owner owner;
    const auto result = owner.control(CudaFunctionControl::kGetAttribute,
                                      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    native_result(result, CUDA_SUCCESS);
    expect(result.value == 37,
           "original native function default remains unchanged by private preparation");
    expect(state.observation_calls == 1 && state.preparation.size() == 6 &&
               state.original_carveout == CU_SHAREDMEM_CARVEOUT_MAX_SHARED,
           "logical observation preserves the qualified original-plus-private physical hints");
    for (std::size_t index = 0; index < state.preparation.size(); ++index)
        expect(state.preparation[index].function == native_function(index) &&
                   state.preparation[index].argument ==
                       CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
                   state.preparation[index].value == CU_SHAREDMEM_CARVEOUT_MAX_SHARED,
               "all six distinct native functions retain the exact physical carveout policy");
    owner.close();
}
void observed_defaults() {
    for (const int initial :
         {std::numeric_limits<int>::min(), -1, 0, std::numeric_limits<int>::max()}) {
        PreparationOwner owner({.initial = initial});
        const auto function = owner.prepare();
        success(function);
        state.records_controls = true;
        const auto result =
            owner.session->function_control(CudaFunctionControl::kGetAttribute, function.id,
                                            CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
        native_result(result, CUDA_SUCCESS);
        expect(result.value == initial && state.observation_calls == 1 &&
                   state.controls.size() == 1 && state.original_carveout == 100,
               "logical preference is the exact observed signed SDK value, never a guess");
        owner.close();
    }
}
void override_lifecycle(bool private_failure) {
    Owner owner;
    const auto query = [&] {
        return owner.control(CudaFunctionControl::kGetAttribute,
                             CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    };
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    native_result(owner.control(CudaFunctionControl::kSetAttribute, 8, 16), CUDA_SUCCESS);
    expect(query().value == 37, "unrelated public controls do not erase the saved preference");
    state.native_status = CUDA_ERROR_INVALID_VALUE;
    const auto failed_query = query();
    native_result(failed_query, CUDA_ERROR_INVALID_VALUE);
    expect(failed_query.value == 0, "logical remapping cannot publish an SDK-failed query");
    if (!private_failure) {
        native_result(owner.control(CudaFunctionControl::kSetAttribute,
                                    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 74),
                      CUDA_ERROR_INVALID_VALUE);
    }
    state.native_status = CUDA_SUCCESS;
    expect(query().value == 37, "failed public setter cannot invent a new logical preference");
    state.private_calls = 0;
    state.private_status = private_failure ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    native_result(owner.control(CudaFunctionControl::kSetAttribute,
                                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 74),
                  CUDA_SUCCESS);
    const auto observed = query();
    native_result(observed, CUDA_SUCCESS);
    expect(observed.value == 63 && state.original_carveout == 63 && state.observation_calls == 1,
           "successful public Set9 exposes subsequent SDK normalization, not saved37/request74");
    owner.launch(false);
    owner.close();
}
void observer_absent() {
    PreparationOwner owner({.has_observer = false});
    const auto function = owner.prepare();
    success(function);
    expect(state.observation_calls == 0 && state.preparation.size() == 6,
           "legacy optional-observer absence preserves existing physical preparation");
    state.records_controls = true;
    const auto result =
        owner.session->function_control(CudaFunctionControl::kGetAttribute, function.id,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
               result.value == 0 && state.controls.empty(),
           "missing observer cannot fabricate new public getter coverage");
    owner.close();
}
void preparation_error(bool is_observer) {
    PreparationOwner owner;
    if (is_observer)
        state.observation_status = CUDA_ERROR_NOT_SUPPORTED;
    else
        state.preparation_status = CUDA_ERROR_INVALID_VALUE;
    const auto result = owner.prepare();
    expect(
        result.disposition == CudaSessionDisposition::kRejected && result.id == 0 &&
            result.status == (is_observer ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE) &&
            result.original_status == CUDA_SUCCESS && owner.session->snapshot().functions == 0 &&
            state.observation_calls == 1 && state.preparation.size() == (is_observer ? 0U : 1U) &&
            !owner.session->snapshot().is_quarantined,
        "private preparation failure withholds function without forging original lookup error");
    owner.close();
}
void preparation_retry(bool is_parameter) {
    PreparationOwner owner;
    state.fails_parameter = is_parameter;
    if (!is_parameter) {
        state.preparation_status = CUDA_ERROR_INVALID_VALUE;
        state.preparation_failure_at = 2;
    }
    const auto failed = owner.prepare();
    expect(failed.disposition == CudaSessionDisposition::kRejected && failed.id == 0 &&
               failed.original_status == CUDA_SUCCESS && state.original_carveout == 100,
           "late preparation failure leaves an owned physical hint but no published function");
    state.fails_parameter = false;
    state.preparation_status = CUDA_SUCCESS;
    const auto function = owner.prepare();
    success(function);
    state.records_controls = true;
    const auto logical =
        owner.session->function_control(CudaFunctionControl::kGetAttribute, function.id,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    native_result(logical, CUDA_SUCCESS);
    expect(logical.value == 37 && state.observation_calls == 1 &&
               std::ranges::count_if(
                   state.preparation,
                   [](const auto& call) { return call.function == native_function(0); }) == 1,
           "retry reuses the module-owned pre-hint observation, never recaptures physical100");
    success(owner.session->unload_module(owner.module));
    state.records_controls = false;
    state.original_carveout = 11;
    const auto reloaded = owner.session->load_ptx(kSource);
    success(reloaded);
    owner.module = reloaded.id;
    const auto next = owner.prepare();
    success(next);
    state.records_controls = true;
    const auto fresh =
        owner.session->function_control(CudaFunctionControl::kGetAttribute, next.id,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    native_result(fresh, CUDA_SUCCESS);
    expect(fresh.value == 11 && state.observation_calls == 2,
           "verified module destruction removes old observations even when native handles reuse");
    owner.close();
}
void ledger_capacity(bool fails_unload) {
    struct LedgerNative {
        std::array<char, 3> modules{};
        std::array<char, 4> functions{};
        std::array<bool, 3> live{};
        std::array<int, 4> physical{37, 41, 43, 47};
        unsigned int observations{}, hints{}, parameter_queries{}, unload_attempts{};
        CUresult unload_status{CUDA_SUCCESS};
    };
    static LedgerNative ledger;
    ledger = {};
    state = {};
    auto sdk = dispatch();
    sdk.module_load = [](CUmodule* output, const void*) {
        expect(state.loads < ledger.modules.size(), "bounded ledger module fixture");
        ledger.live[state.loads] = true;
        *output = reinterpret_cast<CUmodule>(&ledger.modules[state.loads++]);
        state.module_live = true;
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule module) {
        ++ledger.unload_attempts;
        if (ledger.unload_status != CUDA_SUCCESS) return ledger.unload_status;
        const auto found = std::ranges::find_if(ledger.modules, [module](char& candidate) {
            return reinterpret_cast<CUmodule>(&candidate) == module;
        });
        expect(found != ledger.modules.end(), "unload an exact ledger fixture module");
        const auto index = static_cast<std::size_t>(found - ledger.modules.begin());
        expect(ledger.live[index], "verified ledger module unload happens once");
        ledger.live[index] = false;
        ++state.unloads;
        state.module_live = std::ranges::any_of(ledger.live, [](bool live) { return live; });
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        ++state.function_gets;
        const auto found = std::ranges::find_if(ledger.modules, [module](char& candidate) {
            return reinterpret_cast<CUmodule>(&candidate) == module;
        });
        expect(found != ledger.modules.end(), "lookup retains the exact loaded module");
        const auto index = static_cast<std::size_t>(found - ledger.modules.begin());
        expect(ledger.live[index] &&
                   (std::string_view(name) == "kernel" || std::string_view(name) == "other"),
               "only declared original entries reach native ledger lookup");
        *output = reinterpret_cast<CUfunction>(
            &ledger.functions[std::string_view(name) == "kernel" ? index : (index == 0 ? 2 : 3)]);
        return CUDA_SUCCESS;
    };
    sdk.function_get_attribute = [](int* output, CUfunction_attribute attribute,
                                    CUfunction function) {
        expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
               "ledger captures the original queried attribute");
        const auto found = std::ranges::find_if(ledger.functions, [function](char& candidate) {
            return reinterpret_cast<CUfunction>(&candidate) == function;
        });
        expect(found != ledger.functions.end(), "observe an exact original function");
        *output = ledger.physical[static_cast<std::size_t>(found - ledger.functions.begin())];
        ++ledger.observations;
        return CUDA_SUCCESS;
    };
    sdk.function_set_attribute = [](CUfunction function, CUfunction_attribute attribute,
                                    int value) {
        expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT && value == 100,
               "ledger retains the unchanged physical hint");
        const auto found = std::ranges::find_if(ledger.functions, [function](char& candidate) {
            return reinterpret_cast<CUfunction>(&candidate) == function;
        });
        expect(found != ledger.functions.end(), "hint an exact original function");
        ledger.physical[static_cast<std::size_t>(found - ledger.functions.begin())] = value;
        ++ledger.hints;
        return CUDA_SUCCESS;
    };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        ++ledger.parameter_queries;
        return CUDA_ERROR_INVALID_VALUE;  // A later clean failure never publishes a function.
    };
    CudaSessionLimits limits;
    limits.max_objects = 2;
    CudaExecutionSession session({17, 29}, 31, sdk, limits, CudaSessionPersistentConfig{4});
    success(session.initialize(0));
    const auto source = std::string(kSource) +
                        ".visible .entry other(.param .u64 pointer, .param .u32 value) { ret; }\n";
    const auto first = session.load_ptx(source);
    const auto second = session.load_ptx(source);
    success(first);
    success(second);
    for (const auto module : {first.id, second.id}) {
        const auto failed = session.get_function(module, "kernel");
        expect(failed.status == CUDA_ERROR_INVALID_VALUE &&
                   failed.original_status == CUDA_SUCCESS && failed.id == 0 &&
                   session.snapshot().functions == 0,
               "two modules retain observations after late private preparation failures");
    }
    expect(ledger.observations == 2 && ledger.hints == 2 && ledger.parameter_queries == 2,
           "each native function has one observation and physical hint before reaching the bound");
    const auto blocked = session.get_function(first.id, "other");
    expect(blocked.status == CUDA_ERROR_OUT_OF_MEMORY && blocked.original_status == CUDA_SUCCESS &&
               blocked.id == 0 && state.function_gets == 3 && ledger.observations == 2 &&
               ledger.hints == 2 && ledger.parameter_queries == 2,
           "global ledger quota rejects before observer/hint even with room in the first module");
    if (fails_unload) {
        ledger.unload_status = CUDA_ERROR_UNKNOWN;
        const auto failed = session.unload_module(first.id);
        expect(failed.status == CUDA_ERROR_UNKNOWN &&
                   failed.original_status == CUDA_ERROR_UNKNOWN &&
                   session.snapshot().modules == 2 && session.snapshot().is_quarantined &&
                   state.unloads == 0 && ledger.unload_attempts == 1,
               "failed native module destruction cannot erase retained ledger ownership");
        const auto denied = session.get_function(second.id, "other");
        expect(!denied.original_status && ledger.observations == 2 && ledger.hints == 2 &&
                   ledger.unload_attempts == 1 && state.destroys == 0,
               "uncertain unload cannot authorize refunded observer capacity or native retry");
        std::_Exit(EXIT_SUCCESS);  // Isolated process intentionally retains the uncertain owner.
    }
    success(session.unload_module(first.id));
    const auto fresh_module = session.load_ptx(source);
    success(fresh_module);
    const auto fresh = session.get_function(fresh_module.id, "kernel");
    expect(fresh.status == CUDA_ERROR_INVALID_VALUE && fresh.original_status == CUDA_SUCCESS &&
               fresh.id == 0 && ledger.observations == 3 && ledger.hints == 3 &&
               ledger.parameter_queries == 3 && ledger.physical[2] == 100,
           "verified module unload refunds exactly enough for a fresh actual observation");
    const auto full_again = session.get_function(fresh_module.id, "other");
    expect(full_again.status == CUDA_ERROR_OUT_OF_MEMORY && ledger.observations == 3 &&
               ledger.hints == 3 && ledger.parameter_queries == 3,
           "the surviving module still charges its record after another module's refund");
    success(session.unload_module(second.id));
    success(session.unload_module(fresh_module.id));
    success(session.shutdown());
    expect(state.loads == 3 && state.unloads == 3 && state.destroys == 1 &&
               std::ranges::none_of(ledger.live, [](bool live) { return live; }),
           "known ledger modules and current context clean up once");
}
void ledger_failed_unload() {
    const auto child = ::fork();
    expect(child >= 0, "fork isolated ledger unload uncertainty");
    if (child == 0) {
        try {
            ledger_capacity(true);
        } catch (const std::exception& error) {
            std::cerr << "ledger unload: " << error.what() << '\n';
        }
        std::_Exit(EXIT_FAILURE);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "failed module unload cannot refund private observation authority");
}
void preparation_reentry(Phase phase) {
    PreparationOwner owner;
    // The existing lookup cold gate precedes the new observer transaction.
    // Arm from original lookup to target that transaction's FIRST proof.
    state.arms_preparation_proof = phase == Phase::kContext;
    state.armed = phase == Phase::kContext ? Phase::kNone : phase;
    state.probe = [&] {
        const auto contexts = state.context_calls;
        const auto lookups = state.function_gets;
        const auto observations = state.observation_calls;
        const auto hints = state.preparation.size();
        expect(
            owner.session->get_function(owner.module, "kernel").status == CUDA_ERROR_NOT_READY &&
                owner.session->unload_module(owner.module).status == CUDA_ERROR_NOT_READY &&
                owner.session->shutdown().status == CUDA_ERROR_NOT_READY &&
                owner.session->activate().status == CUDA_ERROR_NOT_READY &&
                owner.session->deactivate().status == CUDA_ERROR_NOT_READY &&
                owner.session->function_control(CudaFunctionControl::kGetAttribute, 0, 9).status ==
                    CUDA_ERROR_NOT_READY &&
                owner.session->issue_next().native_status == CUDA_ERROR_NOT_READY &&
                owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                owner.session->drain_issued().status == CUDA_ERROR_NOT_READY,
            "preparation exclusion begins before observer/context/hint callbacks");
        expect(state.context_calls == contexts && state.function_gets == lookups &&
                   state.observation_calls == observations && state.preparation.size() == hints &&
                   state.unloads == 0 && state.destroys == 0,
               "nested preparation cannot issue callbacks or invalidate its owner");
    };
    success(owner.prepare());
    expect(state.probes == 1, "preparation reentry phase actually executes");
    state.probe = {};
    owner.close();
}
void preparation_context_loss(Phase phase) {
    const auto child = ::fork();
    expect(child >= 0, "fork isolated preparation context loss");
    if (child == 0) {
        try {
            PreparationOwner owner;
            state.arms_preparation_proof = phase == Phase::kContext;
            state.armed = phase == Phase::kContext ? Phase::kNone : phase;
            state.probe = [] { state.current = nullptr; };
            const auto result = owner.prepare();
            expect(result.disposition == CudaSessionDisposition::kRejected && result.id == 0 &&
                       result.status == CUDA_ERROR_INVALID_CONTEXT &&
                       owner.session->snapshot().functions == 0 &&
                       owner.session->snapshot().is_quarantined,
                   "preparation context loss never publishes a function");
            expect(result.original_status == CUDA_SUCCESS,
                   "actual original lookup supplies provenance before private preparation proof");
            expect(state.observation_calls == (phase == Phase::kContext ? 0U : 1U) &&
                       state.preparation.size() == (phase == Phase::kHint ? 1U : 0U) &&
                       state.function_gets == 1,
                   "context loss stops all later observation/hint/private preparation callbacks");
            expect(owner.session->shutdown().status != CUDA_SUCCESS && state.unloads == 0 &&
                       state.destroys == 0 && state.module_live,
                   "unknown preparation context is retained without native cleanup");
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "preparation loss phase=" << static_cast<unsigned int>(phase) << ": "
                      << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated preparation context proof");
}
void propagation() {
    for (const auto operation :
         {CudaFunctionControl::kSetAttribute, CudaFunctionControl::kSetCache}) {
        for (const unsigned int aliases : {0U, 1U, 2U}) {
            Owner owner(0, aliases);
            for (std::size_t index = 1; index < kNames.size(); ++index)
                expect(owner.session->get_function(owner.module, kNames[index]).status ==
                           CUDA_ERROR_INVALID_VALUE,
                       "all prepared variant names remain private");
            const auto argument = operation == CudaFunctionControl::kSetAttribute ? 31U : 3U;
            const auto value = operation == CudaFunctionControl::kSetAttribute
                                   ? std::numeric_limits<int>::min()
                                   : 0;
            native_result(owner.control(operation, argument, value), CUDA_SUCCESS);
            expect(state.controls.size() == (aliases ? aliases : 6U),
                   "original plus every distinct prepared variant configured exactly once");
            for (std::size_t index = 0; index < state.controls.size(); ++index) {
                const auto& call = state.controls[index];
                expect(call.operation == operation && call.function == native_function(index) &&
                           call.argument == argument && call.value == value,
                       "exact signed value and configuration propagated in original-first order");
            }
            if (!aliases) owner.launch(true);
            owner.close();
        }
    }
}
std::uint64_t prepare_public_alias(Owner& owner) {
    state.records_controls = false;
    const auto alias = owner.session->get_function(owner.module, "alias");
    state.records_controls = true;
    success(alias);
    expect(alias.id != owner.function && owner.session->snapshot().functions == 2,
           "two public names have distinct IDs but the same original native function");
    return alias.id;
}
void public_alias_group(PublicAliases layout, CudaFunctionControl operation) {
    Owner owner(0, 0, false, layout);
    const auto alias = prepare_public_alias(owner);
    native_result(
        owner.control(operation, 1, operation == CudaFunctionControl::kSetAttribute ? -7 : 0),
        CUDA_SUCCESS);
    const auto count = layout == PublicAliases::kDistinct ? 11U : 6U;
    expect(state.controls.size() == count,
           "same-original public aliases configure original once and every unique private family");
    for (std::size_t index = 0; index < count; ++index)
        expect(std::ranges::count_if(
                   state.controls,
                   [index, operation](const auto& call) {
                       return call.function == native_function(index) &&
                              call.operation == operation && call.argument == 1 &&
                              call.value ==
                                  (operation == CudaFunctionControl::kSetAttribute ? -7 : 0);
                   }) == 1,
               "alias-group configuration deduplicates every native target without omissions");
    owner.launch(true);
    owner.function = alias;
    owner.launch(true, layout == PublicAliases::kDistinct ? 9 : 4);
    owner.close();
}
void public_alias_failure(std::size_t failed_private) {
    Owner owner(0, 0, false, PublicAliases::kDistinct);
    const auto alias = prepare_public_alias(owner);
    state.native_status = failed_private == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    state.private_status = CUDA_ERROR_NOT_SUPPORTED;
    state.private_failure_at = failed_private;
    native_result(owner.control(CudaFunctionControl::kSetAttribute, 8, 16384),
                  failed_private == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS);
    expect(state.controls.size() == failed_private + 1 && !owner.session->snapshot().is_quarantined,
           "alias-group failure stops propagation while preserving the exact requested status");
    const auto first = owner.function;
    owner.launch(false);
    owner.function = alias;
    owner.launch(false);
    state.native_status = state.private_status = CUDA_SUCCESS;
    const auto before = state.controls.size();
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    expect(
        state.controls.size() == before + 1 && state.controls.back().function == native_function(0),
        "later setter through another public alias cannot revive incomplete private configuration");
    owner.launch(false);
    owner.function = first;
    owner.launch(false);
    owner.close();
}
void late_public_alias() {
    Owner owner(0, 0, false, PublicAliases::kDistinct);
    native_result(owner.control(CudaFunctionControl::kSetAttribute,
                                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 74),
                  CUDA_SUCCESS);
    const auto hints = state.preparation.size();
    owner.function = prepare_public_alias(owner);
    expect(state.preparation.size() == hints && state.observation_calls == 1 &&
               state.original_carveout == 63,
           "late aliases cannot overwrite existing settings or fabricate historical private "
           "configuration");
    const auto observed = owner.control(CudaFunctionControl::kGetAttribute,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    native_result(observed, CUDA_SUCCESS);
    expect(observed.value == 63, "late public alias observes the live shared original SDK state");
    owner.launch(false);
    owner.close();
}
void getters_and_validation() {
    Owner owner;
    for (const int value :
         {std::numeric_limits<int>::min(), -1, 0, std::numeric_limits<int>::max()}) {
        state.query_value = value;
        const auto result = owner.control(CudaFunctionControl::kGetAttribute, 31);
        native_result(result, CUDA_SUCCESS);
        expect(result.value == value, "full signed original query value");
    }
    state.native_status = CUDA_ERROR_NOT_SUPPORTED;
    const auto failed = owner.control(CudaFunctionControl::kGetAttribute, 31);
    native_result(failed, CUDA_ERROR_NOT_SUPPORTED);
    expect(failed.value == 0 && !owner.session->snapshot().is_quarantined,
           "native getter error publishes no touched scratch value or ownership fault");
    const auto before = state.controls.size();
    for (const auto operation :
         {CudaFunctionControl::kGetAttribute, CudaFunctionControl::kSetAttribute,
          CudaFunctionControl::kSetCache}) {
        const auto result =
            owner.control(operation, operation == CudaFunctionControl::kSetCache ? 4 : 32);
        expect(!result.original_status && result.status == CUDA_ERROR_INVALID_VALUE,
               "unrepresentable enum rejects before native conversion");
    }
    for (const auto operation :
         {CudaFunctionControl::kGetAttribute, CudaFunctionControl::kSetCache})
        expect(!owner.control(operation, 0, 1).original_status,
               "unused signed value is canonical zero");
    expect(!owner.control(static_cast<CudaFunctionControl>(99)).original_status,
           "unknown operation has no requested native provenance");
    for (const auto id :
         {std::uint64_t{0}, owner.module, owner.allocation,
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(native_function(4))),
          std::numeric_limits<std::uint64_t>::max()})
        expect(!owner.session->function_control(CudaFunctionControl::kGetAttribute, id, 0)
                    .original_status,
               "unknown, wrong-kind and guessed private handles have no authority");
    expect(state.controls.size() == before, "all policy rejects avoid requested native callbacks");
    state.native_status = CUDA_SUCCESS;
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    owner.launch(true);
    success(owner.session->release(owner.allocation));
    success(owner.session->unload_module(owner.module));
    expect(!owner.control(CudaFunctionControl::kGetAttribute).original_status,
           "stale function loses authority after public module unload");
    success(owner.session->shutdown());
}
void missing_callbacks() {
    for (unsigned int missing = 1; missing <= 3; ++missing) {
        Owner owner(missing);
        const auto result = owner.control(static_cast<CudaFunctionControl>(missing));
        expect(!result.original_status && result.status == CUDA_ERROR_NOT_SUPPORTED &&
                   state.controls.empty(),
               "missing optional callback is clean policy rejection, not native provenance");
        native_result(owner.control(missing == 1 ? CudaFunctionControl::kSetCache
                                                 : CudaFunctionControl::kGetAttribute),
                      CUDA_SUCCESS);
        expect(!owner.session->snapshot().is_quarantined,
               "another available callback remains usable after clean rejection");
        owner.close();
    }
}
void original_only() {
    Owner owner(0, 0, true);
    for (const auto operation :
         {CudaFunctionControl::kGetAttribute, CudaFunctionControl::kSetAttribute,
          CudaFunctionControl::kSetCache}) {
        const auto before = state.controls.size();
        native_result(owner.control(operation, 1,
                                    operation == CudaFunctionControl::kSetAttribute
                                        ? std::numeric_limits<int>::max()
                                        : 0),
                      CUDA_SUCCESS);
        expect(state.controls.size() == before + 1 &&
                   state.controls.back().function == native_function(0),
               "valid original-only functions do not require prepared variants");
    }
    owner.launch(false);
    owner.close();
}
void failure_is_sticky() {
    for (const auto operation :
         {CudaFunctionControl::kSetAttribute, CudaFunctionControl::kSetCache}) {
        for (std::size_t failure = 0; failure <= 5; ++failure) {
            Owner owner;
            state.native_status = failure == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
            state.private_status = CUDA_ERROR_NOT_SUPPORTED;
            state.private_failure_at = failure;
            const auto result = owner.control(
                operation, 1, operation == CudaFunctionControl::kSetAttribute ? -7 : 0);
            native_result(result, failure == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS);
            expect(
                state.controls.size() == failure + 1 && !owner.session->snapshot().is_quarantined,
                "stop private propagation on first failure while preserving known ownership");
            owner.launch(false);
            state.native_status = state.private_status = CUDA_SUCCESS;
            const auto before = state.controls.size();
            native_result(owner.control(operation, 1), CUDA_SUCCESS);
            expect(state.controls.size() == before + 1 &&
                       state.controls.back().function == native_function(0),
                   "later setters never retry incomplete private configuration");
            owner.launch(false);
            expect(owner.session->snapshot().persistent_accepted == 0,
                   "later successful mutation cannot restore unverified private configuration");
            owner.close();
        }
    }
}
void pending_lifetime() {
    Owner owner;
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    const auto accepted = owner.session->accept_kernel(owner.next++, owner.kernel());
    expect(accepted.accepted(), "queue one immutable application command");
    const auto before = state.controls.size();
    auto deferred = owner.control(CudaFunctionControl::kSetAttribute, 8, -1);
    expect(deferred.disposition == CudaSessionDisposition::kDeferred && !deferred.original_status,
           "queued owner excludes function mutation");
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued &&
               owner.session->acknowledge_result(accepted.receipt),
           "issue and ACK pending command");
    deferred = owner.control(CudaFunctionControl::kSetCache, 2);
    expect(deferred.disposition == CudaSessionDisposition::kDeferred && !deferred.original_status &&
               state.controls.size() == before,
           "physical owner, not host ACK, excludes mutation callbacks");
    native_result(owner.control(CudaFunctionControl::kGetAttribute, 0), CUDA_SUCCESS);
    expect(owner.session->poll().retained == 1 && state.launches == 1 && state.unloads == 0,
           "getter may observe original while physical work remains pending without drain");
    state.can_stop = true;
    expect(owner.session->poll().retired == 1,
           "physical completion settles immutable kernel owner");
    native_result(owner.control(CudaFunctionControl::kSetCache, 2), CUDA_SUCCESS);
    owner.close();
}
void reentry(Phase phase, CudaFunctionControl operation, bool has_queued = false) {
    Owner owner;
    std::optional<ExecutionAcceptance> queued;
    if (has_queued) {
        native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
        queued = owner.session->accept_kernel(owner.next++, owner.kernel());
        expect(queued->accepted(), "queued command precedes reentrant getter");
    }
    state.armed = phase;
    state.probe = [&] {
        const auto callbacks = state.controls.size();
        const auto contexts = state.context_calls;
        expect(owner.control(CudaFunctionControl::kGetAttribute).status == CUDA_ERROR_NOT_READY &&
                   owner.session->unload_module(owner.module).status == CUDA_ERROR_NOT_READY &&
                   owner.session->shutdown().status == CUDA_ERROR_NOT_READY &&
                   owner.session->activate().status == CUDA_ERROR_NOT_READY &&
                   owner.session->deactivate().status == CUDA_ERROR_NOT_READY &&
                   !owner.session->accept_kernel(owner.next, owner.kernel()).accepted() &&
                   owner.session->issue_next().native_status == CUDA_ERROR_NOT_READY &&
                   owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                   owner.session->drain_issued().status == CUDA_ERROR_NOT_READY &&
                   owner.session->reject_queued(CUDA_ERROR_UNKNOWN) == 0,
               "callback reentry cannot consume work or tear down native owners");
        expect(state.controls.size() == callbacks && state.context_calls == contexts &&
                   state.destroys == 0 && state.unloads == 0,
               "exclusion precedes even the first nested context callback");
    };
    native_result(owner.control(operation), CUDA_SUCCESS);
    expect(state.probes == 1, "requested callback phase was actually reached");
    state.probe = {};
    if (queued) {
        expect(state.launches == 0 &&
                   owner.session->issue_next().state == CudaExecutionIssueState::kIssued &&
                   state.launches == 1 && owner.session->acknowledge_result(queued->receipt),
               "nested issue/rejection leaves the actual queued command intact");
        state.can_stop = true;
        expect(owner.session->poll().retired == 1, "queued reentry control physically retires");
    }
    owner.close();
}
void context_loss(Phase phase, bool native_error, CudaFunctionControl operation) {
    Owner owner;
    state.armed = phase;
    state.native_status = native_error ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    state.probe = [] { state.current = nullptr; };
    const auto result =
        owner.control(operation, 1, operation == CudaFunctionControl::kSetAttribute ? -1 : 0);
    expect(result.disposition == CudaSessionDisposition::kRejected &&
               result.status == CUDA_ERROR_INVALID_CONTEXT && result.value == 0 &&
               owner.session->snapshot().is_quarantined,
           "lost current context rejects and faults before any further native callbacks");
    expect(phase == Phase::kContext ? !result.original_status
                                    : result.original_status == state.native_status,
           "only actually issued original control supplies provenance");
    expect(state.controls.size() == (phase == Phase::kContext    ? 0U
                                     : phase == Phase::kOriginal ? 1U
                                                                 : 2U),
           "no later original/private configuration after context loss");
    const auto callbacks = state.controls.size();
    expect(owner.session->shutdown().status != CUDA_SUCCESS && state.destroys == 0 &&
               state.unloads == 0 && state.controls.size() == callbacks && state.module_live,
           "unproven owners remain retained; no speculative cleanup or retry");
}
void isolated(Phase phase, bool native_error, CudaFunctionControl operation) {
    const auto child = ::fork();
    expect(child >= 0, "fork isolated uncertainty fixture before any session construction");
    if (child == 0) {
        try {
            context_loss(phase, native_error, operation);
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "context-loss phase=" << static_cast<unsigned int>(phase)
                      << ", operation=" << static_cast<unsigned int>(operation)
                      << ", native_error=" << native_error << ": " << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated context-loss/provenance control");
}
void canceled_owner() {
    Owner owner;
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    const auto accepted = owner.session->accept_kernel(owner.next++, owner.kernel());
    expect(accepted.accepted() && owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
               owner.session->acknowledge_result(accepted.receipt),
           "queued cancellation releases only unissued immutable ownership");
    expect(state.launches == 0, "cancellation cannot fabricate an original native launch");
    native_result(owner.control(CudaFunctionControl::kSetAttribute, 8), CUDA_SUCCESS);
    owner.close();
}
void foreign_current_context() {
    const auto child = ::fork();
    expect(child >= 0, "fork foreign-current ownership control");
    if (child == 0) {
        try {
            Owner owner;
            success(owner.session->deactivate());
            CudaExecutionSession other({18, 30}, 32, dispatch());
            success(other.initialize(0));
            const auto result = owner.control(CudaFunctionControl::kGetAttribute);
            expect(result.status == CUDA_ERROR_INVALID_CONTEXT && !result.original_status &&
                       state.controls.empty() && state.destroys == 0 && state.unloads == 0 &&
                       owner.session->snapshot().is_quarantined,
                   "another live session context cannot confer original-function authority");
            // Both owners remain intentionally retained in this isolated fault process.
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "foreign-current: " << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated foreign-current control");
}
void runtime_attribute_values() {
    state.uses_attribute_values = true;
    // Indexed independently by the installed Driver's numeric enum, not the
    // aggregate's output order. Attribute9 remains the actual mutable SDK state.
    state.attribute_values = {1024,
                              0,
                              std::numeric_limits<int>::max(),
                              8192,
                              63,
                              70,
                              80,
                              -7,
                              49152,
                              0,
                              -1,
                              2,
                              -3,
                              4,
                              std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()};
}
void expect_runtime_attributes(const CudaFunctionControlResult& result, std::size_t first,
                               int carveout = 37) {
    native_result(result, CUDA_SUCCESS);
    expect(result.attributes.has_value() && result.value == 0,
           "aggregate success publishes one complete optional and no scalar-result value");
    const std::array<std::uint64_t, 3> sizes{0, std::numeric_limits<int>::max(), 8192};
    const std::array<std::int32_t, 13> values{1024,
                                              63,
                                              70,
                                              80,
                                              -7,
                                              49152,
                                              carveout,
                                              -1,
                                              2,
                                              -3,
                                              4,
                                              std::numeric_limits<int>::max(),
                                              std::numeric_limits<int>::min()};
    expect(result.attributes->sizes == sizes && result.attributes->values == values,
           "all documented size and signed fields retain their independent exact values");
    constexpr std::array<std::uint32_t, 16> order{1, 2, 3,  0,  4,  5,  6,  7,
                                                  8, 9, 10, 11, 12, 13, 15, 14};
    expect(state.controls.size() == first + order.size(),
           "exactly sixteen original Driver observations supply one aggregate");
    for (std::size_t index = 0; index < order.size(); ++index) {
        const auto& call = state.controls[first + index];
        expect(call.operation == CudaFunctionControl::kGetAttribute &&
                   call.function == native_function(0) && call.argument == order[index],
               "aggregate Driver order and original-only identity match the installed contract");
    }
}
void runtime_attributes_success() {
    for (const bool native_only : {false, true}) {
        Owner owner(0, 0, native_only);
        runtime_attribute_values();
        const auto contexts = state.context_calls;
        const auto preparation = state.preparation.size();
        const auto result = owner.control(CudaFunctionControl::kGetAttributes);
        expect_runtime_attributes(result, 0);
        expect(state.context_calls == contexts + 32 && state.launches == 0 &&
                   state.event_queries == 0 && state.stream_syncs == 0 &&
                   state.preparation.size() == preparation &&
                   state.observation_calls == (native_only ? 0U : 1U) &&
                   state.original_carveout == (native_only ? 37 : 100),
               "every original query is bracketed without private queries, hints or GPU work");
        native_result(owner.control(CudaFunctionControl::kSetAttribute,
                                    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 74),
                      CUDA_SUCCESS);
        const auto before = state.controls.size();
        expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), before, 63);
        expect(state.original_carveout == 63 && owner.profiles->snapshot().samples == 0,
               "successful public Set9 clears remapping and exposes actual SDK normalization");
        owner.close();
    }
    for (const int initial :
         {std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max()}) {
        PreparationOwner owner({.initial = initial});
        const auto function = owner.prepare();
        success(function);
        state.records_controls = true;
        runtime_attribute_values();
        const auto result =
            owner.session->function_control(CudaFunctionControl::kGetAttributes, function.id, 0, 0);
        expect_runtime_attributes(result, 0, initial);
        expect(state.original_carveout == 100 && state.observation_calls == 1,
               "aggregate logical carveout preserves observed zero and signed extrema");
        owner.close();
    }
    Owner owner;
    runtime_attribute_values();
    state.private_status = CUDA_ERROR_NOT_SUPPORTED;
    native_result(owner.control(CudaFunctionControl::kSetAttribute,
                                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 74),
                  CUDA_SUCCESS);
    const auto before = state.controls.size();
    expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), before, 63);
    owner.launch(false);
    owner.close();
}
void runtime_attributes_failures() {
    for (std::size_t failure = 1; failure <= 16; ++failure) {
        Owner owner;
        runtime_attribute_values();
        state.attribute_failure_at = failure;
        state.native_status = CUDA_ERROR_NOT_SUPPORTED;
        const auto result = owner.control(CudaFunctionControl::kGetAttributes);
        native_result(result, CUDA_ERROR_NOT_SUPPORTED);
        expect(!result.attributes && result.value == 0 && state.controls.size() == failure &&
                   !owner.session->snapshot().is_quarantined && state.unloads == 0 &&
                   state.destroys == 0,
               "every failing Driver position stops immediately without a partial aggregate");
        state.attribute_failure_at = 0;
        state.native_status = CUDA_SUCCESS;
        const auto before = state.controls.size();
        expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), before);
        owner.close();
    }
}
void runtime_attributes_validation() {
    {
        Owner owner(1);
        const auto result = owner.control(CudaFunctionControl::kGetAttributes);
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                   !result.attributes && state.controls.empty() &&
                   !owner.session->snapshot().is_quarantined,
               "missing optional original getter rejects cleanly without invented provenance");
        owner.close();
    }
    {
        PreparationOwner owner({.has_observer = false});
        const auto function = owner.prepare();
        success(function);
        state.records_controls = true;
        const auto result =
            owner.session->function_control(CudaFunctionControl::kGetAttributes, function.id, 0, 0);
        expect(
            result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                !result.attributes && state.controls.empty() && state.observation_calls == 0 &&
                state.preparation.size() == 6,
            "legacy hidden hints without the optional getter cannot fabricate aggregate metadata");
        owner.close();
    }
    Owner owner;
    runtime_attribute_values();
    const auto contexts = state.context_calls;
    for (const auto& arguments : {std::pair{1U, 0}, std::pair{0U, 1},
                                  std::pair{std::numeric_limits<unsigned int>::max(), -1}}) {
        const auto result =
            owner.control(CudaFunctionControl::kGetAttributes, arguments.first, arguments.second);
        expect(result.status == CUDA_ERROR_INVALID_VALUE && !result.original_status &&
                   !result.attributes && state.controls.empty() && state.context_calls == contexts,
               "noncanonical aggregate arguments reject before even a context SDK callback");
    }
    expect(!owner.control(static_cast<CudaFunctionControl>(5)).original_status,
           "the next unassigned function-control operation remains rejected");
    for (const auto id :
         {std::uint64_t{0}, owner.module, owner.allocation,
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(native_function(4))),
          std::numeric_limits<std::uint64_t>::max()}) {
        const auto result =
            owner.session->function_control(CudaFunctionControl::kGetAttributes, id, 0, 0);
        expect(result.status == CUDA_ERROR_INVALID_HANDLE && !result.original_status &&
                   !result.attributes && state.controls.empty(),
               "aggregate rejects absent, foreign-kind and private native-pointer identities");
    }
    success(owner.session->release(owner.allocation));
    success(owner.session->unload_module(owner.module));
    const auto stale = owner.control(CudaFunctionControl::kGetAttributes);
    expect(stale.status == CUDA_ERROR_INVALID_HANDLE && !stale.original_status &&
               !stale.attributes && state.controls.empty(),
           "unloaded function generation cannot authorize any original attribute query");
    success(owner.session->shutdown());
}
void runtime_attributes_pending() {
    Owner owner;
    runtime_attribute_values();
    native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
    const auto accepted = owner.session->accept_kernel(owner.next++, owner.kernel());
    expect(accepted.accepted(), "queue real immutable function ownership before aggregate query");
    auto before = state.controls.size();
    expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), before);
    expect(state.launches == 0 && state.event_queries == 0 && state.stream_syncs == 0,
           "aggregate does not issue, poll or synchronize a queued command");
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued &&
               owner.session->acknowledge_result(accepted.receipt),
           "issue and acknowledge the retained immutable command");
    const auto queries = state.event_queries, synchronizations = state.stream_syncs;
    before = state.controls.size();
    expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), before);
    expect(state.launches == 1 && state.event_queries == queries &&
               state.stream_syncs == synchronizations && state.unloads == 0 && state.destroys == 0,
           "aggregate leaves pending physical work and every native owner untouched");
    const auto deferred = owner.control(CudaFunctionControl::kSetAttribute, 8, 1);
    expect(deferred.disposition == CudaSessionDisposition::kDeferred && !deferred.original_status &&
               !deferred.attributes,
           "pending-work getter does not weaken the existing setter exclusion");
    state.can_stop = true;
    expect(owner.session->poll().retired == 1, "real event proof retires the one pending command");
    owner.close();
}
template <typename Action>
void runtime_attributes_isolated(Action action) {
    const auto child = ::fork();
    expect(child >= 0, "fork bounded aggregate ownership-fault control");
    if (child == 0) {
        try {
            action();
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "runtime attributes: " << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated aggregate fault retains unproven owners without destructor cleanup");
}
void runtime_attributes_corrupt_sizes() {
    for (std::size_t field = 1; field <= 3; ++field)
        runtime_attributes_isolated([field] {
            Owner owner;
            runtime_attribute_values();
            state.attribute_values[field] = -1;
            const auto allocations = state.memory.size(), host = state.host.size();
            const auto result = owner.control(CudaFunctionControl::kGetAttributes);
            expect(result.disposition == CudaSessionDisposition::kRejected &&
                       result.status == CUDA_ERROR_UNKNOWN &&
                       result.original_status == CUDA_SUCCESS && !result.attributes &&
                       result.value == 0 && state.controls.size() == field &&
                       owner.session->snapshot().is_quarantined,
                   "negative successful size faults ownership without unsigned wrap or partial "
                   "output");
            const auto again = owner.control(CudaFunctionControl::kGetAttributes);
            expect(again.status == CUDA_ERROR_UNKNOWN && !again.original_status &&
                       !again.attributes && state.controls.size() == field && state.unloads == 0 &&
                       state.destroys == 0 && state.memory.size() == allocations &&
                       state.host.size() == host,
                   "metadata fault prevents another query without implicit cleanup or replay");
        });
}
void runtime_attributes_context_faults() {
    for (unsigned int proof = 1; proof <= 32; ++proof)
        runtime_attributes_isolated([proof] {
            Owner owner;
            runtime_attribute_values();
            const auto initial = state.context_calls;
            state.context_probe = [initial, proof](unsigned int call) {
                if (call == initial + proof) state.current = nullptr;
            };
            const auto result = owner.control(CudaFunctionControl::kGetAttributes);
            expect(result.disposition == CudaSessionDisposition::kRejected &&
                       result.status == CUDA_ERROR_INVALID_CONTEXT && !result.attributes &&
                       result.value == 0 && state.controls.size() == proof / 2 &&
                       state.context_calls == initial + proof &&
                       owner.session->snapshot().is_quarantined,
                   "each pre/post context failure stops the aggregate at the exact native prefix");
            expect(proof == 1 ? !result.original_status : result.original_status == CUDA_SUCCESS,
                   "only an actually issued original Driver query supplies failure provenance");
            state.context_probe = {};
            expect(owner.session->shutdown().status != CUDA_SUCCESS && state.unloads == 0 &&
                       state.destroys == 0 && state.module_live,
                   "lost context cannot authorize module or context cleanup");
        });
    for (const unsigned int field : {1U, 8U, 16U})
        runtime_attributes_isolated([field] {
            Owner owner;
            runtime_attribute_values();
            state.attribute_failure_at = field;
            state.native_status = CUDA_ERROR_NOT_SUPPORTED;
            state.attribute_probe = [field](unsigned int call) {
                if (call == field) state.context_status = CUDA_ERROR_OUT_OF_MEMORY;
            };
            const auto result = owner.control(CudaFunctionControl::kGetAttributes);
            expect(result.disposition == CudaSessionDisposition::kRejected &&
                       result.status == CUDA_ERROR_OUT_OF_MEMORY &&
                       result.original_status == CUDA_ERROR_NOT_SUPPORTED && !result.attributes &&
                       state.controls.size() == field && owner.session->snapshot().is_quarantined &&
                       state.unloads == 0 && state.destroys == 0,
                   "failed post-proof preserves the exact separate failing Driver-query status");
        });
}
void runtime_attributes_reentry() {
    for (const bool during_context : {false, true})
        for (unsigned int phase = 1; phase <= (during_context ? 32U : 16U); ++phase) {
            Owner owner;
            runtime_attribute_values();
            native_result(owner.control(CudaFunctionControl::kSetCache, 1), CUDA_SUCCESS);
            const auto queued = owner.session->accept_kernel(owner.next++, owner.kernel());
            expect(queued.accepted(), "queue immutable work before each aggregate reentry phase");
            unsigned int reached{};
            const auto contexts = state.context_calls, queries = state.attribute_calls;
            const auto check = [&](unsigned int call) {
                if (call != (during_context ? contexts : queries) + phase) return;
                ++reached;
                const auto context_count = state.context_calls;
                const auto controls = state.controls.size();
                const auto nested = owner.control(CudaFunctionControl::kGetAttributes);
                expect(
                    nested.disposition == CudaSessionDisposition::kDeferred &&
                        nested.status == CUDA_ERROR_NOT_READY && !nested.original_status &&
                        !nested.attributes &&
                        owner.control(CudaFunctionControl::kSetAttribute, 9, -1).status ==
                            CUDA_ERROR_NOT_READY &&
                        owner.session->unload_module(owner.module).status == CUDA_ERROR_NOT_READY &&
                        owner.session->shutdown().status == CUDA_ERROR_NOT_READY &&
                        owner.session->activate().status == CUDA_ERROR_NOT_READY &&
                        owner.session->deactivate().status == CUDA_ERROR_NOT_READY &&
                        owner.session->issue_next().native_status == CUDA_ERROR_NOT_READY &&
                        owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                        owner.session->drain_issued().status == CUDA_ERROR_NOT_READY &&
                        owner.session->reject_queued(CUDA_ERROR_UNKNOWN) == 0,
                    "whole-aggregate exclusion blocks recursive metadata, mutation and cleanup");
                expect(state.context_calls == context_count && state.controls.size() == controls &&
                           state.launches == 0 && state.unloads == 0 && state.destroys == 0,
                       "reentry cannot invoke a second SDK callback or consume queued ownership");
            };
            if (during_context)
                state.context_probe = check;
            else
                state.attribute_probe = check;
            const auto first = state.controls.size();
            expect_runtime_attributes(owner.control(CudaFunctionControl::kGetAttributes), first);
            expect(reached == 1, "every intended aggregate callback phase was actually exercised");
            state.context_probe = {};
            state.attribute_probe = {};
            expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued &&
                       state.launches == 1 && owner.session->acknowledge_result(queued.receipt),
                   "outer aggregate leaves exactly one original queued command available");
            state.can_stop = true;
            expect(owner.session->poll().retired == 1,
                   "reentry fixture has known physical cleanup");
            owner.close();
        }
}
void runtime_attributes() {
    runtime_attributes_success();
    runtime_attributes_failures();
    runtime_attributes_validation();
    runtime_attributes_pending();
    runtime_attributes_corrupt_sizes();
    runtime_attributes_context_faults();
    runtime_attributes_reentry();
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--original-default") {
            original_default();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--public-alias") {
            public_alias_group(PublicAliases::kDistinct, CudaFunctionControl::kSetAttribute);
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--runtime-attributes") {
            runtime_attributes();
            std::cout << "Atomic original Runtime function-attribute checks passed\n";
            return EXIT_SUCCESS;
        }
        expect(
            argc == 1,
            "supported isolated modes: --original-default, --public-alias, --runtime-attributes");
        original_default();
        observed_defaults();
        override_lifecycle(false);
        override_lifecycle(true);
        observer_absent();
        preparation_error(true);
        preparation_error(false);
        preparation_retry(true);
        preparation_retry(false);
        ledger_capacity(false);
        ledger_failed_unload();
        for (const auto phase : {Phase::kContext, Phase::kObserver, Phase::kHint}) {
            preparation_reentry(phase);
            preparation_context_loss(phase);
        }
        propagation();
        for (const auto layout : {PublicAliases::kDistinct, PublicAliases::kShared})
            for (const auto operation :
                 {CudaFunctionControl::kSetAttribute, CudaFunctionControl::kSetCache})
                public_alias_group(layout, operation);
        for (const std::size_t failure : {0U, 1U, 7U, 10U}) public_alias_failure(failure);
        late_public_alias();
        getters_and_validation();
        missing_callbacks();
        original_only();
        failure_is_sticky();
        pending_lifetime();
        canceled_owner();
        foreign_current_context();
        runtime_attributes();
        for (const auto phase : {Phase::kContext, Phase::kOriginal, Phase::kPrivate}) {
            for (const auto operation :
                 {CudaFunctionControl::kSetAttribute, CudaFunctionControl::kSetCache}) {
                reentry(phase, operation);
                isolated(phase, false, operation);
                if (phase != Phase::kPrivate) isolated(phase, true, operation);
            }
        }
        for (const auto phase : {Phase::kContext, Phase::kOriginal}) {
            reentry(phase, CudaFunctionControl::kGetAttribute, true);
            isolated(phase, false, CudaFunctionControl::kGetAttribute);
            isolated(phase, true, CudaFunctionControl::kGetAttribute);
        }
        std::cout << "Owned function configuration: propagation, sticky fallback, physical leases "
                     "and callback exclusion passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
