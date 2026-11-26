#include "internal/cuda_execution_session.h"
#include "context_control_fixture.h"
#include "internal/cuda_sgemm_image.h"
#include "internal/cuda_ptx_cache.h"
#include "internal/cuda_profile_cache.h"
#include "../transform/cuda_image_fixture.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
std::uint64_t profile_now{100};
std::uint64_t profile_clock() noexcept {
    return profile_now;
}
constexpr std::string_view kPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
// .entry kernel(.param .u64 fake0, .param .u64 fake1, .param .u32 fake2)
.file 1 "ignored .entry kernel(.param .u32 false_parameter)"
.visible .entry kernel(.param .u64 pointer, /* .param .u64 ignored */ .param .u32 value) { ret; }
)ptx";
constexpr std::string_view kPersistentPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry kernel(.param .u64 pointer, .param .u32 value) { ret; }
)ptx";
constexpr std::string_view kUnsupportedPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry kernel(.param .u64 pointer, .param .u32 value) { trap; ret; }
)ptx";
void expect(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
struct Event {
    bool is_live{};
};
struct FakeState {
    char context_storage{}, module_storage{}, function_storage{}, private_function_storage{},
        stream_storage{}, binary_module_storage{}, candidate_function_storage{};
    CUcontext current{};
    CUresult create_status{CUDA_SUCCESS}, destroy_status{CUDA_SUCCESS}, free_status{CUDA_SUCCESS};
    CUresult lookup_status{CUDA_SUCCESS}, parameter_status{CUDA_SUCCESS},
        allocation_status{CUDA_SUCCESS};
    CUresult query_status{CUDA_ERROR_NOT_READY};
    CUresult module_status{CUDA_SUCCESS}, unload_status{CUDA_SUCCESS}, stream_status{CUDA_SUCCESS},
        stream_destroy_status{CUDA_SUCCESS};
    bool returns_partial_context{}, returns_null_context{}, returns_partial_allocation{},
        returns_null_allocation{};
    bool has_overlapping_abi{}, erases_on_free_error{};
    bool returns_partial_module{}, returns_null_module{}, returns_partial_stream{},
        returns_null_stream{};
    unsigned int unloads{}, stream_destroys{}, module_loads{};
    std::vector<CUmodule> unloaded_modules;
    CudaExecutionSession* budget_session{};
    std::vector<std::size_t> module_load_charges;
    bool has_shared_binary{}, validation_module_retired{};
    unsigned int private_load_failure{};
    unsigned int companion_mode{};
    bool has_bad_companion_abi{};
    std::string stateful_source;
    unsigned int calls{}, context_creates{}, context_destroys{}, frees{}, allocations{}, copies{},
        launches{};
    unsigned int event_records{}, event_waits{};
    std::function<void()> on_event_record;
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::array<Event, 16> events{};
    bool is_persistent{}, is_bad_private_abi{}, returns_null_host{}, returns_partial_host{};
    Kind kind{Kind::kPreemptive};
    bool uses_multiple_kinds{};
    std::array<char, 5> variant_storage{};
    std::vector<std::optional<Kind>> launched_kinds;
    std::string loaded_ptx;
    std::optional<Kind> bad_abi_kind, zero_occupancy_kind;
    unsigned char uuid_tag{1};
    std::vector<std::array<unsigned int, 3>> physical_grids, logical_grids;
    std::vector<std::array<std::uint32_t, 2>> slices;
    int occupancy_capacity{4};
    std::vector<CUfunction> configured_functions;
    std::size_t failed_configuration{};
    CUresult host_status{CUDA_SUCCESS}, host_free_status{CUDA_SUCCESS}, map_status{CUDA_SUCCESS};
    CUresult prefix_status{CUDA_SUCCESS};
    unsigned int host_allocations{}, host_frees{}, prefix_copies{}, readbacks{};
    std::uint64_t claimed{3};
    void* host{};
    std::vector<std::uint64_t> counter_before_launch;
    bool fails_generated_load{};
    unsigned int generated_load_failures{};
    bool is_binary{};
    std::vector<std::byte> expected_binary;
    const void* owned_binary{};
    unsigned int binary_loads{}, parameter_queries{};
    unsigned int context_flags{}, flag_queries{}, flag_updates{};
    CUresult flag_status{CUDA_SUCCESS};
    void* imported_host{};
    CUresult registration_status{CUDA_SUCCESS}, unregister_status{CUDA_SUCCESS};
    CUresult imported_map_status{CUDA_SUCCESS}, unregistered_map_status{CUDA_ERROR_INVALID_VALUE};
    bool returns_unaligned_mapping{};
    unsigned int registrations{}, unregistrations{};
    std::array<char, 16> blas_storage{};
    std::map<cublasHandle_t, cudaStream_t> blas;
    std::map<cublasHandle_t, cublasPointerMode_t> blas_modes;
    std::map<cublasHandle_t, cublasMath_t> blas_math;
    struct BlasPolicy {
        CUdeviceptr workspace{};
        std::size_t workspace_bytes{};
        bool has_workspace{};
        unsigned int atomics{};
        int sm_count{};
    };
    std::map<cublasHandle_t, BlasPolicy> blas_policy;
    unsigned int workspace_sets{}, atomics_sets{}, sm_count_sets{};
    cublasStatus_t workspace_status{CUBLAS_STATUS_SUCCESS};
    bool returns_bad_atomics{}, returns_bad_sm_count{};
    cublasStatus_t blas_create_status{CUBLAS_STATUS_SUCCESS},
        blas_destroy_status{CUBLAS_STATUS_SUCCESS}, blas_stream_status{CUBLAS_STATUS_SUCCESS},
        blas_issue_status{CUBLAS_STATUS_SUCCESS};
    bool returns_partial_blas{}, returns_null_blas{};
    bool returns_default_blas_stream{}, returns_bad_blas_mode{}, returns_bad_blas_math{},
        returns_negative_blas_version{};
    unsigned int blas_creates{}, blas_destroys{}, blas_sets{}, blas_issues{};
    bool is_sgemm{};
    std::array<char, 25> gemm_storage{};
    std::array<unsigned int, 25> gemm_argument_queries{};
    unsigned int bad_split_gemm_abi{};
    std::array<std::byte, 128> gemm_parameters{};
    unsigned int gemm_variant{}, gemm_loads{};
    unsigned int ex_issues{};
    unsigned int batch_issues{};
    unsigned int classic_batch_issues{};
    std::array<std::int64_t, 3> batch_strides{};
    int batches{};
    std::size_t ex_scalar_bytes{4};
    unsigned int global_queries{};
    bool has_null_global{};
    std::array<std::byte, 16> ex_alpha{}, ex_beta{};
    cudaDataType ex_input{CUDA_R_32F}, ex_output{CUDA_R_32F};
    cublasComputeType_t ex_compute{CUBLAS_COMPUTE_32F};
    cublasGemmAlgo_t ex_algorithm{CUBLAS_GEMM_DEFAULT};
    float blas_alpha{}, blas_beta{};
    std::array<CUdeviceptr, 3> blas_addresses{};
    char lt_storage{};
    bool is_lt_live{}, returns_partial_lt{}, returns_null_lt{};
    cublasStatus_t lt_create_status{CUBLAS_STATUS_SUCCESS},
        lt_destroy_status{CUBLAS_STATUS_SUCCESS};
    unsigned int lt_creates{}, lt_destroys{};
};
FakeState state;
struct RandomState {
    std::array<char, 4> handles{};
    std::array<bool, 4> live{};
    unsigned int creates{}, destroys{}, seeds{}, failing_destroy{};
    std::uint32_t type{};
    std::uint64_t seed{};
    curandStatus_t create_status{CURAND_STATUS_SUCCESS}, seed_status{CURAND_STATUS_SUCCESS},
        destroy_status{CURAND_STATUS_SUCCESS};
    bool partial{}, null_success{}, duplicate{};
} random_state;
struct LibraryLifetimeState {
    std::array<char, 3> handles{};
    std::array<bool, 3> live{};
    unsigned int creates{}, destroys{}, failing_destroy{};
    bool returns_duplicate{}, consumes_failed_destroy{}, requires_context{true};
    cublasStatus_t create_status{CUBLAS_STATUS_SUCCESS};
} library_lifetime_state;
CUcontext native_context() {
    return reinterpret_cast<CUcontext>(&state.context_storage);
}
CUresult CUDAAPI initialize(unsigned int) {
    ++state.calls;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_device(CUdevice* device, int ordinal) {
    ++state.calls;
    *device = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* context) {
    ++state.calls;
    *context = state.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_context(CUcontext* context, unsigned int flags, CUdevice) {
    ++state.calls;
    ++state.context_creates;
    state.context_flags = flags;
    if (!state.returns_null_context &&
        (state.create_status == CUDA_SUCCESS || state.returns_partial_context)) {
        *context = native_context();
        state.current = *context;
    }
    return state.create_status;
}
CUresult CUDAAPI set_current(CUcontext context) {
    ++state.calls;
    state.current = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_context(CUcontext context) {
    ++state.calls;
    ++state.context_destroys;
    expect(context == native_context(), "destroy only framework-owned context");
    if (state.destroy_status == CUDA_SUCCESS) {
        state.current = nullptr;
        state.memory.clear();
    }
    return state.destroy_status;
}
CUresult CUDAAPI load_module(CUmodule* module, const void* source) {
    ++state.module_loads;
    ++state.calls;
    if (state.budget_session)
        state.module_load_charges.push_back(state.budget_session->snapshot().device_bytes);
    if (state.is_binary && !state.expected_binary.empty() &&
        std::memcmp(source, state.expected_binary.data(), 4) == 0) {
        ++state.binary_loads;
        expect(std::memcmp(source, state.expected_binary.data(), state.expected_binary.size()) == 0,
               "native binary intake receives the exact owned complete image");
        state.owned_binary = source;
        if (!state.returns_null_module &&
            (state.module_status == CUDA_SUCCESS || state.returns_partial_module))
            *module = reinterpret_cast<CUmodule>(&state.binary_module_storage);
        return state.module_status;
    }
    const std::string_view image(static_cast<const char*>(source));
    if (state.uses_multiple_kinds) state.loaded_ptx = image;
    if (state.is_sgemm && image == cuda_sgemm_image()) {
        ++state.gemm_loads;
        if (!state.returns_null_module &&
            (state.module_status == CUDA_SUCCESS || state.returns_partial_module))
            *module = reinterpret_cast<CUmodule>(&state.module_storage);
        return state.module_status;
    }
    if (state.fails_generated_load && image.find("__glimmer_ptb") != std::string_view::npos) {
        ++state.generated_load_failures;
        return CUDA_ERROR_INVALID_PTX;
    }
    if (state.private_load_failure && image.find("__glimmer_ptb") != std::string_view::npos) {
        if (state.private_load_failure == 1) {
            *module = reinterpret_cast<CUmodule>(&state.module_storage);
            return CUDA_ERROR_INVALID_PTX;
        }
        return CUDA_SUCCESS;  // Deliberately malformed null success.
    }
    expect(state.is_persistent
               ? std::string_view(static_cast<const char*>(source)).find("kernel__glimmer_ptb") !=
                         std::string_view::npos ||
                     (state.fails_generated_load && image == kPersistentPtx) ||
                     image == kUnsupportedPtx || image == state.stateful_source
               : std::string_view(static_cast<const char*>(source)) == kPtx,
           "owned terminated module input");
    if (!state.returns_null_module &&
        (state.module_status == CUDA_SUCCESS || state.returns_partial_module))
        *module = reinterpret_cast<CUmodule>(&state.module_storage);
    return state.module_status;
}
CUresult CUDAAPI unload_module(CUmodule module) {
    ++state.calls;
    ++state.unloads;
    state.unloaded_modules.push_back(module);
    if (state.has_shared_binary &&
        module == reinterpret_cast<CUmodule>(&state.binary_module_storage) &&
        state.unload_status == CUDA_SUCCESS)
        state.validation_module_retired = true;
    return state.unload_status;
}
CUresult CUDAAPI lookup_function(CUfunction* function, CUmodule module, const char* name) {
    ++state.calls;
    if (state.is_sgemm && std::string_view(name).starts_with("glimmer_owned_")) {
        constexpr std::array names{
            "glimmer_owned_sgemm_nn",        "glimmer_owned_sgemm_nt",   "glimmer_owned_sgemm_tn",
            "glimmer_owned_sgemm_tt",        "glimmer_owned_hgemm_nn",   "glimmer_owned_hgemm_nt",
            "glimmer_owned_hgemm_tn",        "glimmer_owned_hgemm_tt",   "glimmer_owned_hsgemm_nn",
            "glimmer_owned_hsgemm_nt",       "glimmer_owned_hsgemm_tn",  "glimmer_owned_hsgemm_tt",
            "glimmer_owned_bsgemm_nn",       "glimmer_owned_bsgemm_nt",  "glimmer_owned_bsgemm_tn",
            "glimmer_owned_bsgemm_tt",       "glimmer_owned_bhgemm_nn",  "glimmer_owned_bhgemm_nt",
            "glimmer_owned_bhgemm_tn",       "glimmer_owned_bhgemm_tt",  "glimmer_owned_bhsgemm_nn",
            "glimmer_owned_bhsgemm_nt",      "glimmer_owned_bhsgemm_tn", "glimmer_owned_bhsgemm_tt",
            "glimmer_owned_sgemm_nn_split_k"};
        const auto found = std::find(names.begin(), names.end(), std::string_view(name));
        expect(found != names.end(), "twenty-four ordinary and one dedicated split-K GEMM entry");
        *function = reinterpret_cast<CUfunction>(&state.gemm_storage[found - names.begin()]);
        return state.lookup_status;
    }
    if (state.is_binary) {
        expect(state.owned_binary && std::memcmp(state.owned_binary, state.expected_binary.data(),
                                                 state.expected_binary.size()) == 0,
               "server retains binary bytes after caller mutation and before every lookup");
        const bool private_name =
            std::string_view(name).find("__glimmer_ptb") != std::string_view::npos;
        const bool candidate =
            state.has_shared_binary && module == reinterpret_cast<CUmodule>(&state.module_storage);
        expect(module == reinterpret_cast<CUmodule>(private_name || candidate ||
                                                            state.validation_module_retired
                                                        ? &state.module_storage
                                                        : &state.binary_module_storage),
               "original and generated functions have distinct owned module identities");
        if (candidate && !private_name) {
            *function = reinterpret_cast<CUfunction>(&state.candidate_function_storage);
            return state.lookup_status;
        }
    }
    if (state.uses_multiple_kinds) {
        constexpr std::array names{"kernel__glimmer_ptb", "kernel__glimmer_ptb_block_static",
                                   "kernel__glimmer_ptb_dynamic", "kernel__glimmer_ptb_preemptive",
                                   "kernel__glimmer_ptb_sliced"};
        const auto found = std::ranges::find(names, std::string_view{name});
        if (found != names.end()) {
            *function = reinterpret_cast<CUfunction>(&state.variant_storage[found - names.begin()]);
            return state.lookup_status;
        }
    }
    *function = reinterpret_cast<CUfunction>(std::string_view(name).find("__glimmer_ptb") !=
                                                     std::string_view::npos
                                                 ? &state.private_function_storage
                                                 : &state.function_storage);
    return state.lookup_status;
}
std::optional<Kind> private_kind(CUfunction function) {
    if (state.uses_multiple_kinds)
        for (std::size_t index = 0; index < state.variant_storage.size(); ++index)
            if (function == reinterpret_cast<CUfunction>(&state.variant_storage[index]))
                return static_cast<Kind>(index);
    if (function == reinterpret_cast<CUfunction>(&state.private_function_storage))
        return state.kind;
    return std::nullopt;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): fixed native CUDA ABI.
CUresult CUDAAPI parameter_info(CUfunction function, std::size_t index, std::size_t* offset,
                                std::size_t* bytes) {
    ++state.calls;
    ++state.parameter_queries;
    if (state.parameter_status != CUDA_SUCCESS) return state.parameter_status;
    if (state.is_sgemm && std::ranges::any_of(state.gemm_storage, [&](const auto& value) {
            return function == reinterpret_cast<CUfunction>(const_cast<char*>(&value));
        })) {
        constexpr std::array<std::size_t, 20> offsets{0,  8,  16, 24, 28, 32, 36, 40,  44,  48,
                                                      52, 56, 64, 72, 80, 88, 96, 104, 112, 120};
        constexpr std::array<std::size_t, 20> sizes{8, 8, 8, 4, 4, 4, 4, 4, 4, 4,
                                                    4, 8, 8, 4, 8, 8, 8, 8, 8, 4};
        const bool batch = std::ranges::any_of(
            std::span{state.gemm_storage}.subspan(12, 12), [&](const auto& value) {
                return function == reinterpret_cast<CUfunction>(const_cast<char*>(&value));
            });
        const bool split = function == reinterpret_cast<CUfunction>(&state.gemm_storage[24]);
        expect(index < (split   ? 17U
                        : batch ? 20U
                                : 16U),
               "independently check unchanged ordinary and dedicated split-K private arity");
        const auto found = std::ranges::find_if(state.gemm_storage, [&](const auto& value) {
            return function == reinterpret_cast<CUfunction>(const_cast<char*>(&value));
        });
        ++state.gemm_argument_queries[static_cast<std::size_t>(found - state.gemm_storage.begin())];
        *offset = offsets[index];
        *bytes = state.is_bad_private_abi ? 1 : split && index == 16 ? 4 : sizes[index];
        if (split && index == 16 && state.bad_split_gemm_abi) {
            if (state.bad_split_gemm_abi == 1)
                *offset = 100;
            else
                *bytes = 8;
        }
        return CUDA_SUCCESS;
    }
    if (const auto kind = private_kind(function)) {
        constexpr std::array<std::size_t, 7> offsets{0, 8, 12, 16, 20, 24, 32};
        constexpr std::array<std::size_t, 7> sizes{8, 4, 4, 4, 4, 8, 8};
        const std::size_t arity = *kind == Kind::kStatic        ? 3
                                  : *kind == Kind::kBlockStatic ? 5
                                  : *kind == Kind::kDynamic     ? 6
                                                                : 7;
        expect(index < arity, "private ABI has exactly the selected generated parameters");
        *offset = *kind == Kind::kSliced && index == 6 ? 28 : offsets[index];
        *bytes = state.is_bad_private_abi || state.bad_abi_kind == kind ? 1
                 : *kind == Kind::kSliced && index >= 5                 ? 4
                                                                        : sizes[index];
        return CUDA_SUCCESS;
    }
    expect(index < 2, "owned PTX arity must prevent native out-of-range probing");
    *offset = index == 0 || state.has_overlapping_abi ? 0 : 8;
    *bytes = index == 0 ? 8 : 4;
    if (state.has_bad_companion_abi &&
        function == reinterpret_cast<CUfunction>(&state.candidate_function_storage))
        *bytes = 1;
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult CUDAAPI allocate(CUdeviceptr* pointer, std::size_t bytes) {
    ++state.calls;
    ++state.allocations;
    if (!state.returns_null_allocation &&
        (state.allocation_status == CUDA_SUCCESS || state.returns_partial_allocation)) {
        *pointer = 0x1000 + static_cast<CUdeviceptr>(state.allocations) * 0x10000;
        state.memory[*pointer].resize(bytes);
    }
    return state.allocation_status;
}
CUresult CUDAAPI release(CUdeviceptr pointer) {
    ++state.calls;
    ++state.frees;
    if (state.free_status == CUDA_SUCCESS || state.erases_on_free_error)
        state.memory.erase(pointer);
    return state.free_status;
}
CUresult CUDAAPI upload(CUdeviceptr destination, const void* source, std::size_t bytes) {
    ++state.calls;
    ++state.copies;
    auto found = state.memory.upper_bound(destination);
    expect(found != state.memory.begin(), "known destination");
    --found;
    std::memcpy(found->second.data() + (destination - found->first), source, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download(void* destination, CUdeviceptr source, std::size_t bytes) {
    ++state.calls;
    ++state.copies;
    auto found = state.memory.upper_bound(source);
    expect(found != state.memory.begin(), "known source");
    --found;
    std::memcpy(destination, found->second.data() + (source - found->first), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_stream(CUstream* stream, unsigned int) {
    ++state.calls;
    if (!state.returns_null_stream &&
        (state.stream_status == CUDA_SUCCESS || state.returns_partial_stream))
        *stream = reinterpret_cast<CUstream>(&state.stream_storage);
    return state.stream_status;
}
CUresult CUDAAPI destroy_stream(CUstream) {
    ++state.calls;
    ++state.stream_destroys;
    return state.stream_destroy_status;
}
CUresult CUDAAPI create_event(CUevent* event, unsigned int) {
    ++state.calls;
    for (auto& entry : state.events)
        if (!entry.is_live) {
            entry.is_live = true;
            *event = reinterpret_cast<CUevent>(&entry);
            return CUDA_SUCCESS;
        }
    return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult CUDAAPI record_event(CUevent, CUstream) {
    ++state.calls;
    ++state.event_records;
    if (state.on_event_record) state.on_event_record();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI query_event(CUevent) {
    ++state.calls;
    return state.query_status;
}
CUresult CUDAAPI destroy_event(CUevent event) {
    ++state.calls;
    reinterpret_cast<Event*>(event)->is_live = false;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI drain(CUstream) {
    ++state.calls;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI launch(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                        unsigned int grid_z, unsigned int, unsigned int, unsigned int, unsigned int,
                        CUstream, void**, void** extra) {
    ++state.calls;
    ++state.launches;
    state.physical_grids.push_back({grid_x, grid_y, grid_z});
    if (state.is_sgemm) {
        for (std::size_t variant = 0; variant < state.gemm_storage.size(); ++variant)
            if (function == reinterpret_cast<CUfunction>(&state.gemm_storage[variant]))
                state.gemm_variant = static_cast<unsigned int>(variant);
        const std::size_t parameter_bytes = state.gemm_variant == 24  ? 104
                                            : state.gemm_variant < 12 ? 96
                                                                      : 128;
        expect(extra && *static_cast<std::size_t*>(extra[3]) == parameter_bytes,
               "replacement uses independently checked flat GEMM or batch ABI");
        std::memcpy(state.gemm_parameters.data(), extra[1], parameter_bytes);
        CUdeviceptr counter{};
        std::memcpy(&counter, state.gemm_parameters.data() + 80, 8);
        auto found = state.memory.upper_bound(counter);
        expect(found != state.memory.begin(), "owned private GEMM counter");
        --found;
        std::uint64_t previous{};
        std::memcpy(&previous, found->second.data() + (counter - found->first), 8);
        state.counter_before_launch.push_back(previous);
        std::memcpy(found->second.data() + (counter - found->first), &state.claimed, 8);
        return CUDA_SUCCESS;
    }
    const auto kind = private_kind(function);
    state.launched_kinds.push_back(kind);
    const bool is_private = kind.has_value();
    const std::size_t private_bytes = kind == Kind::kStatic        ? 16
                                      : kind == Kind::kBlockStatic ? 24
                                      : kind == Kind::kPreemptive  ? 40
                                                                   : 32;
    expect(extra && *static_cast<std::size_t*>(extra[3]) == (is_private ? private_bytes : 12U),
           "use actual packed native ABI");
    const auto* bytes = static_cast<const std::byte*>(extra[1]);
    CUdeviceptr pointer{};
    std::uint32_t value{};
    std::memcpy(&pointer, bytes, 8);
    std::memcpy(&value, bytes + 8, 4);
    std::memcpy(state.memory.at(pointer).data(), &value, 4);
    if (is_private) {
        std::array<unsigned int, 3> logical{1, 1, 1};
        std::memcpy(logical.data(), bytes + 12, kind == Kind::kStatic ? 4 : 12);
        state.logical_grids.push_back(logical);
        if (kind == Kind::kSliced) {
            std::array<std::uint32_t, 2> slice{};
            std::memcpy(slice.data(), bytes + 24, 8);
            state.slices.push_back(slice);
        }
        if (kind != Kind::kDynamic && kind != Kind::kPreemptive) return CUDA_SUCCESS;
        CUdeviceptr counter{};
        std::memcpy(&counter, bytes + 24, 8);
        auto found = state.memory.upper_bound(counter);
        --found;
        std::uint64_t previous{};
        std::memcpy(&previous, found->second.data() + (counter - found->first), 8);
        state.counter_before_launch.push_back(previous);
        std::memcpy(found->second.data() + (counter - found->first), &state.claimed, 8);
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_allocate(void** output, std::size_t bytes, unsigned int) {
    ++state.host_allocations;
    if (!state.returns_null_host &&
        (state.host_status == CUDA_SUCCESS || state.returns_partial_host)) {
        state.host = ::operator new(bytes, std::align_val_t{64});
        *output = state.host;
    }
    return state.host_status;
}
CUresult CUDAAPI host_release(void* pointer) {
    ++state.host_frees;
    if (state.host_free_status == CUDA_SUCCESS) {
        ::operator delete(pointer, std::align_val_t{64});
        state.host = nullptr;
    }
    return state.host_free_status;
}
CUresult CUDAAPI host_map(CUdeviceptr* device, void* pointer, unsigned int) {
    *device = reinterpret_cast<CUdeviceptr>(pointer);
    return state.map_status;
}
CUresult CUDAAPI upload_async(CUdeviceptr pointer, const void* source, std::size_t bytes,
                              CUstream) {
    ++state.prefix_copies;
    expect(upload(pointer, source, bytes) == CUDA_SUCCESS, "fake ordered prefix copies owned zero");
    return state.prefix_status;
}
CUresult CUDAAPI download_async(void* destination, CUdeviceptr source, std::size_t bytes,
                                CUstream) {
    ++state.readbacks;
    return download(destination, source, bytes);
}
CUresult CUDAAPI occupancy(int* active, CUfunction function, int, std::size_t) {
    *active = state.zero_occupancy_kind && private_kind(function) == state.zero_occupancy_kind
                  ? 0
                  : state.occupancy_capacity;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_attribute(int* value, CUdevice_attribute attribute, CUdevice) {
    ++state.calls;
    *value = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 7 : 0;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_get_flags(unsigned int* flags) {
    ++state.flag_queries;
    if (state.flag_status == CUDA_SUCCESS) *flags = state.context_flags;
    return state.flag_status;
}
CUresult CUDAAPI context_set_flags(unsigned int flags) {
    ++state.flag_updates;
    if (state.flag_status == CUDA_SUCCESS) state.context_flags = flags;
    return state.flag_status;
}
CUresult CUDAAPI function_set_attribute(CUfunction function, CUfunction_attribute attribute,
                                        int value) {
    expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
               value == CU_SHAREDMEM_CARVEOUT_MAX_SHARED && function,
           "owned functions use the same checked per-function carveout hint");
    state.configured_functions.push_back(function);
    return state.failed_configuration == state.configured_functions.size()
               ? CUDA_ERROR_INVALID_VALUE
               : CUDA_SUCCESS;
}
CudaSessionDispatch dispatch() {
    return {{get_current, launch, create_event, record_event, query_event, destroy_event, drain},
            initialize,
            get_device,
            create_context,
            set_current,
            destroy_context,
            load_module,
            unload_module,
            lookup_function,
            parameter_info,
            allocate,
            release,
            upload,
            download,
            create_stream,
            destroy_stream,
            host_allocate,
            host_release,
            host_map,
            upload_async,
            download_async,
            occupancy,
            device_attribute,
            context_get_flags,
            context_set_flags,
            function_set_attribute};
}
cublasStatus_t CUBLASWINAPI blas_create(cublasHandle_t* handle) {
    ++state.blas_creates;
    expect(state.current == native_context(), "create cuBLAS on the owned context thread");
    if (!state.returns_null_blas &&
        (state.blas_create_status == CUBLAS_STATUS_SUCCESS || state.returns_partial_blas)) {
        *handle = reinterpret_cast<cublasHandle_t>(&state.blas_storage.at(state.blas_creates - 1));
        state.blas.emplace(*handle, nullptr);
        state.blas_modes.emplace(*handle, CUBLAS_POINTER_MODE_HOST);
        state.blas_math.emplace(*handle, CUBLAS_DEFAULT_MATH);
        state.blas_policy.emplace(*handle, FakeState::BlasPolicy{});
    }
    return state.blas_create_status;
}
cublasStatus_t CUBLASWINAPI blas_destroy(cublasHandle_t handle) {
    ++state.blas_destroys;
    expect(state.current == native_context() && state.blas.contains(handle),
           "destroy the exact owned native cuBLAS handle before context destruction");
    if (state.blas_destroy_status == CUBLAS_STATUS_SUCCESS) {
        state.blas.erase(handle);
        state.blas_modes.erase(handle);
        state.blas_math.erase(handle);
        state.blas_policy.erase(handle);
    }
    return state.blas_destroy_status;
}
cublasStatus_t CUBLASWINAPI blas_set_stream(cublasHandle_t handle, cudaStream_t stream) {
    ++state.blas_sets;
    expect(state.blas.contains(handle), "set stream on an owned native cuBLAS handle");
    if (state.blas_stream_status == CUBLAS_STATUS_SUCCESS) {
        state.blas.at(handle) = stream;
        auto& policy = state.blas_policy.at(handle);
        policy.has_workspace = false;
        policy.workspace = 0;
        policy.workspace_bytes = 0;
    }
    return state.blas_stream_status;
}
cublasStatus_t CUBLASWINAPI blas_get_stream(cublasHandle_t handle, cudaStream_t* output) {
    *output = state.returns_default_blas_stream ? nullptr : state.blas.at(handle);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI blas_set_pointer(cublasHandle_t handle, cublasPointerMode_t mode) {
    state.blas_modes.at(handle) = mode;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI blas_get_pointer(cublasHandle_t handle, cublasPointerMode_t* output) {
    if (state.returns_bad_blas_mode) {
        const std::uint32_t bits{999};
        static_assert(sizeof(*output) == sizeof(bits));
        std::memcpy(output, &bits, sizeof(bits));
        return CUBLAS_STATUS_SUCCESS;
    }
    *output = state.blas_modes.at(handle);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI blas_set_math(cublasHandle_t handle, cublasMath_t mode) {
    if (mode != CUBLAS_DEFAULT_MATH && mode != CUBLAS_PEDANTIC_MATH)
        return CUBLAS_STATUS_INVALID_VALUE;
    state.blas_math.at(handle) = mode;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI blas_get_math(cublasHandle_t handle, cublasMath_t* output) {
    if (state.returns_bad_blas_math) {
        const std::uint32_t bits{999};
        static_assert(sizeof(*output) == sizeof(bits));
        std::memcpy(output, &bits, sizeof(bits));
        return CUBLAS_STATUS_SUCCESS;
    }
    *output = state.blas_math.at(handle);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI blas_get_version(cublasHandle_t handle, int* output) {
    expect(state.blas.contains(handle), "version belongs to a real owned library handle");
    *output = state.returns_negative_blas_version ? -1 : 120605;
    return CUBLAS_STATUS_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuBLAS ABI.
cublasStatus_t CUBLASWINAPI blas_sgemm(cublasHandle_t handle, cublasOperation_t, cublasOperation_t,
                                       int, int, int, const float* alpha, const float* a, int,
                                       const float* b, int, const float* beta, float* c, int) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    ++state.blas_issues;
    expect(state.blas.contains(handle) && state.current == native_context(),
           "SGEMM uses the server-owned cuBLAS handle and context");
    const auto read_scalar = [&](const float* pointer) {
        if (state.blas_modes.at(handle) == CUBLAS_POINTER_MODE_HOST) return *pointer;
        const auto address = reinterpret_cast<CUdeviceptr>(pointer);
        auto found = state.memory.upper_bound(address);
        expect(found != state.memory.begin(), "device scalar is an actual owned allocation");
        --found;
        const auto offset = static_cast<std::size_t>(address - found->first);
        expect(offset <= found->second.size() && sizeof(float) <= found->second.size() - offset,
               "device scalar stays in its exact owned range");
        float value{};
        std::memcpy(&value, found->second.data() + offset, sizeof(value));
        return value;
    };
    state.blas_alpha = read_scalar(alpha);
    state.blas_beta = read_scalar(beta);
    state.blas_addresses = {reinterpret_cast<CUdeviceptr>(a), reinterpret_cast<CUdeviceptr>(b),
                            reinterpret_cast<CUdeviceptr>(c)};
    return state.blas_issue_status;
}
void capture_extended_coefficients(cublasHandle_t handle, const void* alpha, const void* beta) {
    const auto copy = [&](const void* source, auto& destination) {
        if (state.blas_modes.at(handle) == CUBLAS_POINTER_MODE_HOST) {
            expect(reinterpret_cast<std::uintptr_t>(source) % alignof(double) == 0,
                   "owned raw coefficient storage is naturally aligned");
            std::memcpy(destination.data(), source, state.ex_scalar_bytes);
        } else {
            const auto address = reinterpret_cast<CUdeviceptr>(source);
            auto found = state.memory.upper_bound(address);
            expect(found != state.memory.begin(), "owned extended GPU scalar");
            --found;
            const auto offset = static_cast<std::size_t>(address - found->first);
            expect(offset <= found->second.size() &&
                       state.ex_scalar_bytes <= found->second.size() - offset,
                   "native scalar has independently checked exact extent");
            std::memcpy(destination.data(), found->second.data() + offset, state.ex_scalar_bytes);
        }
    };
    copy(alpha, state.ex_alpha);
    copy(beta, state.ex_beta);
}
CudaSessionDispatch blas_dispatch() {
    auto functions = dispatch();
    functions.blas = {blas_create,     blas_destroy,     blas_set_stream,  blas_sgemm,
                      blas_get_stream, blas_set_pointer, blas_get_pointer, blas_set_math,
                      blas_get_math,   blas_get_version};
    functions.blas.set_workspace = [](cublasHandle_t handle, void* pointer, std::size_t bytes) {
        ++state.workspace_sets;
        expect(state.current == native_context() && state.blas.contains(handle),
               "configure the actual context-owned original workspace");
        const auto address = reinterpret_cast<CUdeviceptr>(pointer);
        if (address % 256) return CUBLAS_STATUS_INVALID_VALUE;
        auto& policy = state.blas_policy.at(handle);
        policy.workspace = address;
        policy.workspace_bytes = bytes;
        policy.has_workspace = true;
        return state.workspace_status;
    };
    functions.blas.set_atomics_mode = [](cublasHandle_t handle, cublasAtomicsMode_t mode) {
        ++state.atomics_sets;
        state.blas_policy.at(handle).atomics = static_cast<unsigned int>(mode);
        return CUBLAS_STATUS_SUCCESS;
    };
    functions.blas.get_atomics_mode = [](cublasHandle_t handle, cublasAtomicsMode_t* output) {
        const unsigned int bits =
            state.returns_bad_atomics ? 999U : state.blas_policy.at(handle).atomics;
        static_assert(sizeof(bits) == sizeof(*output));
        std::memcpy(output, &bits, sizeof(bits));
        return CUBLAS_STATUS_SUCCESS;
    };
    functions.blas.set_sm_count_target = [](cublasHandle_t handle, int count) {
        ++state.sm_count_sets;
        if (count < 0 || count > 80) return CUBLAS_STATUS_INVALID_VALUE;
        state.blas_policy.at(handle).sm_count = count;
        return CUBLAS_STATUS_SUCCESS;
    };
    functions.blas.get_sm_count_target = [](cublasHandle_t handle, int* output) {
        *output = state.returns_bad_sm_count ? -1 : state.blas_policy.at(handle).sm_count;
        return CUBLAS_STATUS_SUCCESS;
    };
    functions.blas.gemm_ex = [](cublasHandle_t handle, cublasOperation_t, cublasOperation_t, int,
                                int, int, const void* alpha, const void*, cudaDataType input, int,
                                const void*, cudaDataType, int, const void* beta, void*,
                                cudaDataType output, int, cublasComputeType_t compute,
                                cublasGemmAlgo_t algorithm) {
        ++state.ex_issues;
        expect(state.blas.contains(handle) && state.current == native_context(),
               "GemmEx runs on its actual owned native context");
        state.ex_input = input;
        state.ex_output = output;
        state.ex_compute = compute;
        state.ex_algorithm = algorithm;
        capture_extended_coefficients(handle, alpha, beta);
        return state.blas_issue_status;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original SDK callback ABI.
    functions.blas.strided_gemm_ex = [](cublasHandle_t handle, cublasOperation_t, cublasOperation_t,
                                        int, int, int, const void* alpha, const void* a,
                                        cudaDataType input, int, long long stride_a, const void* b,
                                        cudaDataType, int, long long stride_b, const void* beta,
                                        void* c, cudaDataType output, int, long long stride_c,
                                        int batches, cublasComputeType_t compute,
                                        cublasGemmAlgo_t algorithm) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        ++state.batch_issues;
        expect(state.blas.contains(handle) && state.current == native_context(),
               "strided GemmEx uses the actual context-owned original callback");
        state.batch_strides = {stride_a, stride_b, stride_c};
        state.batches = batches;
        state.blas_addresses = {reinterpret_cast<CUdeviceptr>(a), reinterpret_cast<CUdeviceptr>(b),
                                reinterpret_cast<CUdeviceptr>(c)};
        state.ex_input = input;
        state.ex_output = output;
        state.ex_compute = compute;
        state.ex_algorithm = algorithm;
        capture_extended_coefficients(handle, alpha, beta);
        return state.blas_issue_status;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original FP32 SDK ABI.
    functions.blas.strided_sgemm = [](cublasHandle_t handle, cublasOperation_t, cublasOperation_t,
                                      int, int, int, const float* alpha, const float* a, int,
                                      long long stride_a, const float* b, int, long long stride_b,
                                      const float* beta, float* c, int, long long stride_c,
                                      int batches) {
        ++state.batch_issues;
        ++state.classic_batch_issues;
        expect(state.blas.contains(handle) && state.current == native_context(),
               "classic strided API reaches its own original callback, not GemmEx");
        state.batch_strides = {stride_a, stride_b, stride_c};
        state.batches = batches;
        state.blas_addresses = {reinterpret_cast<CUdeviceptr>(a), reinterpret_cast<CUdeviceptr>(b),
                                reinterpret_cast<CUdeviceptr>(c)};
        state.ex_input = state.ex_output = CUDA_R_32F;
        state.ex_compute = CUBLAS_COMPUTE_32F;
        state.ex_algorithm = CUBLAS_GEMM_DEFAULT;
        capture_extended_coefficients(handle, alpha, beta);
        return state.blas_issue_status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return functions;
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "resource/session operation must complete successfully");
}

struct PreparationTrace {
    struct Event {
        CudaPreparationPhase phase{};
        bool is_begin{};
        bool operator==(const Event&) const = default;
    };
    CudaExecutionSession* session{};
    std::array<Event, 64> events{};
    std::size_t count{};
    bool has_overflow{}, has_reconfiguration{}, has_sdk_reentry{};
    static void observe(void* opaque, CudaPreparationPhase phase, bool is_begin) noexcept {
        auto& trace = *static_cast<PreparationTrace*>(opaque);
        const auto calls = state.calls;
        if (trace.count == trace.events.size())
            trace.has_overflow = true;
        else
            trace.events[trace.count++] = {phase, is_begin};
        // The one explicitly tested configuration operation must reject before
        // invoking any SDK or replacing this non-owning callback, on both edges.
        trace.has_reconfiguration |= trace.session->set_preparation_observer({});
        trace.has_sdk_reentry |= state.calls != calls;
    }
    void verify_balanced() const {
        expect(!has_overflow && !has_reconfiguration && !has_sdk_reentry,
               "observer is bounded, cannot change itself mid-phase and issues no SDK calls");
        std::array<CudaPreparationPhase, 8> stack{};
        std::size_t depth{};
        for (std::size_t index = 0; index < count; ++index) {
            const auto event = events[index];
            expect(static_cast<unsigned int>(event.phase) < 6,
                   "observer has only six bounded phases");
            if (event.is_begin) {
                expect(depth < stack.size(), "bounded preparation nesting");
                stack[depth++] = event.phase;
            } else {
                expect(depth && stack[depth - 1] == event.phase,
                       "every phase end matches the most recent begun scope");
                --depth;
            }
        }
        expect(!depth, "all preparation scopes terminate even on rejected or exceptional paths");
    }
};

struct PreparationOutcome {
    CUresult status{}, original{};
    bool has_original{}, is_quarantined{};
    std::size_t modules{}, functions{}, persistent{}, image_bytes{}, device_bytes{},
        private_host_bytes{}, persistent_slots{};
    unsigned int calls{}, loads{}, parameters{}, unloads{}, context_destroys{};
    bool operator==(const PreparationOutcome&) const = default;
};

PreparationOutcome preparation_observer_case(unsigned int mode, bool observed) {
    state = {};
    PreparationTrace trace;
    state.is_persistent = mode == 1 || mode == 6 || mode == 7;
    state.fails_generated_load = mode == 7;
    state.module_status = mode == 2 || mode == 3 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS;
    state.returns_partial_module = mode == 3;
    auto sdk = dispatch();
    if (mode == 10)
        sdk.module_load = [](CUmodule*, const void*) -> CUresult {
            ++state.calls;
            ++state.module_loads;
            throw std::bad_alloc{};
        };
    CudaSessionLimits limits;
    if (mode == 11) limits.max_image_bytes = kPtx.size();
    CudaExecutionSession session({1, 2}, 7, sdk, limits, {state.is_persistent ? 320U : 0U});
    trace.session = &session;
    success(session.initialize(0));
    if (observed)
        expect(session.set_preparation_observer({&trace, PreparationTrace::observe}),
               "owner installs non-owning observer before preparation");
    CudaSessionResult result;
    if (mode == 1 || mode == 4) {
        const std::array<glimmer::transform::CudaImageArgument, 2> layout{{{0, 8}, {8, 4}}};
        state.is_binary = true;
        state.expected_binary =
            glimmer::test::cuda_test_fatbin(glimmer::test::cuda_test_cubin(layout), kPersistentPtx);
        result = mode == 4 ? session.load_binary(std::span(state.expected_binary).first(4))
                           : session.load_binary(state.expected_binary);
    } else if (mode == 5)
        result =
            session.load_ptx(".version 7.0\n.target sm_70\n.address_size 64\n.global .u32 bad[0];");
    else
        result = session.load_ptx(mode == 6 ? kUnsupportedPtx : mode == 7 ? kPersistentPtx : kPtx);
    if (result.id) {
        if (mode == 8) state.lookup_status = CUDA_ERROR_NOT_FOUND;
        if (mode == 9) state.parameter_status = CUDA_ERROR_INVALID_VALUE;
        result = session.get_function(result.id, "kernel");
    }
    trace.verify_balanced();
    if (!observed) expect(!trace.count, "disabled observer emits no records");
    if (observed) {
        using Phase = CudaPreparationPhase;
        std::vector<PreparationTrace::Event> expected;
        const auto pair = [&](Phase phase) {
            expected.push_back({phase, true});
            expected.push_back({phase, false});
        };
        if (mode == 1 || mode == 4) pair(Phase::kInspectImage);
        if (mode != 4 && mode != 11) pair(Phase::kStaticStorage);
        if (state.is_persistent) pair(Phase::kTransformPtx);
        if (mode == 1 || mode == 7) pair(Phase::kStaticStorage);
        if (mode == 7) pair(Phase::kLoadPrepared);
        if (mode != 4 && mode != 5 && mode != 11) pair(Phase::kLoadOriginal);
        if (mode == 0 || mode == 1 || mode == 6 || mode == 7 || mode == 8 || mode == 9) {
            expected.push_back({Phase::kValidateFunction, true});
            if (mode == 1) pair(Phase::kLoadPrepared);
            expected.push_back({Phase::kValidateFunction, false});
        }
        expect(std::span(trace.events).first(trace.count).size() == expected.size() &&
                   std::equal(expected.begin(), expected.end(), trace.events.begin()),
               "independent phase order matches actual work, not a fabricated success trace");
    }
    if (mode == 2 || mode == 3)
        expect(result.original_status == CUDA_ERROR_INVALID_PTX && !result.id &&
                   state.module_loads == 1,
               "known and partial native failures retain exact original provenance once");
    if (mode == 4 || mode == 5 || mode == 11)
        expect(!result.original_status && state.module_loads == 0,
               "preparation policy errors do not invent original module provenance");
    if (mode == 8)
        expect(result.original_status == CUDA_ERROR_NOT_FOUND && !result.id,
               "function lookup error remains the issued SDK result");
    if (mode == 9)
        expect(result.original_status == CUDA_SUCCESS &&
                   result.status == CUDA_ERROR_INVALID_VALUE && !result.id,
               "failed private ABI validation retains original lookup provenance");
    if (mode == 10)
        expect(result.status == CUDA_ERROR_OUT_OF_MEMORY && !result.original_status &&
                   state.module_loads == 1,
               "exception unwinding closes phase without inventing a returned native status");
    const auto snapshot = session.snapshot();
    if (mode == 0 || mode == 1 || mode == 6 || mode == 7) {
        success(result);
        expect(result.original_status == CUDA_SUCCESS && result.id &&
                   result.parameter_bytes == 12 && result.arguments.size() == 2 &&
                   result.arguments[0].offset == 0 && result.arguments[0].bytes == 8 &&
                   result.arguments[1].offset == 8 && result.arguments[1].bytes == 4 &&
                   snapshot.modules == 1 && snapshot.functions == 1 &&
                   snapshot.persistent_functions == (mode == 1 ? 1U : 0U) &&
                   state.module_loads == (mode == 1 || mode == 7 ? 2U : 1U),
               "successful and original-only fallback paths retain exact ABI and native counts");
    }
    PreparationOutcome outcome{result.status,
                               result.original_status.value_or(CUDA_SUCCESS),
                               result.original_status.has_value(),
                               snapshot.is_quarantined,
                               snapshot.modules,
                               snapshot.functions,
                               snapshot.persistent_functions,
                               snapshot.image_bytes,
                               snapshot.device_bytes,
                               snapshot.private_host_bytes,
                               snapshot.persistent_slots};
    expect(session.set_preparation_observer({}),
           "owner removes observer after final balanced phase");
    const auto count = trace.count;
    success(session.shutdown());
    expect(trace.count == count && state.context_destroys == 1,
           "non-preparation teardown never emits a misleading phase");
    outcome.calls = state.calls;
    outcome.loads = state.module_loads;
    outcome.parameters = state.parameter_queries;
    outcome.unloads = state.unloads;
    outcome.context_destroys = state.context_destroys;
    return outcome;
}

CudaExecutionSession* preparation_configuration_owner{};
PreparationTrace* preparation_configuration_trace{};
bool preparation_configuration_rejected{};
CUresult CUDAAPI preparation_configuration_load(CUmodule* module, const void* source) {
    const auto calls = state.calls;
    preparation_configuration_rejected =
        !preparation_configuration_owner->set_preparation_observer(
            {preparation_configuration_trace, PreparationTrace::observe}) &&
        state.calls == calls;
    return load_module(module, source);
}

void preparation_observer_configuration() {
    state = {};
    PreparationTrace trace;
    auto sdk = dispatch();
    sdk.module_load = preparation_configuration_load;
    CudaExecutionSession session({1, 2}, 7, sdk);
    trace.session = &session;
    preparation_configuration_owner = &session;
    preparation_configuration_trace = &trace;
    preparation_configuration_rejected = false;
    success(session.initialize(0));
    const auto module = session.load_ptx(kPtx);
    success(module);
    expect(preparation_configuration_rejected && trace.count == 0,
           "even disabled observation cannot be installed from an active SDK phase");
    const auto calls = state.calls;
    auto foreign = std::async(std::launch::async, [&] {
        return session.set_preparation_observer({&trace, PreparationTrace::observe});
    });
    expect(!foreign.get() && state.calls == calls,
           "foreign thread cannot install or invoke observer");
    expect(session.set_preparation_observer({&trace, PreparationTrace::observe}),
           "install owner observer");
    const auto child = ::fork();
    expect(child >= 0, "fork observer owner control");
    if (child == 0)
        ::_exit(!session.set_preparation_observer({}) && !session.load_ptx(kPtx).original_status &&
                        state.calls == calls && trace.count == 0
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child rejects before observer callback and SDK");
    success(session.get_function(module.id, "kernel"));
    trace.verify_balanced();
    expect(trace.count == 2 && session.set_preparation_observer({}),
           "owner can detach after complete phase");
    success(session.get_function(module.id, "kernel"));
    expect(trace.count == 2, "detached callback is never reused by later lookup");
    success(session.shutdown());
    preparation_configuration_owner = nullptr;
    preparation_configuration_trace = nullptr;
}

void preparation_observer() {
    for (unsigned int mode = 0; mode < 12; ++mode) {
        const auto baseline = preparation_observer_case(mode, false);
        const auto observed = preparation_observer_case(mode, true);
        expect(baseline == observed,
               "observer does not alter native calls, statuses, resource ownership or quotas");
    }
    preparation_observer_configuration();
}
CudaSessionDispatch random_dispatch() {
    auto functions = dispatch();
    functions.random.create = [](curandGenerator_t* output, curandRngType_t type) {
        ++random_state.creates;
        std::memcpy(&random_state.type, &type, sizeof(type));
        expect(state.current == native_context(), "random factory runs on the actual owner");
        if ((random_state.create_status == CURAND_STATUS_SUCCESS && !random_state.null_success) ||
            random_state.partial) {
            std::size_t index{};
            while (!random_state.duplicate && index < random_state.live.size() &&
                   random_state.live[index])
                ++index;
            expect(index < random_state.live.size(), "bounded original random slots");
            random_state.live[index] = true;
            *output = reinterpret_cast<curandGenerator_t>(&random_state.handles[index]);
        }
        return random_state.create_status;
    };
    functions.random.set_seed = [](curandGenerator_t handle, unsigned long long seed) {
        expect(state.current == native_context() &&
                   handle == reinterpret_cast<curandGenerator_t>(&random_state.handles[0]),
               "seed sees the server-owned native generator");
        ++random_state.seeds;
        random_state.seed = seed;
        return random_state.seed_status;
    };
    functions.random.destroy = [](curandGenerator_t handle) {
        expect(state.current == native_context() && !state.context_destroys,
               "generator cleanup precedes context teardown");
        const auto found = std::find_if(
            random_state.handles.begin(), random_state.handles.end(), [&](const auto& value) {
                return reinterpret_cast<curandGenerator_t>(const_cast<char*>(&value)) == handle;
            });
        expect(found != random_state.handles.end(), "only original random handle is destroyed");
        const auto index = static_cast<std::size_t>(found - random_state.handles.begin());
        expect(random_state.live[index], "native destruction must never be repeated");
        random_state.live[index] = false;  // An error may already have released native state.
        ++random_state.destroys;
        if (random_state.destroys == random_state.failing_destroy)
            return CURAND_STATUS_INTERNAL_ERROR;
        return random_state.destroy_status;
    };
    return functions;
}
void owned_random_lifecycle() {
    state = {};
    random_state = {};
    CudaSessionLimits limits;
    limits.max_objects = 1;
    CudaExecutionSession session({41, 42}, 43, random_dispatch(), limits);
    expect(!session.create_random(161).original_status && random_state.creates == 0,
           "random SDK cannot run before owned context initialization");
    success(session.initialize(0));
    expect(session.create_random(UINT32_MAX).status == CUDA_ERROR_INVALID_VALUE &&
               random_state.creates == 0,
           "out-of-range wire bits cannot become an invalid native enum");
    random_state.create_status = CURAND_STATUS_ALLOCATION_FAILED;
    const auto failed = session.create_random(161);
    expect(failed.original_status == CURAND_STATUS_ALLOCATION_FAILED && !failed.id &&
               failed.status == CUDA_SUCCESS && session.snapshot().random_handles == 0,
           "known-empty random error retains original status and no resource");
    random_state.create_status = CURAND_STATUS_SUCCESS;
    const auto handle = session.create_random(161);
    expect(handle.id && handle.original_status == CURAND_STATUS_SUCCESS &&
               random_state.type == 161 && session.snapshot().random_handles == 1 &&
               session.create_random(161).status == CUDA_ERROR_OUT_OF_MEMORY &&
               random_state.creates == 2,
           "random generator identity and pre-SDK metadata bound");
    const auto seed = UINT64_MAX - 17;
    const auto foreign = std::async(std::launch::async, [&] {
                             return session.set_random_seed(handle.id, seed);
                         }).get();
    expect(foreign.status == CUDA_ERROR_INVALID_CONTEXT && random_state.seeds == 0,
           "non-owner thread cannot mutate a native generator");
    random_state.seed_status = CURAND_STATUS_TYPE_ERROR;
    expect(session.set_random_seed(handle.id, seed).original_status == CURAND_STATUS_TYPE_ERROR &&
               random_state.seed == seed && random_state.seeds == 1 &&
               !session.set_random_seed(handle.id + 1, seed).original_status,
           "full-width seed and original generator-type error survive without replay");
    const auto module = session.load_ptx(kPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto memory = session.allocate(8);
    success(memory);
    std::array<std::byte, 12> parameters{};
    std::memcpy(parameters.data(), &memory.address, 8);
    const std::uint32_t value{19};
    std::memcpy(parameters.data() + 8, &value, 4);
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .grid = {80, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    expect(accepted.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt) &&
               session.set_random_seed(handle.id, 0).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.destroy_random(handle.id).disposition == CudaSessionDisposition::kDeferred &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               random_state.seeds == 1 && random_state.destroys == 0,
           "pending physical work defers generator mutation and cleanup");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "actual stop releases random resource gate");
    success(session.shutdown());
    expect(random_state.destroys == 1 && state.context_destroys == 1 &&
               session.snapshot().random_handles == 0,
           "shutdown destroys remaining native generator once before context retirement");

    for (unsigned int mode = 0; mode < 6; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "isolated random ownership control");
        if (!child) {
            try {
                state = {};
                random_state = {};
                CudaExecutionSession isolated({1, 2}, 3, random_dispatch());
                success(isolated.initialize(0));
                if (mode == 0) {
                    random_state.partial = true;
                    random_state.create_status = CURAND_STATUS_ALLOCATION_FAILED;
                } else if (mode == 1)
                    random_state.null_success = true;
                const auto first = isolated.create_random(101);
                if (mode == 5) {
                    expect(isolated.create_random(101).id != first.id,
                           "independent generator identity");
                    random_state.failing_destroy = 2;
                    expect(isolated.shutdown().status == CUDA_ERROR_UNKNOWN &&
                               isolated.snapshot().random_handles == 1 &&
                               isolated.shutdown().status == CUDA_ERROR_UNKNOWN &&
                               random_state.destroys == 2 && !state.context_destroys,
                           "partial shutdown removes proven cleanup and never repeats it");
                    ::_exit(0);
                }
                if (mode == 2) {
                    random_state.duplicate = true;
                    expect(!isolated.create_random(101).id, "duplicate native generator rejects");
                } else if (mode == 3) {
                    random_state.destroy_status = CURAND_STATUS_INTERNAL_ERROR;
                    expect(isolated.destroy_random(first.id).original_status ==
                               CURAND_STATUS_INTERNAL_ERROR,
                           "ambiguous destruction preserves original status");
                } else if (mode == 4) {
                    expect(isolated.destroy_random(first.id).original_status ==
                                   CURAND_STATUS_SUCCESS &&
                               !isolated.destroy_random(first.id).original_status &&
                               !isolated.set_random_seed(first.id, 0).original_status,
                           "stale generator cannot mutate or repeat cleanup");
                    success(isolated.shutdown());
                    ::_exit(random_state.destroys == 1 ? 0 : 1);
                }
                expect(isolated.snapshot().is_quarantined &&
                           isolated.shutdown().status == CUDA_ERROR_UNKNOWN &&
                           isolated.shutdown().status == CUDA_ERROR_UNKNOWN &&
                           random_state.destroys == (mode == 3 ? 1U : 0U) &&
                           !state.context_destroys,
                       "unknown random ownership never retries cleanup or discards the context");
                ::_exit(0);
            } catch (...) {
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "random isolated control passes");
    }
}
void blas_success(const CudaSessionBlasResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete &&
               result.status == CUDA_SUCCESS && result.original_status == CUBLAS_STATUS_SUCCESS,
           "actual library operation returns its original successful SDK status");
}
void owned_blas_issue_ranges_and_retirement() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, blas_dispatch());
    success(session.initialize(0));
    const auto handle = session.create_blas();
    blas_success(handle);
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    const auto a = session.allocate(128), b = session.allocate(128), c = session.allocate(128);
    success(stream);
    success(a);
    success(b);
    success(c);
    blas_success(session.set_blas_stream(handle.id, stream.id));
    expect(session.destroy_stream(stream.id).status == CUDA_ERROR_NOT_READY &&
               state.stream_destroys == 0,
           "an owned cuBLAS binding keeps its stream until explicit rebinding or handle release");
    CudaSessionSgemm problem{.handle = handle.id,
                             .rows = 2,
                             .columns = 3,
                             .depth = 4,
                             .leading_a = 4,
                             .leading_b = 4,
                             .leading_c = 4,
                             .a = {a.id, 4},
                             .b = {b.id, 8},
                             .c = {c.id, 12},
                             .alpha = 2.0F,
                             .beta = 3.0F};
    auto bad = problem;
    bad.a.offset = 127;
    expect(!session.accept_sgemm(1, bad).accepted(), "reject unaligned or out-of-range A");
    bad = problem;
    bad.b.allocation = c.id + 100;
    expect(!session.accept_sgemm(1, bad).accepted(), "reject unowned matrix identity");
    bad = problem;
    bad.leading_b = 3;
    expect(!session.accept_sgemm(1, bad).accepted(), "reject invalid leading dimension");
    bad = problem;
    bad.leading_c = std::numeric_limits<int>::max();
    expect(!session.accept_sgemm(1, bad).accepted(), "reject excessive matrix extent");
    const auto accepted = session.accept_sgemm(1, problem);
    const auto frontier = session.capture_command_frontier(1);
    expect(accepted.accepted() && state.blas_issues == 0 && frontier.status == CUDA_SUCCESS,
           "library admission owns scalar metadata but has no native result yet");
    problem.alpha = 99.0F;
    problem.beta = 99.0F;
    state.blas_issue_status = CUBLAS_STATUS_EXECUTION_FAILED;
    const auto issued = session.issue_next();
    expect(
        issued.state == CudaExecutionIssueState::kIssued && issued.native_status == CUDA_SUCCESS &&
            issued.library_status == CUBLAS_STATUS_EXECUTION_FAILED && state.blas_alpha == 2.0F &&
            state.blas_beta == 3.0F &&
            state.blas_addresses ==
                std::array<CUdeviceptr, 3>{a.address + 4, b.address + 8, c.address + 12} &&
            session.host_result(accepted.receipt)->status == CUBLAS_STATUS_EXECUTION_FAILED,
        "SGEMM sees immutable host scalars, exact owned pointers and its original library error");
    expect(session.acknowledge_result(accepted.receipt) && session.poll().retained == 1 &&
               session.destroy_blas(handle.id).disposition == CudaSessionDisposition::kDeferred &&
               session.set_blas_stream(handle.id, 0).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.release(a.id).disposition == CudaSessionDisposition::kDeferred &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               state.blas_destroys == 0 && state.frees == 0 && state.context_destroys == 0,
           "library error/ACK cannot destroy handles, mutate bindings or release live GPU inputs");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1 && session.query_frontier(frontier.frontier) == CUDA_SUCCESS,
           "true stop settles the library frontier separately from its host error");
    blas_success(session.destroy_blas(handle.id));
    success(session.destroy_stream(stream.id));
    success(session.shutdown());
    expect(state.blas_destroys == 1 && state.context_destroys == 1,
           "settled library handle destroys once before its owned context");
}
CudaSessionDispatch lt_dispatch() {
    auto functions = dispatch();
    functions.lt.create = [](cublasLtHandle_t* output) {
        ++state.lt_creates;
        expect(state.current == native_context(),
               "native Lt creation belongs to its exact GPU context");
        if (state.lt_create_status != CUBLAS_STATUS_SUCCESS && !state.returns_partial_lt)
            return state.lt_create_status;
        expect(!state.is_lt_live, "native Lt object cannot alias a live owner");
        if (!state.returns_null_lt) {
            *output = reinterpret_cast<cublasLtHandle_t>(&state.lt_storage);
            state.is_lt_live = true;
        }
        return state.lt_create_status;
    };
    functions.lt.destroy = [](cublasLtHandle_t handle) {
        ++state.lt_destroys;
        expect(state.current == native_context() && state.context_destroys == 0 &&
                   state.is_lt_live &&
                   handle == reinterpret_cast<cublasLtHandle_t>(&state.lt_storage),
               "only the actual Lt owner destroys its original object before the GPU context");
        if (state.lt_destroy_status == CUBLAS_STATUS_SUCCESS) state.is_lt_live = false;
        return state.lt_destroy_status;
    };
    return functions;
}

// This CPU fixture models descriptor observations and the private launch ABI,
// not vendor arithmetic or the CUDA provider's numerical implementation.
struct LtSplitLayout {
    cudaDataType type{CUDA_R_32F};
    cublasLtOrder_t order{CUBLASLT_ORDER_COL};
    std::uint64_t rows{768}, columns{768};
    std::int64_t leading{768}, stride{}, plane{};
    std::int32_t batches{1};
};
struct LtSplitOperation {
    cudaDataType scale{CUDA_R_32F};
    cublasComputeType_t compute{CUBLAS_COMPUTE_32F};
    cublasLtPointerMode_t pointer_mode{CUBLASLT_POINTER_MODE_HOST};
    cublasLtEpilogue_t epilogue{CUBLASLT_EPILOGUE_BIAS};
    std::array<cublasOperation_t, 3> transposes{CUBLAS_OP_T, CUBLAS_OP_N, CUBLAS_OP_N};
    std::int32_t bias_type{CUDA_R_32F};
    std::int64_t bias_stride{};
    CUdeviceptr bias{};
};
struct LtSplitDescriptors {
    LtSplitOperation operation;
    std::array<LtSplitLayout, 4> layouts;
    char preference{};
};
struct LtSplitState {
    std::size_t version{120604}, required_workspace{4096};
    std::uint32_t split{6}, reduction{CUBLASLT_REDUCTION_SCHEME_INPLACE};
    cublasStatus_t check_status{CUBLAS_STATUS_SUCCESS};
    cublasStatus_t algorithm_status{CUBLAS_STATUS_SUCCESS};
    cublasStatus_t original_status{CUBLAS_STATUS_SUCCESS};
    std::array<char, 3> functions{};
    std::array<std::size_t, 3> argument_queries{};
    std::array<std::byte, 168> parameters{};
    std::size_t native_calls{}, private_launches{}, heuristics{}, checks{};
    bool has_bad_depth_abi{};
} lt_split;

std::optional<std::size_t> lt_split_function(CUfunction function) {
    for (std::size_t index = 0; index < lt_split.functions.size(); ++index)
        if (function == reinterpret_cast<CUfunction>(&lt_split.functions[index])) return index;
    return {};
}
template <class Value>
cublasStatus_t lt_split_output(void* output, std::size_t capacity, std::size_t* written,
                               const Value& value) {
    expect(output && written && capacity == sizeof(Value),
           "Lt native metadata query uses its exact SDK field width");
    std::memcpy(output, &value, sizeof(value));
    *written = sizeof(value);
    return CUBLAS_STATUS_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): fixed CUDA/cuBLASLt callback ABIs.
cublasStatus_t CUBLASWINAPI lt_split_layout(cublasLtMatrixLayout_t native,
                                            cublasLtMatrixLayoutAttribute_t attribute, void* output,
                                            std::size_t capacity, std::size_t* written) {
    const auto& layout = *reinterpret_cast<const LtSplitLayout*>(native);
    switch (attribute) {
        case CUBLASLT_MATRIX_LAYOUT_TYPE:
            return lt_split_output(output, capacity, written, layout.type);
        case CUBLASLT_MATRIX_LAYOUT_ORDER:
            return lt_split_output(output, capacity, written, layout.order);
        case CUBLASLT_MATRIX_LAYOUT_ROWS:
            return lt_split_output(output, capacity, written, layout.rows);
        case CUBLASLT_MATRIX_LAYOUT_COLS:
            return lt_split_output(output, capacity, written, layout.columns);
        case CUBLASLT_MATRIX_LAYOUT_LD:
            return lt_split_output(output, capacity, written, layout.leading);
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            return lt_split_output(output, capacity, written, layout.batches);
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            return lt_split_output(output, capacity, written, layout.stride);
        case CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET:
            return lt_split_output(output, capacity, written, layout.plane);
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}
cublasStatus_t CUBLASWINAPI lt_split_operation(cublasLtMatmulDesc_t native,
                                               cublasLtMatmulDescAttributes_t attribute,
                                               void* output, std::size_t capacity,
                                               std::size_t* written) {
    const auto& operation = *reinterpret_cast<const LtSplitOperation*>(native);
    switch (attribute) {
        case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
            return lt_split_output(output, capacity, written, operation.scale);
        case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
            return lt_split_output(output, capacity, written, operation.compute);
        case CUBLASLT_MATMUL_DESC_POINTER_MODE:
            return lt_split_output(output, capacity, written, operation.pointer_mode);
        case CUBLASLT_MATMUL_DESC_EPILOGUE:
            return lt_split_output(output, capacity, written, operation.epilogue);
        case CUBLASLT_MATMUL_DESC_TRANSA:
            return lt_split_output(output, capacity, written, operation.transposes[0]);
        case CUBLASLT_MATMUL_DESC_TRANSB:
            return lt_split_output(output, capacity, written, operation.transposes[1]);
        case CUBLASLT_MATMUL_DESC_TRANSC:
            return lt_split_output(output, capacity, written, operation.transposes[2]);
        case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            return lt_split_output(output, capacity, written, operation.bias_type);
        case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
            return lt_split_output(output, capacity, written, operation.bias_stride);
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            return lt_split_output(output, capacity, written, operation.bias);
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}
CUresult CUDAAPI lt_split_parameter(CUfunction function, std::size_t index, std::size_t* offset,
                                    std::size_t* bytes) {
    const auto variant = lt_split_function(function);
    if (!variant) return parameter_info(function, index, offset, bytes);
    constexpr std::array<std::size_t, 28> offsets{0,   8,   16,  24,  32,  40,  44,  48, 52,  56,
                                                  60,  64,  68,  72,  76,  80,  88,  96, 100, 104,
                                                  112, 120, 128, 136, 144, 152, 160, 164};
    constexpr std::array<std::size_t, 28> sizes{8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                4, 8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 4, 4};
    expect(index < offsets.size(), "owned Lt has exactly 28 independently described parameters");
    ++lt_split.argument_queries[*variant];
    *offset = offsets[index];
    *bytes = index == 27 && lt_split.has_bad_depth_abi ? 8 : sizes[index];
    return CUDA_SUCCESS;
}
CUresult CUDAAPI lt_split_launch(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                                 unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                                 unsigned int block_z, unsigned int shared, CUstream stream,
                                 void** parameters, void** extra) {
    const auto variant = lt_split_function(function);
    if (!variant)
        return launch(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared, stream,
                      parameters, extra);
    expect(*variant == 0 && grid_x == 320 && grid_y == 1 && grid_z == 1 && block_x == 128 &&
               block_y == 1 && block_z == 1 && extra &&
               *static_cast<std::size_t*>(extra[3]) == lt_split.parameters.size(),
           "qualified Lt uses explicit workers and the complete 168-byte FP32 ABI");
    ++lt_split.private_launches;
    std::memcpy(lt_split.parameters.data(), extra[1], lt_split.parameters.size());
    CUdeviceptr counter{};
    std::memcpy(&counter, lt_split.parameters.data() + 104, sizeof(counter));
    auto found = state.memory.upper_bound(counter);
    expect(found != state.memory.begin(), "owned Lt counter belongs to private storage");
    --found;
    const auto offset = static_cast<std::size_t>(counter - found->first);
    expect(offset + sizeof(std::uint64_t) <= found->second.size(), "complete private counter");
    std::uint64_t previous{};
    std::memcpy(&previous, found->second.data() + offset, sizeof(previous));
    state.counter_before_launch.push_back(previous);
    std::memcpy(found->second.data() + offset, &state.claimed, sizeof(state.claimed));
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaSessionDispatch lt_split_dispatch() {
    auto functions = lt_dispatch();
    functions.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        // The general tiny Driver fixture spaces addresses by 64 KiB. These
        // full-shape matrices need genuinely disjoint synthetic intervals.
        ++state.allocations;
        *output = static_cast<CUdeviceptr>(state.allocations) * 0x10000000ULL;
        state.memory[*output].resize(bytes);
        return CUDA_SUCCESS;
    };
    functions.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        constexpr std::array names{"glimmer_owned_lt_sgemm", "glimmer_owned_lt_hgemm",
                                   "glimmer_owned_lt_hsgemm"};
        const auto found = std::find(names.begin(), names.end(), std::string_view(name));
        if (found == names.end()) return lookup_function(output, module, name);
        *output = reinterpret_cast<CUfunction>(&lt_split.functions[found - names.begin()]);
        return CUDA_SUCCESS;
    };
    functions.parameter_info = lt_split_parameter;
    functions.execution.launch_kernel = lt_split_launch;
    functions.lt.version = [] { return lt_split.version; };
    functions.lt.layout_get = lt_split_layout;
    functions.lt.operation_get = lt_split_operation;
    functions.lt.preference_get = [](cublasLtMatmulPreference_t,
                                     cublasLtMatmulPreferenceAttributes_t attribute, void* output,
                                     std::size_t capacity, std::size_t* written) {
        expect(attribute == CUBLASLT_MATMUL_PREF_SEARCH_MODE, "original heuristic search mode");
        return lt_split_output(output, capacity, written, std::uint32_t{0});
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact SDK callback signatures.
    functions.lt.heuristic = [](cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int requested,
                                cublasLtMatmulHeuristicResult_t* output, int* count) {
        expect(requested == 1, "one original algorithm identity requested");
        ++lt_split.heuristics;
        *output = {};
        output->algo.data[0] = 0x12345678;
        output->state = CUBLAS_STATUS_SUCCESS;
        output->workspaceSize = lt_split.required_workspace;
        *count = 1;
        return CUBLAS_STATUS_SUCCESS;
    };
    functions.lt.algorithm_check =
        [](cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
           cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t* algorithm,
           cublasLtMatmulHeuristicResult_t* output) {
            ++lt_split.checks;
            expect(algorithm->data[0] == 0x12345678, "only the original algorithm is checked");
            output->state = lt_split.algorithm_status;
            output->workspaceSize = lt_split.required_workspace;
            return lt_split.check_status;
        };
    functions.lt.configuration_get = [](const cublasLtMatmulAlgo_t* algorithm,
                                        cublasLtMatmulAlgoConfigAttributes_t attribute,
                                        void* output, std::size_t capacity, std::size_t* written) {
        expect(algorithm->data[0] == 0x12345678, "configuration retains algorithm provenance");
        expect(attribute == CUBLASLT_ALGO_CONFIG_SPLITK_NUM ||
                   attribute == CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
               "only audited algorithm fields are observed");
        return lt_split_output(
            output, capacity, written,
            attribute == CUBLASLT_ALGO_CONFIG_SPLITK_NUM ? lt_split.split : lt_split.reduction);
    };
    functions.lt.matmul =
        [](cublasLtHandle_t handle, cublasLtMatmulDesc_t, const void*, const void*,
           cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t, const void*, const void*,
           cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
           const cublasLtMatmulAlgo_t* algorithm, void*, std::size_t, cudaStream_t) {
            expect(state.current == native_context() && state.is_lt_live &&
                       handle == reinterpret_cast<cublasLtHandle_t>(&state.lt_storage) &&
                       (!algorithm || algorithm->data[0] == 0x12345678),
                   "fallback executes only the exact live original Lt object and algorithm");
            ++lt_split.native_calls;
            return lt_split.original_status;
        };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return functions;
}

struct LtSplitProfile {
    int depth{768};
    std::size_t version{120604};
};
struct LtSplitOwner {
    std::shared_ptr<LtSplitDescriptors> metadata{std::make_shared<LtSplitDescriptors>()};
    std::unique_ptr<CudaExecutionSession> session;
    CudaSessionLtMatmul problem;
    std::array<CudaSessionResult, 4> matrices;
    CudaSessionResult bias, workspace, scalars;

    explicit LtSplitOwner(LtSplitProfile profile) {
        const ExecutionSessionIdentity identity{91, 92};
        const auto depth = profile.depth;
        state = {};
        state.is_sgemm = true;
        lt_split = {};
        lt_split.version = profile.version;
        lt_split.split = depth == 3072 ? 3 : 6;
        session = std::make_unique<CudaExecutionSession>(identity, 93, lt_split_dispatch(),
                                                         CudaSessionLimits{},
                                                         CudaSessionPersistentConfig{320, true});
        success(session->initialize(0));
        const auto handle = session->create_lt();
        blas_success(handle);
        problem.descriptors.handle = handle.id;
        for (std::size_t index = 0; index < matrices.size(); ++index) {
            auto& layout = metadata->layouts[index];
            if (index < 2) layout.rows = static_cast<std::uint64_t>(depth);
            layout.leading = static_cast<std::int64_t>(layout.rows);
            matrices[index] = session->allocate(static_cast<std::size_t>(layout.rows) *
                                                static_cast<std::size_t>(layout.columns) * 4);
            success(matrices[index]);
            problem.matrices[index] = {matrices[index].id, 0};
            problem.descriptors.matrices[index] = {
                CudaLtObjectKind::kLayout, &layout, {}, metadata};
        }
        bias = session->allocate(768 * sizeof(float));
        workspace = session->allocate(4096);
        scalars = session->allocate(16);
        success(bias);
        success(workspace);
        success(scalars);
        metadata->operation.bias = bias.address;
        problem.descriptors.operation = {
            CudaLtObjectKind::kOperation, &metadata->operation,
            CudaLtDeviceReference{identity.client_id, bias.id, 0, bias.address}, metadata};
        problem.descriptors.preference =
            CudaLtObjectView{CudaLtObjectKind::kPreference, &metadata->preference, {}, metadata};
        problem.descriptors.requested = 1;
        problem.workspace = {workspace.id, 0};
        problem.workspace_bytes = 4096;
        problem.scalar_bytes = sizeof(float);
        set_scalars({0.75F, 0.0F});
        const auto algorithms = session->query_lt_algorithms(problem.descriptors);
        blas_success(algorithms.result);
        expect(algorithms.algorithms.size() == 1 && algorithms.sdk_version == profile.version &&
                   lt_split.argument_queries == std::array<std::size_t, 3>{28, 28, 28},
               "all three private Lt providers validate every new ABI parameter before selection");
        problem.descriptors.algorithm = algorithms.algorithms[0].algo;
    }
    void set_scalars(std::array<float, 2> coefficients) {
        std::memcpy(problem.alpha.data(), &coefficients[0], sizeof(float));
        std::memcpy(problem.beta.data(), &coefficients[1], sizeof(float));
    }
    void close() {
        problem = {};
        metadata.reset();
        success(session->shutdown());
        expect(state.memory.empty() && state.lt_destroys == 1 && state.context_destroys == 1,
               "Lt allocations, handle and context retire exactly once after physical proof");
    }
};
template <class Value>
Value lt_split_argument(std::size_t offset) {
    Value value{};
    std::memcpy(&value, lt_split.parameters.data() + offset, sizeof(value));
    return value;
}
void owned_lt_split_k_selection_and_resume() {
    if (cuda_sgemm_image().empty()) return;
    for (const int depth : {768, 3072}) {
        LtSplitOwner owner({.depth = depth});
        const auto handle = owner.problem.descriptors.handle;
        const std::weak_ptr<const void> lifetime = owner.metadata;
        const auto accepted = owner.session->accept_lt_matmul(1, owner.problem);
        expect(accepted.accepted() && !lt_split.native_calls && !lt_split.private_launches &&
                   lt_split.checks == 1,
               "qualified split-K acceptance validates original algorithm without launching");
        owner.set_scalars({99.0F, 99.0F});
        owner.problem = {};
        owner.metadata.reset();
        state.claimed = 4;
        const auto issued = owner.session->issue_next();
        expect(issued.native_status == CUDA_SUCCESS && !issued.library_status &&
                   lt_split.private_launches == 1 && !lt_split.native_calls && !lifetime.expired(),
               "owned split-K issues once without inventing an original Lt status");
        for (std::size_t index = 0; index < owner.matrices.size(); ++index)
            expect(lt_split_argument<CUdeviceptr>(index * 8) == owner.matrices[index].address,
                   "all four immutable native matrix addresses use exact ABI positions");
        expect(lt_split_argument<CUdeviceptr>(32) == owner.bias.address &&
                   lt_split_argument<int>(40) == 768 && lt_split_argument<int>(44) == 768 &&
                   lt_split_argument<int>(48) == depth && lt_split_argument<int>(52) == depth &&
                   lt_split_argument<int>(56) == depth && lt_split_argument<int>(60) == 768 &&
                   lt_split_argument<int>(64) == 768 && lt_split_argument<std::uint32_t>(68) == 1 &&
                   lt_split_argument<float>(72) == 0.75F && lt_split_argument<float>(76) == 0.0F &&
                   lt_split_argument<CUdeviceptr>(80) == 0 &&
                   lt_split_argument<CUdeviceptr>(88) == 0 &&
                   lt_split_argument<std::uint32_t>(96) == 0 &&
                   lt_split_argument<std::uint32_t>(100) == 1 &&
                   lt_split_argument<CUdeviceptr>(104) != 0 &&
                   lt_split_argument<CUdeviceptr>(112) != 0 &&
                   lt_split_argument<std::int64_t>(120) == 0 &&
                   lt_split_argument<std::int64_t>(128) == 0 &&
                   lt_split_argument<std::int64_t>(136) == 0 &&
                   lt_split_argument<std::int64_t>(144) == 0 &&
                   lt_split_argument<std::int64_t>(152) == 0 &&
                   lt_split_argument<std::int32_t>(160) == 1 &&
                   lt_split_argument<std::uint32_t>(164) == (depth == 768 ? 128U : 1024U),
               "all 28 copied Lt fields preserve pinned partition depth and scalar snapshots");
        expect(
            owner.session->host_result(accepted.receipt).has_value() &&
                owner.session->acknowledge_result(accepted.receipt) && !lifetime.expired() &&
                owner.session->release(owner.bias.id).disposition ==
                    CudaSessionDisposition::kDeferred &&
                owner.session->release(owner.workspace.id).disposition ==
                    CudaSessionDisposition::kDeferred &&
                owner.session->destroy_lt(handle).disposition == CudaSessionDisposition::kDeferred,
            "host ACK cannot release metadata, bias, workspace or the original Lt handle");
        expect(owner.session->request_retreat(1) == CUDA_SUCCESS,
               "cooperative retreat is requested only between whole output tiles");
        state.query_status = CUDA_SUCCESS;
        expect(owner.session->poll().paused == 1 && !lifetime.expired(),
               "actual stop retains incomplete split-K owners");
        const auto progress = owner.session->persistent_progress(1);
        expect(progress && progress->completed_blocks == 4 && progress->total_blocks == 144 &&
                   progress->physical_stops == 1,
               "four of 144 complete output tiles stop without publishing a partial partition");
        state.claimed = 144;
        const auto resumed = owner.session->resume(1);
        expect(resumed.native_status == CUDA_SUCCESS && !resumed.library_status &&
                   state.counter_before_launch == std::vector<std::uint64_t>{0, 4} &&
                   state.prefix_copies == 1 && lt_split.private_launches == 2,
               "continuation preserves complete tile claims without resetting or vendor replay");
        expect(owner.session->poll().retired == 1 && !lt_split.native_calls && lifetime.expired(),
               "only final physical completion releases immutable split-K metadata owners");
        owner.close();
    }
}

void owned_lt_split_k_fallback_and_provenance() {
    if (cuda_sgemm_image().empty()) return;
    for (unsigned int mode = 0; mode < 18; ++mode) {
        LtSplitOwner owner({.depth = mode == 17 ? 1536 : 768,
                            .version = mode == 0 ? std::size_t{120605} : std::size_t{120604}});
        auto& operation = owner.metadata->operation;
        switch (mode) {
            case 0:
                break;  // Unqualified SDK version, even with a matching original algorithm.
            case 1:
                lt_split.split = 3;
                break;
            case 2:
                lt_split.reduction = 0;
                break;
            case 3:
                operation.transposes[0] = CUBLAS_OP_C;
                break;
            case 4:
                owner.set_scalars({0.75F, 1.0F});
                break;
            case 5:
                owner.set_scalars({0.0F, 0.0F});
                break;
            case 6:
                owner.set_scalars({std::numeric_limits<float>::infinity(), 0.0F});
                break;
            case 7:
                owner.set_scalars({std::numeric_limits<float>::quiet_NaN(), 0.0F});
                break;
            case 8:
                operation.compute = CUBLAS_COMPUTE_32F_PEDANTIC;
                break;
            case 9:
                operation.epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
                break;
            case 10:
                operation.bias_type = CUDA_R_16F;
                break;
            case 11:
                lt_split.check_status = CUBLAS_STATUS_NOT_SUPPORTED;
                break;
            case 12:
                lt_split.algorithm_status = CUBLAS_STATUS_NOT_SUPPORTED;
                break;
            case 13:
                lt_split.required_workspace = 8192;
                break;
            case 14:
                operation.pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
                owner.problem.uses_device_scalars = true;
                owner.problem.device_scalars = {{{owner.scalars.id, 0}, {owner.scalars.id, 4}}};
                break;
            case 15:
                operation.epilogue = CUBLASLT_EPILOGUE_DEFAULT;
                owner.problem.descriptors.operation.bias.reset();
                break;
            case 16:
                operation.bias_stride = 768;
                break;
            case 17:
                break;
            default:
                expect(false, "enumerated fallback control");
        }
        const auto accepted = owner.session->accept_lt_matmul(1, owner.problem);
        expect(accepted.accepted() && !owner.session->persistent_progress(1) &&
                   !lt_split.private_launches && !lt_split.native_calls,
               "outside the pinned split-K profile selects native before acceptance");
        lt_split.original_status = CUBLAS_STATUS_NOT_SUPPORTED;
        const auto issued = owner.session->issue_next();
        expect(issued.library_status == CUBLAS_STATUS_NOT_SUPPORTED && lt_split.native_calls == 1 &&
                   !lt_split.private_launches,
               "fallback preserves the exact original Lt error without replacement or retry");
        expect(owner.session->acknowledge_result(accepted.receipt), "one fallback host receipt");
        state.query_status = CUDA_SUCCESS;
        expect(owner.session->poll().retired == 1,
               "original error still needs physical retirement");
        owner.close();
    }
    {
        LtSplitOwner owner({});
        owner.problem.descriptors.algorithm->data[0] ^= 1;
        expect(!owner.session->accept_lt_matmul(1, owner.problem).accepted() && !lt_split.checks &&
                   !lt_split.native_calls && !lt_split.private_launches,
               "unissued or modified algorithm identity cannot authorize split-K arithmetic");
        owner.close();
    }
    {
        LtSplitOwner owner({});
        lt_split.split = 1;
        lt_split.reduction = 0;
        const auto accepted = owner.session->accept_lt_matmul(1, owner.problem);
        expect(accepted.accepted(), "previous direct-reduction profile remains selectable");
        state.claimed = 144;
        expect(owner.session->issue_next().native_status == CUDA_SUCCESS &&
                   lt_split.private_launches == 1 && lt_split_argument<std::uint32_t>(164) == 0 &&
                   !lt_split.native_calls,
               "ordinary direct Lt passes zero in the new trailing parameter");
        expect(owner.session->acknowledge_result(accepted.receipt), "direct-reduction receipt");
        state.query_status = CUDA_SUCCESS;
        expect(owner.session->poll().retired == 1, "direct-reduction physical retirement");
        owner.close();
    }
}
void owned_lt_split_k_cancellation_and_abi_rejection() {
    if (cuda_sgemm_image().empty()) return;
    {
        LtSplitOwner owner({});
        const std::weak_ptr<const void> lifetime = owner.metadata;
        const auto accepted = owner.session->accept_lt_matmul(1, owner.problem);
        expect(accepted.accepted(), "split-K command is queued before cancellation");
        owner.problem = {};
        owner.metadata.reset();
        expect(!lifetime.expired() && owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
               "queued cancellation releases no original work before explicit rejection");
        const auto result = owner.session->host_result(accepted.receipt);
        expect(result && result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   result->status == CUDA_ERROR_INVALID_VALUE && !lt_split.native_calls &&
                   !lt_split.private_launches && !state.prefix_copies &&
                   owner.session->acknowledge_result(accepted.receipt) && lifetime.expired(),
               "unissued cancellation has no Lt provenance, launch, event or retained metadata");
        owner.close();
    }
    {
        state = {};
        state.is_sgemm = true;
        lt_split = {};
        lt_split.has_bad_depth_abi = true;
        CudaExecutionSession session({91, 92}, 93, lt_split_dispatch(), {}, {320, true});
        success(session.initialize(0));
        // The original handle is not created until private ABI preparation
        // succeeds, so failure has neither an Lt owner nor original SDK status.
        const auto result = session.create_lt();
        expect(
            result.status == CUDA_ERROR_INVALID_VALUE && !result.original_status && !result.id &&
                !state.lt_creates &&
                lt_split.argument_queries == std::array<std::size_t, 3>{28, 0, 0} &&
                !lt_split.heuristics && !lt_split.native_calls && !lt_split.private_launches,
            "wrong trailing depth width is rejected before native Lt creation or private launch");
        success(session.shutdown());
        expect(state.memory.empty() && !state.lt_creates && !state.lt_destroys &&
                   state.gemm_loads == 1 && state.unloads == 0 && state.context_destroys == 1,
               "successful owned-context destruction settles its one private module without "
               "a separate unload or an absent Lt destroy");
    }
}
void owned_lt_handles_and_physical_retirement() {
    state = {};
    CudaSessionLimits limits;
    limits.max_objects = 1;
    CudaExecutionSession session({41, 42}, 43, lt_dispatch(), limits);
    expect(session.create_lt().status != CUDA_SUCCESS && state.lt_creates == 0,
           "uninitialized context cannot create a native Lt object");
    success(session.initialize(0));
    state.lt_create_status = CUBLAS_STATUS_ALLOC_FAILED;
    const auto rejected = session.create_lt();
    expect(rejected.original_status == CUBLAS_STATUS_ALLOC_FAILED &&
               rejected.status == CUDA_SUCCESS && !rejected.id &&
               session.snapshot().lt_handles == 0,
           "known-empty Lt failure preserves SDK domain and permits fresh retry");
    state.lt_create_status = CUBLAS_STATUS_SUCCESS;
    const auto handle = session.create_lt();
    blas_success(handle);
    const auto creations = state.lt_creates;
    expect(handle.id && session.snapshot().lt_handles == 1 &&
               session.create_lt().status == CUDA_ERROR_OUT_OF_MEMORY &&
               state.lt_creates == creations,
           "Lt quota rejects before SDK invocation");
    auto foreign = std::async(std::launch::async, [&] { return session.destroy_lt(handle.id); });
    expect(foreign.get().status == CUDA_ERROR_INVALID_CONTEXT && state.lt_destroys == 0,
           "foreign owner cannot destroy the native Lt resource");
    const auto child = ::fork();
    expect(child >= 0, "Lt fork exclusion child");
    if (!child) ::_exit(session.destroy_lt(handle.id).status == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before native Lt SDK calls");
    const auto module = session.load_ptx(kPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto memory = session.allocate(8);
    success(memory);
    std::array<std::byte, 12> parameters{};
    std::memcpy(parameters.data(), &memory.address, 8);
    const std::uint32_t value{19};
    std::memcpy(parameters.data() + 8, &value, 4);
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .grid = {80, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    expect(accepted.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt) && session.poll().retained == 1,
           "actual pending kernel event survives SDK result delivery");
    expect(session.destroy_lt(handle.id).disposition == CudaSessionDisposition::kDeferred &&
               session.create_lt().disposition == CudaSessionDisposition::kDeferred &&
               state.lt_destroys == 0 && state.lt_creates == creations,
           "library lifecycle cannot overtake physical GPU work");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "true trailing event permits Lt lifecycle mutation");
    blas_success(session.destroy_lt(handle.id));
    const auto next = session.create_lt();
    blas_success(next);
    expect(next.id > handle.id && session.destroy_lt(handle.id).status == CUDA_ERROR_INVALID_HANDLE,
           "reused native Lt address never recycles a context-scoped identity");
    success(session.shutdown());
    expect(state.lt_destroys == 2 && !state.is_lt_live && state.context_destroys == 1,
           "shutdown destroys a remaining Lt handle exactly once before its context");
}
void owned_lt_partial_failure_containment() {
    for (unsigned int mode = 0; mode < 3; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "isolated ambiguous Lt child");
        if (!child) {
            try {
                state = {};
                CudaExecutionSession session({51, 52}, 53, lt_dispatch());
                success(session.initialize(0));
                if (mode == 2) {
                    const auto handle = session.create_lt();
                    blas_success(handle);
                    state.lt_destroy_status = CUBLAS_STATUS_EXECUTION_FAILED;
                    const auto result = session.destroy_lt(handle.id);
                    expect(result.original_status == CUBLAS_STATUS_EXECUTION_FAILED &&
                               result.status == CUDA_SUCCESS,
                           "original Lt destruction error keeps its library status domain");
                } else {
                    state.returns_partial_lt = mode == 0;
                    state.returns_null_lt = mode == 1;
                    state.lt_create_status =
                        mode == 0 ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_SUCCESS;
                    const auto result = session.create_lt();
                    expect(result.status == CUDA_ERROR_UNKNOWN &&
                               result.original_status == state.lt_create_status,
                           "partial/null Lt creation retains original SDK result separately from "
                           "framework uncertainty");
                }
                const auto attempts = state.lt_destroys;
                expect(session.snapshot().is_quarantined && session.snapshot().lt_handles == 1 &&
                           !session.create_lt().original_status &&
                           !session.destroy_lt(1).original_status &&
                           session.shutdown().status != CUDA_SUCCESS &&
                           session.shutdown().status != CUDA_SUCCESS &&
                           state.lt_destroys == attempts && state.context_destroys == 0,
                       "ambiguous Lt ownership retains context without destruction or issue retry");
                ::_exit(EXIT_SUCCESS);
            } catch (...) {
                ::_exit(EXIT_FAILURE);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "uncertain Lt child is contained, not a leak-free recovery claim");
    }
}
template <typename Handle>
cublasStatus_t library_lifetime_create(Handle* output) {
    auto& fixture = library_lifetime_state;
    ++fixture.creates;
    expect(!fixture.requires_context || state.current == native_context(),
           "library creation runs on its actual owning context");
    const auto index = fixture.returns_duplicate ? 0U : fixture.creates - 1;
    expect(index < fixture.handles.size(), "bounded independent native library identities");
    fixture.live[index] = true;
    *output = reinterpret_cast<Handle>(&fixture.handles[index]);
    return fixture.create_status;
}
template <typename Handle>
cublasStatus_t library_lifetime_destroy(Handle handle) {
    auto& fixture = library_lifetime_state;
    ++fixture.destroys;
    expect(!fixture.requires_context ||
               (state.current == native_context() && state.context_destroys == 0),
           "library retirement precedes its actual owned context");
    const auto found = std::ranges::find_if(fixture.handles, [&](char& storage) {
        return reinterpret_cast<Handle>(&storage) == handle;
    });
    expect(found != fixture.handles.end(), "destroy only the exact native library identity");
    const auto index = static_cast<std::size_t>(found - fixture.handles.begin());
    expect(fixture.live[index], "native library destruction never repeats a consumed identity");
    const bool fails = fixture.destroys == fixture.failing_destroy;
    if (!fails || fixture.consumes_failed_destroy) fixture.live[index] = false;
    return fails ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
}
CudaSessionDispatch library_lifetime_dispatch(bool is_lt) {
    auto functions = blas_dispatch();
    if (is_lt) {
        functions.lt.create = library_lifetime_create<cublasLtHandle_t>;
        functions.lt.destroy = library_lifetime_destroy<cublasLtHandle_t>;
    } else {
        functions.blas.create = library_lifetime_create<cublasHandle_t>;
        functions.blas.destroy = library_lifetime_destroy<cublasHandle_t>;
    }
    return functions;
}
void owned_library_identity_and_partial_shutdown(unsigned int first_mode = 0) {
    for (const bool is_lt : {false, true}) {
        for (unsigned int mode = first_mode; mode < 5; ++mode) {
            const auto child = ::fork();
            expect(child >= 0, "isolated library identity and partial-shutdown control");
            if (!child) {
                std::unique_ptr<CudaExecutionSession> owner;
                try {
                    state = {};
                    library_lifetime_state = {};
                    owner = std::make_unique<CudaExecutionSession>(
                        ExecutionSessionIdentity{61, 62}, 63, library_lifetime_dispatch(is_lt));
                    auto& session = *owner;
                    success(session.initialize(0));
                    const auto create = [&] {
                        return is_lt ? session.create_lt() : session.create_blas();
                    };
                    const auto count = [&] {
                        const auto snapshot = session.snapshot();
                        return is_lt ? snapshot.lt_handles : snapshot.blas_handles;
                    };
                    const auto first = create();
                    blas_success(first);
                    if (mode < 2) {
                        library_lifetime_state.returns_duplicate = true;
                        library_lifetime_state.create_status =
                            mode == 0 ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
                        const auto duplicate = create();
                        expect(
                            duplicate.disposition == CudaSessionDisposition::kRejected &&
                                duplicate.status == CUDA_ERROR_UNKNOWN && !duplicate.id &&
                                duplicate.original_status == library_lifetime_state.create_status &&
                                count() == 2 && session.snapshot().is_quarantined,
                            "duplicate live native identity retains both claims, preserves "
                            "original status and publishes no second owner");
                        expect(!create().original_status &&
                                   session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                   session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                   library_lifetime_state.creates == 2 &&
                                   library_lifetime_state.destroys == 0 &&
                                   state.context_destroys == 0,
                               "duplicate identity quarantines the existing owner too: neither "
                               "native destruction nor context cleanup may be guessed safe");
                    } else {
                        const auto second = create();
                        blas_success(second);
                        expect(first.id != second.id && count() == 2,
                               "distinct native library resources have distinct opaque owners");
                        if (mode < 4) {
                            library_lifetime_state.failing_destroy = 2;
                            library_lifetime_state.consumes_failed_destroy = mode == 3;
                            expect(session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                       count() == 1 && session.snapshot().is_quarantined &&
                                       library_lifetime_state.destroys == 2 &&
                                       state.context_destroys == 0,
                                   "partial shutdown immediately removes only proven-dead native "
                                   "records and retains the failed destruction claim");
                            expect(session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                       count() == 1 && library_lifetime_state.destroys == 2 &&
                                       state.context_destroys == 0,
                                   "partial shutdown never retries a destructor, including one "
                                   "that consumed its resource before returning failure");
                        } else {
                            success(session.shutdown());
                            success(session.shutdown());
                            expect(count() == 0 && library_lifetime_state.destroys == 2 &&
                                       std::ranges::none_of(library_lifetime_state.live,
                                                            [](bool live) { return live; }) &&
                                       state.context_destroys == 1,
                                   "ordinary cleanup destroys both native resources and the "
                                   "context exactly once");
                        }
                    }
                    ::_exit(EXIT_SUCCESS);
                } catch (const std::exception& error) {
                    std::cerr << (is_lt ? "Lt" : "BLAS") << " lifetime mode " << mode << ": "
                              << error.what() << '\n';
                    ::_exit(EXIT_FAILURE);
                }
            }
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated native library lifetime regression passes");
        }
    }
}
CudaLtDispatch library_lifetime_registry_dispatch() {
    CudaLtDispatch functions;
    // Exact native signatures; these fixtures never query or mutate attributes.
    functions.layout_create = [](cublasLtMatrixLayout_t* output, cudaDataType, std::uint64_t,
                                 std::uint64_t,
                                 std::int64_t) { return library_lifetime_create(output); };
    functions.operation_create = [](cublasLtMatmulDesc_t* output, cublasComputeType_t,
                                    cudaDataType) { return library_lifetime_create(output); };
    functions.preference_create = library_lifetime_create<cublasLtMatmulPreference_t>;
    functions.layout_destroy = library_lifetime_destroy<cublasLtMatrixLayout_t>;
    functions.operation_destroy = library_lifetime_destroy<cublasLtMatmulDesc_t>;
    functions.preference_destroy = library_lifetime_destroy<cublasLtMatmulPreference_t>;
    functions.layout_set = [](cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void*,
                              std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    functions.operation_set = [](cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*,
                                 std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    functions.preference_set = [](cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                                  const void*, std::size_t) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    functions.layout_get = [](cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void*,
                              std::size_t, std::size_t*) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    functions.operation_get = [](cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void*,
                                 std::size_t, std::size_t*) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    functions.preference_get = [](cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                                  void*, std::size_t,
                                  std::size_t*) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    return functions;
}
void owned_library_registry_duplicates() {
    const std::array kinds{CudaLtObjectKind::kLayout, CudaLtObjectKind::kOperation,
                           CudaLtObjectKind::kPreference};
    for (std::size_t kind = 0; kind < kinds.size(); ++kind) {
        for (unsigned int mode = 0; mode < 3; ++mode) {
            const auto child = ::fork();
            expect(child >= 0, "isolated duplicate descriptor registry control");
            if (!child) {
                std::unique_ptr<CudaLtRegistry> owner;
                try {
                    state = {};
                    library_lifetime_state = {};
                    library_lifetime_state.requires_context = false;
                    owner = std::make_unique<CudaLtRegistry>(library_lifetime_registry_dispatch());
                    auto& registry = *owner;
                    CudaLtCreate specification{
                        .kind = kinds[kind], .rows = 2, .columns = 2, .leading = 2};
                    const auto first = registry.create(specification);
                    expect(first.id && first.original_status == CUBLAS_STATUS_SUCCESS,
                           "first descriptor has one successful native owner");
                    auto lease = registry.borrow(first.id, kinds[kind]);
                    expect(lease && lease->lifetime,
                           "real immutable descriptor borrow precedes duplicate creation");
                    library_lifetime_state.returns_duplicate = true;
                    library_lifetime_state.create_status =
                        mode == 1 ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_SUCCESS;
                    if (mode == 2) specification.kind = kinds[(kind + 1) % kinds.size()];
                    const auto duplicate = registry.create(specification);
                    expect(duplicate.disposition == CudaLtDisposition::kRejected &&
                               duplicate.status == CUDA_ERROR_UNKNOWN && !duplicate.id &&
                               duplicate.original_status == library_lifetime_state.create_status &&
                               registry.snapshot().objects == 2 &&
                               registry.snapshot().is_quarantined &&
                               !registry.borrow(first.id, kinds[kind]),
                           "same or cross-kind duplicate native descriptor cannot publish a "
                           "second owner or allow new borrows");
                    lease.reset();
                    expect(!registry.destroy(first.id).original_status &&
                               registry.shutdown().status == CUDA_ERROR_UNKNOWN &&
                               registry.shutdown().status == CUDA_ERROR_UNKNOWN &&
                               library_lifetime_state.creates == 2 &&
                               library_lifetime_state.destroys == 0 && state.context_creates == 0 &&
                               state.context_destroys == 0,
                           "uncertain descriptor identities never retry destruction, including "
                           "after earlier immutable borrows have been released");
                    ::_exit(EXIT_SUCCESS);
                } catch (const std::exception& error) {
                    std::cerr << "Lt registry kind " << kind << " mode " << mode << ": "
                              << error.what() << '\n';
                    ::_exit(EXIT_FAILURE);
                }
            }
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated duplicate native descriptor regression passes");
        }
    }
}
void owned_blas_transpose_and_thread_contract() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, blas_dispatch());
    success(session.initialize(0));
    const auto handle = session.create_blas();
    const auto a = session.allocate(256), b = session.allocate(256), c = session.allocate(256);
    blas_success(handle);
    success(a);
    success(b);
    success(c);
    std::uint64_t sequence{1};
    for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
        for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
            const CudaSessionSgemm problem{.handle = handle.id,
                                           .transpose_a = ta,
                                           .transpose_b = tb,
                                           .rows = 2,
                                           .columns = 3,
                                           .depth = 4,
                                           .leading_a = 4,
                                           .leading_b = 4,
                                           .leading_c = 4,
                                           .a = {a.id, 0},
                                           .b = {b.id, 0},
                                           .c = {c.id, 0}};
            const auto accepted = session.accept_sgemm(sequence++, problem);
            expect(accepted.accepted() &&
                       session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
                       session.acknowledge_result(accepted.receipt),
                   "all real FP32 transpose forms issue");
            state.query_status = CUDA_SUCCESS;
            expect(session.poll().retired == 1, "each original SGEMM stops exactly once");
        }
    }
    const auto calls = state.blas_creates + state.blas_sets + state.blas_destroys;
    auto wrong = std::async(std::launch::async, [&] {
        return session.create_blas().status == CUDA_ERROR_INVALID_CONTEXT &&
               session.set_blas_stream(handle.id, 0).status == CUDA_ERROR_INVALID_CONTEXT &&
               session.destroy_blas(handle.id).status == CUDA_ERROR_INVALID_CONTEXT;
    });
    expect(wrong.get() && calls == state.blas_creates + state.blas_sets + state.blas_destroys,
           "foreign thread rejects before any native library call");
    const auto child = fork();
    expect(child >= 0, "fork inherited library session");
    if (child == 0) {
        const bool guarded = session.create_blas().status == CUDA_ERROR_INVALID_CONTEXT &&
                             session.destroy_blas(handle.id).status == CUDA_ERROR_INVALID_CONTEXT;
        _exit(guarded ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child cannot execute or release the parent's cuBLAS handle");
    success(session.shutdown());
    expect(state.blas_issues == 9 && state.blas_destroys == 1,
           "shutdown destroys the live handle after all nine original operations");
}
void owned_blas_device_scalars_and_native_attributes() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, blas_dispatch());
    success(session.initialize(0));
    const auto handle = session.create_blas();
    const auto a = session.allocate(64), b = session.allocate(64), c = session.allocate(64);
    const auto scalar = session.allocate(16);
    blas_success(handle);
    success(a);
    success(b);
    success(c);
    success(scalar);
    const std::array<float, 4> values{99.0F, 2.0F, 0.5F, 99.0F};
    success(session.upload({scalar.id, 0}, std::as_bytes(std::span{values})));
    const auto version = session.get_blas_attribute(handle.id, CudaBlasAttribute::kVersion);
    blas_success(version);
    expect(version.value == 120605, "library version is an actual SDK result");
    blas_success(
        session.set_blas_attribute(handle.id, CudaBlasAttribute::kMathMode, CUBLAS_PEDANTIC_MATH));
    expect(session.get_blas_attribute(handle.id, CudaBlasAttribute::kMathMode).value ==
               CUBLAS_PEDANTIC_MATH,
           "actual math mode survives native configuration");
    expect(session.set_blas_attribute(handle.id, CudaBlasAttribute::kMathMode, 5).original_status ==
                   CUBLAS_STATUS_INVALID_VALUE &&
               !session.snapshot().is_quarantined &&
               session.get_blas_attribute(handle.id, CudaBlasAttribute::kMathMode).value ==
                   CUBLAS_PEDANTIC_MATH,
           "native validation rejection preserves the old math state and permits later work");
    blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode,
                                            CUBLAS_POINTER_MODE_DEVICE));
    CudaSessionSgemm problem{.handle = handle.id,
                             .rows = 2,
                             .columns = 2,
                             .depth = 2,
                             .leading_a = 2,
                             .leading_b = 2,
                             .leading_c = 2,
                             .a = {a.id, 0},
                             .b = {b.id, 0},
                             .c = {c.id, 0},
                             .alpha = 99.0F,
                             .beta = 99.0F,
                             .uses_device_scalars = true,
                             .device_alpha = {scalar.id, 4},
                             .device_beta = {scalar.id, 8}};
    auto invalid = problem;
    invalid.device_beta.offset = 15;
    expect(!session.accept_sgemm(1, invalid).accepted(),
           "reject unaligned/overflowing device scalar extent");
    invalid = problem;
    invalid.uses_device_scalars = false;
    expect(!session.accept_sgemm(1, invalid).accepted(),
           "device pointer mode cannot be reinterpreted as host scalars");
    const auto accepted = session.accept_sgemm(1, problem);
    expect(accepted.accepted() && session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
               state.blas_alpha == 2.0F && state.blas_beta == 0.5F,
           "actual SDK call receives owned device scalar pointers, not host float addresses");
    expect(
        session.get_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode).value ==
                CUBLAS_POINTER_MODE_DEVICE &&
            session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode,
                                       CUBLAS_POINTER_MODE_HOST)
                    .disposition == CudaSessionDisposition::kDeferred &&
            session.release(scalar.id).disposition == CudaSessionDisposition::kDeferred &&
            session.acknowledge_result(accepted.receipt) && session.poll().retained == 1,
        "native metadata query progresses before stop, but scalar/mode lifetime does not retire");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "actual library stop releases device scalar leases");
    blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode,
                                            CUBLAS_POINTER_MODE_HOST));
    success(session.release(scalar.id));
    success(session.shutdown());
}
void owned_blas_workspace_and_policy_lifetimes() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, blas_dispatch());
    success(session.initialize(0));
    const auto handle = session.create_blas();
    blas_success(handle);
    const auto a = session.allocate(64), b = session.allocate(64), c = session.allocate(64);
    const auto workspace = session.allocate(32768 + 256);
    success(a);
    success(b);
    success(c);
    success(workspace);
    const auto offset = static_cast<std::size_t>((256 - workspace.address % 256) % 256);
    const auto native_handle = state.blas.begin()->first;
    blas_success(session.set_blas_workspace(handle.id, {workspace.id, offset}, 16384));
    expect(state.blas_policy.at(native_handle).workspace == workspace.address + offset &&
               session.snapshot().allocations == 4 &&
               session.snapshot().device_bytes == 3 * 64 + 32768 + 256,
           "workspace is an owned interval, not another allocation or quota charge");
    const auto misaligned =
        session.set_blas_workspace(handle.id, {workspace.id, offset + 1}, 16384);
    expect(misaligned.original_status == CUBLAS_STATUS_INVALID_VALUE &&
               !session.snapshot().is_quarantined &&
               state.blas_policy.at(native_handle).workspace == workspace.address + offset,
           "original alignment rejection preserves the previous workspace configuration");
    const auto calls = state.workspace_sets;
    expect(
        !session.set_blas_workspace(handle.id, {UINT64_MAX, 0}, 16).original_status &&
            !session.set_blas_workspace(handle.id, {workspace.id, SIZE_MAX}, 16).original_status &&
            !session.set_blas_workspace(handle.id, {workspace.id, 0}, SIZE_MAX).original_status &&
            !session.set_blas_workspace(handle.id, {workspace.id, 32768 + 256}, 0)
                 .original_status &&
            !session.set_blas_workspace(handle.id, {}, 16).original_status &&
            state.workspace_sets == calls,
        "unowned, overflowing, one-past-end and null nonempty workspaces cannot reach the SDK");
    blas_success(session.get_blas_attribute(handle.id, CudaBlasAttribute::kAtomicsMode));
    blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kAtomicsMode, 1));
    expect(session.get_blas_attribute(handle.id, CudaBlasAttribute::kAtomicsMode).value == 1 &&
               !session.set_blas_attribute(handle.id, CudaBlasAttribute::kAtomicsMode, 2)
                    .original_status &&
               state.atomics_sets == 1,
           "actual atomics mode is retained and invalid raw values reject before enum use");
    blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, 7));
    expect(session.get_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget).value == 7 &&
               session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, 81)
                       .original_status == CUBLAS_STATUS_INVALID_VALUE &&
               !session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, UINT64_MAX)
                    .original_status &&
               session.get_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget).value == 7,
           "original SM hint rejection leaves the old hint and does not fabricate a global policy "
           "change");
    blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, 0));
    CudaSessionSgemm problem{.handle = handle.id,
                             .rows = 2,
                             .columns = 2,
                             .depth = 2,
                             .leading_a = 2,
                             .leading_b = 2,
                             .leading_c = 2,
                             .a = {a.id, 0},
                             .b = {b.id, 0},
                             .c = {c.id, 0},
                             .alpha = 1};
    const auto accepted = session.accept_sgemm(1, problem);
    expect(accepted.accepted() && session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
               session.set_blas_workspace(handle.id, {}, 0).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.release(workspace.id).disposition == CudaSessionDisposition::kDeferred &&
               session.acknowledge_result(accepted.receipt) && session.poll().retained == 1,
           "host result cannot mutate or free a workspace while GPU work still owns it");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "physical stop releases the workspace interval");
    success(session.release(workspace.id));
    expect(!session.accept_sgemm(2, problem).accepted() && state.blas_issues == 1,
           "freeing an idle configured workspace cannot authorize a later stale SDK pointer");
    blas_success(session.set_blas_stream(handle.id, 0));
    expect(!state.blas_policy.at(native_handle).has_workspace &&
               session.get_blas_attribute(handle.id, CudaBlasAttribute::kAtomicsMode).value == 1,
           "same-stream SetStream resets workspace but preserves independent atomics state");
    const auto resumed = session.accept_sgemm(2, problem);
    expect(resumed.accepted() && session.issue_next().library_status == CUBLAS_STATUS_SUCCESS &&
               session.acknowledge_result(resumed.receipt) && session.poll().retired == 1,
           "stream reset recovers default workspace without replaying an old computation");
    blas_success(session.set_blas_workspace(handle.id, {}, 0));
    expect(state.blas_policy.at(native_handle).has_workspace &&
               !state.blas_policy.at(native_handle).workspace_bytes,
           "explicit zero workspace disables the pool rather than meaning default workspace");
    const auto replacement = session.allocate(32768 + 256);
    success(replacement);
    const auto replacement_offset =
        static_cast<std::size_t>((256 - replacement.address % 256) % 256);
    blas_success(
        session.set_blas_workspace(handle.id, {replacement.id, replacement_offset}, 16384));
    blas_success(session.destroy_blas(handle.id));
    expect(session.device_memory({replacement.id, 0}, 1).has_value(),
           "library handle destruction never frees application-owned workspace");
    success(session.release(replacement.id));
    success(session.shutdown());
}
void workspace_mutation_failure_never_replays() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, blas_dispatch());
    success(session.initialize(0));
    const auto handle = session.create_blas();
    const auto workspace = session.allocate(32768 + 256);
    blas_success(handle);
    success(workspace);
    const auto offset = static_cast<std::size_t>((256 - workspace.address % 256) % 256);
    state.workspace_status = CUBLAS_STATUS_EXECUTION_FAILED;
    const auto result = session.set_blas_workspace(handle.id, {workspace.id, offset}, 16384);
    expect(
        result.original_status == CUBLAS_STATUS_EXECUTION_FAILED &&
            session.snapshot().is_quarantined &&
            !session.set_blas_workspace(handle.id, {workspace.id, offset}, 16384).original_status &&
            state.workspace_sets == 1,
        "uncertain workspace mutation retains ownership and cannot replay the setter");
    success(session.shutdown());
    expect(state.blas_destroys == 1 && state.context_destroys == 1,
           "known handle ownership permits one ordinary destruction after configuration failure");
}
void private_workspace_and_sm_policy_selection() {
    if (cuda_sgemm_image().empty()) return;
    for (unsigned int mode = 0; mode < 5; ++mode) {
        state = {};
        state.is_sgemm = true;
        state.claimed = 4;
        CudaExecutionSession session({91, 92}, 93, blas_dispatch(), {}, {320, true});
        success(session.initialize(0));
        const auto handle = session.create_blas();
        blas_success(handle);
        const auto a = session.allocate(sizeof(float) * 72 * 72);
        const auto b = session.allocate(sizeof(float) * 72 * 72);
        const auto c = session.allocate(sizeof(float) * 72 * 72);
        const auto workspace = session.allocate(32768 + 256);
        success(a);
        success(b);
        success(c);
        success(workspace);
        const auto offset = static_cast<std::size_t>((256 - workspace.address % 256) % 256);
        if (mode == 1)
            blas_success(session.set_blas_workspace(handle.id, {}, 0));
        else
            blas_success(session.set_blas_workspace(handle.id, {workspace.id, offset},
                                                    mode == 2 ? 1024 : 16384));
        if (mode >= 3)
            blas_success(
                session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, 7));
        if (mode == 4)
            blas_success(
                session.set_blas_attribute(handle.id, CudaBlasAttribute::kSmCountTarget, 0));
        if (mode == 1 || mode == 2) state.blas_issue_status = CUBLAS_STATUS_ALLOC_FAILED;
        CudaSessionSgemm problem{.handle = handle.id,
                                 .rows = 65,
                                 .columns = 65,
                                 .depth = 9,
                                 .leading_a = 72,
                                 .leading_b = 72,
                                 .leading_c = 72,
                                 .a = {a.id, 0},
                                 .b = {b.id, 0},
                                 .c = {c.id, 0},
                                 .alpha = 1};
        const auto accepted = session.accept_sgemm(1, problem);
        expect(accepted.accepted(), "owned GEMM admits one explicit workspace policy");
        const auto issued = session.issue_next();
        const bool is_private = mode == 0 || mode == 4;
        expect(issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == CUDA_SUCCESS &&
                   (is_private
                        ? !issued.library_status && session.snapshot().persistent_accepted == 1 &&
                              !state.blas_issues
                        : issued.library_status == state.blas_issue_status &&
                              !session.snapshot().persistent_accepted && state.blas_issues == 1),
               "private routing cannot ignore SM hints or hide original small-workspace failures");
        expect(session.acknowledge_result(accepted.receipt),
               "acknowledge exact policy-selected host reply");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().retired == 1,
               "policy-selected work retains resources until actual stop");
        success(session.shutdown());
    }
}
void owned_persistent_sgemm_selection_and_resume() {
    if (cuda_sgemm_image().empty()) return;
    const auto static_bytes = glimmer::transform::inspect_ptx_static_storage(cuda_sgemm_image());
    for (unsigned int mode = 0; mode < 18; ++mode) {
        state = {};
        state.is_sgemm = true;
        CudaExecutionSession session({91, 92}, 93, blas_dispatch(), {}, {320, true});
        success(session.initialize(0));
        const auto private_bytes = session.snapshot().device_bytes;
        const auto handle = session.create_blas();
        blas_success(handle);
        expect(state.gemm_loads == 1 && state.configured_functions.size() == 25 &&
                   session.snapshot().image_bytes == cuda_sgemm_image().size() && static_bytes &&
                   session.snapshot().device_bytes == private_bytes + *static_bytes,
               "one charged private module with twenty-four unchanged and one split-K ABI");
        const auto a = session.allocate(std::size_t{72} * 72 * sizeof(float));
        const auto b = session.allocate(std::size_t{72} * 72 * sizeof(float));
        const auto c = session.allocate(std::size_t{72} * 72 * sizeof(float));
        const auto scalars = session.allocate(16);
        success(a);
        success(b);
        success(c);
        success(scalars);
        const bool device = mode >= 9;
        if (device)
            blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 1));
        CudaSessionSgemm problem{
            .handle = handle.id,
            .transpose_a = static_cast<cublasOperation_t>(mode % 3),
            .transpose_b = static_cast<cublasOperation_t>((mode % 9) / 3),
            .rows = 65,
            .columns = 65,
            .depth = 9,
            .leading_a = 72,
            .leading_b = 72,
            .leading_c = 72,
            .a = {a.id, 0},
            .b = {b.id, 0},
            .c = {c.id, 0},
            .alpha = 2.0F,
            .beta = 0.5F,
            .uses_device_scalars = device,
            .device_alpha = device ? CudaAllocationRange{scalars.id, 4} : CudaAllocationRange{},
            .device_beta = device ? CudaAllocationRange{scalars.id, 8} : CudaAllocationRange{}};
        // Private bindings must not become ordinary public launch handles.
        expect(!session
                    .accept_kernel(1, {.function = handle.id - 1,
                                       .grid = {1, 1, 1},
                                       .block = {128, 1, 1},
                                       .parameters = {}})
                    .accepted(),
               "guessed private GEMM identity is not an application function");
        const auto accepted = session.accept_sgemm(1, problem);
        expect(accepted.accepted() && state.blas_issues == 0 && state.launches == 0,
               "replacement acceptance is neither cuBLAS call nor GPU launch");
        problem.alpha = 99.0F;
        problem.rows = 1;
        state.claimed = 1;
        const auto issued = session.issue_next();
        expect(issued.native_status == CUDA_SUCCESS && !issued.library_status &&
                   state.blas_issues == 0,
               "actual Driver result cannot invent original cuBLAS status");
        float copied_alpha{};
        std::uint32_t scalar_kind{};
        CUdeviceptr scalar_address{};
        std::memcpy(&copied_alpha, state.gemm_parameters.data() + 48, 4);
        std::memcpy(&scalar_kind, state.gemm_parameters.data() + 72, 4);
        std::memcpy(&scalar_address, state.gemm_parameters.data() + 56, 8);
        const auto expected_variant = (mode % 3 ? 2U : 0U) + ((mode % 9) / 3 ? 1U : 0U);
        expect(copied_alpha == 2.0F && scalar_kind == (device ? 1U : 0U) &&
                   scalar_address == (device ? scalars.address + 4 : 0) &&
                   state.gemm_variant == expected_variant,
               "immutable arguments and all transpose/device-scalar selections use the real ABI");
        expect(session.host_result(accepted.receipt).has_value() &&
                   session.acknowledge_result(accepted.receipt),
               "one original host result");
        expect(session.request_retreat(1) == CUDA_SUCCESS &&
                   session.release(a.id).disposition == CudaSessionDisposition::kDeferred,
               "retreat publication does not release active matrix leases");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().paused == 1, "actual event stop pauses incomplete GEMM");
        expect(session.retarget_workers(1, {80}) == CUDA_ERROR_NOT_SUPPORTED,
               "source PTX retargeting cannot silently change private library geometry");
        const auto partial = session.persistent_progress(1);
        expect(partial && partial->completed_blocks == 1 && partial->total_blocks == 4 &&
                   partial->physical_stops == 1,
               "physical progress is neither reset nor fabricated");
        state.claimed = 4;
        const auto resumed = session.resume(1);
        expect(resumed.native_status == CUDA_SUCCESS && !resumed.library_status &&
                   state.counter_before_launch == std::vector<std::uint64_t>{0, 1} &&
                   state.prefix_copies == 1,
               "resume retains claims without reinitializing beta-C work");
        expect(session.poll().retired == 1 && state.blas_issues == 0,
               "replacement retires exactly once without a vendor replay");
        // A supported native math mode outside the provider's initial contract
        // is selected before admission and retains the original SDK result.
        blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kMathMode,
                                                CUBLAS_PEDANTIC_MATH));
        if (device)
            blas_success(session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 0));
        problem.rows = 65;
        problem.uses_device_scalars = false;
        problem.device_alpha = {};
        problem.device_beta = {};
        const auto native = session.accept_sgemm(2, problem);
        expect(native.accepted() && !session.persistent_progress(2),
               "unsupported math selects native before acceptance");
        expect(
            session.issue_next().library_status == CUBLAS_STATUS_SUCCESS && state.blas_issues == 1,
            "native fallback calls original library exactly once");
        expect(session.acknowledge_result(native.receipt) && session.poll().retired == 1,
               "native library still requires real event retirement");
        success(session.shutdown());
    }
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        state.is_sgemm = true;
        CudaSessionLimits limits;
        if (mode == 0) limits.max_image_bytes = 1;
        if (mode == 1) state.is_bad_private_abi = true;
        if (mode == 2) state.module_status = CUDA_ERROR_INVALID_PTX;
        CudaExecutionSession session({91, 92}, 93, blas_dispatch(), limits, {320, true});
        success(session.initialize(0));
        expect(session.create_blas().status != CUDA_SUCCESS && state.blas_creates == 0,
               "image budget, ABI or load failure cannot publish an owned library handle");
        const auto loads = state.gemm_loads;
        expect(session.create_blas().status != CUDA_SUCCESS && state.gemm_loads == loads,
               "failed private preparation never retries an uncertain native load");
        success(session.shutdown());
    }
}

enum class InputGradientCallback { kNone, kContext, kVersion, kOccupancy };
struct InputGradientState {
    CudaExecutionSession* session{};
    CudaSessionGemmEx* problem{};
    InputGradientCallback callback{InputGradientCallback::kNone};
    bool is_extended{}, is_armed{}, loses_context{};
    int version{120604};
    cublasStatus_t version_status{CUBLAS_STATUS_SUCCESS};
    unsigned int version_queries{}, occupancy_queries{}, probes{};
} input_gradient;

void input_gradient_probe(InputGradientCallback callback) {
    if (!input_gradient.is_armed || input_gradient.callback != callback) return;
    input_gradient.is_armed = false;
    ++input_gradient.probes;
    auto& session = *input_gradient.session;
    const auto& problem = *input_gradient.problem;
    const auto calls = state.calls;
    const auto native_issues = state.blas_issues + state.ex_issues;
    const auto nested = input_gradient.is_extended ? session.accept_gemm_ex(1, problem)
                                                   : session.accept_sgemm(1, problem.operation);
    expect(!nested.accepted() &&
               session.destroy_blas(problem.operation.handle).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.release(problem.operation.a.allocation).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.deactivate().disposition == CudaSessionDisposition::kDeferred &&
               session.activate().disposition == CudaSessionDisposition::kDeferred &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               state.calls == calls && state.blas_destroys == 0 && state.context_destroys == 0 &&
               state.blas_issues + state.ex_issues == native_issues,
           "private input-gradient callbacks cannot recursively admit, free, switch or destroy");
    if (input_gradient.loses_context) state.current = nullptr;
}
CudaSessionDispatch input_gradient_dispatch() {
    auto functions = blas_dispatch();
    functions.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        ++state.allocations;
        *output = static_cast<CUdeviceptr>(state.allocations) * 0x10000000ULL;
        state.memory[*output].resize(bytes);
        return CUDA_SUCCESS;
    };
    functions.execution.context_get_current = [](CUcontext* output) {
        input_gradient_probe(InputGradientCallback::kContext);
        return get_current(output);
    };
    functions.blas.get_version = [](cublasHandle_t handle, int* output) {
        expect(state.blas.contains(handle) && state.current == native_context(),
               "private version belongs to the exact original cuBLAS owner");
        ++input_gradient.version_queries;
        input_gradient_probe(InputGradientCallback::kVersion);
        *output = input_gradient.version;
        return input_gradient.version_status;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver occupancy ABI.
    functions.occupancy = [](int* active, CUfunction function, int threads, std::size_t shared) {
        ++input_gradient.occupancy_queries;
        input_gradient_probe(InputGradientCallback::kOccupancy);
        return occupancy(active, function, threads, shared);
    };
    return functions;
}
struct InputGradientOptions {
    bool is_extended{}, has_version{true};
};
struct InputGradientOwner {
    std::unique_ptr<CudaExecutionSession> session;
    CudaSessionGemmEx problem;
    std::array<CudaSessionResult, 3> matrices;
    CudaSessionResult scalars;
    bool is_extended{};

    explicit InputGradientOwner(InputGradientOptions options) : is_extended(options.is_extended) {
        state = {};
        state.is_sgemm = true;
        input_gradient = {};
        auto functions = input_gradient_dispatch();
        if (!options.has_version) functions.blas.get_version = nullptr;
        session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{91, 92}, 93,
                                                         functions, CudaSessionLimits{},
                                                         CudaSessionPersistentConfig{320, true});
        success(session->initialize(0));
        const auto handle = session->create_blas();
        blas_success(handle);
        for (std::size_t index = 0; index < state.gemm_argument_queries.size(); ++index)
            expect(state.gemm_argument_queries[index] == (index == 24  ? 17U
                                                          : index < 12 ? 16U
                                                                       : 20U),
                   "all twenty-four old ABIs and the new seventeen-argument ABI are checked");
        // Extra leading-dimension space permits independent nonpacked controls.
        constexpr std::array<std::size_t, 3> extents{std::size_t{769} * 3072 * 4 + 128,
                                                     std::size_t{3073} * 768 * 4 + 128,
                                                     std::size_t{769} * 768 * 4 + 128};
        for (std::size_t index = 0; index < matrices.size(); ++index) {
            matrices[index] = session->allocate(extents[index]);
            success(matrices[index]);
        }
        scalars = session->allocate(16);
        success(scalars);
        problem.operation = {.handle = handle.id,
                             .rows = 768,
                             .columns = 768,
                             .depth = 3072,
                             .leading_a = 768,
                             .leading_b = 3072,
                             .leading_c = 768,
                             .a = {matrices[0].id, 64},
                             .b = {matrices[1].id, 64},
                             .c = {matrices[2].id, 64}};
        coefficients({1.0F, 0.0F});
        input_gradient.session = session.get();
        input_gradient.problem = &problem;
        input_gradient.is_extended = is_extended;
    }
    void coefficients(std::array<float, 2> values) {
        problem.operation.alpha = values[0];
        problem.operation.beta = values[1];
        std::memcpy(problem.alpha.data(), &values[0], sizeof(float));
        std::memcpy(problem.beta.data(), &values[1], sizeof(float));
    }
    ExecutionAcceptance accept() {
        return is_extended ? session->accept_gemm_ex(1, problem)
                           : session->accept_sgemm(1, problem.operation);
    }
    void close() {
        input_gradient.is_armed = false;
        success(session->shutdown());
        expect(state.memory.empty() && state.blas_destroys == 1 && state.context_destroys == 1,
               "input-gradient owners retire once after actual physical completion");
        input_gradient.session = nullptr;
        input_gradient.problem = nullptr;
    }
};
template <class Value>
Value input_gradient_argument(std::size_t offset) {
    Value value{};
    std::memcpy(&value, state.gemm_parameters.data() + offset, sizeof(value));
    return value;
}
void owned_input_gradient_split_k_and_resume() {
    if (cuda_sgemm_image().empty()) return;
    for (const bool extended : {false, true}) {
        InputGradientOwner owner({.is_extended = extended});
        // GemmEx's raw coefficient bytes, not the convenience SGEMM floats,
        // govern both profile selection and the immutable copied launch.
        if (extended) {
            owner.problem.operation.alpha = 99.0F;
            owner.problem.operation.beta = 99.0F;
        }
        const auto accepted = owner.accept();
        expect(accepted.accepted() && input_gradient.version_queries == 1 && !state.launches &&
                   !state.blas_issues && !state.ex_issues,
               "both original APIs admit the pinned profile without issuing either library");
        owner.coefficients({99.0F, 99.0F});
        owner.problem.operation.depth = 1;
        state.claimed = 4;
        const auto issued = owner.session->issue_next();
        expect(issued.native_status == CUDA_SUCCESS && !issued.library_status &&
                   state.launches == 1 && state.gemm_variant == 24 && !state.blas_issues &&
                   !state.ex_issues,
               "input-gradient replacement uses only its dedicated private entry");
        for (std::size_t index = 0; index < owner.matrices.size(); ++index)
            expect(input_gradient_argument<CUdeviceptr>(index * 8) ==
                       owner.matrices[index].address + 64,
                   "three matrix offsets remain bound to their exact original allocation");
        expect(input_gradient_argument<int>(24) == 768 && input_gradient_argument<int>(28) == 768 &&
                   input_gradient_argument<int>(32) == 3072 &&
                   input_gradient_argument<int>(36) == 768 &&
                   input_gradient_argument<int>(40) == 3072 &&
                   input_gradient_argument<int>(44) == 768 &&
                   input_gradient_argument<float>(48) == 1.0F &&
                   input_gradient_argument<float>(52) == 0.0F &&
                   input_gradient_argument<CUdeviceptr>(56) == 0 &&
                   input_gradient_argument<CUdeviceptr>(64) == 0 &&
                   input_gradient_argument<std::uint32_t>(72) == 0 &&
                   input_gradient_argument<std::uint32_t>(76) == 0 &&
                   input_gradient_argument<CUdeviceptr>(80) != 0 &&
                   input_gradient_argument<CUdeviceptr>(88) != 0 &&
                   input_gradient_argument<std::uint32_t>(96) == 768 &&
                   input_gradient_argument<std::uint32_t>(100) == 0,
               "all seventeen fields and both padding gaps preserve the 104-byte immutable ABI");
        expect(owner.session->acknowledge_result(accepted.receipt) &&
                   owner.session->release(owner.matrices[0].id).disposition ==
                       CudaSessionDisposition::kDeferred &&
                   owner.session->destroy_blas(owner.problem.operation.handle).disposition ==
                       CudaSessionDisposition::kDeferred,
               "host ACK cannot release original matrices or handle before physical completion");
        expect(owner.session->request_retreat(1) == CUDA_SUCCESS, "whole-tile retreat request");
        state.query_status = CUDA_SUCCESS;
        expect(owner.session->poll().paused == 1, "incomplete input-gradient stops physically");
        const auto progress = owner.session->persistent_progress(1);
        expect(progress && progress->completed_blocks == 4 && progress->total_blocks == 144 &&
                   progress->physical_stops == 1,
               "four complete output tiles, not four partial K partitions, were completed");
        state.claimed = 144;
        const auto resumed = owner.session->resume(1);
        expect(resumed.native_status == CUDA_SUCCESS && !resumed.library_status &&
                   state.counter_before_launch == std::vector<std::uint64_t>{0, 4} &&
                   state.prefix_copies == 1 && input_gradient.version_queries == 1 &&
                   input_gradient_argument<std::uint32_t>(96) == 768,
               "resume preserves partition depth and claims without metadata replay or zeroing");
        expect(owner.session->poll().retired == 1 && !state.blas_issues && !state.ex_issues,
               "completed replacement never replays either original API");
        owner.close();
    }
}
void owned_input_gradient_version_and_existing_routes() {
    if (cuda_sgemm_image().empty()) return;
    for (const bool extended : {false, true}) {
        for (unsigned int mode = 0; mode < 5; ++mode) {
            InputGradientOwner owner({.is_extended = extended, .has_version = mode != 0});
            if (mode == 1) input_gradient.version_status = CUBLAS_STATUS_INTERNAL_ERROR;
            if (mode == 2) input_gradient.version = 120605;
            if (mode == 3) input_gradient.version = 0;
            if (mode == 4) input_gradient.version = -1;
            const auto accepted = owner.accept();
            expect(accepted.accepted() && !owner.session->persistent_progress(1) &&
                       !state.launches && input_gradient.version_queries == (mode == 0 ? 0U : 1U) &&
                       !state.blas_issues && !state.ex_issues,
                   "unqualified private version selects original without compute provenance");
            state.blas_issue_status = CUBLAS_STATUS_NOT_SUPPORTED;
            const auto issued = owner.session->issue_next();
            expect(issued.library_status == CUBLAS_STATUS_NOT_SUPPORTED && !state.launches &&
                       state.blas_issues == (extended ? 0U : 1U) &&
                       state.ex_issues == (extended ? 1U : 0U),
                   "version failure cannot replace the exact original compute status or API");
            expect(owner.session->acknowledge_result(accepted.receipt), "original compute receipt");
            state.query_status = CUDA_SUCCESS;
            expect(owner.session->poll().retired == 1, "original error needs physical retirement");
            owner.close();
        }
        for (unsigned int mode = 0; mode < 10; ++mode) {
            InputGradientOwner owner({.is_extended = extended});
            switch (mode) {
                case 0:
                    owner.coefficients({2.0F, 0.0F});
                    break;
                case 1:
                    owner.coefficients({1.0F, 0.5F});
                    break;
                case 2:
                    owner.problem.operation.leading_a = 769;
                    break;
                case 3:
                    owner.problem.operation.depth = 3071;
                    break;
                case 4:
                    owner.problem.operation.transpose_a = CUBLAS_OP_T;
                    owner.problem.operation.leading_a = 3072;
                    break;
                case 5:
                    owner.problem.operation.transpose_b = CUBLAS_OP_T;
                    owner.problem.operation.leading_b = 768;
                    break;
                case 6:
                    blas_success(owner.session->set_blas_attribute(owner.problem.operation.handle,
                                                                   CudaBlasAttribute::kPointerMode,
                                                                   CUBLAS_POINTER_MODE_DEVICE));
                    owner.problem.operation.uses_device_scalars = true;
                    owner.problem.operation.device_alpha = {owner.scalars.id, 0};
                    owner.problem.operation.device_beta = {owner.scalars.id, 4};
                    break;
                case 7:
                    blas_success(owner.session->set_blas_attribute(owner.problem.operation.handle,
                                                                   CudaBlasAttribute::kMathMode,
                                                                   CUBLAS_PEDANTIC_MATH));
                    break;
                case 8:
                    owner.problem.operation.rows = 767;
                    break;
                case 9:
                    owner.problem.operation.leading_c = 769;
                    break;
                default:
                    expect(false, "enumerated existing routing control");
            }
            const auto accepted = owner.accept();
            expect(accepted.accepted() && input_gradient.version_queries == 0,
                   "noncandidate profiles do not acquire a new SDK-version dependency");
            state.claimed = 144;
            const auto issued = owner.session->issue_next();
            const bool native = mode == 7;
            expect(
                issued.native_status == CUDA_SUCCESS &&
                    (native
                         ? issued.library_status == CUBLAS_STATUS_SUCCESS && state.launches == 0 &&
                               state.blas_issues + state.ex_issues == 1
                         : !issued.library_status && state.launches == 1 &&
                               state.gemm_variant < 24 && !state.blas_issues && !state.ex_issues),
                "outside the exact candidate preserves its prior native or ordinary-private route");
            expect(owner.session->acknowledge_result(accepted.receipt), "unchanged route receipt");
            state.query_status = CUDA_SUCCESS;
            expect(owner.session->poll().retired == 1, "unchanged route physical retirement");
            owner.close();
        }
        {
            InputGradientOwner owner({.is_extended = extended});
            const CudaSessionStridedGemmEx batch{
                owner.problem,
                {std::int64_t{768} * 3072, std::int64_t{3072} * 768, std::int64_t{768} * 768},
                1};
            const auto accepted = extended ? owner.session->accept_strided_gemm_ex(1, batch)
                                           : owner.session->accept_strided_sgemm(1, batch);
            expect(accepted.accepted() && input_gradient.version_queries == 0,
                   "single-batch APIs are not silently converted into the nonbatched candidate");
            state.claimed = 144;
            const auto issued = owner.session->issue_next();
            expect(issued.native_status == CUDA_SUCCESS && !issued.library_status &&
                       state.gemm_variant == 12 && !state.batch_issues &&
                       owner.session->acknowledge_result(accepted.receipt),
                   "batch routing retains its original twenty-argument private entry");
            state.query_status = CUDA_SUCCESS;
            expect(owner.session->poll().retired == 1, "single-batch physical retirement");
            owner.close();
        }
    }
    for (unsigned int mode = 0; mode < 4; ++mode) {
        InputGradientOwner owner({.is_extended = true});
        if (mode == 0)
            owner.problem.a_type = owner.problem.b_type = owner.problem.c_type = CUDA_R_16F;
        if (mode == 1) owner.problem.compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
        if (mode == 2) owner.problem.algorithm = CUBLAS_GEMM_ALGO3;
        if (mode == 3) owner.problem.algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        const auto accepted = owner.accept();
        expect(accepted.accepted() && input_gradient.version_queries == (mode == 3 ? 1U : 0U),
               "extended type, compute and algorithm fields independently bound the version gate");
        state.claimed = 144;
        const auto issued = owner.session->issue_next();
        const bool native = mode == 1 || mode == 2;
        expect(issued.native_status == CUDA_SUCCESS &&
                   (native ? issued.library_status == CUBLAS_STATUS_SUCCESS &&
                                 state.ex_issues == 1 && !state.launches
                           : !issued.library_status && state.launches == 1 && !state.ex_issues &&
                                 state.gemm_variant == (mode == 0 ? 4U : 24U)),
               "FP16 and explicit algorithms retain prior routing while default tensor-op is "
               "eligible");
        expect(owner.session->acknowledge_result(accepted.receipt), "typed routing receipt");
        state.query_status = CUDA_SUCCESS;
        expect(owner.session->poll().retired == 1, "typed routing physical retirement");
        owner.close();
    }
}
void owned_input_gradient_cancellation_abi_and_callbacks() {
    if (cuda_sgemm_image().empty()) return;
    for (const bool extended : {false, true}) {
        {
            InputGradientOwner owner({.is_extended = extended});
            const auto accepted = owner.accept();
            expect(
                accepted.accepted() && owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
                "queued input-gradient can be canceled before any launch");
            const auto result = owner.session->host_result(accepted.receipt);
            expect(result && result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       !state.launches && !state.prefix_copies && !state.blas_issues &&
                       !state.ex_issues && owner.session->acknowledge_result(accepted.receipt),
                   "cancellation has no original compute provenance or private activity");
            owner.close();
        }
        for (const auto callback :
             {InputGradientCallback::kContext, InputGradientCallback::kVersion,
              InputGradientCallback::kOccupancy}) {
            for (const bool loss : {false, true}) {
                InputGradientOwner owner({.is_extended = extended});
                input_gradient.callback = callback;
                input_gradient.loses_context = loss;
                input_gradient.is_armed = true;
                const auto accepted = owner.accept();
                expect(
                    input_gradient.probes == 1 && accepted.accepted() != loss && !state.launches &&
                        !state.blas_issues && !state.ex_issues,
                    "every private callback is excluded and context loss prevents queue admission");
                if (!loss) {
                    expect(owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
                               owner.session->acknowledge_result(accepted.receipt),
                           "guarded callback leaves only the outer accepted command");
                } else {
                    expect(owner.session->snapshot().submissions.outstanding == 0,
                           "lost context cannot leave an admitted or abandoned successor");
                    state.current = native_context();
                }
                owner.close();
            }
        }
    }
    for (unsigned int mode : {1U, 2U}) {
        state = {};
        state.is_sgemm = true;
        state.bad_split_gemm_abi = mode;
        input_gradient = {};
        CudaExecutionSession session({91, 92}, 93, input_gradient_dispatch(), {}, {320, true});
        success(session.initialize(0));
        const auto result = session.create_blas();
        expect(
            result.status == CUDA_ERROR_INVALID_VALUE && !result.original_status && !result.id &&
                state.gemm_argument_queries[24] == 17 && !state.blas_creates && !state.launches,
            "new trailing argument offset and width are validated before original handle creation");
        success(session.shutdown());
        expect(state.memory.empty() && state.gemm_loads == 1 && !state.blas_destroys &&
                   !state.unloads && state.context_destroys == 1,
               "failed provider preparation retires through known owned-context destruction");
    }
}

void owned_gemm_ex_raw_scalars_and_native_status() {
    for (unsigned int format = 0; format < 4; ++format) {
        for (const bool device : {false, true}) {
            state = {};
            const std::array types{CUDA_R_16F, CUDA_R_32F, CUDA_R_64F, CUDA_C_64F};
            const std::array computes{CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F_PEDANTIC,
                                      CUBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F};
            const std::array<std::size_t, 4> widths{2, 4, 8, 16};
            state.ex_scalar_bytes = widths[format];
            CudaExecutionSession session({81, 82}, 83, blas_dispatch());
            success(session.initialize(0));
            const auto handle = session.create_blas();
            blas_success(handle);
            const auto a = session.allocate(256), b = session.allocate(256),
                       c = session.allocate(256), scalars = session.allocate(64);
            success(a);
            success(b);
            success(c);
            success(scalars);
            if (device)
                blas_success(
                    session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 1));
            CudaSessionGemmEx problem{
                .operation = {.handle = handle.id,
                              .rows = 2,
                              .columns = 2,
                              .depth = 2,
                              .leading_a = 2,
                              .leading_b = 2,
                              .leading_c = 2,
                              .a = {a.id, 0},
                              .b = {b.id, 0},
                              .c = {c.id, 0},
                              .uses_device_scalars = device,
                              .device_alpha = device ? CudaAllocationRange{scalars.id, 8}
                                                     : CudaAllocationRange{},
                              .device_beta = device ? CudaAllocationRange{scalars.id, 32}
                                                    : CudaAllocationRange{}},
                .a_type = types[format],
                .b_type = types[format],
                .c_type = types[format],
                .compute_type = computes[format],
                .algorithm = CUBLAS_GEMM_ALGO3};
            std::array<std::byte, 16> av{}, bv{};
            for (std::size_t index = 0; index < widths[format]; ++index) {
                av[index] = static_cast<std::byte>(17 + index);
                bv[index] = static_cast<std::byte>(65 + index);
            }
            problem.alpha = av;
            problem.beta = bv;
            auto invalid = problem;
            invalid.operation.a.offset = 1;
            expect(!session.accept_gemm_ex(1, invalid).accepted(),
                   "typed matrix alignment is independently validated");
            if (device) {
                invalid = problem;
                invalid.operation.device_alpha.offset = 64 - widths[format] + 1;
                expect(!session.accept_gemm_ex(1, invalid).accepted(),
                       "full typed device scalar extent is checked");
            }
            const auto accepted = session.accept_gemm_ex(1, problem);
            expect(accepted.accepted() && state.ex_issues == 0,
                   "owned extended acceptance never calls or substitutes a native function");
            problem.alpha.fill(std::byte{99});
            problem.beta.fill(std::byte{99});
            if (device) {
                std::memcpy(state.memory.at(scalars.address).data() + 8, av.data(), widths[format]);
                std::memcpy(state.memory.at(scalars.address).data() + 32, bv.data(),
                            widths[format]);
            }
            state.blas_issue_status = CUBLAS_STATUS_EXECUTION_FAILED;
            const auto issued = session.issue_next();
            expect(issued.native_status == CUDA_SUCCESS &&
                       issued.library_status == CUBLAS_STATUS_EXECUTION_FAILED &&
                       state.ex_issues == 1 && state.blas_issues == 0 && state.ex_alpha == av &&
                       state.ex_beta == bv && state.ex_input == types[format] &&
                       state.ex_output == types[format] && state.ex_compute == computes[format] &&
                       state.ex_algorithm == CUBLAS_GEMM_ALGO3,
                   "actual GemmEx preserves raw 2/4/8/16-byte coefficients, types, algorithm and "
                   "original error domain");
            expect(
                session.host_result(accepted.receipt)->status == CUBLAS_STATUS_EXECUTION_FAILED &&
                    session.acknowledge_result(accepted.receipt) &&
                    session.release(a.id).disposition == CudaSessionDisposition::kDeferred,
                "original library error and host ACK do not retire physical matrix/scalar leases");
            state.query_status = CUDA_SUCCESS;
            expect(session.poll().retired == 1 && state.ex_issues == 1,
                   "true event stop retires once without a hidden retry");
            success(session.shutdown());
        }
    }
}

void owned_persistent_gemm_ex_selection_and_resume() {
    if (cuda_sgemm_image().empty()) return;
    for (unsigned int format = 0; format < 3; ++format) {
        for (unsigned int mode = 0; mode < 18; ++mode) {
            state = {};
            state.is_sgemm = true;
            CudaExecutionSession session({71, 72}, 73, blas_dispatch(), {}, {320, true});
            success(session.initialize(0));
            const auto handle = session.create_blas();
            blas_success(handle);
            const bool device = mode >= 9;
            const auto in_type = format ? CUDA_R_16F : CUDA_R_32F;
            const auto out_type = format == 1 ? CUDA_R_16F : CUDA_R_32F;
            const auto a = session.allocate(std::size_t{72} * 72 * (format ? 2 : 4));
            const auto b = session.allocate(std::size_t{72} * 72 * (format ? 2 : 4));
            const auto c = session.allocate(std::size_t{72} * 72 * (format == 1 ? 2 : 4));
            const auto scalars = session.allocate(16);
            success(a);
            success(b);
            success(c);
            success(scalars);
            if (device)
                blas_success(
                    session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 1));
            CudaSessionGemmEx problem{
                .operation = {.handle = handle.id,
                              .transpose_a = static_cast<cublasOperation_t>(mode % 3),
                              .transpose_b = static_cast<cublasOperation_t>((mode % 9) / 3),
                              .rows = 65,
                              .columns = 65,
                              .depth = 9,
                              .leading_a = 72,
                              .leading_b = 72,
                              .leading_c = 72,
                              .a = {a.id, 0},
                              .b = {b.id, 0},
                              .c = {c.id, 0},
                              .uses_device_scalars = device,
                              .device_alpha = device ? CudaAllocationRange{scalars.id, 0}
                                                     : CudaAllocationRange{},
                              .device_beta = device ? CudaAllocationRange{scalars.id, 4}
                                                    : CudaAllocationRange{}},
                .a_type = in_type,
                .b_type = in_type,
                .c_type = out_type};
            const float av = 2.0F, bv = 0.5F;
            std::memcpy(problem.alpha.data(), &av, 4);
            std::memcpy(problem.beta.data(), &bv, 4);
            const auto accepted = session.accept_gemm_ex(1, problem);
            expect(accepted.accepted(), "typed persistent acceptance");
            problem.alpha.fill(std::byte{99});
            state.claimed = 1;
            const auto issued = session.issue_next();
            const auto expected_variant = (format == 0   ? 0U
                                           : format == 1 ? 4U
                                                         : 8U) +
                                          (mode % 3 ? 2U : 0U) + ((mode % 9) / 3 ? 1U : 0U);
            float copied{};
            std::memcpy(&copied, state.gemm_parameters.data() + 48, 4);
            expect(issued.native_status == CUDA_SUCCESS && !issued.library_status &&
                       state.ex_issues == 0 && state.blas_issues == 0 && copied == 2.0F &&
                       state.gemm_variant == expected_variant,
                   "actual typed Driver replacement is immutable, correctly selected and never an "
                   "original library result");
            expect(session.acknowledge_result(accepted.receipt) &&
                       session.request_retreat(1) == CUDA_SUCCESS,
                   "one original reply and bounded retreat");
            state.query_status = CUDA_SUCCESS;
            expect(
                session.poll().paused == 1 && session.persistent_progress(1)->completed_blocks == 1,
                "typed incomplete physical stop");
            state.claimed = 4;
            expect(session.resume(1).native_status == CUDA_SUCCESS && session.poll().retired == 1 &&
                       state.counter_before_launch == std::vector<std::uint64_t>{0, 1} &&
                       state.prefix_copies == 1 && state.ex_issues == 0,
                   "typed continuation retains original progress without beta-C replay or native "
                   "substitution");
            success(session.shutdown());
        }
    }
}

void owned_strided_gemm_ex_native_hulls_and_status(bool uses_classic_api = false) {
    for (unsigned int format = 0; format < 4; ++format) {
        if (uses_classic_api && format != 1) continue;
        for (unsigned int geometry = 0; geometry < 4; ++geometry) {
            for (const bool device : {false, true}) {
                state = {};
                const std::array types{CUDA_R_16F, CUDA_R_32F, CUDA_R_64F, CUDA_C_64F};
                std::array computes{CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F_PEDANTIC,
                                    CUBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F};
                if (uses_classic_api) computes[1] = CUBLAS_COMPUTE_32F;
                const std::array<std::size_t, 4> widths{2, 4, 8, 16};
                state.ex_scalar_bytes = widths[format];
                CudaExecutionSession session({91, 92}, 93, blas_dispatch());
                success(session.initialize(0));
                const auto handle = session.create_blas();
                blas_success(handle);
                const auto accept = [&](const auto& problem) {
                    return uses_classic_api ? session.accept_strided_sgemm(1, problem)
                                            : session.accept_strided_gemm_ex(1, problem);
                };
                const auto a = session.allocate(512), b = session.allocate(512),
                           c = session.allocate(512), scalars = session.allocate(64);
                success(a);
                success(b);
                success(c);
                success(scalars);
                if (device)
                    blas_success(
                        session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 1));
                const std::size_t start = geometry == 2 ? 256 : 0;
                CudaSessionStridedGemmEx problem{
                    .operation =
                        {.operation = {.handle = handle.id,
                                       .rows = 2,
                                       .columns = 2,
                                       .depth = 2,
                                       .leading_a = 2,
                                       .leading_b = 2,
                                       .leading_c = 2,
                                       .a = {a.id, start},
                                       .b = {b.id, start},
                                       .c = {c.id, start},
                                       .uses_device_scalars = device,
                                       .device_alpha = device ? CudaAllocationRange{scalars.id, 8}
                                                              : CudaAllocationRange{},
                                       .device_beta = device ? CudaAllocationRange{scalars.id, 32}
                                                             : CudaAllocationRange{}},
                         .a_type = types[format],
                         .b_type = types[format],
                         .c_type = types[format],
                         .compute_type = computes[format],
                         .algorithm = uses_classic_api ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_ALGO3},
                    .strides = geometry == 2   ? std::array<std::int64_t, 3>{-8, -8, -8}
                               : geometry == 1 ? std::array<std::int64_t, 3>{0, 8, 8}
                               : geometry == 3 ? std::array<std::int64_t, 3>{8, 8, 0}
                                               : std::array<std::int64_t, 3>{8, 8, 8},
                    .batches = 3};
                std::array<std::byte, 16> av{}, bv{};
                for (std::size_t index = 0; index < widths[format]; ++index) {
                    av[index] = static_cast<std::byte>(17 + index);
                    bv[index] = static_cast<std::byte>(65 + index);
                }
                problem.operation.alpha = av;
                problem.operation.beta = bv;
                if (device) {
                    success(session.upload({scalars.id, 8}, std::span{av}.first(widths[format])));
                    success(session.upload({scalars.id, 32}, std::span{bv}.first(widths[format])));
                }
                auto invalid = problem;
                invalid.strides[0] = std::numeric_limits<std::int64_t>::max();
                expect(!accept(invalid).accepted(),
                       "batch span overflow must reject before native issue");
                invalid.strides[0] = std::numeric_limits<std::int64_t>::min();
                expect(!accept(invalid).accepted(),
                       "reverse INT64_MIN must reject without signed overflow");
                invalid = problem;
                invalid.batches = 0;
                expect(!accept(invalid).accepted(),
                       "zero batch is explicitly outside this positive batch profile");
                const auto expected_strides = problem.strides;
                if (uses_classic_api) {
                    invalid = problem;
                    invalid.operation.a_type = CUDA_R_16F;
                    expect(!accept(invalid).accepted(),
                           "classic API cannot accept a forged non-FP32 header");
                }
                const auto accepted = accept(problem);
                if (geometry == 2) {
                    expect(!accepted.accepted() && state.batch_issues == 0,
                           "known-unsafe real-SDK reverse strides cannot enter the shared native "
                           "owner");
                    success(session.shutdown());
                    continue;
                }
                expect(accepted.accepted() && state.batch_issues == 0,
                       "batch acceptance copies bytes, not SDK success");
                problem.operation.alpha.fill(std::byte{99});
                problem.strides.fill(99);
                problem.batches = 99;
                state.blas_issue_status = CUBLAS_STATUS_EXECUTION_FAILED;
                const auto issue = session.issue_next();
                expect(issue.native_status == CUDA_SUCCESS &&
                           issue.library_status == CUBLAS_STATUS_EXECUTION_FAILED &&
                           state.batch_issues == 1 && state.ex_issues == 0 &&
                           state.blas_issues == 0 && state.batch_strides == expected_strides &&
                           state.batches == 3 && state.ex_alpha == av && state.ex_beta == bv &&
                           state.ex_input == types[format] &&
                           state.ex_compute == computes[format] &&
                           state.ex_algorithm ==
                               (uses_classic_api ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_ALGO3) &&
                           state.classic_batch_issues == (uses_classic_api ? 1U : 0U) &&
                           state.blas_addresses == std::array<CUdeviceptr, 3>{a.address + start,
                                                                              b.address + start,
                                                                              c.address + start},
                       "original strided callback preserves signed element strides, raw "
                       "coefficients, starts, batch and status");
                expect(session.acknowledge_result(accepted.receipt) &&
                           session.release(c.id).disposition == CudaSessionDisposition::kDeferred,
                       "host error or ACK cannot release a matrix still physically leased");
                state.query_status = CUDA_SUCCESS;
                expect(session.poll().retired == 1 && state.batch_issues == 1,
                       "actual stop retires batch once, without retry");
                success(session.shutdown());
            }
        }
    }
}

void owned_strided_gemm_ex_persistent_mapping_and_resume(bool uses_classic_api = false) {
    if (cuda_sgemm_image().empty()) return;
    for (unsigned int format = 0; format < 3; ++format) {
        if (uses_classic_api && format != 0) continue;
        for (unsigned int mode = 0; mode < 18; ++mode) {
            for (unsigned int geometry = 0; geometry < 7; ++geometry) {
                state = {};
                state.is_sgemm = true;
                CudaExecutionSession session({101, 102}, 103, blas_dispatch(), {}, {320, true});
                success(session.initialize(0));
                const auto handle = session.create_blas();
                blas_success(handle);
                constexpr std::int64_t stride = std::int64_t{72} * 72;
                const std::size_t input_width = format ? 2 : 4, output_width = format == 1 ? 2 : 4;
                const bool broadcasts_a = geometry == 1 || geometry == 6;
                const bool broadcasts_b = geometry == 5 || geometry == 6;
                const auto a = session.allocate((broadcasts_a ? 1 : 3) * stride * input_width),
                           b = session.allocate((broadcasts_b ? 1 : 3) * stride * input_width),
                           c = session.allocate(3 * stride * output_width),
                           scalars = session.allocate(16);
                success(a);
                success(b);
                success(c);
                success(scalars);
                const bool device = mode >= 9;
                if (device)
                    blas_success(
                        session.set_blas_attribute(handle.id, CudaBlasAttribute::kPointerMode, 1));
                CudaSessionStridedGemmEx problem{
                    .operation =
                        {.operation = {.handle = handle.id,
                                       .transpose_a = static_cast<cublasOperation_t>(mode % 3),
                                       .transpose_b =
                                           static_cast<cublasOperation_t>((mode % 9) / 3),
                                       .rows = 65,
                                       .columns = 65,
                                       .depth = 9,
                                       .leading_a = 72,
                                       .leading_b = 72,
                                       .leading_c = 72,
                                       .a = {a.id, geometry == 2 ? 2 * stride * input_width : 0},
                                       .b = {b.id, geometry == 2 ? 2 * stride * input_width : 0},
                                       .c = {c.id, geometry == 2 ? 2 * stride * output_width : 0},
                                       .uses_device_scalars = device,
                                       .device_alpha = device ? CudaAllocationRange{scalars.id, 0}
                                                              : CudaAllocationRange{},
                                       .device_beta = device ? CudaAllocationRange{scalars.id, 4}
                                                             : CudaAllocationRange{}},
                         .a_type = format ? CUDA_R_16F : CUDA_R_32F,
                         .b_type = format ? CUDA_R_16F : CUDA_R_32F,
                         .c_type = format == 1 ? CUDA_R_16F : CUDA_R_32F},
                    .strides =
                        geometry == 2   ? std::array<std::int64_t, 3>{-stride, -stride, -stride}
                        : geometry == 1 ? std::array<std::int64_t, 3>{0, stride, stride}
                                        : std::array<std::int64_t, 3>{stride, stride, stride},
                    .batches = 3};
                const float alpha = 2, beta = 0.5F;
                std::memcpy(problem.operation.alpha.data(), &alpha, 4);
                std::memcpy(problem.operation.beta.data(), &beta, 4);
                if (broadcasts_a) problem.strides[0] = 0;
                if (broadcasts_b) problem.strides[1] = 0;
                if (geometry == 3) problem.strides[2] = 0;
                if (geometry == 4) {
                    problem.batches = 1;
                    problem.strides.fill(std::numeric_limits<std::int64_t>::min());
                    if (uses_classic_api)
                        blas_success(session.set_blas_attribute(
                            handle.id, CudaBlasAttribute::kSmCountTarget, 7));
                    else
                        problem.operation.algorithm = CUBLAS_GEMM_ALGO3;
                }
                const auto accepted = uses_classic_api ? session.accept_strided_sgemm(1, problem)
                                                       : session.accept_strided_gemm_ex(1, problem);
                if (geometry == 2) {
                    expect(!accepted.accepted() && state.batch_issues == 0 && state.launches == 0,
                           "reject unqualified reverse-stride compute without native or private "
                           "issue");
                    success(session.shutdown());
                    continue;
                }
                expect(accepted.accepted(), "private batch acceptance");
                if (geometry == 3 || geometry == 4) {
                    const auto original = session.issue_next();
                    expect(original.native_status == CUDA_SUCCESS &&
                               original.library_status == CUBLAS_STATUS_SUCCESS &&
                               state.batch_issues == 1 && state.ex_issues == 0 &&
                               state.classic_batch_issues == (uses_classic_api ? 1U : 0U) &&
                               state.launches == 0 && state.batch_strides == problem.strides &&
                               state.batches == problem.batches,
                           "overlapping outputs or original algorithm select the actual strided "
                           "callback, even for one batch with unused huge strides");
                    expect(session.acknowledge_result(accepted.receipt), "original batch ACK");
                    state.query_status = CUDA_SUCCESS;
                    expect(session.poll().retired == 1, "actual original batch retirement");
                    success(session.shutdown());
                    continue;
                }
                state.claimed = 5;
                const auto issue = session.issue_next();
                std::array<std::int64_t, 3> copied{};
                int batches{};
                std::memcpy(copied.data(), state.gemm_parameters.data() + 96, 24);
                std::memcpy(&batches, state.gemm_parameters.data() + 120, 4);
                const auto variant = 12 +
                                     (format == 0   ? 0U
                                      : format == 1 ? 4U
                                                    : 8U) +
                                     (mode % 3 ? 2U : 0U) + ((mode % 9) / 3 ? 1U : 0U);
                expect(issue.native_status == CUDA_SUCCESS && !issue.library_status &&
                           state.gemm_variant == variant && copied == problem.strides &&
                           batches == 3 && state.batch_issues == 0 && state.ex_issues == 0,
                       "batch ABI chooses a checked private variant, not a native call or "
                       "per-batch public replay");
                expect(session.acknowledge_result(accepted.receipt) &&
                           session.request_retreat(1) == CUDA_SUCCESS,
                       "request actual incomplete batch retreat");
                state.query_status = CUDA_SUCCESS;
                expect(session.poll().paused == 1 &&
                           session.persistent_progress(1)->completed_blocks == 5 &&
                           session.persistent_progress(1)->total_blocks == 12,
                       "global batch counter retains progress across matrix boundary");
                expect(session.release(a.id).disposition == CudaSessionDisposition::kDeferred &&
                           session.release(b.id).disposition == CudaSessionDisposition::kDeferred,
                       "broadcast input storage remains owned while incomplete work is paused");
                state.claimed = 12;
                expect(session.resume(1).native_status == CUDA_SUCCESS &&
                           session.poll().retired == 1 &&
                           state.counter_before_launch == std::vector<std::uint64_t>{0, 5} &&
                           state.prefix_copies == 1,
                       "batch continuation never reinitializes claimed tiles or beta-C epilogues");
                success(session.shutdown());
            }
        }
    }
}

void contradictory_blas_metadata_is_not_resource_authority() {
    for (unsigned int mode = 0; mode < 6; ++mode) {
        state = {};
        CudaExecutionSession session({1, 2}, 7, blas_dispatch());
        success(session.initialize(0));
        const auto handle = session.create_blas();
        blas_success(handle);
        auto attribute = CudaBlasAttribute::kStream;
        if (mode == 0) {
            const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
            success(stream);
            blas_success(session.set_blas_stream(handle.id, stream.id));
            state.returns_default_blas_stream = true;
        } else if (mode == 1) {
            attribute = CudaBlasAttribute::kPointerMode;
            state.returns_bad_blas_mode = true;
        } else if (mode == 2) {
            attribute = CudaBlasAttribute::kMathMode;
            state.returns_bad_blas_math = true;
        } else if (mode == 3) {
            attribute = CudaBlasAttribute::kVersion;
            state.returns_negative_blas_version = true;
        } else if (mode == 4) {
            attribute = CudaBlasAttribute::kAtomicsMode;
            state.returns_bad_atomics = true;
        } else {
            attribute = CudaBlasAttribute::kSmCountTarget;
            state.returns_bad_sm_count = true;
        }
        const auto observed = session.get_blas_attribute(handle.id, attribute);
        expect(observed.status == CUDA_ERROR_UNKNOWN &&
                   observed.original_status == CUBLAS_STATUS_SUCCESS &&
                   observed.disposition == CudaSessionDisposition::kRejected &&
                   session.snapshot().is_quarantined &&
                   session.accept_sgemm(1, {}).status == ExecutionAcceptStatus::kQuarantined &&
                   state.blas_issues == 0,
               "contradictory native metadata retains its true SDK result but cannot authorize "
               "further work");
        success(session.shutdown());
        expect(state.blas_destroys == 1 && state.context_destroys == 1,
               "known valid handle/context ownership permits one actual cleanup after metadata "
               "corruption");
    }
}
void owned_blas_partial_failures() {
    for (const unsigned int mode : {0U, 1U, 2U, 3U}) {
        const auto child = fork();
        expect(child >= 0, "fork partial library ownership control");
        if (child == 0) {
            try {
                state = {};
                CudaExecutionSession session({1, 2}, 7, blas_dispatch());
                success(session.initialize(0));
                if (mode < 2) {
                    state.blas_create_status = CUBLAS_STATUS_ALLOC_FAILED;
                    state.returns_partial_blas = mode == 1;
                    const auto result = session.create_blas();
                    expect(result.original_status == CUBLAS_STATUS_ALLOC_FAILED && !result.id,
                           "retain the actual failed native create status");
                    if (!mode) {
                        state.blas_create_status = CUBLAS_STATUS_SUCCESS;
                        blas_success(session.create_blas());
                        success(session.shutdown());
                        expect(state.blas_destroys == 1,
                               "known-empty rejection permits fresh create");
                    } else {
                        expect(session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                   !session.create_blas().original_status &&
                                   state.blas_destroys == 0,
                               "partial failed handle is retained, not guessed valid or replayed");
                    }
                } else {
                    state.returns_null_blas = mode == 2;
                    const auto result = session.create_blas();
                    if (mode == 2) {
                        expect(result.status == CUDA_ERROR_UNKNOWN &&
                                   result.original_status == CUBLAS_STATUS_SUCCESS &&
                                   session.shutdown().status == CUDA_ERROR_UNKNOWN,
                               "successful create with no resource is not published as usable");
                    } else {
                        blas_success(result);
                        state.blas_destroy_status = CUBLAS_STATUS_EXECUTION_FAILED;
                        expect(session.destroy_blas(result.id).original_status ==
                                       CUBLAS_STATUS_EXECUTION_FAILED &&
                                   !session.destroy_blas(result.id).original_status &&
                                   session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                                   state.blas_destroys == 1 && state.context_destroys == 0,
                               "ambiguous library destruction is never retried or hidden by "
                               "context teardown");
                    }
                }
                _exit(0);
            } catch (...) {
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "isolated library partial ownership control passes without pretending leak-free "
               "recovery");
    }
}

void session_owned_persistent_preparation_and_reuse(std::shared_ptr<CudaPtxCache> cache = {},
                                                    bool retargets_workers = false) {
    state = {};
    state.is_persistent = true;
    CudaSessionLimits limits;
    limits.submissions.max_commands = 1;
    CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320}, std::move(cache));
    success(session.initialize(0));
    expect(session.snapshot().device_bytes == 8 && session.snapshot().private_host_bytes == 64,
           "private progress is bounded and charged before user allocation");
    const auto module = session.load_ptx(kPersistentPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    expect(function.arguments.size() == 2 && function.parameter_bytes == 12 &&
               session.snapshot().persistent_functions == 1,
           "only original ABI is exposed while private native ABI is independently prepared");
    const auto calls = state.calls;
    expect(!session.get_function(module.id, "kernel__glimmer_ptb_preemptive").original_status &&
               state.calls == calls + 1,
           "private generated symbols reject before native function lookup");
    const auto memory = session.allocate(4);
    success(memory);
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 17;
    std::memcpy(parameters.data(), &memory.address, 8);
    std::memcpy(parameters.data() + 8, &value, 4);
    CudaExecutionKernel kernel{.function = function.id,
                               .stream = 0,
                               .grid = {8, 1, 1},
                               .block = {32, 1, 1},
                               .parameters = parameters};
    const auto low = session.accept_kernel(1, kernel);
    const auto frontier = session.capture_frontier();
    expect(low.accepted() && frontier.status == CUDA_SUCCESS && state.allocations == 2 &&
               state.prefix_copies == 0 && session.snapshot().persistent_slots == 1,
           "warm admission owns a slot without any allocation or initialization SDK call");
    if (retargets_workers)
        expect(session.retarget_workers(1, {80}) == CUDA_ERROR_NOT_READY,
               "accepted but unissued PTX cannot retarget without physical stop");
    expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
               state.prefix_copies == 1 &&
               state.counter_before_launch == std::vector<std::uint64_t>{0} &&
               session.acknowledge_result(low.receipt) && session.poll().retained == 1,
           "ordered counter zero and one initial SDK result precede real stop");
    if (retargets_workers)
        expect(session.retarget_workers(1, {80}) == CUDA_ERROR_NOT_READY,
               "active PTX cannot retarget from a host result or pending event");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().paused == 1 && session.persistent_progress(1)->completed_blocks == 3 &&
               session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY &&
               session.snapshot().persistent_slots == 1,
           "incomplete real stop retains framework-owned private progress and logical frontier");
    expect(session.request_retreat(1) == CUDA_SUCCESS, "owner can observe already stopped retreat");
    if (retargets_workers) {
        expect(session.retarget_workers(1, {0}) == CUDA_ERROR_INVALID_VALUE &&
                   session.retarget_workers(1, {321}) == CUDA_ERROR_INVALID_VALUE &&
                   session.persistent_progress(1)->workers == 320 &&
                   session.retarget_workers(1, {80}) == CUDA_SUCCESS &&
                   session.persistent_progress(1)->workers == 80 &&
                   session.persistent_progress(1)->completed_blocks == 3 &&
                   state.prefix_copies == 1 && state.launches == 1,
               "bounded stopped retarget changes no logical progress or native execution");
        state.claimed = 5;
        expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
                   session.poll().paused == 1 &&
                   session.persistent_progress(1)->completed_blocks == 5 &&
                   state.counter_before_launch == std::vector<std::uint64_t>{0, 3} &&
                   state.physical_grids.back() == std::array{80U, 1U, 1U} &&
                   session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY &&
                   session.retarget_workers(1, {160}) == CUDA_SUCCESS,
               "new worker interval preserves the original counter and incomplete frontier");
    }
    state.claimed = 8;
    const auto expected_counters =
        retargets_workers ? std::vector<std::uint64_t>{0, 3, 5} : std::vector<std::uint64_t>{0, 3};
    expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
               state.prefix_copies == 1 && state.counter_before_launch == expected_counters &&
               session.poll().retired == 1 && session.snapshot().persistent_slots == 0 &&
               !session.persistent_progress(1) &&
               session.query_frontier(frontier.frontier) == CUDA_SUCCESS,
           "resume preserves counter; final retirement alone returns the private slot");
    const auto next = session.accept_kernel(2, kernel);
    expect(next.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               state.prefix_copies == 2 && state.counter_before_launch.back() == 0 &&
               state.physical_grids.back() == std::array{320U, 1U, 1U} &&
               session.poll().retired == 1 && session.acknowledge_result(next.receipt),
           "new logical work reuses the slot only with fresh same-stream zero initialization");
    expect(session.snapshot().persistent_accepted == 2, "actual persistent selection is counted");
    success(session.shutdown());
    expect(state.host_frees == 1 && state.frees == 1 && !state.host && state.context_destroys == 1,
           "explicit quiescent release frees only private counters/host once; context retires user "
           "memory");
}

void sliced_worker_retarget_preserves_exact_parameter_updates() {
    state = {};
    state.is_persistent = true;
    state.kind = Kind::kSliced;
    state.query_status = CUDA_SUCCESS;
    CudaExecutionSession session({1, 2}, 7, dispatch(), {}, {320, false, Kind::kSliced});
    success(session.initialize(0));
    const auto module = session.load_ptx(kPersistentPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto memory = session.allocate(4);
    success(memory);
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 17;
    std::memcpy(parameters.data(), &memory.address, 8);
    std::memcpy(parameters.data() + 8, &value, 4);
    const auto original = parameters;
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .grid = {1000, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    const auto frontier = session.capture_command_frontier(1);
    expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
               session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt) && session.poll().paused == 1 &&
               session.persistent_progress(1)->completed_blocks == 320 &&
               session.retarget_workers(1, {80}) == CUDA_SUCCESS &&
               session.resume(1).state == CudaExecutionIssueState::kIssued &&
               session.poll().paused == 1 &&
               session.persistent_progress(1)->completed_blocks == 400 &&
               session.retarget_workers(1, {160}) == CUDA_SUCCESS,
           "sliced retarget preserves its exact completed offset across320 and80 worker intervals");
    for (const auto completed : {560U, 720U, 880U}) {
        expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
                   session.poll().paused == 1 &&
                   session.persistent_progress(1)->completed_blocks == completed &&
                   session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
               "retargeted slice advances exactly160 without settling the logical frontier");
    }
    expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
               session.poll().retired == 1 &&
               state.slices ==
                   std::vector<std::array<std::uint32_t, 2>>{
                       {0, 320}, {320, 80}, {400, 160}, {560, 160}, {720, 160}, {880, 120}} &&
               session.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
               !session.host_result(accepted.receipt) && parameters == original &&
               state.prefix_copies == 0 && !session.persistent_progress(1),
           "final clipped slice completes exactly once without ABI mutation or counter reset");
    success(session.shutdown());
}

void cached_preparation_preserves_native_ownership_and_budget() {
    auto cache = std::make_shared<CudaPtxCache>();
    session_owned_persistent_preparation_and_reuse(cache);
    session_owned_persistent_preparation_and_reuse(cache);
    expect(cache->snapshot().lookups == 2 && cache->snapshot().transforms == 1 &&
               cache->snapshot().hits == 1,
           "cached immutable source still traverses actual private ABI, incomplete retreat and "
           "retained resume");
    for (unsigned int mode = 0; mode < 7; ++mode) {
        state = {};
        state.is_persistent = true;
        state.fails_generated_load = mode == 0;
        state.is_bad_private_abi = mode == 5;
        if (mode == 3) {
            state.module_status = CUDA_ERROR_INVALID_PTX;
            state.returns_partial_module = true;
        }
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        if (mode == 2) limits.max_image_bytes = kPersistentPtx.size() + 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320}, cache);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPersistentPtx);
        if (mode == 2) {
            expect(module.status == CUDA_ERROR_OUT_OF_MEMORY && !module.original_status &&
                       state.module_loads == 0 && session.snapshot().modules == 0 &&
                       session.snapshot().image_bytes == 0,
                   "cache hit pays the full session image budget before any native module load");
        } else if (mode == 3) {
            expect(module.original_status == CUDA_ERROR_INVALID_PTX && !module.id &&
                       session.snapshot().is_quarantined && state.module_loads == 1 &&
                       cache->prune() && cache->snapshot().entries == 1,
                   "partial native ownership pins cached bytes and cannot retry or poison the "
                   "source cache");
        } else {
            success(module);
            const auto function = session.get_function(module.id, "kernel");
            if (mode == 5)
                expect(
                    function.disposition == CudaSessionDisposition::kRejected &&
                        function.original_status == CUDA_SUCCESS && !function.id,
                    "warm source cache cannot override contradictory native private ABI metadata");
            else
                success(function);
            expect(
                state.module_loads == (mode == 0 ? 2U : 1U) &&
                    session.snapshot().persistent_functions == (mode == 0 || mode == 5 ? 0U : 1U),
                "JIT errors remain per-context while later exact hits independently validate "
                "their own native ABI");
            success(session.unload_module(module.id));
            expect(session.snapshot().image_bytes == 0,
                   "native unload refunds only this session's complete image charge");
        }
        success(session.shutdown());
    }
    expect(cache->snapshot().transforms == 1 && cache->snapshot().hits == 8 && cache->prune() &&
               cache->snapshot().entries == 0,
           "native failures never become shared unsupported entries or survive proven idle CPU "
           "cleanup");
}

void large_source_policy_preserves_session_ownership() {
    std::string source(glimmer::transform::kMaxPersistentPtxBytes + 1, ' ');
    source += kPersistentPtx;
    auto cache = std::make_shared<CudaPtxCache>();
    for (unsigned int mode = 0; mode < 4; ++mode) {
        state = {};
        state.is_persistent = true;
        state.uses_multiple_kinds = true;
        CudaSessionLimits limits;
        if (mode == 3) limits.max_image_bytes = source.size() + 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320}, cache);
        success(session.initialize(0));
        const auto module = session.load_ptx(source);
        if (mode == 3) {
            expect(module.status == CUDA_ERROR_OUT_OF_MEMORY && !module.original_status &&
                       !module.id && state.module_loads == 0 && !session.snapshot().image_bytes,
                   "a wide-policy cache hit still pays the complete per-session image quota");
        } else {
            success(module);
            expect(
                state.loaded_ptx.starts_with(source) &&
                    state.loaded_ptx.find("kernel__glimmer_ptb_preemptive") != std::string::npos &&
                    session.snapshot().image_bytes == source.size() + state.loaded_ptx.size() + 2,
                "owned large source retains its whole prefix and charges all generated bytes");
            if (mode == 2) state.parameter_status = CUDA_ERROR_INVALID_VALUE;
            const auto function = session.get_function(module.id, "kernel");
            if (mode == 2) {
                expect(function.status == CUDA_ERROR_INVALID_VALUE &&
                           function.original_status == CUDA_SUCCESS && !function.id &&
                           !session.snapshot().functions && !state.launches,
                       "a large preparation cache hit cannot override the native parameter ABI");
            } else {
                success(function);
                expect(
                    function.arguments.size() == 2 && function.parameter_bytes == 12 &&
                        session.snapshot().persistent_functions == 1,
                    "large-source preparation publishes only an independently validated variant");
            }
            success(session.unload_module(module.id));
            expect(!session.snapshot().image_bytes && state.unloads == 1,
                   "large whole-module source and generated charges retire exactly once");
        }
        success(session.shutdown());
        expect(state.context_destroys == 1 && !session.snapshot().has_context,
               "each large-image session retains its own native owner and context fence");
    }
    expect(cache->snapshot().lookups == 4 && cache->snapshot().transforms == 1 &&
               cache->snapshot().hits == 3 && cache->prune() && !cache->snapshot().entries,
           "large CPU preparation is reusable without sharing native handles or swallowing errors");
}

void selected_ptx_families_preserve_logical_completion() {
    auto cache = std::make_shared<CudaPtxCache>();
    for (const auto kind : {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kSliced}) {
        for (const bool bad_abi : {false, true}) {
            state = {};
            state.is_persistent = true;
            state.kind = kind;
            state.is_bad_private_abi = bad_abi;
            state.claimed = 8;
            CudaExecutionSession session({1, 2}, 7, dispatch(), {}, {3, false, kind}, cache);
            success(session.initialize(0));
            const auto module = session.load_ptx(kPersistentPtx);
            success(module);
            const auto function = session.get_function(module.id, "kernel");
            if (bad_abi) {
                expect(function.status == CUDA_ERROR_INVALID_VALUE && !function.id &&
                           state.launches == 0,
                       "every warm cached kind rejects contradictory native ABI before issue");
                success(session.shutdown());
                continue;
            }
            success(function);
            const auto memory = session.allocate(4);
            success(memory);
            std::array<std::byte, 12> parameters{};
            const std::uint32_t value = 17;
            std::memcpy(parameters.data(), &memory.address, 8);
            std::memcpy(parameters.data() + 8, &value, 4);
            const auto original = parameters;
            const auto grid =
                kind == Kind::kStatic ? std::array{8U, 1U, 1U} : std::array{2U, 2U, 2U};
            CudaExecutionKernel kernel{.function = function.id,
                                       .grid = grid,
                                       .block = {32, 1, 1},
                                       .parameters = parameters};
            const auto first = session.accept_kernel(1, kernel);
            const auto frontier = session.capture_frontier();
            expect(first.accepted() && frontier.status == CUDA_SUCCESS &&
                       session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(first.receipt) && session.poll().retained == 1 &&
                       session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
                   "selected private ABI retains physical ownership after its single SDK reply");
            const auto retreat = session.request_retreat(1);
            expect(retreat == (kind == Kind::kSliced ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED),
                   "static/dynamic have no mid-execution retreat; slices stop at their boundary");
            state.query_status = CUDA_SUCCESS;
            if (kind == Kind::kSliced) {
                expect(session.poll().paused == 1 &&
                           session.persistent_progress(1)->completed_blocks == 3 &&
                           session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY &&
                           state.slices == std::vector<std::array<std::uint32_t, 2>>{{0, 3}},
                       "first slice proves only three completed logical CTAs");
                const auto second = session.accept_kernel(2, kernel);
                expect(second.accepted() &&
                           session.issue_next().state == CudaExecutionIssueState::kIdle,
                       "same-stream successor cannot pass an incomplete slice");
                expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
                           session.poll().paused == 1 &&
                           session.persistent_progress(1)->completed_blocks == 6 &&
                           !session.host_result(first.receipt),
                       "middle slice advances once without a second application receipt");
                expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
                           session.poll().retired == 1 &&
                           session.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
                           state.slices ==
                               std::vector<std::array<std::uint32_t, 2>>{{0, 3}, {3, 3}, {6, 2}},
                       "tail slice covers the remainder exactly and alone completes the frontier");
                expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
                           session.acknowledge_result(second.receipt) &&
                           state.slices.back() == std::array<std::uint32_t, 2>{0, 3},
                       "next logical command restarts at zero with its own retained state");
                expect(session.poll().paused == 1 &&
                           session.resume(2).state == CudaExecutionIssueState::kIssued &&
                           session.poll().paused == 1 &&
                           session.resume(2).state == CudaExecutionIssueState::kIssued &&
                           session.poll().retired == 1,
                       "drain second sliced command without reusing the first receipt");
            } else {
                expect(session.poll().retired == 1 &&
                           session.query_frontier(frontier.frontier) == CUDA_SUCCESS,
                       "nonpreemptible variant completes only at its actual event");
            }
            expect(
                parameters == original &&
                    std::ranges::all_of(state.logical_grids,
                                        [&](auto actual) { return actual == grid; }) &&
                    std::ranges::all_of(
                        state.physical_grids,
                        [](auto actual) { return actual == std::array{3U, 1U, 1U}; }) &&
                    state.prefix_copies == (kind == Kind::kDynamic ? 1U : 0U) &&
                    state.readbacks == (kind == Kind::kDynamic ? 1U : 0U),
                "retain original arguments/grid, explicit geometry and family-specific transfers");
            if (kind == Kind::kStatic) {
                kernel.grid = {1, 1, 1};
                auto one = session.accept_kernel(2, kernel);
                expect(one.accepted() &&
                           session.issue_next().state == CudaExecutionIssueState::kIssued &&
                           state.physical_grids.back() == std::array{1U, 1U, 1U} &&
                           session.poll().retired == 1 && session.acknowledge_result(one.receipt),
                       "independent-thread static never launches excess out-of-range workers");
                kernel.grid = {2, 2, 2};
                one = session.accept_kernel(3, kernel);
                expect(one.accepted() && !session.persistent_progress(3) &&
                           session.issue_next().state == CudaExecutionIssueState::kIssued &&
                           state.physical_grids.back() == kernel.grid &&
                           session.poll().retired == 1 && session.acknowledge_result(one.receipt),
                       "multidimensional input selects original before static admission");
            }
            success(session.shutdown());
            expect(state.host_frees == 1 && state.context_destroys == 1 &&
                       session.snapshot().persistent_slots == 0,
                   "all family leases retire before native context cleanup");
        }
    }
    expect(cache->snapshot().transforms == 4 && cache->snapshot().hits == 4,
           "actual family routing keeps exact-kind cached code distinct");
    // A dynamic variant cannot legitimately stop incomplete: do not authorize
    // replay using the cooperative preemptive counter interpretation.
    state = {};
    state.is_persistent = true;
    state.kind = Kind::kDynamic;
    CudaExecutionSession session({1, 2}, 7, dispatch(), {}, {3, false, Kind::kDynamic});
    success(session.initialize(0));
    const auto module = session.load_ptx(kPersistentPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto memory = session.allocate(4);
    success(memory);
    std::array<std::byte, 12> parameters{};
    std::memcpy(parameters.data(), &memory.address, 8);
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .grid = {8, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    expect(accepted.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt),
           "issue dynamic malformed-progress control");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().status == CUDA_ERROR_UNKNOWN && session.snapshot().is_quarantined &&
               session.resume(1).state != CudaExecutionIssueState::kIssued && state.launches == 1,
           "incomplete dynamic counter quarantines without replay");
    success(session.shutdown());
}

void once_only_profile_selects_actual_worker_geometry(Kind kind = Kind::kDynamic) {
    auto profile = std::make_shared<CudaProfileCache>(
        CudaProfileConfig{128, 1U << 20, 0, 1, {2, 4}}, profile_clock);
    for (unsigned int round = 0; round < 3; ++round) {
        state = {};
        state.is_persistent = true;
        state.kind = kind;
        state.claimed = 8;
        state.uuid_tag = round == 2 ? 2 : 1;
        auto sdk = dispatch();
        sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
            *uuid = {};
            uuid->bytes[0] = static_cast<char>(state.uuid_tag);
            return CUDA_SUCCESS;
        };
        sdk.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
        CudaExecutionSession session({1, 2}, 7, sdk, {}, {4, false, kind}, {}, profile);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPersistentPtx);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const auto memory = session.allocate(4);
        success(memory);
        std::array<std::byte, 12> parameters{};
        std::memcpy(parameters.data(), &memory.address, 8);
        const std::uint32_t value = 17;
        std::memcpy(parameters.data() + 8, &value, 4);
        const auto commands = round == 0 ? 4U : 1U;
        for (unsigned int command = 1; command <= commands; ++command) {
            const auto before = profile->snapshot().samples;
            profile_now = 100;
            state.query_status = CUDA_ERROR_NOT_READY;
            const auto accepted = session.accept_kernel(command, {.function = function.id,
                                                                  .grid = {8, 1, 1},
                                                                  .block = {32, 1, 1},
                                                                  .parameters = parameters});
            expect(accepted.accepted() &&
                       session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt) &&
                       session.poll().retained == 1 && profile->snapshot().samples == before,
                   "acceptance and SDK reply are not a completed timing sample");
            profile_now += command == 1 ? 1000 : command == 2 ? 50 : 100;
            state.query_status = CUDA_SUCCESS;
            auto observed = session.poll();
            while (observed.paused) {
                expect(kind == Kind::kSliced && profile->snapshot().samples == before,
                       "intermediate sliced stop must not publish or restart the logical sample");
                expect(session.resume(command).state == CudaExecutionIssueState::kIssued,
                       "profile retains the same observation across slice continuation");
                observed = session.poll();
            }
            expect(observed.retired == 1, "actual final event retirement publishes one sample");
        }
        const auto logical_workers = round == 0   ? std::vector{8U, 2U, 4U, 2U}
                                     : round == 1 ? std::vector{2U}
                                                  : std::vector{8U};
        std::vector<std::array<unsigned int, 3>> expected;
        for (const auto workers : logical_workers)
            for (unsigned int launch = 0; launch < (kind == Kind::kSliced ? 8 / workers : 1);
                 ++launch)
                expected.push_back({workers, 1, 1});
        expect(
            state.physical_grids == expected && state.launches == expected.size(),
            "actual native launches use original, candidates and learned workers without replay");
        success(session.shutdown());
    }
    expect(profile->snapshot().functions == 2 && profile->snapshot().samples == 4 &&
               profile->snapshot().hits == 2,
           "independent sessions reuse exact platform/code measurements, changed GPU retrains");
}
void profile_selects_distinct_native_families() {
    const std::vector kinds{Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive,
                            Kind::kSliced};
    const std::string source =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry kernel(.param .u64 pointer, .param .u32 value) { ret; }\n";
    for (unsigned int mode = 0; mode < 4; ++mode) {
        state = {};
        state.is_persistent = true;
        state.uses_multiple_kinds = true;
        state.claimed = 8;
        if (mode == 1) state.zero_occupancy_kind = Kind::kDynamic;
        if (mode == 2) state.bad_abi_kind = Kind::kDynamic;
        auto sdk = dispatch();
        sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
            *uuid = {};
            uuid->bytes[0] = 1;
            return CUDA_SUCCESS;
        };
        sdk.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
        auto profile = std::make_shared<CudaProfileCache>(
            CudaProfileConfig{128, 1U << 20, 0, 1, {2}, kinds}, profile_clock);
        auto prepared = std::make_shared<CudaPtxCache>();
        CudaSessionLimits limits;
        if (mode == 3)
            limits.max_image_bytes =
                source.size() + glimmer::transform::transform_persistent_ptx(source)->image.size() +
                2;
        CudaExecutionSession session({1, 2}, 7, sdk, limits, {4}, prepared, profile);
        success(session.initialize(0));
        const auto module = session.load_ptx(source);
        if (mode == 3) {
            expect(module.status == CUDA_ERROR_OUT_OF_MEMORY && state.module_loads == 0,
                   "all appended families must fit the image quota before any SDK module load");
            success(session.shutdown());
            continue;
        }
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        if (mode == 2) {
            expect(function.status == CUDA_ERROR_INVALID_VALUE &&
                       session.snapshot().functions == 0 && profile->snapshot().functions == 0,
                   "a later family's invalid native ABI cannot publish a partial function");
            success(session.shutdown());
            continue;
        }
        success(function);
        expect(state.module_loads == 1 && state.loaded_ptx.starts_with(source) &&
                   state.loaded_ptx.find(".entry kernel(") ==
                       state.loaded_ptx.rfind(".entry kernel(") &&
                   session.snapshot().image_bytes == source.size() + state.loaded_ptx.size() + 2 &&
                   state.configured_functions.size() == 6 && prepared->snapshot().transforms == 5,
               "all independently validated native entries share one original module/entry "
               "definition and the full prepared-image charge");
        for (const auto kind : kinds) {
            const auto generated = glimmer::transform::transform_persistent_ptx(source, kind);
            expect(generated &&
                       session.get_function(module.id, generated->kernels.front().persistent_name)
                               .status == CUDA_ERROR_INVALID_VALUE,
                   "every generated family name remains private to the framework");
        }
        const auto memory = session.allocate(4);
        success(memory);
        std::array<std::byte, 12> parameters{};
        std::memcpy(parameters.data(), &memory.address, 8);
        const std::uint32_t value = 19;
        std::memcpy(parameters.data() + 8, &value, 4);
        std::vector<std::optional<Kind>> expected{std::nullopt};
        for (const auto kind : kinds)
            if (!(mode == 1 && (kind == Kind::kStatic || kind == Kind::kDynamic)))
                expected.push_back(kind);
        expected.push_back(Kind::kSliced);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            profile_now = 100;
            state.query_status = CUDA_ERROR_NOT_READY;
            const auto previous = profile->snapshot().samples;
            const auto launch_begin = state.launches;
            const auto command = index + 1;
            const auto accepted = session.accept_kernel(
                command, {.function = function.id,
                          .grid = mode == 1 ? std::array{2U, 2U, 2U} : std::array{8U, 1U, 1U},
                          .block = {32, 1, 1},
                          .parameters = parameters});
            expect(accepted.accepted() &&
                       session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt) && session.poll().retained == 1,
                   "cross-family selection issues exactly one initial application command");
            profile_now += expected[index] == Kind::kSliced ? 10 : 100;
            state.query_status = CUDA_SUCCESS;
            auto observed = session.poll();
            while (observed.paused) {
                expect(expected[index] == Kind::kSliced &&
                           profile->snapshot().samples == previous &&
                           session.resume(command).state == CudaExecutionIssueState::kIssued,
                       "selected sliced family retains its logical timing and continuation");
                observed = session.poll();
            }
            expect(observed.retired == 1 && state.launches - launch_begin ==
                                                (expected[index] == Kind::kSliced ? 4U : 1U),
                   "no application replay while exploring distinct execution families");
            for (std::size_t launch = launch_begin; launch < state.launched_kinds.size(); ++launch)
                expect(state.launched_kinds[launch] == expected[index],
                       "actual native function identity follows the measured family selection");
        }
        expect(profile->snapshot().samples == expected.size() - 1 &&
                   profile->snapshot().hits == 1 && profile->snapshot().trained_cases == 1 &&
                   profile->snapshot().rejected_samples == 0,
               "learn the fastest eligible family after exactly one sample per candidate");
        success(session.unload_module(module.id));
        success(session.shutdown());
        expect(state.unloads == 1 && state.host_frees == 1,
               "multi-family code shares one module and cleans retained progress once");
    }
}
void restored_profiles_preserve_ordinary_execution() {
    const CudaProfileConfig config{8, 1U << 20, 0,
                                   1, {2, 4},   {Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}};
    CudaProfileCache measured(config, profile_clock);
    const CudaProfilePlatform platform{{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi};
    const auto key = measured.bind(platform, kPersistentPtx, {}, "kernel");
    const CudaProfileGeometry geometry{{8, 1, 1}, {32, 1, 1}, 0, Kind::kPreemptive};
    for (const auto duration : {10U, 15U, 20U, 60U, 30U, 40U, 50U}) {
        auto choice = measured.choose(key, geometry, 4, config.kinds);
        expect(choice.observation != nullptr, "create fixture profile over natural requests");
        profile_now = 100;
        choice.observation->begin();
        profile_now += duration;
        choice.observation->finish(true);
    }
    const auto archive = measured.encode("fixture-runtime");
    expect(archive.has_value(), "encode complete fixture measurements");
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        state.is_persistent = true;
        state.uses_multiple_kinds = true;
        state.claimed = 8;
        state.uuid_tag = mode == 2 ? 2 : 1;
        auto sdk = dispatch();
        sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
            *uuid = {};
            uuid->bytes[0] = static_cast<char>(state.uuid_tag);
            return CUDA_SUCCESS;
        };
        sdk.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
        auto cache = std::make_shared<CudaProfileCache>(config, profile_clock, false);
        expect(cache->restore(*archive, "fixture-runtime"), "restore into a fresh ordinary cache");
        CudaExecutionSession session({1, 2}, 7, sdk, {}, {4}, {}, cache, mode == 1);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPersistentPtx);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const auto memory = session.allocate(4);
        success(memory);
        const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        std::array<std::byte, 12> parameters{};
        std::memcpy(parameters.data(), &memory.address, 8);
        for (unsigned int command = 1; command <= 2; ++command) {
            const auto accepted =
                session.accept_kernel(command, {.function = function.id,
                                                .stream = command == 1 ? 0 : stream.id,
                                                .grid = geometry.grid,
                                                .block = geometry.block,
                                                .parameters = parameters});
            expect(accepted.accepted() &&
                       session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt),
                   "read-only cache never adds isolated-profile serialization to ordinary issue");
        }
        expect(state.launches == 2 && session.poll().retained == 2 &&
                   cache->snapshot().samples == 0 && cache->snapshot().cases == 1,
               "two actual pending physical commands with no training observations or new cases");
        for (std::size_t index = 0; index < state.launched_kinds.size(); ++index)
            expect(state.launched_kinds[index] ==
                           (mode == 0 ? std::optional<Kind>{} : std::optional{Kind::kPreemptive}) &&
                       state.physical_grids[index][0] == (mode == 0 ? 8U : 4U),
                   "actual issue uses learned original, eligible priority minimum or changed-GPU "
                   "configured default");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().retired == 2 && cache->snapshot().samples == 0 &&
                   cache->snapshot().hits == (mode == 2 ? 0U : 2U) &&
                   cache->snapshot().loaded_cases == 1,
               "ordinary completion cannot retrain restored measurements");
        success(session.shutdown());
    }
}

enum class SharingProbe { kNone, kContext, kOccupancy, kDevice };
struct SharingFixture {
    int sm_count{2}, max_threads{2048};
    CUresult attribute_status{CUDA_SUCCESS}, occupancy_status{CUDA_SUCCESS};
    unsigned int attribute_failure{}, sm_queries{}, thread_queries{}, occupancy_queries{}, probes{};
    CudaExecutionSession* session{};
    CudaExecutionKernel* kernel{};
    std::uint64_t module{};
    SharingProbe probe{SharingProbe::kNone};
    bool loses_context{};
} sharing_fixture;

void sharing_probe(SharingProbe phase) {
    auto& fixture = sharing_fixture;
    if (fixture.probe != phase) return;
    fixture.probe = SharingProbe::kNone;
    ++fixture.probes;
    auto& session = *fixture.session;
    const auto calls = state.calls;
    expect(session.shutdown().disposition == CudaSessionDisposition::kDeferred,
           "sharing native observations exclude context destruction, including cold discovery");
    if (fixture.kernel) {
        expect(!session.accept_kernel(1, *fixture.kernel).accepted() &&
                   session.unload_module(fixture.module).disposition ==
                       CudaSessionDisposition::kDeferred &&
                   session.activate().status == CUDA_ERROR_NOT_READY &&
                   session.deactivate().status == CUDA_ERROR_NOT_READY &&
                   session.issue_next().native_status == CUDA_ERROR_NOT_READY &&
                   session.poll().status == CUDA_ERROR_NOT_READY &&
                   session.drain_issued().status == CUDA_ERROR_NOT_READY,
               "sharing context and occupancy callbacks cannot reenter admission or lifecycle");
    }
    expect(state.calls == calls && !state.context_destroys && !state.unloads && !state.launches,
           "reentrant sharing controls perform no nested native call");
    if (fixture.loses_context) state.current = nullptr;
}

CudaSessionDispatch sharing_dispatch() {
    auto sdk = dispatch();
    sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
        *uuid = {};
        uuid->bytes[0] = static_cast<char>(state.uuid_tag);
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    sdk.device_attribute = [](int* value, CUdevice_attribute attribute, CUdevice device) {
        if (attribute == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ||
            attribute == CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) {
            ++state.calls;
            sharing_probe(SharingProbe::kDevice);
            const bool sm = attribute == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
            ++(sm ? sharing_fixture.sm_queries : sharing_fixture.thread_queries);
            *value = sm ? sharing_fixture.sm_count : sharing_fixture.max_threads;
            return sharing_fixture.attribute_failure == (sm ? 1U : 2U)
                       ? sharing_fixture.attribute_status
                       : CUDA_SUCCESS;
        }
        return device_attribute(value, attribute, device);
    };
    sdk.execution.context_get_current = [](CUcontext* current) {
        sharing_probe(SharingProbe::kContext);
        return get_current(current);
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    sdk.occupancy = [](int* active, CUfunction function, int threads, std::size_t bytes) {
        ++sharing_fixture.occupancy_queries;
        sharing_probe(SharingProbe::kOccupancy);
        expect(private_kind(function) != Kind::kPreemptive,
               "sharing never considers the fastest but ineligible retreat family");
        const auto status = occupancy(active, function, threads, bytes);
        return sharing_fixture.occupancy_status == CUDA_SUCCESS ? status
                                                                : sharing_fixture.occupancy_status;
    };
    sdk.function_set_cache = [](CUfunction, CUfunc_cache) { return CUDA_SUCCESS; };
    sdk.context = glimmer::tests::context_control::dispatch([] { return state.current; });
    return sdk;
}

enum class SharingCase {
    kMeasured,
    kThreshold,
    kEquality,
    kMissing,
    kDifferentPlatform,
    kFitting,
    kThreadCap,
    kOccupancyCap,
    kGlobalCap,
    kNoOccupancy,
    kOccupancyError,
    kNoSmMultiple,
    kStatic,
    kSliced,
    kFunctionMutation,
    kContextMutation,
    kFlagsMutation,
    kContextReentry,
    kOccupancyReentry,
    kContextLoss,
    kOccupancyLoss,
    kCancel,
};

std::shared_ptr<CudaProfileCache> sharing_profile(SharingCase mode, CudaProfileGeometry geometry) {
    const CudaProfileConfig config{
        8,
        std::size_t{1024} * 1024,
        0,
        1,
        {2, 4, 8},
        {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}};
    auto cache = std::make_shared<CudaProfileCache>(config, profile_clock, false);
    if (mode == SharingCase::kMissing) return cache;
    CudaProfileCache measured(config, profile_clock);
    const auto function = measured.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                        kPersistentPtx, {}, "kernel");
    // Static workers are capped at the logical grid, so the fitting four-CTA
    // case coalesces its configured four/eight-worker Static candidates.
    const auto samples = mode == SharingCase::kFitting ? 15U : 16U;
    for (unsigned int index = 0; index < samples; ++index) {
        auto choice = measured.choose(function, geometry, 8, config.kinds);
        expect(choice.observation != nullptr,
               "independent broad profile observes every exact case");
        unsigned int duration = 120;
        if (!choice.workers)
            duration = 100;
        else if (choice.kind == Kind::kPreemptive)
            duration = 1;
        else if (choice.workers == 8)
            duration = 20;
        else if ((choice.kind == Kind::kBlockStatic && choice.workers == 2) ||
                 (mode == SharingCase::kStatic && choice.kind == Kind::kStatic) ||
                 (mode == SharingCase::kSliced && choice.kind == Kind::kSliced))
            duration = 105;
        else if (choice.kind == Kind::kDynamic && choice.workers == 4)
            duration = mode == SharingCase::kEquality ? 100 : 110;
        profile_now = 100;
        choice.observation->begin();
        profile_now += duration;
        choice.observation->finish(true);
    }
    expect(measured.snapshot().trained_cases == 1 && measured.snapshot().samples == samples,
           "sharing fixture trains every worker and family once, including excluded alternatives");
    const auto archive = measured.encode("sharing-session-fixture");
    expect(archive && cache->restore(*archive, "sharing-session-fixture"),
           "ordinary sharing session receives a fresh inference-only restored cache");
    return cache;
}

void workload_agnostic_selection() {
    for (const auto mode : {SharingCase::kMeasured,
                            SharingCase::kThreshold,
                            SharingCase::kEquality,
                            SharingCase::kMissing,
                            SharingCase::kDifferentPlatform,
                            SharingCase::kFitting,
                            SharingCase::kThreadCap,
                            SharingCase::kOccupancyCap,
                            SharingCase::kGlobalCap,
                            SharingCase::kNoOccupancy,
                            SharingCase::kOccupancyError,
                            SharingCase::kNoSmMultiple,
                            SharingCase::kStatic,
                            SharingCase::kSliced,
                            SharingCase::kFunctionMutation,
                            SharingCase::kContextMutation,
                            SharingCase::kFlagsMutation,
                            SharingCase::kContextReentry,
                            SharingCase::kOccupancyReentry,
                            SharingCase::kContextLoss,
                            SharingCase::kOccupancyLoss,
                            SharingCase::kCancel}) {
        state = {};
        sharing_fixture = {};
        state.is_persistent = state.uses_multiple_kinds = true;
        state.claimed = 16;
        const CudaProfileGeometry geometry{
            {mode == SharingCase::kFitting ? 4U : 16U, 1, 1}, {32, 1, 1}, 0, Kind::kPreemptive};
        auto profile = sharing_profile(mode, geometry);
        if (mode == SharingCase::kDifferentPlatform) state.uuid_tag = 2;
        if (mode == SharingCase::kOccupancyCap) state.occupancy_capacity = 1;
        if (mode == SharingCase::kNoOccupancy) state.occupancy_capacity = 0;
        if (mode == SharingCase::kOccupancyError)
            sharing_fixture.occupancy_status = CUDA_ERROR_INVALID_VALUE;
        if (mode == SharingCase::kNoSmMultiple) sharing_fixture.sm_count = 3;
        CudaSessionPersistentConfig selection{8, false, Kind::kPreemptive, true, 64, 90};
        if (mode == SharingCase::kThreshold) selection.sharing_minimum_speed_percent = 91;
        if (mode == SharingCase::kEquality) selection.sharing_minimum_speed_percent = 100;
        if (mode == SharingCase::kThreadCap) selection.sharing_threads_per_sm = 32;
        if (mode == SharingCase::kGlobalCap) selection.workers = 2;
        CudaExecutionSession session({1, 2}, 7, sharing_dispatch(), {}, selection, {}, profile);
        sharing_fixture.session = &session;
        success(session.initialize(0));
        expect(sharing_fixture.sm_queries == 1 && sharing_fixture.thread_queries == 1,
               "trusted sharing platform geometry is observed once before resource creation");
        const auto module = session.load_ptx(kPersistentPtx);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const auto memory = session.allocate(4);
        success(memory);
        std::array<std::byte, 12> parameters{};
        std::memcpy(parameters.data(), &memory.address, 8);
        const std::uint32_t value = 23;
        std::memcpy(parameters.data() + 8, &value, 4);
        CudaExecutionKernel kernel{.function = function.id,
                                   .grid = geometry.grid,
                                   .block = geometry.block,
                                   .parameters = parameters};
        if (mode == SharingCase::kFunctionMutation)
            expect(session.function_control(CudaFunctionControl::kSetCache, function.id, 1)
                           .original_status == CUDA_SUCCESS,
                   "original function mutation invalidates measured sharing eligibility");
        if (mode == SharingCase::kContextMutation)
            expect(session.context_control(CudaContextControl::kSetCache, 1).original_status == 0,
                   "original context mutation invalidates measured sharing eligibility");
        if (mode == SharingCase::kFlagsMutation) success(session.set_context_flags(0));
        if (mode == SharingCase::kContextReentry || mode == SharingCase::kContextLoss)
            sharing_fixture.probe = SharingProbe::kContext;
        if (mode == SharingCase::kOccupancyReentry || mode == SharingCase::kOccupancyLoss)
            sharing_fixture.probe = SharingProbe::kOccupancy;
        sharing_fixture.kernel = &kernel;
        sharing_fixture.module = module.id;
        sharing_fixture.loses_context =
            mode == SharingCase::kContextLoss || mode == SharingCase::kOccupancyLoss;
        const auto accepted = session.accept_kernel(1, kernel);
        if (sharing_fixture.loses_context) {
            expect(!accepted.accepted() && sharing_fixture.probes == 1 && !state.launches &&
                       session.snapshot().submissions.outstanding == 0 &&
                       session.snapshot().is_quarantined &&
                       sharing_fixture.occupancy_queries ==
                           (mode == SharingCase::kOccupancyLoss ? 1U : 0U),
                   "lost context stops all later occupancy queries and command admission");
            state.current = native_context();
            success(session.shutdown());
            continue;
        }
        expect(accepted.accepted(), "sharing admits exactly one original application command");
        const bool original =
            mode == SharingCase::kThreshold || mode == SharingCase::kMissing ||
            mode == SharingCase::kDifferentPlatform || mode == SharingCase::kFitting ||
            mode == SharingCase::kNoOccupancy || mode == SharingCase::kOccupancyError ||
            mode == SharingCase::kNoSmMultiple || mode == SharingCase::kFunctionMutation ||
            mode == SharingCase::kContextMutation || mode == SharingCase::kFlagsMutation;
        const auto expected_kind = mode == SharingCase::kStatic   ? Kind::kStatic
                                   : mode == SharingCase::kSliced ? Kind::kSliced
                                   : mode == SharingCase::kThreadCap ||
                                           mode == SharingCase::kOccupancyCap ||
                                           mode == SharingCase::kGlobalCap
                                       ? Kind::kBlockStatic
                                       : Kind::kDynamic;
        const unsigned int workers = expected_kind == Kind::kBlockStatic ? 2U : 4U;
        const auto progress = session.persistent_progress(1);
        expect(original
                   ? !progress
                   : progress && progress->workers == workers && progress->kind == expected_kind,
               "trusted progress exposes actual resource-limited geometry, never global maximum");
        if (mode == SharingCase::kCancel) {
            expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
                       session.acknowledge_result(accepted.receipt) && !state.launches &&
                       !state.prefix_copies,
                   "cancellation never launches or observes a profile candidate");
        } else {
            expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt) && session.poll().retained == 1,
                   "ACK retains live physical ownership; sharing does not replay to tune");
            state.query_status = CUDA_SUCCESS;
            auto completed = session.poll();
            while (completed.paused) {
                expect(expected_kind == Kind::kSliced &&
                           session.resume(1).state == CudaExecutionIssueState::kIssued,
                       "resource-limited slices retain one logical command and exact continuation");
                completed = session.poll();
            }
            expect(completed.retired == 1 &&
                       state.launches == (!original && expected_kind == Kind::kSliced ? 4U : 1U),
                   "one completed command, with only required sliced physical continuations");
            for (std::size_t index = 0; index < state.launched_kinds.size(); ++index)
                expect(
                    state.launched_kinds[index] ==
                            (original ? std::optional<Kind>{} : std::optional{expected_kind}) &&
                        state.physical_grids[index][0] == (original ? geometry.grid[0] : workers),
                    "actual function/grid matches the independent resource and threshold oracle");
            std::uint32_t observed{};
            std::memcpy(&observed, state.memory.at(memory.address).data(), sizeof(observed));
            expect(observed == value, "original argument bank is unchanged by sharing selection");
        }
        expect(profile->snapshot().samples == 0 &&
                   profile->snapshot().cases == (mode == SharingCase::kMissing ? 0U : 1U) &&
                   sharing_fixture.sm_queries == 1 && sharing_fixture.thread_queries == 1,
               "ordinary admission and retirement never learn or repeat cold platform queries");
        if (mode == SharingCase::kContextReentry || mode == SharingCase::kOccupancyReentry)
            expect(sharing_fixture.probes == 1, "the callback exclusion oracle actually executes");
        success(session.shutdown());
    }
}

void workload_agnostic_configuration() {
    for (unsigned int mode = 0; mode < 10; ++mode) {
        state = {};
        sharing_fixture = {};
        auto sdk = sharing_dispatch();
        CudaSessionPersistentConfig selection{8, false, Kind::kPreemptive, true, 64, 90};
        auto cache =
            std::make_shared<CudaProfileCache>(CudaProfileConfig{}, profile_clock, mode == 1);
        if (mode == 0) cache.reset();
        if (mode == 2) selection.workers = 0;
        if (mode == 3) selection.replace_sgemm = true;
        if (mode == 4) selection.sharing_threads_per_sm = 0;
        if (mode == 5) selection.sharing_threads_per_sm = UINT32_MAX;
        if (mode == 6) selection.sharing_minimum_speed_percent = 0;
        if (mode == 7) selection.sharing_minimum_speed_percent = 10001;
        if (mode == 8) sdk.device_attribute = nullptr;
        bool rejected{};
        try {
            CudaExecutionSession session({1, 2}, 7, sdk, {}, selection, {}, cache, mode == 9);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !state.calls && !state.context_creates && !state.allocations,
               "invalid sharing configuration fails before any SDK resource or observation");
    }
    for (unsigned int mode = 0; mode < 7; ++mode) {
        state = {};
        sharing_fixture = {};
        if (mode < 2) {
            sharing_fixture.attribute_failure = mode + 1;
            sharing_fixture.attribute_status = CUDA_ERROR_NOT_SUPPORTED;
        }
        if (mode == 2) sharing_fixture.sm_count = 0;
        if (mode == 3) sharing_fixture.sm_count = -1;
        if (mode == 4) sharing_fixture.max_threads = 0;
        if (mode == 5) sharing_fixture.max_threads = 32;
        auto cache = std::make_shared<CudaProfileCache>(CudaProfileConfig{}, profile_clock, false);
        CudaExecutionSession session({1, 2}, 7, sharing_dispatch(), {},
                                     {8, false, Kind::kPreemptive, true, 64, 90}, {}, cache);
        sharing_fixture.session = &session;
        if (mode == 6) sharing_fixture.probe = SharingProbe::kDevice;
        const auto result = session.initialize(0);
        if (mode == 6) {
            success(result);
            expect(sharing_fixture.probes == 1 && state.context_creates == 1,
                   "cold native discovery excludes teardown while permitting one outer "
                   "initialization");
        } else {
            expect(result.disposition == CudaSessionDisposition::kRejected &&
                       result.status ==
                           (mode < 2 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE) &&
                       !result.original_status && !state.context_creates && !state.allocations &&
                       !state.host_allocations && !session.snapshot().is_quarantined,
                   "failed or malformed platform proof rejects cleanly before native resources");
        }
        success(session.shutdown());
    }
}

void stateful_modules_share_original_native_storage() {
    const std::string source =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .global .align 4 .u32 state[4] = {1, 2, 3, 4};\n"
        ".visible .const .align 4 .u32 bias = 7;\n"
        ".visible .entry kernel(.param .u64 pointer, .param .u32 value) { ret; }\n";
    const auto cache = std::make_shared<CudaPtxCache>();
    for (unsigned int mode = 0; mode < 18; ++mode) {
        state = {};
        state.is_persistent = mode != 13;
        state.stateful_source = source;
        state.is_binary = mode != 0 && mode != 17;
        state.has_shared_binary = state.is_binary;
        state.companion_mode = mode;
        state.has_bad_companion_abi = mode == 7 || mode == 11;
        state.claimed = 8;
        if (mode >= 6 && mode != 17) {
            const std::array<glimmer::transform::CudaImageArgument, 2> args{{{0, 8}, {8, 4}}};
            const std::array<glimmer::transform::CudaImageVariable, 2> variables{
                {{"state", 16}, {"bias", 4}}};
            const auto cubin = glimmer::test::cuda_test_cubin(args, 0, variables);
            if (mode == 10)
                state.stateful_source.replace(state.stateful_source.find(".entry kernel"),
                                              std::string(".entry kernel").size(),
                                              ".entry other_kernel");
            state.expected_binary =
                mode == 12 ? cubin : glimmer::test::cuda_test_fatbin(cubin, state.stateful_source);
        } else if (state.is_binary) {
            state.expected_binary = glimmer::test::cuda_test_fatbin({}, source);
        }
        state.fails_generated_load = mode == 2;
        if (mode == 3 || mode == 4) state.private_load_failure = mode - 2;
        if (mode == 5 || mode == 11) state.unload_status = CUDA_ERROR_UNKNOWN;
        auto sdk = dispatch();
        sdk.module_global = [](CUdeviceptr* address, std::size_t* bytes, CUmodule module,
                               const char* name) {
            ++state.global_queries;
            const bool candidate = module == reinterpret_cast<CUmodule>(&state.module_storage);
            expect(
                (candidate || module == reinterpret_cast<CUmodule>(&state.binary_module_storage)) &&
                    std::ranges::find(state.unloaded_modules, module) ==
                        state.unloaded_modules.end() &&
                    (std::string_view{name} == "state" || std::string_view{name} == "bias"),
                "symbol validation/public lookup use only the live original or candidate module");
            if (candidate && state.companion_mode == 9) return CUDA_ERROR_NOT_FOUND;
            *address = candidate ? 0x900000 : 0xa00000;
            *bytes = std::string_view{name} == "state" ? 16 : 4;
            if (std::string_view{name} == "bias") *address += 32;
            if (candidate && state.companion_mode == 8) *bytes /= 2;
            state.memory.try_emplace(*address, *bytes);
            return CUDA_SUCCESS;
        };
        CudaSessionLimits limits;
        if (mode >= 14) {
            limits.submissions.max_commands = 1;
            limits.max_device_bytes = mode == 14 ? 27 : mode == 15 ? 28 : mode == 16 ? 54 : 33;
        }
        CudaExecutionSession session({1, 2}, 7, sdk, limits, {mode == 13 ? 0U : 320U}, cache);
        state.budget_session = &session;
        success(session.initialize(0));
        const auto private_bytes = session.snapshot().device_bytes;
        const auto module =
            state.is_binary ? session.load_binary(state.expected_binary) : session.load_ptx(source);
        if (mode == 14 || mode == 17) {
            expect(module.status == CUDA_ERROR_OUT_OF_MEMORY && !module.original_status &&
                       !session.snapshot().is_quarantined && !session.snapshot().modules &&
                       session.snapshot().device_bytes == private_bytes &&
                       state.module_load_charges.empty(),
                   "native CUBIN and direct PTX static quota reject before any SDK module load");
            success(session.shutdown());
            continue;
        }
        const auto original_static = mode < 6 ? 26U : 20U;
        expect(!state.module_load_charges.empty() &&
                   state.module_load_charges.front() == private_bytes + original_static &&
                   (state.module_load_charges.size() != 2 ||
                    state.module_load_charges.back() == private_bytes + original_static + 26),
               "original and candidate static reservations exist during their actual SDK overlap");
        if ((mode >= 3 && mode <= 5) || mode == 11) {
            expect(module.disposition == CudaSessionDisposition::kRejected && !module.id &&
                       module.original_status == CUDA_SUCCESS &&
                       session.snapshot().is_quarantined && session.snapshot().modules == 1 &&
                       state.module_loads == 2 &&
                       state.unloads == (mode == 5 || mode == 11 ? 1U : 0U) &&
                       session.snapshot().device_bytes == private_bytes + original_static + 26,
                   "partial/null private load and uncertain validation unload retain all "
                   "native ownership without publishing a module");
            const auto unloads = state.unloads;
            success(session.shutdown());
            expect(state.unloads == unloads && state.context_destroys == 1 &&
                       session.snapshot().device_bytes == 0,
                   "context recovery never retries an ambiguous module cleanup");
            continue;
        }
        success(module);
        const bool transformed = mode < 2 || mode == 6 || mode == 16;
        expect(state.validation_module_retired == (mode == 1 || mode == 6 || mode == 16) &&
                   state.module_loads ==
                       (mode == 0 || mode == 12 || mode == 13 || mode == 15 ? 1U : 2U),
               "both PTX-selected and validated CUBIN companions resolve state before publication");
        if (mode == 6)
            expect(state.global_queries == 4 && state.parameter_queries == 4,
                   "all native/candidate variable sizes and parameter layouts are checked before "
                   "publication");
        const auto global = session.get_global(module.id, "state");
        success(global);
        expect(session.snapshot().device_bytes ==
                   private_bytes + (transformed ? 26U : original_static),
               "lookup borrows the existing charge including unqueried constant storage");
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        expect(session.snapshot().persistent_functions == (transformed ? 1U : 0U),
               "only validated stateful companions promote; incompatibility and original mode stay "
               "native");
        std::array<std::byte, 12> parameters{};
        std::memcpy(parameters.data(), &global.address, 8);
        for (unsigned int command = 1; command <= 2; ++command) {
            const std::uint32_t value = 10 + command;
            std::memcpy(parameters.data() + 8, &value, 4);
            state.occupancy_capacity = command == 1 ? 4 : 0;
            state.query_status = CUDA_ERROR_NOT_READY;
            const auto accepted = session.accept_kernel(command, {.function = function.id,
                                                                  .grid = {8, 1, 1},
                                                                  .block = {32, 1, 1},
                                                                  .parameters = parameters});
            expect(accepted.accepted() &&
                       session.issue_next().state == CudaExecutionIssueState::kIssued &&
                       session.acknowledge_result(accepted.receipt) &&
                       session.poll().retained == 1 &&
                       session.unload_module(module.id).disposition ==
                           CudaSessionDisposition::kDeferred,
                   "original/private work retains the shared module range through actual stop");
            success(session.get_global(module.id, "state"));
            state.query_status = CUDA_SUCCESS;
            expect(session.poll().retired == 1,
                   "shared module retires only after physical completion");
            std::array<std::uint32_t, 4> observed{};
            success(session.download({global.id, 0}, std::as_writable_bytes(std::span{observed})));
            expect(observed.front() == value,
                   "both injected native paths use the public module range");
        }
        expect(session.snapshot().persistent_accepted == (transformed ? 1U : 0U),
               "mixed private/original selection never creates another copy of module state");
        success(session.unload_module(module.id));
        expect(
            state.unloads ==
                    (mode == 0 || mode == 2 || mode == 12 || mode == 13 || mode == 15 ? 1U : 2U) &&
                session.snapshot().image_bytes == 0 &&
                session.snapshot().device_bytes == private_bytes &&
                !session.device_memory({global.id, 0}, 1),
            "the single surviving module and discovered ranges release once");
        success(session.shutdown());
    }
}
void module_static_failure_refunds() {
    const std::string source =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".global .align 4 .u32 state[4];\n"
        ".entry kernel(.param .u64 pointer, .param .u32 value) { ret; }\n";
    const std::array<glimmer::transform::CudaImageArgument, 2> args{{{0, 8}, {8, 4}}};
    const std::array<glimmer::transform::CudaImageVariable, 1> variables{{{"state", 16}}};
    for (const bool binary : {false, true})
        for (unsigned int mode = 0; mode < 5; ++mode) {
            state = {};
            state.is_persistent = true;  // Fake SDK accepts the stateful source; no private mode.
            state.stateful_source = source;
            state.is_binary = binary;
            if (binary) state.expected_binary = glimmer::test::cuda_test_cubin(args, 0, variables);
            state.module_status = mode < 2 ? CUDA_ERROR_INVALID_IMAGE : CUDA_SUCCESS;
            state.returns_partial_module = mode == 1;
            state.returns_null_module = mode == 2;
            CudaExecutionSession session({1, 2}, 7, dispatch());
            state.budget_session = &session;
            success(session.initialize(0));
            const auto module =
                binary ? session.load_binary(state.expected_binary) : session.load_ptx(source);
            const auto charge = binary ? 16U : 19U;
            expect(state.module_load_charges == std::vector<std::size_t>{charge},
                   "even a failing original load is preceded by its static reservation");
            if (mode == 0) {
                expect(module.status == CUDA_ERROR_INVALID_IMAGE && !module.id &&
                           !session.snapshot().modules && !session.snapshot().device_bytes &&
                           !session.snapshot().is_quarantined,
                       "known empty original load failure refunds only its reservation");
            } else if (mode <= 2) {
                expect(!module.id && session.snapshot().is_quarantined &&
                           session.snapshot().device_bytes == charge,
                       "partial original handle and null success retain static ownership");
            } else {
                success(module);
                state.unload_status = mode == 3 ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
                const auto unloaded = session.unload_module(module.id);
                expect(unloaded.status == state.unload_status &&
                           session.snapshot().device_bytes == (mode == 3 ? charge : 0),
                       "unqueried static storage refunds only on proven original unload");
            }
            const auto unloads = state.unloads;
            success(session.shutdown());
            expect(state.unloads == unloads && !session.snapshot().device_bytes &&
                       state.context_destroys == 1,
                   "owned context destruction settles uncertain charges without unload retries");
        }
}
void persistent_factory_rejections_and_prefix_failure() {
    for (unsigned int mode = 0; mode < 6; ++mode) {
        state = {};
        state.is_persistent = true;
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        if (mode == 0) limits.max_image_bytes = kPersistentPtx.size() + 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320});
        success(session.initialize(0));
        state.fails_generated_load = mode == 5;
        const auto module = session.load_ptx(kPersistentPtx);
        if (mode == 0) {
            expect(module.status == CUDA_ERROR_OUT_OF_MEMORY && !module.original_status &&
                       session.snapshot().modules == 0,
                   "generated bytes, not just original PTX, must fit the image budget before SDK "
                   "load");
            success(session.shutdown());
            continue;
        }
        success(module);
        state.is_bad_private_abi = mode == 1;
        const auto function = session.get_function(module.id, "kernel");
        if (mode == 1) {
            expect(function.status == CUDA_ERROR_INVALID_VALUE &&
                       function.original_status == CUDA_SUCCESS &&
                       session.snapshot().functions == 0,
                   "invalid private ABI is preparation rejection, not a rewritten original lookup "
                   "result");
            success(session.shutdown());
            continue;
        }
        success(function);
        const auto memory = session.allocate(4);
        success(memory);
        std::array<std::byte, 12> parameters{};
        const std::uint32_t value = 19;
        std::memcpy(parameters.data(), &memory.address, 8);
        std::memcpy(parameters.data() + 8, &value, 4);
        if (mode == 2) state.occupancy_capacity = 0;
        if (mode == 3) state.prefix_status = CUDA_ERROR_INVALID_VALUE;
        if (mode == 4) {
            state.current = nullptr;
            const auto calls = state.calls;
            expect(!session.accept_kernel(1, {.function = function.id,
                                              .stream = 0,
                                              .grid = {8, 1, 1},
                                              .block = {32, 1, 1},
                                              .parameters = parameters})
                           .accepted() &&
                       state.calls == calls + 1 && session.snapshot().persistent_slots == 0,
                   "wrong current context cannot call preparation SDKs or claim private slots");
            state.current = native_context();
        }
        const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                        .stream = 0,
                                                        .grid = {8, 1, 1},
                                                        .block = {32, 1, 1},
                                                        .parameters = parameters});
        expect(accepted.accepted(), "owned factory launch acceptance");
        const auto issued = session.issue_next();
        const auto reply = session.host_result(accepted.receipt);
        expect(reply.has_value(),
               "host result available only after actual issue or explicit rejection");
        if (mode == 2 || mode == 5)
            expect(issued.state == CudaExecutionIssueState::kIssued && state.launches == 1 &&
                       state.prefix_copies == 0 && session.snapshot().persistent_accepted == 0,
                   "unsupported occupancy stays native before admission, never accepted-work "
                   "fallback");
        else if (mode == 3)
            expect(issued.state == CudaExecutionIssueState::kRejected && state.launches == 0 &&
                       reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       reply->status == CUDA_ERROR_INVALID_VALUE && session.poll().retained == 1 &&
                       session.snapshot().persistent_slots == 1 && state.host_frees == 0,
                   "prefix failure may follow private GPU activity, so no target launch or early "
                   "release");
        if (mode == 5)
            expect(
                state.generated_load_failures == 1 && session.snapshot().persistent_functions == 0,
                "known JIT failure tries original intake only before ownership publication");
        state.claimed = 8;
        if (mode == 4) {
            state.query_status = CUDA_SUCCESS;
            expect(session.poll().paused == 1 &&
                       session.resume(1).state == CudaExecutionIssueState::kIssued,
                   "valid restored context may continue only after real partial stop");
        }
        state.query_status = CUDA_SUCCESS;
        expect(
            session.poll().retired == 1 && session.acknowledge_result(accepted.receipt),
            "actual event stop retires either native or rejected private preparation exactly once");
        success(session.shutdown());
    }
    for (const bool fails_mapping : {false, true}) {
        state = {};
        if (fails_mapping)
            state.map_status = CUDA_ERROR_INVALID_VALUE;
        else
            state.host_status = CUDA_ERROR_OUT_OF_MEMORY;
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320});
        const auto initialized = session.initialize(0);
        expect(initialized.disposition == CudaSessionDisposition::kRejected &&
                   initialized.original_status == CUDA_SUCCESS &&
                   initialized.status != CUDA_SUCCESS,
               "private pool failure preserves actual context-create success separately");
        success(session.shutdown());
        expect(!state.host && state.memory.empty(),
               "known partial initialization cleans only owned handles");
    }
}

void private_pool_ambiguity_never_retries_cleanup() {
    for (unsigned int mode = 0; mode < 4; ++mode) {
        const auto child = fork();
        expect(child >= 0, "isolate intentionally retained private allocation ambiguity");
        if (child == 0) {
            try {
                state = {};
                if (mode == 0) {
                    state.host_status = CUDA_ERROR_OUT_OF_MEMORY;
                    state.returns_partial_host = true;
                }
                if (mode == 3) state.returns_null_host = true;
                CudaSessionLimits limits;
                limits.submissions.max_commands = 1;
                auto session = std::make_unique<CudaExecutionSession>(
                    ExecutionSessionIdentity{1, 2}, 7, dispatch(), limits,
                    CudaSessionPersistentConfig{320});
                const auto initialized = session->initialize(0);
                if (mode == 1 || mode == 2)
                    success(initialized);
                else
                    expect(
                        initialized.original_status == CUDA_SUCCESS &&
                            initialized.status != CUDA_SUCCESS,
                        "uncertain private host creation is not original context-create failure");
                if (mode == 1) state.host_free_status = CUDA_ERROR_UNKNOWN;
                if (mode == 2) state.free_status = CUDA_ERROR_UNKNOWN;
                expect(session->shutdown().status != CUDA_SUCCESS &&
                           session->shutdown().status != CUDA_SUCCESS &&
                           state.context_destroys == 0 &&
                           state.host_frees == (mode == 1 || mode == 2 ? 1U : 0U) &&
                           state.frees == (mode == 2 ? 1U : 0U) &&
                           session->snapshot().device_bytes == 8,
                       "unknown private allocation/free remains charged and cannot retry or "
                       "destroy context");
                session.reset();
                expect(state.host_frees == (mode == 1 || mode == 2 ? 1U : 0U),
                       "destructor cannot repeat an ambiguous host release");
                _exit(0);
            } catch (...) {
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "private ambiguity retention is isolated, never a leak-free recovery claim");
    }
}

void persistent_configuration_and_private_quota_bounds() {
    for (unsigned int mode = 0; mode < 5; ++mode) {
        state = {};
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        auto functions = dispatch();
        if (mode == 0) limits.max_private_host_bytes = 63;
        if (mode == 1) limits.max_device_bytes = 7;
        if (mode == 2) functions.upload_async = nullptr;
        if (mode == 3) functions.occupancy = nullptr;
        if (mode == 4) functions.function_set_attribute = nullptr;
        bool rejected = false;
        try {
            CudaExecutionSession invalid({1, 2}, 7, functions, limits, {320});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && state.calls == 0 && state.host_allocations == 0,
               "invalid private budgets/dispatch reject before any SDK or resource creation");
    }
    state = {};
    CudaSessionLimits limits;
    limits.max_device_bytes = 16;
    limits.submissions.max_commands = 1;
    CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {320});
    success(session.initialize(0));
    const auto calls = state.allocations;
    expect(session.allocate(9).status == CUDA_ERROR_OUT_OF_MEMORY && state.allocations == calls,
           "private device counters cannot disappear from the user allocation quota");
    success(session.allocate(8));
    expect(session.snapshot().device_bytes == 16, "exact private plus public device budget");
    success(session.shutdown());
}

void persistent_function_carveout_preparation() {
    for (unsigned int mode = 0; mode < 5; ++mode) {
        state = {};
        const bool is_persistent = mode != 0;
        state.is_persistent = is_persistent;
        if (mode == 2) state.failed_configuration = 1;
        if (mode == 3) state.failed_configuration = 2;
        CudaExecutionSession session({1, 2}, 7, dispatch(), {}, {is_persistent ? 320U : 0U});
        success(session.initialize(0));
        const auto module = session.load_ptx(mode == 0   ? kPtx
                                             : mode == 4 ? kUnsupportedPtx
                                                         : kPersistentPtx);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        if (mode == 2 || mode == 3) {
            expect(function.status == CUDA_ERROR_INVALID_VALUE &&
                       function.disposition == CudaSessionDisposition::kRejected &&
                       function.original_status == CUDA_SUCCESS && !function.id &&
                       session.snapshot().functions == 0 && state.launches == 0 &&
                       state.configured_functions.size() == state.failed_configuration,
                   "failed original/private preparation publishes no function or kernel success");
        } else {
            success(function);
            const auto count = mode == 0 ? 0U : mode == 4 ? 1U : 2U;
            expect(state.configured_functions.size() == count,
                   "persistent profile configures both domains including native-only helpers");
            const auto repeated = session.get_function(module.id, "kernel");
            success(repeated);
            expect(repeated.id == function.id && state.configured_functions.size() == count,
                   "repeated resolved handles do not repeat attribute preparation");
        }
        success(session.shutdown());
        expect(state.context_destroys == 1 && state.host_frees == (is_persistent ? 1U : 0U),
               "partial function preparation preserves once-only module/context/pool cleanup");
    }
}

void complete_owned_session() {
    state = {};
    CudaExecutionSession session({1, 2}, 7, dispatch());
    success(session.initialize(0));
    const auto module = session.load_ptx(kPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    expect(function.arguments.size() == 2 && function.arguments[1].offset == 8 &&
               function.parameter_bytes == 12,
           "native ABI metadata must determine packed storage");
    const auto memory = session.allocate(8);
    success(memory);
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(stream);
    std::array<std::byte, 12> parameters{};
    std::uint32_t value = 41;
    std::memcpy(parameters.data(), &memory.address, 8);
    std::memcpy(parameters.data() + 8, &value, 4);
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .stream = stream.id,
                                                    .grid = {80, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    expect(accepted.accepted(), "own registered session launch");
    parameters.fill(std::byte{0});
    const auto calls = state.calls;
    expect(session.release(memory.id).disposition == CudaSessionDisposition::kDeferred &&
               session.unload_module(module.id).disposition == CudaSessionDisposition::kDeferred &&
               session.destroy_stream(stream.id).disposition == CudaSessionDisposition::kDeferred &&
               state.frees == 0 && state.calls >= calls,
           "accepted work must defer native resource mutation");
    expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.poll().retained == 1,
           "session must issue its own native resources and preserve pending work");
    expect(session.acknowledge_result(accepted.receipt) &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               state.context_destroys == 0,
           "receipt or shutdown does not prove stop");
    expect(session.allocate(1).disposition == CudaSessionDisposition::kRejected &&
               !session.allocate(1).original_status && state.allocations == 1,
           "closing a session cannot create new resources while physical cleanup is pending");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "real backend proof retires owned launch");
    std::uint32_t actual{};
    success(session.download({memory.id, 0}, std::as_writable_bytes(std::span{&actual, 1})));
    expect(actual == value, "caller-overwritten packed arguments must retain original output");
    success(session.release(memory.id));
    success(session.destroy_stream(stream.id));
    success(session.unload_module(module.id));
    expect(!session.release(memory.id).original_status && session.snapshot().device_bytes == 0,
           "stale release must not reach SDK or change accounting");
    success(session.shutdown());
    success(session.shutdown());
    expect(state.context_creates == 1 && state.context_destroys == 1 && state.frees == 1,
           "context and allocation ownership must retire exactly once");
}

void bounds_abi_and_nonreuse() {
    state = {};
    CudaSessionLimits limits;
    limits.max_objects = 2;
    limits.max_device_bytes = 8;
    limits.max_image_bytes = kPtx.size() + 1;
    CudaExecutionSession session({1, 2}, 7, dispatch(), limits);
    success(session.initialize(0));
    expect(!session.load_ptx(std::string_view("bad\0PTX", 7)).original_status &&
               !session.load_ptx(std::string(limits.max_image_bytes, 'x')).original_status,
           "unowned/oversize module input rejects");
    const auto allocation = session.allocate(8);
    success(allocation);
    const auto before = state.allocations;
    expect(!session.allocate(1).original_status && state.allocations == before,
           "memory cap rejects before SDK");
    std::array<std::byte, 2> bytes{};
    const auto copy_calls = state.copies;
    expect(!session.upload({allocation.id, 7}, bytes).original_status &&
               !session.download({allocation.id, std::numeric_limits<std::size_t>::max()}, bytes)
                    .original_status &&
               state.copies == copy_calls,
           "overflow/out-of-range copies must not reach native API");
    success(session.release(allocation.id));
    const auto replacement = session.allocate(8);
    success(replacement);
    expect(replacement.id != allocation.id && !session.release(allocation.id).original_status,
           "resource identities never recycle after successful release");
    const auto module = session.load_ptx(kPtx);
    success(module);
    state.has_overlapping_abi = true;
    const auto malformed = session.get_function(module.id, "kernel");
    expect(malformed.disposition == CudaSessionDisposition::kRejected &&
               malformed.original_status == CUDA_SUCCESS && !malformed.id,
           "invalid ABI preparation must not overwrite successful original lookup or publish a "
           "binding");
    state.has_overlapping_abi = false;
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto again = session.get_function(module.id, "kernel");
    success(again);
    expect(again.id == function.id, "stable module/function mapping is generation scoped");
    success(session.deactivate());
    expect(session.allocate(1).status == CUDA_ERROR_INVALID_CONTEXT,
           "inactive context cannot mutate resources");
    success(session.activate());
    success(session.shutdown());
    expect(state.memory.empty(), "owned context fence retires remaining legacy allocations");
}

void ambiguous_release_and_partial_initialization() {
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        CudaExecutionSession session({1, 2}, 7, dispatch());
        success(session.initialize(0));
        if (mode == 1) {
            state.allocation_status = CUDA_ERROR_OUT_OF_MEMORY;
            state.returns_partial_allocation = true;
        }
        if (mode == 2) state.returns_null_allocation = true;
        const auto memory = session.allocate(8);
        if (mode == 0) {
            success(memory);
            state.free_status = CUDA_ERROR_UNKNOWN;
            state.erases_on_free_error = true;
            const auto released = session.release(memory.id);
            expect(released.original_status == CUDA_ERROR_UNKNOWN && state.frees == 1,
                   "native free failure must remain unchanged even if actual free occurred");
        }
        expect(session.snapshot().is_quarantined && session.snapshot().device_bytes == 8 &&
                   !session.allocate(1).original_status,
               "unknown allocation/release retains charged ownership and rejects new work");
        success(session.shutdown());
        expect(state.context_destroys == 1 && state.frees == (mode == 0 ? 1U : 0U) &&
                   state.memory.empty(),
               "only context destruction retires ambiguity without repeated native release");
    }
    for (bool partial : {false, true}) {
        state = {};
        state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
        state.returns_partial_context = partial;
        {
            CudaExecutionSession session({1, 2}, 7, dispatch());
            expect(session.initialize(0).original_status == CUDA_ERROR_OUT_OF_MEMORY,
                   "preserve actual startup error");
            expect(session.snapshot().is_quarantined == partial,
                   "checked no-handle rejection differs from retained partial creation");
            success(session.shutdown());
            expect(!session.snapshot().is_quarantined,
                   "real owned-context fence settles GPU uncertainty, not a missing handle");
            const auto calls = state.calls;
            expect(!session.initialize(0).original_status && state.calls == calls,
                   "a closed factory cannot replay its original context creation");
        }
        expect(state.context_destroys == (partial ? 1U : 0U),
               "partial initialization cleans only returned owned context");
    }
}

void owner_fork_and_exhaustion() {
    state = {};
    CudaSessionLimits limits;
    limits.first_resource_id = std::numeric_limits<std::uint64_t>::max();
    CudaExecutionSession session({1, 2}, 7, dispatch(), limits);
    success(session.initialize(0));
    const auto first = session.allocate(4);
    success(first);
    success(session.release(first.id));
    expect(first.id == std::numeric_limits<std::uint64_t>::max() &&
               !session.allocate(4).original_status,
           "resource exhaustion cannot wrap or reuse old IDs");
    const auto before = state.calls;
    auto wrong = std::async(std::launch::async, [&] {
        return session.activate().status == CUDA_ERROR_INVALID_CONTEXT &&
               session.allocate(4).status == CUDA_ERROR_INVALID_CONTEXT &&
               session.shutdown().status == CUDA_ERROR_INVALID_CONTEXT;
    });
    expect(wrong.get() && state.calls == before, "wrong owner must reject before any SDK call");
    const auto child = fork();
    expect(child >= 0, "fork inherited session fixture");
    if (child == 0) {
        const bool guarded = session.activate().status == CUDA_ERROR_INVALID_CONTEXT &&
                             session.shutdown().status == CUDA_ERROR_INVALID_CONTEXT &&
                             state.calls == before;
        session.~CudaExecutionSession();
        _exit(guarded && state.calls == before ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child must not release parent resources");
    success(session.shutdown());
}

void dead_owner_and_context_destroy_ambiguity() {
    for (bool owner_died : {false, true}) {
        const auto child = fork();
        expect(child >= 0, "isolate intentionally retained resource owner");
        if (child == 0) {
            state = {};
            std::unique_ptr<CudaExecutionSession> session;
            bool valid = true;
            if (owner_died) {
                auto owner = std::async(std::launch::async, [&] {
                    session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{1, 2},
                                                                     7, dispatch());
                    return session->initialize(0).status == CUDA_SUCCESS;
                });
                valid = owner.get();
                const auto before = state.calls;
                auto replacement = std::async(std::launch::async, [&] {
                    return session->activate().status == CUDA_ERROR_INVALID_CONTEXT;
                });
                valid = valid && replacement.get() && state.calls == before;
                session.reset();
                valid = valid && state.context_destroys == 0;
            } else {
                session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{1, 2}, 7,
                                                                 dispatch());
                valid = session->initialize(0).status == CUDA_SUCCESS;
                state.destroy_status = CUDA_ERROR_UNKNOWN;
                valid = valid && session->shutdown().original_status == CUDA_ERROR_UNKNOWN &&
                        !session->shutdown().original_status && state.context_destroys == 1;
                session.reset();
                valid = valid && state.context_destroys == 1;
            }
            _exit(valid ? 0 : 1);
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "dead owner or ambiguous context destruction cannot transfer ownership or retry "
               "cleanup");
    }
    state = {};
    state.current = native_context();
    {
        CudaExecutionSession session({1, 2}, 7, dispatch());
        expect(session.initialize(0).disposition == CudaSessionDisposition::kRejected,
               "existing native context cannot be silently migrated");
    }
    expect(state.context_creates == 0 && state.context_destroys == 0 &&
               state.current == native_context(),
           "foreign current context must remain untouched");
}

void module_and_stream_partial_results() {
    for (unsigned int mode = 0; mode < 6; ++mode) {
        state = {};
        CudaExecutionSession session({1, 2}, 7, dispatch());
        success(session.initialize(0));
        if (mode < 3) {
            if (mode == 0) {
                state.module_status = CUDA_ERROR_INVALID_PTX;
                state.returns_partial_module = true;
            }
            if (mode == 1) state.returns_null_module = true;
            const auto module = session.load_ptx(kPtx);
            if (mode == 2) {
                success(module);
                const auto function = session.get_function(module.id, "kernel");
                success(function);
                state.unload_status = CUDA_ERROR_UNKNOWN;
                expect(
                    session.unload_module(module.id).original_status == CUDA_ERROR_UNKNOWN &&
                        !session.unload_module(module.id).original_status && state.unloads == 1,
                    "ambiguous unload cannot repeat release or restore a stale function binding");
            } else
                expect(!module.id && module.original_status ==
                                         (mode == 0 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS),
                       "partial/malformed module output preserves original SDK status");
            expect(session.snapshot().modules == 1 &&
                       session.snapshot().image_bytes == kPtx.size() + 1,
                   "unknown module stays charged");
        } else {
            if (mode == 3) {
                state.stream_status = CUDA_ERROR_OUT_OF_MEMORY;
                state.returns_partial_stream = true;
            }
            if (mode == 4) state.returns_null_stream = true;
            const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
            if (mode == 5) {
                success(stream);
                state.stream_destroy_status = CUDA_ERROR_UNKNOWN;
                expect(session.destroy_stream(stream.id).original_status == CUDA_ERROR_UNKNOWN &&
                           !session.destroy_stream(stream.id).original_status &&
                           state.stream_destroys == 1,
                       "ambiguous stream destruction cannot repeat native cleanup");
            } else
                expect(!stream.id && stream.original_status ==
                                         (mode == 3 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS),
                       "partial/malformed stream output preserves original SDK status");
            expect(session.snapshot().streams == 1, "uncertain stream metadata remains owned");
        }
        expect(session.snapshot().is_quarantined, "resource ambiguity contains all successor work");
        success(session.shutdown());
        expect(state.context_destroys == 1 && !session.snapshot().has_context,
               "real context fence retires residual resources once");
    }
}
void binary_owned_images_and_private_modules(const std::shared_ptr<CudaPtxCache>& cache = {}) {
    const std::array<glimmer::transform::CudaImageArgument, 2> layout{{{0, 8}, {8, 4}}};
    const auto cubin = glimmer::test::cuda_test_cubin(layout);
    for (unsigned int mode = 0; mode < 4; ++mode) {
        state = {};
        state.is_binary = true;
        state.is_persistent = mode == 1 || mode == 2;
        state.fails_generated_load = mode == 2;
        state.expected_binary =
            mode == 0 ? cubin : glimmer::test::cuda_test_fatbin(cubin, kPersistentPtx);
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits,
                                     {state.is_persistent ? 320U : 0U}, cache);
        success(session.initialize(0));
        auto caller = state.expected_binary;
        const auto module = session.load_binary(caller);
        success(module);
        std::fill(caller.begin(), caller.end(), std::byte{0});
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const bool is_prepared = mode == 1;
        expect(function.original_status == CUDA_SUCCESS && function.arguments.size() == 2 &&
                   function.parameter_bytes == 12 && state.binary_loads == 1 &&
                   session.snapshot().persistent_functions == (is_prepared ? 1U : 0U),
               "owned binary metadata binds native and optional private ABI independently");
        if (is_prepared) {
            const auto denied = session.get_function(module.id, "kernel__glimmer_ptb_preemptive");
            expect(denied.status == CUDA_ERROR_INVALID_VALUE && !denied.original_status,
                   "private module symbols never become application function handles");
        }
        const auto allocation = session.allocate(4);
        success(allocation);
        std::array<std::byte, 12> arguments{};
        const std::uint32_t value = 29;
        std::memcpy(arguments.data(), &allocation.address, 8);
        std::memcpy(arguments.data() + 8, &value, 4);
        CudaExecutionKernel kernel{.function = function.id,
                                   .stream = 0,
                                   .grid = {8, 1, 1},
                                   .block = {32, 1, 1},
                                   .parameters = arguments};
        const auto accepted = session.accept_kernel(1, kernel);
        expect(accepted.accepted(), "accept original ABI only, not client private state");
        std::fill(arguments.begin(), arguments.end(), std::byte{0});
        expect(session.issue_next().state == CudaExecutionIssueState::kIssued,
               "binary function issues exactly once from copied arguments");
        expect(session.host_result(accepted.receipt)->status == CUDA_SUCCESS &&
                   session.acknowledge_result(accepted.receipt),
               "one actual initial SDK reply");
        if (is_prepared)
            expect(session.request_retreat(1) == CUDA_SUCCESS, "retreat prepared binary companion");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().status == CUDA_SUCCESS,
               "real trailing event precedes binary progress");
        if (is_prepared) {
            expect(session.snapshot().submissions.paused == 1 &&
                       session.persistent_progress(1)->completed_blocks == 3,
                   "incomplete binary companion stop remains logically pending");
            state.claimed = 20;
            expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
                       session.poll().status == CUDA_SUCCESS &&
                       state.counter_before_launch == std::vector<std::uint64_t>{0, 3} &&
                       state.prefix_copies == 1,
                   "exact retained binary progress is never reset");
        }
        expect(session.snapshot().submissions.outstanding == 0 &&
                   state.generated_load_failures == (mode == 2 ? 1U : 0U),
               "known optional no-handle JIT failure stays native without retry");
        success(session.release(allocation.id));
        success(session.unload_module(module.id));
        expect(state.unloads == (is_prepared ? 2U : 1U) && session.snapshot().image_bytes == 0,
               "original/private modules release once with complete image accounting");
        success(session.shutdown());
        expect(state.context_destroys == 1 && state.host_frees == (state.is_persistent ? 1U : 0U),
               "binary and private preparation teardown remains owner-fenced");
    }
    if (cache)
        expect(cache->snapshot().lookups == 2 && cache->snapshot().hits == 1 &&
                   cache->snapshot().transforms == 1,
               "selected binary PTX reuses pure preparation without sharing cubin metadata, JIT "
               "errors or SDK handles");
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        state.is_binary = true;
        state.expected_binary = cubin;
        state.module_status = mode == 1 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_IMAGE;
        state.returns_null_module = mode == 1;
        state.returns_partial_module = mode == 2;
        CudaExecutionSession session({1, 2}, 7, dispatch());
        success(session.initialize(0));
        const auto result = session.load_binary(cubin);
        expect(
            result.original_status == state.module_status && state.binary_loads == 1 && !result.id,
            "binary original status is never replaced or retried on malformed/partial output");
        expect(session.snapshot().modules == (mode ? 1U : 0U) &&
                   session.snapshot().is_quarantined == (mode != 0),
               "unknown module output retains exact owned input through a real context fence");
        success(session.shutdown());
    }
    state = {};
    state.is_binary = true;
    state.expected_binary = cubin;
    CudaSessionLimits limits;
    limits.max_image_bytes = cubin.size();
    CudaExecutionSession bounded({1, 2}, 7, dispatch(), limits);
    success(bounded.initialize(0));
    const auto rejected = bounded.load_binary(cubin);
    expect(rejected.status == CUDA_ERROR_OUT_OF_MEMORY && !rejected.original_status &&
               state.binary_loads == 0 && bounded.snapshot().image_bytes == 0,
           "retained binary metadata is charged before any target SDK operation");
    success(bounded.shutdown());
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        state.is_binary = true;
        const std::array<glimmer::transform::CudaImageArgument, 2> mismatched{{{0, 4}, {8, 4}}};
        const auto inconsistent = glimmer::test::cuda_test_cubin(mismatched);
        state.expected_binary = mode == 0
                                    ? inconsistent
                                    : glimmer::test::cuda_test_fatbin(
                                          cubin,
                                          ".version 7.0\n.target sm_70\n.address_size 64\n"
                                          ".visible .entry kernel(.param .u64 pointer) { ret; }\n");
        auto native_dispatch = dispatch();
        if (mode == 2) native_dispatch.device_attribute = nullptr;
        CudaExecutionSession session({1, 2}, 7, native_dispatch);
        success(session.initialize(0));
        const auto module = session.load_binary(state.expected_binary);
        if (mode == 2) {
            expect(module.status == CUDA_ERROR_NOT_SUPPORTED && !module.original_status &&
                       state.binary_loads == 0,
                   "missing optional architecture dispatch rejects before original module load");
        } else {
            success(module);
            const auto result = session.get_function(module.id, "kernel");
            expect(result.status == CUDA_ERROR_INVALID_VALUE &&
                       result.original_status == CUDA_SUCCESS && !result.id &&
                       state.parameter_queries == (mode == 0 ? 1U : 0U) && state.launches == 0,
                   "SDK layout and fallback PTX arity conflicts cannot produce a launchable ABI");
            success(session.unload_module(module.id));
        }
        success(session.shutdown());
    }
    for (unsigned int mode = 0; mode < 2; ++mode) {
        state = {};
        state.is_binary = state.is_persistent = true;
        state.expected_binary = glimmer::test::cuda_test_fatbin(cubin, kPersistentPtx);
        CudaSessionLimits private_limits;
        private_limits.submissions.max_commands = 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), private_limits, {320});
        success(session.initialize(0));
        const auto module = session.load_binary(state.expected_binary);
        success(module);
        state.module_status = mode ? CUDA_SUCCESS : CUDA_ERROR_INVALID_IMAGE;
        state.returns_partial_module = mode == 0;
        state.returns_null_module = mode == 1;
        const auto result = session.get_function(module.id, "kernel");
        expect(result.original_status == CUDA_SUCCESS && result.status != CUDA_SUCCESS &&
                   session.snapshot().is_quarantined && session.snapshot().modules == 1,
               "partial/NULL-success private JIT retains both domains without pretending original "
               "lookup failed");
        expect(!session.unload_module(module.id).original_status && state.unloads == 0,
               "uncertain private cleanup is never retried or relabeled as original unload");
        success(session.shutdown());
        expect(state.context_destroys == 1 && state.host_frees == 1,
               "only real owned-context teardown settles ambiguous private GPU modules");
    }
}
void context_control_ownership() {
    namespace fixture = glimmer::tests::context_control;
    using Operation = CudaContextControl;
    state = {};
    auto sdk = dispatch();
    sdk.context = fixture::dispatch([] { return state.current; });
    sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
        *uuid = {};
        uuid->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    auto profiles = std::make_shared<CudaProfileCache>();
    CudaExecutionSession session({1, 2}, 7, sdk, {}, {}, {}, profiles);
    expect(!session.context_control(Operation::kGetLimit).original_status,
           "context controls never initialize a caller or owner context implicitly");
    success(session.initialize(0));
    const auto module = session.load_ptx(kPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto memory = session.allocate(8);
    success(memory);
    expect(profiles->snapshot().functions == 1, "initial context has a bound profile identity");
    for (bool runtime : {false, true}) {
        const auto control = [&](Operation operation, std::uint32_t argument = 0,
                                 std::uint64_t value = 0) {
            return session.context_control(operation, argument, value, runtime);
        };
        expect(control(Operation::kSetLimit, 0, UINT64_MAX - 17).original_status == 0 &&
                   control(Operation::kGetLimit).value == UINT64_MAX - 17 &&
                   control(Operation::kSetCache, 2).original_status == 0 &&
                   control(Operation::kGetCache).value == 2 &&
                   control(Operation::kSetSharedMemory, 1).original_status == 0 &&
                   control(Operation::kGetSharedMemory).value == 1 &&
                   control(Operation::kResetPersistingCache).original_status == 0,
               "full-width limits and native cache/shared-memory controls round-trip");
        expect(control(Operation::kGetLimit, 7).original_status == 215 &&
                   control(Operation::kSetSharedMemory, 3).original_status == 1 &&
                   !control(Operation::kGetLimit, UINT32_MAX).original_status &&
                   !control(Operation::kSetCache, UINT32_MAX).original_status &&
                   !control(Operation::kGetCache, 1).original_status &&
                   !control(Operation::kGetCache, 0, 1).original_status,
               "native enum errors remain original; unrepresentable or noncanonical wire rejects");
        auto wrong_thread =
            std::async(std::launch::async, [&] { return control(Operation::kGetLimit); });
        expect(!wrong_thread.get().original_status, "foreign thread never reaches control SDK");
    }
    fixture::state.driver_error = CUDA_ERROR_INVALID_CONTEXT;
    fixture::state.runtime_error = cudaErrorInvalidDevice;
    const auto driver_error = session.context_control(Operation::kGetLimit);
    const auto runtime_error = session.context_control(Operation::kGetLimit, 0, 0, true);
    expect(driver_error.original_status == CUDA_ERROR_INVALID_CONTEXT && !driver_error.value &&
               runtime_error.original_status == cudaErrorInvalidDevice && !runtime_error.value,
           "original Driver and Runtime errors remain distinct and do not publish touched outputs");
    fixture::state.driver_error = CUDA_SUCCESS;
    fixture::state.runtime_error = cudaSuccess;
    expect(session.context_control(Operation::kGetApiVersion).value == 3020 &&
               session.context_control(Operation::kGetAffinity).value == 80 &&
               session.context_control(Operation::kGetPriorityRange).auxiliary == UINT32_MAX - 1,
           "native explicit context, affinity and signed priority queries");
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 19;
    std::memcpy(parameters.data(), &memory.address, sizeof(memory.address));
    std::memcpy(parameters.data() + 8, &value, sizeof(value));
    const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                    .grid = {80, 1, 1},
                                                    .block = {32, 1, 1},
                                                    .parameters = parameters});
    expect(accepted.accepted() && session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt),
           "actual pending command");
    const auto calls = fixture::state.driver_calls + fixture::state.runtime_calls;
    expect(session.context_control(Operation::kSetCache, 1).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.context_control(Operation::kGetLimit, 0, 0, true).disposition ==
                   CudaSessionDisposition::kComplete &&
               fixture::state.driver_calls + fixture::state.runtime_calls == calls + 1,
           "pending work defers mutations without turning read-only queries into GPU barriers");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1 &&
               session.context_control(Operation::kSetCache, 1).original_status == 0 &&
               profiles->snapshot().cases == 0 && profiles->snapshot().samples == 0,
           "actual physical retirement unblocks configuration exactly once");
    expect(!profiles->snapshot().hits,
           "changed controls cannot reuse or train initial-context profiles");
    success(session.shutdown());
    expect(!session.context_control(Operation::kGetLimit).original_status,
           "retired context cannot be reconfigured");
}
void context_flag_ownership() {
    for (const bool persistent : {false, true}) {
        state = {};
        state.is_persistent = persistent;
        CudaSessionLimits limits;
        limits.submissions.max_commands = 1;
        CudaExecutionSession session({1, 2}, 7, dispatch(), limits, {persistent ? 320U : 0U});
        expect(session.initialize(0, {3}).status == CUDA_ERROR_INVALID_VALUE && state.calls == 0 &&
                   session.initialize(0, {CU_CTX_COREDUMP_ENABLE}).status ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   state.calls == 0,
               "invalid/unauthorized context flags reject before native initialization");
        success(session.initialize(0, {CU_CTX_SCHED_YIELD}));
        const auto original_flags = static_cast<unsigned int>(CU_CTX_SCHED_YIELD) |
                                    (persistent ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U);
        const auto query = session.get_context_flags();
        expect(query.original_status == CUDA_SUCCESS && query.id == original_flags,
               "native creation preserves requested flags and private map-host support");
        state.flag_status = CUDA_ERROR_NOT_SUPPORTED;
        expect(session.set_context_flags(CU_CTX_SCHED_SPIN).original_status ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   state.context_flags == original_flags,
               "actual flag failure stays native and does not mutate metadata");
        state.flag_status = CUDA_SUCCESS;
        success(session.set_context_flags(CU_CTX_SCHED_SPIN));
        expect(
            state.context_flags == (CU_CTX_SCHED_SPIN |
                                    (persistent ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U)),
            "changing wait policy cannot remove private mapped progress support");
        const auto before = state.flag_updates;
        expect(!session.set_context_flags(3).original_status &&
                   !session.set_context_flags(CU_CTX_USER_COREDUMP_ENABLE).original_status &&
                   state.flag_updates == before,
               "invalid flag updates do not issue a target SDK call");
        auto foreign = std::async(std::launch::async, [&] {
            return !session.get_context_flags().original_status &&
                   !session.set_context_flags(0).original_status;
        });
        expect(foreign.get() && state.flag_updates == before,
               "wrong owner cannot query or change native flags");
        success(session.shutdown());
    }
    state = {};
    auto optional = dispatch();
    optional.context_get_flags = nullptr;
    optional.context_set_flags = nullptr;
    CudaExecutionSession session({1, 2}, 7, optional);
    success(session.initialize(0));
    expect(session.get_context_flags().status == CUDA_ERROR_NOT_SUPPORTED &&
               session.set_context_flags(0).status == CUDA_ERROR_NOT_SUPPORTED,
           "optional flag dispatch absence does not break existing native sessions");
    success(session.shutdown());
}

void pending_commands_revalidate_actual_context_control_attempts() {
    for (const bool persistent : {false, true}) {
        state = {};
        state.is_persistent = persistent;
        CudaExecutionSession session({1, 2}, 7, dispatch(), {}, {persistent ? 320U : 0U});
        success(session.initialize(0));
        const auto module = session.load_ptx(persistent ? kPersistentPtx : kPtx);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        const auto memory = session.allocate(8);
        success(function);
        success(memory);
        std::array<std::byte, 12> parameters{};
        const std::uint32_t value = 19;
        std::memcpy(parameters.data(), &memory.address, sizeof(memory.address));
        std::memcpy(parameters.data() + 8, &value, sizeof(value));
        const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                        .grid = {80, 1, 1},
                                                        .block = {32, 1, 1},
                                                        .parameters = parameters});
        const auto calls = state.calls;
        const auto pending = session.pending_command();
        expect(accepted.accepted() && pending.command && state.calls == calls &&
                   pending.command.description().control_epoch == 1 &&
                   !pending.command.description().member_generation &&
                   !pending.command.description().physical_configuration_epoch,
               "isolated pending capture is SDK-free and has no fabricated shared-domain epochs");
        const auto getter = session.get_context_flags();
        expect(getter.status == CUDA_SUCCESS && !session.set_context_flags(3).original_status &&
                   session.pending_command().command.description() == pending.command.description(),
               "getter and clean invalid flag rejection cannot invalidate an actual pending grant");
        state.flag_status = CUDA_ERROR_NOT_SUPPORTED;
        expect(session.set_context_flags(CU_CTX_SCHED_YIELD).original_status ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   session.pending_command().command.description().control_epoch == 2,
               "an actual SDK mutation error still invalidates the old control revision");
        const auto before_stale = state.calls;
        expect(session.issue(pending.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
                   state.calls == before_stale && !state.launches,
               "stale context configuration rejects before all SDK proof and command issue");
        state.flag_status = CUDA_SUCCESS;
        success(session.set_context_flags(CU_CTX_SCHED_SPIN));
        const auto current = session.pending_command();
        auto foreign = std::async(std::launch::async, [&] {
            return session.pending_command().status == CUDA_ERROR_INVALID_CONTEXT &&
                   session.issue(current.command).native_status == CUDA_ERROR_INVALID_CONTEXT;
        });
        expect(foreign.get() && current.command.description().control_epoch == 3,
               "nonrecycled Session owner checks protect exact pending commands");
        state.claimed = 80;
        state.query_status = CUDA_SUCCESS;
        expect(session.issue(current.command).state == CudaExecutionIssueState::kIssued &&
                   session.acknowledge_result(accepted.receipt) && session.poll().retired == 1 &&
                   !session.pending_command().command,
               "fresh exact issue preserves native/persistent completion and once-only host reply");
        success(session.shutdown());
    }
}
struct PendingPairResourceFixture {
    unsigned int device_queries{}, function_queries{};
    CUresult status{CUDA_SUCCESS};
    std::function<void()> on_query;
} pending_pair_resource;

CudaSessionDispatch pending_pair_dispatch() {
    auto sdk = dispatch();
    sdk.device_uuid = [](CUuuid* uuid, CUdevice) {
        *uuid = {};
        uuid->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    sdk.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice device) {
        switch (attribute) {
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
                return device_attribute(output, attribute, device);
        }
        ++state.calls;
        ++pending_pair_resource.device_queries;
        return CUDA_SUCCESS;
    };
    sdk.function_get_attribute = [](int* output, CUfunction_attribute attribute,
                                    CUfunction function) {
        ++state.calls;
        if (attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
            *output = -1;
            return CUDA_SUCCESS;
        }
        expect(private_kind(function).has_value(),
               "pair resources come from the actual private function");
        ++pending_pair_resource.function_queries;
        if (pending_pair_resource.on_query) pending_pair_resource.on_query();
        *output = attribute == CU_FUNC_ATTRIBUTE_NUM_REGS ? 17 : 128;
        expect(attribute == CU_FUNC_ATTRIBUTE_NUM_REGS ||
                   attribute == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
               "exact requested register/static-shared native metadata");
        return pending_pair_resource.status;
    };
    return sdk;
}

void pending_pair_catalog_binds_actual_resources_and_control_epochs() {
    state = {};
    state.is_persistent = true;
    pending_pair_resource = {};
    CudaProfileConfig config;
    config.workers = {320};
    config.warmup_calls = 0;
    config.samples_per_candidate = 1;
    const CudaProfileGeometry geometry{{80, 1, 1}, {32, 1, 1}, 64, Kind::kPreemptive};
    CudaProfileCache learning(config, profile_clock);
    const auto identity = learning.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                        kPersistentPtx, {}, "kernel");
    for (const auto duration : {100U, 80U}) {
        auto choice = learning.choose(identity, geometry, 320, true);
        expect(choice.observation != nullptr,
               "train actual single-cache original/private identities");
        profile_now = 100;
        choice.observation->begin();
        profile_now += duration;
        choice.observation->finish(true);
    }
    const auto archive = learning.encode("pending-pair-fixture");
    auto profiles = std::make_shared<CudaProfileCache>(config, profile_clock, false);
    expect(archive && profiles->restore(*archive, "pending-pair-fixture"),
           "pair metadata resolves an exact qualified readonly original identity");
    CudaExecutionSession session({1, 2}, 7, pending_pair_dispatch(), {}, {320}, {}, profiles);
    success(session.initialize(0));
    const auto module = session.load_ptx(kPersistentPtx);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    const auto memory = session.allocate(8);
    success(function);
    success(memory);
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 19;
    std::memcpy(parameters.data(), &memory.address, sizeof(memory.address));
    std::memcpy(parameters.data() + 8, &value, sizeof(value));
    const auto accepted =
        session.accept_kernel(1, {.function = function.id,
                                  .grid = geometry.grid,
                                  .block = geometry.block,
                                  .shared_memory_bytes = geometry.shared_memory_bytes,
                                  .parameters = parameters});
    const auto pending = session.pending_command();
    expect(accepted.accepted() && pending.command.description().command.is_persistent,
           "exact actual persistent record precedes resource preparation");
    pending_pair_resource.status = CUDA_ERROR_NOT_SUPPORTED;
    expect(session.prepare_pair_kernel(pending.command).status == CUDA_ERROR_NOT_SUPPORTED &&
               !session.snapshot().is_quarantined && !state.launches,
           "original metadata error invents no capacity, execution or quarantine");
    pending_pair_resource.status = CUDA_SUCCESS;
    pending_pair_resource.on_query = [&] {
        const auto calls = state.calls;
        expect(session.pending_command().status == CUDA_ERROR_NOT_READY &&
                   session.prepare_pair_kernel(pending.command).status == CUDA_ERROR_NOT_READY &&
                   session.select_pending_kernel(pending.command, {.workers = 80}).status ==
                       CUDA_ERROR_NOT_READY &&
                   state.calls == calls,
               "actual resource getters exclude nested owner work without SDK recursion");
    };
    const auto prepared = session.prepare_pair_kernel(pending.command);
    pending_pair_resource.on_query = {};
    const auto data = prepared.kernel.description();
    expect(prepared.status == CUDA_SUCCESS && prepared.kernel && data.catalog_generation == 1 &&
               data.pending == pending.command.description() && data.geometry == geometry &&
               data.limits == CudaPairProfileLimits{80, 32, 2048, 65536, 320, 98304} &&
               data.variant == CudaPairProfileVariant{Kind::kPreemptive, 320, 17, 4, 128} &&
               data.profile && !data.execution_domain && !state.launches && !state.prefix_copies,
           "catalog binds genuine readonly code identity, geometry, hardware and private resource "
           "queries");
    CudaPairProfileCache pairs(profiles, {}, profile_clock, false);
    expect(static_cast<bool>(pairs.bind(data.profile, data.geometry)),
           "actual prepared function resolves the paired cache's qualified original key");
    const auto calls = state.calls;
    const auto selected = session.select_pending_kernel(pending.command, {.workers = 160});
    const auto refreshed = session.refresh_pair_kernel(prepared.kernel, selected.command);
    expect(selected.status == CUDA_SUCCESS && refreshed.status == CUDA_SUCCESS &&
               refreshed.kernel.description().catalog_generation == 1 &&
               refreshed.kernel.description().variant.workers == 160 && state.calls == calls &&
               session.prepare_pair_kernel(pending.command).status == CUDA_ERROR_INVALID_HANDLE,
           "geometry-only refresh keeps actual resource facts without another SDK query");
    success(session.set_context_flags(CU_CTX_SCHED_YIELD));
    const auto changed = session.pending_command();
    const auto after_control = state.calls;
    expect(session.refresh_pair_kernel(refreshed.kernel, changed.command).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               state.calls == after_control,
           "a control transition cannot reuse an old native resource catalog");
    const auto unqualified = session.prepare_pair_kernel(changed.command);
    expect(
        unqualified.status == CUDA_SUCCESS && unqualified.kernel &&
            unqualified.kernel.description().catalog_generation == 2 &&
            !unqualified.kernel.description().profile,
        "changed context keeps real resources but cannot borrow initial trained profile identity");
    const auto original = session.select_pending_kernel(changed.command, {.uses_original = true});
    const auto original_metadata =
        session.refresh_pair_kernel(unqualified.kernel, original.command);
    expect(original.status == CUDA_SUCCESS && original_metadata.status == CUDA_SUCCESS &&
               !original_metadata.kernel.description().pending.command.is_persistent &&
               !original_metadata.kernel.description().variant.workers && !state.prefix_copies,
           "serial fallback changes representation before issue, not logical identity or private "
           "initialization");
    state.query_status = CUDA_SUCCESS;
    expect(session.issue(original.command).state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt) && session.poll().retired == 1 &&
               state.launched_kinds == std::vector<std::optional<Kind>>{std::nullopt} &&
               !state.prefix_copies && !state.readbacks,
           "selected original executes its exact saved public ABI without private replay");
    std::uint32_t actual{};
    std::memcpy(&actual, state.memory.at(memory.address).data(), sizeof(actual));
    expect(actual == value, "complete original result survives resource preparation and selection");
    success(session.shutdown());
    expect(!prepared.kernel && !refreshed.kernel && !original_metadata.kernel,
           "CPU profile catalogs do not pin retired native resources through shutdown");
}

void pending_pair_resource_proof_failures_are_retained() {
    for (const bool throws : {false, true}) {
        const auto child = fork();
        expect(child >= 0, "fork retained pair resource proof failure");
        if (!child) {
            try {
                state = {};
                state.is_persistent = true;
                pending_pair_resource = {};
                CudaExecutionSession session({1, 2}, 7, pending_pair_dispatch(), {}, {320});
                success(session.initialize(0));
                const auto module = session.load_ptx(kPersistentPtx);
                success(module);
                const auto function = session.get_function(module.id, "kernel");
                const auto memory = session.allocate(8);
                success(function);
                success(memory);
                std::array<std::byte, 12> parameters{};
                std::memcpy(parameters.data(), &memory.address, sizeof(memory.address));
                const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                                .grid = {80, 1, 1},
                                                                .block = {32, 1, 1},
                                                                .parameters = parameters});
                const auto pending = session.pending_command();
                expect(accepted.accepted() && pending.command,
                       "retain actual pending resource owner");
                pending_pair_resource.on_query = [throws] {
                    if (throws) throw std::runtime_error("injected native metadata exception");
                    state.current = nullptr;
                };
                const auto result = session.prepare_pair_kernel(pending.command);
                const auto calls = state.calls;
                expect(
                    result.status != CUDA_SUCCESS && !result.kernel &&
                        session.snapshot().is_quarantined &&
                        session.issue(pending.command).state == CudaExecutionIssueState::kError &&
                        session.prepare_pair_kernel(pending.command).status != CUDA_SUCCESS &&
                        state.calls == calls && !state.launches && !state.context_destroys,
                    "unproved/throwing resource getters quarantine, retain and never retry or "
                    "issue");
                _exit(0);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "retained resource-proof control has no fake cleanup or recovery");
    }
}

void owned_device_memory_ranges_and_retirement() {
    state = {};
    auto functions = dispatch();
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native SDK ABI.
    functions.execution.copy_device = [](CUdeviceptr destination, CUdeviceptr source,
                                         std::size_t bytes) {
        ++state.calls;
        ++state.copies;
        if (bytes) {
            auto to = state.memory.upper_bound(destination);
            auto from = state.memory.upper_bound(source);
            expect(to != state.memory.begin() && from != state.memory.begin(),
                   "native copy sees owned ranges");
            --to;
            --from;
            std::memmove(to->second.data() + (destination - to->first),
                         from->second.data() + (source - from->first), bytes);
        }
        return CUDA_SUCCESS;
    };
    functions.execution.fill32_async = [](CUdeviceptr destination, unsigned int bits,
                                          std::size_t count, CUstream stream) {
        ++state.calls;
        ++state.copies;
        expect(stream == reinterpret_cast<CUstream>(&state.stream_storage),
               "fill uses its exact owned stream");
        auto to = state.memory.upper_bound(destination);
        expect(to != state.memory.begin(), "native fill sees owned destination");
        --to;
        for (std::size_t index = 0; index < count; ++index)
            std::memcpy(to->second.data() + (destination - to->first) + index * sizeof(bits), &bits,
                        sizeof(bits));
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    CudaExecutionSession session({1, 2}, 7, functions);
    expect(!session.device_memory({1, 0}, 1),
           "uninitialized sessions cannot authenticate device references");
    success(session.initialize(0));
    const auto source = session.allocate(64), destination = session.allocate(64);
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(source);
    success(destination);
    success(stream);
    std::array<std::byte, 64> initial{}, result{};
    const auto metadata_calls = state.calls;
    const auto metadata = session.device_memory({source.id, 5}, 59);
    expect(metadata && metadata->address == source.address + 5 && metadata->bytes == 59 &&
               metadata->lifetime && !session.device_memory({source.id, 5}, 60) &&
               !session.device_memory({source.id, 64}, 1) && !session.device_memory({0, 0}, 1),
           "metadata references validate complete allocation generations and bounds");
    auto foreign_metadata =
        std::async(std::launch::async, [&] { return session.device_memory({source.id, 0}, 1); });
    expect(!foreign_metadata.get() && state.calls == metadata_calls,
           "CPU reference lookup never invokes CUDA or grants a foreign owner access");
    for (std::size_t index = 0; index < initial.size(); ++index)
        initial[index] = static_cast<std::byte>(index);
    success(session.upload({source.id, 0}, initial));
    CudaSessionMemory copy{
        .destination = {destination.id, 16}, .source = {source.id, 8}, .count = 24};
    auto invalid = copy;
    invalid.destination.offset = 60;
    const auto calls = state.calls;
    expect(!session.accept_memory(1, invalid).accepted(),
           "copy destination extent checked independently");
    invalid = copy;
    invalid.source.allocation = UINT64_MAX;
    expect(!session.accept_memory(1, invalid).accepted(),
           "copy cannot borrow an unknown allocation identity");
    invalid = copy;
    invalid.kind = CudaMemoryKind::kFill32;
    invalid.source = {};
    invalid.count = std::numeric_limits<std::size_t>::max();
    expect(!session.accept_memory(1, invalid).accepted(), "typed fill count cannot overflow bytes");
    invalid.count = 1;
    invalid.destination.offset = 1;
    expect(!session.accept_memory(1, invalid).accepted(),
           "typed fill alignment checked before native SDK work");
    expect(state.calls == calls,
           "rejected memory metadata invokes no SDK and consumes no sequence");
    const auto copied = session.accept_memory(1, copy);
    const auto filled = session.accept_memory(2, {.kind = CudaMemoryKind::kFill32,
                                                  .destination = {destination.id, 48},
                                                  .count = 4,
                                                  .value = 0x4BADC0DE,
                                                  .stream = stream.id,
                                                  .is_asynchronous = true});
    expect(copied.accepted() && filled.accepted() && state.calls == calls &&
               session.issue_next().sequence == 1 && session.issue_next().sequence == 2 &&
               session.acknowledge_result(copied.receipt) &&
               session.acknowledge_result(filled.receipt),
           "copy and fill pipeline through their actual SDK and retain separate physical events");
    expect(session.release(source.id).disposition == CudaSessionDisposition::kDeferred &&
               session.download({destination.id, 0}, result).disposition ==
                   CudaSessionDisposition::kDeferred &&
               session.poll().retained == 2,
           "host result delivery cannot free either endpoint or expose unfinished device output");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 2, "both real trailing events complete");
    success(session.download({destination.id, 0}, result));
    for (std::size_t index = 0; index < 24; ++index)
        expect(result[16 + index] == initial[8 + index], "complete original copied subrange");
    for (std::size_t index = 0; index < 4; ++index) {
        std::uint32_t bits{};
        std::memcpy(&bits, result.data() + 48 + index * sizeof(bits), sizeof(bits));
        expect(bits == 0x4BADC0DE, "typed fill preserves full scalar bits, not byte replication");
    }
    const auto empty = session.accept_memory(3, {});
    expect(empty.accepted() && session.issue_next().sequence == 3 && session.poll().retired == 1 &&
               session.acknowledge_result(empty.receipt),
           "zero-count/null copy preserves a real native SDK outcome");
    success(session.release(source.id));
    expect(!session.device_memory({source.id, 0}, 1) && metadata->lifetime,
           "retained CPU metadata is not authority to reuse a retired GPU allocation");
    success(session.release(destination.id));
    success(session.destroy_stream(stream.id));
    success(session.shutdown());
}

void context_preparation_does_not_claim_creation() {
    for (unsigned int stage = 0; stage < 3; ++stage) {
        state = {};
        auto functions = dispatch();
        if (stage == 0)
            functions.initialize = [](unsigned int) { return CUDA_ERROR_NOT_INITIALIZED; };
        if (stage == 1)
            functions.device_get = [](CUdevice*, int) { return CUDA_ERROR_INVALID_DEVICE; };
        if (stage == 2)
            functions.execution.context_get_current = [](CUcontext*) { return CUDA_ERROR_UNKNOWN; };
        CudaExecutionSession session({1, 2}, 7, functions);
        const auto result = session.initialize(0);
        expect(result.status != CUDA_SUCCESS && !result.original_status &&
                   state.context_creates == 0 && !session.snapshot().has_context &&
                   !session.snapshot().is_quarantined,
               "preparation SDK errors preserve their status without inventing a context creation "
               "call");
        success(session.shutdown());
    }
}
struct SharedBacking {
    int descriptor{-1};
    SharedBacking(bool should_seal = true) {
        descriptor = ::memfd_create("glimmer-host-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        expect(descriptor >= 0 && ::ftruncate(descriptor, 4096) == 0, "create shared backing");
        if (should_seal)
            expect(::fcntl(descriptor, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
                   "seal only backing extent, not shared writes");
    }
    ~SharedBacking() {
        if (descriptor >= 0 && ::close(descriptor) != 0) std::terminate();
    }
    SharedBacking(const SharedBacking&) = delete;
    SharedBacking& operator=(const SharedBacking&) = delete;
};
CUresult CUDAAPI register_shared_host(void* pointer, std::size_t bytes, unsigned int flags) {
    ++state.registrations;
    expect(pointer && bytes == 4096 &&
               flags == (CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
           "register exact owned portable mapped extent");
    if (state.registration_status == CUDA_SUCCESS) state.imported_host = pointer;
    return state.registration_status;
}
CUresult CUDAAPI unregister_shared_host(void* pointer) {
    ++state.unregistrations;
    expect(pointer && pointer == state.imported_host, "unregister exact native registration once");
    if (state.unregister_status == CUDA_SUCCESS) state.imported_host = nullptr;
    return state.unregister_status;
}
CUresult CUDAAPI map_shared_host(CUdeviceptr* address, void* pointer, unsigned int flags) {
    expect(!flags, "mapped-pointer query keeps original flags");
    if (pointer != state.imported_host) return state.unregistered_map_status;
    *address = reinterpret_cast<CUdeviceptr>(pointer) + (state.returns_unaligned_mapping ? 1U : 0U);
    return state.imported_map_status;
}
CudaSessionDispatch host_dispatch() {
    auto result = dispatch();
    result.host_register = register_shared_host;
    result.host_unregister = unregister_shared_host;
    result.host_map = map_shared_host;
    return result;
}
void shared_host_validation_dataflow_and_physical_leases() {
    state = {};
    SharedBacking backing;
    SharedBacking unsealed(false);
    CudaSessionLimits limits;
    limits.max_host_memory_bytes = 4096;
    CudaExecutionSession session({31, 32}, 33, host_dispatch(), limits);
    success(session.initialize(0));
    for (const auto& invalid :
         {session.import_host_memory(-1, 256, 4096, 0),
          session.import_host_memory(unsealed.descriptor, 256, 4096, 0),
          session.import_host_memory(backing.descriptor, 0, 4096, 0),
          session.import_host_memory(backing.descriptor, 4097, 4096, 0),
          session.import_host_memory(backing.descriptor, 256, 4095, 0),
          session.import_host_memory(backing.descriptor, 256, 8192, 0),
          session.import_host_memory(backing.descriptor, 256, 4096, 0x80000000U)})
        expect(invalid.status == CUDA_ERROR_INVALID_VALUE && !invalid.original_status,
               "malformed backing/ranges/flags reject before registration");
    expect(session.import_host_memory(backing.descriptor, 256, 4096, CU_MEMHOSTALLOC_WRITECOMBINED)
                       .status == CUDA_ERROR_NOT_SUPPORTED &&
               state.registrations == 0,
           "ordinary backing must not pretend write-combined native semantics");
    const auto allocation = session.allocate(256);
    success(allocation);
    const auto imported = session.import_host_memory(backing.descriptor, 256, 4096, 0);
    success(imported);
    expect(imported.id && imported.address && session.snapshot().host_memory_bytes == 4096 &&
               session.snapshot().host_allocations == 1 && state.registrations == 1,
           "native registration is charged by rounded extent and has a scoped identity");
    const auto full = session.import_host_memory(backing.descriptor, 256, 4096, 0);
    expect(full.status == CUDA_ERROR_OUT_OF_MEMORY && !full.original_status &&
               state.registrations == 1,
           "separate host quota rejects before another registration");
    expect(!session.host_memory({imported.id, 250}, 7) &&
               !session.host_memory({imported.id + 1, 0}, 1),
           "borrow validates live logical range");
    auto view = session.host_memory({imported.id, 0}, 256);
    expect(view && view->address == imported.address && view->lifetime,
           "borrow carries shared physical lifetime, not just a pointer");
    expect(!session.accept_host_transfer(1, {allocation.id + 1, 0}, *view, 0, true).accepted() &&
               !session.accept_host_transfer(1, {allocation.id, 1}, *view, 0, false).accepted() &&
               !session.accept_host_transfer(1, {allocation.id, 0}, *view, UINT64_MAX, true)
                    .accepted(),
           "invalid owned device range or stream cannot consume sequence or execute a native copy");
    void* client = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, backing.descriptor, 0);
    expect(client != MAP_FAILED, "second mapping observes actual shared backing");
    std::memset(client, 17, 256);
    expect(std::memcmp(client, view->pointer, 256) == 0, "both owners observe the same bytes");
    const auto first = session.accept_host_transfer(1, {allocation.id, 0}, *view, 0, true);
    expect(first.accepted() && session.issue_next().sequence == 1 &&
               session.host_result(first.receipt)->status == CUDA_SUCCESS &&
               session.acknowledge_result(first.receipt),
           "actual upload SDK result is separate");
    std::memset(state.memory.at(allocation.address).data(), 41, 256);
    const auto second = session.accept_host_transfer(2, {allocation.id, 0}, *view, 0, false);
    const auto third = session.accept_host_transfer(3, {allocation.id, 0}, *view, 0, true);
    expect(second.accepted() && third.accepted(), "accept dependent shared-buffer transfers");
    const auto deferred = session.release_host_memory(imported.id);
    expect(deferred.disposition == CudaSessionDisposition::kDeferred && !deferred.original_status &&
               state.unregistrations == 0 && !session.host_memory({imported.id, 0}, 1),
           "free seals new borrows and cannot unregister an accepted transfer");
    view.reset();
    for (const auto& receipt : {second.receipt, third.receipt}) {
        expect(session.issue_next().sequence == receipt.sequence() &&
                   session.host_result(receipt)->status == CUDA_SUCCESS &&
                   session.acknowledge_result(receipt),
               "ordered transfers issue exactly once");
    }
    for (std::size_t index = 0; index < 256; ++index)
        expect(static_cast<const unsigned char*>(client)[index] == 41 &&
                   state.memory.at(allocation.address)[index] == std::byte{41},
               "D2H followed by H2D reads the shared GPU-produced bytes, not an early snapshot");
    expect(session.poll().retired == 0 &&
               session.release_host_memory(imported.id).disposition ==
                   CudaSessionDisposition::kDeferred &&
               state.unregistrations == 0,
           "host SDK success and acknowledgment are not GPU stop");
    state.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 3, "all actual trailing events retire");
    success(session.release_host_memory(imported.id));
    expect(state.unregistrations == 1 && !session.snapshot().host_memory_bytes &&
               !session.snapshot().host_allocations && !state.imported_host,
           "last physical lease permits one unregister, unmap and quota release");
    expect(session.release_host_memory(imported.id).status == CUDA_ERROR_INVALID_VALUE &&
               state.unregistrations == 1,
           "duplicate free cannot reduce quota twice");
    const auto replacement = session.import_host_memory(backing.descriptor, 256, 4096, 0);
    success(replacement);
    expect(replacement.id > imported.id, "host resource identities never recycle");
    auto held = session.host_memory({replacement.id, 0}, 256);
    session.close();
    expect(!session.host_memory({replacement.id, 0}, 0) &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred &&
               state.context_destroys == 0,
           "context close retains a borrowed host registration");
    held.reset();
    success(session.shutdown());
    expect(state.unregistrations == 2 && state.context_destroys == 1,
           "registration retires before its owning context exactly once");
    expect(::munmap(client, 4096) == 0, "independent application mapping retires separately");
}
void shared_host_registration_failure_containment() {
    for (const auto failure :
         {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_SUPPORTED,
          CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED}) {
        state = {};
        state.registration_status = failure;
        SharedBacking backing;
        CudaExecutionSession session({41, 42}, 43, host_dispatch());
        success(session.initialize(0));
        const auto result = session.import_host_memory(backing.descriptor, 256, 4096, 0);
        expect(result.status == failure && result.original_status == failure &&
                   !session.snapshot().host_memory_bytes && !session.snapshot().is_quarantined &&
                   state.unregistrations == 0,
               "known empty native rejection safely rolls back");
        success(session.shutdown());
    }
    for (const bool is_unaligned : {false, true}) {
        state = {};
        state.returns_unaligned_mapping = is_unaligned;
        state.imported_map_status = is_unaligned ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
        SharedBacking backing;
        CudaExecutionSession session({61, 62}, 63, host_dispatch());
        success(session.initialize(0));
        const auto result = session.import_host_memory(backing.descriptor, 256, 4096, 0);
        expect(result.status == (is_unaligned ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_NOT_SUPPORTED) &&
                   result.original_status == CUDA_SUCCESS &&
                   result.disposition == CudaSessionDisposition::kRejected && !result.id &&
                   !session.snapshot().host_memory_bytes && !session.snapshot().is_quarantined &&
                   state.registrations == 1 && state.unregistrations == 1,
               "failed or malformed mapped-address preparation rolls back the known unpublished "
               "registration once");
        success(session.shutdown());
    }
    for (const bool fails_unregister : {false, true}) {
        const auto child = ::fork();
        expect(child >= 0, "fork isolated unknown-ownership control");
        if (child == 0) {
            try {
                state = {};
                SharedBacking backing;
                CudaExecutionSession session({51, 52}, 53, host_dispatch());
                success(session.initialize(0));
                if (fails_unregister) {
                    const auto imported =
                        session.import_host_memory(backing.descriptor, 256, 4096, 0);
                    success(imported);
                    state.unregister_status = CUDA_ERROR_UNKNOWN;
                    const auto result = session.release_host_memory(imported.id);
                    expect(result.original_status == CUDA_ERROR_UNKNOWN,
                           "preserve exact unregister failure");
                } else {
                    state.registration_status = CUDA_ERROR_UNKNOWN;
                    const auto result =
                        session.import_host_memory(backing.descriptor, 256, 4096, 0);
                    expect(result.original_status == CUDA_ERROR_UNKNOWN,
                           "preserve exact registration failure");
                }
                const auto attempts = state.unregistrations;
                expect(session.snapshot().host_memory_bytes == 4096 &&
                           session.snapshot().is_quarantined &&
                           session.shutdown().status != CUDA_SUCCESS &&
                           session.shutdown().status != CUDA_SUCCESS &&
                           state.unregistrations == attempts && state.context_destroys == 0,
                       "uncertain native registration retains backing/quota/context without "
                       "cleanup retry");
                ::_exit(EXIT_SUCCESS);
            } catch (...) {
                ::_exit(EXIT_FAILURE);
            }
        }
        int status{};
        expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                   WEXITSTATUS(status) == EXIT_SUCCESS,
               "unknown host ownership is contained");
    }
}
}  // namespace

void owned_module_global_ranges_and_retirement() {
    constexpr CUdeviceptr global_address = 0x900000;
    auto functions = dispatch();
    functions.module_global = [](CUdeviceptr* address, std::size_t* bytes, CUmodule module,
                                 const char* name) {
        expect(module == reinterpret_cast<CUmodule>(&state.module_storage) &&
                   state.current == native_context(),
               "global lookup runs on the original module/context owner");
        ++state.global_queries;
        *address = state.has_null_global ? 0 : CUdeviceptr{0x900000};
        *bytes = 16;
        if (std::string_view(name) == "missing") return CUDA_ERROR_NOT_FOUND;
        if (*address) state.memory.try_emplace(*address, 16);
        return CUDA_SUCCESS;
    };
    functions.module_unload = [](CUmodule module) {
        const auto result = unload_module(module);
        if (result == CUDA_SUCCESS) state.memory.erase(0x900000);
        return result;
    };
    {
        state = {};
        CudaSessionLimits limits;
        limits.max_objects = 1;
        CudaExecutionSession session({1, 2}, 7, functions, limits);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPtx);
        success(module);
        const auto global = session.get_global(module.id, "global");
        success(global);
        const auto alias = session.get_global(module.id, "alias");
        success(alias);
        expect(global.id == alias.id && global.address == global_address &&
                   global.parameter_bytes == 16 && session.snapshot().allocations == 1 &&
                   session.snapshot().device_bytes == 16 && state.allocations == 0,
               "global alias reuses one module range and charge even at the object limit");
        const auto missing = session.get_global(module.id, "missing");
        expect(missing.status == CUDA_ERROR_NOT_FOUND &&
                   missing.original_status == CUDA_ERROR_NOT_FOUND && !missing.id &&
                   session.snapshot().device_bytes == 16 && !session.snapshot().is_quarantined,
               "read-only native query error ignores partial metadata without resource creation");
        expect(session.release(global.id).status == CUDA_ERROR_INVALID_VALUE && !state.frees,
               "module globals cannot reach allocation free or reduce their charge");
        std::array<std::uint32_t, 4> values{3, 5, 7, 11}, copied{};
        success(session.upload({global.id, 0}, std::as_bytes(std::span{values})));
        success(session.download({global.id, 0}, std::as_writable_bytes(std::span{copied})));
        expect(values == copied, "discovered global supports checked original transfer dataflow");
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        std::array<std::byte, 12> parameters{};
        const std::uint32_t value = 41;
        std::memcpy(parameters.data(), &global.address, 8);
        std::memcpy(parameters.data() + 8, &value, 4);
        const auto accepted = session.accept_kernel(1, {.function = function.id,
                                                        .grid = {80, 1, 1},
                                                        .block = {32, 1, 1},
                                                        .parameters = parameters});
        const auto queries = state.global_queries;
        expect(
            accepted.accepted() && session.get_global(module.id, "global").status == CUDA_SUCCESS &&
                state.global_queries == queries + 1 &&
                session.unload_module(module.id).disposition == CudaSessionDisposition::kDeferred,
            "read-only global lookup does not drain queued work or release its module owner");
        expect(
            session.issue_next().state == CudaExecutionIssueState::kIssued &&
                session.poll().retained == 1 &&
                session.unload_module(module.id).disposition == CudaSessionDisposition::kDeferred &&
                !state.unloads,
            "original launch return cannot retire a global or unload its live module");
        success(session.get_global(module.id, "alias"));
        expect(
            session.snapshot().submissions.outstanding == 1 && state.global_queries == queries + 2,
            "read-only original metadata progresses while an actual native event is pending");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().retired == 1 && session.acknowledge_result(accepted.receipt),
               "actual physical completion retires global-bearing work");
        success(session.unload_module(module.id));
        expect(state.unloads == 1 && state.frees == 0 && session.snapshot().device_bytes == 0 &&
                   session.snapshot().allocations == 0 && !session.device_memory({global.id, 0}, 1),
               "module unload, not allocation free, retires the range and its exact charge");
        success(session.shutdown());
    }
    {
        state = {};
        CudaSessionLimits limits;
        limits.max_device_bytes = 8;
        CudaExecutionSession session({1, 2}, 7, functions, limits);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPtx);
        success(module);
        const auto global = session.get_global(module.id, "global");
        expect(
            global.status == CUDA_ERROR_OUT_OF_MEMORY && global.original_status == CUDA_SUCCESS &&
                session.snapshot().is_quarantined && session.snapshot().device_bytes == 16 &&
                !state.frees,
            "discovered over-budget storage remains charged and retained rather than undercounted");
        success(session.shutdown());
        expect(!session.snapshot().device_bytes && state.context_destroys == 1,
               "only actual context destruction clears unexposed over-budget storage");
    }
    {
        state = {};
        state.has_null_global = true;
        CudaExecutionSession session({1, 2}, 7, functions);
        success(session.initialize(0));
        const auto module = session.load_ptx(kPtx);
        success(module);
        expect(session.get_global(module.id, "global").status == CUDA_ERROR_UNKNOWN &&
                   session.snapshot().is_quarantined,
               "malformed original success does not publish an invalid device range");
        const auto queries = state.global_queries;
        expect(session.get_global(module.id, "global").status != CUDA_SUCCESS &&
                   state.global_queries == queries,
               "retained uncertain global metadata cannot retry a native target");
        success(session.shutdown());
    }
}
void immutable_event_diagnostic_identities() {
    state = {};
    auto functions = dispatch();
    functions.stream_wait_event = [](CUstream, CUevent, unsigned int flags) {
        expect(!flags, "diagnostic fixture preserves the exact native wait flags");
        ++state.calls;
        ++state.event_waits;
        return CUDA_SUCCESS;
    };
    CudaExecutionSession source({1, 2}, 7, functions);
    success(source.initialize(0));
    CudaExecutionFrontier expired_frontier;
    {
        const auto event = source.create_event(CU_EVENT_DISABLE_TIMING);
        success(event);
        const auto initial = source.capture_event(event.id);
        expect(initial.status == CUDA_SUCCESS && initial.generation &&
                   *initial.generation == CudaSessionEventIdentity{event.id, 0, 0} &&
                   initial.event.diagnostic_identity() == initial.generation &&
                   !initial.dependency && !CudaSessionEvent{}.diagnostic_identity(),
               "never-recorded generation has an explicit immutable object identity");
        const auto unrecorded = source.wait_event_with_frontier(1, event.id, 0);
        const auto first = source.record_event_with_frontier(2, event.id, 0);
        const auto old = source.capture_event(event.id);
        const auto local = source.wait_event_with_frontier(3, event.id, 0);
        const auto second = source.record_event_with_frontier(4, event.id, 0);
        const auto current = source.capture_event(event.id);
        expect(unrecorded.result.status == CUDA_SUCCESS &&
                   unrecorded.generation == initial.generation &&
                   unrecorded.frontier.diagnostic_id() && first.result.status == CUDA_SUCCESS &&
                   first.generation && first.generation->event_object == event.id &&
                   first.generation->record_sequence == 2 &&
                   first.generation->frontier_capture == first.frontier.diagnostic_id() &&
                   first.generation->frontier_capture && old.generation == first.generation &&
                   local.result.status == CUDA_SUCCESS && local.generation == first.generation &&
                   second.result.status == CUDA_SUCCESS && second.generation &&
                   second.generation->record_sequence == 4 &&
                   second.generation->frontier_capture == second.frontier.diagnostic_id() &&
                   second.generation->frontier_capture != first.generation->frontier_capture &&
                   current.generation == second.generation && !state.event_records &&
                   !state.event_waits,
               "each admission reports its actual selected generation before any original issue");
        const auto calls = state.calls;
        expect(old.event.diagnostic_identity() == first.generation &&
                   initial.event.diagnostic_identity() == initial.generation &&
                   current.event.diagnostic_identity() == second.generation && state.calls == calls,
               "identity reads do not call SDK, recapture or observe readiness");
        auto wrong = std::async(std::launch::async, [&] {
            return !old.event.diagnostic_identity() && !first.frontier.diagnostic_id();
        });
        expect(wrong.get() && state.calls == calls,
               "foreign diagnostic reads reject before native callbacks");
        const auto child = ::fork();
        expect(child >= 0, "fork immutable event diagnostic control");
        if (!child)
            ::_exit(!old.event.diagnostic_identity() && !first.frontier.diagnostic_id() &&
                            state.calls == calls
                        ? 0
                        : 1);
        int status{};
        expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                   WEXITSTATUS(status) == 0 && state.calls == calls,
               "fork cannot read inherited event diagnostic identities");

        success(source.deactivate());
        CudaExecutionSession target({3, 4}, 8, functions);
        success(target.initialize(0));
        const auto foreign = target.wait_event_with_frontier(1, old.event, 0);
        expect(foreign.result.status == CUDA_SUCCESS && foreign.generation == first.generation &&
                   foreign.frontier.diagnostic_id(),
               "foreign wait snapshots the borrowed old generation, not the source's rerecord");
        success(target.deactivate());
        success(source.activate());
        success(source.destroy_event(event.id));
        expect(old.event.diagnostic_identity() == first.generation &&
                   current.event.diagnostic_identity() == second.generation &&
                   source.capture_event(event.id).status == CUDA_ERROR_INVALID_HANDLE,
               "logical event deletion does not rewrite retained generation identities");
        unsigned int reentries{};
        state.on_event_record = [&] {
            ++reentries;
            const auto before = state.calls;
            expect(!old.event.diagnostic_identity() && !first.frontier.diagnostic_id() &&
                       state.calls == before,
                   "native callback reentry sees no diagnostic identity or nested SDK call");
        };
        expect(source.issue_next().sequence == 1 && source.issue_next().sequence == 2 && reentries,
               "diagnostic guards leave both original ordering calls intact");
        state.on_event_record = {};
        expect(!first.frontier.retirement().is_retired &&
                   old.event.diagnostic_identity() == first.generation,
               "a valid diagnostic identity is not completion proof");
        success(source.deactivate());
        success(target.activate());
        const auto waits = state.event_waits;
        expect(target.issue_next().state == CudaExecutionIssueState::kIdle &&
                   state.event_waits == waits,
               "foreign old-generation wait still honors actual source completion");
        success(target.deactivate());
        success(source.activate());
        state.query_status = CUDA_SUCCESS;
        expect(source.poll().retired == 2 && first.frontier.retirement().is_retired &&
                   !second.frontier.retirement().is_retired,
               "old source generation retires independently of its queued rerecord");
        success(source.deactivate());
        success(target.activate());
        expect(target.issue_next().sequence == 1 && target.poll().retired == 1 &&
                   state.event_waits == waits + 1 && foreign.generation == first.generation,
               "foreign wait uses exactly its old generation while the newer record stays queued");
        success(target.shutdown());
        success(source.activate());
        expect(source.issue_next().sequence == 3 && source.issue_next().sequence == 4 &&
                   source.poll().retired == 2 && second.frontier.retirement().is_retired,
               "later local wait and rerecord retain their ordinary sequence and exact stop");
        const auto replacement = source.create_event(CU_EVENT_DISABLE_TIMING);
        success(replacement);
        const auto fresh = source.capture_event(replacement.id);
        expect(replacement.id != event.id && fresh.generation &&
                   *fresh.generation == CudaSessionEventIdentity{replacement.id, 0, 0},
               "new event objects never recycle an old generation identity");
        success(source.destroy_event(replacement.id));
        source.close();
        const auto closing_calls = state.calls;
        expect(old.event.diagnostic_identity() == first.generation &&
                   first.frontier.diagnostic_id() == first.generation->frontier_capture &&
                   state.calls == closing_calls,
               "admission close does not invalidate a still-owned immutable diagnostic identity");
        expired_frontier = first.frontier;
    }
    expect(source.poll().status == CUDA_SUCCESS &&
               std::ranges::none_of(state.events, [](const auto& event) { return event.is_live; }),
           "last borrow releases every actual native generation exactly once");
    success(source.shutdown());
    expect(state.context_destroys == 2 && !expired_frontier.diagnostic_id(),
           "actual Session shutdown expires its dispatcher identity without reusing the capture");
}

void event_diagnostics_preserve_full_width_sequences() {
    state = {};
    CudaSessionLimits limits;
    limits.first_resource_id = UINT64_MAX;
    CudaExecutionSession session({9, 10, UINT64_MAX}, 17, dispatch(), limits);
    success(session.initialize(0));
    {
        const auto event = session.create_event(CU_EVENT_DISABLE_TIMING);
        success(event);
        const auto initial = session.capture_event(event.id);
        const auto recorded = session.record_event_with_frontier(UINT64_MAX, event.id, 0);
        const auto current = session.capture_event(event.id);
        expect(event.id == UINT64_MAX && initial.generation &&
                   *initial.generation == CudaSessionEventIdentity{UINT64_MAX, 0, 0} &&
                   recorded.result.status == CUDA_SUCCESS && recorded.generation &&
                   recorded.generation->event_object == UINT64_MAX &&
                   recorded.generation->record_sequence == UINT64_MAX &&
                   recorded.generation->frontier_capture == recorded.frontier.diagnostic_id() &&
                   current.generation == recorded.generation,
               "diagnostics preserve complete unsigned event and ordinary sequence values");
        const auto calls = state.calls;
        expect(session.create_event(0).status == CUDA_ERROR_OUT_OF_MEMORY &&
                   state.calls == calls + 1 &&
                   current.event.diagnostic_identity() == recorded.generation,
               "resource exhaustion preserves the required context proof without a new event or "
               "generation");
        expect(session.issue_next().sequence == UINT64_MAX,
               "diagnostics leave the final valid ordinary sequence executable");
        state.query_status = CUDA_SUCCESS;
        expect(session.poll().retired == 1 && recorded.frontier.retirement().is_retired,
               "only actual event completion settles the full-width record");
        success(session.destroy_event(event.id));
    }
    expect(session.poll().status == CUDA_SUCCESS,
           "released full-width generation borrows permit native cleanup");
    success(session.shutdown());
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--workload-agnostic") {
            workload_agnostic_configuration();
            workload_agnostic_selection();
            std::cout << "Workload-agnostic resource selection checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--preparation-observer") {
            preparation_observer();
            std::cout << "Bounded preparation observer checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--large-image-policy") {
            large_source_policy_preserves_session_ownership();
            binary_owned_images_and_private_modules();
            std::cout << "Large-image owned preparation policy checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--input-gradient-split-k") {
            owned_input_gradient_split_k_and_resume();
            owned_input_gradient_version_and_existing_routes();
            owned_input_gradient_cancellation_abi_and_callbacks();
            std::cout << "Owned input-gradient split-K checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--lt-split-k") {
            owned_lt_split_k_selection_and_resume();
            owned_lt_split_k_fallback_and_provenance();
            owned_lt_split_k_cancellation_and_abi_rejection();
            std::cout << "Owned Lt split-K selection and lifetime checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--library-lifetime-review") {
            owned_library_identity_and_partial_shutdown();
            std::cout << "Native library lifetime review checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--library-shutdown-review") {
            owned_library_identity_and_partial_shutdown(2);
            std::cout << "Native library partial-shutdown review checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--library-registry-review") {
            owned_library_registry_duplicates();
            std::cout << "Native library descriptor identity review checks passed\n";
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown owned-session test mode");
        preparation_observer();
        immutable_event_diagnostic_identities();
        event_diagnostics_preserve_full_width_sequences();
        complete_owned_session();
        owned_module_global_ranges_and_retirement();
        session_owned_persistent_preparation_and_reuse();
        session_owned_persistent_preparation_and_reuse({}, true);
        sliced_worker_retarget_preserves_exact_parameter_updates();
        selected_ptx_families_preserve_logical_completion();
        once_only_profile_selects_actual_worker_geometry();
        once_only_profile_selects_actual_worker_geometry(Kind::kSliced);
        profile_selects_distinct_native_families();
        restored_profiles_preserve_ordinary_execution();
        workload_agnostic_configuration();
        workload_agnostic_selection();
        stateful_modules_share_original_native_storage();
        module_static_failure_refunds();
        cached_preparation_preserves_native_ownership_and_budget();
        large_source_policy_preserves_session_ownership();
        persistent_factory_rejections_and_prefix_failure();
        private_pool_ambiguity_never_retries_cleanup();
        persistent_configuration_and_private_quota_bounds();
        persistent_function_carveout_preparation();
        bounds_abi_and_nonreuse();
        ambiguous_release_and_partial_initialization();
        owner_fork_and_exhaustion();
        dead_owner_and_context_destroy_ambiguity();
        module_and_stream_partial_results();
        binary_owned_images_and_private_modules();
        binary_owned_images_and_private_modules(std::make_shared<CudaPtxCache>());
        context_flag_ownership();
        pending_commands_revalidate_actual_context_control_attempts();
        pending_pair_catalog_binds_actual_resources_and_control_epochs();
        pending_pair_resource_proof_failures_are_retained();
        context_control_ownership();
        context_preparation_does_not_claim_creation();
        owned_device_memory_ranges_and_retirement();
        owned_blas_issue_ranges_and_retirement();
        owned_random_lifecycle();
        owned_lt_handles_and_physical_retirement();
        owned_lt_partial_failure_containment();
        owned_lt_split_k_selection_and_resume();
        owned_lt_split_k_fallback_and_provenance();
        owned_lt_split_k_cancellation_and_abi_rejection();
        owned_library_identity_and_partial_shutdown();
        owned_library_registry_duplicates();
        owned_blas_transpose_and_thread_contract();
        owned_blas_partial_failures();
        owned_blas_device_scalars_and_native_attributes();
        owned_blas_workspace_and_policy_lifetimes();
        workspace_mutation_failure_never_replays();
        private_workspace_and_sm_policy_selection();
        owned_persistent_sgemm_selection_and_resume();
        owned_input_gradient_split_k_and_resume();
        owned_input_gradient_version_and_existing_routes();
        owned_input_gradient_cancellation_abi_and_callbacks();
        owned_gemm_ex_raw_scalars_and_native_status();
        owned_persistent_gemm_ex_selection_and_resume();
        owned_strided_gemm_ex_native_hulls_and_status();
        owned_strided_gemm_ex_persistent_mapping_and_resume();
        owned_strided_gemm_ex_native_hulls_and_status(true);
        owned_strided_gemm_ex_persistent_mapping_and_resume(true);
        contradictory_blas_metadata_is_not_resource_authority();
        shared_host_validation_dataflow_and_physical_leases();
        shared_host_registration_failure_containment();
        std::cout << "Server-owned CUDA session checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
