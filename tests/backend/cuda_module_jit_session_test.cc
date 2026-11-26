#include "cuda_execution_domain_fixture.h"
#include "../transform/cuda_image_fixture.h"

#include <bit>
#include <string>

namespace {
namespace fixture = glimmer::test::cuda_domain;
using namespace glimmer::backend::cuda;
using fixture::expect;
using fixture::success;
unsigned int mode{}, calls{}, unload_attempts{};
bool uses_binary{};
bool uses_names{};
std::string referenced_kernel{"kernel"};
const char** previous_kernel_names{};
std::vector<std::byte> original_binary;
std::vector<std::byte> names_blob(std::string_view name) {
    std::vector<std::byte> result(8 + name.size() + 1);
    const std::uint32_t count = 1, bytes = static_cast<std::uint32_t>(name.size() + 1);
    std::memcpy(result.data(), &count, 4);
    std::memcpy(result.data() + 4, &bytes, 4);
    std::memcpy(result.data() + 8, name.data(), name.size());
    return result;
}
std::vector<CudaModuleJitOption> options() {
    const auto bits = std::bit_cast<std::array<std::byte, 4>>(-99.5F);
    std::vector<CudaModuleJitOption> result{
        {CU_JIT_MAX_REGISTERS, 64, {}},
        {CU_JIT_THREADS_PER_BLOCK, 128, {}},
        {CU_JIT_WALL_TIME, 0, {bits.begin(), bits.end()}},
        {CU_JIT_INFO_LOG_BUFFER, 0, std::vector<std::byte>(64, std::byte{0xa5})},
        {CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, 64, {}}};
    if (uses_names) {
        result.push_back({CU_JIT_REFERENCED_KERNEL_NAMES, 1, names_blob(referenced_kernel)});
        result.push_back({CU_JIT_REFERENCED_KERNEL_COUNT, 1, {}});
        result.push_back({CU_JIT_REFERENCED_VARIABLE_NAMES, 1, names_blob("variable")});
        result.push_back({CU_JIT_REFERENCED_VARIABLE_COUNT, 1, {}});
    }
    return result;
}
CUresult CUDAAPI compile(CUmodule* output, const void* image, unsigned int count,
                         CUjit_option* keys, void** values) {
    ++calls;
    const bool prepared = calls % 2 == 0;
    if (prepared) {
        const std::string_view source(static_cast<const char*>(image));
        expect(source.starts_with(fixture::kSource) &&
                   source.find("__glimmer") != std::string_view::npos,
               "private compile uses the qualified source prefix plus generated entries");
    } else {
        expect(uses_binary ? std::memcmp(image, original_binary.data(), original_binary.size()) == 0
                           : std::string_view(static_cast<const char*>(image)) == fixture::kSource,
               "original SDK call receives exact unmodified caller image bytes");
    }
    expect(
        (count == (uses_names ? 9U : 5U) ||
         (!uses_names && count == 6 && keys[5] == CU_JIT_OPTIMIZATION_LEVEL &&
          (reinterpret_cast<std::uintptr_t>(values[5]) == 3 ||
           reinterpret_cast<std::uintptr_t>(values[5]) == 4))) &&
            keys[0] == CU_JIT_MAX_REGISTERS && keys[1] == CU_JIT_THREADS_PER_BLOCK &&
            keys[2] == CU_JIT_WALL_TIME && keys[3] == CU_JIT_INFO_LOG_BUFFER &&
            keys[4] == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES &&
            reinterpret_cast<std::uintptr_t>(values[0]) == 64 &&
            reinterpret_cast<std::uintptr_t>(values[1]) == 128 &&
            reinterpret_cast<std::uintptr_t>(values[2]) == std::bit_cast<std::uint32_t>(-99.5F) &&
            reinterpret_cast<std::uintptr_t>(values[4]) == 64 &&
            std::all_of(static_cast<const unsigned char*>(values[3]),
                        static_cast<const unsigned char*>(values[3]) + 64,
                        [](auto byte) { return byte == 0xa5; }),
        "every compile receives immutable original scalar, float and log inputs");
    if (uses_names) {
        auto** kernels = static_cast<const char**>(values[5]);
        auto** variables = static_cast<const char**>(values[7]);
        expect(keys[5] == CU_JIT_REFERENCED_KERNEL_NAMES &&
                   keys[6] == CU_JIT_REFERENCED_KERNEL_COUNT &&
                   keys[7] == CU_JIT_REFERENCED_VARIABLE_NAMES &&
                   keys[8] == CU_JIT_REFERENCED_VARIABLE_COUNT && kernels && variables &&
                   kernels != variables &&
                   (prepared ? kernels[1] && variables[1] && !kernels[2] && !variables[2] &&
                                   std::strcmp(kernels[1], "*__glimmer*") == 0 &&
                                   std::strcmp(variables[1], "*__glimmer*") == 0
                             : !kernels[1] && !variables[1]) &&
                   std::strcmp(kernels[0], referenced_kernel.c_str()) == 0 &&
                   std::strcmp(variables[0], "variable") == 0 &&
                   reinterpret_cast<std::uintptr_t>(values[6]) == (prepared ? 2U : 1U) &&
                   reinterpret_cast<std::uintptr_t>(values[8]) == (prepared ? 2U : 1U),
               "original uses exact caller names; private compile also retains generated entries");
        if (!prepared)
            previous_kernel_names = kernels;
        else
            expect(previous_kernel_names == kernels,
                   "private compile resets the same stable array storage");
    }
    const float elapsed = prepared ? 22.5F : 11.25F;
    std::memcpy(&values[2], &elapsed, sizeof(elapsed));
    std::memcpy(values[3], prepared ? "private" : "native", prepared ? 8 : 7);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    values[1] = reinterpret_cast<void*>(prepared ? 512 : 256);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    values[4] = reinterpret_cast<void*>(prepared ? 8 : 7);
    if (mode == 1 || (mode == 3 && prepared)) {
        ++fixture::driver.module_load_calls;
        return mode == 1 ? CUDA_ERROR_INVALID_PTX : CUDA_ERROR_OUT_OF_MEMORY;
    }
    if ((mode == 10 && prepared) || (mode == 11 && !prepared)) {
        ++fixture::driver.module_load_calls;
        return CUDA_SUCCESS;
    }
    const auto loaded = fixture::module_load(output, image);
    expect(loaded == CUDA_SUCCESS, "independent native handle factory");
    if (mode == 5 || (mode == 9 && prepared)) values[3] = nullptr;
    if (mode == 12 && !prepared)
        static_cast<const char**>(values[5])[0] = reinterpret_cast<const char*>(1);
    if (mode == 13 && prepared) const_cast<char*>(static_cast<const char**>(values[5])[0])[0] = '!';
    if (mode == 2 || (mode == 4 && prepared))
        return mode == 2 ? CUDA_ERROR_INVALID_PTX : CUDA_ERROR_OUT_OF_MEMORY;
    return CUDA_SUCCESS;
}
void verify_original_outputs(std::span<const CudaModuleJitOption> outputs) {
    expect(outputs.size() == (uses_names ? 9U : 5U) && outputs[0].value == 64 &&
               outputs[1].value == 256 && outputs[4].value == 7,
           "original SDK output slots survive private compile/reset");
    float time{};
    std::memcpy(&time, outputs[2].data.data(), sizeof(time));
    expect(time == 11.25F && outputs[3].data.size() == 64 &&
               std::memcmp(outputs[3].data.data(), "native", 7) == 0 &&
               std::ranges::all_of(std::span(outputs[3].data).subspan(7),
                                   [](auto byte) { return byte == std::byte{0xa5}; }),
           "original float, log prefix and all untouched tail bytes are exact");
    if (uses_names) {
        const auto input = options();
        expect(std::ranges::equal(outputs.subspan(5), std::span(input).subspan(5)),
               "input-only names retain original bytes, never native pointers or private changes");
    }
}
void exercise(bool binary, unsigned int scenario, bool named = false) {
    fixture::driver = {};
    mode = scenario;
    calls = 0;
    uses_binary = binary;
    uses_names = named;
    referenced_kernel = "kernel";
    unload_attempts = 0;
    auto dispatch = fixture::pair_dispatch();
    dispatch.module_load_ex = scenario == 6 ? nullptr : compile;
    if (scenario == 8)
        dispatch.module_unload = [](CUmodule) {
            ++unload_attempts;
            return CUDA_ERROR_UNKNOWN;
        };
    auto limits = fixture::limits(1000);
    if (scenario == 7) limits.max_image_bytes = fixture::kSource.size() + 1;
    CudaExecutionSession session({31, 37}, 41, dispatch, limits, {320});
    success(session.initialize(0));
    const std::array<glimmer::transform::CudaImageArgument, 2> layout{{{0, 8}, {8, 4}}};
    const auto image =
        glimmer::test::cuda_test_fatbin(glimmer::test::cuda_test_cubin(layout), fixture::kSource);
    const auto input = options();
    original_binary = image;
    std::vector<CudaModuleJitOption> outputs;
    const auto invoke = [&] {
        return binary ? session.load_binary_jit(image, input, &outputs)
                      : session.load_ptx_jit(fixture::kSource, input, &outputs);
    };
    const auto result = invoke();
    if (scenario == 6 || scenario == 7) {
        expect(!result.original_status && !result.id && !calls && !session.snapshot().modules &&
                   !session.snapshot().image_bytes && !session.snapshot().is_quarantined,
               "missing compiler callback or bounded storage rejects before any SDK load");
        expect(
            result.status == (scenario == 6 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_OUT_OF_MEMORY),
            "exact pre-entry policy status");
    } else {
        expect(result.original_status ==
                   (scenario == 1 || scenario == 2 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS),
               "original image result is independent of private compilation");
        if (scenario != 5 && scenario != 12)
            verify_original_outputs(outputs);
        else
            expect(outputs.empty(), "malformed native pointer cannot become fabricated outputs");
        if (scenario == 0 || scenario == 3) {
            success(result);
            expect(result.id && calls == 2 && !session.snapshot().is_quarantined,
                   "one original and one private compile, without option-free retries");
            success(session.get_function(result.id, "kernel"));
            expect(
                session.snapshot().persistent_functions == (scenario == 3 ? 0U : 1U),
                "known no-handle private rejection stays original, not false persistent selection");
            success(session.unload_module(result.id));
        } else if (scenario == 1) {
            expect(result.disposition == CudaSessionDisposition::kComplete &&
                       result.status == CUDA_ERROR_INVALID_PTX && calls == 1 &&
                       !session.snapshot().modules && !session.snapshot().image_bytes &&
                       !session.snapshot().is_quarantined,
                   "clean original compiler error has full logs and reusable ownership");
            mode = calls = 0;
            const auto retried = invoke();
            success(retried);
            verify_original_outputs(outputs);
            expect(calls == 2 && retried.id,
                   "explicit caller retry compiles a new original and private image");
            success(session.unload_module(retried.id));
        } else {
            expect(result.disposition == CudaSessionDisposition::kRejected && !result.id &&
                       session.snapshot().is_quarantined && session.snapshot().modules == 1,
                   "partial original/private or malformed SDK outputs retain module ownership");
            expect(result.status == (scenario == 2   ? CUDA_ERROR_INVALID_PTX
                                     : scenario == 4 ? CUDA_ERROR_OUT_OF_MEMORY
                                                     : CUDA_ERROR_UNKNOWN),
                   "original/private/cleanup uncertainty keeps its exact requested status");
            const auto before = calls;
            expect(!invoke().original_status && calls == before,
                   "quarantine never replays compilation");
        }
    }
    success(session.shutdown());
    expect(unload_attempts == (scenario == 8 ? 1U : 0U),
           "uncertain original unload is never repeated");
    expect(fixture::driver.context_destroys == 1 && !fixture::driver.live_resources(),
           "isolated owner finishes physical context cleanup without another compiler call");
}
void profile_keys_include_compiler_controls() {
    fixture::driver = {};
    uses_binary = false;
    uses_names = false;
    auto settings = CudaProfileConfig{};
    settings.workers = {320};
    auto cache = std::make_shared<CudaProfileCache>(settings, fixture::pair_clock);
    auto dispatch = fixture::pair_dispatch();
    dispatch.module_load_ex = compile;
    CudaExecutionSession session({31, 37}, 41, dispatch, fixture::limits(1000), {320}, {}, cache);
    success(session.initialize(0));
    mode = calls = 0;
    auto input = options();
    std::vector<CudaModuleJitOption> outputs;
    for (unsigned int iteration = 0; iteration < 3; ++iteration) {
        // Use a code-affecting option while preserving the fake SDK's independent
        // exact register/thread/output contract above.
        input.resize(5);
        input.push_back({CU_JIT_OPTIMIZATION_LEVEL, iteration == 1 ? 3U : 4U, {}});
        const auto module = session.load_ptx_jit(fixture::kSource, input, &outputs);
        success(module);
        success(session.get_function(module.id, "kernel"));
        expect(cache->snapshot().functions == (iteration == 0 ? 1U : 2U),
               "different compiler controls separate code profiles, equal controls reuse exact "
               "identity");
        success(session.unload_module(module.id));
    }
    const auto plain = session.load_ptx(fixture::kSource);
    success(plain);
    success(session.get_function(plain.id, "kernel"));
    expect(cache->snapshot().functions == 3,
           "option-free profiles cannot collide with JIT identities");
    success(session.unload_module(plain.id));
    success(session.shutdown());
}
void profile_keys_include_names_not_just_counts() {
    fixture::driver = {};
    uses_names = true;
    uses_binary = false;
    mode = calls = 0;
    auto settings = CudaProfileConfig{};
    settings.workers = {320};
    auto cache = std::make_shared<CudaProfileCache>(settings, fixture::pair_clock);
    auto dispatch = fixture::pair_dispatch();
    dispatch.module_load_ex = compile;
    CudaExecutionSession session({31, 37}, 41, dispatch, fixture::limits(1000), {320}, {}, cache);
    success(session.initialize(0));
    for (unsigned int iteration = 0; iteration < 3; ++iteration) {
        referenced_kernel = iteration == 1 ? "another_kernel" : "kernel";
        std::vector<CudaModuleJitOption> outputs;
        const auto module = session.load_ptx_jit(fixture::kSource, options(), &outputs);
        success(module);
        success(session.get_function(module.id, "kernel"));
        verify_original_outputs(outputs);
        expect(cache->snapshot().functions == (iteration == 0 ? 1U : 2U),
               "different exact names separate compiled profiles despite identical counts/scalars");
        success(session.unload_module(module.id));
    }
    success(session.shutdown());
}
}  // namespace
int main() {
    try {
        for (const bool binary : {false, true})
            for (unsigned int scenario = 0; scenario < 12; ++scenario) exercise(binary, scenario);
        profile_keys_include_compiler_controls();
        for (const bool binary : {false, true})
            for (unsigned int scenario = 0; scenario < 14; ++scenario)
                exercise(binary, scenario, true);
        profile_keys_include_names_not_just_counts();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
