#include "cuda_execution_domain_fixture.h"
#include "../transform/cuda_image_fixture.h"

namespace {
namespace fixture = glimmer::test::cuda_domain;
using namespace glimmer::backend::cuda;
using fixture::expect;
using fixture::success;
unsigned int mode{};
CUresult CUDAAPI load(CUmodule* output, const void* image) {
    if (mode == 0 || mode == 2) {
        ++fixture::driver.module_load_calls;
        return mode == 0 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS;
    }
    const auto result = fixture::module_load(output, image);
    expect(result == CUDA_SUCCESS, "independent original module creation succeeds");
    return mode == 1 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS;
}
void exercise(bool binary, unsigned int scenario) {
    fixture::driver = {};
    mode = scenario;
    auto dispatch = fixture::session_dispatch();
    dispatch.module_load = load;
    dispatch.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice) {
        expect(attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ||
                   attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
               "only original module architecture metadata is queried");
        *output = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 7 : 0;
        return CUDA_SUCCESS;
    };
    CudaExecutionSession session({31, 37}, 41, dispatch);
    success(session.initialize(0));
    const std::array<glimmer::transform::CudaImageArgument, 2> arguments{{{0, 8}, {8, 4}}};
    const auto image = glimmer::test::cuda_test_cubin(arguments);
    const auto invoke = [&] {
        return binary ? session.load_binary(image) : session.load_ptx(fixture::kSource);
    };
    const auto result = invoke();
    expect(result.original_status == (scenario < 2 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS),
           "every actual original SDK status is retained exactly");
    expect(!result.address && !result.parameter_bytes && result.arguments.empty(),
           "module factories never publish unrelated fields");
    if (scenario == 0) {
        expect(result.disposition == CudaSessionDisposition::kComplete &&
                   result.status == CUDA_ERROR_INVALID_PTX && !result.id &&
                   !session.snapshot().is_quarantined && !session.snapshot().modules &&
                   !session.snapshot().image_bytes && fixture::driver.module_load_calls == 1,
               "known no-handle native error is a reusable completed error");
        mode = 3;
        const auto retried = invoke();
        success(retried);
        expect(retried.id && fixture::driver.module_load_calls == 2,
               "one explicit caller retry owns a fresh successful module");
        success(session.unload_module(retried.id));
    } else if (scenario < 3) {
        expect(result.disposition == CudaSessionDisposition::kRejected && !result.id &&
                   result.status == (scenario == 1 ? CUDA_ERROR_INVALID_PTX : CUDA_ERROR_UNKNOWN) &&
                   session.snapshot().is_quarantined && session.snapshot().modules == 1 &&
                   session.snapshot().image_bytes && fixture::driver.module_load_calls == 1,
               "partial/native-null result exposes retained uncertainty without erasing SDK error");
        expect(!invoke().original_status && fixture::driver.module_load_calls == 1 &&
                   !fixture::driver.module_unloads,
               "uncertain owner rejects successor preparation without SDK replay or early unload");
    } else {
        success(result);
        expect(result.id && !session.snapshot().is_quarantined && session.snapshot().modules == 1,
               "normal factory creates one unique known owner");
        success(session.unload_module(result.id));
    }
    success(session.shutdown());
    expect(fixture::driver.context_destroys == 1 &&
               fixture::driver.module_unloads == (scenario == 0 || scenario == 3 ? 1U : 0U) &&
               std::ranges::none_of(fixture::driver.modules,
                                    [](const auto& module) { return module.is_live; }) &&
               !session.snapshot().modules && !session.snapshot().image_bytes,
           "known handles unload once; exact isolated context destruction owns partial cleanup");
}
}  // namespace
int main() {
    try {
        for (const bool binary : {false, true})
            for (unsigned int scenario = 0; scenario < 4; ++scenario) exercise(binary, scenario);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
