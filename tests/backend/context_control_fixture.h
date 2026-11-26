#pragma once

#include "internal/cuda_execution_session.h"

#include <array>
#include <map>
#include <stdexcept>
#include <thread>

namespace glimmer::tests::context_control {
struct Configuration {
    std::size_t limit{1024};
    unsigned int cache{}, shared{}, version{3020};
};
struct State {
    CUcontext (*current)(){};
    std::thread::id owner{};
    std::map<CUcontext, Configuration> configurations;
    unsigned int driver_calls{}, runtime_calls{}, resets{};
    CUresult driver_error{CUDA_SUCCESS};
    cudaError_t runtime_error{cudaSuccess};
};
inline State state;
inline void require(bool condition) {
    if (!condition) throw std::runtime_error("context control original SDK invariant");
}
inline std::int32_t touch(bool is_runtime) {
    require(std::this_thread::get_id() == state.owner && state.current && state.current());
    if (is_runtime) {
        ++state.runtime_calls;
        return state.runtime_error;
    }
    ++state.driver_calls;
    return state.driver_error;
}
inline Configuration& configuration() {
    return state.configurations[state.current()];
}
inline backend::cuda::CudaContextDispatch dispatch(CUcontext (*current)()) {
    state = {};
    state.current = current;
    state.owner = std::this_thread::get_id();
    backend::cuda::CudaContextDispatch sdk;
    sdk.get_limit = [](std::size_t* output, CUlimit limit) {
        const auto status = touch(false);
        *output = configuration().limit;  // Even an original error may touch its output.
        return status                         ? static_cast<CUresult>(status)
               : limit == CU_LIMIT_STACK_SIZE ? CUDA_SUCCESS
                                              : CUDA_ERROR_UNSUPPORTED_LIMIT;
    };
    sdk.set_limit = [](CUlimit limit, std::size_t value) {
        const auto status = touch(false);
        if (status) return static_cast<CUresult>(status);
        if (limit != CU_LIMIT_STACK_SIZE) return CUDA_ERROR_UNSUPPORTED_LIMIT;
        configuration().limit = value;
        return CUDA_SUCCESS;
    };
    sdk.get_cache = [](CUfunc_cache* output) {
        const auto status = touch(false);
        *output = static_cast<CUfunc_cache>(configuration().cache);
        return static_cast<CUresult>(status);
    };
    sdk.set_cache = [](CUfunc_cache value) {
        const auto status = touch(false);
        if (!status) configuration().cache = value;
        return static_cast<CUresult>(status);
    };
    sdk.get_shared = [](CUsharedconfig* output) {
        const auto status = touch(false);
        *output = static_cast<CUsharedconfig>(configuration().shared);
        return static_cast<CUresult>(status);
    };
    sdk.set_shared = [](CUsharedconfig value) {
        const auto status = touch(false);
        if (status) return static_cast<CUresult>(status);
        if (static_cast<unsigned int>(value) == 3) return CUDA_ERROR_INVALID_VALUE;
        configuration().shared = value;
        return CUDA_SUCCESS;
    };
    sdk.reset_cache = [] {
        const auto status = touch(false);
        if (!status) ++state.resets;
        return static_cast<CUresult>(status);
    };
    sdk.api_version = [](CUcontext context, unsigned int* output) {
        const auto status = touch(false);
        require(context == state.current());
        *output = configuration().version;
        return static_cast<CUresult>(status);
    };
    sdk.affinity = [](CUexecAffinityParam* output, CUexecAffinityType type) {
        const auto status = touch(false);
        if (status) return static_cast<CUresult>(status);
        if (type != CU_EXEC_AFFINITY_TYPE_SM_COUNT) return CUDA_ERROR_INVALID_VALUE;
        *output = {};
        output->type = type;
        output->param.smCount.val = 80;
        return CUDA_SUCCESS;
    };
    sdk.priority_range = [](int* least, int* greatest) {
        const auto status = touch(false);
        *least = 0;
        *greatest = -2;
        return static_cast<CUresult>(status);
    };
    sdk.runtime.get_limit = [](std::size_t* output, cudaLimit limit) {
        const auto status = touch(true);
        *output = configuration().limit;
        return status                        ? static_cast<cudaError_t>(status)
               : limit == cudaLimitStackSize ? cudaSuccess
                                             : cudaErrorUnsupportedLimit;
    };
    sdk.runtime.set_limit = [](cudaLimit limit, std::size_t value) {
        const auto status = touch(true);
        if (status) return static_cast<cudaError_t>(status);
        if (limit != cudaLimitStackSize) return cudaErrorUnsupportedLimit;
        configuration().limit = value;
        return cudaSuccess;
    };
    sdk.runtime.get_cache = [](cudaFuncCache* output) {
        const auto status = touch(true);
        *output = static_cast<cudaFuncCache>(configuration().cache);
        return static_cast<cudaError_t>(status);
    };
    sdk.runtime.set_cache = [](cudaFuncCache value) {
        const auto status = touch(true);
        if (!status) configuration().cache = value;
        return static_cast<cudaError_t>(status);
    };
    sdk.runtime.get_shared = [](cudaSharedMemConfig* output) {
        const auto status = touch(true);
        *output = static_cast<cudaSharedMemConfig>(configuration().shared);
        return static_cast<cudaError_t>(status);
    };
    sdk.runtime.set_shared = [](cudaSharedMemConfig value) {
        const auto status = touch(true);
        if (status) return static_cast<cudaError_t>(status);
        if (static_cast<unsigned int>(value) == 3) return cudaErrorInvalidValue;
        configuration().shared = value;
        return cudaSuccess;
    };
    sdk.runtime.reset_cache = [] {
        const auto status = touch(true);
        if (!status) ++state.resets;
        return static_cast<cudaError_t>(status);
    };
    return sdk;
}
}  // namespace glimmer::tests::context_control
