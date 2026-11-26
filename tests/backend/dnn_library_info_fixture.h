#pragma once

#include "dnn_fixture.h"
#include "internal/cuda_execution_session.h"

#include <atomic>

namespace glimmer::tests::dnn_library_info {
namespace base = glimmer::tests::dnn;
using namespace glimmer::backend::cuda;

// Injected host-only SDK observations, not vendor-runtime qualification.
struct Control {
    std::array<std::atomic<std::size_t>, 3> values{};
    std::array<std::atomic<unsigned int>, 4> calls{};
    std::atomic<int> property_value{}, property_status{};
    std::atomic<unsigned int> property_selector{}, null_outputs{}, driver_calls{};
};
inline Control control;

inline void reset() {
    base::reset();
    control.values[0].store(91002);
    control.values[1].store(12080);
    control.values[2].store(1000);
    for (auto& calls : control.calls) calls.store(0);
    control.property_value.store(9);
    control.property_status.store(CUDNN_STATUS_SUCCESS);
    control.property_selector.store(0);
    control.null_outputs.store(0);
    control.driver_calls.store(0);
}
inline std::size_t value(unsigned int index) {
    base::owner();
    ++control.calls[index];
    return control.values[index].load();
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto result = base::dispatch();
    result.version = [] { return value(0); };
    result.cudart_version = [] { return value(1); };
    result.max_device_version = [] { return value(2); };
    result.property = [](libraryPropertyType property, int* output) {
        base::owner();
        ++control.calls[3];
        control.property_selector.store(static_cast<unsigned int>(property));
        if (!output) {
            ++control.null_outputs;
            return CUDNN_STATUS_BAD_PARAM;
        }
        // Deliberately write even on error: the caller must not see this value.
        *output = control.property_value.load();
        if (property == static_cast<libraryPropertyType>(3)) return CUDNN_STATUS_BAD_PARAM;
        return static_cast<cudnnStatus_t>(control.property_status.load());
    };
    if (missing == 1) result.version = nullptr;
    if (missing == 2) result.cudart_version = nullptr;
    if (missing == 3) result.max_device_version = nullptr;
    if (missing == 4) result.property = nullptr;
    return result;
}
inline CUresult unexpected_driver() {
    ++control.driver_calls;
    return CUDA_ERROR_NOT_SUPPORTED;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    CudaSessionDispatch result;
    result.initialize = [](unsigned int) { return unexpected_driver(); };
    result.device_get = [](CUdevice*, int) { return unexpected_driver(); };
    result.context_create = [](CUcontext*, unsigned int, CUdevice) { return unexpected_driver(); };
    result.context_set_current = [](CUcontext) { return unexpected_driver(); };
    result.context_destroy = [](CUcontext) { return unexpected_driver(); };
    result.execution.context_get_current = [](CUcontext*) { return unexpected_driver(); };
    result.module_load = [](CUmodule*, const void*) { return unexpected_driver(); };
    result.module_unload = [](CUmodule) { return unexpected_driver(); };
    result.function_get = [](CUfunction*, CUmodule, const char*) { return unexpected_driver(); };
    result.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return unexpected_driver();
    };
    result.allocate = [](CUdeviceptr*, std::size_t) { return unexpected_driver(); };
    result.release = [](CUdeviceptr) { return unexpected_driver(); };
    result.upload = [](CUdeviceptr, const void*, std::size_t) { return unexpected_driver(); };
    result.download = [](void*, CUdeviceptr, std::size_t) { return unexpected_driver(); };
    result.stream_create = [](CUstream*, unsigned int) { return unexpected_driver(); };
    result.stream_destroy = [](CUstream) { return unexpected_driver(); };
    result.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**) { return unexpected_driver(); };
    result.execution.event_create = [](CUevent*, unsigned int) { return unexpected_driver(); };
    result.execution.event_record = [](CUevent, CUstream) { return unexpected_driver(); };
    result.execution.event_query = [](CUevent) { return unexpected_driver(); };
    result.execution.event_destroy = [](CUevent) { return unexpected_driver(); };
    result.execution.stream_synchronize = [](CUstream) { return unexpected_driver(); };
    result.context_synchronize = [] { return unexpected_driver(); };
    // Profile configuration requires capabilities, but metadata must never call them.
    result.device_uuid = [](CUuuid*, CUdevice) { return unexpected_driver(); };
    result.driver_version = [](int*) { return unexpected_driver(); };
    result.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    result.verified_dnn_header_version = CUDNN_VERSION;
    return result;
}
inline unsigned int calls() {
    unsigned int total{};
    for (const auto& count : control.calls) total += count.load();
    return total;
}
inline bool is_resource_free() {
    return !control.driver_calls.load() && !base::state.creates && !base::state.destroys &&
           !base::state.sets && !base::state.gets && !base::state.sizes &&
           !base::state.handle_creates && !base::state.handle_destroys &&
           !base::state.activation_calls && !base::state.add_calls;
}
}  // namespace glimmer::tests::dnn_library_info
