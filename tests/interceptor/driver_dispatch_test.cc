#include "internal/driver_dispatch.h"

#include <cstdlib>
#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

using glimmer::interceptor::DriverDispatch;
using glimmer::interceptor::DriverFunctionTable;
using glimmer::interceptor::is_inside_driver_call;

bool g_guard_failed = false;
const DriverDispatch* g_recursive_dispatch = nullptr;

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }

    std::cerr << message << '\n';
    return false;
}

void check_driver_guard() {
    if (!is_inside_driver_call()) {
        g_guard_failed = true;
    }
}

struct PrimaryIdentityProbe {
    CUcontext current{};
    CUcontext primary{};
    CUdevice device = 3;
    int active = 0;
    CUresult current_status = CUDA_SUCCESS;
    CUresult device_status = CUDA_SUCCESS;
    CUresult state_status = CUDA_SUCCESS;
    CUresult retain_status = CUDA_SUCCESS;
    CUresult release_status = CUDA_SUCCESS;
    unsigned int state_calls{};
    unsigned int retain_calls{};
    unsigned int release_calls{};
};
PrimaryIdentityProbe g_primary_probe;

CUresult CUDAAPI probe_current_context(CUcontext* output) {
    check_driver_guard();
    *output = g_primary_probe.current;
    return g_primary_probe.current_status;
}
CUresult CUDAAPI probe_current_device(CUdevice* output) {
    check_driver_guard();
    *output = g_primary_probe.device;
    return g_primary_probe.device_status;
}
CUresult CUDAAPI probe_primary_state(CUdevice device, unsigned int* flags, int* active) {
    check_driver_guard();
    ++g_primary_probe.state_calls;
    if (device != g_primary_probe.device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *flags = CU_CTX_SCHED_AUTO;
    *active = g_primary_probe.active;
    return g_primary_probe.state_status;
}
CUresult CUDAAPI probe_primary_retain(CUcontext* output, CUdevice device) {
    check_driver_guard();
    ++g_primary_probe.retain_calls;
    if (device != g_primary_probe.device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *output = g_primary_probe.primary;
    return g_primary_probe.retain_status;
}
CUresult CUDAAPI probe_primary_release(CUdevice device) {
    check_driver_guard();
    ++g_primary_probe.release_calls;
    return device == g_primary_probe.device ? g_primary_probe.release_status
                                            : CUDA_ERROR_INVALID_DEVICE;
}

bool test_primary_context_state() {
    DriverFunctionTable functions;
    functions.primary_context_get_state = probe_primary_state;
    const DriverDispatch dispatch(functions);
    bool passed = true;
    for (const int active : {0, 1, -1, 2}) {
        g_primary_probe = {};
        g_primary_probe.active = active;
        bool output = true;
        const auto expected = active < 0 || active > 1 ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        passed &= expect(dispatch.primary_context_is_active(3, output) == expected &&
                             output == (active != 0) && g_primary_probe.state_calls == 1 &&
                             g_primary_probe.retain_calls == 0 &&
                             g_primary_probe.release_calls == 0 && !is_inside_driver_call(),
                         "primary state is output-atomic metadata, never context ownership");
    }
    g_primary_probe = {};
    g_primary_probe.state_status = CUDA_ERROR_INVALID_DEVICE;
    bool output = true;
    passed &= expect(dispatch.primary_context_is_active(3, output) == CUDA_ERROR_INVALID_DEVICE &&
                         output && g_primary_probe.state_calls == 1,
                     "failed original primary metadata preserves output");
    g_primary_probe = {};
    passed &=
        expect(dispatch.primary_context_is_active(-1, output) == CUDA_ERROR_INVALID_DEVICE &&
                   output && g_primary_probe.state_calls == 0 &&
                   DriverDispatch(DriverFunctionTable{}).primary_context_is_active(3, output) ==
                       CUDA_ERROR_NOT_SUPPORTED &&
                   output && !is_inside_driver_call(),
               "invalid or missing primary metadata issues nothing");
    return passed;
}

bool test_primary_context_identity() {
    DriverFunctionTable functions;
    functions.context_get_current = probe_current_context;
    functions.context_get_device = probe_current_device;
    functions.primary_context_get_state = probe_primary_state;
    functions.primary_context_retain = probe_primary_retain;
    functions.primary_context_release = probe_primary_release;
    const DriverDispatch dispatch(functions);
    int current_token{}, other_token{};
    const auto context = reinterpret_cast<CUcontext>(&current_token);
    const auto other = reinterpret_cast<CUcontext>(&other_token);
    bool passed = true;
    const auto initialize = [&] {
        g_primary_probe = {};
        g_primary_probe.current = context;
        g_primary_probe.primary = context;
    };
    initialize();
    bool output = true;
    passed &= expect(dispatch.current_context_is_active_primary(context, output) == CUDA_SUCCESS &&
                         !output && g_primary_probe.state_calls == 1 &&
                         g_primary_probe.retain_calls == 0 && g_primary_probe.release_calls == 0,
                     "inactive primary must not be retained or initialized");
    for (bool is_primary : {false, true}) {
        initialize();
        g_primary_probe.active = 1;
        g_primary_probe.primary = is_primary ? context : other;
        output = !is_primary;
        passed &=
            expect(dispatch.current_context_is_active_primary(context, output) == CUDA_SUCCESS &&
                       output == is_primary && g_primary_probe.retain_calls == 1 &&
                       g_primary_probe.release_calls == 1 && g_primary_probe.current == context,
                   "active primary identity needs an exact balanced reference");
    }
    for (int failure = 0; failure < 8; ++failure) {
        initialize();
        g_primary_probe.active = 1;
        CUresult expected = CUDA_ERROR_LAUNCH_FAILED;
        switch (failure) {
            case 0:
                g_primary_probe.current_status = expected;
                break;
            case 1:
                g_primary_probe.current = other;
                expected = CUDA_ERROR_INVALID_CONTEXT;
                break;
            case 2:
                g_primary_probe.device_status = expected;
                break;
            case 3:
                g_primary_probe.state_status = expected;
                break;
            case 4:
                g_primary_probe.active = -1;
                expected = CUDA_ERROR_UNKNOWN;
                break;
            case 5:
                g_primary_probe.retain_status = expected;
                break;
            case 6:
                g_primary_probe.release_status = expected;
                break;
            default:
                g_primary_probe.primary = nullptr;
                expected = CUDA_ERROR_UNKNOWN;
                break;
        }
        output = true;
        passed &= expect(dispatch.current_context_is_active_primary(context, output) == expected &&
                             output && g_primary_probe.retain_calls == (failure >= 5 ? 1U : 0U) &&
                             g_primary_probe.release_calls == (failure >= 6 ? 1U : 0U),
                         "failed identity proof preserves output and never retries a reference");
    }
    for (int missing = 0; missing < 5; ++missing) {
        auto partial = functions;
        switch (missing) {
            case 0:
                partial.context_get_current = nullptr;
                break;
            case 1:
                partial.context_get_device = nullptr;
                break;
            case 2:
                partial.primary_context_get_state = nullptr;
                break;
            case 3:
                partial.primary_context_retain = nullptr;
                break;
            default:
                partial.primary_context_release = nullptr;
                break;
        }
        initialize();
        const DriverDispatch missing_dispatch(partial);
        output = true;
        passed &=
            expect(missing_dispatch.current_context_is_active_primary(context, output) ==
                           CUDA_ERROR_NOT_SUPPORTED &&
                       output && g_primary_probe.state_calls == 0 &&
                       g_primary_probe.retain_calls == 0 && g_primary_probe.release_calls == 0,
                   "missing primary proof must reject before acquiring a reference");
    }
    initialize();
    output = true;
    passed &= expect(
        dispatch.current_context_is_active_primary(nullptr, output) == CUDA_ERROR_INVALID_CONTEXT &&
            output && !is_inside_driver_call() && !g_guard_failed,
        "null identity and native scope must remain checked");
    return passed;
}

bool test_unbound_active_primary_identity() {
    DriverFunctionTable functions;
    // Selected-device lookup must not require a current context or bind one.
    functions.primary_context_get_state = probe_primary_state;
    functions.primary_context_retain = probe_primary_retain;
    functions.primary_context_release = probe_primary_release;
    const DriverDispatch dispatch(functions);
    int token{}, sentinel{};
    const auto primary = reinterpret_cast<CUcontext>(&token);
    const auto unchanged = reinterpret_cast<CUcontext>(&sentinel);
    const auto initialize = [&] {
        g_primary_probe = {};
        g_primary_probe.primary = primary;
    };
    bool passed = true;
    for (bool is_active : {false, true}) {
        initialize();
        g_primary_probe.active = is_active ? 1 : 0;
        CUcontext output = unchanged;
        passed &= expect(
            dispatch.active_primary_context(3, output) == CUDA_SUCCESS &&
                output == (is_active ? primary : nullptr) && g_primary_probe.state_calls == 1 &&
                g_primary_probe.retain_calls == (is_active ? 1U : 0U) &&
                g_primary_probe.release_calls == (is_active ? 1U : 0U) && !g_primary_probe.current,
            "selected primary is borrowed without binding or cold activation");
    }
    for (int failure = 0; failure < 7; ++failure) {
        initialize();
        g_primary_probe.active = 1;
        CUresult expected = CUDA_ERROR_LAUNCH_FAILED;
        CUdevice device = 3;
        switch (failure) {
            case 0:
                device = -1;
                expected = CUDA_ERROR_INVALID_DEVICE;
                break;
            case 1:
                device = 4;
                expected = CUDA_ERROR_INVALID_DEVICE;
                break;
            case 2:
                g_primary_probe.state_status = expected;
                break;
            case 3:
                g_primary_probe.active = 2;
                expected = CUDA_ERROR_UNKNOWN;
                break;
            case 4:
                g_primary_probe.retain_status = expected;
                break;
            case 5:
                g_primary_probe.release_status = expected;
                break;
            default:
                g_primary_probe.primary = nullptr;
                expected = CUDA_ERROR_UNKNOWN;
                break;
        }
        CUcontext output = unchanged;
        passed &= expect(dispatch.active_primary_context(device, output) == expected &&
                             output == unchanged &&
                             g_primary_probe.state_calls == (failure == 0 ? 0U : 1U) &&
                             g_primary_probe.retain_calls == (failure >= 4 ? 1U : 0U) &&
                             g_primary_probe.release_calls == (failure >= 5 ? 1U : 0U) &&
                             !g_primary_probe.current,
                         "failed selected-primary proof is output-atomic with no reference retry");
    }
    for (int missing = 0; missing < 3; ++missing) {
        auto partial = functions;
        if (missing == 0) {
            partial.primary_context_get_state = nullptr;
        } else if (missing == 1) {
            partial.primary_context_retain = nullptr;
        } else {
            partial.primary_context_release = nullptr;
        }
        initialize();
        CUcontext output = unchanged;
        passed &= expect(
            DriverDispatch(partial).active_primary_context(3, output) == CUDA_ERROR_NOT_SUPPORTED &&
                output == unchanged && g_primary_probe.state_calls == 0 &&
                g_primary_probe.retain_calls == 0 && g_primary_probe.release_calls == 0,
            "missing selected-primary proof rejects before native identity issue");
    }
    for (int phase = 0; phase < 4; ++phase) {
        initialize();
        g_primary_probe.active = 1;
        if (phase == 0) {
            g_primary_probe.state_status = CUDA_ERROR_DEINITIALIZED;
        } else if (phase == 1) {
            g_primary_probe.retain_status = CUDA_ERROR_DEINITIALIZED;
        } else if (phase == 2) {
            g_primary_probe.release_status = CUDA_ERROR_DEINITIALIZED;
        }
        CUcontext output = unchanged;
        bool is_deinitialized_before_identity = true;
        const auto status =
            dispatch.active_primary_context(3, output, &is_deinitialized_before_identity);
        passed &= expect(status == (phase == 3 ? CUDA_SUCCESS : CUDA_ERROR_DEINITIALIZED) &&
                             output == (phase == 3 ? primary : unchanged) &&
                             is_deinitialized_before_identity == (phase == 0) &&
                             g_primary_probe.state_calls == 1 &&
                             g_primary_probe.retain_calls == (phase == 0 ? 0U : 1U) &&
                             g_primary_probe.release_calls == (phase < 2 ? 0U : 1U),
                         "deinitialized proof is stage-specific, output-atomic and reset on every "
                         "identity call");
    }
    initialize();
    CUcontext output = unchanged;
    bool is_deinitialized_before_identity = true;
    passed &=
        expect(dispatch.active_primary_context(-1, output, &is_deinitialized_before_identity) ==
                       CUDA_ERROR_INVALID_DEVICE &&
                   !is_deinitialized_before_identity && output == unchanged &&
                   g_primary_probe.state_calls == 0,
               "pre-query rejection also clears stale deinitialization proof without native calls");
    return passed && !is_inside_driver_call() && !g_guard_failed;
}

void CUDAAPI fake_symbol() {}

std::array<unsigned int, 2> g_primary_reset_calls{};
std::array<CUdevice, 2> g_primary_reset_devices{};

CUresult CUDAAPI fake_primary_reset_legacy(CUdevice device) {
    check_driver_guard();
    ++g_primary_reset_calls[0];
    g_primary_reset_devices[0] = device;
    return CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
}

CUresult CUDAAPI fake_primary_reset_modern(CUdevice device) {
    check_driver_guard();
    ++g_primary_reset_calls[1];
    g_primary_reset_devices[1] = device;
    return CUDA_ERROR_INVALID_CONTEXT;
}

bool test_primary_reset_dispatch() {
    DriverFunctionTable functions;
    functions.primary_context_reset = fake_primary_reset_legacy;
    functions.primary_context_reset_v2 = fake_primary_reset_modern;
    const DriverDispatch dispatch(functions);
    g_primary_reset_calls = {};
    bool passed = expect(
        dispatch.has_primary_context_reset(false) && dispatch.has_primary_context_reset(true),
        "distinct optional reset entries are available");
    passed &= expect(
        dispatch.primary_context_reset(3, false) == CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE &&
            dispatch.primary_context_reset(-7, true) == CUDA_ERROR_INVALID_CONTEXT &&
            g_primary_reset_calls == std::array<unsigned int, 2>{1, 1} &&
            g_primary_reset_devices == std::array<CUdevice, 2>{3, -7} && !is_inside_driver_call(),
        "reset preserves exact native version, input, error, count and guard");
    for (bool modern : {false, true}) {
        auto partial = functions;
        if (modern) {
            partial.primary_context_reset_v2 = nullptr;
        } else {
            partial.primary_context_reset = nullptr;
        }
        const DriverDispatch missing(partial);
        const auto before = g_primary_reset_calls;
        passed &= expect(!missing.has_primary_context_reset(modern) &&
                             missing.has_primary_context_reset(!modern) &&
                             missing.primary_context_reset(3, modern) == CUDA_ERROR_NOT_SUPPORTED &&
                             g_primary_reset_calls == before && !is_inside_driver_call(),
                         "missing reset ownership version never aliases or retries another entry");
    }
    return passed;
}

std::array<unsigned int, 2> g_primary_release_calls{};
std::array<CUdevice, 2> g_primary_release_devices{};
CUresult CUDAAPI fake_primary_release_legacy(CUdevice device) {
    check_driver_guard();
    ++g_primary_release_calls[0];
    g_primary_release_devices[0] = device;
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult CUDAAPI fake_primary_release_modern(CUdevice device) {
    check_driver_guard();
    ++g_primary_release_calls[1];
    g_primary_release_devices[1] = device;
    return CUDA_ERROR_INVALID_CONTEXT;
}
bool test_primary_release_dispatch() {
    DriverFunctionTable functions;
    functions.primary_context_release_legacy = fake_primary_release_legacy;
    functions.primary_context_release = fake_primary_release_modern;
    const DriverDispatch dispatch(functions);
    g_primary_release_calls = {};
    bool passed = expect(
        dispatch.has_primary_context_release(false) && dispatch.has_primary_context_release(true),
        "release ownership entries are optional and distinct");
    passed &= expect(dispatch.primary_context_release(3, false) == CUDA_ERROR_INVALID_DEVICE &&
                         dispatch.primary_context_release(-7, true) == CUDA_ERROR_INVALID_CONTEXT &&
                         g_primary_release_calls == std::array<unsigned int, 2>{1, 1} &&
                         g_primary_release_devices == std::array<CUdevice, 2>{3, -7} &&
                         !is_inside_driver_call(),
                     "release preserves exact native version, input, error, count and guard");
    for (const bool is_modern : {false, true}) {
        auto partial = functions;
        if (is_modern) {
            partial.primary_context_release = nullptr;
        } else {
            partial.primary_context_release_legacy = nullptr;
        }
        const DriverDispatch missing(partial);
        const auto before = g_primary_release_calls;
        passed &=
            expect(!missing.has_primary_context_release(is_modern) &&
                       missing.has_primary_context_release(!is_modern) &&
                       missing.primary_context_release(3, is_modern) == CUDA_ERROR_NOT_SUPPORTED &&
                       g_primary_release_calls == before && !is_inside_driver_call(),
                   "missing release ABI never aliases or retries another ownership version");
    }
    return passed;
}

std::array<unsigned int, 2> g_context_destroy_calls{};
std::array<CUcontext, 2> g_context_destroy_inputs{};
CUresult CUDAAPI fake_context_destroy_legacy(CUcontext context) {
    check_driver_guard();
    ++g_context_destroy_calls[0];
    g_context_destroy_inputs[0] = context;
    return CUDA_ERROR_INVALID_HANDLE;
}
CUresult CUDAAPI fake_context_destroy_modern(CUcontext context) {
    check_driver_guard();
    ++g_context_destroy_calls[1];
    g_context_destroy_inputs[1] = context;
    return CUDA_ERROR_INVALID_CONTEXT;
}
bool test_context_destroy_dispatch() {
    DriverFunctionTable functions;
    functions.context_destroy_legacy = fake_context_destroy_legacy;
    functions.context_destroy = fake_context_destroy_modern;
    const DriverDispatch dispatch(functions);
    g_context_destroy_calls = {};
    const auto context = reinterpret_cast<CUcontext>(0x1234);
    bool passed = expect(dispatch.has_context_destroy(false) && dispatch.has_context_destroy(),
                         "context destroy versions are optional and distinct");
    passed &= expect(dispatch.context_destroy(context, false) == CUDA_ERROR_INVALID_HANDLE &&
                         dispatch.context_destroy(nullptr) == CUDA_ERROR_INVALID_CONTEXT &&
                         g_context_destroy_calls == std::array<unsigned int, 2>{1, 1} &&
                         g_context_destroy_inputs == std::array<CUcontext, 2>{context, nullptr} &&
                         !is_inside_driver_call(),
                     "destroy preserves exact native version, input, error, count and guard");
    for (const bool is_modern : {false, true}) {
        auto partial = functions;
        if (is_modern) {
            partial.context_destroy = nullptr;
        } else {
            partial.context_destroy_legacy = nullptr;
        }
        const DriverDispatch missing(partial);
        const auto before = g_context_destroy_calls;
        passed &= expect(
            !missing.has_context_destroy(is_modern) && missing.has_context_destroy(!is_modern) &&
                missing.context_destroy(context, is_modern) == CUDA_ERROR_NOT_SUPPORTED &&
                g_context_destroy_calls == before && !is_inside_driver_call(),
            "missing destroy ABI never aliases or retries another ownership version");
    }
    return passed;
}

unsigned int g_context_version_calls{};
CUresult CUDAAPI fake_context_version(CUcontext context, unsigned int* version) {
    check_driver_guard();
    ++g_context_version_calls;
    if (!context || !version) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *version = 3020;
    return CUDA_SUCCESS;
}
bool test_context_version_dispatch() {
    DriverFunctionTable functions;
    const auto context = reinterpret_cast<CUcontext>(0x1234);
    unsigned int version = 777;
    g_context_version_calls = 0;
    const DriverDispatch missing(functions);
    bool passed =
        expect(missing.context_get_api_version(context, &version) == CUDA_ERROR_NOT_SUPPORTED &&
                   version == 777 && g_context_version_calls == 0 && !is_inside_driver_call(),
               "optional context version never mutates output or issues an absent entry");
    functions.context_get_api_version = fake_context_version;
    const DriverDispatch dispatch(functions);
    passed &=
        expect(dispatch.context_get_api_version(context, &version) == CUDA_SUCCESS &&
                   version == 3020 && g_context_version_calls == 1 && !is_inside_driver_call(),
               "original context metadata preserves guarded ABI and exact native output");
    version = 777;
    {
        const glimmer::interceptor::NativeDriverCallScope native;
        passed &= expect(
            dispatch.context_get_api_version(nullptr, &version) == CUDA_ERROR_INVALID_VALUE &&
                version == 777 &&
                dispatch.context_get_api_version(context, nullptr) == CUDA_ERROR_INVALID_VALUE &&
                g_context_version_calls == 3 && is_inside_driver_call(),
            "metadata delegates validation and preserves a nested original guard");
    }
    return passed && !is_inside_driver_call();
}

CUresult CUDAAPI fake_init(unsigned int) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

unsigned int g_function_load_calls{};
CUfunction g_expected_loaded_function{};
CUresult CUDAAPI fake_function_load(CUfunction function) {
    check_driver_guard();
    ++g_function_load_calls;
    return function == g_expected_loaded_function ? CUDA_ERROR_LAUNCH_FAILED
                                                  : CUDA_ERROR_INVALID_VALUE;
}

bool test_function_loading_dispatch() {
    DriverFunctionTable functions;
    functions.function_load = fake_function_load;
    DriverDispatch dispatch(functions);
    DriverDispatch missing(DriverFunctionTable{});
    int token{};
    g_expected_loaded_function = reinterpret_cast<CUfunction>(&token);
    const auto before = g_function_load_calls;
    return dispatch.has_function_load() && !missing.has_function_load() &&
           dispatch.function_load(g_expected_loaded_function) == CUDA_ERROR_LAUNCH_FAILED &&
           g_function_load_calls == before + 1 &&
           missing.function_load(g_expected_loaded_function) == CUDA_ERROR_NOT_SUPPORTED &&
           g_function_load_calls == before + 1 && !is_inside_driver_call() && !g_guard_failed;
}

unsigned int g_function_module_calls{};
CUmodule* g_expected_module_output{};
CUresult CUDAAPI fake_function_module(CUmodule* output, CUfunction function) {
    check_driver_guard();
    ++g_function_module_calls;
    if (output != g_expected_module_output || function != g_expected_loaded_function) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *output = reinterpret_cast<CUmodule>(function);
    return CUDA_ERROR_NOT_FOUND;
}

bool test_function_module_dispatch() {
    DriverFunctionTable functions;
    functions.function_get_module = fake_function_module;
    DriverDispatch dispatch(functions);
    DriverDispatch missing(DriverFunctionTable{});
    CUmodule output{};
    int token{};
    g_expected_loaded_function = reinterpret_cast<CUfunction>(&token);
    g_expected_module_output = &output;
    const auto before = g_function_module_calls;
    return dispatch.has_function_get_module() && !missing.has_function_get_module() &&
           dispatch.function_get_module(&output, g_expected_loaded_function) ==
               CUDA_ERROR_NOT_FOUND &&
           output == reinterpret_cast<CUmodule>(g_expected_loaded_function) &&
           g_function_module_calls == before + 1 &&
           missing.function_get_module(&output, g_expected_loaded_function) ==
               CUDA_ERROR_NOT_SUPPORTED &&
           g_function_module_calls == before + 1 && !is_inside_driver_call() && !g_guard_failed;
}

CUresult CUDAAPI fake_launch_kernel(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int, unsigned int,
                                    CUstream, void**, void**) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_graph_launch(CUgraphExec, CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_launch_kernel_ex(const CUlaunchConfig* config, CUfunction function,
                                       void** arguments, void** extra) {
    check_driver_guard();
    return config != nullptr && arguments != nullptr && arguments[0] == config &&
                   reinterpret_cast<const void*>(function) == config && extra == arguments + 1
               ? CUDA_ERROR_LAUNCH_FAILED
               : CUDA_ERROR_INVALID_VALUE;
}

bool test_extended_dispatch() {
    DriverFunctionTable functions;
    functions.launch_kernel_ex = fake_launch_kernel_ex;
    functions.launch_kernel_ex_ptsz = fake_launch_kernel_ex;
    DriverDispatch dispatch(functions);
    DriverDispatch missing(DriverFunctionTable{});
    CUlaunchConfig config{};
    void* arguments[] = {&config, nullptr};
    bool passed = true;
    for (const bool ptds : {false, true}) {
        passed &= dispatch.has_launch_kernel_ex(ptds) && !missing.has_launch_kernel_ex(ptds);
        passed &=
            dispatch.launch_kernel_ex(&config, reinterpret_cast<CUfunction>(&config), arguments,
                                      arguments + 1, ptds) == CUDA_ERROR_LAUNCH_FAILED;
        passed &= missing.launch_kernel_ex(&config, nullptr, nullptr, nullptr, ptds) ==
                  CUDA_ERROR_NOT_SUPPORTED;
    }
    return passed && !is_inside_driver_call();
}

CUresult CUDAAPI fake_capture_legacy(CUstream, CUstreamCaptureStatus* status) {
    check_driver_guard();
    if (status == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_capture_ptds(CUstream, CUstreamCaptureStatus* status) {
    check_driver_guard();
    if (status == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *status = CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    return CUDA_SUCCESS;
}

bool test_capture_dispatch() {
    DriverFunctionTable functions;
    functions.stream_capture_status = fake_capture_legacy;
    functions.stream_capture_status_ptsz = fake_capture_ptds;
    DriverDispatch dispatch(functions);
    DriverDispatch missing(DriverFunctionTable{});
    bool passed = true;
    for (const bool ptds : {false, true}) {
        CUstreamCaptureStatus status{};
        passed &= dispatch.stream_capture_status(nullptr, &status, ptds) == CUDA_SUCCESS;
        passed &= status ==
                  (ptds ? CU_STREAM_CAPTURE_STATUS_INVALIDATED : CU_STREAM_CAPTURE_STATUS_ACTIVE);
        passed &=
            dispatch.stream_capture_status(nullptr, nullptr, ptds) == CUDA_ERROR_INVALID_VALUE;
        passed &= missing.stream_capture_status(nullptr, &status, ptds) == CUDA_ERROR_NOT_SUPPORTED;
    }
    return passed && !is_inside_driver_call();
}

CUresult CUDAAPI fake_mem_alloc(CUdeviceptr* device_pointer, std::size_t) {
    check_driver_guard();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x1234;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t, unsigned int) {
    return fake_mem_alloc(device_pointer, 0);
}

CUresult CUDAAPI fake_mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch, std::size_t,
                                      std::size_t, unsigned int) {
    check_driver_guard();
    if (device_pointer == nullptr || pitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x5678;
    *pitch = 64;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                      CUstream) {
    return fake_mem_alloc(device_pointer, memory_bytes);
}

CUresult CUDAAPI fake_mem_alloc_async_ptsz(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           CUstream stream) {
    return fake_mem_alloc_async(device_pointer, memory_bytes, stream);
}

CUresult CUDAAPI fake_mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, CUmemoryPool, CUstream) {
    return fake_mem_alloc(device_pointer, memory_bytes);
}

CUresult CUDAAPI fake_mem_alloc_from_pool_async_ptsz(CUdeviceptr* device_pointer,
                                                     std::size_t memory_bytes, CUmemoryPool pool,
                                                     CUstream stream) {
    return fake_mem_alloc_from_pool_async(device_pointer, memory_bytes, pool, stream);
}

CUresult CUDAAPI fake_mem_create(CUmemGenericAllocationHandle* handle, std::size_t,
                                 const CUmemAllocationProp*, unsigned long long) {
    check_driver_guard();
    if (handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x42;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_release(CUmemGenericAllocationHandle) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_address_reserve(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                          std::size_t, CUdeviceptr, unsigned long long) {
    check_driver_guard();
    if (device_pointer == nullptr || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x4000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_address_free(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes, std::size_t,
                              CUmemGenericAllocationHandle handle, unsigned long long) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 || handle == 0 ? CUDA_ERROR_INVALID_VALUE
                                                                   : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    check_driver_guard();
    return device_pointer == 0 || memory_bytes == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     const CUmemAccessDesc* access_descriptors,
                                     std::size_t descriptor_count) {
    check_driver_guard();
    if (device_pointer == 0 || memory_bytes == 0 ||
        (descriptor_count != 0 && access_descriptors == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_get_address_range(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                            CUdeviceptr device_pointer) {
    check_driver_guard();
    if (base_pointer == nullptr || memory_bytes == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *base_pointer = device_pointer;
    *memory_bytes = 4096;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                     CUdeviceptr device_pointer) {
    check_driver_guard();
    if (flags == nullptr || location == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_export(void* shareable_handle, CUmemGenericAllocationHandle handle,
                                 CUmemAllocationHandleType, unsigned long long) {
    check_driver_guard();
    return shareable_handle == nullptr || handle == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_import(CUmemGenericAllocationHandle* handle, void* os_handle,
                                 CUmemAllocationHandleType) {
    check_driver_guard();
    if (handle == nullptr || os_handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x43;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_granularity(std::size_t* granularity, const CUmemAllocationProp* prop,
                                      CUmemAllocationGranularity_flags) {
    check_driver_guard();
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = 65536;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_properties(CUmemAllocationProp* prop,
                                     CUmemGenericAllocationHandle handle) {
    check_driver_guard();
    if (prop == nullptr || handle == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *prop = {};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_retain(CUmemGenericAllocationHandle* handle, void* device_pointer) {
    check_driver_guard();
    if (handle == nullptr || device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = 0x44;
    return CUDA_SUCCESS;
}

std::uint8_t g_pool_token = 0;

CUresult CUDAAPI fake_pool_trim(CUmemoryPool pool, std::size_t) {
    check_driver_guard();
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute, void* value) {
    check_driver_guard();
    return pool == nullptr || value == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute, void* value) {
    check_driver_guard();
    return pool == nullptr || value == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_set_access(CUmemoryPool pool, const CUmemAccessDesc* descriptors,
                                      std::size_t count) {
    check_driver_guard();
    return pool == nullptr || (count != 0 && descriptors == nullptr) ? CUDA_ERROR_INVALID_VALUE
                                                                     : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                      CUmemLocation* location) {
    check_driver_guard();
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_create(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    check_driver_guard();
    if (pool == nullptr || properties == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_destroy(CUmemoryPool pool) {
    check_driver_guard();
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_get_pool(CUmemoryPool* pool, CUdevice device) {
    check_driver_guard();
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_set_pool(CUdevice device, CUmemoryPool pool) {
    check_driver_guard();
    return device < 0 || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_get_default_pool(CUmemoryPool* pool, CUdevice device) {
    check_driver_guard();
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_pool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType) {
    check_driver_guard();
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_set_pool(CUmemLocation* location, CUmemAllocationType, CUmemoryPool pool) {
    check_driver_guard();
    return location == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_export(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType,
                                  unsigned long long) {
    check_driver_guard();
    return handle_out == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_import(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType,
                                  unsigned long long) {
    check_driver_guard();
    if (pool_out == nullptr || handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool_out = reinterpret_cast<CUmemoryPool>(&g_pool_token);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                          CUdeviceptr device_pointer) {
    check_driver_guard();
    return share_data_out == nullptr || device_pointer == 0 ? CUDA_ERROR_INVALID_VALUE
                                                            : CUDA_SUCCESS;
}

CUresult CUDAAPI fake_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                          CUmemPoolPtrExportData* share_data) {
    check_driver_guard();
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer_out = 0x7000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_free(CUdeviceptr) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_mem_free_async(CUdeviceptr device_pointer, CUstream) {
    return fake_mem_free(device_pointer);
}

CUresult CUDAAPI fake_mem_free_async_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return fake_mem_free_async(device_pointer, stream);
}

CUresult CUDAAPI fake_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
    check_driver_guard();
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *free_bytes = 100;
    *total_bytes = 200;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_device_total_mem(std::size_t* total_bytes, CUdevice) {
    check_driver_guard();
    if (total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *total_bytes = 200;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_get_current(CUcontext* context) {
    check_driver_guard();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *context = nullptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_get_device(CUdevice* device) {
    check_driver_guard();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_destroy(CUcontext) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_context_synchronize() {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_device(CUstream, CUdevice* device) {
    check_driver_guard();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_device_ptsz(CUstream stream, CUdevice* device) {
    return fake_stream_get_device(stream, device);
}

CUresult CUDAAPI fake_stream_get_context(CUstream, CUcontext* context) {
    check_driver_guard();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *context = nullptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_get_context_ptsz(CUstream stream, CUcontext* context) {
    return fake_stream_get_context(stream, context);
}

CUresult CUDAAPI fake_stream_query(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_query_ptsz(CUstream stream) {
    return fake_stream_query(stream);
}

CUresult CUDAAPI fake_stream_synchronize(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_stream_synchronize_ptsz(CUstream stream) {
    return fake_stream_synchronize(stream);
}

CUresult CUDAAPI fake_stream_destroy(CUstream) {
    check_driver_guard();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_proc_address(const char*, void** function_pointer, int, cuuint64_t) {
    check_driver_guard();
    if (function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *function_pointer = reinterpret_cast<void*>(&fake_symbol);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_get_proc_address_v2(const char*, void** function_pointer, int, cuuint64_t,
                                          CUdriverProcAddressQueryResult* symbol_status) {
    const CUresult result = fake_get_proc_address(nullptr, function_pointer, 0, 0);
    if (symbol_status != nullptr) {
        *symbol_status = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return result;
}

CUresult CUDAAPI recursive_get_proc_address_v2(const char* symbol, void** function_pointer,
                                               int cuda_version, cuuint64_t flags,
                                               CUdriverProcAddressQueryResult* symbol_status) {
    if (g_recursive_dispatch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return g_recursive_dispatch->get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                     symbol_status);
}

struct LibraryProbe {
    CUlibrary* output{};
    const void* image{};
    CUjit_option* options{};
    void** values{};
    unsigned int count{};
    CUlibraryOption* library_options{};
    void** library_values{};
    unsigned int library_count{};
    unsigned int calls{};
};
LibraryProbe g_library_probe;

CUresult probe_library_load(CUlibrary* output, const void* image, CUjit_option* options,
                            void** values, unsigned int count, CUlibraryOption* library_options,
                            void** library_values, unsigned int library_count) {
    check_driver_guard();
    g_library_probe = {output,         image,         options,
                       values,         count,         library_options,
                       library_values, library_count, g_library_probe.calls + 1};
    return CUDA_ERROR_INVALID_IMAGE;
}
CUresult probe_library_file(CUlibrary* output, const char* path, CUjit_option* options,
                            void** values, unsigned int count, CUlibraryOption* library_options,
                            void** library_values, unsigned int library_count) {
    return probe_library_load(output, path, options, values, count, library_options, library_values,
                              library_count);
}
CUresult probe_library_module(CUmodule* module, CUlibrary library) {
    check_driver_guard();
    ++g_library_probe.calls;
    return module && library ? CUDA_ERROR_NOT_FOUND : CUDA_ERROR_INVALID_VALUE;
}

bool test_library_dispatch() {
    DriverFunctionTable functions;
    functions.library_load_data = probe_library_load;
    functions.library_load_from_file = probe_library_file;
    functions.library_get_module = probe_library_module;
    DriverDispatch driver(functions);
    CUlibrary library{};
    CUjit_option option = CU_JIT_MAX_REGISTERS;
    CUlibraryOption library_option = CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE;
    void* value = &library;
    const char image[] = "opaque";
    bool passed = true;
    for (const bool is_file : {false, true}) {
        const auto before = g_library_probe.calls;
        const auto result = is_file
                                ? driver.library_load_from_file(&library, image, &option, &value, 3,
                                                                &library_option, &value, 4)
                                : driver.library_load_data(&library, image, &option, &value, 3,
                                                           &library_option, &value, 4);
        passed &= expect(
            result == CUDA_ERROR_INVALID_IMAGE && library == nullptr &&
                g_library_probe.output == &library && g_library_probe.image == image &&
                g_library_probe.options == &option && g_library_probe.values == &value &&
                g_library_probe.count == 3 && g_library_probe.library_options == &library_option &&
                g_library_probe.library_values == &value && g_library_probe.library_count == 4 &&
                g_library_probe.calls == before + 1,
            "library dispatch forwards all original options and errors exactly once");
    }
    const auto before = g_library_probe.calls;
    passed &= expect(driver.library_get_module(nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE &&
                         g_library_probe.calls == before + 1 && !g_guard_failed &&
                         !is_inside_driver_call(),
                     "library module dispatch is guarded exact-once");
    DriverDispatch absent(DriverFunctionTable{});
    passed &=
        expect(absent.library_load_data(nullptr, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
                   absent.library_get_module(nullptr, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
                   absent.library_load_from_file(nullptr, nullptr, nullptr, nullptr, 0, nullptr,
                                                 nullptr, 0) == CUDA_ERROR_NOT_SUPPORTED &&
                   g_library_probe.calls == before + 1,
               "missing library APIs do not replay");
    return passed;
}

}  // namespace

int main() {
    if (!test_library_dispatch()) return EXIT_FAILURE;
    if (!expect(test_function_module_dispatch(),
                "typed original function/module dispatch failed") ||
        !expect(test_function_loading_dispatch(),
                "function loading dispatch or native guard failed")) {
        return EXIT_FAILURE;
    }
    if (!expect(test_extended_dispatch(), "extended dispatch or missing-symbol handling failed")) {
        return EXIT_FAILURE;
    }
    if (!expect(test_capture_dispatch(), "capture dispatch or missing-symbol handling failed")) {
        return EXIT_FAILURE;
    }
    const glimmer::interceptor::DriverFunctionTable functions{
        .init = &fake_init,
        .launch_kernel = &fake_launch_kernel,
        .launch_kernel_ptsz = &fake_launch_kernel,
        .graph_launch = &fake_graph_launch,
        .graph_launch_ptsz = &fake_graph_launch,
        .mem_alloc = &fake_mem_alloc,
        .mem_alloc_managed = &fake_mem_alloc_managed,
        .mem_alloc_pitch = &fake_mem_alloc_pitch,
        .mem_alloc_async = &fake_mem_alloc_async,
        .mem_alloc_async_ptsz = &fake_mem_alloc_async_ptsz,
        .mem_alloc_from_pool_async = &fake_mem_alloc_from_pool_async,
        .mem_alloc_from_pool_async_ptsz = &fake_mem_alloc_from_pool_async_ptsz,
        .mem_create = &fake_mem_create,
        .mem_release = &fake_mem_release,
        .mem_address_reserve = &fake_mem_address_reserve,
        .mem_address_free = &fake_mem_address_free,
        .mem_map = &fake_mem_map,
        .mem_unmap = &fake_mem_unmap,
        .mem_set_access = &fake_mem_set_access,
        .mem_get_address_range = &fake_mem_get_address_range,
        .mem_get_access = &fake_mem_get_access,
        .mem_export_to_shareable_handle = &fake_mem_export,
        .mem_import_from_shareable_handle = &fake_mem_import,
        .mem_get_allocation_granularity = &fake_mem_granularity,
        .mem_get_allocation_properties = &fake_mem_properties,
        .mem_retain_allocation_handle = &fake_mem_retain,
        .mem_pool_trim_to = &fake_pool_trim,
        .mem_pool_set_attribute = &fake_pool_set_attribute,
        .mem_pool_get_attribute = &fake_pool_get_attribute,
        .mem_pool_set_access = &fake_pool_set_access,
        .mem_pool_get_access = &fake_pool_get_access,
        .mem_pool_create = &fake_pool_create,
        .mem_pool_destroy = &fake_pool_destroy,
        .device_get_mem_pool = &fake_device_get_pool,
        .device_set_mem_pool = &fake_device_set_pool,
        .device_get_default_mem_pool = &fake_device_get_default_pool,
        .mem_get_default_mem_pool = &fake_get_pool,
        .mem_get_mem_pool = &fake_get_pool,
        .mem_set_mem_pool = &fake_set_pool,
        .mem_pool_export_to_shareable_handle = &fake_pool_export,
        .mem_pool_import_from_shareable_handle = &fake_pool_import,
        .mem_pool_export_pointer = &fake_pool_export_pointer,
        .mem_pool_import_pointer = &fake_pool_import_pointer,
        .mem_free = &fake_mem_free,
        .mem_free_async = &fake_mem_free_async,
        .mem_free_async_ptsz = &fake_mem_free_async_ptsz,
        .mem_get_info = &fake_mem_get_info,
        .device_total_mem = &fake_device_total_mem,
        .context_synchronize = &fake_context_synchronize,
        .context_get_current = &fake_context_get_current,
        .context_get_device = &fake_context_get_device,
        .context_destroy = &fake_context_destroy,
        .stream_get_device = &fake_stream_get_device,
        .stream_get_device_ptsz = &fake_stream_get_device_ptsz,
        .stream_get_context = &fake_stream_get_context,
        .stream_get_context_ptsz = &fake_stream_get_context_ptsz,
        .stream_query = &fake_stream_query,
        .stream_query_ptsz = &fake_stream_query_ptsz,
        .stream_synchronize = &fake_stream_synchronize,
        .stream_synchronize_ptsz = &fake_stream_synchronize_ptsz,
        .stream_destroy = &fake_stream_destroy,
        .get_proc_address = &fake_get_proc_address,
        .get_proc_address_v2 = &fake_get_proc_address_v2,
    };
    const DriverDispatch dispatch(functions);
    bool all_passed = true;
    all_passed &= test_primary_context_state();
    all_passed &= test_primary_context_identity();
    all_passed &= test_unbound_active_primary_identity();
    all_passed &= test_primary_reset_dispatch();
    all_passed &= test_primary_release_dispatch();
    all_passed &= test_context_destroy_dispatch();
    all_passed &= test_context_version_dispatch();
    all_passed &=
        expect(glimmer::interceptor::is_interceptor_driver_function(
                   reinterpret_cast<void*>(&fake_symbol)) &&
                   !glimmer::interceptor::is_interceptor_driver_function(
                       reinterpret_cast<void*>(&std::malloc)) &&
                   !glimmer::interceptor::is_interceptor_driver_function(nullptr),
               "native dispatch ownership must distinguish this module from foreign native code");

    all_passed &= expect(dispatch.has_get_proc_address(), "legacy resolver was not available");
    all_passed &= expect(dispatch.has_get_proc_address_v2(), "v2 resolver was not available");
    all_passed &= expect(
        dispatch.is_proc_address_function(reinterpret_cast<void*>(&fake_get_proc_address), false) &&
            !dispatch.is_proc_address_function(reinterpret_cast<void*>(&fake_get_proc_address),
                                               true) &&
            dispatch.is_proc_address_function(reinterpret_cast<void*>(&fake_get_proc_address_v2),
                                              true) &&
            !dispatch.is_proc_address_function(reinterpret_cast<void*>(&fake_get_proc_address_v2),
                                               false) &&
            !dispatch.is_proc_address_function(nullptr, false),
        "resolver ABI identity did not match its native entry point");
    all_passed &= expect(dispatch.has_launch_kernel(), "kernel launch was not available");
    all_passed &= expect(dispatch.has_launch_kernel_ptsz(), "PTDS kernel launch was not available");
    all_passed &= expect(dispatch.has_mem_alloc_managed(), "managed allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_pitch(), "pitched allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_async(), "async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_async_ptsz(), "PTDS async allocator was not available");
    all_passed &=
        expect(dispatch.has_mem_alloc_from_pool_async(), "pool async allocator was not available");
    all_passed &= expect(dispatch.has_mem_alloc_from_pool_async_ptsz(),
                         "PTDS pool async allocator was not available");
    all_passed &= expect(dispatch.has_mem_create(), "VMM allocator was not available");
    all_passed &= expect(dispatch.has_mem_release(), "VMM release was not available");
    all_passed &=
        expect(dispatch.has_mem_address_reserve(), "VMM address reserve was not available");
    all_passed &= expect(dispatch.has_mem_address_free(), "VMM address free was not available");
    all_passed &= expect(dispatch.has_mem_map(), "VMM map was not available");
    all_passed &= expect(dispatch.has_mem_unmap(), "VMM unmap was not available");
    all_passed &= expect(dispatch.has_mem_set_access(), "VMM access update was not available");
    all_passed &=
        expect(dispatch.has_mem_get_address_range(), "VMM address query was not available");
    all_passed &= expect(dispatch.has_mem_get_access(), "VMM access query was not available");
    all_passed &= expect(dispatch.has_mem_export_to_shareable_handle(),
                         "VMM handle export was not available");
    all_passed &= expect(dispatch.has_mem_import_from_shareable_handle(),
                         "VMM handle import was not available");
    all_passed &= expect(dispatch.has_mem_get_allocation_granularity(),
                         "VMM granularity query was not available");
    all_passed &= expect(dispatch.has_mem_get_allocation_properties(),
                         "VMM properties query was not available");
    all_passed &=
        expect(dispatch.has_mem_retain_allocation_handle(), "VMM handle retain was not available");
    all_passed &= expect(dispatch.has_mem_pool_trim_to(), "memory-pool trim was not available");
    all_passed &= expect(dispatch.has_mem_pool_set_attribute(),
                         "memory-pool attribute update was not available");
    all_passed &= expect(dispatch.has_mem_pool_get_attribute(),
                         "memory-pool attribute query was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_set_access(), "memory-pool access update was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_get_access(), "memory-pool access query was not available");
    all_passed &= expect(dispatch.has_mem_pool_create(), "memory-pool creation was not available");
    all_passed &=
        expect(dispatch.has_mem_pool_destroy(), "memory-pool destruction was not available");
    all_passed &=
        expect(dispatch.has_device_get_mem_pool(), "device memory-pool query was not available");
    all_passed &=
        expect(dispatch.has_device_set_mem_pool(), "device memory-pool update was not available");
    all_passed &= expect(dispatch.has_device_get_default_mem_pool(),
                         "device default memory-pool query was not available");
    all_passed &= expect(dispatch.has_mem_get_default_mem_pool(),
                         "default memory-pool query was not available");
    all_passed &= expect(dispatch.has_mem_get_mem_pool(), "memory-pool lookup was not available");
    all_passed &=
        expect(dispatch.has_mem_set_mem_pool(), "memory-pool selection was not available");
    all_passed &= expect(dispatch.has_mem_pool_export_to_shareable_handle(),
                         "memory-pool handle export was not available");
    all_passed &= expect(dispatch.has_mem_pool_import_from_shareable_handle(),
                         "memory-pool handle import was not available");
    all_passed &= expect(dispatch.has_mem_pool_export_pointer(),
                         "memory-pool pointer export was not available");
    all_passed &= expect(dispatch.has_mem_pool_import_pointer(),
                         "memory-pool pointer import was not available");
    all_passed &= expect(dispatch.has_mem_free_async(), "async free was not available");
    all_passed &= expect(dispatch.has_mem_free_async_ptsz(), "PTDS async free was not available");
    all_passed &= expect(dispatch.has_device_total_mem(), "capacity query was not available");
    all_passed &=
        expect(dispatch.has_context_synchronize(), "context synchronization was not available");
    all_passed &= expect(dispatch.has_context_queries(), "context queries were not available");
    all_passed &= expect(dispatch.has_context_destroy(), "context destruction was not available");
    all_passed &= expect(dispatch.has_stream_identity(), "stream identity was not available");
    all_passed &=
        expect(dispatch.has_stream_identity_ptsz(), "PTDS stream identity was not available");
    all_passed &= expect(dispatch.has_stream_query(), "stream query was not available");
    all_passed &= expect(dispatch.has_stream_query_ptsz(), "PTDS stream query was not available");
    all_passed &=
        expect(dispatch.has_stream_synchronize(), "stream synchronization was not available");
    all_passed &= expect(dispatch.has_stream_synchronize_ptsz(),
                         "PTDS stream synchronization was not available");
    all_passed &= expect(dispatch.has_stream_destroy(), "stream destruction was not available");

    CUdeviceptr device_pointer{};
    std::size_t pitch{};
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    std::size_t address_size{};
    std::size_t granularity{};
    CUdeviceptr address_base{};
    unsigned long long access_flags{};
    CUmemAllocationProp allocation_properties{};
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id = 0;
    CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_NONE;
    unsigned long long shareable_handle{};
    CUmemGenericAllocationHandle imported_handle{};
    CUmemGenericAllocationHandle retained_handle{};
    CUmemPoolProps pool_properties{};
    pool_properties.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUmemoryPool pool{};
    CUmemoryPool queried_pool{};
    CUmemLocation pool_location = pool_properties.location;
    std::uint64_t pool_attribute_value{};
    CUmemAccess_flags pool_access_flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    CUmemAccessDesc pool_access_descriptor{};
    CUmemPoolPtrExportData pool_pointer_data{};
    CUdeviceptr imported_pointer{};
    void* function_pointer = nullptr;
    CUdriverProcAddressQueryResult symbol_status{};
    CUcontext context{};
    CUdevice device{};
    all_passed &= expect(dispatch.init(0) == CUDA_SUCCESS, "fake cuInit failed");
    all_passed &= expect(dispatch.launch_kernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                                                nullptr) == CUDA_SUCCESS,
                         "fake cuLaunchKernel failed");
    all_passed &= expect(dispatch.launch_kernel_ptsz(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                                                     nullptr) == CUDA_SUCCESS,
                         "fake cuLaunchKernel_ptsz failed");
    all_passed &= expect(dispatch.graph_launch(nullptr, nullptr) == CUDA_SUCCESS,
                         "fake cuGraphLaunch failed");
    all_passed &= expect(dispatch.graph_launch_ptsz(nullptr, nullptr) == CUDA_SUCCESS,
                         "fake cuGraphLaunch_ptsz failed");
    all_passed &=
        expect(dispatch.mem_alloc(&device_pointer, 16) == CUDA_SUCCESS, "fake cuMemAlloc failed");
    all_passed &= expect(dispatch.mem_alloc_managed(&device_pointer, 16, 0) == CUDA_SUCCESS,
                         "fake cuMemAllocManaged failed");
    all_passed &= expect(dispatch.mem_alloc_async(&device_pointer, 16, nullptr) == CUDA_SUCCESS,
                         "fake cuMemAllocAsync failed");
    all_passed &=
        expect(dispatch.mem_alloc_async_ptsz(&device_pointer, 16, nullptr) == CUDA_SUCCESS,
               "fake cuMemAllocAsync_ptsz failed");
    all_passed &= expect(
        dispatch.mem_alloc_from_pool_async(&device_pointer, 16, nullptr, nullptr) == CUDA_SUCCESS,
        "fake cuMemAllocFromPoolAsync failed");
    all_passed &= expect(dispatch.mem_alloc_from_pool_async_ptsz(&device_pointer, 16, nullptr,
                                                                 nullptr) == CUDA_SUCCESS,
                         "fake cuMemAllocFromPoolAsync_ptsz failed");
    CUmemGenericAllocationHandle vmm_handle = 0;
    all_passed &= expect(
        dispatch.mem_create(&vmm_handle, 16, nullptr, 0) == CUDA_SUCCESS && vmm_handle == 0x42,
        "fake cuMemCreate failed");
    all_passed &=
        expect(dispatch.mem_release(vmm_handle) == CUDA_SUCCESS, "fake cuMemRelease failed");
    CUdeviceptr virtual_address = 0;
    all_passed &=
        expect(dispatch.mem_address_reserve(&virtual_address, 16, 0, 0, 0) == CUDA_SUCCESS &&
                   virtual_address != 0,
               "fake cuMemAddressReserve failed");
    all_passed &= expect(dispatch.mem_map(virtual_address, 16, 0, 1, 0) == CUDA_SUCCESS,
                         "fake cuMemMap failed");
    all_passed &= expect(dispatch.mem_set_access(virtual_address, 16, nullptr, 0) == CUDA_SUCCESS,
                         "fake cuMemSetAccess failed");
    all_passed &= expect(dispatch.mem_get_address_range(&address_base, &address_size,
                                                        virtual_address) == CUDA_SUCCESS &&
                             address_base == virtual_address && address_size == 4096,
                         "fake cuMemGetAddressRange failed");
    all_passed &= expect(dispatch.mem_get_access(&access_flags, &allocation_prop.location,
                                                 virtual_address) == CUDA_SUCCESS &&
                             access_flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                         "fake cuMemGetAccess failed");
    all_passed &= expect(dispatch.mem_export_to_shareable_handle(&shareable_handle, vmm_handle,
                                                                 handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemExportToShareableHandle failed");
    all_passed &= expect(dispatch.mem_import_from_shareable_handle(
                             &imported_handle, &shareable_handle, handle_type) == CUDA_SUCCESS &&
                             imported_handle != 0,
                         "fake cuMemImportFromShareableHandle failed");
    all_passed &= expect(
        dispatch.mem_get_allocation_granularity(&granularity, &allocation_prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
            granularity == 65536,
        "fake cuMemGetAllocationGranularity failed");
    all_passed &= expect(
        dispatch.mem_get_allocation_properties(&allocation_properties, vmm_handle) == CUDA_SUCCESS,
        "fake cuMemGetAllocationPropertiesFromHandle failed");
    all_passed &= expect(
        dispatch.mem_retain_allocation_handle(&retained_handle, &virtual_address) == CUDA_SUCCESS &&
            retained_handle != 0,
        "fake cuMemRetainAllocationHandle failed");
    all_passed &=
        expect(dispatch.mem_pool_create(&pool, &pool_properties) == CUDA_SUCCESS && pool != nullptr,
               "fake cuMemPoolCreate failed");
    all_passed &= expect(dispatch.mem_pool_set_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                                         &pool_attribute_value) == CUDA_SUCCESS,
                         "fake cuMemPoolSetAttribute failed");
    all_passed &= expect(dispatch.mem_pool_get_attribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                                         &pool_attribute_value) == CUDA_SUCCESS,
                         "fake cuMemPoolGetAttribute failed");
    all_passed &=
        expect(dispatch.mem_pool_set_access(pool, &pool_access_descriptor, 1) == CUDA_SUCCESS,
               "fake cuMemPoolSetAccess failed");
    all_passed &= expect(
        dispatch.mem_pool_get_access(&pool_access_flags, pool, &pool_location) == CUDA_SUCCESS,
        "fake cuMemPoolGetAccess failed");
    all_passed &=
        expect(dispatch.mem_pool_trim_to(pool, 0) == CUDA_SUCCESS, "fake cuMemPoolTrimTo failed");
    all_passed &= expect(
        dispatch.device_get_mem_pool(&queried_pool, 0) == CUDA_SUCCESS && queried_pool != nullptr,
        "fake cuDeviceGetMemPool failed");
    all_passed &= expect(dispatch.device_set_mem_pool(0, pool) == CUDA_SUCCESS,
                         "fake cuDeviceSetMemPool failed");
    all_passed &= expect(dispatch.device_get_default_mem_pool(&queried_pool, 0) == CUDA_SUCCESS &&
                             queried_pool != nullptr,
                         "fake cuDeviceGetDefaultMemPool failed");
    all_passed &=
        expect(dispatch.mem_get_default_mem_pool(&queried_pool, &pool_location,
                                                 CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
               "fake cuMemGetDefaultMemPool failed");
    all_passed &= expect(dispatch.mem_get_mem_pool(&queried_pool, &pool_location,
                                                   CU_MEM_ALLOCATION_TYPE_PINNED) == CUDA_SUCCESS,
                         "fake cuMemGetMemPool failed");
    all_passed &= expect(dispatch.mem_set_mem_pool(&pool_location, CU_MEM_ALLOCATION_TYPE_PINNED,
                                                   pool) == CUDA_SUCCESS,
                         "fake cuMemSetMemPool failed");
    all_passed &= expect(dispatch.mem_pool_export_to_shareable_handle(
                             &shareable_handle, pool, handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemPoolExportToShareableHandle failed");
    all_passed &= expect(dispatch.mem_pool_import_from_shareable_handle(
                             &queried_pool, &shareable_handle, handle_type, 0) == CUDA_SUCCESS,
                         "fake cuMemPoolImportFromShareableHandle failed");
    all_passed &= expect(
        dispatch.mem_pool_export_pointer(&pool_pointer_data, virtual_address) == CUDA_SUCCESS,
        "fake cuMemPoolExportPointer failed");
    all_passed &= expect(dispatch.mem_pool_import_pointer(&imported_pointer, pool,
                                                          &pool_pointer_data) == CUDA_SUCCESS,
                         "fake cuMemPoolImportPointer failed");
    all_passed &=
        expect(dispatch.mem_pool_destroy(pool) == CUDA_SUCCESS, "fake cuMemPoolDestroy failed");
    all_passed &=
        expect(dispatch.mem_unmap(virtual_address, 16) == CUDA_SUCCESS, "fake cuMemUnmap failed");
    all_passed &= expect(dispatch.mem_address_free(virtual_address, 16) == CUDA_SUCCESS,
                         "fake cuMemAddressFree failed");
    all_passed &=
        expect(dispatch.mem_alloc_pitch(&device_pointer, &pitch, 16, 16, 4) == CUDA_SUCCESS,
               "fake cuMemAllocPitch failed");
    all_passed &=
        expect(dispatch.mem_free(device_pointer) == CUDA_SUCCESS, "fake cuMemFree failed");
    all_passed &= expect(dispatch.mem_free_async(device_pointer, nullptr) == CUDA_SUCCESS,
                         "fake cuMemFreeAsync failed");
    all_passed &= expect(dispatch.mem_free_async_ptsz(device_pointer, nullptr) == CUDA_SUCCESS,
                         "fake cuMemFreeAsync_ptsz failed");
    all_passed &= expect(dispatch.mem_get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
                             free_bytes == 100 && total_bytes == 200,
                         "fake cuMemGetInfo failed");
    all_passed &=
        expect(dispatch.device_total_mem(&total_bytes, 0) == CUDA_SUCCESS && total_bytes == 200,
               "fake cuDeviceTotalMem failed");
    all_passed &= expect(dispatch.context_get_current(&context) == CUDA_SUCCESS,
                         "fake cuCtxGetCurrent failed");
    all_passed &= expect(dispatch.context_get_device(&device) == CUDA_SUCCESS && device == 0,
                         "fake cuCtxGetDevice failed");
    all_passed &=
        expect(dispatch.context_destroy(context) == CUDA_SUCCESS, "fake cuCtxDestroy failed");
    all_passed &=
        expect(dispatch.context_synchronize() == CUDA_SUCCESS, "fake cuCtxSynchronize failed");
    all_passed &= expect(dispatch.stream_get_device(nullptr, &device) == CUDA_SUCCESS,
                         "fake cuStreamGetDevice failed");
    all_passed &= expect(dispatch.stream_get_device_ptsz(nullptr, &device) == CUDA_SUCCESS,
                         "fake cuStreamGetDevice_ptsz failed");
    all_passed &= expect(dispatch.stream_get_context(nullptr, &context) == CUDA_SUCCESS,
                         "fake cuStreamGetCtx failed");
    all_passed &= expect(dispatch.stream_get_context_ptsz(nullptr, &context) == CUDA_SUCCESS,
                         "fake cuStreamGetCtx_ptsz failed");
    all_passed &=
        expect(dispatch.stream_query(nullptr) == CUDA_SUCCESS, "fake cuStreamQuery failed");
    all_passed &= expect(dispatch.stream_query_ptsz(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamQuery_ptsz failed");
    all_passed &= expect(dispatch.stream_synchronize(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamSynchronize failed");
    all_passed &= expect(dispatch.stream_synchronize_ptsz(nullptr) == CUDA_SUCCESS,
                         "fake cuStreamSynchronize_ptsz failed");
    all_passed &=
        expect(dispatch.stream_destroy(nullptr) == CUDA_SUCCESS, "fake cuStreamDestroy failed");
    all_passed &=
        expect(dispatch.get_proc_address("fake", &function_pointer, 0, 0) == CUDA_SUCCESS &&
                   function_pointer != nullptr,
               "fake legacy resolver failed");
    function_pointer = nullptr;
    all_passed &=
        expect(dispatch.get_proc_address_v2("fake", &function_pointer, 0, 0, &symbol_status) ==
                       CUDA_SUCCESS &&
                   function_pointer != nullptr && symbol_status == CU_GET_PROC_ADDRESS_SUCCESS,
               "fake v2 resolver failed");
    all_passed &= expect(!g_guard_failed, "Driver call guard was not active");

    DriverFunctionTable recursive_functions = functions;
    recursive_functions.get_proc_address_v2 = &recursive_get_proc_address_v2;
    const DriverDispatch recursive_dispatch(recursive_functions);
    g_recursive_dispatch = &recursive_dispatch;
    function_pointer = nullptr;
    all_passed &=
        expect(recursive_dispatch.get_proc_address_v2("recursive", &function_pointer, 0, 0,
                                                      &symbol_status) == CUDA_ERROR_NOT_SUPPORTED,
               "recursive proc-address dispatch was not rejected");
    g_recursive_dispatch = nullptr;

    DriverFunctionTable legacy_only_functions = functions;
    legacy_only_functions.mem_alloc_async_ptsz = nullptr;
    legacy_only_functions.mem_alloc_from_pool_async_ptsz = nullptr;
    legacy_only_functions.mem_free_async_ptsz = nullptr;
    legacy_only_functions.mem_address_reserve = nullptr;
    legacy_only_functions.mem_address_free = nullptr;
    legacy_only_functions.mem_map = nullptr;
    legacy_only_functions.mem_unmap = nullptr;
    legacy_only_functions.mem_set_access = nullptr;
    legacy_only_functions.mem_get_address_range = nullptr;
    legacy_only_functions.mem_get_access = nullptr;
    legacy_only_functions.mem_pool_create = nullptr;
    legacy_only_functions.mem_pool_destroy = nullptr;
    legacy_only_functions.device_get_default_mem_pool = nullptr;
    legacy_only_functions.stream_get_device_ptsz = nullptr;
    legacy_only_functions.stream_get_context_ptsz = nullptr;
    legacy_only_functions.stream_query_ptsz = nullptr;
    legacy_only_functions.stream_synchronize_ptsz = nullptr;
    const DriverDispatch legacy_only_dispatch(legacy_only_functions);
    all_passed &= expect(!legacy_only_dispatch.has_mem_alloc_async_ptsz(),
                         "legacy allocator was reported as a PTDS allocator");
    all_passed &= expect(!legacy_only_dispatch.has_mem_alloc_from_pool_async_ptsz(),
                         "legacy pool allocator was reported as a PTDS allocator");
    all_passed &= expect(!legacy_only_dispatch.has_mem_free_async_ptsz(),
                         "legacy free was reported as a PTDS free");
    all_passed &= expect(!legacy_only_dispatch.has_mem_address_reserve(),
                         "missing VMM address reserve was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_address_free(),
                         "missing VMM address free was reported as available");
    all_passed &=
        expect(!legacy_only_dispatch.has_mem_map(), "missing VMM map was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_unmap(),
                         "missing VMM unmap was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_set_access(),
                         "missing VMM access update was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_get_address_range(),
                         "missing VMM address query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_get_access(),
                         "missing VMM access query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_pool_create(),
                         "missing memory-pool creation was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_mem_pool_destroy(),
                         "missing memory-pool destruction was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_device_get_default_mem_pool(),
                         "missing device default memory-pool query was reported as available");
    all_passed &= expect(!legacy_only_dispatch.has_stream_identity_ptsz(),
                         "legacy stream identity was reported as PTDS identity");
    all_passed &= expect(!legacy_only_dispatch.has_stream_query_ptsz(),
                         "legacy stream query was reported as PTDS query");
    all_passed &= expect(!legacy_only_dispatch.has_stream_synchronize_ptsz(),
                         "legacy stream synchronization was reported as PTDS synchronization");
    all_passed &= expect(legacy_only_dispatch.mem_alloc_async_ptsz(&device_pointer, 16, nullptr) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing PTDS allocator did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_free_async_ptsz(device_pointer, nullptr) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing PTDS free did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_address_reserve(&virtual_address, 16, 0, 0, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing VMM address reserve did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_address_free(virtual_address, 16) == CUDA_ERROR_NOT_SUPPORTED,
        "missing VMM address free did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_map(virtual_address, 16, 0, 1, 0) == CUDA_ERROR_NOT_SUPPORTED,
        "missing VMM map did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.mem_unmap(virtual_address, 16) == CUDA_ERROR_NOT_SUPPORTED,
               "missing VMM unmap did not fail closed");
    all_passed &= expect(legacy_only_dispatch.mem_set_access(virtual_address, 16, nullptr, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing VMM access update did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.mem_get_address_range(
                   &address_base, &address_size, virtual_address) == CUDA_ERROR_NOT_SUPPORTED,
               "missing VMM address query did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.mem_pool_create(&pool, &pool_properties) == CUDA_ERROR_NOT_SUPPORTED,
        "missing memory-pool creation did not fail closed");
    all_passed &= expect(legacy_only_dispatch.device_get_default_mem_pool(&queried_pool, 0) ==
                             CUDA_ERROR_NOT_SUPPORTED,
                         "missing device default memory-pool query did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.stream_query_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS query did not fail closed");
    all_passed &=
        expect(legacy_only_dispatch.stream_synchronize_ptsz(nullptr) == CUDA_ERROR_NOT_SUPPORTED,
               "missing PTDS synchronization did not fail closed");
    all_passed &= expect(
        legacy_only_dispatch.event_record_ptsz(nullptr, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
            legacy_only_dispatch.event_record_with_flags_ptsz(nullptr, nullptr, 0) ==
                CUDA_ERROR_NOT_SUPPORTED &&
            legacy_only_dispatch.stream_wait_event_ptsz(nullptr, nullptr, 0) ==
                CUDA_ERROR_NOT_SUPPORTED,
        "missing PTDS event APIs must not fall back to legacy semantics");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
