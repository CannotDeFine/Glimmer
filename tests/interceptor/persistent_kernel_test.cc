#include "internal/persistent_kernel.h"
#include "internal/qualified_cubin_source.h"
#include "internal/persistent_sgemm.h"
#include "persistent_ptx_fixture.h"
#include "cooperative_ptx_fixture.h"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <limits>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <fstream>

namespace {

using glimmer::interceptor::DriverDispatch;
using glimmer::interceptor::DriverFunctionTable;
using glimmer::interceptor::PersistentKernelRuntime;

std::uint8_t g_module_token = 0;
std::uint8_t g_function_token = 0;
std::uint8_t g_second_function_token = 0;
std::uint8_t g_variant_token = 0;
std::uint8_t g_second_variant_token = 0;
std::uint8_t g_context_token = 0;
std::uint8_t g_stream_token = 0;

CUmodule module_handle() {
    return reinterpret_cast<CUmodule>(&g_module_token);
}
CUfunction original_handle() {
    return reinterpret_cast<CUfunction>(&g_function_token);
}
CUfunction variant_handle() {
    return reinterpret_cast<CUfunction>(&g_variant_token);
}
CUcontext context_handle() {
    return reinterpret_cast<CUcontext>(&g_context_token);
}
CUstream stream_handle() {
    return reinterpret_cast<CUstream>(&g_stream_token);
}

struct StubState {
    bool guard_failed = false;
    bool should_fail_jit = false;
    bool should_fail_original = false;
    bool should_fail_lookup = false;
    bool should_fail_unload = false;
    bool should_fail_context = false;
    bool should_fail_occupancy = false;
    bool should_fail_capture_query = false;
    bool should_fail_library_load = false;
    bool should_fail_library_lookup = false;
    bool should_fail_library_unload = false;
    bool should_publish_failed_library = false;
    bool should_fail_function_load = false;
    bool should_fail_function_module = false;
    bool should_return_null_function_module = false;
    bool should_fail_original_lookup = false;
    CUcontext current_context = context_handle();
    CUmodule function_module = module_handle();
    CUfunction rejected_function_load = nullptr;
    bool should_fail_device_attribute = false;
    bool is_block_static = false;
    bool was_transformed = false;
    bool used_ptds = false;
    bool used_capture_ptds = false;
    int resident_blocks = 1;
    int compute_major = 7;
    int grid_x_limit = std::numeric_limits<int>::max();
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    unsigned int load_count = 0;
    unsigned int launch_count = 0;
    unsigned int library_load_count = 0;
    unsigned int library_unload_count = 0;
    unsigned int library_lookup_count = 0;
    unsigned int kernel_function_count = 0;
    unsigned int native_call_count = 0;
    unsigned int module_lookup_count = 0;
    unsigned int function_load_count = 0;
    unsigned int function_module_count = 0;
    unsigned int module_unload_count = 0;
    unsigned int original_grid = 0;
    unsigned int last_grid = 0;
    unsigned int last_grid_y = 0;
    unsigned int last_grid_z = 0;
    unsigned int original_grid_y = 0;
    unsigned int original_grid_z = 0;
    int occupancy_threads = 0;
    unsigned int last_shared_bytes = 0;
    CUfunction last_function = nullptr;
    CUstream last_stream = nullptr;
    void** last_arguments = nullptr;
    void** last_extra = nullptr;
    const void* last_image = nullptr;
    CUjit_option* last_options = nullptr;
    void** last_option_values = nullptr;
    void* output_argument = nullptr;
    void* count_argument = nullptr;
    CUresult launch_result = CUDA_SUCCESS;
};

StubState g_state;
std::mutex g_function_module_mutex;

bool expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

void check_guard() {
    ++g_state.native_call_count;
    g_state.guard_failed |= !glimmer::interceptor::is_inside_driver_call();
}

CUresult load_module(CUmodule* module, const void* image) {
    check_guard();
    ++g_state.load_count;
    g_state.last_image = image;
    if (module == nullptr || image == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto* text = static_cast<const char*>(image);
    const bool transformed =
        text[0] != '\x7f' && std::string_view(text).find("__glimmer_ptb") != std::string_view::npos;
    if ((transformed && g_state.should_fail_jit) ||
        (!transformed && g_state.should_fail_original)) {
        return CUDA_ERROR_INVALID_PTX;
    }
    g_state.was_transformed = transformed;
    *module = module_handle();
    return CUDA_SUCCESS;
}

CUresult load_module_ex(CUmodule* module, const void* image, unsigned int, CUjit_option* options,
                        void** option_values) {
    g_state.last_options = options;
    g_state.last_option_values = option_values;
    return load_module(module, image);
}

CUresult get_function(CUfunction* function, CUmodule, const char* name) {
    check_guard();
    ++g_state.module_lookup_count;
    if (function == nullptr || name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (std::string_view(name) == "missing") {
        return CUDA_ERROR_NOT_FOUND;
    }
    if (std::string_view(name).find("__glimmer_ptb") != std::string_view::npos) {
        if (g_state.should_fail_lookup) {
            return CUDA_ERROR_NOT_FOUND;
        }
        *function = variant_handle();
        g_state.is_block_static = std::string_view(name).ends_with("_block_static");
    } else {
        if (g_state.should_fail_original_lookup) {
            return CUDA_ERROR_NOT_FOUND;
        }
        *function = std::string_view(name) == "native_barrier"
                        ? reinterpret_cast<CUfunction>(&g_second_function_token)
                        : original_handle();
    }
    return CUDA_SUCCESS;
}

CUresult unload_module(CUmodule) {
    check_guard();
    ++g_state.module_unload_count;
    return g_state.should_fail_unload ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

CUresult get_context(CUcontext* context) {
    check_guard();
    *context = g_state.current_context;
    return g_state.should_fail_context ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
}

CUresult get_device(CUdevice* device) {
    check_guard();
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult occupancy(int* blocks, CUfunction, int threads, std::size_t) {
    check_guard();
    *blocks = g_state.resident_blocks;
    g_state.occupancy_threads = threads;
    return g_state.should_fail_occupancy ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUresult device_attribute(int* value, CUdevice_attribute attribute, CUdevice) {
    check_guard();
    if (g_state.should_fail_device_attribute) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    switch (attribute) {
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *value = g_state.compute_major;
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *value = 0;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
            *value = g_state.grid_x_limit;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *value = 65535;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *value = 64;
            break;
        default:
            *value = 1024;
            break;
    }
    return CUDA_SUCCESS;
}

CUresult stream_capture(CUstream, CUstreamCaptureStatus* status) {
    check_guard();
    *status = g_state.capture_status;
    return g_state.should_fail_capture_query ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

CUresult stream_capture_ptds(CUstream stream, CUstreamCaptureStatus* status) {
    g_state.used_capture_ptds = true;
    return stream_capture(stream, status);
}

CUresult library_load(CUlibrary* library, const void* image, CUjit_option*, void**, unsigned int,
                      CUlibraryOption*, void**, unsigned int) {
    check_guard();
    ++g_state.library_load_count;
    if (g_state.should_fail_library_load) {
        if (g_state.should_publish_failed_library) {
            *library = reinterpret_cast<CUlibrary>(module_handle());
        }
        return CUDA_ERROR_INVALID_PTX;
    }
    if (static_cast<const char*>(image)[0] == '\x7f') {
        *library = reinterpret_cast<CUlibrary>(module_handle());
        return CUDA_SUCCESS;
    }
    const std::string_view source(static_cast<const char*>(image));
    if (source.find("write_indices__glimmer_ptb") == std::string_view::npos &&
        source.find(glimmer::interceptor::kPersistentSgemmName) == std::string_view::npos) {
        return CUDA_ERROR_INVALID_PTX;
    }
    *library = reinterpret_cast<CUlibrary>(module_handle());
    return CUDA_SUCCESS;
}

CUresult library_kernel(CUkernel* kernel, CUlibrary, const char* name) {
    check_guard();
    ++g_state.library_lookup_count;
    if (!g_state.should_fail_library_lookup &&
        std::string_view(name) == "native_barrier__glimmer_ptb_preemptive") {
        *kernel = reinterpret_cast<CUkernel>(&g_second_variant_token);
        return CUDA_SUCCESS;
    }
    if (g_state.should_fail_library_lookup ||
        (std::string_view(name) != "write_indices__glimmer_ptb" &&
         std::string_view(name) != "write_indices__glimmer_ptb_preemptive" &&
         std::string_view(name) != glimmer::interceptor::kQualifiedAddEntry &&
         std::string_view(name) != glimmer::interceptor::kPersistentSgemmName &&
         std::string_view(name) != glimmer::interceptor::kQualifiedGemmEntry)) {
        return CUDA_ERROR_NOT_FOUND;
    }
    *kernel = reinterpret_cast<CUkernel>(variant_handle());
    return CUDA_SUCCESS;
}

CUresult kernel_function(CUfunction* function, CUkernel kernel) {
    check_guard();
    ++g_state.kernel_function_count;
    *function = kernel == reinterpret_cast<CUkernel>(&g_second_variant_token)
                    ? reinterpret_cast<CUfunction>(&g_second_variant_token)
                    : variant_handle();
    return CUDA_SUCCESS;
}

CUresult library_unload(CUlibrary) {
    check_guard();
    ++g_state.library_unload_count;
    return g_state.should_fail_library_unload ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}

CUresult load_function(CUfunction function) {
    check_guard();
    ++g_state.function_load_count;
    return function && !g_state.should_fail_function_load &&
                   function != g_state.rejected_function_load
               ? CUDA_SUCCESS
               : CUDA_ERROR_INVALID_VALUE;
}

CUresult function_module(CUmodule* module, CUfunction function) {
    const std::scoped_lock lock(g_function_module_mutex);
    check_guard();
    ++g_state.function_module_count;
    if (g_state.should_fail_function_module || !module || !function) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *module = g_state.should_return_null_function_module ? nullptr : g_state.function_module;
    return CUDA_SUCCESS;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): fake the exact CUDA launch ABI.
CUresult launch(CUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                unsigned int, unsigned int, unsigned int, unsigned int shared_bytes,
                CUstream stream, void** arguments, void** extra) {
    check_guard();
    ++g_state.launch_count;
    g_state.last_function = function;
    g_state.last_grid = grid_x;
    g_state.last_grid_y = grid_y;
    g_state.last_grid_z = grid_z;
    g_state.last_shared_bytes = shared_bytes;
    g_state.last_stream = stream;
    g_state.last_arguments = arguments;
    g_state.last_extra = extra;
    if (function == variant_handle()) {
        const auto private_begin = g_state.is_block_static ? 3U : 2U;
        g_state.original_grid = *static_cast<unsigned int*>(arguments[private_begin]);
        if (g_state.is_block_static) {
            g_state.original_grid_y = *static_cast<unsigned int*>(arguments[private_begin + 1]);
            g_state.original_grid_z = *static_cast<unsigned int*>(arguments[private_begin + 2]);
        }
        g_state.output_argument = arguments[0];
        g_state.count_argument = arguments[1];
    }
    return g_state.launch_result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

CUresult launch_ptds(CUfunction function, unsigned int grid_x, unsigned int grid_y,
                     unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                     unsigned int block_z, unsigned int shared_bytes, CUstream stream,
                     void** arguments, void** extra) {
    g_state.used_ptds = true;
    return launch(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes, stream,
                  arguments, extra);
}

DriverFunctionTable functions() {
    DriverFunctionTable table;
    table.module_load_data = load_module;
    table.library_load_data = library_load;
    table.library_get_kernel = library_kernel;
    table.kernel_get_function = kernel_function;
    table.function_load = load_function;
    table.function_get_module = function_module;
    table.library_unload = library_unload;
    table.module_load_data_ex = load_module_ex;
    table.module_get_function = get_function;
    table.module_unload = unload_module;
    table.context_get_current = get_context;
    table.context_get_device = get_device;
    table.occupancy = occupancy;
    table.device_get_attribute = device_attribute;
    table.stream_capture_status = stream_capture;
    table.stream_capture_status_ptsz = stream_capture_ptds;
    table.launch_kernel = launch;
    table.launch_kernel_ptsz = launch_ptds;
    return table;
}

bool prepare(PersistentKernelRuntime& runtime) {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    return expect(runtime.load_module(&module, kPersistentTestPtx) == CUDA_SUCCESS &&
                      runtime.get_function(&function, module, "write_indices") == CUDA_SUCCESS &&
                      function == original_handle(),
                  "module and original application function must load");
}

bool test_owned_sgemm() {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    if (!expect(!runtime.owned_sgemm(context_handle()) && g_state.native_call_count == 0,
                "cold SGEMM lookup must be metadata-only") ||
        !expect(runtime.prepare_owned_sgemm(context_handle()) == CUDA_SUCCESS,
                "owned SGEMM cold preparation failed"))
        return false;
    const auto native_calls = g_state.native_call_count;
    auto code = runtime.owned_sgemm(context_handle());
    if (!expect(code && code->function == variant_handle() && code->worker_blocks == 3 &&
                    code->resources && g_state.native_call_count == native_calls,
                "warm owned SGEMM lookup changed native state") ||
        !expect(runtime.prepare_owned_sgemm(context_handle()) == CUDA_SUCCESS &&
                    g_state.library_load_count == 1 && g_state.occupancy_threads == 256,
                "owned SGEMM was reloaded") ||
        !expect(runtime.forget_context_checked(context_handle()) == CUDA_ERROR_NOT_READY &&
                    !runtime.owned_sgemm(context_handle()) && g_state.library_unload_count == 0,
                "retained SGEMM code lease was unloaded"))
        return false;
    code.reset();
    if (!expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                    g_state.library_unload_count == 1 && !g_state.guard_failed,
                "retired owned SGEMM was not released after its last lease"))
        return false;
    for (const int mode : {0, 1, 2, 3, 4, 5}) {
        g_state = {};
        PersistentKernelRuntime failed(driver, 3, false, true);
        g_state.should_fail_library_load = mode == 0 || mode == 5;
        g_state.should_publish_failed_library = mode == 5;
        g_state.should_fail_library_lookup = mode == 1;
        g_state.should_fail_occupancy = mode == 2;
        g_state.resident_blocks = mode == 3 ? 0 : 1;
        g_state.should_fail_library_unload = mode == 4;
        const auto prepared = failed.prepare_owned_sgemm(context_handle());
        if (!expect((mode == 4 ? prepared == CUDA_SUCCESS : prepared != CUDA_SUCCESS),
                    "SGEMM preparation failure was not contained"))
            return false;
        if (mode != 4 &&
            !expect(!failed.owned_sgemm(context_handle()), "failed SGEMM code became selectable"))
            return false;
        const auto retired = failed.forget_context_checked(context_handle());
        const auto unloads = g_state.library_unload_count;
        (void)failed.forget_context_checked(context_handle());
        if (!expect((mode == 4 || mode == 5) ? retired != CUDA_SUCCESS : retired == CUDA_SUCCESS,
                    "SGEMM uncertain cleanup was treated as success") ||
            !expect(g_state.library_unload_count == unloads && !g_state.guard_failed,
                    "SGEMM cleanup repeated an uncertain unload"))
            return false;
    }
    g_state = {};
    PersistentKernelRuntime isolated(driver, 3, false, true);
    g_state.current_context = nullptr;
    if (!expect(isolated.prepare_owned_sgemm(context_handle()) == CUDA_ERROR_INVALID_CONTEXT &&
                    g_state.library_load_count == 0,
                "wrong-context SGEMM was prepared"))
        return false;
    g_state.current_context = context_handle();
    g_state.compute_major = 8;
    return expect(isolated.prepare_owned_sgemm(context_handle()) == CUDA_ERROR_NOT_SUPPORTED &&
                      g_state.library_load_count == 0,
                  "out-of-scope device prepared owned SGEMM");
}

bool test_launch_and_cleanup() {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false);
    if (!prepare(runtime)) {
        return false;
    }
    unsigned int output = 0;
    unsigned int count = 999;
    void* arguments[] = {&output, &count};
    const auto run = [&](bool ptds = false, unsigned int grid_y = 1, unsigned int shared_bytes = 0,
                         void** extra = nullptr) {
        return runtime.launch(original_handle(), 17, grid_y, 1, 64, 1, 1, shared_bytes,
                              stream_handle(), arguments, extra, ptds);
    };
    bool passed = expect(run(true) == CUDA_SUCCESS && g_state.was_transformed &&
                             g_state.last_function == variant_handle() && g_state.last_grid == 3 &&
                             g_state.original_grid == 17 && g_state.output_argument == &output &&
                             g_state.count_argument == &count && g_state.used_ptds &&
                             g_state.last_stream == stream_handle(),
                         "PTB must preserve arguments/stream/PTDS and cap physical worker blocks");
    passed &= expect(run(false, 2) == CUDA_SUCCESS && g_state.last_function == original_handle() &&
                         g_state.last_arguments == arguments,
                     "multidimensional launch must stay native");
    passed &=
        expect(run(false, 1, 32) == CUDA_SUCCESS && g_state.last_function == original_handle() &&
                   g_state.last_shared_bytes == 32,
               "dynamic shared-memory launch must stay native");
    passed &=
        expect(run(false, 1, 0, arguments) == CUDA_SUCCESS &&
                   g_state.last_function == original_handle() && g_state.last_extra == arguments,
               "packed extra arguments must be forwarded unchanged");
    g_state.resident_blocks = 0;
    passed &= expect(run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "nonresident variant must stay native");
    g_state.resident_blocks = 1;
    g_state.should_fail_occupancy = true;
    passed &= expect(run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "failed occupancy query must stay native");
    g_state.should_fail_occupancy = false;
    g_state.capture_status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    passed &= expect(run(true) == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "stream capture must retain the original graph function and parameter ABI");
    g_state.capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    g_state.should_fail_capture_query = true;
    passed &= expect(run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "failed capture-status query must keep original launch");
    g_state.should_fail_capture_query = false;
    g_state.launch_result = CUDA_ERROR_LAUNCH_FAILED;
    const auto launches_before = g_state.launch_count;
    passed &=
        expect(run() == CUDA_ERROR_LAUNCH_FAILED && g_state.launch_count == launches_before + 1 &&
                   g_state.last_function == variant_handle(),
               "failed real launch must not be retried");
    g_state.launch_result = CUDA_SUCCESS;
    g_state.should_fail_unload = true;
    passed &= expect(runtime.unload_module(module_handle()) == CUDA_ERROR_INVALID_HANDLE &&
                         run() == CUDA_SUCCESS && g_state.last_function == variant_handle(),
                     "failed unload must retain mappings");
    g_state.should_fail_unload = false;
    passed &= expect(runtime.unload_module(module_handle()) == CUDA_SUCCESS &&
                         run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "successful unload must remove mappings");
    passed &= prepare(runtime);
    const char reused_binary[] = {'\x7f', 'E', 'L', 'F', '\0'};
    CUmodule reused_module = nullptr;
    passed &= expect(runtime.load_module(&reused_module, reused_binary) == CUDA_SUCCESS &&
                         run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "native module load must clear recycled-handle metadata");
    passed &= prepare(runtime);
    runtime.forget_context(context_handle());
    passed &= expect(run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "destroyed context must remove mappings");
    CUfunction missing = nullptr;
    passed &=
        expect(runtime.get_function(&missing, module_handle(), "missing") == CUDA_ERROR_NOT_FOUND,
               "native lookup error must remain unchanged");
    return passed && expect(!g_state.guard_failed, "all internal CUDA calls need Driver guards");
}

bool test_checked_context_retirement() {
    bool passed = true;
    for (const bool fail_unload : {false, true}) {
        g_state = {};
        DriverDispatch driver(functions());
        PersistentKernelRuntime runtime(driver, 3, false, true);
        CUmodule module{};
        CUfunction function{};
        passed &= expect(
            runtime.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr, nullptr) ==
                    CUDA_SUCCESS &&
                runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS,
            "checked retirement owns an original source generation");
        runtime.prepare_module_source(function);
        passed &= expect(
            runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(), false)
                .has_value(),
            "checked retirement owns prepared private code");
        const auto calls = g_state.native_call_count;
        passed &= expect(runtime.forget_context_checked(nullptr) == CUDA_ERROR_INVALID_CONTEXT &&
                             g_state.native_call_count == calls,
                         "empty retirement cannot select or retry quarantined code");
        g_state.should_fail_library_unload = fail_unload;
        passed &= expect(runtime.forget_context_checked(context_handle()) ==
                                 (fail_unload ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS) &&
                             g_state.library_unload_count == 1 &&
                             !runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0,
                                                         stream_handle(), false),
                         "checked retirement reports cleanup effect and removes old bindings");
        const auto retired_calls = g_state.native_call_count;
        passed &= expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                             g_state.native_call_count == retired_calls && !g_state.guard_failed,
                         "successful or uncertain library cleanup is never replayed");
    }
    return passed;
}

bool test_block_static_launch() {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false);
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!expect(runtime.load_module(&module, kCooperativePtx) == CUDA_SUCCESS &&
                    runtime.get_function(&function, module, "cooperative") == CUDA_SUCCESS,
                "shared-memory kernel must gain a transparent block-static entry")) {
        return false;
    }
    unsigned int value = 0;
    void* arguments[] = {&value, &value, &value};
    const auto run = [&](unsigned int grid_y = 5, unsigned int grid_z = 3,
                         unsigned int block_z = 2) {
        return runtime.launch(function, 7, grid_y, grid_z, 8, 4, block_z, 0, stream_handle(),
                              arguments, nullptr, true);
    };
    bool passed =
        expect(run() == CUDA_SUCCESS && g_state.last_function == variant_handle() &&
                   g_state.last_grid == 3 && g_state.last_grid_y == 1 && g_state.last_grid_z == 1 &&
                   g_state.original_grid == 7 && g_state.original_grid_y == 5 &&
                   g_state.original_grid_z == 3 && g_state.occupancy_threads == 64 &&
                   g_state.used_ptds && g_state.last_stream == stream_handle(),
               "block-static launch must flatten only physical grid and append 3D ABI");
    for (const auto grid_y : {0U, 65536U, std::numeric_limits<unsigned int>::max()}) {
        passed &=
            expect(run(grid_y) == CUDA_SUCCESS && g_state.last_function == original_handle() &&
                       g_state.last_grid_y == grid_y && g_state.last_arguments == arguments,
                   "zero, invalid-device, and overflowing original grids must remain native");
    }
    passed &= expect(run(5, 3, 65) == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "invalid original block shape must remain native");
    g_state.should_fail_device_attribute = true;
    passed &= expect(run() == CUDA_SUCCESS && g_state.last_function == original_handle(),
                     "failed device limits query must remain native");
    g_state.should_fail_device_attribute = false;
    g_state.grid_x_limit = 2;
    passed &= expect(runtime.launch(function, 1, 5, 3, 8, 4, 2, 0, stream_handle(), arguments,
                                    nullptr, false) == CUDA_SUCCESS &&
                         g_state.last_function == original_handle(),
                     "physical worker grid must also fit the actual device limit");
    g_state.grid_x_limit = std::numeric_limits<int>::max();
    g_state.launch_result = CUDA_ERROR_LAUNCH_FAILED;
    const auto before = g_state.launch_count;
    passed &= expect(run() == CUDA_ERROR_LAUNCH_FAILED && g_state.launch_count == before + 1 &&
                         g_state.last_function == variant_handle(),
                     "failed block-static launch must not retry the original");
    auto table = functions();
    table.device_get_attribute = nullptr;
    DriverDispatch missing_driver(table);
    PersistentKernelRuntime missing_runtime(missing_driver, 3, false);
    g_state.launch_result = CUDA_SUCCESS;
    passed &=
        expect(missing_runtime.load_module(&module, kCooperativePtx) == CUDA_SUCCESS &&
                   missing_runtime.get_function(&function, module, "cooperative") == CUDA_SUCCESS &&
                   missing_runtime.launch(function, 7, 5, 3, 8, 4, 2, 0, stream_handle(), arguments,
                                          nullptr, false) == CUDA_SUCCESS &&
                   g_state.last_function == original_handle(),
               "missing device-limit API must remain native");
    return passed &&
           expect(!g_state.guard_failed, "block-static preparation requires Driver guards");
}

bool test_deferred_module_preparation() {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    CUmodule module{};
    CUfunction function{};
    std::string image = kPersistentTestPtx;
    bool passed = expect(
        runtime.load_source_module(&module, image.c_str(), false, 0, nullptr, nullptr) ==
                CUDA_SUCCESS &&
            !g_state.was_transformed && g_state.load_count == 1 &&
            g_state.library_load_count == 0 &&
            runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS &&
            function == original_handle() && g_state.module_lookup_count == 1 &&
            !runtime.module_source_is_prepared(function),
        "source load/lookup issues original SDK once without private lookup or JIT");
    image.assign(image.size(), 'x');
    runtime.prepare_module_source(function);
    const auto variant =
        runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(), false);
    passed &= expect(
        variant && variant->function == variant_handle() && variant->parameter_layout.size() == 2 &&
            runtime.module_source_is_prepared(function) && g_state.library_load_count == 1 &&
            g_state.library_lookup_count == 2 && g_state.kernel_function_count == 2 &&
            g_state.load_count == 1 && g_state.module_lookup_count == 3 &&
            g_state.function_load_count == 4 && g_state.launch_count == 0,
        "owned source survives caller overwrite and eagerly resolves both private entries");
    const auto calls = g_state.native_call_count;
    runtime.prepare_module_source(function);
    passed &= expect(g_state.native_call_count == calls,
                     "warm preparation must not repeat context queries, JIT or native lookup");
    passed &= expect(runtime.get_function(&function, module, "native_barrier") == CUDA_SUCCESS &&
                         runtime.module_source_is_prepared(function),
                     "new original function from a prepared module reuses the same generation");
    const auto second =
        runtime.cooperative_kernel(function, {3, 1, 1}, {32, 1, 1}, 0, stream_handle(), false);
    passed &= expect(
        second && second->function == reinterpret_cast<CUfunction>(&g_second_variant_token) &&
            g_state.library_load_count == 1 && g_state.module_lookup_count == 4,
        "warm new function uses eager metadata, not another private SDK lookup");
    passed &= expect(
        runtime.unload_module(module) == CUDA_SUCCESS && g_state.library_unload_count == 1 &&
            !runtime.cooperative_kernel(function, {3, 1, 1}, {32, 1, 1}, 0, stream_handle(), false),
        "successful module unload retires source association and private library once");
    passed &= expect(
        runtime.load_module(&module, kPersistentTestPtx) == CUDA_SUCCESS &&
            runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS &&
            !runtime.module_source_is_prepared(function),
        "existing transformed module supports its first function lookup during capture");
    const auto module_lookups = g_state.module_lookup_count;
    runtime.prepare_module_source(function);
    passed &= expect(
        runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(), false)
                .has_value() &&
            g_state.module_lookup_count == module_lookups + 4 && g_state.library_load_count == 1,
        "existing module prepares native private functions without an additional library");
    runtime.forget_context(context_handle());
    passed &= expect(
        !runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(), false),
        "context retirement removes source-created function bindings");
    for (const bool publishes_failed_handle : {false, true}) {
        g_state.should_fail_library_load = true;
        g_state.should_publish_failed_library = publishes_failed_handle;
        const auto loads = g_state.library_load_count;
        const auto unloads = g_state.library_unload_count;
        passed &= expect(
            runtime.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr, nullptr) ==
                    CUDA_SUCCESS &&
                runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS,
            "source original success is independent of later private JIT failure");
        runtime.prepare_module_source(function);
        runtime.prepare_module_source(function);
        passed &= expect(g_state.library_load_count == loads + 1 &&
                             runtime.module_source_is_prepared(function) &&
                             !runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0,
                                                         stream_handle(), false),
                         "failed private acquisition is terminal and cannot publish a variant");
        passed &= expect(runtime.unload_module(module) == CUDA_SUCCESS &&
                             g_state.library_unload_count == unloads,
                         "unknown published-error library is retained, not automatically unloaded");
    }
    g_state.should_fail_library_load = false;
    g_state.should_publish_failed_library = false;
    const auto concurrent_loads = g_state.library_load_count;
    passed &=
        expect(runtime.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr,
                                          nullptr) == CUDA_SUCCESS &&
                   runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS,
               "prepare a fresh source generation after quarantined handle reuse");
    std::vector<std::thread> callers;
    for (unsigned int index = 0; index < 8; ++index) {
        callers.emplace_back([&] { runtime.prepare_module_source(function); });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    passed &= expect(g_state.library_load_count == concurrent_loads + 1 &&
                         runtime.module_lookup_is_prepared(module, "write_indices"),
                     "eight cold callers share one terminal eager module initialization");
    g_state.should_fail_unload = true;
    const auto unloads = g_state.library_unload_count;
    passed &= expect(runtime.unload_module(module) == CUDA_ERROR_INVALID_HANDLE &&
                         runtime.module_lookup_is_prepared(module, "write_indices") &&
                         g_state.library_unload_count == unloads,
                     "failed original module unload preserves its private owner and associations");
    g_state.should_fail_unload = false;
    g_state.should_fail_library_unload = true;
    passed &=
        expect(runtime.unload_module(module) == CUDA_SUCCESS &&
                   g_state.library_unload_count == unloads + 1 &&
                   !runtime.module_lookup_is_prepared(module, "write_indices"),
               "original unload success retires the module despite uncertain private cleanup");
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == unloads + 1,
                     "uncertain private unload cannot be retried by later context cleanup");
    g_state.should_fail_library_unload = false;
    CUjit_option option = CU_JIT_TARGET_FROM_CUCONTEXT;
    void* option_value{};
    const auto option_loads = g_state.load_count;
    passed &= expect(
        runtime.load_source_module(&module, kPersistentTestPtx, true, 1, &option, &option_value) ==
                CUDA_SUCCESS &&
            g_state.load_count == option_loads + 1 && !g_state.was_transformed &&
            g_state.last_options == &option && g_state.last_option_values == &option_value &&
            runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS &&
            runtime.module_source_is_prepared(function) &&
            !runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(), false),
        "nonzero original JIT options are unchanged and do not create guessed private metadata");
    passed &= expect(runtime.unload_module(module) == CUDA_SUCCESS,
                     "clean the original non-transformed option module");
    for (const bool is_missing : {false, true}) {
        auto table = functions();
        if (is_missing) {
            table.function_load = nullptr;
        }
        DriverDispatch limited_driver(table);
        PersistentKernelRuntime limited(limited_driver, 3, false, true);
        g_state.should_fail_function_load = true;
        const auto native_library_loads = g_state.library_load_count;
        passed &= expect(
            limited.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr, nullptr) ==
                    CUDA_SUCCESS &&
                limited.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS,
            "missing or failed loading proof preserves original source issue");
        limited.prepare_module_source(function);
        const auto prepared_calls = g_state.native_call_count;
        limited.prepare_module_source(function);
        passed &=
            expect(limited.module_source_is_prepared(function) &&
                       !limited.module_lookup_is_prepared(module, "write_indices") &&
                       !limited.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0,
                                                   stream_handle(), false) &&
                       g_state.library_load_count == native_library_loads &&
                       g_state.native_call_count == prepared_calls,
                   "missing or failed function loading is terminal, not guessed GPU readiness");
        passed &= expect(limited.unload_module(module) == CUDA_SUCCESS,
                         "release original limited-loading module without a private CUDA owner");
        g_state.should_fail_function_load = false;
    }
    return passed &&
           expect(!g_state.guard_failed, "all deferred native operations use leaf guards");
}

bool test_source_module_retirement() {
    bool passed = true;
    for (const bool has_ambiguous_acquisition : {false, true}) {
        g_state = {};
        g_state.should_fail_library_load = has_ambiguous_acquisition;
        g_state.should_publish_failed_library = has_ambiguous_acquisition;
        DriverDispatch driver(functions());
        PersistentKernelRuntime runtime(driver, 3, false, true);
        CUmodule module{};
        CUfunction function{};
        passed &= expect(
            runtime.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr, nullptr) ==
                    CUDA_SUCCESS &&
                runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS,
            "retirement fixture owns original source module metadata");
        runtime.prepare_module_source(function);
        const auto kernel = [&] {
            return runtime.cooperative_kernel(function, {7, 1, 1}, {32, 1, 1}, 0, stream_handle(),
                                              false);
        };
        passed &= expect(
            kernel().has_value() != has_ambiguous_acquisition && g_state.library_load_count == 1,
            "retirement covers known and uncertain private acquisition");
        g_state.should_fail_unload = true;
        passed &= expect(runtime.unload_source_module(module) == CUDA_ERROR_INVALID_HANDLE &&
                             kernel().has_value() != has_ambiguous_acquisition &&
                             g_state.module_unload_count == 1 && g_state.library_unload_count == 0,
                         "failed original source unload preserves association and private owner");
        g_state.should_fail_unload = false;
        passed &=
            expect(runtime.unload_source_module(module) == CUDA_SUCCESS && !kernel() &&
                       !runtime.module_lookup_is_prepared(module, "write_indices") &&
                       g_state.module_unload_count == 2 && g_state.library_unload_count == 0,
                   "successful source unload retires exact CPU metadata without private cleanup");
        g_state.should_fail_library_load = false;
        g_state.should_publish_failed_library = false;
        passed &= expect(
            runtime.load_source_module(&module, kPersistentTestPtx, false, 0, nullptr, nullptr) ==
                    CUDA_SUCCESS &&
                runtime.get_source_function(&function, module, "write_indices") == CUDA_SUCCESS &&
                !runtime.module_source_is_prepared(function) && !kernel(),
            "recycled native handles cannot recover a retired private function");
        runtime.prepare_module_source(function);
        passed &= expect(kernel() && g_state.library_load_count == 2 &&
                             g_state.library_unload_count == (has_ambiguous_acquisition ? 0U : 1U),
                         "later normal preparation cleans only a known retired owner once");
        passed &= expect(runtime.unload_source_module(module) == CUDA_SUCCESS && !kernel() &&
                             g_state.module_unload_count == 3 &&
                             g_state.library_unload_count == (has_ambiguous_acquisition ? 0U : 1U),
                         "another source generation still cannot clean private CUDA");
        g_state.should_fail_library_unload = true;
        runtime.forget_context(context_handle());
        const auto attempted_unloads = g_state.library_unload_count;
        passed &= expect(attempted_unloads == (has_ambiguous_acquisition ? 1U : 2U),
                         "checked context retirement attempts only known private cleanup");
        runtime.forget_context(context_handle());
        passed &=
            expect(g_state.library_unload_count == attempted_unloads &&
                       g_state.module_unload_count == 3 && !g_state.guard_failed,
                   "ambiguous cleanup is retained and never replayed or turned into SDK unload");
    }
    return passed;
}

bool test_cooperative_preparation() {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!expect(runtime.load_module(&module, kCooperativePtx) == CUDA_SUCCESS &&
                    runtime.get_function(&function, module, "cooperative") == CUDA_SUCCESS,
                "cooperative preparation must retain original function handle")) {
        return false;
    }
    const auto prepare_kernel = [&](std::array<unsigned int, 3> grid = {7, 5, 3},
                                    std::array<unsigned int, 3> block = {8, 4, 2},
                                    unsigned int shared_bytes = 0) {
        return runtime.cooperative_kernel(function, grid, block, shared_bytes, stream_handle(),
                                          true);
    };
    const auto prepared = prepare_kernel();
    bool passed =
        expect(prepared && prepared->function == variant_handle() &&
                   prepared->context == context_handle() && prepared->shape.logical_blocks == 105 &&
                   prepared->shape.block_threads == 64 && prepared->worker_blocks == 3 &&
                   prepared->parameter_layout.size() == 3 &&
                   prepared->parameter_layout[0].size_bytes == 8 &&
                   prepared->parameter_layout[1].size_bytes == 8 &&
                   prepared->parameter_layout[2].size_bytes == 4 && g_state.used_capture_ptds &&
                   g_state.launch_count == 0,
               "prepare must own exact original ABI and validate shape without launch");
    const auto small = prepare_kernel({1, 1, 1}, {1, 1, 1});
    passed &= expect(small && small->worker_blocks == 1 && small->shape.logical_blocks == 1,
                     "small priority launches need a bounded owned cooperative ABI too");
    passed &= expect(!prepare_kernel({0, 1, 1}) && !prepare_kernel({7, 65536, 3}) &&
                         !prepare_kernel({7, 5, 3}, {8, 4, 65}) &&
                         !prepare_kernel({7, 5, 3}, {8, 4, 2}, 4),
                     "invalid dimensions and dynamic shared memory must not prepare");
    g_state.capture_status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    passed &= expect(!prepare_kernel(), "captured kernel must never prepare cooperative ABI");
    g_state.capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    g_state.should_fail_capture_query = true;
    passed &= expect(!prepare_kernel(), "failed capture query must reject preparation");
    g_state.should_fail_capture_query = false;
    g_state.should_fail_context = true;
    passed &= expect(!prepare_kernel(), "failed context query must reject preparation");
    g_state.should_fail_context = false;
    g_state.resident_blocks = 0;
    passed &= expect(!prepare_kernel(), "nonresident cooperative function must not prepare");
    g_state.resident_blocks = 1;
    unsigned int value = 0;
    void* arguments[]{&value, &value, &value};
    passed &= expect(!runtime.try_launch(function, 7, 5, 3, 8, 4, 2, 0, stream_handle(), arguments,
                                         nullptr, false) &&
                         g_state.launch_count == 0,
                     "legacy substitution cannot issue an incomplete cooperative ABI");
    passed &= expect(runtime.launch(function, 7, 5, 3, 8, 4, 2, 0, stream_handle(), arguments,
                                    nullptr, false) == CUDA_SUCCESS &&
                         g_state.last_function == original_handle(),
                     "legacy cooperative-mode launch must preserve original native entry");
    passed &= expect(runtime.unload_module(module) == CUDA_SUCCESS && !prepare_kernel() &&
                         prepared->parameter_layout.size() == 3,
                     "unload removes mappings, not previously owned host metadata");
    return passed && expect(!g_state.guard_failed, "cooperative metadata calls need native guards");
}

bool test_load_fallback() {
    bool passed = true;
    for (const bool fail_original : {false, true}) {
        g_state = {};
        g_state.should_fail_jit = true;
        g_state.should_fail_original = fail_original;
        DriverDispatch driver(functions());
        PersistentKernelRuntime runtime(driver, 3, false);
        CUmodule module = nullptr;
        passed &= expect(runtime.load_module(&module, kPersistentTestPtx) ==
                                 (fail_original ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS) &&
                             g_state.load_count == 2 && g_state.last_image == kPersistentTestPtx,
                         "JIT failure must retry original image and return its result");
    }
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false);
    CUmodule module = nullptr;
    CUjit_option option = CU_JIT_OPTIMIZATION_LEVEL;
    void* option_value = reinterpret_cast<void*>(2);
    passed &=
        expect(runtime.load_module(&module, kPersistentTestPtx, true, 1, &option, &option_value) ==
                       CUDA_SUCCESS &&
                   !g_state.was_transformed && g_state.last_image == kPersistentTestPtx &&
                   g_state.last_options == &option && g_state.last_option_values == &option_value,
               "caller JIT options and input must be forwarded unchanged");
    const char binary[] = {'\x7f', 'E', 'L', 'F', '\0'};
    passed &= expect(runtime.load_module(&module, binary) == CUDA_SUCCESS &&
                         g_state.last_image == binary && !g_state.was_transformed,
                     "binary module must stay native");
    passed &= expect(runtime.load_module(&module, nullptr) == CUDA_ERROR_INVALID_VALUE,
                     "native invalid-image error must remain unchanged");
    g_state.should_fail_context = true;
    passed &= expect(runtime.load_module(&module, kPersistentTestPtx) == CUDA_SUCCESS &&
                         !g_state.was_transformed,
                     "missing current context must disable transformation");
    g_state.should_fail_context = false;
    g_state.should_fail_lookup = true;
    passed &= prepare(runtime);
    unsigned int value = 0;
    void* arguments[] = {&value, &value};
    passed &= expect(runtime.launch(original_handle(), 17, 1, 1, 64, 1, 1, 0, stream_handle(),
                                    arguments, nullptr, false) == CUDA_SUCCESS &&
                         g_state.last_function == original_handle(),
                     "missing variant function must keep original launch");
    auto missing_table = functions();
    missing_table.occupancy = nullptr;
    DriverDispatch missing_driver(missing_table);
    PersistentKernelRuntime missing_runtime(missing_driver, 3, false);
    g_state.was_transformed = false;
    passed &= expect(missing_runtime.load_module(&module, kPersistentTestPtx) == CUDA_SUCCESS &&
                         !g_state.was_transformed,
                     "missing optional Driver support must keep original module");
    g_state.should_fail_lookup = false;
    missing_table = functions();
    missing_table.stream_capture_status = nullptr;
    missing_table.stream_capture_status_ptsz = nullptr;
    DriverDispatch missing_capture_driver(missing_table);
    PersistentKernelRuntime missing_capture_runtime(missing_capture_driver, 3, false);
    passed &= prepare(missing_capture_runtime);
    for (const bool ptds : {false, true}) {
        passed &= expect(missing_capture_runtime.launch(original_handle(), 17, 1, 1, 64, 1, 1, 0,
                                                        stream_handle(), arguments, nullptr,
                                                        ptds) == CUDA_SUCCESS &&
                             g_state.last_function == original_handle(),
                         "missing capture query must keep native launch for either stream ABI");
    }
    return passed && expect(!g_state.guard_failed, "load fallback calls need Driver guards");
}

bool test_runtime_source_and_cleanup(const std::string& tool) {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false);
    const auto prepare_kernel = [&](const std::string& path) {
        return runtime.runtime_kernel(context_handle(), original_handle(), "write_indices", path,
                                      tool);
    };
    const auto kernel = prepare_kernel("/good");
    bool passed = expect(kernel.has_value() && kernel->parameter_count == 2 &&
                             kernel->kernel == reinterpret_cast<CUkernel>(variant_handle()) &&
                             g_state.library_load_count == 1,
                         "ordinary Runtime symbol must resolve an embedded PTX variant");
    passed &= expect(prepare_kernel("/good").has_value() && g_state.library_load_count == 1,
                     "warm Runtime lookup must not load or extract again");
    passed &= expect(!prepare_kernel("/duplicate").has_value() && g_state.library_load_count == 1,
                     "ambiguous architecture definitions must stay native without a JIT load");
    passed &= expect(
        !runtime.runtime_kernel(context_handle(), original_handle(), "missing", "/good", tool)
             .has_value(),
        "absent Runtime entry must stay native");
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1 && prepare_kernel("/good").has_value() &&
                         g_state.library_load_count == 2,
                     "context teardown must release private libraries and reset function caches");
    runtime.forget_context(context_handle());
    g_state.should_fail_library_load = true;
    passed &= expect(!prepare_kernel("/good").has_value(), "Runtime JIT failure must stay native");
    runtime.forget_context(context_handle());
    g_state.should_fail_library_load = false;
    g_state.should_fail_library_lookup = true;
    const auto unloads_before = g_state.library_unload_count;
    passed &= expect(
        !prepare_kernel("/good").has_value() && g_state.library_unload_count == unloads_before + 1,
        "failed Runtime lookup must clean its partially initialized library");
    runtime.forget_context(context_handle());
    g_state.should_fail_library_lookup = false;
    PersistentKernelRuntime cooperative(driver, 3, false, true);
    const auto cooperative_variant = cooperative.runtime_kernel(context_handle(), original_handle(),
                                                                "write_indices", "/good", tool);
    passed &= expect(cooperative_variant &&
                         cooperative_variant->execution_kind ==
                             glimmer::interceptor::PtxExecutionKind::kPreemptive &&
                         cooperative_variant->parameter_layout.size() == 2 &&
                         cooperative_variant->parameter_layout[0].size_bytes == 8 &&
                         cooperative_variant->parameter_layout[1].size_bytes == 4,
                     "Runtime cooperative recovery must own exact original parameter layout");
    cooperative.forget_context(context_handle());
    return passed &&
           expect(!g_state.guard_failed, "private Runtime library calls need Driver guards");
}

bool test_checked_runtime_retirement(const std::string& tool) {
    bool passed = true;
    for (const bool cooperative : {false, true}) {
        for (const bool fail_unload : {false, true}) {
            g_state = {};
            DriverDispatch driver(functions());
            PersistentKernelRuntime runtime(driver, 3, false, cooperative);
            passed &= expect(
                runtime
                    .runtime_kernel(context_handle(), cooperative ? nullptr : original_handle(),
                                    "write_indices", "/good", tool)
                    .has_value(),
                "checked Runtime retirement owns private function or source code");
            g_state.should_fail_library_unload = fail_unload;
            passed &= expect(runtime.forget_context_checked(context_handle()) ==
                                     (fail_unload ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS) &&
                                 g_state.library_unload_count == 1,
                             "checked Runtime retirement reports the original cleanup error");
            const auto calls = g_state.native_call_count;
            passed &=
                expect(runtime.forget_context_checked(nullptr) == CUDA_ERROR_INVALID_CONTEXT &&
                           runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                           g_state.native_call_count == calls && !g_state.guard_failed,
                       "retired successful or uncertain Runtime library owner is not retried");
        }
    }
    return passed;
}

bool test_runtime_original_preparation(const std::string& tool) {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    const auto second_original = reinterpret_cast<CUfunction>(&g_second_function_token);
    const auto first = runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good",
                                              tool, original_handle());
    bool passed =
        expect(first && first->original_function == original_handle() &&
                   first->original_module == module_handle() &&
                   g_state.function_module_count == 2 && g_state.module_lookup_count == 2 &&
                   g_state.function_load_count == 4 && g_state.library_load_count == 1,
               "cold Runtime preparation must prove both original/private functions");
    const auto warm_calls = g_state.native_call_count;
    const auto second = runtime.runtime_kernel(context_handle(), nullptr, "native_barrier", "/good",
                                               tool, second_original);
    passed &= expect(second && second->original_function == second_original &&
                         g_state.native_call_count == warm_calls + 1 &&
                         g_state.function_module_count == 3 && g_state.function_load_count == 4 &&
                         g_state.module_lookup_count == 2 && g_state.library_lookup_count == 2,
                     "warm original validation may query identity but must not load private code");
    passed &= expect(!runtime.runtime_kernel(context_handle(), nullptr, "native_barrier", "/good",
                                             tool, original_handle()) &&
                         !runtime.runtime_kernel(context_handle(), nullptr, "missing", "/good",
                                                 tool, original_handle()) &&
                         g_state.native_call_count == warm_calls + 1,
                     "mismatched original or absent entry cannot select a private function");
    g_state.function_module = reinterpret_cast<CUmodule>(&g_second_function_token);
    passed &= expect(!runtime.runtime_kernel(context_handle(), nullptr, "native_barrier", "/good",
                                             tool, second_original) &&
                         g_state.function_load_count == 4 && g_state.library_load_count == 1,
                     "changed actual module identity rejects without another JIT/load");
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1 && g_state.module_unload_count == 0 &&
                         !g_state.guard_failed,
                     "context retirement unloads private code, never the borrowed SDK module");

    for (unsigned int fault = 0; fault < 5; ++fault) {
        g_state = {};
        auto table = functions();
        if (fault == 0) {
            table.function_get_module = nullptr;
        }
        g_state.should_fail_function_module = fault == 1;
        g_state.should_return_null_function_module = fault == 2;
        g_state.should_fail_original_lookup = fault == 3;
        g_state.should_fail_function_load = fault == 4;
        DriverDispatch failed_driver(table);
        PersistentKernelRuntime failed_runtime(failed_driver, 3, false, true);
        passed &= expect(
            !failed_runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good",
                                           tool, original_handle()) &&
                failed_runtime.runtime_cooperative_source_is_prepared(context_handle(), "/good") &&
                g_state.library_load_count == 0 && g_state.launch_count == 0,
            "missing/failed original proof must be terminal without private acquisition/issue");
        const auto calls = g_state.native_call_count;
        g_state.should_fail_function_module = false;
        g_state.should_return_null_function_module = false;
        g_state.should_fail_original_lookup = false;
        g_state.should_fail_function_load = false;
        passed &= expect(!failed_runtime.runtime_kernel(context_handle(), nullptr, "native_barrier",
                                                        "/good", tool, second_original) &&
                             g_state.native_call_count == calls,
                         "restored original support must not retry terminal initialization");
        failed_runtime.forget_context(context_handle());
        passed &= expect(g_state.library_unload_count == 0 && g_state.module_unload_count == 0 &&
                             !g_state.guard_failed,
                         "failed original proof cannot manufacture an unload owner");
    }

    g_state = {};
    g_state.rejected_function_load = second_original;
    DriverDispatch partial_driver(functions());
    PersistentKernelRuntime partial(partial_driver, 3, false, true);
    passed &= expect(partial.runtime_kernel(context_handle(), nullptr, "write_indices", "/good",
                                            tool, original_handle()) &&
                         !partial.runtime_kernel(context_handle(), nullptr, "native_barrier",
                                                 "/good", tool, second_original) &&
                         g_state.library_lookup_count == 1 && g_state.function_load_count == 3,
                     "only original-loaded siblings may acquire and publish private counterparts");
    partial.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1 && g_state.module_unload_count == 0 &&
                         !g_state.guard_failed,
                     "partially ready original/private source retains one private unload owner");

    g_state = {};
    DriverDispatch concurrent_driver(functions());
    PersistentKernelRuntime concurrent(concurrent_driver, 3, false, true);
    std::atomic<bool> is_ready{true};
    std::vector<std::thread> callers;
    for (unsigned int index = 0; index < 8; ++index) {
        callers.emplace_back([&, index] {
            const auto entry = concurrent.runtime_kernel(
                context_handle(), nullptr, index % 2 == 0 ? "write_indices" : "native_barrier",
                "/good", tool, index % 2 == 0 ? original_handle() : second_original);
            if (!entry) {
                is_ready.store(false);
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    passed &= expect(is_ready.load() && g_state.library_load_count == 1 &&
                         g_state.module_lookup_count == 2 && g_state.function_load_count == 4 &&
                         g_state.function_module_count == 9,
                     "concurrent callers share one cold proof and validate their exact identities");
    concurrent.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1 && g_state.module_unload_count == 0,
                     "concurrent original proof never takes SDK module ownership");
    return passed && !g_state.guard_failed;
}

bool test_cooperative_source_preparation(const std::string& tool) {
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    bool passed = expect(!runtime.runtime_cooperative_source_is_prepared(context_handle(), "/good"),
                         "cold cooperative source must require a native preparation boundary");
    const auto first =
        runtime.runtime_kernel(context_handle(), original_handle(), "write_indices", "/good", tool);
    passed &= expect(first && first->function == variant_handle() &&
                         g_state.library_load_count == 1 && g_state.library_lookup_count == 2 &&
                         g_state.kernel_function_count == 2 && g_state.function_load_count == 2 &&
                         runtime.runtime_cooperative_source_is_prepared(context_handle(), "/good"),
                     "one source load must eagerly finalize both qualified context functions");
    const auto native_calls = g_state.native_call_count;
    const auto second = runtime.runtime_kernel(context_handle(), &g_second_variant_token,
                                               "native_barrier", "/good", tool);
    passed &=
        expect(second && second->parameter_count == 1 &&
                   second->function == reinterpret_cast<CUfunction>(&g_second_variant_token) &&
                   g_state.native_call_count == native_calls && g_state.library_load_count == 1,
               "a new host function from a warm source must make no native CUDA/JIT call");
    passed &= expect(!runtime.runtime_kernel(context_handle(), nullptr, "missing", "/good", tool) &&
                         g_state.native_call_count == native_calls,
                     "an absent warm source entry must remain a metadata-only miss");
    passed &= expect(
        !runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/duplicate", tool) &&
            g_state.library_load_count == 1,
        "duplicate architecture entries must never cause cooperative library JIT");
    const auto foreign = reinterpret_cast<CUcontext>(&g_second_variant_token);
    passed &= expect(!runtime.runtime_kernel(foreign, nullptr, "write_indices", "/good", tool) &&
                         g_state.library_load_count == 1,
                     "cooperative preparation must reject a foreign current context before JIT");
    runtime.forget_context(foreign);
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1 &&
                         !runtime.runtime_cooperative_source_is_prepared(context_handle(), "/good"),
                     "checked context forgetting must unload the shared library once");

    g_state.should_fail_library_load = true;
    g_state.should_publish_failed_library = true;
    const auto loads_before_failure = g_state.library_load_count;
    passed &=
        expect(!runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good", tool),
               "failed native load with a published handle must not publish a kernel");
    const auto calls_after_failure = g_state.native_call_count;
    passed &= expect(
        !runtime.runtime_kernel(context_handle(), nullptr, "native_barrier", "/good", tool) &&
            g_state.native_call_count == calls_after_failure &&
            g_state.library_load_count == loads_before_failure + 1,
        "failed source initialization must be terminal rather than replayed");
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1,
                     "a failed load's uncertain published handle must not be freed speculatively");
    g_state.should_fail_library_load = false;
    g_state.should_publish_failed_library = false;
    g_state.should_fail_library_lookup = true;
    g_state.should_fail_library_unload = true;
    passed &=
        expect(!runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good", tool) &&
                   g_state.library_unload_count == 2,
               "failed eager lookup must attempt known library cleanup exactly once");
    runtime.forget_context(context_handle());
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 2,
                     "ambiguous native unload must never be replayed at context forgetting");
    g_state.should_fail_library_lookup = false;
    g_state.should_fail_library_unload = false;
    passed &=
        expect(runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good", tool)
                       .has_value() &&
                   g_state.library_load_count == loads_before_failure + 3,
               "a reused context gets a new source generation, not quarantined handles");
    runtime.forget_context(context_handle());
    passed &= expect(!runtime.runtime_cooperative_source_is_prepared(nullptr, "/good"),
                     "retired uncertain owners must not become a null-context source");
    const auto concurrent_loads_before = g_state.library_load_count;
    const auto concurrent_unloads_before = g_state.library_unload_count;
    std::atomic<bool> concurrent_passed{true};
    std::vector<std::thread> callers;
    for (unsigned int index = 0; index < 8; ++index) {
        callers.emplace_back([&, index] {
            const auto entry = runtime.runtime_kernel(
                context_handle(), nullptr, index % 2 == 0 ? "write_indices" : "native_barrier",
                "/good", tool);
            if (!entry || entry->function == nullptr) {
                concurrent_passed.store(false);
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    passed &= expect(
        concurrent_passed.load() && g_state.library_load_count == concurrent_loads_before + 1,
        "concurrent cold callers must share one eager source initialization");
    runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == concurrent_unloads_before + 1,
                     "concurrent source owners must still release their one library exactly once");
    passed &=
        expect(!g_state.guard_failed, "all earlier source preparation calls need native guards");

    for (const bool has_loader : {false, true}) {
        g_state = {};
        auto table = functions();
        if (!has_loader) {
            table.function_load = nullptr;
        }
        g_state.should_fail_function_load = true;
        DriverDispatch loading_driver(table);
        PersistentKernelRuntime loading_runtime(loading_driver, 3, false, true);
        passed &= expect(
            !loading_runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good",
                                            tool) &&
                loading_runtime.runtime_cooperative_source_is_prepared(context_handle(), "/good") &&
                g_state.library_load_count == (has_loader ? 1U : 0U) &&
                g_state.function_load_count == (has_loader ? 2U : 0U) &&
                g_state.library_unload_count == (has_loader ? 1U : 0U) && g_state.launch_count == 0,
            "missing/failed code loading must be terminal native-only without kernel issue");
        const auto calls = g_state.native_call_count;
        g_state.should_fail_function_load = false;
        passed &= expect(!loading_runtime.runtime_kernel(context_handle(), nullptr,
                                                         "native_barrier", "/good", tool) &&
                             g_state.native_call_count == calls,
                         "restored code loading must not retry a terminal source initialization");
        loading_runtime.forget_context(context_handle());
        passed &=
            expect(g_state.library_unload_count == (has_loader ? 1U : 0U) && !g_state.guard_failed,
                   "terminal loading failure must not replay known library cleanup");
    }
    g_state = {};
    g_state.rejected_function_load = reinterpret_cast<CUfunction>(&g_second_variant_token);
    DriverDispatch partial_driver(functions());
    PersistentKernelRuntime partial_runtime(partial_driver, 3, false, true);
    passed &= expect(
        partial_runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good", tool) &&
            !partial_runtime.runtime_kernel(context_handle(), nullptr, "native_barrier", "/good",
                                            tool) &&
            g_state.function_load_count == 2 && g_state.library_unload_count == 0,
        "only individually loaded variants may publish from a partially usable source");
    const auto partial_calls = g_state.native_call_count;
    passed &= expect(
        partial_runtime.runtime_kernel(context_handle(), nullptr, "write_indices", "/good", tool) &&
            g_state.native_call_count == partial_calls,
        "a proven warm entry must not retry another entry's failed loading");
    partial_runtime.forget_context(context_handle());
    passed &= expect(g_state.library_unload_count == 1,
                     "a partially usable library must retain one checked unload owner");
    return passed &&
           expect(!g_state.guard_failed, "all source preparation calls need native guards");
}

template <bool IsGemm = false>
bool test_binary_source(const char* path, bool is_fatbinary = false, bool is_library = false) {
    constexpr auto entry = IsGemm ? glimmer::interceptor::kQualifiedGemmEntry
                                  : glimmer::interceptor::kQualifiedAddEntry;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::size_t image_bytes =
        IsGemm ? (is_fatbinary ? 364384 : 3055056) : (is_fatbinary ? 5072120 : 5336008);
    if (!expect(input.good() && input.tellg() == static_cast<std::streamoff>(image_bytes),
                "qualified original fixture"))
        return false;
    std::vector<char> original(image_bytes);
    input.seekg(0);
    input.read(original.data(), static_cast<std::streamsize>(original.size()));
    if (!expect(input.good(), "qualified original read")) return false;
    bool passed = true;
    for (int mode = 0; mode != (IsGemm ? 14 : 9); ++mode) {
        g_state = {};
        auto api = functions();
        if (mode == 13) api.function_load = nullptr;
        DriverDispatch driver(api);
        PersistentKernelRuntime runtime(driver, 3, false, true);
        auto image = original;
        CUmodule module{};
        CUfunction function{};
        const auto get_variant = [&] {
            if constexpr (IsGemm)
                return runtime.mapped_gemm_kernel(function);
            else
                return runtime.mapped_binary_kernel(function);
        };
        const auto library = reinterpret_cast<CUlibrary>(&g_stream_token);
        CUresult loaded = CUDA_SUCCESS;
        if (is_library) {
            runtime.observe_library_source(library, image.data());
            module = module_handle();
            runtime.observe_library_module(library, module, context_handle());
        } else {
            loaded = runtime.load_module(&module, image.data());
        }
        passed &= expect(
            loaded == CUDA_SUCCESS && module == module_handle() &&
                g_state.load_count == (is_library ? 0U : 1U) && g_state.library_load_count == 0 &&
                runtime.get_function(&function, module, entry) == CUDA_SUCCESS &&
                function == original_handle() && !runtime.module_source_is_prepared(function),
            "binary source keeps original handles and defers all private CUDA");
        std::fill(image.begin(), image.end(), 0);
        if (mode == 1 || mode == 2) g_state.should_fail_library_load = true;
        if (mode == 2) g_state.should_publish_failed_library = true;
        if (mode == 3) g_state.should_fail_library_lookup = true;
        if (mode == 4) g_state.compute_major = 8;
        if (mode == 5) g_state.should_fail_context = true;
        if (mode == 6) g_state.should_fail_function_load = true;
        if (mode == 9)
            g_state.current_context = reinterpret_cast<CUcontext>(&g_second_variant_token);
        if (mode == 10) g_state.should_fail_original_lookup = true;
        if (mode == 11) g_state.rejected_function_load = variant_handle();
        if (mode == 12) g_state.should_fail_device_attribute = true;
        runtime.prepare_module_source(function);
        const auto calls = g_state.native_call_count;
        runtime.prepare_module_source(function);
        auto variant = get_variant();
        alignas(8) std::array<std::byte, 176> argument_storage{};
        std::array<void*, 3> arguments{argument_storage.data(), argument_storage.data(),
                                       argument_storage.data()};
        const bool qualified = mode == 0 || mode == 7 || mode == 8;
        passed &=
            expect(runtime.module_source_is_prepared(function) &&
                       g_state.native_call_count == calls && variant.has_value() == qualified &&
                       !runtime.cooperative_kernel(function, {33, 1, 1}, {128, 1, 1}, 0,
                                                   stream_handle(), false) &&
                       !runtime.try_launch(function, 33, 1, 1, 128, 1, 1, 0, stream_handle(),
                                           arguments.data(), nullptr, false) &&
                       g_state.launch_count == 0 && g_state.native_call_count == calls,
                   "terminal binary preparation is never selected as a PTX launch");
        if (variant) {
            passed &= expect(variant->context == context_handle() &&
                                 variant->function == variant_handle() && variant->resources &&
                                 g_state.library_load_count == 1 &&
                                 runtime.module_lookup_is_prepared(module, entry),
                             "owned private variant comes from the original module association");
            if constexpr (!IsGemm)
                passed &=
                    expect(variant->mapping == glimmer::interceptor::QualifiedAddMapping::kRange,
                           "add retains its range mapping");
            else
                passed &=
                    expect(variant->mapping == glimmer::interceptor::QualifiedGemmMapping::kRange,
                           "GEMM retains its separately typed range mapping");
        }
        if constexpr (IsGemm)
            passed &=
                expect(!runtime.mapped_binary_kernel(function), "GEMM cannot select the add ABI");
        else
            passed &=
                expect(!runtime.mapped_gemm_kernel(function), "add cannot select the GEMM ABI");
        g_state.should_fail_context = false;
        g_state.should_fail_original_lookup = false;
        g_state.current_context = context_handle();
        if (is_library) {
            const auto before = g_state.native_call_count;
            runtime.observe_library_module(library, module, context_handle());
            passed &= expect(
                get_variant().has_value() == qualified && g_state.native_call_count == before,
                "repeated library association is metadata-only and preserves warm code");
            runtime.retire_library_source(library);
            passed &= expect(!get_variant() && g_state.native_call_count == before &&
                                 g_state.module_unload_count == 0,
                             "library retirement owns neither the original library nor its module");
            variant.reset();
            passed &= expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                                 g_state.library_unload_count ==
                                     (qualified || mode == 3 || mode == 11 ? 1U : 0U),
                             "retired library source releases only its known private owner");
            runtime.observe_library_module(library, module, context_handle());
            g_state.should_fail_library_load = false;
            g_state.should_fail_library_lookup = false;
            g_state.should_fail_function_load = false;
            passed &=
                expect(runtime.get_source_function(&function, module, entry) == CUDA_SUCCESS &&
                           !get_variant(),
                       "unloaded library cannot resurrect a recycled module and function");
            passed &= expect(!g_state.guard_failed, "library preparation uses native guards");
            continue;
        }
        g_state.should_fail_unload = true;
        passed &=
            expect(runtime.unload_module(module) == CUDA_ERROR_INVALID_HANDLE &&
                       get_variant().has_value() == qualified && g_state.library_unload_count == 0,
                   "failed original unload preserves the exact source binding");
        g_state.should_fail_unload = false;
        if (mode == 7) {
            passed &= expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS,
                             "context retirement with retained variant");
        } else if (mode == 8) {
            passed &= expect(runtime.unload_source_module(module) == CUDA_SUCCESS &&
                                 g_state.library_unload_count == 0,
                             "source-only retirement has no private CUDA");
        } else {
            passed &=
                expect(runtime.unload_module(module) == CUDA_SUCCESS, "original binary unload");
        }
        passed &= expect(!get_variant(), "retired source is unreachable");
        if (qualified)
            passed &= expect(g_state.library_unload_count == 0,
                             "outstanding binary lease retains library");
        variant.reset();
        g_state.should_fail_library_unload = mode == 7;
        const auto cleanup = runtime.forget_context_checked(context_handle());
        passed &= expect(cleanup == (mode == 7 ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS),
                         "checked binary library cleanup");
        const auto expected_unloads = qualified || mode == 3 || mode == 11 ? 1U : 0U;
        passed &= expect(g_state.library_unload_count == expected_unloads,
                         "exactly one known-owner cleanup");
        runtime.forget_context(context_handle());
        passed &= expect(g_state.library_unload_count == expected_unloads,
                         "uncertain cleanup is never retried");
        g_state.should_fail_library_load = false;
        g_state.should_fail_library_lookup = false;
        g_state.should_fail_function_load = false;
        // The native fake deliberately recycles the same module/function handles.
        passed &= expect(runtime.load_module(&module, "opaque") == CUDA_SUCCESS &&
                             runtime.get_function(&function, module, entry) == CUDA_SUCCESS &&
                             !get_variant(),
                         "same name and recycled handles cannot resurrect binary ownership");
        passed &= expect(!g_state.guard_failed, "binary preparation uses native leaf guards");
    }
    std::cout << "binary_owner gemm=" << IsGemm << " fatbinary=" << is_fatbinary
              << " library=" << is_library << " modes=" << (IsGemm ? 14 : 9) << " passed=" << passed
              << " nonnull_isolation=1\n";
    return passed;
}

template <bool IsGemm = false>
bool test_library_identity(const char* path) {
    constexpr auto entry = IsGemm ? glimmer::interceptor::kQualifiedGemmEntry
                                  : glimmer::interceptor::kQualifiedAddEntry;
    std::ifstream input(path, std::ios::binary);
    std::vector<char> image(IsGemm ? 364384 : 5072120);
    input.read(image.data(), static_cast<std::streamsize>(image.size()));
    if (!expect(input.good(), "library identity source")) return false;
    g_state = {};
    DriverDispatch driver(functions());
    PersistentKernelRuntime runtime(driver, 3, false, true);
    const auto library = reinterpret_cast<CUlibrary>(&g_stream_token);
    const auto unknown = reinterpret_cast<CUlibrary>(&g_function_token);
    const auto second_module = reinterpret_cast<CUmodule>(&g_second_function_token);
    const auto second_context = reinterpret_cast<CUcontext>(&g_second_variant_token);
    CUfunction function{};
    const auto get_variant = [&] {
        if constexpr (IsGemm)
            return runtime.mapped_gemm_kernel(function);
        else
            return runtime.mapped_binary_kernel(function);
    };
    runtime.observe_library_source(library, image.data());
    runtime.observe_library_module(library, module_handle(), context_handle());
    runtime.observe_library_module(library, second_module, second_context);
    bool passed = expect(g_state.native_call_count == 0, "source associations issue no CUDA");
    passed &= expect(runtime.get_source_function(&function, module_handle(), entry) == CUDA_SUCCESS,
                     "first context original lookup");
    runtime.prepare_module_source(function);
    auto first = get_variant();
    passed &= expect(first && first->context == context_handle(), "first context private owner");
    passed &= expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                         !get_variant() && g_state.library_unload_count == 0,
                     "context retirement retains a leased private owner");
    first.reset();
    passed &= expect(runtime.forget_context_checked(context_handle()) == CUDA_SUCCESS &&
                         g_state.library_unload_count == 1,
                     "last lease permits drained cleanup");
    g_state.current_context = second_context;
    passed &= expect(runtime.get_source_function(&function, second_module, entry) == CUDA_SUCCESS &&
                         !runtime.module_source_is_prepared(function),
                     "retiring one context preserves another library module");
    runtime.prepare_module_source(function);
    auto second = get_variant();
    passed &= expect(second && second->context == second_context && g_state.library_load_count == 2,
                     "the second context owns separate private code");
    runtime.observe_library_module(unknown, second_module, second_context);
    passed &=
        expect(!get_variant(), "unknown library identity invalidates a reused module and function");
    second.reset();
    passed &= expect(runtime.forget_context_checked(second_context) == CUDA_SUCCESS &&
                         g_state.library_unload_count == 2,
                     "unknown owner cannot take private ownership");
    runtime.observe_library_module(library, second_module, second_context);
    passed &= expect(runtime.get_source_function(&function, second_module, entry) == CUDA_SUCCESS &&
                         !runtime.module_source_is_prepared(function),
                     "live context-independent library survives context retirement");
    runtime.observe_library_source(library, "opaque");
    passed &= expect(!get_variant() && runtime.module_source_is_prepared(function),
                     "unqualified load retires old source identity");
    runtime.observe_library_module(library, second_module, second_context);
    passed &= expect(runtime.get_source_function(&function, second_module, entry) == CUDA_SUCCESS &&
                         runtime.module_source_is_prepared(function),
                     "same symbol cannot qualify an unknown replacement image");
    runtime.observe_library_source(library, image.data());
    runtime.observe_library_module(library, second_module, nullptr);
    passed &= expect(runtime.get_source_function(&function, second_module, entry) == CUDA_SUCCESS &&
                         runtime.module_source_is_prepared(function),
                     "missing current context cannot establish an association");
    runtime.retire_library_source(library);
    passed &= expect(runtime.forget_context_checked(second_context) == CUDA_SUCCESS &&
                         g_state.library_unload_count == 2 && g_state.module_unload_count == 0 &&
                         !g_state.guard_failed,
                     "cleanup never unloads borrowed original modules");
    std::cout << "library_identity gemm=" << IsGemm << " passed=" << passed << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    return (argc == 2 || argc == 3 || argc == 4 || argc == 6) && test_owned_sgemm() &&
                   test_launch_and_cleanup() && test_checked_context_retirement() &&
                   test_block_static_launch() && test_cooperative_preparation() &&
                   test_deferred_module_preparation() && test_source_module_retirement() &&
                   test_load_fallback() && test_runtime_source_and_cleanup(argv[1]) &&
                   test_checked_runtime_retirement(argv[1]) &&
                   test_runtime_original_preparation(argv[1]) &&
                   test_cooperative_source_preparation(argv[1]) &&
                   (argc < 3 || test_binary_source(argv[2])) &&
                   (argc < 4 ||
                    (test_binary_source(argv[3], true) && test_binary_source(argv[3], true, true) &&
                     test_library_identity(argv[3]))) &&
                   (argc < 6 ||
                    (test_binary_source<true>(argv[4]) && test_binary_source<true>(argv[5], true) &&
                     test_binary_source<true>(argv[5], true, true) &&
                     test_library_identity<true>(argv[5])))
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
