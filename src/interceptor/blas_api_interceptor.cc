#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/persistent_sgemm.h"
#include "internal/runtime_api_bridge.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace {

// PyTorch may load its CUDA dependencies RTLD_LOCAL. Retain the already loaded
// SDK instead of loading another installation or relying on RTLD_NEXT visibility.
enum class BlasKind { kBlas, kLt };

struct BlasLibrary {
    void* handle{};
    explicit BlasLibrary(BlasKind kind) noexcept {
        const auto* entry = kind == BlasKind::kBlas ? "cublasSgemm_v2" : "cublasLtMatmul";
        const auto* soname = kind == BlasKind::kBlas ? "libcublas.so.12" : "libcublasLt.so.12";
        const auto lookup = glimmer::interceptor::resolve_real_dlsym();
        const auto symbol = lookup ? lookup(RTLD_NEXT, entry) : nullptr;
        Dl_info info{};
        if (symbol && dladdr(symbol, &info) && info.dli_fname)
            handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        else
            handle = dlopen(soname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    }
    ~BlasLibrary() {
        if (handle && dlclose(handle) != 0)
            glimmer::interceptor::report_diagnostic(
                "[glimmer] cuBLAS loader reference release failed\n");
    }
    BlasLibrary(const BlasLibrary&) = delete;
    BlasLibrary& operator=(const BlasLibrary&) = delete;
};

void* original_symbol(const char* name) noexcept {
    static const BlasLibrary library(BlasKind::kBlas);
    const auto lookup = glimmer::interceptor::resolve_real_dlsym();
    return lookup && library.handle ? lookup(library.handle, name) : nullptr;
}

template <typename Function>
Function original(const char* name) noexcept {
    return reinterpret_cast<Function>(original_symbol(name));
}

void* original_lt_symbol(const char* name) noexcept {
    static const BlasLibrary library(BlasKind::kLt);
    const auto lookup = glimmer::interceptor::resolve_real_dlsym();
    return lookup && library.handle ? lookup(library.handle, name) : nullptr;
}

template <typename Function>
Function original_lt(const char* name) noexcept {
    return reinterpret_cast<Function>(original_lt_symbol(name));
}

bool is_selected() noexcept {
    static const bool selected = [] {
        const auto* value = std::getenv("GLIMMER_PERSISTENT_BLAS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return selected;
}

struct HandleRegistry {
    std::mutex mutex;
    std::unordered_map<cublasHandle_t, glimmer::interceptor::PersistentBlasContext> contexts;
    std::unordered_set<cublasLtHandle_t> lt_handles;
};

HandleRegistry& handles() {
    // CPU-only process owner: SDK TLS destructors may call Destroy during loader
    // teardown. Keep bounded metadata reachable rather than race static teardown.
    static auto* registry = new HandleRegistry;
    return *registry;
}

void observe_handle(cublasHandle_t handle,
                    std::optional<glimmer::interceptor::PersistentBlasContext> context) noexcept {
    try {
        auto& registry = handles();
        const std::lock_guard lock(registry.mutex);
        registry.contexts.erase(handle);
        if (handle && context && registry.contexts.size() < 8192)
            registry.contexts.emplace(handle, *context);
    } catch (...) {
        // No work accepted. Failed metadata insertion remains native-only.
    }
}

std::optional<glimmer::interceptor::PersistentBlasContext> handle_context(cublasHandle_t handle) {
    auto& registry = handles();
    const std::lock_guard lock(registry.mutex);
    const auto entry = registry.contexts.find(handle);
    return entry == registry.contexts.end() ? std::nullopt : std::optional{entry->second};
}

void observe_lt_handle(cublasLtHandle_t handle, bool is_live) noexcept {
    try {
        auto& registry = handles();
        const std::lock_guard lock(registry.mutex);
        registry.lt_handles.erase(handle);
        if (handle && is_live && registry.lt_handles.size() < 8192)
            registry.lt_handles.insert(handle);
    } catch (...) {
        // Missing provenance declines before acceptance.
    }
}

bool has_lt_handle(cublasLtHandle_t handle,
                   const glimmer::interceptor::PersistentBlasContext& context) {
    auto& registry = handles();
    const std::lock_guard lock(registry.mutex);
    if (registry.lt_handles.contains(handle)) return true;
    // The SDK explicitly permits valid cuBLAS handles as Lt handles. This cast
    // is a metadata lookup only, never an opaque-handle dereference.
    const auto entry = registry.contexts.find(reinterpret_cast<cublasHandle_t>(handle));
    return entry != registry.contexts.end() && entry->second == context;
}

template <typename Value>
bool operation_attribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attribute,
                         Value& value) noexcept {
    static const auto get =
        original_lt<decltype(&cublasLtMatmulDescGetAttribute)>("cublasLtMatmulDescGetAttribute");
    std::size_t written{};
    return get && get(desc, attribute, &value, sizeof(value), &written) == CUBLAS_STATUS_SUCCESS &&
           written == sizeof(value);
}

template <typename Value>
bool has_operation_attribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attribute,
                             Value expected) noexcept {
    Value value{};
    return operation_attribute(desc, attribute, value) && value == expected;
}

struct LtLayout {
    std::uint64_t rows{}, columns{};
    std::int64_t leading{};
    bool operator==(const LtLayout&) const = default;
};

bool lt_layout(cublasLtMatrixLayout_t desc, LtLayout& layout) noexcept {
    static const auto get = original_lt<decltype(&cublasLtMatrixLayoutGetAttribute)>(
        "cublasLtMatrixLayoutGetAttribute");
    const auto query = [&](cublasLtMatrixLayoutAttribute_t attribute, auto& value) {
        std::size_t written{};
        return get &&
               get(desc, attribute, &value, sizeof(value), &written) == CUBLAS_STATUS_SUCCESS &&
               written == sizeof(value);
    };
    std::uint32_t type{};
    std::int32_t order{}, batch{};
    std::int64_t plane{};
    return query(CUBLASLT_MATRIX_LAYOUT_TYPE, type) && type == CUDA_R_32F &&
           query(CUBLASLT_MATRIX_LAYOUT_ORDER, order) && order == CUBLASLT_ORDER_COL &&
           query(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch) && batch == 1 &&
           query(CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, plane) && plane == 0 &&
           query(CUBLASLT_MATRIX_LAYOUT_ROWS, layout.rows) &&
           query(CUBLASLT_MATRIX_LAYOUT_COLS, layout.columns) &&
           query(CUBLASLT_MATRIX_LAYOUT_LD, layout.leading) && layout.rows > 0 &&
           layout.rows <= std::numeric_limits<std::uint32_t>::max() && layout.columns > 0 &&
           layout.columns <= std::numeric_limits<std::uint32_t>::max() && layout.leading > 0 &&
           layout.leading <= std::numeric_limits<std::uint32_t>::max();
}

bool lt_operation(cublasLtMatmulDesc_t desc, const void*& bias) noexcept {
    if (!has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
                                 std::int32_t{CUBLAS_COMPUTE_32F}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, std::int32_t{CUDA_R_32F}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                 std::int32_t{CUBLASLT_POINTER_MODE_HOST}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, std::int32_t{CUBLAS_OP_T}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, std::int32_t{CUBLAS_OP_N}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_TRANSC, std::int32_t{CUBLAS_OP_N}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_FILL_MODE,
                                 std::int32_t{CUBLAS_FILL_MODE_FULL}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 std::uint32_t{CUBLASLT_EPILOGUE_BIAS}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, std::int32_t{0}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS,
                                 std::int32_t{0}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS,
                                 std::int32_t{0}) ||
        !has_operation_attribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, std::int8_t{0}) ||
        !operation_attribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias) || !bias)
        return false;
    std::int32_t bias_type{};
    if (!operation_attribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_type) ||
        (bias_type != -1 && bias_type != CUDA_R_32F))
        return false;
    // Exclude optional input scaling and output/atomic side effects. A valid
    // descriptor does not imply those semantics are implemented by our kernel.
    for (const auto attribute :
         {CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
          CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER,
          CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER,
          CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER}) {
        if (!has_operation_attribute(desc, attribute, static_cast<const void*>(nullptr)))
            return false;
    }
    return true;
}

bool lt_alignment(const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoCapAttributes_t attribute,
                  const void* pointer, std::int64_t leading) noexcept {
    static const auto get = original_lt<decltype(&cublasLtMatmulAlgoCapGetAttribute)>(
        "cublasLtMatmulAlgoCapGetAttribute");
    std::uint32_t alignment{};
    std::size_t written{};
    return get &&
           get(algo, attribute, &alignment, sizeof(alignment), &written) == CUBLAS_STATUS_SUCCESS &&
           written == sizeof(alignment) && alignment &&
           reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0 &&
           (static_cast<std::uint64_t>(leading) * sizeof(float)) % alignment == 0;
}

cublasStatus_t blas_status(CUresult result) noexcept {
    if (result == CUDA_SUCCESS) return CUBLAS_STATUS_SUCCESS;
    if (result == CUDA_ERROR_OUT_OF_MEMORY) return CUBLAS_STATUS_ALLOC_FAILED;
    if (result == CUDA_ERROR_NOT_SUPPORTED) return CUBLAS_STATUS_NOT_SUPPORTED;
    return CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace

extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle) {
    static const auto native = original<decltype(&cublasCreate_v2)>("cublasCreate_v2");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    try {
        if (!is_selected() || !glimmer::interceptor::cooperative_driver_preview_requested())
            return native(handle);
        const glimmer::interceptor::CooperativeCallScope call;
        const auto result = native(handle);
        if (result == CUBLAS_STATUS_SUCCESS && handle)
            observe_handle(*handle, glimmer::interceptor::persistent_blas_current_context());
        return result;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

extern "C" cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    static const auto native = original<decltype(&cublasDestroy_v2)>("cublasDestroy_v2");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    try {
        namespace interceptor = glimmer::interceptor;
        if (!is_selected() || !interceptor::cooperative_driver_preview_requested())
            return native(handle);
        const interceptor::CooperativeCallScope call;
        if (!interceptor::persistent_blas_current_context()) {
            observe_handle(handle, std::nullopt);
            return native(handle);
        }
        // Destroy's SDK device fence alone cannot see not-yet-issued queue work.
        // Drain first, then keep SDK event/free calls on their ordinary routing
        // paths: a private native scope would suppress their handle retirement.
        const auto boundary =
            interceptor::intercept_cooperative_native_boundary([] { return CUDA_SUCCESS; });
        if (boundary != CUDA_SUCCESS) return blas_status(boundary);
        observe_handle(handle, std::nullopt);
        return native(handle);
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

extern "C" cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle,
                                                      cublasOperation_t trans_a,
                                                      cublasOperation_t trans_b, int m, int n,
                                                      int k, const float* alpha, const float* a,
                                                      int lda, const float* b, int ldb,
                                                      const float* beta, float* c, int ldc) {
    static const auto native = original<decltype(&cublasSgemm_v2)>("cublasSgemm_v2");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    const auto fallback = [&] {
        return native(handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    };
    try {
        namespace interceptor = glimmer::interceptor;
        if (!is_selected() || !interceptor::cooperative_driver_preview_requested() || !handle ||
            !alpha || !beta || !a || !b || !c || trans_a != CUBLAS_OP_N ||
            (trans_b != CUBLAS_OP_N && trans_b != CUBLAS_OP_T) || m <= 0 || n <= 0 || k <= 0 ||
            lda <= 0 || ldb <= 0 || ldc <= 0)
            return fallback();
        interceptor::PersistentSgemmProblem problem{
            static_cast<unsigned int>(m),    static_cast<unsigned int>(n),
            static_cast<unsigned int>(k),    static_cast<unsigned int>(lda),
            static_cast<unsigned int>(ldb),  static_cast<unsigned int>(ldc),
            trans_b == CUBLAS_OP_T ? 1U : 0U};
        // Validate dimensions before querying a handle or reading scalar storage.
        if (!interceptor::persistent_sgemm_layout(problem)) return fallback();
        const interceptor::CooperativeCallScope call;
        if (interceptor::is_inside_driver_call() || interceptor::is_inside_runtime_call() ||
            interceptor::cooperative_process_call_gate().is_native_only_on_current_thread())
            return fallback();
        const auto context = handle_context(handle);
        if (!context || context != interceptor::persistent_blas_current_context())
            return fallback();
        static const auto pointer_mode =
            original<decltype(&cublasGetPointerMode_v2)>("cublasGetPointerMode_v2");
        static const auto math_mode = original<decltype(&cublasGetMathMode)>("cublasGetMathMode");
        static const auto get_stream =
            original<decltype(&cublasGetStream_v2)>("cublasGetStream_v2");
        static const auto sm_target =
            original<decltype(&cublasGetSmCountTarget)>("cublasGetSmCountTarget");
        if (!pointer_mode || !math_mode || !get_stream || !sm_target) return fallback();
        cublasPointerMode_t pointers{};
        cublasMath_t math{};
        cudaStream_t stream{};
        int target{};
        if (pointer_mode(handle, &pointers) != CUBLAS_STATUS_SUCCESS ||
            pointers != CUBLAS_POINTER_MODE_HOST ||
            math_mode(handle, &math) != CUBLAS_STATUS_SUCCESS || math != CUBLAS_DEFAULT_MATH ||
            get_stream(handle, &stream) != CUBLAS_STATUS_SUCCESS || stream != nullptr ||
            sm_target(handle, &target) != CUBLAS_STATUS_SUCCESS || target != 0)
            return fallback();
        problem.alpha = *alpha;
        problem.beta = *beta;
        const auto result = interceptor::try_intercept_persistent_sgemm(
            context->context, problem, reinterpret_cast<CUdeviceptr>(a),
            reinterpret_cast<CUdeviceptr>(b), reinterpret_cast<CUdeviceptr>(c));
        return result ? blas_status(*result) : fallback();
    } catch (const std::bad_alloc&) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* handle) {
    static const auto native = original_lt<decltype(&cublasLtCreate)>("cublasLtCreate");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    try {
        if (!is_selected() || !glimmer::interceptor::cooperative_driver_preview_requested())
            return native(handle);
        const glimmer::interceptor::CooperativeCallScope call;
        const auto result = native(handle);
        if (result == CUBLAS_STATUS_SUCCESS && handle) observe_lt_handle(*handle, true);
        return result;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t handle) {
    static const auto native = original_lt<decltype(&cublasLtDestroy)>("cublasLtDestroy");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    try {
        namespace interceptor = glimmer::interceptor;
        if (!is_selected() || !interceptor::cooperative_driver_preview_requested())
            return native(handle);
        const interceptor::CooperativeCallScope call;
        if (interceptor::persistent_blas_current_context()) {
            const auto boundary =
                interceptor::intercept_cooperative_native_boundary([] { return CUDA_SUCCESS; });
            if (boundary != CUDA_SUCCESS) return blas_status(boundary);
        }
        observe_lt_handle(handle, false);
        return native(handle);
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t a_desc, const void* b, cublasLtMatrixLayout_t b_desc, const void* beta,
    const void* c, cublasLtMatrixLayout_t c_desc, void* d, cublasLtMatrixLayout_t d_desc,
    const cublasLtMatmulAlgo_t* algo, void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
    static const auto native = original_lt<decltype(&cublasLtMatmul)>("cublasLtMatmul");
    if (!native) return CUBLAS_STATUS_NOT_INITIALIZED;
    const auto fallback = [&] {
        return native(handle, operation, alpha, a, a_desc, b, b_desc, beta, c, c_desc, d, d_desc,
                      algo, workspace, workspace_bytes, stream);
    };
    try {
        namespace interceptor = glimmer::interceptor;
        if (!is_selected() || !interceptor::cooperative_driver_preview_requested() || !handle ||
            !operation || !alpha || !beta || !a || !b || !c || c != d || !a_desc || !b_desc ||
            !c_desc || !d_desc || !algo || stream ||
            (workspace_bytes && (!workspace || reinterpret_cast<std::uintptr_t>(workspace) % 256)))
            return fallback();
        const interceptor::CooperativeCallScope call;
        const auto context = interceptor::persistent_blas_current_context();
        if (!context || !has_lt_handle(handle, *context)) return fallback();
        LtLayout left, right, input, output;
        const void* bias{};
        if (!lt_layout(a_desc, left) || !lt_layout(b_desc, right) || !lt_layout(c_desc, input) ||
            !lt_layout(d_desc, output) || input != output || left.rows != right.rows ||
            left.columns != output.rows || right.columns != output.columns ||
            !lt_operation(operation, bias))
            return fallback();
        interceptor::PersistentSgemmProblem problem{static_cast<std::uint32_t>(output.rows),
                                                    static_cast<std::uint32_t>(output.columns),
                                                    static_cast<std::uint32_t>(left.rows),
                                                    static_cast<std::uint32_t>(left.leading),
                                                    static_cast<std::uint32_t>(right.leading),
                                                    static_cast<std::uint32_t>(output.leading),
                                                    0,
                                                    1.0F,
                                                    0.0F,
                                                    1};
        if (!interceptor::persistent_sgemm_layout(problem)) return fallback();
        static const auto check =
            original_lt<decltype(&cublasLtMatmulAlgoCheck)>("cublasLtMatmulAlgoCheck");
        cublasLtMatmulHeuristicResult_t checked{};
        if (!check ||
            check(handle, operation, a_desc, b_desc, c_desc, d_desc, algo, &checked) !=
                CUBLAS_STATUS_SUCCESS ||
            checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > workspace_bytes ||
            !lt_alignment(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, a, left.leading) ||
            !lt_alignment(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES, b, right.leading) ||
            !lt_alignment(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, c, input.leading) ||
            !lt_alignment(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES, d, output.leading))
            return fallback();
        problem.alpha = *static_cast<const float*>(alpha);
        problem.beta = *static_cast<const float*>(beta);
        const auto result = interceptor::try_intercept_persistent_sgemm(
            context->context, problem, reinterpret_cast<CUdeviceptr>(a),
            reinterpret_cast<CUdeviceptr>(b), reinterpret_cast<CUdeviceptr>(d),
            reinterpret_cast<CUdeviceptr>(bias), reinterpret_cast<CUdeviceptr>(workspace),
            workspace_bytes);
        return result ? blas_status(*result) : fallback();
    } catch (const std::bad_alloc&) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
