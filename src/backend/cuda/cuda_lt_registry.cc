#include "internal/cuda_lt_registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> registry_owner;
CudaLtResult rejected(CUresult status) {
    CudaLtResult result;
    result.status = status;
    return result;
}
CudaLtResult completed(cublasStatus_t status) {
    CudaLtResult result;
    result.disposition = CudaLtDisposition::kComplete;
    result.status = CUDA_SUCCESS;
    result.original_status = status;
    return result;
}
bool known_rejection(cublasStatus_t status) {
    return status == CUBLAS_STATUS_INVALID_VALUE || status == CUBLAS_STATUS_NOT_SUPPORTED;
}
bool known_capability(std::uint32_t attribute) noexcept {
    switch (attribute) {
        case CUBLASLT_ALGO_CAP_SPLITK_SUPPORT:
        case CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK:
        case CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT:
        case CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT:
        case CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT:
        case CUBLASLT_ALGO_CAP_UPLO_SUPPORT:
        case CUBLASLT_ALGO_CAP_TILE_IDS:
        case CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX:
        case CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER:
        case CUBLASLT_ALGO_CAP_POINTER_MODE_MASK:
        case CUBLASLT_ALGO_CAP_EPILOGUE_MASK:
        case CUBLASLT_ALGO_CAP_STAGES_IDS:
        case CUBLASLT_ALGO_CAP_LD_NEGATIVE:
        case CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS:
        case CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES:
        case CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES:
        case CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES:
        case CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES:
        case CUBLASLT_ALGO_CAP_ATOMIC_SYNC:
            return true;
        default:
            return false;
    }
}
}  // namespace

std::optional<CudaLtAttribute> cuda_lt_attribute(CudaLtObjectKind kind,
                                                 std::uint32_t attribute) noexcept {
    if (kind == CudaLtObjectKind::kLayout) {
        switch (attribute) {
            case CUBLASLT_MATRIX_LAYOUT_TYPE:
            case CUBLASLT_MATRIX_LAYOUT_ORDER:
            case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                return CudaLtAttribute{4, false};
            case CUBLASLT_MATRIX_LAYOUT_ROWS:
            case CUBLASLT_MATRIX_LAYOUT_COLS:
            case CUBLASLT_MATRIX_LAYOUT_LD:
            case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            case CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET:
                return CudaLtAttribute{8, false};
            default:
                return std::nullopt;
        }
    }
    if (kind == CudaLtObjectKind::kOperation) {
        switch (attribute) {
            case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
            case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
            case CUBLASLT_MATMUL_DESC_POINTER_MODE:
            case CUBLASLT_MATMUL_DESC_TRANSA:
            case CUBLASLT_MATMUL_DESC_TRANSB:
            case CUBLASLT_MATMUL_DESC_TRANSC:
            case CUBLASLT_MATMUL_DESC_FILL_MODE:
            case CUBLASLT_MATMUL_DESC_EPILOGUE:
            case CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
            case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                return CudaLtAttribute{4, false};
            case CUBLASLT_MATMUL_DESC_FAST_ACCUM:
                return CudaLtAttribute{1, false};
            case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
            case CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE:
                return CudaLtAttribute{8, false};
            case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
                return CudaLtAttribute{sizeof(CUdeviceptr), true};
            default:
                // Unknown attributes may contain native pointers; never pass
                // unclassified caller bytes to the original SDK.
                return std::nullopt;
        }
    }
    if (kind == CudaLtObjectKind::kPreference) {
        switch (attribute) {
            case CUBLASLT_MATMUL_PREF_SEARCH_MODE:
            case CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK:
            case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES:
            case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES:
            case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES:
            case CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES:
            case CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT:
                return CudaLtAttribute{4, false};
            case CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            case CUBLASLT_MATMUL_PREF_IMPL_MASK:
                return CudaLtAttribute{8, false};
            default:
                return std::nullopt;
        }
    }
    return std::nullopt;
}

struct CudaLtRegistry::Implementation {
    struct Object {
        std::uint64_t id{};
        CudaLtObjectKind kind{};
        void* native{};
        std::optional<CudaLtDeviceReference> bias;
        bool is_uncertain{}, has_destroy_attempt{};
    };
    const CudaLtDispatch dispatch;
    const std::size_t max_objects;
    const pid_t process{::getpid()};
    std::weak_ptr<const void> owner;
    std::vector<std::shared_ptr<Object>> objects;
    std::vector<cublasLtMatmulAlgo_t> algorithms;
    std::size_t algorithm_sdk_version{};
    std::uint64_t next_id{1};
    bool has_id_space{true}, is_closing{}, is_faulted{};

    Implementation(CudaLtDispatch functions, std::size_t maximum)
        : dispatch(functions), max_objects(maximum) {
        if (!maximum || !functions.layout_create || !functions.layout_destroy ||
            !functions.layout_set || !functions.layout_get || !functions.operation_create ||
            !functions.operation_destroy || !functions.operation_set || !functions.operation_get ||
            !functions.preference_create || !functions.preference_destroy ||
            !functions.preference_set || !functions.preference_get)
            throw std::invalid_argument("invalid owned Lt metadata dispatch or bound");
        objects.reserve(maximum);
        if (!registry_owner) registry_owner = std::make_shared<int>(0);
        owner = registry_owner;
    }
    bool owns() const noexcept {
        return ::getpid() == process && registry_owner && owner.lock() == registry_owner;
    }
    CudaLtResult gate(bool mutating = false) const noexcept {
        if (!owns() || is_faulted) return rejected(CUDA_ERROR_UNKNOWN);
        if (is_closing && mutating) return rejected(CUDA_ERROR_DEINITIALIZED);
        return rejected(CUDA_SUCCESS);
    }
    auto find(std::uint64_t id) {
        return std::ranges::find_if(objects, [id](const auto& object) { return object->id == id; });
    }
    cublasStatus_t destroy_native(Object& object) noexcept {
        object.has_destroy_attempt = true;
        switch (object.kind) {
            case CudaLtObjectKind::kLayout:
                return dispatch.layout_destroy(static_cast<cublasLtMatrixLayout_t>(object.native));
            case CudaLtObjectKind::kOperation:
                return dispatch.operation_destroy(static_cast<cublasLtMatmulDesc_t>(object.native));
            case CudaLtObjectKind::kPreference:
                return dispatch.preference_destroy(
                    static_cast<cublasLtMatmulPreference_t>(object.native));
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
};

CudaLtRegistry::CudaLtRegistry(CudaLtDispatch dispatch, std::size_t max_objects)
    : implementation_(std::make_unique<Implementation>(dispatch, max_objects)) {}
CudaLtRegistry::~CudaLtRegistry() {
    if (implementation_->owns()) {
        const auto result = shutdown();
        if (result.status == CUDA_SUCCESS && result.disposition == CudaLtDisposition::kComplete)
            return;
    }
    // Native destruction cannot move to a foreign thread, repeat an uncertain
    // SDK call, or race a descriptor retained by a real GPU command.
    try {
        spdlog::critical("Retaining unsettled native Lt metadata registry");
    } catch (...) {
    }
    (void)implementation_.release();
}
CudaLtResult CudaLtRegistry::create(const CudaLtCreate& specification) noexcept {
    auto& state = *implementation_;
    auto gate = state.gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (!state.has_id_space || state.objects.size() == state.max_objects)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto object = std::make_shared<Implementation::Object>();
        object->id = state.next_id;
        object->kind = specification.kind;
        cublasStatus_t status{};
        switch (specification.kind) {
            case CudaLtObjectKind::kLayout: {
                cublasLtMatrixLayout_t native{};
                status =
                    state.dispatch.layout_create(&native, specification.type, specification.rows,
                                                 specification.columns, specification.leading);
                object->native = native;
                break;
            }
            case CudaLtObjectKind::kOperation: {
                cublasLtMatmulDesc_t native{};
                status = state.dispatch.operation_create(&native, specification.compute,
                                                         specification.scale);
                object->native = native;
                break;
            }
            case CudaLtObjectKind::kPreference: {
                cublasLtMatmulPreference_t native{};
                status = state.dispatch.preference_create(&native);
                object->native = native;
                break;
            }
            default:
                return rejected(CUDA_ERROR_INVALID_VALUE);
        }
        auto result = completed(status);
        if (status != CUBLAS_STATUS_SUCCESS && !object->native) return result;
        if (state.next_id == std::numeric_limits<std::uint64_t>::max())
            state.has_id_space = false;
        else
            ++state.next_id;
        object->is_uncertain = status != CUBLAS_STATUS_SUCCESS || !object->native;
        for (auto& existing : state.objects)
            if (existing->native == object->native)
                existing->is_uncertain = object->is_uncertain = true;
        state.objects.push_back(object);  // Capacity was reserved before any SDK call.
        if (object->is_uncertain) {
            state.is_faulted = true;
            result.status = CUDA_ERROR_UNKNOWN;
            result.disposition = CudaLtDisposition::kRejected;
        } else
            result.id = object->id;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaLtResult CudaLtRegistry::destroy(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find(id);
    if (found == state.objects.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (found->use_count() != 1)
        return {.disposition = CudaLtDisposition::kDeferred, .status = CUDA_SUCCESS};
    auto result = completed(state.destroy_native(**found));
    if (result.original_status != CUBLAS_STATUS_SUCCESS) {
        (*found)->is_uncertain = state.is_faulted = true;
    } else
        state.objects.erase(found);
    return result;
}
CudaLtResult CudaLtRegistry::set_attribute(
    CudaLtAttributeKey key, std::span<const std::byte> data,
    std::optional<CudaLtDeviceReference> reference) noexcept {
    const auto [id, attribute] = key;
    auto& state = *implementation_;
    auto gate = state.gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find(id);
    if (found == state.objects.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    const auto description = cuda_lt_attribute((*found)->kind, attribute);
    if (!description) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (data.size() != description->bytes || (!description->is_device_pointer && reference))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaLtDisposition::kDeferred, .status = CUDA_SUCCESS};
    // An absent reference means a genuine null pointer. Otherwise the trusted
    // composition layer supplied an independently validated native GPU address.
    CUdeviceptr pointer = reference ? reference->address : 0;
    if (description->is_device_pointer && reference &&
        (!reference->session || !reference->allocation || !reference->address))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (description->is_device_pointer &&
        std::ranges::any_of(data, [](auto byte) { return byte != std::byte{}; }))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    alignas(16) std::array<std::byte, 16> copied{};
    if (description->is_device_pointer)
        std::memcpy(copied.data(), &pointer, sizeof(pointer));
    else
        std::memcpy(copied.data(), data.data(), data.size());
    const void* input = copied.data();
    cublasStatus_t status{};
    switch ((*found)->kind) {
        case CudaLtObjectKind::kLayout:
            status = state.dispatch.layout_set(
                static_cast<cublasLtMatrixLayout_t>((*found)->native),
                static_cast<cublasLtMatrixLayoutAttribute_t>(attribute), input, data.size());
            break;
        case CudaLtObjectKind::kOperation:
            status = state.dispatch.operation_set(
                static_cast<cublasLtMatmulDesc_t>((*found)->native),
                static_cast<cublasLtMatmulDescAttributes_t>(attribute), input, data.size());
            break;
        case CudaLtObjectKind::kPreference:
            status = state.dispatch.preference_set(
                static_cast<cublasLtMatmulPreference_t>((*found)->native),
                static_cast<cublasLtMatmulPreferenceAttributes_t>(attribute), input, data.size());
            break;
    }
    if (status == CUBLAS_STATUS_SUCCESS && description->is_device_pointer)
        (*found)->bias = reference;
    if (status != CUBLAS_STATUS_SUCCESS && !known_rejection(status))
        (*found)->is_uncertain = state.is_faulted = true;
    return completed(status);
}
CUresult CudaLtRegistry::remember_algorithms(
    std::span<const cublasLtMatmulHeuristicResult_t> algorithms, std::size_t sdk_version) noexcept {
    auto& state = *implementation_;
    const auto permission = state.gate(true);
    if (permission.status != CUDA_SUCCESS) return permission.status;
    if (!sdk_version || algorithms.size() > 64) return CUDA_ERROR_INVALID_VALUE;
    if (state.algorithm_sdk_version && state.algorithm_sdk_version != sdk_version)
        return CUDA_ERROR_NOT_SUPPORTED;
    try {
        auto fresh = state.algorithms;
        for (const auto& candidate : algorithms) {
            if (candidate.state != CUBLAS_STATUS_SUCCESS ||
                std::ranges::any_of(fresh, [&](const auto& known) {
                    return std::memcmp(&known, &candidate.algo, sizeof(known)) == 0;
                }))
                continue;
            if (fresh.size() == state.max_objects) return CUDA_ERROR_OUT_OF_MEMORY;
            fresh.push_back(candidate.algo);
        }
        state.algorithms = std::move(fresh);
        state.algorithm_sdk_version = sdk_version;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}
CudaLtResult CudaLtRegistry::algorithm_capability(const cublasLtMatmulAlgo_t& algorithm,
                                                  std::uint32_t attribute, std::size_t bytes,
                                                  std::uint32_t presence,
                                                  std::span<const std::byte> initial,
                                                  std::size_t initial_written) noexcept {
    static_assert(sizeof(cublasLtMatmulAlgo_t) == 64);
    auto& state = *implementation_;
    auto permission = state.gate();
    if (permission.status != CUDA_SUCCESS) return permission;
    if (bytes > 4096 || presence > 3 || initial.size() != ((presence & 1U) ? bytes : 0))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!known_capability(attribute) || !state.dispatch.capability_get || !state.dispatch.version ||
        !std::ranges::any_of(state.algorithms, [&](const auto& known) {
            return std::memcmp(&known, &algorithm, sizeof(known)) == 0;
        }))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (state.dispatch.version() != state.algorithm_sdk_version) {
        state.is_faulted = true;
        return rejected(CUDA_ERROR_UNKNOWN);
    }
    try {
        std::vector<std::byte> output(initial.begin(), initial.end());
        alignas(16) std::array<std::byte, 16> empty{};
        void* buffer = !(presence & 1U) ? nullptr : bytes ? output.data() : empty.data();
        auto written = initial_written;
        const auto status = state.dispatch.capability_get(
            &algorithm, static_cast<cublasLtMatmulAlgoCapAttributes_t>(attribute), buffer, bytes,
            (presence & 2U) ? &written : nullptr);
        auto result = completed(status);
        if (status == CUBLAS_STATUS_SUCCESS && (presence & 2U) && bytes && written > bytes) {
            state.is_faulted = true;
            result.status = CUDA_ERROR_UNKNOWN;
            result.disposition = CudaLtDisposition::kRejected;
            return result;
        }
        // Preserve native error-side writes and untouched tails. sizeWritten
        // is an output value, never a client/server address or resource grant.
        result.data = std::move(output);
        result.value = written;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaLtResult CudaLtRegistry::get_attribute(CudaLtAttributeKey key, std::size_t bytes) noexcept {
    const auto [id, attribute] = key;
    auto& state = *implementation_;
    auto gate = state.gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find(id);
    if (found == state.objects.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    const auto description = cuda_lt_attribute((*found)->kind, attribute);
    if (!description) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (bytes > 4096) return rejected(CUDA_ERROR_INVALID_VALUE);
    try {
        CudaLtResult result;
        result.data.resize(bytes);
        std::size_t written{};
        cublasStatus_t status{};
        void* output = bytes ? result.data.data() : nullptr;
        switch ((*found)->kind) {
            case CudaLtObjectKind::kLayout:
                status = state.dispatch.layout_get(
                    static_cast<cublasLtMatrixLayout_t>((*found)->native),
                    static_cast<cublasLtMatrixLayoutAttribute_t>(attribute), output, bytes,
                    &written);
                break;
            case CudaLtObjectKind::kOperation:
                status = state.dispatch.operation_get(
                    static_cast<cublasLtMatmulDesc_t>((*found)->native),
                    static_cast<cublasLtMatmulDescAttributes_t>(attribute), output, bytes,
                    &written);
                break;
            case CudaLtObjectKind::kPreference:
                status = state.dispatch.preference_get(
                    static_cast<cublasLtMatmulPreference_t>((*found)->native),
                    static_cast<cublasLtMatmulPreferenceAttributes_t>(attribute), output, bytes,
                    &written);
                break;
        }
        result.disposition = CudaLtDisposition::kComplete;
        result.status = CUDA_SUCCESS;
        result.original_status = status;
        if (status != CUBLAS_STATUS_SUCCESS) {
            result.data.clear();
            return result;
        }
        CUdeviceptr observed{};
        if (description->is_device_pointer && bytes && written == sizeof(observed) &&
            written <= bytes)
            std::memcpy(&observed, result.data.data(), sizeof(observed));
        const bool invalid_pointer = description->is_device_pointer && bytes &&
                                     observed != ((*found)->bias ? (*found)->bias->address : 0);
        if (written != description->bytes || (bytes && written > bytes) || invalid_pointer) {
            state.is_faulted = (*found)->is_uncertain = true;
            result.data.clear();
            result.status = CUDA_ERROR_UNKNOWN;
            result.disposition = CudaLtDisposition::kRejected;
            return result;
        }
        result.value = written;
        if (bytes) result.data.resize(written);
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaLtResult CudaLtRegistry::version(bool runtime) noexcept {
    auto& state = *implementation_;
    auto gate = state.gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto function = runtime ? state.dispatch.runtime_version : state.dispatch.version;
    if (!function) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    auto result = rejected(CUDA_SUCCESS);
    result.disposition = CudaLtDisposition::kComplete;
    result.value = function();
    return result;  // A native size_t return is not an original cublasStatus_t.
}
std::optional<CudaLtObjectView> CudaLtRegistry::borrow(std::uint64_t id,
                                                       CudaLtObjectKind kind) noexcept {
    auto& state = *implementation_;
    if (state.gate(true).status != CUDA_SUCCESS) return std::nullopt;
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != kind || (*found)->is_uncertain ||
        (*found)->has_destroy_attempt)
        return std::nullopt;
    return CudaLtObjectView{kind, (*found)->native, (*found)->bias, *found};
}
void CudaLtRegistry::close() noexcept {
    if (implementation_->owns()) implementation_->is_closing = true;
}
CudaLtResult CudaLtRegistry::shutdown() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted) return rejected(CUDA_ERROR_UNKNOWN);
    state.is_closing = true;
    for (auto item = state.objects.begin(); item != state.objects.end();) {
        if (item->use_count() != 1) {
            ++item;
            continue;
        }
        if (state.destroy_native(**item) != CUBLAS_STATUS_SUCCESS) {
            (*item)->is_uncertain = state.is_faulted = true;
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        item = state.objects.erase(item);
    }
    return {.disposition =
                state.objects.empty() ? CudaLtDisposition::kComplete : CudaLtDisposition::kDeferred,
            .status = CUDA_SUCCESS};
}
CudaLtRegistrySnapshot CudaLtRegistry::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    return {state.objects.size(),
            static_cast<std::size_t>(std::ranges::count_if(
                state.objects, [](const auto& object) { return object.use_count() > 1; })),
            state.is_closing, state.is_closing && state.objects.empty(), state.is_faulted};
}
}  // namespace glimmer::backend::cuda
