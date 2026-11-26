#pragma once

// Private state shared by the CudaExecutionSession implementation files.
// Only cuda_execution_session*.cc may include this header.

#include "internal/cuda_execution_session.h"
#include "internal/cuda_external_execution_bridge.h"
#include "internal/cuda_sgemm_image.h"
#include "internal/cuda_gemm_geometry.h"
#include "internal/cuda_gemm_signal_layout.h"
#include "glimmer/transform/ptx_transform.h"
#include "internal/cuda_ptx_cache.h"
#include "internal/cuda_profile_cache.h"
#include "glimmer/transform/cuda_module_image.h"

#ifdef GLIMMER_HAS_CUDNN
#include "../../internal/cudnn_sdk_profile.h"
#include "internal/cuda_dnn_execution.h"
#include "internal/cuda_dnn_backend_execution.h"
#include "internal/cuda_dnn_legacy.h"
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::backend::cuda {
namespace session_detail {
#ifdef GLIMMER_HAS_CUDNN
namespace dnn_profile = ::glimmer::internal::cudnn_profile;
#endif
inline thread_local std::shared_ptr<const void> resource_owner;
inline std::shared_ptr<CudaProfileCache> session_profile_cache(
    const CudaSessionPersistentConfig& persistent, std::shared_ptr<CudaProfileCache> profiling) {
    if (!persistent.observes_block_completion) return profiling;
    if (profiling &&
        (profiling->is_learning() || profiling->snapshot().cases || !profiling->kinds().empty()))
        throw std::invalid_argument("block completion diagnostics cannot use measured profiles");
    // Do not retain an initially empty external cache: a later restore must
    // never feed timings from ordinary code into an observed command.
    return {};
}
inline std::shared_ptr<CudaExecutionOwnerActivity> session_activity(
    const CudaExecutionDomainMember& member) {
    if (!member) return std::make_shared<CudaExecutionOwnerActivity>();
    auto activity = member.owner_activity();
    if (!activity) throw std::invalid_argument("unavailable CUDA execution domain member");
    return activity;
}
inline CudaSessionStreamConfig session_streams(CudaSessionStreamConfig streams,
                                               const CudaExecutionDomainMember& member) noexcept {
    if (member) streams.uses_default_proxy = true;
    return streams;
}
// Service-owned eligibility; clients cannot supply preparation budgets.
constexpr transform::PtxTransformLimits kOwnedPtxTransformLimits{
    transform::kMaxPtxTransformSourceBytes, transform::kMaxPtxTransformOutputBytes,
    transform::kMaxPtxTransformTokens};

struct NativeExecutionScope {
    bool& active;
    const bool previous;
    explicit NativeExecutionScope(bool& value) noexcept : active(value), previous(value) {
        active = true;
    }
    ~NativeExecutionScope() {
        active = previous;
    }
};

class ConfigurationObservation final : public CudaExecutionObserver {
   public:
    ConfigurationObservation(CudaExecutionDomainMember member,
                             CudaProfileContextConfiguration configuration,
                             std::shared_ptr<CudaExecutionObserver> observer)
        : member_(std::move(member)),
          expected{CUDA_SUCCESS, configuration.available, configuration.values},
          observation(std::move(observer)) {}
    void begin() noexcept override {
        if (!owns() || is_terminal_) return;
        if (is_begun_) {
            is_terminal_ = true;
            observation->finish(false);
            return;
        }
        is_begun_ = true;
        observation->begin();
    }
    void finish(bool is_successful) noexcept override {
        if (!owns() || is_terminal_) return;
        is_terminal_ = true;
        const bool qualified = is_successful && is_begun_ &&
                               member_.verify_profile_configuration(expected, true) == CUDA_SUCCESS;
        observation->finish(qualified);
    }

   private:
    bool owns() const noexcept {
        return ::getpid() == process && member_.owner_activity() != nullptr;
    }
    const pid_t process{::getpid()};
    CudaExecutionDomainMember member_;
    const CudaExecutionDomainConfigurationView expected;
    const std::shared_ptr<CudaExecutionObserver> observation;
    bool is_begun_{}, is_terminal_{};
};

inline std::shared_ptr<CudaExecutionObserver> configuration_observation(
    const CudaExecutionDomainMember& member, CudaProfileContextConfiguration configuration,
    std::shared_ptr<CudaExecutionObserver> observation) {
    if (!observation || !configuration.available) return observation;
    return std::make_shared<ConfigurationObservation>(member, configuration,
                                                      std::move(observation));
}

using FunctionProfileTargets = std::array<CUfunction, 6>;

template <class Proof>
std::optional<CudaProfileFunctionConfiguration> capture_function_configuration(
    const CudaSessionDispatch& dispatch, const FunctionProfileTargets& targets,
    std::uint32_t cache_preference, Proof&& proof) noexcept {
    if (!dispatch.function_get_attribute) return std::nullopt;
    CudaProfileFunctionConfiguration result;
    result.cache_preference = cache_preference;
    try {
        for (std::size_t role = 0; role < targets.size(); ++role) {
            if (!targets[role]) continue;
            result.functions |= 1U << role;
            for (std::size_t attribute = 0; attribute < result.values[role].size(); ++attribute) {
                if (!proof()) return std::nullopt;
                int value{};
                const auto status = dispatch.function_get_attribute(
                    &value, static_cast<CUfunction_attribute>(attribute), targets[role]);
                if (!proof()) return std::nullopt;
                if (status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_NOT_SUPPORTED)
                    continue;
                if (status != CUDA_SUCCESS) return std::nullopt;
                if (attribute == 9 ? value < -1 || value > 100 : value < 0) return std::nullopt;
                result.available[role] |= 1U << attribute;
                result.values[role][attribute] = value;
            }
        }
    } catch (...) {
        return std::nullopt;
    }
    return cuda_profile_function_configuration_valid(result) ? std::optional{result} : std::nullopt;
}

class FunctionConfigurationObservation final : public CudaExecutionObserver {
   public:
    FunctionConfigurationObservation(std::shared_ptr<CudaExecutionObserver> observer,
                                     std::function<bool()> verifier)
        : observation(std::move(observer)), verify(std::move(verifier)) {}
    void begin() noexcept override {
        if (!owns() || is_terminal_) return;
        if (is_begun_) {
            is_terminal_ = true;
            observation->finish(false);
            return;
        }
        is_begun_ = true;
        observation->begin();
    }
    void finish(bool is_successful) noexcept override {
        if (!owns() || is_terminal_) return;
        is_terminal_ = true;
        observation->finish(is_successful && is_begun_ && verify());
    }

   private:
    bool owns() const noexcept {
        return ::getpid() == process && std::this_thread::get_id() == thread;
    }
    const pid_t process{::getpid()};
    const std::thread::id thread{std::this_thread::get_id()};
    const std::shared_ptr<CudaExecutionObserver> observation;
    const std::function<bool()> verify;
    bool is_begun_{}, is_terminal_{};
};

inline bool geam_scalar_is_zero(cudaDataType type,
                                const std::array<std::byte, 16>& value) noexcept {
    if (type == CUDA_R_32F) {
        std::uint32_t bits{};
        std::memcpy(&bits, value.data(), sizeof(bits));
        return (bits & 0x7fffffffU) == 0;
    }
    std::uint64_t bits{};
    std::memcpy(&bits, value.data(), sizeof(bits));
    return (bits & 0x7fffffffffffffffULL) == 0;
}

struct PreparationScope {
    const CudaPreparationObserver observer;
    const CudaPreparationPhase phase;
    bool& active;
    const bool was_active;

    PreparationScope(CudaPreparationObserver value, CudaPreparationPhase phase_value,
                     bool& active_value) noexcept
        : observer(value), phase(phase_value), active(active_value), was_active(active_value) {
        active = true;
        if (observer.callback) observer.callback(observer.context, phase, true);
    }
    ~PreparationScope() {
        if (observer.callback) observer.callback(observer.context, phase, false);
        active = was_active;
    }
    PreparationScope(const PreparationScope&) = delete;
    PreparationScope& operator=(const PreparationScope&) = delete;
};

#ifdef GLIMMER_HAS_CUDNN
inline CudaDnnResult dnn_gate(const CudaSessionResult& gate) {
    const auto disposition = gate.disposition == CudaSessionDisposition::kDeferred
                                 ? CudaDnnDisposition::kDeferred
                                 : CudaDnnDisposition::kRejected;
    return {.disposition = disposition, .status = gate.status};
}
inline CudaDnnResult dnn_completed(cudnnStatus_t status) {
    return {.disposition = CudaDnnDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
#endif

inline CUresult context_flags_status(unsigned int flags) noexcept {
    const auto scheduling = flags & CU_CTX_SCHED_MASK;
    if ((flags & ~static_cast<unsigned int>(CU_CTX_FLAGS_MASK)) ||
        (scheduling != CU_CTX_SCHED_AUTO && scheduling != CU_CTX_SCHED_SPIN &&
         scheduling != CU_CTX_SCHED_YIELD && scheduling != CU_CTX_SCHED_BLOCKING_SYNC))
        return CUDA_ERROR_INVALID_VALUE;
    // Application flags cannot opt into a new coredump/security policy here.
    if (flags & (CU_CTX_COREDUMP_ENABLE | CU_CTX_USER_COREDUMP_ENABLE))
        return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_SUCCESS;
}
inline bool clean_context_creation_rejection(CUresult status) noexcept {
    return status == CUDA_ERROR_OUT_OF_MEMORY || status == CUDA_ERROR_INVALID_VALUE ||
           status == CUDA_ERROR_INVALID_DEVICE || status == CUDA_ERROR_NOT_SUPPORTED ||
           status == CUDA_ERROR_NOT_PERMITTED;
}

// The Driver validates PTX at load. This bounded lexer only locates an entry's
// parameter count; the Driver still supplies and validates every byte layout.
// Do not probe one past the last native parameter as an enumeration sentinel.
class PtxTokens {
   public:
    explicit PtxTokens(std::string_view source) noexcept : source_(source) {}
    [[nodiscard]] std::string_view next() noexcept {
        while (position_ < source_.size()) {
            if (std::isspace(static_cast<unsigned char>(source_[position_]))) {
                ++position_;
                continue;
            }
            if (source_.substr(position_, 2) == "//") {
                const auto end = source_.find('\n', position_ + 2);
                position_ = end == std::string_view::npos ? source_.size() : end + 1;
                continue;
            }
            if (source_.substr(position_, 2) == "/*") {
                const auto end = source_.find("*/", position_ + 2);
                position_ = end == std::string_view::npos ? source_.size() : end + 2;
                continue;
            }
            if (source_[position_] == '"') {
                ++position_;
                while (position_ < source_.size()) {
                    const auto value = source_[position_++];
                    if (value == '"') break;
                    if (value == '\\' && position_ < source_.size()) ++position_;
                }
                continue;
            }
            const auto begin = position_++;
            constexpr std::string_view punctuation = "()[]{},;";
            if (punctuation.find(source_[begin]) == std::string_view::npos)
                while (position_ < source_.size() &&
                       !std::isspace(static_cast<unsigned char>(source_[position_])) &&
                       punctuation.find(source_[position_]) == std::string_view::npos &&
                       source_[position_] != '"' && source_.substr(position_, 2) != "//" &&
                       source_.substr(position_, 2) != "/*")
                    ++position_;
            return source_.substr(begin, position_ - begin);
        }
        return {};
    }

   private:
    std::string_view source_;
    std::size_t position_{};
};

inline std::optional<std::size_t> parameter_count(PtxTokens tokens, std::string_view name,
                                                  std::size_t maximum) noexcept {
    for (auto token = tokens.next(); !token.empty(); token = tokens.next()) {
        if (token != ".entry" || tokens.next() != name) continue;
        if (tokens.next() != "(") return std::nullopt;
        std::size_t depth = 1, count{};
        bool is_next_parameter = true;
        while (!(token = tokens.next()).empty()) {
            if (token == "(") {
                ++depth;
                continue;
            }
            if (token == ")") {
                if (--depth == 0)
                    return is_next_parameter && count ? std::nullopt : std::optional{count};
                continue;
            }
            if (depth != 1) continue;
            if (token == ",") {
                if (is_next_parameter) return std::nullopt;
                is_next_parameter = true;
            }
            if (token == ".param") {
                if (!is_next_parameter || count == maximum) return std::nullopt;
                ++count;
                is_next_parameter = false;
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

inline CudaSessionResult rejected(CUresult status) noexcept {
    CudaSessionResult result;
    result.status = status;
    return result;
}
inline CudaSessionResult completed(CUresult status) noexcept {
    CudaSessionResult result;
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = status;
    result.original_status = status;
    return result;
}
inline CudaSessionResult settled() noexcept {
    auto result = completed(CUDA_SUCCESS);
    result.original_status.reset();
    return result;
}
inline CudaSessionResult deferred_resource_release() noexcept {
    auto result = rejected(CUDA_ERROR_NOT_READY);
    result.disposition = CudaSessionDisposition::kDeferred;
    return result;
}
inline CudaSessionBlasResult blas_gate(const CudaSessionResult& result) noexcept {
    return {.disposition = result.disposition, .status = result.status, .original_status = {}};
}
inline CudaSessionBlasResult blas_completed(cublasStatus_t status) noexcept {
    return {.disposition = CudaSessionDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
inline CudaSessionRandResult random_gate(const CudaSessionResult& result) noexcept {
    return {.disposition = result.disposition, .status = result.status, .original_status = {}};
}
inline CudaSessionRandResult random_completed(curandStatus_t status) noexcept {
    return {.disposition = CudaSessionDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
struct MatrixDimensions {
    int rows{}, columns{}, leading{};
};
inline std::optional<std::size_t> matrix_bytes(MatrixDimensions dimensions,
                                               std::size_t width = sizeof(float)) noexcept {
    const auto [rows, columns, leading] = dimensions;
    if (rows <= 0 || columns <= 0 || leading < rows) return std::nullopt;
    const auto count = static_cast<std::size_t>(columns - 1);
    const auto stride = static_cast<std::size_t>(leading);
    const auto last = static_cast<std::size_t>(rows);
    const auto maximum = std::numeric_limits<std::size_t>::max() / width;
    if (last > maximum || count > (maximum - last) / stride) return std::nullopt;
    return (count * stride + last) * width;
}
inline std::optional<std::size_t> vector_bytes(int length, int increment) noexcept {
    if (length <= 0 || !increment) return std::nullopt;
    const auto signed_stride = static_cast<std::int64_t>(increment);
    const auto stride =
        static_cast<std::uint64_t>(signed_stride < 0 ? -signed_stride : signed_stride);
    const auto intervals = static_cast<std::uint64_t>(length - 1);
    constexpr auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (intervals > (maximum - 1) / stride) return std::nullopt;
    return static_cast<std::size_t>(intervals * stride + 1) * sizeof(float);
}
inline std::size_t gemm_element_bytes(cudaDataType type) noexcept {
    switch (type) {
        case CUDA_R_8I:
        case CUDA_R_8U:
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
            return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF:
        case CUDA_R_16I:
        case CUDA_R_16U:
        case CUDA_C_8I:
        case CUDA_C_8U:
            return 2;
        case CUDA_R_32F:
        case CUDA_R_32I:
        case CUDA_R_32U:
        case CUDA_C_16F:
        case CUDA_C_16BF:
        case CUDA_C_16I:
        case CUDA_C_16U:
            return 4;
        case CUDA_R_64F:
        case CUDA_R_64I:
        case CUDA_R_64U:
        case CUDA_C_32F:
        case CUDA_C_32I:
        case CUDA_C_32U:
            return 8;
        case CUDA_C_64F:
        case CUDA_C_64I:
        case CUDA_C_64U:
            return 16;
        default:
            return 0;
    }
}
inline bool gemm_is_complex(cudaDataType type) noexcept {
    return type == CUDA_C_16F || type == CUDA_C_16BF || type == CUDA_C_32F || type == CUDA_C_64F ||
           type == CUDA_C_8I || type == CUDA_C_8U || type == CUDA_C_16I || type == CUDA_C_16U ||
           type == CUDA_C_32I || type == CUDA_C_32U || type == CUDA_C_64I || type == CUDA_C_64U;
}
inline std::size_t gemm_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
    std::size_t width{};
    switch (compute) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            width = 2;
            break;
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_PEDANTIC:
        case CUBLAS_COMPUTE_32F_FAST_16F:
        case CUBLAS_COMPUTE_32F_FAST_16BF:
        case CUBLAS_COMPUTE_32F_FAST_TF32:
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            width = 4;
            break;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            width = 8;
            break;
        default:
            return 0;
    }
    return width * (gemm_is_complex(output) ? 2U : 1U);
}
class SgemmCall final : public CudaExecutionLibraryCall {
   public:
    SgemmCall(decltype(&cublasSgemm_v2) dispatch, cublasHandle_t handle, CudaSessionSgemm problem,
              std::array<CUdeviceptr, 5> addresses) noexcept
        : dispatch_(dispatch), handle_(handle), problem_(problem), addresses_(addresses) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        // The cuBLAS ABI represents owned device addresses as pointers. These
        // values are never dereferenced by host code.
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const auto* alpha = problem_.uses_device_scalars
                                ? reinterpret_cast<const float*>(addresses_[3])
                                : &problem_.alpha;
        const auto* beta = problem_.uses_device_scalars
                               ? reinterpret_cast<const float*>(addresses_[4])
                               : &problem_.beta;
        return static_cast<std::int32_t>(dispatch_(
            handle_, problem_.transpose_a, problem_.transpose_b, problem_.rows, problem_.columns,
            problem_.depth, alpha, reinterpret_cast<const float*>(addresses_[0]),
            problem_.leading_a, reinterpret_cast<const float*>(addresses_[1]), problem_.leading_b,
            beta, reinterpret_cast<float*>(addresses_[2]), problem_.leading_c));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    decltype(&cublasSgemm_v2) dispatch_;
    cublasHandle_t handle_;
    CudaSessionSgemm problem_;
    std::array<CUdeviceptr, 5> addresses_;
};
class GemmExCall final : public CudaExecutionLibraryCall {
   public:
    GemmExCall(CudaGemmExFunction dispatch, cublasHandle_t handle, CudaSessionGemmEx problem,
               std::array<CUdeviceptr, 5> addresses) noexcept
        : dispatch_(dispatch), handle_(handle), problem_(problem), addresses_(addresses) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto& shape = problem_.operation;
        // NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK address ABI.
        const auto* alpha = shape.uses_device_scalars ? reinterpret_cast<const void*>(addresses_[3])
                                                      : problem_.alpha.data();
        const auto* beta = shape.uses_device_scalars ? reinterpret_cast<const void*>(addresses_[4])
                                                     : problem_.beta.data();
        return static_cast<std::int32_t>(dispatch_(
            handle_, shape.transpose_a, shape.transpose_b, shape.rows, shape.columns, shape.depth,
            alpha, reinterpret_cast<const void*>(addresses_[0]), problem_.a_type, shape.leading_a,
            reinterpret_cast<const void*>(addresses_[1]), problem_.b_type, shape.leading_b, beta,
            reinterpret_cast<void*>(addresses_[2]), problem_.c_type, shape.leading_c,
            problem_.compute_type, problem_.algorithm));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    CudaGemmExFunction dispatch_;
    cublasHandle_t handle_;
    CudaSessionGemmEx problem_;
    std::array<CUdeviceptr, 5> addresses_;
};
class StridedSgemmCall final : public CudaExecutionLibraryCall {
   public:
    StridedSgemmCall(decltype(&cublasSgemmStridedBatched) dispatch, cublasHandle_t handle,
                     CudaSessionStridedGemmEx problem,
                     std::array<CUdeviceptr, 5> addresses) noexcept
        : dispatch_(dispatch), handle_(handle), problem_(problem), addresses_(addresses) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto& typed = problem_.operation;
        const auto& shape = typed.operation;
        alignas(16) float av{}, bv{};
        std::memcpy(&av, typed.alpha.data(), sizeof(av));
        std::memcpy(&bv, typed.beta.data(), sizeof(bv));
        // NOLINTBEGIN(performance-no-int-to-ptr): exact native device pointer ABI.
        const auto* alpha =
            shape.uses_device_scalars ? reinterpret_cast<const float*>(addresses_[3]) : &av;
        const auto* beta =
            shape.uses_device_scalars ? reinterpret_cast<const float*>(addresses_[4]) : &bv;
        return static_cast<std::int32_t>(dispatch_(
            handle_, shape.transpose_a, shape.transpose_b, shape.rows, shape.columns, shape.depth,
            alpha, reinterpret_cast<const float*>(addresses_[0]), shape.leading_a,
            problem_.strides[0], reinterpret_cast<const float*>(addresses_[1]), shape.leading_b,
            problem_.strides[1], beta, reinterpret_cast<float*>(addresses_[2]), shape.leading_c,
            problem_.strides[2], problem_.batches));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    decltype(&cublasSgemmStridedBatched) dispatch_;
    cublasHandle_t handle_;
    CudaSessionStridedGemmEx problem_;
    std::array<CUdeviceptr, 5> addresses_;
};
class StridedGemmExCall final : public CudaExecutionLibraryCall {
   public:
    StridedGemmExCall(CudaStridedGemmExFunction dispatch, cublasHandle_t handle,
                      CudaSessionStridedGemmEx problem,
                      std::array<CUdeviceptr, 5> addresses) noexcept
        : dispatch_(dispatch), handle_(handle), problem_(problem), addresses_(addresses) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto& typed = problem_.operation;
        const auto& shape = typed.operation;
        // NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK address ABI.
        const auto* alpha = shape.uses_device_scalars ? reinterpret_cast<const void*>(addresses_[3])
                                                      : typed.alpha.data();
        const auto* beta = shape.uses_device_scalars ? reinterpret_cast<const void*>(addresses_[4])
                                                     : typed.beta.data();
        return static_cast<std::int32_t>(dispatch_(
            handle_, shape.transpose_a, shape.transpose_b, shape.rows, shape.columns, shape.depth,
            alpha, reinterpret_cast<const void*>(addresses_[0]), typed.a_type, shape.leading_a,
            problem_.strides[0], reinterpret_cast<const void*>(addresses_[1]), typed.b_type,
            shape.leading_b, problem_.strides[1], beta, reinterpret_cast<void*>(addresses_[2]),
            typed.c_type, shape.leading_c, problem_.strides[2], problem_.batches,
            typed.compute_type, typed.algorithm));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    CudaStridedGemmExFunction dispatch_;
    cublasHandle_t handle_;
    CudaSessionStridedGemmEx problem_;
    std::array<CUdeviceptr, 5> addresses_;
};
template <typename Value, typename Attribute>
bool lt_property(const CudaLtDispatch& sdk, const CudaLtObjectView& object, Attribute attribute,
                 Value& value, const auto& proof) noexcept {
    if (!proof()) return false;
    std::size_t written{};
    cublasStatus_t status{};
    if constexpr (std::is_same_v<Attribute, cublasLtMatrixLayoutAttribute_t>) {
        if (!sdk.layout_get) return false;
        status = sdk.layout_get(static_cast<cublasLtMatrixLayout_t>(object.native), attribute,
                                &value, sizeof(value), &written);
    } else {
        if (!sdk.operation_get) return false;
        status = sdk.operation_get(static_cast<cublasLtMatmulDesc_t>(object.native), attribute,
                                   &value, sizeof(value), &written);
    }
    return proof() && status == CUBLAS_STATUS_SUCCESS && written == sizeof(value);
}
struct LtMatrixState {
    cudaDataType type{};
    cublasLtOrder_t order{};
    std::uint64_t rows{}, columns{};
    std::int64_t leading{}, stride{};
    std::int32_t batches{};
    std::size_t bytes{};
};
inline bool lt_matrix_state(const CudaLtDispatch& sdk, const CudaLtObjectView& view, bool is_output,
                            LtMatrixState& matrix, const auto& proof) noexcept {
    std::int64_t plane{};
    if (!view.native || !view.lifetime || view.kind != CudaLtObjectKind::kLayout ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_TYPE, matrix.type, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_ORDER, matrix.order, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_ROWS, matrix.rows, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_COLS, matrix.columns, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_LD, matrix.leading, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, matrix.batches, proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, matrix.stride,
                     proof) ||
        !lt_property(sdk, view, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, plane, proof) || plane ||
        !matrix.rows || !matrix.columns || matrix.leading < 0 || matrix.batches <= 0 ||
        matrix.stride < 0 ||
        (matrix.order != CUBLASLT_ORDER_COL && matrix.order != CUBLASLT_ORDER_ROW))
        return false;
    const auto minor = matrix.order == CUBLASLT_ORDER_COL ? matrix.rows : matrix.columns;
    const auto major = matrix.order == CUBLASLT_ORDER_COL ? matrix.columns : matrix.rows;
    const auto leading = static_cast<std::uint64_t>(matrix.leading);
    // A zero C leading dimension is a documented read-only broadcast. Other
    // invalid leading dimensions cannot authorize an undersized memory lease.
    if ((leading && leading < minor) || (is_output && leading < minor)) return false;
    const auto width = gemm_element_bytes(matrix.type);
    if (!width) return false;
    const auto maximum = std::numeric_limits<std::size_t>::max() / width;
    if (minor > maximum || (leading && major - 1 > (maximum - minor) / leading)) return false;
    const auto elements = (major - 1) * leading + minor;
    const auto batches = static_cast<std::uint64_t>(matrix.batches - 1);
    const auto stride = static_cast<std::uint64_t>(matrix.stride);
    if ((is_output && batches && stride < elements) ||
        (batches && stride > (maximum - elements) / batches))
        return false;
    matrix.bytes = static_cast<std::size_t>(elements + batches * stride) * width;
    return true;
}
class LtMatmulCall final : public CudaExecutionLibraryCall {
   public:
    LtMatmulCall(decltype(&cublasLtMatmul) function, cublasLtHandle_t handle,
                 CudaSessionLtMatmul problem, std::array<CUdeviceptr, 7> addresses)
        : function_(function),
          handle_(handle),
          problem_(std::move(problem)),
          addresses_(addresses) {}
    std::int32_t issue(CUstream stream, std::span<const std::byte>) noexcept override {
        const auto& descriptors = problem_.descriptors;
        const auto layout = [&](std::size_t index) {
            return static_cast<cublasLtMatrixLayout_t>(descriptors.matrices[index].native);
        };
        // NOLINTBEGIN(performance-no-int-to-ptr): original SDK GPU address ABI.
        const auto* alpha = problem_.uses_device_scalars
                                ? reinterpret_cast<const void*>(addresses_[4])
                                : problem_.alpha.data();
        const auto* beta = problem_.uses_device_scalars
                               ? reinterpret_cast<const void*>(addresses_[5])
                               : problem_.beta.data();
        return static_cast<std::int32_t>(function_(
            handle_, static_cast<cublasLtMatmulDesc_t>(descriptors.operation.native), alpha,
            reinterpret_cast<const void*>(addresses_[0]), layout(0),
            reinterpret_cast<const void*>(addresses_[1]), layout(1), beta,
            reinterpret_cast<const void*>(addresses_[2]),
            descriptors.has_c_layout ? layout(2) : nullptr, reinterpret_cast<void*>(addresses_[3]),
            layout(3), descriptors.algorithm ? &*descriptors.algorithm : nullptr,
            reinterpret_cast<void*>(addresses_[6]), problem_.workspace_bytes,
            reinterpret_cast<cudaStream_t>(stream)));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    decltype(&cublasLtMatmul) function_;
    cublasLtHandle_t handle_;
    CudaSessionLtMatmul problem_;
    std::array<CUdeviceptr, 7> addresses_;
};
}  // namespace session_detail
using namespace session_detail;

struct CudaSessionEvent::State {
    CUevent native{};
    std::optional<CudaExecutionFrontier> frontier;
    CUresult record_status{CUDA_SUCCESS};
    bool has_destroy_attempt{};
    CudaSessionEventIdentity identity;
    std::weak_ptr<const CudaExecutionOwnerActivity> activity;
};
struct CudaSessionExternalContext::State {
    std::uint64_t id{};
    CUcontext native{};
    int device_ordinal{-1};
    pid_t process{};
    std::shared_ptr<const void> lifetime;
};
struct CudaSessionExternalCommand::State {
    std::vector<std::shared_ptr<const void>> process_lifetimes;
    CudaSessionExternalContext context;
    std::uint64_t stream_id{};
    CUstream stream{};
    std::vector<CudaSessionExternalRange> ranges;
    std::vector<std::uint64_t> parent_modules;
};

struct CudaExecutionSession::Implementation {
    struct Event {
        std::uint64_t id{};
        unsigned int flags{};
        std::shared_ptr<CudaSessionEvent::State> current;
    };
    class EventCall final : public CudaExecutionEventCall {
       public:
        std::shared_ptr<CudaSessionEvent::State> generation;
        CudaSessionEvent source;
        CudaSessionDispatch dispatch;
        unsigned int flags{};
        bool is_wait{};
        CUresult dependency_status() const noexcept override {
            if (!is_wait) return CUDA_SUCCESS;
            if (source.lifetime_) return source.observe();
            if (generation->record_status != CUDA_SUCCESS) return generation->record_status;
            return generation->frontier ? generation->frontier->observe() : CUDA_SUCCESS;
        }
        CUresult issue(CUstream stream) noexcept override {
            if (is_wait) return dispatch.stream_wait_event(stream, generation->native, flags);
            const auto result = flags
                                    ? dispatch.event_record_flags(generation->native, stream, flags)
                                    : dispatch.execution.event_record(generation->native, stream);
            generation->record_status = result;
            return result;
        }
    };
    struct PendingEvent {
        ExecutionReceipt receipt;
        std::shared_ptr<EventCall> call;
    };
    struct Allocation {
        std::uint64_t id{};
        CUdeviceptr address{};
        std::size_t bytes{};
        std::uint64_t module{};
    };
    struct FunctionCarveout {
        CUfunction native{};
        std::optional<int> logical_value;
        bool is_hint_applied{}, is_override_active{};
    };
    struct Module {
        std::uint64_t id{};
        CUmodule native{};
        std::string source;
        std::unique_ptr<CudaModuleJitArguments> jit;
        std::string profile_source;
        std::shared_ptr<const transform::PersistentPtx> transformed;
        std::vector<std::byte> binary;
        std::optional<transform::CudaModuleMetadata> metadata;
        CUmodule prepared_native{};
        bool has_preparation_attempt{};
        bool shares_original_module{};
        std::size_t charged_bytes{};
        std::optional<std::size_t> source_static_bytes;
        std::size_t prepared_static_budget{}, native_static_bytes{}, prepared_static_bytes{};
        std::size_t discovered_static_bytes{};
        // Retain the first original observation even if subsequent function
        // preparation fails. The module is the native function's lifetime owner.
        std::vector<FunctionCarveout> function_carveouts;
    };
    struct Stream {
        std::uint64_t id{};
        CUstream native{};
        bool is_nonblocking{};
    };
    struct Lt {
        std::uint64_t id{};
        cublasLtHandle_t native{};
        bool is_uncertain{}, has_destroy_attempt{};
        std::size_t sdk_version{};
        std::vector<cublasLtMatmulAlgo_t> algorithms{};
    };
    struct Blas {
        std::uint64_t id{}, stream{};
        cublasHandle_t native{};
        bool is_uncertain{}, has_destroy_attempt{};
        cublasPointerMode_t pointer_mode{CUBLAS_POINTER_MODE_HOST};
        std::uint64_t math_mode{};
        CudaAllocationRange workspace{};
        std::size_t workspace_bytes{};
        bool has_workspace{};
        cublasAtomicsMode_t atomics_mode{CUBLAS_ATOMICS_NOT_ALLOWED};
        int sm_count_target{};
        // The SDK documents cuBLAS handles as valid Lt compute handles. This
        // alias never owns/destroys the underlying native object a second time.
        Lt lt{};
    };
    struct Random {
        std::uint64_t id{};
        curandGenerator_t native{};
        bool is_uncertain{}, has_destroy_attempt{};
    };
#ifdef GLIMMER_HAS_CUDNN
    struct Dnn {
        std::uint64_t id{}, stream{};
        cudnnHandle_t native{};
        bool is_uncertain{}, has_destroy_attempt{};
    };
#endif
    struct PrivateFunction {
        CUfunction native{};
        transform::PtxExecutionKind kind{};
        std::vector<CudaArgumentLayout> arguments;
        std::size_t bytes{};
        bool observes_block_completion{};
    };
    struct Function {
        std::uint64_t id{}, module{};
        CUfunction native{};
        std::string name;
        std::vector<CudaArgumentLayout> arguments;
        std::size_t parameter_bytes{};
        CudaProfileFunction profile;
        std::vector<PrivateFunction> variants;
        bool has_configuration_attempt{}, is_persistent_disabled{};
        bool has_unknown_configuration{};
        std::uint32_t cache_preference{};
        std::optional<CudaProfileFunctionConfiguration> configuration{
            CudaProfileFunctionConfiguration{}};
    };
    struct alignas(64) HostSlot {
        std::uint32_t retreat{};
        std::uint32_t completed{};
        std::uint64_t zero{}, claimed{};
    };
    static_assert(sizeof(HostSlot) == kPersistentProgressHostSlotBytes &&
                  offsetof(HostSlot, completed) == 4 && offsetof(HostSlot, zero) == 8 &&
                  offsetof(HostSlot, claimed) == 16);
    struct alignas(kGemmSignalShardBytes) GemmSignalShard {
        std::uint32_t retreat{};
    };
    static_assert(sizeof(GemmSignalShard) == kGemmSignalShardBytes);
    // A process-local fallback lock cannot synchronize mapped words with GPU
    // loads or stores. These checks do not prove cross-device visibility.
    using MappedWordAtomic = std::atomic_ref<std::uint32_t>;
    static_assert(MappedWordAtomic::is_always_lock_free);
    static_assert(alignof(HostSlot) % MappedWordAtomic::required_alignment == 0 &&
                  offsetof(HostSlot, retreat) % MappedWordAtomic::required_alignment == 0 &&
                  offsetof(HostSlot, completed) % MappedWordAtomic::required_alignment == 0);
    static_assert(alignof(GemmSignalShard) % MappedWordAtomic::required_alignment == 0 &&
                  offsetof(GemmSignalShard, retreat) % MappedWordAtomic::required_alignment == 0);
    struct HostMapping {
        void* pointer{MAP_FAILED};
        std::size_t bytes{};
        bool can_unmap{true};
        ~HostMapping() {
            if (pointer == MAP_FAILED) return;
            if (!can_unmap || ::munmap(pointer, bytes) != 0) {
                try {
                    spdlog::critical("Retaining unsettled shared CUDA host mapping: bytes={}",
                                     bytes);
                } catch (...) {
                }
            }
        }
        HostMapping() = default;
        HostMapping(const HostMapping&) = delete;
        HostMapping& operator=(const HostMapping&) = delete;
    };
    struct HostAllocation {
        std::uint64_t id{};
        std::size_t bytes{};
        unsigned int flags{};
        CUdeviceptr address{};
        std::shared_ptr<HostMapping> mapping;
        bool is_retiring{}, is_uncertain{}, has_unregister_attempt{};
        bool is_pageable{};
    };
    // Lease destruction is CPU-only. Only this session's actual owner performs
    // SDK cleanup, after dispatcher retirement or a real owned-context fence.
    struct Storage {
        CUcontext context{};
        CudaExecutionDomainMember domain_member;
        CUstream default_proxy{};
        bool has_default_proxy_destroy_attempt{}, is_default_proxy_uncertain{};
        std::size_t staging_bytes{};
        std::vector<Allocation> allocations;
        std::vector<Module> modules;
        std::vector<Stream> streams;
        std::vector<Event> events;
        std::vector<std::shared_ptr<CudaSessionEvent::State>> event_generations;
        std::atomic<std::size_t> event_borrows{};
        std::vector<Blas> blas;
        std::vector<Lt> lt;
        std::vector<Random> random;
#ifdef GLIMMER_HAS_CUDNN
        std::vector<Dnn> dnn;
#endif
        std::vector<Function> functions;
        std::vector<HostAllocation> host_allocations;
        std::size_t host_memory_bytes{};
        std::size_t device_bytes{}, image_bytes{};
        CUdeviceptr counters{}, mapped{};
        HostSlot* host{};
        GemmSignalShard* gemm_signals{};
        std::vector<bool> occupied;
        std::size_t counter_bytes{}, host_bytes{};
        std::size_t counter_stride{sizeof(std::uint64_t)};
        std::size_t counter_initialization_bytes{sizeof(std::uint64_t)};
        bool is_counter_uncertain{}, is_host_uncertain{}, has_host_free_attempt{},
            has_counter_free_attempt{}, is_progress_ready{};
        CUmodule gemm_module{};
        std::size_t gemm_static_bytes{};
        std::array<CUfunction, 25> gemm_functions{};
        std::array<std::uint64_t, 25> gemm_ids{};
        std::array<CUfunction, 3> lt_functions{};
        std::array<std::uint64_t, 3> lt_ids{};
        bool has_gemm_preparation_attempt{};
    };

    struct Staging {
        std::shared_ptr<Storage> storage;
        std::shared_ptr<const void> process_lifetime;
        std::unique_ptr<std::byte[]> data;
        const std::size_t bytes;
        Staging(std::shared_ptr<Storage> owner, std::size_t count)
            : storage(std::move(owner)), data(std::make_unique<std::byte[]>(count)), bytes(count) {
            storage->staging_bytes += bytes;
        }
        ~Staging() {
            storage->staging_bytes -= bytes;
        }
        Staging(const Staging&) = delete;
        Staging& operator=(const Staging&) = delete;
    };
    struct ProcessTail final : CudaProcessMemoryCompletion {
        std::shared_ptr<Storage> storage;
        bool stopped{};
        explicit ProcessTail(std::shared_ptr<Storage> value) : storage(std::move(value)) {}
        CUresult observe() noexcept override {
            return stopped ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
        }
        void complete() noexcept {
            stopped = true;
            storage.reset();
        }
    };
    struct ProcessAllocation final : CudaExecutionLibraryCall, CudaSessionProcessAllocation {
        const pid_t process{::getpid()};
        std::shared_ptr<CudaProcessMemoryBackend> backend;
        CudaProcessMemoryRequest request;
        std::uint64_t id{};
        std::shared_ptr<CudaNativeOutputCell> output;
        std::shared_ptr<ProcessTail> tail;
        std::optional<CudaProcessMemoryReply> result;
        bool is_cancelled{};
        ~ProcessAllocation() override {
            if (::getpid() == process && id && !is_cancelled &&
                (!result || !result->original_status))
                (void)backend->cancel(id);
        }
        bool borrows_process_memory() const noexcept override {
            return false;
        }
        std::int32_t issue(CUstream stream, std::span<const std::byte>) noexcept override {
            if (result) return CUDA_ERROR_UNKNOWN;
            result = backend->allocate(id, request, stream, output, tail);
            return result->original_status.value_or(static_cast<std::int32_t>(result->status));
        }
        CudaExecutionLibraryCompletion completion() const noexcept override {
            if (!result) return {CUDA_ERROR_UNKNOWN, true, false};
            return {.status = result->is_quarantined    ? CUDA_ERROR_UNKNOWN
                              : result->original_status ? CUDA_SUCCESS
                                                        : result->status,
                    .is_cleanup_uncertain = result->is_quarantined,
                    .was_native_issued = result->original_status.has_value(),
                    .is_clean_preparation_rejection =
                        !result->original_status && !result->is_quarantined};
        }
        CUresult cleanup_after_physical_stop() noexcept override {
            if (!result) return CUDA_ERROR_UNKNOWN;
            if (!result->original_status) {
                const auto status = backend->cancel(id);
                if (status != CUDA_SUCCESS) return status;
                is_cancelled = true;
            }
            tail->complete();
            return CUDA_SUCCESS;
        }
        std::optional<CudaProcessMemoryReply> reply() const noexcept override {
            return ::getpid() == process ? result : std::nullopt;
        }
    };
    struct EventLease {
        std::shared_ptr<Storage> storage;
        explicit EventLease(std::shared_ptr<Storage> owner) : storage(std::move(owner)) {
            storage->event_borrows.fetch_add(1, std::memory_order_relaxed);
        }
        ~EventLease() {
            storage->event_borrows.fetch_sub(1, std::memory_order_release);
        }
        EventLease(const EventLease&) = delete;
        EventLease& operator=(const EventLease&) = delete;
    };
    struct ProgressParameters {
        std::size_t slot{};
        std::uint64_t sequence{}, total{};
        unsigned int workers{};
        unsigned int max_workers{};
        bool can_retarget_workers{};
        transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
        std::array<std::size_t, 2> slice_offsets{};
        bool observes_block_completion{};
        bool uses_sharded_host_retreat{};
    };
    class Progress final : public CudaExecutionPersistentProgress {
       public:
        Progress(std::shared_ptr<Storage> storage, const CudaSessionDispatch& dispatch,
                 ProgressParameters parameters, std::shared_ptr<const void> operation = {})
            : storage_(std::move(storage)),
              dispatch_(dispatch),
              slot_(parameters.slot),
              sequence(parameters.sequence),
              total_blocks_(parameters.total),
              workers_(parameters.workers),
              max_workers(parameters.max_workers ? parameters.max_workers : parameters.workers),
              can_retarget_workers(parameters.can_retarget_workers),
              operation_(std::move(operation)),
              kind_(parameters.kind),
              slice_updates_{{{parameters.slice_offsets[0], 0}, {parameters.slice_offsets[1], 0}}},
              observes_block_completion(parameters.observes_block_completion),
              uses_sharded_host_retreat(parameters.uses_sharded_host_retreat) {}
        ~Progress() override {
            storage_->occupied[slot_] = false;
        }
        CUresult prepare_launch() noexcept override {
            if (completed_ >= total_blocks_) return CUDA_ERROR_INVALID_VALUE;
            if (kind_ == transform::PtxExecutionKind::kPreemptive) publish_host_retreat(0);
            if (kind_ == transform::PtxExecutionKind::kSliced) {
                slice_updates_[0].value = static_cast<std::uint32_t>(completed_);
                slice_updates_[1].value = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(workers_, total_blocks_ - completed_));
            }
            return CUDA_SUCCESS;
        }
        CUresult request_retreat() noexcept override {
            if (kind_ == transform::PtxExecutionKind::kSliced) return CUDA_SUCCESS;
            if (kind_ != transform::PtxExecutionKind::kPreemptive) return CUDA_ERROR_NOT_SUPPORTED;
            publish_host_retreat(1);
            return CUDA_SUCCESS;
        }
        CUresult retarget_workers(unsigned int workers) noexcept override {
            if (!can_retarget_workers || (kind_ != transform::PtxExecutionKind::kPreemptive &&
                                          kind_ != transform::PtxExecutionKind::kSliced))
                return CUDA_ERROR_NOT_SUPPORTED;
            if (!workers || workers > max_workers || completed_ >= total_blocks_)
                return CUDA_ERROR_INVALID_VALUE;
            workers_ = workers;
            return CUDA_SUCCESS;
        }
        std::span<const CudaExecutionParameterUpdate> parameter_updates() const noexcept override {
            return kind_ == transform::PtxExecutionKind::kSliced
                       ? std::span<const CudaExecutionParameterUpdate>{slice_updates_}
                       : std::span<const CudaExecutionParameterUpdate>{};
        }
        CUresult record_initialization(CUstream stream) noexcept override {
            ++physical_launches_;
            if (!uses_counter()) return CUDA_SUCCESS;
            auto& host = storage_->host[slot_];
            if (!is_initialized_) {
                if (observes_block_completion)
                    std::atomic_ref(host.completed).store(0, std::memory_order_release);
                host.claimed = 0;
                host.zero = 0;
                const auto initialization =
                    std::as_bytes(std::span{&host, 1})
                        .subspan(offsetof(HostSlot, zero), storage_->counter_initialization_bytes);
                const auto status = dispatch_.upload_async(counter(), initialization.data(),
                                                           initialization.size(), stream);
                if (status != CUDA_SUCCESS) return status;
                is_initialized_ = true;
            }
            return CUDA_SUCCESS;
        }
        CUresult record_readback(CUstream stream) noexcept override {
            if (!uses_counter()) return CUDA_SUCCESS;
            return dispatch_.download_async(&storage_->host[slot_].claimed, counter(),
                                            sizeof(std::uint64_t), stream);
        }
        CudaExecutionPersistentStop observe_physical_stop() noexcept override {
            const auto claimed = uses_counter() ? storage_->host[slot_].claimed
                                 : kind_ == transform::PtxExecutionKind::kSliced
                                     ? completed_ + slice_updates_[1].value
                                     : total_blocks_;
            if (claimed < completed_ || claimed > total_blocks_ + workers_)
                return {CUDA_ERROR_UNKNOWN, false};
            if (kind_ == transform::PtxExecutionKind::kDynamic && claimed < total_blocks_)
                return {CUDA_ERROR_UNKNOWN, false};
            completed_ = std::min(claimed, total_blocks_);
            ++physical_stops_;
            return {CUDA_SUCCESS, completed_ == total_blocks_};
        }
        [[nodiscard]] CUdeviceptr counter() const noexcept {
            return storage_->counters + slot_ * storage_->counter_stride;
        }
        [[nodiscard]] CUdeviceptr retreat() const noexcept {
            if (uses_sharded_host_retreat)
                return storage_->mapped + storage_->occupied.size() * sizeof(HostSlot) +
                       slot_ * kGemmSignalSlotBytes;
            return storage_->mapped + slot_ * sizeof(HostSlot) + offsetof(HostSlot, retreat);
        }
        [[nodiscard]] CUdeviceptr block_completion() const noexcept {
            return storage_->mapped + slot_ * sizeof(HostSlot) + offsetof(HostSlot, completed);
        }
        [[nodiscard]] CudaSessionPersistentProgress snapshot() const noexcept {
            const bool completed =
                observes_block_completion && physical_launches_ &&
                std::atomic_ref(storage_->host[slot_].completed).load(std::memory_order_acquire) ==
                    1;
            return {completed_, total_blocks_, physical_launches_,        physical_stops_,
                    workers_,   kind_,         observes_block_completion, completed};
        }

       private:
        void publish_host_retreat(std::uint32_t value) noexcept {
            if (!uses_sharded_host_retreat) {
                std::atomic_ref(storage_->host[slot_].retreat)
                    .store(value, std::memory_order_release);
                return;
            }
            auto* shards = storage_->gemm_signals + slot_ * kGemmSignalShards;
            for (std::size_t index = 0; index < kGemmSignalShards; ++index)
                std::atomic_ref(shards[index].retreat).store(value, std::memory_order_release);
        }
        [[nodiscard]] bool uses_counter() const noexcept {
            return kind_ == transform::PtxExecutionKind::kPreemptive ||
                   kind_ == transform::PtxExecutionKind::kDynamic;
        }
        std::shared_ptr<Storage> storage_;
        CudaSessionDispatch dispatch_;
        std::size_t slot_;

       public:
        const std::uint64_t sequence;

       private:
        std::uint64_t total_blocks_, completed_{}, physical_launches_{}, physical_stops_{};
        unsigned int workers_;
        const unsigned int max_workers;
        const bool can_retarget_workers;
        std::shared_ptr<const void> operation_;
        transform::PtxExecutionKind kind_;
        std::array<CudaExecutionParameterUpdate, 2> slice_updates_;
        const bool observes_block_completion;
        const bool uses_sharded_host_retreat;
        bool is_initialized_{};
    };

    struct DeferredKernel {
        std::shared_ptr<Storage> storage;
        CudaExecutionKernel kernel;
        std::size_t parameter_bytes{};
        std::optional<std::size_t> reserved_slot;
        ExecutionReceipt receipt;
        ~DeferredKernel() {
            if (reserved_slot) storage->occupied[*reserved_slot] = false;
        }
    };

    const ExecutionSessionIdentity identity;
    const std::uint64_t context_id;
    const CudaSessionDispatch dispatch;
    const CudaSessionLimits limits;
    const CudaSessionPersistentConfig persistent;
    const std::shared_ptr<CudaPtxCache> ptx_cache;
    const std::shared_ptr<CudaProfileCache> profile_cache;
    const bool requires_profile_preemption;
    const CudaSessionStreamConfig stream_config;
    CudaProfilePlatform profile_platform;
    const pid_t process{getpid()};
    std::weak_ptr<const void> owner;
    std::shared_ptr<Storage> storage{std::make_shared<Storage>()};
    std::shared_ptr<CudaProcessMemoryBackend> process_memory;
    std::unique_ptr<CudaExecutionGraph> graphs;
    [[nodiscard]] bool has_graph_capture() const noexcept {
        return graphs && graphs->snapshot().has_capture;
    }
    [[nodiscard]] std::optional<Allocation> find_allocation(std::uint64_t id) const noexcept {
        constexpr auto tag = UINT64_C(1) << 63;
        if (process_memory && (id & tag)) {
            const auto found = process_memory->owner()->allocation(id & ~tag);
            if (!found) return std::nullopt;
            return Allocation{id, found->address, found->bytes, 0};
        }
        const auto found = std::ranges::find(storage->allocations, id, &Allocation::id);
        return found == storage->allocations.end() ? std::nullopt : std::optional{*found};
    }
    std::unique_ptr<CudaExecutionDispatcher> executor;
    std::uint64_t next_id;
    std::vector<std::weak_ptr<Progress>> progress;
    std::vector<std::weak_ptr<DeferredKernel>> deferred_kernels;
    std::uint64_t persistent_accepted{};
    std::uint64_t control_epoch{1};
    std::uint64_t next_pair_catalog{1};
    CUdevice native_device{};
    unsigned int sharing_sm_count{};
    std::optional<unsigned int> compute_capability;
    bool has_id_space{true}, has_started{}, has_destroy_attempt{}, is_closing{};
    bool is_creation_unproven{};
    bool has_context_configuration_attempt{};
    // Synchronous metadata calls exclude lifecycle reentry without draining.
    const bool uses_shared_domain;
    const std::shared_ptr<CudaExecutionOwnerActivity> owner_activity;
    bool& is_faulted{owner_activity->is_quarantined};
    bool& is_folded_active{owner_activity->is_folded_active};
    bool& is_rnn_query_active{owner_activity->is_rnn_query_active};
    bool& is_native_issue_active{owner_activity->is_native_issue_active};
    CudaPreparationObserver preparation_observer;
    bool& is_preparation_active{owner_activity->is_preparation_active};
    int device_ordinal{-1};
    // Only the owner mutates these bounded weak registries. Foreign workers
    // observe immutable state and drop shared references, never touch vectors.
    std::vector<std::weak_ptr<const CudaSessionExternalContext::State>> external_contexts;
    std::vector<std::weak_ptr<const CudaSessionExternalCommand::State>> external_commands;
    // Address-only metadata input. Never registered, dereferenced by a query,
    // exposed on the wire or promoted to application allocation authority.
    std::byte pointer_metadata_sentinel{};

    Implementation(ExecutionSessionIdentity identity_value, std::uint64_t context_value,
                   CudaSessionDispatch dispatch_value, CudaSessionLimits limits_value,
                   CudaSessionPersistentConfig persistent_value,
                   std::shared_ptr<CudaPtxCache> cache, std::shared_ptr<CudaProfileCache> profiling,
                   bool requires_preemption, CudaSessionStreamConfig streams,
                   CudaExecutionDomainMember domain_member)
        : identity(identity_value),
          context_id(context_value),
          dispatch(std::move(dispatch_value)),
          limits(limits_value),
          persistent(persistent_value),
          ptx_cache(std::move(cache)),
          profile_cache(session_profile_cache(persistent, std::move(profiling))),
          requires_profile_preemption(requires_preemption && !persistent.observes_block_completion),
          stream_config(session_streams(streams, domain_member)),
          next_id(limits.first_resource_id),
          uses_shared_domain(static_cast<bool>(domain_member)),
          owner_activity(session_activity(domain_member)) {
        switch (persistent.kind) {
            case transform::PtxExecutionKind::kStatic:
            case transform::PtxExecutionKind::kBlockStatic:
            case transform::PtxExecutionKind::kDynamic:
            case transform::PtxExecutionKind::kPreemptive:
            case transform::PtxExecutionKind::kSliced:
                break;
            default:
                throw std::invalid_argument("invalid persistent execution kind");
        }
        if (!persistent.workers && persistent.kind != transform::PtxExecutionKind::kPreemptive)
            throw std::invalid_argument("selected PTX family requires explicit workers");
        if (profile_cache && (!dispatch.device_uuid || !dispatch.driver_version ||
                              (profile_cache->is_learning() && persistent.replace_sgemm)))
            throw std::invalid_argument(
                "profile mode requires platform identity and PTX/native kernels");
        if (requires_profile_preemption &&
            (!profile_cache || profile_cache->is_learning() || !persistent.workers ||
             (persistent.kind != transform::PtxExecutionKind::kPreemptive &&
              persistent.kind != transform::PtxExecutionKind::kSliced)))
            throw std::invalid_argument("learned priority requires a preemptible default");
        if (persistent.uses_workload_agnostic &&
            (!profile_cache || profile_cache->is_learning() || !persistent.workers ||
             persistent.replace_sgemm || requires_profile_preemption ||
             !persistent.sharing_threads_per_sm ||
             persistent.sharing_threads_per_sm >
                 static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
             !persistent.sharing_minimum_speed_percent ||
             persistent.sharing_minimum_speed_percent > 10000 || !dispatch.device_attribute))
            throw std::invalid_argument("sharing requires bounded inference-only resource policy");
        if (!identity.client_id || !identity.server_incarnation || !identity.first_sequence ||
            !context_id || !limits.max_objects ||
            limits.max_objects > std::numeric_limits<std::size_t>::max() - 28 ||
            !limits.max_image_bytes || !limits.max_device_bytes || !limits.max_parameters ||
            limits.max_parameters == std::numeric_limits<std::size_t>::max() || !next_id ||
            !limits.submissions.max_commands || !limits.submissions.max_parameter_bytes ||
            !limits.submissions.max_retained_host_bytes || !dispatch.initialize ||
            !dispatch.device_get || !dispatch.context_create || !dispatch.context_set_current ||
            !dispatch.context_destroy || !dispatch.module_load || !dispatch.module_unload ||
            !dispatch.function_get || !dispatch.parameter_info || !dispatch.allocate ||
            !dispatch.release || !dispatch.upload || !dispatch.download ||
            !dispatch.stream_create || !dispatch.stream_destroy ||
            !dispatch.execution.context_get_current || !dispatch.execution.launch_kernel ||
            !dispatch.execution.event_create || !dispatch.execution.event_record ||
            !dispatch.execution.event_query || !dispatch.execution.event_destroy ||
            !dispatch.execution.stream_synchronize)
            throw std::invalid_argument("invalid CUDA session identity, limits or native dispatch");
        if (stream_config.uses_default_proxy &&
            (!dispatch.stream_wait_event || !stream_config.maximum_staging_bytes))
            throw std::invalid_argument("default-stream proxy requires native event waits");
        if (stream_config.scheduling_priority &&
            (!stream_config.uses_default_proxy || !dispatch.context.priority_range ||
             !dispatch.stream_create_priority || !dispatch.stream_get_priority ||
             persistent.workers || profile_cache))
            throw std::invalid_argument(
                "native scheduling priority requires unprofiled originals, a proxy and checked "
                "priority dispatch");
        if (persistent.uses_sharded_host_retreat &&
            (!persistent.replace_sgemm || !persistent.workers ||
             persistent.kind != transform::PtxExecutionKind::kPreemptive ||
             cuda_sgemm_sharded_image(persistent.gemm_tile_side).empty()))
            throw std::invalid_argument("sharded retreat requires the supported GEMM provider");
        if (persistent.workers &&
            (persistent.workers > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
             limits.max_parameters > std::numeric_limits<std::size_t>::max() -
                                         (persistent.observes_block_completion ? 6 : 5) ||
             limits.submissions.max_commands >
                 limits.max_private_host_bytes /
                     (sizeof(HostSlot) +
                      (persistent.uses_sharded_host_retreat ? kGemmSignalSlotBytes : 0)) ||
             limits.submissions.max_commands >
                 limits.max_device_bytes /
                     (sizeof(std::uint64_t) *
                      (persistent.observes_block_completion && persistent.replace_sgemm ? 2 : 1)) ||
             !dispatch.host_allocate || !dispatch.host_release || !dispatch.host_map ||
             !dispatch.upload_async || !dispatch.download_async || !dispatch.occupancy ||
             !dispatch.function_set_attribute))
            throw std::invalid_argument("invalid persistent session geometry, budget or dispatch");
        if (!cuda_gemm_tile_side_is_valid(persistent.gemm_tile_side) ||
            (!persistent.replace_sgemm && persistent.gemm_tile_side != 64))
            throw std::invalid_argument("GEMM tile side requires supported replacement geometry");
        if (persistent.replace_sgemm &&
            (!persistent.workers || persistent.kind != transform::PtxExecutionKind::kPreemptive ||
             cuda_sgemm_image(persistent.gemm_tile_side).empty() || !dispatch.device_attribute))
            throw std::invalid_argument("persistent GEMM requires workers and compiled provider");
        if (persistent.observes_block_completion &&
            (!persistent.workers || persistent.kind != transform::PtxExecutionKind::kPreemptive ||
             persistent.uses_workload_agnostic))
            throw std::invalid_argument(
                "block completion diagnostics require preemptive workers and no measured profiles");
        if (domain_member) {
            const auto view = domain_member.view();
            if (view.status != CUDA_SUCCESS || view.is_attached ||
                view.logical.context != context_id ||
                view.logical.session.client_id != identity.client_id ||
                view.logical.session.server_incarnation != identity.server_incarnation ||
                view.logical.session.first_sequence != identity.first_sequence)
                throw std::invalid_argument("execution domain member identity mismatch");
            storage->domain_member = std::move(domain_member);
        }
        storage->allocations.reserve(limits.max_objects);
        storage->modules.reserve(limits.max_objects);
        storage->streams.reserve(limits.max_objects);
        storage->events.reserve(limits.max_objects);
        storage->event_generations.reserve(limits.max_objects);
        pending_events.reserve(limits.submissions.max_commands);
        deferred_kernels.reserve(limits.submissions.max_commands);
        storage->blas.reserve(limits.max_objects);
        storage->lt.reserve(limits.max_objects);
        storage->random.reserve(limits.max_objects);
#ifdef GLIMMER_HAS_CUDNN
        storage->dnn.reserve(limits.max_objects);
#endif
        storage->functions.reserve(limits.max_objects);
        storage->host_allocations.reserve(limits.max_objects);
        external_contexts.reserve(limits.max_objects);
        external_commands.reserve(limits.submissions.max_commands);
        if (persistent.workers) {
            storage->occupied.resize(limits.submissions.max_commands, false);
            progress.reserve(limits.submissions.max_commands);
        }
        if (!resource_owner) resource_owner = std::make_shared<int>(0);
        owner = resource_owner;
    }

    [[nodiscard]] bool owns() const noexcept {
        return getpid() == process && resource_owner && owner.lock() == resource_owner;
    }
    template <class Callback, class... Arguments>
    auto invoke(Callback callback, Arguments&&... arguments) {
        if (!uses_shared_domain) return callback(std::forward<Arguments>(arguments)...);
        const NativeExecutionScope scope(is_native_issue_active);
        const auto result = callback(std::forward<Arguments>(arguments)...);
        if (storage->context) {
            auto expected = storage->context;
            if constexpr (std::is_same_v<Callback, decltype(dispatch.context_set_current)>) {
                if (callback == dispatch.context_set_current && result == CUDA_SUCCESS)
                    expected = std::get<0>(std::forward_as_tuple(arguments...));
            }
            CUcontext current{};
            const auto proof = dispatch.execution.context_get_current(&current);
            if (proof != CUDA_SUCCESS || current != expected) fault();
        }
        return result;
    }
    struct SchedulingStreamPriorities {
        CUresult status{};
        int native{}, logical{};
    };
    SchedulingStreamPriorities scheduling_stream_priorities(int requested) noexcept {
        if (!stream_config.scheduling_priority) return {CUDA_ERROR_INVALID_VALUE};
        int least{}, greatest{};
        const auto status = invoke(dispatch.context.priority_range, &least, &greatest);
        const auto proof = check_context(true);
        if (proof != CUDA_SUCCESS) return {proof};
        if (status != CUDA_SUCCESS) return {status};
        if (greatest > least || greatest > 0 || least < 0) {
            fault();
            return {CUDA_ERROR_UNKNOWN};
        }
        // Configured priorities are full-width unsigned values; negating an
        // int conversion would overflow or invert high priorities.
        const auto policy = -static_cast<std::int64_t>(*stream_config.scheduling_priority);
        return {CUDA_SUCCESS, static_cast<int>(std::clamp<std::int64_t>(policy, greatest, least)),
                std::clamp(requested, greatest, least)};
    }
    [[nodiscard]] bool reserve_domain_bytes(std::size_t bytes) noexcept {
        if (!bytes) return true;
        if (process_memory && process_memory->owner()->reserve_context_bytes(bytes) != CUDA_SUCCESS)
            return false;
        if (storage->domain_member &&
            storage->domain_member.reserve_device_bytes(bytes) != CUDA_SUCCESS) {
            if (process_memory &&
                process_memory->owner()->release_context_bytes(bytes) != CUDA_SUCCESS)
                fault();
            return false;
        }
        return true;
    }
    void release_domain_bytes(std::size_t bytes) noexcept {
        if (bytes && process_memory &&
            process_memory->owner()->release_context_bytes(bytes) != CUDA_SUCCESS)
            fault();
        if (bytes && storage->domain_member &&
            storage->domain_member.release_device_bytes(bytes) != CUDA_SUCCESS)
            fault();
    }
    [[nodiscard]] bool clean_shared_allocation_rejection(CUresult status) const noexcept {
        return uses_shared_domain && !is_faulted &&
               (status == CUDA_ERROR_OUT_OF_MEMORY || status == CUDA_ERROR_INVALID_VALUE);
    }
    void update_domain_physical() noexcept {
        if (!storage->domain_member || !executor) return;
        const auto state = executor->snapshot();
        if (state.is_quarantined) fault();
        if (is_faulted) return;
        const bool physical =
            state.issuing || state.awaiting_physical_stop || executor->started_external_count();
        if (storage->domain_member.set_physical_reservation(physical) != CUDA_SUCCESS) fault();
        if (!physical && !is_faulted &&
            storage->domain_member.verify_materialized_configuration(true) != CUDA_SUCCESS)
            fault();
    }
    [[nodiscard]] bool reserve_domain_physical() noexcept {
        return !storage->domain_member ||
               storage->domain_member.set_physical_reservation(true) == CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult domain_cold_status() const noexcept {
        return storage->domain_member ? storage->domain_member.other_members_quiescent()
                                      : CUDA_SUCCESS;
    }
    [[nodiscard]] std::optional<CudaProfileContextConfiguration> profile_context(
        bool owns_native_scope = false) const noexcept {
        if (!has_context_configuration_attempt) {
            return CudaProfileContextConfiguration{};  // Legacy INITIAL-ONLY identity.
        }
        if (!storage->domain_member) return std::nullopt;
        const auto observed = storage->domain_member.configuration(owns_native_scope);
        if (observed.status != CUDA_SUCCESS) return std::nullopt;
        CudaProfileContextConfiguration result{observed.available, observed.values};
        return cuda_profile_context_configuration_valid(result) ? std::optional{result}
                                                                : std::nullopt;
    }
    static FunctionProfileTargets profile_targets(const Function& function) noexcept {
        FunctionProfileTargets result{};
        result[0] = function.native;
        for (const auto& variant : function.variants)
            result[1 + static_cast<std::size_t>(variant.kind)] = variant.native;
        return result;
    }
    std::shared_ptr<CudaExecutionObserver> profile_observation(
        const Function& function, const CudaProfileGeometry& geometry,
        std::shared_ptr<CudaExecutionObserver> observation) {
        observation = configuration_observation(storage->domain_member, geometry.context,
                                                std::move(observation));
        if (!observation || !geometry.function.functions) return observation;
        // Retain native lifetime and owner activity, never a pointer to this
        // Session or a vector element that can move during later admission.
        auto verify = [retained = storage, callbacks = dispatch, activity = owner_activity,
                       targets = profile_targets(function),
                       expected = geometry.function]() noexcept {
            const auto fault = [&] {
                activity->is_quarantined = true;
                if (retained->domain_member) retained->domain_member.quarantine();
            };
            const auto proof = [&] {
                if (!activity->is_native_issue_active || activity->is_quarantined) return false;
                CUcontext current{};
                try {
                    if (callbacks.execution.context_get_current(&current) == CUDA_SUCCESS &&
                        current == retained->context)
                        return true;
                } catch (...) {
                }
                fault();
                return false;
            };
            const auto actual = capture_function_configuration(callbacks, targets,
                                                               expected.cache_preference, proof);
            if (!actual || *actual != expected) {
                fault();
                return false;
            }
            return true;
        };
        return std::make_shared<FunctionConfigurationObservation>(std::move(observation),
                                                                  std::move(verify));
    }
    template <class Callback, class... Arguments>
    CudaExecutionDomainConfigurationResult mutate_domain_configuration(
        CudaExecutionDomainConfigurationMutation mutation, std::uint32_t selector,
        Callback callback, Arguments... arguments) {
        if (control_epoch == UINT64_MAX)
            return {.status = CUDA_ERROR_OUT_OF_MEMORY, .original_status = std::nullopt};
        // The domain owns shared callback exclusion and post-call proof here.
        // Publish the actual requested return before a later proof can fail.
        auto request = [&] { return static_cast<std::int32_t>(callback(arguments...)); };
        const auto invoke_request = [](void* state) {
            return (*static_cast<decltype(request)*>(state))();
        };
        const auto result = storage->domain_member.mutate_configuration(mutation, selector,
                                                                        {&request, invoke_request});
        if (result.original_status) {
            has_context_configuration_attempt = true;
            ++control_epoch;
        }
        if (owner_activity->is_quarantined) fault();
        return result;
    }
    [[nodiscard]] CUresult check_context(bool owns_native_scope = false,
                                         bool allows_enqueue_context_transition = false,
                                         bool requires_exclusive_domain = false) const noexcept {
        if (!owns() || !storage->context || has_destroy_attempt) return CUDA_ERROR_INVALID_CONTEXT;
        if (uses_shared_domain && is_faulted) return CUDA_ERROR_UNKNOWN;
        // Dispatcher issue/retirement owns the native callback boundary. Its
        // own context proof bypasses this session entry gate; reentrant public
        // queries and admissions must not call another SDK while it is active.
        if (!owns_native_scope &&
            (is_native_issue_active ||
             (uses_shared_domain &&
              (is_folded_active || is_rnn_query_active || is_preparation_active))))
            return CUDA_ERROR_NOT_READY;
        if (!allows_enqueue_context_transition && executor && executor->has_active_enqueue_turn())
            return CUDA_ERROR_NOT_READY;
        std::optional<NativeExecutionScope> shared_scope;
        if (uses_shared_domain && !owns_native_scope) shared_scope.emplace(is_native_issue_active);
        if (storage->domain_member) {
            const auto configuration = storage->domain_member.configuration_status(true);
            if (configuration != CUDA_SUCCESS) return configuration;
        }
        CUcontext current{};
        const auto result = dispatch.execution.context_get_current(&current);
        if (result != CUDA_SUCCESS) return result;
        if (current != storage->context) return CUDA_ERROR_INVALID_CONTEXT;
        return requires_exclusive_domain ? domain_cold_status() : CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult external_capture_gate() const noexcept {
        if (!owns()) return CUDA_ERROR_INVALID_CONTEXT;
        if (is_folded_active || is_rnn_query_active || is_native_issue_active ||
            is_preparation_active)
            return CUDA_ERROR_NOT_READY;
        if (is_closing) return CUDA_ERROR_DEINITIALIZED;
        if (!storage->context || has_destroy_attempt) return CUDA_ERROR_INVALID_CONTEXT;
        if (is_faulted || !executor || executor->snapshot().is_quarantined)
            return CUDA_ERROR_UNKNOWN;
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult external_transition_gate() const noexcept {
        if (!owns()) return CUDA_ERROR_INVALID_CONTEXT;
        if (is_folded_active || is_rnn_query_active || is_native_issue_active ||
            is_preparation_active)
            return CUDA_ERROR_NOT_READY;
        if (!storage->context || has_destroy_attempt || !executor)
            return CUDA_ERROR_INVALID_CONTEXT;
        // Closing/fault does not forbid known no-work cancellation or fencing
        // already accepted work. New native entry checks fault independently.
        return CUDA_SUCCESS;
    }
    [[nodiscard]] bool has_external_contexts() const noexcept {
        return std::ranges::any_of(external_contexts,
                                   [](const auto& entry) { return !entry.expired(); });
    }
    [[nodiscard]] bool has_external_allocation(std::uint64_t id) const noexcept {
        return std::ranges::any_of(external_commands, [id](const auto& entry) {
            const auto command = entry.lock();
            return command && std::ranges::any_of(command->ranges, [id](const auto& range) {
                       return range.range.allocation == id;
                   });
        });
    }
    [[nodiscard]] bool has_external_module(std::uint64_t id) const noexcept {
        return std::ranges::any_of(external_commands, [id](const auto& entry) {
            const auto command = entry.lock();
            return command &&
                   std::ranges::find(command->parent_modules, id) != command->parent_modules.end();
        });
    }
    [[nodiscard]] bool has_external_stream(std::uint64_t id) const noexcept {
        return std::ranges::any_of(external_commands, [id](const auto& entry) {
            const auto command = entry.lock();
            return command && command->stream_id == id;
        });
    }
    std::vector<PendingEvent> pending_events;
    [[nodiscard]] bool reserve_static(std::size_t bytes, std::size_t& charge) noexcept {
        if (bytes >
            limits.max_device_bytes - std::min(storage->device_bytes, limits.max_device_bytes))
            return false;
        if (!reserve_domain_bytes(bytes)) return false;
        storage->device_bytes += bytes;
        charge = bytes;
        return true;
    }
    void release_static(std::size_t& charge) noexcept {
        release_domain_bytes(charge);
        storage->device_bytes -= charge;
        charge = 0;
    }
    [[nodiscard]] std::optional<std::size_t> inspect_static_storage(
        std::string_view source,
        std::size_t source_bytes = transform::kMaxCudaModuleBytes) noexcept {
        const PreparationScope phase(preparation_observer, CudaPreparationPhase::kStaticStorage,
                                     is_preparation_active);
        return transform::inspect_ptx_static_storage(source, source_bytes);
    }
    CUresult load_preparation(CUmodule* destination, const void* image, CudaPreparationPhase kind) {
        const PreparationScope phase(preparation_observer, kind, is_preparation_active);
        return invoke(dispatch.module_load, destination, image);
    }
    CUresult attach_jit(Module& module, std::span<const CudaModuleJitOption> options,
                        std::vector<CudaModuleJitOption>* outputs) {
        if (options.empty()) return CUDA_SUCCESS;
        if (!outputs || !CudaModuleJitArguments::accepts(options)) return CUDA_ERROR_INVALID_VALUE;
        if (!dispatch.module_load_ex) return CUDA_ERROR_NOT_SUPPORTED;
        module.jit = CudaModuleJitArguments::create(options);
        if (!module.jit) return CUDA_ERROR_OUT_OF_MEMORY;
        outputs->assign(module.jit->inputs().begin(), module.jit->inputs().end());
        if (profile_cache) {
            // Identity data, never a compiler image. Keep compiler controls out
            // of the pure transform cache, but separate compiled measurements.
            module.profile_source = module.source;
            module.profile_source.append("\0JIT1", 5);
            for (const auto& option : module.jit->inputs()) {
                if (option.option >= 2 && option.option <= 6) continue;
                for (unsigned int byte = 0; byte < 4; ++byte)
                    module.profile_source.push_back(static_cast<char>(option.option >> (8 * byte)));
                for (unsigned int byte = 0; byte < 8; ++byte)
                    module.profile_source.push_back(static_cast<char>(option.value >> (8 * byte)));
                if (option.option == CU_JIT_REFERENCED_KERNEL_NAMES ||
                    option.option == CU_JIT_REFERENCED_VARIABLE_NAMES) {
                    const auto bytes = static_cast<std::uint64_t>(option.data.size());
                    for (unsigned int byte = 0; byte < 8; ++byte)
                        module.profile_source.push_back(static_cast<char>(bytes >> (8 * byte)));
                    module.profile_source.append(reinterpret_cast<const char*>(option.data.data()),
                                                 option.data.size());
                }
            }
            if (module.jit->has_private_references()) {
                module.profile_source.append("\0JITREF1", 8);
                module.profile_source.append(kCudaModuleJitPrivateReferencePattern,
                                             sizeof(kCudaModuleJitPrivateReferencePattern));
            }
        }
        const auto available = limits.max_image_bytes - storage->image_bytes;
        const auto extra = module.jit->storage_bytes() + module.profile_source.size();
        if (module.charged_bytes > available || extra > available - module.charged_bytes)
            return CUDA_ERROR_OUT_OF_MEMORY;
        module.charged_bytes += extra;
        return CUDA_SUCCESS;
    }
    CUresult load_module_preparation(Module& module, CUmodule* destination, const void* image,
                                     CudaPreparationPhase kind,
                                     std::vector<CudaModuleJitOption>* outputs = nullptr) {
        if (!module.jit) return load_preparation(destination, image, kind);
        const PreparationScope phase(preparation_observer, kind, is_preparation_active);
        if (kind == CudaPreparationPhase::kLoadPrepared)
            module.jit->reset_for_private();
        else
            module.jit->reset();
        auto keys = module.jit->native_options();
        auto values = module.jit->native_values();
        const auto result =
            invoke(dispatch.module_load_ex, destination, image,
                   static_cast<unsigned int>(keys.size()), keys.data(), values.data());
        const auto captured = module.jit->capture();
        if (!captured) {
            fault();
            if (outputs) outputs->clear();
            return result;  // Original status remains independent of malformed output.
        }
        if (outputs) {
            for (std::size_t index = 0; index < captured->size(); ++index) {
                (*outputs)[index].value = (*captured)[index].value;
                std::ranges::copy((*captured)[index].data, (*outputs)[index].data.begin());
            }
        }
        return result;
    }
    CUresult transform_source(Module& module) {
        if (has_graph_capture()) return CUDA_SUCCESS;
        if (!persistent.workers || !module.source_static_bytes) return CUDA_SUCCESS;
        const PreparationScope phase(preparation_observer, CudaPreparationPhase::kTransformPtx,
                                     is_preparation_active);
        // This owner resolves stateful binary companions into one module
        // before publication. Legacy separate-module adapters do not opt in.
        constexpr bool shares_original_module = true;
        const std::array default_kind{persistent.kind};
        const auto kinds = profile_cache && !profile_cache->kinds().empty()
                               ? profile_cache->kinds()
                               : std::span<const transform::PtxExecutionKind>{default_kind};
        std::shared_ptr<transform::PersistentPtx> combined;
        for (const auto kind : kinds) {
            auto preparation_limits = kOwnedPtxTransformLimits;
            preparation_limits.observes_block_completion = persistent.observes_block_completion;
            std::shared_ptr<const transform::PersistentPtx> image;
            bool was_prepared{};
            if (ptx_cache) {
                auto prepared = ptx_cache->lookup(module.source, kind, shares_original_module,
                                                  preparation_limits);
                if (prepared.status == CudaPtxCacheStatus::kWrongProcess)
                    return CUDA_ERROR_INVALID_CONTEXT;
                was_prepared = prepared.status == CudaPtxCacheStatus::kReady ||
                               prepared.status == CudaPtxCacheStatus::kUnsupported;
                image = std::move(prepared.image);
            }
            if (!was_prepared)
                if (auto prepared = transform::transform_persistent_ptx(
                        module.source, kind, shares_original_module, preparation_limits))
                    image = std::make_shared<const transform::PersistentPtx>(std::move(*prepared));
            if (!image) continue;
            if (kinds.size() == 1) {
                module.transformed = std::move(image);
                return CUDA_SUCCESS;
            }
            // Generators preserve the original bytes and append distinct private
            // entries. Never duplicate original declarations across families.
            if (!image->image.starts_with(module.source)) return CUDA_ERROR_INVALID_IMAGE;
            if (!combined) {
                combined = std::make_shared<transform::PersistentPtx>(*image);
                continue;
            }
            const auto extra = image->image.size() - module.source.size();
            if (combined->image.size() > kOwnedPtxTransformLimits.output_bytes ||
                extra > kOwnedPtxTransformLimits.output_bytes - combined->image.size())
                return CUDA_ERROR_OUT_OF_MEMORY;
            combined->image.append(image->image, module.source.size(), extra);
            combined->kernels.insert(combined->kernels.end(), image->kernels.begin(),
                                     image->kernels.end());
        }
        module.transformed = std::move(combined);
        return CUDA_SUCCESS;
    }
    [[nodiscard]] bool shared_companion_matches(const Module& module) noexcept {
        const PreparationScope phase(preparation_observer, CudaPreparationPhase::kValidateFunction,
                                     is_preparation_active);
        if (!module.metadata)
            return module.binary.empty() && module.transformed &&
                   module.transformed->image.starts_with(module.source);
        const auto& metadata = *module.metadata;
        if (metadata.is_ptx_selected) return true;
        if (!metadata.has_complete_kernel_metadata || !metadata.has_complete_variable_metadata ||
            metadata.kernels.empty())
            return false;
        for (const auto& kernel : metadata.kernels) {
            const auto count =
                parameter_count(PtxTokens{module.source}, kernel.name, limits.max_parameters);
            if (!count || *count != kernel.arguments.size()) return false;
            CUfunction original{}, candidate{};
            if (invoke(dispatch.function_get, &original, module.native, kernel.name.c_str()) !=
                    CUDA_SUCCESS ||
                is_faulted || !original ||
                invoke(dispatch.function_get, &candidate, module.prepared_native,
                       kernel.name.c_str()) != CUDA_SUCCESS ||
                is_faulted || !candidate)
                return false;
            for (std::size_t index = 0; index < *count; ++index) {
                const auto& expected = kernel.arguments[index];
                for (const auto function : {original, candidate}) {
                    std::size_t offset{}, bytes{};
                    if (invoke(dispatch.parameter_info, function, index, &offset, &bytes) !=
                            CUDA_SUCCESS ||
                        is_faulted || offset != expected.offset || bytes != expected.bytes)
                        return false;
                }
            }
        }
        if (!metadata.variables.empty() && !dispatch.module_global) return false;
        for (const auto& variable : metadata.variables)
            for (const auto native : {module.native, module.prepared_native}) {
                CUdeviceptr address{};
                std::size_t bytes{};
                if (invoke(dispatch.module_global, &address, &bytes, native,
                           variable.name.c_str()) != CUDA_SUCCESS ||
                    is_faulted || !address || bytes != variable.bytes ||
                    bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address)
                    return false;
            }
        return true;
    }
    static void discard_transformation(Module& module) noexcept {
        module.charged_bytes -= module.transformed->image.size() + 1;
        module.transformed.reset();
    }
    CUresult prepare_shared_binary_module(Module& module) noexcept {
        if (has_graph_capture()) return CUDA_SUCCESS;
        if (!module.transformed || (!module.transformed->has_module_variables && !module.jit))
            return CUDA_SUCCESS;
        if (!module.metadata && !module.binary.empty()) return CUDA_ERROR_INVALID_IMAGE;
        module.has_preparation_attempt = true;
        if (!reserve_static(module.prepared_static_budget, module.prepared_static_bytes)) {
            discard_transformation(module);
            return CUDA_SUCCESS;  // Optional preparation must fit its overlap with the original.
        }
        const auto prepared = load_module_preparation(module, &module.prepared_native,
                                                      module.transformed->image.c_str(),
                                                      CudaPreparationPhase::kLoadPrepared);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (prepared != CUDA_SUCCESS && !module.prepared_native) {
            release_static(module.prepared_static_bytes);
            discard_transformation(module);
            return CUDA_SUCCESS;  // Original validated module remains authoritative.
        }
        if (prepared != CUDA_SUCCESS || !module.prepared_native)
            return prepared == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : prepared;
        if (!shared_companion_matches(module)) {
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            const auto released = invoke(dispatch.module_unload, module.prepared_native);
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            if (released != CUDA_SUCCESS) return released;  // Retain both, never retry.
            module.prepared_native = nullptr;
            release_static(module.prepared_static_bytes);
            discard_transformation(module);
            return CUDA_SUCCESS;
        }
        const auto released = invoke(dispatch.module_unload, module.native);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (released != CUDA_SUCCESS) return released;  // Retain both handles, never retry.
        release_static(module.native_static_bytes);
        module.native_static_bytes = std::exchange(module.prepared_static_bytes, 0);
        module.native = module.prepared_native;
        module.prepared_native = nullptr;
        module.shares_original_module = true;
        return CUDA_SUCCESS;
    }
    [[nodiscard]] bool workspace_is_valid(const Blas& handle) const noexcept {
        if (!handle.has_workspace) return true;
        if (!handle.workspace.allocation)
            return !handle.workspace_bytes && !handle.workspace.offset;
        const auto found = find_allocation(handle.workspace.allocation);
        const auto needed = handle.workspace_bytes ? handle.workspace_bytes : 1;
        return found && handle.workspace.offset <= found->bytes &&
               needed <= found->bytes - handle.workspace.offset &&
               handle.workspace.offset <= std::numeric_limits<CUdeviceptr>::max() - found->address;
    }
    [[nodiscard]] CudaSessionResult make_event(
        unsigned int flags, std::shared_ptr<CudaSessionEvent::State>& generation) noexcept {
        if (storage->event_generations.size() == limits.max_objects)
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        try {
            generation = std::make_shared<CudaSessionEvent::State>();
            generation->activity = owner_activity;
            storage->event_generations.push_back(generation);
        } catch (...) {
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        }
        auto result =
            completed(invoke(dispatch.execution.event_create, &generation->native, flags));
        if (result.status != CUDA_SUCCESS && !generation->native) {
            storage->event_generations.pop_back();
            return result;
        }
        if (result.status != CUDA_SUCCESS || !generation->native) {
            generation->has_destroy_attempt =
                true;  // Uncertain factory output, never replay cleanup.
            fault();
            if (result.status == CUDA_SUCCESS) {
                result.status = CUDA_ERROR_UNKNOWN;
                result.disposition = CudaSessionDisposition::kRejected;
            }
        }
        return result;
    }
    [[nodiscard]] CUresult collect_events() noexcept {
        CUresult failure = CUDA_SUCCESS;
        for (auto item = pending_events.begin(); item != pending_events.end();) {
            const auto reply = executor->host_result(item->receipt);
            if (!reply) {
                ++item;
                continue;
            }
            if (!item->call->is_wait) {
                item->call->generation->record_status = static_cast<CUresult>(reply->status);
            }
            if (reply->status && failure == CUDA_SUCCESS)
                failure = static_cast<CUresult>(reply->status);
            if (!executor->acknowledge_result(item->receipt)) {
                fault();
                return CUDA_ERROR_UNKNOWN;
            }
            item = pending_events.erase(item);
        }
        for (auto item = storage->event_generations.begin();
             item != storage->event_generations.end();) {
            auto& generation = **item;
            if (item->use_count() != 1 || generation.has_destroy_attempt) {
                ++item;
                continue;
            }
            generation.has_destroy_attempt = true;
            const auto result = invoke(dispatch.execution.event_destroy, generation.native);
            if (result != CUDA_SUCCESS) {
                fault();
                return result;
            }
            item = storage->event_generations.erase(item);
        }
        return failure;
    }
    [[nodiscard]] CudaSessionEventSubmission submit_event(EventCommand parameters, bool is_wait,
                                                          bool captures_frontier) noexcept {
        const auto sequence = parameters.sequence;
        const auto id = parameters.event;
        const auto stream = parameters.stream;
        const auto flags = parameters.flags;
        const auto failure = [](CUresult status) {
            return CudaSessionEventSubmission{rejected(status), {}};
        };
        if (!owns()) return failure(CUDA_ERROR_INVALID_CONTEXT);
        if (has_graph_capture()) return failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
        if (is_native_issue_active || is_folded_active || is_rnn_query_active)
            return failure(CUDA_ERROR_NOT_READY);
        const NativeExecutionScope scope(is_native_issue_active);
        const auto checked = check_context(true);
        if (checked != CUDA_SUCCESS) return failure(checked);
        if (!executor || is_faulted || is_closing) return failure(CUDA_ERROR_UNKNOWN);
        // This execution domain has no active graph capture. EXTERNAL, as
        // well as unknown bits, is invalid outside capture in CUDA 12.6.
        if (flags) return failure(CUDA_ERROR_INVALID_VALUE);
        if (is_wait && !dispatch.stream_wait_event) return failure(CUDA_ERROR_NOT_SUPPORTED);
        const auto event = std::ranges::find(storage->events, id, &Event::id);
        if (event == storage->events.end() ||
            (stream && !std::ranges::any_of(storage->streams, [stream](const auto& value) {
                 return value.id == stream;
             })))
            return failure(CUDA_ERROR_INVALID_HANDLE);
        const auto cleaned = collect_events();
        if (cleaned != CUDA_SUCCESS) return failure(cleaned);
        if (executor->snapshot().outstanding >= limits.submissions.max_commands) {
            auto deferred = rejected(CUDA_ERROR_NOT_READY);
            deferred.disposition = CudaSessionDisposition::kDeferred;
            return {deferred, {}};
        }
        try {
            auto call = std::make_shared<EventCall>();
            call->dispatch = dispatch;
            call->flags = flags;
            call->is_wait = is_wait;
            if (is_wait)
                call->generation = event->current;
            else {
                auto prepared = make_event(event->flags, call->generation);
                if (prepared.status != CUDA_SUCCESS) {
                    prepared.original_status.reset();  // Factory, not the requested record call.
                    return {prepared, {}};
                }
                call->generation->identity = {id, sequence, 0};
            }
            CudaExecutionEvent command{stream, call, {}};
            if (is_wait) command.dependency = call->generation->frontier;
            auto submitted =
                !is_wait || captures_frontier
                    ? executor->accept_event_with_frontier(sequence, command)
                    : CudaExecutionEventAcceptance{executor->accept_event(sequence, command), {}};
            if (!submitted.acceptance.accepted()) return failure(CUDA_ERROR_OUT_OF_MEMORY);
            pending_events.push_back({submitted.acceptance.receipt, call});
            if (!is_wait) {
                call->generation->identity.frontier_capture = submitted.frontier_capture;
                call->generation->frontier = submitted.frontier;
                event->current = call->generation;
            }
            auto result = settled();
            result.id = sequence;
            return {result,
                    captures_frontier ? std::move(submitted.frontier) : CudaExecutionFrontier{},
                    call->generation->identity};
        } catch (...) {
            return failure(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    [[nodiscard]] Lt* lt_handle(std::uint64_t id) noexcept {
        const auto lt = std::ranges::find(storage->lt, id, &Lt::id);
        if (lt != storage->lt.end()) return &*lt;
        const auto blas = std::ranges::find(storage->blas, id, &Blas::id);
        return blas == storage->blas.end() || blas->is_uncertain || blas->has_destroy_attempt
                   ? nullptr
                   : &blas->lt;
    }
#ifdef GLIMMER_HAS_CUDNN
    struct AttnOperation {
        CudaDnnAttnPreparation preparation;
        CudaDnnAttnAcceptance acceptance;
    };
    [[nodiscard]] AttnOperation observe_attn(CudaExecutionSession& session,
                                             const CudaSessionDnnAttn& problem, bool is_preparation,
                                             std::uint64_t sequence) noexcept;
    // Only callers that already hold the session's scoped native-call guard
    // may use this observer. The public entry acquires that guard itself.
    [[nodiscard]] CudaDnnResult observe_dnn_stream(std::uint64_t id) noexcept {
        if (!owns() || is_faulted) return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
        if (const auto status = check_context(true); status != CUDA_SUCCESS)
            return dnn_gate(rejected(status));
        const auto found = std::ranges::find(storage->dnn, id, &Dnn::id);
        if (found == storage->dnn.end()) return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
        if (found->is_uncertain || found->has_destroy_attempt)
            return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
        CUstream expected = storage->default_proxy;
        if (found->stream) {
            const auto owned = std::ranges::find(storage->streams, found->stream, &Stream::id);
            if (owned == storage->streams.end()) {
                fault();
                return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
            }
            expected = owned->native;
        }
        cudaStream_t observed{};
        auto result = dnn_completed(invoke(dispatch.dnn->get_stream, found->native, &observed));
        if (check_context(true) != CUDA_SUCCESS ||
            (result.original_status == CUDNN_STATUS_SUCCESS &&
             reinterpret_cast<CUstream>(observed) != expected)) {
            fault();
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        } else if (result.original_status == CUDNN_STATUS_SUCCESS)
            result.value = found->stream;
        return result;
    }
#endif
    [[nodiscard]] CudaSessionResult cold_gate(
        bool is_creation = false, bool allows_graph_preparation = false) const noexcept {
        if (graphs && graphs->snapshot().is_unknown) return rejected(CUDA_ERROR_UNKNOWN);
        if (has_graph_capture() && !allows_graph_preparation)
            return rejected(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
        if (process_memory) {
            const auto health = process_memory->native_health();
            if (health != CUDA_SUCCESS) return rejected(health);
        }
        if (owns() && (is_folded_active || is_rnn_query_active || is_native_issue_active)) {
            auto deferred = rejected(CUDA_ERROR_NOT_READY);
            deferred.disposition = CudaSessionDisposition::kDeferred;
            return deferred;
        }
        const auto result = check_context();
        if (result != CUDA_SUCCESS) return rejected(result);
        if (is_creation && is_closing) return rejected(CUDA_ERROR_DEINITIALIZED);
        if (!executor || is_faulted || executor->snapshot().is_quarantined)
            return rejected(CUDA_ERROR_UNKNOWN);
        if (!executor->resources_quiescent() || domain_cold_status() == CUDA_ERROR_NOT_READY) {
            auto deferred = rejected(CUDA_ERROR_NOT_READY);
            deferred.disposition = CudaSessionDisposition::kDeferred;
            return deferred;
        }
        if (domain_cold_status() != CUDA_SUCCESS) return rejected(domain_cold_status());
        return settled();
    }
    void fault() noexcept {
        is_faulted = true;
        if (storage->domain_member) storage->domain_member.quarantine();
        if (executor) executor->close();
    }
    [[nodiscard]] CudaSessionResult release_host_at(std::size_t index) noexcept {
        auto& allocation = storage->host_allocations[index];
        allocation.is_retiring = true;
        if (allocation.is_uncertain || allocation.has_unregister_attempt)
            return rejected(CUDA_ERROR_UNKNOWN);
        if (allocation.mapping.use_count() != 1) {
            auto result = rejected(CUDA_ERROR_NOT_READY);
            result.disposition = CudaSessionDisposition::kDeferred;
            return result;
        }
        auto result = settled();
        if (!allocation.is_pageable) {
            allocation.has_unregister_attempt = true;
            result = completed(invoke(dispatch.host_unregister, allocation.mapping->pointer));
        }
        if (result.status != CUDA_SUCCESS) {
            allocation.is_uncertain = true;
            fault();
            return result;
        }
        allocation.mapping->can_unmap = true;
        if (::munmap(allocation.mapping->pointer, allocation.mapping->bytes) != 0) {
            allocation.mapping->can_unmap = false;
            allocation.is_uncertain = true;
            fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        allocation.mapping->pointer = MAP_FAILED;
        storage->host_memory_bytes -= allocation.mapping->bytes;
        storage->host_allocations.erase(storage->host_allocations.begin() +
                                        static_cast<std::ptrdiff_t>(index));
        return result;
    }
    [[nodiscard]] std::uint64_t consume_id() noexcept {
        const auto id = next_id;
        if (id == std::numeric_limits<std::uint64_t>::max() ||
            (process_memory && id == (UINT64_C(1) << 63) - 1))
            has_id_space = false;
        else
            ++next_id;
        return id;
    }

    [[nodiscard]] CUresult initialize_progress() noexcept {
        if (!persistent.workers) return CUDA_SUCCESS;
        storage->counter_initialization_bytes =
            sizeof(std::uint64_t) *
            (persistent.observes_block_completion && persistent.replace_sgemm ? 2 : 1);
        storage->counter_stride = storage->counter_initialization_bytes;
        const auto counter_bytes = storage->occupied.size() * storage->counter_stride;
        if (!reserve_domain_bytes(counter_bytes)) return CUDA_ERROR_OUT_OF_MEMORY;
        auto status = invoke(dispatch.allocate, &storage->counters, counter_bytes);
        const bool clean_rejection =
            status != CUDA_SUCCESS && !storage->counters &&
            (!uses_shared_domain || clean_shared_allocation_rejection(status));
        if (clean_rejection) release_domain_bytes(counter_bytes);
        if (!clean_rejection) {
            storage->counter_bytes = counter_bytes;
            storage->device_bytes += counter_bytes;
        }
        if (is_faulted) {
            storage->is_counter_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS || !storage->counters ||
            counter_bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - storage->counters ||
            storage->counters % alignof(std::uint64_t)) {
            storage->is_counter_uncertain = !clean_rejection;
            return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        }
        const auto slot_count = storage->occupied.size();
        const auto base_host_bytes = slot_count * sizeof(HostSlot);
        const auto host_bytes =
            base_host_bytes +
            (persistent.uses_sharded_host_retreat ? slot_count * kGemmSignalSlotBytes : 0);
        status = invoke(dispatch.host_allocate, reinterpret_cast<void**>(&storage->host),
                        host_bytes, CU_MEMHOSTALLOC_DEVICEMAP);
        if (status == CUDA_SUCCESS || storage->host) storage->host_bytes = host_bytes;
        if (is_faulted) {
            storage->is_host_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS || !storage->host ||
            reinterpret_cast<std::uintptr_t>(storage->host) % alignof(HostSlot)) {
            storage->is_host_uncertain =
                status == CUDA_SUCCESS || storage->host ||
                (uses_shared_domain && !clean_shared_allocation_rejection(status));
            return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        }
        for (std::size_t index = 0; index < storage->occupied.size(); ++index)
            std::construct_at(storage->host + index);
        if (persistent.uses_sharded_host_retreat) {
            storage->gemm_signals = reinterpret_cast<GemmSignalShard*>(
                reinterpret_cast<std::byte*>(storage->host) + base_host_bytes);
            for (std::size_t index = 0; index < slot_count * kGemmSignalShards; ++index)
                std::construct_at(storage->gemm_signals + index);
        }
        status = invoke(dispatch.host_map, &storage->mapped, storage->host, 0);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (status != CUDA_SUCCESS || !storage->mapped ||
            host_bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - storage->mapped ||
            storage->mapped % alignof(HostSlot))
            return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        storage->is_progress_ready = true;
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult release_progress() noexcept {
        if (storage->is_host_uncertain ||
            std::ranges::any_of(storage->occupied, [](bool used) { return used; }))
            return CUDA_ERROR_UNKNOWN;
        if (storage->host) {
            if (storage->has_host_free_attempt) return CUDA_ERROR_UNKNOWN;
            storage->has_host_free_attempt = true;
            const auto status = invoke(dispatch.host_release, storage->host);
            if (status != CUDA_SUCCESS) return status;
            if (uses_shared_domain && is_faulted) return CUDA_ERROR_UNKNOWN;
            storage->host = nullptr;
            storage->gemm_signals = nullptr;
            storage->host_bytes = 0;
        }
        if (storage->counters && !storage->is_counter_uncertain) {
            if (storage->has_counter_free_attempt) return CUDA_ERROR_UNKNOWN;
            storage->has_counter_free_attempt = true;
            const auto status = invoke(dispatch.release, storage->counters);
            if (status != CUDA_SUCCESS) return status;
            if (uses_shared_domain && is_faulted) return CUDA_ERROR_UNKNOWN;
            storage->counters = 0;
            release_domain_bytes(storage->counter_bytes);
            storage->device_bytes -= storage->counter_bytes;
            storage->counter_bytes = 0;
        }
        storage->is_progress_ready = false;
        return CUDA_SUCCESS;
    }

    [[nodiscard]] bool rollback_shared_initialization() noexcept {
        if (!uses_shared_domain || is_faulted || storage->is_counter_uncertain ||
            storage->is_host_uncertain || storage->is_default_proxy_uncertain)
            return false;
        if (release_progress() != CUDA_SUCCESS || is_faulted) return false;
        if (storage->default_proxy) {
            storage->has_default_proxy_destroy_attempt = true;
            const auto status = invoke(dispatch.stream_destroy, storage->default_proxy);
            if (status != CUDA_SUCCESS || is_faulted) return false;
            storage->default_proxy = nullptr;
        }
        const auto status = invoke(dispatch.context_set_current, nullptr);
        if (status != CUDA_SUCCESS || is_faulted) return false;
        storage->context = nullptr;
        return true;
    }

    [[nodiscard]] CUresult release_shared_resources() noexcept {
        const NativeExecutionScope scope(is_native_issue_active);
        // The caller has proved own dispatcher retirement, no exported resource
        // owners and no other member's physical work. Any ambiguous native
        // cleanup poisons the entire domain before a later cleanup can retry it.
        const auto failed = [&](CUresult result) {
            fault();
            return result;
        };
        storage->events.clear();
        pending_events.clear();
        for (auto& generation : storage->event_generations) {
            if (generation.use_count() != 1 || generation->has_destroy_attempt)
                return failed(CUDA_ERROR_UNKNOWN);
            generation->has_destroy_attempt = true;
            const auto status = invoke(dispatch.execution.event_destroy, generation->native);
            if (status != CUDA_SUCCESS || is_faulted)
                return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            generation->native = nullptr;
        }
        storage->event_generations.clear();
        for (auto item = storage->allocations.begin(); item != storage->allocations.end();) {
            if (item->module) {
                ++item;
                continue;
            }
            const auto status = invoke(dispatch.release, item->address);
            if (status != CUDA_SUCCESS || is_faulted)
                return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            storage->device_bytes -= item->bytes;
            release_domain_bytes(item->bytes);
            item = storage->allocations.erase(item);
        }
        for (auto& module : storage->modules) {
            if (module.prepared_native) {
                const auto status = invoke(dispatch.module_unload, module.prepared_native);
                if (status != CUDA_SUCCESS || is_faulted)
                    return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
                module.prepared_native = nullptr;
                release_static(module.prepared_static_bytes);
            }
            const auto status = invoke(dispatch.module_unload, module.native);
            if (status != CUDA_SUCCESS || is_faulted)
                return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            module.native = nullptr;
            release_static(module.native_static_bytes);
        }
        storage->modules.clear();
        storage->allocations.clear();
        storage->functions.clear();
        storage->image_bytes = 0;
        if (storage->gemm_module) {
            const auto status = invoke(dispatch.module_unload, storage->gemm_module);
            if (status != CUDA_SUCCESS || is_faulted)
                return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            storage->gemm_module = nullptr;
            release_static(storage->gemm_static_bytes);
        }
        for (auto& stream : storage->streams) {
            const auto status = invoke(dispatch.stream_destroy, stream.native);
            if (status != CUDA_SUCCESS || is_faulted)
                return failed(status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            stream.native = nullptr;
        }
        storage->streams.clear();
        if (storage->device_bytes || is_faulted) return failed(CUDA_ERROR_UNKNOWN);
        return CUDA_SUCCESS;
    }

    [[nodiscard]] CUresult prepare_sgemm() noexcept {
        if (!persistent.replace_sgemm || storage->gemm_ids.back()) return CUDA_SUCCESS;
        if (storage->has_gemm_preparation_attempt) return CUDA_ERROR_UNKNOWN;
        const auto image = persistent.uses_sharded_host_retreat
                               ? cuda_sgemm_sharded_image(persistent.gemm_tile_side)
                               : cuda_sgemm_image(persistent.gemm_tile_side);
        if (image.size() > limits.max_image_bytes - storage->image_bytes || !has_id_space ||
            next_id > std::numeric_limits<std::uint64_t>::max() - 25)
            return CUDA_ERROR_OUT_OF_MEMORY;
        int major{}, minor{};
        auto status = invoke(dispatch.device_attribute, &major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, native_device);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (status == CUDA_SUCCESS)
            status = invoke(dispatch.device_attribute, &minor,
                            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, native_device);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (status != CUDA_SUCCESS) return status;
        if (major != 7 || minor != 0) return CUDA_ERROR_NOT_SUPPORTED;
        // The compiled provider image never changes. Cache only pure CPU
        // inventory, not context-bound SDK handles or session reservations.
        const auto static_bytes = [](unsigned int side, bool sharded) {
            if (sharded) {
                static const auto bytes =
                    transform::inspect_ptx_static_storage(cuda_sgemm_sharded_image(32));
                return bytes;
            }
            if (side == 32) {
                static const auto bytes =
                    transform::inspect_ptx_static_storage(cuda_sgemm_image(32));
                return bytes;
            }
            static const auto bytes = transform::inspect_ptx_static_storage(cuda_sgemm_image(64));
            return bytes;
        }(persistent.gemm_tile_side, persistent.uses_sharded_host_retreat);
        if (!static_bytes) return CUDA_ERROR_NOT_SUPPORTED;
        if (!reserve_static(*static_bytes, storage->gemm_static_bytes))
            return CUDA_ERROR_OUT_OF_MEMORY;
        storage->has_gemm_preparation_attempt = true;
        status = invoke(dispatch.module_load, &storage->gemm_module, image.data());
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (status != CUDA_SUCCESS && !storage->gemm_module)
            release_static(storage->gemm_static_bytes);
        // A partial/unknown load is retained and never retried, like user modules.
        storage->image_bytes += image.size();
        if (status != CUDA_SUCCESS || !storage->gemm_module) {
            fault();
            return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        }
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
        constexpr std::array<std::size_t, 20> offsets{0,  8,  16, 24, 28, 32, 36, 40,  44,  48,
                                                      52, 56, 64, 72, 80, 88, 96, 104, 112, 120};
        constexpr std::array<std::size_t, 20> sizes{8, 8, 8, 4, 4, 4, 4, 4, 4, 4,
                                                    4, 8, 8, 4, 8, 8, 8, 8, 8, 4};
        for (std::size_t index = 0; index < names.size(); ++index) {
            auto& function = storage->gemm_functions[index];
            const auto name = std::string(names[index]) +
                              (persistent.observes_block_completion ? "_observed" : "");
            status = invoke(dispatch.function_get, &function, storage->gemm_module, name.c_str());
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            if (status == CUDA_SUCCESS && !function) status = CUDA_ERROR_UNKNOWN;
            const bool is_split = index == names.size() - 1;
            const std::size_t ordinary_arity = is_split ? 17 : index < 12 ? 16 : 20;
            const auto arity = ordinary_arity + (persistent.observes_block_completion ? 1 : 0);
            for (std::size_t argument = 0; status == CUDA_SUCCESS && argument < arity; ++argument) {
                std::size_t offset{}, bytes{};
                status = invoke(dispatch.parameter_info, function, argument, &offset, &bytes);
                if (is_faulted) return CUDA_ERROR_UNKNOWN;
                const bool is_completion = argument == ordinary_arity;
                const auto expected_offset = is_completion ? (is_split     ? 104
                                                              : index < 12 ? 96
                                                                           : 128)
                                                           : offsets[argument];
                const auto expected_bytes = is_completion                ? 8
                                            : is_split && argument == 16 ? 4
                                                                         : sizes[argument];
                if (status == CUDA_SUCCESS &&
                    (offset != expected_offset || bytes != expected_bytes))
                    status = CUDA_ERROR_INVALID_VALUE;
            }
            if (status == CUDA_SUCCESS)
                status = invoke(dispatch.function_set_attribute, function,
                                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            const auto id = consume_id();
            if (status != CUDA_SUCCESS || !executor->bind_function(id, function, 0, storage)) {
                fault();
                return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
            }
            storage->gemm_ids[index] = id;
        }
        return CUDA_SUCCESS;
    }

    [[nodiscard]] ExecutionAcceptance accept_gemm(std::uint64_t sequence,
                                                  const CudaSessionSgemm& problem,
                                                  const CudaSessionGemmEx* extended,
                                                  const CudaSessionStridedGemmEx* batch = nullptr,
                                                  bool uses_sgemm_batch = false) noexcept;

    [[nodiscard]] CUresult prepare_lt() noexcept {
        if (!persistent.replace_sgemm || storage->lt_ids.back()) return CUDA_SUCCESS;
        auto status = prepare_sgemm();
        if (status != CUDA_SUCCESS) return status;
        if (!has_id_space || next_id > std::numeric_limits<std::uint64_t>::max() - 3)
            return CUDA_ERROR_OUT_OF_MEMORY;
        constexpr std::array names{"glimmer_owned_lt_sgemm", "glimmer_owned_lt_hgemm",
                                   "glimmer_owned_lt_hsgemm"};
        constexpr std::array<std::size_t, 28> offsets{
            0,  8,  16, 24, 32,  40,  44,  48,  52,  56,  60,  64,  68,  72,
            76, 80, 88, 96, 100, 104, 112, 120, 128, 136, 144, 152, 160, 164};
        constexpr std::array<std::size_t, 28> sizes{8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                    4, 8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 4, 4};
        for (std::size_t index = 0; index < names.size(); ++index) {
            auto& function = storage->lt_functions[index];
            const auto name = std::string(names[index]) +
                              (persistent.observes_block_completion ? "_observed" : "");
            status = invoke(dispatch.function_get, &function, storage->gemm_module, name.c_str());
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            if (status == CUDA_SUCCESS && !function) status = CUDA_ERROR_UNKNOWN;
            const auto arity = offsets.size() + (persistent.observes_block_completion ? 1 : 0);
            for (std::size_t argument = 0; status == CUDA_SUCCESS && argument < arity; ++argument) {
                std::size_t offset{}, bytes{};
                status = invoke(dispatch.parameter_info, function, argument, &offset, &bytes);
                if (is_faulted) return CUDA_ERROR_UNKNOWN;
                const bool is_completion = argument == offsets.size();
                if (status == CUDA_SUCCESS &&
                    (offset != (is_completion ? 168 : offsets[argument]) ||
                     bytes != (is_completion ? 8 : sizes[argument])))
                    status = CUDA_ERROR_INVALID_VALUE;
            }
            if (status == CUDA_SUCCESS)
                status = invoke(dispatch.function_set_attribute, function,
                                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            const auto id = consume_id();
            if (status != CUDA_SUCCESS || !executor->bind_function(id, function, 0, storage)) {
                fault();
                return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
            }
            storage->lt_ids[index] = id;
        }
        return CUDA_SUCCESS;
    }

    class GeamCall;
    template <typename Problem>
    class BlasExtensionCall;
    template <typename Problem>
    [[nodiscard]] ExecutionAcceptance accept_blas_extension(std::uint64_t sequence,
                                                            const Problem& problem) noexcept;

    [[nodiscard]] CUresult prepare_persistent_function(Function& function,
                                                       const transform::PtxKernelVariant& variant) {
        if (variant.parameter_count != function.arguments.size() ||
            variant.parameter_layout.size() != function.arguments.size())
            return CUDA_ERROR_INVALID_VALUE;
        PrivateFunction prepared;
        prepared.kind = variant.execution_kind;
        prepared.observes_block_completion = variant.observes_block_completion;
        if (prepared.observes_block_completion != persistent.observes_block_completion ||
            (prepared.observes_block_completion &&
             prepared.kind != transform::PtxExecutionKind::kPreemptive))
            return CUDA_ERROR_INVALID_VALUE;
        const auto module = std::ranges::find_if(
            storage->modules, [&](const auto& entry) { return entry.id == function.module; });
        if (module == storage->modules.end()) return CUDA_ERROR_INVALID_HANDLE;
        const auto native = module->binary.empty() || module->shares_original_module
                                ? module->native
                                : module->prepared_native;
        if (!native) return CUDA_ERROR_INVALID_HANDLE;
        const auto status = invoke(dispatch.function_get, &prepared.native, native,
                                   variant.persistent_name.c_str());
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (status != CUDA_SUCCESS) return status;
        if (!prepared.native) return CUDA_ERROR_UNKNOWN;
        const auto configured = invoke(dispatch.function_set_attribute, prepared.native,
                                       CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                       CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
        if (is_faulted) return CUDA_ERROR_UNKNOWN;
        if (configured != CUDA_SUCCESS) return configured;
        const auto kind = variant.execution_kind;
        const std::size_t appended = kind == transform::PtxExecutionKind::kStatic        ? 1
                                     : kind == transform::PtxExecutionKind::kBlockStatic ? 3
                                     : kind == transform::PtxExecutionKind::kDynamic     ? 4
                                     : prepared.observes_block_completion                ? 6
                                                                                         : 5;
        prepared.arguments.reserve(function.arguments.size() + appended);
        for (std::size_t index = 0; index < function.arguments.size() + appended; ++index) {
            std::size_t offset{}, bytes{};
            const auto queried =
                invoke(dispatch.parameter_info, prepared.native, index, &offset, &bytes);
            if (is_faulted) return CUDA_ERROR_UNKNOWN;
            if (queried != CUDA_SUCCESS) return queried;
            if (!bytes || offset < prepared.bytes ||
                offset > limits.submissions.max_parameter_bytes ||
                bytes > limits.submissions.max_parameter_bytes - offset)
                return CUDA_ERROR_INVALID_VALUE;
            if (index < function.arguments.size()) {
                const auto& original = function.arguments[index];
                const auto& layout = variant.parameter_layout[index];
                if (offset != original.offset || bytes != original.bytes ||
                    bytes != layout.size_bytes || !layout.alignment_bytes ||
                    offset % layout.alignment_bytes)
                    return CUDA_ERROR_INVALID_VALUE;
            } else if (bytes != (index - function.arguments.size() < 3 ||
                                         kind == transform::PtxExecutionKind::kSliced
                                     ? 4U
                                     : 8U)) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            prepared.arguments.push_back({offset, bytes});
            prepared.bytes = offset + bytes;
        }
        function.variants.push_back(std::move(prepared));
        return CUDA_SUCCESS;
    }
};

class CudaExecutionSession::Implementation::GeamCall final : public CudaExecutionLibraryCall {
   public:
    GeamCall(Implementation& owner, CudaSessionGeam problem, std::array<CUdeviceptr, 5> addresses,
             std::array<bool, 2> safe_inputs, bool stages_scalars, bool requires_version)
        : owner_(owner),
          storage_(owner.storage),
          problem_(problem),
          addresses_(addresses),
          safe_inputs_(safe_inputs),
          stages_scalars_(stages_scalars),
          requires_version_(requires_version) {}

    std::int32_t issue(CUstream stream, std::span<const std::byte>) noexcept override {
        auto found = std::ranges::find(storage_->blas, problem_.handle, &Blas::id);
        if (found == storage_->blas.end() || found->is_uncertain || found->has_destroy_attempt) {
            fail(CUDA_ERROR_INVALID_HANDLE, true);
            return 0;
        }
        handle_ = &*found;
        if (!observe_state(stream, problem_.pointer_mode, false)) return 0;
        const auto& sdk = owner_.dispatch.blas;
        if (requires_version_) {
            int version{};
            const auto status = sdk.get_version(handle_->native, &version);
            if (!prove_context()) return 0;
            if (status != CUBLAS_STATUS_SUCCESS) {
                fail(CUDA_ERROR_UNKNOWN, false);
                return 0;
            }
            if (version != 120604) {
                fail(CUDA_ERROR_NOT_SUPPORTED, false, true);
                return 0;
            }
            if (!observe_state(stream, problem_.pointer_mode, false)) return 0;
        }
        if (stages_scalars_) {
            const auto synchronized = owner_.dispatch.execution.stream_synchronize(stream);
            if (!prove_context()) return 0;
            if (synchronized != CUDA_SUCCESS) {
                fail(synchronized, true);
                return 0;
            }
            const auto width = problem_.type == CUDA_R_32F ? sizeof(float) : sizeof(double);
            const auto copied_alpha =
                owner_.dispatch.download(problem_.alpha.data(), addresses_[3], width);
            if (!prove_context()) return 0;
            if (copied_alpha != CUDA_SUCCESS) {
                fail(copied_alpha, true);
                return 0;
            }
            const auto copied_beta =
                owner_.dispatch.download(problem_.beta.data(), addresses_[4], width);
            if (!prove_context()) return 0;
            if (copied_beta != CUDA_SUCCESS) {
                fail(copied_beta, true);
                return 0;
            }
            if (!observe_state(stream, problem_.pointer_mode, true)) return 0;
            if ((!safe_inputs_[0] && !geam_scalar_is_zero(problem_.type, problem_.alpha)) ||
                (!safe_inputs_[1] && !geam_scalar_is_zero(problem_.type, problem_.beta))) {
                fail(CUDA_ERROR_INVALID_VALUE, false, true);
                return 0;
            }
            const auto changed = sdk.set_pointer_mode(handle_->native, CUBLAS_POINTER_MODE_HOST);
            if (!prove_context()) return 0;
            if (changed != CUBLAS_STATUS_SUCCESS) {
                fail(CUDA_ERROR_UNKNOWN, true);
                return 0;
            }
            // A successful setter alone is not authority to interpret device
            // coefficients as host pointers. Prove the temporary native state.
            if (!observe_state(stream, CUBLAS_POINTER_MODE_HOST, true)) return 0;
        }
        // NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK device address ABI.
        const auto scalar = [&](std::size_t index) -> const void* {
            if (!(problem_.presence & (1U << index))) return nullptr;
            if (problem_.pointer_mode == CUBLAS_POINTER_MODE_DEVICE && !stages_scalars_)
                return reinterpret_cast<const void*>(addresses_[index + 3]);
            return index == 0 ? problem_.alpha.data() : problem_.beta.data();
        };
        const auto op_a = static_cast<cublasOperation_t>(problem_.transpose_a);
        const auto op_b = static_cast<cublasOperation_t>(problem_.transpose_b);
        completion_.was_native_issued = true;
        const auto status =
            problem_.type == CUDA_R_32F
                ? sdk.sgeam(handle_->native, op_a, op_b, problem_.rows, problem_.columns,
                            static_cast<const float*>(scalar(0)),
                            reinterpret_cast<const float*>(addresses_[0]), problem_.leading_a,
                            static_cast<const float*>(scalar(1)),
                            reinterpret_cast<const float*>(addresses_[1]), problem_.leading_b,
                            reinterpret_cast<float*>(addresses_[2]), problem_.leading_c)
                : sdk.dgeam(handle_->native, op_a, op_b, problem_.rows, problem_.columns,
                            static_cast<const double*>(scalar(0)),
                            reinterpret_cast<const double*>(addresses_[0]), problem_.leading_a,
                            static_cast<const double*>(scalar(1)),
                            reinterpret_cast<const double*>(addresses_[1]), problem_.leading_b,
                            reinterpret_cast<double*>(addresses_[2]), problem_.leading_c);
        // NOLINTEND(performance-no-int-to-ptr)
        if (!prove_context()) return status;
        if (stages_scalars_) {
            const auto restored = sdk.set_pointer_mode(handle_->native, CUBLAS_POINTER_MODE_DEVICE);
            if (!prove_context()) return status;
            if (restored != CUBLAS_STATUS_SUCCESS) {
                fail(CUDA_ERROR_UNKNOWN, true);
                return status;
            }
        }
        (void)observe_state(stream, problem_.pointer_mode, true);
        return status;
    }

    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    void fail(CUresult status, bool uncertain, bool clean = false) noexcept {
        completion_.status = status;
        completion_.is_cleanup_uncertain = uncertain;
        completion_.is_clean_preparation_rejection = clean;
        if (uncertain) {
            if (handle_) handle_->is_uncertain = true;
            owner_.fault();
        }
    }
    bool prove_context() noexcept {
        const auto status = owner_.check_context(true);
        if (status == CUDA_SUCCESS) return true;
        fail(status, true);
        return false;
    }
    bool observe_state(CUstream expected_stream, std::uint32_t expected_mode,
                       bool has_possible_effects) noexcept {
        if (!prove_context()) return false;
        CUstream stream{};
        const auto stream_status = owner_.dispatch.blas.get_stream(handle_->native, &stream);
        if (!prove_context()) return false;
        if (stream_status != CUBLAS_STATUS_SUCCESS) {
            fail(CUDA_ERROR_UNKNOWN, has_possible_effects);
            return false;
        }
        if (stream != expected_stream) {
            fail(CUDA_ERROR_INVALID_HANDLE, true);
            return false;
        }
        cublasPointerMode_t mode{};
        const auto mode_status = owner_.dispatch.blas.get_pointer_mode(handle_->native, &mode);
        if (!prove_context()) return false;
        if (mode_status != CUBLAS_STATUS_SUCCESS) {
            fail(CUDA_ERROR_UNKNOWN, has_possible_effects);
            return false;
        }
        std::uint32_t raw_mode{};
        static_assert(sizeof(raw_mode) == sizeof(mode));
        std::memcpy(&raw_mode, &mode, sizeof(raw_mode));
        if (raw_mode != expected_mode) {
            fail(CUDA_ERROR_INVALID_VALUE, true);
            return false;
        }
        return true;
    }
    Implementation& owner_;
    std::shared_ptr<Storage> storage_;
    CudaSessionGeam problem_;
    std::array<CUdeviceptr, 5> addresses_;
    std::array<bool, 2> safe_inputs_;
    bool stages_scalars_, requires_version_;
    Blas* handle_{};
    CudaExecutionLibraryCompletion completion_{.was_native_issued = false};
};

// These two native-only APIs share ownership/state proof, not the GEMM private
// selector or another public SDK entry. The copied request owns host scalars.
template <typename Problem>
class CudaExecutionSession::Implementation::BlasExtensionCall final
    : public CudaExecutionLibraryCall {
   public:
    BlasExtensionCall(Implementation& owner, Problem problem, std::array<CUdeviceptr, 5> addresses,
                      bool requires_version) noexcept
        : owner_(owner),
          storage_(owner.storage),
          problem_(problem),
          addresses_(addresses),
          requires_version_(requires_version) {}

    std::int32_t issue(CUstream stream, std::span<const std::byte>) noexcept override {
        const auto found = std::ranges::find(storage_->blas, problem_.handle, &Blas::id);
        if (found == storage_->blas.end() || found->is_uncertain || found->has_destroy_attempt) {
            fail(CUDA_ERROR_INVALID_HANDLE, true);
            return 0;
        }
        handle_ = &*found;
        if (!observe_state(stream, false)) return 0;
        const auto& sdk = owner_.dispatch.blas;
        if (requires_version_) {
            int version{};
            const auto status = sdk.get_version(handle_->native, &version);
            if (!prove_context() || !observe_state(stream, false)) return 0;
            if (status != CUBLAS_STATUS_SUCCESS) {
                fail(CUDA_ERROR_UNKNOWN, false);
                return 0;
            }
            if (version != 120604) {
                fail(CUDA_ERROR_NOT_SUPPORTED, false, true);
                return 0;
            }
        }
        // NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK device-address ABI.
        const auto* alpha = !(problem_.presence & 1U) ? nullptr
                            : problem_.pointer_mode == CUBLAS_POINTER_MODE_DEVICE
                                ? reinterpret_cast<const float*>(addresses_[3])
                                : reinterpret_cast<const float*>(problem_.alpha.data());
        const auto* beta = !(problem_.presence & 2U) ? nullptr
                           : problem_.pointer_mode == CUBLAS_POINTER_MODE_DEVICE
                               ? reinterpret_cast<const float*>(addresses_[4])
                               : reinterpret_cast<const float*>(problem_.beta.data());
        completion_.was_native_issued = true;
        cublasStatus_t status{};
        if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
            status = sdk.sgemv(handle_->native, static_cast<cublasOperation_t>(problem_.transpose),
                               problem_.rows, problem_.columns, alpha,
                               reinterpret_cast<const float*>(addresses_[0]), problem_.leading_a,
                               reinterpret_cast<const float*>(addresses_[1]), problem_.increment_x,
                               beta, reinterpret_cast<float*>(addresses_[2]), problem_.increment_y);
        } else {
            status =
                sdk.sgemm_ex(handle_->native, static_cast<cublasOperation_t>(problem_.transpose_a),
                             static_cast<cublasOperation_t>(problem_.transpose_b), problem_.rows,
                             problem_.columns, problem_.depth, alpha,
                             reinterpret_cast<const void*>(addresses_[0]),
                             static_cast<cudaDataType>(problem_.type_a), problem_.leading_a,
                             reinterpret_cast<const void*>(addresses_[1]),
                             static_cast<cudaDataType>(problem_.type_b), problem_.leading_b, beta,
                             reinterpret_cast<void*>(addresses_[2]),
                             static_cast<cudaDataType>(problem_.type_c), problem_.leading_c);
        }
        // NOLINTEND(performance-no-int-to-ptr)
        // The original result remains authoritative even if this postcondition
        // cannot prove safe physical cleanup. Never replay the SDK call.
        (void)observe_state(stream, true);
        return static_cast<std::int32_t>(status);
    }

    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    void fail(CUresult status, bool uncertain, bool clean = false) noexcept {
        completion_.status = status;
        completion_.is_cleanup_uncertain = uncertain;
        completion_.is_clean_preparation_rejection = clean;
        if (uncertain) {
            if (handle_) handle_->is_uncertain = true;
            owner_.fault();
        }
    }
    bool prove_context() noexcept {
        const auto status = owner_.check_context(true);
        if (status == CUDA_SUCCESS) return true;
        fail(status, true);
        return false;
    }
    bool observe_state(CUstream expected_stream, bool has_possible_effects) noexcept {
        if (!prove_context()) return false;
        cudaStream_t stream{};
        const auto stream_status = owner_.dispatch.blas.get_stream(handle_->native, &stream);
        if (!prove_context()) return false;
        if (stream_status != CUBLAS_STATUS_SUCCESS) {
            fail(CUDA_ERROR_UNKNOWN, has_possible_effects);
            return false;
        }
        if (stream != expected_stream) {
            fail(CUDA_ERROR_INVALID_HANDLE, true);
            return false;
        }
        cublasPointerMode_t mode{};
        const auto mode_status = owner_.dispatch.blas.get_pointer_mode(handle_->native, &mode);
        if (!prove_context()) return false;
        if (mode_status != CUBLAS_STATUS_SUCCESS) {
            fail(CUDA_ERROR_UNKNOWN, has_possible_effects);
            return false;
        }
        std::uint32_t raw_mode{};
        static_assert(sizeof(raw_mode) == sizeof(mode));
        std::memcpy(&raw_mode, &mode, sizeof(raw_mode));
        if (raw_mode != problem_.pointer_mode) {
            fail(CUDA_ERROR_INVALID_VALUE, true);
            return false;
        }
        return true;
    }
    Implementation& owner_;
    std::shared_ptr<Storage> storage_;
    Problem problem_;
    std::array<CUdeviceptr, 5> addresses_;
    bool requires_version_{};
    Blas* handle_{};
    CudaExecutionLibraryCompletion completion_{.was_native_issued = false};
};
}  // namespace glimmer::backend::cuda
