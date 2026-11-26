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
struct CudaSessionHostFunction::State {
    std::atomic<bool> entered{}, completed{};
};
CudaSessionHostFunction::CudaSessionHostFunction() : state_(std::make_unique<State>()) {}
CudaSessionHostFunction::~CudaSessionHostFunction() = default;
bool CudaSessionHostFunction::entered() const noexcept {
    return state_->entered.load(std::memory_order_acquire);
}
void CudaSessionHostFunction::finish() noexcept {
    state_->completed.store(true, std::memory_order_release);
    state_->completed.notify_all();
}
void CUDA_CB CudaSessionHostFunction::wait(void* data) noexcept {
    auto& state = *static_cast<State*>(data);
    state.entered.store(true, std::memory_order_release);
    while (!state.completed.load(std::memory_order_acquire))
        state.completed.wait(false, std::memory_order_acquire);
}
namespace {
#ifdef GLIMMER_HAS_CUDNN
namespace dnn_profile = ::glimmer::internal::cudnn_profile;
#endif
thread_local std::shared_ptr<const void> resource_owner;
std::shared_ptr<CudaProfileCache> session_profile_cache(
    const CudaSessionPersistentConfig& persistent, std::shared_ptr<CudaProfileCache> profiling) {
    if (!persistent.observes_block_completion) return profiling;
    if (profiling &&
        (profiling->is_learning() || profiling->snapshot().cases || !profiling->kinds().empty()))
        throw std::invalid_argument("block completion diagnostics cannot use measured profiles");
    // Do not retain an initially empty external cache: a later restore must
    // never feed timings from ordinary code into an observed command.
    return {};
}
std::shared_ptr<CudaExecutionOwnerActivity> session_activity(
    const CudaExecutionDomainMember& member) {
    if (!member) return std::make_shared<CudaExecutionOwnerActivity>();
    auto activity = member.owner_activity();
    if (!activity) throw std::invalid_argument("unavailable CUDA execution domain member");
    return activity;
}
CudaSessionStreamConfig session_streams(CudaSessionStreamConfig streams,
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

std::shared_ptr<CudaExecutionObserver> configuration_observation(
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

bool geam_scalar_is_zero(cudaDataType type, const std::array<std::byte, 16>& value) noexcept {
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
CudaDnnResult dnn_gate(const CudaSessionResult& gate) {
    const auto disposition = gate.disposition == CudaSessionDisposition::kDeferred
                                 ? CudaDnnDisposition::kDeferred
                                 : CudaDnnDisposition::kRejected;
    return {.disposition = disposition, .status = gate.status};
}
CudaDnnResult dnn_completed(cudnnStatus_t status) {
    return {.disposition = CudaDnnDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
#endif

CUresult context_flags_status(unsigned int flags) noexcept {
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
bool clean_context_creation_rejection(CUresult status) noexcept {
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

std::optional<std::size_t> parameter_count(PtxTokens tokens, std::string_view name,
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

CudaSessionResult rejected(CUresult status) noexcept {
    CudaSessionResult result;
    result.status = status;
    return result;
}
CudaSessionResult completed(CUresult status) noexcept {
    CudaSessionResult result;
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = status;
    result.original_status = status;
    return result;
}
CudaSessionResult settled() noexcept {
    auto result = completed(CUDA_SUCCESS);
    result.original_status.reset();
    return result;
}
CudaSessionResult deferred_resource_release() noexcept {
    auto result = rejected(CUDA_ERROR_NOT_READY);
    result.disposition = CudaSessionDisposition::kDeferred;
    return result;
}
CudaSessionBlasResult blas_gate(const CudaSessionResult& result) noexcept {
    return {.disposition = result.disposition, .status = result.status, .original_status = {}};
}
CudaSessionBlasResult blas_completed(cublasStatus_t status) noexcept {
    return {.disposition = CudaSessionDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
CudaSessionRandResult random_gate(const CudaSessionResult& result) noexcept {
    return {.disposition = result.disposition, .status = result.status, .original_status = {}};
}
CudaSessionRandResult random_completed(curandStatus_t status) noexcept {
    return {.disposition = CudaSessionDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
struct MatrixDimensions {
    int rows{}, columns{}, leading{};
};
std::optional<std::size_t> matrix_bytes(MatrixDimensions dimensions,
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
std::optional<std::size_t> vector_bytes(int length, int increment) noexcept {
    if (length <= 0 || !increment) return std::nullopt;
    const auto signed_stride = static_cast<std::int64_t>(increment);
    const auto stride =
        static_cast<std::uint64_t>(signed_stride < 0 ? -signed_stride : signed_stride);
    const auto intervals = static_cast<std::uint64_t>(length - 1);
    constexpr auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (intervals > (maximum - 1) / stride) return std::nullopt;
    return static_cast<std::size_t>(intervals * stride + 1) * sizeof(float);
}
std::size_t gemm_element_bytes(cudaDataType type) noexcept {
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
bool gemm_is_complex(cudaDataType type) noexcept {
    return type == CUDA_C_16F || type == CUDA_C_16BF || type == CUDA_C_32F || type == CUDA_C_64F ||
           type == CUDA_C_8I || type == CUDA_C_8U || type == CUDA_C_16I || type == CUDA_C_16U ||
           type == CUDA_C_32I || type == CUDA_C_32U || type == CUDA_C_64I || type == CUDA_C_64U;
}
std::size_t gemm_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
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
bool lt_matrix_state(const CudaLtDispatch& sdk, const CudaLtObjectView& view, bool is_output,
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
}  // namespace

std::size_t cuda_gemm_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
    return gemm_scalar_bytes(compute, output);
}

struct CudaSessionEvent::State {
    CUevent native{};
    std::optional<CudaExecutionFrontier> frontier;
    CUresult record_status{CUDA_SUCCESS};
    bool has_destroy_attempt{};
    CudaSessionEventIdentity identity;
    std::weak_ptr<const CudaExecutionOwnerActivity> activity;
};
CUresult CudaSessionEvent::observe() const noexcept {
    if (!state_ || !lifetime_) return CUDA_ERROR_INVALID_HANDLE;
    if (static_cast<std::uint64_t>(::getpid()) != process_ || !resource_owner ||
        thread_.lock() != resource_owner)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (owner_.expired()) return CUDA_ERROR_INVALID_HANDLE;
    if (state_->record_status != CUDA_SUCCESS) return state_->record_status;
    return state_->frontier ? state_->frontier->observe() : CUDA_SUCCESS;
}
std::optional<CudaSessionEventIdentity> CudaSessionEvent::diagnostic_identity() const noexcept {
    if (!state_ || !lifetime_ || static_cast<std::uint64_t>(::getpid()) != process_ ||
        !resource_owner || thread_.lock() != resource_owner || owner_.expired())
        return std::nullopt;
    const auto activity = state_->activity.lock();
    if (!activity || activity->is_native_issue_active || activity->is_folded_active ||
        activity->is_rnn_query_active || activity->is_preparation_active ||
        !state_->identity.event_object)
        return std::nullopt;
    return state_->identity;
}

struct CudaSessionExternalContext::State {
    std::uint64_t id{};
    CUcontext native{};
    int device_ordinal{-1};
    pid_t process{};
    std::shared_ptr<const void> lifetime;
};
CUcontext CudaSessionExternalContext::native() const noexcept {
    return state_ && state_->process == ::getpid() ? state_->native : nullptr;
}
std::uint64_t CudaSessionExternalContext::id() const noexcept {
    return native() ? state_->id : 0;
}
int CudaSessionExternalContext::device_ordinal() const noexcept {
    return native() ? state_->device_ordinal : -1;
}
struct CudaSessionExternalCommand::State {
    std::vector<std::shared_ptr<const void>> process_lifetimes;
    CudaSessionExternalContext context;
    std::uint64_t stream_id{};
    CUstream stream{};
    std::vector<CudaSessionExternalRange> ranges;
    std::vector<std::uint64_t> parent_modules;
};
CudaSessionExternalContext CudaSessionExternalCommand::context() const noexcept {
    return state_ && state_->context.native() ? state_->context : CudaSessionExternalContext{};
}
CUstream CudaSessionExternalCommand::stream() const noexcept {
    return state_ && state_->context.native() ? state_->stream : nullptr;
}
std::uint64_t CudaSessionExternalCommand::stream_id() const noexcept {
    return state_ && state_->context.native() ? state_->stream_id : 0;
}
std::span<const CudaSessionExternalRange> CudaSessionExternalCommand::ranges() const noexcept {
    return state_ && state_->context.native()
               ? std::span<const CudaSessionExternalRange>{state_->ranges}
               : std::span<const CudaSessionExternalRange>{};
}

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

CudaExecutionSession::CudaExecutionSession(
    ExecutionSessionIdentity identity, std::uint64_t context_id, CudaSessionDispatch dispatch,
    CudaSessionLimits limits, CudaSessionPersistentConfig persistent,
    std::shared_ptr<CudaPtxCache> ptx_cache, std::shared_ptr<CudaProfileCache> profile_cache,
    bool requires_profile_preemption, CudaSessionStreamConfig streams,
    CudaExecutionDomainMember domain_member)
    : implementation_(std::make_unique<Implementation>(
          identity, context_id, std::move(dispatch), limits, persistent, std::move(ptx_cache),
          std::move(profile_cache), requires_profile_preemption, streams,
          std::move(domain_member))) {}

CUresult CudaExecutionSession::attach_process_memory(
    std::shared_ptr<CudaProcessMemoryBackend> owner) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!owner || state.has_started || state.storage->context || state.process_memory ||
        state.next_id >= (UINT64_C(1) << 63))
        return CUDA_ERROR_INVALID_VALUE;
    state.process_memory = std::move(owner);
    return CUDA_SUCCESS;
}
CudaSessionProcessAllocationAcceptance CudaExecutionSession::allocate_process_memory(
    std::uint64_t sequence, const CudaProcessMemoryRequest& request) noexcept {
    auto& state = *implementation_;
    const auto invalid = [] {
        return CudaSessionProcessAllocationAcceptance{
            {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}}, {}};
    };
    if (!state.process_memory || state.process_memory->native_health() != CUDA_SUCCESS ||
        !state.owns() || state.is_closing || state.is_native_issue_active ||
        state.check_context() != CUDA_SUCCESS || !state.executor ||
        (request.operation != CudaProcessMemoryOperation::kAllocateAsync &&
         request.operation != CudaProcessMemoryOperation::kAllocateFromPoolAsync))
        return invalid();
    if (request.stream &&
        std::ranges::find(state.storage->streams, request.stream, &Implementation::Stream::id) ==
            state.storage->streams.end())
        return invalid();
    try {
        auto operation = std::make_shared<Implementation::ProcessAllocation>();
        operation->backend = state.process_memory;
        operation->request = request;
        operation->tail = std::make_shared<Implementation::ProcessTail>(state.storage);
        if (request.has_output) {
            auto created =
                CudaNativeOutputCell::create(std::as_bytes(std::span{&request.output_seed, 1}));
            if (created.status != CUDA_SUCCESS)
                return {{.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}}, {}};
            operation->output = std::move(created.cell);
        }
        const auto reserved = state.process_memory->reserve(request);
        if (reserved.status != CUDA_SUCCESS)
            return {{.status = reserved.status == CUDA_ERROR_OUT_OF_MEMORY
                                   ? ExecutionAcceptStatus::kOutOfMemory
                                   : ExecutionAcceptStatus::kInvalid,
                     .receipt = {}},
                    {}};
        operation->id = reserved.id;
        auto accepted = state.executor->accept_library(sequence, {request.stream, {}, operation});
        return {accepted, accepted.accepted() ? operation : nullptr};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}}, {}};
    }
}
CudaExecutionSession::~CudaExecutionSession() {
    auto& state = *implementation_;
    if (getpid() != state.process) {
        (void)implementation_.release();
        return;
    }
    if (!state.owns() || shutdown().status != CUDA_SUCCESS) {
        const auto* retained = implementation_.release();
        try {
            spdlog::critical("Retaining unsettled CUDA session: client={}, context={}",
                             retained->identity.client_id, retained->context_id);
        } catch (...) {
        }
    }
}

CudaSessionResult CudaExecutionSession::initialize(int device_ordinal,
                                                   CudaSessionContextConfig config) noexcept {
    const auto flags = config.flags;
    auto& state = *implementation_;
    if (state.process_memory) {
        const auto health = state.process_memory->native_health();
        if (health != CUDA_SUCCESS) return rejected(health);
    }
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    if (state.is_closing) return rejected(CUDA_ERROR_DEINITIALIZED);
    if (device_ordinal < 0 || state.has_started) return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto flags_status = context_flags_status(flags);
    if (flags_status != CUDA_SUCCESS) return rejected(flags_status);
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        const auto quiescent = state.domain_cold_status();
        if (quiescent == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (quiescent != CUDA_SUCCESS) return rejected(quiescent);
        const auto effective_flags =
            flags | ((state.persistent.workers || (member.flags & CU_CTX_MAP_HOST))
                         ? static_cast<unsigned int>(CU_CTX_MAP_HOST)
                         : 0U);
        if (member.device_ordinal != device_ordinal || member.logical_flags != effective_flags)
            return rejected(CUDA_ERROR_INVALID_VALUE);
        const auto attached =
            state.storage->domain_member.attach({state.identity, state.context_id});
        if (attached != CUDA_SUCCESS) return rejected(attached);
    }
    // Sharing adds cold native observations. Exclude callback lifecycle reentry
    // before the first observation, without changing legacy initialization.
    std::optional<NativeExecutionScope> sharing_scope;
    if (state.persistent.uses_workload_agnostic || state.stream_config.uses_default_proxy)
        sharing_scope.emplace(state.is_native_issue_active);
    state.has_started = true;
    auto result = CUDA_SUCCESS;
    CUdevice device{};
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        device = member.native_device;
    } else {
        result = state.invoke(state.dispatch.initialize, 0);
        if (result == CUDA_SUCCESS)
            result = state.invoke(state.dispatch.device_get, &device, device_ordinal);
    }
    if (result == CUDA_SUCCESS) state.device_ordinal = device_ordinal;
    if (result != CUDA_SUCCESS) return rejected(result);
    state.native_device = device;
    if (state.persistent.uses_workload_agnostic) {
        int multiprocessors{}, native_threads{};
        result = state.invoke(state.dispatch.device_attribute, &multiprocessors,
                              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        if (result != CUDA_SUCCESS) return rejected(result);
        result = state.invoke(state.dispatch.device_attribute, &native_threads,
                              CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
        if (result != CUDA_SUCCESS) return rejected(result);
        if (multiprocessors <= 0 || native_threads <= 0 ||
            state.persistent.sharing_threads_per_sm > static_cast<unsigned int>(native_threads))
            return rejected(CUDA_ERROR_INVALID_VALUE);
        state.sharing_sm_count = static_cast<unsigned int>(multiprocessors);
    }
    if (state.profile_cache) {
        CUuuid uuid{};
        int driver{};
        result = state.invoke(state.dispatch.device_uuid, &uuid, device);
        if (result == CUDA_SUCCESS) result = state.invoke(state.dispatch.driver_version, &driver);
        if (result != CUDA_SUCCESS) return rejected(result);
        if (driver <= 0) return rejected(CUDA_ERROR_INVALID_VALUE);
        static_assert(sizeof(uuid.bytes) == sizeof(state.profile_platform.uuid));
        std::memcpy(state.profile_platform.uuid.data(), uuid.bytes, sizeof(uuid.bytes));
        if (std::ranges::all_of(state.profile_platform.uuid,
                                [](auto value) { return value == std::byte{}; }))
            return rejected(CUDA_ERROR_INVALID_VALUE);
        state.profile_platform.driver_version = static_cast<unsigned int>(driver);
        state.profile_platform.toolkit_version = CUDA_VERSION;
        state.profile_platform.generator_abi = kCudaPtxGeneratorAbi;
    }
    CUcontext current{};
    result = state.invoke(state.dispatch.execution.context_get_current, &current);
    if (result != CUDA_SUCCESS) return rejected(result);
    if (current) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        state.storage->context = member.context.native;
        result = state.invoke(state.dispatch.context_set_current, state.storage->context);
    } else {
        result = state.invoke(
            state.dispatch.context_create, &state.storage->context,
            flags | (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U),
            device);
    }
    CUcontext after{};
    const auto observed = state.invoke(state.dispatch.execution.context_get_current, &after);
    const bool known_extent =
        observed == CUDA_SUCCESS && (!after || after == state.storage->context);
    if (result != CUDA_SUCCESS || !state.storage->context) {
        const bool is_clean =
            !state.storage->context && clean_context_creation_rejection(result) && known_extent;
        state.is_creation_unproven =
            !known_extent || (!state.storage->context && !clean_context_creation_rejection(result));
        if (!is_clean) state.fault();
        auto failed = completed(result);
        if (result == CUDA_SUCCESS) {
            failed.disposition = CudaSessionDisposition::kRejected;
            failed.status = CUDA_ERROR_UNKNOWN;
        }
        return failed;
    }
    if (observed != CUDA_SUCCESS || after != state.storage->context) {
        state.is_creation_unproven = true;
        state.fault();
        auto failed = rejected(observed == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.materialize_configuration(true);
        if (configured != CUDA_SUCCESS) return rejected(configured);
    }
    if (state.stream_config.uses_default_proxy) {
        int native_priority{};
        if (state.stream_config.scheduling_priority) {
            const auto priorities = state.scheduling_stream_priorities(0);
            if (priorities.status != CUDA_SUCCESS) {
                auto failed = rejected(priorities.status);
                failed.original_status = CUDA_SUCCESS;
                return failed;  // The created/borrowed context remains owned.
            }
            native_priority = priorities.native;
        }
        result = state.check_context(true);
        if (result == CUDA_SUCCESS) {
            result = state.stream_config.scheduling_priority
                         ? state.invoke(state.dispatch.stream_create_priority,
                                        &state.storage->default_proxy, CU_STREAM_NON_BLOCKING,
                                        native_priority)
                         : state.invoke(state.dispatch.stream_create, &state.storage->default_proxy,
                                        CU_STREAM_NON_BLOCKING);
            if (result == CUDA_SUCCESS && state.storage->default_proxy &&
                state.stream_config.scheduling_priority) {
                int actual{};
                result = state.invoke(state.dispatch.stream_get_priority,
                                      state.storage->default_proxy, &actual);
                if (result == CUDA_SUCCESS && actual != native_priority)
                    result = CUDA_ERROR_UNKNOWN;
            }
        }
        const auto proof = state.check_context(true);
        if (result != CUDA_SUCCESS || proof != CUDA_SUCCESS || !state.storage->default_proxy ||
            state.storage->default_proxy == CU_STREAM_LEGACY ||
            state.storage->default_proxy == CU_STREAM_PER_THREAD) {
            state.storage->is_default_proxy_uncertain = true;
            state.fault();
            auto failed = rejected(proof != CUDA_SUCCESS    ? proof
                                   : result != CUDA_SUCCESS ? result
                                                            : CUDA_ERROR_UNKNOWN);
            failed.original_status = CUDA_SUCCESS;
            return failed;
        }
    }
    result = state.initialize_progress();
    if (result != CUDA_SUCCESS) {
        if (!state.clean_shared_allocation_rejection(result) ||
            !state.rollback_shared_initialization())
            state.fault();
        auto failed = rejected(result);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
    try {
        auto execution = state.dispatch.execution;
        execution.upload_async = state.dispatch.upload_async;
        execution.download_async = state.dispatch.download_async;
        execution.stream_wait_event = state.dispatch.stream_wait_event;
        state.executor = std::make_unique<CudaExecutionDispatcher>(
            state.identity,
            CudaExecutionContext{state.context_id, state.storage->context, state.storage,
                                 state.process_memory ? state.process_memory->owner() : nullptr,
                                 state.process_memory
                                     ? std::function<CUresult()>{[memory = state.process_memory] {
                                           return memory->native_health();
                                       }}
                                     : std::function<CUresult()>{}},
            execution, state.limits.submissions, state.limits.max_objects + 28,
            state.profile_cache && state.profile_cache->is_learning(),
            CudaExecutionStreamConfig{state.storage->default_proxy,
                                      state.storage->default_proxy ? state.storage : nullptr},
            state.owner_activity,
            state.storage->domain_member
                ? std::function<CUresult()>{[member = state.storage->domain_member] {
                      return member.other_members_quiescent();
                  }}
                : std::function<CUresult()>{});
        result = state.executor->bind_executor();
        if (result != CUDA_SUCCESS) {
            state.executor.reset();
            state.fault();
            auto failed = rejected(result);
            failed.original_status = CUDA_SUCCESS;
            return failed;
        }
        return completed(result);
    } catch (...) {
        state.fault();
        auto failed = rejected(CUDA_ERROR_OUT_OF_MEMORY);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
}

CudaSessionResult CudaExecutionSession::get_context_flags() noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!state.dispatch.context_get_flags) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    unsigned int flags{};
    auto result = completed(state.invoke(state.dispatch.context_get_flags, &flags));
    if (result.status == CUDA_SUCCESS) result.id = flags;
    return result;
}

CudaSessionResult CudaExecutionSession::set_context_flags(unsigned int flags) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    const auto flags_status = context_flags_status(flags);
    if (flags_status != CUDA_SUCCESS) return rejected(flags_status);
    if (!state.dispatch.context_set_flags) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (state.uses_shared_domain) {
        auto gate = state.cold_gate(true);
        if (gate.status != CUDA_SUCCESS) return gate;
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        const auto configured = state.mutate_domain_configuration(
            CudaExecutionDomainConfigurationMutation::kFlags, 0, state.dispatch.context_set_flags,
            flags | (member.flags & CU_CTX_MAP_HOST) |
                (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U));
        auto result = configured.status == CUDA_ERROR_NOT_READY ? deferred_resource_release()
                                                                : rejected(configured.status);
        result.original_status =
            configured.original_status
                ? std::optional{static_cast<CUresult>(*configured.original_status)}
                : std::nullopt;
        if (configured.original_status && configured.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kComplete;
            result.status = static_cast<CUresult>(*configured.original_status);
        }
        return result;
    }
    state.has_context_configuration_attempt = true;
    if (state.control_epoch == UINT64_MAX) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    ++state.control_epoch;
    const NativeExecutionScope active{state.is_native_issue_active};
    return completed(state.invoke(
        state.dispatch.context_set_flags,
        flags | (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U)));
}

CudaContextControlResult CudaExecutionSession::context_control(CudaContextControl operation,
                                                               std::uint32_t argument,
                                                               std::uint64_t value,
                                                               bool is_runtime) noexcept {
    auto& state = *implementation_;
    CudaContextControlResult result;
    using Operation = CudaContextControl;
    const bool is_mutation =
        operation == Operation::kSetLimit || operation == Operation::kSetCache ||
        operation == Operation::kSetSharedMemory || operation == Operation::kResetPersistingCache;
    auto gate = is_mutation ? state.cold_gate(true) : rejected(state.check_context());
    if (!is_mutation && gate.status == CUDA_SUCCESS) {
        if (state.is_closing || state.is_faulted || !state.executor ||
            state.executor->snapshot().is_quarantined)
            gate.status = CUDA_ERROR_UNKNOWN;
        else
            gate = settled();
    }
    result.status = gate.status;
    if (gate.disposition != CudaSessionDisposition::kComplete) {
        result.disposition = gate.disposition;
        return result;
    }
    const bool has_argument =
        operation == Operation::kGetLimit || operation == Operation::kSetLimit ||
        operation == Operation::kSetCache || operation == Operation::kSetSharedMemory ||
        operation == Operation::kGetAffinity;
    if ((!has_argument && argument) || (operation != Operation::kSetLimit && value) ||
        value > std::numeric_limits<std::size_t>::max() ||
        ((operation == Operation::kGetLimit || operation == Operation::kSetLimit) &&
         argument > (is_runtime ? 7U : 15U)) ||
        ((operation == Operation::kSetCache || operation == Operation::kSetSharedMemory) &&
         argument > 3) ||
        (operation == Operation::kGetAffinity && argument > 1)) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    // Reject only values outside the SDK enum's representable range. Unknown
    // but representable enumerators reach the SDK and retain its original error.
    if (operation == Operation::kGetLimit && state.uses_shared_domain) {
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) {
            result.status = member.status;
            return result;
        }
        if (member.has_unobserved_configuration_effects) {
            result.status = CUDA_ERROR_NOT_READY;
            result.disposition = CudaSessionDisposition::kDeferred;
            return result;
        }
    }
    result.status = CUDA_ERROR_NOT_SUPPORTED;
    const auto complete = [&](auto status) {
        // Existing profiles are qualified for the context's initial controls.
        // A native mutation attempt may affect state even on an error. Do not
        // train/reuse those profiles afterward; ordinary configured execution
        // and persistent preparation remain enabled.
        if (is_mutation) state.has_context_configuration_attempt = true;
        result.disposition = CudaSessionDisposition::kComplete;
        result.status = CUDA_SUCCESS;
        result.original_status = static_cast<std::int32_t>(status);
        if (*result.original_status != 0) result.value = result.auxiliary = 0;
        return result;
    };
    const auto mutate = [&](auto callback, auto... arguments) {
        if (!state.uses_shared_domain) {
            if (state.control_epoch == UINT64_MAX) {
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            ++state.control_epoch;
            const NativeExecutionScope active{state.is_native_issue_active};
            return complete(state.invoke(callback, arguments...));
        }
        using Mutation = CudaExecutionDomainConfigurationMutation;
        const auto mutation = operation == Operation::kSetLimit   ? Mutation::kLimit
                              : operation == Operation::kSetCache ? Mutation::kCache
                              : operation == Operation::kSetSharedMemory
                                  ? Mutation::kSharedMemory
                                  : Mutation::kResetPersistingCache;
        const auto configured = state.mutate_domain_configuration(
            mutation, mutation == Mutation::kLimit ? argument : 0, callback, arguments...);
        result.status = configured.status;
        result.original_status = configured.original_status;
        result.disposition = configured.status == CUDA_ERROR_NOT_READY
                                 ? CudaSessionDisposition::kDeferred
                             : configured.original_status && configured.status == CUDA_SUCCESS
                                 ? CudaSessionDisposition::kComplete
                                 : CudaSessionDisposition::kRejected;
        return result;
    };
    const auto& sdk = state.dispatch.context;
    if (is_runtime) {
        const auto& runtime = sdk.runtime;
        switch (operation) {
            case Operation::kGetLimit: {
                if (!runtime.get_limit) return result;
                std::size_t bytes{};
                const auto status =
                    state.invoke(runtime.get_limit, &bytes, static_cast<cudaLimit>(argument));
                result.value = bytes;
                return complete(status);
            }
            case Operation::kSetLimit:
                if (!runtime.set_limit) return result;
                return mutate(runtime.set_limit, static_cast<cudaLimit>(argument),
                              static_cast<std::size_t>(value));
            case Operation::kGetCache: {
                if (!runtime.get_cache) return result;
                cudaFuncCache configuration{};
                const auto status = state.invoke(runtime.get_cache, &configuration);
                result.value = static_cast<std::uint32_t>(configuration);
                return complete(status);
            }
            case Operation::kSetCache:
                if (!runtime.set_cache) return result;
                return mutate(runtime.set_cache, static_cast<cudaFuncCache>(argument));
            case Operation::kGetSharedMemory: {
                if (!runtime.get_shared) return result;
                cudaSharedMemConfig configuration{};
                const auto status = state.invoke(runtime.get_shared, &configuration);
                result.value = static_cast<std::uint32_t>(configuration);
                return complete(status);
            }
            case Operation::kSetSharedMemory:
                if (!runtime.set_shared) return result;
                return mutate(runtime.set_shared, static_cast<cudaSharedMemConfig>(argument));
            case Operation::kResetPersistingCache:
                if (!runtime.reset_cache) return result;
                return mutate(runtime.reset_cache);
            default:
                return result;
        }
    }
    switch (operation) {
        case Operation::kGetLimit: {
            if (!sdk.get_limit) return result;
            std::size_t bytes{};
            const auto status = state.invoke(sdk.get_limit, &bytes, static_cast<CUlimit>(argument));
            result.value = bytes;
            return complete(status);
        }
        case Operation::kSetLimit:
            if (!sdk.set_limit) return result;
            return mutate(sdk.set_limit, static_cast<CUlimit>(argument),
                          static_cast<std::size_t>(value));
        case Operation::kGetCache: {
            if (!sdk.get_cache) return result;
            CUfunc_cache configuration{};
            const auto status = state.invoke(sdk.get_cache, &configuration);
            result.value = static_cast<std::uint32_t>(configuration);
            return complete(status);
        }
        case Operation::kSetCache:
            if (!sdk.set_cache) return result;
            return mutate(sdk.set_cache, static_cast<CUfunc_cache>(argument));
        case Operation::kGetSharedMemory: {
            if (!sdk.get_shared) return result;
            CUsharedconfig configuration{};
            const auto status = state.invoke(sdk.get_shared, &configuration);
            result.value = static_cast<std::uint32_t>(configuration);
            return complete(status);
        }
        case Operation::kSetSharedMemory:
            if (!sdk.set_shared) return result;
            return mutate(sdk.set_shared, static_cast<CUsharedconfig>(argument));
        case Operation::kResetPersistingCache:
            if (!sdk.reset_cache) return result;
            return mutate(sdk.reset_cache);
        case Operation::kGetApiVersion: {
            if (!sdk.api_version) return result;
            unsigned int version{};
            const auto status = state.invoke(sdk.api_version, state.storage->context, &version);
            result.value = version;
            return complete(status);
        }
        case Operation::kGetAffinity: {
            if (!sdk.affinity) return result;
            CUexecAffinityParam parameter{};
            const auto status =
                state.invoke(sdk.affinity, &parameter, static_cast<CUexecAffinityType>(argument));
            if (status == CUDA_SUCCESS &&
                (argument != 0 || parameter.type != CU_EXEC_AFFINITY_TYPE_SM_COUNT)) {
                state.fault();
                result.status = CUDA_ERROR_UNKNOWN;
                result.original_status = CUDA_SUCCESS;
                return result;
            }
            if (status == CUDA_SUCCESS) result.value = parameter.param.smCount.val;
            return complete(status);
        }
        case Operation::kGetPriorityRange: {
            if (!sdk.priority_range) return result;
            int least{}, greatest{};
            const auto status = state.invoke(sdk.priority_range, &least, &greatest);
            result.value = std::bit_cast<std::uint32_t>(least);
            result.auxiliary = std::bit_cast<std::uint32_t>(greatest);
            return complete(status);
        }
        default:
            return result;
    }
}

CudaSessionResult CudaExecutionSession::activate() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt)
        return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        (state.uses_shared_domain && state.is_preparation_active)) {
        auto result = rejected(CUDA_ERROR_NOT_READY);
        result.disposition = CudaSessionDisposition::kDeferred;
        return result;
    }
    CUcontext previous{};
    std::optional<NativeExecutionScope> configuration_scope;
    if (state.storage->domain_member) {
        const auto gate = state.storage->domain_member.configuration_status();
        if (gate != CUDA_SUCCESS && gate != CUDA_ERROR_NOT_READY) return rejected(gate);
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        if (member.can_virtualize_configuration) {
            configuration_scope.emplace(state.is_native_issue_active);
            const auto observed = state.dispatch.execution.context_get_current(&previous);
            if (observed != CUDA_SUCCESS) {
                state.fault();
                return rejected(observed);
            }
        }
    }
    auto selected =
        completed(state.invoke(state.dispatch.context_set_current, state.storage->context));
    if (selected.status != CUDA_SUCCESS || !state.storage->domain_member) return selected;
    const auto configured =
        state.storage->domain_member.materialize_configuration(configuration_scope.has_value());
    if ((configured == CUDA_ERROR_NOT_READY || configured == CUDA_ERROR_OUT_OF_MEMORY) &&
        configuration_scope) {
        // A clean, unissued configuration wait must not leak this activation's
        // TLS selection into a different logical owner's next scheduling turn.
        const auto restored = state.invoke(state.dispatch.context_set_current, previous);
        if (restored != CUDA_SUCCESS || state.is_faulted) {
            state.fault();
            return rejected(restored == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : restored);
        }
    }
    if (configured == CUDA_ERROR_NOT_READY) return deferred_resource_release();
    return configured == CUDA_SUCCESS ? selected : rejected(configured);
}
CudaSessionResult CudaExecutionSession::deactivate() noexcept {
    auto& state = *implementation_;
    if (state.owns() &&
        (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)) {
        auto result = rejected(CUDA_ERROR_NOT_READY);
        result.disposition = CudaSessionDisposition::kDeferred;
        return result;
    }
    if (state.owns() && state.executor && state.executor->has_active_enqueue_turn()) {
        // An actor must restore its own thread to NULL after beginning a source
        // turn, so another retained source can be visited. This changes ONLY the
        // owner's TLS context, never the worker's context or native issue grant.
        // Keep native callbacks/reentry excluded throughout both independent proofs.
        const NativeExecutionScope scope(state.is_native_issue_active);
        const auto before = state.check_context(true, true);
        if (before != CUDA_SUCCESS) return rejected(before);
        const auto selected = state.invoke(state.dispatch.context_set_current, nullptr);
        CUcontext current{};
        const auto proof = state.invoke(state.dispatch.execution.context_get_current, &current);
        if (proof != CUDA_SUCCESS || current != nullptr) {
            state.fault();
            return rejected(proof != CUDA_SUCCESS ? proof : CUDA_ERROR_INVALID_CONTEXT);
        }
        return completed(selected);
    }
    const auto result = state.check_context();
    if (result != CUDA_SUCCESS) return rejected(result);
    return completed(state.invoke(state.dispatch.context_set_current, nullptr));
}

bool CudaExecutionSession::set_preparation_observer(CudaPreparationObserver observer) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_preparation_active) return false;
    state.preparation_observer = observer;
    return true;
}
bool CudaExecutionSession::set_retirement_observer(
    std::shared_ptr<CudaRetirementObserver> observer) noexcept {
    auto& state = *implementation_;
    return state.owns() && state.executor &&
           state.executor->set_retirement_observer(std::move(observer));
}

CudaSessionResult CudaExecutionSession::load_ptx(std::string_view source) noexcept {
    return load_ptx_jit(source, {}, nullptr);
}
CudaSessionResult CudaExecutionSession::load_ptx_jit(
    std::string_view source, std::span<const CudaModuleJitOption> options,
    std::vector<CudaModuleJitOption>* outputs) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (source.empty() || source.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->modules.size() == state.limits.max_objects ||
        source.size() >= state.limits.max_image_bytes - state.storage->image_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        Implementation::Module module;
        module.source = source;
        module.source_static_bytes = state.inspect_static_storage(source);
        if (!module.source_static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        module.charged_bytes = module.source.size() + 1;
        const auto attached = state.attach_jit(module, options, outputs);
        if (attached != CUDA_SUCCESS) return rejected(attached);
        const auto prepared = state.transform_source(module);
        if (prepared != CUDA_SUCCESS) return rejected(prepared);
        if (module.transformed) {
            const auto static_bytes = state.inspect_static_storage(
                module.transformed->image, kOwnedPtxTransformLimits.output_bytes);
            if (!static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            module.prepared_static_budget = *static_bytes;
            const auto extra = module.transformed->image.size() + 1;
            if (extra >
                state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes)
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            module.charged_bytes += extra;
        }
        const bool is_combined = module.transformed && !module.jit;
        const auto* image = is_combined ? module.transformed->image.c_str() : module.source.c_str();
        if (!state.reserve_static(
                is_combined ? module.prepared_static_budget : *module.source_static_bytes,
                module.native_static_bytes))
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        auto result = completed(state.load_module_preparation(
            module, &module.native, image,
            is_combined ? CudaPreparationPhase::kLoadPrepared : CudaPreparationPhase::kLoadOriginal,
            outputs));
        if (result.status != CUDA_SUCCESS && !module.native && module.transformed && !module.jit &&
            !state.is_faulted) {
            // Retry original intake only before any native handle or launch is
            // published. Partial handles and malformed success never retry.
            module.transformed.reset();
            module.charged_bytes = module.source.size() + 1;
            state.release_static(module.native_static_bytes);
            if (!state.reserve_static(*module.source_static_bytes, module.native_static_bytes))
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            result = completed(state.load_preparation(&module.native, module.source.c_str(),
                                                      CudaPreparationPhase::kLoadOriginal));
        }
        if (result.status != CUDA_SUCCESS && !module.native && !state.is_faulted) {
            state.release_static(module.native_static_bytes);
            return result;
        }
        if (module.jit && result.status == CUDA_SUCCESS && module.native && !state.is_faulted) {
            const auto prepared = state.prepare_shared_binary_module(module);
            if (prepared != CUDA_SUCCESS) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = prepared;
            }
        }
        module.id = state.consume_id();
        const auto id = module.id;
        state.storage->image_bytes += module.charged_bytes;
        const bool is_valid = result.status == CUDA_SUCCESS && module.native && !state.is_faulted;
        state.storage->modules.push_back(std::move(module));
        if (is_valid)
            result.id = id;
        else {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            if (result.status == CUDA_SUCCESS) {
                result.status = CUDA_ERROR_UNKNOWN;
            }
        }
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}

CudaSessionResult CudaExecutionSession::load_binary(std::span<const std::byte> image) noexcept {
    return load_binary_jit(image, {}, nullptr);
}
CudaSessionResult CudaExecutionSession::load_binary_jit(
    std::span<const std::byte> image, std::span<const CudaModuleJitOption> options,
    std::vector<CudaModuleJitOption>* outputs) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (image.empty()) return rejected(CUDA_ERROR_INVALID_IMAGE);
    if (!state.has_id_space || state.storage->modules.size() == state.limits.max_objects ||
        image.size() > state.limits.max_image_bytes - state.storage->image_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (!state.dispatch.device_attribute) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (!state.compute_capability) {
        int major{}, minor{};
        auto status =
            state.invoke(state.dispatch.device_attribute, &major,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, state.native_device);
        if (status == CUDA_SUCCESS)
            status =
                state.invoke(state.dispatch.device_attribute, &minor,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, state.native_device);
        if (status != CUDA_SUCCESS) return rejected(status);
        if (major <= 0 || major > 99 || minor < 0 || minor > 9)
            return rejected(CUDA_ERROR_NOT_SUPPORTED);
        state.compute_capability = static_cast<unsigned int>(major * 10 + minor);
    }
    try {
        Implementation::Module module;
        module.binary.assign(image.begin(), image.end());
        transform::CudaImageLimits limits;
        limits.image_bytes = std::min(state.limits.max_image_bytes, transform::kMaxCudaModuleBytes);
        limits.parameters = state.limits.max_parameters;
        limits.parameter_bytes = state.limits.submissions.max_parameter_bytes;
        limits.work_bytes = transform::kMaxCudaModuleWorkBytes;
        limits.compact_ptx_arities = true;
        {
            const PreparationScope phase(state.preparation_observer,
                                         CudaPreparationPhase::kInspectImage,
                                         state.is_preparation_active);
            module.metadata =
                transform::inspect_cuda_module(module.binary, *state.compute_capability, limits);
        }
        if (!module.metadata) return rejected(CUDA_ERROR_INVALID_IMAGE);
        module.charged_bytes = module.binary.size();
        auto remaining =
            state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes;
        if (module.metadata->retained_bytes > remaining) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        module.charged_bytes += module.metadata->retained_bytes;
        remaining -= module.metadata->retained_bytes;
        if (module.metadata->ptx) {
            module.source = std::move(*module.metadata->ptx);
            module.metadata->ptx.reset();
            module.source_static_bytes = state.inspect_static_storage(module.source);
            if (module.metadata->is_ptx_selected && !module.source_static_bytes)
                return rejected(CUDA_ERROR_NOT_SUPPORTED);
            const auto prepared = state.transform_source(module);
            if (prepared != CUDA_SUCCESS) return rejected(prepared);
        }
        const auto attached = state.attach_jit(module, options, outputs);
        if (attached != CUDA_SUCCESS) return rejected(attached);
        remaining =
            state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes;
        if (module.transformed) {
            const auto static_bytes = state.inspect_static_storage(
                module.transformed->image, kOwnedPtxTransformLimits.output_bytes);
            if (!static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            module.prepared_static_budget = *static_bytes;
            if (module.transformed->image.size() >= remaining)
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            module.charged_bytes += module.transformed->image.size() + 1;
        }
        const auto static_bytes = module.metadata->is_ptx_selected
                                      ? *module.source_static_bytes
                                      : module.metadata->static_storage_bytes;
        if (!state.reserve_static(static_bytes, module.native_static_bytes))
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        auto result =
            completed(state.load_module_preparation(module, &module.native, module.binary.data(),
                                                    CudaPreparationPhase::kLoadOriginal, outputs));
        if (result.status != CUDA_SUCCESS && !module.native && !state.is_faulted) {
            state.release_static(module.native_static_bytes);
            return result;
        }
        if (result.status == CUDA_SUCCESS && module.native && !state.is_faulted) {
            const auto prepared = state.prepare_shared_binary_module(module);
            if (prepared != CUDA_SUCCESS) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = prepared;
            }
        }
        module.id = state.consume_id();
        const auto id = module.id;
        const bool valid = result.status == CUDA_SUCCESS && module.native && !state.is_faulted;
        state.storage->image_bytes += module.charged_bytes;
        state.storage->modules.push_back(std::move(module));
        if (valid)
            result.id = id;
        else {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            if (result.status == CUDA_SUCCESS) {
                result.status = CUDA_ERROR_UNKNOWN;
            }
        }
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}

CudaSessionResult CudaExecutionSession::get_function(std::uint64_t module_id,
                                                     std::string_view name) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto module = std::ranges::find_if(
        state.storage->modules, [module_id](const auto& entry) { return entry.id == module_id; });
    if (module == state.storage->modules.end() || name.empty() || name.size() > 4096 ||
        name.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (module->transformed &&
        std::ranges::any_of(module->transformed->kernels,
                            [&](const auto& entry) { return entry.persistent_name == name; }))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    auto existing = std::ranges::find_if(state.storage->functions, [&](const auto& entry) {
        return entry.module == module_id && entry.name == name;
    });
    if (existing == state.storage->functions.end() &&
        (!state.has_id_space || state.storage->functions.size() == state.limits.max_objects))
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    const PreparationScope phase(state.preparation_observer,
                                 CudaPreparationPhase::kValidateFunction,
                                 state.is_preparation_active);
    std::optional<CUresult> original_result;
    try {
        Implementation::Function function;
        function.module = module_id;
        function.name = name;
        function.arguments.reserve(state.limits.max_parameters);
        auto result = completed(state.invoke(state.dispatch.function_get, &function.native,
                                             module->native, function.name.c_str()));
        original_result = result.original_status;
        if (state.is_faulted) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (result.status != CUDA_SUCCESS) return result;
        if (!function.native) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (existing != state.storage->functions.end()) {
            if (existing->native != function.native) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            result.id = existing->id;
            result.parameter_bytes = existing->parameter_bytes;
            result.arguments = existing->arguments;
            return result;
        }
        if (std::ranges::any_of(state.storage->functions, [&](const auto& alias) {
                return alias.module == module_id && alias.native == function.native &&
                       alias.has_configuration_attempt;
            })) {
            // A newly resolved public alias cannot reconstruct earlier public
            // settings. Preparing its private family could also reset a native
            // variant shared with an already configured alias.
            function.has_configuration_attempt = true;
            function.is_persistent_disabled = true;
            function.has_unknown_configuration = true;
            function.configuration.reset();
        }
        if (!state.has_graph_capture() && state.persistent.workers &&
            !function.has_configuration_attempt) {
            const NativeExecutionScope active(state.is_native_issue_active);
            const auto proof = [&] {
                const auto status = state.check_context(true);
                if (status != CUDA_SUCCESS) state.fault();
                return status;
            };
            const auto fail_preparation = [&](CUresult status) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = status;
                return result;
            };
            Implementation::FunctionCarveout* carveout{};
            if (state.dispatch.function_get_attribute) {
                auto found = std::ranges::find(module->function_carveouts, function.native,
                                               &Implementation::FunctionCarveout::native);
                if (found == module->function_carveouts.end()) {
                    std::size_t retained{};
                    for (const auto& owner : state.storage->modules)
                        retained += owner.function_carveouts.size();
                    if (retained >= state.limits.max_objects)
                        return fail_preparation(CUDA_ERROR_OUT_OF_MEMORY);
                    // Allocate before the observer/hint. A later preparation
                    // error must not make a retry recapture our physical hint.
                    module->function_carveouts.push_back({function.native, {}, false, false});
                    carveout = &module->function_carveouts.back();
                } else
                    carveout = &*found;
            }
            if (const auto status = proof(); status != CUDA_SUCCESS)
                return fail_preparation(status);
            if (carveout && !carveout->logical_value) {
                int observed{};
                const auto status = state.invoke(state.dispatch.function_get_attribute, &observed,
                                                 CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                                 function.native);
                if (const auto checked = proof(); checked != CUDA_SUCCESS)
                    return fail_preparation(checked);
                if (status != CUDA_SUCCESS) return fail_preparation(status);
                carveout->logical_value = observed;
                carveout->is_override_active = true;
            }
            // Private broadcast storage must not trigger a partition transition
            // behind an independent native helper. Keep both function domains
            // on the same physical hint while retaining the observed logical
            // value. A public setter ends that override; retries never renew it.
            if (!carveout || !carveout->is_hint_applied) {
                const auto configured =
                    state.invoke(state.dispatch.function_set_attribute, function.native,
                                 CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                 CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
                if (const auto checked = proof(); checked != CUDA_SUCCESS)
                    return fail_preparation(checked);
                if (configured != CUDA_SUCCESS) return fail_preparation(configured);
                if (carveout) carveout->is_hint_applied = true;
            }
        }
        const transform::CudaImageKernel* binary_kernel{};
        std::optional<std::size_t> count;
        if (module->metadata && !module->metadata->is_ptx_selected) {
            const auto found = std::ranges::find(module->metadata->kernels, name,
                                                 &transform::CudaImageKernel::name);
            if (found != module->metadata->kernels.end()) {
                binary_kernel = &*found;
                count = found->arguments.size();
            }
        } else
            count = parameter_count(PtxTokens{module->source}, name, state.limits.max_parameters);
        if (!count) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_NOT_SUPPORTED;
            return result;
        }
        if (module->metadata) {
            const auto alternative = std::ranges::find(module->metadata->ptx_arities, name,
                                                       &transform::CudaImagePtxArity::name);
            if (alternative != module->metadata->ptx_arities.end() &&
                (alternative->has_conflicting_counts || alternative->parameter_count != *count)) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_INVALID_VALUE;
                return result;
            }
            for (const auto& source : module->metadata->validation_ptx) {
                const auto alternative =
                    parameter_count(PtxTokens{source}, name, state.limits.max_parameters);
                if (alternative && *alternative != *count) {
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = CUDA_ERROR_INVALID_VALUE;
                    return result;
                }
            }
        }
        for (std::size_t index = 0; index < *count; ++index) {
            std::size_t offset{}, bytes{};
            const auto status = state.invoke(state.dispatch.parameter_info, function.native, index,
                                             &offset, &bytes);
            if (state.is_faulted || status != CUDA_SUCCESS || !bytes ||
                offset < function.parameter_bytes ||
                offset > state.limits.submissions.max_parameter_bytes ||
                bytes > state.limits.submissions.max_parameter_bytes - offset ||
                (binary_kernel && (offset != binary_kernel->arguments[index].offset ||
                                   bytes != binary_kernel->arguments[index].bytes))) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = state.is_faulted         ? CUDA_ERROR_UNKNOWN
                                : status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE
                                                         : status;
                return result;
            }
            function.arguments.push_back({offset, bytes});
            function.parameter_bytes = offset + bytes;
        }
        if (!state.has_graph_capture() && module->transformed && !function.is_persistent_disabled &&
            !module->binary.empty() && !module->has_preparation_attempt) {
            module->has_preparation_attempt = true;
            const auto status =
                state.reserve_static(module->prepared_static_budget, module->prepared_static_bytes)
                    ? state.load_module_preparation(*module, &module->prepared_native,
                                                    module->transformed->image.c_str(),
                                                    CudaPreparationPhase::kLoadPrepared)
                    : CUDA_ERROR_OUT_OF_MEMORY;
            if (state.is_faulted) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            if (status != CUDA_SUCCESS || !module->prepared_native) {
                if (!module->prepared_native && status != CUDA_SUCCESS) {
                    state.release_static(module->prepared_static_bytes);
                    // Optional JIT failed before returning any private handle.
                    // Original module/function stay native; never retry this JIT.
                    const auto released = module->transformed->image.size() + 1;
                    module->charged_bytes -= released;
                    state.storage->image_bytes -= released;
                    module->transformed.reset();
                } else {
                    state.fault();
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
                    return result;
                }
            }
        }
        if (!state.has_graph_capture() && module->transformed && !function.is_persistent_disabled) {
            for (const auto& variant : module->transformed->kernels) {
                if (variant.original_name != name) continue;
                const auto status = state.prepare_persistent_function(function, variant);
                if (status != CUDA_SUCCESS) {
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = status;
                    return result;
                }
            }
        }
        if (state.profile_cache && !function.has_configuration_attempt)
            function.profile = state.profile_cache->bind(
                state.profile_platform,
                module->profile_source.empty() ? module->source : module->profile_source,
                module->binary, function.name);
        function.id = state.next_id;
        function.module = module_id;
        if (!state.executor->bind_function(function.id, function.native, function.parameter_bytes,
                                           state.storage)) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        result.id = state.consume_id();
        state.storage->functions.push_back(std::move(function));
        result.parameter_bytes = state.storage->functions.back().parameter_bytes;
        result.arguments = state.storage->functions.back().arguments;
        return result;
    } catch (...) {
        auto result = rejected(CUDA_ERROR_OUT_OF_MEMORY);
        result.original_status = original_result;
        return result;
    }
}

CudaPointerMetadataResult CudaExecutionSession::pointer_metadata(
    const CudaPointerQuery& query, CudaExecutionSession* querying, CudaExecutionSession* source,
    std::optional<CudaHostMemoryView> host) noexcept {
    auto& carrier = *implementation_;
    CudaPointerMetadataResult result;
    const auto reject = [&](CUresult status) {
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = status;
        result.observation.reset();
        return result;
    };
    std::array<Implementation*, 3> owners{};
    std::size_t count{};
    for (auto* candidate :
         std::array{&carrier, querying ? querying->implementation_.get() : nullptr,
                    source ? source->implementation_.get() : nullptr}) {
        if (!candidate || std::ranges::find(owners, candidate) != owners.end()) continue;
        if (!candidate->owns() || candidate->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_CONTEXT);
        if (candidate->is_native_issue_active || candidate->is_folded_active ||
            candidate->is_rnn_query_active) {
            result.disposition = CudaSessionDisposition::kDeferred;
            result.status = CUDA_ERROR_NOT_READY;
            return result;
        }
        if (candidate->is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
        if (candidate->is_faulted ||
            (candidate->executor && candidate->executor->snapshot().is_quarantined))
            return reject(CUDA_ERROR_UNKNOWN);
        owners[count++] = candidate;
    }
    std::array<std::optional<NativeExecutionScope>, 3> scopes;
    for (std::size_t index = 0; index < count; ++index)
        scopes[index].emplace(owners[index]->is_native_issue_active);
    const auto fault = [&](CUresult status) {
        result.context_selection = CudaPointerContextSelection::kUnproven;
        for (std::size_t index = 0; index < count; ++index) owners[index]->fault();
        return reject(status);
    };
    const bool is_driver = query.domain == CudaPointerDomain::kDriver;
    if ((!is_driver && query.domain != CudaPointerDomain::kRuntime) ||
        (is_driver ? (query.selector != 3 && query.selector != 9 &&
                      !(query.selector == 1 && query.kind == CudaPointerKind::kProcessDevice))
                   : query.selector != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    if (querying &&
        (!querying->implementation_->storage->context || !querying->implementation_->executor))
        return reject(CUDA_ERROR_INVALID_CONTEXT);
    CUdeviceptr input{}, device_alias{}, host_alias{};
    std::shared_ptr<const void> process_input;
    switch (query.kind) {
        case CudaPointerKind::kNull:
        case CudaPointerKind::kUnregistered:
            if (source || host || query.range.allocation || query.range.offset)
                return reject(CUDA_ERROR_INVALID_VALUE);
            if (query.kind == CudaPointerKind::kUnregistered)
                input = reinterpret_cast<CUdeviceptr>(&carrier.pointer_metadata_sentinel);
            host_alias = input;
            break;
        case CudaPointerKind::kDevice:
        case CudaPointerKind::kProcessDevice: {
            if (!source || host) return reject(CUDA_ERROR_INVALID_VALUE);
            auto& owner = *source->implementation_;
            const bool is_process = query.kind == CudaPointerKind::kProcessDevice;
            if (is_process
                    ? (!owner.process_memory || !(query.range.allocation & (UINT64_C(1) << 63)))
                    : (!owner.storage->context || !owner.executor))
                return reject(CUDA_ERROR_INVALID_CONTEXT);
            const auto found = owner.find_allocation(query.range.allocation);
            if (!found || query.range.offset >= found->bytes ||
                query.range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address ||
                (found->module && !std::ranges::any_of(owner.storage->modules, [&](const auto& m) {
                     return m.id == found->module && m.native;
                 })))
                return reject(CUDA_ERROR_INVALID_VALUE);
            input = device_alias = found->address + query.range.offset;
            if (is_process) {
                auto borrow = owner.process_memory->owner()->borrow(
                    query.range.allocation & ~(UINT64_C(1) << 63), query.range.offset, 1);
                if (borrow.status != CUDA_SUCCESS) return reject(borrow.status);
                process_input = std::move(borrow.lifetime);
            }
            break;
        }
        case CudaPointerKind::kHost:
            if (source || query.range.allocation || query.range.offset || !host || !host->pointer ||
                !host->address || !host->bytes || !host->lifetime)
                return reject(CUDA_ERROR_INVALID_VALUE);
            input = host_alias = reinterpret_cast<CUdeviceptr>(host->pointer);
            device_alias = host->address;
            break;
        default:
            return reject(CUDA_ERROR_INVALID_VALUE);
    }
    const auto& sdk = carrier.dispatch;
    if ((is_driver && !sdk.pointer_get_attribute) ||
        (!is_driver && (!sdk.runtime.pointer_attributes || !sdk.runtime.version)))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // The initial context is deliberately not inferred from any resource owner.
    // cuInit may be the first Driver operation; NOT_INITIALIZED is a permitted
    // initial observation, not permission to skip all later context proofs.
    CUcontext before{};
    auto status = sdk.execution.context_get_current(&before);
    const bool was_uninitialized = status == CUDA_ERROR_NOT_INITIALIZED;
    if (status != CUDA_SUCCESS && !was_uninitialized) return fault(status);
    if (was_uninitialized && before) return fault(CUDA_ERROR_UNKNOWN);
    const auto initialized = sdk.initialize(0);
    CUcontext after{};
    status = sdk.execution.context_get_current(&after);
    if (initialized != CUDA_SUCCESS && was_uninitialized && status == CUDA_ERROR_NOT_INITIALIZED &&
        !after)
        return reject(initialized);
    if (status != CUDA_SUCCESS) return fault(status);
    if (after != before) return fault(CUDA_ERROR_INVALID_CONTEXT);
    if (initialized != CUDA_SUCCESS) return reject(initialized);
    const auto prove = [&](CUcontext expected) {
        CUcontext current{};
        const auto observed = sdk.execution.context_get_current(&current);
        return observed != CUDA_SUCCESS ? observed
               : current != expected    ? CUDA_ERROR_INVALID_CONTEXT
                                        : CUDA_SUCCESS;
    };
    if (const auto observed = prove(before); observed != CUDA_SUCCESS) return fault(observed);
    const auto desired = querying ? querying->implementation_->storage->context : nullptr;
    const auto selected = sdk.context_set_current(desired);
    if (const auto observed = prove(selected == CUDA_SUCCESS ? desired : before);
        observed != CUDA_SUCCESS)
        return fault(observed);
    if (selected != CUDA_SUCCESS) return reject(selected);
    result.context_selection = CudaPointerContextSelection::kQuery;
    if (!is_driver) {
        if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
        int version{};
        const auto version_status = sdk.runtime.version(&version);
        if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
        if (version_status != cudaSuccess || version != 12060 || CUDART_VERSION != 12060 ||
            sizeof(void*) != 8)
            return reject(CUDA_ERROR_NOT_SUPPORTED);
    }
    CUdeviceptr poison = std::numeric_limits<CUdeviceptr>::max();
    while (!poison || poison == input || poison == device_alias || poison == host_alias) --poison;
    constexpr auto scalar_poison = std::numeric_limits<std::int32_t>::min();
    static_assert(sizeof(int) == sizeof(std::int32_t));
    static_assert(sizeof(cudaMemoryType) == sizeof(std::int32_t));
    CUdeviceptr device_value = poison;
    int ordinal = scalar_poison;
    cudaPointerAttributes attributes{};
    std::memcpy(&attributes.type, &scalar_poison, sizeof(scalar_poison));
    attributes.device = scalar_poison;
    // These deliberately invalid output sentinels are compared as bits only;
    // neither the framework nor the original metadata getter dereferences them.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    attributes.devicePointer = reinterpret_cast<void*>(poison);
    attributes.hostPointer = reinterpret_cast<void*>(poison);
    // NOLINTEND(performance-no-int-to-ptr)
    if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
    if (is_driver) {
        void* output = query.selector == 3 || query.selector == 1
                           ? static_cast<void*>(&device_value)
                           : static_cast<void*>(&ordinal);
        result.original_status = static_cast<std::int32_t>(sdk.pointer_get_attribute(
            output, static_cast<CUpointer_attribute>(query.selector), input));
    } else {
        // The Runtime ABI takes an address instead of Driver CUdeviceptr bits.
        // This validated source/sentinel is observed, never read as a pointee.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* runtime_input = reinterpret_cast<const void*>(input);
        result.original_status =
            static_cast<std::int32_t>(sdk.runtime.pointer_attributes(&attributes, runtime_input));
    }
    if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
    CudaPointerObservation observation;
    enum class PointerField : std::uint32_t { kDevice = 4, kHost = 8 };
    const auto project_alias = [&](CUdeviceptr value, CUdeviceptr expected, PointerField effect,
                                   CudaPointerAlias& output) {
        if (value == poison) return true;
        if (value && (!expected || value != expected)) return false;
        observation.effects |= static_cast<std::uint32_t>(effect);
        output = value ? CudaPointerAlias::kInput : CudaPointerAlias::kNull;
        return true;
    };
    const bool succeeded = *result.original_status == 0;
    if (is_driver && query.selector == 1) {
        if (!project_alias(device_value, 0, PointerField::kDevice, observation.device_alias) ||
            (succeeded && observation.effects != 4))
            return fault(CUDA_ERROR_UNKNOWN);
    } else if (is_driver && query.selector == 3) {
        if (!project_alias(device_value, device_alias, PointerField::kDevice,
                           observation.device_alias) ||
            (succeeded &&
             (observation.effects != 4 || observation.device_alias != CudaPointerAlias::kInput)))
            return fault(CUDA_ERROR_UNKNOWN);
    } else if (is_driver) {
        if (ordinal != scalar_poison) {
            observation.effects = 2;
            observation.device = ordinal;
        }
        if (succeeded && (observation.effects != 2 || ordinal < 0 ||
                          (query.kind != CudaPointerKind::kDevice &&
                           query.kind != CudaPointerKind::kProcessDevice &&
                           query.kind != CudaPointerKind::kHost)))
            return fault(CUDA_ERROR_UNKNOWN);
    } else {
        std::int32_t type{};
        std::memcpy(&type, &attributes.type, sizeof(type));
        if (type != scalar_poison) {
            observation.effects |= 1;
            observation.type = type;
        }
        if (attributes.device != scalar_poison) {
            observation.effects |= 2;
            observation.device = attributes.device;
        }
        if (!project_alias(reinterpret_cast<CUdeviceptr>(attributes.devicePointer), device_alias,
                           PointerField::kDevice, observation.device_alias) ||
            !project_alias(reinterpret_cast<CUdeviceptr>(attributes.hostPointer), host_alias,
                           PointerField::kHost, observation.host_alias))
            return fault(CUDA_ERROR_UNKNOWN);
        const int expected_type =
            query.kind == CudaPointerKind::kDevice || query.kind == CudaPointerKind::kProcessDevice
                ? 2
            : query.kind == CudaPointerKind::kHost ? 1
                                                   : 0;
        if (succeeded &&
            (observation.effects != 15 || type != expected_type ||
             (expected_type ? attributes.device < 0 : attributes.device != -2) ||
             (!expected_type && reinterpret_cast<CUdeviceptr>(attributes.hostPointer) != input)))
            return fault(CUDA_ERROR_UNKNOWN);
    }
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = CUDA_SUCCESS;
    result.observation = observation;
    return result;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): distinct wire fields retain their typed ABI.
CudaFunctionControlResult CudaExecutionSession::function_control(CudaFunctionControl operation,
                                                                 std::uint64_t function_id,
                                                                 std::uint32_t argument,
                                                                 std::int32_t value) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    CudaFunctionControlResult result;
    const auto reject = [&](CUresult status) {
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = status;
        result.value = 0;
        result.attributes.reset();
        return result;
    };
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        result.disposition = CudaSessionDisposition::kDeferred;
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    if (operation == CudaFunctionControl::kGetAttributes && (argument != 0 || value != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    // Acquire exclusion before even the first context observation: an injected
    // Driver callback must not destroy owners or consume another queued call.
    const NativeExecutionScope active(state.is_native_issue_active);
    const auto proof = [&] {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (state.is_faulted || !state.executor || state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    using Operation = CudaFunctionControl;
    const bool is_query =
        operation == Operation::kGetAttribute || operation == Operation::kGetAttributes;
    if ((operation != Operation::kGetAttribute && operation != Operation::kSetAttribute &&
         operation != Operation::kSetCache && operation != Operation::kGetAttributes) ||
        (operation == Operation::kSetCache ? argument > 3 : argument > 31) ||
        (operation != Operation::kSetAttribute && value != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto function =
        std::ranges::find(state.storage->functions, function_id, &Implementation::Function::id);
    if (function == state.storage->functions.end() || !function->native ||
        !std::ranges::any_of(state.storage->modules, [&](const auto& module) {
            return module.id == function->module && module.native;
        }))
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if ((is_query && !state.dispatch.function_get_attribute) ||
        (operation == Operation::kSetAttribute && !state.dispatch.function_set_attribute) ||
        (operation == Operation::kSetCache && !state.dispatch.function_set_cache))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // The outer scope already owns cold_gate's exclusion. Only mutations need
    // its physical-quiescence condition; a getter never drains pending work.
    if (!is_query && !state.executor->resources_quiescent()) {
        result.disposition = CudaSessionDisposition::kDeferred;
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    if (operation == Operation::kGetAttributes) {
        constexpr std::array attributes{CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                        CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        CU_FUNC_ATTRIBUTE_PTX_VERSION,
                                        CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                                        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                        CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
                                        CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                                        CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED};
        CudaFunctionAttributes observed;
        for (std::size_t index = 0; index < attributes.size(); ++index) {
            // The initial proof covers the first callback. Later callbacks also
            // have their own pre-proof; the whole aggregate holds one exclusion.
            if (index != 0)
                if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
            int field{};
            result.original_status = state.invoke(state.dispatch.function_get_attribute, &field,
                                                  attributes[index], function->native);
            result.disposition = CudaSessionDisposition::kComplete;
            result.status = CUDA_SUCCESS;
            if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
            if (*result.original_status != CUDA_SUCCESS) return result;
            if (index < observed.sizes.size()) {
                if (field < 0) {
                    state.fault();
                    return reject(CUDA_ERROR_UNKNOWN);
                }
                observed.sizes[index] = static_cast<std::uint64_t>(field);
            } else {
                observed.values[index - observed.sizes.size()] = field;
            }
        }
        const auto module = std::ranges::find(state.storage->modules, function->module,
                                              &Implementation::Module::id);
        const auto carveout = std::ranges::find(module->function_carveouts, function->native,
                                                &Implementation::FunctionCarveout::native);
        if (carveout != module->function_carveouts.end() && carveout->is_override_active &&
            carveout->logical_value)
            observed.values[6] = *carveout->logical_value;
        result.attributes = observed;
        return result;
    }
    std::vector<Implementation::Function*> aliases;
    std::vector<CUfunction> private_targets;
    bool is_persistent_disabled{};
    if (!is_query) {
        try {
            // Existing owner/variant storage bounds these temporary lists.
            // Finish every allocation before the requested native mutation.
            std::size_t alias_count{}, private_count{};
            for (const auto& alias : state.storage->functions) {
                if (alias.module != function->module || alias.native != function->native) continue;
                if (alias_count == aliases.max_size() ||
                    alias.variants.size() > private_targets.max_size() - private_count)
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                ++alias_count;
                private_count += alias.variants.size();
                is_persistent_disabled = is_persistent_disabled || alias.is_persistent_disabled;
            }
            aliases.reserve(alias_count);
            if (!is_persistent_disabled) private_targets.reserve(private_count);
            for (auto& alias : state.storage->functions) {
                if (alias.module != function->module || alias.native != function->native) continue;
                aliases.push_back(&alias);
                if (is_persistent_disabled) continue;
                for (const auto& variant : alias.variants) {
                    if (!variant.native) {
                        state.fault();
                        return reject(CUDA_ERROR_UNKNOWN);
                    }
                    if (variant.native != function->native &&
                        std::ranges::find(private_targets, variant.native) == private_targets.end())
                        private_targets.push_back(variant.native);
                }
            }
        } catch (...) {
            return reject(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    const auto disable_private = [&] {
        is_persistent_disabled = true;
        for (auto* alias : aliases) alias->is_persistent_disabled = true;
    };
    const auto configure = [&](CUfunction native) {
        return operation == Operation::kSetAttribute
                   ? state.invoke(state.dispatch.function_set_attribute, native,
                                  static_cast<CUfunction_attribute>(argument), value)
                   : state.invoke(state.dispatch.function_set_cache, native,
                                  static_cast<CUfunc_cache>(argument));
    };
    int observed{};
    if (!is_query) {
        if (state.control_epoch == UINT64_MAX) return reject(CUDA_ERROR_OUT_OF_MEMORY);
        ++state.control_epoch;
        for (auto* alias : aliases) {
            alias->has_configuration_attempt = true;
            alias->configuration.reset();
        }
        if (is_persistent_disabled) disable_private();
    }
    result.original_status =
        is_query ? state.invoke(state.dispatch.function_get_attribute, &observed,
                                static_cast<CUfunction_attribute>(argument), function->native)
                 : configure(function->native);
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = CUDA_SUCCESS;
    if (!is_query && *result.original_status != CUDA_SUCCESS) disable_private();
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (*result.original_status != CUDA_SUCCESS) return result;
    const auto module =
        std::ranges::find(state.storage->modules, function->module, &Implementation::Module::id);
    const auto carveout = std::ranges::find(module->function_carveouts, function->native,
                                            &Implementation::FunctionCarveout::native);
    if (is_query) {
        result.value = argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
                               carveout != module->function_carveouts.end() &&
                               carveout->is_override_active && carveout->logical_value
                           ? *carveout->logical_value
                           : observed;
        return result;
    }
    if (operation == Operation::kSetAttribute &&
        argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
        carveout != module->function_carveouts.end())
        carveout->is_override_active = false;
    // An alias may lack complete historical configuration. A later setter
    // cannot restore that proof for the group, even if this setting succeeds.
    if (is_persistent_disabled) return result;
    for (const auto native : private_targets) {
        const auto configured = configure(native);
        if (configured != CUDA_SUCCESS) disable_private();
        if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
        if (configured != CUDA_SUCCESS) {
            // Preserve requested original success. Future admission uses the
            // original function, never a partly configured private replacement.
            return result;
        }
    }
    if (operation == Operation::kSetCache)
        for (auto* alias : aliases) alias->cache_preference = argument + 1;
    if (operation == Operation::kSetAttribute && argument >= 16)
        for (auto* alias : aliases) alias->has_unknown_configuration = true;
    return result;
}

CudaSessionResult CudaExecutionSession::unload_module(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto found = std::ranges::find_if(state.storage->modules,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->modules.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_module(id)) return deferred_resource_release();
    for (const auto& function : state.storage->functions)
        if (function.module == id && !state.executor->unbind_function(function.id)) {
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
    if (found->prepared_native) {
        const auto status = state.invoke(state.dispatch.module_unload, found->prepared_native);
        if (status != CUDA_SUCCESS) {
            state.fault();
            return rejected(status);  // Original target unload was not called.
        }
        found->prepared_native = nullptr;
        state.release_static(found->prepared_static_bytes);
    }
    auto result = completed(state.invoke(state.dispatch.module_unload, found->native));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    std::erase_if(state.storage->functions,
                  [id](const auto& function) { return function.module == id; });
    std::erase_if(state.storage->allocations,
                  [&](const auto& allocation) { return allocation.module == id; });
    state.release_static(found->native_static_bytes);
    state.storage->image_bytes -= found->charged_bytes;
    state.storage->modules.erase(found);
    return result;
}

CudaSessionResult CudaExecutionSession::get_global(std::uint64_t module_id,
                                                   std::string_view name) noexcept {
    auto& state = *implementation_;
    const auto context = state.check_context();
    if (context != CUDA_SUCCESS) return rejected(context);
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return rejected(CUDA_ERROR_UNKNOWN);
    if (!state.dispatch.module_global) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto module =
        std::ranges::find(state.storage->modules, module_id, &Implementation::Module::id);
    if (module == state.storage->modules.end() || name.empty() || name.size() > 4096 ||
        name.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    try {
        const std::string symbol{name};
        CUdeviceptr address{};
        std::size_t bytes{};
        auto result = completed(state.invoke(state.dispatch.module_global, &address, &bytes,
                                             module->native, symbol.c_str()));
        const auto after = state.check_context();
        if (after != CUDA_SUCCESS) {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = after;
            return result;
        }
        if (result.status != CUDA_SUCCESS) return result;
        if (!address || !bytes || bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address) {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        auto existing =
            std::ranges::find_if(state.storage->allocations, [&](const auto& allocation) {
                return allocation.address == address && allocation.bytes == bytes &&
                       allocation.module == module_id;
            });
        if (existing != state.storage->allocations.end()) {
            result.id = existing->id;
        } else {
            const bool overlap =
                std::ranges::any_of(state.storage->allocations, [&](const auto& allocation) {
                    return address >= allocation.address
                               ? address - allocation.address < allocation.bytes
                               : allocation.address - address < bytes;
                });
            if (overlap) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_NOT_SUPPORTED;
                return result;
            }
            const auto maximum = std::numeric_limits<std::size_t>::max();
            bool has_overflow = bytes > maximum - module->discovered_static_bytes;
            module->discovered_static_bytes =
                has_overflow ? maximum : module->discovered_static_bytes + bytes;
            const auto new_charge =
                std::max(module->native_static_bytes, module->discovered_static_bytes);
            const auto extra = new_charge - module->native_static_bytes;
            if (!state.reserve_domain_bytes(extra)) {
                if (state.storage->domain_member)
                    state.storage->domain_member.retain_uncertain_device_bytes(extra);
                if (state.process_memory)
                    (void)state.process_memory->owner()->quarantine(CUDA_ERROR_OUT_OF_MEMORY);
                has_overflow = true;
            }
            const auto used = state.storage->device_bytes;
            has_overflow |= extra > maximum - used;
            state.storage->device_bytes = extra > maximum - used ? maximum : used + extra;
            module->native_static_bytes = new_charge;
            if (!state.has_id_space ||
                state.storage->allocations.size() == state.limits.max_objects) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            const auto id = state.consume_id();
            state.storage->allocations.push_back({id, address, bytes, module_id});
            if (has_overflow || extra > state.limits.max_device_bytes -
                                            std::min(used, state.limits.max_device_bytes)) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            result.id = id;
        }
        result.address = address;
        result.parameter_bytes = bytes;
        return result;
    } catch (const std::bad_alloc&) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
}
CudaSessionResult CudaExecutionSession::allocate(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (!bytes) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->allocations.size() == state.limits.max_objects ||
        bytes > state.limits.max_device_bytes - state.storage->device_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (!state.reserve_domain_bytes(bytes)) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    Implementation::Allocation allocation{.id = state.next_id, .address = 0, .bytes = bytes};
    auto result = completed(state.invoke(state.dispatch.allocate, &allocation.address, bytes));
    if (result.status != CUDA_SUCCESS && !allocation.address &&
        (!state.uses_shared_domain || state.clean_shared_allocation_rejection(result.status))) {
        state.release_domain_bytes(bytes);
        return result;
    }
    const bool is_valid = result.status == CUDA_SUCCESS && !state.is_faulted &&
                          allocation.address &&
                          bytes - 1 <= std::numeric_limits<CUdeviceptr>::max() - allocation.address;
    (void)state.consume_id();
    state.storage->device_bytes += bytes;
    state.storage->allocations.push_back(allocation);
    if (is_valid) {
        result.id = allocation.id;
        result.address = allocation.address;
    } else {
        state.fault();
        if (result.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        }
    }
    return result;
}
CudaSessionResult CudaExecutionSession::release(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (state.process_memory && (id & (UINT64_C(1) << 63))) {
        const auto observed = state.process_memory->release(id);
        auto result = observed.original_status
                          ? completed(static_cast<CUresult>(*observed.original_status))
                          : rejected(observed.status);
        result.status = observed.status;
        if (observed.status == CUDA_ERROR_NOT_READY && !observed.original_status)
            result.disposition = CudaSessionDisposition::kDeferred;
        if (observed.is_quarantined) state.fault();
        return result;
    }
    const auto found = std::ranges::find_if(state.storage->allocations,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->allocations.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->module) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_allocation(id)) return deferred_resource_release();
    auto result = completed(state.invoke(state.dispatch.release, found->address));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->device_bytes -= found->bytes;
    state.release_domain_bytes(found->bytes);
    state.storage->allocations.erase(found);
    return result;
}
CudaSessionResult CudaExecutionSession::upload(CudaAllocationRange range,
                                               std::span<const std::byte> bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes.size() > found->bytes - range.offset ||
        (!bytes.empty() && !bytes.data()))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (bytes.empty()) return settled();
    if (state.storage->default_proxy) {
        if (bytes.size() > state.stream_config.maximum_staging_bytes - state.storage->staging_bytes)
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        try {
            // Page-locked/pageable SDK staging may outlive its host return.
            // Own the exact bytes until the real legacy-to-proxy tail stops.
            auto staging = std::make_shared<Implementation::Staging>(state.storage, bytes.size());
            if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(range, bytes.size());
                if (!view) return rejected(CUDA_ERROR_INVALID_VALUE);
                staging->process_lifetime = view->lifetime;
            }
            std::copy(bytes.begin(), bytes.end(), staging->data.get());
            const auto address = found->address + range.offset;
            const std::function<CUresult()> call = [sdk = state.dispatch.upload, address, staging] {
                return sdk(address, staging->data.get(), staging->bytes);
            };
            const NativeExecutionScope scope(state.is_native_issue_active);
            if (!state.reserve_domain_physical()) return rejected(CUDA_ERROR_UNKNOWN);
            const auto observed = state.executor->execute_synchronous_legacy(call, staging);
            state.update_domain_physical();
            auto result = observed.original_status ? completed(*observed.original_status)
                                                   : rejected(observed.status);
            result.frontier = observed.frontier;
            if (observed.status != CUDA_SUCCESS) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = observed.status;
            }
            return result;
        } catch (...) {
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    return completed(state.invoke(state.dispatch.upload, found->address + range.offset,
                                  bytes.data(), bytes.size()));
}
CudaSessionResult CudaExecutionSession::download(CudaAllocationRange range,
                                                 std::span<std::byte> bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes.size() > found->bytes - range.offset ||
        (!bytes.empty() && !bytes.data()))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (bytes.empty()) return settled();
    if (state.storage->default_proxy) {
        if (bytes.size() > state.stream_config.maximum_staging_bytes - state.storage->staging_bytes)
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        try {
            auto staging = std::make_shared<Implementation::Staging>(state.storage, bytes.size());
            if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(range, bytes.size());
                if (!view) return rejected(CUDA_ERROR_INVALID_VALUE);
                staging->process_lifetime = view->lifetime;
            }
            const auto address = found->address + range.offset;
            const std::function<CUresult()> call = [sdk = state.dispatch.download, address,
                                                    staging] {
                return sdk(staging->data.get(), address, staging->bytes);
            };
            const NativeExecutionScope scope(state.is_native_issue_active);
            if (!state.reserve_domain_physical()) return rejected(CUDA_ERROR_UNKNOWN);
            const auto observed = state.executor->execute_synchronous_legacy(call, staging);
            state.update_domain_physical();
            auto result = observed.original_status ? completed(*observed.original_status)
                                                   : rejected(observed.status);
            result.frontier = observed.frontier;
            if (observed.status != CUDA_SUCCESS) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = observed.status;
            } else if (observed.original_status == CUDA_SUCCESS) {
                // The requested DtoH call itself supplies host-completion;
                // the private event preserves subsequent stream ordering.
                std::copy_n(staging->data.get(), staging->bytes, bytes.begin());
            }
            return result;
        } catch (...) {
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    return completed(state.invoke(state.dispatch.download, bytes.data(),
                                  found->address + range.offset, bytes.size()));
}

std::optional<CudaDeviceMemoryView> CudaExecutionSession::device_memory(
    CudaAllocationRange range, std::size_t bytes) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.is_faulted || state.is_closing)
        return std::nullopt;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes > found->bytes - range.offset ||
        range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
        return std::nullopt;
    if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
        auto captured = state.process_memory->owner()->borrow(
            range.allocation & ~(UINT64_C(1) << 63), range.offset, bytes);
        if (captured.status != CUDA_SUCCESS) return std::nullopt;
        return CudaDeviceMemoryView{captured.address, bytes, std::move(captured.lifetime)};
    }
    return CudaDeviceMemoryView{found->address + range.offset, bytes, state.storage};
}

CudaSessionExternalContextCapture CudaExecutionSession::capture_external_context() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return {.status = status};
    std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
    if (state.external_contexts.size() >= state.limits.max_objects)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        CudaSessionExternalContext context;
        context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        state.external_contexts.push_back(context.state_);
        return {.status = CUDA_SUCCESS, .context = std::move(context)};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}

CudaSessionExternalCommandCapture CudaExecutionSession::capture_external_command(
    std::uint64_t stream, std::span<const CudaSessionExternalRangeRequest> ranges) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return {.status = status};
    if (ranges.size() > state.limits.max_parameters || (!ranges.empty() && !ranges.data()))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    CUstream native_stream = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
        native_stream = found->native;
    }
    std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
    std::erase_if(state.external_commands, [](const auto& entry) { return entry.expired(); });
    if (state.external_contexts.size() >= state.limits.max_objects ||
        state.external_commands.size() >= state.limits.submissions.max_commands)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        auto command = std::make_shared<CudaSessionExternalCommand::State>();
        command->stream_id = stream;
        command->stream = native_stream;
        command->ranges.reserve(ranges.size());
        command->parent_modules.reserve(ranges.size());
        for (const auto& request : ranges) {
            const auto found = state.find_allocation(request.range.allocation);
            if (!found || request.range.offset > found->bytes ||
                request.bytes > found->bytes - request.range.offset ||
                request.range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
                return {.status = CUDA_ERROR_INVALID_VALUE};
            const auto address = found->address + request.range.offset;
            if (request.bytes &&
                request.bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address)
                return {.status = CUDA_ERROR_INVALID_VALUE};
            command->ranges.push_back({request.range, address, request.bytes});
            if (state.process_memory && (request.range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(request.range, request.bytes);
                if (!view) return {.status = CUDA_ERROR_INVALID_VALUE};
                command->process_lifetimes.push_back(view->lifetime);
            }
            if (found->module) command->parent_modules.push_back(found->module);
        }
        command->context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        // Capacity is reserved at construction. Publish only after all resource
        // checks/allocation succeed; the two weak insertions cannot allocate.
        state.external_contexts.push_back(command->context.state_);
        state.external_commands.push_back(command);
        CudaSessionExternalCommand capsule;
        capsule.state_ = std::move(command);
        return {.status = CUDA_SUCCESS, .command = std::move(capsule)};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}

CudaExecutionExternalAcceptance CudaExecutionSession::accept_external_barrier(
    std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_external_barrier(stream, lifetime);
}
CudaExecutionExternalCohortAcceptance CudaExecutionSession::accept_external_cohort(
    std::span<const std::uint64_t> streams, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_external_cohort(streams, lifetime);
}
CUresult CudaExecutionSession::external_barrier_ready(
    const CudaExecutionExternalBarrier& barrier) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    return state.executor->external_barrier_ready(barrier);
}
CUresult CudaExecutionSession::begin_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (!state.reserve_domain_physical()) return CUDA_ERROR_UNKNOWN;
    const auto result = state.executor->begin_external_barrier(barrier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::record_external_enqueue(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->record_external_enqueue(barrier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::enqueue_host_function(
    const CudaExecutionExternalBarrier& barrier, std::uint64_t stream,
    const std::shared_ptr<CudaSessionHostFunction>& callback) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (!callback || !state.dispatch.launch_host_function) return CUDA_ERROR_NOT_SUPPORTED;
    CUstream native = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return CUDA_ERROR_INVALID_HANDLE;
        native = found->native;
    }
    const auto begun = begin_external_barrier(barrier);
    if (begun != CUDA_SUCCESS) return begun;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto original = state.dispatch.launch_host_function(
        native, &CudaSessionHostFunction::wait, callback->state_.get());
    const auto proof = state.check_context(true);
    if (original != CUDA_SUCCESS || proof != CUDA_SUCCESS) {
        const auto status = proof != CUDA_SUCCESS ? proof : original;
        (void)state.executor->fail_external_barrier(barrier, status);
        state.fault();
        state.update_domain_physical();
        return status;
    }
    const auto recorded = state.executor->record_external_enqueue(barrier);
    if (recorded != CUDA_SUCCESS) state.fault();
    state.update_domain_physical();
    return recorded;
}
CUresult CudaExecutionSession::cancel_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    return state.executor->cancel_external_barrier(barrier);
}
CUresult CudaExecutionSession::fail_external_barrier(const CudaExecutionExternalBarrier& barrier,
                                                     CUresult failure) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->fail_external_barrier(barrier, failure);
    state.update_domain_physical();
    return result;
}

CudaExecutionEnqueueAcceptance CudaExecutionSession::accept_enqueue_turn(
    std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_enqueue_turn(stream, lifetime);
}
CudaExecutionEnqueueAcceptance CudaExecutionSession::accept_enqueue_cohort(
    std::span<const std::uint64_t> streams, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_enqueue_cohort(streams, lifetime);
}
CUresult CudaExecutionSession::enqueue_turn_ready(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    return state.executor->enqueue_turn_ready(turn);
}
CUresult CudaExecutionSession::begin_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (!state.reserve_domain_physical()) return CUDA_ERROR_UNKNOWN;
    const auto result = state.executor->begin_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CudaExecutionIssueResult CudaExecutionSession::progress_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_enqueue_dependency(turn);
    state.update_domain_physical();
    return result;
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_enqueue_dependency(turn);
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_frontier_dependency(
    const CudaExecutionFrontier& frontier) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_frontier_dependency(frontier);
}
bool CudaExecutionSession::owns_frontier(const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    return state.owns() && state.executor && state.executor->owns_frontier(frontier);
}
CudaExecutionIssueResult CudaExecutionSession::progress_frontier_dependency(
    const CudaExecutionFrontier& frontier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.state = CudaExecutionIssueState::kError, .native_status = status};
    if (state.is_faulted)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_frontier_dependency(frontier);
    state.update_domain_physical();
    return result;
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_frontier_retirement(
    const CudaExecutionFrontier& frontier) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_frontier_retirement(frontier);
}
CudaExecutionIssueResult CudaExecutionSession::progress_frontier_retirement(
    const CudaExecutionFrontier& frontier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.state = CudaExecutionIssueState::kError, .native_status = status};
    if (state.is_faulted)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_frontier_retirement(frontier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::finish_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->finish_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::cancel_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->cancel_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::fail_enqueue_turn(const CudaExecutionEnqueueTurn& turn,
                                                 CUresult failure) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->fail_enqueue_turn(turn, failure);
    state.update_domain_physical();
    return result;
}

CUresult CudaExecutionSession::prepare_external_bridge(
    CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.prepare_source();
}
CUresult CudaExecutionSession::join_external_bridge(CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.join_source();
}
CUresult CudaExecutionSession::cleanup_external_bridge(
    CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.cleanup_source();
}

CudaRuntimeMetadataResult CudaExecutionSession::runtime_metadata(
    const CudaRuntimeMetadataQuery& query) noexcept {
    auto& state = *implementation_;
    CudaRuntimeMetadataResult result;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) {
        result.status = checked;
        return result;
    }
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return result;
    const auto& sdk = state.dispatch.runtime;
    if ((query.kind == CudaRuntimeMetadataKind::kAttribute &&
         (query.attribute < 0 || query.attribute >= static_cast<int>(cudaDevAttrMax))) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAttribute &&
         (query.attribute < 0 || query.attribute > 7))) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    if (!sdk.version || (query.kind == CudaRuntimeMetadataKind::kProperties && !sdk.properties) ||
        (query.kind == CudaRuntimeMetadataKind::kAttribute && !sdk.attribute) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAttribute && !sdk.peer_attribute) ||
        (query.kind == CudaRuntimeMetadataKind::kStreamPriorityRange &&
         !sdk.stream_priority_range) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAccess && !sdk.peer_access) ||
        (query.kind == CudaRuntimeMetadataKind::kMemoryInfo && !sdk.memory_info)) {
        result.status = CUDA_ERROR_NOT_SUPPORTED;
        return result;
    }
    try {
        // Fixed CUDA-12.6 SDK ABI: scalar/array properties only, no pointers or
        // resource handles. An unverified SDK cannot supply this wire profile.
        const auto version_status = state.invoke(sdk.version, &result.version);
        const auto after_version = state.check_context();
        if (after_version != CUDA_SUCCESS) {
            state.fault();
            result.status = after_version;
            return result;
        }
        if (version_status != cudaSuccess || result.version != 12060 || CUDART_VERSION != 12060 ||
            sizeof(void*) != 8) {
            result.status = CUDA_ERROR_NOT_SUPPORTED;
            return result;
        }
        cudaDeviceProp properties{};
        int value{};
        std::array<int, 2> priorities{};
        std::array<std::size_t, 2> memory{};
        const void* data{};
        std::size_t bytes{};
        switch (query.kind) {
            case CudaRuntimeMetadataKind::kProperties:
                result.original_status = state.invoke(sdk.properties, &properties, query.device);
                data = &properties;
                bytes = sizeof(properties);
                break;
            case CudaRuntimeMetadataKind::kAttribute:
                result.original_status =
                    state.invoke(sdk.attribute, &value,
                                 static_cast<cudaDeviceAttr>(query.attribute), query.device);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kPeerAttribute:
                result.original_status = state.invoke(
                    sdk.peer_attribute, &value, static_cast<cudaDeviceP2PAttr>(query.attribute),
                    query.device, query.peer);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kStreamPriorityRange:
                result.original_status =
                    state.invoke(sdk.stream_priority_range, &priorities[0], &priorities[1]);
                data = priorities.data();
                bytes = sizeof(priorities);
                break;
            case CudaRuntimeMetadataKind::kPeerAccess:
                result.original_status =
                    state.invoke(sdk.peer_access, &value, query.device, query.peer);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kMemoryInfo:
                result.original_status = state.invoke(sdk.memory_info, &memory[0], &memory[1]);
                data = memory.data();
                bytes = sizeof(memory);
                break;
            default:
                result.status = CUDA_ERROR_INVALID_VALUE;
                return result;
        }
        const auto after = state.check_context();
        if (after != CUDA_SUCCESS) {
            state.fault();
            result.status = after;
            return result;
        }
        result.disposition = CudaSessionDisposition::kComplete;
        result.status = CUDA_SUCCESS;
        if (result.original_status == cudaSuccess) {
            const auto* first = static_cast<const std::byte*>(data);
            result.data.assign(first, first + bytes);
        }
        return result;
    } catch (const std::bad_alloc&) {
        result.status = CUDA_ERROR_OUT_OF_MEMORY;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;
    } catch (...) {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;
    }
}
CudaSessionResult CudaExecutionSession::create_stream(unsigned int flags,
                                                      std::optional<int> priority) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (flags != CU_STREAM_DEFAULT && flags != CU_STREAM_NON_BLOCKING)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->streams.size() == state.limits.max_objects)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (priority && (!state.dispatch.stream_create_priority || !state.dispatch.stream_get_priority))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    int native_priority = priority.value_or(0), logical_priority = priority.value_or(0);
    if (state.stream_config.scheduling_priority) {
        const auto priorities = state.scheduling_stream_priorities(logical_priority);
        if (priorities.status != CUDA_SUCCESS) return rejected(priorities.status);
        native_priority = priorities.native;
        logical_priority = priorities.logical;
    }
    Implementation::Stream stream{
        .id = state.next_id, .native = nullptr, .is_nonblocking = flags == CU_STREAM_NON_BLOCKING};
    const auto native_flags =
        state.uses_shared_domain ? static_cast<unsigned int>(CU_STREAM_NON_BLOCKING) : flags;
    auto result =
        completed(priority || state.stream_config.scheduling_priority
                      ? state.invoke(state.dispatch.stream_create_priority, &stream.native,
                                     native_flags, native_priority)
                      : state.invoke(state.dispatch.stream_create, &stream.native, native_flags));
    if (result.status != CUDA_SUCCESS && !stream.native) {
        if (state.check_context(state.stream_config.uses_default_proxy) != CUDA_SUCCESS)
            state.fault();
        return result;
    }
    (void)state.consume_id();
    state.storage->streams.push_back(stream);
    if (state.storage->default_proxy && stream.native == state.storage->default_proxy) {
        state.storage->is_default_proxy_uncertain = true;
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    const auto after = state.check_context(state.stream_config.uses_default_proxy);
    if (after != CUDA_SUCCESS) {
        state.fault();
        auto failed = rejected(after);
        failed.original_status = result.original_status;
        return failed;
    }
    if ((priority || state.stream_config.scheduling_priority) && result.status == CUDA_SUCCESS &&
        stream.native) {
        int actual{};
        const auto queried =
            state.invoke(state.dispatch.stream_get_priority, stream.native, &actual);
        const auto restored = state.check_context(state.stream_config.uses_default_proxy);
        if (queried != CUDA_SUCCESS || restored != CUDA_SUCCESS ||
            (state.stream_config.scheduling_priority && actual != native_priority)) {
            state.fault();
            auto failed = rejected(queried != CUDA_SUCCESS    ? queried
                                   : restored != CUDA_SUCCESS ? restored
                                                              : CUDA_ERROR_UNKNOWN);
            failed.original_status = result.original_status;
            return failed;
        }
        result.address = std::bit_cast<std::uint32_t>(
            state.stream_config.scheduling_priority ? logical_priority : actual);
    }
    if (result.status == CUDA_SUCCESS && stream.native &&
        state.executor->bind_stream(stream.id, stream.native, state.storage,
                                    flags == CU_STREAM_NON_BLOCKING))
        result.id = stream.id;
    else {
        state.fault();
        if (result.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        }
    }
    return result;
}
std::optional<bool> CudaExecutionSession::stream_is_nonblocking(std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt || state.is_faulted)
        return std::nullopt;
    if (!id) return false;
    const auto stream = std::ranges::find(state.storage->streams, id, &Implementation::Stream::id);
    if (stream == state.storage->streams.end()) return std::nullopt;
    return stream->is_nonblocking;
}

CudaSessionResult CudaExecutionSession::destroy_stream(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto found = std::ranges::find_if(state.storage->streams,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->streams.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_stream(id)) return deferred_resource_release();
    if (std::ranges::any_of(state.storage->blas,
                            [id](const auto& handle) { return handle.stream == id; }))
        return rejected(CUDA_ERROR_NOT_READY);
#ifdef GLIMMER_HAS_CUDNN
    if (std::ranges::any_of(state.storage->dnn,
                            [id](const auto& handle) { return handle.stream == id; }))
        return rejected(CUDA_ERROR_NOT_READY);
#endif
    if (!state.executor->unbind_stream(id)) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
    auto result = completed(state.invoke(state.dispatch.stream_destroy, found->native));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->streams.erase(found);
    return result;
}

#ifdef GLIMMER_HAS_CUDNN
CudaDnnResult CudaExecutionSession::create_dnn() noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto& sdk = state.dispatch.dnn;
    if (!sdk || !sdk->create || !sdk->destroy || !sdk->set_stream || !sdk->get_stream)
        return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->dnn.size() == state.limits.max_objects)
        return dnn_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    Implementation::Dnn handle{.id = state.next_id};
    auto result = dnn_completed(state.invoke(sdk->create, &handle.native));
    const auto after_create = state.check_context(state.stream_config.uses_default_proxy);
    if (result.original_status != CUDNN_STATUS_SUCCESS && !handle.native) {
        if (after_create != CUDA_SUCCESS) {
            state.fault();
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = after_create;
        }
        return result;
    }
    (void)state.consume_id();
    handle.is_uncertain = result.original_status != CUDNN_STATUS_SUCCESS || !handle.native ||
                          after_create != CUDA_SUCCESS;
    for (auto& existing : state.storage->dnn)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    state.storage->dnn.push_back(handle);
    if (!handle.is_uncertain && state.storage->default_proxy) {
        const auto configured =
            state.invoke(sdk->set_stream, handle.native,
                         reinterpret_cast<cudaStream_t>(state.storage->default_proxy));
        const auto proof = state.check_context(true);
        handle.is_uncertain = configured != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS;
    }
    if (!handle.is_uncertain) {
        cudaStream_t observed{};
        const auto queried = state.invoke(sdk->get_stream, handle.native, &observed);
        handle.is_uncertain = queried != CUDNN_STATUS_SUCCESS ||
                              reinterpret_cast<CUstream>(observed) != state.storage->default_proxy;
    }
    const auto final_proof = state.stream_config.uses_default_proxy || !handle.is_uncertain
                                 ? state.check_context(state.stream_config.uses_default_proxy)
                                 : CUDA_SUCCESS;
    if (handle.is_uncertain || final_proof != CUDA_SUCCESS) {
        state.storage->dnn.back().is_uncertain = true;
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = handle.id;
    return result;
}
CudaDnnResult CudaExecutionSession::destroy_dnn(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto found = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (found == state.storage->dnn.end()) return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = dnn_completed(state.invoke(state.dispatch.dnn->destroy, found->native));
    if (result.original_status == CUDNN_STATUS_SUCCESS)
        state.storage->dnn.erase(found);
    else {
        found->is_uncertain = true;
        state.fault();
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    }
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): logical handle and stream wire IDs.
CudaDnnResult CudaExecutionSession::set_dnn_stream(std::uint64_t id,
                                                   std::uint64_t stream_id) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return dnn_gate(gate);
    const auto found = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (found == state.storage->dnn.end()) return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    CUstream stream = state.storage->default_proxy;
    if (stream_id) {
        const auto owned =
            std::ranges::find(state.storage->streams, stream_id, &Implementation::Stream::id);
        if (owned == state.storage->streams.end())
            return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
        stream = owned->native;
    }
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    auto result = dnn_completed(state.invoke(state.dispatch.dnn->set_stream, found->native,
                                             reinterpret_cast<cudaStream_t>(stream)));
    if (result.original_status == CUDNN_STATUS_SUCCESS)
        found->stream = stream_id;
    else {
        // A failed native mutation does not prove the old stream is still set.
        found->is_uncertain = true;
        state.fault();
    }
    if (state.check_context(state.stream_config.uses_default_proxy) != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    }
    return result;
}
CudaDnnResult CudaExecutionSession::get_dnn_stream(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted) return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    return state.observe_dnn_stream(id);
}
CudaDnnResult CudaExecutionSession::query_dnn_dropout_states_size(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (!sdk->dropout_states_size) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // This describes storage only and must not drain previously accepted work.
    std::size_t bytes{};
    auto result = dnn_completed(state.invoke(sdk->dropout_states_size, handle->native, &bytes));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = status;
    } else if (result.original_status == CUDNN_STATUS_SUCCESS) {
        result.value = bytes;
    }
    return result;
}
std::optional<CudaDeviceMemoryView> CudaExecutionSession::resolve_dnn_dropout(
    const CudaDnnDropoutView& dropout) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_closing || state.is_faulted ||
        state.executor->snapshot().is_quarantined || state.check_context() != CUDA_SUCCESS ||
        dropout.object.kind != CudaDnnObjectKind::kDropout || !dropout.object.native ||
        !dropout.object.lifetime)
        return std::nullopt;
    const auto& binding = dropout.binding;
    // A newly created/null-only descriptor has no proven state allocation.
    // Preserve that null state for the original consumer's legality decision.
    if (!binding.allocation) return CudaDeviceMemoryView{};
    if (binding.context.lock() != state.storage) return std::nullopt;
    return device_memory({binding.allocation, binding.offset}, binding.bytes);
}
CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_dropout(
    std::uint64_t sequence, const CudaSessionDnnDropout& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    const bool is_set = problem.kind == CudaDnnDropoutKind::kSet;
    const bool is_restore = problem.kind == CudaDnnDropoutKind::kRestore;
    if (!sdk || (!is_set && !is_restore) || (is_set && !sdk->dropout_set) ||
        (is_restore && !sdk->dropout_restore) || !state.dispatch.context_synchronize ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !problem.mutation || !problem.mutation->native())
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    CudaDeviceMemoryView states;
    auto binding = problem.mutation->old_binding();
    if (problem.states.allocation) {
        const auto memory = device_memory(problem.states, problem.state_bytes);
        if (!memory) return {};
        states = *memory;
        binding = {state.storage, problem.states.allocation, problem.states.offset,
                   problem.state_bytes};
    } else if (problem.states.offset) {
        return {};
    }
    // Null Set updates only metadata; null Restore has no documented detach
    // contract. Both forward the original count and preserve prior provenance.
    try {
        auto call = make_cuda_dnn_dropout_call(
            sdk, handle->native, problem, std::move(states), std::move(binding),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize});
        return {
            state.executor->accept_library(
                sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)}),
            {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}
CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_reorder(
    std::uint64_t sequence, const CudaSessionDnnReorder& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_reorder_descriptors(problem) ||
        problem.reorder > 1)
        return {};
    if (!sdk->version || !sdk->reorder_filter_bias || !sdk->filter_get_nd || !sdk->filter_size)
        return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    // The two native branches have different bias footprints. This exact
    // version gate does not replace runtime qualification of mapped libraries.
    const auto version = state.invoke(sdk->version);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    if (version != 91002) return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        if (index >= 2 && !problem.reorder_bias) {
            if (range.allocation || range.offset) return {};
        } else if (!range.allocation) {
            // Negative flags also access bias in the inspected native binary.
            // Nonzero/null rejection is an explicit compatibility restriction.
            return {};
        }
    }
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    const auto descriptor = static_cast<CudaDnnFilterHandle>(problem.filter.native);
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> sizes{};
    auto status = state.invoke(sdk->filter_get_nd, descriptor, CUDNN_DIM_MAX, &type, &format,
                               &dimensions, sizes.data());
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
    std::uint32_t type_bits{}, format_bits{};
    static_assert(sizeof(type) == sizeof(type_bits) && sizeof(format) == sizeof(format_bits));
    std::memcpy(&type_bits, &type, sizeof(type_bits));
    std::memcpy(&format_bits, &format, sizeof(format_bits));
    if (dimensions <= 0 || dimensions > CUDNN_DIM_MAX ||
        type_bits > dnn_profile::data_type_enum_max() || format_bits > 3 ||
        std::ranges::any_of(std::span{sizes.data(), static_cast<std::size_t>(dimensions)},
                            [](int size) { return size <= 0; })) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    std::size_t filter_bytes{};
    status = state.invoke(sdk->filter_size, descriptor, &filter_bytes);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
    }
    if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
    if (!filter_bytes) return {};
    // Native FilterNd expands vectorized C, while FilterSize already includes
    // the vector width. Never multiply logical INT8x32 K*C*R*S by 32 again.
    constexpr std::array<std::size_t, 17> native_widths{4, 8, 2, 1, 4, 4, 1, 4, 32,
                                                        2, 8, 1, 1, 1, 4, 1, 1};
    std::size_t bias_bytes{};
    if (problem.reorder == 1 && type_bits >= native_widths.size())
        return {{}, {}, CUDA_ERROR_NOT_SUPPORTED};
    if (problem.reorder_bias) {
        const auto width = problem.reorder == 0 ? sizeof(float) : native_widths[type_bits];
        const auto channels = static_cast<std::size_t>(sizes[0]);
        if (channels > std::numeric_limits<std::size_t>::max() / width) return {};
        bias_bytes = channels * width;
    }
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != buffers.size(); ++index) {
        if (index >= 2 && !problem.reorder_bias) continue;
        const auto memory =
            device_memory(problem.buffers[index], index < 2 ? filter_bytes : bias_bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    for (std::size_t output : {std::size_t{1}, std::size_t{3}}) {
        const auto& destination = buffers[output];
        if (!destination.bytes) continue;
        for (std::size_t index = 0; index != buffers.size(); ++index) {
            const auto& other = buffers[index];
            if (index == output || !other.bytes) continue;
            // General in-place reordering is not qualified. Read/read aliases
            // and disjoint views of one allocation remain valid.
            const bool is_disjoint = destination.address <= other.address
                                         ? other.address - destination.address >= destination.bytes
                                         : destination.address - other.address >= other.bytes;
            if (!is_disjoint) return {};
        }
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_reorder_call(sdk, handle->native, problem,
                                                                  std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnResult CudaExecutionSession::query_dnn_reduction(const CudaSessionDnnReduction& problem,
                                                        CudaDnnReductionQueryKind kind) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_reduction_descriptors(problem))
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (kind != CudaDnnReductionQueryKind::kIndices &&
        kind != CudaDnnReductionQueryKind::kWorkspace)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    const auto query = kind == CudaDnnReductionQueryKind::kIndices ? sdk->reduction_indices_size
                                                                   : sdk->reduction_workspace_size;
    if (!query) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // Both queries describe storage; neither issues computation nor needs to
    // drain unrelated accepted work. Immutable descriptor borrows cover them.
    std::size_t bytes{};
    auto result = dnn_completed(state.invoke(
        query, handle->native, static_cast<CudaDnnReduction>(problem.descriptors[1].native),
        static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native),
        static_cast<cudnnTensorDescriptor_t>(problem.descriptors[2].native), &bytes));
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else if (result.original_status == CUDNN_STATUS_SUCCESS)
        result.value = bytes;
    return result;
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_reduction(
    std::uint64_t sequence, const CudaSessionDnnReduction& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->reduce_tensor || !sdk->tensor_get_nd || !sdk->tensor_size ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_reduction_descriptors(problem))
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != 2; ++index) {
        const auto descriptor =
            static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index * 2].native);
        cudnnDataType_t type{};
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (rank <= 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        // Reduction permits mixed non-double storage types, all with float
        // coefficients. The native SDK validates the actual combination.
        const auto width = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
        if (problem.scalar_bytes != width) return {};
        std::size_t bytes{};
        status = state.invoke(sdk->tensor_size, descriptor, &bytes);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    for (std::size_t index = 2; index != buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const auto bytes = index == 2 ? problem.indices_bytes : problem.workspace_bytes;
        if (range.allocation) {
            const auto memory = device_memory(range, bytes);
            if (!memory) return {};
            buffers[index] = *memory;
        } else if (range.offset) {
            return {};
        }
        // A null optional pointer preserves its original count. The SDK
        // decides whether the space is ignored or insufficient, not the wire.
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_reduction_call(sdk, handle->native, problem,
                                                                    std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnResult CudaExecutionSession::query_dnn_batch_norm(
    const CudaSessionDnnBatchNorm& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return dnn_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (state.is_closing) return dnn_gate(rejected(CUDA_ERROR_DEINITIALIZED));
    if (const auto status = state.check_context(); status != CUDA_SUCCESS)
        return dnn_gate(rejected(status));
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_batch_norm_descriptors(problem))
        return dnn_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (problem.mode > 3 || problem.operation > 3)
        return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    const auto tensor = [&](std::size_t index) {
        return static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index].native);
    };
    const auto activation = static_cast<CudaDnnActivation>(problem.descriptors[6].native);
    const auto mode = static_cast<decltype(CUDNN_BATCHNORM_SPATIAL)>(problem.mode);
    const auto operation = static_cast<decltype(CUDNN_BATCHNORM_OPS_BN)>(problem.operation);
    std::size_t bytes{};
    cudnnStatus_t status{};
    // These original storage queries issue no computation and do not acquire
    // the physical-quiescence gate used by handle mutation.
    switch (problem.kind) {
        case CudaDnnBatchNormKind::kForward:
            if (!sdk->batch_norm_forward_workspace)
                return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status =
                state.invoke(sdk->batch_norm_forward_workspace, handle->native, mode, operation,
                             tensor(0), tensor(3), tensor(1), tensor(5), activation, &bytes);
            break;
        case CudaDnnBatchNormKind::kBackward:
            if (!sdk->batch_norm_backward_workspace)
                return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status = state.invoke(sdk->batch_norm_backward_workspace, handle->native, mode,
                                  operation, tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
                                  tensor(5), activation, &bytes);
            break;
        case CudaDnnBatchNormKind::kReserve:
            if (!sdk->batch_norm_reserve_size) return dnn_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
            status = state.invoke(sdk->batch_norm_reserve_size, handle->native, mode, operation,
                                  activation, tensor(0), &bytes);
            break;
        default:
            return dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    }
    auto result = dnn_completed(status);
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else if (status == CUDNN_STATUS_SUCCESS)
        result.value = bytes;
    return result;
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_batch_norm(
    std::uint64_t sequence, const CudaSessionDnnBatchNorm& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const bool is_forward = problem.kind == CudaDnnBatchNormKind::kForward;
    if (!is_forward && problem.kind != CudaDnnBatchNormKind::kBackward) return {};
    if (problem.mode > 3 || problem.operation > 3) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->tensor_get_nd || !sdk->tensor_size ||
        (is_forward ? !sdk->batch_norm_forward : !sdk->batch_norm_backward) ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_batch_norm_descriptors(problem))
        return {};
    if (is_forward)
        for (const auto index : {2U, 4U})
            if (problem.buffers[index].allocation || problem.buffers[index].offset) return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<std::size_t, 6> extents{};
    for (std::size_t index = 0; index != extents.size(); ++index) {
        if (!problem.descriptors[index].native) continue;
        const auto descriptor =
            static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index].native);
        cudnnDataType_t type{};
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (rank <= 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (index == 0 && problem.scalar_bytes !=
                              (type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float)))
            return {};
        status = state.invoke(sdk->tensor_size, descriptor, &extents[index]);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!extents[index]) return {};
    }
    std::array<CudaDeviceMemoryView, 13> buffers{};
    for (std::size_t index = 0; index != buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const bool is_required =
            index == 0 || index == 5 ||
            (is_forward ? index == 1 || index == 6
                        : index == 2 || index == 4 || index == 7 || index == 8);
        if (!range.allocation) {
            if (range.offset || is_required) return {};
            continue;
        }
        // An ignored optional tensor may lack its descriptor. Still retain its
        // owned address; active tensor shapes and optional pair legality remain
        // original SDK decisions. Parameter tensors have independent extents.
        const auto bytes = index == 11   ? problem.workspace_bytes
                           : index == 12 ? problem.reserve_bytes
                                         : std::max<std::size_t>(1, extents[index < 5 ? index : 5]);
        const auto memory = device_memory(range, bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_batch_norm_call(sdk, handle->native, problem,
                                                                     std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnConvolutionQueryResult CudaExecutionSession::query_dnn_convolution(
    const CudaSessionDnnConvolution& problem, CudaDnnConvolutionQueryKind kind,
    int requested) noexcept {
    auto& state = *implementation_;
    CudaDnnConvolutionQueryResult output;
    auto reject = [&](CUresult status) {
        output.result = dnn_gate(rejected(status));
        return output;
    };
    const auto quarantine = [&] {
        state.fault();
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = CUDA_ERROR_UNKNOWN;
        return output;  // Retain the successful native call's original status.
    };
    if (!state.owns() || !state.executor || state.is_faulted) return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return reject(status);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    // These are original metadata/heuristic queries, not an exhaustive Find
    // benchmark. They do not drain unrelated accepted GPU work.
    const auto finished = [&](cudnnStatus_t status) {
        output.result = dnn_completed(status);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status = CUDA_ERROR_UNKNOWN;
        }
        return output.result.status == CUDA_SUCCESS && status == CUDNN_STATUS_SUCCESS;
    };
    if (kind == CudaDnnConvolutionQueryKind::kMaxCount) {
        if (!sdk->convolution_forward_max_count) return reject(CUDA_ERROR_NOT_SUPPORTED);
        int count{};
        if (!finished(state.invoke(sdk->convolution_forward_max_count, handle->native, &count)))
            return output;
        if (count < 0) return quarantine();
        output.result.value = static_cast<std::uint64_t>(count);
        return output;
    }
    if (!valid_cuda_dnn_convolution_descriptors(problem)) return reject(CUDA_ERROR_INVALID_HANDLE);
    const auto x = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native);
    const auto w = static_cast<CudaDnnFilterHandle>(problem.descriptors[1].native);
    const auto convolution = static_cast<CudaDnnConvolution>(problem.descriptors[2].native);
    const auto y = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[3].native);
    if (kind == CudaDnnConvolutionQueryKind::kWorkspace) {
        if (!sdk->convolution_forward_workspace) return reject(CUDA_ERROR_NOT_SUPPORTED);
        constexpr auto algorithm_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_FWD_ALGO_COUNT) + 1U) - 1U;
        if (problem.algorithm > algorithm_max) return reject(CUDA_ERROR_INVALID_VALUE);
        std::size_t bytes{};
        if (finished(
                state.invoke(sdk->convolution_forward_workspace, handle->native, x, w, convolution,
                             y, static_cast<cudnnConvolutionFwdAlgo_t>(problem.algorithm), &bytes)))
            output.result.value = bytes;
        return output;
    }
    if (kind != CudaDnnConvolutionQueryKind::kHeuristic) return reject(CUDA_ERROR_INVALID_VALUE);
    if (!sdk->convolution_forward_heuristic || !sdk->convolution_forward_max_count)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    int maximum{};
    if (!finished(state.invoke(sdk->convolution_forward_max_count, handle->native, &maximum)))
        return output;
    if (maximum < 0) return quarantine();
    if (static_cast<std::size_t>(maximum) > kCudaDnnMaxConvolutionAlgorithms)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // Preserve zero/negative request errors. Larger capacities need no larger
    // native buffer than the SDK's actual maximum (including math variants).
    const auto capacity = std::min(requested, maximum);
    std::array<cudnnConvolutionFwdAlgoPerfStruct, kCudaDnnMaxConvolutionAlgorithms> native{};
    int count{};
    if (!finished(state.invoke(sdk->convolution_forward_heuristic, handle->native, x, w,
                               convolution, y, capacity, &count, native.data())))
        return output;
    if (count < 0 || count > capacity) return quarantine();
    decltype(output.performance) candidate{};
    for (int index = 0; index < count; ++index) {
        const auto& value = native[static_cast<std::size_t>(index)];
        auto& result = candidate[static_cast<std::size_t>(index)];
        std::memcpy(&result.algorithm, &value.algo, sizeof(result.algorithm));
        std::memcpy(&result.status, &value.status, sizeof(result.status));
        std::memcpy(&result.determinism, &value.determinism, sizeof(result.determinism));
        std::memcpy(&result.math, &value.mathType, sizeof(result.math));
        result.time_bits = std::bit_cast<std::uint32_t>(value.time);
        result.memory = value.memory;
        std::copy(std::begin(value.reserved), std::end(value.reserved), result.reserved.begin());
        if (result.algorithm > 15 || result.status > dnn_profile::status_enum_max() ||
            result.determinism > 1 || result.math > 3)
            return quarantine();
    }
    output.count = static_cast<std::uint32_t>(count);
    output.performance = candidate;
    return output;
}

CudaDnnResult CudaExecutionSession::query_dnn_backward_convolution(
    const CudaSessionDnnConvolution& problem, CudaDnnBackwardConvolutionQueryKind kind,
    bool has_output) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) { return dnn_gate(rejected(status)); };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataMaxCount &&
        kind != CudaDnnBackwardConvolutionQueryKind::kFilterMaxCount &&
        kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace)
        return reject(CUDA_ERROR_INVALID_VALUE);
    // The configured native MaxCount implementations write through count
    // without checking it. Reject absence instead of fabricating a temporary
    // output for a call that would otherwise crash.
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace && !has_output)
        return reject(CUDA_ERROR_INVALID_VALUE);
    if (const auto status = state.check_context(); status != CUDA_SUCCESS) return reject(status);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);

    CudaDnnResult output;
    const auto finished = [&](cudnnStatus_t status) {
        output = dnn_completed(status);
        if (const auto checked = state.check_context(); checked != CUDA_SUCCESS) {
            state.fault();
            output.disposition = CudaDnnDisposition::kRejected;
            output.status = checked;
        }
        return output.status == CUDA_SUCCESS && status == CUDNN_STATUS_SUCCESS;
    };
    // These original host-planning queries do not drain accepted work or
    // record events. Vendor loading/planning may still perform Driver work.
    if (kind != CudaDnnBackwardConvolutionQueryKind::kDataWorkspace) {
        const auto query = kind == CudaDnnBackwardConvolutionQueryKind::kDataMaxCount
                               ? sdk->convolution_backward_data_max_count
                               : sdk->convolution_backward_filter_max_count;
        if (!query) return reject(CUDA_ERROR_NOT_SUPPORTED);
        int count{};
        if (!finished(state.invoke(query, handle->native, &count))) return output;
        if (count < 0) {
            state.fault();
            output.disposition = CudaDnnDisposition::kRejected;
            output.status = CUDA_ERROR_UNKNOWN;
            return output;  // Preserve native SUCCESS but publish no count.
        }
        output.value = static_cast<std::uint64_t>(count);
        return output;
    }
    if (!sdk->convolution_backward_data_workspace) return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!valid_cuda_dnn_convolution_descriptors(problem)) return reject(CUDA_ERROR_INVALID_HANDLE);
    constexpr auto algorithm_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT) + 1U) - 1U;
    if (problem.algorithm > algorithm_max) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto dy = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[0].native);
    const auto w = static_cast<CudaDnnFilterHandle>(problem.descriptors[1].native);
    const auto convolution = static_cast<CudaDnnConvolution>(problem.descriptors[2].native);
    const auto dx = static_cast<cudnnTensorDescriptor_t>(problem.descriptors[3].native);
    std::size_t bytes{};
    if (finished(state.invoke(sdk->convolution_backward_data_workspace, handle->native, w, dy,
                              convolution, dx,
                              static_cast<cudnnConvolutionBwdDataAlgo_t>(problem.algorithm),
                              has_output ? &bytes : nullptr)) &&
        has_output)
        output.value = bytes;
    return output;
}

CudaDnnBackendContextResult CudaExecutionSession::backend_context(
    std::uint64_t owner, std::uint64_t id, CudaDnnRegistry& registry,
    const CudaDnnBackendContextRequest& request) noexcept {
    auto& state = *implementation_;
    CudaDnnBackendContextResult output;
    const auto reject = [&](CUresult status) {
        output.result = dnn_gate(rejected(status));
        return output;
    };
    if (!owner || !state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        output.result.disposition = CudaDnnDisposition::kDeferred;
        output.result.status = CUDA_ERROR_NOT_READY;
        return output;
    }
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    cudnnHandle_t native{};
    if (id) {
        const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
        if (handle == state.storage->dnn.end() || !handle->native || handle->is_uncertain ||
            handle->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        native = handle->native;
    }
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    const auto check = [](void* opaque) noexcept {
        auto& session = *static_cast<Implementation*>(opaque);
        const auto status = session.check_context(true);
        if (status != CUDA_SUCCESS) session.fault();
        return status;
    };
    if (const auto status = check(&state); status != CUDA_SUCCESS) return reject(status);
    const CudaDnnBackendContext context{
        native,
        owner,
        id,
        state.storage,
        check,
        &state,
        state.device_ordinal,
        [](std::uint64_t handle_id, void* opaque) noexcept -> cudnnHandle_t {
            const auto& session = *static_cast<const Implementation*>(opaque);
            const auto found =
                std::ranges::find(session.storage->dnn, handle_id, &Implementation::Dnn::id);
            return found != session.storage->dnn.end() && !found->is_uncertain &&
                           !found->has_destroy_attempt
                       ? found->native
                       : nullptr;
        }};
    switch (request.operation) {
        case CudaDnnBackendContextOperation::kPrepare:
            output.preparation = registry.prepare_backend_attribute(request.attribute_operation,
                                                                    request.header, &context);
            output.result = output.preparation.result;
            break;
        case CudaDnnBackendContextOperation::kSet:
            output.attribute = registry.set_backend_attribute(request.header, request.generation,
                                                              request.values, &context);
            output.result = output.attribute.result;
            break;
        case CudaDnnBackendContextOperation::kGet:
            output.attribute = registry.get_backend_attribute(request.header, request.generation,
                                                              request.values, &context);
            output.result = output.attribute.result;
            break;
        case CudaDnnBackendContextOperation::kFinalize:
            output.result = registry.finalize_backend(request.id, &context);
            break;
        case CudaDnnBackendContextOperation::kInitialize:
            output.result = registry.initialize_backend(request.id);
            break;
        case CudaDnnBackendContextOperation::kDestroy:
            if (!registry.borrow(request.id, CudaDnnObjectKind::kBackend))
                return reject(CUDA_ERROR_INVALID_HANDLE);
            output.result = registry.destroy(request.id);
            break;
        default:
            return reject(CUDA_ERROR_INVALID_VALUE);
    }
    if (const auto status = check(&state); status != CUDA_SUCCESS) {
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = status;
    }
    return output;
}
CudaDnnBackendExecutePreparation CudaExecutionSession::prepare_dnn_backend_execute(
    std::uint64_t owner, std::uint64_t handle, CudaDnnRegistry& registry, std::uint64_t plan_id,
    std::uint64_t variant_id) noexcept {
    CudaDnnBackendExecutePreparation output;
    output.result = dnn_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted || state.is_closing) return output;
    try {
        const auto plan = registry.borrow_backend(plan_id);
        const auto variant = registry.borrow_backend(variant_id);
        if (!plan || !variant || plan->snapshot->kind != 5 || variant->snapshot->kind != 16 ||
            !plan->snapshot->is_valid || !plan->snapshot->is_finalized ||
            !variant->snapshot->is_valid || !variant->snapshot->is_finalized ||
            !plan->snapshot->context ||
            plan->snapshot->context->device_ordinal != state.device_ordinal ||
            variant->snapshot->variant_uids.size() != variant->snapshot->pointer_slots.size() ||
            variant->snapshot->variant_uids.size() > 100)
            return output;
        const auto tensors = cuda_dnn_backend_tensor_requirements(*plan->snapshot);
        if (tensors.status != CUDA_SUCCESS) {
            output.result.status = tensors.status;
            return output;
        }
        CudaDnnBackendContextRequest request;
        request.operation = CudaDnnBackendContextOperation::kGet;
        request.header = {plan_id, 402, 3, 1, 3};
        request.generation = plan->snapshot->generation;
        const auto workspace = backend_context(owner, handle, registry, request);
        if (workspace.result.disposition == CudaDnnDisposition::kDeferred) {
            output.result = workspace.result;
            return output;
        }
        if (workspace.result.status != CUDA_SUCCESS ||
            workspace.result.original_status != CUDNN_STATUS_SUCCESS ||
            workspace.attribute.values.scalars.size() != sizeof(std::int64_t)) {
            // The private workspace query is never requested Execute provenance.
            output.result.status = workspace.result.status != CUDA_SUCCESS ? workspace.result.status
                                                                           : CUDA_ERROR_UNKNOWN;
            return output;
        }
        std::int64_t bytes{};
        std::memcpy(&bytes, workspace.attribute.values.scalars.data(), sizeof(bytes));
        if (bytes < 0 || static_cast<std::uint64_t>(bytes) > SIZE_MAX) return output;
        output.plan_generation = plan->snapshot->generation;
        output.variant_generation = variant->snapshot->generation;
        for (std::size_t i = 0; i != variant->snapshot->variant_uids.size(); ++i) {
            const auto uid = variant->snapshot->variant_uids[i];
            CudaDnnBackendExecuteRequirement requirement{static_cast<std::uint32_t>(i),
                                                         CudaDnnBackendBindingClass::kIgnored,
                                                         uid,
                                                         variant->snapshot->pointer_slots[i],
                                                         0,
                                                         0};
            const auto tensor =
                std::ranges::find(tensors.tensors, uid, &CudaDnnBackendTensorRequirement::uid);
            if (tensor != tensors.tensors.end()) {
                requirement.value_class = tensor->is_by_value ? CudaDnnBackendBindingClass::kByValue
                                                              : CudaDnnBackendBindingClass::kDevice;
                requirement.bytes = tensor->bytes;
                requirement.alignment = tensor->alignment;
            }
            output.requirements.push_back(requirement);
        }
        for (const auto& tensor : tensors.tensors)
            if (std::ranges::find(variant->snapshot->variant_uids, tensor.uid) ==
                variant->snapshot->variant_uids.end()) {
                output.requirements.clear();
                return output;
            }
        output.requirements.push_back({UINT32_MAX, CudaDnnBackendBindingClass::kDevice, 0,
                                       variant->snapshot->workspace_slot,
                                       static_cast<std::size_t>(bytes), 1});
        output.result.disposition = CudaDnnDisposition::kComplete;
        output.result.status = CUDA_SUCCESS;
    } catch (...) {
        output.result.status = CUDA_ERROR_OUT_OF_MEMORY;
        output.requirements.clear();
    }
    return output;
}
CudaDnnBackendExecuteAcceptance CudaExecutionSession::accept_dnn_backend_execute(
    std::uint64_t sequence, CudaDnnRegistry& registry, std::uint64_t owner,
    const CudaSessionDnnBackendExecute& problem) noexcept {
    auto& state = *implementation_;
    CudaDnnBackendExecuteAcceptance output;
    const auto reject = [&](CUresult status) {
        output.native.rejection_status = status;
        return std::move(output);
    };
    const auto prepared =
        prepare_dnn_backend_execute(owner, problem.handle, registry, problem.plan, problem.variant);
    if (prepared.result.status != CUDA_SUCCESS) return reject(prepared.result.status);
    if (prepared.plan_generation != problem.plan_generation ||
        prepared.variant_generation != problem.variant_generation ||
        prepared.requirements.size() != problem.bindings.size())
        return reject(CUDA_ERROR_INVALID_VALUE);
    try {
        auto plan = registry.borrow_backend_call(problem.plan);
        const auto variant = registry.borrow_backend(problem.variant);
        if (!plan || !variant || plan->descriptor.snapshot->generation != problem.plan_generation ||
            variant->snapshot->generation != problem.variant_generation)
            return reject(CUDA_ERROR_INVALID_VALUE);
        const auto handle =
            std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
        if (handle == state.storage->dnn.end() || handle->is_uncertain ||
            handle->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        CudaDnnBackendExecutionArguments arguments;
        arguments.plan = std::move(*plan);
        arguments.uids = variant->snapshot->variant_uids;
        arguments.pointers.resize(arguments.uids.size());
        std::size_t retained{};
        for (std::size_t i = 0; i != prepared.requirements.size(); ++i) {
            const auto& requirement = prepared.requirements[i];
            const auto& binding = problem.bindings[i];
            if (binding.index != requirement.index || binding.slot != requirement.slot ||
                binding.value_class != requirement.value_class)
                return reject(CUDA_ERROR_INVALID_VALUE);
            if (!binding.slot) {
                if (binding.device.allocation || binding.device.offset ||
                    !binding.host_bytes.empty())
                    return reject(CUDA_ERROR_INVALID_VALUE);
                continue;
            }
            if (requirement.value_class == CudaDnnBackendBindingClass::kDevice) {
                if (!binding.host_bytes.empty()) return reject(CUDA_ERROR_INVALID_VALUE);
                const auto memory = device_memory(binding.device, requirement.bytes);
                if (!memory || !memory->address || memory->address % requirement.alignment ||
                    requirement.bytes > std::numeric_limits<CUdeviceptr>::max() - memory->address)
                    return reject(CUDA_ERROR_INVALID_VALUE);
                if (requirement.index == UINT32_MAX)
                    arguments.workspace = *memory;
                else
                    arguments.pointers[requirement.index] = {
                        CudaDnnBackendPointerKind::kDevice, *memory, {}, requirement.alignment};
            } else {
                if (binding.device.allocation || binding.device.offset ||
                    (requirement.value_class == CudaDnnBackendBindingClass::kByValue
                         ? binding.host_bytes.size() != requirement.bytes
                         : !binding.host_bytes.empty()))
                    return reject(CUDA_ERROR_INVALID_VALUE);
                auto& pointer = arguments.pointers[requirement.index];
                pointer.kind = CudaDnnBackendPointerKind::kHostValue;
                pointer.alignment = requirement.alignment ? requirement.alignment : 1;
                pointer.value = requirement.value_class == CudaDnnBackendBindingClass::kByValue
                                    ? binding.host_bytes
                                    : std::vector<std::byte>{std::byte{}};
                // Zero-size live pointers still preserve non-NULL identity.
                if (pointer.value.empty()) pointer.value.push_back(std::byte{});
                if (pointer.alignment > state.limits.submissions.max_parameter_bytes ||
                    pointer.value.size() >
                        state.limits.submissions.max_parameter_bytes - (pointer.alignment - 1))
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                const auto bytes = pointer.value.size() + pointer.alignment - 1;
                if (bytes > state.limits.submissions.max_parameter_bytes - retained)
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                retained += bytes;
            }
        }
        const auto metadata_call = arguments.plan.call;
        if (!metadata_call->begin_call()) return reject(CUDA_ERROR_NOT_READY);
        struct ActiveCall {
            bool& active;
            CudaDnnRnnLease& metadata;
            ~ActiveCall() {
                active = false;
                metadata.end_call();
            }
        };
        if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
            return reject(status);
        state.is_rnn_query_active = true;
        const ActiveCall active{state.is_rnn_query_active, *metadata_call};
        const auto stream = state.observe_dnn_stream(problem.handle);
        if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS)
            return reject(stream.status != CUDA_SUCCESS ? stream.status : CUDA_ERROR_UNKNOWN);
        auto result = std::make_shared<CudaDnnBackendExecutionCompletion>();
        auto call = make_cuda_dnn_backend_call(
            state.dispatch.dnn, handle->native, std::move(arguments),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result);
        // Account retained host snapshots in the submission byte limit too.
        const std::vector<std::byte> charge(retained);
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = charge, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaExecutionSession::fold_dnn_backward_convolution(
    std::uint64_t id, CudaDnnRegistry& metadata,
    const CudaDnnFoldedDescriptors& descriptors) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) { return dnn_gate(rejected(status)); };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_folded_active = true;
    const ActiveCall active{state.is_folded_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    // The registry holds every unique descriptor owner and excludes new reads
    // through getter, original call and both post-call context checks. Session
    // lifecycle reentry cannot destroy/rebind this handle during those calls.
    const auto result = metadata.folded_backward_data(
        handle->native, descriptors,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    // Metadata-only failure does not erase known handle cleanup ownership.
    // The helper's diagnostic status is not the requested Folded's provenance.
    return result.result;
}

CudaDnnRnnQueryResult CudaExecutionSession::query_dnn_rnn(std::uint64_t id,
                                                          CudaDnnRegistry& metadata,
                                                          const CudaDnnRnnQuery& query) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnQueryResult output;
        output.result = dnn_gate(rejected(status));
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    // Do not call cold_gate: pending physical work need not finish for these
    // original planning queries. Their vendor loader/device behavior remains
    // context-owned and may block internally, without a framework queue drain.
    auto result = metadata.query_rnn_sizes(
        handle->native, query,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    // Metadata-only uncertainty does not erase proven native handle ownership.
    return std::move(result.result);
}

CudaDnnRnnBuildResult CudaExecutionSession::build_dnn_rnn(std::uint64_t id,
                                                          CudaDnnRegistry& metadata,
                                                          std::uint64_t descriptor,
                                                          int mini_batch) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnBuildResult result{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            result.result.disposition = CudaDnnDisposition::kDeferred;
        return result;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return reject(status);
    auto result = metadata.build_rnn_dynamic(
        handle->native, descriptor, mini_batch,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (result.has_context_loss) state.fault();
    return result;
}

CudaDnnRnnWeightParamsResult CudaExecutionSession::query_dnn_rnn_weight_params(
    std::uint64_t id, CudaDnnRegistry& metadata,
    const CudaSessionDnnRnnWeightParams& problem) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnWeightParamsResult output{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS) {
        state.fault();
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    }
    std::optional<CudaDeviceMemoryView> weights;
    if (problem.weights.allocation) {
        weights = device_memory(problem.weights, problem.weight_bytes);
        if (!weights || !weights->address ||
            problem.weight_bytes > std::numeric_limits<CUdeviceptr>::max() - weights->address)
            return reject(CUDA_ERROR_INVALID_VALUE);
    } else if (problem.weights.offset) {
        return reject(CUDA_ERROR_INVALID_VALUE);
    }
    // This host metadata query neither consumes the weight payload nor drains
    // pending physical readers. The scoped session guard prevents explicit
    // free/context teardown while the complete declared range is observed.
    auto output = metadata.rnn_weight_params(
        handle->native, problem.parameters, weights ? weights->address : 0, problem.weight_bytes,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (output.has_context_loss) state.fault();
    return output;
}

CudaDnnAttnWeightsResult CudaExecutionSession::query_dnn_attn_weights(
    std::uint64_t id, CudaDnnRegistry& metadata, const CudaDnnAttnWeights& parameters) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnAttnWeightsResult output{dnn_gate(rejected(status))};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    const auto handle = std::ranges::find(state.storage->dnn, id, &Implementation::Dnn::id);
    if (!state.dispatch.dnn || handle == state.storage->dnn.end() || !handle->native ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active};
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS) {
        state.fault();
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    }
    // The original only calculates addresses. Neither GPU allocation ownership
    // nor RNG-state resolution is required, and pending work is not drained.
    auto output = metadata.attn_weights(
        handle->native, parameters,
        [](const void* opaque) noexcept {
            return static_cast<const Implementation*>(opaque)->check_context(true);
        },
        &state);
    if (output.has_context_loss) state.fault();
    return output;
}

CudaExecutionSession::Implementation::AttnOperation
CudaExecutionSession::Implementation::observe_attn(CudaExecutionSession& session,
                                                   const CudaSessionDnnAttn& problem,
                                                   bool is_preparation,
                                                   std::uint64_t sequence) noexcept {
    AttnOperation output;
    const auto reject = [&](CUresult status,
                            ExecutionAcceptStatus disposition = ExecutionAcceptStatus::kInvalid) {
        output.preparation.result.disposition = CudaDnnDisposition::kRejected;
        output.preparation.result.status = status;
        output.preparation.first = output.preparation.count = 0;
        output.acceptance.native.rejection_status = status;
        output.acceptance.native.acceptance.status = disposition;
        return std::move(output);
    };
    if (!owns() || !executor || is_faulted || executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN, ExecutionAcceptStatus::kQuarantined);
    if (is_closing) return reject(CUDA_ERROR_DEINITIALIZED, ExecutionAcceptStatus::kClosed);
    if (is_folded_active || is_rnn_query_active || is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    if (!valid_cuda_dnn_attn_descriptors(problem) ||
        (is_preparation && (!problem.windows[0].empty() || !problem.windows[1].empty())))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = dispatch.dnn;
    const auto handle = std::ranges::find(storage->dnn, problem.handle, &Dnn::id);
    if (!sdk || handle == storage->dnn.end() || !handle->native || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    const bool forward = problem.kind == CudaDnnAttnComputeKind::kForward;
    const bool data = problem.kind == CudaDnnAttnComputeKind::kBackwardData;
    const bool weights = problem.kind == CudaDnnAttnComputeKind::kBackwardWeights;
    if (!sdk->version || !sdk->get_stream || !dispatch.context_synchronize ||
        (forward && !sdk->attn_forward) || (data && !sdk->attn_backward_data) ||
        (weights && !sdk->attn_backward_weights))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (const auto status = domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status, status == CUDA_ERROR_NOT_READY ? ExecutionAcceptStatus::kFull
                                                             : ExecutionAcceptStatus::kQuarantined);
    if (!problem.attn.lease->begin_call())
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    struct ActiveCall {
        bool& active;
        CudaDnnRnnLease& lease;
        ~ActiveCall() {
            active = false;
            lease.end_call();
        }
    };
    is_rnn_query_active = true;
    const ActiveCall active{is_rnn_query_active, *problem.attn.lease};
    const auto context = [&]() {
        const auto status = check_context(true);
        if (status != CUDA_SUCCESS) {
            fault();
            output.preparation.has_context_loss = true;
        }
        return status;
    };
    const auto observed_status = [&](cudnnStatus_t status) {
        output.preparation.preflight_status = status;
        output.acceptance.preflight_status = status;
        const auto proof = context();
        return proof != CUDA_SUCCESS            ? proof
               : status == CUDNN_STATUS_SUCCESS ? CUDA_SUCCESS
                                                : CUDA_ERROR_UNKNOWN;
    };
    const auto malformed = [&]() {
        problem.attn.lease->quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    const auto version = invoke(sdk->version);
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    if (version != 91002) return reject(CUDA_ERROR_NOT_SUPPORTED);
    const auto stream = observe_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.preparation.preflight_status = output.acceptance.preflight_status =
            stream.original_status;
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        output.preparation.has_context_loss = is_faulted;
        return reject(
            stream.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stream.status,
            is_faulted ? ExecutionAcceptStatus::kQuarantined : ExecutionAcceptStatus::kInvalid);
    }
    const auto present = [&](std::size_t index) { return problem.buffers[index].allocation != 0; };
    bool is_early_error = std::ranges::any_of(
        problem.descriptors, [](const auto& descriptor) { return !descriptor.native; });
    is_early_error |= !present(2) || !present(3) || !present(4) || !present(forward ? 6 : 7);
    is_early_error |= (problem.weight_bytes && !present(11)) ||
                      (problem.workspace_bytes && !present(13)) ||
                      (problem.reserve_bytes && !present(14));
    if (!weights) is_early_error |= problem.window_presence != 3 || !present(0) || !present(1);
    if (forward)
        is_early_error |= problem.current >= 0 && present(14);
    else {
        is_early_error |= !present(14) || !problem.reserve_bytes;
        if (data) is_early_error |= !present(8) || !present(9) || !present(10);
        if (weights) is_early_error |= problem.weight_bytes && !present(12);
    }
    const auto multiply = [](std::size_t& product, std::size_t value) {
        if (value && product > std::numeric_limits<std::size_t>::max() / value) return false;
        product *= value;
        return true;
    };
    const auto width = [](std::uint32_t type) -> std::size_t {
        switch (type) {
            case 0:
                return 4;
            case 1:
                return 8;
            case 2:
            case 9:
                return 2;
            default:
                return 0;
        }
    };
    struct SeqObservation {
        std::uint32_t type{};
        std::array<int, 4> sizes{};
        std::array<std::uint32_t, 4> axes{};
        std::size_t count{}, bytes{};
    };
    std::array<SeqObservation, 4> observed{};
    const auto observe_sequence = [&](std::size_t index, SeqObservation& value,
                                      std::size_t capacity, int* lengths) -> CUresult {
        cudnnDataType_t type{};
        int rank{};
        std::array<cudnnSeqDataAxis_t, 4> axes{};
        const auto status =
            invoke(sdk->seq_data_get,
                   static_cast<CudaDnnSeqData>(problem.descriptors[index].native), &type, &rank, 4,
                   value.sizes.data(), axes.data(), &value.count, capacity, lengths, nullptr);
        if (const auto proof = observed_status(status); proof != CUDA_SUCCESS) return proof;
        std::memcpy(&value.type, &type, sizeof(value.type));
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            std::memcpy(&value.axes[axis], &axes[axis], sizeof(value.axes[axis]));
        if (rank != 4 || value.type > dnn_profile::data_type_enum_max()) return malformed();
        // Fresh descriptors have rank four but zero dimensions/axes/count.
        // This is an unusable extent, not proof of metadata corruption.
        if (std::ranges::any_of(value.sizes, [](int size) { return size <= 0; }))
            return CUDA_ERROR_INVALID_VALUE;
        if (!width(value.type)) return CUDA_ERROR_NOT_SUPPORTED;
        auto sorted = value.axes;
        std::sort(sorted.begin(), sorted.begin() + 3);
        if (sorted != std::array<std::uint32_t, 4>{0, 1, 2, 3}) return malformed();
        auto count = static_cast<std::size_t>(value.sizes[1]);
        if (!multiply(count, static_cast<std::size_t>(value.sizes[2])) ||
            count > kCudaDnnMaxSequenceLengths)
            return CUDA_ERROR_INVALID_VALUE;
        if (value.count != count) return malformed();
        return CUDA_SUCCESS;
    };
    std::array<std::size_t, 15> sizes{};
    sizes.fill(1);
    sizes[11] = sizes[12] = problem.weight_bytes;
    sizes[13] = problem.workspace_bytes;
    sizes[14] = problem.reserve_bytes;
    bool is_noop{};
    try {
        if (!is_early_error) {
            if (!sdk->seq_data_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            if (const auto status = observe_sequence(0, observed[0], 0, nullptr);
                status != CUDA_SUCCESS)
                return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                 : ExecutionAcceptStatus::kInvalid);
            // This original check precedes every host-window payload access.
            is_early_error = forward && problem.current >= observed[0].sizes[0];
        }
        if (!is_early_error) {
            if (!sdk->attn_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            cudnnDataType_t type{};
            int maximum_query{};
            const auto native =
                invoke(sdk->attn_get, static_cast<CudaDnnAttn>(problem.attn.object.native), nullptr,
                       nullptr, nullptr, &type, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &maximum_query,
                       nullptr, nullptr, nullptr);
            if (const auto status = observed_status(native); status != CUDA_SUCCESS)
                return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                 : ExecutionAcceptStatus::kInvalid);
            std::uint32_t type_bits{};
            std::memcpy(&type_bits, &type, sizeof(type_bits));
            if (type_bits > dnn_profile::data_type_enum_max()) return reject(malformed());
            const auto parent_width = width(type_bits);
            if (!parent_width) return reject(CUDA_ERROR_NOT_SUPPORTED);
            for (std::size_t index = 1; index < observed.size(); ++index)
                if (const auto status = observe_sequence(index, observed[index], 0, nullptr);
                    status != CUDA_SUCCESS)
                    return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                     : ExecutionAcceptStatus::kInvalid);
            for (auto& value : observed) {
                value.bytes = std::max(parent_width, width(value.type));
                for (const int size : value.sizes)
                    if (!multiply(value.bytes, static_cast<std::size_t>(size)))
                        return reject(CUDA_ERROR_INVALID_VALUE);
            }
            sizes[0] = observed[0].count * sizeof(int);
            sizes[1] = observed[1].count * sizeof(int);
            sizes[2] = sizes[8] = observed[0].bytes;
            sizes[3] = sizes[9] = observed[1].bytes;
            sizes[4] = sizes[10] = observed[2].bytes;
            sizes[6] = sizes[7] = observed[3].bytes;
            sizes[5] = std::max(observed[0].bytes, observed[3].bytes);
            if (!weights) {
                if (forward && problem.current >= 0) {
                    std::vector<int> lengths(observed[0].count);
                    SeqObservation again{};
                    if (const auto status =
                            observe_sequence(0, again, lengths.size(), lengths.data());
                        status != CUDA_SUCCESS)
                        return reject(status, is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                         : ExecutionAcceptStatus::kInvalid);
                    if (again.type != observed[0].type || again.sizes != observed[0].sizes ||
                        again.axes != observed[0].axes || again.count != observed[0].count)
                        return reject(malformed());
                    int maximum_length{};
                    for (const int length : lengths) {
                        if (length < 0 || length > observed[0].sizes[0]) return reject(malformed());
                        maximum_length = std::max(maximum_length, length);
                    }
                    is_noop = problem.current >= maximum_length;
                    if (!is_noop) {
                        output.preparation.first = static_cast<std::uint32_t>(problem.current);
                        output.preparation.count = 1;
                    }
                } else {
                    if (maximum_query <= 0 ||
                        static_cast<std::size_t>(maximum_query) > kCudaDnnMaxSequenceLengths)
                        return reject(CUDA_ERROR_INVALID_VALUE);
                    output.preparation.count = static_cast<std::uint32_t>(maximum_query);
                }
            }
        }
        const auto first = static_cast<std::size_t>(output.preparation.first);
        const auto count = static_cast<std::size_t>(output.preparation.count);
        if (first > kCudaDnnMaxSequenceLengths || count > kCudaDnnMaxSequenceLengths - first)
            return reject(CUDA_ERROR_INVALID_VALUE);
        if (!is_preparation)
            for (std::size_t index = 0; index < problem.windows.size(); ++index)
                if (problem.windows[index].size() !=
                    ((problem.window_presence & (1U << index)) ? count : 0))
                    return reject(CUDA_ERROR_INVALID_VALUE);
        std::array<CudaDeviceMemoryView, 15> buffers{};
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (!present(index)) continue;
            const auto memory = session.device_memory(problem.buffers[index], sizes[index]);
            if (!memory || !memory->address ||
                sizes[index] > std::numeric_limits<CUdeviceptr>::max() - memory->address)
                return reject(CUDA_ERROR_INVALID_VALUE);
            buffers[index] = *memory;
        }
        std::array<CudaDeviceMemoryView, 4> dropout_states{};
        if (forward && present(14) && !is_early_error && !is_noop)
            for (std::size_t index = 0; index < dropout_states.size(); ++index) {
                const auto& dropout = problem.attn.dropouts[index];
                if (!dropout.object.native) continue;
                const auto memory = session.resolve_dnn_dropout(dropout);
                if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
                dropout_states[index] = *memory;
            }
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        output.preparation.result.disposition = CudaDnnDisposition::kComplete;
        output.preparation.result.status = CUDA_SUCCESS;
        if (is_preparation) return output;
        auto result = std::make_shared<CudaDnnAttnCompletion>();
        auto call =
            make_cuda_dnn_attn_call(sdk, handle->native, problem, std::move(buffers),
                                    {storage->context, dispatch.execution.context_get_current,
                                     dispatch.context_synchronize},
                                    result, std::move(dropout_states), output.preparation.first);
        output.acceptance.native.acceptance = executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.acceptance.native.acceptance.accepted())
            output.acceptance.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY, ExecutionAcceptStatus::kFull);
    }
}

CudaDnnAttnPreparation CudaExecutionSession::prepare_dnn_attn(
    const CudaSessionDnnAttn& problem) noexcept {
    return implementation_->observe_attn(*this, problem, true, 0).preparation;
}
CudaDnnAttnAcceptance CudaExecutionSession::accept_dnn_attn(
    std::uint64_t sequence, const CudaSessionDnnAttn& problem) noexcept {
    return implementation_->observe_attn(*this, problem, false, sequence).acceptance;
}

CudaDnnRnnAcceptance CudaExecutionSession::accept_dnn_rnn(
    std::uint64_t sequence, const CudaSessionDnnRnn& problem) noexcept {
    auto& state = *implementation_;
    CudaDnnRnnAcceptance output;
    const auto reject = [&](CUresult status,
                            ExecutionAcceptStatus disposition = ExecutionAcceptStatus::kInvalid) {
        output.native.rejection_status = status;
        output.native.acceptance.status = disposition;
        return output;
    };
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN, ExecutionAcceptStatus::kQuarantined);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED, ExecutionAcceptStatus::kClosed);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active)
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    if (!valid_cuda_dnn_rnn_descriptors(problem)) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || handle == state.storage->dnn.end() || !handle->native || handle->is_uncertain ||
        handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (!sdk->get_stream || !sdk->rnn_data_get || !state.dispatch.context_synchronize ||
        (problem.kind == CudaDnnRnnComputeKind::kForward && !sdk->rnn_forward) ||
        (problem.kind == CudaDnnRnnComputeKind::kBackwardData && !sdk->rnn_backward_data) ||
        (problem.kind == CudaDnnRnnComputeKind::kBackwardWeights && !sdk->rnn_backward_weights))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!problem.rnn.lease->begin_call())
        return reject(CUDA_ERROR_NOT_READY, ExecutionAcceptStatus::kFull);
    struct ActiveCall {
        bool& active;
        CudaDnnRnnLease& lease;
        ~ActiveCall() {
            active = false;
            lease.end_call();
        }
    };
    if (const auto status = state.domain_cold_status(); status != CUDA_SUCCESS)
        return reject(status);
    state.is_rnn_query_active = true;
    const ActiveCall active{state.is_rnn_query_active, *problem.rnn.lease};
    const auto context = [&]() {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto proof = context(); proof != CUDA_SUCCESS)
        return reject(proof, ExecutionAcceptStatus::kQuarantined);
    const auto stream = state.observe_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.preflight_status = stream.original_status;
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        return reject(stream.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : stream.status,
                      state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                       : ExecutionAcceptStatus::kInvalid);
    }
    const auto malformed = [&]() {
        problem.rnn.lease->quarantine();
        return CUDA_ERROR_UNKNOWN;
    };
    const auto multiply = [](std::size_t& product, std::size_t value) {
        if (value && product > std::numeric_limits<std::size_t>::max() / value) return false;
        product *= value;
        return true;
    };
    struct DataExtent {
        std::size_t bytes{}, batch{}, width{};
    };
    std::array<DataExtent, 2> data{};
    const auto data_extent = [&](std::size_t index) -> CUresult {
        cudnnDataType_t type{};
        cudnnRNNDataLayout_t layout{};
        int maximum{}, batch{}, vector{};
        alignas(double) std::array<std::byte, 8> padding{};
        const auto native = static_cast<CudaDnnRnnData>(problem.descriptors[index].native);
        auto status = state.invoke(sdk->rnn_data_get, native, &type, &layout, &maximum, &batch,
                                   &vector, 0, nullptr, padding.data());
        output.preflight_status = status;
        if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
        if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_UNKNOWN;
        std::uint32_t type_bits{}, layout_bits{};
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        std::memcpy(&layout_bits, &layout, sizeof(layout_bits));
        if (type_bits > dnn_profile::data_type_enum_max() || layout_bits > 3) return malformed();
        if (type_bits > 2 || layout_bits > 2) return CUDA_ERROR_NOT_SUPPORTED;
        if (maximum <= 0 || batch <= 0 || vector <= 0 ||
            static_cast<std::size_t>(batch) > kCudaDnnMaxSequenceLengths)
            return CUDA_ERROR_INVALID_VALUE;
        std::size_t elements = static_cast<std::size_t>(maximum);
        if (!multiply(elements, static_cast<std::size_t>(batch))) return CUDA_ERROR_INVALID_VALUE;
        if (layout_bits == 1) {
            std::vector<int> lengths(static_cast<std::size_t>(batch));
            cudnnDataType_t again_type{};
            cudnnRNNDataLayout_t again_layout{};
            int again_maximum{}, again_batch{}, again_vector{};
            status =
                state.invoke(sdk->rnn_data_get, native, &again_type, &again_layout, &again_maximum,
                             &again_batch, &again_vector, batch, lengths.data(), padding.data());
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS) return proof;
            if (status != CUDNN_STATUS_SUCCESS) return CUDA_ERROR_UNKNOWN;
            std::uint32_t again_type_bits{}, again_layout_bits{};
            std::memcpy(&again_type_bits, &again_type, sizeof(again_type_bits));
            std::memcpy(&again_layout_bits, &again_layout, sizeof(again_layout_bits));
            if (again_type_bits != type_bits || again_layout_bits != layout_bits ||
                again_maximum != maximum || again_batch != batch || again_vector != vector)
                return malformed();
            elements = 0;
            for (const int length : lengths) {
                if (length < 0 || length > maximum) return malformed();
                const auto value = static_cast<std::size_t>(length);
                if (elements > std::numeric_limits<std::size_t>::max() - value)
                    return CUDA_ERROR_INVALID_VALUE;
                elements += value;
            }
        }
        constexpr std::array<std::size_t, 3> widths{4, 8, 2};
        if (!multiply(elements, static_cast<std::size_t>(vector)) ||
            !multiply(elements, widths[type_bits]))
            return CUDA_ERROR_INVALID_VALUE;
        data[index] = {elements, static_cast<std::size_t>(batch), widths[type_bits]};
        return CUDA_SUCCESS;
    };
    try {
        for (std::size_t index = 0; index < data.size(); ++index)
            if (const auto status = data_extent(index); status != CUDA_SUCCESS)
                return reject(status, state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                                       : ExecutionAcceptStatus::kInvalid);
        std::array<std::size_t, 2> state_bytes{};
        for (std::size_t index = 0; index < state_bytes.size(); ++index) {
            const auto begin = index == 0 ? std::size_t{5} : std::size_t{9};
            bool needed{};
            for (auto slot = begin; slot < begin + 4; ++slot)
                needed |= problem.buffers[slot].allocation != 0;
            if (!needed) continue;
            if (!sdk->tensor_get_nd || !sdk->tensor_size) return reject(CUDA_ERROR_NOT_SUPPORTED);
            const auto descriptor =
                static_cast<cudnnTensorDescriptor_t>(problem.descriptors[index + 2].native);
            cudnnDataType_t type{};
            int rank{};
            std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
            auto status = state.invoke(sdk->tensor_get_nd, descriptor, CUDNN_DIM_MAX, &type, &rank,
                                       sizes.data(), strides.data());
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS)
                return reject(proof, ExecutionAcceptStatus::kQuarantined);
            if (status != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_UNKNOWN);
            std::uint32_t type_bits{};
            std::memcpy(&type_bits, &type, sizeof(type_bits));
            if (rank < 0 || rank > CUDNN_DIM_MAX || type_bits > dnn_profile::data_type_enum_max())
                return reject(malformed());
            // Native RNN reads three dimension slots; smaller ranks can hide
            // historical tail dimensions that the public getter cannot prove.
            if (rank < 3) return reject(CUDA_ERROR_INVALID_VALUE);
            if (type_bits > 2) return reject(CUDA_ERROR_NOT_SUPPORTED);
            std::size_t hull = 1;
            std::size_t dense = 1;
            for (int dim = 0; dim < rank; ++dim) {
                if (sizes[dim] <= 0 || strides[dim] <= 0) return reject(CUDA_ERROR_INVALID_VALUE);
                auto offset = static_cast<std::size_t>(sizes[dim] - 1);
                if (!multiply(dense, static_cast<std::size_t>(sizes[dim])) ||
                    !multiply(offset, static_cast<std::size_t>(strides[dim])) ||
                    hull > std::numeric_limits<std::size_t>::max() - offset)
                    return reject(CUDA_ERROR_INVALID_VALUE);
                hull += offset;
            }
            constexpr std::array<std::size_t, 3> widths{4, 8, 2};
            // Native packed checks use narrow arithmetic, and state accesses
            // follow RNNData type dispatch rather than every state descriptor.
            const auto compute_width = std::max({widths[type_bits], data[0].width, data[1].width});
            if (!multiply(hull, widths[type_bits]) || !multiply(dense, compute_width))
                return reject(CUDA_ERROR_INVALID_VALUE);
            status = state.invoke(sdk->tensor_size, descriptor, &state_bytes[index]);
            output.preflight_status = status;
            if (const auto proof = context(); proof != CUDA_SUCCESS)
                return reject(proof, ExecutionAcceptStatus::kQuarantined);
            if (status != CUDNN_STATUS_SUCCESS) return reject(CUDA_ERROR_UNKNOWN);
            if (state_bytes[index] < hull) return reject(malformed());
            state_bytes[index] = std::max(state_bytes[index], dense);
        }
        std::array<std::size_t, 16> sizes{};
        sizes[0] = std::max(data[0].batch, data[1].batch) * sizeof(std::int32_t);
        sizes[1] = sizes[4] = data[0].bytes;
        sizes[2] = sizes[3] = data[1].bytes;
        std::fill(sizes.begin() + 5, sizes.begin() + 9, state_bytes[0]);
        std::fill(sizes.begin() + 9, sizes.begin() + 13, state_bytes[1]);
        sizes[13] = problem.weight_bytes;
        sizes[14] = problem.workspace_bytes;
        sizes[15] = problem.reserve_bytes;
        std::array<CudaDeviceMemoryView, 16> buffers{};
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (!problem.buffers[index].allocation) continue;
            const auto memory = device_memory(problem.buffers[index], sizes[index]);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            buffers[index] = *memory;
        }
        std::array<CudaDeviceMemoryView, 4> dropout_states{};
        for (std::size_t index = 0; index < dropout_states.size(); ++index) {
            const auto& dropout = problem.rnn.dropouts[index];
            if (!dropout.object.native) continue;
            const auto memory = resolve_dnn_dropout(dropout);
            if (!memory) return reject(CUDA_ERROR_INVALID_VALUE);
            dropout_states[index] = *memory;
        }
        if (const auto proof = context(); proof != CUDA_SUCCESS)
            return reject(proof, ExecutionAcceptStatus::kQuarantined);
        auto result = std::make_shared<CudaDnnRnnCompletion>();
        auto call = make_cuda_dnn_rnn_call(
            sdk, handle->native, problem, std::move(buffers),
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result, std::move(dropout_states));
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY, ExecutionAcceptStatus::kFull);
    }
}

CudaDnnFindAcceptance CudaExecutionSession::accept_dnn_find(
    std::uint64_t sequence, const CudaSessionDnnConvolution& problem, int requested) noexcept {
    auto& state = *implementation_;
    CudaDnnFindAcceptance output;
    if (!state.owns() || !state.executor || state.is_faulted ||
        state.executor->snapshot().is_quarantined) {
        output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    if (state.is_closing) {
        output.native.acceptance.status = ExecutionAcceptStatus::kClosed;
        return output;
    }
    if (state.check_context() != CUDA_SUCCESS) return output;
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->convolution_forward_find || !state.dispatch.context_synchronize ||
        handle == state.storage->dnn.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !valid_cuda_dnn_convolution_descriptors(problem))
        return output;
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS || stream.original_status != CUDNN_STATUS_SUCCESS) {
        output.native.original_status = stream.original_status;
        if (state.is_faulted) output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    // The SDK maximum includes math variants, not just the algorithm enums.
    // This metadata query does not execute/replay the benchmark itself.
    const auto maximum = query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kMaxCount);
    if (maximum.result.status != CUDA_SUCCESS ||
        maximum.result.original_status != CUDNN_STATUS_SUCCESS) {
        output.native.original_status = maximum.result.original_status;
        if (state.is_faulted) output.native.acceptance.status = ExecutionAcceptStatus::kQuarantined;
        return output;
    }
    if (maximum.result.value > kCudaDnnMaxConvolutionAlgorithms) return output;
    const auto capacity = std::min(requested, static_cast<int>(maximum.result.value));
    try {
        auto result = std::make_shared<CudaDnnConvolutionFindResult>();
        auto call = make_cuda_dnn_convolution_find_call(
            sdk, handle->native, problem, capacity,
            {state.storage->context, state.dispatch.execution.context_get_current,
             state.dispatch.context_synchronize},
            result);
        output.native.acceptance = state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
        if (output.native.acceptance.accepted()) output.result = std::move(result);
        return output;
    } catch (...) {
        output.native.acceptance.status = ExecutionAcceptStatus::kFull;
        return output;
    }
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_convolution(
    std::uint64_t sequence, const CudaSessionDnnConvolution& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    constexpr auto algorithm_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDNN_CONVOLUTION_FWD_ALGO_COUNT) + 1U) - 1U;
    if (!sdk || !sdk->convolution_forward || !sdk->filter_size || !sdk->tensor_get_nd ||
        !sdk->tensor_size || handle == state.storage->dnn.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !valid_cuda_dnn_convolution_descriptors(problem) ||
        problem.algorithm > algorithm_max)
        return {};
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 4> buffers{};
    for (std::size_t index = 0; index != 3; ++index) {
        const auto descriptor = problem.descriptors[index == 2 ? 3 : index].native;
        std::size_t bytes{};
        cudnnStatus_t status{};
        if (index == 1) {
            status = state.invoke(sdk->filter_size, static_cast<CudaDnnFilterHandle>(descriptor),
                                  &bytes);
        } else {
            if (!index) {
                cudnnDataType_t type{};
                int rank{};
                std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
                status = state.invoke(sdk->tensor_get_nd,
                                      static_cast<cudnnTensorDescriptor_t>(descriptor),
                                      CUDNN_DIM_MAX, &type, &rank, sizes.data(), strides.data());
                if (state.check_context() != CUDA_SUCCESS) {
                    state.fault();
                    return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
                }
                if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
                std::uint32_t type_bits{};
                std::memcpy(&type_bits, &type, sizeof(type_bits));
                if (rank <= 0 || rank > CUDNN_DIM_MAX ||
                    type_bits > dnn_profile::data_type_enum_max()) {
                    state.fault();
                    return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
                }
                const auto width = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
                if (problem.scalar_bytes != width) return {};
            }
            status = state.invoke(sdk->tensor_size,
                                  static_cast<cudnnTensorDescriptor_t>(descriptor), &bytes);
        }
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    const auto& workspace = problem.buffers[3];
    if (workspace.allocation) {
        const auto memory = device_memory(workspace, problem.workspace_bytes);
        if (!memory) return {};
        buffers[3] = *memory;
    } else if (workspace.offset || problem.workspace_bytes)
        return {};
    try {
        return {state.executor->accept_library(
                    sequence, {.stream = handle->stream,
                               .parameters = {},
                               .call = make_cuda_dnn_convolution_call(sdk, handle->native, problem,
                                                                      std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}

CudaDnnComputeAcceptance CudaExecutionSession::accept_dnn_compute(
    std::uint64_t sequence, const CudaSessionDnnCompute& problem) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    if (state.is_closing) return {{.status = ExecutionAcceptStatus::kClosed, .receipt = {}}, {}};
    if (state.check_context() != CUDA_SUCCESS) return {};
    const auto& sdk = state.dispatch.dnn;
    const auto handle =
        std::ranges::find(state.storage->dnn, problem.handle, &Implementation::Dnn::id);
    if (!sdk || !sdk->tensor_get_nd || !sdk->tensor_size || handle == state.storage->dnn.end() ||
        handle->is_uncertain || handle->has_destroy_attempt)
        return {};
    if (problem.kind == CudaDnnComputeKind::kActivationForward) {
        if (!sdk->activation_forward || problem.activation.kind != CudaDnnObjectKind::kActivation ||
            !problem.activation.native || !problem.activation.lifetime)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kPoolingForward) {
        if (!sdk->pooling_forward || problem.pooling.kind != CudaDnnObjectKind::kPooling ||
            !problem.pooling.native || !problem.pooling.lifetime || problem.algorithm ||
            problem.mode)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kSoftmaxForward) {
        // The unfixed SDK algorithm enum represents [0,3]. Unknown but
        // representable value 3 still reaches the original native error path.
        if (!sdk->softmax_forward || problem.algorithm > 3 || problem.mode > 1 ||
            problem.pooling.native || problem.pooling.lifetime)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kLrnForward) {
        if (!sdk->lrn_forward || problem.lrn.kind != CudaDnnObjectKind::kLrn ||
            !problem.lrn.native || !problem.lrn.lifetime || problem.algorithm || problem.mode > 1)
            return {};
    } else if (problem.kind == CudaDnnComputeKind::kTransformTensor) {
        if (!sdk->transform_tensor || problem.algorithm || problem.mode ||
            problem.activation.native || problem.activation.lifetime || problem.pooling.native ||
            problem.pooling.lifetime || problem.lrn.native || problem.lrn.lifetime)
            return {};
    } else if (problem.kind != CudaDnnComputeKind::kAddTensor || !sdk->add_tensor) {
        return {};
    }
    const auto stream = get_dnn_stream(problem.handle);
    if (stream.status != CUDA_SUCCESS)
        return {{.status = state.is_faulted ? ExecutionAcceptStatus::kQuarantined
                                            : ExecutionAcceptStatus::kInvalid,
                 .receipt = {}},
                stream.original_status};
    if (stream.original_status != CUDNN_STATUS_SUCCESS) return {{}, stream.original_status};
    std::array<CudaDeviceMemoryView, 2> buffers;
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto& descriptor = problem.tensors[index];
        if (descriptor.kind != CudaDnnObjectKind::kTensor || !descriptor.native ||
            !descriptor.lifetime)
            return {};
        const auto native = static_cast<cudnnTensorDescriptor_t>(descriptor.native);
        cudnnDataType_t type{};
        int dimensions{};
        std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
        auto status = state.invoke(sdk->tensor_get_nd, native, CUDNN_DIM_MAX, &type, &dimensions,
                                   sizes.data(), strides.data());
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        std::uint32_t type_bits{};
        static_assert(sizeof(type) == sizeof(type_bits));
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (dimensions <= 0 || dimensions > CUDNN_DIM_MAX ||
            type_bits > dnn_profile::data_type_enum_max()) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        const auto scalar_bytes = type_bits == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
        if (problem.scalar_bytes != scalar_bytes) return {};
        std::size_t bytes{};
        status = state.invoke(sdk->tensor_size, native, &bytes);
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, status};
        }
        if (status != CUDNN_STATUS_SUCCESS) return {{}, status};
        if (!bytes) return {};
        const auto memory = device_memory(problem.buffers[index], bytes);
        if (!memory) return {};
        buffers[index] = *memory;
    }
    if (problem.kind == CudaDnnComputeKind::kTransformTensor) {
        const auto& x = buffers[0];
        const auto& y = buffers[1];
        // Unlike softmax, the original transformation contract forbids
        // overlapping tensor storage. Subtraction avoids address-end overflow
        // and still permits adjacent/disjoint views of the same allocation.
        const bool is_disjoint = x.address <= y.address ? y.address - x.address >= x.bytes
                                                        : x.address - y.address >= y.bytes;
        if (!is_disjoint) return {};
    }
    if (state.check_context() != CUDA_SUCCESS) {
        state.fault();
        return {{.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}}, {}};
    }
    try {
        return {state.executor->accept_library(
                    sequence,
                    {.stream = handle->stream,
                     .parameters = {},
                     .call = make_cuda_dnn_call(sdk, handle->native, problem, std::move(buffers))}),
                {}};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kFull, .receipt = {}}, {}};
    }
}
#include "internal/cuda_dnn_legacy_session.inc"
#endif

CudaSessionRandResult CudaExecutionSession::create_random(std::uint32_t type) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto& sdk = state.dispatch.random;
    if (!sdk.create || !sdk.set_seed || !sdk.destroy)
        return random_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    // CUDA 12.6's unfixed enum has the C++ value range [0,255]. Unknown
    // representable values still reach the SDK; arbitrary wire bits must not
    // become an invalid enum load before that boundary.
    if (type > 255) return random_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    if (!state.has_id_space || state.storage->random.size() == state.limits.max_objects)
        return random_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    Implementation::Random handle{.id = state.next_id};
    curandRngType_t native_type{};
    static_assert(sizeof(native_type) == sizeof(type));
    std::memcpy(&native_type, &type, sizeof(type));
    auto result = random_completed(state.invoke(sdk.create, &handle.native, native_type));
    if (*result.original_status != CURAND_STATUS_SUCCESS && !handle.native) return result;
    (void)state.consume_id();
    handle.is_uncertain = *result.original_status != CURAND_STATUS_SUCCESS || !handle.native;
    for (auto& existing : state.storage->random)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    state.storage->random.push_back(handle);  // Capacity reserved before any SDK call.
    if (handle.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = handle.id;
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): logical generator ID and SDK seed ABI.
CudaSessionRandResult CudaExecutionSession::set_random_seed(std::uint64_t id,
                                                            std::uint64_t seed) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto found = std::ranges::find(state.storage->random, id, &Implementation::Random::id);
    if (found == state.storage->random.end())
        return random_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return random_gate(rejected(CUDA_ERROR_UNKNOWN));
    return random_completed(state.invoke(state.dispatch.random.set_seed, found->native, seed));
}
CudaSessionRandResult CudaExecutionSession::destroy_random(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return random_gate(gate);
    const auto found = std::ranges::find(state.storage->random, id, &Implementation::Random::id);
    if (found == state.storage->random.end())
        return random_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return random_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    const auto result =
        random_completed(state.invoke(state.dispatch.random.destroy, found->native));
    if (*result.original_status == CURAND_STATUS_SUCCESS)
        state.storage->random.erase(found);
    else {
        found->is_uncertain = true;
        state.fault();
    }
    return result;
}

CudaSessionBlasResult CudaExecutionSession::create_blas() noexcept {
    // Existing cuBLAS preparation and callback identity remain unchanged.
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto& dispatch = state.dispatch.blas;
    if (!dispatch.create || !dispatch.destroy || !dispatch.set_stream || !dispatch.sgemm)
        return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (state.stream_config.uses_default_proxy && !dispatch.get_stream)
        return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->blas.size() == state.limits.max_objects)
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    const auto prepared = state.prepare_sgemm();
    if (prepared != CUDA_SUCCESS) return blas_gate(rejected(prepared));
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    Implementation::Blas handle{.id = state.next_id};
    auto result = blas_completed(state.invoke(dispatch.create, &handle.native));
    const auto proof =
        state.stream_config.uses_default_proxy ? state.check_context(true) : CUDA_SUCCESS;
    if (*result.original_status != CUBLAS_STATUS_SUCCESS && !handle.native) {
        if (proof != CUDA_SUCCESS) {
            state.fault();
            result.status = proof;
            result.disposition = CudaSessionDisposition::kRejected;
        }
        return result;
    }
    (void)state.consume_id();
    handle.is_uncertain =
        *result.original_status != CUBLAS_STATUS_SUCCESS || !handle.native || proof != CUDA_SUCCESS;
    handle.lt.id = handle.id;
    handle.lt.native = reinterpret_cast<cublasLtHandle_t>(handle.native);
    for (auto& existing : state.storage->blas)
        if (existing.native == handle.native) {
            existing.is_uncertain = existing.lt.is_uncertain = handle.is_uncertain = true;
        }
    for (auto& existing : state.storage->lt)
        if (static_cast<void*>(existing.native) == static_cast<void*>(handle.native))
            existing.is_uncertain = handle.is_uncertain = true;
    handle.lt.is_uncertain = handle.is_uncertain;
    state.storage->blas.push_back(handle);
    if (!handle.is_uncertain && state.storage->default_proxy) {
        const auto configured =
            state.invoke(dispatch.set_stream, handle.native,
                         reinterpret_cast<cudaStream_t>(state.storage->default_proxy));
        const auto configured_proof = state.check_context(true);
        handle.is_uncertain =
            configured != CUBLAS_STATUS_SUCCESS || configured_proof != CUDA_SUCCESS;
        if (!handle.is_uncertain) {
            cudaStream_t observed{};
            const auto queried = state.invoke(dispatch.get_stream, handle.native, &observed);
            const auto queried_proof = state.check_context(true);
            handle.is_uncertain =
                queried != CUBLAS_STATUS_SUCCESS || queried_proof != CUDA_SUCCESS ||
                reinterpret_cast<CUstream>(observed) != state.storage->default_proxy;
        }
        state.storage->blas.back().is_uncertain = handle.is_uncertain;
        state.storage->blas.back().lt.is_uncertain = handle.is_uncertain;
    }
    if (handle.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else {
        result.id = handle.id;
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::destroy_blas(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto found = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (found == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = blas_completed(state.invoke(state.dispatch.blas.destroy, found->native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS) {
        found->is_uncertain = true;
        state.fault();
    } else {
        state.storage->blas.erase(found);
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::create_lt() noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto& dispatch = state.dispatch.lt;
    if (!dispatch.create || !dispatch.destroy) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    if (!state.has_id_space || state.storage->lt.size() == state.limits.max_objects)
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    const auto prepared = state.prepare_lt();
    if (prepared != CUDA_SUCCESS) return blas_gate(rejected(prepared));
    Implementation::Lt handle{.id = state.next_id};
    try {
        handle.algorithms.reserve(state.limits.max_objects);
    } catch (...) {
        return blas_gate(rejected(CUDA_ERROR_OUT_OF_MEMORY));
    }
    auto result = blas_completed(state.invoke(dispatch.create, &handle.native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS && !handle.native) return result;
    (void)state.consume_id();
    handle.is_uncertain =
        *result.original_status != CUBLAS_STATUS_SUCCESS || !handle.native || state.is_faulted;
    for (auto& existing : state.storage->lt)
        if (existing.native == handle.native) existing.is_uncertain = handle.is_uncertain = true;
    for (auto& existing : state.storage->blas)
        if (static_cast<void*>(existing.native) == static_cast<void*>(handle.native))
            existing.is_uncertain = existing.lt.is_uncertain = handle.is_uncertain = true;
    state.storage->lt.push_back(std::move(handle));
    const auto& retained = state.storage->lt.back();
    if (retained.is_uncertain) {
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
    } else
        result.id = retained.id;
    return result;
}
CudaSessionLtAlgorithms CudaExecutionSession::query_lt_algorithms(
    const CudaSessionLtQuery& query) noexcept {
    auto& state = *implementation_;
    CudaSessionLtAlgorithms output;
    const auto reject = [&](CUresult status) {
        output.result.status = status;
        output.result.disposition = CudaSessionDisposition::kRejected;
        output.algorithms.clear();
        return std::move(output);
    };
    if (!state.owns()) return reject(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return reject(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope active(state.is_native_issue_active);
    const auto context_status = state.check_context(true);
    if (context_status != CUDA_SUCCESS) return reject(context_status);
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    auto* handle = state.lt_handle(query.handle);
    if (!handle || handle->is_uncertain || handle->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (state.persistent.replace_sgemm && !state.storage->lt_ids.back()) {
        // The outer native scope already excludes reentry. Only the physical
        // quiescence part of cold_gate applies to this internal preparation.
        if (!state.executor->resources_quiescent()) {
            output.result = blas_gate(rejected(CUDA_ERROR_NOT_READY));
            output.result.disposition = CudaSessionDisposition::kDeferred;
            return output;
        }
        const auto prepared = state.prepare_lt();
        if (prepared != CUDA_SUCCESS) return reject(prepared);
    }
    const auto proof = [&] {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    const auto valid_view = [](const CudaLtObjectView& view, CudaLtObjectKind kind) {
        return view.kind == kind && view.native && view.lifetime;
    };
    if (!valid_view(query.operation, CudaLtObjectKind::kOperation) ||
        !std::ranges::all_of(query.matrices, [&](const auto& view) {
            return valid_view(view, CudaLtObjectKind::kLayout);
        }))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto& sdk = state.dispatch.lt;
    if (!sdk.version || (query.algorithm ? !sdk.algorithm_check : !sdk.heuristic))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (!query.algorithm && (query.requested <= 0 || query.requested > 64 || !query.preference ||
                             !valid_view(*query.preference, CudaLtObjectKind::kPreference)))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto version = state.invoke(sdk.version);
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (!version || (handle->sdk_version && handle->sdk_version != version)) {
        state.fault();
        return reject(CUDA_ERROR_UNKNOWN);
    }
    const auto same_algorithm = [](const auto& left, const auto& right) {
        static_assert(sizeof(cublasLtMatmulAlgo_t) == 64);
        return std::memcmp(&left, &right, sizeof(left)) == 0;
    };
    if (query.algorithm && !std::ranges::any_of(handle->algorithms, [&](const auto& known) {
            return same_algorithm(known, *query.algorithm);
        }))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    try {
        output.algorithms.resize(query.algorithm ? 1 : static_cast<std::size_t>(query.requested));
        const auto operation = static_cast<cublasLtMatmulDesc_t>(query.operation.native);
        std::array<cublasLtMatrixLayout_t, 4> layouts{};
        for (std::size_t index = 0; index < layouts.size(); ++index)
            layouts[index] = static_cast<cublasLtMatrixLayout_t>(query.matrices[index].native);
        if (!query.has_c_layout) layouts[2] = nullptr;
        int count{};
        if (query.algorithm) {
            output.algorithms[0].algo = *query.algorithm;
            output.result = blas_completed(
                state.invoke(sdk.algorithm_check, handle->native, operation, layouts[0], layouts[1],
                             layouts[2], layouts[3], &*query.algorithm, output.algorithms.data()));
            count = 1;
        } else {
            // Limited-by-algorithm search consumes caller algorithm state and
            // requires its own validated initialization/configuration contract.
            if (!sdk.preference_get) return reject(CUDA_ERROR_NOT_SUPPORTED);
            std::uint32_t mode{};
            std::size_t written{};
            const auto status =
                state.invoke(sdk.preference_get,
                             static_cast<cublasLtMatmulPreference_t>(query.preference->native),
                             CUBLASLT_MATMUL_PREF_SEARCH_MODE, &mode, sizeof(mode), &written);
            if (const auto context = proof(); context != CUDA_SUCCESS) return reject(context);
            if (status != CUBLAS_STATUS_SUCCESS || written != sizeof(mode))
                return reject(CUDA_ERROR_UNKNOWN);
            if (mode != 0) return reject(CUDA_ERROR_NOT_SUPPORTED);
            output.result = blas_completed(state.invoke(
                sdk.heuristic, handle->native, operation, layouts[0], layouts[1], layouts[2],
                layouts[3], static_cast<cublasLtMatmulPreference_t>(query.preference->native),
                query.requested, output.algorithms.data(), &count));
        }
        if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
        output.sdk_version = version;
        if (output.result.original_status != CUBLAS_STATUS_SUCCESS) {
            output.algorithms.clear();
            return output;
        }
        if (count < 0 || static_cast<std::size_t>(count) > output.algorithms.size()) {
            state.fault();
            output.result.status = CUDA_ERROR_UNKNOWN;
            output.result.disposition = CudaSessionDisposition::kRejected;
            output.algorithms.clear();
            return output;
        }
        output.algorithms.resize(static_cast<std::size_t>(count));
        // Reserve all provenance records before publishing any algorithm. The
        // query performs no application GPU work and may reject a full cache.
        auto fresh = handle->algorithms;
        for (const auto& item : output.algorithms) {
            if (item.state != CUBLAS_STATUS_SUCCESS ||
                std::ranges::any_of(
                    fresh, [&](const auto& known) { return same_algorithm(known, item.algo); }))
                continue;
            if (fresh.size() == state.limits.max_objects) return reject(CUDA_ERROR_OUT_OF_MEMORY);
            fresh.push_back(item.algo);
        }
        handle->algorithms = std::move(fresh);
        handle->sdk_version = version;
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionBlasResult CudaExecutionSession::destroy_lt(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto found = std::ranges::find(state.storage->lt, id, &Implementation::Lt::id);
    if (found == state.storage->lt.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (found->is_uncertain || found->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    found->has_destroy_attempt = true;
    auto result = blas_completed(state.invoke(state.dispatch.lt.destroy, found->native));
    if (*result.original_status != CUBLAS_STATUS_SUCCESS) {
        found->is_uncertain = true;
        state.fault();
    } else
        state.storage->lt.erase(found);
    return result;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): distinct scoped resource IDs.
CudaSessionBlasResult CudaExecutionSession::set_blas_stream(std::uint64_t id,
                                                            std::uint64_t stream_id) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    CUstream stream = state.storage->default_proxy;
    if (stream_id) {
        const auto found =
            std::ranges::find(state.storage->streams, stream_id, &Implementation::Stream::id);
        if (found == state.storage->streams.end())
            return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
        stream = found->native;
    }
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    auto result = blas_completed(state.invoke(state.dispatch.blas.set_stream, handle->native,
                                              reinterpret_cast<cudaStream_t>(stream)));
    if (*result.original_status == CUBLAS_STATUS_SUCCESS) {
        handle->stream = stream_id;
        // The SDK resets workspace even when the stream identity is unchanged.
        handle->has_workspace = false;
        handle->workspace = {};
        handle->workspace_bytes = 0;
    } else
        state.fault();
    if (state.stream_config.uses_default_proxy && state.check_context(true) != CUDA_SUCCESS) {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::set_blas_workspace(std::uint64_t id,
                                                               CudaAllocationRange workspace,
                                                               std::size_t bytes) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    if (handle->is_uncertain || handle->has_destroy_attempt)
        return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    if (!state.dispatch.blas.set_workspace) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
    CUdeviceptr address{};
    if (workspace.allocation) {
        const auto memory = device_memory(workspace, bytes ? bytes : 1);
        if (!memory) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        address = memory->address;
    } else if (bytes || workspace.offset)
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    // Let the original SDK reject misalignment and preserve its own status.
    // NOLINTBEGIN(performance-no-int-to-ptr): exact SDK device-address ABI, never a host access.
    auto result = blas_completed(state.invoke(state.dispatch.blas.set_workspace, handle->native,
                                              reinterpret_cast<void*>(address), bytes));
    // NOLINTEND(performance-no-int-to-ptr)
    if (*result.original_status == CUBLAS_STATUS_SUCCESS) {
        handle->has_workspace = true;
        handle->workspace = workspace;
        handle->workspace_bytes = bytes;
    } else if (*result.original_status != CUBLAS_STATUS_INVALID_VALUE &&
               *result.original_status != CUBLAS_STATUS_NOT_SUPPORTED) {
        state.fault();
    }
    return result;
}
CudaSessionBlasResult CudaExecutionSession::set_blas_attribute(std::uint64_t id,
                                                               CudaBlasAttribute attribute,
                                                               std::uint64_t value) noexcept {
    if (attribute == CudaBlasAttribute::kStream) return set_blas_stream(id, value);
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return blas_gate(gate);
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    cublasStatus_t status{};
    if (attribute == CudaBlasAttribute::kPointerMode) {
        if (value > 1) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_pointer_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto mode = static_cast<cublasPointerMode_t>(value);
        status = state.invoke(state.dispatch.blas.set_pointer_mode, handle->native, mode);
        if (status == CUBLAS_STATUS_SUCCESS) handle->pointer_mode = mode;
    } else if (attribute == CudaBlasAttribute::kMathMode) {
        // SDK enum values, including combined reduction flags, fit 0..31.
        // The actual library rejects unsupported combinations in that range.
        if (value > 31) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_math_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        status = state.invoke(state.dispatch.blas.set_math_mode, handle->native,
                              static_cast<cublasMath_t>(value));
        if (status == CUBLAS_STATUS_SUCCESS) handle->math_mode = value;
    } else if (attribute == CudaBlasAttribute::kAtomicsMode) {
        if (value > 1) return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_atomics_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto mode = static_cast<cublasAtomicsMode_t>(value);
        status = state.invoke(state.dispatch.blas.set_atomics_mode, handle->native, mode);
        if (status == CUBLAS_STATUS_SUCCESS) handle->atomics_mode = mode;
    } else if (attribute == CudaBlasAttribute::kSmCountTarget) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
        if (!state.dispatch.blas.set_sm_count_target)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        const auto count = static_cast<int>(value);
        status = state.invoke(state.dispatch.blas.set_sm_count_target, handle->native, count);
        if (status == CUBLAS_STATUS_SUCCESS) handle->sm_count_target = count;
    } else
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    if (status != CUBLAS_STATUS_SUCCESS && status != CUBLAS_STATUS_INVALID_VALUE &&
        status != CUBLAS_STATUS_NOT_SUPPORTED)
        state.fault();
    return blas_completed(status);
}
CudaSessionBlasResult CudaExecutionSession::get_blas_attribute(
    std::uint64_t id, CudaBlasAttribute attribute) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return blas_gate(rejected(checked));
    if (state.is_faulted) return blas_gate(rejected(CUDA_ERROR_UNKNOWN));
    const auto handle = std::ranges::find(state.storage->blas, id, &Implementation::Blas::id);
    if (handle == state.storage->blas.end()) return blas_gate(rejected(CUDA_ERROR_INVALID_HANDLE));
    CudaSessionBlasResult result;
    const auto invalid_observation = [&] {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;  // Preserve the actual successful SDK return separately.
    };
    if (attribute == CudaBlasAttribute::kStream) {
        if (!state.dispatch.blas.get_stream) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        std::optional<NativeExecutionScope> proxy_scope;
        if (state.stream_config.uses_default_proxy)
            proxy_scope.emplace(state.is_native_issue_active);
        cudaStream_t stream{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_stream, handle->native, &stream));
        if (state.stream_config.uses_default_proxy && state.check_context(true) != CUDA_SUCCESS)
            return invalid_observation();
        if (result.original_status != CUBLAS_STATUS_SUCCESS) return result;
        if (!handle->stream)
            return reinterpret_cast<CUstream>(stream) == state.storage->default_proxy
                       ? result
                       : invalid_observation();
        if (!stream) return handle->stream ? invalid_observation() : result;
        const auto found =
            std::ranges::find(state.storage->streams, reinterpret_cast<CUstream>(stream),
                              &Implementation::Stream::native);
        if (found == state.storage->streams.end() || found->id != handle->stream)
            return invalid_observation();
        result.value = found->id;
    } else if (attribute == CudaBlasAttribute::kPointerMode) {
        if (!state.dispatch.blas.get_pointer_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasPointerMode_t mode{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_pointer_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 1 || bits != static_cast<std::uint32_t>(handle->pointer_mode))
                return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kMathMode) {
        if (!state.dispatch.blas.get_math_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasMath_t mode{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_math_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 31 || bits != handle->math_mode) return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kAtomicsMode) {
        if (!state.dispatch.blas.get_atomics_mode)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        cublasAtomicsMode_t mode{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_atomics_mode, handle->native, &mode));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            std::uint32_t bits{};
            static_assert(sizeof(mode) == sizeof(bits));
            std::memcpy(&bits, &mode, sizeof(bits));
            if (bits > 1 || bits != static_cast<std::uint32_t>(handle->atomics_mode))
                return invalid_observation();
            result.value = bits;
        }
    } else if (attribute == CudaBlasAttribute::kSmCountTarget) {
        if (!state.dispatch.blas.get_sm_count_target)
            return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        int count{};
        result = blas_completed(
            state.invoke(state.dispatch.blas.get_sm_count_target, handle->native, &count));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            if (count < 0 || count != handle->sm_count_target) return invalid_observation();
            result.value = static_cast<std::uint64_t>(count);
        }
    } else if (attribute == CudaBlasAttribute::kVersion) {
        if (!state.dispatch.blas.get_version) return blas_gate(rejected(CUDA_ERROR_NOT_SUPPORTED));
        int version{};
        result =
            blas_completed(state.invoke(state.dispatch.blas.get_version, handle->native, &version));
        if (result.original_status == CUBLAS_STATUS_SUCCESS) {
            if (version < 0) return invalid_observation();
            result.value = static_cast<std::uint64_t>(version);
        }
    } else
        return blas_gate(rejected(CUDA_ERROR_INVALID_VALUE));
    return result;
}
ExecutionAcceptance CudaExecutionSession::accept_sgemm(std::uint64_t sequence,
                                                       const CudaSessionSgemm& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem, nullptr);
}
ExecutionAcceptance CudaExecutionSession::accept_geam(std::uint64_t sequence,
                                                      const CudaSessionGeam& problem) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) {
        state.fault();
        return invalid;
    }
    const auto& sdk = state.dispatch.blas;
    if ((problem.type != CUDA_R_32F && problem.type != CUDA_R_64F) || problem.transpose_a > 3 ||
        problem.transpose_b > 3 || problem.pointer_mode > 1 || problem.presence > 31 ||
        !sdk.get_stream || !sdk.get_pointer_mode ||
        (problem.type == CUDA_R_32F ? !sdk.sgeam : !sdk.dgeam))
        return invalid;
    const auto width = problem.type == CUDA_R_32F ? sizeof(float) : sizeof(double);
    const auto is_zero = [](auto bytes) {
        return std::ranges::all_of(bytes, [](std::byte value) { return value == std::byte{}; });
    };
    const bool is_device = problem.pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
    const std::array<std::span<const std::byte>, 2> coefficients{problem.alpha, problem.beta};
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        const auto bytes = coefficients[index];
        const auto used = !is_device && (problem.presence & (1U << index)) ? width : 0;
        if (!is_zero(bytes.subspan(used))) return invalid;
        if (!is_device &&
            (problem.buffers[index + 3].allocation || problem.buffers[index + 3].offset))
            return invalid;
    }
    const auto handle =
        std::ranges::find(state.storage->blas, problem.handle, &Implementation::Blas::id);
    if (handle == state.storage->blas.end() || handle->is_uncertain ||
        handle->has_destroy_attempt || !state.workspace_is_valid(*handle) ||
        static_cast<std::uint32_t>(handle->pointer_mode) != problem.pointer_mode)
        return invalid;
    if (handle->stream &&
        std::ranges::find(state.storage->streams, handle->stream, &Implementation::Stream::id) ==
            state.storage->streams.end())
        return invalid;
    std::array<CUdeviceptr, 5> addresses{};
    std::array<std::size_t, 5> remaining{};
    constexpr std::array<unsigned int, 5> presence_bits{2, 3, 4, 0, 1};
    for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
        const auto range = problem.buffers[index];
        if ((!range.allocation && range.offset) ||
            (!(problem.presence & (1U << presence_bits[index])) &&
             (range.allocation || range.offset)))
            return invalid;
        if (!range.allocation) continue;
        const auto allocation = state.find_allocation(range.allocation);
        if (!allocation || range.offset > allocation->bytes ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - allocation->address)
            return invalid;
        addresses[index] = allocation->address + range.offset;
        remaining[index] = allocation->bytes - range.offset;
    }
    const bool is_a_transposed = problem.transpose_a != CUBLAS_OP_N;
    const bool is_b_transposed = problem.transpose_b != CUBLAS_OP_N;
    const bool has_valid_shape =
        problem.transpose_a <= CUBLAS_OP_C && problem.transpose_b <= CUBLAS_OP_C &&
        problem.rows >= 0 && problem.columns >= 0 &&
        problem.leading_a >= std::max(1, is_a_transposed ? problem.columns : problem.rows) &&
        problem.leading_b >= std::max(1, is_b_transposed ? problem.columns : problem.rows) &&
        problem.leading_c >= std::max(1, problem.rows);
    const bool is_empty = has_valid_shape && (!problem.rows || !problem.columns);
    const bool has_invalid_alias =
        addresses[2] && ((addresses[2] == addresses[0] &&
                          (is_a_transposed || problem.leading_a != problem.leading_c)) ||
                         (addresses[2] == addresses[1] &&
                          (is_b_transposed || problem.leading_b != problem.leading_c)));
    // These exact SDK paths reject or return before reading any scalar/matrix.
    // Unknown client addresses stay NULL; resolved exact aliases stay intact.
    const bool has_no_reads = !has_valid_shape || is_empty || has_invalid_alias ||
                              (is_device && (problem.presence & 3U) != 3U);
    bool requires_version = has_no_reads;
    bool stages_scalars{};
    std::array<bool, 2> safe_inputs{};
    if (!has_no_reads) {
        const std::array extents{
            matrix_bytes({is_a_transposed ? problem.columns : problem.rows,
                          is_a_transposed ? problem.rows : problem.columns, problem.leading_a},
                         width),
            matrix_bytes({is_b_transposed ? problem.columns : problem.rows,
                          is_b_transposed ? problem.rows : problem.columns, problem.leading_b},
                         width),
            matrix_bytes({problem.rows, problem.columns, problem.leading_c}, width)};
        const auto covers = [&](std::size_t index, std::size_t bytes) {
            return addresses[index] && addresses[index] % width == 0 && bytes <= remaining[index] &&
                   bytes <= std::numeric_limits<CUdeviceptr>::max() - addresses[index];
        };
        if (!extents[2] || !covers(2, *extents[2])) return invalid;
        for (std::size_t index = 0; index < safe_inputs.size(); ++index) {
            if (!extents[index] || !covers(index, *extents[index])) continue;
            const bool overlaps = addresses[index] < addresses[2] + *extents[2] &&
                                  addresses[2] < addresses[index] + *extents[index];
            // Invalid exact aliases took the native no-read path above. All
            // remaining equal addresses obey the documented in-place rule.
            safe_inputs[index] = !overlaps || addresses[index] == addresses[2];
        }
        if (is_device) {
            if (!covers(3, width) || !covers(4, width)) return invalid;
            stages_scalars = !safe_inputs[0] || !safe_inputs[1];
            requires_version = stages_scalars;
            if (stages_scalars &&
                (!state.dispatch.download || !state.dispatch.execution.stream_synchronize ||
                 !sdk.set_pointer_mode))
                return invalid;
        } else {
            for (std::size_t index = 0; index < safe_inputs.size(); ++index) {
                const auto& value = index == 0 ? problem.alpha : problem.beta;
                if (!safe_inputs[index] && !geam_scalar_is_zero(problem.type, value))
                    return invalid;
                if (!safe_inputs[index] || !(problem.presence & (1U << index)))
                    requires_version = true;
            }
        }
    }
    if (requires_version && !sdk.get_version) return invalid;
    try {
        auto call = std::make_shared<Implementation::GeamCall>(
            state, problem, addresses, safe_inputs, stages_scalars, requires_version);
        // Session cold gates retain every referenced allocation/module, handle,
        // stream and workspace until this record's real physical retirement.
        return state.executor->accept_library(
            sequence, {.stream = handle->stream, .parameters = {}, .call = std::move(call)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
}
template <typename Problem>
ExecutionAcceptance CudaExecutionSession::Implementation::accept_blas_extension(
    std::uint64_t sequence, const Problem& problem) noexcept {
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!owns() || !executor || is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (is_native_issue_active || is_folded_active || is_rnn_query_active) return invalid;
    if (is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(is_native_issue_active);
    if (check_context(true) != CUDA_SUCCESS) {
        fault();
        return invalid;
    }
    if (problem.pointer_mode > 1 || problem.presence > 31 || !dispatch.blas.get_stream ||
        !dispatch.blas.get_pointer_mode)
        return invalid;
    const auto handle = std::ranges::find(storage->blas, problem.handle, &Blas::id);
    if (handle == storage->blas.end() || handle->is_uncertain || handle->has_destroy_attempt ||
        !workspace_is_valid(*handle) ||
        static_cast<std::uint32_t>(handle->pointer_mode) != problem.pointer_mode ||
        (handle->stream && std::ranges::find(storage->streams, handle->stream, &Stream::id) ==
                               storage->streams.end()))
        return invalid;
    const bool is_device = problem.pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
    const auto is_zero = [](std::span<const std::byte> bytes) {
        return std::ranges::all_of(bytes, [](std::byte value) { return value == std::byte{}; });
    };
    for (std::size_t index = 0; index != 2; ++index) {
        const auto& scalar = index ? problem.beta : problem.alpha;
        const auto used = !is_device && (problem.presence & (1U << index)) ? sizeof(float) : 0;
        if (!is_zero(std::span{scalar}.subspan(used)) ||
            (!is_device &&
             (problem.buffers[index + 3].allocation || problem.buffers[index + 3].offset)))
            return invalid;
    }
    // Optional operands do not turn stale or foreign identities into NULL.
    // Resolve every supplied allocation before applying access exceptions.
    std::array<CUdeviceptr, 5> addresses{};
    std::array<std::size_t, 5> remaining{};
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
        const auto range = problem.buffers[index];
        if ((!range.allocation &&
             (range.offset || ((problem.presence & bits[index]) && (index < 3 || is_device)))) ||
            (!(problem.presence & bits[index]) && (range.allocation || range.offset)))
            return invalid;
        if (!range.allocation) continue;
        const auto allocation = find_allocation(range.allocation);
        if (!allocation || range.offset > allocation->bytes ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - allocation->address)
            return invalid;
        addresses[index] = allocation->address + range.offset;
        remaining[index] = allocation->bytes - range.offset;
    }
    // Candidate predicates require independent API/mode/pointer-pattern access
    // qualification, not GEAM's no-read rules or merely an unchanged output.
    bool metadata_only{}, zero_depth{};
    std::array<std::size_t, 5> alignments{4, 4, 4, 4, 4};
    std::array<std::optional<std::size_t>, 5> extents;
    extents.fill(0);
    if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
        if (!dispatch.blas.sgemv || problem.transpose > CUBLAS_OP_CONJG) return invalid;
        metadata_only = problem.rows <= 0 || problem.columns <= 0;
        if (!metadata_only) {
            // Actual SDK access controls refute a no-read permission for these
            // positive-dimension errors; they must retain strict geometry.
            if (problem.transpose > CUBLAS_OP_C || problem.leading_a < std::max(1, problem.rows) ||
                !problem.increment_x || !problem.increment_y)
                return invalid;
            extents[0] = matrix_bytes({problem.rows, problem.columns, problem.leading_a});
            const bool is_transposed = problem.transpose != CUBLAS_OP_N;
            extents[1] =
                vector_bytes(is_transposed ? problem.rows : problem.columns, problem.increment_x);
            extents[2] =
                vector_bytes(is_transposed ? problem.columns : problem.rows, problem.increment_y);
        }
    } else {
        if (!dispatch.blas.sgemm_ex || problem.transpose_a > CUBLAS_OP_CONJG ||
            problem.transpose_b > CUBLAS_OP_CONJG || problem.type_a > CUDA_R_8F_E5M2 ||
            problem.type_b > CUDA_R_8F_E5M2 || problem.type_c > CUDA_R_8F_E5M2)
            return invalid;
        const bool transposed_a = problem.transpose_a != CUBLAS_OP_N;
        const bool transposed_b = problem.transpose_b != CUBLAS_OP_N;
        metadata_only =
            problem.rows <= 0 || problem.columns <= 0 || problem.depth < 0 ||
            problem.transpose_a == CUBLAS_OP_CONJG || problem.transpose_b == CUBLAS_OP_CONJG ||
            problem.leading_a < std::max(1, transposed_a ? problem.depth : problem.rows) ||
            problem.leading_b < std::max(1, transposed_b ? problem.columns : problem.depth) ||
            problem.leading_c < std::max(1, problem.rows);
        zero_depth = !metadata_only && !problem.depth;
        if (!metadata_only) {
            const std::array types{static_cast<cudaDataType>(problem.type_a),
                                   static_cast<cudaDataType>(problem.type_b),
                                   static_cast<cudaDataType>(problem.type_c)};
            std::array<std::size_t, 3> widths{};
            for (std::size_t index = 0; index < types.size(); ++index) {
                widths[index] = gemm_element_bytes(types[index]);
                if (!widths[index]) return invalid;
                alignments[index] = widths[index] / (gemm_is_complex(types[index]) ? 2U : 1U);
            }
            if (!zero_depth) {
                extents[0] =
                    matrix_bytes({transposed_a ? problem.depth : problem.rows,
                                  transposed_a ? problem.rows : problem.depth, problem.leading_a},
                                 widths[0]);
                extents[1] = matrix_bytes(
                    {transposed_b ? problem.columns : problem.depth,
                     transposed_b ? problem.depth : problem.columns, problem.leading_b},
                    widths[1]);
            }
            extents[2] =
                matrix_bytes({problem.rows, problem.columns, problem.leading_c}, widths[2]);
        }
    }
    // Native access qualification covers grouped scalar/data presence only.
    // Mixed presence remains a policy gap, not an invented original SDK error.
    if (metadata_only && problem.presence != 0 && problem.presence != 3 && problem.presence != 28 &&
        problem.presence != 31)
        return invalid;
    const bool requires_version = metadata_only || zero_depth;
    if (requires_version && !dispatch.blas.get_version) return invalid;
    if (!metadata_only) {
        if (zero_depth && problem.presence != 19 && problem.presence != 31) return invalid;
        const auto required_presence = zero_depth ? 19U : 31U;
        if ((problem.presence & required_presence) != required_presence) return invalid;
        extents[3] = extents[4] = is_device ? sizeof(float) : 0;
    }
    for (std::size_t index = 0; index != extents.size(); ++index) {
        if (!extents[index]) return invalid;
        if (!*extents[index]) continue;
        if (!addresses[index] || *extents[index] > remaining[index] ||
            addresses[index] % alignments[index] ||
            *extents[index] > std::numeric_limits<CUdeviceptr>::max() - addresses[index])
            return invalid;
    }
    const auto overlaps = [](CUdeviceptr first, std::size_t first_bytes, CUdeviceptr second,
                             std::size_t second_bytes) {
        return first_bytes && second_bytes && first < second + second_bytes &&
               second < first + first_bytes;
    };
    for (std::size_t index = 0; index < addresses.size(); ++index)
        if (index != 2 && overlaps(addresses[2], *extents[2], addresses[index], *extents[index]))
            return invalid;
    if (handle->has_workspace && handle->workspace_bytes) {
        const auto allocation = find_allocation(handle->workspace.allocation);
        if (!allocation) return invalid;
        const auto address = allocation->address + handle->workspace.offset;
        if (handle->workspace_bytes > std::numeric_limits<CUdeviceptr>::max() - address)
            return invalid;
        for (std::size_t index = 0; index < addresses.size(); ++index)
            if (overlaps(address, handle->workspace_bytes, addresses[index], *extents[index]))
                return invalid;
    }
    try {
        // Fixed-size snapshots are bounded by max_commands. The dispatcher
        // captures logical dependencies and retains Storage through physical
        // retirement; ACK cannot open cold resource/mode/workspace mutation.
        return executor->accept_library(sequence,
                                        {.stream = handle->stream,
                                         .parameters = {},
                                         .call = std::make_shared<BlasExtensionCall<Problem>>(
                                             *this, problem, addresses, requires_version)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
ExecutionAcceptance CudaExecutionSession::accept_sgemv(std::uint64_t sequence,
                                                       const CudaSessionSgemv& problem) noexcept {
    return implementation_->accept_blas_extension(sequence, problem);
}
ExecutionAcceptance CudaExecutionSession::accept_sgemm_ex(
    std::uint64_t sequence, const CudaSessionSgemmEx& problem) noexcept {
    return implementation_->accept_blas_extension(sequence, problem);
}
ExecutionAcceptance CudaExecutionSession::accept_lt_matmul(
    std::uint64_t sequence, const CudaSessionLtMatmul& problem) noexcept {
    auto& state = *implementation_;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) return invalid;
    bool has_context_loss{};
    const auto proof = [&] {
        if (has_context_loss) return false;
        has_context_loss = state.check_context(true) != CUDA_SUCCESS;
        if (has_context_loss) state.fault();
        return !has_context_loss;
    };
    const auto& descriptors = problem.descriptors;
    auto* handle = state.lt_handle(descriptors.handle);
    const auto& sdk = state.dispatch.lt;
    if (!handle || handle->is_uncertain || handle->has_destroy_attempt || !sdk.matmul ||
        !sdk.version || !descriptors.operation.native || !descriptors.operation.lifetime ||
        descriptors.operation.kind != CudaLtObjectKind::kOperation)
        return invalid;
    if (descriptors.algorithm) {
        if (!proof()) return invalid;
        const auto version = state.invoke(sdk.version);
        if (!proof() || handle->sdk_version != version ||
            !std::ranges::any_of(handle->algorithms, [&](const auto& known) {
                return std::memcmp(&known, &*descriptors.algorithm, sizeof(known)) == 0;
            }))
            return invalid;
    }
    std::array<LtMatrixState, 4> matrices;
    std::array<CUdeviceptr, 7> addresses{};
    for (std::size_t index = 0; index < matrices.size(); ++index) {
        if (!lt_matrix_state(sdk, descriptors.matrices[index], index == 3, matrices[index], proof))
            return invalid;
        const auto memory = device_memory(problem.matrices[index], matrices[index].bytes);
        if (!memory) return invalid;
        addresses[index] = memory->address;
    }
    cudaDataType scale{};
    cublasLtPointerMode_t pointer_mode{};
    cublasLtEpilogue_t epilogue{};
    if (!lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_SCALE_TYPE, scale, proof) ||
        !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode,
                     proof) ||
        !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue, proof))
        return invalid;
    const auto width = gemm_element_bytes(scale);
    if (!width || width > problem.alpha.size() || problem.scalar_bytes != width ||
        (pointer_mode != CUBLASLT_POINTER_MODE_HOST &&
         pointer_mode != CUBLASLT_POINTER_MODE_DEVICE) ||
        problem.uses_device_scalars != (pointer_mode == CUBLASLT_POINTER_MODE_DEVICE))
        return invalid;
    if (problem.uses_device_scalars) {
        for (std::size_t index = 0; index < 2; ++index) {
            const auto memory = device_memory(problem.device_scalars[index], width);
            if (!memory) return invalid;
            addresses[4 + index] = memory->address;
        }
    }
    if (problem.workspace_bytes) {
        const auto memory = device_memory(problem.workspace, problem.workspace_bytes);
        if (!memory) return invalid;
        addresses[6] = memory->address;
    } else if (problem.workspace.allocation) {
        const auto memory = device_memory(problem.workspace, 0);
        if (!memory) return invalid;
        addresses[6] = memory->address;
    }
    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT && epilogue != CUBLASLT_EPILOGUE_BIAS &&
        epilogue != CUBLASLT_EPILOGUE_RELU && epilogue != CUBLASLT_EPILOGUE_RELU_BIAS &&
        epilogue != CUBLASLT_EPILOGUE_GELU && epilogue != CUBLASLT_EPILOGUE_GELU_BIAS)
        return invalid;
    CUdeviceptr bias_address{};
    cudaDataType bias_type = matrices[3].type;
    std::int64_t bias_stride{};
    if ((static_cast<unsigned int>(epilogue) & CUBLASLT_EPILOGUE_BIAS) != 0) {
        const auto& bias = descriptors.operation.bias;
        std::int32_t type{};
        std::int64_t stride{};
        CUdeviceptr pointer{};
        if (!bias || bias->session != state.identity.client_id ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, type,
                         proof) ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, stride,
                         proof) ||
            !lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, pointer,
                         proof) ||
            pointer != bias->address || stride < 0)
            return invalid;
        bias_type = type == -1 ? matrices[3].type : static_cast<cudaDataType>(type);
        if (type == -1 && matrices[3].type == CUDA_R_8I && scale == CUDA_R_32F)
            bias_type = CUDA_R_32F;
        const auto bias_width = gemm_element_bytes(bias_type);
        const auto count = static_cast<std::uint64_t>(matrices[3].batches - 1);
        if (!bias_width || matrices[3].rows > SIZE_MAX / bias_width ||
            (count && static_cast<std::uint64_t>(stride) >
                          (SIZE_MAX / bias_width - matrices[3].rows) / count))
            return invalid;
        const auto bytes =
            (matrices[3].rows + count * static_cast<std::uint64_t>(stride)) * bias_width;
        const auto memory =
            device_memory({bias->allocation, static_cast<std::size_t>(bias->offset)}, bytes);
        if (!memory || memory->address != pointer) return invalid;
        bias_address = pointer;
        bias_stride = stride;
    }
    try {
        cublasComputeType_t compute{};
        bool has_direct_reduction = !descriptors.algorithm;
        bool has_split_reduction{};
        std::uint32_t split{}, reduction{};
        if (descriptors.algorithm && sdk.configuration_get) {
            std::size_t split_bytes{}, reduction_bytes{};
            const bool has_configuration =
                state.invoke(sdk.configuration_get, &*descriptors.algorithm,
                             CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split),
                             &split_bytes) == CUBLAS_STATUS_SUCCESS &&
                proof() &&
                state.invoke(sdk.configuration_get, &*descriptors.algorithm,
                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction),
                             &reduction_bytes) == CUBLAS_STATUS_SUCCESS &&
                proof() && split_bytes == sizeof(split) && reduction_bytes == sizeof(reduction);
            has_direct_reduction = has_configuration && split <= 1 && !reduction;
            has_split_reduction = has_configuration && (split == 3 || split == 6) &&
                                  reduction == CUBLASLT_REDUCTION_SCHEME_INPLACE;
        }
        if ((has_direct_reduction || has_split_reduction) && descriptors.algorithm) {
            cublasLtMatmulHeuristicResult_t checked{};
            checked.algo = *descriptors.algorithm;
            const auto layout = [&](std::size_t index) {
                return static_cast<cublasLtMatrixLayout_t>(descriptors.matrices[index].native);
            };
            const bool has_valid_algorithm =
                sdk.algorithm_check &&
                state.invoke(sdk.algorithm_check, handle->native,
                             static_cast<cublasLtMatmulDesc_t>(descriptors.operation.native),
                             layout(0), layout(1), descriptors.has_c_layout ? layout(2) : nullptr,
                             layout(3), &*descriptors.algorithm,
                             &checked) == CUBLAS_STATUS_SUCCESS &&
                proof() && checked.state == CUBLAS_STATUS_SUCCESS &&
                checked.workspaceSize <= problem.workspace_bytes;
            has_direct_reduction &= has_valid_algorithm;
            has_split_reduction &= has_valid_algorithm;
        }
        std::array<cublasOperation_t, 3> transposes{};
        const bool has_operations =
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute,
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, transposes[0],
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, transposes[1],
                        proof) &&
            lt_property(sdk, descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSC, transposes[2],
                        proof);
        if (!proof()) return invalid;
        const bool is_float = matrices[0].type == CUDA_R_32F && matrices[1].type == CUDA_R_32F &&
                              matrices[3].type == CUDA_R_32F;
        const bool is_half = matrices[0].type == CUDA_R_16F && matrices[1].type == CUDA_R_16F &&
                             (matrices[3].type == CUDA_R_16F || matrices[3].type == CUDA_R_32F);
        std::uint32_t reduction_depth{};
        float host_alpha{}, host_beta{};
        std::memcpy(&host_alpha, problem.alpha.data(), sizeof(host_alpha));
        std::memcpy(&host_beta, problem.beta.data(), sizeof(host_beta));
        // This arithmetic is pinned to these exact BERT forward
        // profiles, not a promise about arbitrary split-K SDK algorithms.
        const bool has_pinned_split_profile =
            has_split_reduction && handle->sdk_version == 120604 && is_float && has_operations &&
            !problem.uses_device_scalars && std::isfinite(host_alpha) && host_alpha != 0.0F &&
            host_beta == 0.0F && epilogue == CUBLASLT_EPILOGUE_BIAS && bias_type == CUDA_R_32F &&
            bias_address && bias_stride == 0 && transposes[0] == CUBLAS_OP_T &&
            transposes[1] == CUBLAS_OP_N && transposes[2] == CUBLAS_OP_N &&
            matrices[3].rows == 768 && matrices[3].columns == 768 &&
            ((matrices[0].rows == 768 && split == 6) || (matrices[0].rows == 3072 && split == 3)) &&
            std::ranges::all_of(matrices, [](const auto& matrix) {
                return matrix.order == CUBLASLT_ORDER_COL && matrix.batches == 1 &&
                       matrix.leading == static_cast<std::int64_t>(matrix.rows);
            });
        if (has_pinned_split_profile)
            reduction_depth = static_cast<std::uint32_t>(matrices[0].rows / split);
        const auto disjoint = [](CUdeviceptr left, std::size_t left_bytes, CUdeviceptr right,
                                 std::size_t right_bytes) {
            return left <= right ? right - left >= left_bytes : left - right >= right_bytes;
        };
        const bool safe_output =
            disjoint(addresses[3], matrices[3].bytes, addresses[0], matrices[0].bytes) &&
            disjoint(addresses[3], matrices[3].bytes, addresses[1], matrices[1].bytes) &&
            (disjoint(addresses[3], matrices[3].bytes, addresses[2], matrices[2].bytes) ||
             (addresses[2] == addresses[3] && matrices[2].order == matrices[3].order &&
              matrices[2].leading == matrices[3].leading &&
              matrices[2].stride == matrices[3].stride && transposes[2] == CUBLAS_OP_N));
        bool has_safe_alignment = !addresses[6] || addresses[6] % 256 == 0;
        for (std::size_t operand = 0; operand < 4; ++operand)
            has_safe_alignment =
                has_safe_alignment &&
                addresses[operand] % gemm_element_bytes(matrices[operand].type) == 0;
        if (bias_address)
            has_safe_alignment =
                has_safe_alignment && bias_address % gemm_element_bytes(bias_type) == 0;
        if (problem.uses_device_scalars)
            has_safe_alignment =
                has_safe_alignment && addresses[4] % width == 0 && addresses[5] % width == 0;
        if (state.persistent.replace_sgemm && state.storage->lt_ids.back() && has_operations &&
            (has_direct_reduction || has_pinned_split_profile) && (is_float || is_half) &&
            matrices[2].type == matrices[3].type && matrices[0].leading > 0 &&
            matrices[1].leading > 0 &&
            (epilogue != CUBLASLT_EPILOGUE_DEFAULT || !descriptors.operation.bias) &&
            (epilogue == CUBLASLT_EPILOGUE_DEFAULT || matrices[3].order == CUBLASLT_ORDER_COL) &&
            has_safe_alignment && compute == CUBLAS_COMPUTE_32F && scale == CUDA_R_32F &&
            safe_output &&
            (epilogue == CUBLASLT_EPILOGUE_DEFAULT || epilogue == CUBLASLT_EPILOGUE_BIAS) &&
            (bias_type == matrices[3].type || bias_type == CUDA_R_32F) &&
            std::ranges::all_of(transposes,
                                [](auto value) {
                                    return value == CUBLAS_OP_N || value == CUBLAS_OP_T ||
                                           value == CUBLAS_OP_C;
                                }) &&
            std::ranges::all_of(matrices, [&](const auto& matrix) {
                return matrix.leading <= INT_MAX && matrix.rows <= INT_MAX - 64 &&
                       matrix.columns <= INT_MAX - 64 &&
                       (matrix.batches == 1 || matrix.batches == matrices[3].batches);
            })) {
            const auto rows = transposes[0] == CUBLAS_OP_N ? matrices[0].rows : matrices[0].columns;
            const auto depth =
                transposes[0] == CUBLAS_OP_N ? matrices[0].columns : matrices[0].rows;
            const auto columns =
                transposes[1] == CUBLAS_OP_N ? matrices[1].columns : matrices[1].rows;
            const auto b_depth =
                transposes[1] == CUBLAS_OP_N ? matrices[1].rows : matrices[1].columns;
            const auto c_rows =
                transposes[2] == CUBLAS_OP_N ? matrices[2].rows : matrices[2].columns;
            const auto c_columns =
                transposes[2] == CUBLAS_OP_N ? matrices[2].columns : matrices[2].rows;
            const auto tiles =
                cuda_gemm_matrix_tiles(rows, columns, state.persistent.gemm_tile_side);
            const auto batches = static_cast<std::uint64_t>(matrices[3].batches);
            if (rows != matrices[3].rows || columns != matrices[3].columns || b_depth != depth ||
                c_rows != rows || c_columns != columns || !tiles || tiles > UINT32_MAX / batches)
                return invalid;
            const auto total = tiles * batches;
            const auto free_slot = std::ranges::find(state.storage->occupied, false);
            if (free_slot == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            const auto variant = is_float ? 0U : matrices[3].type == CUDA_R_16F ? 1U : 2U;
            int capacity{};
            if (state.invoke(state.dispatch.occupancy, &capacity,
                             state.storage->lt_functions[variant], 128, 0) != CUDA_SUCCESS ||
                capacity <= 0)
                return invalid;
            if (!proof()) return invalid;
            const auto index =
                static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
            auto progress = std::make_shared<Implementation::Progress>(
                state.storage, state.dispatch,
                Implementation::ProgressParameters{
                    .slot = index,
                    .sequence = sequence,
                    .total = total,
                    .workers = state.persistent.workers,
                    .observes_block_completion = state.persistent.observes_block_completion,
                    .uses_sharded_host_retreat = state.persistent.uses_sharded_host_retreat},
                std::make_shared<CudaSessionLtMatmul>(problem));
            *free_slot = true;
            std::array<std::byte, 176> parameters{};
            const auto put = [&](std::size_t offset, const auto& value) {
                std::memcpy(parameters.data() + offset, &value, sizeof(value));
            };
            for (std::size_t operand = 0; operand < 4; ++operand)
                put(operand * 8, addresses[operand]);
            put(32, bias_address);
            const std::array dimensions{static_cast<int>(rows),
                                        static_cast<int>(columns),
                                        static_cast<int>(depth),
                                        static_cast<int>(matrices[0].leading),
                                        static_cast<int>(matrices[1].leading),
                                        static_cast<int>(matrices[2].leading),
                                        static_cast<int>(matrices[3].leading)};
            for (std::size_t field = 0; field < dimensions.size(); ++field)
                put(40 + field * 4, dimensions[field]);
            std::uint32_t flags{};
            for (std::size_t operand = 0; operand < 3; ++operand)
                if ((transposes[operand] != CUBLAS_OP_N) !=
                    (matrices[operand].order == CUBLASLT_ORDER_ROW))
                    flags |= 1U << operand;
            if (matrices[3].order == CUBLASLT_ORDER_ROW) flags |= 8U;
            put(68, flags);
            std::memcpy(parameters.data() + 72, problem.alpha.data(), 4);
            std::memcpy(parameters.data() + 76, problem.beta.data(), 4);
            put(80, addresses[4]);
            put(88, addresses[5]);
            put(96, std::uint32_t{problem.uses_device_scalars ? 1U : 0U});
            put(100, std::uint32_t{bias_type == CUDA_R_32F ? 1U : 0U});
            put(104, progress->counter());
            put(112, progress->retreat());
            for (std::size_t operand = 0; operand < 4; ++operand)
                put(120 + operand * 8,
                    matrices[operand].batches == 1 ? std::int64_t{0} : matrices[operand].stride);
            put(152, bias_stride);
            put(160, matrices[3].batches);
            put(164, reduction_depth);
            if (state.persistent.observes_block_completion) put(168, progress->block_completion());
            std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
            auto accepted = state.executor->accept_persistent_kernel(
                sequence,
                {.function = state.storage->lt_ids[variant],
                 .stream = problem.stream,
                 .grid = {static_cast<unsigned int>(total), 1, 1},
                 .block = {128, 1, 1},
                 .parameters = {}},
                {state.storage->lt_functions[variant], state.persistent.workers,
                 std::span{parameters}.first(state.persistent.observes_block_completion ? 176
                                                                                        : 168),
                 progress});
            if (accepted.accepted()) {
                state.progress.push_back(progress);
                if (state.persistent_accepted != UINT64_MAX) ++state.persistent_accepted;
            }
            return accepted;
        }
        if (!proof()) return invalid;
        return state.executor->accept_library(
            sequence, {.stream = problem.stream,
                       .parameters = {},
                       .call = std::make_shared<LtMatmulCall>(sdk.matmul, handle->native, problem,
                                                              addresses)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
}
ExecutionAcceptance CudaExecutionSession::accept_gemm_ex(
    std::uint64_t sequence, const CudaSessionGemmEx& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem.operation, &problem);
}
ExecutionAcceptance CudaExecutionSession::accept_strided_gemm_ex(
    std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept {
    return implementation_->accept_gemm(sequence, problem.operation.operation, &problem.operation,
                                        &problem);
}
ExecutionAcceptance CudaExecutionSession::accept_strided_sgemm(
    std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept {
    const auto& typed = problem.operation;
    if (typed.a_type != CUDA_R_32F || typed.b_type != CUDA_R_32F || typed.c_type != CUDA_R_32F ||
        typed.compute_type != CUBLAS_COMPUTE_32F || typed.algorithm != CUBLAS_GEMM_DEFAULT)
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    return implementation_->accept_gemm(sequence, typed.operation, &typed, &problem, true);
}
ExecutionAcceptance CudaExecutionSession::Implementation::accept_gemm(
    std::uint64_t sequence, const CudaSessionSgemm& problem, const CudaSessionGemmEx* extended,
    const CudaSessionStridedGemmEx* batch, bool uses_sgemm_batch) noexcept {
    auto& state = *this;
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return invalid;
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const NativeExecutionScope active(state.is_native_issue_active);
    if (state.check_context(true) != CUDA_SUCCESS) return invalid;
    const auto proof = [&] {
        if (state.check_context(true) == CUDA_SUCCESS) return true;
        state.fault();
        return false;
    };
    const auto handle =
        std::ranges::find(state.storage->blas, problem.handle, &Implementation::Blas::id);
    if (handle == state.storage->blas.end() || handle->is_uncertain ||
        !state.workspace_is_valid(*handle) ||
        (batch      ? (uses_sgemm_batch ? !state.dispatch.blas.strided_sgemm
                                        : !state.dispatch.blas.strided_gemm_ex)
         : extended ? !state.dispatch.blas.gemm_ex
                    : !state.dispatch.blas.sgemm) ||
        (batch && (batch->batches <= 0 ||
                   (batch->batches > 1 &&
                    std::ranges::any_of(batch->strides, [](auto stride) { return stride < 0; })))))
        return invalid;
    const std::array types{extended ? extended->a_type : CUDA_R_32F,
                           extended ? extended->b_type : CUDA_R_32F,
                           extended ? extended->c_type : CUDA_R_32F};
    const std::array widths{gemm_element_bytes(types[0]), gemm_element_bytes(types[1]),
                            gemm_element_bytes(types[2])};
    const auto scalar_bytes =
        extended ? gemm_scalar_bytes(extended->compute_type, extended->c_type) : sizeof(float);
    if (!scalar_bytes || std::ranges::any_of(widths, [](auto width) { return width == 0; }) ||
        (extended && (extended->algorithm < CUBLAS_GEMM_DEFAULT || extended->algorithm > 127)))
        return invalid;
    const auto valid_transpose = [](auto value) {
        return value == CUBLAS_OP_N || value == CUBLAS_OP_T || value == CUBLAS_OP_C;
    };
    if (!valid_transpose(problem.transpose_a) || !valid_transpose(problem.transpose_b))
        return invalid;
    const bool is_a_transposed = problem.transpose_a != CUBLAS_OP_N;
    const bool is_b_transposed = problem.transpose_b != CUBLAS_OP_N;
    const std::array extents{
        matrix_bytes({is_a_transposed ? problem.depth : problem.rows,
                      is_a_transposed ? problem.rows : problem.depth, problem.leading_a},
                     widths[0]),
        matrix_bytes({is_b_transposed ? problem.columns : problem.depth,
                      is_b_transposed ? problem.depth : problem.columns, problem.leading_b},
                     widths[1]),
        matrix_bytes({problem.rows, problem.columns, problem.leading_c}, widths[2])};
    const std::array ranges{problem.a, problem.b, problem.c};
    if (problem.uses_device_scalars != (handle->pointer_mode == CUBLAS_POINTER_MODE_DEVICE))
        return invalid;
    std::array<CUdeviceptr, 5> addresses{};
    bool is_disjoint_batch = true;
    bool is_signed_indexable = true;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        if (!extents[index]) return invalid;
        const auto found = state.find_allocation(ranges[index].allocation);
        if (!found || ranges[index].offset > found->bytes ||
            *extents[index] > found->bytes - ranges[index].offset ||
            ranges[index].offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return invalid;
        addresses[index] = found->address + ranges[index].offset;
        const auto alignment = widths[index] / (gemm_is_complex(types[index]) ? 2U : 1U);
        if (addresses[index] % alignment) return invalid;
        if (batch && batch->batches > 1) {
            const auto stride = batch->strides[index];
            // This unsigned magnitude also handles INT64_MIN without signed overflow.
            const auto magnitude = stride < 0 ? static_cast<std::uint64_t>(-(stride + 1)) + 1
                                              : static_cast<std::uint64_t>(stride);
            const auto intervals = static_cast<std::uint64_t>(batch->batches - 1);
            const auto maximum = std::numeric_limits<std::uint64_t>::max();
            if (magnitude > maximum / widths[index]) return invalid;
            const auto stride_bytes = magnitude * widths[index];
            if (stride_bytes > maximum / intervals) return invalid;
            const auto span = stride_bytes * intervals;
            if (stride < 0 ? span > ranges[index].offset
                           : span > found->bytes - ranges[index].offset - *extents[index])
                return invalid;
            is_signed_indexable =
                is_signed_indexable &&
                magnitude <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                                 intervals;
            if (index == 2) is_disjoint_batch = stride_bytes >= *extents[index];
        }
    }
    if (problem.uses_device_scalars) {
        const std::array scalars{problem.device_alpha, problem.device_beta};
        for (std::size_t index = 0; index < scalars.size(); ++index) {
            const auto found = state.find_allocation(scalars[index].allocation);
            if (!found || scalars[index].offset > found->bytes ||
                scalar_bytes > found->bytes - scalars[index].offset ||
                scalars[index].offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
                return invalid;
            addresses[index + 3] = found->address + scalars[index].offset;
            const auto alignment = scalar_bytes / (gemm_is_complex(types[2]) ? 2U : 1U);
            if (addresses[index + 3] % alignment) return invalid;
        }
    } else if (problem.device_alpha.allocation || problem.device_alpha.offset ||
               problem.device_beta.allocation || problem.device_beta.offset) {
        return invalid;
    }
    try {
        // Selection precedes queue admission. No original library call is made
        // for this path; the returned host result is the actual Driver launch.
        const auto matrix_tiles = cuda_gemm_matrix_tiles(
            static_cast<std::uint64_t>(problem.rows), static_cast<std::uint64_t>(problem.columns),
            state.persistent.gemm_tile_side);
        const auto batches = batch ? static_cast<std::uint64_t>(batch->batches) : 1;
        if (matrix_tiles > std::numeric_limits<std::uint64_t>::max() / batches) return invalid;
        const auto total = matrix_tiles * batches;
        const bool is_default_ex =
            !extended || (extended->compute_type == CUBLAS_COMPUTE_32F &&
                          (extended->algorithm == CUBLAS_GEMM_DEFAULT ||
                           extended->algorithm == CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        const bool is_float =
            types[0] == CUDA_R_32F && types[1] == CUDA_R_32F && types[2] == CUDA_R_32F;
        const bool is_half = types[0] == CUDA_R_16F && types[1] == CUDA_R_16F &&
                             (types[2] == CUDA_R_16F || types[2] == CUDA_R_32F);
        bool is_indexable = true;
        for (std::size_t index = 0; index < extents.size(); ++index)
            is_indexable =
                is_indexable && *extents[index] / widths[index] <=
                                    static_cast<std::size_t>(std::numeric_limits<int>::max());
        bool is_replaced =
            is_default_ex && (is_float || is_half) && is_disjoint_batch && is_signed_indexable &&
            // Zero input strides reuse read-only matrices. Disjoint output
            // hulls above still exclude overlapping C from private execution.
            (!batch || batch->batches == 1 ||
             std::ranges::all_of(batch->strides, [](auto stride) { return stride >= 0; })) &&
            state.persistent.replace_sgemm && handle->math_mode == 0 &&
            handle->sm_count_target == 0 &&
            (!handle->has_workspace || handle->workspace_bytes >= std::size_t{16} * 1024) &&
            state.storage->gemm_ids.back() &&
            problem.rows <= std::numeric_limits<int>::max() - 64 &&
            problem.columns <= std::numeric_limits<int>::max() - 64 &&
            problem.depth <= std::numeric_limits<int>::max() - 8 &&
            total <= std::numeric_limits<std::uint32_t>::max() && is_indexable;
        float alpha = problem.alpha, beta = problem.beta;
        if (extended && is_float && is_default_ex) {
            std::memcpy(&alpha, extended->alpha.data(), sizeof(alpha));
            std::memcpy(&beta, extended->beta.data(), sizeof(beta));
        }
        const bool is_input_gradient =
            !batch && is_float && is_default_ex && !problem.uses_device_scalars &&
            problem.transpose_a == CUBLAS_OP_N && problem.transpose_b == CUBLAS_OP_N &&
            problem.rows == 768 && problem.columns == 768 && problem.depth == 3072 &&
            problem.leading_a == 768 && problem.leading_b == 3072 && problem.leading_c == 768 &&
            alpha == 1.0F && beta == 0.0F;
        bool uses_split_reduction{};
        if (is_replaced && is_input_gradient) {
            // This arithmetic is qualified for the pinned SDK, not for
            // every library algorithm. A private query failure keeps the
            // original route and never becomes an original compute result.
            if (state.dispatch.blas.get_version) {
                if (!proof()) return invalid;
                int version{};
                const auto status =
                    state.invoke(state.dispatch.blas.get_version, handle->native, &version);
                if (!proof()) return invalid;
                uses_split_reduction = status == CUBLAS_STATUS_SUCCESS && version == 120604;
            }
            is_replaced = uses_split_reduction;
        }
        if (is_replaced) {
            const auto free_slot = std::ranges::find(state.storage->occupied, false);
            if (free_slot == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            const auto variant = uses_split_reduction
                                     ? 24U
                                     : (is_a_transposed ? 2U : 0U) + (is_b_transposed ? 1U : 0U) +
                                           (is_half ? (types[2] == CUDA_R_16F ? 4U : 8U) : 0U) +
                                           (batch ? 12U : 0U);
            if (!proof()) return invalid;
            int capacity{};
            const auto occupancy = state.invoke(state.dispatch.occupancy, &capacity,
                                                state.storage->gemm_functions[variant], 128, 0);
            if (!proof() || occupancy != CUDA_SUCCESS || capacity <= 0) return invalid;
            const auto index =
                static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
            auto progress = std::make_shared<Implementation::Progress>(
                state.storage, state.dispatch,
                Implementation::ProgressParameters{
                    .slot = index,
                    .sequence = sequence,
                    .total = total,
                    .workers = state.persistent.workers,
                    .observes_block_completion = state.persistent.observes_block_completion,
                    .uses_sharded_host_retreat = state.persistent.uses_sharded_host_retreat});
            *free_slot = true;
            std::array<std::byte, 136> parameters{};
            const auto put = [&parameters](std::size_t offset, const auto& value) {
                std::memcpy(parameters.data() + offset, &value, sizeof(value));
            };
            for (std::size_t operand = 0; operand < 3; ++operand)
                put(operand * 8, addresses[operand]);
            const std::array dimensions{problem.rows,      problem.columns,   problem.depth,
                                        problem.leading_a, problem.leading_b, problem.leading_c};
            for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
                put(24 + axis * 4, dimensions[axis]);
            put(48, problem.alpha);
            put(52, problem.beta);
            if (extended) {
                std::memcpy(parameters.data() + 48, extended->alpha.data(), sizeof(float));
                std::memcpy(parameters.data() + 52, extended->beta.data(), sizeof(float));
            }
            put(56, addresses[3]);
            put(64, addresses[4]);
            const std::uint32_t device_scalars = problem.uses_device_scalars ? 1U : 0U;
            put(72, device_scalars);
            put(80, progress->counter());
            put(88, progress->retreat());
            if (batch) {
                for (std::size_t operand = 0; operand < batch->strides.size(); ++operand)
                    put(96 + operand * 8, batch->strides[operand]);
                put(120, batch->batches);
            } else if (uses_split_reduction) {
                put(96, std::uint32_t{768});
            }
            const std::size_t ordinary_bytes = batch ? 128 : uses_split_reduction ? 104 : 96;
            if (state.persistent.observes_block_completion)
                put(ordinary_bytes, progress->block_completion());
            std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
            auto accepted = state.executor->accept_persistent_kernel(
                sequence,
                {.function = state.storage->gemm_ids[variant],
                 .stream = handle->stream,
                 .grid = {static_cast<unsigned int>(total), 1, 1},
                 .block = {128, 1, 1},
                 .parameters = {}},
                {state.storage->gemm_functions[variant], state.persistent.workers,
                 std::span{parameters}.first(ordinary_bytes +
                                             (state.persistent.observes_block_completion ? 8 : 0)),
                 progress});
            if (accepted.accepted()) {
                state.progress.push_back(progress);
                if (state.persistent_accepted != std::numeric_limits<std::uint64_t>::max())
                    ++state.persistent_accepted;
            }
            return accepted;
        }
        if (batch && uses_sgemm_batch)
            return state.executor->accept_library(
                sequence,
                {.stream = handle->stream,
                 .parameters = {},
                 .call = std::make_shared<StridedSgemmCall>(state.dispatch.blas.strided_sgemm,
                                                            handle->native, *batch, addresses)});
        if (batch)
            return state.executor->accept_library(
                sequence,
                {.stream = handle->stream,
                 .parameters = {},
                 .call = std::make_shared<StridedGemmExCall>(state.dispatch.blas.strided_gemm_ex,
                                                             handle->native, *batch, addresses)});
        if (extended)
            return state.executor->accept_library(
                sequence, {.stream = handle->stream,
                           .parameters = {},
                           .call = std::make_shared<GemmExCall>(
                               state.dispatch.blas.gemm_ex, handle->native, *extended, addresses)});
        return state.executor->accept_library(
            sequence, {.stream = handle->stream,
                       .parameters = {},
                       .call = std::make_shared<SgemmCall>(state.dispatch.blas.sgemm,
                                                           handle->native, problem, addresses)});
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

// Logical bytes, mapped extent and native flags are separately validated.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
CudaSessionResult CudaExecutionSession::import_host_memory(int descriptor, std::size_t bytes,
                                                           std::size_t mapping_bytes,
                                                           unsigned int flags,
                                                           bool is_pageable) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (!is_pageable && (!state.dispatch.host_register || !state.dispatch.host_unregister ||
                         !state.dispatch.host_map))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (is_pageable && flags) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (flags & ~static_cast<unsigned int>(CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (flags & CU_MEMHOSTALLOC_WRITECOMBINED) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto page = ::sysconf(_SC_PAGESIZE);
    struct stat info {};
    const auto seals = descriptor < 0 ? -1 : ::fcntl(descriptor, F_GET_SEALS);
    constexpr int required_seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if (descriptor < 0 || page <= 0 || !bytes || mapping_bytes < bytes ||
        mapping_bytes % static_cast<std::size_t>(page) ||
        mapping_bytes - bytes >= static_cast<std::size_t>(page) ||
        ::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<std::uintmax_t>(info.st_size) != mapping_bytes || seals < 0 ||
        (seals & required_seals) != required_seals ||
        (::fcntl(descriptor, F_GETFL) & O_ACCMODE) != O_RDWR)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->host_allocations.size() == state.limits.max_objects ||
        mapping_bytes > state.limits.max_host_memory_bytes - state.storage->host_memory_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto mapping = std::make_shared<Implementation::HostMapping>();
        mapping->bytes = mapping_bytes;
        mapping->pointer =
            ::mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
        if (mapping->pointer == MAP_FAILED)
            return rejected(errno == ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE);
        // Capacity is reserved before any native resource exists. Unknown native
        // state retains this charged mapping, even when no public ID is returned.
        state.storage->host_allocations.push_back({0, bytes, flags, 0, std::move(mapping)});
        state.storage->host_memory_bytes += mapping_bytes;
        auto& allocation = state.storage->host_allocations.back();
        allocation.is_pageable = is_pageable;
        if (is_pageable) {
            // This mapping has never been registered. Preserve its host memory
            // class for the exact generic SDK call; physical borrowers still
            // retain the same charged mapping until independent retirement.
            allocation.id = state.consume_id();
            auto result = settled();
            result.id = allocation.id;
            return result;
        }
        const auto discard = [&](CudaSessionResult result) {
            if (::munmap(allocation.mapping->pointer, mapping_bytes) != 0) {
                allocation.mapping->can_unmap = false;
                allocation.is_uncertain = true;
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            allocation.mapping->pointer = MAP_FAILED;
            state.storage->host_memory_bytes -= mapping_bytes;
            state.storage->host_allocations.pop_back();
            return result;
        };
        CUdeviceptr before{};
        const auto observed =
            state.invoke(state.dispatch.host_map, &before, allocation.mapping->pointer, 0);
        if (state.is_faulted) {
            allocation.mapping->can_unmap = false;
            allocation.is_uncertain = true;
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        if (observed != CUDA_ERROR_INVALID_VALUE || before) {
            allocation.mapping->can_unmap = false;
            allocation.is_uncertain = true;
            state.fault();
            return rejected(observed == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed);
        }
        allocation.mapping->can_unmap = false;
        auto result = completed(
            state.invoke(state.dispatch.host_register, allocation.mapping->pointer, mapping_bytes,
                         CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
        const auto retain_fault = [&] {
            allocation.is_uncertain = true;
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        };
        if (state.is_faulted) return retain_fault();
        if (result.status != CUDA_SUCCESS) {
            CUdeviceptr after{};
            const auto empty =
                state.invoke(state.dispatch.host_map, &after, allocation.mapping->pointer, 0);
            if (state.is_faulted) return retain_fault();
            const bool is_known_rejection =
                result.status == CUDA_ERROR_INVALID_VALUE ||
                result.status == CUDA_ERROR_OUT_OF_MEMORY ||
                result.status == CUDA_ERROR_NOT_SUPPORTED ||
                result.status == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
            if (is_known_rejection && empty == CUDA_ERROR_INVALID_VALUE && !after)
                return discard(result);
            allocation.is_uncertain = true;
            state.fault();
            return result;
        }
        const auto mapped = state.invoke(state.dispatch.host_map, &allocation.address,
                                         allocation.mapping->pointer, 0);
        if (state.is_faulted) return retain_fault();
        if (mapped != CUDA_SUCCESS || !allocation.address ||
            allocation.address % static_cast<std::size_t>(page) ||
            mapping_bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - allocation.address) {
            const auto failure = mapped == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : mapped;
            // No application work or public capability exists. One checked
            // unregister can settle this known successful private registration.
            allocation.has_unregister_attempt = true;
            const auto cleaned =
                state.invoke(state.dispatch.host_unregister, allocation.mapping->pointer);
            if (state.is_faulted) return retain_fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = failure;
            if (cleaned == CUDA_SUCCESS) return discard(result);
            allocation.is_uncertain = true;
            state.fault();
            return result;
        }
        allocation.id = state.consume_id();
        result.id = allocation.id;
        result.address = allocation.address;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionResult CudaExecutionSession::release_host_memory(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto status = state.check_context();
    if (status != CUDA_SUCCESS) return rejected(status);
    if (state.is_faulted || !id) return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto found =
        std::ranges::find(state.storage->host_allocations, id, &Implementation::HostAllocation::id);
    if (found == state.storage->host_allocations.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    return state.release_host_at(
        static_cast<std::size_t>(found - state.storage->host_allocations.begin()));
}
std::optional<CudaHostMemoryView> CudaExecutionSession::host_memory(CudaAllocationRange range,
                                                                    std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted || state.is_closing || !range.allocation)
        return std::nullopt;
    const auto found = std::ranges::find(state.storage->host_allocations, range.allocation,
                                         &Implementation::HostAllocation::id);
    if (found == state.storage->host_allocations.end() || found->is_retiring ||
        found->is_uncertain || range.offset > found->bytes || bytes > found->bytes - range.offset ||
        range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
        return std::nullopt;
    return CudaHostMemoryView{static_cast<std::byte*>(found->mapping->pointer) + range.offset,
                              found->is_pageable ? 0 : found->address + range.offset, bytes,
                              found->mapping, found->is_pageable};
}
std::vector<std::shared_ptr<const void>> CudaExecutionSession::host_lifetimes() {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted || state.is_closing)
        throw std::logic_error("unavailable CUDA host owner");
    std::vector<std::shared_ptr<const void>> lifetimes;
    lifetimes.reserve(state.storage->host_allocations.size());
    for (const auto& allocation : state.storage->host_allocations)
        if (!allocation.is_retiring) lifetimes.push_back(allocation.mapping);
    return lifetimes;
}
ExecutionAcceptance CudaExecutionSession::accept_host_transfer(std::uint64_t sequence,
                                                               CudaAllocationRange device,
                                                               const CudaHostMemoryView& host,
                                                               std::uint64_t stream,
                                                               bool is_upload) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture() || host.is_pageable)
        return invalid;
    CUdeviceptr address{};
    if (device.allocation || device.offset || host.bytes) {
        const auto found = state.find_allocation(device.allocation);
        if (!found || device.offset > found->bytes || host.bytes > found->bytes - device.offset ||
            device.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return invalid;
        address = found->address + device.offset;
    }
    if (host.bytes && (!host.pointer || !host.lifetime)) return invalid;
    return state.executor->accept_memory(
        sequence, {.kind = is_upload ? CudaMemoryKind::kUpload : CudaMemoryKind::kDownload,
                   .destination = is_upload ? address : 0,
                   .source = is_upload ? 0 : address,
                   .count = host.bytes,
                   .stream = stream,
                   .is_asynchronous = true,
                   .host = host.pointer,
                   .host_lifetime = host.lifetime});
}

ExecutionAcceptance CudaExecutionSession::accept_memory(std::uint64_t sequence,
                                                        const CudaSessionMemory& memory) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture()) return invalid;
    std::size_t width{1};
    switch (memory.kind) {
        case CudaMemoryKind::kCopy:
        case CudaMemoryKind::kFill8:
            break;
        case CudaMemoryKind::kFill16:
            width = 2;
            break;
        case CudaMemoryKind::kFill32:
            width = 4;
            break;
        default:
            return invalid;
    }
    if (memory.count > std::numeric_limits<std::size_t>::max() / width) return invalid;
    const auto bytes = memory.count * width;
    const auto address = [&](CudaAllocationRange range) -> std::optional<CUdeviceptr> {
        if (!range.allocation && !range.offset && !bytes) return CUdeviceptr{};
        const auto found = state.find_allocation(range.allocation);
        if (!found || range.offset > found->bytes || bytes > found->bytes - range.offset ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return std::nullopt;
        return found->address + range.offset;
    };
    const auto destination = address(memory.destination);
    auto source = std::optional<CUdeviceptr>{0};
    if (memory.kind == CudaMemoryKind::kCopy)
        source = address(memory.source);
    else if (memory.source.allocation || memory.source.offset)
        return invalid;
    if (!destination || !source) return invalid;
    return state.executor->accept_memory(sequence, {.kind = memory.kind,
                                                    .destination = *destination,
                                                    .source = *source,
                                                    .count = memory.count,
                                                    .value = memory.value,
                                                    .stream = memory.stream,
                                                    .is_asynchronous = memory.is_asynchronous});
}

CudaSessionGraphSnapshot CudaExecutionSession::graph_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {};
    return state.graphs ? state.graphs->snapshot()
                        : CudaSessionGraphSnapshot{.status = CUDA_SUCCESS};
}
CudaSessionGraphResult CudaExecutionSession::begin_graph_capture(std::uint64_t stream, int mode,
                                                                 std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return {.status = gate.status};
    const auto found =
        std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
    if (!stream || found == state.storage->streams.end() || !found->is_nonblocking)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (!state.dispatch.graph.begin) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    try {
        if (!state.graphs)
            state.graphs = std::make_unique<CudaExecutionGraph>(
                state.dispatch.graph, state.storage->context,
                state.dispatch.execution.context_get_current, state.limits.max_objects,
                state.limits.submissions.max_commands);
        const auto context = capture_external_context();
        if (context.status != CUDA_SUCCESS) return {.status = context.status};
        auto lifetime = std::make_shared<const CudaSessionExternalContext>(context.context);
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->begin(stream, found->native, mode, caller, std::move(lifetime));
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CudaSessionGraphResult CudaExecutionSession::end_graph_capture(std::uint64_t stream,
                                                               std::uint64_t caller,
                                                               bool has_output,
                                                               std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->end(stream, caller, has_output, seed);
}
CudaSessionGraphResult CudaExecutionSession::graph_capture_state(std::uint64_t stream,
                                                                 std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.graphs && state.graphs->snapshot().has_capture) {
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->state(stream, caller);
    }
    CUstream native = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
        native = found->native;
    }
    if (!state.dispatch.graph.state) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS)
        return {.status = proof};
    cudaStreamCaptureStatus observed{};
    const auto original = state.dispatch.graph.state(native, &observed);
    auto proof = state.check_context(true);
    if (proof != CUDA_SUCCESS) state.fault();
    if (proof == CUDA_SUCCESS && original == cudaSuccess &&
        observed != cudaStreamCaptureStatusNone) {
        state.fault();
        proof = CUDA_ERROR_UNKNOWN;
    }
    return {.status = proof,
            .original_status = static_cast<std::int32_t>(original),
            .capture_status = static_cast<std::uint32_t>(observed),
            .is_quarantined = state.is_faulted};
}
CudaSessionGraphResult CudaExecutionSession::instantiate_graph(std::uint64_t graph,
                                                               std::uint64_t flags, bool has_output,
                                                               std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return {.status = gate.status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->instantiate(graph, flags, has_output, seed);
}
CudaSessionGraphResult CudaExecutionSession::destroy_graph(std::uint64_t graph) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->destroy(graph);
}
CudaSessionGraphResult CudaExecutionSession::destroy_graph_exec(std::uint64_t executable) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->destroy_exec(executable);
}
CudaSessionGraphResult CudaExecutionSession::invalidate_graph_capture(
    std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    return state.graphs ? state.graphs->invalidate(caller)
                        : CudaSessionGraphResult{.status = CUDA_ERROR_INVALID_HANDLE};
}
CudaSessionGraphResult CudaExecutionSession::abort_graph_capture() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_SUCCESS};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->abort();
}
CudaSessionGraphResult CudaExecutionSession::capture_memory(const CudaSessionMemory& memory,
                                                            std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.graphs) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (!memory.is_asynchronous ||
        (memory.kind != CudaMemoryKind::kCopy && memory.kind != CudaMemoryKind::kFill8 &&
         memory.kind != CudaMemoryKind::kFill16 && memory.kind != CudaMemoryKind::kFill32))
        return state.graphs->invalidate(caller);
    const auto width = memory.kind == CudaMemoryKind::kFill32   ? 4U
                       : memory.kind == CudaMemoryKind::kFill16 ? 2U
                                                                : 1U;
    if (memory.count > SIZE_MAX / width) return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto bytes = memory.count * width;
    const std::array ranges{CudaSessionExternalRangeRequest{memory.destination, bytes},
                            CudaSessionExternalRangeRequest{memory.source, bytes}};
    const auto capture = capture_external_command(
        memory.stream, std::span{ranges}.first(memory.kind == CudaMemoryKind::kCopy ? 2 : 1));
    if (capture.status != CUDA_SUCCESS) return {.status = capture.status};
    const auto& sdk = state.dispatch.execution;
    if ((memory.kind == CudaMemoryKind::kCopy && !sdk.copy_device_async) ||
        (memory.kind == CudaMemoryKind::kFill8 && !sdk.fill8_async) ||
        (memory.kind == CudaMemoryKind::kFill16 && !sdk.fill16_async) ||
        (memory.kind == CudaMemoryKind::kFill32 && !sdk.fill32_async))
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    try {
        auto lifetime = std::make_shared<const CudaSessionExternalCommand>(capture.command);
        const auto& command = *capture.command.state_;
        const auto destination = command.ranges[0].address;
        const auto source = memory.kind == CudaMemoryKind::kCopy ? command.ranges[1].address : 0;
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->append(memory.stream, caller, std::move(lifetime), [&] {
            switch (memory.kind) {
                case CudaMemoryKind::kCopy:
                    return sdk.copy_device_async(destination, source, bytes, command.stream);
                case CudaMemoryKind::kFill8:
                    return sdk.fill8_async(destination, static_cast<unsigned char>(memory.value),
                                           memory.count, command.stream);
                case CudaMemoryKind::kFill16:
                    return sdk.fill16_async(destination, static_cast<unsigned short>(memory.value),
                                            memory.count, command.stream);
                case CudaMemoryKind::kFill32:
                    return sdk.fill32_async(destination, memory.value, memory.count,
                                            command.stream);
                default:
                    return CUDA_ERROR_INVALID_VALUE;
            }
        });
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CudaSessionGraphResult CudaExecutionSession::capture_kernel(const CudaExecutionKernel& kernel,
                                                            std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto function =
        std::ranges::find(state.storage->functions, kernel.function, &Implementation::Function::id);
    const auto stream =
        std::ranges::find(state.storage->streams, kernel.stream, &Implementation::Stream::id);
    if (function == state.storage->functions.end() || stream == state.storage->streams.end() ||
        kernel.parameters.size() != function->parameter_bytes ||
        (!kernel.parameters.empty() && !kernel.parameters.data()) ||
        std::ranges::any_of(kernel.grid, [](auto value) { return !value; }) ||
        std::ranges::any_of(kernel.block, [](auto value) { return !value; }))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    try {
        std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
        std::erase_if(state.external_commands, [](const auto& entry) { return entry.expired(); });
        if (state.external_contexts.size() == state.limits.max_objects ||
            state.external_commands.size() == state.limits.submissions.max_commands)
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        auto command = std::make_shared<CudaSessionExternalCommand::State>();
        command->stream_id = kernel.stream;
        command->stream = stream->native;
        command->parent_modules.push_back(function->module);
        command->ranges.reserve(state.storage->allocations.size());
        for (const auto& allocation : state.storage->allocations) {
            command->ranges.push_back({{allocation.id, 0}, allocation.address, allocation.bytes});
            if (allocation.module) command->parent_modules.push_back(allocation.module);
        }
        if (state.process_memory) {
            auto leases = state.process_memory->owner()->borrow_live();
            if (leases.status != CUDA_SUCCESS) return {.status = leases.status};
            if (leases.lifetime) command->process_lifetimes.push_back(std::move(leases.lifetime));
        }
        for (auto& lease : host_lifetimes()) command->process_lifetimes.push_back(std::move(lease));
        if (kernel.lifetime) command->process_lifetimes.push_back(kernel.lifetime);
        command->context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        std::vector<std::byte> parameters(kernel.parameters.begin(), kernel.parameters.end());
        std::vector<void*> arguments;
        arguments.reserve(function->arguments.size());
        for (const auto& layout : function->arguments)
            arguments.push_back(parameters.data() + layout.offset);
        state.external_contexts.push_back(command->context.state_);
        state.external_commands.push_back(command);
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->append(kernel.stream, caller, command, [&] {
            return state.dispatch.execution.launch_kernel(
                function->native, kernel.grid[0], kernel.grid[1], kernel.grid[2], kernel.block[0],
                kernel.block[1], kernel.block[2], kernel.shared_memory_bytes, stream->native,
                arguments.empty() ? nullptr : arguments.data(), nullptr);
        });
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): distinct scoped wire identities.
CudaSessionGraphLaunchAcceptance CudaExecutionSession::accept_graph_launch(
    std::uint64_t sequence, std::uint64_t executable, std::uint64_t stream) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || !state.graphs || state.is_faulted || state.is_closing ||
        state.is_native_issue_active || state.has_graph_capture())
        return {};
    auto prepared = state.graphs->prepare_launch(executable);
    if (prepared.status != CUDA_SUCCESS) return {};
    auto accepted = state.executor->accept_library(
        sequence, {.stream = stream, .parameters = {}, .call = prepared.operation});
    return {accepted, accepted.accepted() ? std::move(prepared.operation) : nullptr};
}

ExecutionAcceptance CudaExecutionSession::accept_uva_copy(std::uint64_t sequence,
                                                          const CudaSessionUvaCopy& copy) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture()) return invalid;
    CudaExecutionMemory memory{.kind = CudaMemoryKind::kCopyUva,
                               .count = copy.bytes,
                               .stream = copy.stream,
                               .is_asynchronous = copy.is_asynchronous};
    const auto resolve = [&](const CudaSessionUvaEndpoint& endpoint,
                             std::size_t index) -> std::optional<CUdeviceptr> {
        const bool has_device = endpoint.device.allocation || endpoint.device.offset;
        const auto& host = endpoint.host;
        const bool has_host =
            host.pointer || host.address || host.bytes || host.lifetime || host.is_pageable;
        switch (endpoint.kind) {
            case CudaUvaEndpointKind::kNull:
                if (copy.bytes || has_device || has_host) return std::nullopt;
                return CUdeviceptr{};
            case CudaUvaEndpointKind::kDevice: {
                if (has_host || !endpoint.device.allocation) return std::nullopt;
                const auto view = device_memory(endpoint.device, copy.bytes);
                if (!view) return std::nullopt;
                memory.endpoint_lifetimes[index] = view->lifetime;
                return view->address;
            }
            case CudaUvaEndpointKind::kHost: {
                if (has_device || !host.pointer || !host.lifetime || copy.bytes > host.bytes ||
                    (host.is_pageable && host.address))
                    return std::nullopt;
                const auto address =
                    static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host.pointer));
                if (copy.bytes > std::numeric_limits<CUdeviceptr>::max() - address)
                    return std::nullopt;
                memory.endpoint_lifetimes[index] = host.lifetime;
                // The original generic SDK infers host/device direction. A
                // registered host's device alias is not its host endpoint.
                return address;
            }
        }
        return std::nullopt;
    };
    const auto destination = resolve(copy.destination, 0);
    const auto source = resolve(copy.source, 1);
    if (!destination || !source) return invalid;
    if (copy.bytes && copy.destination.kind == CudaUvaEndpointKind::kHost &&
        copy.source.kind == CudaUvaEndpointKind::kHost && *destination < *source + copy.bytes &&
        *source < *destination + copy.bytes)
        return invalid;
    memory.destination = *destination;
    memory.source = *source;
    return state.executor->accept_memory(sequence, memory);
}

ExecutionAcceptance CudaExecutionSession::accept_async_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& input) noexcept {
    return accept_async_kernel(sequence, input, nullptr);
}

ExecutionAcceptance CudaExecutionSession::accept_async_kernel(std::uint64_t sequence,
                                                              const CudaExecutionKernel& input,
                                                              bool* requires_preparation) noexcept {
    if (requires_preparation) *requires_preparation = false;
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.has_destroy_attempt || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const auto function =
        std::ranges::find(state.storage->functions, input.function, &Implementation::Function::id);
    if (function == state.storage->functions.end() ||
        input.parameters.size() != function->parameter_bytes ||
        (!input.parameters.empty() && !input.parameters.data()) ||
        std::ranges::any_of(input.grid, [](auto value) { return !value; }) ||
        std::ranges::any_of(input.block, [](auto value) { return !value; }) ||
        (input.stream && !std::ranges::any_of(state.storage->streams, [&](const auto& stream) {
             return stream.id == input.stream;
         })))
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    std::size_t capacity = input.parameters.size();
    const bool may_transform = state.persistent.workers && !function->variants.empty() &&
                               !function->is_persistent_disabled;
    const bool uses_original_admission =
        !may_transform && !state.profile_cache && !state.persistent.uses_workload_agnostic;
    if (!uses_original_admission) {
        std::erase_if(state.deferred_kernels, [](const auto& value) { return value.expired(); });
        if (state.deferred_kernels.size() == state.limits.submissions.max_commands)
            return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
    if (may_transform)
        for (const auto& variant : function->variants) capacity = std::max(capacity, variant.bytes);
    if (capacity > state.limits.submissions.max_parameter_bytes)
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    try {
        auto retained = input;
        if (!state.storage->host_allocations.empty()) {
            auto lifetimes = host_lifetimes();
            if (input.lifetime) lifetimes.push_back(input.lifetime);
            if (!lifetimes.empty())
                retained.lifetime = std::make_shared<std::vector<std::shared_ptr<const void>>>(
                    std::move(lifetimes));
        }
        if (uses_original_admission) {
            // Session observers come only from profile selection, as in the
            // prepared original path. No native preparation or profiling is due.
            retained.observation.reset();
            return state.executor->accept_async_original_kernel(sequence, retained);
        }
        auto deferred = std::make_shared<Implementation::DeferredKernel>();
        deferred->storage = state.storage;
        deferred->kernel = retained;
        deferred->kernel.parameters = {};
        deferred->parameter_bytes = input.parameters.size();
        if (may_transform) {
            if (!state.storage->is_progress_ready)
                return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
            const auto free = std::ranges::find(state.storage->occupied, false);
            if (free == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            deferred->reserved_slot =
                static_cast<std::size_t>(free - state.storage->occupied.begin());
            *free = true;
        }
        auto kernel = input;
        kernel.lifetime = deferred;
        auto accepted = state.executor->accept_deferred_kernel(sequence, kernel, capacity);
        if (accepted.accepted()) {
            deferred->receipt = accepted.receipt;
            state.deferred_kernels.push_back(deferred);
            if (requires_preparation) *requires_preparation = true;
        }
        return accepted;
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

CudaSessionPendingResult CudaExecutionSession::pending_preparation(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    CudaSessionPendingResult result;
    result.command.description_.control_epoch = state.control_epoch;
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        result.command.description_.member_generation = member.generation;
        result.command.description_.configuration_epoch = member.configuration_epoch;
        result.command.description_.physical_configuration_epoch =
            member.physical_configuration_epoch;
    }
    auto pending = state.executor->pending_preparation(sequence);
    if (pending.status != CUDA_SUCCESS || !pending.command) return {.status = pending.status};
    result.command.description_.command = pending.command.description();
    result.command.command_ = std::move(pending.command);
    result.command.owner_ = state.storage;
    return result;
}

CudaSessionKernelPreparationResult CudaExecutionSession::prepare_pending_kernel(
    const CudaSessionPendingCommand& command) noexcept {
    using Disposition = CudaSessionKernelPreparationDisposition;
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = CUDA_ERROR_NOT_READY, .disposition = Disposition::kDeferred};
    if (state.process_memory && state.process_memory->native_health() != CUDA_SUCCESS)
        return {.status = state.process_memory->native_health()};
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY, .disposition = Disposition::kDeferred};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_preparation(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    std::shared_ptr<Implementation::DeferredKernel> deferred;
    for (const auto& weak : state.deferred_kernels)
        if (auto value = weak.lock();
            value && value->receipt.sequence() == command.description_.command.sequence) {
            deferred = std::move(value);
            break;
        }
    if (!deferred) return {.status = CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionPendingCommand prepared_command;
    const auto uncertain = [&](CUresult failure) {
        const auto observed = state.executor->quarantine_deferred_kernel(
            prepared_command ? prepared_command : command.command_, failure);
        state.fault();
        return CudaSessionKernelPreparationResult{
            .status = observed == CUDA_SUCCESS ? failure : observed,
            .disposition = Disposition::kUncertain};
    };
    const auto rejected = [&](CUresult failure) {
        if (!state.executor->reject_command(deferred->receipt, failure)) return uncertain(failure);
        return CudaSessionKernelPreparationResult{.status = failure,
                                                  .disposition = Disposition::kRejected};
    };
    std::vector<std::byte> parameters;
    try {
        parameters.resize(deferred->parameter_bytes);
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
    const auto copied = state.executor->copy_deferred_parameters(command.command_, parameters);
    if (copied != CUDA_SUCCESS) return rejected(copied);
    try {
        if (state.storage->domain_member) {
            const auto configured = state.storage->domain_member.configuration_status(true);
            if (configured == CUDA_ERROR_NOT_READY)
                return {.status = configured, .disposition = Disposition::kDeferred};
            if (configured != CUDA_SUCCESS) return uncertain(configured);
        }
        const auto before = state.check_context(true);
        if (before != CUDA_SUCCESS) return uncertain(before);
        auto kernel = deferred->kernel;
        kernel.parameters = parameters;
        CUresult failure = CUDA_SUCCESS;
        ExecutionAcceptance prepared;
        {
            const NativeExecutionScope preparation_scope(state.is_preparation_active);
            prepared = accept_kernel_impl(deferred->receipt.sequence(), kernel, &command.command_,
                                          &deferred->receipt, &deferred->reserved_slot, &failure,
                                          &prepared_command);
        }
        if (state.is_faulted)
            return uncertain(failure != CUDA_SUCCESS ? failure : CUDA_ERROR_UNKNOWN);
        if (!prepared.accepted())
            return rejected(failure == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : failure);
        if (deferred->reserved_slot) {
            state.storage->occupied[*deferred->reserved_slot] = false;
            deferred->reserved_slot.reset();
        }
        const auto pending = pending_command(deferred->receipt.sequence());
        if (pending.status != CUDA_SUCCESS && pending.status != CUDA_ERROR_NOT_READY)
            return uncertain(pending.status);
        return {.status = CUDA_SUCCESS,
                .disposition = Disposition::kPrepared,
                .command = pending.command};
    } catch (...) {
        return uncertain(CUDA_ERROR_UNKNOWN);
    }
}

ExecutionAcceptance CudaExecutionSession::accept_kernel(std::uint64_t sequence,
                                                        const CudaExecutionKernel& input) noexcept {
    try {
        return accept_kernel_impl(sequence, input);
    } catch (...) {
        implementation_->fault();
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    }
}

ExecutionAcceptance CudaExecutionSession::accept_kernel_impl(
    std::uint64_t sequence, const CudaExecutionKernel& input,
    const CudaExecutionPendingCommand* preparation, const ExecutionReceipt* receipt,
    std::optional<std::size_t>* reserved_slot, CUresult* preparation_status,
    CudaExecutionPendingCommand* prepared_command) {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    const auto note_failure = [&](CUresult failure) {
        if (preparation_status && *preparation_status == CUDA_SUCCESS)
            *preparation_status = failure;
    };
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || (state.persistent.uses_workload_agnostic &&
                                         (state.is_folded_active || state.is_rnn_query_active)))
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_closing && !preparation)
        return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    auto kernel = input;
    try {
        if (!preparation && !state.storage->host_allocations.empty()) {
            auto lifetimes = host_lifetimes();
            if (kernel.lifetime) lifetimes.push_back(kernel.lifetime);
            if (!lifetimes.empty())
                kernel.lifetime = std::make_shared<std::vector<std::shared_ptr<const void>>>(
                    std::move(lifetimes));
        }
    } catch (...) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
    const auto function = std::ranges::find_if(
        state.storage->functions, [&](const auto& entry) { return entry.id == kernel.function; });
    // Private library bindings are not public launch handles, even if guessed.
    if (function == state.storage->functions.end())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    const bool sharing = state.persistent.uses_workload_agnostic;
    std::optional<NativeExecutionScope> sharing_scope;
    if (sharing) sharing_scope.emplace(state.is_native_issue_active);
    if (state.profile_cache && function->has_configuration_attempt &&
        !function->is_persistent_disabled && !function->has_unknown_configuration &&
        !function->configuration) {
        const NativeExecutionScope capture_scope(state.is_native_issue_active);
        function->configuration = capture_function_configuration(
            state.dispatch, Implementation::profile_targets(*function), function->cache_preference,
            [&] {
                if (state.check_context(true) == CUDA_SUCCESS) return true;
                state.fault();
                return false;
            });
        if (state.is_faulted) return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    }
    const auto context_configuration =
        state.profile_cache ? state.profile_context(sharing_scope.has_value() || preparation)
                            : std::optional<CudaProfileContextConfiguration>{};
    const CudaProfileGeometry profile_geometry{
        kernel.grid,
        kernel.block,
        kernel.shared_memory_bytes,
        state.persistent.kind,
        context_configuration.value_or(CudaProfileContextConfiguration{}),
        function->configuration.value_or(CudaProfileFunctionConfiguration{})};
    bool has_profile_allocation_failure{};
    const auto profile_choice = [&](std::span<const transform::PtxExecutionKind> kinds) {
        auto choice = !state.persistent.uses_workload_agnostic && state.profile_cache &&
                              context_configuration && function->configuration
                          ? state.profile_cache->choose(function->profile, profile_geometry,
                                                        state.persistent.workers, kinds,
                                                        state.requires_profile_preemption)
                          : CudaProfileChoice{};
        try {
            choice.observation =
                state.profile_observation(*function, profile_geometry, choice.observation);
        } catch (...) {
            if (choice.observation) choice.observation->finish(false);
            has_profile_allocation_failure = true;
            return CudaProfileChoice{};
        }
        return choice;
    };
    const auto publish_original = [&] {
        if (!preparation) return state.executor->accept_kernel(sequence, kernel);
        const auto proof = state.check_context(true);
        if (proof != CUDA_SUCCESS) {
            note_failure(proof);
            state.fault();
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kQuarantined,
                                       .receipt = {}};
        }
        const auto prepared =
            state.executor->complete_kernel_preparation(*preparation, kernel.observation);
        if (prepared.status != CUDA_SUCCESS) {
            note_failure(prepared.status);
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
        }
        if (prepared_command) *prepared_command = prepared.command;
        return ExecutionAcceptance{.status = ExecutionAcceptStatus::kAccepted, .receipt = *receipt};
    };
    const auto accept_original = [&] {
        auto choice = profile_choice({});
        if (has_profile_allocation_failure) {
            note_failure(CUDA_ERROR_OUT_OF_MEMORY);
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kOutOfMemory,
                                       .receipt = {}};
        }
        kernel.observation = std::move(choice.observation);
        return publish_original();
    };
    if (!state.persistent.workers || function->variants.empty() || function->is_persistent_disabled)
        return accept_original();
    if (sharing && (!context_configuration || !function->configuration)) return accept_original();
    const auto proof = [&] {
        const auto status = state.check_context(sharing || preparation);
        if (status != CUDA_SUCCESS) {
            note_failure(status);
            if (sharing || preparation) state.fault();
        }
        return status == CUDA_SUCCESS;
    };
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!proof() || !state.storage->is_progress_ready ||
        kernel.parameters.size() != function->parameter_bytes ||
        (!kernel.parameters.empty() && !kernel.parameters.data()) ||
        (kernel.stream && !std::ranges::any_of(state.storage->streams, [&](const auto& entry) {
             return entry.id == kernel.stream;
         })))
        return invalid;
    std::uint64_t total = 1, threads = 1;
    for (const auto dimension : kernel.grid) {
        if (!dimension || total > std::numeric_limits<std::uint32_t>::max() / dimension)
            return accept_original();
        total *= dimension;
    }
    for (const auto dimension : kernel.block) {
        if (!dimension ||
            threads > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / dimension)
            return accept_original();
        threads *= dimension;
    }
    std::array<transform::PtxExecutionKind, 5> eligible{};
    std::array<CudaProfileSharingCandidate, 4> sharing_candidates{};
    std::size_t eligible_count{};
    const Implementation::PrivateFunction* selected{};
    for (const auto& variant : function->variants) {
        if (sharing && variant.kind == transform::PtxExecutionKind::kPreemptive) continue;
        if (variant.kind == transform::PtxExecutionKind::kStatic &&
            (kernel.grid[1] != 1 || kernel.grid[2] != 1))
            continue;
        int capacity{};
        const auto occupancy = state.invoke(state.dispatch.occupancy, &capacity, variant.native,
                                            static_cast<int>(threads), kernel.shared_memory_bytes);
        if ((sharing || preparation) && !proof()) return invalid;
        if (preparation && occupancy != CUDA_SUCCESS) {
            note_failure(occupancy);
            return invalid;
        }
        if (occupancy != CUDA_SUCCESS || capacity <= 0) continue;
        if (sharing) {
            const auto cap = static_cast<std::uint64_t>(state.sharing_sm_count) *
                             static_cast<unsigned int>(capacity);
            sharing_candidates[eligible_count] = {
                variant.kind,
                static_cast<unsigned int>(std::min<std::uint64_t>(state.persistent.workers, cap))};
        }
        eligible[eligible_count++] = variant.kind;
        if (variant.kind == state.persistent.kind) selected = &variant;
    }
    if (!eligible_count) return accept_original();
    auto choice =
        sharing && context_configuration && function->configuration
            ? state.profile_cache->choose_sharing(
                  function->profile, profile_geometry,
                  {state.sharing_sm_count, state.persistent.sharing_threads_per_sm,
                   state.persistent.workers, state.persistent.sharing_minimum_speed_percent},
                  std::span{sharing_candidates}.first(eligible_count))
            : profile_choice(std::span{eligible}.first(eligible_count));
    if (has_profile_allocation_failure) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
    kernel.observation = std::move(choice.observation);
    if (sharing && !choice.was_selected) return publish_original();
    if (choice.was_selected && !choice.workers) return publish_original();
    if (choice.was_selected) {
        const auto found = std::ranges::find(function->variants, choice.kind,
                                             &Implementation::PrivateFunction::kind);
        if (found == function->variants.end()) return invalid;
        selected = &*found;
    }
    if (!selected) return publish_original();
    const auto kind = selected->kind;
    const auto configured_workers = choice.was_selected ? choice.workers : state.persistent.workers;
    const auto free_slot =
        reserved_slot && *reserved_slot
            ? state.storage->occupied.begin() + static_cast<std::ptrdiff_t>(**reserved_slot)
            : std::ranges::find(state.storage->occupied, false);
    if (free_slot == state.storage->occupied.end()) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
    try {
        const auto index = static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
        const auto count = function->arguments.size();
        const auto workers =
            kind == transform::PtxExecutionKind::kStatic
                ? static_cast<unsigned int>(std::min<std::uint64_t>(configured_workers, total))
                : configured_workers;
        const std::array<std::size_t, 2> slice_offsets =
            kind == transform::PtxExecutionKind::kSliced
                ? std::array{selected->arguments[count + 3].offset,
                             selected->arguments[count + 4].offset}
                : std::array<std::size_t, 2>{};
        auto progress = std::make_shared<Implementation::Progress>(
            state.storage, state.dispatch,
            Implementation::ProgressParameters{
                .slot = index,
                .sequence = sequence,
                .total = total,
                .workers = workers,
                .max_workers = state.persistent.workers,
                .can_retarget_workers = true,
                .kind = kind,
                .slice_offsets = slice_offsets,
                .observes_block_completion = selected->observes_block_completion});
        *free_slot = true;
        if (reserved_slot) reserved_slot->reset();
        std::vector<std::byte> parameters(selected->bytes);
        if (!kernel.parameters.empty())
            std::memcpy(parameters.data(), kernel.parameters.data(), kernel.parameters.size());
        const std::size_t axes = kind == transform::PtxExecutionKind::kStatic ? 1 : 3;
        for (std::size_t axis = 0; axis < axes; ++axis)
            std::memcpy(parameters.data() + selected->arguments[count + axis].offset,
                        &kernel.grid[axis], sizeof(kernel.grid[axis]));
        if (kind == transform::PtxExecutionKind::kDynamic ||
            kind == transform::PtxExecutionKind::kPreemptive) {
            const auto counter = progress->counter();
            std::memcpy(parameters.data() + selected->arguments[count + 3].offset, &counter,
                        sizeof(counter));
        }
        if (kind == transform::PtxExecutionKind::kPreemptive) {
            const auto retreat = progress->retreat();
            std::memcpy(parameters.data() + selected->arguments[count + 4].offset, &retreat,
                        sizeof(retreat));
            if (selected->observes_block_completion) {
                const auto completion = progress->block_completion();
                std::memcpy(parameters.data() + selected->arguments[count + 5].offset, &completion,
                            sizeof(completion));
            }
        }
        std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
        ExecutionAcceptance accepted;
        if (preparation) {
            const auto proof = state.check_context(true);
            if (proof != CUDA_SUCCESS) {
                note_failure(proof);
                state.fault();
                return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
            }
            const CudaExecutionPersistentKernel prepared{selected->native, workers, parameters,
                                                         progress};
            const auto completed = state.executor->complete_kernel_preparation(
                *preparation, kernel.observation, &prepared);
            if (completed.status == CUDA_SUCCESS)
                accepted = {.status = ExecutionAcceptStatus::kAccepted, .receipt = *receipt};
            else
                note_failure(completed.status);
            if (completed.status == CUDA_SUCCESS && prepared_command)
                *prepared_command = completed.command;
        } else {
            accepted = state.executor->accept_persistent_kernel(
                sequence, kernel, {selected->native, workers, parameters, progress});
        }
        if (accepted.accepted()) {
            state.progress.push_back(progress);
            if (state.persistent_accepted != std::numeric_limits<std::uint64_t>::max())
                ++state.persistent_accepted;
        }
        return accepted;
    } catch (...) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
CudaExecutionIssueResult CudaExecutionSession::issue_next() noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.state = CudaExecutionIssueState::kIdle, .native_status = CUDA_ERROR_NOT_READY};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->issue_next();
    state.update_domain_physical();
    return result;
}

CudaSessionPendingResult CudaExecutionSession::pending_command(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    CudaSessionPendingResult result;
    result.command.description_.control_epoch = state.control_epoch;
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        result.command.description_.member_generation = member.generation;
        result.command.description_.configuration_epoch = member.configuration_epoch;
        result.command.description_.physical_configuration_epoch =
            member.physical_configuration_epoch;
    }
    auto pending = state.executor->pending_command(sequence);
    result.status = pending.status;
    if (pending.status != CUDA_SUCCESS || !pending.command) return {.status = pending.status};
    result.command.description_.command = pending.command.description();
    result.command.command_ = std::move(pending.command);
    result.command.owner_ = state.storage;
    return result;
}

CudaExecutionIssueResult CudaExecutionSession::issue(
    const CudaSessionPendingCommand& command) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.state = CudaExecutionIssueState::kIdle, .native_status = CUDA_ERROR_NOT_READY};
    const auto failure = [&](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .sequence = command.description_.command.sequence,
                                        .native_status = status};
    };
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return failure(CUDA_ERROR_NOT_READY);
    if (state.is_faulted) return failure(CUDA_ERROR_UNKNOWN);
    if (command.owner_.lock() != state.storage) return failure(CUDA_ERROR_INVALID_HANDLE);
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return failure(current.status);
    if (!current.command || current.command.description_ != command.description_)
        return failure(CUDA_ERROR_INVALID_HANDLE);
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS) return failure(configured);
    }
    if (!state.reserve_domain_physical()) return failure(CUDA_ERROR_UNKNOWN);
    const auto result = state.executor->issue(command.command_);
    state.update_domain_physical();
    return result;
}
CudaSessionPendingResult CudaExecutionSession::select_pending_kernel(
    const CudaSessionPendingCommand& command,
    const CudaExecutionPendingKernelSelection& selection) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (!selection.uses_original && selection.workers > state.persistent.workers)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    {
        const NativeExecutionScope active{state.is_native_issue_active};
        if (state.storage->domain_member) {
            const auto configured = state.storage->domain_member.configuration_status(true);
            if (configured != CUDA_SUCCESS) return {.status = configured};
        }
        const auto selected = state.executor->select_pending_kernel(command.command_, selection);
        if (state.executor->snapshot().is_quarantined) state.fault();
        if (selected.status != CUDA_SUCCESS) return {.status = selected.status};
    }
    return pending_command(command.description_.command.sequence);
}

CudaSessionPairKernelResult CudaExecutionSession::prepare_pair_kernel(
    const CudaSessionPendingCommand& command) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
    if (state.process_memory && state.process_memory->native_health() != CUDA_SUCCESS)
        return {.status = state.process_memory->native_health()};
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status();
        if (configured != CUDA_SUCCESS) return {.status = configured};
    }
    if (!state.next_pair_catalog) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    if (!state.dispatch.device_attribute || !state.dispatch.function_get_attribute ||
        !state.dispatch.occupancy || !command.description_.command.is_persistent)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const auto function =
        std::ranges::find(state.storage->functions, command.description_.command.function,
                          &Implementation::Function::id);
    const auto progress = persistent_progress(command.description_.command.sequence);
    if (!progress) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const NativeExecutionScope scope{state.is_native_issue_active};
    const auto proof = [&] {
        CUresult status{CUDA_ERROR_UNKNOWN};
        try {
            status = state.check_context(true);
        } catch (...) {
            state.fault();
            return CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return {.status = status};
    if (function == state.storage->functions.end()) {
        const auto is_private_binding = [&](const auto& ids, const auto& functions) {
            const auto id = command.description_.command.function;
            const auto found = std::ranges::find(ids, id);
            return id && found != ids.end() &&
                   functions[static_cast<std::size_t>(found - ids.begin())];
        };
        // Private replacement bindings are real persistent commands but have
        // no public-PTX measurement catalog. Do not classify an arbitrary
        // missing function, lost progress or native metadata error this way.
        if (progress->kind == transform::PtxExecutionKind::kPreemptive && progress->workers &&
            (is_private_binding(state.storage->gemm_ids, state.storage->gemm_functions) ||
             is_private_binding(state.storage->lt_ids, state.storage->lt_functions)))
            return {.status = CUDA_ERROR_NOT_SUPPORTED,
                    .disposition = CudaSessionPairKernelDisposition::kUnsupported};
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto variant = std::ranges::find(function->variants, progress->kind,
                                           &Implementation::PrivateFunction::kind);
    if (variant == function->variants.end() || !variant->native)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    switch (progress->kind) {
        case transform::PtxExecutionKind::kPreemptive:
        case transform::PtxExecutionKind::kSliced:
            break;
        case transform::PtxExecutionKind::kStatic:
        case transform::PtxExecutionKind::kBlockStatic:
        case transform::PtxExecutionKind::kDynamic:
            return {.status = CUDA_ERROR_NOT_SUPPORTED,
                    .disposition = CudaSessionPairKernelDisposition::kUnsupported};
        default:
            return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto query = [&](auto callback, auto... arguments) {
        const auto before = proof();
        if (before != CUDA_SUCCESS) return before;
        CUresult status{CUDA_ERROR_UNKNOWN};
        try {
            status = state.invoke(callback, arguments...);
        } catch (...) {
            state.fault();
            return CUDA_ERROR_UNKNOWN;
        }
        const auto after = proof();
        return after == CUDA_SUCCESS ? status : after;
    };
    CudaSessionPairKernelResult result;
    auto& data = result.kernel.description_;
    data.pending = command.description_;
    data.geometry = {command.description_.command.grid, command.description_.command.block,
                     command.description_.command.shared_memory_bytes, state.persistent.kind};
    constexpr std::array attributes{CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR};
    std::array<unsigned int, attributes.size()> hardware{};
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        int value{};
        const auto status =
            query(state.dispatch.device_attribute, &value, attributes[index], state.native_device);
        if (status != CUDA_SUCCESS) return {.status = status};
        if (value <= 0) return {.status = CUDA_ERROR_INVALID_VALUE};
        hardware[index] = static_cast<unsigned int>(value);
    }
    data.limits = {hardware[0], hardware[1], hardware[2], hardware[3], state.persistent.workers,
                   hardware[4]};
    int registers{}, shared{}, occupancy{};
    auto status = query(state.dispatch.function_get_attribute, &registers,
                        CU_FUNC_ATTRIBUTE_NUM_REGS, variant->native);
    if (status != CUDA_SUCCESS) return {.status = status};
    status = query(state.dispatch.function_get_attribute, &shared,
                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, variant->native);
    if (status != CUDA_SUCCESS) return {.status = status};
    std::uint64_t threads{1};
    for (const auto dimension : data.geometry.block) {
        if (!dimension || threads > static_cast<unsigned int>(INT32_MAX) / dimension)
            return {.status = CUDA_ERROR_INVALID_VALUE};
        threads *= dimension;
    }
    status = query(state.dispatch.occupancy, &occupancy, variant->native, static_cast<int>(threads),
                   data.geometry.shared_memory_bytes);
    if (status != CUDA_SUCCESS) return {.status = status};
    if (registers < 0 || shared < 0 || occupancy <= 0 || threads > data.limits.threads_per_sm ||
        static_cast<unsigned int>(occupancy) > data.limits.blocks_per_sm)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    data.variant = {progress->kind, progress->workers, static_cast<unsigned int>(registers),
                    static_cast<unsigned int>(occupancy), static_cast<std::uint64_t>(shared)};
    if (state.profile_cache && !state.profile_cache->is_learning() && function->configuration) {
        const auto configuration = state.profile_context(true);
        if (configuration) {
            data.profile = function->profile;
            data.geometry.context = *configuration;
            data.geometry.function = *function->configuration;
        }
    }
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        data.execution_domain = member.context.id;
        data.is_configuration_materialized = member.is_configuration_materialized;
    } else {
        data.is_configuration_materialized = true;
    }
    data.catalog_generation = state.next_pair_catalog;
    state.next_pair_catalog =
        state.next_pair_catalog == UINT64_MAX ? 0 : state.next_pair_catalog + 1;
    result.kernel.command_ = command;
    result.kernel.owner_ = state.storage;
    result.status = CUDA_SUCCESS;
    result.disposition = CudaSessionPairKernelDisposition::kPrepared;
    return result;
}

CudaSessionPairKernelResult CudaExecutionSession::refresh_pair_kernel(
    const CudaSessionPairKernel& kernel, const CudaSessionPendingCommand& command) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (kernel.owner_.lock() != state.storage || command.owner_.lock() != state.storage ||
        !kernel.description_.catalog_generation)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    auto expected = kernel.description_.pending;
    expected.command.geometry_epoch = command.description_.command.geometry_epoch;
    expected.command.physical_grid = command.description_.command.physical_grid;
    expected.command.is_persistent = command.description_.command.is_persistent;
    if (expected != command.description_) return {.status = CUDA_ERROR_INVALID_HANDLE};
    CudaSessionPairKernelResult result;
    result.kernel = kernel;
    result.kernel.command_ = command;
    result.kernel.description_.pending = command.description_;
    result.kernel.description_.variant.workers = command.description_.command.is_persistent
                                                     ? command.description_.command.physical_grid[0]
                                                     : 0;
    result.status = CUDA_SUCCESS;
    result.disposition = CudaSessionPairKernelDisposition::kPrepared;
    return result;
}

bool CudaExecutionSession::can_share_configuration(
    const CudaExecutionSession& other) const noexcept {
    const auto& state = *implementation_;
    const auto& peer = *other.implementation_;
    return state.owns() && peer.owns() && !state.has_destroy_attempt && !peer.has_destroy_attempt &&
           !state.is_faulted && !peer.is_faulted &&
           state.storage->domain_member.can_share_configuration(peer.storage->domain_member);
}

core::ExecutionPairProposal CudaSessionPairSelection::proposal(
    std::array<std::uint64_t, 2> policy_ids) const noexcept {
    core::ExecutionPairProposal result;
    result.is_shared = choice_.is_shared;
    result.first_serial = choice_.is_shared ? 0 : choice_.first_original;
    for (std::size_t leg = 0; leg < kernels_.size(); ++leg) {
        const auto data = kernels_[leg].description();
        const auto& pending = data.pending;
        result.legs[leg] = {
            .session = policy_ids[leg],
            .command = {.sequence = pending.command.sequence,
                        .acceptance_order = pending.command.acceptance_order,
                        .physical_generation = pending.command.physical_generation,
                        .geometry_epoch = pending.command.geometry_epoch,
                        .command_epoch = pending.command.command_epoch,
                        .control_epoch = pending.control_epoch,
                        .member_generation = pending.member_generation,
                        .configuration_epoch = pending.configuration_epoch,
                        .physical_configuration_epoch = pending.physical_configuration_epoch,
                        .is_resume = pending.command.is_resume},
            .catalog_generation = data.catalog_generation,
            .workers = data.variant.workers};
    }
    return result;
}

std::array<core::ExecutionPairContext, 2> CudaSessionPairSelection::contexts() const noexcept {
    std::array<core::ExecutionPairContext, 2> result{};
    for (std::size_t leg = 0; leg < kernels_.size(); ++leg) {
        const auto data = kernels_[leg].description();
        result[leg] = {data.execution_domain, data.catalog_generation, data.variant.workers,
                       data.is_configuration_materialized};
    }
    return result;
}

namespace {
struct PairResourceCatalog {
    CudaPairProfileLimits limits;
    std::array<CudaProfileGeometry, 2> geometry;
    std::array<CudaPairProfileCandidate, 64> candidates{};
    std::size_t count{};
};
std::optional<PairResourceCatalog> pair_resource_catalog(
    const std::array<CudaSessionPairKernelDescription, 2>& kernels,
    std::span<const unsigned int> worker_counts) noexcept {
    if (worker_counts.empty() || worker_counts.size() > 8) return std::nullopt;
    PairResourceCatalog result;
    result.limits = kernels[0].limits;
    auto other = kernels[1].limits;
    result.limits.max_workers = other.max_workers =
        std::max(result.limits.max_workers, other.max_workers);
    if (result.limits != other) return std::nullopt;
    for (std::size_t index = 0; index < worker_counts.size(); ++index) {
        const auto workers = worker_counts[index];
        if (!workers || workers > static_cast<unsigned int>(INT32_MAX) || !result.limits.sm_count ||
            workers % result.limits.sm_count ||
            std::ranges::find(worker_counts.first(index), workers) !=
                worker_counts.first(index).end())
            return std::nullopt;
    }
    result.geometry = {kernels[0].geometry, kernels[1].geometry};
    for (const auto left : worker_counts)
        for (const auto right : worker_counts) {
            if (left > kernels[0].limits.max_workers || right > kernels[1].limits.max_workers)
                continue;
            CudaPairProfileCandidate candidate{{kernels[0].variant, kernels[1].variant}};
            candidate.legs[0].workers = left;
            candidate.legs[1].workers = right;
            if (cuda_pair_profile_candidate_fits(result.geometry, result.limits, candidate))
                result.candidates[result.count++] = candidate;
        }
    return result;
}
}  // namespace

CudaSessionPairSelectionResult CudaExecutionSession::inspect_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    if (&first == &second || worker_counts.empty() || worker_counts.size() > 8)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    std::array<CudaSessionPairKernel, 2> kernels{first_kernel, second_kernel};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto refreshed =
            sessions[leg]->refresh_pair_kernel(kernels[leg], kernels[leg].command_);
        if (refreshed.status != CUDA_SUCCESS) return {.status = refreshed.status};
        kernels[leg] = refreshed.kernel;
    }
    const auto cache = profiles.snapshot();
    if (!cache.is_valid || cache.is_learning) return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto catalog =
        pair_resource_catalog({kernels[0].description_, kernels[1].description_}, worker_counts);
    if (!catalog) return {.status = CUDA_ERROR_INVALID_VALUE};
    const bool can_share = first.can_share_configuration(second);
    const CudaPairProfileDomain domain =
        can_share ? CudaPairProfileDomain{CudaPairProfileDomainKind::kSameContext,
                                          {kernels[0].description_.execution_domain,
                                           kernels[1].description_.execution_domain}}
                  : CudaPairProfileDomain{CudaPairProfileDomainKind::kDistinctContexts, {1, 2}};
    auto choice = profiles.lookup(
        {kernels[0].description_.profile, kernels[1].description_.profile}, catalog->geometry,
        domain, catalog->limits, std::span{catalog->candidates}.first(catalog->count));
    // Host overlap between separate native contexts is not spatial sharing.
    if (!can_share) {
        const auto originals = choice.original;
        const auto first_original = choice.first_original;
        choice = {.first_original = first_original, .original = originals};
    }
    CudaSessionPairSelectionResult result;
    result.selection.kernels_ = std::move(kernels);
    result.selection.choice_ = choice;
    return result;
}

CudaSessionPairChoiceResult CudaExecutionSession::evaluate_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    const auto inspected =
        inspect_pair(first, first_kernel, second, second_kernel, profiles, worker_counts);
    return {inspected.status, inspected.selection.choice_};
}

CudaSessionPairSelectionResult CudaExecutionSession::select_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    auto inspected =
        inspect_pair(first, first_kernel, second, second_kernel, profiles, worker_counts);
    if (inspected.status != CUDA_SUCCESS) return inspected;
    return configure_pair({&first, &second}, std::move(inspected.selection.kernels_),
                          inspected.selection.choice_);
}

CudaSessionPairSelectionResult CudaExecutionSession::configure_pair(
    std::array<CudaExecutionSession*, 2> sessions, std::array<CudaSessionPairKernel, 2> kernels,
    CudaPairProfileChoice choice,
    const std::array<std::shared_ptr<CudaExecutionObserver>, 2>& observations) noexcept {
    if (choice.first_original > 1 || (choice.is_shared && !choice.candidate) ||
        (bool(observations[0]) != bool(observations[1])) || (observations[0] && !choice.is_shared))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto& pending = kernels[leg].description_.pending.command;
        // A cache miss cannot restart a partially executed original. Keep its
        // prepared family/counter/observer and resume only the selected serial leg.
        const bool uses_original = !choice.is_shared && !pending.is_resume;
        const auto workers = choice.is_shared ? choice.configuration.legs[leg].workers
                             : uses_original  ? 0
                                              : pending.physical_grid[0];
        std::shared_ptr<CudaExecutionObserver> observation;
        try {
            auto& state = *sessions[leg]->implementation_;
            const auto function = std::ranges::find(state.storage->functions, pending.function,
                                                    &Implementation::Function::id);
            if (function == state.storage->functions.end())
                return {.status = CUDA_ERROR_INVALID_HANDLE};
            observation = state.profile_observation(*function, kernels[leg].description_.geometry,
                                                    observations[leg]);
        } catch (...) {
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        }
        const auto selected = sessions[leg]->select_pending_kernel(
            kernels[leg].command_, {.workers = workers,
                                    .uses_original = uses_original,
                                    .observation = std::move(observation)});
        if (selected.status != CUDA_SUCCESS) return {.status = selected.status};
        const auto refreshed = sessions[leg]->refresh_pair_kernel(kernels[leg], selected.command);
        if (refreshed.status != CUDA_SUCCESS) return {.status = refreshed.status};
        kernels[leg] = refreshed.kernel;
    }
    CudaSessionPairSelectionResult result;
    result.selection.kernels_ = std::move(kernels);
    result.selection.choice_ = choice;
    result.selection.is_profile_sample_ = observations[0] != nullptr;
    return result;
}

CudaSessionPairSampleResult CudaExecutionSession::reserve_pair_sample(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    if (&first == &second) return {.status = CUDA_ERROR_INVALID_VALUE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    std::array<CudaSessionPairKernel, 2> kernels{first_kernel, second_kernel};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto fresh = sessions[leg]->refresh_pair_kernel(kernels[leg], kernels[leg].command_);
        if (fresh.status != CUDA_SUCCESS) return {.status = fresh.status};
        kernels[leg] = fresh.kernel;
        const auto& command = kernels[leg].description_.pending.command;
        if (command.is_resume || command.physical_generation || command.command_epoch != 1)
            return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto cache = profiles.snapshot();
    if (!cache.is_valid || !cache.is_learning) return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!first.can_share_configuration(second)) return {.status = CUDA_ERROR_NOT_READY};
    const auto catalog =
        pair_resource_catalog({kernels[0].description_, kernels[1].description_}, worker_counts);
    if (!catalog) return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!catalog->count) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const CudaPairProfileDomain domain{
        CudaPairProfileDomainKind::kSameContext,
        {kernels[0].description_.execution_domain, kernels[1].description_.execution_domain}};
    std::array<CudaPairProfileKernel, 2> bound{};
    for (std::size_t leg = 0; leg < bound.size(); ++leg) {
        if (!kernels[leg].description_.profile) return {.status = CUDA_ERROR_NOT_SUPPORTED};
        auto binding =
            profiles.bind_checked(kernels[leg].description_.profile, catalog->geometry[leg]);
        switch (binding.status) {
            case CudaPairProfileBindStatus::kBound:
                bound[leg] = std::move(binding.kernel);
                break;
            case CudaPairProfileBindStatus::kCapacity:
                return {.status = CUDA_ERROR_OUT_OF_MEMORY};
            case CudaPairProfileBindStatus::kMissingOriginal:
                return {.status = CUDA_ERROR_NOT_SUPPORTED};
            case CudaPairProfileBindStatus::kInvalid:
                return {.status = CUDA_ERROR_INVALID_VALUE};
        }
    }
    const auto candidates = std::span{catalog->candidates}.first(catalog->count);
    auto reserved = profiles.reserve_checked(bound, domain, catalog->limits, candidates);
    using Outcome = CudaPairProfileReserveStatus;
    if (reserved.status == Outcome::kBusy) return {.status = CUDA_ERROR_NOT_READY};
    if (reserved.status == Outcome::kCapacity) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    if (reserved.status == Outcome::kInvalid) return {.status = CUDA_ERROR_INVALID_VALUE};
    auto reservation = std::move(reserved.reservation);
    auto choice = profiles.choose(bound, domain, catalog->limits, candidates);
    if (reserved.status == Outcome::kReserved && reservation)
        choice = {.is_shared = true,
                  .candidate = reservation.candidate,
                  .configuration = reservation.configuration};
    else if (!choice.is_trained)
        return {.status = CUDA_ERROR_UNKNOWN};
    auto configured =
        configure_pair(sessions, std::move(kernels), choice, reservation.observations);
    if (configured.status != CUDA_SUCCESS) {
        for (const auto& observer : reservation.observations)
            if (observer) observer->finish(false);
        return {.status = configured.status};
    }
    return {CUDA_SUCCESS, std::move(configured.selection), std::move(reservation)};
}

CudaSessionPairIssueResult CudaExecutionSession::issue_pair(
    CudaExecutionSession& first, CudaExecutionSession& second,
    const CudaSessionPairSelection& selection, const core::ExecutionPairPlan& authorization,
    std::array<std::uint64_t, 2> policy_ids) noexcept {
    for (const auto* session : {&first, &second}) {
        if (session->implementation_->has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
        const auto& memory = session->implementation_->process_memory;
        if (memory && memory->native_health() != CUDA_SUCCESS)
            return {.status = memory->native_health()};
    }
    if (&first == &second || !selection || !policy_ids[0] || !policy_ids[1] ||
        policy_ids[0] == policy_ids[1] || !authorization.is_valid ||
        !authorization.priority.is_valid ||
        authorization.priority.size > authorization.priority.decisions.size())
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!authorization.grant_count) return {.status = CUDA_ERROR_NOT_READY};
    const auto proposed = selection.proposal(policy_ids);
    const auto count = proposed.is_shared ? 2U : 1U;
    if (authorization.grant_count != count || authorization.is_shared != proposed.is_shared ||
        proposed.first_serial > 1)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    for (std::size_t index = 0; index < count; ++index) {
        const auto leg = proposed.is_shared ? index : proposed.first_serial;
        if (authorization.grants[index] != proposed.legs[leg])
            return {.status = CUDA_ERROR_INVALID_HANDLE};
    }
    std::size_t executes{};
    for (const auto& decision :
         std::span{authorization.priority.decisions}.first(authorization.priority.size)) {
        if (decision.action == core::ExecutionPriorityAction::kExecuteProgress ||
            decision.action == core::ExecutionPriorityAction::kRequestRetreat ||
            decision.action == core::ExecutionPriorityAction::kQuarantined)
            return {.status = CUDA_ERROR_NOT_READY};
        if (decision.action != core::ExecutionPriorityAction::kExecute) continue;
        const auto grant = std::ranges::find(std::span{authorization.grants}.first(count),
                                             decision.session, &core::ExecutionPairLeg::session);
        if (grant == std::span{authorization.grants}.first(count).end() ||
            decision.command != grant->command ||
            std::ranges::count(
                std::span{authorization.priority.decisions}.first(authorization.priority.size),
                decision.session, &core::ExecutionPriorityDecision::session) != 1)
            return {.status = CUDA_ERROR_INVALID_HANDLE};
        ++executes;
    }
    if (executes != count) return {.status = CUDA_ERROR_INVALID_HANDLE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    // Validate BOTH opaque commands before any first-leg native effect. There
    // is no intake/poll/callback turn between these exact owner-serialized grants.
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto& kernel = selection.kernels_[leg];
        const auto current = sessions[leg]->refresh_pair_kernel(kernel, kernel.command_);
        if (current.status != CUDA_SUCCESS) return {.status = current.status};
    }
    if (proposed.is_shared && !first.can_share_configuration(second))
        return {.status = CUDA_ERROR_NOT_READY};
    CudaSessionPairIssueResult result;
    for (std::size_t index = 0; index < count; ++index) {
        const auto leg = proposed.is_shared ? index : proposed.first_serial;
        const auto& kernel = selection.kernels_[leg];
        const auto current = sessions[leg]->refresh_pair_kernel(kernel, kernel.command_);
        if (current.status != CUDA_SUCCESS) {
            result.status = current.status;
            return result;
        }
        if (proposed.is_shared && !first.can_share_configuration(second)) {
            result.status = CUDA_ERROR_NOT_READY;
            return result;
        }
        const auto before =
            sessions[leg]->persistent_progress(kernel.description_.pending.command.sequence);
        result.attempted[leg] = true;
        auto& issued = result.issues[leg];
        issued = sessions[leg]->issue(kernel.command_);
        if (issued.state != CudaExecutionIssueState::kIssued ||
            issued.native_status != CUDA_SUCCESS || issued.completion_status != CUDA_SUCCESS ||
            sessions[leg]->snapshot().is_quarantined) {
            result.status = issued.native_status != CUDA_SUCCESS       ? issued.native_status
                            : issued.completion_status != CUDA_SUCCESS ? issued.completion_status
                                                                       : CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (proposed.is_shared) {
            const auto after = sessions[leg]->persistent_progress(issued.sequence);
            if (!before || !after || before->physical_launches == UINT64_MAX ||
                after->physical_launches != before->physical_launches + 1 ||
                after->physical_stops >= after->physical_launches) {
                sessions[leg]->implementation_->fault();
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
        }
    }
    return result;
}

CudaSessionResult CudaExecutionSession::create_event(unsigned int flags) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return rejected(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return rejected(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (state.is_faulted || state.is_closing) return rejected(CUDA_ERROR_UNKNOWN);
    if (!state.has_id_space || state.storage->events.size() == state.limits.max_objects)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    const auto cleaned = state.collect_events();
    if (cleaned != CUDA_SUCCESS) return rejected(cleaned);
    std::shared_ptr<CudaSessionEvent::State> generation;
    auto result = state.make_event(flags, generation);
    if (result.status != CUDA_SUCCESS) return result;
    result.id = state.consume_id();
    generation->identity = {result.id, 0, 0};
    state.storage->events.push_back({result.id, flags, std::move(generation)});
    return result;
}
CudaSessionResult CudaExecutionSession::destroy_event(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return rejected(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return rejected(checked);
    const auto event = std::ranges::find(state.storage->events, id, &Implementation::Event::id);
    if (event == state.storage->events.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    state.storage->events.erase(event);  // Existing immutable borrows remain valid.
    const auto cleaned = state.collect_events();
    return cleaned == CUDA_SUCCESS ? settled() : rejected(cleaned);
}
CudaSessionResult CudaExecutionSession::record_event(std::uint64_t sequence, std::uint64_t event,
                                                     std::uint64_t stream,
                                                     unsigned int flags) noexcept {
    return implementation_
        ->submit_event({.sequence = sequence, .event = event, .stream = stream, .flags = flags},
                       false, false)
        .result;
}
CudaSessionResult CudaExecutionSession::wait_event(std::uint64_t sequence, std::uint64_t event,
                                                   std::uint64_t stream,
                                                   unsigned int flags) noexcept {
    return implementation_
        ->submit_event({.sequence = sequence, .event = event, .stream = stream, .flags = flags},
                       true, false)
        .result;
}
CudaSessionEventSubmission CudaExecutionSession::record_event_with_frontier(
    std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
    unsigned int flags) noexcept {
    return implementation_->submit_event(
        {.sequence = sequence, .event = event, .stream = stream, .flags = flags}, false, true);
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_with_frontier(
    std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
    unsigned int flags) noexcept {
    return implementation_->submit_event(
        {.sequence = sequence, .event = event, .stream = stream, .flags = flags}, true, true);
}
CudaSessionEventCapture CudaExecutionSession::capture_event(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt || state.is_closing)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .event = {}};
    if (state.is_faulted || !state.executor || state.executor->snapshot().is_quarantined)
        return {.status = CUDA_ERROR_UNKNOWN, .event = {}};
    const auto event = std::ranges::find(state.storage->events, id, &Implementation::Event::id);
    if (event == state.storage->events.end()) return {};
    if (state.storage->event_borrows.load(std::memory_order_acquire) >= state.limits.max_objects)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .event = {}};
    try {
        CudaSessionEvent result;
        result.state_ = event->current;
        result.owner_ = state.storage;
        result.thread_ = resource_owner;
        result.process_ = static_cast<std::uint64_t>(state.process);
        result.lifetime_ = std::make_shared<Implementation::EventLease>(state.storage);
        const auto generation = result.diagnostic_identity();
        return {CUDA_SUCCESS, std::move(result), event->current->frontier, generation};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .event = {}};
    }
}
CudaSessionResult CudaExecutionSession::wait_event(std::uint64_t sequence,
                                                   const CudaSessionEvent& source,
                                                   std::uint64_t stream,
                                                   unsigned int flags) noexcept {
    return wait_event_impl({.sequence = sequence, .stream = stream, .flags = flags}, source, false)
        .result;
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_with_frontier(
    std::uint64_t sequence, const CudaSessionEvent& source, std::uint64_t stream,
    unsigned int flags) noexcept {
    return wait_event_impl({.sequence = sequence, .stream = stream, .flags = flags}, source, true);
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_impl(EventCommand parameters,
                                                                 const CudaSessionEvent& source,
                                                                 bool captures_frontier) noexcept {
    const auto sequence = parameters.sequence;
    const auto stream = parameters.stream;
    const auto flags = parameters.flags;
    auto& state = *implementation_;
    const auto failure = [](CUresult status) {
        return CudaSessionEventSubmission{rejected(status), {}};
    };
    if (!state.owns()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return failure(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return failure(checked);
    if (!state.executor || state.is_faulted || state.is_closing) return failure(CUDA_ERROR_UNKNOWN);
    const auto dependency = source.observe();
    if (dependency != CUDA_SUCCESS && dependency != CUDA_ERROR_NOT_READY)
        return failure(dependency);
    if (flags) return failure(CUDA_ERROR_INVALID_VALUE);
    if (!state.dispatch.stream_wait_event) return failure(CUDA_ERROR_NOT_SUPPORTED);
    if (stream && !std::ranges::any_of(state.storage->streams,
                                       [stream](const auto& value) { return value.id == stream; }))
        return failure(CUDA_ERROR_INVALID_HANDLE);
    const auto cleaned = state.collect_events();
    if (cleaned != CUDA_SUCCESS) return failure(cleaned);
    if (state.executor->snapshot().outstanding >= state.limits.submissions.max_commands) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return {deferred, {}};
    }
    try {
        auto call = std::make_shared<Implementation::EventCall>();
        call->generation = source.state_;
        call->source = source;
        call->dispatch = state.dispatch;
        call->flags = flags;
        call->is_wait = true;
        CudaExecutionEvent command{stream, call, {}};
        if (source.owner_.lock() == state.storage)
            command.dependency = source.state_->frontier;
        else
            command.foreign_dependency = source.state_->frontier;
        auto submitted =
            captures_frontier
                ? state.executor->accept_event_with_frontier(sequence, command)
                : CudaExecutionEventAcceptance{state.executor->accept_event(sequence, command), {}};
        if (!submitted.acceptance.accepted()) return failure(CUDA_ERROR_OUT_OF_MEMORY);
        const auto generation = call->generation->identity;
        state.pending_events.push_back({submitted.acceptance.receipt, std::move(call)});
        auto result = settled();
        result.id = sequence;
        return {result, std::move(submitted.frontier), generation};
    } catch (...) {
        return failure(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionResult CudaExecutionSession::query_event(const CudaSessionEvent& event,
                                                    bool should_synchronize) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!event.state_ || event.owner_.lock() != state.storage)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    // An earlier record/preparation error is observable here, but does not
    // mean that this requested query/synchronize SDK operation was called.
    if (event.state_->record_status != CUDA_SUCCESS) return rejected(event.state_->record_status);
    if (event.state_->frontier) {
        const auto status = state.executor->query_frontier(*event.state_->frontier);
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (should_synchronize) {
        if (!state.dispatch.event_synchronize) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        return completed(state.invoke(state.dispatch.event_synchronize, event.state_->native));
    }
    return completed(state.invoke(state.dispatch.execution.event_query, event.state_->native));
}
CudaSessionResult CudaExecutionSession::event_elapsed_time(std::uint64_t start,
                                                           std::uint64_t end) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!state.dispatch.event_elapsed_time) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto first = capture_event(start);
    if (first.status != CUDA_SUCCESS) return rejected(first.status);
    const auto second = start == end ? first : capture_event(end);
    if (second.status != CUDA_SUCCESS) return rejected(second.status);
    for (const auto* event : {&first.event, &second.event}) {
        auto ready = query_event(*event);
        if (ready.status != CUDA_SUCCESS) {
            ready.original_status.reset();  // Preparatory observation, not elapsed-time issue.
            return ready;
        }
    }
    float milliseconds{};
    auto result = completed(state.invoke(state.dispatch.event_elapsed_time, &milliseconds,
                                         first.event.state_->native, second.event.state_->native));
    if (result.status == CUDA_SUCCESS) result.address = std::bit_cast<std::uint32_t>(milliseconds);
    return result;
}
CudaExecutionPollResult CudaExecutionSession::poll() noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
    if (!state.owns() || !state.executor) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    auto result = state.executor->poll();
    if (result.status == CUDA_SUCCESS) result.status = state.collect_events();
    if (state.uses_shared_domain && result.status != CUDA_SUCCESS &&
        result.status != CUDA_ERROR_NOT_READY)
        state.fault();
    state.update_domain_physical();
    if (state.uses_shared_domain && state.is_faulted && result.status == CUDA_SUCCESS)
        result.status = CUDA_ERROR_UNKNOWN;
    return result;
}
CUresult CudaExecutionSession::request_retreat(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return CUDA_ERROR_NOT_READY;
    const NativeExecutionScope active{state.is_native_issue_active};
    return state.executor->request_retreat(sequence);
}
CUresult CudaExecutionSession::retarget_workers(std::uint64_t sequence,
                                                CudaExecutionWorkerGeometry geometry) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return CUDA_ERROR_NOT_READY;
    if (!geometry.workers || geometry.workers > state.persistent.workers)
        return CUDA_ERROR_INVALID_VALUE;
    const NativeExecutionScope active{state.is_native_issue_active};
    return state.executor->retarget_workers(sequence, geometry);
}

CudaExecutionIssueResult CudaExecutionSession::resume(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->resume(sequence);
    state.update_domain_physical();
    return result;
}
std::optional<CudaSessionPersistentProgress> CudaExecutionSession::persistent_progress(
    std::uint64_t sequence) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return std::nullopt;
    for (const auto& entry : state.progress)
        if (const auto progress = entry.lock(); progress && progress->sequence == sequence)
            return progress->snapshot();
    return std::nullopt;
}
CudaExecutionPollResult CudaExecutionSession::drain_issued() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    auto result = state.executor->drain_issued();
    if (state.uses_shared_domain && result.status != CUDA_SUCCESS &&
        result.status != CUDA_ERROR_NOT_READY)
        state.fault();
    state.update_domain_physical();
    if (state.uses_shared_domain && state.is_faulted && result.status == CUDA_SUCCESS)
        result.status = CUDA_ERROR_UNKNOWN;
    return result;
}
std::optional<ExecutionHostReply> CudaExecutionSession::host_result(
    const ExecutionReceipt& receipt) const noexcept {
    const auto& state = *implementation_;
    if (getpid() != state.process || !state.executor) return std::nullopt;
    return state.executor->host_result(receipt);
}
std::optional<std::array<std::uint64_t, 2>> CudaExecutionSession::native_kernel_pair()
    const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return std::nullopt;
    return state.executor->native_kernel_pair();
}
CudaExecutionFrontierResult CudaExecutionSession::capture_frontier(
    std::optional<std::uint64_t> stream, bool includes_submission_dependencies) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .frontier = {}};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN, .frontier = {}};
    return state.executor->capture_frontier(stream, includes_submission_dependencies);
}
CudaExecutionFrontierResult CudaExecutionSession::capture_command_frontier(
    std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .frontier = {}};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN, .frontier = {}};
    return state.executor->capture_command_frontier(sequence);
}
CudaExecutionFrontierReservationResult CudaExecutionSession::reserve_command_frontier(
    std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->reserve_command_frontier(sequence);
}
CudaExecutionFrontierResult CudaExecutionSession::reserved_command_frontier(
    const CudaExecutionFrontierReservation& reservation) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status, .frontier = {}};
    return state.executor->reserved_command_frontier(reservation);
}
CUresult CudaExecutionSession::cancel_frontier_reservation(
    const CudaExecutionFrontierReservation& reservation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    return state.executor->cancel_frontier_reservation(reservation);
}
bool CudaExecutionSession::reject_command(const ExecutionReceipt& receipt,
                                          CUresult failure) noexcept {
    auto& state = *implementation_;
    if (state.external_transition_gate() != CUDA_SUCCESS || state.is_faulted) return false;
    return state.executor->reject_command(receipt, failure);
}
CUresult CudaExecutionSession::query_frontier(
    const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor) return CUDA_ERROR_INVALID_CONTEXT;
    return state.executor->query_frontier(frontier);
}
bool CudaExecutionSession::acknowledge_result(const ExecutionReceipt& receipt) noexcept {
    auto& state = *implementation_;
    return getpid() == state.process && state.executor &&
           state.executor->acknowledge_result(receipt);
}
CudaSessionSnapshot CudaExecutionSession::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) {
        CudaSessionSnapshot rejected;
        rejected.is_quarantined = true;
        return rejected;
    }
    CudaSessionSnapshot result;
    result.device_bytes = state.storage->device_bytes;
    result.image_bytes = state.storage->image_bytes;
    result.allocations = state.storage->allocations.size();
    result.modules = state.storage->modules.size();
    result.streams = state.storage->streams.size();
    result.functions = state.storage->functions.size();
    result.private_host_bytes = state.storage->host_bytes;
    result.default_proxy_staging_bytes = state.storage->staging_bytes;
    result.persistent_slots =
        static_cast<std::size_t>(std::ranges::count(state.storage->occupied, true));
    result.persistent_functions = static_cast<std::size_t>(std::ranges::count_if(
        state.storage->functions, [](const auto& function) { return !function.variants.empty(); }));
    result.persistent_accepted = state.persistent_accepted;
    result.host_memory_bytes = state.storage->host_memory_bytes;
    result.host_allocations = state.storage->host_allocations.size();
    result.blas_handles = state.storage->blas.size();
    result.lt_handles = state.storage->lt.size();
    result.random_handles = state.storage->random.size();
    result.external_context_leases = static_cast<std::size_t>(std::ranges::count_if(
        state.external_contexts, [](const auto& entry) { return !entry.expired(); }));
    result.external_command_leases = static_cast<std::size_t>(std::ranges::count_if(
        state.external_commands, [](const auto& entry) { return !entry.expired(); }));
#ifdef GLIMMER_HAS_CUDNN
    result.dnn_handles = state.storage->dnn.size();
#endif
    // Preserve resource observation before the dispatcher's readiness callbacks.
    static_cast<CudaSessionSchedulingSnapshot&>(result) = scheduling_snapshot();
    return result;
}
CudaSessionHealthSnapshot CudaExecutionSession::health_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    return {.has_context = state.storage->context != nullptr,
            .is_quarantined = state.is_faulted || state.is_creation_unproven ||
                              (state.executor && state.executor->is_quarantined())};
}
CudaSessionSchedulingSnapshot CudaExecutionSession::scheduling_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    CudaSessionSchedulingSnapshot result{
        .has_context = state.storage->context != nullptr,
        .is_quarantined = state.is_faulted || state.is_creation_unproven};
    if (state.executor) {
        const auto scheduling = state.executor->scheduling_snapshot();
        result.submissions = scheduling.submissions;
        result.external_barriers = scheduling.external_barriers;
        result.enqueue_turns = scheduling.enqueue_turns;
        result.external_work_in_flight = scheduling.external_work_in_flight;
        result.has_ready_work = scheduling.has_ready_work;
        result.is_quarantined = result.is_quarantined || result.submissions.is_quarantined;
        if (scheduling.has_pending_preparation && !state.has_destroy_attempt &&
            !state.is_native_issue_active && !state.is_folded_active &&
            !state.is_rnn_query_active && !state.is_preparation_active && !state.is_faulted) {
            result.has_pending_preparation =
                !state.storage->domain_member ||
                state.storage->domain_member.view(true).status == CUDA_SUCCESS;
        }
    }
    return result;
}
void CudaExecutionSession::close() noexcept {
    auto& state = *implementation_;
    if (state.owns() && !state.is_native_issue_active) {
        state.is_closing = true;
        if (state.executor) state.executor->close();
    }
}
bool CudaExecutionSession::can_cleanup_quarantined() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || !state.executor || state.is_faulted ||
        state.is_creation_unproven || state.has_destroy_attempt || state.is_native_issue_active ||
        state.is_folded_active || state.is_rnn_query_active || state.has_external_contexts() ||
        state.storage->event_borrows.load(std::memory_order_acquire))
        return false;
    const auto queue = state.executor->snapshot();
    if (!queue.is_quarantined || !state.executor->physical_readers_quiescent()) return false;
    const auto known = [](const auto& handle) {
        return handle.native && !handle.is_uncertain && !handle.has_destroy_attempt;
    };
    if (!std::ranges::all_of(state.storage->blas, known) ||
        !std::ranges::all_of(state.storage->lt, known) ||
        !std::ranges::all_of(state.storage->random, known))
        return false;
#ifdef GLIMMER_HAS_CUDNN
    if (!std::ranges::all_of(state.storage->dnn, known)) return false;
#endif
    return true;
}
std::size_t CudaExecutionSession::reject_queued(CUresult failure) noexcept {
    auto& state = *implementation_;
    return state.owns() && state.executor && !state.is_native_issue_active &&
                   !state.is_folded_active && !state.is_rnn_query_active
               ? state.executor->reject_queued(failure)
               : 0;
}
CudaSessionResult CudaExecutionSession::shutdown() noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    state.is_closing = true;
    if (!state.storage->context) {
        if (state.is_creation_unproven || (state.uses_shared_domain && state.is_faulted))
            return rejected(CUDA_ERROR_UNKNOWN);
        if (state.storage->domain_member) {
            const auto member = state.storage->domain_member.view();
            if (member.status != CUDA_SUCCESS) return rejected(member.status);
            if (member.is_attached && state.storage->domain_member.retire() != CUDA_SUCCESS)
                return rejected(CUDA_ERROR_UNKNOWN);
            state.storage->domain_member = {};
        }
        return settled();
    }
    if (state.process_memory) {
        const auto health = state.process_memory->native_health();
        if (health != CUDA_SUCCESS) return rejected(health);
    }
    if (state.has_destroy_attempt) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.uses_shared_domain && state.is_faulted) return rejected(CUDA_ERROR_UNKNOWN);
    if (state.graphs) {
        const auto status = state.graphs->shutdown();
        if (status == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (state.has_external_contexts()) return deferred_resource_release();
    if (state.storage->event_borrows.load(std::memory_order_acquire)) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    if (state.executor) {
        const auto status = state.check_context();
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (state.executor && !state.executor->shutdown()) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    state.update_domain_physical();
    if (state.uses_shared_domain) {
        const auto quiescent = state.domain_cold_status();
        if (quiescent == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (quiescent != CUDA_SUCCESS) return rejected(quiescent);
    }
    for (auto& allocation : state.storage->host_allocations) allocation.is_retiring = true;
    while (!state.storage->host_allocations.empty()) {
        auto released = state.release_host_at(0);
        if (released.status != CUDA_SUCCESS) return released;
    }
#ifdef GLIMMER_HAS_CUDNN
    while (!state.storage->dnn.empty()) {
        auto& handle = state.storage->dnn.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.dnn->destroy, handle.native);
        if (status != CUDNN_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->dnn.pop_back();
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
    }
#endif
    while (!state.storage->random.empty()) {
        auto& handle = state.storage->random.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.random.destroy, handle.native);
        if (status != CURAND_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error("Owned cuRAND shutdown failed: context={}, handle={}, library_status={}",
                          state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->random.pop_back();
    }
    while (!state.storage->blas.empty()) {
        auto& handle = state.storage->blas.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.blas.destroy, handle.native);
        if (status != CUBLAS_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error("Owned cuBLAS shutdown failed: context={}, handle={}, library_status={}",
                          state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->blas.pop_back();
    }
    while (!state.storage->lt.empty()) {
        auto& handle = state.storage->lt.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.lt.destroy, handle.native);
        if (status != CUBLAS_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error(
                "Owned cuBLASLt shutdown failed: context={}, handle={}, library_status={}",
                state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->lt.pop_back();
    }
    state.executor.reset();
    const auto cleanup = state.release_progress();
    if (cleanup != CUDA_SUCCESS) {
        state.fault();
        return rejected(cleanup);
    }
    if (state.storage->default_proxy) {
        if (state.storage->has_default_proxy_destroy_attempt ||
            state.storage->is_default_proxy_uncertain)
            return rejected(CUDA_ERROR_UNKNOWN);
        const NativeExecutionScope scope(state.is_native_issue_active);
        auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) return rejected(status);
        state.storage->has_default_proxy_destroy_attempt = true;
        status = state.invoke(state.dispatch.stream_destroy, state.storage->default_proxy);
        const auto proof = state.check_context(true);
        if (status != CUDA_SUCCESS || proof != CUDA_SUCCESS) {
            state.storage->is_default_proxy_uncertain = true;
            state.fault();
            return rejected(proof != CUDA_SUCCESS ? proof : status);
        }
        state.storage->default_proxy = nullptr;
    }
    if (state.storage->domain_member) {
        const auto released = state.release_shared_resources();
        if (released != CUDA_SUCCESS) return rejected(released);
        const auto detached = state.invoke(state.dispatch.context_set_current, nullptr);
        if (detached != CUDA_SUCCESS || state.is_faulted) {
            state.fault();
            return rejected(detached == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : detached);
        }
        const auto retired = state.storage->domain_member.retire();
        if (retired != CUDA_SUCCESS) {
            state.fault();
            return rejected(retired);
        }
        state.storage->context = nullptr;
        state.storage->domain_member = {};
        return settled();
    }
    state.has_destroy_attempt = true;
    auto result = completed(state.invoke(state.dispatch.context_destroy, state.storage->context));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->context = nullptr;
    state.storage->allocations.clear();
    state.storage->modules.clear();
    state.storage->streams.clear();
    state.storage->events.clear();
    state.pending_events.clear();
    for (auto& generation : state.storage->event_generations) generation->native = nullptr;
    state.storage->event_generations.clear();
    state.storage->functions.clear();
    if (state.process_memory && state.storage->device_bytes &&
        state.process_memory->owner()->release_context_bytes(state.storage->device_bytes) !=
            CUDA_SUCCESS) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
    state.storage->device_bytes = 0;
    state.storage->gemm_static_bytes = 0;
    state.storage->gemm_module = nullptr;
    state.storage->image_bytes = 0;
    state.storage->counters = 0;
    state.storage->counter_bytes = 0;
    // Successful owned-context destruction, after checked host cleanup, settles
    // prior GPU uncertainty. This closed, started session still cannot reopen.
    state.is_faulted = false;
    return result;
}

}  // namespace glimmer::backend::cuda
