#pragma once
#include "cuda_process_memory_backend.h"
#include "cuda_execution_graph.h"

#include "glimmer/transform/ptx_transform.h"
#include "glimmer/core/execution_priority.h"

#include "cuda_execution_dispatcher.h"
#include "cuda_execution_domain.h"
#include "cuda_lt_registry.h"
#include "cuda_pair_profile_cache.h"
#include "cuda_module_jit.h"

#include <cublas_v2.h>
#include <curand.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace glimmer::backend::cuda {
class CudaPtxCache;
class CudaProfileCache;
class CudaExternalExecutionBridge;
class CudaRetirementObserver;
struct CudaDnnDispatch;
struct CudaDnnResult;
class CudaDnnRegistry;
struct CudaDnnFoldedDescriptors;
struct CudaDnnRnnQuery;
struct CudaDnnRnnQueryResult;
struct CudaDnnRnnBuildResult;
struct CudaDnnRnnWeightParamsResult;
struct CudaDnnAttnWeights;
struct CudaDnnAttnWeightsResult;
struct CudaSessionDnnRnnWeightParams;
struct CudaSessionDnnRnn;
struct CudaDnnRnnAcceptance;
struct CudaDnnLegacyControl;
struct CudaDnnLegacyResult;
struct CudaSessionDnnLegacyRnn;
struct CudaSessionDnnAttn;
struct CudaDnnAttnPreparation;
struct CudaDnnAttnAcceptance;
struct CudaDnnBackendContextRequest;
struct CudaDnnBackendContextResult;
struct CudaSessionDnnBackendExecute;
struct CudaDnnBackendExecutePreparation;
struct CudaDnnBackendExecuteAcceptance;
struct CudaSessionDnnCompute;
struct CudaDnnComputeAcceptance;
struct CudaSessionDnnConvolution;
struct CudaDnnConvolutionQueryResult;
struct CudaDnnFindAcceptance;
enum class CudaDnnConvolutionQueryKind : std::uint32_t;
enum class CudaDnnBackwardConvolutionQueryKind : std::uint32_t;
struct CudaSessionDnnReduction;
struct CudaSessionDnnBatchNorm;
struct CudaSessionDnnDropout;
struct CudaSessionDnnReorder;
struct CudaDnnDropoutView;
enum class CudaDnnReductionQueryKind : std::uint32_t;

// The installed C++ SDK also has a legacy inline overload; retain the exact
// modern C ABI explicitly instead of using an ambiguous decltype expression.
using CudaGemmExFunction = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasOperation_t,
                                                         cublasOperation_t, int, int, int,
                                                         const void*, const void*, cudaDataType,
                                                         int, const void*, cudaDataType, int,
                                                         const void*, void*, cudaDataType, int,
                                                         cublasComputeType_t, cublasGemmAlgo_t);
using CudaStridedGemmExFunction = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType, int, long long, const void*, cudaDataType, int, long long, const void*, void*,
    cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
[[nodiscard]] std::size_t cuda_gemm_scalar_bytes(cublasComputeType_t compute,
                                                 cudaDataType output) noexcept;

struct CudaBlasDispatch {
    decltype(&cublasCreate_v2) create{};
    decltype(&cublasDestroy_v2) destroy{};
    decltype(&cublasSetStream_v2) set_stream{};
    decltype(&cublasSgemm_v2) sgemm{};
    decltype(&cublasGetStream_v2) get_stream{};
    decltype(&cublasSetPointerMode_v2) set_pointer_mode{};
    decltype(&cublasGetPointerMode_v2) get_pointer_mode{};
    decltype(&cublasSetMathMode) set_math_mode{};
    decltype(&cublasGetMathMode) get_math_mode{};
    decltype(&cublasGetVersion_v2) get_version{};
    CudaGemmExFunction gemm_ex{};
    CudaStridedGemmExFunction strided_gemm_ex{};
    decltype(&cublasSetWorkspace_v2) set_workspace{};
    decltype(&cublasSetAtomicsMode) set_atomics_mode{};
    decltype(&cublasGetAtomicsMode) get_atomics_mode{};
    decltype(&cublasSetSmCountTarget) set_sm_count_target{};
    decltype(&cublasGetSmCountTarget) get_sm_count_target{};
    decltype(&cublasSgemmStridedBatched) strided_sgemm{};
    decltype(&cublasSgeam) sgeam{};
    decltype(&cublasDgeam) dgeam{};
    decltype(&cublasSgemv_v2) sgemv{};
    decltype(&cublasSgemmEx) sgemm_ex{};
};
enum class CudaBlasAttribute : std::uint8_t {
    kStream,
    kPointerMode,
    kMathMode,
    kVersion,
    kAtomicsMode,
    kSmCountTarget
};

struct CudaRandDispatch {
    decltype(&curandCreateGenerator) create{};
    decltype(&curandSetPseudoRandomGeneratorSeed) set_seed{};
    decltype(&curandDestroyGenerator) destroy{};
};

struct CudaContextDispatch {
    decltype(&cuCtxGetLimit) get_limit{};
    decltype(&cuCtxSetLimit) set_limit{};
    decltype(&cuCtxGetCacheConfig) get_cache{};
    decltype(&cuCtxSetCacheConfig) set_cache{};
    CUresult(CUDAAPI* get_shared)(CUsharedconfig*) {};
    CUresult(CUDAAPI* set_shared)(CUsharedconfig) {};
    decltype(&cuCtxResetPersistingL2Cache) reset_cache{};
    decltype(&cuCtxGetApiVersion) api_version{};
    decltype(&cuCtxGetExecAffinity) affinity{};
    decltype(&cuCtxGetStreamPriorityRange) priority_range{};
    struct Runtime {
        decltype(&cudaDeviceGetLimit) get_limit{};
        decltype(&cudaDeviceSetLimit) set_limit{};
        decltype(&cudaDeviceGetCacheConfig) get_cache{};
        decltype(&cudaDeviceSetCacheConfig) set_cache{};
        cudaError_t(CUDARTAPI* get_shared)(cudaSharedMemConfig*) {};
        cudaError_t(CUDARTAPI* set_shared)(cudaSharedMemConfig) {};
        decltype(&cudaCtxResetPersistingL2Cache) reset_cache{};
    } runtime{};
};

struct CudaSessionDispatch {
    CudaExecutionDispatch execution;
    decltype(&cuInit) initialize{};
    decltype(&cuDeviceGet) device_get{};
    decltype(&cuCtxCreate) context_create{};
    decltype(&cuCtxSetCurrent) context_set_current{};
    decltype(&cuCtxDestroy) context_destroy{};
    decltype(&cuModuleLoadData) module_load{};
    decltype(&cuModuleUnload) module_unload{};
    decltype(&cuModuleGetFunction) function_get{};
    decltype(&cuFuncGetParamInfo) parameter_info{};
    decltype(&cuMemAlloc) allocate{};
    decltype(&cuMemFree) release{};
    decltype(&cuMemcpyHtoD) upload{};
    decltype(&cuMemcpyDtoH) download{};
    decltype(&cuStreamCreate) stream_create{};
    decltype(&cuStreamDestroy) stream_destroy{};
    decltype(&cuMemHostAlloc) host_allocate{};
    decltype(&cuMemFreeHost) host_release{};
    decltype(&cuMemHostGetDevicePointer) host_map{};
    decltype(&cuMemcpyHtoDAsync) upload_async{};
    decltype(&cuMemcpyDtoHAsync) download_async{};
    decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor) occupancy{};
    decltype(&cuDeviceGetAttribute) device_attribute{};
    decltype(&cuCtxGetFlags) context_get_flags{};
    decltype(&cuCtxSetFlags) context_set_flags{};
    decltype(&cuFuncSetAttribute) function_set_attribute{};
    decltype(&cuMemHostRegister) host_register{};
    decltype(&cuMemHostUnregister) host_unregister{};
    CudaBlasDispatch blas{};
    CudaLtDispatch lt{};
    struct RuntimeMetadata {
        decltype(&cudaRuntimeGetVersion) version{};
        decltype(&cudaGetDeviceProperties) properties{};
        decltype(&cudaDeviceGetAttribute) attribute{};
        decltype(&cudaDeviceGetP2PAttribute) peer_attribute{};
        decltype(&cudaDeviceGetStreamPriorityRange) stream_priority_range{};
        decltype(&cudaDeviceCanAccessPeer) peer_access{};
        decltype(&cudaMemGetInfo) memory_info{};
        decltype(&cudaPointerGetAttributes) pointer_attributes{};
    } runtime{};
    decltype(&cuStreamCreateWithPriority) stream_create_priority{};
    decltype(&cuStreamGetPriority) stream_get_priority{};
    decltype(&cuModuleGetGlobal) module_global{};
    decltype(&cuEventRecordWithFlags) event_record_flags{};
    decltype(&cuStreamWaitEvent) stream_wait_event{};
    decltype(&cuEventElapsedTime) event_elapsed_time{};
    decltype(&cuEventSynchronize) event_synchronize{};
    decltype(&cuDeviceGetUuid) device_uuid{};
    decltype(&cuDriverGetVersion) driver_version{};
    CudaRandDispatch random{};
    CudaContextDispatch context{};
    std::shared_ptr<const CudaDnnDispatch> dnn{};
    // The composition owner verifies the actual SDK before publishing dispatch.
    // Zero means no such proof. This is not user-controlled wire metadata.
    std::uint32_t verified_dnn_header_version{};
    // Optional original context-wide proof for opaque native benchmark work.
    decltype(&cuCtxSynchronize) context_synchronize{};
    decltype(&cuFuncGetAttribute) function_get_attribute{};
    decltype(&cuFuncSetCacheConfig) function_set_cache{};
    decltype(&cuPointerGetAttribute) pointer_get_attribute{};
    decltype(&cuModuleLoadDataEx) module_load_ex{};
    CudaProcessMemoryDispatch process_memory{};
    decltype(&cuLaunchHostFunc) launch_host_function{};
    CudaSessionGraphDispatch graph{};
};

// Service-owned callback data. The native callback has no SDK or application
// access; exact transport Finish (or proven peer death) releases only this wait.
class CudaSessionHostFunction final {
   public:
    CudaSessionHostFunction();
    ~CudaSessionHostFunction();
    [[nodiscard]] bool entered() const noexcept;
    void finish() noexcept;

   private:
    friend class CudaExecutionSession;
    struct State;
    std::unique_ptr<State> state_;
    static void CUDA_CB wait(void* data) noexcept;
};

struct CudaSessionEventIdentity {
    std::uint64_t event_object{}, record_sequence{}, frontier_capture{};
    bool operator==(const CudaSessionEventIdentity&) const = default;
};

// Immutable application-event generation. CPU-only copying/destruction; native
// resources remain with the context owner through every accepted command lease.
class CudaSessionEvent final {
   public:
    CudaSessionEvent() = default;
    [[nodiscard]] CUresult observe() const noexcept;
    // CPU-only owner/PID/thread/reentry-checked diagnostic correlation. Sequence
    // zero denotes the initial never-recorded generation, not completion proof.
    [[nodiscard]] std::optional<CudaSessionEventIdentity> diagnostic_identity() const noexcept;

   private:
    friend class CudaExecutionSession;
    struct State;
    std::shared_ptr<State> state_;
    std::weak_ptr<const void> owner_;
    std::weak_ptr<const void> thread_;
    std::shared_ptr<const void> lifetime_;
    std::uint64_t process_{};
};
struct CudaSessionEventCapture {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaSessionEvent event;
    // Exact recorded generation, absent for a never-recorded event. Capturing
    // it never recaptures a stream or includes a later event rerecord.
    std::optional<CudaExecutionFrontier> dependency{};
    std::optional<CudaSessionEventIdentity> generation{};
};

struct CudaSessionLimits {
    std::size_t max_objects{256};
    // Charged owned raw/parsed/transformed bytes, allocated only on intake.
    std::size_t max_image_bytes{std::size_t{2} * 1024 * 1024 * 1024};
    std::size_t max_device_bytes{std::size_t{1024} * 1024 * 1024};
    std::size_t max_parameters{1024};
    std::uint64_t first_resource_id{1};
    ExecutionSubmissionLimits submissions;
    std::size_t max_private_host_bytes{std::size_t{1024} * 1024};
    std::size_t max_host_memory_bytes{std::size_t{64} * 1024 * 1024};
};

struct CudaSessionPersistentConfig {
    // Zero preserves native mode. Nonzero explicitly enables framework-owned
    // preparation; unsupported inputs remain native before admission.
    unsigned int workers{};
    // Trusted owner selection only; never enabled by application arguments.
    // Requires the optional immutable CUTLASS image and nonzero workers.
    bool replace_sgemm{};
    transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
    // Read-only measured sharing policy; a missing qualified choice is original.
    bool uses_workload_agnostic{};
    unsigned int sharing_threads_per_sm{1024}, sharing_minimum_speed_percent{90};
    // Default-off diagnostic code; excludes measured/learning profiles.
    // Private GEMM has its own observed entries. Never grants physical stop.
    bool observes_block_completion{};
    // Trusted provider choice, fixed for the whole Session and all resumes.
    // Nondefault geometry requires private GEMM replacement, not native work.
    unsigned int gemm_tile_side{64};
    // Experimental mapped-host GEMM layout; scalar and generic paths stay unchanged.
    bool uses_sharded_host_retreat{};
};
struct CudaSessionPersistentProgress {
    std::uint64_t completed_blocks{}, total_blocks{}, physical_launches{}, physical_stops{};
    unsigned int workers{};
    transform::PtxExecutionKind kind{transform::PtxExecutionKind::kPreemptive};
    bool observes_block_completion{}, has_completed_block{};
};
struct CudaSessionPendingDescription {
    CudaExecutionPendingDescription command;
    std::uint64_t control_epoch{}, member_generation{}, configuration_epoch{},
        physical_configuration_epoch{};
    bool operator==(const CudaSessionPendingDescription&) const = default;
};
class CudaSessionPendingCommand final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(command_) && !owner_.expired();
    }
    [[nodiscard]] CudaSessionPendingDescription description() const noexcept {
        return description_;
    }

   private:
    friend class CudaExecutionSession;
    CudaExecutionPendingCommand command_;
    CudaSessionPendingDescription description_;
    std::weak_ptr<const void> owner_;
};
struct CudaSessionPendingResult {
    CUresult status{CUDA_SUCCESS};
    CudaSessionPendingCommand command{};
};
enum class CudaSessionKernelPreparationDisposition : std::uint8_t {
    kFailed,
    kPrepared,
    kDeferred,
    kRejected,
    kUncertain,
};
struct CudaSessionKernelPreparationResult {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    CudaSessionKernelPreparationDisposition disposition{
        CudaSessionKernelPreparationDisposition::kFailed};
    CudaSessionPendingCommand command{};
};
struct CudaSessionPairKernelDescription {
    CudaSessionPendingDescription pending;
    CudaProfileFunction profile;
    CudaProfileGeometry geometry;
    CudaPairProfileLimits limits;
    CudaPairProfileVariant variant;
    std::uint64_t catalog_generation{}, execution_domain{};
    bool is_configuration_materialized{};
};
// Prepared metadata is not native execution authority. It retains only code
// identity and weak command/Session observations, not native resources. A
// catalog describes the actual retained family; partial work cannot change it.
class CudaSessionPairKernel final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(command_) && !owner_.expired();
    }
    [[nodiscard]] CudaSessionPairKernelDescription description() const noexcept {
        return description_;
    }

   private:
    friend class CudaExecutionSession;
    CudaSessionPendingCommand command_;
    CudaSessionPairKernelDescription description_;
    std::weak_ptr<const void> owner_;
};
enum class CudaSessionPairKernelDisposition : std::uint8_t {
    kFailed,
    kPrepared,
    kUnsupported,
};
struct CudaSessionPairKernelResult {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    CudaSessionPairKernel kernel{};
    // Only proven structural exclusions are skippable during an ordinary
    // readonly search. A native NOT_SUPPORTED result remains kFailed.
    CudaSessionPairKernelDisposition disposition{CudaSessionPairKernelDisposition::kFailed};
};
class CudaSessionPairSelection final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(kernels_[0]) && static_cast<bool>(kernels_[1]);
    }
    [[nodiscard]] CudaPairProfileChoice choice() const noexcept {
        return choice_;
    }
    [[nodiscard]] bool is_profile_sample() const noexcept {
        return is_profile_sample_;
    }
    [[nodiscard]] std::array<CudaSessionPairKernelDescription, 2> kernels() const noexcept {
        return {kernels_[0].description(), kernels_[1].description()};
    }
    [[nodiscard]] core::ExecutionPairProposal proposal(
        std::array<std::uint64_t, 2> policy_ids) const noexcept;
    [[nodiscard]] std::array<core::ExecutionPairContext, 2> contexts() const noexcept;

   private:
    friend class CudaExecutionSession;
    std::array<CudaSessionPairKernel, 2> kernels_{};
    CudaPairProfileChoice choice_{};
    bool is_profile_sample_{};
};
struct CudaSessionPairSelectionResult {
    CUresult status{CUDA_SUCCESS};
    CudaSessionPairSelection selection{};
};
struct CudaSessionPairChoiceResult {
    CUresult status{CUDA_SUCCESS};
    CudaPairProfileChoice choice{};
};
struct CudaSessionPairSampleResult {
    CUresult status{CUDA_SUCCESS};
    CudaSessionPairSelection selection{};
    CudaPairProfileReservation reservation{};
};
struct CudaSessionPairIssueResult {
    CUresult status{CUDA_SUCCESS};
    // Per-leg method results, never physical-stop/rollback proof. Any partial
    // native attempt remains independently owned by its original Session.
    std::array<CudaExecutionIssueResult, 2> issues{};
    std::array<bool, 2> attempted{};
};
struct CudaSessionContextConfig {
    unsigned int flags{};
};

enum class CudaPreparationPhase : std::uint8_t {
    kInspectImage,
    kStaticStorage,
    kTransformPtx,
    kLoadOriginal,
    kLoadPrepared,
    kValidateFunction
};
// Trusted, non-owning owner-thread observation. The callback and context must
// outlive their installation. Callbacks are synchronous, may nest across phases,
// and must not invoke the SDK, reenter the session or replace the observer.
// Logging callbacks must contain their own failures without changing the call.
// Scope plumbing adds no allocation or clock read; phase completion is not
// operation success or physical completion. Disabled by default.
struct CudaPreparationObserver {
    void* context{};
    void (*callback)(void*, CudaPreparationPhase, bool is_begin) noexcept {};
};

struct CudaArgumentLayout {
    std::size_t offset{};
    std::size_t bytes{};
};
struct CudaAllocationRange {
    std::uint64_t allocation{};
    std::size_t offset{};
};
struct CudaSessionMemory {
    CudaMemoryKind kind{CudaMemoryKind::kCopy};
    CudaAllocationRange destination{}, source{};
    std::size_t count{};
    std::uint32_t value{};
    std::uint64_t stream{};
    bool is_asynchronous{};
};
struct CudaSessionSgemm {
    std::uint64_t handle{};
    cublasOperation_t transpose_a{CUBLAS_OP_N}, transpose_b{CUBLAS_OP_N};
    int rows{}, columns{}, depth{}, leading_a{}, leading_b{}, leading_c{};
    CudaAllocationRange a{}, b{}, c{};
    float alpha{1.0F}, beta{};
    bool uses_device_scalars{};
    CudaAllocationRange device_alpha{}, device_beta{};
};
struct CudaSessionGemmEx {
    CudaSessionSgemm operation;
    cudaDataType a_type{CUDA_R_32F}, b_type{CUDA_R_32F}, c_type{CUDA_R_32F};
    cublasComputeType_t compute_type{CUBLAS_COMPUTE_32F};
    cublasGemmAlgo_t algorithm{CUBLAS_GEMM_DEFAULT};
    // Raw original bytes, not a float conversion. Native scalar pointers must
    // satisfy SDK alignment even for complex-double coefficients.
    alignas(16) std::array<std::byte, 16> alpha{}, beta{};
};
struct CudaSessionGeam {
    cudaDataType type{CUDA_R_32F};
    std::uint32_t transpose_a{}, transpose_b{}, pointer_mode{}, presence{};
    int rows{}, columns{}, leading_a{}, leading_b{}, leading_c{};
    std::uint64_t handle{};
    // A, B, C, alpha, beta. Presence bits are alpha, beta, A, B, C.
    // A present zero range is unresolved, not native pointer authority.
    std::array<CudaAllocationRange, 5> buffers{};
    alignas(16) std::array<std::byte, 16> alpha{}, beta{};
};
struct CudaSessionSgemv {
    std::uint32_t transpose{};
    int rows{}, columns{}, leading_a{}, increment_x{}, increment_y{};
    std::uint64_t handle{};
    std::uint32_t pointer_mode{}, presence{};
    // A, x, y, alpha, beta; presence bits are alpha, beta, A, x, y.
    // Negative increments retain the original physical BLAS base address.
    std::array<CudaAllocationRange, 5> buffers{};
    alignas(16) std::array<std::byte, 16> alpha{}, beta{};
};
struct CudaSessionSgemmEx {
    std::uint32_t transpose_a{}, transpose_b{}, type_a{}, type_b{}, type_c{};
    int rows{}, columns{}, depth{}, leading_a{}, leading_b{}, leading_c{};
    std::uint64_t handle{};
    std::uint32_t pointer_mode{}, presence{};
    // A, B, C, alpha, beta. Scalars always use the exact FP32 SDK ABI;
    // this is not a GemmEx request or permission to select private compute.
    std::array<CudaAllocationRange, 5> buffers{};
    alignas(16) std::array<std::byte, 16> alpha{}, beta{};
};
struct CudaSessionStridedGemmEx {
    CudaSessionGemmEx operation;
    // Original signed element strides; zero permits read-only input broadcast.
    std::array<std::int64_t, 3> strides{};
    int batches{};
};
struct CudaHostMemoryView {
    void* pointer{};
    CUdeviceptr address{};
    std::size_t bytes{};
    // CPU-only lease; the native owner separately unregisters after all users
    // have retired. Neither lease destruction nor socket close performs CUDA.
    std::shared_ptr<const void> lifetime;
    // Private pageable backing has no registered device alias. Its original
    // host address must remain pageable at the generic Driver copy boundary.
    bool is_pageable{};
};
enum class CudaUvaEndpointKind : std::uint8_t { kNull, kDevice, kHost };
struct CudaSessionUvaEndpoint {
    CudaUvaEndpointKind kind{CudaUvaEndpointKind::kNull};
    CudaAllocationRange device{};
    // Trusted authenticated HostOwner borrow, never a wire-supplied pointer.
    CudaHostMemoryView host{};
};
struct CudaSessionUvaCopy {
    CudaSessionUvaEndpoint destination{}, source{};
    std::size_t bytes{};
    std::uint64_t stream{};
    bool is_asynchronous{};
};
struct CudaDeviceMemoryView {
    CUdeviceptr address{};
    std::size_t bytes{};
    // CPU-only backing reference for trusted preparation. Only an accepted
    // command and its actual GPU event can establish a physical device lease.
    std::shared_ptr<const void> lifetime;
};
// Trusted immutable resource retention, not command admission or physical-stop
// proof. Only the original session owner captures these capabilities. Copies,
// observations and destruction may run on any thread in the same process and
// never invoke CUDA; mutable session/frontier access is not transferred.
class CudaSessionExternalContext final {
   public:
    CudaSessionExternalContext() = default;
    [[nodiscard]] CUcontext native() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] int device_ordinal() const noexcept;

   private:
    friend class CudaExecutionSession;
    struct State;
    std::shared_ptr<const State> state_;
};
struct CudaSessionExternalRangeRequest {
    CudaAllocationRange range{};
    std::size_t bytes{};
};
struct CudaSessionExternalRange {
    CudaAllocationRange range{};
    CUdeviceptr address{};
    std::size_t bytes{};
};
class CudaSessionExternalCommand final {
   public:
    CudaSessionExternalCommand() = default;
    [[nodiscard]] CudaSessionExternalContext context() const noexcept;
    [[nodiscard]] CUstream stream() const noexcept;
    [[nodiscard]] std::uint64_t stream_id() const noexcept;
    [[nodiscard]] std::span<const CudaSessionExternalRange> ranges() const noexcept;

   private:
    friend class CudaExecutionSession;
    struct State;
    std::shared_ptr<const State> state_;
};
struct CudaSessionExternalContextCapture {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    CudaSessionExternalContext context{};
};
struct CudaSessionExternalCommandCapture {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    CudaSessionExternalCommand command{};
};
struct CudaSessionStreamConfig {
    // Cold opt-in; public stream zero remains zero while native execution uses
    // one source-owned nonblocking proxy with explicit legacy dependencies.
    bool uses_default_proxy{};
    std::size_t maximum_staging_bytes{std::size_t{64} * 1024 * 1024};
    // Trusted cold original-only policy, not a profile/transform cache key.
    // Hidden native priorities never replace the
    // application's source-visible stream priority metadata.
    std::optional<std::uint32_t> scheduling_priority{};
};
enum class CudaSessionDisposition : std::uint8_t { kComplete, kRejected, kDeferred };
enum class CudaPointerDomain : std::uint32_t { kDriver = 1, kRuntime = 2 };
enum class CudaPointerKind : std::uint32_t {
    kNull = 0,
    kUnregistered = 1,
    kDevice = 2,
    kHost = 3,
    kProcessDevice = 4
};
enum class CudaPointerAlias : std::uint32_t { kNull = 0, kInput = 1 };
struct CudaPointerQuery {
    CudaPointerDomain domain{CudaPointerDomain::kDriver};
    std::uint32_t selector{};
    CudaPointerKind kind{CudaPointerKind::kNull};
    CudaAllocationRange range{};
};
struct CudaPointerObservation {
    std::uint32_t effects{};
    std::int32_t type{}, device{};
    CudaPointerAlias device_alias{CudaPointerAlias::kNull}, host_alias{CudaPointerAlias::kNull};
};
enum class CudaPointerContextSelection : std::uint8_t {
    kUnchanged,
    // The exact supplied querying Session, or explicit NULL when absent.
    // This proves selection only, never execution, completion or cleanup.
    kQuery,
    kUnproven,
};
struct CudaPointerMetadataResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<std::int32_t> original_status;
    std::optional<CudaPointerObservation> observation;
    CudaPointerContextSelection context_selection{CudaPointerContextSelection::kUnchanged};
};
struct CudaSessionBlasResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<cublasStatus_t> original_status;
    std::uint64_t id{};
    std::uint64_t value{};
};
class CudaSessionProcessAllocation {
   public:
    virtual ~CudaSessionProcessAllocation() = default;
    [[nodiscard]] virtual std::optional<CudaProcessMemoryReply> reply() const noexcept = 0;
};
struct CudaSessionProcessAllocationAcceptance {
    ExecutionAcceptance acceptance;
    std::shared_ptr<CudaSessionProcessAllocation> operation;
};
struct CudaSessionRandResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<curandStatus_t> original_status;
    std::uint64_t id{};
};
struct CudaSessionLtQuery {
    std::uint64_t handle{};
    CudaLtObjectView operation;
    std::array<CudaLtObjectView, 4> matrices;
    std::optional<CudaLtObjectView> preference;
    int requested{};
    std::optional<cublasLtMatmulAlgo_t> algorithm;
    bool has_c_layout{true};
};
struct CudaSessionLtAlgorithms {
    CudaSessionBlasResult result;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    std::size_t sdk_version{};
};
struct CudaSessionLtMatmul {
    CudaSessionLtQuery descriptors;
    std::array<CudaAllocationRange, 4> matrices;
    std::uint64_t stream{};
    CudaAllocationRange workspace;
    std::size_t workspace_bytes{}, scalar_bytes{};
    alignas(16) std::array<std::byte, 16> alpha{}, beta{};
    bool uses_device_scalars{};
    std::array<CudaAllocationRange, 2> device_scalars;
};
struct CudaSessionResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    // Present only when the targeted native operation was actually called.
    // Preparation failure after successful lookup does not overwrite its result.
    std::optional<CUresult> original_status;
    std::uint64_t id{};
    CUdeviceptr address{};
    std::size_t parameter_bytes{};
    // Borrowed until the next session resource mutation, never a wire field.
    std::span<const CudaArgumentLayout> arguments;
    // Private exact synchronous prefix, not a public reply or completion claim.
    CudaExecutionFrontier frontier{};
};
struct CudaSessionEventSubmission {
    CudaSessionResult result{};
    // Exact accepted target command, allocated before acceptance. Empty on
    // rejection/defer; later admissions and event rerecords cannot widen it.
    CudaExecutionFrontier frontier{};
    // Snapshot of the actual selected generation, not a later event lookup.
    std::optional<CudaSessionEventIdentity> generation{};
};
enum class CudaRuntimeMetadataKind : std::uint32_t {
    kProperties = 1,
    kAttribute,
    kPeerAttribute,
    kStreamPriorityRange,
    kPeerAccess,
    kMemoryInfo,
};
struct CudaRuntimeMetadataQuery {
    CudaRuntimeMetadataKind kind{CudaRuntimeMetadataKind::kProperties};
    int device{}, attribute{}, peer{};
};
struct CudaRuntimeMetadataResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<cudaError_t> original_status;
    int version{};
    std::vector<std::byte> data;
};
enum class CudaContextControl : std::uint32_t {
    kGetLimit = 1,
    kSetLimit,
    kGetCache,
    kSetCache,
    kGetSharedMemory,
    kSetSharedMemory,
    kResetPersistingCache,
    kGetApiVersion,
    kGetAffinity,
    kGetPriorityRange
};
struct CudaContextControlResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    // Selected operation determines Driver or Runtime provenance. Policy errors
    // have no original status; outputs are populated only on native success.
    std::optional<std::int32_t> original_status;
    std::uint64_t value{}, auxiliary{};
};
enum class CudaFunctionControl : std::uint32_t {
    kGetAttribute = 1,
    kSetAttribute,
    kSetCache,
    kGetAttributes = 4
};
struct CudaFunctionAttributes {
    std::array<std::uint64_t, 3> sizes{};
    std::array<std::int32_t, 13> values{};
};
struct CudaFunctionControlResult {
    CudaSessionDisposition disposition{CudaSessionDisposition::kRejected};
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    // Only the requested original function call supplies native provenance.
    // A failed private configuration disables replacement, not original status.
    std::optional<CUresult> original_status;
    int value{};
    // All sixteen original Driver observations, or no aggregate at all.
    std::optional<CudaFunctionAttributes> attributes;
};
// Owner-only context and error state; does not inspect scheduling readiness.
struct CudaSessionHealthSnapshot {
    bool has_context{}, is_quarantined{};
};
// Fresh owner-only scheduling observation; no diagnostic resource census.
struct CudaSessionSchedulingSnapshot {
    bool has_context{}, is_quarantined{};
    ExecutionSubmissionSnapshot submissions{};
    bool has_ready_work{};
    std::size_t external_barriers{};
    // Live host turns and their posted-prefix events; no application sequence
    // or host receipt is allocated. Physical poll.retired includes prefixes.
    std::size_t enqueue_turns{};
    // Retained started external barriers/turns, not accepted unstarted cuts.
    // This is a physical reservation count, never completion or SDK status.
    std::size_t external_work_in_flight{};
    // Same eligibility as pending_preparation(), without a resource/issue token.
    bool has_pending_preparation{};
};
struct CudaSessionSnapshot : CudaSessionSchedulingSnapshot {
    std::size_t device_bytes{}, image_bytes{}, allocations{}, modules{}, streams{}, functions{};
    std::size_t private_host_bytes{}, persistent_slots{}, persistent_functions{};
    std::uint64_t persistent_accepted{};
    std::size_t host_memory_bytes{}, host_allocations{};
    std::size_t blas_handles{}, lt_handles{}, random_handles{}, dnn_handles{};
    std::size_t external_context_leases{}, external_command_leases{};
    std::size_t default_proxy_staging_bytes{};
};

// Private server-owned resources and native execution; not a public CUDA ABI.
// Construct and mutate on one nonrecycled owner thread. Reply query/ACK may run
// concurrently after initialization, with lifecycle externally serialized.
// initialize requires no current context; activate/deactivate are explicit actor
// context switches. Numeric IDs require authenticated session/incarnation scope.
// Deferred/rejected preparation must not be returned as an original SDK result.
class CudaExecutionSession final {
   public:
    CudaExecutionSession(ExecutionSessionIdentity identity, std::uint64_t context_id,
                         CudaSessionDispatch dispatch, CudaSessionLimits limits = {},
                         CudaSessionPersistentConfig persistent = {},
                         std::shared_ptr<CudaPtxCache> ptx_cache = {},
                         std::shared_ptr<CudaProfileCache> profile_cache = {},
                         bool requires_profile_preemption = false,
                         CudaSessionStreamConfig streams = {},
                         CudaExecutionDomainMember domain_member = {});
    ~CudaExecutionSession();
    CudaExecutionSession(const CudaExecutionSession&) = delete;
    CudaExecutionSession& operator=(const CudaExecutionSession&) = delete;

    [[nodiscard]] CUresult attach_process_memory(
        std::shared_ptr<CudaProcessMemoryBackend>) noexcept;
    [[nodiscard]] CudaSessionProcessAllocationAcceptance allocate_process_memory(
        std::uint64_t sequence, const CudaProcessMemoryRequest&) noexcept;

    [[nodiscard]] CudaSessionResult initialize(int device_ordinal,
                                               CudaSessionContextConfig config = {}) noexcept;
    [[nodiscard]] CudaSessionResult get_context_flags() noexcept;
    [[nodiscard]] CudaSessionResult set_context_flags(unsigned int flags) noexcept;
    [[nodiscard]] CudaContextControlResult context_control(CudaContextControl operation,
                                                           std::uint32_t argument = 0,
                                                           std::uint64_t value = 0,
                                                           bool is_runtime = false) noexcept;
    [[nodiscard]] CudaSessionResult activate() noexcept;
    // An accepted source host turn still permits the original owner to restore
    // its own TLS context to NULL with independent before/after proof. This is
    // not native command permission and never changes the distinct worker TLS.
    [[nodiscard]] CudaSessionResult deactivate() noexcept;
    // CPU-only configuration; wrong process/thread or an active preparation
    // scope rejects without changing the installed observer. {} disables it.
    [[nodiscard]] bool set_preparation_observer(CudaPreparationObserver observer) noexcept;
    [[nodiscard]] bool set_retirement_observer(
        std::shared_ptr<CudaRetirementObserver> observer) noexcept;
    [[nodiscard]] CudaSessionResult load_ptx(std::string_view source) noexcept;
    [[nodiscard]] CudaSessionResult load_binary(std::span<const std::byte> image) noexcept;
    // Original compiler outputs are captured before private preparation. Output
    // records are preallocated before native issue; no client pointer is stored.
    [[nodiscard]] CudaSessionResult load_ptx_jit(
        std::string_view source, std::span<const CudaModuleJitOption> options,
        std::vector<CudaModuleJitOption>* outputs) noexcept;
    [[nodiscard]] CudaSessionResult load_binary_jit(
        std::span<const std::byte> image, std::span<const CudaModuleJitOption> options,
        std::vector<CudaModuleJitOption>* outputs) noexcept;
    [[nodiscard]] CudaSessionResult get_function(std::uint64_t module,
                                                 std::string_view name) noexcept;
    [[nodiscard]] CudaFunctionControlResult function_control(CudaFunctionControl operation,
                                                             std::uint64_t function,
                                                             std::uint32_t argument,
                                                             std::int32_t value = 0) noexcept;
    // The carrier may be uninitialized. Trusted composition authenticates the
    // distinct query/source owners; this method excludes their native mutation.
    [[nodiscard]] CudaPointerMetadataResult pointer_metadata(
        const CudaPointerQuery& query, CudaExecutionSession* querying = nullptr,
        CudaExecutionSession* source = nullptr,
        std::optional<CudaHostMemoryView> host = std::nullopt) noexcept;
    [[nodiscard]] CudaSessionResult unload_module(std::uint64_t module) noexcept;
    [[nodiscard]] CudaSessionResult get_global(std::uint64_t module,
                                               std::string_view name) noexcept;
    [[nodiscard]] CudaSessionResult allocate(std::size_t bytes) noexcept;
    [[nodiscard]] CudaSessionResult release(std::uint64_t allocation) noexcept;
    [[nodiscard]] std::optional<CudaDeviceMemoryView> device_memory(
        CudaAllocationRange range, std::size_t bytes) const noexcept;
    // Context retention does not freeze unrelated resource operations. Command
    // retention additionally protects exactly its stream/ranges/global parents.
    // No native work, stream sealing, waiting or physical event is introduced.
    // Caller keeps each capsule through every actual physical tail/uncertainty.
    [[nodiscard]] CudaSessionExternalContextCapture capture_external_context() noexcept;
    [[nodiscard]] CudaSessionExternalCommandCapture capture_external_command(
        std::uint64_t stream, std::span<const CudaSessionExternalRangeRequest> ranges) noexcept;
    // Trusted external actors retain immutable source capsules in lifetime.
    // Acceptance/readiness/cancel/fail are CPU-only, owner-thread operations;
    // begin/record require the actual bound native context. No host receipt or
    // application sequence is consumed. Already accepted work may finish after
    // close. Foreign execution needs incoming and outgoing dependency bridges;
    // record is valid only after actual enqueue reaches this exact source stream.
    [[nodiscard]] CudaExecutionExternalAcceptance accept_external_barrier(
        std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept;
    // One owner-session transaction, not cross-session progress admission.
    // Deduplicate streams while retaining the union lifetime through every tail.
    [[nodiscard]] CudaExecutionExternalCohortAcceptance accept_external_cohort(
        std::span<const std::uint64_t> streams,
        const std::shared_ptr<const void>& lifetime) noexcept;
    [[nodiscard]] CUresult external_barrier_ready(
        const CudaExecutionExternalBarrier& barrier) const noexcept;
    [[nodiscard]] CUresult begin_external_barrier(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    [[nodiscard]] CUresult record_external_enqueue(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    [[nodiscard]] CUresult enqueue_host_function(
        const CudaExecutionExternalBarrier& barrier, std::uint64_t stream,
        const std::shared_ptr<CudaSessionHostFunction>& callback) noexcept;
    [[nodiscard]] CUresult cancel_external_barrier(
        const CudaExecutionExternalBarrier& barrier) noexcept;
    [[nodiscard]] CUresult fail_external_barrier(const CudaExecutionExternalBarrier& barrier,
                                                 CUresult failure) noexcept;
    [[nodiscard]] CudaExecutionEnqueueAcceptance accept_enqueue_turn(
        std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept;
    [[nodiscard]] CudaExecutionEnqueueAcceptance accept_enqueue_cohort(
        std::span<const std::uint64_t> streams,
        const std::shared_ptr<const void>& lifetime) noexcept;
    [[nodiscard]] CUresult enqueue_turn_ready(const CudaExecutionEnqueueTurn& turn) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult progress_enqueue_dependency(
        const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CudaExecutionDependencyInspection inspect_enqueue_dependency(
        const CudaExecutionEnqueueTurn& turn) const noexcept;
    [[nodiscard]] CudaExecutionDependencyInspection inspect_frontier_dependency(
        const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] bool owns_frontier(const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult progress_frontier_dependency(
        const CudaExecutionFrontier& frontier) noexcept;
    // Exact cold-call quiescence view; historical errors remain observable on
    // the frontier. This does not relax cold_gate or authorize later admissions.
    [[nodiscard]] CudaExecutionDependencyInspection inspect_frontier_retirement(
        const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult progress_frontier_retirement(
        const CudaExecutionFrontier& frontier) noexcept;
    [[nodiscard]] CUresult begin_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult finish_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult cancel_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept;
    [[nodiscard]] CUresult fail_enqueue_turn(const CudaExecutionEnqueueTurn& turn,
                                             CUresult failure) noexcept;
    // Source bridge callbacks share all existing native session exclusion.
    // The worker endpoint never receives this session.
    [[nodiscard]] CUresult prepare_external_bridge(CudaExternalExecutionBridge& bridge) noexcept;
    [[nodiscard]] CUresult join_external_bridge(CudaExternalExecutionBridge& bridge) noexcept;
    [[nodiscard]] CUresult cleanup_external_bridge(CudaExternalExecutionBridge& bridge) noexcept;
    [[nodiscard]] CudaSessionResult upload(CudaAllocationRange range,
                                           std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] CudaSessionResult download(CudaAllocationRange range,
                                             std::span<std::byte> bytes) noexcept;
    [[nodiscard]] CudaSessionResult create_stream(
        unsigned int flags, std::optional<int> priority = std::nullopt) noexcept;
    // Logical application stream policy, independent of native proxy flags.
    // Owner-only CPU metadata; an absent value is not a known stream.
    [[nodiscard]] std::optional<bool> stream_is_nonblocking(std::uint64_t stream) const noexcept;
    [[nodiscard]] CudaRuntimeMetadataResult runtime_metadata(
        const CudaRuntimeMetadataQuery& query) noexcept;
    [[nodiscard]] CudaSessionResult destroy_stream(std::uint64_t stream) noexcept;
    [[nodiscard]] CudaSessionBlasResult create_blas() noexcept;
    [[nodiscard]] CudaDnnResult create_dnn() noexcept;
    [[nodiscard]] CudaDnnResult destroy_dnn(std::uint64_t handle) noexcept;
    [[nodiscard]] CudaDnnResult set_dnn_stream(std::uint64_t handle, std::uint64_t stream) noexcept;
    [[nodiscard]] CudaDnnResult get_dnn_stream(std::uint64_t handle) noexcept;
    [[nodiscard]] CudaDnnBackendContextResult backend_context(
        std::uint64_t owner, std::uint64_t handle, CudaDnnRegistry& registry,
        const CudaDnnBackendContextRequest& request) noexcept;
    [[nodiscard]] CudaDnnBackendExecutePreparation prepare_dnn_backend_execute(
        std::uint64_t owner, std::uint64_t handle, CudaDnnRegistry& registry, std::uint64_t plan,
        std::uint64_t variant) noexcept;
    [[nodiscard]] CudaDnnBackendExecuteAcceptance accept_dnn_backend_execute(
        std::uint64_t sequence, CudaDnnRegistry& registry, std::uint64_t owner,
        const CudaSessionDnnBackendExecute& problem) noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_compute(
        std::uint64_t sequence, const CudaSessionDnnCompute& problem) noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_convolution(
        std::uint64_t sequence, const CudaSessionDnnConvolution& problem) noexcept;
    [[nodiscard]] CudaDnnFindAcceptance accept_dnn_find(std::uint64_t sequence,
                                                        const CudaSessionDnnConvolution& problem,
                                                        int requested) noexcept;
    [[nodiscard]] CudaDnnConvolutionQueryResult query_dnn_convolution(
        const CudaSessionDnnConvolution& problem, CudaDnnConvolutionQueryKind kind,
        int requested = 0) noexcept;
    [[nodiscard]] CudaDnnResult query_dnn_backward_convolution(
        const CudaSessionDnnConvolution& problem, CudaDnnBackwardConvolutionQueryKind kind,
        bool has_output = true) noexcept;
    // Synchronous same-registry mutation: no queue drain or physical event.
    [[nodiscard]] CudaDnnResult fold_dnn_backward_convolution(
        std::uint64_t handle, CudaDnnRegistry& metadata,
        const CudaDnnFoldedDescriptors& descriptors) noexcept;
    [[nodiscard]] CudaDnnRnnQueryResult query_dnn_rnn(std::uint64_t handle,
                                                      CudaDnnRegistry& registry,
                                                      const CudaDnnRnnQuery& query) noexcept;
    [[nodiscard]] CudaDnnRnnWeightParamsResult query_dnn_rnn_weight_params(
        std::uint64_t handle, CudaDnnRegistry& registry,
        const CudaSessionDnnRnnWeightParams& problem) noexcept;
    [[nodiscard]] CudaDnnAttnWeightsResult query_dnn_attn_weights(
        std::uint64_t handle, CudaDnnRegistry& registry,
        const CudaDnnAttnWeights& parameters) noexcept;
    [[nodiscard]] CudaDnnRnnBuildResult build_dnn_rnn(std::uint64_t handle,
                                                      CudaDnnRegistry& registry,
                                                      std::uint64_t descriptor,
                                                      int mini_batch) noexcept;
    [[nodiscard]] CudaDnnRnnAcceptance accept_dnn_rnn(std::uint64_t sequence,
                                                      const CudaSessionDnnRnn& problem) noexcept;
    [[nodiscard]] CudaDnnLegacyResult dnn_legacy_control(std::uint64_t handle, CudaDnnRegistry&,
                                                         const CudaDnnLegacyControl&) noexcept;
    [[nodiscard]] CudaDnnRnnAcceptance accept_dnn_legacy_rnn(
        std::uint64_t sequence, CudaDnnRegistry&, const CudaSessionDnnLegacyRnn&) noexcept;
    [[nodiscard]] CudaDnnAttnPreparation prepare_dnn_attn(
        const CudaSessionDnnAttn& problem) noexcept;
    [[nodiscard]] CudaDnnAttnAcceptance accept_dnn_attn(std::uint64_t sequence,
                                                        const CudaSessionDnnAttn& problem) noexcept;
    [[nodiscard]] CudaDnnResult query_dnn_reduction(const CudaSessionDnnReduction& problem,
                                                    CudaDnnReductionQueryKind kind) noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_reduction(
        std::uint64_t sequence, const CudaSessionDnnReduction& problem) noexcept;
    [[nodiscard]] CudaDnnResult query_dnn_batch_norm(
        const CudaSessionDnnBatchNorm& problem) noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_batch_norm(
        std::uint64_t sequence, const CudaSessionDnnBatchNorm& problem) noexcept;
    [[nodiscard]] CudaDnnResult query_dnn_dropout_states_size(std::uint64_t handle) noexcept;
    [[nodiscard]] std::optional<CudaDeviceMemoryView> resolve_dnn_dropout(
        const CudaDnnDropoutView& dropout) const noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_dropout(
        std::uint64_t sequence, const CudaSessionDnnDropout& problem) noexcept;
    [[nodiscard]] CudaDnnComputeAcceptance accept_dnn_reorder(
        std::uint64_t sequence, const CudaSessionDnnReorder& problem) noexcept;
    [[nodiscard]] CudaSessionRandResult create_random(std::uint32_t type) noexcept;
    [[nodiscard]] CudaSessionRandResult set_random_seed(std::uint64_t handle,
                                                        std::uint64_t seed) noexcept;
    [[nodiscard]] CudaSessionRandResult destroy_random(std::uint64_t handle) noexcept;
    [[nodiscard]] CudaSessionBlasResult destroy_blas(std::uint64_t handle) noexcept;
    [[nodiscard]] CudaSessionBlasResult create_lt() noexcept;
    [[nodiscard]] CudaSessionBlasResult destroy_lt(std::uint64_t handle) noexcept;
    [[nodiscard]] CudaSessionLtAlgorithms query_lt_algorithms(
        const CudaSessionLtQuery& query) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_lt_matmul(std::uint64_t sequence,
                                                       const CudaSessionLtMatmul& problem) noexcept;
    [[nodiscard]] CudaSessionBlasResult set_blas_stream(std::uint64_t handle,
                                                        std::uint64_t stream) noexcept;
    [[nodiscard]] CudaSessionBlasResult set_blas_workspace(std::uint64_t handle,
                                                           CudaAllocationRange workspace,
                                                           std::size_t bytes) noexcept;
    [[nodiscard]] CudaSessionBlasResult set_blas_attribute(std::uint64_t handle,
                                                           CudaBlasAttribute attribute,
                                                           std::uint64_t value) noexcept;
    [[nodiscard]] CudaSessionBlasResult get_blas_attribute(std::uint64_t handle,
                                                           CudaBlasAttribute attribute) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_sgemm(std::uint64_t sequence,
                                                   const CudaSessionSgemm& problem) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_gemm_ex(std::uint64_t sequence,
                                                     const CudaSessionGemmEx& problem) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_geam(std::uint64_t sequence,
                                                  const CudaSessionGeam& problem) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_sgemv(std::uint64_t sequence,
                                                   const CudaSessionSgemv& problem) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_sgemm_ex(std::uint64_t sequence,
                                                      const CudaSessionSgemmEx& problem) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_strided_gemm_ex(
        std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept;
    // Canonical FP32 payload, but preserve the exact original non-Ex SDK entry.
    [[nodiscard]] ExecutionAcceptance accept_strided_sgemm(
        std::uint64_t sequence, const CudaSessionStridedGemmEx& problem) noexcept;
    [[nodiscard]] CudaSessionResult import_host_memory(int descriptor, std::size_t bytes,
                                                       std::size_t mapping_bytes,
                                                       unsigned int flags,
                                                       bool is_pageable = false) noexcept;
    [[nodiscard]] CudaSessionResult release_host_memory(std::uint64_t allocation) noexcept;
    [[nodiscard]] std::optional<CudaHostMemoryView> host_memory(CudaAllocationRange range,
                                                                std::size_t bytes) noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<const void>> host_lifetimes();
    [[nodiscard]] ExecutionAcceptance accept_host_transfer(std::uint64_t sequence,
                                                           CudaAllocationRange device,
                                                           const CudaHostMemoryView& host,
                                                           std::uint64_t stream,
                                                           bool is_upload) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_kernel(std::uint64_t sequence,
                                                    const CudaExecutionKernel& kernel) noexcept;
    // SDK-free ordinary admission, including when another native context is
    // current. All payload/progress/resource capacity precedes acceptance.
    [[nodiscard]] ExecutionAcceptance accept_async_kernel(
        std::uint64_t sequence, const CudaExecutionKernel& kernel) noexcept;
    // Exact admission provenance, not a fresh readiness or preparation query.
    // False on direct-original admission and every unsuccessful return.
    [[nodiscard]] ExecutionAcceptance accept_async_kernel(std::uint64_t sequence,
                                                          const CudaExecutionKernel& kernel,
                                                          bool* requires_preparation) noexcept;
    [[nodiscard]] CudaSessionPendingResult pending_preparation(
        std::optional<std::uint64_t> sequence = std::nullopt) const noexcept;
    // Owner-only SDK metadata preparation of that exact admitted record.
    // kRejected publishes a known-not-issued result; kUncertain retains all
    // owners and has no fabricated host result. kPrepared may have no ready
    // command yet when a stream predecessor still blocks physical issue.
    [[nodiscard]] CudaSessionKernelPreparationResult prepare_pending_kernel(
        const CudaSessionPendingCommand& command) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_memory(std::uint64_t sequence,
                                                    const CudaSessionMemory& memory) noexcept;
    [[nodiscard]] CudaSessionGraphSnapshot graph_snapshot() const noexcept;
    [[nodiscard]] CudaSessionGraphResult begin_graph_capture(std::uint64_t stream, int mode,
                                                             std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult end_graph_capture(std::uint64_t stream,
                                                           std::uint64_t caller, bool has_output,
                                                           std::uint64_t seed) noexcept;
    [[nodiscard]] CudaSessionGraphResult graph_capture_state(std::uint64_t stream,
                                                             std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult instantiate_graph(std::uint64_t graph, std::uint64_t flags,
                                                           bool has_output,
                                                           std::uint64_t seed) noexcept;
    [[nodiscard]] CudaSessionGraphResult destroy_graph(std::uint64_t graph) noexcept;
    [[nodiscard]] CudaSessionGraphResult destroy_graph_exec(std::uint64_t executable) noexcept;
    [[nodiscard]] CudaSessionGraphResult capture_kernel(const CudaExecutionKernel& kernel,
                                                        std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult capture_memory(const CudaSessionMemory& memory,
                                                        std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult invalidate_graph_capture(std::uint64_t caller) noexcept;
    [[nodiscard]] CudaSessionGraphResult abort_graph_capture() noexcept;
    [[nodiscard]] CudaSessionGraphLaunchAcceptance accept_graph_launch(
        std::uint64_t sequence, std::uint64_t executable, std::uint64_t stream) noexcept;
    [[nodiscard]] ExecutionAcceptance accept_uva_copy(std::uint64_t sequence,
                                                      const CudaSessionUvaCopy& copy) noexcept;
    [[nodiscard]] CudaExecutionIssueResult issue_next() noexcept;
    // CPU-only exact pending view, independent of selected native TLS. Issue
    // rejects a different Session/control/member/configuration epoch before SDK
    // work; ordinary native context/dependency/physical proof gates still apply.
    [[nodiscard]] CudaSessionPendingResult pending_command(
        std::optional<std::uint64_t> sequence = std::nullopt) const noexcept;
    [[nodiscard]] CudaExecutionIssueResult issue(const CudaSessionPendingCommand& command) noexcept;
    [[nodiscard]] CudaSessionPendingResult select_pending_kernel(
        const CudaSessionPendingCommand& command,
        const CudaExecutionPendingKernelSelection& selection) noexcept;
    // Explicit owner-only SDK metadata preparation. Native query errors never
    // invent resource capacity or a trained profile. Known unsupported families
    // are distinguished only after exact command/owner/configuration/context
    // validation. No kernel/replay/clock; isolated and final selection stay strict.
    [[nodiscard]] CudaSessionPairKernelResult prepare_pair_kernel(
        const CudaSessionPendingCommand& command) noexcept;
    // CPU-only refresh after this exact command's pre-issue/stopped geometry
    // selection; no issue/generation/control/configuration transition is hidden.
    [[nodiscard]] CudaSessionPairKernelResult refresh_pair_kernel(
        const CudaSessionPairKernel& kernel,
        const CudaSessionPendingCommand& command) const noexcept;
    [[nodiscard]] bool can_share_configuration(const CudaExecutionSession& other) const noexcept;
    // CPU-only readonly evaluation of exact prepared catalogs. Does not select
    // geometry, attach observers, change original fallback or issue work. The
    // returned values are not capabilities; final selection/issue revalidate.
    [[nodiscard]] static CudaSessionPairChoiceResult evaluate_pair(
        CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
        CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
        CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept;
    [[nodiscard]] static CudaSessionPairSelectionResult select_pair(
        CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
        CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
        CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept;
    // Explicit isolated inputs only. Normal readonly selection never samples.
    // Observers attach only before any initial preparation; the caller must
    // retain the selected exact cohort until both logical frontiers retire.
    [[nodiscard]] static CudaSessionPairSampleResult reserve_pair_sample(
        CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
        CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
        CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept;
    // Exact owner-serialized one/two-leg grant, not resume-all. The caller first
    // activates the selected owner and builds fresh authenticated global Core
    // observations. No public wire/client can supply these private capabilities.
    [[nodiscard]] static CudaSessionPairIssueResult issue_pair(
        CudaExecutionSession& first, CudaExecutionSession& second,
        const CudaSessionPairSelection& selection, const core::ExecutionPairPlan& authorization,
        std::array<std::uint64_t, 2> policy_ids) noexcept;
    [[nodiscard]] CUresult request_retreat(std::uint64_t sequence) noexcept;
    [[nodiscard]] CUresult retarget_workers(std::uint64_t sequence,
                                            CudaExecutionWorkerGeometry geometry) noexcept;
    [[nodiscard]] CudaExecutionIssueResult resume(std::uint64_t sequence) noexcept;
    [[nodiscard]] std::optional<CudaSessionPersistentProgress> persistent_progress(
        std::uint64_t sequence) const noexcept;
    [[nodiscard]] CudaExecutionPollResult poll() noexcept;
    [[nodiscard]] CudaSessionResult create_event(unsigned int flags) noexcept;
    [[nodiscard]] CudaSessionResult destroy_event(std::uint64_t event) noexcept;
    [[nodiscard]] CudaSessionResult record_event(std::uint64_t sequence, std::uint64_t event,
                                                 std::uint64_t stream,
                                                 unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionResult wait_event(std::uint64_t sequence, std::uint64_t event,
                                               std::uint64_t stream,
                                               unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionEventSubmission record_event_with_frontier(
        std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
        unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionEventSubmission wait_event_with_frontier(
        std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
        unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionEventCapture capture_event(std::uint64_t event) noexcept;
    // Trusted composition authenticates the source peer before passing its
    // immutable token. Source context lifetime is retained until retirement.
    [[nodiscard]] CudaSessionResult wait_event(std::uint64_t sequence,
                                               const CudaSessionEvent& source, std::uint64_t stream,
                                               unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionEventSubmission wait_event_with_frontier(
        std::uint64_t sequence, const CudaSessionEvent& source, std::uint64_t stream,
        unsigned int flags = 0) noexcept;
    [[nodiscard]] CudaSessionResult query_event(const CudaSessionEvent& event,
                                                bool should_synchronize = false) noexcept;
    // address carries the original float timestamp difference's exact u32 bits.
    [[nodiscard]] CudaSessionResult event_elapsed_time(std::uint64_t start,
                                                       std::uint64_t end) noexcept;
    [[nodiscard]] CudaExecutionPollResult drain_issued() noexcept;
    [[nodiscard]] CudaExecutionFrontierResult capture_frontier(
        std::optional<std::uint64_t> stream = std::nullopt,
        bool includes_submission_dependencies = false) noexcept;
    [[nodiscard]] CudaExecutionFrontierResult capture_command_frontier(
        std::uint64_t sequence) noexcept;
    [[nodiscard]] CudaExecutionFrontierReservationResult reserve_command_frontier(
        std::uint64_t sequence) noexcept;
    [[nodiscard]] CudaExecutionFrontierResult reserved_command_frontier(
        const CudaExecutionFrontierReservation& reservation) const noexcept;
    [[nodiscard]] CUresult cancel_frontier_reservation(
        const CudaExecutionFrontierReservation& reservation) noexcept;
    [[nodiscard]] bool reject_command(const ExecutionReceipt& receipt, CUresult failure) noexcept;
    [[nodiscard]] CUresult query_frontier(const CudaExecutionFrontier& frontier) const noexcept;
    [[nodiscard]] std::optional<ExecutionHostReply> host_result(
        const ExecutionReceipt& receipt) const noexcept;
    [[nodiscard]] bool acknowledge_result(const ExecutionReceipt& receipt) noexcept;
    [[nodiscard]] CudaSessionSnapshot snapshot() const noexcept;
    [[nodiscard]] CudaSessionHealthSnapshot health_snapshot() const noexcept;
    [[nodiscard]] CudaSessionSchedulingSnapshot scheduling_snapshot() const noexcept;
    [[nodiscard]] std::optional<std::array<std::uint64_t, 2>> native_kernel_pair() const noexcept;
    void close() noexcept;
    // CPU-only proof for cleanup scheduling, never computation/admission authority.
    [[nodiscard]] bool can_cleanup_quarantined() const noexcept;
    [[nodiscard]] std::size_t reject_queued(CUresult failure) noexcept;
    [[nodiscard]] CudaSessionResult shutdown() noexcept;

   private:
    [[nodiscard]] ExecutionAcceptance accept_kernel_impl(
        std::uint64_t sequence, const CudaExecutionKernel& kernel,
        const CudaExecutionPendingCommand* preparation = nullptr,
        const ExecutionReceipt* receipt = nullptr,
        std::optional<std::size_t>* reserved_slot = nullptr, CUresult* preparation_status = nullptr,
        CudaExecutionPendingCommand* prepared_command = nullptr);
    [[nodiscard]] static CudaSessionPairSelectionResult inspect_pair(
        CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
        CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
        CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept;
    [[nodiscard]] static CudaSessionPairSelectionResult configure_pair(
        std::array<CudaExecutionSession*, 2> sessions, std::array<CudaSessionPairKernel, 2> kernels,
        CudaPairProfileChoice choice,
        const std::array<std::shared_ptr<CudaExecutionObserver>, 2>& observations = {}) noexcept;
    struct EventCommand {
        std::uint64_t sequence{}, event{}, stream{};
        unsigned int flags{};
    };
    [[nodiscard]] CudaSessionEventSubmission wait_event_impl(EventCommand parameters,
                                                             const CudaSessionEvent& source,
                                                             bool captures_frontier) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
