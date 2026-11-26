#include "runtime.h"
#include "host_stage_metrics.h"
#include "internal/cuda_execution_domain.h"
#include "internal/cuda_sgemm_image.h"
#include "internal/cuda_gemm_geometry.h"
#include "internal/process_identity.h"

#ifdef GLIMMER_HAS_CUDNN
#include "internal/cuda_dnn_registry.h"
#include "../internal/cudnn_provider_identity.h"
#endif

#include <atomic>
#include <array>
#include <algorithm>
#include <charconv>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace {
struct RangeLibraryCloser {
    void operator()(void* library) const noexcept {
        if (library) (void)::dlclose(library);
    }
};
#ifdef GLIMMER_ENABLE_NCCL_EXECUTION
glimmer::app::ExecutionNcclDispatch nccl_dispatch() {
    return {{ncclGetUniqueId, ncclCommInitRank, ncclCommInitRankConfig, ncclCommGetAsyncError,
             ncclCommDestroy, ncclCommAbort, ncclGroupStart, ncclGroupEnd, ncclAllGather,
             ncclAllReduce, ncclBcast},
            {cuDeviceGet, cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRelease, cuCtxGetCurrent,
             cuCtxSetCurrent, cuStreamCreate, cuStreamQuery, cuStreamDestroy}};
}
#endif
#ifdef GLIMMER_HAS_CUDNN
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
glimmer::backend::cuda::CudaDnnDispatch dnn_dispatch() {
    auto result = glimmer::backend::cuda::CudaDnnDispatch{
        cudnnCreate, cudnnDestroy, cudnnSetStream, cudnnGetStream, cudnnCreateTensorDescriptor,
        cudnnDestroyTensorDescriptor, cudnnSetTensor4dDescriptor, cudnnSetTensor4dDescriptorEx,
        cudnnGetTensor4dDescriptor, cudnnSetTensorNdDescriptor, cudnnSetTensorNdDescriptorEx,
        cudnnGetTensorNdDescriptor, cudnnGetTensorSizeInBytes, cudnnCreateActivationDescriptor,
        cudnnDestroyActivationDescriptor, cudnnSetActivationDescriptor,
        cudnnGetActivationDescriptor, cudnnActivationForward, cudnnAddTensor,
        cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor, cudnnSetFilter4dDescriptor,
        cudnnGetFilter4dDescriptor, cudnnSetFilterNdDescriptor, cudnnGetFilterNdDescriptor,
        cudnnGetFilterSizeInBytes, cudnnCreateConvolutionDescriptor,
        cudnnDestroyConvolutionDescriptor, cudnnSetConvolution2dDescriptor,
        cudnnGetConvolution2dDescriptor, cudnnSetConvolutionNdDescriptor,
        cudnnGetConvolutionNdDescriptor, cudnnSetConvolutionGroupCount,
        cudnnGetConvolutionGroupCount, cudnnSetConvolutionMathType, cudnnGetConvolutionMathType,
        cudnnSetConvolutionReorderType, cudnnGetConvolutionReorderType,
        cudnnGetConvolution2dForwardOutputDim, cudnnGetConvolutionNdForwardOutputDim,
        cudnnConvolutionForward, cudnnGetConvolutionForwardWorkspaceSize,
        cudnnGetConvolutionForwardAlgorithm_v7, cudnnGetConvolutionForwardAlgorithmMaxCount,
        cudnnFindConvolutionForwardAlgorithm, cudnnCreatePoolingDescriptor,
        cudnnDestroyPoolingDescriptor, cudnnSetPoolingNdDescriptor, cudnnGetPoolingNdDescriptor,
        cudnnGetPoolingNdForwardOutputDim, cudnnPoolingForward, cudnnSoftmaxForward,
        cudnnCreateLRNDescriptor, cudnnDestroyLRNDescriptor, cudnnSetLRNDescriptor,
        cudnnLRNCrossChannelForward, cudnnTransformTensor, cudnnCreateReduceTensorDescriptor,
        cudnnDestroyReduceTensorDescriptor, cudnnSetReduceTensorDescriptor,
        cudnnGetReductionIndicesSize, cudnnGetReductionWorkspaceSize, cudnnReduceTensor,
        cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
        cudnnGetBatchNormalizationBackwardExWorkspaceSize,
        cudnnGetBatchNormalizationTrainingExReserveSpaceSize,
        cudnnBatchNormalizationForwardTrainingEx, cudnnBatchNormalizationBackwardEx,
        cudnnCreateDropoutDescriptor, cudnnDestroyDropoutDescriptor, cudnnDropoutGetStatesSize,
        cudnnSetDropoutDescriptor, cudnnRestoreDropoutDescriptor, cudnnCreateOpTensorDescriptor,
        cudnnDestroyOpTensorDescriptor, cudnnSetOpTensorDescriptor, cudnnGetOpTensorDescriptor,
        cudnnCreateTensorTransformDescriptor, cudnnDestroyTensorTransformDescriptor,
        cudnnSetTensorTransformDescriptor,
        // This route requires the configured SDK's host-only descriptor
        // implementation; actual SDK compatibility is a separate gate.
        cudnnInitTransformDest,
        // Version is also the internal reorder-footprint qualification.
        cudnnGetVersion, cudnnReorderFilterAndBias, cudnnGetCudartVersion, cudnnGetMaxDeviceVersion,
        cudnnGetProperty, cudnnGetConvolutionBackwardDataAlgorithmMaxCount,
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount,
        cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnGetFoldedConvBackwardDataDescriptors,
        cudnnCreateRNNDataDescriptor, cudnnDestroyRNNDataDescriptor, cudnnSetRNNDataDescriptor,
        cudnnCreateSeqDataDescriptor, cudnnDestroySeqDataDescriptor, cudnnSetSeqDataDescriptor,
        cudnnGetSeqDataDescriptor, cudnnCreateRNNDescriptor, cudnnDestroyRNNDescriptor,
        cudnnSetRNNDescriptor_v8, cudnnCreateAttnDescriptor, cudnnDestroyAttnDescriptor,
        cudnnSetAttnDescriptor, cudnnGetRNNWeightSpaceSize, cudnnGetRNNTempSpaceSizes,
        cudnnGetMultiHeadAttnBuffers, cudnnRNNSetClip_v8, cudnnGetRNNDataDescriptor,
        cudnnRNNForward, cudnnRNNBackwardData_v8, cudnnRNNBackwardWeights_v8, cudnnBuildRNNDynamic,
        cudnnGetRNNWeightParams, cudnnGetMultiHeadAttnWeights, cudnnMultiHeadAttnForward,
        cudnnMultiHeadAttnBackwardData, cudnnMultiHeadAttnBackwardWeights, cudnnGetAttnDescriptor,
        cudnnBackendCreateDescriptor, cudnnBackendDestroyDescriptor, cudnnBackendInitialize,
        cudnnBackendSetAttribute, cudnnBackendGetAttribute, cudnnBackendFinalize,
        cudnnBackendExecute};
#if CUDNN_MAJOR == 8
    result.legacy = {cudnnOpsInferVersionCheck,
                     cudnnOpsTrainVersionCheck,
                     cudnnCnnInferVersionCheck,
                     cudnnCnnTrainVersionCheck,
                     cudnnAdvInferVersionCheck,
                     cudnnAdvTrainVersionCheck,
                     cudnnCreateAlgorithmDescriptor,
                     cudnnSetAlgorithmDescriptor,
                     cudnnDestroyAlgorithmDescriptor,
                     cudnnSetRNNMatrixMathType,
                     cudnnGetRNNMatrixMathType,
                     cudnnSetRNNBiasMode,
                     cudnnGetRNNBiasMode,
                     cudnnSetRNNDescriptor_v6,
                     cudnnRNNSetClip,
                     cudnnGetRNNWorkspaceSize,
                     cudnnGetRNNTrainingReserveSize,
                     cudnnGetRNNParamsSize,
                     cudnnGetRNNLinLayerMatrixParams,
                     cudnnGetRNNLinLayerBiasParams,
                     cudnnSetRNNAlgorithmDescriptor,
                     cudnnGetRNNForwardInferenceAlgorithmMaxCount,
                     cudnnRNNForwardTraining,
                     cudnnRNNBackwardData,
                     cudnnRNNBackwardWeights};
#endif
    return result;
}
#pragma GCC diagnostic pop
#endif

// Required compatibility ABI: retain the original deprecated shared-bank calls.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
glimmer::backend::cuda::CudaContextDispatch context_dispatch() {
    return {cuCtxGetLimit,
            cuCtxSetLimit,
            cuCtxGetCacheConfig,
            cuCtxSetCacheConfig,
            cuCtxGetSharedMemConfig,
            cuCtxSetSharedMemConfig,
            cuCtxResetPersistingL2Cache,
            cuCtxGetApiVersion,
            cuCtxGetExecAffinity,
            cuCtxGetStreamPriorityRange,
            {cudaDeviceGetLimit, cudaDeviceSetLimit, cudaDeviceGetCacheConfig,
             cudaDeviceSetCacheConfig, cudaDeviceGetSharedMemConfig, cudaDeviceSetSharedMemConfig,
             cudaCtxResetPersistingL2Cache}};
}
#pragma GCC diagnostic pop
std::atomic<bool> should_stop{};
static_assert(std::atomic<bool>::is_always_lock_free);
void stop_requested(int) {
    should_stop.store(true, std::memory_order_relaxed);
}
bool number(std::string_view text, std::uint64_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}
bool profile_workers(std::string_view text, std::vector<unsigned int>& workers,
                     std::size_t maximum = 7) {
    workers.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        std::uint64_t value{};
        if (!number(text.substr(0, comma), value) || !value ||
            value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            workers.size() == maximum ||
            std::find(workers.begin(), workers.end(), value) != workers.end())
            return false;
        workers.push_back(static_cast<unsigned int>(value));
        if (comma == std::string_view::npos) return true;
        text.remove_prefix(comma + 1);
    }
    return false;
}
std::optional<glimmer::transform::PtxExecutionKind> execution_kind(std::string_view text) {
    using Kind = glimmer::transform::PtxExecutionKind;
    if (text == "static") return Kind::kStatic;
    if (text == "block-static") return Kind::kBlockStatic;
    if (text == "dynamic") return Kind::kDynamic;
    if (text == "preemptive") return Kind::kPreemptive;
    if (text == "sliced") return Kind::kSliced;
    return std::nullopt;
}
bool profile_kinds(std::string_view text,
                   std::vector<glimmer::transform::PtxExecutionKind>& kinds) {
    kinds.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto kind = execution_kind(text.substr(0, comma));
        if (!kind || kinds.size() == 5 || std::ranges::find(kinds, *kind) != kinds.end())
            return false;
        kinds.push_back(*kind);
        if (comma == std::string_view::npos) return true;
        text.remove_prefix(comma + 1);
    }
    return false;
}
bool client_priority(std::string_view text, glimmer::app::ExecutionClientPriority& result) {
    const auto first = text.find(':'),
               second = text.find(':', first == std::string_view::npos ? first : first + 1);
    if (first == std::string_view::npos || second == std::string_view::npos) return false;
    const auto third = text.find(':', second + 1);
    auto kernel_mode = glimmer::app::ExecutionClientKernelMode::kAuto;
    if (third != std::string_view::npos) {
        const auto selection = text.substr(third + 1);
        if (selection == "original")
            kernel_mode = glimmer::app::ExecutionClientKernelMode::kOriginal;
        else if (selection == "configured")
            kernel_mode = glimmer::app::ExecutionClientKernelMode::kConfigured;
        else if (selection != "auto")
            return false;
    }
    std::uint64_t pid{}, start{}, priority{};
    if (!number(text.substr(0, first), pid) ||
        !number(text.substr(first + 1, second - first - 1), start) ||
        !number(
            text.substr(second + 1, third == std::string_view::npos ? third : third - second - 1),
            priority) ||
        !pid || !start || pid > std::numeric_limits<std::uint32_t>::max() ||
        priority > std::numeric_limits<std::uint32_t>::max())
        return false;
    result = {{static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start},
              static_cast<std::uint32_t>(priority),
              kernel_mode};
    return true;
}
struct BlockCompletionPin {
    glimmer::app::ExecutionClientPriority client;
    std::uint64_t last_session{}, last_sequence{};
};

void observe_blocks(glimmer::app::ExecutionService& service,
                    std::vector<BlockCompletionPin>& pins) noexcept {
    for (auto& pin : pins) {
        const auto observation = service.block_completion_observation(pin.client.peer);
        if (!observation || (observation->session == pin.last_session &&
                             observation->sequence == pin.last_sequence))
            continue;
        const auto& progress = observation->progress;
        if (!progress.observes_block_completion || !progress.has_completed_block ||
            progress.physical_launches != 1 || progress.physical_stops ||
            progress.completed_blocks || !progress.total_blocks || !progress.workers)
            continue;
        if (service.publish_block_completion_audit(pin.client.peer, pin.client.priority,
                                                   *observation)) {
            pin.last_session = observation->session;
            pin.last_sequence = observation->sequence;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    glimmer::app::ExecutionServiceConfig config;
    std::string sgemm_range_library, host_stage_output, issue_depth_output, poll_journal_output;
    bool should_record_issue_events{}, should_record_issue_selections{};
    bool has_kernel_mode{}, has_profile_options{}, has_sharing_options{};
    glimmer::app::ExecutionServicePairConfig paired;
    bool has_pair_options{}, has_pair_workers{};
    bool has_priority_idle{};
    bool has_gemm_tile_side{};
    config.transport.max_pending_requests = 8;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--help") {
            std::cout
                << "Usage: glimmer_execution_service --socket PATH [--device N] [--max-clients N] "
                   "[--memory-per-client BYTES] [--host-memory-per-client BYTES] "
                   "[--max-pending-requests N] "
                   "[--enable-nccl] "
                   "[--persistent-workers N] [--replace-gemm|--replace-sgemm] "
                   "[--gemm-tile-side 32|64] [--experimental-sharded-gemm-signal] "
                   "[--exit-when-idle] "
                   "[--scheduler "
                   "priority|naive|profile|workload-agnostic|workload-aware|priority-space-share] "
                   "[--priority-idle-us N (default 100; 0 disables)] "
                   "[--pair-workers N,N,...] "
                   "[--pair-profile-input ABSOLUTE_PATH] [--pair-profile-output NEW_ABSOLUTE_PATH] "
                   "[--pair-cache-pairs N] [--pair-cache-bytes N] [--pair-cache-candidates N] "
                   "[--pair-warmup N] [--pair-samples N] "
                   "[--sharing-threads-per-sm N] [--sharing-min-speed-percent N] "
                   "[--kernel-mode original|preemptive|static|block-static|dynamic|sliced] "
                   "[--ptx-cache-bytes N] [--ptx-cache-entries N] "
                   "[--ptx-cache-input ABSOLUTE_PATH] [--ptx-cache-output NEW_ABSOLUTE_PATH] "
                   "[--profile-workers N,N,...] [--profile-warmup N] [--profile-samples N] "
                   "[--profile-kinds static,block-static,dynamic,preemptive,sliced] "
                   "[--profile-cache-bytes N] [--profile-cache-functions N] "
                   "[--profile-input ABSOLUTE_PATH] [--profile-output NEW_ABSOLUTE_PATH] "
                   "[--audit-native] [--audit-sgemm-target SESSION:SEQUENCE] "
                   "[--audit-sgemm-target-client PID:START:SEQUENCE] "
                   "[--audit-sgemm-range-library ABSOLUTE_PATH] "
                   "[--audit-copy-requests] "
                   "[--host-stage-output NEW_ABSOLUTE_PATH] "
                   "[--poll-journal-output NEW_ABSOLUTE_PATH] "
                   "[--issue-depth-output NEW_ABSOLUTE_PATH] "
                   "[--issue-depth-events] "
                   "[--issue-depth-selections] "
                   "[--skip-idle-helper-plans] "
                   "[--async-intake-burst] "
                   "[--client-priority PID:START:PRIORITY[:auto|original|configured]] "
                   "[--audit-priority] [--observe-block-completion]\n";
            return EXIT_SUCCESS;
        }
        if (option == "--exit-when-idle") {
            config.exit_when_idle = true;
            continue;
        }
        if (option == "--enable-nccl") {
#ifdef GLIMMER_ENABLE_NCCL_EXECUTION
            if (config.uses_nccl) {
                spdlog::error("Duplicate NCCL execution selection");
                return EXIT_FAILURE;
            }
            config.uses_nccl = true;
            continue;
#else
            spdlog::error("NCCL execution is unavailable in this build");
            return EXIT_FAILURE;
#endif
        }
        if (option == "--replace-sgemm" || option == "--replace-gemm") {
            config.persistent.replace_sgemm = true;
            continue;
        }
        if (option == "--experimental-sharded-gemm-signal") {
            if (config.persistent.uses_sharded_host_retreat) {
                spdlog::error("Duplicate sharded GEMM signal selection");
                return EXIT_FAILURE;
            }
            config.persistent.uses_sharded_host_retreat = true;
            continue;
        }
        if (option == "--audit-native") {
            config.should_audit_native = true;
            continue;
        }
        if (option == "--audit-priority") {
            config.should_audit_priority = true;
            continue;
        }
        if (option == "--audit-copy-requests") {
            if (config.should_audit_copy_requests) {
                spdlog::error("Duplicate copy-request audit selection");
                return EXIT_FAILURE;
            }
            config.should_audit_copy_requests = true;
            continue;
        }
        if (option == "--observe-block-completion") {
            if (config.persistent.observes_block_completion) {
                spdlog::error("Duplicate block-completion observation selection");
                return EXIT_FAILURE;
            }
            config.persistent.observes_block_completion = true;
            continue;
        }
        if (option == "--async-intake-burst") {
            if (config.should_receive_async_burst) {
                spdlog::error("Duplicate async-intake burst selection");
                return EXIT_FAILURE;
            }
            config.should_receive_async_burst = true;
            continue;
        }
        if (option == "--skip-idle-helper-plans") {
            if (config.should_skip_idle_helper_plans) {
                spdlog::error("Duplicate idle-helper plan selection");
                return EXIT_FAILURE;
            }
            config.should_skip_idle_helper_plans = true;
            continue;
        }
        if (option == "--issue-depth-events") {
            if (should_record_issue_events) {
                spdlog::error("Duplicate issue-event selection");
                return EXIT_FAILURE;
            }
            should_record_issue_events = true;
            continue;
        }
        if (option == "--issue-depth-selections") {
            if (should_record_issue_selections) {
                spdlog::error("Duplicate issue-selection recording");
                return EXIT_FAILURE;
            }
            should_record_issue_selections = true;
            continue;
        }
        if (index + 1 >= argc) {
            spdlog::error("Missing execution-service option value");
            return EXIT_FAILURE;
        }
        const std::string_view value{argv[++index]};
        if (option == "--host-stage-output" || option == "--issue-depth-output" ||
            option == "--poll-journal-output") {
            auto& output = option == "--host-stage-output"    ? host_stage_output
                           : option == "--issue-depth-output" ? issue_depth_output
                                                              : poll_journal_output;
            if (!output.empty() ||
                !glimmer::app::HostStageMetrics::fresh_output(std::string{value})) {
                spdlog::error("Diagnostic output requires a fresh absolute owned path");
                return EXIT_FAILURE;
            }
            output = value;
            continue;
        }
        if (option == "--socket") {
            config.transport.socket_path = value;
            continue;
        }
        if (option == "--ptx-cache-input" || option == "--ptx-cache-output") {
            if (value.size() < 2 || value.size() > 4096 || value.front() != '/' ||
                value.back() == '/') {
                spdlog::error("PTX archive requires an absolute file path");
                return EXIT_FAILURE;
            }
            auto& path =
                option == "--ptx-cache-input" ? config.ptx_cache_input : config.ptx_cache_output;
            if (!path.empty()) {
                spdlog::error("Duplicate PTX archive option");
                return EXIT_FAILURE;
            }
            path = value;
            continue;
        }
        if (option == "--profile-input" || option == "--profile-output") {
            if (value.size() < 2 || value.size() > 4096 || value.front() != '/' ||
                value.back() == '/') {
                spdlog::error("Profile archive requires an absolute file path");
                return EXIT_FAILURE;
            }
            auto& path = option == "--profile-input" ? config.profile_input : config.profile_output;
            if (!path.empty()) {
                spdlog::error("Duplicate profile archive option");
                return EXIT_FAILURE;
            }
            path = value;
            continue;
        }
        if (option == "--pair-profile-input" || option == "--pair-profile-output") {
            if (value.size() < 2 || value.size() > 4096 || value.front() != '/' ||
                value.back() == '/') {
                spdlog::error("Pair profile archive requires an absolute file path");
                return EXIT_FAILURE;
            }
            auto& path =
                option == "--pair-profile-input" ? paired.profile_input : paired.profile_output;
            if (!path.empty()) {
                spdlog::error("Duplicate pair profile archive option");
                return EXIT_FAILURE;
            }
            path = value;
            has_pair_options = true;
            continue;
        }
        if (option == "--pair-workers") {
            if (has_pair_workers || !profile_workers(value, paired.workers, 8)) {
                spdlog::error("Invalid or duplicate pair worker list");
                return EXIT_FAILURE;
            }
            has_pair_workers = has_pair_options = true;
            continue;
        }
        if (option == "--client-priority") {
            glimmer::app::ExecutionClientPriority entry;
            if (config.client_priorities.size() == 64 || !client_priority(value, entry)) {
                spdlog::error("Invalid or oversized trusted execution priority table");
                return EXIT_FAILURE;
            }
            if (std::ranges::any_of(config.client_priorities,
                                    [&](const auto& prior) { return prior.peer == entry.peer; })) {
                spdlog::error("Duplicate or conflicting trusted execution priority identity");
                return EXIT_FAILURE;
            }
            config.client_priorities.push_back(entry);
            continue;
        }
        if (option == "--audit-sgemm-target") {
            const auto separator = value.find(':');
            glimmer::app::ExecutionSgemmAuditTarget target;
            if (config.sgemm_audit_target || separator == std::string_view::npos ||
                !number(value.substr(0, separator), target.session) || !target.session ||
                !number(value.substr(separator + 1), target.sequence) || !target.sequence) {
                spdlog::error("Invalid or duplicate SGEMM audit target");
                return EXIT_FAILURE;
            }
            config.sgemm_audit_target = target;
            continue;
        }
        if (option == "--audit-sgemm-target-client") {
            const auto first = value.find(':'),
                       second =
                           value.find(':', first == std::string_view::npos ? first : first + 1);
            std::uint64_t pid{}, start{};
            glimmer::app::ExecutionSgemmAuditTarget target;
            if (config.sgemm_audit_target || first == std::string_view::npos ||
                second == std::string_view::npos || !number(value.substr(0, first), pid) || !pid ||
                pid > std::numeric_limits<std::uint32_t>::max() ||
                !number(value.substr(first + 1, second - first - 1), start) || !start ||
                !number(value.substr(second + 1), target.sequence) || !target.sequence) {
                spdlog::error("Invalid or duplicate SGEMM audit target");
                return EXIT_FAILURE;
            }
            target.client = glimmer::control::TaskPeerIdentity{
                static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start};
            config.sgemm_audit_target = target;
            continue;
        }
        if (option == "--audit-sgemm-range-library") {
            if (!sgemm_range_library.empty() || value.size() < 2 || value.size() > 4096 ||
                value.front() != '/' || value.back() == '/') {
                spdlog::error("Invalid or duplicate SGEMM range library path");
                return EXIT_FAILURE;
            }
            sgemm_range_library = value;
            continue;
        }
        if (option == "--scheduler") {
            if (value == "priority")
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kPriority;
            else if (value == "naive")
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
            else if (value == "profile")
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
            else if (value == "workload-agnostic")
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic;
            else if (value == "workload-aware")
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAware;
            else if (value == "priority-space-share")
                config.scheduling_mode =
                    glimmer::core::ExecutionSchedulingMode::kPrioritySpaceShare;
            else {
                spdlog::error(
                    "Unknown execution scheduler; expected priority, naive, profile or "
                    "workload-agnostic, workload-aware or priority-space-share");
                return EXIT_FAILURE;
            }
            continue;
        }
        if (option == "--profile-workers") {
            has_profile_options = true;
            if (!profile_workers(value, config.profile.workers)) {
                spdlog::error("Invalid profile worker list");
                return EXIT_FAILURE;
            }
            continue;
        }
        if (option == "--profile-kinds") {
            has_profile_options = true;
            if (!profile_kinds(value, config.profile.kinds)) {
                spdlog::error("Invalid profile family list");
                return EXIT_FAILURE;
            }
            continue;
        }
        if (option == "--kernel-mode") {
            const auto kind = execution_kind(value == "original" ? "preemptive" : value);
            if (!kind) {
                spdlog::error("Unknown kernel mode");
                return EXIT_FAILURE;
            }
            config.persistent.kind = *kind;
            config.uses_original_kernels = value == "original";
            has_kernel_mode = true;
            continue;
        }
        std::uint64_t parsed{};
        if (!number(value, parsed)) {
            spdlog::error("Invalid execution-service numeric value");
            return EXIT_FAILURE;
        }
        if (option == "--priority-idle-us") {
            if (has_priority_idle || parsed > 1000000) {
                spdlog::error("Duplicate or out-of-range priority idle interval");
                return EXIT_FAILURE;
            }
            has_priority_idle = true;
            config.priority_idle_us = static_cast<std::uint32_t>(parsed);
        } else if (option == "--device" &&
                   parsed <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            config.device_ordinal = static_cast<int>(parsed);
        else if (option == "--max-clients" && parsed > 0 && parsed <= 64)
            config.transport.max_peers = static_cast<std::size_t>(parsed);
        else if (option == "--max-pending-requests" && parsed > 0 && parsed <= 64)
            config.transport.max_pending_requests = static_cast<std::size_t>(parsed);
        else if (option == "--memory-per-client" && parsed > 0 &&
                 parsed <= std::numeric_limits<std::size_t>::max())
            config.session_limits.max_device_bytes = static_cast<std::size_t>(parsed);
        else if (option == "--host-memory-per-client" && parsed > 0 &&
                 parsed <= std::numeric_limits<std::size_t>::max())
            config.session_limits.max_host_memory_bytes = static_cast<std::size_t>(parsed);
        else if (option == "--persistent-workers" && parsed > 0 &&
                 parsed <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            config.persistent.workers = static_cast<unsigned int>(parsed);
        else if (option == "--gemm-tile-side" && !has_gemm_tile_side &&
                 (parsed == 32 || parsed == 64)) {
            has_gemm_tile_side = true;
            config.persistent.gemm_tile_side = static_cast<unsigned int>(parsed);
        } else if (option == "--sharing-threads-per-sm" && parsed > 0 && parsed <= 65536) {
            has_sharing_options = true;
            config.persistent.sharing_threads_per_sm = static_cast<unsigned int>(parsed);
        } else if (option == "--sharing-min-speed-percent" && parsed > 0 && parsed <= 10000) {
            has_sharing_options = true;
            config.persistent.sharing_minimum_speed_percent = static_cast<unsigned int>(parsed);
        } else if (option == "--ptx-cache-bytes" &&
                   parsed <= std::numeric_limits<std::size_t>::max())
            config.ptx_cache.max_bytes = static_cast<std::size_t>(parsed);
        else if (option == "--ptx-cache-entries" && parsed <= 4096)
            config.ptx_cache.max_entries = static_cast<std::size_t>(parsed);
        else if (option == "--profile-warmup" && parsed <= 1000) {
            has_profile_options = true;
            config.profile.warmup_calls = static_cast<std::uint32_t>(parsed);
        } else if (option == "--profile-samples" && parsed > 0 && parsed <= 1000) {
            has_profile_options = true;
            config.profile.samples_per_candidate = static_cast<std::uint32_t>(parsed);
        } else if (option == "--profile-cache-functions" && parsed > 0 && parsed <= 1024) {
            has_profile_options = true;
            config.profile.max_functions = static_cast<std::size_t>(parsed);
        } else if (option == "--profile-cache-bytes" && parsed > 0 &&
                   parsed <= std::numeric_limits<std::size_t>::max()) {
            has_profile_options = true;
            config.profile.max_bytes = static_cast<std::size_t>(parsed);
        } else if (option == "--pair-cache-pairs" && parsed > 0 && parsed <= 1024) {
            has_pair_options = true;
            paired.profile.max_pairs = static_cast<std::size_t>(parsed);
        } else if (option == "--pair-cache-bytes" && parsed > 0 &&
                   parsed <= glimmer::backend::cuda::kCudaPairProfileArchiveMaxBytes) {
            has_pair_options = true;
            paired.profile.max_bytes = static_cast<std::size_t>(parsed);
        } else if (option == "--pair-cache-candidates" && parsed > 0 && parsed <= 64) {
            has_pair_options = true;
            paired.profile.max_candidates = static_cast<std::size_t>(parsed);
        } else if (option == "--pair-warmup" && parsed <= 1000) {
            has_pair_options = true;
            paired.profile.warmup_calls = static_cast<std::uint32_t>(parsed);
        } else if (option == "--pair-samples" && parsed > 0 && parsed <= 1000) {
            has_pair_options = true;
            paired.profile.samples_per_candidate = static_cast<std::uint32_t>(parsed);
        } else {
            spdlog::error("Unknown or out-of-range execution-service option");
            return EXIT_FAILURE;
        }
    }
    if (config.sgemm_audit_target && !config.should_audit_native) {
        spdlog::error("SGEMM audit target requires native auditing");
        return EXIT_FAILURE;
    }
    if (!sgemm_range_library.empty() && !config.sgemm_audit_target) {
        spdlog::error("SGEMM range library requires an exact SGEMM audit target");
        return EXIT_FAILURE;
    }
    if (should_record_issue_selections && !should_record_issue_events) {
        spdlog::error("Issue selections require --issue-depth-events");
        return EXIT_FAILURE;
    }
    if (should_record_issue_events && issue_depth_output.empty()) {
        spdlog::error("Issue events require --issue-depth-output");
        return EXIT_FAILURE;
    }
    if (config.transport.socket_path.empty()) {
        spdlog::error("Execution-service socket path is required");
        return EXIT_FAILURE;
    }
    if (!poll_journal_output.empty()) {
        if (!config.exit_when_idle) {
            spdlog::error("Poll journal publication requires --exit-when-idle");
            return EXIT_FAILURE;
        }
        const std::array outputs{poll_journal_output,         host_stage_output,
                                 issue_depth_output,          config.ptx_cache_output,
                                 config.profile_output,       paired.profile_output,
                                 config.transport.socket_path};
        std::array<std::filesystem::path, outputs.size()> resolved;
        std::size_t used{};
        for (const auto& output : outputs) {
            if (output.empty()) continue;
            std::error_code error;
            const auto path = std::filesystem::weakly_canonical(output, error);
            if (error || !path.is_absolute() ||
                std::find(resolved.begin(), resolved.begin() + static_cast<std::ptrdiff_t>(used),
                          path) != resolved.begin() + static_cast<std::ptrdiff_t>(used)) {
                spdlog::error("Diagnostic and execution output paths must be pairwise distinct");
                return EXIT_FAILURE;
            }
            resolved[used++] = path;
        }
    }
    const bool uses_space_share =
        config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kPrioritySpaceShare;
    if (uses_space_share) {
        if ((has_kernel_mode && !config.uses_original_kernels) || config.persistent.workers ||
            config.persistent.replace_sgemm || has_profile_options || has_sharing_options ||
            has_pair_options || !config.profile_input.empty() || !config.profile_output.empty() ||
            !config.ptx_cache_input.empty() || !config.ptx_cache_output.empty() ||
            config.persistent.observes_block_completion || has_priority_idle ||
            config.client_priorities.size() < 2 ||
            config.client_priorities.size() > config.transport.max_peers ||
            config.transport.max_peers > glimmer::core::kMaxExecutionPriorityPeers ||
            config.session_limits.max_device_bytes >
                std::numeric_limits<std::size_t>::max() / config.transport.max_peers) {
            spdlog::error(
                "Priority space sharing requires at least two bounded pinned original clients "
                "without transform, replacement, profile, pair or idle-hold options");
            return EXIT_FAILURE;
        }
        config.uses_original_kernels = true;
        config.shared =
            glimmer::app::ExecutionServiceSharedConfig{.should_initialize_domain = true};
    }
    if (has_priority_idle && (!glimmer::core::is_priority_execution_mode(config.scheduling_mode) ||
                              config.client_priorities.empty())) {
        spdlog::error("Explicit priority idle interval requires priority mode and pinned clients");
        return EXIT_FAILURE;
    }
    if (config.persistent.observes_block_completion &&
        (!config.should_audit_priority || config.client_priorities.empty() ||
         config.scheduling_mode != glimmer::core::ExecutionSchedulingMode::kPriority ||
         config.uses_original_kernels || !config.persistent.workers ||
         config.persistent.kind != glimmer::transform::PtxExecutionKind::kPreemptive ||
         !config.profile_input.empty() || !config.profile_output.empty() || has_profile_options)) {
        spdlog::error(
            "Block-completion observation requires unmeasured preemptive priority workers, "
            "pinned clients and priority auditing");
        return EXIT_FAILURE;
    }
    if (std::ranges::any_of(config.client_priorities,
                            [](const auto& entry) {
                                return entry.kernel_mode ==
                                       glimmer::app::ExecutionClientKernelMode::kConfigured;
                            }) &&
        (!glimmer::core::is_priority_execution_mode(config.scheduling_mode) ||
         config.uses_original_kernels || !config.persistent.workers)) {
        spdlog::error(
            "Configured client kernels require priority scheduling and enabled "
            "transformation/workers");
        return EXIT_FAILURE;
    }
    const bool uses_pairs =
        config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kWorkloadAware;
    if (has_pair_options && !uses_pairs) {
        spdlog::error("Pair options require workload-aware scheduling");
        return EXIT_FAILURE;
    }
    if (uses_pairs) {
        const auto configured =
            std::ranges::count_if(config.client_priorities, [](const auto& pin) {
                return pin.kernel_mode == glimmer::app::ExecutionClientKernelMode::kConfigured;
            });
        if (!has_pair_workers || !config.persistent.workers || config.uses_original_kernels ||
            config.profile_input.empty() || !config.profile_output.empty() || configured < 2 ||
            config.client_priorities.size() > config.transport.max_peers ||
            (config.persistent.kind != glimmer::transform::PtxExecutionKind::kPreemptive &&
             config.persistent.kind != glimmer::transform::PtxExecutionKind::kSliced) ||
            std::ranges::any_of(paired.workers, [&](auto workers) {
                return workers > config.persistent.workers;
            })) {
            spdlog::error(
                "Workload-aware scheduling requires readonly singles, explicit compatible pair "
                "workers and at least two configured pins");
            return EXIT_FAILURE;
        }
        paired.is_isolated_profile = !paired.profile_output.empty();
        paired.should_rebalance_on_arrival = !paired.is_isolated_profile;
        paired.should_initialize_domain = true;
        if (paired.is_isolated_profile &&
            (config.client_priorities.size() != 2 || configured != 2 || config.uses_nccl ||
             config.client_priorities[0].priority != config.client_priorities[1].priority ||
             paired.workers.size() * paired.workers.size() > paired.profile.max_candidates)) {
            spdlog::error(
                "Pair profiling requires exactly two equal configured pins, a bounded catalog and "
                "no native actor");
            return EXIT_FAILURE;
        }
        const std::array<std::string_view, 5> aliases{
            config.profile_input, config.profile_output, config.ptx_cache_input,
            config.ptx_cache_output, paired.profile_input};
        if ((!paired.profile_output.empty() &&
             std::ranges::find(aliases, paired.profile_output) != aliases.end()) ||
            (!config.ptx_cache_output.empty() && config.ptx_cache_output == paired.profile_input) ||
            config.session_limits.max_device_bytes >
                std::numeric_limits<std::size_t>::max() / config.transport.max_peers) {
            spdlog::error(
                "Pair archive outputs must be distinct and aggregate memory bounds must not "
                "overflow");
            return EXIT_FAILURE;
        }
        config.paired = std::move(paired);
    }
    if ((!config.ptx_cache_input.empty() || !config.ptx_cache_output.empty()) &&
        (config.uses_original_kernels || !config.persistent.workers ||
         !config.ptx_cache.max_entries || !config.ptx_cache.max_bytes)) {
        spdlog::error("PTX archives require enabled transformation and cache");
        return EXIT_FAILURE;
    }
    if ((!config.ptx_cache_output.empty() && (config.ptx_cache_output == config.ptx_cache_input ||
                                              config.ptx_cache_output == config.profile_input ||
                                              config.ptx_cache_output == config.profile_output)) ||
        (!config.profile_output.empty() && config.profile_output == config.ptx_cache_input)) {
        spdlog::error("Execution archive output paths must be distinct");
        return EXIT_FAILURE;
    }
    const bool uses_sharing =
        config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic;
    if ((has_sharing_options && !uses_sharing) ||
        (uses_sharing && (config.uses_original_kernels || config.persistent.replace_sgemm ||
                          !config.persistent.workers || config.profile_input.empty() ||
                          !config.profile_output.empty() || !config.client_priorities.empty()))) {
        spdlog::error(
            "Workload-agnostic scheduling needs a readonly profile and explicit workers; "
            "sharing options cannot be used by another scheduler");
        return EXIT_FAILURE;
    }
    if (has_profile_options &&
        config.scheduling_mode != glimmer::core::ExecutionSchedulingMode::kProfile &&
        config.profile_input.empty()) {
        spdlog::error("Profile options require the profile scheduler or a profile input");
        return EXIT_FAILURE;
    }
    if (!config.profile_output.empty() &&
        config.scheduling_mode != glimmer::core::ExecutionSchedulingMode::kProfile) {
        spdlog::error("Profile output requires the profile scheduler");
        return EXIT_FAILURE;
    }
    if (!config.profile_input.empty() && config.profile_input == config.profile_output) {
        spdlog::error("Profile output must not replace the input archive");
        return EXIT_FAILURE;
    }
    const bool learns = config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kProfile;
    if ((learns || !config.profile_input.empty()) &&
        ((learns && (!config.client_priorities.empty() || config.persistent.replace_sgemm)) ||
         (config.uses_original_kernels && !config.profile.kinds.empty()) ||
         (!config.uses_original_kernels &&
          (!config.persistent.workers ||
           (!uses_sharing && std::ranges::any_of(config.profile.workers, [&](auto workers) {
               return workers > config.persistent.workers;
           })))))) {
        spdlog::error(
            "Profile requires explicit compatible workers or original mode, "
            "without a priority table or private library replacement during measurement");
        return EXIT_FAILURE;
    }
    if (config.persistent.uses_sharded_host_retreat &&
        (!config.persistent.replace_sgemm || !config.persistent.workers || !has_gemm_tile_side ||
         config.persistent.gemm_tile_side != 32 || config.uses_original_kernels ||
         config.persistent.kind != glimmer::transform::PtxExecutionKind::kPreemptive ||
         glimmer::backend::cuda::cuda_sgemm_sharded_image(32).empty())) {
        spdlog::error(
            "Sharded GEMM signal requires private replacement, explicit tile side 32, "
            "preemptive workers and the compiled sharded provider");
        return EXIT_FAILURE;
    }
    if (config.uses_original_kernels && config.persistent.replace_sgemm) {
        spdlog::error("Original kernel mode conflicts with private GEMM replacement");
        return EXIT_FAILURE;
    }
    if (has_kernel_mode && !config.uses_original_kernels && !config.persistent.workers) {
        spdlog::error("Transformed kernel mode requires explicit persistent workers");
        return EXIT_FAILURE;
    }
    if (config.persistent.replace_sgemm &&
        config.persistent.kind != glimmer::transform::PtxExecutionKind::kPreemptive) {
        spdlog::error("Private GEMM replacement requires preemptive kernel mode");
        return EXIT_FAILURE;
    }
    if (has_gemm_tile_side && !config.persistent.replace_sgemm) {
        spdlog::error("Explicit GEMM tile geometry requires private replacement");
        return EXIT_FAILURE;
    }
    if (config.persistent.replace_sgemm &&
        (!config.persistent.workers ||
         glimmer::backend::cuda::cuda_sgemm_image(config.persistent.gemm_tile_side).empty())) {
        spdlog::error(
            "SGEMM replacement requires explicit workers and the compiled CUTLASS provider");
        return EXIT_FAILURE;
    }
    if (config.uses_nccl && config.transport.max_peers >= 64) {
        spdlog::error("NCCL execution requires fewer than 64 source clients");
        return EXIT_FAILURE;
    }
    glimmer::backend::cuda::CudaSessionDispatch dispatch{
        {cuCtxGetCurrent, cuLaunchKernel, cuEventCreate, cuEventRecord, cuEventQuery,
         cuEventDestroy, cuStreamSynchronize, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemsetD8,
         cuMemsetD8Async, cuMemsetD16, cuMemsetD16Async, cuMemsetD32, cuMemsetD32Async},
        cuInit,
        cuDeviceGet,
        cuCtxCreate,
        cuCtxSetCurrent,
        cuCtxDestroy,
        cuModuleLoadData,
        cuModuleUnload,
        cuModuleGetFunction,
        cuFuncGetParamInfo,
        cuMemAlloc,
        cuMemFree,
        cuMemcpyHtoD,
        cuMemcpyDtoH,
        cuStreamCreate,
        cuStreamDestroy,
        cuMemHostAlloc,
        cuMemFreeHost,
        cuMemHostGetDevicePointer,
        cuMemcpyHtoDAsync,
        cuMemcpyDtoHAsync,
        cuOccupancyMaxActiveBlocksPerMultiprocessor,
        cuDeviceGetAttribute,
        cuCtxGetFlags,
        cuCtxSetFlags,
        cuFuncSetAttribute,
        cuMemHostRegister,
        cuMemHostUnregister,
        {cublasCreate_v2,
         cublasDestroy_v2,
         cublasSetStream_v2,
         cublasSgemm_v2,
         cublasGetStream_v2,
         cublasSetPointerMode_v2,
         cublasGetPointerMode_v2,
         cublasSetMathMode,
         cublasGetMathMode,
         cublasGetVersion_v2,
         static_cast<glimmer::backend::cuda::CudaGemmExFunction>(&cublasGemmEx),
         static_cast<glimmer::backend::cuda::CudaStridedGemmExFunction>(
             &cublasGemmStridedBatchedEx),
         cublasSetWorkspace_v2,
         cublasSetAtomicsMode,
         cublasGetAtomicsMode,
         cublasSetSmCountTarget,
         cublasGetSmCountTarget,
         cublasSgemmStridedBatched,
         cublasSgeam,
         cublasDgeam,
         cublasSgemv_v2,
         cublasSgemmEx},
        {cublasLtCreate,
         cublasLtDestroy,
         cublasLtMatrixLayoutCreate,
         cublasLtMatrixLayoutDestroy,
         cublasLtMatrixLayoutSetAttribute,
         cublasLtMatrixLayoutGetAttribute,
         cublasLtMatmulDescCreate,
         cublasLtMatmulDescDestroy,
         cublasLtMatmulDescSetAttribute,
         cublasLtMatmulDescGetAttribute,
         cublasLtMatmulPreferenceCreate,
         cublasLtMatmulPreferenceDestroy,
         cublasLtMatmulPreferenceSetAttribute,
         cublasLtMatmulPreferenceGetAttribute,
         cublasLtGetVersion,
         cublasLtGetCudartVersion,
         cublasLtMatmulAlgoGetHeuristic,
         cublasLtMatmulAlgoCheck,
         cublasLtMatmul,
         cublasLtMatmulAlgoConfigGetAttribute,
         cublasLtMatmulAlgoCapGetAttribute},
        {cudaRuntimeGetVersion, cudaGetDeviceProperties, cudaDeviceGetAttribute,
         cudaDeviceGetP2PAttribute, cudaDeviceGetStreamPriorityRange, cudaDeviceCanAccessPeer,
         cudaMemGetInfo},
        cuStreamCreateWithPriority,
        cuStreamGetPriority,
        cuModuleGetGlobal,
        cuEventRecordWithFlags,
        cuStreamWaitEvent,
        cuEventElapsedTime,
        cuEventSynchronize,
        cuDeviceGetUuid,
        cuDriverGetVersion,
        {curandCreateGenerator, curandSetPseudoRandomGeneratorSeed, curandDestroyGenerator},
        context_dispatch()};
    try {
        // The optional profiler is loaded only for a named diagnostic. Keep
        // its exact provider alive until after the service and final range pop.
        std::unique_ptr<void, RangeLibraryCloser> range_library;
        if (!sgemm_range_library.empty()) {
            range_library.reset(::dlopen(sgemm_range_library.c_str(), RTLD_NOW | RTLD_LOCAL));
            if (!range_library) throw std::runtime_error("cannot load SGEMM range provider");
            void* push = ::dlsym(range_library.get(), "nvtxRangePushA");
            void* pop = ::dlsym(range_library.get(), "nvtxRangePop");
            Dl_info push_info{}, pop_info{};
            struct stat expected {
            }, pushed{}, popped{};
            if (!push || !pop || !::dladdr(push, &push_info) || !::dladdr(pop, &pop_info) ||
                !push_info.dli_fname || !pop_info.dli_fname ||
                ::stat(sgemm_range_library.c_str(), &expected) ||
                ::stat(push_info.dli_fname, &pushed) || ::stat(pop_info.dli_fname, &popped) ||
                expected.st_dev != pushed.st_dev || expected.st_ino != pushed.st_ino ||
                expected.st_dev != popped.st_dev || expected.st_ino != popped.st_ino)
                throw std::runtime_error("SGEMM range symbols differ from selected provider");
            config.sgemm_range_hooks = glimmer::app::ExecutionSgemmRangeHooks{
                reinterpret_cast<int (*)(const char*)>(push), reinterpret_cast<int (*)()>(pop)};
        }
        dispatch.execution.copy_unified = cuMemcpy;
        dispatch.execution.copy_unified_async = cuMemcpyAsync;
        dispatch.context_synchronize = cuCtxSynchronize;
        dispatch.function_get_attribute = cuFuncGetAttribute;
        dispatch.function_set_cache = cuFuncSetCacheConfig;
        dispatch.pointer_get_attribute = cuPointerGetAttribute;
        dispatch.module_load_ex = cuModuleLoadDataEx;
        dispatch.process_memory = {cuInit,
                                   cuCtxGetCurrent,
                                   cuDeviceGetDefaultMemPool,
                                   cuDeviceGetMemPool,
                                   cuDeviceSetMemPool,
                                   cuMemAllocAsync,
                                   cuMemAllocFromPoolAsync,
                                   cudaMemPoolTrimTo,
                                   cuMemFree};
        dispatch.launch_host_function = cuLaunchHostFunc;
        dispatch.graph = {cudaStreamBeginCapture, cudaStreamEndCapture,
                          cudaStreamIsCapturing,  cudaGraphInstantiateWithFlags,
                          cudaGraphDestroy,       cudaGraphExecDestroy,
                          cuGraphLaunch};
        dispatch.runtime.pointer_attributes = cudaPointerGetAttributes;
        if (uses_pairs || uses_space_share) {
            glimmer::backend::cuda::CudaExecutionDomainDispatch domain_dispatch{
                dispatch.initialize,
                dispatch.device_get,
                dispatch.context_create,
                dispatch.context_set_current,
                dispatch.execution.context_get_current,
                dispatch.context_destroy};
            domain_dispatch.configuration = {
                dispatch.context_get_flags,  dispatch.context_set_flags, dispatch.context.get_limit,
                dispatch.context.set_limit,  dispatch.context.get_cache, dispatch.context.set_cache,
                dispatch.context.get_shared, dispatch.context.set_shared};
            auto domain = std::make_shared<glimmer::backend::cuda::CudaExecutionDomainOwner>(
                domain_dispatch,
                glimmer::backend::cuda::CudaExecutionDomainConfig{
                    1, config.device_ordinal, CU_CTX_MAP_HOST, config.transport.max_peers,
                    config.session_limits.max_device_bytes * config.transport.max_peers});
            if (uses_pairs)
                config.paired->domain = std::move(domain);
            else
                config.shared->domain = std::move(domain);
        }
#ifdef GLIMMER_HAS_CUDNN
        if (!glimmer::internal::cudnn_profile::process_providers_match()) {
            spdlog::error("Loaded cuDNN provider does not match configured SDK identity");
            return EXIT_FAILURE;
        }
        const auto dnn_version = cudnnGetVersion();
        if (dnn_version != CUDNN_VERSION ||
            !glimmer::internal::cudnn_profile::process_providers_match()) {
            spdlog::error("cuDNN profile mismatch: headers={}, native={}", CUDNN_VERSION,
                          dnn_version);
            return EXIT_FAILURE;
        }
        dispatch.dnn =
            std::make_shared<const glimmer::backend::cuda::CudaDnnDispatch>(dnn_dispatch());
        dispatch.verified_dnn_header_version = CUDNN_VERSION;
#endif
        struct sigaction action {};
        action.sa_handler = stop_requested;
        if (::sigemptyset(&action.sa_mask) != 0 || ::sigaction(SIGINT, &action, nullptr) != 0 ||
            ::sigaction(SIGTERM, &action, nullptr) != 0) {
            spdlog::error("Cannot install execution-service stop handlers");
            return EXIT_FAILURE;
        }
        const auto scheduler_name =
            config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kNaive     ? "naive"
            : config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kProfile ? "profile"
            : config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic
                ? "workload-agnostic"
            : config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kWorkloadAware
                ? "workload-aware"
            : uses_space_share ? "priority-space-share"
                               : "priority";
        constexpr std::array kind_names{"static", "block-static", "dynamic", "preemptive",
                                        "sliced"};
        const auto kernel_mode =
            config.uses_original_kernels || !config.persistent.workers
                ? "original"
                : kind_names.at(static_cast<std::size_t>(config.persistent.kind));
        const auto workers = config.persistent.workers;
        const bool uses_sharded_gemm_signal = config.persistent.uses_sharded_host_retreat;
        if (glimmer::core::is_priority_execution_mode(config.scheduling_mode) &&
            !config.client_priorities.empty())
            spdlog::info("Priority idle interval: {} us after highest-priority drain",
                         config.priority_idle_us);
        const bool saves_pairs = uses_pairs && config.paired->is_isolated_profile;
        std::shared_ptr<glimmer::app::HostStageMetrics> host_stages;
        std::shared_ptr<glimmer::app::IssueDepthMetrics> issue_depth;
        std::shared_ptr<glimmer::control::ExecutionSocketPollJournal> poll_journal;
        if (!poll_journal_output.empty()) {
            poll_journal = std::make_shared<glimmer::control::ExecutionSocketPollJournal>(
                glimmer::control::ExecutionSocketPollJournal::Limits{524288, 3145728, 524288,
                                                                     1572864, 524288});
            config.transport.should_record_receive_time = true;
        }
        if (!issue_depth_output.empty()) {
            issue_depth = std::make_shared<glimmer::app::IssueDepthMetrics>(
                glimmer::app::IssueDepthMetrics::system_clock,
                should_record_issue_events ? glimmer::app::IssueDepthMetrics::event_capacity_limit
                                           : 0,
                glimmer::app::IssueDepthMetrics::SelectionConfig{
                    should_record_issue_selections
                        ? glimmer::app::IssueDepthMetrics::selection_capacity_limit
                        : 0});
            config.issue_depth_metrics = issue_depth;
        }
        if (!host_stage_output.empty()) {
            host_stages = std::make_shared<glimmer::app::HostStageMetrics>(
                glimmer::app::HostStageMetrics::system_clock,
                glimmer::app::HostStageMetrics::lifecycle_capacity_limit);
            std::uint32_t highest{};
            for (const auto& pin : config.client_priorities)
                highest = std::max(highest, pin.priority);
            host_stages->set_idle_enabled(
                glimmer::core::is_priority_execution_mode(config.scheduling_mode) && highest != 0 &&
                config.priority_idle_us != 0);
            config.host_stage_metrics = host_stages;
        }
        if (uses_pairs)
            spdlog::info(
                "Workload-aware selection: isolated_pair_profile={}, pair_worker_choices={}, "
                "arrival_checkpoints={}; resource geometry is not SM participation or co-residency",
                saves_pairs, config.paired->workers.size(),
                config.paired->should_rebalance_on_arrival);
        if (uses_sharing)
            spdlog::info(
                "Workload-agnostic selection: thread_budget_per_sm={}, minimum_speed_percent={}, "
                "readonly_profile=true; resource geometry is not SM participation or co-residency",
                config.persistent.sharing_threads_per_sm,
                config.persistent.sharing_minimum_speed_percent);
        if (config.scheduling_mode == glimmer::core::ExecutionSchedulingMode::kProfile) {
            spdlog::info(
                "Isolated profile mode: independently terminating kernels only; once-only "
                "requests, host-observed elapsed time, external GPU isolation unverified");
            std::string families;
            for (const auto kind : config.profile.kinds) {
                if (!families.empty()) families += ',';
                families += kind_names.at(static_cast<std::size_t>(kind));
            }
            spdlog::info("Profile candidates: families={}, worker_choices={}, original=true",
                         families.empty() ? kernel_mode : families, config.profile.workers.size());
        }
        const auto uses_nccl = config.uses_nccl;
        std::vector<BlockCompletionPin> completion_pins;
        const bool observes_completion = config.persistent.observes_block_completion;
        if (observes_completion) {
            completion_pins.reserve(config.client_priorities.size());
            for (const auto& client : config.client_priorities)
                if (client.kernel_mode != glimmer::app::ExecutionClientKernelMode::kOriginal)
                    completion_pins.push_back({client});
        }
#ifdef GLIMMER_ENABLE_NCCL_EXECUTION
        auto service =
            poll_journal
                ? glimmer::app::ExecutionService(poll_journal, std::move(config), dispatch,
                                                 nccl_dispatch())
                : glimmer::app::ExecutionService(std::move(config), dispatch, nccl_dispatch());
#else
        auto service =
            poll_journal ? glimmer::app::ExecutionService(poll_journal, std::move(config), dispatch)
                         : glimmer::app::ExecutionService(std::move(config), dispatch);
#endif
        if (!service.start()) {
            spdlog::error("Cannot start authenticated execution service");
            return EXIT_FAILURE;
        }
        const auto identity = glimmer::control::current_process_identity();
        if (!identity) {
            spdlog::error("Cannot identify authenticated execution-service process");
            return EXIT_FAILURE;
        }
        spdlog::info(
            "Execution service configuration: scheduler={}, kernel_mode={}, configured_workers={}, "
            "nccl_enabled={}, sharded_gemm_signal={}; "
            "full_device_qualified=false",
            scheduler_name, kernel_mode, workers, uses_nccl, uses_sharded_gemm_signal);
        spdlog::info(
            "Execution service ready: pid={}, start={}; routing=explicit_opt_in, "
            "application_qualified=false",
            identity->pid, identity->start_time_ticks);
        spdlog::default_logger()->flush();
        while (!should_stop.load(std::memory_order_relaxed) && !service.finished()) {
            if (!service.step()) {
                spdlog::error("Execution-service owner loop failed");
                return EXIT_FAILURE;
            }
            if (observes_completion) observe_blocks(service, completion_pins);
        }
        const bool finished_normally =
            !should_stop.load(std::memory_order_relaxed) && service.finished();
        if (!service.request_stop()) {
            spdlog::critical("Execution service cannot prove complete native shutdown");
            return EXIT_FAILURE;
        }
        if (host_stages && !host_stages->publish(host_stage_output))
            spdlog::warn(
                "Host-stage diagnostic publication failed; application execution is unchanged");
        if (issue_depth) {
            issue_depth->finish();
            if (!issue_depth->publish(issue_depth_output))
                spdlog::warn(
                    "Issue-depth diagnostic publication failed; application execution is "
                    "unchanged");
        }
        if (!service.audit_healthy()) {
            spdlog::error("Execution audit evidence was lost; application results are unchanged");
            return EXIT_FAILURE;
        }
        if (!service.save_profile()) {
            spdlog::error("Cannot publish profile archive; existing files are never overwritten");
            return EXIT_FAILURE;
        }
        if (!service.save_ptx_cache()) {
            spdlog::error("Cannot publish PTX archive; existing files are never overwritten");
            return EXIT_FAILURE;
        }
        if (saves_pairs && !service.save_pair_profile()) {
            spdlog::error(
                "Cannot publish complete pair profile archive; existing files are never "
                "overwritten");
            return EXIT_FAILURE;
        }
        const auto cache = service.ptx_cache_snapshot();
        if (cache.is_valid)
            spdlog::info(
                "PTX preparation cache: lookups={}, hits={}, transforms={}, evictions={}, "
                "bypasses={}, retained_bytes={}, loaded_entries={}; "
                "performance_qualified=false",
                cache.lookups, cache.hits, cache.transforms, cache.evictions, cache.bypasses,
                cache.retained_bytes, cache.loaded_entries);
        const auto profile = service.profile_snapshot();
        if (profile.is_valid)
            spdlog::info(
                "Kernel profile cache: functions={}, cases={}, trained={}, samples={}, "
                "rejected={}, cancelled={}, hits={}, bypasses={}, retained_bytes={}, "
                "loaded_cases={}, learning={}; "
                "timing=host_observed_ns; performance_qualified=false",
                profile.functions, profile.cases, profile.trained_cases, profile.samples,
                profile.rejected_samples, profile.cancelled, profile.hits, profile.bypasses,
                profile.retained_bytes, profile.loaded_cases, profile.is_learning);
        const auto pairs = service.pair_profile_snapshot();
        if (pairs.is_valid)
            spdlog::info(
                "Kernel pair profile cache: pairs={}, trained={}, samples={}, rejected={}, "
                "cancelled={}, active={}, loaded_pairs={}, learning={}; "
                "performance_qualified=false",
                pairs.pairs, pairs.trained_pairs, pairs.samples, pairs.rejected_samples,
                pairs.cancelled, pairs.active, pairs.loaded_pairs, pairs.is_learning);
        if (!should_stop.load(std::memory_order_relaxed)) {
            if (const auto count = service.async_intake_second_takes()) {
                spdlog::info(
                    "Async intake receipt: schema=1, pid={}, start={}, enabled=1, "
                    "successful_second_takes={}",
                    identity->pid, identity->start_time_ticks, *count);
                spdlog::default_logger()->flush();
            }
        }
        if (poll_journal) {
            if (!finished_normally || should_stop.load(std::memory_order_relaxed) ||
                !poll_journal->valid() || !poll_journal->publish(poll_journal_output)) {
                spdlog::error(
                    "Poll journal requires complete normal shutdown and checked publication");
                return EXIT_FAILURE;
            }
            // This is a checked terminal protocol receipt, not a best-effort log.
            // Direct writes have no userspace buffer left to flush at process exit.
            const auto receipt = fmt::format(
                "Poll journal receipt: schema=1, pid={}, start={}, complete=1, "
                "cycles={}, polls={}, peers={}, reads={}, frames={}\n",
                identity->pid, identity->start_time_ticks, poll_journal->cycles().size(),
                poll_journal->polls().size(), poll_journal->peers().size(),
                poll_journal->reads().size(), poll_journal->frames().size());
            std::size_t written{};
            while (written < receipt.size()) {
                const auto count =
                    ::write(STDOUT_FILENO, receipt.data() + written, receipt.size() - written);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return EXIT_FAILURE;
                written += static_cast<std::size_t>(count);
            }
            if (should_stop.load(std::memory_order_relaxed)) return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        spdlog::error("Execution service failed: {}", error.what());
        return EXIT_FAILURE;
    }
}
