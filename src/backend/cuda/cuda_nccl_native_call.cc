#include "internal/cuda_nccl_native_call.h"
#include "internal/cuda_native_output_cell.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(ncclComm_t) == 8);
constexpr std::array<std::size_t, 14> kScalarOffsets{16, 20, 24, 28, 40, 44, 56,
                                                     60, 64, 68, 72, 76, 80, 84};
static_assert(offsetof(ncclConfig_t, cgaClusterSize) == 20 &&
              offsetof(ncclConfig_t, minCTAs) == 24 && offsetof(ncclConfig_t, maxCTAs) == 28 &&
              offsetof(ncclConfig_t, splitShare) == 40 &&
              offsetof(ncclConfig_t, trafficClass) == 44 &&
              offsetof(ncclConfig_t, collnetEnable) == 56 &&
              offsetof(ncclConfig_t, CTAPolicy) == 60 &&
              offsetof(ncclConfig_t, shrinkShare) == 64 && offsetof(ncclConfig_t, nvlsCTAs) == 68 &&
              offsetof(ncclConfig_t, nChannelsPerNetPeer) == 72 &&
              offsetof(ncclConfig_t, nvlinkCentricSched) == 76 &&
              offsetof(ncclConfig_t, graphUsageMode) == 80);

thread_local std::shared_ptr<const void> nccl_call_thread;

class CallScope final {
   public:
    explicit CallScope(std::atomic_flag& active) noexcept
        : active_(active), acquired_(!active.test_and_set(std::memory_order_acquire)) {}
    ~CallScope() {
        if (acquired_) active_.clear(std::memory_order_release);
    }
    CallScope(const CallScope&) = delete;
    CallScope& operator=(const CallScope&) = delete;
    [[nodiscard]] bool acquired() const noexcept {
        return acquired_;
    }

   private:
    std::atomic_flag& active_;
    bool acquired_;
};

bool has_callback(const CudaNcclNativeDispatch& dispatch, CudaNcclNativeOperation operation) {
    switch (operation) {
        case CudaNcclNativeOperation::kGetUniqueId:
            return dispatch.get_unique_id != nullptr;
        case CudaNcclNativeOperation::kInitRank:
            return dispatch.init_rank != nullptr;
        case CudaNcclNativeOperation::kInitRankConfig:
            return dispatch.init_rank_config != nullptr;
        case CudaNcclNativeOperation::kGetAsyncError:
            return dispatch.get_async_error != nullptr;
        case CudaNcclNativeOperation::kDestroy:
            return dispatch.destroy != nullptr;
        case CudaNcclNativeOperation::kAbort:
            return dispatch.abort != nullptr;
        case CudaNcclNativeOperation::kGroupStart:
            return dispatch.group_start != nullptr;
        case CudaNcclNativeOperation::kGroupEnd:
            return dispatch.group_end != nullptr;
        case CudaNcclNativeOperation::kAllGather:
            return dispatch.all_gather != nullptr;
        case CudaNcclNativeOperation::kAllReduce:
            return dispatch.all_reduce != nullptr;
        case CudaNcclNativeOperation::kBcast:
            return dispatch.bcast != nullptr;
    }
    return false;
}
struct ConfigScalar {
    std::uint32_t value;
    std::size_t offset;
};
bool scalar_prefix(ConfigScalar scalar, std::size_t prefix) {
    const auto copied =
        prefix <= scalar.offset ? 0 : std::min(prefix - scalar.offset, std::size_t{4});
    if (copied == 4) return true;
    const auto mask = (std::uint32_t{1} << (copied * 8)) - 1;
    return (scalar.value & ~mask) == 0;
}
bool valid_config(const CudaNcclNativeConfig& config) {
    if (config.declared_bytes > std::numeric_limits<std::size_t>::max()) return false;
    const auto prefix = static_cast<std::size_t>(std::min(config.declared_bytes, UINT64_C(88)));
    if ((prefix > 32 && prefix < 40) || (prefix > 48 && prefix < 56)) return false;
    if (!scalar_prefix({config.magic, 8}, prefix) || !scalar_prefix({config.version, 12}, prefix))
        return false;
    for (std::size_t index = 0; index != kScalarOffsets.size(); ++index)
        if (!scalar_prefix(
                {std::bit_cast<std::uint32_t>(config.fields[index]), kScalarOffsets[index]},
                prefix))
            return false;
    const auto string_is_valid = [](const std::optional<std::string>& value) {
        return !value || (value->size() <= kCudaNcclNativeConfigStringLimit &&
                          value->find('\0') == std::string::npos);
    };
    return (!config.net_name || prefix >= 40) && (!config.comm_name || prefix >= 56) &&
           string_is_valid(config.net_name) && string_is_valid(config.comm_name);
}
bool is_collective(CudaNcclNativeOperation operation) {
    return operation == CudaNcclNativeOperation::kAllGather ||
           operation == CudaNcclNativeOperation::kAllReduce ||
           operation == CudaNcclNativeOperation::kBcast;
}
}  // namespace

struct CudaNcclNativeCall::Implementation {
    const pid_t process{::getpid()};
    const CudaNcclNativeDispatch dispatch;
    const CudaNcclNativeArguments arguments;
    ncclUniqueId input_id{}, output_id{};
    ncclConfig_t config{};
    ncclComm_t output_communicator{};
    std::array<std::byte, sizeof(ncclComm_t)> initial_communicator{};
    alignas(ncclResult_t) std::array<std::byte, sizeof(ncclResult_t)> output_async{};
    std::weak_ptr<const void> worker;
    bool has_worker{};
    std::atomic_flag active = ATOMIC_FLAG_INIT;
    std::atomic<bool> invoked{}, returned{};
    std::int32_t original_status{};
    std::shared_ptr<CudaNativeOutputCell> output_cell;

    Implementation(CudaNcclNativeDispatch callbacks, CudaNcclNativeArguments request)
        : dispatch(callbacks), arguments(std::move(request)) {
        std::memcpy(&input_id, arguments.unique_id.data(), arguments.unique_id.size());
        std::memset(&output_id, 0xa5, sizeof(output_id));
        output_async.fill(std::byte{0xa5});
        std::memcpy(initial_communicator.data(), &output_communicator, initial_communicator.size());
        if (arguments.config) {
            const auto& source = *arguments.config;
            config.size = static_cast<std::size_t>(source.declared_bytes);
            config.magic = source.magic;
            config.version = source.version;
            auto* bytes = reinterpret_cast<std::byte*>(&config);
            for (std::size_t index = 0; index != kScalarOffsets.size(); ++index)
                std::memcpy(bytes + kScalarOffsets[index], &source.fields[index],
                            sizeof(source.fields[index]));
            config.netName = source.net_name ? source.net_name->c_str() : nullptr;
            config.commName = source.comm_name ? source.comm_name->c_str() : nullptr;
        }
    }
    [[nodiscard]] bool owns_worker() const noexcept {
        return has_worker && nccl_call_thread && worker.lock() == nccl_call_thread;
    }
    CUresult prepare_output() {
        if (arguments.output_contract != CudaNcclNativeOutputContract::kPinnedHostStable ||
            !arguments.has_output)
            return CUDA_SUCCESS;
        std::span<const std::byte> initial;
        switch (arguments.operation) {
            case CudaNcclNativeOperation::kGetUniqueId:
                initial = std::as_bytes(std::span{&output_id, 1});
                break;
            case CudaNcclNativeOperation::kInitRank:
            case CudaNcclNativeOperation::kInitRankConfig:
                initial = initial_communicator;
                break;
            case CudaNcclNativeOperation::kGetAsyncError:
                initial = output_async;
                break;
            default:
                return CUDA_SUCCESS;
        }
        auto created = CudaNativeOutputCell::create(initial);
        if (created.status != CUDA_SUCCESS) return created.status;
        output_cell = std::move(created.cell);
        // Start trivial typed object lifetimes without value-initializing: only
        // the sealed backing seed may initialize this isolated output page.
        switch (arguments.operation) {
            case CudaNcclNativeOperation::kGetUniqueId:
                ::new (output_cell->data()) ncclUniqueId;
                break;
            case CudaNcclNativeOperation::kInitRank:
            case CudaNcclNativeOperation::kInitRankConfig:
                ::new (output_cell->data()) ncclComm_t;
                break;
            case CudaNcclNativeOperation::kGetAsyncError:
                ::new (output_cell->data()) ncclResult_t;
                break;
            default:
                return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }
    ncclResult_t issue() {
        const auto& request = arguments;
        auto* communicator = request.has_output
                                 ? (output_cell ? static_cast<ncclComm_t*>(output_cell->data())
                                                : &output_communicator)
                                 : nullptr;
        switch (request.operation) {
            case CudaNcclNativeOperation::kGetUniqueId:
                return dispatch.get_unique_id(
                    request.has_output
                        ? (output_cell ? static_cast<ncclUniqueId*>(output_cell->data())
                                       : &output_id)
                        : nullptr);
            case CudaNcclNativeOperation::kInitRank:
                return dispatch.init_rank(communicator, request.ranks, input_id, request.rank);
            case CudaNcclNativeOperation::kInitRankConfig:
                return dispatch.init_rank_config(communicator, request.ranks, input_id,
                                                 request.rank, request.config ? &config : nullptr);
            case CudaNcclNativeOperation::kGetAsyncError:
                return dispatch.get_async_error(
                    request.communicator.native,
                    request.has_output
                        ? (output_cell ? static_cast<ncclResult_t*>(output_cell->data())
                                       : reinterpret_cast<ncclResult_t*>(output_async.data()))
                        : nullptr);
            case CudaNcclNativeOperation::kDestroy:
                return dispatch.destroy(request.communicator.native);
            case CudaNcclNativeOperation::kAbort:
                return dispatch.abort(request.communicator.native);
            case CudaNcclNativeOperation::kGroupStart:
                return dispatch.group_start();
            case CudaNcclNativeOperation::kGroupEnd:
                return dispatch.group_end();
            case CudaNcclNativeOperation::kAllGather:
                return dispatch.all_gather(
                    request.send, request.receive, static_cast<std::size_t>(request.count),
                    request.datatype, request.communicator.native, request.stream);
            case CudaNcclNativeOperation::kAllReduce:
                return dispatch.all_reduce(request.send, request.receive,
                                           static_cast<std::size_t>(request.count),
                                           request.datatype, request.reduction,
                                           request.communicator.native, request.stream);
            case CudaNcclNativeOperation::kBcast:
                return dispatch.bcast(request.receive, static_cast<std::size_t>(request.count),
                                      request.datatype, request.root, request.communicator.native,
                                      request.stream);
        }
        // create() rejects unknown operations before any packet is published.
        throw std::logic_error("unvalidated NCCL packet operation");
    }
};

CudaNcclNativeCall::CudaNcclNativeCall(std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {}
CudaNcclNativeCall::~CudaNcclNativeCall() = default;

CudaNcclNativeCallCreation CudaNcclNativeCall::create(CudaNcclNativeDispatch dispatch,
                                                      CudaNcclNativeArguments arguments) noexcept {
    const auto operation = static_cast<std::uint32_t>(arguments.operation);
    if (operation < 1 || operation > 11 ||
        (arguments.output_contract != CudaNcclNativeOutputContract::kDeferred &&
         arguments.output_contract != CudaNcclNativeOutputContract::kPinnedHostStable) ||
        arguments.count > std::numeric_limits<std::size_t>::max() ||
        (arguments.config && !valid_config(*arguments.config)))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!has_callback(dispatch, arguments.operation)) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (arguments.communicator.native && !arguments.communicator.lifetime)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (is_collective(arguments.operation) && !arguments.resources_lifetime)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    try {
        auto implementation = std::make_unique<Implementation>(dispatch, std::move(arguments));
        const auto prepared = implementation->prepare_output();
        if (prepared != CUDA_SUCCESS) return {.status = prepared};
        auto call =
            std::shared_ptr<CudaNcclNativeCall>(new CudaNcclNativeCall(std::move(implementation)));
        return {CUDA_SUCCESS, std::move(call)};
    } catch (const std::bad_alloc&) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    } catch (...) {
        return {.status = CUDA_ERROR_UNKNOWN};
    }
}

CUresult CudaNcclNativeCall::invoke() noexcept {
    auto& state = *implementation_;
    if (state.process != ::getpid()) return CUDA_ERROR_INVALID_CONTEXT;
    const CallScope scope(state.active);
    if (!scope.acquired()) return CUDA_ERROR_NOT_READY;
    if (state.has_worker && !state.owns_worker()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.invoked.load(std::memory_order_acquire)) return CUDA_ERROR_INVALID_VALUE;
    try {
        if (!nccl_call_thread) nccl_call_thread = std::make_shared<int>(0);
        state.worker = nccl_call_thread;
        state.has_worker = true;
        state.invoked.store(true, std::memory_order_release);
        state.original_status = static_cast<std::int32_t>(state.issue());
        state.returned.store(true, std::memory_order_release);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return state.invoked.load(std::memory_order_acquire) ? CUDA_ERROR_UNKNOWN
                                                             : CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
std::optional<std::int32_t> CudaNcclNativeCall::requested_status() const noexcept {
    const auto& state = *implementation_;
    if (state.process != ::getpid() || !state.returned.load(std::memory_order_acquire))
        return std::nullopt;
    return state.original_status;
}
bool CudaNcclNativeCall::was_invoked() const noexcept {
    const auto& state = *implementation_;
    return state.process == ::getpid() && state.invoked.load(std::memory_order_acquire);
}
bool CudaNcclNativeCall::has_stable_output() const noexcept {
    const auto& state = *implementation_;
    return state.process == ::getpid() &&
           state.arguments.output_contract == CudaNcclNativeOutputContract::kPinnedHostStable;
}
std::optional<CudaNcclNativeObservation> CudaNcclNativeCall::snapshot_after_host_return() noexcept {
    if (!has_stable_output()) return std::nullopt;
    return snapshot_after_quiescence();
}
std::optional<CudaNcclNativeObservation> CudaNcclNativeCall::snapshot_after_quiescence() noexcept {
    auto& state = *implementation_;
    if (state.process != ::getpid()) return std::nullopt;
    const CallScope scope(state.active);
    if (!scope.acquired() || !state.owns_worker() ||
        !state.returned.load(std::memory_order_acquire))
        return std::nullopt;
    CudaNcclNativeObservation result{.operation = state.arguments.operation};
    if (!state.arguments.has_output) return result;
    if (state.output_cell) {
        std::span<std::byte> bytes;
        switch (state.arguments.operation) {
            case CudaNcclNativeOperation::kGetUniqueId:
                bytes = std::as_writable_bytes(std::span{&state.output_id, 1});
                break;
            case CudaNcclNativeOperation::kInitRank:
            case CudaNcclNativeOperation::kInitRankConfig:
                bytes = std::as_writable_bytes(std::span{&state.output_communicator, 1});
                break;
            case CudaNcclNativeOperation::kGetAsyncError:
                bytes = state.output_async;
                break;
            default:
                return std::nullopt;
        }
        const auto observed = state.output_cell->snapshot(bytes);
        if (observed.status != CUDA_SUCCESS) return std::nullopt;
        result.output_written = observed.was_written;
    }
    switch (state.arguments.operation) {
        case CudaNcclNativeOperation::kGetUniqueId:
            result.has_output = true;
            std::memcpy(result.unique_id.data(), &state.output_id, result.unique_id.size());
            for (std::size_t index = 0; index != result.unique_id.size(); ++index)
                if (result.unique_id[index] != std::byte{0xa5})
                    result.unique_id_changed[index / 8] |=
                        static_cast<std::byte>(1U << (index % 8));
            break;
        case CudaNcclNativeOperation::kInitRank:
        case CudaNcclNativeOperation::kInitRankConfig: {
            result.has_output = true;
            std::array<std::byte, sizeof(ncclComm_t)> bytes{};
            std::memcpy(bytes.data(), &state.output_communicator, bytes.size());
            std::memcpy(&result.communicator, bytes.data(), bytes.size());
            for (std::size_t index = 0; index != bytes.size(); ++index)
                if (bytes[index] != state.initial_communicator[index])
                    result.communicator_changed |= 1U << index;
            break;
        }
        case CudaNcclNativeOperation::kGetAsyncError:
            result.has_output = true;
            std::memcpy(&result.async_error, state.output_async.data(), sizeof(result.async_error));
            for (std::size_t index = 0; index != state.output_async.size(); ++index)
                if (state.output_async[index] != std::byte{0xa5})
                    result.async_error_changed |= 1U << index;
            break;
        default:
            break;
    }
    return result;
}

}  // namespace glimmer::backend::cuda
