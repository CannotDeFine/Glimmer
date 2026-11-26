#include "runtime.h"
#include "host_stage_metrics.h"
#include "business_owner_lease.h"
#include "async_audit_log.h"
#include "cleanup_idle.h"
#include "internal/cuda_gemm_geometry.h"
#include "internal/cuda_gemm_signal_layout.h"
#include "internal/cuda_sgemm_image.h"
#include "internal/cuda_retirement_observer.h"
#include "host_owner.h"
#include "profile_archive.h"
#ifdef GLIMMER_HAS_NCCL
#include "nccl_actor.h"
#include "nccl_output_effects.h"
#endif
#include "glimmer/control/execution_blas_protocol.h"
#include "glimmer/control/execution_function_protocol.h"
#include "glimmer/control/execution_jit_protocol.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/control/execution_process_memory_protocol.h"
#include "glimmer/control/execution_graph_protocol.h"
#include "glimmer/core/execution_priority.h"
#include "glimmer/transform/cuda_module_image.h"
#ifdef GLIMMER_HAS_CUDNN
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "glimmer/control/execution_dnn_legacy_protocol.h"
#include "internal/cuda_dnn_execution.h"
#include "internal/cuda_dnn_legacy.h"
#include "internal/cuda_dnn_backend_execution.h"
#include "../internal/cudnn_sdk_profile.h"
#endif

#include <algorithm>
#include <bit>
#include <charconv>
#include <climits>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::app {
namespace {
std::shared_ptr<control::ExecutionSocketPollJournal> require_poll_journal(
    std::shared_ptr<control::ExecutionSocketPollJournal> journal) {
    if (!journal) throw std::invalid_argument("diagnostic service requires a poll journal");
    return journal;
}
ExecutionServiceConfig execution_configuration(ExecutionServiceConfig config) {
    if (config.should_audit_copy_requests ||
        (config.host_stage_metrics && config.host_stage_metrics->records_intake()))
        config.transport.should_record_receive_time = true;
    if (config.scheduling_mode == core::ExecutionSchedulingMode::kWorkloadAware &&
        (!config.paired ||
         config.paired->should_rebalance_on_arrival == config.paired->is_isolated_profile))
        throw std::invalid_argument(
            "workload-aware scheduling requires paired selection and ordinary arrival checkpoints");
    const bool uses_sharing =
        config.scheduling_mode == core::ExecutionSchedulingMode::kWorkloadAgnostic;
    if (config.persistent.uses_workload_agnostic && !uses_sharing)
        throw std::invalid_argument("sharing selection requires the workload-agnostic scheduler");
    config.persistent.uses_workload_agnostic = uses_sharing;
    if (!config.transport.wake)
        config.transport.wake = std::make_shared<control::ExecutionSocketWake>();
    if (config.transport.wake->descriptor() < 0)
        throw std::invalid_argument("execution wake belongs to a different process");
    return config;
}
using namespace control;
using namespace backend;
using namespace backend::cuda;
using HostStage = HostStageMetrics::Stage;
using HostScope = HostStageMetrics::Scope;
thread_local std::shared_ptr<const void> service_owner;
// Bind the pidfd to the transport's authenticated incarnation, not merely its
// reusable PID. Opening the pidfd first closes the exit/reuse race with /proc.
bool memory_peer_matches(const TaskPeerIdentity& peer) noexcept {
    try {
        const auto path = "/proc/" + std::to_string(peer.pid) + "/stat";
        ExecutionOwnedDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
        if (file.get() < 0 || !peer.start_time_ticks) return false;
        std::array<char, 4096> bytes{};
        const auto count = ::read(file.get(), bytes.data(), bytes.size());
        if (count <= 0 || static_cast<std::size_t>(count) == bytes.size()) return false;
        std::string_view input(bytes.data(), static_cast<std::size_t>(count));
        const auto end = input.rfind(')');
        if (end == std::string_view::npos || end + 2 >= input.size()) return false;
        input.remove_prefix(end + 2);
        for (unsigned int field = 3; field <= 22; ++field) {
            const auto next = input.find(' ');
            if (next == std::string_view::npos) return false;
            if (field == 22) {
                std::uint64_t ticks{};
                const auto result = std::from_chars(input.data(), input.data() + next, ticks);
                return result.ec == std::errc{} && result.ptr == input.data() + next &&
                       ticks == peer.start_time_ticks;
            }
            input.remove_prefix(next + 1);
        }
    } catch (...) {
    }
    return false;
}
// Only constant field names and integral diagnostic values enter this bounded
// builder. It never formats native pointers, caller strings or request payloads.
class SemanticAuditJson final {
   public:
    SemanticAuditJson() noexcept {
        append("{");
    }
    template <class Value>
    void add(std::string_view name, Value value) noexcept {
        static_assert(std::is_integral_v<Value>);
        if (has_field_) append(",");
        append("\"");
        append(name);
        append("\":");
        using Number = std::conditional_t<std::is_signed_v<Value>, std::int64_t, std::uint64_t>;
        if (!is_valid_) return;
        const auto result = std::to_chars(bytes_.data() + size_, bytes_.data() + bytes_.size(),
                                          static_cast<Number>(value));
        if (result.ec != std::errc{}) {
            is_valid_ = false;
            return;
        }
        size_ = static_cast<std::size_t>(result.ptr - bytes_.data());
        has_field_ = true;
    }
    [[nodiscard]] std::optional<std::string_view> finish() noexcept {
        append("}");
        return is_valid_ ? std::optional<std::string_view>{{bytes_.data(), size_}} : std::nullopt;
    }

   private:
    void append(std::string_view value) noexcept {
        if (!is_valid_ || value.size() > bytes_.size() - size_) {
            is_valid_ = false;
            return;
        }
        std::memcpy(bytes_.data() + size_, value.data(), value.size());
        size_ += value.size();
    }
    std::array<char, 8192> bytes_{};
    std::size_t size_{};
    bool has_field_{}, is_valid_{true};
};
// The request owns its descriptor across cold-gate deferrals. This temporary
// CPU view cannot outlive dispatch; successful backend intake owns another copy.
struct ModuleFileMapping {
    void* pointer{MAP_FAILED};
    std::size_t bytes{};
    ~ModuleFileMapping() {
        if (pointer != MAP_FAILED && ::munmap(pointer, bytes) != 0)
            spdlog::error("Execution module view cleanup failed: errno={}", errno);
    }
    ModuleFileMapping() = default;
    ModuleFileMapping(const ModuleFileMapping&) = delete;
    ModuleFileMapping& operator=(const ModuleFileMapping&) = delete;
    CUresult open(int descriptor, std::uint64_t count) noexcept {
        if (descriptor < 0 || !count || count > transform::kMaxCudaModuleBytes)
            return CUDA_ERROR_INVALID_VALUE;
        struct stat info {};
        constexpr auto required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK;
        const auto seals = ::fcntl(descriptor, F_GET_SEALS);
        if (seals < 0 || (seals & required_seals) != required_seals ||
            ::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
            static_cast<std::uint64_t>(info.st_size) != count)
            return CUDA_ERROR_INVALID_VALUE;
        bytes = static_cast<std::size_t>(count);
        pointer = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (pointer == MAP_FAILED)
            return errno == ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    }
};
ExecutionWireReply rejection(CUresult status) {
    ExecutionWireReply result;
    result.status = static_cast<std::int32_t>(status);
    return result;
}
ExecutionWireReply project(const CudaSessionResult& source) {
    ExecutionWireReply result;
    result.disposition = source.disposition == CudaSessionDisposition::kComplete
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    if (source.original_status)
        result.original_status = static_cast<std::int32_t>(*source.original_status);
    result.id = source.id;
    result.address = source.address;
    result.parameter_bytes = source.parameter_bytes;
    result.arguments.reserve(source.arguments.size());
    for (const auto& argument : source.arguments)
        result.arguments.push_back({argument.offset, argument.bytes});
    return result;
}
ExecutionWireReply project(const CudaSessionGraphResult& source) {
    ExecutionWireReply result;
    result.disposition = source.status == CUDA_SUCCESS && !source.is_quarantined
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    result.original_status = source.original_status;
    result.id = source.id;
    ExecutionPayloadWriter observation(kExecutionGraphObservationBytes);
    if (!write_execution_graph_observation(
            observation,
            {source.output_written, source.is_quarantined, source.capture_status, source.output}))
        return rejection(CUDA_ERROR_OUT_OF_MEMORY);
    result.data = observation.take();
    return result;
}
ExecutionWireReply project(const CudaSessionBlasResult& source) {
    ExecutionWireReply result;
    result.disposition = source.disposition == CudaSessionDisposition::kComplete
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    if (source.original_status) {
        result.original_status = static_cast<std::int32_t>(*source.original_status);
        if (source.status == CUDA_SUCCESS) result.status = *result.original_status;
    }
    result.id = source.id;
    result.address = source.value;
    return result;
}
ExecutionWireReply project(const CudaSessionRandResult& source) {
    ExecutionWireReply result;
    result.disposition = source.disposition == CudaSessionDisposition::kComplete
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    if (source.original_status) {
        result.original_status = static_cast<std::int32_t>(*source.original_status);
        if (source.status == CUDA_SUCCESS) result.status = *result.original_status;
    }
    result.id = source.id;
    return result;
}
ExecutionWireReply project(CudaLtResult source) {
    ExecutionWireReply result;
    result.disposition = source.disposition == CudaLtDisposition::kComplete
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    if (source.original_status) {
        result.original_status = static_cast<std::int32_t>(*source.original_status);
        if (source.status == CUDA_SUCCESS) result.status = *result.original_status;
    }
    result.id = source.id;
    result.address = source.value;
    result.data = std::move(source.data);
    return result;
}
#ifdef GLIMMER_HAS_CUDNN
// The wire is little-endian; native attributes use host scalar fields. Fraction
// values contain two independent int64 fields, not one 128-bit integer.
void convert_dnn_backend_scalar_order(std::uint32_t type, std::vector<std::byte>& bytes) {
    static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big);
    if constexpr (std::endian::native == std::endian::big) {
        const auto width = type == 26 ? 8 : execution_dnn_backend_scalar_width(type);
        if (width > 1 && bytes.size() % width == 0)
            for (std::size_t offset = 0; offset != bytes.size(); offset += width)
                std::reverse(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                             bytes.begin() + static_cast<std::ptrdiff_t>(offset + width));
    }
}
ExecutionWireReply project(const CudaDnnResult& source) {
    ExecutionWireReply result;
    result.disposition = source.disposition == CudaDnnDisposition::kComplete
                             ? ExecutionWireDisposition::kComplete
                             : ExecutionWireDisposition::kRejected;
    result.status = static_cast<std::int32_t>(source.status);
    if (source.original_status) {
        result.original_status = static_cast<std::int32_t>(*source.original_status);
        if (source.status == CUDA_SUCCESS) result.status = *result.original_status;
    }
    result.id = source.id;
    result.address = source.value;
    return result;
}
#endif
ExecutionWireReply project(const CudaContextControlResult& source) {
    auto reply = rejection(source.status);
    if (source.disposition == CudaSessionDisposition::kComplete)
        reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = source.original_status;
    if (source.status == CUDA_SUCCESS && source.original_status)
        reply.status = *source.original_status;
    reply.id = source.value;
    reply.address = source.auxiliary;
    return reply;
}
ExecutionWireReply project(CudaRuntimeMetadataResult source, bool is_quarantined) {
    auto reply = rejection(source.status);
    if (source.disposition == CudaSessionDisposition::kComplete)
        reply.disposition = ExecutionWireDisposition::kComplete;
    if (source.original_status) {
        reply.original_status = static_cast<std::int32_t>(*source.original_status);
        if (source.status == CUDA_SUCCESS) reply.status = *reply.original_status;
    }
    reply.address = static_cast<std::uint64_t>(source.version);
    reply.parameter_bytes = is_quarantined ? 1 : 0;
    reply.data = std::move(source.data);
    return reply;
}
std::optional<CudaRuntimeMetadataQuery> decode_runtime_metadata(ExecutionPayloadReader& input) {
    std::uint32_t kind{}, device{}, attribute{}, peer{};
    if (!input.read_u32(kind) || kind < 1 || kind > 6 || !input.read_u32(device) ||
        !input.read_u32(attribute) || !input.read_u32(peer) || !input.empty())
        return std::nullopt;
    return CudaRuntimeMetadataQuery{
        static_cast<CudaRuntimeMetadataKind>(kind), std::bit_cast<std::int32_t>(device),
        std::bit_cast<std::int32_t>(attribute), std::bit_cast<std::int32_t>(peer)};
}
bool decode_geam(ExecutionPayloadReader& input, CudaSessionGeam& problem) noexcept {
    ExecutionGeam source;
    if (!read_execution_geam(input, source) || !input.empty()) return false;
    CudaSessionGeam decoded;
    decoded.type = source.type == ExecutionGeamType::kFloat ? CUDA_R_32F : CUDA_R_64F;
    decoded.transpose_a = source.transpose_a;
    decoded.transpose_b = source.transpose_b;
    decoded.pointer_mode = source.pointer_mode;
    decoded.presence = source.presence;
    decoded.rows = source.rows;
    decoded.columns = source.columns;
    decoded.leading_a = source.leading_a;
    decoded.leading_b = source.leading_b;
    decoded.leading_c = source.leading_c;
    decoded.handle = source.handle;
    for (std::size_t index = 0; index != decoded.buffers.size(); ++index) {
        const auto& range = source.buffers[index];
        if (range.offset > std::numeric_limits<std::size_t>::max()) return false;
        decoded.buffers[index] = {range.allocation, static_cast<std::size_t>(range.offset)};
    }
    decoded.alpha = source.alpha;
    decoded.beta = source.beta;
    problem = decoded;
    return true;
}
template <typename Source, typename Destination>
bool decode_original_blas_tail(const Source& source, Destination& destination) noexcept {
    destination.handle = source.handle;
    destination.pointer_mode = source.pointer_mode;
    destination.presence = source.presence;
    for (std::size_t index = 0; index != destination.buffers.size(); ++index) {
        const auto& range = source.buffers[index];
        if (range.offset > std::numeric_limits<std::size_t>::max()) return false;
        destination.buffers[index] = {range.allocation, static_cast<std::size_t>(range.offset)};
    }
    destination.alpha = source.alpha;
    destination.beta = source.beta;
    return true;
}
bool decode_sgemv(ExecutionPayloadReader& input, CudaSessionSgemv& problem) noexcept {
    ExecutionSgemv source;
    if (!read_execution_sgemv(input, source) || !input.empty()) return false;
    CudaSessionSgemv decoded;
    decoded.transpose = source.transpose;
    decoded.rows = source.rows;
    decoded.columns = source.columns;
    decoded.leading_a = source.leading_a;
    decoded.increment_x = source.increment_x;
    decoded.increment_y = source.increment_y;
    if (!decode_original_blas_tail(source, decoded)) return false;
    problem = decoded;
    return true;
}
bool decode_sgemm_ex(ExecutionPayloadReader& input, CudaSessionSgemmEx& problem) noexcept {
    ExecutionSgemmEx source;
    if (!read_execution_sgemm_ex(input, source) || !input.empty()) return false;
    CudaSessionSgemmEx decoded;
    decoded.transpose_a = source.transpose_a;
    decoded.transpose_b = source.transpose_b;
    decoded.type_a = source.type_a;
    decoded.type_b = source.type_b;
    decoded.type_c = source.type_c;
    decoded.rows = source.rows;
    decoded.columns = source.columns;
    decoded.depth = source.depth;
    decoded.leading_a = source.leading_a;
    decoded.leading_b = source.leading_b;
    decoded.leading_c = source.leading_c;
    if (!decode_original_blas_tail(source, decoded)) return false;
    problem = decoded;
    return true;
}
bool decode_sgemm(ExecutionPayloadReader& input, CudaSessionSgemm& problem,
                  CudaSessionGemmEx* extended = nullptr) noexcept {
    std::uint32_t ta{}, tb{}, scalar{};
    if (!input.read_u64(problem.handle) || !input.read_u32(ta) || !input.read_u32(tb) || ta > 2 ||
        tb > 2)
        return false;
    problem.transpose_a = static_cast<cublasOperation_t>(ta);
    problem.transpose_b = static_cast<cublasOperation_t>(tb);
    for (auto* value : {&problem.rows, &problem.columns, &problem.depth, &problem.leading_a,
                        &problem.leading_b, &problem.leading_c}) {
        std::uint32_t bits{};
        if (!input.read_u32(bits)) return false;
        *value = std::bit_cast<std::int32_t>(bits);
    }
    const auto range = [&](CudaAllocationRange& value) {
        std::uint64_t offset{};
        if (!input.read_u64(value.allocation) || !input.read_u64(offset) ||
            offset > std::numeric_limits<std::size_t>::max())
            return false;
        value.offset = static_cast<std::size_t>(offset);
        return true;
    };
    if (!range(problem.a) || !range(problem.b) || !range(problem.c) || !input.read_u32(scalar) ||
        scalar > 1)
        return false;
    problem.uses_device_scalars = scalar != 0;
    if (scalar) {
        if (!range(problem.device_alpha) || !range(problem.device_beta)) return false;
    } else if (extended) {
        std::span<const std::byte> alpha, beta;
        const auto width = cuda_gemm_scalar_bytes(extended->compute_type, extended->c_type);
        if (!width || !input.read_blob(alpha) || !input.read_blob(beta) || alpha.size() != width ||
            beta.size() != width)
            return false;
        std::memcpy(extended->alpha.data(), alpha.data(), width);
        std::memcpy(extended->beta.data(), beta.data(), width);
    } else {
        std::uint32_t alpha{}, beta{};
        if (!input.read_u32(alpha) || !input.read_u32(beta)) return false;
        problem.alpha = std::bit_cast<float>(alpha);
        problem.beta = std::bit_cast<float>(beta);
    }
    return input.empty();
}
bool decode_gemm_ex(ExecutionPayloadReader& input, CudaSessionGemmEx& problem) noexcept {
    std::uint32_t a{}, b{}, c{}, compute{}, algorithm{};
    if (!input.read_u32(a) || !input.read_u32(b) || !input.read_u32(c) ||
        !input.read_u32(compute) || !input.read_u32(algorithm) || a > 29 || b > 29 || c > 29 ||
        compute < 64 || compute > 77)
        return false;
    const auto native_algorithm = std::bit_cast<std::int32_t>(algorithm);
    if (native_algorithm < -1 || native_algorithm > 127) return false;
    problem.a_type = static_cast<cudaDataType>(a);
    problem.b_type = static_cast<cudaDataType>(b);
    problem.c_type = static_cast<cudaDataType>(c);
    problem.compute_type = static_cast<cublasComputeType_t>(compute);
    problem.algorithm = static_cast<cublasGemmAlgo_t>(native_algorithm);
    if (!cuda_gemm_scalar_bytes(problem.compute_type, problem.c_type)) return false;
    return decode_sgemm(input, problem.operation, &problem);
}
void describe_bootstrap(ExecutionWireReply& reply, ExecutionBootstrapOwnership ownership) {
    const auto encoded = encode_execution_bootstrap_ownership(ownership);
    if (!encoded) throw std::logic_error("invalid internal bootstrap ownership");
    reply.data.assign(encoded->begin(), encoded->end());
}
bool random_incarnation(std::array<std::uint64_t, 2>& value) noexcept {
    auto bytes = std::as_writable_bytes(std::span{value});
    std::size_t done{};
    while (done < bytes.size()) {
        const auto count = ::getrandom(bytes.data() + done, bytes.size() - done, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        done += static_cast<std::size_t>(count);
    }
    return value[0] != 0 && value[1] != 0;
}
CUresult issue_failure(const CudaExecutionIssueResult& result) noexcept {
    if (result.native_status != CUDA_SUCCESS) return result.native_status;
    if (result.completion_status != CUDA_SUCCESS) return result.completion_status;
    return CUDA_ERROR_UNKNOWN;
}

// Cold-path observation only. The callback borrows the stable service-owned
// pointer slot, not this stack object, so clearing it also contains an unexpected
// failed observer reset. No source text, handles or payload bytes are logged.
class PreparationAudit final {
   public:
    PreparationAudit(PreparationAudit*& active, CudaExecutionSession& session,
                     std::shared_ptr<AsyncAuditLog> log, std::uint64_t connection,
                     const ExecutionWireHeader& header, std::uint64_t attempt) noexcept
        : active_(active),
          session_(session),
          log_(std::move(log)),
          connection_(connection),
          request_(header.sequence),
          operation_(header.operation),
          attempt_(attempt),
          started_(std::chrono::steady_clock::now()) {
        owns_active_ = !active_;
        if (owns_active_) {
            active_ = this;
            has_observer_ = session_.set_preparation_observer(
                {&active_, [](void* context, CudaPreparationPhase phase, bool is_begin) noexcept {
                     auto* audit = *static_cast<PreparationAudit**>(context);
                     if (audit) audit->phase(phase, is_begin);
                 }});
        }
        request_record("begin", "pending", 0, std::nullopt);
    }
    ~PreparationAudit() {
        if (!is_finished_) finish("exception", 0, std::nullopt);
    }
    PreparationAudit(const PreparationAudit&) = delete;
    PreparationAudit& operator=(const PreparationAudit&) = delete;
    void finish(const std::optional<ExecutionWireReply>& reply) noexcept {
        finish(reply ? "reply" : "deferred", reply ? reply->status : 0,
               reply ? reply->original_status : std::nullopt);
    }

   private:
    static std::string_view phase_name(CudaPreparationPhase phase) noexcept {
        switch (phase) {
            case CudaPreparationPhase::kInspectImage:
                return "inspect_image";
            case CudaPreparationPhase::kStaticStorage:
                return "static_storage";
            case CudaPreparationPhase::kTransformPtx:
                return "transform_ptx";
            case CudaPreparationPhase::kLoadOriginal:
                return "load_original";
            case CudaPreparationPhase::kLoadPrepared:
                return "load_prepared";
            case CudaPreparationPhase::kValidateFunction:
                return "validate_function";
        }
        return "invalid";
    }
    [[nodiscard]] std::uint64_t elapsed_ns() const noexcept {
        const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - started_)
                               .count();
        return value > 0 ? static_cast<std::uint64_t>(value) : 0;
    }
    void phase(CudaPreparationPhase phase, bool is_begin) noexcept {
        try {
            if (log_)
                log_->info(
                    "Preparation phase audit: session={}, request={}, attempt={}, operation={}, "
                    "phase={}, event={}, elapsed_ns={}",
                    connection_, request_, attempt_, static_cast<unsigned int>(operation_),
                    phase_name(phase), is_begin ? "begin" : "end", elapsed_ns());
        } catch (...) {
            // Missing records fail evidence qualification, not the CUDA call.
        }
    }
    void request_record(std::string_view event, std::string_view outcome, std::int32_t code,
                        std::optional<std::int32_t> native) noexcept {
        try {
            if (log_)
                log_->info(
                    "Preparation request audit: session={}, request={}, attempt={}, operation={}, "
                    "event={}, elapsed_ns={}, outcome={}, code={}, native_present={}, "
                    "native_code={}, observer_enabled={}",
                    connection_, request_, attempt_, static_cast<unsigned int>(operation_), event,
                    elapsed_ns(), outcome, code, native.has_value(), native.value_or(0),
                    has_observer_);
        } catch (...) {
        }
    }
    void finish(std::string_view outcome, std::int32_t code,
                std::optional<std::int32_t> native) noexcept {
        if (owns_active_) {
            if (!session_.set_preparation_observer({})) has_observer_ = false;
            active_ = nullptr;
        }
        request_record("end", outcome, code, native);
        is_finished_ = true;
    }
    PreparationAudit*& active_;
    CudaExecutionSession& session_;
    std::shared_ptr<AsyncAuditLog> log_;
    std::uint64_t connection_{}, request_{};
    ExecutionOpcode operation_;
    std::uint64_t attempt_{};
    std::chrono::steady_clock::time_point started_;
    bool owns_active_{}, has_observer_{}, is_finished_{};
};

// Retains only bounded sequence numbers, not frontiers or native resources.
// The dispatcher reports actual retirement on its ordinary collection path.
class NativeRetirementAudit final : public CudaRetirementObserver {
   public:
    struct Settings {
        std::uint64_t connection{};
        std::size_t capacity{};
    };
    NativeRetirementAudit(Settings settings, const std::shared_ptr<AsyncAuditLog>& log,
                          std::shared_ptr<IssueDepthMetrics> depth)
        : connection_(settings.connection), log_(log), depth_(std::move(depth)) {
        pending_.reserve(settings.capacity);
    }
    [[nodiscard]] bool track(std::uint64_t sequence) noexcept {
        if (!sequence || pending_.size() == pending_.capacity() ||
            std::ranges::find(pending_, sequence) != pending_.end())
            return false;
        pending_.push_back(sequence);
        return true;
    }
    void retired(std::uint64_t sequence, int status) noexcept override {
        if (depth_) depth_->retired(connection_, sequence, status == 0);
        const auto found = std::ranges::find(pending_, sequence);
        if (found == pending_.end()) return;
        pending_.erase(found);
        if (const auto log = log_.lock())
            (void)log->info(
                "Native execution audit: session={}, issued={}, retained=0, "
                "retired=1, status={}",
                connection_, sequence, status);
    }

   private:
    std::uint64_t connection_{};
    std::weak_ptr<AsyncAuditLog> log_;
    std::shared_ptr<IssueDepthMetrics> depth_;
    std::vector<std::uint64_t> pending_;
};
}  // namespace

struct ExecutionService::Implementation {
    PreparationAudit* preparation_audit{};
    struct ForeignEventEdge {
        TaskPeerIdentity peer;
        std::uint64_t source_id{}, target_id{}, target_sequence{};
        std::shared_ptr<CudaExecutionSession> source, target;
        CudaSessionEvent generation;
        std::optional<CudaExecutionFrontier> source_frontier;
        CudaExecutionFrontier target_frontier;
    };
    struct Slot {
        // Common-domain lifecycle marker; actual measured pairing separately
        // requires config.paired, including original-only space-sharing members.
        bool is_paired_member{};
        struct AdmissionFence {
            std::uint64_t continuation{};
            std::shared_ptr<CudaExecutionSession> owner;
            CudaExecutionFrontier before;
        };
        struct ContinuationInput {
            ExecutionWireMessage wire;
            ExecutionOwnedDescriptor descriptor;
        };
        struct Request {
            ExecutionWireMessage message;
            std::optional<ExecutionReceipt> receipt;
            ExecutionOwnedDescriptor descriptor;
            bool has_host_job{};
            std::optional<CudaExecutionFrontier> transfer_frontier{};
            bool is_library_replacement{};
            std::optional<CudaSessionEvent> event_generation{};
            std::shared_ptr<ForeignEventEdge> foreign_event{};
            std::shared_ptr<CudaSessionProcessAllocation> process_allocation{};
            std::shared_ptr<CudaSessionGraphLaunch> graph_launch{};
#ifdef GLIMMER_HAS_CUDNN
            std::shared_ptr<const CudaDnnConvolutionFindResult> dnn_find{};
            std::uint64_t dnn_dropout_metadata{};
            std::shared_ptr<const CudaDnnDropoutMutation> dnn_dropout{};
            std::uint64_t dnn_rnn_metadata{};
            std::shared_ptr<const CudaDnnRnnCompletion> dnn_rnn{};
            std::uint64_t dnn_attn_metadata{};
            std::shared_ptr<const CudaDnnAttnCompletion> dnn_attn{};
            std::uint64_t dnn_backend_metadata{};
            std::shared_ptr<const CudaDnnBackendExecutionCompletion> dnn_backend{};
#endif
            std::uint64_t preparation_attempt{};
            std::uint64_t frame_received_ns{}, owner_received_ns{}, first_read_ns{};
            std::uint64_t nccl_request{};
            // Allocation/quota is reserved before admission. All consumers
            // share the same immutable target instead of recapturing afterward.
            std::optional<CudaExecutionFrontierReservation> command_reservation{};
            std::optional<CudaExecutionFrontier> command_frontier{};
            std::uint64_t command_capture{};
            std::optional<ExecutionSourceContinuation> continuation{};
            std::shared_ptr<const ContinuationInput> continuation_input{};
            std::uint64_t ordinary_continuation{};
            std::optional<ExecutionWireReply> continuation_rejection{};
            bool has_continuation_dependencies{};
            bool has_host_shutdown{};
            ExecutionOpcode reply_operation() const noexcept {
                return continuation_input ? continuation_input->wire.header.operation
                                          : message.header.operation;
            }
            int borrowed_descriptor() const noexcept {
                return continuation_input ? continuation_input->descriptor.get() : descriptor.get();
            }
        };
        struct Frontier {
            std::uint64_t id{};
            CudaExecutionFrontier token;
            // A capture also owns already-retired asynchronous failures from
            // its exact accepted prefix. Later admissions never join this set.
            std::vector<CudaExecutionFrontier> asynchronous_prefix{};
            std::uint64_t asynchronous_sequence{}, stream{};
            bool is_nonblocking{};
        };
        struct AsynchronousReceipt {
            ExecutionReceipt receipt;
            CudaExecutionFrontier frontier;
        };
        struct HostFunction {
            std::uint64_t id{};
            CudaExecutionExternalBarrier barrier;
            CudaExecutionFrontier predecessor;
            // Publication is irreversible: transport loss cannot prove that
            // the application did not receive permission and enter its callback.
            bool has_permission{}, is_uncertain{};
            std::shared_ptr<CudaSessionHostFunction> native;
            std::uint64_t stream{};
            bool is_enqueued{};
        };
        struct LowEpoch {
            std::uint64_t sequence{};
            CudaExecutionFrontier frontier;
            CudaSessionPersistentProgress progress;
            bool is_issued{}, was_successful{}, is_paused{}, retreat_requested{},
                retreat_unsupported{};
            bool is_event{};
            bool is_asynchronous{};
            // ACK may erase queue metadata before the next scheduling pass.
            // Preserve the exact receipt and its immutable native/preparation origin.
            ExecutionReceipt receipt{};
            std::optional<ExecutionHostReply> host_reply{};
            ExecutionOpcode ordering_operation{};
            std::uint64_t event_source{}, ordering_capture{};
            std::optional<CudaSessionEventIdentity> event_identity{};
        };
        std::uint64_t connection{}, next_request{2}, next_command{1};
        std::uint32_t dnn_profile_version{};
        bool has_dnn_profile_mismatch{};
        std::uint64_t considered_arrival_sequence{};
        TaskPeerIdentity peer;
        std::shared_ptr<CudaExecutionSession> session;
        std::shared_ptr<HostOwner> host_owner;
        BusinessOwnerLease<CudaExecutionSession> session_leases;
        BusinessOwnerLease<HostOwner> host_owner_leases;
        std::shared_ptr<NativeRetirementAudit> retirement_audit;
        std::array<std::uint64_t, 2> last_native_pair_audit{};
        std::shared_ptr<CudaProcessMemoryBackend> process_memory;
        std::shared_ptr<const AdmissionFence> admission_fence;
        std::size_t bound_cleanup_parents{};
#ifdef GLIMMER_HAS_NCCL
        std::optional<NcclActorCleanupRetirement> cleanup_retirement;
#endif
        std::unique_ptr<CudaLtRegistry> lt_metadata;
#ifdef GLIMMER_HAS_CUDNN
        std::unique_ptr<CudaDnnRegistry> dnn_metadata;
#endif
        std::vector<Request> requests;
        std::optional<CUresult> asynchronous_failure;
        std::vector<Frontier> frontiers;
        std::vector<AsynchronousReceipt> asynchronous_receipts;
        std::vector<HostFunction> host_functions;
        // Also retains transformed epochs for unranked/naive peers: sliced work
        // needs continuation even when there is no lower-priority participant.
        std::vector<LowEpoch> low_epochs;
        std::uint64_t next_frontier{1};
        bool has_frontier_space{true};
        std::uint64_t next_host_function{1};
        bool has_host_function_space{true};
        bool is_retiring{}, is_retirement_started{}, is_initialized{}, is_closing{},
            has_request_space{true}, has_command_space{true};
        std::uint32_t priority{};
        ExecutionClientKernelMode kernel_mode{};
        bool uses_original{};
        // Retained epoch/audit cohort, not numeric priority authority. Core
        // still compares priorities; configured highest peers also retain work.
        bool is_low{};
        bool is_bootstrap_pending{}, is_staging_owner{}, is_pageable_owner{}, is_pointer_metadata{};
        bool is_process_memory{};
        bool is_nccl{};
        // Validated cold client description, not authority to share/issue work.
        ExecutionContextBootstrap context_bootstrap;
        std::uint64_t native_peer{}, policy_id{};
        std::optional<CUresult> policy_failure;
        bool is_metadata_owner() const noexcept {
            if (is_pointer_metadata || is_process_memory) return true;
#ifdef GLIMMER_HAS_CUDNN
            if (dnn_metadata) return true;
#endif
            return lt_metadata != nullptr;
        }
        CudaSessionSchedulingSnapshot snapshot() const noexcept {
            if (is_nccl) return {};
#ifdef GLIMMER_HAS_CUDNN
            if (dnn_metadata) {
                CudaSessionSchedulingSnapshot result;
                result.is_quarantined = dnn_metadata->snapshot().is_quarantined;
                return result;
            }
#endif
            if (lt_metadata) {
                CudaSessionSchedulingSnapshot result;
                result.is_quarantined = lt_metadata->snapshot().is_quarantined;
                return result;
            }
            return host_owner ? host_owner->snapshot().session : session->scheduling_snapshot();
        }
    };
    const ExecutionServiceConfig config;
    const CudaSessionDispatch dispatch;
    const pid_t process{::getpid()};
    std::weak_ptr<const void> owner;
    ExecutionSocketServer transport;
    std::shared_ptr<CudaPtxCache> ptx_cache;
    std::shared_ptr<CudaProfileCache> profile_cache;
    std::shared_ptr<CudaPairProfileCache> pair_profile_cache;
    std::shared_ptr<AsyncAuditLog> audit_log;
    struct SerialPair {
        struct Tail {
            std::uint64_t sequence{};
            CudaExecutionFrontier frontier;
            bool is_ready{};
        };
        std::array<std::shared_ptr<CudaExecutionSession>, 2> sessions;
        std::array<std::uint64_t, 2> policies{};
        CudaExecutionFrontier first;
        std::uint64_t sequence{};
        std::optional<Tail> tail{};
    };
    std::optional<SerialPair> serial_pair;
    struct ProfilePair {
        std::array<std::shared_ptr<CudaExecutionSession>, 2> sessions;
        std::array<std::uint64_t, 2> policies{}, sequences{};
        std::array<CudaExecutionFrontier, 2> frontiers;
        CudaSessionPairSelection selection;
        CudaPairProfileReservation reservation;
        bool needs_initial_pair{true};
    };
    std::optional<ProfilePair> profile_pair;
    struct ArrivalCheckpoint {
        std::shared_ptr<CudaExecutionSession> trigger;
        std::array<std::shared_ptr<CudaExecutionSession>, 2> sessions;
        std::array<CudaExecutionFrontier, 2> frontiers;
        std::array<core::ExecutionRebalanceLeg, 2> legs{};
        std::array<bool, 2> is_stopped{};
        std::size_t size{};
        std::uint64_t trigger_policy{}, trigger_sequence{};
    };
    std::optional<ArrivalCheckpoint> arrival_checkpoint;
    std::optional<CUresult> pair_profile_failure;
    std::weak_ptr<CudaExecutionSession> selected_session;
    bool execution_domain_stopped{};
    std::array<bool, core::kMaxExecutionPriorityPeers> space_share_issued{};
    std::string profile_identity, ptx_identity;
    std::array<std::uint64_t, 2> incarnation{};
    std::vector<Slot> slots;
    struct AsyncIntakePeer {
        std::uint64_t connection{}, first_sequence{};
    };
    std::vector<AsyncIntakePeer> async_intake_peers;
    std::uint64_t async_intake_second_takes{};
    struct ProcessMemoryPeer {
        TaskPeerIdentity identity;
        ExecutionOwnedDescriptor pidfd;
        std::shared_ptr<CudaProcessMemoryBackend> backend;
        std::shared_ptr<CudaExecutionSession> cleanup{};
        bool has_exited{}, is_retired{};
    };
    std::vector<ProcessMemoryPeer> memory_peers;
    std::shared_ptr<CudaProcessMemoryDevice> memory_device;
    std::vector<std::shared_ptr<ForeignEventEdge>> foreign_events;
    struct GraphConstruction {
        std::uint64_t connection{};
        ExecutionOwnedDescriptor pidfd;
        bool is_unknown{};
    };
    std::optional<GraphConstruction> graph_construction;
    std::vector<ExecutionReceivedRequest> graph_deferred_intake;
    bool is_started{}, is_stopping{}, has_client{}, is_active{};
    std::uint64_t semantic_serial{};
    bool has_semantic_loss{}, is_semantic_closed{}, sgemm_audit_captured{};
    std::uint64_t sgemm_audit_session{};
    std::optional<SgemmIssueRangeTracker> sgemm_issue_range;
    enum class IssueAuditKind : std::uint8_t { kInitial, kResumeAttempt };
    struct ActiveScope {
        bool& active;
        explicit ActiveScope(bool& value) noexcept : active(value) {
            active = true;
        }
        ~ActiveScope() {
            active = false;
        }
    };
    std::uint32_t highest_priority{};
    core::ExecutionPriorityIdleWindow priority_idle;
    std::uint64_t next_policy_id{1};
#ifdef GLIMMER_HAS_NCCL
    struct NativePeer {
        TaskPeerIdentity identity;
        std::uint64_t id{};
        std::uint32_t priority{};
        bool disconnected{};
    };
    struct NativePolicy {
        std::uint64_t reservation{}, generation{}, identity{};
    };
    struct NativeSourceKey {
        std::uint64_t peer{}, session{};
    };
    struct NativeDependencyPath {
        CUresult status{CUDA_ERROR_INVALID_HANDLE};
        CUresult rejection{CUDA_SUCCESS};
        std::uint64_t source_id{};
        std::shared_ptr<CudaExecutionSession> source;
        std::optional<CudaExecutionFrontier> frontier;
        std::optional<std::uint64_t> preparation_sequence;
        std::vector<std::shared_ptr<ForeignEventEdge>> edges;
        bool is_retirement{}, has_progress{}, is_parent_rejection{};
    };
    enum class NativeAuditTarget : std::uint32_t { kRequest = 1, kContinuation };
    enum class NativeAuditMode : std::uint32_t { kNative = 1, kSource, kHandler, kReject };
    std::unique_ptr<NcclActor> nccl_actor;
    std::vector<NativePeer> native_peers;
    std::vector<NativePolicy> native_policies;
    std::optional<CUresult> nccl_failure;
    // Exact last owner selected by this runtime. Never clear an unrecognized
    // native context simply to make a subsequent actor call appear valid.
    bool nccl_stopped{}, nccl_driver_ready{};
    bool has_continuation_attempt{};
#endif

    Implementation(ExecutionServiceConfig value, CudaSessionDispatch functions,
                   std::shared_ptr<ExecutionSocketPollJournal> journal = {}
#ifdef GLIMMER_HAS_NCCL
                   ,
                   ExecutionNcclDispatch nccl = {}
#endif
                   )
        : config(execution_configuration(std::move(value))),
          dispatch(std::move(functions)),
          transport(journal ? ExecutionSocketServer(config.transport, std::move(journal))
                            : ExecutionSocketServer(config.transport)) {
        if (config.should_receive_async_burst) {
            if (config.scheduling_mode != core::ExecutionSchedulingMode::kPriority ||
                config.paired || config.shared || config.uses_nccl ||
                config.persistent.uses_workload_agnostic || !config.profile_input.empty() ||
                !config.profile_output.empty() ||
                std::ranges::any_of(config.client_priorities,
                                    [](const auto& pin) { return pin.priority != 0; }))
                throw std::invalid_argument("asynchronous burst requires ordinary equal priority");
            async_intake_peers.reserve(config.transport.max_peers);
        }
        if (config.issue_depth_metrics) {
            const bool is_supported =
                config.scheduling_mode == core::ExecutionSchedulingMode::kPriority &&
                !config.paired && !config.shared && !config.uses_nccl &&
                std::ranges::all_of(config.client_priorities,
                                    [](const auto& pin) { return pin.priority == 0; });
            if (!is_supported)
                config.issue_depth_metrics->invalidate(IssueDepthMetrics::Failure::kUnsupported);
            if (!config.transport.max_peers ||
                config.transport.max_peers > IssueDepthMetrics::slot_capacity ||
                config.session_limits.submissions.max_commands >
                    IssueDepthMetrics::entry_capacity / config.transport.max_peers)
                config.issue_depth_metrics->invalidate(IssueDepthMetrics::Failure::kCapacity);
        }
        if (config.device_ordinal < 0 || !dispatch.initialize || !dispatch.context_set_current)
            throw std::invalid_argument("invalid execution service device or native dispatch");
        if (const auto& target = config.sgemm_audit_target) {
            if (!config.should_audit_native || !target->sequence ||
                (static_cast<bool>(target->session) == target->client.has_value()) ||
                (target->client &&
                 (!target->client->pid || !target->client->start_time_ticks ||
                  target->client->uid != config.transport.allowed_uid.value_or(
                                             static_cast<std::uint32_t>(::geteuid())))))
                throw std::invalid_argument(
                    "SGEMM audit target requires native audit and one nonzero identity");
            sgemm_audit_session = target->session;
        }
        if (config.sgemm_range_hooks) {
            if (!config.sgemm_audit_target || !config.sgemm_range_hooks->push ||
                !config.sgemm_range_hooks->pop)
                throw std::invalid_argument(
                    "SGEMM range requires an exact audit target and callbacks");
            sgemm_issue_range.emplace(*config.sgemm_range_hooks);
        }
        if (!config.maximum_foreign_event_edges || config.maximum_foreign_event_edges > 65536)
            throw std::invalid_argument("invalid foreign event dependency capacity");
        if (config.priority_idle_us > 1000000)
            throw std::invalid_argument("invalid priority idle interval");
        if ((!core::is_priority_execution_mode(config.scheduling_mode) &&
             config.scheduling_mode != core::ExecutionSchedulingMode::kNaive &&
             config.scheduling_mode != core::ExecutionSchedulingMode::kProfile &&
             config.scheduling_mode != core::ExecutionSchedulingMode::kWorkloadAgnostic &&
             config.scheduling_mode != core::ExecutionSchedulingMode::kPrioritySpaceShare) ||
            (config.uses_original_kernels && config.persistent.replace_sgemm))
            throw std::invalid_argument(
                "invalid execution mode or conflicting original/replacement selection");
        switch (config.persistent.kind) {
            case transform::PtxExecutionKind::kStatic:
            case transform::PtxExecutionKind::kBlockStatic:
            case transform::PtxExecutionKind::kDynamic:
            case transform::PtxExecutionKind::kPreemptive:
            case transform::PtxExecutionKind::kSliced:
                break;
            default:
                throw std::invalid_argument("invalid persistent execution kind");
        }
        if (!cuda_gemm_tile_side_is_valid(config.persistent.gemm_tile_side) ||
            (!config.persistent.replace_sgemm && config.persistent.gemm_tile_side != 64))
            throw std::invalid_argument("invalid private GEMM tile geometry or replacement mode");
        if (!config.uses_original_kernels && !config.persistent.workers &&
            config.persistent.kind != transform::PtxExecutionKind::kPreemptive)
            throw std::invalid_argument("selected PTX family requires explicit workers");
        if (config.persistent.replace_sgemm &&
            config.persistent.kind != transform::PtxExecutionKind::kPreemptive)
            throw std::invalid_argument("private GEMM replacement requires preemptive mode");
        if (config.persistent.uses_sharded_host_retreat &&
            (!config.persistent.replace_sgemm || !config.persistent.workers ||
             config.persistent.kind != transform::PtxExecutionKind::kPreemptive ||
             cuda_sgemm_sharded_image(config.persistent.gemm_tile_side).empty() ||
             !config.session_limits.submissions.max_commands ||
             config.session_limits.submissions.max_commands >
                 config.session_limits.max_private_host_bytes /
                     (kPersistentProgressHostSlotBytes + kGemmSignalSlotBytes)))
            throw std::invalid_argument("invalid sharded GEMM signal configuration or budget");
        if (config.persistent.uses_workload_agnostic &&
            (config.uses_original_kernels || config.persistent.replace_sgemm ||
             !config.persistent.workers || config.profile_input.empty() ||
             !config.profile_output.empty() || !config.client_priorities.empty() ||
             !dispatch.device_attribute || !config.persistent.sharing_threads_per_sm ||
             config.persistent.sharing_threads_per_sm > 65536 ||
             !config.persistent.sharing_minimum_speed_percent ||
             config.persistent.sharing_minimum_speed_percent > 10000))
            throw std::invalid_argument(
                "workload-agnostic scheduling requires a readonly profile, explicit workers, "
                "bounded sharing limits and no priority/original/replacement conflicts");
        if (config.client_priorities.size() > core::kMaxExecutionPriorityPeers ||
            (config.should_audit_priority && config.client_priorities.empty()) ||
            ((!config.client_priorities.empty() ||
              config.scheduling_mode != core::ExecutionSchedulingMode::kPriority) &&
             config.transport.max_peers > core::kMaxExecutionPriorityPeers) ||
            (!config.client_priorities.empty() && !config.persistent.workers &&
             !config.uses_original_kernels &&
             core::is_priority_execution_mode(config.scheduling_mode)))
            throw std::invalid_argument(
                "execution scheduling requires bounded peers; priority needs original mode or "
                "persistent workers");
        for (std::size_t index = 0; index < config.client_priorities.size(); ++index) {
            const auto& entry = config.client_priorities[index];
            if (!entry.peer.pid || !entry.peer.start_time_ticks ||
                entry.peer.uid != config.transport.allowed_uid.value_or(
                                      static_cast<std::uint32_t>(::geteuid())) ||
                std::ranges::any_of(std::span{config.client_priorities}.first(index),
                                    [&](const auto& prior) { return prior.peer == entry.peer; }))
                throw std::invalid_argument("invalid or duplicate trusted priority identity");
            if (entry.kernel_mode != ExecutionClientKernelMode::kAuto &&
                entry.kernel_mode != ExecutionClientKernelMode::kOriginal &&
                entry.kernel_mode != ExecutionClientKernelMode::kConfigured)
                throw std::invalid_argument("invalid trusted client kernel mode");
            if (entry.kernel_mode == ExecutionClientKernelMode::kConfigured &&
                (!core::is_priority_execution_mode(config.scheduling_mode) ||
                 config.uses_original_kernels || !config.persistent.workers))
                throw std::invalid_argument(
                    "configured client kernels require priority scheduling and enabled "
                    "transformation/workers");
            highest_priority = std::max(highest_priority, entry.priority);
        }
        if (core::is_priority_execution_mode(config.scheduling_mode))
            priority_idle = core::ExecutionPriorityIdleWindow{
                {highest_priority, static_cast<std::uint64_t>(config.priority_idle_us) * 1000}};
        if (!service_owner) service_owner = std::make_shared<int>(0);
        ptx_cache = std::make_shared<CudaPtxCache>(config.ptx_cache);
        if (!config.ptx_cache_input.empty() || !config.ptx_cache_output.empty()) {
            if (config.uses_original_kernels || !config.persistent.workers ||
                !config.ptx_cache.max_entries || !config.ptx_cache.max_bytes)
                throw std::invalid_argument(
                    "PTX archives require enabled transformation and cache");
            if ((!config.ptx_cache_output.empty() &&
                 (config.ptx_cache_output == config.ptx_cache_input ||
                  config.ptx_cache_output == config.profile_input ||
                  config.ptx_cache_output == config.profile_output)) ||
                (!config.profile_output.empty() && config.profile_output == config.ptx_cache_input))
                throw std::invalid_argument("execution archive output paths must be distinct");
            const auto identity = profile_runtime_identity();
            if (!identity) throw std::runtime_error("cannot identify PTX preparation program");
            ptx_identity = *identity;
            if (!config.ptx_cache_input.empty()) {
                const auto archive =
                    read_execution_archive(config.ptx_cache_input, kCudaPtxArchiveMaxBytes);
                if (!archive || !ptx_cache->restore(*archive, ptx_identity))
                    throw std::invalid_argument(
                        "invalid, incompatible or inaccessible PTX preparation archive");
            }
        }
        const bool learns = config.scheduling_mode == core::ExecutionSchedulingMode::kProfile;
        if (!config.profile_output.empty() && !learns)
            throw std::invalid_argument("profile output requires the profile scheduler");
        if (learns || !config.profile_input.empty()) {
            if ((learns &&
                 (!config.client_priorities.empty() || config.persistent.replace_sgemm)) ||
                (config.uses_original_kernels && !config.profile.kinds.empty()) ||
                !dispatch.device_uuid || !dispatch.driver_version ||
                (!config.uses_original_kernels &&
                 (!config.persistent.workers ||
                  (!config.persistent.uses_workload_agnostic &&
                   std::ranges::any_of(config.profile.workers, [&](auto workers) {
                       return workers > config.persistent.workers;
                   })))))
                throw std::invalid_argument(
                    "profile selection requires platform identity and compatible candidates");
            if (!learns && !config.uses_original_kernels &&
                core::is_priority_execution_mode(config.scheduling_mode) && has_priority_policy() &&
                config.persistent.kind != transform::PtxExecutionKind::kPreemptive &&
                config.persistent.kind != transform::PtxExecutionKind::kSliced)
                throw std::invalid_argument("learned priority requires a preemptible default");
            profile_cache = std::make_shared<CudaProfileCache>(config.profile, nullptr, learns);
            if (!config.profile_input.empty() || !config.profile_output.empty()) {
                const auto identity = profile_runtime_identity();
                if (!identity)
                    throw std::runtime_error("cannot identify profile execution program");
                profile_identity = *identity;
            }
            if (!config.profile_input.empty()) {
                const auto archive = read_profile_archive(config.profile_input);
                if (!archive || !profile_cache->restore(*archive, profile_identity))
                    throw std::invalid_argument(
                        "invalid, incompatible or inaccessible profile archive");
            }
        }
        if (config.persistent.observes_block_completion &&
            (config.scheduling_mode != core::ExecutionSchedulingMode::kPriority ||
             config.uses_original_kernels || !config.persistent.workers ||
             config.persistent.kind != transform::PtxExecutionKind::kPreemptive ||
             config.persistent.uses_workload_agnostic ||
             (profile_cache && (profile_cache->is_learning() || profile_cache->snapshot().cases ||
                                !profile_cache->kinds().empty()))))
            throw std::invalid_argument(
                "block completion observation requires unmeasured preemptive priority execution");
        if (uses_priority_space_share() != config.shared.has_value() ||
            (config.shared && config.paired))
            throw std::invalid_argument(
                "priority space sharing requires its own non-paired common domain");
        if (config.shared) {
            const auto& shared = *config.shared;
            const auto domain =
                shared.domain ? shared.domain->snapshot() : CudaExecutionDomainSnapshot{};
            if (!shared.domain || domain.status != CUDA_SUCCESS ||
                (shared.should_initialize_domain ? domain.has_context : !domain.has_context) ||
                domain.is_closed || domain.is_shutdown || domain.is_quarantined ||
                domain.retained_members || domain.reserved_device_bytes ||
                config.client_priorities.size() < 2 || !config.uses_original_kernels ||
                config.persistent.workers || config.persistent.replace_sgemm || profile_cache ||
                !config.profile_input.empty() || !config.profile_output.empty() ||
                !config.ptx_cache_input.empty() || !config.ptx_cache_output.empty() ||
                config.persistent.observes_block_completion || !dispatch.context.priority_range ||
                !dispatch.stream_create_priority || !dispatch.stream_get_priority ||
                !dispatch.stream_wait_event || !dispatch.execution.context_get_current)
                throw std::invalid_argument(
                    "priority space sharing requires original pinned clients, an empty common "
                    "domain and checked priority streams without profiling or transformation");
        }
        if (config.paired) {
            const auto& paired = *config.paired;
            if (paired.should_rebalance_on_arrival &&
                (paired.is_isolated_profile ||
                 std::ranges::count_if(config.client_priorities, [](const auto& pin) {
                     return pin.kernel_mode == ExecutionClientKernelMode::kConfigured;
                 }) < 2))
                throw std::invalid_argument(
                    "arrival rebalancing requires at least two configured pins without sampling");
            if (config.persistent.observes_block_completion &&
                (paired.is_isolated_profile || !paired.profile_input.empty() ||
                 std::ranges::count_if(config.client_priorities, [](const auto& pin) {
                     return pin.kernel_mode == ExecutionClientKernelMode::kConfigured;
                 }) > 1))
                throw std::invalid_argument(
                    "block completion diagnostics cannot use paired measurements");
            const auto domain =
                paired.domain ? paired.domain->snapshot() : CudaExecutionDomainSnapshot{};
            if (!paired.domain || domain.status != CUDA_SUCCESS ||
                (paired.should_initialize_domain ? domain.has_context : !domain.has_context) ||
                domain.is_closed || domain.is_shutdown || domain.is_quarantined ||
                domain.retained_members || domain.reserved_device_bytes ||
                !core::is_priority_execution_mode(config.scheduling_mode) ||
                config.client_priorities.size() < 2 || config.uses_original_kernels ||
                !config.persistent.workers || !profile_cache || profile_cache->is_learning() ||
                !config.profile_output.empty() || paired.workers.empty() ||
                paired.workers.size() > 8 || !dispatch.execution.context_get_current ||
                !dispatch.device_attribute || !dispatch.function_get_attribute ||
                !dispatch.occupancy ||
                (config.persistent.kind != transform::PtxExecutionKind::kPreemptive &&
                 config.persistent.kind != transform::PtxExecutionKind::kSliced))
                throw std::invalid_argument(
                    "paired service requires an empty trusted domain, readonly profiles, pinned "
                    "clients and explicit preemptible workers");
            for (std::size_t index = 0; index < paired.workers.size(); ++index) {
                const auto workers = paired.workers[index];
                if (!workers || workers > config.persistent.workers ||
                    std::ranges::find(std::span{paired.workers}.first(index), workers) !=
                        std::span{paired.workers}.first(index).end())
                    throw std::invalid_argument("invalid paired worker catalog");
            }
            if (paired.is_isolated_profile) {
                if (config.client_priorities.size() != 2 || config.uses_nccl ||
                    paired.profile_output.empty() ||
                    paired.workers.size() * paired.workers.size() > paired.profile.max_candidates ||
                    config.client_priorities[0].priority != config.client_priorities[1].priority ||
                    std::ranges::any_of(config.client_priorities, [](const auto& pin) {
                        return pin.kernel_mode != ExecutionClientKernelMode::kConfigured;
                    }))
                    throw std::invalid_argument(
                        "isolated pair profiling requires exactly two equal configured pins, a "
                        "bounded catalog and output, and no native actor");
                const std::array<std::string_view, 5> inputs{
                    config.profile_input, config.profile_output, config.ptx_cache_input,
                    config.ptx_cache_output, paired.profile_input};
                if (std::ranges::find(inputs, paired.profile_output) != inputs.end())
                    throw std::invalid_argument(
                        "paired profile output must not alias another execution archive");
            } else if (!paired.profile_output.empty()) {
                throw std::invalid_argument("paired output requires explicit isolated profiling");
            }
            pair_profile_cache = std::make_shared<CudaPairProfileCache>(
                profile_cache, paired.profile, nullptr, paired.is_isolated_profile);
            if (!pair_profile_cache->snapshot().is_valid)
                throw std::invalid_argument("invalid paired profile bounds");
            if (!paired.profile_input.empty()) {
                const auto archive =
                    read_execution_archive(paired.profile_input, kCudaPairProfileArchiveMaxBytes);
                if (!archive || !pair_profile_cache->restore(*archive, profile_identity))
                    throw std::invalid_argument("invalid or incompatible paired profile archive");
            }
        }
        owner = service_owner;
        if (config.should_audit_native || config.should_audit_priority ||
            config.should_audit_copy_requests) {
            try {
                audit_log = std::make_shared<AsyncAuditLog>();
            } catch (...) {
                has_semantic_loss = true;
            }
        }
        slots.reserve(config.transport.max_peers);
        foreign_events.reserve(config.maximum_foreign_event_edges);
#ifdef GLIMMER_HAS_NCCL
        if (config.uses_nccl) {
            if (config.transport.max_peers >= core::kMaxExecutionPriorityPeers ||
                config.device_ordinal != 0)
                throw std::invalid_argument(
                    "NCCL requires bounded source/batch policy capacity and device zero");
            const auto& native = nccl.native;
            const auto& primary = nccl.primary;
            // A missing optional provider is a clean role-bootstrap rejection,
            // not permission to enter any partial SDK table.
            if (native.get_unique_id && native.init_rank && native.init_rank_config &&
                native.get_async_error && native.destroy && native.abort && native.group_start &&
                native.group_end && native.all_gather && native.all_reduce && native.bcast &&
                primary.device_get && primary.primary_retain && primary.primary_release &&
                primary.context_get_current && primary.context_set_current &&
                primary.stream_create && primary.stream_query && primary.stream_destroy &&
                dispatch.execution.context_get_current) {
                NcclActorConfig settings;
                settings.device_ordinal = config.device_ordinal;
                settings.maximum_reservations = std::min<std::size_t>(
                    16, core::kMaxExecutionPriorityPeers - config.transport.max_peers);
                settings.maximum_groups =
                    std::min(settings.maximum_groups, settings.maximum_reservations);
                settings.coordinator.maximum_retained_groups = settings.maximum_groups;
                native_peers.reserve(settings.maximum_callers);
                native_policies.reserve(settings.maximum_reservations);
                NcclActorResolver resolver;
                resolver.session = [this](std::uint64_t peer, std::uint64_t source) {
                    return nccl_source({.peer = peer, .session = source});
                };
                resolver.stream = [this](std::uint64_t peer, const ExecutionNcclScope& scope,
                                         const ExecutionNcclStream& stream) {
                    const auto source = stream.kind == ExecutionNcclStreamKind::kOwned
                                            ? stream.owner
                                            : scope.context;
                    // Public PTDS is materialized as its authenticated owned
                    // stream by the frontend. An unbound PTDS tag has no lane.
                    if (stream.kind == ExecutionNcclStreamKind::kPerThread)
                        return NcclActorSource{};
                    return NcclActorSource{
                        source, nccl_source({.peer = peer, .session = source}),
                        stream.kind == ExecutionNcclStreamKind::kOwned ? stream.object : 0};
                };
                resolver.admit_source = [this](std::uint64_t peer, std::uint64_t source) {
                    const auto found = std::ranges::find(slots, source, &Slot::connection);
                    const auto identity = std::ranges::find(native_peers, peer, &NativePeer::id);
                    if (found == slots.end() || identity == native_peers.end() ||
                        identity->disconnected || found->peer != identity->identity)
                        return CUDA_ERROR_INVALID_HANDLE;
                    return found->admission_fence ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
                };
                resolver.cleanup_owner_retired = [this](std::uint64_t peer,
                                                        std::uint64_t connection,
                                                        const std::shared_ptr<const void>& helper) {
                    const NativeSourceKey key{peer, connection};
                    const auto found = std::ranges::find(slots, key.session, &Slot::connection);
                    const auto identity =
                        std::ranges::find(native_peers, key.peer, &NativePeer::id);
                    if (found == slots.end() || identity == native_peers.end() ||
                        found->peer != identity->identity || !found->host_owner ||
                        found->host_owner != helper || !found->cleanup_retirement ||
                        (!found->is_retiring && !found->is_closing))
                        return CUDA_ERROR_INVALID_HANDLE;
                    const auto observed = found->host_owner->snapshot();
                    if (found->policy_failure || observed.session.is_quarantined)
                        return CUDA_ERROR_UNKNOWN;
                    return observed.is_closed && !observed.has_reservation &&
                                   !observed.session.has_context &&
                                   !observed.session.host_allocations &&
                                   !observed.session.host_memory_bytes
                               ? CUDA_SUCCESS
                               : CUDA_ERROR_NOT_READY;
                };
                nccl_actor = std::make_unique<NcclActor>(
                    settings, native,
                    CudaNativeExecutionLaneDispatch{dispatch.context_set_current,
                                                    dispatch.execution.context_get_current},
                    primary, std::move(resolver), make_pinned_nccl_effect_producer());
            }
        }
#endif
        // Last startup operation: every archive, bound, pin, metadata allocation
        // and optional actor validation has completed before touching the SDK.
        if ((config.paired && config.paired->should_initialize_domain) ||
            (config.shared && config.shared->should_initialize_domain)) {
            const auto status = execution_domain()->initialize();
            if (status != CUDA_SUCCESS)
                throw std::runtime_error(
                    std::string(config.paired
                                    ? "cannot initialize workload-aware execution domain: "
                                    : "cannot initialize priority-space-share execution domain: ") +
                    std::to_string(status));
        }
    }
    bool owns() const noexcept {
        return ::getpid() == process && service_owner && owner.lock() == service_owner;
    }
    template <class Fields>
    std::uint64_t semantic(std::uint32_t family, Fields&& fields) noexcept {
        if ((!config.should_audit_native && !config.should_audit_priority) || !is_started ||
            is_semantic_closed || has_semantic_loss)
            return 0;
        if (!audit_log || audit_log->lost()) {
            has_semantic_loss = true;
            return 0;
        }
        if (semantic_serial == std::numeric_limits<std::uint64_t>::max()) {
            has_semantic_loss = true;
            return 0;
        }
        const auto serial = ++semantic_serial;
        SemanticAuditJson json;
        json.add("v", 1);
        json.add("service", incarnation[0]);
        json.add("serial", serial);
        json.add("family", family);
        fields(json);
        const auto payload = json.finish();
        if (!payload) {
            has_semantic_loss = true;
            return 0;
        }
        if (!audit_log->info("Execution semantic audit: {}", *payload)) {
            // Qualification requires contiguous records and a healthy final
            // marker. Diagnostic loss never changes an application result.
            has_semantic_loss = true;
            return 0;
        }
        return serial;
    }
    void semantic_lost(std::uint32_t reason) noexcept {
        (void)semantic(2, [&](auto& json) {
            json.add("event", 3);
            json.add("reason", reason);
        });
        has_semantic_loss = true;
    }
    void start_sgemm_audit() noexcept {
        const auto& target = config.sgemm_audit_target;
        if (!target || !sgemm_audit_session) return;
        if (!audit_log ||
            !audit_log->info("SGEMM diagnostic: v=1, event=start, session={}, sequence={}",
                             sgemm_audit_session, target->sequence))
            has_semantic_loss = true;
        if (sgemm_issue_range &&
            (!audit_log || !audit_log->info("SGEMM range diagnostic: v=1, event=start")))
            has_semantic_loss = true;
    }
    void bind_sgemm_audit(const Slot& slot) noexcept {
        const auto& target = config.sgemm_audit_target;
        if (!target || !target->client || slot.peer != *target->client || !slot.session ||
            slot.host_owner || slot.is_metadata_owner() || slot.is_nccl)
            return;
        // Connection identity is service-local. Never rebind on reconnection or
        // choose between multiple compute connections belonging to one client.
        if (sgemm_audit_session) {
            has_semantic_loss = true;
            return;
        }
        sgemm_audit_session = slot.connection;
        if (!audit_log ||
            !audit_log->info("SGEMM target binding: v=1, pid={}, start={}, uid={}, session={}, "
                             "sequence={}",
                             slot.peer.pid, slot.peer.start_time_ticks, slot.peer.uid,
                             sgemm_audit_session, target->sequence))
            has_semantic_loss = true;
        start_sgemm_audit();
    }
    void audit_peer(const Slot& slot, std::uint32_t event) noexcept {
        (void)semantic(4, [&](auto& json) {
            json.add("event", event);
            json.add("connection", slot.connection);
            json.add("peer", slot.native_peer);
            json.add("pid", slot.peer.pid);
            json.add("start", slot.peer.start_time_ticks);
            json.add("uid", slot.peer.uid);
            json.add("priority", slot.priority);
            json.add("role", slot.is_nccl               ? 3
                             : slot.is_metadata_owner() ? 2
                             : slot.host_owner          ? 1
                                                        : 0);
            json.add("policy", slot.policy_id);
            json.add("kernel_mode", static_cast<unsigned int>(slot.kernel_mode));
            json.add("effective", slot.session && !slot.is_metadata_owner() &&
                                          !slot.uses_original && config.persistent.workers
                                      ? 1
                                      : 0);
            json.add("retained", slot.is_low);
        });
    }
#ifdef GLIMMER_HAS_NCCL
    static void actor_semantic(void* context, const NcclActorAuditRecord& record) noexcept {
        auto& runtime = *static_cast<Implementation*>(context);
        (void)runtime.semantic(1, [&](auto& json) {
            json.add("actor_serial", record.actor_serial);
            json.add("event", static_cast<unsigned int>(record.event));
            json.add("subject", static_cast<unsigned int>(record.subject));
            json.add("action", static_cast<unsigned int>(record.action));
            json.add("binding", static_cast<unsigned int>(record.binding));
            json.add("id", record.id);
            json.add("peer", record.peer);
            json.add("priority", record.priority);
            json.add("caller0", record.caller[0]);
            json.add("caller1", record.caller[1]);
            json.add("group", record.group);
            json.add("sequence", record.sequence);
            json.add("ordinary_sequence", record.ordinary_sequence);
            json.add("cleanup_parent", record.cleanup_parent);
            json.add("origin_source", record.origin_source);
            json.add("caller_primary_domain", record.caller_primary_domain);
            json.add("has_context", record.has_context);
            json.add("before_kind", record.before_kind);
            json.add("before_identity", record.before_identity);
            json.add("before_context", record.before_context);
            json.add("before_device", record.before_device);
            json.add("after_kind", record.after_kind);
            json.add("after_identity", record.after_identity);
            json.add("after_context", record.after_context);
            json.add("after_device", record.after_device);
            json.add("reservation", record.reservation);
            json.add("generation", record.generation);
            json.add("cutoff", record.cutoff);
            json.add("input_reservation", record.input_reservation);
            json.add("input_generation", record.input_generation);
            json.add("input_cutoff", record.input_cutoff);
            json.add("native_sequence", record.native_sequence);
            json.add("native_operation", record.native_operation);
            json.add("native_kind", record.native_kind);
            json.add("communicator_domain", record.communicator_domain);
            json.add("communicator_context", record.communicator_context);
            json.add("communicator_object", record.communicator_object);
            json.add("output_contract", record.output_contract);
            json.add("ordinary_opcode", record.ordinary_opcode);
            json.add("connection", record.connection);
            json.add("wire_sequence", record.wire_sequence);
            json.add("attempt", record.attempt);
            json.add("source", record.source);
            json.add("stream", record.stream);
            json.add("capture", record.capture);
            json.add("command_sequence", record.command_sequence);
            json.add("related", record.related);
            json.add("index", record.index);
            json.add("count", record.count);
            json.add("allocation", record.allocation);
            json.add("ticket", record.ticket);
            json.add("depth", record.depth);
            json.add("stage", record.stage);
            json.add("previous_stage", record.previous_stage);
            json.add("phase", record.phase);
            json.add("outcome", record.outcome);
            json.add("disposition", record.disposition);
            json.add("host_origin", record.host_origin);
            json.add("status", record.status);
            json.add("carrier", record.carrier);
            json.add("original_status", record.original_status);
            json.add("async_status", record.async_status);
            json.add("native_status", record.native_status);
            json.add("completion_status", record.completion_status);
            json.add("library_status", record.library_status);
            json.add("accepted", record.accepted);
            json.add("sequence_consumed", record.sequence_consumed);
            json.add("has_original", record.has_original);
            json.add("has_async", record.has_async);
            json.add("has_library", record.has_library);
            json.add("has_host", record.has_host);
            json.add("has_output_written", record.has_output_written);
            json.add("output_written", record.output_written);
            json.add("reply_ready", record.reply_ready);
            json.add("reply_released", record.reply_released);
            json.add("enqueue_complete", record.enqueue_complete);
            json.add("physical_complete", record.physical_complete);
            json.add("has_physical_work", record.has_physical_work);
            json.add("open_group", record.open_group);
            json.add("unknown", record.unknown);
            json.add("retired", record.retired);
            json.add("bound", record.bound);
            json.add("consumed", record.consumed);
            json.add("active_call", record.active_call);
            json.add("ready_continuation", record.ready_continuation);
        });
    }
#endif
    bool has_priority_policy() const noexcept {
        return !config.client_priorities.empty();
    }
    bool receives_priority_before_observation() const noexcept {
        return config.scheduling_mode == core::ExecutionSchedulingMode::kPriority &&
               has_priority_policy() && config.persistent.workers &&
               config.persistent.kind == transform::PtxExecutionKind::kPreemptive &&
               !config.uses_original_kernels && !config.paired && !profile_cache &&
               !config.uses_nccl;
    }
    bool uses_priority_space_share() const noexcept {
        return config.scheduling_mode == core::ExecutionSchedulingMode::kPrioritySpaceShare;
    }
    CudaExecutionDomainOwner* execution_domain() const noexcept {
        return config.paired   ? config.paired->domain.get()
               : config.shared ? config.shared->domain.get()
                               : nullptr;
    }
    CUresult memory_native_health() const noexcept {
        return memory_device ? memory_device->native_health() : CUDA_SUCCESS;
    }
    CUresult process_memory_native_gate() const noexcept {
        try {
            if (graph_construction)
                return graph_construction->is_unknown ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_NOT_READY;
            if (const auto health = memory_native_health(); health != CUDA_SUCCESS) return health;
            for (const auto& slot : slots) {
                if (!slot.host_owner) continue;
                const auto observed = slot.host_owner->snapshot();
                if (observed.session.is_quarantined) return CUDA_ERROR_UNKNOWN;
                if (observed.has_reservation) return CUDA_ERROR_NOT_READY;
            }
#ifdef GLIMMER_HAS_NCCL
            if (nccl_actor) {
                const auto observed = nccl_actor->snapshot();
                if (observed.status != CUDA_SUCCESS) return observed.status;
                for (const auto& reservation : observed.reservations) {
                    if (reservation.has_unknown_state) return CUDA_ERROR_UNKNOWN;
                    for (const auto& request : reservation.requests)
                        if (request.stage == NcclActorStage::kNativePending ||
                            request.stage == NcclActorStage::kWaitingAsync ||
                            request.stage == NcclActorStage::kUncertain)
                            return CUDA_ERROR_NOT_READY;
                }
            }
#endif
            return CUDA_SUCCESS;
        } catch (...) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    std::shared_ptr<CudaProcessMemoryBackend> memory_owner(const TaskPeerIdentity& peer,
                                                           std::uint64_t identity) {
        const auto found = std::ranges::find(memory_peers, peer, &ProcessMemoryPeer::identity);
        if (found != memory_peers.end()) return found->has_exited ? nullptr : found->backend;
        if (!dispatch.process_memory.allocate || memory_peers.size() == config.transport.max_peers)
            return {};
        ExecutionOwnedDescriptor pidfd(static_cast<int>(::syscall(SYS_pidfd_open, peer.pid, 0)));
        if (pidfd.get() < 0 || !memory_peer_matches(peer)) return {};
        if (!memory_device)
            memory_device = std::make_shared<CudaProcessMemoryDevice>(
                dispatch.process_memory, [this] { return process_memory_native_gate(); },
                config.paired   ? config.paired->domain
                : config.shared ? config.shared->domain
                                : nullptr);
        auto backend = std::make_shared<CudaProcessMemoryBackend>(
            CudaProcessMemoryConfig{identity, config.device_ordinal,
                                    config.session_limits.max_objects,
                                    config.session_limits.max_objects, 65536},
            std::make_shared<core::QuotaLedger>(config.session_limits.max_device_bytes),
            memory_device);
        memory_peers.push_back({peer, std::move(pidfd), backend});
        return backend;
    }
    bool space_share_budget_exhausted(
        const Slot& slot, std::optional<std::uint32_t> represented_priority) const noexcept {
        // Reuse this fresh Core observation, including retiring/uncertain
        // owners, rather than rescanning clients or retaining departed pins.
        if (!uses_priority_space_share() || !represented_priority ||
            slot.priority >= *represented_priority || slot.host_owner || slot.is_nccl ||
            slot.is_metadata_owner())
            return false;
        const auto pin =
            std::ranges::find(config.client_priorities, slot.peer, &ExecutionClientPriority::peer);
        return pin == config.client_priorities.end() ||
               space_share_issued[static_cast<std::size_t>(pin - config.client_priorities.begin())];
    }
    bool has_scheduling_policy() const noexcept {
        return config.uses_nccl || has_priority_policy() ||
               config.scheduling_mode != core::ExecutionSchedulingMode::kPriority;
    }
    static bool has_private_boundaries(const CudaSessionSchedulingSnapshot& snapshot) noexcept {
        return snapshot.enqueue_turns || snapshot.external_barriers;
    }
    static bool has_physical_reservation(const CudaSessionSchedulingSnapshot& snapshot) noexcept {
        // An accepted, unstarted boundary needs observation/admission progress,
        // but only a started native boundary reserves physical execution.
        return snapshot.submissions.awaiting_physical_stop || snapshot.submissions.issuing ||
               snapshot.external_work_in_flight;
    }
    CudaSessionResult activate(Slot& slot) noexcept {
        auto result = slot.session->activate();
        if (result.status == CUDA_SUCCESS) selected_session = slot.session;
        return result;
    }
    CUresult detach_selected_context() noexcept {
        CUcontext current{};
        auto status = dispatch.execution.context_get_current(&current);
        if (status != CUDA_SUCCESS) return status;
        if (current) {
            const auto selected = selected_session.lock();
            if (!selected || !selected->health_snapshot().has_context)
                return CUDA_ERROR_INVALID_CONTEXT;
            status = selected->deactivate().status;
            if (status != CUDA_SUCCESS) return status;
            status = dispatch.execution.context_get_current(&current);
            if (status != CUDA_SUCCESS) return status;
            if (current) return CUDA_ERROR_INVALID_CONTEXT;
        }
        selected_session.reset();
        return CUDA_SUCCESS;
    }
    bool stop_execution_domain() noexcept {
        if (has_process_memory()) return false;
        if (!execution_domain() || execution_domain_stopped) return true;
        if (!slots.empty() || serial_pair || profile_pair ||
            detach_selected_context() != CUDA_SUCCESS)
            return false;
        if (execution_domain()->close() != CUDA_SUCCESS ||
            execution_domain()->shutdown() != CUDA_SUCCESS)
            return false;
        execution_domain_stopped = true;
        return true;
    }
    bool has_process_memory() const noexcept {
        return std::ranges::any_of(memory_peers, [](const auto& peer) {
            const auto state = peer.backend->owner()->snapshot();
            return peer.cleanup || state.allocations || state.quota.used_bytes() ||
                   state.is_quarantined || state.has_native_call;
        });
    }
    void observe_process_memory() noexcept {
        for (auto& peer : memory_peers) {
            if (peer.is_retired) continue;
            if (!peer.has_exited) {
                pollfd descriptor{peer.pidfd.get(), POLLIN, 0};
                peer.has_exited = ::poll(&descriptor, 1, 0) == 1 && (descriptor.revents & POLLIN);
                if (peer.has_exited) (void)peer.backend->close();
            }
            if (peer.has_exited)
                for (auto& slot : slots)
                    if (slot.peer == peer.identity)
                        for (auto& callback : slot.host_functions)
                            if (callback.native && callback.is_enqueued) callback.native->finish();
            (void)peer.backend->progress();
            // Cleanup is an idle-only native actor. Socket/Context Close never
            // supplies this authority, nor does process exit prove GPU stop.
            if (!peer.has_exited || !slots.empty() || process_memory_native_gate() != CUDA_SUCCESS)
                continue;
            const auto state = peer.backend->owner()->snapshot();
            if (state.status != CUDA_SUCCESS || state.borrowed || state.has_native_call) continue;
            try {
                if (state.allocations) {
                    if (!peer.cleanup) {
                        if (!next_policy_id || detach_selected_context() != CUDA_SUCCESS) continue;
                        const auto identity = next_policy_id++;
                        peer.cleanup = std::make_shared<CudaExecutionSession>(
                            ExecutionSessionIdentity{identity, incarnation[0]}, identity, dispatch,
                            config.session_limits);
                        const auto initialized = peer.cleanup->initialize(config.device_ordinal);
                        if (peer.cleanup->health_snapshot().has_context)
                            selected_session = peer.cleanup;
                        if (initialized.status != CUDA_SUCCESS) continue;
                    } else {
                        if (peer.cleanup->activate().status != CUDA_SUCCESS) continue;
                        selected_session = peer.cleanup;
                    }
                    for (const auto& allocation : peer.backend->owner()->live_allocations()) {
                        const auto released =
                            peer.backend->release(allocation.id | kExecutionProcessAllocationTag);
                        if (released.status != CUDA_SUCCESS) break;
                    }
                    if (peer.backend->has_allocations()) continue;
                }
                if (peer.cleanup) {
                    const auto stopped = peer.cleanup->shutdown();
                    if (stopped.status != CUDA_SUCCESS ||
                        stopped.disposition != CudaSessionDisposition::kComplete)
                        continue;
                    selected_session.reset();
                    peer.cleanup.reset();
                }
                if (peer.backend->shutdown() == CUDA_SUCCESS) peer.is_retired = true;
            } catch (...) {
                // CPU allocation failure cannot discard the process's leases.
            }
        }
        std::erase_if(memory_peers, [](const auto& peer) { return peer.is_retired; });
    }
#ifdef GLIMMER_HAS_NCCL
    CUresult nccl_entry() noexcept {
        if (nccl_failure) return *nccl_failure;
        CUcontext current{};
        auto status = dispatch.execution.context_get_current(&current);
        if (status == CUDA_SUCCESS && current) {
            const auto selected = selected_session.lock();
            if (!selected || !selected->health_snapshot().has_context)
                status = CUDA_ERROR_INVALID_CONTEXT;
            else
                status = selected->deactivate().status;
            if (status == CUDA_SUCCESS) {
                status = dispatch.execution.context_get_current(&current);
                if (status == CUDA_SUCCESS && current) status = CUDA_ERROR_INVALID_CONTEXT;
            }
        }
        if (status != CUDA_SUCCESS)
            nccl_failure = status;
        else
            selected_session.reset();
        return status;
    }
    std::shared_ptr<CudaExecutionSession> nccl_source(NativeSourceKey key) const noexcept {
        const auto identity = std::ranges::find(native_peers, key.peer, &NativePeer::id);
        if (identity == native_peers.end() || identity->disconnected) return {};
        const auto slot = std::ranges::find(slots, key.session, &Slot::connection);
        if (slot == slots.end() || slot->peer != identity->identity || !slot->is_initialized ||
            slot->is_retiring || slot->is_closing || slot->is_bootstrap_pending || slot->is_nccl ||
            slot->host_owner || slot->is_metadata_owner() || slot->policy_failure || !slot->session)
            return {};
        const auto state = slot->session->health_snapshot();
        return state.has_context && !state.is_quarantined ? slot->session_leases.acquire()
                                                          : nullptr;
    }
    std::uint64_t native_peer(const TaskPeerIdentity& identity, std::uint32_t priority) {
        const auto found = std::ranges::find(native_peers, identity, &NativePeer::identity);
        if (found != native_peers.end())
            return !found->disconnected && found->priority == priority ? found->id : 0;
        if (native_peers.size() == 64) return 0;
        const auto id = static_cast<std::uint64_t>(native_peers.size()) + 1;
        native_peers.push_back({identity, id, priority, false});
        return id;
    }
    std::optional<std::uint64_t> continuation_progress_source(
        std::uint64_t peer, const NcclActorContinuationSnapshot& request) noexcept {
        for (const auto source : request.dependencies) {
            const auto path = trace_continuation_dependency({peer, source}, request);
            if (path.status != CUDA_ERROR_NOT_READY &&
                (path.status != CUDA_SUCCESS || path.has_progress ||
                 path.rejection != CUDA_SUCCESS))
                return source;
        }
        return std::nullopt;
    }
    std::optional<std::uint64_t> native_progress_source(
        std::uint64_t peer, const NcclActorRequestSnapshot& request) noexcept {
        if (request.stage != NcclActorStage::kSourceBlocked) return std::nullopt;
        for (const auto& source : request.dependencies) {
            if (source.is_ready) continue;
            const auto path = trace_nccl_dependency({peer, source.session}, request);
            if (path.status != CUDA_ERROR_NOT_READY &&
                (path.status != CUDA_SUCCESS || path.has_progress ||
                 path.rejection != CUDA_SUCCESS))
                return source.session;
        }
        return std::nullopt;
    }
    std::optional<NcclActorGrant> native_progress_grant(
        const NcclActorReservation& reservation) noexcept {
        std::optional<NcclActorGrant> selected;
        for (const auto& request : reservation.requests)
            if ((request.is_ready || native_progress_source(reservation.peer, request)) &&
                (!selected || request.grant.cutoff < selected->cutoff))
                selected = request.grant;
        for (const auto& request : reservation.continuations)
            if ((request.is_ready || continuation_progress_source(reservation.peer, request)) &&
                (!selected || request.grant.cutoff < selected->cutoff))
                selected = request.grant;
        return selected;
    }
    NcclActorSnapshot nccl_state() noexcept {
        if (!nccl_actor) return {CUDA_SUCCESS};
        auto state = nccl_actor->snapshot();
        if (state.status != CUDA_SUCCESS) {
            nccl_failure = state.status;
            return state;
        }
        std::erase_if(native_policies, [&](const auto& policy) {
            return std::ranges::none_of(state.reservations, [&](const auto& item) {
                return item.reservation == policy.reservation &&
                       item.generation == policy.generation;
            });
        });
        for (const auto& reservation : state.reservations) {
            if (std::ranges::any_of(native_policies, [&](const auto& item) {
                    return item.reservation == reservation.reservation &&
                           item.generation == reservation.generation;
                }))
                continue;
            if (!next_policy_id || native_policies.size() == native_policies.capacity()) {
                state.status = CUDA_ERROR_OUT_OF_MEMORY;
                nccl_failure = state.status;
                return state;
            }
            native_policies.push_back(
                {reservation.reservation, reservation.generation, next_policy_id++});
        }
        return state;
    }
    void observe_nccl() noexcept {
        if (!nccl_actor) return;
        const auto status = nccl_actor->observe();
        if (status != CUDA_SUCCESS) nccl_failure = status;
        (void)nccl_state();
    }
    ExecutionWireReply nccl_rejection(CUresult status) const {
        auto reply = rejection(status);
        ExecutionPayloadWriter output(config.transport.max_payload_bytes);
        if (!write_execution_nccl_reply(output, {})) throw std::bad_alloc();
        reply.data = output.take();
        return reply;
    }
    void reject_continuation(Slot& slot, Slot::Request& request, CUresult status) {
        auto reply = rejection(status);
        if (slot.is_bootstrap_pending) {
            slot.is_bootstrap_pending = false;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kNone);
        }
        request.continuation_rejection = std::move(reply);
    }
    void prepare_continuation(Slot& slot, Slot::Request& request) {
        if (!request.continuation || request.ordinary_continuation ||
            request.continuation_rejection)
            return;
        if (!config.uses_nccl || !nccl_actor || !request.continuation_input) {
            reject_continuation(slot, request, CUDA_ERROR_NOT_SUPPORTED);
            return;
        }
        const auto& scope = *request.continuation;
        const auto operation = request.message.header.operation;
        NcclActorContinuationTarget target;
        target.connection = slot.connection;
        target.wire_sequence = request.message.header.sequence;
        target.operation = operation;
        target.request_lifetime = request.continuation_input;
        target.helper_owner = slot.host_owner_leases.acquire();
        target.is_context_free_helper =
            !scope.origin_source && (slot.host_owner || slot.is_metadata_owner());
        if (!scope.flags && operation == ExecutionOpcode::kClose &&
            scope.origin_source == slot.connection && slot.session && !slot.is_metadata_owner())
            target.closing_source = slot.session_leases.acquire();
        if (operation == ExecutionOpcode::kReleaseHostMemory &&
            scope.flags == kExecutionSourceContinuationCleanup) {
            ExecutionPayloadReader input(request.message.payload);
            if (!input.read_u64(target.cleanup_allocation) || !input.empty()) {
                reject_continuation(slot, request, CUDA_ERROR_INVALID_VALUE);
                return;
            }
        }
        const auto previous_peer = slot.native_peer;
        slot.native_peer = native_peer(slot.peer, slot.priority);
        if (!slot.native_peer) {
            reject_continuation(slot, request, CUDA_ERROR_OUT_OF_MEMORY);
            return;
        }
        if (slot.native_peer != previous_peer) audit_peer(slot, 3);
        const auto accepted = nccl_actor->prepare_continuation({slot.native_peer, slot.priority},
                                                               scope, std::move(target));
        if (accepted.status != CUDA_SUCCESS) {
            reject_continuation(slot, request, accepted.status);
            return;
        }
        if (slot.is_nccl) {
            const auto cancelled = nccl_actor->cancel_continuation(accepted.request);
            if (cancelled != CUDA_SUCCESS) nccl_failure = cancelled;
            reject_continuation(slot, request,
                                cancelled == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED : cancelled);
            return;
        }
        request.ordinary_continuation = accepted.request;
        (void)nccl_state();
    }
    void release_continuation_fences(std::uint64_t continuation) noexcept {
        for (auto& source : slots)
            if (source.admission_fence && source.admission_fence->continuation == continuation)
                source.admission_fence.reset();
    }
    CUresult bind_continuation_dependencies(Slot& slot, Slot::Request& request) {
        if (request.has_continuation_dependencies) return CUDA_SUCCESS;
        // Reserve all CPU owners/captures before publishing any source fence.
        // No later ordinary/native admission may extend a cold handler's cut.
        std::vector<Slot*> sources;
        sources.reserve(4);
        const auto append = [&](std::uint64_t connection) {
            if (!connection) return CUDA_SUCCESS;
            const auto found = std::ranges::find(slots, connection, &Slot::connection);
            if (found == slots.end() || found->peer != slot.peer || !found->session ||
                found->host_owner || found->is_metadata_owner() || found->is_nccl ||
                ((found->is_retiring || found->is_closing) &&
                 request.continuation->flags != kExecutionSourceContinuationCleanup &&
                 !(found->connection == slot.connection && !found->is_retiring &&
                   request.message.header.operation == ExecutionOpcode::kClose)) ||
                found->is_bootstrap_pending || found->policy_failure)
                return CUDA_ERROR_INVALID_HANDLE;
            const auto state = found->session->health_snapshot();
            if (!state.has_context)
                return request.continuation->flags == kExecutionSourceContinuationCleanup &&
                               connection == request.continuation->origin_source &&
                               !state.is_quarantined
                           ? CUDA_SUCCESS
                           : CUDA_ERROR_INVALID_HANDLE;
            if (found->admission_fence &&
                found->admission_fence->continuation != request.ordinary_continuation)
                return CUDA_ERROR_NOT_READY;
            if (std::ranges::find(sources, &*found) == sources.end()) sources.push_back(&*found);
            return CUDA_SUCCESS;
        };
        auto status = append(request.continuation->origin_source);
        if (status != CUDA_SUCCESS) return status;
        if (slot.session && !slot.is_metadata_owner() && !slot.is_bootstrap_pending) {
            status = append(slot.connection);
            if (status != CUDA_SUCCESS) return status;
        }
        if (request.message.header.operation == ExecutionOpcode::kPointerMetadata &&
            !slot.is_bootstrap_pending) {
            ExecutionPointerQuery query;
            ExecutionPayloadReader input(request.message.payload);
            if (!read_execution_pointer_query(input, query) || !input.empty())
                return CUDA_ERROR_INVALID_VALUE;
            status = append(query.query_session);
            if (status != CUDA_SUCCESS) return status;
            if (query.kind == ExecutionPointerKind::kDevice) {
                status = append(query.source_session);
                if (status != CUDA_SUCCESS) return status;
            }
        }
        std::vector<std::shared_ptr<const Slot::AdmissionFence>> fences;
        fences.reserve(sources.size());
        for (const auto* source : sources) {
            if (source->admission_fence) {
                fences.push_back(source->admission_fence);
                continue;
            }
            auto captured = source->session->capture_frontier(std::nullopt, true);
            if (captured.status != CUDA_SUCCESS) return captured.status;
            auto fence = std::make_shared<Slot::AdmissionFence>();
            fence->continuation = request.ordinary_continuation;
            fence->owner = source->session_leases.acquire();
            fence->before = std::move(captured.frontier);
            fences.push_back(std::move(fence));
        }
        // Actor dependency capacity is reserved at prepare. Publication is CPU
        // only; a partial bind never permits handler entry or a broader capture.
        for (std::size_t index = 0; index != sources.size(); ++index) {
            status = nccl_actor->bind_continuation_dependency(
                request.ordinary_continuation,
                {sources[index]->connection, fences[index]->owner, fences[index]->before, {}});
            if (status != CUDA_SUCCESS) return status;
        }
        for (std::size_t index = 0; index != sources.size(); ++index)
            sources[index]->admission_fence = std::move(fences[index]);
        request.has_continuation_dependencies = true;
        return CUDA_SUCCESS;
    }
    void retire_continuation_requests(Slot& slot) noexcept {
        if (!nccl_actor || has_continuation_attempt) return;
        for (std::size_t index = 0; index != slot.requests.size();) {
            auto& request = slot.requests[index];
            if (!request.continuation) {
                ++index;
                continue;
            }
            const auto id = request.ordinary_continuation;
            if (id) {
                if (nccl_actor->cancel_continuation(id) == CUDA_SUCCESS) {
                    release_continuation_fences(id);
                } else {
                    try {
                        poll_continuation_result(slot, request);
                    } catch (...) {
                        nccl_failure = CUDA_ERROR_UNKNOWN;
                        ++index;
                        continue;
                    }
                    const auto observed = nccl_actor->continuation_reply(id);
                    if (!observed.ready) {
                        ++index;
                        continue;
                    }
                    if (request.receipt) {
                        remember_epoch_reply(slot, *request.receipt);
                        if (!slot.session->acknowledge_result(*request.receipt)) {
                            nccl_failure = CUDA_ERROR_UNKNOWN;
                            ++index;
                            continue;
                        }
                    }
                    const auto released = nccl_actor->release_continuation_reply(id);
                    if (released != CUDA_SUCCESS) {
                        nccl_failure = released;
                        ++index;
                        continue;
                    }
                    if (observed.status == CUDA_SUCCESS) release_continuation_fences(id);
                }
            }
            slot.requests.erase(slot.requests.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }
    CUresult begin_helper_retirement(Slot& slot, const Slot::Request* closing = nullptr) noexcept {
        if (!slot.bound_cleanup_parents || slot.cleanup_retirement) return CUDA_SUCCESS;
        if (!nccl_actor || !slot.host_owner) return CUDA_ERROR_INVALID_HANDLE;
        // A completed mailbox still owns an unconsumed result. In particular,
        // close() must not erase a successful import before its parent is bound.
        if (std::ranges::any_of(slot.requests, [&](const auto& request) {
                return &request != closing && (request.has_host_job || request.has_host_shutdown ||
                                               request.ordinary_continuation);
            }))
            return CUDA_ERROR_NOT_READY;
        const auto accepted = nccl_actor->begin_cleanup_retirement(
            {slot.native_peer, slot.priority}, slot.connection, slot.host_owner_leases.acquire(),
            closing ? closing->ordinary_continuation : 0);
        if (accepted.status != CUDA_SUCCESS) return accepted.status;
        slot.cleanup_retirement = accepted.ticket;
        if (accepted.ticket.parents != slot.bound_cleanup_parents) {
            policy_failed(slot, CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }
    void finish_helper_retirement(Slot& slot) noexcept {
        if (!slot.cleanup_retirement || !nccl_actor || !slot.host_owner->snapshot().is_closed)
            return;
        // Parent pins survive through native shutdown. The actor independently
        // authenticates the actual closed owner before consuming its exact set.
        const auto status = nccl_actor->complete_cleanup_retirement(*slot.cleanup_retirement);
        if (status == CUDA_ERROR_NOT_READY) return;
        if (status != CUDA_SUCCESS ||
            slot.cleanup_retirement->parents != slot.bound_cleanup_parents) {
            policy_failed(slot, status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status);
            return;
        }
        slot.bound_cleanup_parents = 0;
        slot.cleanup_retirement.reset();
    }
    std::optional<ExecutionWireReply> execute_nccl(Slot& slot, Slot::Request& request,
                                                   bool can_issue) {
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{}, flags{};
            if (operation != ExecutionOpcode::kHello || !input.read_u32(role) ||
                role != kExecutionNcclRole || !input.read_u32(flags) || flags || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            if (!nccl_driver_ready) {
                if (!can_issue) return std::nullopt;
                const auto initialized = dispatch.initialize(0);
                if (initialized != CUDA_SUCCESS) {
                    slot.is_bootstrap_pending = false;
                    auto reply = rejection(initialized);
                    describe_bootstrap(reply, ExecutionBootstrapOwnership::kNone);
                    return reply;
                }
                nccl_driver_ready = true;
            }
            slot.is_bootstrap_pending = false;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kRetained);
            return reply;
        }
        if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            return reply;  // Closing transport is not native communicator cleanup.
        }
        if (operation != ExecutionOpcode::kNcclCall || !nccl_actor)
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        if (!request.nccl_request) {
            ExecutionNcclRequest wire;
            std::uint64_t primary{};
            if (request.borrowed_descriptor() >= 0 || !input.read_u64(primary) ||
                !read_execution_nccl_request(input, wire) || !input.empty())
                return nccl_rejection(CUDA_ERROR_INVALID_VALUE);
            const auto prepared =
                nccl_actor->prepare({slot.native_peer, slot.priority}, std::move(wire), primary);
            if (prepared.status != CUDA_SUCCESS) return nccl_rejection(prepared.status);
            request.nccl_request = prepared.request;
            (void)nccl_state();
        }
        const auto observed = nccl_actor->reply(request.nccl_request);
        if (!observed.ready || !observed.envelope) return std::nullopt;
        ExecutionPayloadWriter output(config.transport.max_payload_bytes);
        if (!write_execution_nccl_reply(output, *observed.envelope)) throw std::bad_alloc();
        auto reply = rejection(observed.carrier);
        reply.data = output.take();
        reply.original_status = observed.envelope->original_status;
        if (observed.envelope->is_accepted) {
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.id = observed.envelope->operation;
            reply.parameter_bytes = observed.carrier == CUDA_SUCCESS ? 0U : 1U;
        }
        return reply;
    }
#endif
    static bool is_dnn_backend_metadata(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kDnnBackendCreate ||
               operation == ExecutionOpcode::kDnnBackendDestroy ||
               operation == ExecutionOpcode::kDnnBackendInitialize ||
               operation == ExecutionOpcode::kDnnBackendFinalize ||
               operation == ExecutionOpcode::kDnnBackendPrepare ||
               operation == ExecutionOpcode::kDnnBackendSet ||
               operation == ExecutionOpcode::kDnnBackendGet;
    }
    static bool is_host_function_observation(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kQueryHostFunction ||
               operation == ExecutionOpcode::kFinishHostFunction;
    }
    static bool is_host_function_command(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kEnqueueHostFunction ||
               is_host_function_observation(operation);
    }
    static bool is_async_frontier_batch(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kQueryAsyncFrontiers ||
               operation == ExecutionOpcode::kReleaseAsyncFrontiers;
    }
    static bool decode_kernel_request(ExecutionPayloadReader& input, CudaExecutionKernel& kernel,
                                      bool& asynchronous) noexcept {
        asynchronous = false;
        if (!input.read_u64(kernel.function) || !input.read_u64(kernel.stream)) return false;
        for (auto& dimension : kernel.grid)
            if (!input.read_u32(dimension)) return false;
        for (auto& dimension : kernel.block)
            if (!input.read_u32(dimension)) return false;
        if (!input.read_u32(kernel.shared_memory_bytes) || !input.read_blob(kernel.parameters))
            return false;
        if (input.empty()) return true;
        std::uint32_t marker{};
        if (!input.read_u32(marker) || marker != kExecutionAsyncLaunchMagic || !input.empty())
            return false;
        asynchronous = true;  // Explicit versioned ASY1, not an SDK result request.
        return true;
    }
#include "graph_runtime.inc"
    static bool is_asynchronous_admission(const Slot::Request& request) noexcept {
        if (request.continuation || request.message.header.operation != ExecutionOpcode::kLaunch)
            return false;
        ExecutionPayloadReader input(request.message.payload);
        CudaExecutionKernel kernel;
        bool asynchronous{};
        return decode_kernel_request(input, kernel, asynchronous) && asynchronous;
    }
    bool can_receive_async_burst(const ExecutionSocketRequestView& request,
                                 std::optional<std::uint64_t> first_sequence = std::nullopt) const {
        if (graph_construction || !graph_deferred_intake.empty() ||
            priority_idle.has_idle_window() || request.has_descriptor ||
            request.header.kind != ExecutionMessageKind::kRequest ||
            request.header.operation != ExecutionOpcode::kLaunch ||
            config.transport.max_pending_requests < 2 ||
            config.session_limits.submissions.max_commands < 2 ||
            config.session_limits.max_objects < 3)
            return false;
        const auto found = std::ranges::find(slots, request.connection, &Slot::connection);
        if (found == slots.end()) return false;
        const auto& slot = *found;
        if (!slot.session || slot.host_owner || slot.is_metadata_owner() || slot.is_nccl ||
            slot.is_staging_owner || slot.is_pageable_owner || !slot.is_initialized ||
            slot.is_bootstrap_pending || slot.is_closing || slot.is_retiring ||
            slot.policy_failure || slot.asynchronous_failure || !slot.has_request_space ||
            !slot.has_command_space || !slot.has_frontier_space || slot.priority ||
            slot.requests.size() != (first_sequence ? 1 : 0) || request.peer != slot.peer ||
            request.header.session != slot.connection ||
            request.header.incarnation != incarnation ||
            request.header.sequence != slot.next_request ||
            slot.next_command == std::numeric_limits<std::uint64_t>::max() ||
            slot.next_frontier == std::numeric_limits<std::uint64_t>::max())
            return false;
        if (first_sequence) {
            if (*first_sequence == std::numeric_limits<std::uint64_t>::max() ||
                request.header.sequence != *first_sequence + 1)
                return false;
            const auto& first = slot.requests.front();
            if (first.message.header.sequence != *first_sequence ||
                first.message.header.session != slot.connection ||
                first.message.header.incarnation != incarnation || first.descriptor.get() >= 0 ||
                !is_asynchronous_admission(first))
                return false;
        } else if (slot.next_request == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        ExecutionPayloadReader input(request.payload);
        CudaExecutionKernel kernel;
        bool asynchronous{};
        if (!decode_kernel_request(input, kernel, asynchronous) || !asynchronous) return false;
        const auto capacity = config.session_limits.submissions.max_commands - 2;
        if (slot.low_epochs.size() > capacity || slot.asynchronous_receipts.size() > capacity ||
            slot.frontiers.size() > config.session_limits.max_objects - 3)
            return false;
        const auto snapshot = slot.session->scheduling_snapshot();
        return !snapshot.is_quarantined && !snapshot.submissions.is_closing &&
               !snapshot.submissions.is_quarantined && snapshot.submissions.outstanding <= capacity;
    }
    bool receive_async_burst() {
        if (async_intake_peers.empty()) return true;
        {
            const HostScope transport_scope(config.host_stage_metrics.get(), HostStage::kTransport);
            if (!transport.poll_for_ns(0)) return false;
        }
        for (const auto& candidate : async_intake_peers) {
            const auto view = transport.peek_request(candidate.connection);
            if (!view || !can_receive_async_burst(*view, candidate.first_sequence)) continue;
            auto request = transport.take_request(*view);
            if (!request) continue;
            if (async_intake_second_takes != std::numeric_limits<std::uint64_t>::max())
                ++async_intake_second_takes;
            try {
                const HostScope receive_scope(config.host_stage_metrics.get(), HostStage::kReceive);
                receive(std::move(*request));
            } catch (...) {
                transport.close_peer(candidate.connection);
            }
        }
        return true;
    }
    static bool is_metadata(const ExecutionWireMessage& message) noexcept {
        const auto operation = message.header.operation;
        if (operation == ExecutionOpcode::kFunctionControl) {
            ExecutionPayloadReader input(message.payload);
            std::uint32_t control{};
            return input.read_u32(control) &&
                   (control ==
                        static_cast<std::uint32_t>(ExecutionFunctionControl::kGetAttribute) ||
                    control ==
                        static_cast<std::uint32_t>(ExecutionFunctionControl::kGetAttributes));
        }
        if (operation == ExecutionOpcode::kDriverContextControl ||
            operation == ExecutionOpcode::kRuntimeContextControl) {
            ExecutionPayloadReader input(message.payload);
            std::uint32_t control{};
            return input.read_u32(control) && execution_context_control_is_query(
                                                  static_cast<ExecutionContextControl>(control));
        }
        return is_host_function_command(operation) || is_async_frontier_batch(operation) ||
               execution_is_dnn_descriptor_observation(operation) ||
               operation == ExecutionOpcode::kBindDnnProfile ||
               operation == ExecutionOpcode::kQueryStagingIdle ||
               operation == ExecutionOpcode::kPointerMetadata ||
               is_dnn_backend_metadata(operation) ||
               operation == ExecutionOpcode::kDnnBackendContext ||
               operation == ExecutionOpcode::kPrepareDnnBackendExecute ||
               operation == ExecutionOpcode::kSnapshot || operation == ExecutionOpcode::kDrain ||
               operation == ExecutionOpcode::kCaptureFrontier ||
               operation == ExecutionOpcode::kQueryFrontier ||
               operation == ExecutionOpcode::kReleaseFrontier ||
               operation == ExecutionOpcode::kQueryEvent ||
               operation == ExecutionOpcode::kSynchronizeEvent ||
               operation == ExecutionOpcode::kEventElapsedTime ||
               operation == ExecutionOpcode::kGetContextFlags ||
               operation == ExecutionOpcode::kSetContextFlags ||
               operation == ExecutionOpcode::kCreateLtMetadata ||
               operation == ExecutionOpcode::kDestroyLtMetadata ||
               operation == ExecutionOpcode::kSetLtAttribute ||
               operation == ExecutionOpcode::kGetLtAttribute ||
               operation == ExecutionOpcode::kLtVersion ||
               operation == ExecutionOpcode::kLtAlgorithmCapability ||
               operation == ExecutionOpcode::kCreateDnnMetadata ||
               operation == ExecutionOpcode::kDestroyDnnMetadata ||
               operation == ExecutionOpcode::kSetDnnTensor ||
               operation == ExecutionOpcode::kSetDnnActivation ||
               operation == ExecutionOpcode::kGetDnnStream ||
               operation == ExecutionOpcode::kSetDnnFilter ||
               operation == ExecutionOpcode::kSetDnnConvolution ||
               operation == ExecutionOpcode::kSetDnnConvolutionAttribute ||
               operation == ExecutionOpcode::kDnnConvolutionQuery ||
               operation == ExecutionOpcode::kDnnBackwardConvolutionQuery ||
               operation == ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors ||
               operation == ExecutionOpcode::kDnnRnnQuery ||
               operation == ExecutionOpcode::kDnnBuildRnnDynamic ||
               operation == ExecutionOpcode::kDnnRnnWeightParams ||
               operation == ExecutionOpcode::kDnnAttnWeights ||
               operation == ExecutionOpcode::kPrepareDnnAttn ||
               operation == ExecutionOpcode::kSetDnnPooling ||
               operation == ExecutionOpcode::kSetDnnLrn ||
               operation == ExecutionOpcode::kSetDnnReduction ||
               operation == ExecutionOpcode::kDnnReductionQuery ||
               operation == ExecutionOpcode::kDnnBatchNormQuery ||
               operation == ExecutionOpcode::kDnnDropoutStatesSize ||
               operation == ExecutionOpcode::kSetDnnOpTensor ||
               operation == ExecutionOpcode::kSetDnnTensorTransform ||
               operation == ExecutionOpcode::kDnnInitTransformDest ||
               operation == ExecutionOpcode::kSetDnnRnnData ||
               operation == ExecutionOpcode::kSetDnnSeqData ||
               operation == ExecutionOpcode::kSetDnnRnn ||
               operation == ExecutionOpcode::kSetDnnAttn ||
               operation == ExecutionOpcode::kSetDnnRnnClip ||
               operation == ExecutionOpcode::kDnnLibraryInfo ||
               operation == ExecutionOpcode::kDnnLegacyMetadata ||
               operation == ExecutionOpcode::kDnnLegacyControl ||
               operation == ExecutionOpcode::kLtHeuristic ||
               operation == ExecutionOpcode::kLtAlgorithmCheck;
    }
    static bool is_original_blas_command(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kGeam || operation == ExecutionOpcode::kSgemv ||
               operation == ExecutionOpcode::kSgemmEx;
    }
    static bool is_execution_command(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kGraphLaunch ||
               operation == ExecutionOpcode::kProcessMemory ||
               operation == ExecutionOpcode::kLaunch || operation == ExecutionOpcode::kMemory ||
               operation == ExecutionOpcode::kUnifiedCopy ||
               operation == ExecutionOpcode::kHostTransfer ||
               operation == ExecutionOpcode::kStagedHostTransfer ||
               operation == ExecutionOpcode::kSgemm || operation == ExecutionOpcode::kGemmEx ||
               operation == ExecutionOpcode::kStridedGemmEx ||
               operation == ExecutionOpcode::kStridedSgemm ||
               operation == ExecutionOpcode::kLtMatmul || is_original_blas_command(operation) ||
               is_dnn_computation(operation);
    }
    static bool is_ordering_command(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kEnqueueHostFunction ||
               operation == ExecutionOpcode::kRecordEvent ||
               operation == ExecutionOpcode::kStreamWaitEvent ||
               operation == ExecutionOpcode::kWaitForeignEvent;
    }
    static bool is_dnn_computation(ExecutionOpcode operation) noexcept {
        return operation == ExecutionOpcode::kDnnCompute ||
               operation == ExecutionOpcode::kDnnConvolutionForward ||
               operation == ExecutionOpcode::kDnnConvolutionFind ||
               operation == ExecutionOpcode::kDnnTensorForward ||
               operation == ExecutionOpcode::kDnnReduction ||
               operation == ExecutionOpcode::kDnnBatchNorm ||
               operation == ExecutionOpcode::kDnnDropout ||
               operation == ExecutionOpcode::kDnnRnnCompute ||
               operation == ExecutionOpcode::kDnnLegacyCompute ||
               operation == ExecutionOpcode::kDnnAttnCompute ||
               operation == ExecutionOpcode::kDnnBackendExecute ||
               operation == ExecutionOpcode::kDnnReorderFilterAndBias;
    }
    void audit(std::string_view event, const Slot& slot, std::uint64_t sequence,
               CUresult status = CUDA_SUCCESS) const noexcept {
        if (!config.should_audit_priority || !audit_log) return;
        const auto epoch = std::ranges::find(slot.low_epochs, sequence, &Slot::LowEpoch::sequence);
        const auto progress =
            epoch == slot.low_epochs.end() ? CudaSessionPersistentProgress{} : epoch->progress;
        try {
            (void)audit_log->info(
                "Priority execution audit: event={}, session={}, priority={}, sequence={}, "
                "completed_blocks={}, total_blocks={}, physical_launches={}, status={}",
                event, slot.connection, slot.priority, sequence, progress.completed_blocks,
                progress.total_blocks, progress.physical_launches, static_cast<int>(status));
        } catch (...) {
            // Diagnostics must never publish execution/physical authority.
        }
    }
    void policy_failed(Slot& slot, CUresult status) noexcept {
        if (slot.policy_failure) return;
        slot.policy_failure = status;
        if (!slot.asynchronous_failure) slot.asynchronous_failure = status;
        audit("policy_failure", slot, 0, status);
    }
    void remember_epoch_reply(Slot& slot, const ExecutionReceipt& receipt) noexcept {
        const auto epoch =
            std::ranges::find(slot.low_epochs, receipt.sequence(), &Slot::LowEpoch::sequence);
        if (epoch != slot.low_epochs.end() && !epoch->is_event && !epoch->host_reply)
            epoch->host_reply = slot.session->host_result(epoch->receipt);
    }
    void harvest_asynchronous_results(Slot& slot) noexcept {
        for (std::size_t index = 0; index < slot.asynchronous_receipts.size();) {
            const auto& pending = slot.asynchronous_receipts[index];
            const auto host = slot.session->host_result(pending.receipt);
            if (!host) {
                ++index;
                continue;
            }
            // The backend publishes initial errors into the retained exact
            // frontier before exposing its host reply. Never erase the sole
            // error provenance to make this queue slot available again.
            const auto observation = pending.frontier.observe();
            if (host->status != CUDA_SUCCESS &&
                (observation == CUDA_SUCCESS || observation == CUDA_ERROR_NOT_READY)) {
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                ++index;
                continue;
            }
            remember_epoch_reply(slot, pending.receipt);
            if (!slot.session->acknowledge_result(pending.receipt)) {
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                ++index;
                continue;
            }
            slot.asynchronous_receipts.erase(slot.asynchronous_receipts.begin() +
                                             static_cast<std::ptrdiff_t>(index));
        }
    }
    CUresult reserve_request_frontier(Slot& slot, Slot::Request& request) noexcept {
        if (request.command_reservation) return CUDA_SUCCESS;
        auto reserved = slot.session->reserve_command_frontier(slot.next_command);
        if (reserved.status == CUDA_SUCCESS)
            request.command_reservation = std::move(reserved.reservation);
        return reserved.status;
    }
    bool bind_request_frontier(Slot& slot, Slot::Request& request) noexcept {
        if (!request.command_reservation || !request.receipt) {
            policy_failed(slot, CUDA_ERROR_UNKNOWN);
            return false;
        }
        auto captured = slot.session->reserved_command_frontier(*request.command_reservation);
        if (captured.status != CUDA_SUCCESS || !slot.session->owns_frontier(captured.frontier)) {
            // Acceptance cannot be rolled back by rejecting unrelated work.
            // Retain the receipt and seal this owner if its promised binding is
            // unavailable; no allocating recovery capture or native retry.
            policy_failed(slot,
                          captured.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : captured.status);
            return false;
        }
        remember_command_frontier(request, std::move(captured.frontier));
        request.command_reservation.reset();
        return true;
    }
    void remember_command_frontier(Slot::Request& request,
                                   CudaExecutionFrontier frontier) noexcept {
        if (config.should_audit_native || config.should_audit_priority)
            request.command_capture = frontier.diagnostic_id();
        request.command_frontier = std::move(frontier);
    }
    void remember_synchronous_frontier(Slot& slot, Slot::Request& request,
                                       const CudaSessionResult& result) noexcept {
        if (slot.session->owns_frontier(result.frontier))
            remember_command_frontier(request, result.frontier);
    }
    void release_unaccepted_foreign_events(Slot& slot) noexcept {
        for (auto& request : slot.requests)
            if (request.foreign_event && !request.foreign_event->target_sequence)
                request.foreign_event.reset();
        std::erase_if(foreign_events, [&](const auto& edge) {
            return edge->target_id == slot.connection && !edge->target_sequence;
        });
    }
    void retire_foreign_events() noexcept {
        std::erase_if(foreign_events, [](const auto& edge) {
            // Error status alone cannot release either native owner. A known
            // rejected wait may retire without ever issuing native work.
            return edge->target_sequence && edge->target->owns_frontier(edge->target_frontier) &&
                   edge->target_frontier.retirement().is_retired;
        });
    }
    void audit_ordering(const Slot& slot, const Slot::LowEpoch& epoch, std::uint32_t event,
                        CUresult status) noexcept {
        if (!config.should_audit_native && !config.should_audit_priority) return;
        const auto capture = epoch.ordering_capture;
        if (!epoch.event_identity || !epoch.event_identity->event_object || !capture) {
            semantic_lost(4);
            return;
        }
        (void)semantic(3, [&](auto& json) {
            json.add("event", event);
            json.add("connection", slot.connection);
            json.add("priority", slot.priority);
            json.add("sequence", epoch.sequence);
            json.add("opcode", static_cast<unsigned int>(epoch.ordering_operation));
            json.add("capture", capture);
            json.add("event_source", epoch.event_source);
            json.add("event_object", epoch.event_identity->event_object);
            json.add("record_sequence", epoch.event_identity->record_sequence);
            json.add("event_capture", epoch.event_identity->frontier_capture);
            json.add("status", static_cast<int>(status));
        });
    }
    void account_event(Slot& slot, const CudaSessionEventSubmission& accepted,
                       ExecutionOpcode operation, std::uint64_t source = 0) noexcept {
        if (!accepted.result.id) return;
        // The caller preflights the reserved vector before backend acceptance.
        // Track high events too, to distinguish ordering from compute audits.
        slot.low_epochs.push_back(
            {.sequence = accepted.result.id,
             .frontier = accepted.frontier,
             .progress = {},
             .is_event = true,
             .ordering_operation = operation,
             .event_source = source ? source : slot.connection,
             .ordering_capture = config.should_audit_native || config.should_audit_priority
                                     ? accepted.frontier.diagnostic_id()
                                     : 0,
             .event_identity = accepted.generation});
        audit_ordering(slot, slot.low_epochs.back(), 1, accepted.result.status);
        audit("event_accepted", slot, accepted.result.id, accepted.result.status);
        if (slot.next_command == std::numeric_limits<std::uint64_t>::max())
            slot.has_command_space = false;
        else
            ++slot.next_command;
    }
    void drain_retiring_host_requests(Slot& slot) noexcept {
        for (std::size_t index = 0; index != slot.requests.size();) {
            auto& request = slot.requests[index];
            if (request.ordinary_continuation) {
                ++index;
                continue;
            }
            if (request.has_host_job || request.has_host_shutdown) {
                try {
                    const auto reply = execute_host(slot, request, false);
                    if (!reply && (request.has_host_job || request.has_host_shutdown)) {
                        ++index;
                        continue;
                    }
                    // Consume accepted mailbox outcomes even without an SCN1
                    // parent. A known Deferred permits only fresh-grant owner
                    // cleanup, not replay of the vanished caller's request.
                } catch (...) {
                    policy_failed(slot, CUDA_ERROR_UNKNOWN);
                    ++index;
                    continue;
                }
            }
            slot.requests.erase(slot.requests.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }
    void retire_host_functions(Slot& slot) noexcept {
        for (std::size_t index = 0; index != slot.host_functions.size();) {
            auto& function = slot.host_functions[index];
            if (function.has_permission || function.is_uncertain) {
                // No finish receipt exists after disconnect. Keep the logical
                // boundary and its stream/context owners; no GPU stop is implied.
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                function.is_uncertain = true;
                ++index;
                continue;
            }
            if (slot.session_leases.has_leases()) {
                // A native actor or foreign dependency can retain this exact
                // source beyond its connection. Settling the unrun callback as
                // success would release that owner's already accepted suffix,
                // even though ordinary source retirement cannot reject it.
                // The callback is known absent, not executed: retain its bounded
                // boundary and fail scheduling instead of inventing success.
                policy_failed(slot, CUDA_ERROR_DEINITIALIZED);
                function.is_uncertain = true;
                ++index;
                continue;
            }
            const auto cancelled = slot.session->cancel_external_barrier(function.barrier);
            if (cancelled != CUDA_SUCCESS) {
                policy_failed(slot, cancelled);
                function.is_uncertain = true;
                ++index;
                continue;
            }
            slot.host_functions.erase(slot.host_functions.begin() +
                                      static_cast<std::ptrdiff_t>(index));
        }
    }
    void retire(Slot& slot) noexcept {
        if (!slot.is_retiring) lifecycle(HostStageMetrics::Lifecycle::kRetire, slot);
        if (!slot.is_retiring) audit_peer(slot, 2);
        slot.is_retiring = true;
        // Capture construction is a distinct native lifetime. Ordinary peer
        // retirement may submit helper cleanup or touch private events.
        if (graph_construction) return;
        retire_host_functions(slot);
        if (profile_pair && (slot.policy_id == profile_pair->policies[0] ||
                             slot.policy_id == profile_pair->policies[1]))
            invalidate_profile_pair();
#ifdef GLIMMER_HAS_NCCL
        retire_continuation_requests(slot);
#endif
        if (slot.host_owner) {
            drain_retiring_host_requests(slot);
            if (!slot.requests.empty()) return;
        }
#ifdef GLIMMER_HAS_NCCL
        if (slot.host_owner && (slot.bound_cleanup_parents || slot.cleanup_retirement)) {
            const auto prepared = begin_helper_retirement(slot);
            if (prepared != CUDA_SUCCESS) {
                if (prepared != CUDA_ERROR_NOT_READY) policy_failed(slot, prepared);
                return;
            }
            finish_helper_retirement(slot);
        }
#endif
        // A staged, never-accepted wait owns no native command. Drop these
        // reservations before their shared target pins could defer retirement.
        release_unaccepted_foreign_events(slot);
        if (slot.is_retirement_started) return;
        // Accepted native groups/calls own these exact source Sessions, including
        // their queued pre-cut producers. Socket loss cannot reject that work.
        if (slot.session && !slot.is_metadata_owner() && slot.session_leases.has_leases()) return;
        if (slot.host_owner && slot.host_owner_leases.has_leases()
#ifdef GLIMMER_HAS_NCCL
            && !slot.cleanup_retirement
#endif
        )
            return;
        slot.is_retirement_started = true;
        slot.frontiers.clear();  // CPU metadata only; never releases command/GPU ownership.
#ifdef GLIMMER_HAS_NCCL
        if (slot.is_nccl) {
            const bool connected = std::ranges::any_of(slots, [&](const auto& other) {
                return &other != &slot && other.is_nccl && !other.is_retiring &&
                       other.native_peer == slot.native_peer;
            });
            if (!connected && nccl_actor) {
                (void)nccl_actor->disconnect(slot.native_peer);
                const auto peer =
                    std::ranges::find(native_peers, slot.native_peer, &NativePeer::id);
                if (peer != native_peers.end()) peer->disconnected = true;
            }
            // Pending envelopes remain until actually observed/released. Their
            // resource owners are actor records, not the vanished connection.
            return;
        }
#endif
#ifdef GLIMMER_HAS_CUDNN
        if (slot.dnn_metadata) {
            slot.dnn_metadata->close();
            slot.requests.clear();
            return;
        }
#endif
        if (slot.lt_metadata) {
            slot.lt_metadata->close();
            slot.requests.clear();
            return;
        }
        if (slot.host_owner) {
            slot.host_owner->close();
            slot.requests.clear();
            return;
        }
        slot.session->close();
        (void)slot.session->reject_queued(CUDA_ERROR_DEINITIALIZED);
        harvest_asynchronous_results(slot);
        for (const auto& request : slot.requests)
            if (request.receipt && slot.session->host_result(*request.receipt)) {
                remember_epoch_reply(slot, *request.receipt);
                (void)slot.session->acknowledge_result(*request.receipt);
            }
        slot.requests.clear();
    }
    void lifecycle(HostStageMetrics::Lifecycle kind, const Slot& slot, std::uint64_t wire = 0,
                   std::uint64_t command = 0, std::uint64_t detail = 0,
                   std::uint64_t frame_received_ns = 0, std::uint64_t owner_enter_ns = 0,
                   std::uint64_t first_read_ns = 0) noexcept {
        if (config.host_stage_metrics) {
            if (kind == HostStageMetrics::Lifecycle::kReceive) {
                const std::uint64_t slot_kind = slot.is_metadata_owner() ? 1
                                                : slot.host_owner        ? 2
                                                : slot.is_nccl           ? 3
                                                                         : 0;
                detail |= slot_kind << 32;
            }
            config.host_stage_metrics->lifecycle(
                kind, {slot.connection, slot.peer.pid, slot.peer.start_time_ticks}, wire, command,
                detail, frame_received_ns, owner_enter_ns, first_read_ns);
        }
    }
    void drop(Slot& slot) noexcept {
        transport.close_peer(slot.connection);
        retire(slot);
    }
    bool queue_reply(Slot& slot, const Slot::Request& request,
                     const ExecutionWireReply& reply) noexcept {
        const auto operation = request.message.header.operation;
        const bool should_audit_copy = config.should_audit_copy_requests &&
                                       (operation == ExecutionOpcode::kImportHostMemory ||
                                        operation == ExecutionOpcode::kHostTransfer ||
                                        operation == ExecutionOpcode::kStagedHostTransfer ||
                                        operation == ExecutionOpcode::kReleaseHostMemory);
        std::uint64_t queued_ns{};
        auto payload = encode_execution_reply(reply, config.transport.max_payload_bytes);
        if (!payload ||
            !transport.reply(
                slot.connection,
                {{ExecutionMessageKind::kReply, request.reply_operation(), 0, slot.connection,
                  request.message.header.sequence, incarnation},
                 payload ? std::move(*payload) : std::vector<std::byte>{}},
                should_audit_copy || config.host_stage_metrics ? &queued_ns : nullptr)) {
            drop(slot);
            return false;
        }
        if (config.host_stage_metrics)
            config.host_stage_metrics->reply_at(
                queued_ns, {slot.connection, slot.peer.pid, slot.peer.start_time_ticks},
                request.message.header.sequence, request.receipt ? request.receipt->sequence() : 0,
                static_cast<std::uint32_t>(reply.disposition) |
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(reply.status)) << 32));
        // Transport now owns the exact result bytes. Acknowledging host metadata
        // cannot release parameters or native resources before physical proof.
        if (request.receipt && reply.disposition != ExecutionWireDisposition::kAdmissionAccepted) {
            remember_epoch_reply(slot, *request.receipt);
            if (!slot.session->acknowledge_result(*request.receipt)) {
                drop(slot);
                return false;
            }
        }
#ifdef GLIMMER_HAS_NCCL
        if (request.nccl_request && nccl_actor) {
            const auto status = nccl_actor->release_reply(request.nccl_request);
            if (status != CUDA_SUCCESS) {
                nccl_failure = status;
                drop(slot);
                return false;
            }
        }
        if (request.ordinary_continuation && nccl_actor) {
            const auto status =
                nccl_actor->release_continuation_reply(request.ordinary_continuation);
            if (status != CUDA_SUCCESS) {
                nccl_failure = status;
                drop(slot);
                return false;
            }
        }
#endif
        if (request.message.header.operation == ExecutionOpcode::kHello)
            slot.is_initialized = reply.status == CUDA_SUCCESS &&
                                  reply.disposition == ExecutionWireDisposition::kComplete;
        if (should_audit_copy) {
            try {
                if (audit_log)
                    (void)audit_log->info(
                        "Copy request audit: v=1, connection={}, session={}, request={}, "
                        "command={}, "
                        "pid={}, "
                        "operation={}, frame_ns={}, owner_ns={}, reply_queued_ns={}, status={}; "
                        "gpu_completion_qualified=false",
                        slot.connection, request.message.header.session,
                        request.message.header.sequence,
                        request.receipt ? request.receipt->sequence() : 0, slot.peer.pid,
                        static_cast<std::uint32_t>(operation), request.frame_received_ns,
                        request.owner_received_ns, queued_ns, reply.status);
            } catch (...) {
                // Missing diagnostics cannot change accepted work or SDK results.
            }
        }
        return true;
    }
    void reject_bootstrap(const ExecutionReceivedRequest& request, CUresult status,
                          ExecutionOpcode wire_operation = ExecutionOpcode::kHello,
                          std::optional<ExecutionSourceContinuation> continuation = std::nullopt,
                          // Ownership transfers to the NCCL actor when it is enabled.
                          // NOLINTNEXTLINE(performance-unnecessary-value-param)
                          std::shared_ptr<const Slot::ContinuationInput> lifetime = {}) noexcept {
#ifdef GLIMMER_HAS_NCCL
        if (continuation && !continuation->flags && nccl_actor && config.uses_nccl && lifetime) {
            const auto trusted = std::ranges::find(config.client_priorities, request.peer,
                                                   &ExecutionClientPriority::peer);
            if (!has_priority_policy() || trusted != config.client_priorities.end()) {
                const auto priority = has_priority_policy() ? trusted->priority : 0;
                const auto peer = native_peer(request.peer, priority);
                if (peer) {
                    NcclActorContinuationTarget target;
                    target.connection = request.connection;
                    target.wire_sequence = request.message.header.sequence;
                    target.operation = ExecutionOpcode::kHello;
                    target.request_lifetime = std::move(lifetime);
                    const auto prepared = nccl_actor->prepare_continuation(
                        {peer, priority}, *continuation, std::move(target));
                    if (prepared.status == CUDA_SUCCESS) {
                        // Canonical no-entry rejection still consumes its
                        // authenticated dense ordinal. Cancellation is CPU-only.
                        const auto cancelled = nccl_actor->cancel_continuation(prepared.request);
                        if (cancelled != CUDA_SUCCESS) {
                            nccl_failure = cancelled;
                            status = cancelled;
                        }
                    } else {
                        status = prepared.status;
                    }
                }
            }
        }
#else
        (void)continuation;
        (void)lifetime;
#endif
        auto reply = rejection(status);
        try {
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kNone);
        } catch (...) {
            transport.close_peer(request.connection);
            return;
        }
        auto payload = encode_execution_reply(reply, config.transport.max_payload_bytes);
        if (!payload ||
            !transport.reply(request.connection,
                             {{ExecutionMessageKind::kReply, wire_operation, 0, request.connection,
                               request.message.header.sequence, incarnation},
                              payload ? std::move(*payload) : std::vector<std::byte>{}}))
            transport.close_peer(request.connection);
    }
    void receive(ExecutionReceivedRequest request) {
        const auto owner_received_ns =
            config.should_audit_copy_requests ||
                    (config.host_stage_metrics && config.host_stage_metrics->records_intake())
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count()
                : 0;
        if (graph_construction && request.connection != graph_construction->connection) {
            // Defer even cold Hello factories: native metadata initialization
            // must not escape capture exclusion. The parked envelopes are bounded.
            if (graph_deferred_intake.size() == config.transport.max_pending_requests) {
                transport.close_peer(request.connection);
                return;
            }
            graph_deferred_intake.push_back(std::move(request));
            return;
        }
        const auto normalized =
            normalize_execution_request(request.message.header.operation, request.message.payload,
                                        config.transport.max_payload_bytes);
        if (!normalized) {
            transport.close_peer(request.connection);
            return;
        }
        const auto wire_operation = request.message.header.operation;
        std::shared_ptr<const Slot::ContinuationInput> continuation_input;
        if (normalized->continuation) {
            auto owned = std::make_shared<Slot::ContinuationInput>();
            owned->wire = std::move(request.message);
            owned->descriptor = std::move(request.descriptor);
            request.message.header = owned->wire.header;
            request.message.header.operation = normalized->operation;
            request.message.header.payload_bytes =
                static_cast<std::uint32_t>(normalized->payload.size());
            request.message.payload.assign(normalized->payload.begin(), normalized->payload.end());
            continuation_input = std::move(owned);
        }
        const auto retain_request = [&] {
            Slot::Request pending{std::move(request.message), std::nullopt,
                                  std::move(request.descriptor)};
            pending.frame_received_ns = request.frame_received_ns;
            pending.first_read_ns = request.first_read_ns;
            pending.owner_received_ns =
                owner_received_ns > 0 ? static_cast<std::uint64_t>(owner_received_ns) : 0;
            pending.continuation = normalized->continuation;
            pending.continuation_input = std::move(continuation_input);
            return pending;
        };
        const auto reject_initial = [&](CUresult status) {
            reject_bootstrap(request, status, wire_operation, normalized->continuation,
                             continuation_input);
        };
        auto found = std::ranges::find_if(
            slots, [&](const auto& slot) { return slot.connection == request.connection; });
        const auto& header = request.message.header;
        if (found == slots.end()) {
            if (header.operation != ExecutionOpcode::kHello || header.sequence != 1 ||
                header.session || header.incarnation != std::array<std::uint64_t, 2>{}) {
                transport.close_peer(request.connection);
                return;
            }
            if (slots.size() == config.transport.max_peers || !next_policy_id) {
                reject_initial(CUDA_ERROR_OUT_OF_MEMORY);
                return;
            }
            const auto trusted =
                std::ranges::find_if(config.client_priorities,
                                     [&](const auto& entry) { return entry.peer == request.peer; });
            if (has_priority_policy() && trusted == config.client_priorities.end()) {
                reject_initial(CUDA_ERROR_NOT_PERMITTED);
                return;
            }
            bool is_host_owner = request.message.payload.size() == 8;
            bool is_staging_owner{}, is_pageable_owner{};
            bool is_lt_metadata{};
            bool is_dnn_metadata{};
            bool is_pointer_metadata{};
            bool is_process_memory{};
            bool is_nccl{};
            std::optional<ExecutionContextBootstrap> context_bootstrap;
            if (is_host_owner) {
                ExecutionPayloadReader input(request.message.payload);
                std::uint32_t role{}, flags{};
                if (!input.read_u32(role) ||
                    (role != kExecutionHostOwnerRole && role != kExecutionStagingOwnerRole &&
                     role != kExecutionPageableOwnerRole && role != kExecutionLtMetadataRole &&
                     role != kExecutionDnnMetadataRole && role != kExecutionPointerMetadataRole &&
                     role != kExecutionProcessMemoryRole && role != kExecutionNcclRole) ||
                    !input.read_u32(flags) || !input.empty()) {
                    reject_initial(CUDA_ERROR_INVALID_VALUE);
                    return;
                }
                is_pageable_owner = role == kExecutionPageableOwnerRole;
                is_staging_owner = role == kExecutionStagingOwnerRole || is_pageable_owner;
                is_lt_metadata = role == kExecutionLtMetadataRole;
                is_dnn_metadata = role == kExecutionDnnMetadataRole;
                is_pointer_metadata = role == kExecutionPointerMetadataRole;
                is_process_memory = role == kExecutionProcessMemoryRole;
                is_nccl = role == kExecutionNcclRole;
                if (is_nccl && normalized->continuation) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                }
                if ((is_lt_metadata || is_dnn_metadata || is_pointer_metadata ||
                     is_process_memory || is_nccl) &&
                    flags) {
                    reject_initial(CUDA_ERROR_INVALID_VALUE);
                    return;
                }
                is_host_owner = !is_lt_metadata && !is_dnn_metadata && !is_pointer_metadata &&
                                !is_process_memory && !is_nccl;
                if (is_process_memory && !dispatch.process_memory.allocate) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                }
                if (is_nccl) {
#ifdef GLIMMER_HAS_NCCL
                    if (!config.uses_nccl || !nccl_actor) {
#else
                    {
#endif
                        reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                        return;
                    }
                }
#ifndef GLIMMER_HAS_CUDNN
                if (is_dnn_metadata) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                }
#endif
                if (is_host_owner && !is_pageable_owner &&
                    (!dispatch.host_register || !dispatch.host_unregister || !dispatch.host_map)) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                }
            } else {
                context_bootstrap = decode_execution_context_bootstrap(request.message.payload);
                if (!context_bootstrap) {
                    reject_initial(CUDA_ERROR_INVALID_VALUE);
                    return;
                }
                if (config.paired && config.paired->is_isolated_profile &&
                    (context_bootstrap->purpose != ExecutionContextPurpose::kRuntimePrimary ||
                     std::ranges::any_of(slots, [&](const auto& prior) {
                         return prior.is_paired_member && prior.peer == request.peer;
                     }))) {
                    reject_initial(CUDA_ERROR_NOT_PERMITTED);
                    return;
                }
            }
            Slot slot;
            slot.requests.reserve(config.transport.max_pending_requests);
            slot.frontiers.reserve(config.session_limits.max_objects);
            slot.asynchronous_receipts.reserve(config.session_limits.submissions.max_commands);
            slot.host_functions.reserve(config.session_limits.max_objects);
            slot.low_epochs.reserve(config.session_limits.submissions.max_commands);
            slot.connection = request.connection;
            slot.policy_id = next_policy_id++;
            slot.peer = request.peer;
            slot.is_staging_owner = is_staging_owner;
            slot.is_pageable_owner = is_pageable_owner;
            slot.is_pointer_metadata = is_pointer_metadata;
            slot.is_process_memory = is_process_memory;
            slot.is_nccl = is_nccl;
            if (context_bootstrap) slot.context_bootstrap = *context_bootstrap;
            const auto kernel_mode =
                has_priority_policy() ? trusted->kernel_mode : ExecutionClientKernelMode::kAuto;
            slot.kernel_mode = kernel_mode;
            if (has_priority_policy()) {
                slot.priority = trusted->priority;
                // This is the retained epoch/audit cohort, not Core's numeric
                // priority comparison. Configured highest peers need it too.
                slot.is_low = slot.priority < highest_priority ||
                              kernel_mode == ExecutionClientKernelMode::kConfigured;
            }
            slot.is_bootstrap_pending = true;
            const bool uses_original = config.uses_original_kernels ||
                                       kernel_mode == ExecutionClientKernelMode::kOriginal ||
                                       (core::is_priority_execution_mode(config.scheduling_mode) &&
                                        has_priority_policy() && !slot.is_low);
            slot.uses_original = uses_original;
            const bool excludes_profile =
                kernel_mode == ExecutionClientKernelMode::kOriginal ||
                (core::is_priority_execution_mode(config.scheduling_mode) &&
                 has_priority_policy() && !slot.is_low);
            // These owners have distinct initialization when their SDKs are enabled.
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (is_nccl) {
#ifdef GLIMMER_HAS_NCCL
                slot.native_peer = native_peer(slot.peer, slot.priority);
                if (!slot.native_peer) {
                    reject_initial(CUDA_ERROR_OUT_OF_MEMORY);
                    return;
                }
#endif
            } else if (is_dnn_metadata) {
#ifdef GLIMMER_HAS_CUDNN
                if (!dispatch.dnn) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                }
                try {
                    slot.dnn_metadata = std::make_unique<CudaDnnRegistry>(
                        *dispatch.dnn, config.session_limits.max_objects);
                } catch (const std::invalid_argument&) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                } catch (const std::bad_alloc&) {
                    reject_initial(CUDA_ERROR_OUT_OF_MEMORY);
                    return;
                }
#endif
            } else if (is_lt_metadata) {
                try {
                    slot.lt_metadata = std::make_unique<CudaLtRegistry>(
                        dispatch.lt, config.session_limits.max_objects);
                } catch (const std::invalid_argument&) {
                    reject_initial(CUDA_ERROR_NOT_SUPPORTED);
                    return;
                } catch (const std::bad_alloc&) {
                    reject_initial(CUDA_ERROR_OUT_OF_MEMORY);
                    return;
                }
            } else if (is_host_owner)
                slot.host_owner = std::make_shared<HostOwner>(
                    ExecutionSessionIdentity{request.connection, incarnation[0]},
                    request.connection, dispatch, config.session_limits, config.device_ordinal,
                    config.transport.wake);
            else {
                CudaExecutionDomainMember member;
                if (execution_domain() && context_bootstrap &&
                    (context_bootstrap->purpose == ExecutionContextPurpose::kRuntimePrimary ||
                     (uses_priority_space_share() &&
                      (context_bootstrap->purpose == ExecutionContextPurpose::kProcessPrimary ||
                       context_bootstrap->purpose == ExecutionContextPurpose::kExplicitUser))) &&
                    trusted != config.client_priorities.end()) {
                    auto created = execution_domain()->create_member(
                        {{request.connection, incarnation[0]}, request.connection},
                        uses_priority_space_share()
                            ? std::optional<unsigned int>{context_bootstrap->flags}
                            : std::nullopt);
                    if (created.status != CUDA_SUCCESS) {
                        reject_initial(created.status);
                        return;
                    }
                    member = std::move(created.member);
                    slot.is_paired_member = true;
                }
                slot.session = std::make_shared<CudaExecutionSession>(
                    ExecutionSessionIdentity{request.connection, incarnation[0]},
                    request.connection, dispatch, config.session_limits,
                    uses_original ? CudaSessionPersistentConfig{} : config.persistent, ptx_cache,
                    excludes_profile ? nullptr : profile_cache,
                    // Configured highest peers need the same preemptible learned
                    // profile filter as lower configured peers; original never does.
                    profile_cache && !profile_cache->is_learning() &&
                        core::is_priority_execution_mode(config.scheduling_mode) &&
                        has_priority_policy() && slot.is_low && !config.uses_original_kernels &&
                        kernel_mode != ExecutionClientKernelMode::kOriginal,
                    CudaSessionStreamConfig{
                        .uses_default_proxy =
                            (config.uses_nccl && !is_pointer_metadata && !is_process_memory) ||
                            uses_priority_space_share(),
                        .scheduling_priority = uses_priority_space_share()
                                                   ? std::optional<std::uint32_t>{slot.priority}
                                                   : std::nullopt},
                    std::move(member));
                if (dispatch.process_memory.allocate && !is_pointer_metadata) {
                    slot.process_memory = memory_owner(slot.peer, slot.policy_id);
                    if (!slot.process_memory) {
                        reject_initial(CUDA_ERROR_OUT_OF_MEMORY);
                        return;
                    }
                    const auto attached = slot.session->attach_process_memory(slot.process_memory);
                    if (attached != CUDA_SUCCESS) {
                        reject_initial(attached);
                        return;
                    }
                }
            }
            if (slot.session) slot.session_leases.bind(slot.session);
            if (slot.host_owner) slot.host_owner_leases.bind(slot.host_owner);
            if (slot.session &&
                ((config.should_audit_native && audit_log) || config.issue_depth_metrics)) {
                try {
                    slot.retirement_audit = std::make_shared<NativeRetirementAudit>(
                        NativeRetirementAudit::Settings{
                            slot.connection, config.session_limits.submissions.max_commands},
                        audit_log, config.issue_depth_metrics);
                } catch (...) {
                    if (config.should_audit_native && audit_log) has_semantic_loss = true;
                    if (config.issue_depth_metrics)
                        config.issue_depth_metrics->invalidate(
                            IssueDepthMetrics::Failure::kCapacity);
                }
            }
            if (slot.session && !slot.is_metadata_owner() &&
                (config.should_audit_native || config.should_audit_priority)) {
                try {
                    const auto* requested =
                        kernel_mode == ExecutionClientKernelMode::kConfigured ? "configured"
                        : kernel_mode == ExecutionClientKernelMode::kOriginal ? "original"
                                                                              : "auto";
                    if (audit_log)
                        (void)audit_log->info(
                            "Execution client selection audit: session={}, pid={}, start={}, "
                            "priority={}, requested={}, effective={}, retained_epochs={}",
                            slot.connection, slot.peer.pid, slot.peer.start_time_ticks,
                            slot.priority, requested,
                            uses_original || !config.persistent.workers ? "original" : "configured",
                            slot.is_low);
                } catch (...) {
                    // Selection evidence cannot change execution or physical authority.
                }
            }
            slot.requests.push_back(retain_request());
            slots.push_back(std::move(slot));
            lifecycle(
                HostStageMetrics::Lifecycle::kReceive, slots.back(),
                slots.back().requests.back().message.header.sequence, 0,
                static_cast<std::uint32_t>(slots.back().requests.back().message.header.operation),
                slots.back().requests.back().frame_received_ns,
                slots.back().requests.back().owner_received_ns,
                slots.back().requests.back().first_read_ns);
            audit_peer(slots.back(), 1);
            bind_sgemm_audit(slots.back());
            has_client = true;
            return;
        }
        auto& slot = *found;
        if (slot.is_retiring || !slot.is_initialized ||
            (slot.is_closing && !is_host_function_observation(header.operation)) ||
            slot.requests.size() == config.transport.max_pending_requests ||
            (header.operation == ExecutionOpcode::kClose && !slot.requests.empty()) ||
            !slot.has_request_space || request.peer != slot.peer ||
            header.operation == ExecutionOpcode::kHello || header.session != slot.connection ||
            header.incarnation != incarnation || header.sequence != slot.next_request) {
            drop(slot);
            return;
        }
        if (slot.next_request == std::numeric_limits<std::uint64_t>::max())
            slot.has_request_space = false;
        else
            ++slot.next_request;
        // A pending close can wait for an already accepted CPU callback. Its
        // exact query/finish receipts may still drain that boundary, but they
        // never reopen admission or allow a new callback/native operation.
        slot.is_closing = slot.is_closing || header.operation == ExecutionOpcode::kClose;
        slot.requests.push_back(retain_request());
        lifecycle(HostStageMetrics::Lifecycle::kReceive, slot,
                  slot.requests.back().message.header.sequence, 0,
                  static_cast<std::uint32_t>(slot.requests.back().message.header.operation),
                  slot.requests.back().frame_received_ns, slot.requests.back().owner_received_ns,
                  slot.requests.back().first_read_ns);
    }
    ExecutionWireReply finish_bootstrap(Slot& slot, ExecutionWireReply reply) {
        auto snapshot = slot.session->snapshot();
        if (reply.status != CUDA_SUCCESS && snapshot.has_context &&
            !snapshot.submissions.outstanding && slot.next_command == 1) {
            // No application command has ever been accepted in this factory.
            // Cleanup is not a planner grant to issue/replay GPU computation.
            const auto activated = activate(slot);
            const auto cleaned =
                activated.status == CUDA_SUCCESS ? slot.session->shutdown() : activated;
            if (cleaned.status != CUDA_SUCCESS)
                spdlog::warn(
                    "Execution bootstrap retains ownership: session={}, native_status={}, "
                    "has_original={}, original_status={}, cleanup_status={}",
                    slot.connection, reply.status, reply.original_status.has_value(),
                    reply.original_status.value_or(0), static_cast<int>(cleaned.status));
            snapshot = slot.session->snapshot();
        }
        const bool retained = snapshot.has_context || snapshot.device_bytes ||
                              snapshot.private_host_bytes || snapshot.image_bytes ||
                              snapshot.submissions.outstanding;
        const auto ownership = retained                  ? ExecutionBootstrapOwnership::kRetained
                               : snapshot.is_quarantined ? ExecutionBootstrapOwnership::kUnproven
                                                         : ExecutionBootstrapOwnership::kNone;
        describe_bootstrap(reply, ownership);
        return reply;
    }
    std::optional<ExecutionWireReply> execute_host(Slot& slot, Slot::Request& request,
                                                   bool can_issue) {
        const auto operation = request.message.header.operation;
        ExecutionPayloadReader input(request.message.payload);
        HostOwnerCommand command;
        command.sequence = request.message.header.sequence;
        if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            if (slot.bound_cleanup_parents) {
#ifdef GLIMMER_HAS_NCCL
                // Wrapped Close is sealed before its exact handler permit is
                // armed. Other actor entry is forbidden during that attempt.
                if (request.continuation && !slot.cleanup_retirement) return std::nullopt;
                const auto prepared = begin_helper_retirement(slot, &request);
                if (prepared != CUDA_SUCCESS) {
                    if (prepared != CUDA_ERROR_NOT_READY) policy_failed(slot, prepared);
                    return std::nullopt;
                }
#else
                return std::nullopt;
#endif
            }
            if (request.has_host_shutdown) {
                auto completed = slot.host_owner->take_shutdown_reply(command.sequence);
                if (!completed) return std::nullopt;
                request.has_host_shutdown = false;
                if (completed->disposition == CudaSessionDisposition::kDeferred) {
                    if (completed->original_status) {
                        policy_failed(slot, CUDA_ERROR_UNKNOWN);
                        return project(*completed);
                    }
                    // Only this actual consumed result permits a fresh attempt.
                    // An idle reservation alone cannot move HostPending back to
                    // Blocked or justify retrying a native cleanup operation.
                    return std::nullopt;
                }
                if (completed->status != CUDA_SUCCESS ||
                    completed->disposition != CudaSessionDisposition::kComplete) {
                    policy_failed(slot, completed->status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN
                                                                          : completed->status);
                    return project(*completed);
                }
                // Composite helper shutdown is not a public single SDK call.
                // Preserve the existing successful Close wire representation.
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                return reply;
            }
            slot.host_owner->close();
            if (slot.host_owner->snapshot().is_closed) {
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                return reply;
            }
            if (can_issue && memory_native_health() == CUDA_SUCCESS)
                request.has_host_shutdown = slot.host_owner->permit_shutdown(command.sequence);
            return std::nullopt;
        }
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{};
            if (!input.read_u32(role) ||
                role != (slot.is_pageable_owner  ? kExecutionPageableOwnerRole
                         : slot.is_staging_owner ? kExecutionStagingOwnerRole
                                                 : kExecutionHostOwnerRole) ||
                !input.read_u32(command.flags) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            command.operation = HostOwnerOperation::kInitialize;
        } else if (operation == ExecutionOpcode::kQueryStagingIdle) {
            if (!slot.is_staging_owner || request.continuation ||
                request.borrowed_descriptor() >= 0)
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
            if (!input.read_u64(command.allocation) || !command.allocation || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            if (slot.host_owner->snapshot().session.is_quarantined)
                return rejection(CUDA_ERROR_UNKNOWN);
            const auto idle = slot.host_owner->allocation_idle(command.allocation);
            if (!idle) return rejection(CUDA_ERROR_INVALID_VALUE);
            if (!*idle) return std::nullopt;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.id = command.allocation;
            return reply;
        } else if (operation == ExecutionOpcode::kImportHostMemory) {
            std::uint64_t bytes{}, mapping_bytes{};
            if (!input.read_u64(bytes) || !input.read_u64(mapping_bytes) ||
                !input.read_u32(command.flags) || !input.empty() ||
                bytes > std::numeric_limits<std::size_t>::max() ||
                mapping_bytes > std::numeric_limits<std::size_t>::max())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            command.operation = HostOwnerOperation::kImport;
            command.is_pageable = slot.is_pageable_owner;
            command.bytes = static_cast<std::size_t>(bytes);
            command.mapping_bytes = static_cast<std::size_t>(mapping_bytes);
        } else if (operation == ExecutionOpcode::kRuntimeMetadata && !slot.is_staging_owner) {
            const auto query = decode_runtime_metadata(input);
            if (!query || query->kind == CudaRuntimeMetadataKind::kMemoryInfo)
                return rejection(CUDA_ERROR_INVALID_VALUE);
            command.operation = HostOwnerOperation::kRuntimeMetadata;
            command.metadata = *query;
        } else if (operation == ExecutionOpcode::kReleaseHostMemory) {
            if (!input.read_u64(command.allocation) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            command.operation = HostOwnerOperation::kRelease;
        } else
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        if (!request.has_host_job) {
            if (!can_issue || memory_native_health() != CUDA_SUCCESS ||
                !slot.host_owner->submit(command, request.borrowed_descriptor()))
                return std::nullopt;
            request.has_host_job = true;
        }
        auto result = slot.host_owner->take_reply(command.sequence);
        if (!result) return std::nullopt;
        request.has_host_job = false;
        if (result->metadata) {
            return project(std::move(*result->metadata),
                           slot.host_owner->snapshot().session.is_quarantined);
        }
        if (result->result.disposition == CudaSessionDisposition::kDeferred) {
            // No SDK free has occurred. Release the reservation, then obtain a
            // fresh priority permission before trying after actual GPU leases.
            if (result->result.original_status) drop(slot);
            return std::nullopt;
        }
        const bool is_pageable_completion =
            slot.is_pageable_owner &&
            (command.operation == HostOwnerOperation::kImport ||
             command.operation == HostOwnerOperation::kRelease) &&
            result->result.disposition == CudaSessionDisposition::kComplete &&
            !result->result.original_status;
        if (request.continuation && result->result.status == CUDA_SUCCESS &&
            (result->result.original_status == CUDA_SUCCESS || is_pageable_completion)) {
            if (command.operation == HostOwnerOperation::kImport && !request.continuation->flags) {
                ++slot.bound_cleanup_parents;  // Credit was reserved before native import.
            } else if (command.operation == HostOwnerOperation::kRelease &&
                       request.continuation->flags == kExecutionSourceContinuationCleanup) {
                if (!slot.bound_cleanup_parents)
                    policy_failed(slot, CUDA_ERROR_UNKNOWN);
                else
                    --slot.bound_cleanup_parents;
            }
        }
        auto reply = project(result->result);
        if (command.operation == HostOwnerOperation::kImport) {
            ExecutionPayloadWriter ownership;
            if (!ownership.write_u32(slot.host_owner->snapshot().session.is_quarantined ? 1U : 0U))
                throw std::bad_alloc();
            reply.data = ownership.take();
        }
        if (slot.is_bootstrap_pending) {
            slot.is_bootstrap_pending = false;
            describe_bootstrap(reply, result->ownership);
        }
        return reply;
    }
    static CudaProcessMemoryRequest process_request(const ExecutionProcessMemoryRequest& wire) {
        return {static_cast<CudaProcessMemoryOperation>(wire.operation),
                wire.device,
                wire.has_output,
                wire.pool,
                wire.bytes,
                wire.stream,
                wire.output_seed};
    }
    static ExecutionWireReply project_process_memory(const CudaProcessMemoryReply& observed) {
        auto reply = rejection(observed.status);
        reply.original_status = observed.original_status;
        reply.parameter_bytes = observed.is_quarantined ? 1 : 0;
        if (observed.original_status) {
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.id = observed.id;
            reply.address = observed.output;
            ExecutionPayloadWriter output(kExecutionProcessMemoryObservationBytes);
            if (!write_execution_process_memory_observation(output, {observed.output_written})) {
                reply.status = CUDA_ERROR_UNKNOWN;
                reply.parameter_bytes = 1;
            } else
                reply.data = output.take();
        }
        return reply;
    }
    std::optional<ExecutionWireReply> execute_process_memory(Slot& slot, Slot::Request& request) {
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{}, flags{};
            if (operation != ExecutionOpcode::kHello || !input.read_u32(role) ||
                role != kExecutionProcessMemoryRole || !input.read_u32(flags) || flags ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            slot.is_bootstrap_pending = false;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kMetadataRetained);
            return reply;
        }
        if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            // Closing one carrier is not process death or allocation release.
            return project(slot.session->shutdown());
        }
        if (operation != ExecutionOpcode::kProcessMemory)
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        ExecutionProcessMemoryRequest wire;
        if (request.borrowed_descriptor() >= 0 ||
            !read_execution_process_memory_request(input, wire) || !input.empty() ||
            wire.operation == ExecutionProcessMemoryOperation::kAllocateAsync ||
            wire.operation == ExecutionProcessMemoryOperation::kAllocateFromPoolAsync)
            return rejection(CUDA_ERROR_INVALID_VALUE);
        if (!slot.process_memory) return rejection(CUDA_ERROR_NOT_SUPPORTED);
        const auto detached = detach_selected_context();
        if (detached == CUDA_ERROR_NOT_READY) return std::nullopt;
        if (detached != CUDA_SUCCESS && detached != CUDA_ERROR_NOT_INITIALIZED)
            return rejection(detached);
        const auto observed = slot.process_memory->pool_control(process_request(wire));
        if (observed.status == CUDA_ERROR_NOT_READY && !observed.original_status)
            return std::nullopt;
        return project_process_memory(observed);
    }
    std::optional<ExecutionWireReply> execute_pointer_metadata(Slot& slot, Slot::Request& request) {
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{}, flags{};
            if (operation != ExecutionOpcode::kHello || !input.read_u32(role) ||
                role != kExecutionPointerMetadataRole || !input.read_u32(flags) || flags ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            slot.is_bootstrap_pending = false;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kMetadataRetained);
            return reply;
        }
        if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            return project(slot.session->shutdown());
        }
        if (operation != ExecutionOpcode::kPointerMetadata)
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        if (slot.policy_failure || slot.session->health_snapshot().is_quarantined) {
            auto reply = rejection(CUDA_ERROR_UNKNOWN);
            reply.parameter_bytes = 1;
            return reply;
        }
        ExecutionPointerQuery wire;
        if (!read_execution_pointer_query(input, wire) || !input.empty() || wire.offset > SIZE_MAX)
            return rejection(CUDA_ERROR_INVALID_VALUE);
        ExecutionPayloadWriter output(config.transport.max_payload_bytes);
        if (!output.reserve_additional_capacity(kExecutionPointerObservationBytes))
            return rejection(CUDA_ERROR_OUT_OF_MEMORY);
        const auto execution_owner = [&](std::uint64_t connection) -> Slot* {
            const auto found = std::ranges::find(slots, connection, &Slot::connection);
            if (found == slots.end() || found->peer != slot.peer || !found->is_initialized ||
                found->is_bootstrap_pending || found->is_retiring || found->is_closing ||
                found->policy_failure || found->host_owner || found->is_metadata_owner() ||
                !found->session)
                return nullptr;
            const auto health = found->session->health_snapshot();
            if (!health.has_context || health.is_quarantined) return nullptr;
            return &*found;
        };
        auto* querying = wire.query_session ? execution_owner(wire.query_session) : nullptr;
        if (wire.query_session && !querying) return rejection(CUDA_ERROR_INVALID_CONTEXT);
        CudaPointerQuery query;
        query.domain = static_cast<CudaPointerDomain>(wire.domain);
        query.selector = wire.selector;
        query.kind = static_cast<CudaPointerKind>(wire.kind);
        Slot* source{};
        std::optional<CudaHostMemoryView> host;
        if (wire.kind == ExecutionPointerKind::kDevice ||
            wire.kind == ExecutionPointerKind::kProcessDevice) {
            if (wire.kind == ExecutionPointerKind::kProcessDevice) {
                const auto found = std::ranges::find(slots, wire.source_session, &Slot::connection);
                if (found != slots.end() && found->peer == slot.peer && found->is_initialized &&
                    !found->is_bootstrap_pending && !found->is_retiring && !found->is_closing &&
                    !found->policy_failure && found->is_process_memory && found->process_memory &&
                    !found->process_memory->owner()->snapshot().is_quarantined)
                    source = &*found;
            } else
                source = execution_owner(wire.source_session);
            if (!source) return rejection(CUDA_ERROR_INVALID_VALUE);
            query.range = {wire.allocation, static_cast<std::size_t>(wire.offset)};
            // CPU-only positive containment. The backend acquires native source
            // exclusion before any callback; this view alone is not a lease.
            if (wire.kind == ExecutionPointerKind::kDevice &&
                !source->session->device_memory(query.range, 1))
                return rejection(CUDA_ERROR_INVALID_VALUE);
        } else if (wire.kind == ExecutionPointerKind::kHost) {
            const auto found = std::ranges::find(slots, wire.source_session, &Slot::connection);
            if (found == slots.end() || found->peer != slot.peer || !found->is_initialized ||
                found->is_bootstrap_pending || found->is_retiring || found->is_closing ||
                found->policy_failure || !found->host_owner || found->is_staging_owner)
                return rejection(CUDA_ERROR_INVALID_VALUE);
            host = found->host_owner->borrow(
                {wire.allocation, static_cast<std::size_t>(wire.offset)}, 1);
            if (!host) return rejection(CUDA_ERROR_INVALID_VALUE);
        }
        const auto native = slot.session->pointer_metadata(
            query, querying ? querying->session.get() : nullptr,
            source ? source->session.get() : nullptr, std::move(host));
        // Native selection is independent of requested getter provenance: a
        // later preparation rejection may still leave this exact query owner
        // selected. Never infer it from status or matching context addresses.
        switch (native.context_selection) {
            case CudaPointerContextSelection::kUnchanged:
                break;
            case CudaPointerContextSelection::kQuery:
                selected_session =
                    querying ? querying->session : std::weak_ptr<CudaExecutionSession>{};
                break;
            case CudaPointerContextSelection::kUnproven:
                selected_session.reset();  // Retain unknown TLS; do not detach it.
                break;
        }
        if (native.disposition == CudaSessionDisposition::kDeferred) return std::nullopt;
        auto reply = rejection(native.status);
        reply.original_status = native.original_status;
        const bool is_quarantined =
            slot.session->health_snapshot().is_quarantined ||
            (querying && querying->session->health_snapshot().is_quarantined) ||
            (source && source->session->health_snapshot().is_quarantined);
        reply.parameter_bytes = is_quarantined ? 1 : 0;
        if (native.disposition != CudaSessionDisposition::kComplete ||
            native.status != CUDA_SUCCESS || is_quarantined)
            return reply;
        if (native.original_status && native.observation) {
            const auto& observed = *native.observation;
            const ExecutionPointerObservation value{
                observed.effects, observed.type, observed.device,
                static_cast<ExecutionPointerAlias>(observed.device_alias),
                static_cast<ExecutionPointerAlias>(observed.host_alias)};
            if (write_execution_pointer_observation(output, value)) {
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.status = *native.original_status;
                reply.data = output.take();
                return reply;
            }
        }
        policy_failed(slot, CUDA_ERROR_UNKNOWN);
        reply.status = CUDA_ERROR_UNKNOWN;
        reply.parameter_bytes = 1;
        return reply;
    }
    std::optional<ExecutionWireReply> execute_lt_metadata(Slot& slot, Slot::Request& request) {
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{}, flags{};
            if (!input.read_u32(role) || role != kExecutionLtMetadataRole ||
                !input.read_u32(flags) || flags || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            slot.is_bootstrap_pending = false;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kMetadataRetained);
            return reply;
        }
        CudaLtResult result;
        std::uint64_t id{};
        if (operation == ExecutionOpcode::kCreateLtMetadata) {
            CudaLtCreate specification;
            std::uint32_t kind{}, type{}, compute{}, scale{};
            std::uint64_t leading{};
            if (!input.read_u32(kind) || kind < 1 || kind > 3 || !input.read_u32(type) ||
                type > 29 || !input.read_u32(compute) || compute < 64 || compute > 77 ||
                !input.read_u32(scale) || scale > 29 || !input.read_u64(specification.rows) ||
                !input.read_u64(specification.columns) || !input.read_u64(leading) ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            specification.kind = static_cast<CudaLtObjectKind>(kind);
            specification.type = static_cast<cudaDataType>(type);
            specification.compute = static_cast<cublasComputeType_t>(compute);
            specification.scale = static_cast<cudaDataType>(scale);
            specification.leading = std::bit_cast<std::int64_t>(leading);
            result = slot.lt_metadata->create(specification);
        } else if (operation == ExecutionOpcode::kDestroyLtMetadata) {
            if (!input.read_u64(id) || !input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.lt_metadata->destroy(id);
        } else if (operation == ExecutionOpcode::kGetLtAttribute ||
                   operation == ExecutionOpcode::kSetLtAttribute) {
            std::uint32_t attribute{};
            if (!input.read_u64(id) || !input.read_u32(attribute))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            if (operation == ExecutionOpcode::kGetLtAttribute) {
                std::uint64_t bytes{};
                if (!input.read_u64(bytes) || bytes > 4096 || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.lt_metadata->get_attribute({id, attribute},
                                                         static_cast<std::size_t>(bytes));
            } else {
                std::span<const std::byte> data;
                std::uint32_t has_reference{};
                std::optional<CudaLtDeviceReference> reference;
                if (!input.read_blob(data) || data.size() > 4096 ||
                    !input.read_u32(has_reference) || has_reference > 1)
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (has_reference) {
                    CudaLtDeviceReference value;
                    if (!input.read_u64(value.session) || !input.read_u64(value.allocation) ||
                        !input.read_u64(value.offset) ||
                        value.offset > std::numeric_limits<std::size_t>::max())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    const auto origin = std::ranges::find(slots, value.session, &Slot::connection);
                    if (origin == slots.end() || !origin->session || origin->peer != slot.peer ||
                        !origin->is_initialized || origin->is_retiring || origin->is_closing)
                        return rejection(CUDA_ERROR_INVALID_HANDLE);
                    const auto view = origin->session->device_memory(
                        {value.allocation, static_cast<std::size_t>(value.offset)}, 1);
                    if (!view) return rejection(CUDA_ERROR_INVALID_VALUE);
                    value.address = view->address;
                    reference = value;
                }
                if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.lt_metadata->set_attribute({id, attribute}, data, reference);
            }
        } else if (operation == ExecutionOpcode::kLtAlgorithmCapability) {
            std::span<const std::byte> algorithm, initial;
            std::uint32_t attribute{}, presence{};
            std::uint64_t bytes{}, written{};
            if (!input.read_blob(algorithm) || algorithm.size() != sizeof(cublasLtMatmulAlgo_t) ||
                !input.read_u32(attribute) || !input.read_u64(bytes) || bytes > 4096 ||
                !input.read_u32(presence) || presence > 3 || !input.read_blob(initial) ||
                initial.size() != ((presence & 1U) ? bytes : 0) || !input.read_u64(written) ||
                written > SIZE_MAX || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            cublasLtMatmulAlgo_t copied{};
            std::memcpy(&copied, algorithm.data(), algorithm.size());
            result = slot.lt_metadata->algorithm_capability(
                copied, attribute, static_cast<std::size_t>(bytes), presence, initial,
                static_cast<std::size_t>(written));
        } else if (operation == ExecutionOpcode::kLtVersion) {
            std::uint32_t runtime{};
            if (!input.read_u32(runtime) || runtime > 1 || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.lt_metadata->version(runtime != 0);
        } else if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.lt_metadata->shutdown();
        } else
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        if (result.disposition == CudaLtDisposition::kDeferred) return std::nullopt;
        auto reply = project(std::move(result));
        reply.parameter_bytes = slot.lt_metadata->snapshot().is_quarantined ? 1 : 0;
        return reply;
    }
#ifdef GLIMMER_HAS_CUDNN
    bool decode_backend_values(const ExecutionDnnBackendAttribute& wire, const Slot& metadata,
                               const Slot* execution, CudaDnnBackendAttributeValues& values) {
        const auto value_class = execution_dnn_backend_value_class(wire.header.type);
        if (value_class == ExecutionDnnBackendValueClass::kInvalid) return false;
        values.scalars = wire.values.scalars;
        convert_dnn_backend_scalar_order(wire.header.type, values.scalars);
        for (const auto& reference : wire.values.references) {
            const bool is_null = reference.kind == ExecutionDnnBackendReferenceKind::kNull;
            if (value_class == ExecutionDnnBackendValueClass::kDescriptor &&
                (is_null || (reference.kind == ExecutionDnnBackendReferenceKind::kDescriptor &&
                             reference.owner == metadata.connection))) {
                values.descriptors.push_back(is_null ? 0 : reference.id);
            } else if (value_class == ExecutionDnnBackendValueClass::kPointerSlot &&
                       (is_null ||
                        (reference.kind == ExecutionDnnBackendReferenceKind::kPointerSlot &&
                         reference.owner == metadata.connection))) {
                values.pointer_slots.push_back(is_null ? 0 : reference.id);
            } else if (value_class == ExecutionDnnBackendValueClass::kHandle && execution &&
                       (is_null || (reference.kind == ExecutionDnnBackendReferenceKind::kHandle &&
                                    reference.owner == execution->connection))) {
                values.handles.push_back(
                    is_null ? CudaDnnBackendHandleReference{}
                            : CudaDnnBackendHandleReference{reference.owner, reference.id});
            } else {
                return false;
            }
        }
        return true;
    }
    bool encode_backend_get(ExecutionPayloadWriter& output, CudaDnnBackendAttributeResult& result,
                            std::uint32_t type, const Slot& metadata) try {
        convert_dnn_backend_scalar_order(type, result.values.scalars);
        ExecutionDnnBackendGetResult returned{result.count_bytes,
                                              std::bit_cast<std::uint64_t>(result.count),
                                              {type, std::move(result.values.scalars), {}}};
        if (!result.values.descriptors.empty()) return false;
        for (const auto& handle : result.values.handles)
            returned.values.references.push_back(
                handle.id ? ExecutionDnnBackendReference{ExecutionDnnBackendReferenceKind::kHandle,
                                                         handle.owner, handle.id}
                          : ExecutionDnnBackendReference{});
        for (const auto slot : result.values.pointer_slots)
            returned.values.references.push_back(
                slot ? ExecutionDnnBackendReference{ExecutionDnnBackendReferenceKind::kPointerSlot,
                                                    metadata.connection, slot}
                     : ExecutionDnnBackendReference{});
        return write_execution_dnn_backend_get_result(output, returned);
    } catch (...) {
        return false;
    }
    std::optional<ExecutionWireReply> execute_dnn_backend_context(Slot& execution,
                                                                  ExecutionPayloadReader& input) {
        ExecutionDnnBackendContext envelope;
        if (!read_execution_dnn_backend_context(input, envelope) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, envelope.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != execution.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        ExecutionPayloadReader payload(envelope.payload);
        CudaDnnBackendContextRequest request;
        if (envelope.operation == ExecutionOpcode::kDnnBackendPrepare) {
            ExecutionDnnBackendPrepare wire;
            if (!read_execution_dnn_backend_prepare(payload, wire))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            request.operation = CudaDnnBackendContextOperation::kPrepare;
            request.attribute_operation =
                static_cast<CudaDnnBackendAttributeOperation>(wire.operation);
            const auto& header = wire.header;
            request.header = {header.descriptor, header.name, header.type, header.count,
                              header.presence};
        } else if (envelope.operation == ExecutionOpcode::kDnnBackendSet ||
                   envelope.operation == ExecutionOpcode::kDnnBackendGet) {
            ExecutionDnnBackendAttribute wire;
            if (!read_execution_dnn_backend_attribute(payload, wire))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            if (!decode_backend_values(wire, *owner, &execution, request.values))
                return rejection(CUDA_ERROR_INVALID_HANDLE);
            request.operation = envelope.operation == ExecutionOpcode::kDnnBackendSet
                                    ? CudaDnnBackendContextOperation::kSet
                                    : CudaDnnBackendContextOperation::kGet;
            const auto& header = wire.header;
            request.header = {header.descriptor, header.name, header.type, header.count,
                              header.presence};
            request.generation = wire.generation;
        } else {
            if (!payload.read_u64(request.id)) return rejection(CUDA_ERROR_INVALID_VALUE);
            request.operation = envelope.operation == ExecutionOpcode::kDnnBackendFinalize
                                    ? CudaDnnBackendContextOperation::kFinalize
                                : envelope.operation == ExecutionOpcode::kDnnBackendInitialize
                                    ? CudaDnnBackendContextOperation::kInitialize
                                    : CudaDnnBackendContextOperation::kDestroy;
        }
        if (!payload.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
        ExecutionPayloadWriter output;
        if (request.operation == CudaDnnBackendContextOperation::kGet &&
            !output.reserve_capacity(20 + kExecutionDnnBackendMaxValueBytes))
            return rejection(CUDA_ERROR_OUT_OF_MEMORY);
        const auto activated = activate(execution);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        auto result = execution.session->backend_context(execution.connection, envelope.handle,
                                                         *owner->dnn_metadata, request);
        if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = project(result.result);
        reply.parameter_bytes = rnn_fault_bits(execution, owner->connection);
        if (reply.disposition == ExecutionWireDisposition::kRejected &&
            reply.original_status == CUDNN_STATUS_SUCCESS && !reply.parameter_bytes)
            reply.parameter_bytes = 1U;
        if (result.result.status != CUDA_SUCCESS || reply.parameter_bytes ||
            (result.result.original_status &&
             result.result.original_status != CUDNN_STATUS_SUCCESS))
            return reply;
        if (request.operation == CudaDnnBackendContextOperation::kPrepare) {
            const auto& prepared = result.preparation;
            if (result.result.original_status ||
                !write_execution_dnn_backend_preparation(
                    output, {prepared.generation,
                             static_cast<ExecutionDnnBackendAttributeDirection>(prepared.direction),
                             prepared.input_elements, prepared.output_elements,
                             prepared.count_bytes, prepared.output_bytes}))
                return rejection(CUDA_ERROR_UNKNOWN);
            reply.data = output.take();
        } else if (request.operation == CudaDnnBackendContextOperation::kGet) {
            if (!encode_backend_get(output, result.attribute, request.header.type, *owner)) {
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_UNKNOWN;
                reply.parameter_bytes |= 1U;
            } else {
                reply.data = output.take();
            }
        }
        return reply;
    }
    std::optional<ExecutionWireReply> execute_dnn_backend_metadata(Slot& slot,
                                                                   ExecutionOpcode operation,
                                                                   ExecutionPayloadReader& input) {
        auto& registry = *slot.dnn_metadata;
        const auto finish =
            [&registry](const CudaDnnResult& result) -> std::optional<ExecutionWireReply> {
            if (result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
            auto reply = project(result);
            reply.parameter_bytes = registry.snapshot().is_quarantined ? 1 : 0;
            return reply;
        };
        if (operation == ExecutionOpcode::kDnnBackendCreate) {
            ExecutionDnnBackendCreate wire;
            if (!read_execution_dnn_backend_create(input, wire) || !input.empty() ||
                !wire.has_output)
                return rejection(CUDA_ERROR_INVALID_VALUE);
            return finish(registry.create_backend(wire.kind));
        }
        if (operation == ExecutionOpcode::kDnnBackendDestroy ||
            operation == ExecutionOpcode::kDnnBackendInitialize ||
            operation == ExecutionOpcode::kDnnBackendFinalize) {
            std::uint64_t id{};
            if (!input.read_u64(id) || !input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            if (operation == ExecutionOpcode::kDnnBackendDestroy) {
                // Check kind on the serialized owner thread, then release the
                // temporary view before the exclusive native destruction.
                if (!registry.borrow(id, CudaDnnObjectKind::kBackend))
                    return rejection(CUDA_ERROR_INVALID_HANDLE);
                return finish(registry.destroy(id));
            }
            return finish(operation == ExecutionOpcode::kDnnBackendInitialize
                              ? registry.initialize_backend(id)
                              : registry.finalize_backend(id));
        }
        if (operation == ExecutionOpcode::kDnnBackendPrepare) {
            ExecutionDnnBackendPrepare wire;
            if (!read_execution_dnn_backend_prepare(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            const auto& header = wire.header;
            const auto prepared = registry.prepare_backend_attribute(
                static_cast<CudaDnnBackendAttributeOperation>(wire.operation),
                {header.descriptor, header.name, header.type, header.count, header.presence});
            auto reply = finish(prepared.result);
            if (reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                reply->status == CUDA_SUCCESS && !reply->parameter_bytes &&
                !reply->original_status) {
                ExecutionPayloadWriter output;
                if (!write_execution_dnn_backend_preparation(
                        output,
                        {prepared.generation,
                         static_cast<ExecutionDnnBackendAttributeDirection>(prepared.direction),
                         prepared.input_elements, prepared.output_elements, prepared.count_bytes,
                         prepared.output_bytes}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply->data = output.take();
            }
            return reply;
        }
        ExecutionDnnBackendAttribute wire;
        if (!read_execution_dnn_backend_attribute(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto value_class = execution_dnn_backend_value_class(wire.header.type);
        if (value_class == ExecutionDnnBackendValueClass::kInvalid)
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        CudaDnnBackendAttributeValues values;
        if (!decode_backend_values(wire, slot, nullptr, values))
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        ExecutionPayloadWriter output;
        // Reserve before a mutating descriptor-output Get. Native results are
        // already bounded by the same materialized-value quota; publication
        // cannot allocate after the native destination change.
        if (operation == ExecutionOpcode::kDnnBackendGet &&
            !output.reserve_capacity(20 + kExecutionDnnBackendMaxValueBytes))
            return rejection(CUDA_ERROR_OUT_OF_MEMORY);
        const auto& header = wire.header;
        const CudaDnnBackendAttributeHeader native_header{
            header.descriptor, header.name, header.type, header.count, header.presence};
        auto result = operation == ExecutionOpcode::kDnnBackendSet
                          ? registry.set_backend_attribute(native_header, wire.generation, values)
                          : registry.get_backend_attribute(native_header, wire.generation, values);
        auto reply = finish(result.result);
        if (reply && operation == ExecutionOpcode::kDnnBackendGet &&
            reply->disposition == ExecutionWireDisposition::kComplete &&
            reply->status == CUDNN_STATUS_SUCCESS &&
            reply->original_status == CUDNN_STATUS_SUCCESS && !reply->parameter_bytes) {
            if (!encode_backend_get(output, result, header.type, slot)) {
                // An inconsistent post-issue receipt retains native provenance
                // but cannot authorize caller publication or a retry.
                reply->disposition = ExecutionWireDisposition::kRejected;
                reply->status = CUDA_ERROR_UNKNOWN;
            } else {
                reply->data = output.take();
            }
        }
        return reply;
    }
#include "dnn_legacy_service.inc"
    std::optional<ExecutionWireReply> execute_dnn_metadata(Slot& slot, Slot::Request& request) {
        static_assert(CUDNN_DIM_MAX == kExecutionDnnMaxDimensions);
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        if (slot.is_bootstrap_pending) {
            std::uint32_t role{}, flags{};
            if (!input.read_u32(role) || role != kExecutionDnnMetadataRole ||
                !input.read_u32(flags) || flags || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            slot.is_bootstrap_pending = false;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            describe_bootstrap(reply, ExecutionBootstrapOwnership::kMetadataRetained);
            return reply;
        }
        if (is_dnn_backend_metadata(operation))
            return execute_dnn_backend_metadata(slot, operation, input);
        if (operation == ExecutionOpcode::kDnnLegacyMetadata)
            return execute_dnn_legacy_metadata(slot, input);
        CudaDnnResult result;
        std::uint64_t id{};
        if (operation != ExecutionOpcode::kCreateDnnMetadata &&
            operation != ExecutionOpcode::kClose && operation != ExecutionOpcode::kDnnLibraryInfo &&
            !input.read_u64(id))
            return rejection(CUDA_ERROR_INVALID_VALUE);
        if (operation == ExecutionOpcode::kDnnLibraryInfo) {
            ExecutionDnnLibraryInfo query;
            if (!read_execution_dnn_library_info(input, query) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result =
                slot.dnn_metadata->library_info(static_cast<CudaDnnLibraryInfoKind>(query.kind),
                                                query.property, query.has_output != 0);
        } else if (operation == ExecutionOpcode::kCreateDnnMetadata) {
            std::uint32_t kind{};
            if (!input.read_u32(kind) || kind < 1 || (kind > 14 && kind != 16) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->create(static_cast<CudaDnnObjectKind>(kind));
        } else if (operation == ExecutionOpcode::kDestroyDnnMetadata) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->destroy(id);
        } else if (operation == ExecutionOpcode::kSetDnnTensor) {
            std::uint32_t setter{};
            ExecutionDnnTensor tensor;
            if (!input.read_u32(setter) || setter < 1 || setter > 4 ||
                !read_execution_dnn_tensor(input, tensor) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            const CudaDnnTensor native{tensor.type, tensor.format, tensor.dimensions, tensor.sizes,
                                       tensor.strides};
            constexpr std::array setters{CudaDnnTensorSetter::kFour,
                                         CudaDnnTensorSetter::kFourStrided,
                                         CudaDnnTensorSetter::kNd, CudaDnnTensorSetter::kNdPacked};
            result = slot.dnn_metadata->set_tensor(id, setters[setter - 1], native);
        } else if (operation == ExecutionOpcode::kGetDnnTensor) {
            std::uint32_t requested{}, is_four{};
            if (!input.read_u32(requested) || requested > kExecutionDnnMaxDimensions ||
                !input.read_u32(is_four) || is_four > 1 || (is_four && requested != 4) ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_tensor(id, static_cast<int>(requested), is_four != 0);
        } else if (operation == ExecutionOpcode::kDnnTensorSize) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->tensor_size(id);
        } else if (operation == ExecutionOpcode::kSetDnnActivation) {
            ExecutionDnnActivation activation;
            if (!read_execution_dnn_activation(input, activation) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_activation(
                id, {activation.mode, activation.nan,
                     std::bit_cast<double>(activation.coefficient_bits)});
        } else if (operation == ExecutionOpcode::kGetDnnActivation) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_activation(id);
        } else if (operation == ExecutionOpcode::kSetDnnFilter) {
            std::uint32_t is_four{};
            ExecutionDnnFilter filter;
            if (!input.read_u32(is_four) || is_four > 1 ||
                !read_execution_dnn_filter(input, filter) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_filter(
                id, {filter.type, filter.format, filter.dimensions, filter.sizes}, is_four != 0);
        } else if (operation == ExecutionOpcode::kGetDnnFilter) {
            std::uint32_t requested{}, is_four{};
            if (!input.read_u32(requested) || !input.read_u32(is_four) || is_four > 1 ||
                (is_four && requested != 4) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_filter(id, std::bit_cast<std::int32_t>(requested),
                                                   is_four != 0);
        } else if (operation == ExecutionOpcode::kDnnFilterSize) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->filter_size(id);
        } else if (operation == ExecutionOpcode::kSetDnnConvolution) {
            std::uint32_t is_two{};
            ExecutionDnnConvolutionState convolution;
            if (!input.read_u32(is_two) || is_two > 1 ||
                !read_execution_dnn_convolution(input, convolution) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_convolution(
                id,
                {convolution.mode, convolution.type, convolution.dimensions, convolution.padding,
                 convolution.strides, convolution.dilation},
                is_two != 0);
        } else if (operation == ExecutionOpcode::kGetDnnConvolution) {
            std::uint32_t requested{}, is_two{};
            if (!input.read_u32(requested) || !input.read_u32(is_two) || is_two > 1 ||
                (is_two && requested != 2) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_convolution(id, std::bit_cast<std::int32_t>(requested),
                                                        is_two != 0);
        } else if (operation == ExecutionOpcode::kSetDnnConvolutionAttribute ||
                   operation == ExecutionOpcode::kGetDnnConvolutionAttribute) {
            std::uint32_t attribute{}, value{};
            if (!input.read_u32(attribute) || attribute < 1 || attribute > 3 ||
                (operation == ExecutionOpcode::kSetDnnConvolutionAttribute &&
                 !input.read_u32(value)) ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            const auto kind = static_cast<CudaDnnConvolutionAttribute>(attribute);
            result = operation == ExecutionOpcode::kSetDnnConvolutionAttribute
                         ? slot.dnn_metadata->set_convolution_attribute(id, kind, value)
                         : slot.dnn_metadata->get_convolution_attribute(id, kind);
        } else if (operation == ExecutionOpcode::kDnnConvolutionOutput) {
            std::uint64_t x{}, w{};
            std::uint32_t requested{}, is_two{};
            if (!input.read_u64(x) || !input.read_u64(w) || !input.read_u32(requested) ||
                !input.read_u32(is_two) || is_two > 1 || (is_two && requested != 4) ||
                !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->convolution_output(
                id, x, w, std::bit_cast<std::int32_t>(requested), is_two != 0);
        } else if (operation == ExecutionOpcode::kSetDnnPooling) {
            ExecutionDnnPooling pooling;
            if (!read_execution_dnn_pooling(input, pooling) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_pooling(
                id, {pooling.mode, pooling.nan, pooling.dimensions, pooling.window, pooling.padding,
                     pooling.strides});
        } else if (operation == ExecutionOpcode::kGetDnnPooling) {
            std::uint32_t requested{};
            if (!input.read_u32(requested) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_pooling(id, std::bit_cast<std::int32_t>(requested));
        } else if (operation == ExecutionOpcode::kDnnPoolingOutput) {
            std::uint64_t tensor{};
            std::uint32_t dimensions{};
            if (!input.read_u64(tensor) || !input.read_u32(dimensions) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->pooling_output(id, tensor,
                                                       std::bit_cast<std::int32_t>(dimensions));
        } else if (operation == ExecutionOpcode::kSetDnnReduction) {
            ExecutionDnnReduction reduction;
            if (!read_execution_dnn_reduction(input, reduction) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_reduction(
                id, {reduction.operation, reduction.type, reduction.nan, reduction.indices,
                     reduction.indices_type});
        } else if (operation == ExecutionOpcode::kSetDnnLrn) {
            ExecutionDnnLrn lrn;
            if (!read_execution_dnn_lrn(input, lrn) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_lrn(
                id, {lrn.n, std::bit_cast<double>(lrn.alpha_bits),
                     std::bit_cast<double>(lrn.beta_bits), std::bit_cast<double>(lrn.k_bits)});
        } else if (operation == ExecutionOpcode::kSetDnnOpTensor) {
            ExecutionDnnOpTensor state;
            if (!read_execution_dnn_op_tensor(input, state) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_op_tensor(id, {state.operation, state.type, state.nan});
        } else if (operation == ExecutionOpcode::kGetDnnOpTensor) {
            std::uint32_t mask{};
            if (!input.read_u32(mask) || mask > 7 || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_op_tensor(id, mask);
        } else if (operation == ExecutionOpcode::kSetDnnTensorTransform) {
            ExecutionDnnTensorTransform state;
            if (!read_execution_dnn_tensor_transform(input, state) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_tensor_transform(
                id, {state.dimensions, state.format, state.direction, state.presence, state.before,
                     state.after, state.fold});
        } else if (operation == ExecutionOpcode::kSetDnnRnn) {
            ExecutionDnnRnnState wire;
            if (!read_execution_dnn_rnn(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            CudaDnnRnnState native;
            native.algorithm = wire.algorithm;
            native.cell_mode = wire.cell_mode;
            native.bias_mode = wire.bias_mode;
            native.direction = wire.direction;
            native.input_mode = wire.input_mode;
            native.type = wire.type;
            native.math_precision = wire.math_precision;
            native.math_type = wire.math_type;
            native.input_size = wire.input_size;
            native.hidden_size = wire.hidden_size;
            native.projection_size = wire.projection_size;
            native.layers = wire.layers;
            native.dropout = wire.dropout;
            native.flags = wire.flags;
            result = slot.dnn_metadata->set_rnn(id, native);
        } else if (operation == ExecutionOpcode::kSetDnnRnnClip) {
            ExecutionDnnRnnClip wire;
            if (!read_execution_dnn_rnn_clip(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->set_rnn_clip(
                id, {wire.mode, wire.nan, std::bit_cast<double>(wire.lower_bits),
                     std::bit_cast<double>(wire.upper_bits)});
        } else if (operation == ExecutionOpcode::kSetDnnAttn) {
            ExecutionDnnAttnState wire;
            if (!read_execution_dnn_attn(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            CudaDnnAttnState native;
            native.mode = wire.mode;
            native.heads = wire.heads;
            native.scale = std::bit_cast<double>(wire.scale_bits);
            native.type = wire.type;
            native.precision = wire.precision;
            native.math_type = wire.math_type;
            native.dropouts = wire.dropouts;
            native.query_size = wire.query_size;
            native.key_size = wire.key_size;
            native.value_size = wire.value_size;
            native.query_projection = wire.query_projection;
            native.key_projection = wire.key_projection;
            native.value_projection = wire.value_projection;
            native.output_projection = wire.output_projection;
            native.max_query_sequence = wire.max_query_sequence;
            native.max_key_sequence = wire.max_key_sequence;
            native.max_batch = wire.max_batch;
            native.max_beam = wire.max_beam;
            result = slot.dnn_metadata->set_attn(id, native);
        } else if (operation == ExecutionOpcode::kSetDnnRnnData) {
            ExecutionDnnRnnDataState wire;
            if (!read_execution_dnn_rnn_data(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            CudaDnnRnnDataState native;
            native.type = wire.type;
            native.layout = wire.layout;
            native.presence = wire.presence;
            native.padding_bytes = wire.padding_bytes;
            native.max_sequence_length = wire.max_sequence_length;
            native.batch_size = wire.batch_size;
            native.vector_size = wire.vector_size;
            native.lengths = std::move(wire.lengths);
            native.padding = wire.padding;
            result = slot.dnn_metadata->set_rnn_data(id, native);
        } else if (operation == ExecutionOpcode::kSetDnnSeqData) {
            ExecutionDnnSeqDataState wire;
            if (!read_execution_dnn_seq_data(input, wire) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            CudaDnnSeqDataState native;
            native.type = wire.type;
            native.presence = wire.presence;
            native.padding_bytes = wire.padding_bytes;
            native.dimensions = wire.dimensions;
            native.sizes = wire.sizes;
            native.axes = wire.axes;
            native.sequence_count = wire.sequence_count;
            native.lengths = std::move(wire.lengths);
            native.padding = wire.padding;
            result = slot.dnn_metadata->set_seq_data(id, native);
        } else if (operation == ExecutionOpcode::kGetDnnSeqData) {
            ExecutionDnnSeqDataQuery query;
            if (!read_execution_dnn_seq_data_query(input, query) || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->get_seq_data(
                id, {query.dimensions_requested, query.lengths_requested, query.outputs});
        } else if (operation == ExecutionOpcode::kDnnInitTransformDest) {
            std::uint64_t source{}, destination{};
            std::uint32_t has_size{};
            if (!input.read_u64(source) || !input.read_u64(destination) ||
                !input.read_u32(has_size) || has_size > 1 || !input.empty())
                return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->init_transform_dest(id, source, destination, has_size != 0);
        } else if (operation == ExecutionOpcode::kClose) {
            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
            result = slot.dnn_metadata->shutdown();
        } else
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
        if (result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = project(result);
        reply.parameter_bytes = slot.dnn_metadata->snapshot().is_quarantined ? 1 : 0;
        if (result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
            result.original_status == CUDNN_STATUS_SUCCESS) {
            ExecutionPayloadWriter output;
            if (operation == ExecutionOpcode::kGetDnnTensor) {
                const auto& native = result.tensor;
                if (!write_execution_dnn_tensor(
                        output, {native.type, native.format, native.dimensions, native.sizes,
                                 native.strides}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnActivation) {
                const auto& native = result.activation;
                if (!write_execution_dnn_activation(
                        output, {native.mode, native.nan,
                                 std::bit_cast<std::uint64_t>(native.coefficient)}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnFilter) {
                const auto& native = result.filter;
                if (!write_execution_dnn_filter(
                        output, {native.type, native.format, native.dimensions, native.sizes}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnConvolution) {
                const auto& native = result.convolution;
                if (!write_execution_dnn_convolution(
                        output, {native.mode, native.type, native.dimensions, native.padding,
                                 native.strides, native.dilation}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kDnnConvolutionOutput) {
                if (!output.write_u32(static_cast<std::uint32_t>(result.tensor.dimensions)))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                for (auto size : result.tensor.sizes)
                    if (!output.write_u32(std::bit_cast<std::uint32_t>(size)))
                        return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnPooling) {
                const auto& native = result.pooling;
                if (!write_execution_dnn_pooling(
                        output, {native.mode, native.nan, native.dimensions, native.window,
                                 native.padding, native.strides}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kDnnPoolingOutput) {
                const auto& native = result.tensor;
                if (!write_execution_dnn_tensor(
                        output, {native.type, native.format, native.dimensions, native.sizes,
                                 native.strides}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnOpTensor) {
                const auto& native = result.op_tensor;
                if (!write_execution_dnn_op_tensor(output,
                                                   {native.operation, native.type, native.nan}))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            } else if (operation == ExecutionOpcode::kGetDnnSeqData) {
                auto& native = result.seq_data;
                ExecutionDnnSeqDataState wire;
                wire.type = native.type;
                wire.presence = native.presence;
                wire.padding_bytes = native.padding_bytes;
                wire.dimensions = native.dimensions;
                wire.sizes = native.sizes;
                wire.axes = native.axes;
                wire.sequence_count = native.sequence_count;
                wire.lengths = std::move(native.lengths);
                wire.padding = native.padding;
                if (!write_execution_dnn_seq_data(output, wire))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                reply.data = output.take();
            }
        }
        return reply;
    }
    bool decode_dnn_compute(Slot& slot, ExecutionPayloadReader& input,
                            CudaSessionDnnCompute& problem) {
        ExecutionDnnCompute wire;
        if (!read_execution_dnn_compute(input, wire) || !input.empty()) return false;
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        problem.kind = wire.kind == ExecutionDnnComputeKind::kActivationForward
                           ? CudaDnnComputeKind::kActivationForward
                           : CudaDnnComputeKind::kAddTensor;
        problem.handle = wire.handle;
        if (problem.kind == CudaDnnComputeKind::kActivationForward) {
            const auto descriptor =
                owner->dnn_metadata->borrow(wire.activation, CudaDnnObjectKind::kActivation);
            if (!descriptor) return false;
            problem.activation = *descriptor;
        }
        for (std::size_t index = 0; index < problem.tensors.size(); ++index) {
            const auto descriptor =
                owner->dnn_metadata->borrow(wire.tensors[index], CudaDnnObjectKind::kTensor);
            if (!descriptor || wire.buffers[index].offset > SIZE_MAX) return false;
            problem.tensors[index] = *descriptor;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        problem.scalar_bytes = wire.scalar_bytes;
        problem.alpha = wire.alpha;
        problem.beta = wire.beta;
        return true;
    }
    bool decode_dnn_tensor_forward(Slot& slot, ExecutionPayloadReader& input,
                                   CudaSessionDnnCompute& problem) {
        ExecutionDnnTensorForward wire;
        if (!read_execution_dnn_tensor_forward(input, wire) || !input.empty()) return false;
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        switch (wire.kind) {
            case ExecutionDnnTensorForwardKind::kPoolingForward:
                problem.kind = CudaDnnComputeKind::kPoolingForward;
                break;
            case ExecutionDnnTensorForwardKind::kSoftmaxForward:
                problem.kind = CudaDnnComputeKind::kSoftmaxForward;
                break;
            case ExecutionDnnTensorForwardKind::kLrnForward:
                problem.kind = CudaDnnComputeKind::kLrnForward;
                break;
            case ExecutionDnnTensorForwardKind::kTransformTensor:
                problem.kind = CudaDnnComputeKind::kTransformTensor;
                break;
            default:
                return false;
        }
        problem.handle = wire.handle;
        if (problem.kind == CudaDnnComputeKind::kPoolingForward ||
            problem.kind == CudaDnnComputeKind::kLrnForward) {
            const bool is_pooling = problem.kind == CudaDnnComputeKind::kPoolingForward;
            const auto descriptor = owner->dnn_metadata->borrow(
                wire.descriptor,
                is_pooling ? CudaDnnObjectKind::kPooling : CudaDnnObjectKind::kLrn);
            if (!descriptor) return false;
            (is_pooling ? problem.pooling : problem.lrn) = *descriptor;
        }
        for (std::size_t index = 0; index < problem.tensors.size(); ++index) {
            const auto descriptor =
                owner->dnn_metadata->borrow(wire.tensors[index], CudaDnnObjectKind::kTensor);
            if (!descriptor || wire.buffers[index].offset > SIZE_MAX) return false;
            problem.tensors[index] = *descriptor;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        problem.algorithm = wire.algorithm;
        problem.mode = wire.mode;
        problem.scalar_bytes = wire.scalar_bytes;
        problem.alpha = wire.alpha;
        problem.beta = wire.beta;
        return true;
    }
    bool borrow_dnn_reduction(Slot& slot, std::uint64_t metadata,
                              const std::array<std::uint64_t, 3>& ids,
                              CudaSessionDnnReduction& problem) {
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kReduction,
                                   CudaDnnObjectKind::kTensor};
        for (std::size_t index = 0; index != ids.size(); ++index) {
            const auto descriptor = owner->dnn_metadata->borrow(ids[index], kinds[index]);
            if (!descriptor) return false;
            problem.descriptors[index] = *descriptor;
        }
        return true;
    }
    bool decode_dnn_reduction(Slot& slot, ExecutionPayloadReader& input,
                              CudaSessionDnnReduction& problem) {
        ExecutionDnnReductionCompute wire;
        if (!read_execution_dnn_reduction_compute(input, wire) || !input.empty() ||
            wire.indices_bytes > SIZE_MAX || wire.workspace_bytes > SIZE_MAX ||
            !borrow_dnn_reduction(slot, wire.metadata, wire.descriptors, problem))
            return false;
        problem.handle = wire.handle;
        problem.indices_bytes = static_cast<std::size_t>(wire.indices_bytes);
        problem.workspace_bytes = static_cast<std::size_t>(wire.workspace_bytes);
        for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return false;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        problem.scalar_bytes = wire.scalar_bytes;
        problem.alpha = wire.alpha;
        problem.beta = wire.beta;
        return true;
    }
    ExecutionWireReply query_dnn_reduction(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnReductionQuery wire;
        if (!read_execution_dnn_reduction_query(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        CudaSessionDnnReduction problem;
        problem.handle = wire.handle;
        if (!borrow_dnn_reduction(slot, wire.metadata, wire.descriptors, problem))
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        auto reply = project(slot.session->query_dnn_reduction(
            problem, static_cast<CudaDnnReductionQueryKind>(wire.kind)));
        reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
        return reply;
    }
    bool decode_dnn_reorder(Slot& slot, ExecutionPayloadReader& input,
                            CudaSessionDnnReorder& problem) {
        ExecutionDnnReorderFilterAndBias wire;
        if (!read_execution_dnn_reorder_filter_and_bias(input, wire) || !input.empty())
            return false;
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        const auto filter = owner->dnn_metadata->borrow(wire.filter, CudaDnnObjectKind::kFilter);
        if (!filter) return false;
        problem.filter = *filter;
        problem.handle = wire.handle;
        problem.reorder = wire.reorder;
        problem.reorder_bias = wire.reorder_bias;
        for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return false;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        return valid_cuda_dnn_reorder_descriptors(problem);
    }
    bool borrow_dnn_batch_norm(Slot& slot, const ExecutionDnnBatchNormQuery& wire,
                               CudaSessionDnnBatchNorm& problem) {
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        problem.kind = static_cast<CudaDnnBatchNormKind>(wire.kind);
        problem.mode = wire.mode;
        problem.operation = wire.operation;
        problem.handle = wire.handle;
        for (std::size_t index = 0; index != wire.descriptors.size(); ++index) {
            if (!wire.descriptors[index]) continue;
            const auto descriptor = owner->dnn_metadata->borrow(
                wire.descriptors[index],
                index == 6 ? CudaDnnObjectKind::kActivation : CudaDnnObjectKind::kTensor);
            if (!descriptor) return false;
            problem.descriptors[index] = *descriptor;
        }
        return valid_cuda_dnn_batch_norm_descriptors(problem);
    }
    bool decode_dnn_batch_norm(Slot& slot, ExecutionPayloadReader& input,
                               CudaSessionDnnBatchNorm& problem) {
        ExecutionDnnBatchNormCompute wire;
        if (!read_execution_dnn_batch_norm_compute(input, wire) || !input.empty() ||
            wire.workspace_bytes > SIZE_MAX || wire.reserve_bytes > SIZE_MAX ||
            !borrow_dnn_batch_norm(slot, wire.query, problem))
            return false;
        problem.workspace_bytes = static_cast<std::size_t>(wire.workspace_bytes);
        problem.reserve_bytes = static_cast<std::size_t>(wire.reserve_bytes);
        problem.epsilon = std::bit_cast<double>(wire.epsilon_bits);
        problem.average_factor = std::bit_cast<double>(wire.average_factor_bits);
        problem.scalar_bytes = wire.scalar_bytes;
        problem.coefficients = wire.coefficients;
        for (std::size_t index = 0; index != wire.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return false;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        return true;
    }
    ExecutionWireReply query_dnn_batch_norm(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnBatchNormQuery wire;
        if (!read_execution_dnn_batch_norm_query(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        CudaSessionDnnBatchNorm problem;
        if (!borrow_dnn_batch_norm(slot, wire, problem))
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        auto reply = project(slot.session->query_dnn_batch_norm(problem));
        reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
        return reply;
    }
    CudaDnnResult decode_dnn_dropout(Slot& slot, ExecutionPayloadReader& input,
                                     CudaSessionDnnDropout& problem, std::uint64_t& metadata) {
        ExecutionDnnDropout wire;
        if (!read_execution_dnn_dropout(input, wire) || !input.empty() ||
            wire.states.offset > SIZE_MAX || wire.state_bytes > SIZE_MAX)
            return {.status = CUDA_ERROR_INVALID_VALUE};
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return {.status = CUDA_ERROR_INVALID_HANDLE};
        metadata = wire.metadata;
        auto reservation = owner->dnn_metadata->begin_dropout_mutation(wire.descriptor);
        if (reservation.result.disposition == CudaDnnDisposition::kDeferred ||
            reservation.result.status != CUDA_SUCCESS)
            return reservation.result;
        if (!reservation.mutation) return {.status = CUDA_ERROR_UNKNOWN};
        problem.kind = static_cast<CudaDnnDropoutKind>(wire.kind);
        problem.handle = wire.handle;
        problem.mutation = std::move(reservation.mutation);
        problem.states = {wire.states.allocation, static_cast<std::size_t>(wire.states.offset)};
        problem.state_bytes = static_cast<std::size_t>(wire.state_bytes);
        problem.probability_bits = wire.probability_bits;
        problem.seed = wire.seed;
        return {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
    }
    bool dropout_metadata_quarantined(std::uint64_t metadata) const {
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        return owner != slots.end() && owner->dnn_metadata &&
               owner->dnn_metadata->snapshot().is_quarantined;
    }
    std::uint64_t rnn_fault_bits(const Slot& slot, std::uint64_t metadata) const {
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        const bool metadata_fault = owner != slots.end() && owner->dnn_metadata &&
                                    owner->peer == slot.peer &&
                                    owner->dnn_metadata->snapshot().is_quarantined;
        return (metadata_fault ? 1U : 0U) |
               (slot.session->health_snapshot().is_quarantined ? 2U : 0U);
    }
    Slot* backend_metadata_owner(const Slot& slot, std::uint64_t metadata) {
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return nullptr;
        return &*owner;
    }
    CudaDnnResult decode_dnn_backend_execute(Slot& slot, ExecutionPayloadReader& input,
                                             CudaSessionDnnBackendExecute& problem,
                                             std::uint64_t& metadata) {
        ExecutionDnnBackendExecute wire;
        if (!read_execution_dnn_backend_execute(input, wire) || !input.empty())
            return {.status = CUDA_ERROR_INVALID_VALUE};
        metadata = wire.metadata;
        if (!backend_metadata_owner(slot, metadata)) return {.status = CUDA_ERROR_INVALID_HANDLE};
        problem.handle = wire.handle;
        problem.plan = wire.plan;
        problem.variant = wire.variant;
        problem.plan_generation = wire.plan_generation;
        problem.variant_generation = wire.variant_generation;
        problem.bindings.reserve(wire.bindings.size());
        for (auto& binding : wire.bindings) {
            if (binding.offset > SIZE_MAX) return {.status = CUDA_ERROR_INVALID_VALUE};
            problem.bindings.push_back(
                {binding.index,
                 static_cast<CudaDnnBackendBindingClass>(binding.value_class),
                 binding.slot,
                 {binding.allocation, static_cast<std::size_t>(binding.offset)},
                 std::move(binding.host_bytes)});
        }
        return {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
    }
    std::optional<ExecutionWireReply> prepare_dnn_backend_execute(Slot& slot,
                                                                  ExecutionPayloadReader& input) {
        ExecutionDnnBackendExecutePrepare wire;
        if (!read_execution_dnn_backend_execute_prepare(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        auto* metadata = backend_metadata_owner(slot, wire.metadata);
        if (!metadata) return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = rnn_fault_bits(slot, wire.metadata);
            return reply;
        };
        ExecutionPayloadWriter output;
        if (!output.reserve_capacity(20 + 48 * kExecutionDnnBackendMaxBindings))
            return receipt(rejection(CUDA_ERROR_OUT_OF_MEMORY));
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        const auto prepared = slot.session->prepare_dnn_backend_execute(
            slot.connection, wire.handle, *metadata->dnn_metadata, wire.plan, wire.variant);
        if (prepared.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = receipt(project(prepared.result));
        if (reply.original_status || reply.id || reply.address || !reply.arguments.empty()) {
            slot.session->close();
            reply = rejection(CUDA_ERROR_UNKNOWN);
            reply.parameter_bytes = rnn_fault_bits(slot, wire.metadata) | 2U;
            return reply;
        }
        if (prepared.result.disposition != CudaDnnDisposition::kComplete ||
            prepared.result.status != CUDA_SUCCESS || reply.parameter_bytes) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            if (reply.status == CUDA_SUCCESS) reply.status = CUDA_ERROR_UNKNOWN;
            return reply;
        }
        ExecutionDnnBackendExecutePreparation returned{
            prepared.plan_generation, prepared.variant_generation, {}};
        returned.requirements.reserve(prepared.requirements.size());
        for (const auto& requirement : prepared.requirements) {
            const bool is_workspace = requirement.index == kExecutionDnnBackendWorkspaceIndex;
            const auto value_class =
                static_cast<ExecutionDnnBackendBindingClass>(requirement.value_class);
            const auto access = value_class == ExecutionDnnBackendBindingClass::kIgnored
                                    ? ExecutionDnnBackendBindingAccess::kNone
                                : value_class == ExecutionDnnBackendBindingClass::kByValue
                                    ? ExecutionDnnBackendBindingAccess::kRead
                                    : ExecutionDnnBackendBindingAccess::kReadWrite;
            returned.requirements.push_back({requirement.index,
                                             is_workspace
                                                 ? ExecutionDnnBackendBindingRole::kWorkspace
                                                 : ExecutionDnnBackendBindingRole::kTensor,
                                             value_class, access, requirement.uid, requirement.slot,
                                             requirement.bytes, requirement.alignment});
        }
        if (!write_execution_dnn_backend_execute_preparation(output, returned))
            return receipt(rejection(CUDA_ERROR_UNKNOWN));
        reply.data = output.take();
        return reply;
    }
    CudaDnnResult decode_dnn_rnn(Slot& slot, ExecutionPayloadReader& input,
                                 CudaSessionDnnRnn& problem, std::uint64_t& metadata) {
        ExecutionDnnRnnCompute wire;
        if (!read_execution_dnn_rnn_compute(input, wire) || !input.empty() ||
            wire.weight_bytes > SIZE_MAX || wire.workspace_bytes > SIZE_MAX ||
            wire.reserve_bytes > SIZE_MAX)
            return {.status = CUDA_ERROR_INVALID_VALUE};
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return {.status = CUDA_ERROR_INVALID_HANDLE};
        metadata = wire.metadata;
        // Pending dropout mutation defers before any partial descriptor set is
        // retained. This local product is dropped on every pre-admission retry.
        auto parent = owner->dnn_metadata->borrow_rnn(wire.descriptors[0]);
        if (parent.result.disposition == CudaDnnDisposition::kDeferred ||
            parent.result.status != CUDA_SUCCESS)
            return parent.result;
        problem.rnn = std::move(parent.view);
        problem.kind = static_cast<CudaDnnRnnComputeKind>(wire.kind);
        problem.mode = wire.mode;
        problem.handle = wire.handle;
        problem.weight_bytes = static_cast<std::size_t>(wire.weight_bytes);
        problem.workspace_bytes = static_cast<std::size_t>(wire.workspace_bytes);
        problem.reserve_bytes = static_cast<std::size_t>(wire.reserve_bytes);
        for (std::size_t index = 0; index != problem.descriptors.size(); ++index) {
            const auto id = wire.descriptors[index + 1];
            if (!id && index >= 2) continue;
            const auto view = owner->dnn_metadata->borrow(
                id, index < 2 ? CudaDnnObjectKind::kRnnData : CudaDnnObjectKind::kTensor);
            if (!view) return {.status = CUDA_ERROR_INVALID_HANDLE};
            problem.descriptors[index] = *view;
        }
        for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return {.status = CUDA_ERROR_INVALID_VALUE};
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        return {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
    }
    CudaDnnResult decode_dnn_attn(Slot& slot, ExecutionPayloadReader& input,
                                  CudaSessionDnnAttn& problem, std::uint64_t& metadata,
                                  bool is_preparation = false) {
        ExecutionDnnAttnCompute wire;
        if (!read_execution_dnn_attn_compute(input, wire) || !input.empty() ||
            wire.weight_bytes > SIZE_MAX || wire.workspace_bytes > SIZE_MAX ||
            wire.reserve_bytes > SIZE_MAX ||
            (is_preparation && (!wire.lower.empty() || !wire.upper.empty())))
            return {.status = CUDA_ERROR_INVALID_VALUE};
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return {.status = CUDA_ERROR_INVALID_HANDLE};
        metadata = wire.metadata;
        auto parent = owner->dnn_metadata->borrow_attn(wire.descriptors[0]);
        if (parent.result.disposition == CudaDnnDisposition::kDeferred ||
            parent.result.status != CUDA_SUCCESS)
            return parent.result;
        problem.attn = std::move(parent.view);
        problem.kind = static_cast<CudaDnnAttnComputeKind>(wire.kind);
        problem.mode = wire.mode;
        problem.current = wire.current;
        problem.window_presence = wire.window_presence;
        problem.handle = wire.handle;
        problem.weight_bytes = static_cast<std::size_t>(wire.weight_bytes);
        problem.workspace_bytes = static_cast<std::size_t>(wire.workspace_bytes);
        problem.reserve_bytes = static_cast<std::size_t>(wire.reserve_bytes);
        for (std::size_t index = 0; index != problem.descriptors.size(); ++index) {
            const auto id = wire.descriptors[index + 1];
            if (!id) continue;
            const auto view = owner->dnn_metadata->borrow(id, CudaDnnObjectKind::kSeqData);
            if (!view) return {.status = CUDA_ERROR_INVALID_HANDLE};
            problem.descriptors[index] = *view;
        }
        for (std::size_t index = 0; index != problem.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return {.status = CUDA_ERROR_INVALID_VALUE};
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        problem.windows[0].assign(wire.lower.begin(), wire.lower.end());
        problem.windows[1].assign(wire.upper.begin(), wire.upper.end());
        return {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
    }
    std::optional<ExecutionWireReply> prepare_dnn_attn(Slot& slot, ExecutionPayloadReader& input) {
        CudaSessionDnnAttn problem;
        std::uint64_t metadata{};
        const auto decoded = decode_dnn_attn(slot, input, problem, metadata, true);
        if (decoded.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = rnn_fault_bits(slot, metadata);
            return reply;
        };
        if (decoded.status != CUDA_SUCCESS) return receipt(rejection(decoded.status));
        ExecutionPayloadWriter output(8);
        if (!output.reserve_capacity(8)) return receipt(rejection(CUDA_ERROR_OUT_OF_MEMORY));
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        const auto prepared = slot.session->prepare_dnn_attn(problem);
        if (prepared.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = receipt(project(prepared.result));
        if (prepared.has_context_loss) reply.parameter_bytes |= 2U;
        // A private observation can never publish requested-compute provenance.
        if (reply.original_status || reply.id || reply.address || !reply.arguments.empty()) {
            slot.session->close();
            reply = rejection(CUDA_ERROR_UNKNOWN);
            reply.parameter_bytes = rnn_fault_bits(slot, metadata) | 2U;
            return reply;
        }
        if (prepared.result.disposition == CudaDnnDisposition::kComplete &&
            prepared.result.status == CUDA_SUCCESS && !reply.parameter_bytes) {
            if (!write_execution_dnn_attn_window_range(output, {prepared.first, prepared.count})) {
                slot.session->close();
                reply = rejection(CUDA_ERROR_UNKNOWN);
                reply.parameter_bytes = rnn_fault_bits(slot, metadata) | 2U;
            } else {
                reply.data = output.take();
            }
        } else {
            reply.disposition = ExecutionWireDisposition::kRejected;
            if (reply.status == CUDA_SUCCESS) reply.status = CUDA_ERROR_UNKNOWN;
        }
        return reply;
    }
    std::optional<ExecutionWireReply> build_dnn_rnn(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnBuildRnnDynamic wire;
        if (!read_execution_dnn_build_rnn_dynamic(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) {
            auto reply = rejection(activated.status);
            reply.parameter_bytes = rnn_fault_bits(slot, wire.metadata);
            return reply;
        }
        const auto result = slot.session->build_dnn_rnn(wire.handle, *owner->dnn_metadata,
                                                        wire.descriptor, wire.mini_batch);
        if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = project(result.result);
        reply.parameter_bytes =
            rnn_fault_bits(slot, wire.metadata) | (result.has_context_loss ? 2U : 0U);
        return reply;
    }
    std::optional<ExecutionWireReply> query_dnn_rnn_weight_params(Slot& slot,
                                                                  ExecutionPayloadReader& input) {
        ExecutionDnnRnnWeightParams wire;
        if (!read_execution_dnn_rnn_weight_params(input, wire) || !input.empty() ||
            wire.weights.offset > SIZE_MAX || wire.weight_bytes > SIZE_MAX)
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = rnn_fault_bits(slot, wire.metadata);
            return reply;
        };
        // Reserve the whole fixed result before a native call can mutate outputs.
        ExecutionPayloadWriter output(20);
        if (!output.reserve_capacity(20)) return receipt(rejection(CUDA_ERROR_OUT_OF_MEMORY));
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        const CudaSessionDnnRnnWeightParams problem{
            {wire.descriptor, wire.pseudo_layer, wire.linear_layer, wire.destinations,
             wire.outputs},
            {wire.weights.allocation, static_cast<std::size_t>(wire.weights.offset)},
            static_cast<std::size_t>(wire.weight_bytes)};
        // The registry checks unique mutable destinations before its own pins;
        // no partial descriptor borrow or physical drain belongs in the service.
        const auto result =
            slot.session->query_dnn_rnn_weight_params(wire.handle, *owner->dnn_metadata, problem);
        if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = receipt(project(result.result));
        if (result.has_context_loss) reply.parameter_bytes |= 2U;
        if (result.result.disposition == CudaDnnDisposition::kComplete &&
            result.result.status == CUDA_SUCCESS &&
            result.result.original_status == CUDNN_STATUS_SUCCESS && !reply.parameter_bytes) {
            if (!write_execution_dnn_rnn_weight_params_result(output,
                                                              {result.non_null, result.offsets})) {
                // With valid trusted fields and reserved capacity this is an
                // internal contract failure, never an excuse to replay mutation.
                owner->is_closing = slot.is_closing = true;
                owner->dnn_metadata->close();
                slot.session->close();
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_UNKNOWN;
                reply.parameter_bytes = 3;
                return reply;  // Preserve the original SUCCESS provenance.
            }
            reply.data = output.take();
        }
        return reply;
    }
    std::optional<ExecutionWireReply> query_dnn_attn_weights(Slot& slot,
                                                             ExecutionPayloadReader& input) {
        ExecutionDnnAttnWeights wire;
        if (!read_execution_dnn_attn_weights(input, wire) || !input.empty() ||
            wire.weight_bytes > SIZE_MAX)
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = rnn_fault_bits(slot, wire.metadata);
            return reply;
        };
        // No caller address or allocation authority crosses this layout query.
        // Reserve the complete result before the original can mutate metadata.
        ExecutionPayloadWriter output(12);
        if (!output.reserve_capacity(12)) return receipt(rejection(CUDA_ERROR_OUT_OF_MEMORY));
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        const CudaDnnAttnWeights parameters{
            wire.descriptor,  wire.kind,
            wire.has_base,    static_cast<std::size_t>(wire.weight_bytes),
            wire.destination, wire.has_output};
        const auto result =
            slot.session->query_dnn_attn_weights(wire.handle, *owner->dnn_metadata, parameters);
        if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = receipt(project(result.result));
        if (result.has_context_loss) reply.parameter_bytes |= 2U;
        if (result.result.disposition == CudaDnnDisposition::kComplete &&
            result.result.status == CUDA_SUCCESS &&
            result.result.original_status == CUDNN_STATUS_SUCCESS && !reply.parameter_bytes) {
            if (!write_execution_dnn_attn_weights_result(output,
                                                         {result.non_null, result.offset})) {
                owner->is_closing = slot.is_closing = true;
                owner->dnn_metadata->close();
                slot.session->close();
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_UNKNOWN;
                reply.parameter_bytes = 3;
                return reply;  // Never replay a successful native mutation.
            }
            reply.data = output.take();
        }
        return reply;
    }
    bool borrow_dnn_convolution(Slot& slot, std::uint64_t metadata,
                                const std::array<std::uint64_t, 4>& ids,
                                CudaSessionDnnConvolution& problem) {
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return false;
        constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kFilter,
                                   CudaDnnObjectKind::kConvolution, CudaDnnObjectKind::kTensor};
        for (std::size_t index = 0; index != ids.size(); ++index) {
            auto descriptor = owner->dnn_metadata->borrow(ids[index], kinds[index]);
            if (!descriptor) return false;
            problem.descriptors[index] = std::move(*descriptor);
        }
        return true;
    }
    bool decode_dnn_convolution(Slot& slot, ExecutionPayloadReader& input,
                                CudaSessionDnnConvolution& problem) {
        ExecutionDnnConvolutionForward wire;
        if (!read_execution_dnn_convolution_forward(input, wire) || !input.empty() ||
            wire.workspace_bytes > SIZE_MAX ||
            !borrow_dnn_convolution(slot, wire.metadata, wire.descriptors, problem))
            return false;
        problem.handle = wire.handle;
        problem.algorithm = wire.algorithm;
        problem.workspace_bytes = static_cast<std::size_t>(wire.workspace_bytes);
        for (std::size_t index = 0; index != wire.buffers.size(); ++index) {
            if (wire.buffers[index].offset > SIZE_MAX) return false;
            problem.buffers[index] = {wire.buffers[index].allocation,
                                      static_cast<std::size_t>(wire.buffers[index].offset)};
        }
        problem.scalar_bytes = wire.scalar_bytes;
        problem.alpha = wire.alpha;
        problem.beta = wire.beta;
        return true;
    }
    bool decode_dnn_find(Slot& slot, ExecutionPayloadReader& input,
                         CudaSessionDnnConvolution& problem, int& requested) {
        ExecutionDnnConvolutionFind wire;
        if (!read_execution_dnn_convolution_find(input, wire) || !input.empty() ||
            !borrow_dnn_convolution(slot, wire.metadata, wire.descriptors, problem))
            return false;
        problem.handle = wire.handle;
        requested = std::bit_cast<std::int32_t>(wire.requested);
        return true;
    }
    ExecutionWireReply project_dnn_convolution_result(Slot& slot,
                                                      const CudaDnnConvolutionQueryResult& result,
                                                      bool has_performance) {
        auto reply = project(result.result);
        reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
        if (result.result.disposition == CudaDnnDisposition::kComplete &&
            result.result.status == CUDA_SUCCESS &&
            result.result.original_status == CUDNN_STATUS_SUCCESS && has_performance) {
            ExecutionDnnConvolutionPerformances performances;
            performances.count = result.count;
            static_assert(kCudaDnnMaxConvolutionAlgorithms ==
                          kExecutionDnnMaxConvolutionAlgorithms);
            if (result.count > performances.values.size()) {
                slot.session->close();
                reply = rejection(CUDA_ERROR_UNKNOWN);
                reply.parameter_bytes = 1;
                return reply;
            }
            for (std::size_t index = 0; index != result.count; ++index) {
                const auto& value = result.performance[index];
                performances.values[index] = {value.algorithm, value.status,      value.time_bits,
                                              value.memory,    value.determinism, value.math,
                                              value.reserved};
            }
            ExecutionPayloadWriter output;
            if (!write_execution_dnn_convolution_performances(output, performances))
                return rejection(CUDA_ERROR_OUT_OF_MEMORY);
            reply.data = output.take();
        }
        return reply;
    }
    ExecutionWireReply query_dnn_convolution(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnConvolutionQuery wire;
        if (!read_execution_dnn_convolution_query(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        CudaSessionDnnConvolution problem;
        problem.handle = wire.handle;
        problem.algorithm = wire.algorithm;
        if (wire.kind != ExecutionDnnConvolutionQueryKind::kMaxCount &&
            !borrow_dnn_convolution(slot, wire.metadata, wire.descriptors, problem))
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        const auto result = slot.session->query_dnn_convolution(
            problem, static_cast<CudaDnnConvolutionQueryKind>(wire.kind),
            std::bit_cast<std::int32_t>(wire.requested));
        return project_dnn_convolution_result(
            slot, result, wire.kind == ExecutionDnnConvolutionQueryKind::kHeuristic);
    }
    ExecutionWireReply query_dnn_backward_convolution(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnBackwardConvolutionQuery wire;
        if (!read_execution_dnn_backward_convolution_query(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        CudaSessionDnnConvolution problem;
        problem.handle = wire.handle;
        problem.algorithm = wire.algorithm;
        if (wire.kind == ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace) {
            // Wire follows the SDK ABI (w,dy,conv,dx); the shared typed
            // borrower and backend problem use tensor/filter order (dy,w,conv,dx).
            std::swap(wire.descriptors[0], wire.descriptors[1]);
            if (!borrow_dnn_convolution(slot, wire.metadata, wire.descriptors, problem))
                return rejection(CUDA_ERROR_INVALID_HANDLE);
        }
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        auto reply = project(slot.session->query_dnn_backward_convolution(
            problem, static_cast<CudaDnnBackwardConvolutionQueryKind>(wire.kind),
            wire.has_output != 0));
        reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
        return reply;
    }
    std::optional<ExecutionWireReply> query_dnn_rnn(Slot& slot, ExecutionPayloadReader& input) {
        ExecutionDnnRnnQuery wire;
        if (!read_execution_dnn_rnn_query(input, wire) || !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = (owner->dnn_metadata->snapshot().is_quarantined ? 1U : 0U) |
                                    (slot.session->health_snapshot().is_quarantined ? 2U : 0U);
            return reply;
        };
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        // The registry atomically retains the parent, input and dropout dependencies.
        // Do not drain the session or borrow a partial dependency set here.
        const auto result =
            slot.session->query_dnn_rnn(wire.handle, *owner->dnn_metadata,
                                        {static_cast<CudaDnnRnnQueryKind>(wire.kind),
                                         wire.descriptor, wire.input, wire.mode, wire.outputs});
        if (result.result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        auto reply = receipt(project(result.result));
        if (result.result.disposition == CudaDnnDisposition::kComplete &&
            result.result.status == CUDA_SUCCESS &&
            result.result.original_status == CUDNN_STATUS_SUCCESS && !reply.parameter_bytes) {
            ExecutionPayloadWriter output;
            for (const auto size : result.sizes)
                if (!output.write_u64(size)) return receipt(rejection(CUDA_ERROR_OUT_OF_MEMORY));
            reply.data = output.take();
        }
        return reply;
    }
    std::optional<ExecutionWireReply> fold_dnn_backward_convolution(Slot& slot,
                                                                    ExecutionPayloadReader& input) {
        ExecutionDnnFoldedConvBackwardDataDescriptors wire;
        if (!read_execution_dnn_folded_conv_backward_data_descriptors(input, wire) ||
            !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto owner = std::ranges::find(slots, wire.metadata, &Slot::connection);
        if (owner == slots.end() || !owner->dnn_metadata || owner->peer != slot.peer ||
            !owner->is_initialized || owner->is_retiring || owner->is_closing)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        // Do not borrow inputs here: an input/output alias must not create its
        // own immutable-borrow conflict before the registry's exclusive gate.
        const auto receipt = [&](ExecutionWireReply reply) {
            reply.parameter_bytes = (owner->dnn_metadata->snapshot().is_quarantined ? 1U : 0U) |
                                    (slot.session->health_snapshot().is_quarantined ? 2U : 0U);
            return reply;
        };
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return receipt(rejection(activated.status));
        const auto result = slot.session->fold_dnn_backward_convolution(
            wire.handle, *owner->dnn_metadata, {wire.inputs, wire.outputs, wire.format});
        // Only a not-issued immutable-borrow/lifecycle conflict can defer.
        // An original return, including failure, is never replayed.
        if (result.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
        return receipt(project(result));
    }
#endif
    bool borrow_lt_query(Slot& slot, ExecutionPayloadReader& input, CudaSessionLtQuery& query) {
        std::uint64_t metadata{}, operation{};
        std::array<std::uint64_t, 4> matrices{};
        if (!input.read_u64(query.handle) || !input.read_u64(metadata) ||
            !input.read_u64(operation))
            return false;
        for (auto& matrix : matrices)
            if (!input.read_u64(matrix)) return false;
        const auto owner = std::ranges::find(slots, metadata, &Slot::connection);
        if (owner == slots.end() || !owner->lt_metadata || owner->peer != slot.peer ||
            owner->is_bootstrap_pending || owner->is_retiring || owner->is_closing)
            return false;
        const auto descriptor = owner->lt_metadata->borrow(operation, CudaLtObjectKind::kOperation);
        if (!descriptor) return false;
        query.operation = *descriptor;
        query.has_c_layout = matrices[2] != 0;
        if (!matrices[2]) matrices[2] = matrices[3];
        for (std::size_t index = 0; index < matrices.size(); ++index) {
            const auto view =
                owner->lt_metadata->borrow(matrices[index], CudaLtObjectKind::kLayout);
            if (!view) return false;
            query.matrices[index] = *view;
        }
        return true;
    }
    std::optional<ExecutionWireReply> execute_lt_query(Slot& slot, Slot::Request& request) {
        ExecutionPayloadReader input(request.message.payload);
        CudaSessionLtQuery query;
        if (!borrow_lt_query(slot, input, query)) return rejection(CUDA_ERROR_INVALID_HANDLE);
        if (request.message.header.operation == ExecutionOpcode::kLtHeuristic) {
            std::uint64_t preference{};
            std::uint32_t count{};
            if (!input.read_u64(preference) || !input.read_u32(count) || !count || count > 64)
                return rejection(CUDA_ERROR_INVALID_VALUE);
            // Operation and preference share the exact authenticated CPU owner.
            ExecutionPayloadReader prefix(request.message.payload);
            std::uint64_t ignored{}, metadata{};
            if (!prefix.read_u64(ignored) || !prefix.read_u64(metadata))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            const auto origin = std::ranges::find(slots, metadata, &Slot::connection);
            query.preference =
                origin->lt_metadata->borrow(preference, CudaLtObjectKind::kPreference);
            if (!query.preference) return rejection(CUDA_ERROR_INVALID_HANDLE);
            query.requested = static_cast<int>(count);
        } else {
            std::span<const std::byte> algorithm;
            if (!input.read_blob(algorithm) || algorithm.size() != sizeof(cublasLtMatmulAlgo_t))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            query.algorithm.emplace();
            std::memcpy(&*query.algorithm, algorithm.data(), algorithm.size());
        }
        if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
        // These metadata queries bypass generic issue-time activation. Borrowed
        // CPU descriptors do not prove that their GPU handle's context is current.
        const auto activated = activate(slot);
        if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
        const auto result = slot.session->query_lt_algorithms(query);
        if (result.result.disposition == CudaSessionDisposition::kDeferred) return std::nullopt;
        auto reply = project(result.result);
        reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
        if (result.result.disposition != CudaSessionDisposition::kComplete ||
            result.result.original_status != CUBLAS_STATUS_SUCCESS)
            return reply;
        // Register value provenance with the SAME authenticated metadata owner
        // before publishing any algorithm. This retains no GPU owner/handle.
        ExecutionPayloadReader prefix(request.message.payload);
        std::uint64_t ignored{}, metadata{};
        if (!prefix.read_u64(ignored) || !prefix.read_u64(metadata))
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto origin = std::ranges::find(slots, metadata, &Slot::connection);
        if (origin == slots.end() || origin->peer != slot.peer || !origin->lt_metadata)
            return rejection(CUDA_ERROR_INVALID_HANDLE);
        const auto remembered =
            origin->lt_metadata->remember_algorithms(result.algorithms, result.sdk_version);
        if (remembered != CUDA_SUCCESS) return rejection(remembered);
        ExecutionPayloadWriter output;
        if (!output.write_u32(static_cast<std::uint32_t>(result.algorithms.size())))
            throw std::bad_alloc();
        for (const auto& item : result.algorithms) {
            if (!output.write_blob(std::as_bytes(std::span{&item.algo, 1})) ||
                !output.write_u64(item.workspaceSize) ||
                !output.write_u32(static_cast<std::uint32_t>(item.state)) ||
                !output.write_u32(std::bit_cast<std::uint32_t>(item.wavesCount)))
                throw std::bad_alloc();
            for (const auto reserved : item.reserved)
                if (!output.write_u32(std::bit_cast<std::uint32_t>(reserved)))
                    throw std::bad_alloc();
        }
        reply.address = result.sdk_version;
        reply.data = output.take();
        return reply;
    }
    bool decode_lt_matmul(Slot& slot, ExecutionPayloadReader& input, CudaSessionLtMatmul& problem) {
        if (!borrow_lt_query(slot, input, problem.descriptors)) return false;
        std::uint32_t has_algorithm{}, device_mode{}, scalar_bytes{};
        std::span<const std::byte> algorithm;
        if (!input.read_u32(has_algorithm) || has_algorithm > 1) return false;
        if (has_algorithm) {
            if (!input.read_blob(algorithm) || algorithm.size() != sizeof(cublasLtMatmulAlgo_t))
                return false;
            problem.descriptors.algorithm.emplace();
            std::memcpy(&*problem.descriptors.algorithm, algorithm.data(), algorithm.size());
        }
        std::uint64_t workspace{};
        if (!input.read_u64(problem.stream) || !input.read_u64(workspace) || workspace > SIZE_MAX)
            return false;
        problem.workspace_bytes = static_cast<std::size_t>(workspace);
        const auto range = [&](CudaAllocationRange& value) {
            std::uint64_t offset{};
            if (!input.read_u64(value.allocation) || !input.read_u64(offset) || offset > SIZE_MAX)
                return false;
            value.offset = static_cast<std::size_t>(offset);
            return value.allocation || !offset;
        };
        if (!range(problem.workspace)) return false;
        for (auto& matrix : problem.matrices)
            if (!range(matrix)) return false;
        if (!input.read_u32(device_mode) || device_mode > 1 || !input.read_u32(scalar_bytes) ||
            !scalar_bytes || scalar_bytes > 16)
            return false;
        problem.scalar_bytes = scalar_bytes;
        problem.uses_device_scalars = device_mode != 0;
        if (device_mode) {
            for (auto& scalar : problem.device_scalars)
                if (!range(scalar)) return false;
        } else {
            for (auto* scalar : {&problem.alpha, &problem.beta}) {
                std::span<const std::byte> bytes;
                if (!input.read_blob(bytes) || bytes.size() != scalar_bytes) return false;
                std::memcpy(scalar->data(), bytes.data(), bytes.size());
            }
        }
        return input.empty();
    }
    static ExecutionAsyncFrontierObservation observe_frontier(const Slot& slot,
                                                              const Slot::Frontier& frontier) {
        auto observed = slot.session->query_frontier(frontier.token);
        bool retired = frontier.token.retirement().is_retired;
        for (const auto& prior : frontier.asynchronous_prefix) {
            const auto status = slot.session->query_frontier(prior);
            retired = retired && prior.retirement().is_retired;
            if (observed == CUDA_SUCCESS ||
                (observed == CUDA_ERROR_NOT_READY && status != CUDA_SUCCESS))
                observed = status;
        }
        return {{frontier.id, frontier.asynchronous_sequence},
                static_cast<std::int32_t>(slot.policy_failure.value_or(observed)),
                retired};
    }
    static ExecutionWireReply execute_async_frontier_batch(Slot& slot, const Slot::Request& request,
                                                           bool exact_continuation) {
        if (request.continuation || exact_continuation || request.borrowed_descriptor() >= 0)
            return rejection(CUDA_ERROR_INVALID_VALUE);
        const auto keys = decode_execution_async_frontier_batch(request.message.payload);
        if (!keys) return rejection(CUDA_ERROR_INVALID_VALUE);
        std::array<const Slot::Frontier*, kExecutionMaxAsyncFrontiers> frontiers{};
        for (std::size_t index = 0; index != keys->size(); ++index) {
            const auto& key = (*keys)[index];
            const auto found = std::ranges::find(slot.frontiers, key.id, &Slot::Frontier::id);
            if (found == slot.frontiers.end() || !found->asynchronous_sequence ||
                found->asynchronous_sequence != key.sequence)
                return rejection(CUDA_ERROR_INVALID_HANDLE);
            frontiers[index] = &*found;
        }
        const bool is_release =
            request.message.header.operation == ExecutionOpcode::kReleaseAsyncFrontiers;
        std::vector<ExecutionAsyncFrontierObservation> observations;
        if (!is_release) observations.reserve(keys->size());
        for (const auto* frontier : std::span{frontiers}.first(keys->size())) {
            const auto observed = observe_frontier(slot, *frontier);
            if (is_release) {
                // A reported error is independent of physical retirement. No
                // accepted observation disappears while its ownership is live.
                if (!observed.is_retired) return rejection(CUDA_ERROR_NOT_READY);
            } else {
                observations.push_back(observed);
            }
        }
        auto encoded = is_release ? encode_execution_async_frontier_batch(*keys)
                                  : encode_execution_async_frontier_observations(observations);
        if (!encoded) return rejection(CUDA_ERROR_OUT_OF_MEMORY);
        auto reply = rejection(CUDA_SUCCESS);
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.data = std::move(*encoded);
        if (is_release) {
            // Every identity and physical stop, plus the complete echoed reply
            // allocation, is proven before the first metadata deletion.
            std::erase_if(slot.frontiers, [&](const auto& frontier) {
                return std::ranges::any_of(*keys,
                                           [&](const auto& key) { return frontier.id == key.id; });
            });
        }
        return reply;
    }
    std::optional<ExecutionWireReply> execute_host_function(Slot& slot, Slot::Request& request,
                                                            bool can_issue) {
        ExecutionPayloadReader input(request.message.payload);
        const auto operation = request.message.header.operation;
        std::uint64_t id{};
        std::uint32_t has_returned{};
        if (request.borrowed_descriptor() >= 0 || !input.read_u64(id) ||
            (operation == ExecutionOpcode::kFinishHostFunction &&
             (!input.read_u32(has_returned) || has_returned > 1)) ||
            !input.empty())
            return rejection(CUDA_ERROR_INVALID_VALUE);
        if (operation == ExecutionOpcode::kEnqueueHostFunction) {
            if (slot.policy_failure) return rejection(*slot.policy_failure);
            if (slot.asynchronous_failure) return rejection(*slot.asynchronous_failure);
            if (!slot.has_host_function_space ||
                slot.host_functions.size() == config.session_limits.max_objects)
                return rejection(CUDA_ERROR_OUT_OF_MEMORY);
            // Capture the upcoming stream's physical predecessors BEFORE the
            // new logical boundary, including legacy/blocking-stream edges.
            // The client never supplies this frontier or a callback address.
            auto predecessor = slot.session->capture_frontier(id, true);
            if (predecessor.status != CUDA_SUCCESS) return rejection(predecessor.status);
            const auto assigned = slot.next_host_function;
            const auto native = dispatch.launch_host_function
                                    ? std::make_shared<CudaSessionHostFunction>()
                                    : nullptr;
            const std::shared_ptr<const void> lifetime =
                native ? std::static_pointer_cast<const void>(native)
                       : std::static_pointer_cast<const void>(
                             std::make_shared<const std::uint64_t>(assigned));
            auto accepted = slot.session->accept_external_barrier(id, lifetime);
            if (accepted.status != CUDA_SUCCESS) return rejection(accepted.status);
            // Capacity was reserved with the Slot. Nothing after acceptance
            // allocates or invokes native work; every admitted boundary is owned.
            slot.host_functions.push_back({assigned, std::move(accepted.barrier),
                                           std::move(predecessor.frontier), false, false, native,
                                           id, false});
            remember_command_frontier(request, std::move(accepted.frontier));
            if (assigned == std::numeric_limits<std::uint64_t>::max())
                slot.has_host_function_space = false;
            else
                ++slot.next_host_function;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.id = assigned;
            return reply;
        }
        const auto found = std::ranges::find(slot.host_functions, id, &Slot::HostFunction::id);
        if (found == slot.host_functions.end()) return rejection(CUDA_ERROR_INVALID_HANDLE);
        if (found->is_uncertain) return rejection(CUDA_ERROR_UNKNOWN);
        if (operation == ExecutionOpcode::kQueryHostFunction) {
            if (found->has_permission) return rejection(CUDA_ERROR_INVALID_VALUE);
            if (found->native) {
                if (!found->is_enqueued) {
                    if (!can_issue) return std::nullopt;
                    const auto activated = activate(slot);
                    if (activated.status == CUDA_ERROR_NOT_READY && !activated.original_status)
                        return std::nullopt;
                    if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                    const auto ready = slot.session->external_barrier_ready(found->barrier);
                    if (ready == CUDA_ERROR_NOT_READY) return std::nullopt;
                    if (ready != CUDA_SUCCESS) return rejection(ready);
                    const auto status = slot.session->enqueue_host_function(
                        found->barrier, found->stream, found->native);
                    // After native entry any error is retained, never replayed.
                    found->is_enqueued = true;
                    if (status != CUDA_SUCCESS) {
                        policy_failed(slot, status);
                        found->is_uncertain = true;
                        return rejection(status);
                    }
                }
                if (!found->native->entered()) return std::nullopt;
            }
            const auto status = slot.policy_failure.value_or(slot.asynchronous_failure.value_or(
                slot.session->health_snapshot().is_quarantined
                    ? CUDA_ERROR_UNKNOWN
                    : slot.session->query_frontier(found->predecessor)));
            if (status == CUDA_ERROR_NOT_READY) return std::nullopt;
            if (status != CUDA_SUCCESS) {
                if (slot.session_leases.has_leases()) {
                    // A sticky error can come from an unrelated stream while
                    // this callback's physical predecessor remains healthy.
                    // External owners retain dependencies beyond this request;
                    // successful cancellation would let them skip the unrun
                    // callback. Preserve its boundary and original error, and
                    // give no known-removed marker to the client.
                    policy_failed(slot, status);
                    found->is_uncertain = true;
                    return rejection(status);
                }
                if (found->native && found->is_enqueued) {
                    policy_failed(slot, status);
                    found->is_uncertain = true;
                    return rejection(status);
                }
                const auto cancelled = slot.session->cancel_external_barrier(found->barrier);
                if (cancelled != CUDA_SUCCESS) {
                    policy_failed(slot, cancelled);
                    found->is_uncertain = true;
                    return rejection(cancelled);
                }
                // This exact marker proves only that permission was never
                // published. It cannot clear the source's independent GPU fault.
                slot.host_functions.erase(found);
                auto reply = rejection(status);
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.id = id;
                reply.parameter_bytes = 1;
                return reply;
            }
            // Set before serialization/publication. Even a failed transport
            // reply is conservatively treated as possibly delivered permission.
            found->has_permission = true;
        } else {
            if (!found->has_permission) return rejection(CUDA_ERROR_INVALID_VALUE);
            if (!has_returned) {
                // An exception or unknown callback outcome is not a normal
                // return and must not release subsequent stream computation.
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                found->is_uncertain = true;
                return rejection(CUDA_ERROR_UNKNOWN);
            }
            const auto cancelled = found->native && found->is_enqueued
                                       ? (found->native->finish(), CUDA_SUCCESS)
                                       : slot.session->cancel_external_barrier(found->barrier);
            if (cancelled != CUDA_SUCCESS) {
                policy_failed(slot, cancelled);
                found->is_uncertain = true;
                return rejection(cancelled);
            }
            // Native FIFO remains held until the callback returns. Its recorded
            // event, not this Finish receipt, proves actual physical retirement.
            slot.host_functions.erase(found);
        }
        auto reply = rejection(CUDA_SUCCESS);
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.id = id;
        return reply;
    }
    bool is_dnn_profile_control(const Slot& slot, const Slot::Request& request) const noexcept {
        const auto operation = request.message.header.operation;
        if (operation == ExecutionOpcode::kBindDnnProfile) return true;
#ifdef GLIMMER_HAS_CUDNN
        return execution_opcode_is_dnn(operation) && !request.receipt &&
               (slot.has_dnn_profile_mismatch ||
                slot.dnn_profile_version != glimmer::internal::cudnn_profile::kHeaderVersion);
#else
        (void)slot;
        return false;
#endif
    }
    std::optional<ExecutionWireReply> execute(Slot& slot, Slot::Request& request, bool can_issue,
                                              bool exact_continuation = false) {
        if (config.issue_depth_metrics &&
            issue_depth_compute_class(request.message.header.operation) ==
                IssueDepthComputeClass::kUnsupported)
            config.issue_depth_metrics->invalidate(IssueDepthMetrics::Failure::kUnsupported);
        // These requests cannot issue SDK work. Preserve canonical binding or
        // rejection even when another member owns shared native configuration.
        if (is_dnn_profile_control(slot, request))
            return execute_unobserved(slot, request, false, exact_continuation);
        if (memory_native_health() != CUDA_SUCCESS)
            return execute_unobserved(slot, request, false, exact_continuation);
        if (slot.is_paired_member && !slot.is_bootstrap_pending &&
            slot.session->health_snapshot().has_context) {
            const auto operation = request.message.header.operation;
            const bool cpu_only =
                is_asynchronous_admission(request) || is_host_function_command(operation) ||
                is_async_frontier_batch(operation) || operation == ExecutionOpcode::kSnapshot ||
                operation == ExecutionOpcode::kCaptureFrontier ||
                operation == ExecutionOpcode::kQueryFrontier ||
                operation == ExecutionOpcode::kReleaseFrontier ||
                (request.receipt && slot.session->host_result(*request.receipt));
            if (!cpu_only) {
                const auto activated = activate(slot);
                if (activated.status == CUDA_ERROR_NOT_READY && !activated.original_status)
                    return std::nullopt;
                if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
            }
        }
        if (!config.should_audit_native || !slot.session ||
            !execution_uses_preparation_deadline(request.message.header.operation))
            return execute_unobserved(slot, request, can_issue, exact_continuation);
        if (request.preparation_attempt != std::numeric_limits<std::uint64_t>::max())
            ++request.preparation_attempt;
        PreparationAudit audit(preparation_audit, *slot.session, audit_log, slot.connection,
                               request.message.header, request.preparation_attempt);
        auto reply = execute_unobserved(slot, request, can_issue, exact_continuation);
        audit.finish(reply);
        return reply;
    }
    std::optional<ExecutionWireReply> execute_unobserved(Slot& slot, Slot::Request& request,
                                                         bool can_issue, bool exact_continuation) {
#ifdef GLIMMER_HAS_CUDNN
        const auto dnn_operation = request.message.header.operation;
        if (dnn_operation == ExecutionOpcode::kBindDnnProfile) {
            if (slot.is_bootstrap_pending || request.continuation ||
                request.descriptor.get() >= 0 || slot.host_owner || slot.lt_metadata ||
                slot.is_pointer_metadata || slot.is_process_memory || slot.is_nccl)
                return rejection(CUDA_ERROR_INVALID_VALUE);
            std::uint32_t version{};
            ExecutionPayloadReader input(request.message.payload);
            if (!read_execution_dnn_profile(input, version))
                return rejection(CUDA_ERROR_INVALID_VALUE);
            if (!dispatch.dnn || slot.has_dnn_profile_mismatch ||
                dispatch.verified_dnn_header_version !=
                    glimmer::internal::cudnn_profile::kHeaderVersion ||
                version != dispatch.verified_dnn_header_version ||
                (slot.dnn_profile_version && slot.dnn_profile_version != version)) {
                slot.has_dnn_profile_mismatch = true;
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
            }
            ExecutionPayloadWriter output;
            if (!write_execution_dnn_profile(output, version))
                return rejection(CUDA_ERROR_OUT_OF_MEMORY);
            slot.dnn_profile_version = version;
            auto reply = rejection(CUDA_SUCCESS);
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.data = output.take();
            return reply;
        }
        // An accepted command belongs to the immutable original profile. A
        // later conflicting bind cannot replace its real result or retirement.
        if (execution_opcode_is_dnn(dnn_operation) && !request.receipt &&
            (slot.has_dnn_profile_mismatch ||
             slot.dnn_profile_version != glimmer::internal::cudnn_profile::kHeaderVersion))
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
#else
        if (request.message.header.operation == ExecutionOpcode::kBindDnnProfile)
            return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
        const auto health = memory_native_health();
        if (health != CUDA_SUCCESS) {
            const auto operation = request.message.header.operation;
            // Preserve exact already-produced host results. The issue fence
            // cannot rewrite their SDK provenance or invent physical retirement.
            const bool has_host_result = request.receipt && slot.session &&
                                         slot.session->host_result(*request.receipt).has_value();
            const bool cpu_only = is_host_function_observation(operation) ||
                                  operation == ExecutionOpcode::kSnapshot ||
                                  operation == ExecutionOpcode::kCaptureFrontier ||
                                  operation == ExecutionOpcode::kQueryFrontier ||
                                  operation == ExecutionOpcode::kReleaseFrontier ||
                                  is_async_frontier_batch(operation) ||
                                  (slot.is_metadata_owner() && slot.is_bootstrap_pending);
            if (!cpu_only && !has_host_result && !request.has_host_job &&
                !request.has_host_shutdown && !(slot.is_nccl && request.nccl_request)) {
                auto reply = rejection(health);
                if (operation == ExecutionOpcode::kProcessMemory ||
                    operation == ExecutionOpcode::kPointerMetadata ||
                    operation == ExecutionOpcode::kRelease)
                    reply.parameter_bytes = 1;
                if (slot.is_bootstrap_pending) {
                    slot.is_bootstrap_pending = false;
                    return finish_bootstrap(slot, std::move(reply));
                }
                return reply;
            }
            can_issue = false;
        }
#ifdef GLIMMER_HAS_NCCL
        if (slot.is_nccl) return execute_nccl(slot, request, can_issue);
#endif
        if (slot.is_pointer_metadata) return execute_pointer_metadata(slot, request);
        if (slot.is_process_memory) return execute_process_memory(slot, request);
#ifdef GLIMMER_HAS_CUDNN
        if (slot.dnn_metadata) return execute_dnn_metadata(slot, request);
#endif
        if (slot.lt_metadata) return execute_lt_metadata(slot, request);
        if (slot.host_owner) return execute_host(slot, request, can_issue);
        if (slot.is_bootstrap_pending) {
            const auto initialized = dispatch.initialize(0);
            if (initialized != CUDA_SUCCESS) {
                slot.is_bootstrap_pending = false;
                return finish_bootstrap(slot, rejection(initialized));
            }
#ifdef GLIMMER_HAS_NCCL
            if (config.uses_nccl) nccl_driver_ready = true;
#endif
            const auto detached = execution_domain() ? detach_selected_context()
                                                     : dispatch.context_set_current(nullptr);
            if (detached != CUDA_SUCCESS) {
                slot.is_bootstrap_pending = false;
                return finish_bootstrap(slot, rejection(detached));
            }
            const auto result =
                slot.session->initialize(config.device_ordinal, {slot.context_bootstrap.flags});
            if (slot.is_paired_member && result.disposition == CudaSessionDisposition::kDeferred &&
                !result.original_status && !slot.session->health_snapshot().has_context)
                return std::nullopt;
            slot.is_bootstrap_pending = false;
            if (slot.session->health_snapshot().has_context) selected_session = slot.session;
            audit(slot.is_low ? "low_bound" : "native_bound", slot, 0, result.status);
            return finish_bootstrap(slot, project(result));
        }
        const auto& message = request.message;
        ExecutionPayloadReader input(message.payload);
        CudaSessionResult result;
        std::uint64_t id{}, offset{}, count{};
        std::uint32_t flags{};
        std::string_view text;
        std::span<const std::byte> data;
        switch (message.header.operation) {
            case ExecutionOpcode::kBindDnnProfile:
                // The profile gate above consumes this opcode in every build.
                return rejection(CUDA_ERROR_INVALID_VALUE);
            case ExecutionOpcode::kGraphControl:
                return execute_graph_control(slot, request);
            case ExecutionOpcode::kProcessMemory: {
                ExecutionProcessMemoryRequest wire;
                if (request.borrowed_descriptor() >= 0 ||
                    !read_execution_process_memory_request(input, wire) || !input.empty() ||
                    (wire.operation != ExecutionProcessMemoryOperation::kAllocateAsync &&
                     wire.operation != ExecutionProcessMemoryOperation::kAllocateFromPoolAsync) ||
                    wire.bytes > SIZE_MAX)
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (!slot.process_memory) return rejection(CUDA_ERROR_NOT_SUPPORTED);
                if (!request.receipt) {
                    if (!slot.has_command_space) return rejection(CUDA_ERROR_INVALID_VALUE);
                    if (slot.is_low &&
                        slot.low_epochs.size() == config.session_limits.submissions.max_commands)
                        return std::nullopt;
                    const auto reserved = reserve_request_frontier(slot, request);
                    if (reserved != CUDA_SUCCESS) return rejection(reserved);
                    const auto accepted = slot.session->allocate_process_memory(
                        slot.next_command, process_request(wire));
                    if (accepted.acceptance.status != ExecutionAcceptStatus::kAccepted)
                        return rejection(accepted.acceptance.status ==
                                                 ExecutionAcceptStatus::kOutOfMemory
                                             ? CUDA_ERROR_OUT_OF_MEMORY
                                             : CUDA_ERROR_INVALID_VALUE);
                    request.receipt = accepted.acceptance.receipt;
                    lifecycle(HostStageMetrics::Lifecycle::kAdmit, slot,
                              request.message.header.sequence, request.receipt->sequence());
                    request.process_allocation = accepted.operation;
                    if (!bind_request_frontier(slot, request)) return std::nullopt;
                    if (slot.next_command == UINT64_MAX)
                        slot.has_command_space = false;
                    else
                        ++slot.next_command;
                    if (slot.is_low)
                        slot.low_epochs.push_back({.sequence = request.receipt->sequence(),
                                                   .frontier = *request.command_frontier,
                                                   .progress = {},
                                                   .receipt = *request.receipt});
                }
                if (can_issue && !exact_continuation && !uses_priority_space_share() &&
                    !slot.session->host_result(*request.receipt)) {
                    const auto ready = process_memory_native_gate();
                    if (ready == CUDA_ERROR_NOT_READY) return std::nullopt;
                    if (ready != CUDA_SUCCESS)
                        (void)slot.session->reject_queued(ready);
                    else {
                        bool resume{};
                        const auto issued = issue_scheduled_command(slot, resume);
                        if (issued.state == CudaExecutionIssueState::kError &&
                            issue_failure(issued) != CUDA_ERROR_NOT_READY)
                            (void)slot.session->reject_queued(issue_failure(issued));
                        note_scheduled_issue(slot, issued, resume);
                    }
                }
                const auto host = slot.session->host_result(*request.receipt);
                if (!host) return std::nullopt;
                const auto observed =
                    request.process_allocation ? request.process_allocation->reply() : std::nullopt;
                if (!observed) return rejection(static_cast<CUresult>(host->status));
                return project_process_memory(*observed);
            }
            case ExecutionOpcode::kEnqueueHostFunction:
            case ExecutionOpcode::kQueryHostFunction:
            case ExecutionOpcode::kFinishHostFunction:
                return execute_host_function(slot, request, can_issue);
            case ExecutionOpcode::kNcclCall:
                // The wire identity is reserved before public activation. Its
                // native actor must be composed under Core grants, not routed
                // as metadata or silently forwarded to a native fallback.
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
            case ExecutionOpcode::kDnnDropoutStatesSize: {
#ifdef GLIMMER_HAS_CUDNN
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto activated = activate(slot);
                if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                auto reply = project(slot.session->query_dnn_dropout_states_size(id));
                reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
                return reply;
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            }
            // Each opcode has a distinct handler when cuDNN is enabled.
            // NOLINTNEXTLINE(bugprone-branch-clone)
            case ExecutionOpcode::kDnnBatchNormQuery:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_batch_norm(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnReductionQuery:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_reduction(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnConvolutionQuery:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_convolution(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnBackwardConvolutionQuery:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_backward_convolution(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors:
#ifdef GLIMMER_HAS_CUDNN
                return fold_dnn_backward_convolution(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnRnnQuery:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_rnn(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnLegacyControl:
#ifdef GLIMMER_HAS_CUDNN
                return execute_dnn_legacy_control(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnBuildRnnDynamic:
#ifdef GLIMMER_HAS_CUDNN
                return build_dnn_rnn(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnRnnWeightParams:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_rnn_weight_params(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnAttnWeights:
#ifdef GLIMMER_HAS_CUDNN
                return query_dnn_attn_weights(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kDnnBackendContext:
#ifdef GLIMMER_HAS_CUDNN
                return execute_dnn_backend_context(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kPrepareDnnAttn:
#ifdef GLIMMER_HAS_CUDNN
                return prepare_dnn_attn(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kPrepareDnnBackendExecute:
#ifdef GLIMMER_HAS_CUDNN
                return prepare_dnn_backend_execute(slot, input);
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            case ExecutionOpcode::kCreateDnn:
            case ExecutionOpcode::kDestroyDnn:
            case ExecutionOpcode::kSetDnnStream:
            case ExecutionOpcode::kGetDnnStream: {
#ifdef GLIMMER_HAS_CUDNN
                const auto operation = message.header.operation;
                if (operation == ExecutionOpcode::kGetDnnStream) {
                    const auto activated = activate(slot);
                    if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                }
                CudaDnnResult library;
                if (operation == ExecutionOpcode::kCreateDnn) {
                    if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                    library = slot.session->create_dnn();
                } else {
                    if (!input.read_u64(id)) return rejection(CUDA_ERROR_INVALID_VALUE);
                    if (operation == ExecutionOpcode::kSetDnnStream) {
                        if (!input.read_u64(count) || !input.empty())
                            return rejection(CUDA_ERROR_INVALID_VALUE);
                        library = slot.session->set_dnn_stream(id, count);
                    } else {
                        if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                        library = operation == ExecutionOpcode::kDestroyDnn
                                      ? slot.session->destroy_dnn(id)
                                      : slot.session->get_dnn_stream(id);
                    }
                }
                if (library.disposition == CudaDnnDisposition::kDeferred) return std::nullopt;
                auto reply = project(library);
                reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
                return reply;
#else
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
            }
            case ExecutionOpcode::kFunctionControl: {
                std::uint32_t operation{}, argument{}, value{};
                if (!input.read_u32(operation) || !input.read_u64(id) ||
                    !input.read_u32(argument) || !input.read_u32(value) || !input.empty() ||
                    operation < static_cast<std::uint32_t>(CudaFunctionControl::kGetAttribute) ||
                    operation > static_cast<std::uint32_t>(CudaFunctionControl::kGetAttributes) ||
                    (operation != static_cast<std::uint32_t>(CudaFunctionControl::kSetAttribute) &&
                     value) ||
                    (operation == static_cast<std::uint32_t>(CudaFunctionControl::kGetAttributes) &&
                     argument))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                const bool is_aggregate =
                    operation == static_cast<std::uint32_t>(CudaFunctionControl::kGetAttributes);
                ExecutionPayloadWriter attributes_output(kExecutionFunctionAttributesBytes);
                if (is_aggregate &&
                    !attributes_output.reserve_capacity(kExecutionFunctionAttributesBytes))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                // Queries do not acquire GPU scheduling demand, so activate
                // their owner here. Mutations already hold the normal grant.
                if (is_aggregate ||
                    operation == static_cast<std::uint32_t>(CudaFunctionControl::kGetAttribute)) {
                    const auto activated = activate(slot);
                    if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                }
                const auto controlled =
                    slot.session->function_control(static_cast<CudaFunctionControl>(operation), id,
                                                   argument, std::bit_cast<std::int32_t>(value));
                if (controlled.disposition == CudaSessionDisposition::kDeferred)
                    return std::nullopt;
                auto reply = rejection(controlled.status);
                if (controlled.disposition == CudaSessionDisposition::kComplete)
                    reply.disposition = ExecutionWireDisposition::kComplete;
                if (controlled.original_status) {
                    reply.original_status = static_cast<std::int32_t>(*controlled.original_status);
                    if (controlled.status == CUDA_SUCCESS) reply.status = *reply.original_status;
                }
                reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
                if (operation == static_cast<std::uint32_t>(CudaFunctionControl::kGetAttribute) &&
                    reply.disposition == ExecutionWireDisposition::kComplete && !reply.status &&
                    reply.original_status == CUDA_SUCCESS && !reply.parameter_bytes)
                    reply.id = std::bit_cast<std::uint32_t>(controlled.value);
                if (is_aggregate && reply.disposition == ExecutionWireDisposition::kComplete &&
                    !reply.status && reply.original_status == CUDA_SUCCESS &&
                    !reply.parameter_bytes) {
                    if (!controlled.attributes ||
                        !write_execution_function_attributes(
                            attributes_output,
                            {controlled.attributes->sizes, controlled.attributes->values})) {
                        reply.disposition = ExecutionWireDisposition::kRejected;
                        reply.status = CUDA_ERROR_UNKNOWN;
                    } else {
                        reply.data = attributes_output.take();
                    }
                }
                return reply;
            }
            case ExecutionOpcode::kDriverContextControl:
            case ExecutionOpcode::kRuntimeContextControl: {
                std::uint32_t operation{}, argument{};
                if (!input.read_u32(operation) || !input.read_u32(argument) ||
                    !input.read_u64(count) || !input.empty() ||
                    operation < static_cast<std::uint32_t>(CudaContextControl::kGetLimit) ||
                    operation > static_cast<std::uint32_t>(CudaContextControl::kGetPriorityRange))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto activated = activate(slot);
                if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                const auto controlled = slot.session->context_control(
                    static_cast<CudaContextControl>(operation), argument, count,
                    message.header.operation == ExecutionOpcode::kRuntimeContextControl);
                if (controlled.disposition == CudaSessionDisposition::kDeferred)
                    return std::nullopt;
                return project(controlled);
            }
            case ExecutionOpcode::kLtHeuristic:
            case ExecutionOpcode::kLtAlgorithmCheck:
                return execute_lt_query(slot, request);
            case ExecutionOpcode::kCreateLt:
            case ExecutionOpcode::kDestroyLt: {
                CudaSessionBlasResult library;
                if (message.header.operation == ExecutionOpcode::kCreateLt) {
                    if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                    library = slot.session->create_lt();
                } else {
                    if (!input.read_u64(id) || !input.empty())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    library = slot.session->destroy_lt(id);
                }
                if (library.disposition == CudaSessionDisposition::kDeferred) return std::nullopt;
                auto reply = project(library);
                reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
                return reply;
            }
            case ExecutionOpcode::kCreateRand:
            case ExecutionOpcode::kSetRandSeed:
            case ExecutionOpcode::kDestroyRand: {
                CudaSessionRandResult library;
                const auto operation = message.header.operation;
                if (operation == ExecutionOpcode::kCreateRand) {
                    if (!input.read_u32(flags) || !input.empty())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    library = slot.session->create_random(flags);
                } else {
                    if (!input.read_u64(id)) return rejection(CUDA_ERROR_INVALID_VALUE);
                    if (operation == ExecutionOpcode::kSetRandSeed) {
                        if (!input.read_u64(count) || !input.empty())
                            return rejection(CUDA_ERROR_INVALID_VALUE);
                        library = slot.session->set_random_seed(id, count);
                    } else {
                        if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                        library = slot.session->destroy_random(id);
                    }
                }
                if (library.disposition == CudaSessionDisposition::kDeferred) return std::nullopt;
                return project(library);
            }
            case ExecutionOpcode::kCreateBlas:
            case ExecutionOpcode::kDestroyBlas:
            case ExecutionOpcode::kGetBlasAttribute:
            case ExecutionOpcode::kSetBlasAttribute: {
                CudaSessionBlasResult library;
                const auto operation = message.header.operation;
                if (operation == ExecutionOpcode::kCreateBlas) {
                    if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                    library = slot.session->create_blas();
                } else {
                    if (!input.read_u64(id)) return rejection(CUDA_ERROR_INVALID_VALUE);
                    if (operation == ExecutionOpcode::kDestroyBlas) {
                        if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                        library = slot.session->destroy_blas(id);
                    } else {
                        if (!input.read_u32(flags) ||
                            flags > static_cast<std::uint32_t>(CudaBlasAttribute::kSmCountTarget))
                            return rejection(CUDA_ERROR_INVALID_VALUE);
                        const auto attribute = static_cast<CudaBlasAttribute>(flags);
                        if (operation == ExecutionOpcode::kSetBlasAttribute) {
                            if (!input.read_u64(count) || !input.empty())
                                return rejection(CUDA_ERROR_INVALID_VALUE);
                            library = slot.session->set_blas_attribute(id, attribute, count);
                        } else {
                            if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                            library = slot.session->get_blas_attribute(id, attribute);
                        }
                    }
                }
                if (library.disposition == CudaSessionDisposition::kDeferred) return std::nullopt;
                if (operation == ExecutionOpcode::kDestroyBlas && completed_idle_cleanup(library))
                    priority_idle.note_completed_cleanup(slot.priority);
                return project(library);
            }
            case ExecutionOpcode::kSetBlasWorkspace: {
                std::uint64_t allocation{}, workspace_offset{};
                if (!input.read_u64(id) || !input.read_u64(allocation) ||
                    !input.read_u64(workspace_offset) || !input.read_u64(count) || !input.empty() ||
                    workspace_offset > SIZE_MAX || count > SIZE_MAX)
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto configured = slot.session->set_blas_workspace(
                    id, {allocation, static_cast<std::size_t>(workspace_offset)},
                    static_cast<std::size_t>(count));
                if (configured.disposition == CudaSessionDisposition::kDeferred)
                    return std::nullopt;
                return project(configured);
            }
            case ExecutionOpcode::kGetContextFlags:
            case ExecutionOpcode::kSetContextFlags: {
                if ((message.header.operation == ExecutionOpcode::kSetContextFlags &&
                     !input.read_u32(flags)) ||
                    !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = activate(slot);
                if (result.status != CUDA_SUCCESS) return rejection(result.status);
                result = message.header.operation == ExecutionOpcode::kGetContextFlags
                             ? slot.session->get_context_flags()
                             : slot.session->set_context_flags(flags);
                break;
            }
            case ExecutionOpcode::kCreateEvent:
                if (!input.read_u32(flags) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->create_event(flags);
                break;
            case ExecutionOpcode::kDestroyEvent:
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->destroy_event(id);
                break;
            case ExecutionOpcode::kRecordEvent:
            case ExecutionOpcode::kStreamWaitEvent: {
                if (!input.read_u64(id) || !input.read_u64(offset) || !input.read_u32(flags) ||
                    !input.empty() || !slot.has_command_space)
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (slot.low_epochs.size() == config.session_limits.submissions.max_commands)
                    return std::nullopt;
                const auto accepted = message.header.operation == ExecutionOpcode::kRecordEvent
                                          ? slot.session->record_event_with_frontier(
                                                slot.next_command, id, offset, flags)
                                          : slot.session->wait_event_with_frontier(
                                                slot.next_command, id, offset, flags);
                result = accepted.result;
                if (result.id) remember_command_frontier(request, accepted.frontier);
                account_event(slot, accepted, message.header.operation);
                break;
            }
            case ExecutionOpcode::kWaitForeignEvent: {
                if (!input.read_u64(id) || !input.read_u64(count) || !input.read_u64(offset) ||
                    !input.read_u32(flags) || !input.empty() || !slot.has_command_space)
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (slot.low_epochs.size() == config.session_limits.submissions.max_commands)
                    return std::nullopt;
                if (!request.foreign_event) {
                    retire_foreign_events();
                    if (foreign_events.size() == config.maximum_foreign_event_edges)
                        return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                    const auto source = std::ranges::find(slots, id, &Slot::connection);
                    if (source == slots.end() || source->peer != slot.peer || !source->session ||
                        source->host_owner || source->is_metadata_owner() || source->is_nccl ||
                        source->is_retiring || source->is_closing || source->is_bootstrap_pending)
                        return rejection(CUDA_ERROR_INVALID_HANDLE);
                    auto edge = std::make_shared<ForeignEventEdge>();
                    auto captured = source->session->capture_event(count);
                    if (captured.status != CUDA_SUCCESS) return rejection(captured.status);
                    edge->peer = slot.peer;
                    edge->source_id = source->connection;
                    edge->target_id = slot.connection;
                    edge->source = source->session_leases.acquire();
                    edge->target = slot.session_leases.acquire();
                    edge->generation = std::move(captured.event);
                    edge->source_frontier = std::move(captured.dependency);
                    foreign_events.push_back(edge);
                    request.foreign_event = std::move(edge);
                }
                const auto edge = request.foreign_event;
                const auto accepted = slot.session->wait_event_with_frontier(
                    slot.next_command, edge->generation, offset, flags);
                result = accepted.result;
                if (result.id) {
                    edge->target_sequence = result.id;
                    edge->target_frontier = accepted.frontier;
                    remember_command_frontier(request, accepted.frontier);
                    account_event(slot, accepted, message.header.operation, edge->source_id);
                    audit_ordering(slot, slot.low_epochs.back(), 4, result.status);
                    audit("foreign_event_accepted", slot, result.id, result.status);
                    request.foreign_event.reset();
                } else if (result.disposition != CudaSessionDisposition::kDeferred) {
                    request.foreign_event.reset();
                    std::erase(foreign_events, edge);
                }
                break;
            }
            case ExecutionOpcode::kQueryEvent:
            case ExecutionOpcode::kSynchronizeEvent:
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (const auto activated = activate(slot); activated.status != CUDA_SUCCESS)
                    return rejection(activated.status);
                if (!request.event_generation) {
                    auto captured = slot.session->capture_event(id);
                    if (captured.status != CUDA_SUCCESS) return rejection(captured.status);
                    request.event_generation = std::move(captured.event);
                }
                result = slot.session->query_event(
                    *request.event_generation,
                    message.header.operation == ExecutionOpcode::kSynchronizeEvent);
                if (result.status == CUDA_ERROR_NOT_READY &&
                    message.header.operation == ExecutionOpcode::kSynchronizeEvent)
                    return std::nullopt;
                break;
            case ExecutionOpcode::kEventElapsedTime:
                if (!input.read_u64(id) || !input.read_u64(offset) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (const auto activated = activate(slot); activated.status != CUDA_SUCCESS)
                    return rejection(activated.status);
                result = slot.session->event_elapsed_time(id, offset);
                break;
            case ExecutionOpcode::kLoadModuleJit: {
                std::span<const std::byte> encoded;
                if (!input.read_u32(flags) || !input.read_u64(count) || !input.read_blob(encoded) ||
                    !input.empty() ||
                    (flags != static_cast<std::uint32_t>(ExecutionModuleImageKind::kBinary) &&
                     flags != static_cast<std::uint32_t>(ExecutionModuleImageKind::kPtx)))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                auto options = decode_execution_jit_options(encoded);
                if (!options || options->empty() || !valid_execution_jit_input(*options))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                ModuleFileMapping mapping;
                const auto mapped = mapping.open(request.borrowed_descriptor(), count);
                if (mapped != CUDA_SUCCESS) return rejection(mapped);
                std::vector<CudaModuleJitOption> native_options;
                native_options.reserve(options->size());
                for (const auto& option : *options)
                    native_options.push_back({option.option, option.value, option.data});
                std::vector<CudaModuleJitOption> outputs;
                result = flags == static_cast<std::uint32_t>(ExecutionModuleImageKind::kBinary)
                             ? slot.session->load_binary_jit(
                                   {static_cast<const std::byte*>(mapping.pointer), mapping.bytes},
                                   native_options, &outputs)
                             : slot.session->load_ptx_jit(
                                   {static_cast<const char*>(mapping.pointer), mapping.bytes},
                                   native_options, &outputs);
                if (result.disposition == CudaSessionDisposition::kDeferred) {
                    if (result.original_status) drop(slot);
                    return std::nullopt;
                }
                auto reply = project(result);
                if (result.original_status && outputs.size() == options->size()) {
                    for (std::size_t index = 0; index < outputs.size(); ++index) {
                        (*options)[index].value = outputs[index].value;
                        std::ranges::copy(outputs[index].data, (*options)[index].data.begin());
                    }
                    if (auto data = encode_execution_jit_options(*options))
                        reply.data = std::move(*data);
                    else {
                        slot.session->close();
                        reply.id = 0;
                        reply.status = CUDA_ERROR_UNKNOWN;
                        reply.disposition = ExecutionWireDisposition::kRejected;
                    }
                }
                return reply;
            }
            case ExecutionOpcode::kLoadModuleFile: {
                if (!input.read_u32(flags) || !input.read_u64(count) || !input.empty() ||
                    (flags != static_cast<std::uint32_t>(ExecutionModuleImageKind::kBinary) &&
                     flags != static_cast<std::uint32_t>(ExecutionModuleImageKind::kPtx)))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                ModuleFileMapping mapping;
                const auto mapped = mapping.open(request.borrowed_descriptor(), count);
                if (mapped != CUDA_SUCCESS) return rejection(mapped);
                if (flags == static_cast<std::uint32_t>(ExecutionModuleImageKind::kBinary))
                    result = slot.session->load_binary(
                        {static_cast<const std::byte*>(mapping.pointer), mapping.bytes});
                else
                    result = slot.session->load_ptx(
                        {static_cast<const char*>(mapping.pointer), mapping.bytes});
                break;
            }
            case ExecutionOpcode::kLoadBinary:
                if (!input.read_blob(data) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->load_binary(data);
                break;
            case ExecutionOpcode::kLoadPtx:
                if (!input.read_string(text) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->load_ptx(text);
                break;
            case ExecutionOpcode::kGetFunction:
                if (!input.read_u64(id) || !input.read_string(text) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->get_function(id, text);
                break;
            case ExecutionOpcode::kGetGlobal:
                if (!input.read_u64(id) || !input.read_string(text) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->get_global(id, text);
                break;
            case ExecutionOpcode::kUnloadModule:
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->unload_module(id);
                break;
            case ExecutionOpcode::kAllocate:
                if (!input.read_u64(count) || !input.empty() ||
                    count > std::numeric_limits<std::size_t>::max())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->allocate(static_cast<std::size_t>(count));
                break;
            case ExecutionOpcode::kRelease:
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->release(id);
                if (completed_idle_cleanup(id, result))
                    priority_idle.note_completed_cleanup(slot.priority);
                break;
            case ExecutionOpcode::kUpload:
                if (!input.read_u64(id) || !input.read_u64(offset) ||
                    offset > std::numeric_limits<std::size_t>::max() || !input.read_blob(data) ||
                    !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->upload({id, static_cast<std::size_t>(offset)}, data);
                break;
            case ExecutionOpcode::kDownload: {
                if (!input.read_u64(id) || !input.read_u64(offset) || !input.read_u64(count) ||
                    !input.empty() || offset > std::numeric_limits<std::size_t>::max() ||
                    count > config.transport.max_payload_bytes -
                                std::min<std::size_t>(64, config.transport.max_payload_bytes))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                std::vector<std::byte> bytes(static_cast<std::size_t>(count));
                result = slot.session->download({id, static_cast<std::size_t>(offset)}, bytes);
                remember_synchronous_frontier(slot, request, result);
                if (result.disposition == CudaSessionDisposition::kDeferred) {
                    if (result.original_status) drop(slot);
                    return std::nullopt;
                }
                auto reply = project(result);
                if (result.status == CUDA_SUCCESS) reply.data = std::move(bytes);
                return reply;
            }
            case ExecutionOpcode::kCreatePriorityStream: {
                std::uint32_t priority{};
                if (!input.read_u32(flags) || !input.read_u32(priority) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->create_stream(flags, std::bit_cast<std::int32_t>(priority));
                break;
            }
            case ExecutionOpcode::kRuntimeMetadata: {
                const auto query = decode_runtime_metadata(input);
                if (!query) return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto activated = activate(slot);
                if (activated.status != CUDA_SUCCESS) return rejection(activated.status);
                return project(slot.session->runtime_metadata(*query),
                               slot.session->health_snapshot().is_quarantined);
            }
            case ExecutionOpcode::kCreateStream:
                if (!input.read_u32(flags) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->create_stream(flags);
                break;
            case ExecutionOpcode::kDestroyStream:
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->destroy_stream(id);
                break;
            case ExecutionOpcode::kStagedHostTransfer:
            case ExecutionOpcode::kHostTransfer:
            case ExecutionOpcode::kMemory:
            case ExecutionOpcode::kUnifiedCopy:
            case ExecutionOpcode::kSgemm:
            case ExecutionOpcode::kGeam:
            case ExecutionOpcode::kSgemv:
            case ExecutionOpcode::kSgemmEx:
            case ExecutionOpcode::kGemmEx:
            case ExecutionOpcode::kStridedGemmEx:
            case ExecutionOpcode::kStridedSgemm:
            case ExecutionOpcode::kLtMatmul:
            case ExecutionOpcode::kDnnCompute:
            case ExecutionOpcode::kDnnConvolutionForward:
            case ExecutionOpcode::kDnnConvolutionFind:
            case ExecutionOpcode::kDnnTensorForward:
            case ExecutionOpcode::kDnnReduction:
            case ExecutionOpcode::kDnnBatchNorm:
            case ExecutionOpcode::kDnnDropout:
            case ExecutionOpcode::kDnnRnnCompute:
            case ExecutionOpcode::kDnnLegacyCompute:
            case ExecutionOpcode::kDnnAttnCompute:
            case ExecutionOpcode::kDnnBackendExecute:
            case ExecutionOpcode::kDnnReorderFilterAndBias:
            case ExecutionOpcode::kGraphLaunch:
            case ExecutionOpcode::kLaunch: {
                CudaExecutionKernel kernel;
                ExecutionGraphLaunch graph;
                bool asynchronous_kernel{};
                bool is_nonblocking_kernel{};
                CudaSessionSgemm sgemm;
                CudaSessionGeam geam;
                CudaSessionSgemv sgemv;
                CudaSessionSgemmEx sgemm_ex;
                CudaSessionGemmEx gemm_ex;
                CudaSessionStridedGemmEx strided_gemm;
                CudaSessionLtMatmul lt_matmul;
#ifdef GLIMMER_HAS_CUDNN
                CudaSessionDnnCompute dnn_compute;
                CudaSessionDnnConvolution dnn_convolution;
                CudaSessionDnnReduction dnn_reduction;
                CudaSessionDnnBatchNorm dnn_batch_norm;
                CudaSessionDnnDropout dnn_dropout;
                CudaSessionDnnRnn dnn_rnn;
                CudaSessionDnnLegacyRnn dnn_legacy;
                CudaSessionDnnAttn dnn_attn;
                CudaSessionDnnBackendExecute dnn_backend;
                CudaSessionDnnReorder dnn_reorder;
                int dnn_requested{};
#endif
                CudaSessionMemory memory;
                ExecutionUnifiedCopy unified_wire;
                CudaSessionUvaCopy unified_copy;
                CudaHostMemoryView host_memory;
                CudaAllocationRange device;
                std::uint64_t host_connection{}, host_allocation{}, host_offset{}, device_offset{};
                std::uint32_t direction{};
                const bool is_host_transfer =
                    message.header.operation == ExecutionOpcode::kHostTransfer ||
                    message.header.operation == ExecutionOpcode::kStagedHostTransfer;
                if (is_dnn_computation(message.header.operation)) {
#ifdef GLIMMER_HAS_CUDNN
                    // The accepted command retains immutable native views. A
                    // later metadata close cannot revoke an already owned reply.
                    if (!request.receipt) {
                        if (message.header.operation == ExecutionOpcode::kDnnBackendExecute) {
                            const auto decoded = decode_dnn_backend_execute(
                                slot, input, dnn_backend, request.dnn_backend_metadata);
                            if (decoded.status != CUDA_SUCCESS) {
                                auto reply = rejection(decoded.status);
                                reply.parameter_bytes =
                                    rnn_fault_bits(slot, request.dnn_backend_metadata);
                                return reply;
                            }
                        } else if (message.header.operation == ExecutionOpcode::kDnnAttnCompute) {
                            const auto reserved =
                                decode_dnn_attn(slot, input, dnn_attn, request.dnn_attn_metadata);
                            if (reserved.disposition == CudaDnnDisposition::kDeferred)
                                return std::nullopt;
                            if (reserved.status != CUDA_SUCCESS) {
                                auto reply = rejection(reserved.status);
                                reply.parameter_bytes =
                                    rnn_fault_bits(slot, request.dnn_attn_metadata);
                                return reply;
                            }
                        } else if (message.header.operation == ExecutionOpcode::kDnnLegacyCompute) {
                            const auto decoded = decode_dnn_legacy_compute(
                                slot, input, dnn_legacy, request.dnn_rnn_metadata);
                            if (decoded != CUDA_SUCCESS) return rejection(decoded);
                        } else if (message.header.operation == ExecutionOpcode::kDnnRnnCompute) {
                            const auto reserved =
                                decode_dnn_rnn(slot, input, dnn_rnn, request.dnn_rnn_metadata);
                            if (reserved.disposition == CudaDnnDisposition::kDeferred)
                                return std::nullopt;
                            if (reserved.status != CUDA_SUCCESS) {
                                auto reply = project(reserved);
                                reply.parameter_bytes =
                                    rnn_fault_bits(slot, request.dnn_rnn_metadata);
                                return reply;
                            }
                        } else if (message.header.operation == ExecutionOpcode::kDnnDropout) {
                            const auto reserved = decode_dnn_dropout(slot, input, dnn_dropout,
                                                                     request.dnn_dropout_metadata);
                            if (reserved.disposition == CudaDnnDisposition::kDeferred)
                                return std::nullopt;
                            if (reserved.status != CUDA_SUCCESS) {
                                auto reply = project(reserved);
                                // The token is not reacquired on receipt polling.
                                // Native mutation uncertainty belongs to DNM1 too.
                                reply.parameter_bytes =
                                    dropout_metadata_quarantined(request.dnn_dropout_metadata) ? 1
                                                                                               : 0;
                                return reply;
                            }
                        } else {
                            const bool decoded =
                                message.header.operation == ExecutionOpcode::kDnnCompute
                                    ? decode_dnn_compute(slot, input, dnn_compute)
                                : message.header.operation == ExecutionOpcode::kDnnTensorForward
                                    ? decode_dnn_tensor_forward(slot, input, dnn_compute)
                                : message.header.operation == ExecutionOpcode::kDnnReduction
                                    ? decode_dnn_reduction(slot, input, dnn_reduction)
                                : message.header.operation == ExecutionOpcode::kDnnBatchNorm
                                    ? decode_dnn_batch_norm(slot, input, dnn_batch_norm)
                                : message.header.operation ==
                                        ExecutionOpcode::kDnnReorderFilterAndBias
                                    ? decode_dnn_reorder(slot, input, dnn_reorder)
                                : message.header.operation == ExecutionOpcode::kDnnConvolutionFind
                                    ? decode_dnn_find(slot, input, dnn_convolution, dnn_requested)
                                    : decode_dnn_convolution(slot, input, dnn_convolution);
                            if (!decoded) return rejection(CUDA_ERROR_INVALID_VALUE);
                        }
                    }
#else
                    return rejection(CUDA_ERROR_NOT_SUPPORTED);
#endif
                } else if (message.header.operation == ExecutionOpcode::kUnifiedCopy) {
                    if (!read_execution_unified_copy(input, unified_wire) ||
                        unified_wire.bytes > SIZE_MAX ||
                        unified_wire.destination.offset > SIZE_MAX ||
                        unified_wire.source.offset > SIZE_MAX)
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    unified_copy.bytes = static_cast<std::size_t>(unified_wire.bytes);
                    unified_copy.stream = unified_wire.stream;
                    unified_copy.is_asynchronous = unified_wire.is_asynchronous;
                    if (unified_wire.destination.kind != ExecutionCopyEndpointKind::kDevice &&
                        unified_wire.source.kind != ExecutionCopyEndpointKind::kDevice) {
                        const auto& destination = unified_wire.destination;
                        const auto& source = unified_wire.source;
                        if (!unified_wire.bytes ||
                            (destination.connection == source.connection &&
                             destination.allocation == source.allocation &&
                             destination.offset < source.offset + unified_wire.bytes &&
                             source.offset < destination.offset + unified_wire.bytes))
                            return rejection(CUDA_ERROR_NOT_SUPPORTED);
                    }
                } else if (message.header.operation == ExecutionOpcode::kGeam) {
                    if (!decode_geam(input, geam)) return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kSgemv) {
                    if (!decode_sgemv(input, sgemv)) return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kSgemmEx) {
                    if (!decode_sgemm_ex(input, sgemm_ex))
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kLtMatmul) {
                    if (!decode_lt_matmul(slot, input, lt_matmul))
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kStridedGemmEx ||
                           message.header.operation == ExecutionOpcode::kStridedSgemm) {
                    for (auto& stride : strided_gemm.strides) {
                        std::uint64_t bits{};
                        if (!input.read_u64(bits)) return rejection(CUDA_ERROR_INVALID_VALUE);
                        stride = std::bit_cast<std::int64_t>(bits);
                    }
                    std::uint32_t batches{};
                    if (!input.read_u32(batches) ||
                        batches > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    strided_gemm.batches = static_cast<int>(batches);
                    if (!decode_gemm_ex(input, strided_gemm.operation))
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kGemmEx) {
                    if (!decode_gemm_ex(input, gemm_ex)) return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kSgemm) {
                    if (!decode_sgemm(input, sgemm)) return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kGraphLaunch) {
                    if (request.continuation || !read_execution_graph_launch(input, graph) ||
                        !input.empty())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (message.header.operation == ExecutionOpcode::kLaunch) {
                    if (!decode_kernel_request(input, kernel, asynchronous_kernel) ||
                        (asynchronous_kernel && (request.continuation || exact_continuation)))
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                } else if (is_host_transfer) {
                    if (!input.read_u64(host_connection) || !input.read_u64(host_allocation) ||
                        !input.read_u64(host_offset) || !input.read_u64(device.allocation) ||
                        !input.read_u64(device_offset) || !input.read_u64(count) ||
                        !input.read_u64(memory.stream) || !input.read_u32(direction) ||
                        direction > 1 || !input.empty() ||
                        host_offset > std::numeric_limits<std::size_t>::max() ||
                        device_offset > std::numeric_limits<std::size_t>::max() ||
                        count > std::numeric_limits<std::size_t>::max())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    device.offset = static_cast<std::size_t>(device_offset);
                } else {
                    std::uint32_t kind{}, asynchronous{};
                    std::uint64_t destination_offset{}, source_offset{};
                    if (!input.read_u32(kind) ||
                        kind > static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32) ||
                        !input.read_u32(asynchronous) || asynchronous > 1 ||
                        !input.read_u64(memory.destination.allocation) ||
                        !input.read_u64(destination_offset) ||
                        !input.read_u64(memory.source.allocation) ||
                        !input.read_u64(source_offset) || !input.read_u64(count) ||
                        !input.read_u32(memory.value) || !input.read_u64(memory.stream) ||
                        !input.empty() ||
                        destination_offset > std::numeric_limits<std::size_t>::max() ||
                        source_offset > std::numeric_limits<std::size_t>::max() ||
                        count > std::numeric_limits<std::size_t>::max())
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    constexpr std::array kinds{CudaMemoryKind::kCopy, CudaMemoryKind::kFill8,
                                               CudaMemoryKind::kFill16, CudaMemoryKind::kFill32};
                    memory.kind = kinds[kind];
                    memory.is_asynchronous = asynchronous != 0;
                    memory.destination.offset = static_cast<std::size_t>(destination_offset);
                    memory.source.offset = static_cast<std::size_t>(source_offset);
                    memory.count = static_cast<std::size_t>(count);
                }
                if (!request.receipt) {
                    if (asynchronous_kernel && slot.policy_failure)
                        return rejection(*slot.policy_failure);
                    if (message.header.operation == ExecutionOpcode::kLaunch &&
                        pair_profile_failure)
                        return rejection(*pair_profile_failure);
                    if (!can_issue && !asynchronous_kernel) return std::nullopt;
                    if (asynchronous_kernel &&
                        (!slot.has_frontier_space || config.session_limits.max_objects <= 1))
                        return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                    if (asynchronous_kernel && !slot.has_command_space)
                        return rejection(CUDA_ERROR_INVALID_VALUE);
                    if (slot.low_epochs.size() == config.session_limits.submissions.max_commands) {
                        if (asynchronous_kernel) {
                            ExecutionWireReply busy;
                            busy.disposition = ExecutionWireDisposition::kAdmissionBusy;
                            return busy;
                        }
                        return std::nullopt;  // Reap settled epochs before accepting more work.
                    }
                    if (asynchronous_kernel &&
                        (slot.frontiers.size() >= config.session_limits.max_objects - 1 ||
                         slot.asynchronous_receipts.size() ==
                             config.session_limits.submissions.max_commands)) {
                        ExecutionWireReply busy;
                        busy.disposition = ExecutionWireDisposition::kAdmissionBusy;
                        return busy;
                    }
                    if (asynchronous_kernel) {
                        const auto mode = slot.session->stream_is_nonblocking(kernel.stream);
                        if (!mode) return rejection(CUDA_ERROR_INVALID_HANDLE);
                        is_nonblocking_kernel = *mode;
                    }
                    if (!slot.has_command_space) return rejection(CUDA_ERROR_INVALID_VALUE);
                    const auto reserved = reserve_request_frontier(slot, request);
                    if (reserved != CUDA_SUCCESS) return rejection(reserved);
                    if (message.header.operation == ExecutionOpcode::kLaunch) {
                        std::vector<std::shared_ptr<const void>> lifetimes;
                        for (auto& candidate : slots)
                            if (candidate.host_owner && !candidate.is_staging_owner &&
                                !candidate.is_retiring && !candidate.is_closing &&
                                candidate.peer == slot.peer) {
                                auto leases = candidate.host_owner->lifetimes();
                                lifetimes.insert(lifetimes.end(), leases.begin(), leases.end());
                            }
                        if (!lifetimes.empty())
                            kernel.lifetime =
                                std::make_shared<std::vector<std::shared_ptr<const void>>>(
                                    std::move(lifetimes));
                    } else if (message.header.operation == ExecutionOpcode::kUnifiedCopy) {
                        const auto resolve_endpoint = [&](const ExecutionCopyEndpoint& wire,
                                                          CudaSessionUvaEndpoint& endpoint) {
                            if (wire.kind == ExecutionCopyEndpointKind::kNull) return true;
                            if (wire.kind == ExecutionCopyEndpointKind::kDevice) {
                                endpoint.kind = CudaUvaEndpointKind::kDevice;
                                endpoint.device = {wire.allocation,
                                                   static_cast<std::size_t>(wire.offset)};
                                return true;
                            }
                            const auto owner =
                                std::ranges::find(slots, wire.connection, &Slot::connection);
                            const bool is_pageable =
                                wire.kind == ExecutionCopyEndpointKind::kPageable;
                            if (owner == slots.end() || !owner->host_owner ||
                                owner->peer != slot.peer || !owner->is_initialized ||
                                owner->is_bootstrap_pending || owner->is_retiring ||
                                owner->is_closing || owner->policy_failure ||
                                owner->is_pageable_owner != is_pageable ||
                                owner->is_staging_owner != is_pageable)
                                return false;
                            auto view = owner->host_owner->borrow(
                                {wire.allocation, static_cast<std::size_t>(wire.offset)},
                                unified_copy.bytes);
                            if (!view || view->is_pageable != is_pageable) return false;
                            endpoint.kind = CudaUvaEndpointKind::kHost;
                            endpoint.host = std::move(*view);
                            return true;
                        };
                        if (!resolve_endpoint(unified_wire.destination, unified_copy.destination) ||
                            !resolve_endpoint(unified_wire.source, unified_copy.source))
                            return rejection(CUDA_ERROR_INVALID_VALUE);
                    } else if (is_host_transfer) {
                        const bool is_empty = !count && !host_connection && !host_allocation &&
                                              !host_offset && !device.allocation && !device.offset;
                        if (!is_empty) {
                            auto owner_slot =
                                std::ranges::find(slots, host_connection, &Slot::connection);
                            if (owner_slot == slots.end() || !owner_slot->host_owner ||
                                owner_slot->peer != slot.peer || !owner_slot->is_initialized ||
                                owner_slot->is_retiring || owner_slot->is_closing ||
                                owner_slot->is_pageable_owner ||
                                owner_slot->is_staging_owner !=
                                    (message.header.operation ==
                                     ExecutionOpcode::kStagedHostTransfer))
                                return rejection(CUDA_ERROR_INVALID_VALUE);
                            auto view = owner_slot->host_owner->borrow(
                                {host_allocation, static_cast<std::size_t>(host_offset)},
                                static_cast<std::size_t>(count));
                            if (!view) return rejection(CUDA_ERROR_INVALID_VALUE);
                            host_memory = std::move(*view);
                        }
                    }
                    ExecutionAcceptance accepted;
                    bool requires_preparation{};
                    CUresult rejection_status = CUDA_ERROR_INVALID_VALUE;
#ifdef GLIMMER_HAS_CUDNN
                    if (is_dnn_computation(message.header.operation)) {
                        CudaDnnComputeAcceptance native;
                        if (message.header.operation == ExecutionOpcode::kDnnBackendExecute) {
                            auto* metadata =
                                backend_metadata_owner(slot, request.dnn_backend_metadata);
                            if (!metadata) return rejection(CUDA_ERROR_INVALID_HANDLE);
                            auto found = slot.session->accept_dnn_backend_execute(
                                slot.next_command, *metadata->dnn_metadata, slot.connection,
                                dnn_backend);
                            native = found.native;
                            if (native.acceptance.accepted())
                                request.dnn_backend = std::move(found.result);
                        } else if (message.header.operation == ExecutionOpcode::kDnnAttnCompute) {
                            auto found = slot.session->accept_dnn_attn(slot.next_command, dnn_attn);
                            native = found.native;
                            if (native.acceptance.accepted())
                                request.dnn_attn = std::move(found.result);
                        } else if (message.header.operation == ExecutionOpcode::kDnnLegacyCompute) {
                            auto* metadata = backend_metadata_owner(slot, request.dnn_rnn_metadata);
                            if (!metadata) return rejection(CUDA_ERROR_INVALID_HANDLE);
                            auto found = slot.session->accept_dnn_legacy_rnn(
                                slot.next_command, *metadata->dnn_metadata, dnn_legacy);
                            native = found.native;
                            if (native.acceptance.accepted())
                                request.dnn_rnn = std::move(found.result);
                        } else if (message.header.operation == ExecutionOpcode::kDnnRnnCompute) {
                            auto found = slot.session->accept_dnn_rnn(slot.next_command, dnn_rnn);
                            native = found.native;
                            if (native.acceptance.accepted())
                                request.dnn_rnn = std::move(found.result);
                        } else if (message.header.operation ==
                                   ExecutionOpcode::kDnnConvolutionFind) {
                            auto found = slot.session->accept_dnn_find(
                                slot.next_command, dnn_convolution, dnn_requested);
                            native = found.native;
                            if (native.acceptance.accepted())
                                request.dnn_find = std::move(found.result);
                        } else if (message.header.operation == ExecutionOpcode::kDnnReduction) {
                            native = slot.session->accept_dnn_reduction(slot.next_command,
                                                                        dnn_reduction);
                        } else if (message.header.operation == ExecutionOpcode::kDnnBatchNorm) {
                            native = slot.session->accept_dnn_batch_norm(slot.next_command,
                                                                         dnn_batch_norm);
                        } else if (message.header.operation == ExecutionOpcode::kDnnDropout) {
                            native =
                                slot.session->accept_dnn_dropout(slot.next_command, dnn_dropout);
                            if (native.acceptance.accepted())
                                request.dnn_dropout = std::move(dnn_dropout.mutation);
                        } else if (message.header.operation ==
                                   ExecutionOpcode::kDnnReorderFilterAndBias) {
                            native =
                                slot.session->accept_dnn_reorder(slot.next_command, dnn_reorder);
                        } else {
                            native =
                                (message.header.operation == ExecutionOpcode::kDnnCompute ||
                                 message.header.operation == ExecutionOpcode::kDnnTensorForward)
                                    ? slot.session->accept_dnn_compute(slot.next_command,
                                                                       dnn_compute)
                                    : slot.session->accept_dnn_convolution(slot.next_command,
                                                                           dnn_convolution);
                        }
                        accepted = native.acceptance;
                        rejection_status = native.rejection_status;
                        if (!accepted.accepted() &&
                            (message.header.operation == ExecutionOpcode::kDnnRnnCompute ||
                             message.header.operation == ExecutionOpcode::kDnnLegacyCompute ||
                             message.header.operation == ExecutionOpcode::kDnnAttnCompute ||
                             message.header.operation == ExecutionOpcode::kDnnBackendExecute)) {
                            if (accepted.status == ExecutionAcceptStatus::kFull)
                                return std::nullopt;
                            // Private safety observations are not provenance for
                            // the requested, not-issued computation.
                            const auto faults = rnn_fault_bits(
                                slot,
                                (message.header.operation == ExecutionOpcode::kDnnRnnCompute ||
                                 message.header.operation == ExecutionOpcode::kDnnLegacyCompute)
                                    ? request.dnn_rnn_metadata
                                : message.header.operation == ExecutionOpcode::kDnnBackendExecute
                                    ? request.dnn_backend_metadata
                                    : request.dnn_attn_metadata);
                            auto reply = rejection(faults ? CUDA_ERROR_UNKNOWN : rejection_status);
                            reply.parameter_bytes = faults;
                            return reply;
                        }
                        if (!accepted.accepted() && native.original_status) {
                            // An original query failure is not an issued GPU
                            // command. Contradictory successful metadata instead
                            // retains policy rejection and original provenance.
                            const bool is_quarantined =
                                slot.session->health_snapshot().is_quarantined;
                            auto reply = project(CudaDnnResult{
                                .disposition = is_quarantined ? CudaDnnDisposition::kRejected
                                                              : CudaDnnDisposition::kComplete,
                                .status = is_quarantined ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS,
                                .original_status = native.original_status});
                            reply.parameter_bytes = is_quarantined ? 1 : 0;
                            return reply;
                        }
                    } else
#endif
                        if (message.header.operation == ExecutionOpcode::kGraphLaunch) {
                        auto submitted = slot.session->accept_graph_launch(
                            slot.next_command, graph.executable, graph.stream);
                        accepted = submitted.acceptance;
                        request.graph_launch = std::move(submitted.operation);
                    } else
                        accepted =
                            message.header.operation == ExecutionOpcode::kLtMatmul
                                ? slot.session->accept_lt_matmul(slot.next_command, lt_matmul)
                            : message.header.operation == ExecutionOpcode::kStridedGemmEx
                                ? slot.session->accept_strided_gemm_ex(slot.next_command,
                                                                       strided_gemm)
                            : message.header.operation == ExecutionOpcode::kStridedSgemm
                                ? slot.session->accept_strided_sgemm(slot.next_command,
                                                                     strided_gemm)
                            : message.header.operation == ExecutionOpcode::kGemmEx
                                ? slot.session->accept_gemm_ex(slot.next_command, gemm_ex)
                            : message.header.operation == ExecutionOpcode::kGeam
                                ? slot.session->accept_geam(slot.next_command, geam)
                            : message.header.operation == ExecutionOpcode::kSgemv
                                ? slot.session->accept_sgemv(slot.next_command, sgemv)
                            : message.header.operation == ExecutionOpcode::kSgemmEx
                                ? slot.session->accept_sgemm_ex(slot.next_command, sgemm_ex)
                            : message.header.operation == ExecutionOpcode::kSgemm
                                ? slot.session->accept_sgemm(slot.next_command, sgemm)
                            : message.header.operation == ExecutionOpcode::kLaunch
                                ? (asynchronous_kernel
                                       ? (config.issue_depth_metrics &&
                                                  config.issue_depth_metrics->records_events()
                                              ? slot.session->accept_async_kernel(
                                                    slot.next_command, kernel,
                                                    &requires_preparation)
                                              : slot.session->accept_async_kernel(slot.next_command,
                                                                                  kernel))
                                       : slot.session->accept_kernel(slot.next_command, kernel))
                            : message.header.operation == ExecutionOpcode::kUnifiedCopy
                                ? slot.session->accept_uva_copy(slot.next_command, unified_copy)
                            : is_host_transfer
                                ? slot.session->accept_host_transfer(slot.next_command, device,
                                                                     host_memory, memory.stream,
                                                                     direction != 0)
                                : slot.session->accept_memory(slot.next_command, memory);
                    if (!accepted.accepted()) {
                        if (accepted.status == ExecutionAcceptStatus::kFull &&
                            asynchronous_kernel) {
                            ExecutionWireReply busy;
                            busy.disposition = ExecutionWireDisposition::kAdmissionBusy;
                            return busy;
                        }
                        if (accepted.status == ExecutionAcceptStatus::kFull && !asynchronous_kernel)
                            return std::nullopt;
                        if (asynchronous_kernel) {
                            switch (accepted.status) {
                                case ExecutionAcceptStatus::kFull:
                                case ExecutionAcceptStatus::kOutOfMemory:
                                    rejection_status = CUDA_ERROR_OUT_OF_MEMORY;
                                    break;
                                case ExecutionAcceptStatus::kClosed:
                                    rejection_status = CUDA_ERROR_CONTEXT_IS_DESTROYED;
                                    break;
                                case ExecutionAcceptStatus::kQuarantined:
                                    rejection_status = CUDA_ERROR_UNKNOWN;
                                    break;
                                default:
                                    break;
                            }
                        }
                        const bool has_fault_flag =
                            is_dnn_computation(message.header.operation) ||
                            is_original_blas_command(message.header.operation);
                        const bool is_quarantined =
                            has_fault_flag && slot.session->health_snapshot().is_quarantined;
                        if (is_original_blas_command(message.header.operation) ||
                            message.header.operation == ExecutionOpcode::kUnifiedCopy)
                            rejection_status =
                                admission_rejection_status(accepted.status, rejection_status);
                        auto reply =
                            rejection(is_quarantined ? CUDA_ERROR_UNKNOWN : rejection_status);
                        if (has_fault_flag) reply.parameter_bytes = is_quarantined ? 1 : 0;
                        return reply;
                    }
                    request.receipt = accepted.receipt;
                    if (config.issue_depth_metrics &&
                        issue_depth_compute_class(message.header.operation) ==
                            IssueDepthComputeClass::kIncluded) {
                        if (!slot.retirement_audit ||
                            !slot.session->set_retirement_observer(slot.retirement_audit))
                            config.issue_depth_metrics->invalidate(
                                IssueDepthMetrics::Failure::kTransition);
                        config.issue_depth_metrics->admit(
                            {slot.connection, slot.peer.pid, slot.peer.start_time_ticks},
                            accepted.receipt.sequence(),
                            static_cast<std::uint32_t>(message.header.operation),
                            requires_preparation);
                    }
                    if (!bind_request_frontier(slot, request)) return std::nullopt;
                    request.is_library_replacement =
                        execution_allows_compute_replacement(message.header.operation) &&
                        slot.session->persistent_progress(accepted.receipt.sequence()).has_value();
                    lifecycle(HostStageMetrics::Lifecycle::kAdmit, slot,
                              request.message.header.sequence, request.receipt->sequence(),
                              request.is_library_replacement ? 1 : 0);
                    if ((message.header.operation == ExecutionOpcode::kStagedHostTransfer &&
                         !direction) ||
                        (message.header.operation == ExecutionOpcode::kUnifiedCopy &&
                         unified_wire.destination.kind == ExecutionCopyEndpointKind::kPageable)) {
                        request.transfer_frontier = request.command_frontier;
                    }
                    if (config.should_audit_native) {
                        try {
                            if (audit_log)
                                (void)audit_log->info(
                                    "Native admission audit: session={}, sequence={}, "
                                    "accepted=true, "
                                    "operation={}",
                                    slot.connection, request.receipt->sequence(),
                                    static_cast<std::uint32_t>(message.header.operation));
                        } catch (...) {
                            // Admission metadata and physical ownership do not depend on logging.
                        }
                    }
                    if (config.sgemm_audit_target &&
                        message.header.operation == ExecutionOpcode::kSgemm &&
                        slot.connection == sgemm_audit_session &&
                        request.receipt->sequence() == config.sgemm_audit_target->sequence) {
                        if (sgemm_audit_captured || !audit_log ||
                            !audit_log->info(
                                "SGEMM diagnostic: v=1, event=descriptor, session={}, sequence={}, "
                                "transa={}, transb={}, m={}, n={}, k={}, lda={}, ldb={}, ldc={}, "
                                "device_scalars={}, replacement={}",
                                slot.connection, request.receipt->sequence(),
                                static_cast<unsigned int>(sgemm.transpose_a),
                                static_cast<unsigned int>(sgemm.transpose_b), sgemm.rows,
                                sgemm.columns, sgemm.depth, sgemm.leading_a, sgemm.leading_b,
                                sgemm.leading_c, sgemm.uses_device_scalars ? 1 : 0,
                                request.is_library_replacement ? 1 : 0))
                            semantic_lost(5);
                        else
                            sgemm_audit_captured = true;
                    }
                    if (slot.next_command == std::numeric_limits<std::uint64_t>::max())
                        slot.has_command_space = false;
                    else
                        ++slot.next_command;
                    if (asynchronous_kernel || slot.is_low ||
                        slot.session->persistent_progress(accepted.receipt.sequence())) {
                        slot.low_epochs.push_back({.sequence = accepted.receipt.sequence(),
                                                   .frontier = *request.command_frontier,
                                                   .progress = {},
                                                   .is_asynchronous = asynchronous_kernel,
                                                   .receipt = accepted.receipt});
                    }
                    if (asynchronous_kernel) {
                        const auto frontier_id = slot.next_frontier;
                        slot.frontiers.push_back(
                            {.id = frontier_id,
                             .token = *request.command_frontier,
                             .asynchronous_sequence = accepted.receipt.sequence(),
                             .stream = kernel.stream,
                             .is_nonblocking = is_nonblocking_kernel});
                        slot.asynchronous_receipts.push_back(
                            {accepted.receipt, *request.command_frontier});
                        if (frontier_id == std::numeric_limits<std::uint64_t>::max())
                            slot.has_frontier_space = false;
                        else
                            ++slot.next_frontier;
                        ExecutionWireReply reply;
                        reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
                        reply.status = CUDA_SUCCESS;
                        reply.id = accepted.receipt.sequence();
                        reply.address = frontier_id;
                        return reply;
                    }
                }
                // Pair selection must see this turn's accepted persistent
                // replacements before any of them obtains an ordinary grant.
                if (can_issue && !exact_continuation && !uses_priority_space_share() &&
                    !(slot.is_paired_member &&
                      (message.header.operation == ExecutionOpcode::kLaunch ||
                       request.is_library_replacement)) &&
                    !slot.session->host_result(*request.receipt)) {
                    bool is_resume{};
                    const auto issued = issue_scheduled_command(slot, is_resume);
                    if (issued.state == CudaExecutionIssueState::kError &&
                        issue_failure(issued) != CUDA_ERROR_NOT_READY)
                        (void)slot.session->reject_queued(issue_failure(issued));
                    note_scheduled_issue(slot, issued, is_resume);
                }
                const auto host = slot.session->host_result(*request.receipt);
                if (!host) return std::nullopt;  // Logical predecessor may still be incomplete.
                ExecutionWireReply reply;
                reply.status = host->status;
                if (is_dnn_computation(message.header.operation) ||
                    is_original_blas_command(message.header.operation))
                    reply.parameter_bytes = slot.session->health_snapshot().is_quarantined ? 1 : 0;
                if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                    reply.disposition = request.is_library_replacement
                                            ? ExecutionWireDisposition::kComputeReplacement
                                            : ExecutionWireDisposition::kComplete;
                    if (!request.is_library_replacement) reply.original_status = host->status;
                }
                if (message.header.operation == ExecutionOpcode::kGraphLaunch &&
                    request.graph_launch) {
                    if (const auto graph_result = request.graph_launch->reply())
                        reply = project(*graph_result);
                    else if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                        slot.session->close();
                        reply = project(CudaSessionGraphResult{.status = CUDA_ERROR_UNKNOWN,
                                                               .is_quarantined = true});
                    }
                }
#ifdef GLIMMER_HAS_CUDNN
                if (message.header.operation == ExecutionOpcode::kDnnBackendExecute) {
                    const auto faults = rnn_fault_bits(slot, request.dnn_backend_metadata);
                    reply.parameter_bytes = faults;
                    const auto& completed = request.dnn_backend;
                    if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                        if (!completed || !completed->is_ready ||
                            completed->result.original_status != host->status) {
                            slot.session->close();
                            reply = rejection(CUDA_ERROR_UNKNOWN);
                            reply.parameter_bytes = faults | 2U;
                        } else {
                            reply = project(completed->result);
                            reply.parameter_bytes =
                                faults |
                                (completed->has_context_loss || completed->is_cleanup_uncertain
                                     ? 2U
                                     : 0U);
                            if (reply.parameter_bytes &&
                                (reply.status == CUDNN_STATUS_SUCCESS ||
                                 reply.disposition != ExecutionWireDisposition::kRejected)) {
                                reply = rejection(CUDA_ERROR_UNKNOWN);
                                reply.original_status = host->status;
                                reply.parameter_bytes = faults | 2U;
                            }
                        }
                    } else if (completed && completed->is_ready) {
                        // Private pack preparation is not Execute provenance.
                        reply = rejection(completed->result.status != CUDA_SUCCESS
                                              ? completed->result.status
                                              : CUDA_ERROR_UNKNOWN);
                        reply.parameter_bytes =
                            faults |
                            (completed->has_context_loss || completed->is_cleanup_uncertain ? 2U
                                                                                            : 0U);
                    }
                }
                if (message.header.operation == ExecutionOpcode::kDnnAttnCompute) {
                    const auto faults = rnn_fault_bits(slot, request.dnn_attn_metadata);
                    reply.parameter_bytes = faults;
                    if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                        const auto& completed = request.dnn_attn;
                        if (!completed || !completed->is_ready ||
                            completed->result.original_status != host->status) {
                            slot.session->close();
                            reply = rejection(CUDA_ERROR_UNKNOWN);
                            reply.parameter_bytes = faults | 2U;
                        } else {
                            reply = project(completed->result);
                            reply.parameter_bytes =
                                faults | (completed->has_metadata_quarantine ? 1U : 0U) |
                                (completed->has_context_loss || completed->is_cleanup_uncertain
                                     ? 2U
                                     : 0U);
                            if (reply.status == CUDNN_STATUS_SUCCESS && reply.parameter_bytes) {
                                const auto all_faults = reply.parameter_bytes;
                                slot.session->close();
                                reply = rejection(CUDA_ERROR_UNKNOWN);
                                reply.original_status = host->status;
                                reply.parameter_bytes = all_faults | 2U;
                            }
                        }
                    }
                }
                if (message.header.operation == ExecutionOpcode::kDnnRnnCompute ||
                    message.header.operation == ExecutionOpcode::kDnnLegacyCompute) {
                    const auto faults = rnn_fault_bits(slot, request.dnn_rnn_metadata);
                    reply.parameter_bytes = faults;
                    if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                        const auto& completed = request.dnn_rnn;
                        if (!completed || !completed->is_ready ||
                            completed->result.original_status != host->status) {
                            slot.session->close();
                            reply = rejection(CUDA_ERROR_UNKNOWN);
                            // Inconsistent evidence is not authority to invent
                            // requested-native provenance from a queue scalar.
                            reply.parameter_bytes = faults | 2U;
                        } else {
                            reply = project(completed->result);
                            reply.parameter_bytes =
                                faults | (completed->has_metadata_quarantine ? 1U : 0U) |
                                (completed->has_context_loss || completed->is_cleanup_uncertain
                                     ? 2U
                                     : 0U);
                            if (reply.status == CUDNN_STATUS_SUCCESS && reply.parameter_bytes) {
                                const auto all_faults = reply.parameter_bytes;
                                slot.session->close();
                                reply = rejection(CUDA_ERROR_UNKNOWN);
                                reply.original_status = host->status;
                                reply.parameter_bytes = all_faults | 2U;
                            }
                        }
                    }
                }
                if (message.header.operation == ExecutionOpcode::kDnnDropout) {
                    if (host->origin == ExecutionReplyOrigin::kNativeSdk) {
                        const auto completed = request.dnn_dropout
                                                   ? request.dnn_dropout->completion_result()
                                                   : std::nullopt;
                        if (!completed || completed->original_status != host->status) {
                            slot.session->close();
                            reply = rejection(CUDA_ERROR_UNKNOWN);
                            reply.original_status = host->status;
                            reply.parameter_bytes = 1;
                        } else {
                            // Original success is not a substitute for the
                            // command's same-context physical completion proof.
                            reply = project(*completed);
                            reply.parameter_bytes =
                                slot.session->health_snapshot().is_quarantined ? 1 : 0;
                        }
                    }
                    if (dropout_metadata_quarantined(request.dnn_dropout_metadata))
                        reply.parameter_bytes = 1;
                }
                if (message.header.operation == ExecutionOpcode::kDnnConvolutionFind &&
                    host->origin == ExecutionReplyOrigin::kNativeSdk) {
                    // Only the accepted native command owns this result. Host
                    // receipt/ACK does not release its physical descriptor lease.
                    if (!request.dnn_find || !request.dnn_find->is_ready ||
                        request.dnn_find->query.result.original_status != host->status) {
                        slot.session->close();
                        reply = rejection(CUDA_ERROR_UNKNOWN);
                        reply.original_status = host->status;
                        reply.parameter_bytes = 1;
                    } else {
                        reply = project_dnn_convolution_result(slot, request.dnn_find->query, true);
                    }
                }
#endif
                // Graph's typed receipt carries no object/command capability.
                // Its transport header already binds this exact replay request.
                if (message.header.operation != ExecutionOpcode::kGraphLaunch)
                    reply.id = request.receipt->sequence();
                if (request.transfer_frontier && host->status == CUDA_SUCCESS) {
                    const auto completed = slot.session->query_frontier(*request.transfer_frontier);
                    if (completed == CUDA_ERROR_NOT_READY) return std::nullopt;
                    reply.status = completed;
                }
                return reply;
            }
            case ExecutionOpcode::kImportHostMemory:
            case ExecutionOpcode::kReleaseHostMemory:
            case ExecutionOpcode::kQueryStagingIdle:
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
            case ExecutionOpcode::kDrain: {
                if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto snapshot = slot.session->scheduling_snapshot();
                if (snapshot.is_quarantined) {
                    auto reply = rejection(slot.asynchronous_failure.value_or(CUDA_ERROR_UNKNOWN));
                    reply.disposition = ExecutionWireDisposition::kComplete;
                    return reply;
                }
                if (snapshot.submissions.outstanding || has_private_boundaries(snapshot))
                    return std::nullopt;
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                return reply;
            }
            case ExecutionOpcode::kCaptureFrontier: {
                if (!input.read_u32(flags) || flags > 2 || !input.read_u64(id) || !input.empty() ||
                    (!flags && id))
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                if (!slot.has_frontier_space ||
                    slot.frontiers.size() == config.session_limits.max_objects)
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                std::vector<CudaExecutionFrontier> asynchronous_prefix;
                asynchronous_prefix.reserve(slot.frontiers.size());
                const auto stream_mode =
                    flags ? slot.session->stream_is_nonblocking(id) : std::optional{false};
                if (!stream_mode) return rejection(CUDA_ERROR_INVALID_HANDLE);
                for (const auto& prior : slot.frontiers)
                    if (prior.asynchronous_sequence &&
                        (!flags || prior.stream == id ||
                         (flags == 2 && ((id == 0 && !prior.is_nonblocking) ||
                                         (prior.stream == 0 && !*stream_mode)))))
                        asynchronous_prefix.push_back(prior.token);
                auto captured = slot.session->capture_frontier(
                    flags ? std::optional{id} : std::nullopt, flags == 2);
                if (captured.status != CUDA_SUCCESS)
                    return rejection(slot.asynchronous_failure.value_or(captured.status));
                const auto assigned = slot.next_frontier;
                slot.frontiers.push_back({.id = assigned,
                                          .token = std::move(captured.frontier),
                                          .asynchronous_prefix = std::move(asynchronous_prefix)});
                if (assigned == std::numeric_limits<std::uint64_t>::max())
                    slot.has_frontier_space = false;
                else
                    ++slot.next_frontier;
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.id = assigned;
                reply.address = slot.has_command_space ? slot.next_command - 1
                                                       : std::numeric_limits<std::uint64_t>::max();
                return reply;
            }
            case ExecutionOpcode::kQueryAsyncFrontiers:
            case ExecutionOpcode::kReleaseAsyncFrontiers:
                return execute_async_frontier_batch(slot, request, exact_continuation);
            case ExecutionOpcode::kQueryFrontier:
            case ExecutionOpcode::kReleaseFrontier: {
                if (!input.read_u64(id) || !input.empty())
                    return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto found = std::ranges::find_if(
                    slot.frontiers, [id](const auto& item) { return item.id == id; });
                if (found == slot.frontiers.end()) return rejection(CUDA_ERROR_INVALID_HANDLE);
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                if (message.header.operation == ExecutionOpcode::kQueryFrontier) {
                    const auto observed = observe_frontier(slot, *found);
                    reply.status = observed.status;
                    reply.parameter_bytes = observed.is_retired ? 1 : 0;
                } else
                    slot.frontiers.erase(found);
                return reply;
            }
            case ExecutionOpcode::kSnapshot: {
                if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                const auto snapshot = slot.session->snapshot();
                ExecutionPayloadWriter writer;
                if (!writer.write_u64(snapshot.submissions.outstanding) ||
                    !writer.write_u64(snapshot.submissions.awaiting_physical_stop) ||
                    !writer.write_u64(snapshot.submissions.retained_parameter_bytes) ||
                    !writer.write_u64(snapshot.device_bytes) ||
                    !writer.write_u64(snapshot.modules) || !writer.write_u64(snapshot.streams) ||
                    !writer.write_u64(snapshot.functions) ||
                    !writer.write_u32(snapshot.is_quarantined ? 1 : 0))
                    return rejection(CUDA_ERROR_OUT_OF_MEMORY);
                auto reply = rejection(CUDA_SUCCESS);
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.data = writer.take();
                return reply;
            }
            case ExecutionOpcode::kClose:
                if (!input.empty()) return rejection(CUDA_ERROR_INVALID_VALUE);
                result = slot.session->shutdown();
                break;
            case ExecutionOpcode::kHello:
            case ExecutionOpcode::kSourceContinuation:
                return rejection(CUDA_ERROR_INVALID_VALUE);
            case ExecutionOpcode::kCreateLtMetadata:
            case ExecutionOpcode::kPointerMetadata:
            case ExecutionOpcode::kDestroyLtMetadata:
            case ExecutionOpcode::kSetLtAttribute:
            case ExecutionOpcode::kGetLtAttribute:
            case ExecutionOpcode::kLtVersion:
            case ExecutionOpcode::kLtAlgorithmCapability:
            case ExecutionOpcode::kDnnLibraryInfo:
            case ExecutionOpcode::kDnnLegacyMetadata:
            case ExecutionOpcode::kDnnBackendCreate:
            case ExecutionOpcode::kDnnBackendDestroy:
            case ExecutionOpcode::kDnnBackendInitialize:
            case ExecutionOpcode::kDnnBackendFinalize:
            case ExecutionOpcode::kDnnBackendPrepare:
            case ExecutionOpcode::kDnnBackendSet:
            case ExecutionOpcode::kDnnBackendGet:
            case ExecutionOpcode::kCreateDnnMetadata:
            case ExecutionOpcode::kDestroyDnnMetadata:
            case ExecutionOpcode::kSetDnnTensor:
            case ExecutionOpcode::kGetDnnTensor:
            case ExecutionOpcode::kDnnTensorSize:
            case ExecutionOpcode::kSetDnnActivation:
            case ExecutionOpcode::kGetDnnActivation:
            case ExecutionOpcode::kSetDnnFilter:
            case ExecutionOpcode::kGetDnnFilter:
            case ExecutionOpcode::kDnnFilterSize:
            case ExecutionOpcode::kSetDnnConvolution:
            case ExecutionOpcode::kGetDnnConvolution:
            case ExecutionOpcode::kSetDnnConvolutionAttribute:
            case ExecutionOpcode::kGetDnnConvolutionAttribute:
            case ExecutionOpcode::kDnnConvolutionOutput:
            case ExecutionOpcode::kSetDnnPooling:
            case ExecutionOpcode::kGetDnnPooling:
            case ExecutionOpcode::kDnnPoolingOutput:
            case ExecutionOpcode::kSetDnnLrn:
            case ExecutionOpcode::kSetDnnReduction:
            case ExecutionOpcode::kSetDnnOpTensor:
            case ExecutionOpcode::kGetDnnOpTensor:
            case ExecutionOpcode::kSetDnnTensorTransform:
            case ExecutionOpcode::kDnnInitTransformDest:
            case ExecutionOpcode::kSetDnnRnnData:
            case ExecutionOpcode::kSetDnnSeqData:
            case ExecutionOpcode::kGetDnnSeqData:
            case ExecutionOpcode::kSetDnnRnn:
            case ExecutionOpcode::kSetDnnAttn:
            case ExecutionOpcode::kSetDnnRnnClip:
                return rejection(CUDA_ERROR_NOT_SUPPORTED);
        }
        remember_synchronous_frontier(slot, request, result);
        if (result.disposition == CudaSessionDisposition::kDeferred) {
            if (result.original_status) {
                drop(slot);
            }
            return std::nullopt;
        }
        auto reply = project(result);
        if (message.header.operation == ExecutionOpcode::kRelease &&
            execution_is_process_allocation(id) && slot.process_memory)
            reply.parameter_bytes = slot.process_memory->owner()->snapshot().is_quarantined ? 1 : 0;
        return reply;
    }
#ifdef GLIMMER_HAS_NCCL
    NcclActorContinuationOutcome continuation_outcome(Slot& slot, Slot::Request& request,
                                                      std::optional<ExecutionWireReply> reply) {
        NcclActorContinuationOutcome outcome;
        const auto state = slot.snapshot();
        const bool uncertain = state.is_quarantined || slot.policy_failure;
        if (uncertain) {
            outcome.kind = NcclActorContinuationOutcomeKind::kUncertain;
            outcome.status = slot.policy_failure.value_or(CUDA_ERROR_UNKNOWN);
        } else if (reply) {
            outcome.kind = NcclActorContinuationOutcomeKind::kReturned;
            outcome.status = CUDA_SUCCESS;
        } else if (request.receipt) {
            outcome.kind = NcclActorContinuationOutcomeKind::kCommandAccepted;
            outcome.status = CUDA_SUCCESS;
        } else if (request.has_host_job || request.has_host_shutdown) {
            outcome.kind = NcclActorContinuationOutcomeKind::kHostJobPending;
            outcome.status = CUDA_SUCCESS;
            outcome.pending_lifetime = slot.host_owner_leases.acquire();
            outcome.pending_sequence = request.message.header.sequence;
        } else {
            outcome.kind = NcclActorContinuationOutcomeKind::kDeferred;
            outcome.status = CUDA_ERROR_NOT_READY;
        }
        if (request.command_frontier)
            outcome.binding = NcclActorContinuationBinding{
                slot.connection, slot.session_leases.acquire(), *request.command_frontier,
                request.receipt, request.command_capture};
        if (reply && reply->status == CUDA_SUCCESS &&
            reply->disposition == ExecutionWireDisposition::kComplete &&
            request.message.header.operation == ExecutionOpcode::kImportHostMemory)
            outcome.imported_allocation = reply->id;
        outcome.reply = std::move(reply);
        return outcome;
    }
    void poll_continuation_result(Slot& slot, Slot::Request& request) {
        const auto observed = nccl_actor->continuation_reply(request.ordinary_continuation);
        if (observed.ready || (request.receipt && !slot.session->host_result(*request.receipt)) ||
            (!request.receipt && !request.has_host_job && !request.has_host_shutdown))
            return;
        // These branches only read an already accepted queue/mailbox result.
        // In particular can_issue=false cannot submit another helper job or
        // issue/resume ordinary source computation.
        auto reply = execute(slot, request, false, true);
        if (!reply && (request.receipt || request.has_host_job || request.has_host_shutdown))
            return;
        auto outcome = continuation_outcome(slot, request, std::move(reply));
        const bool keeps_fence =
            outcome.kind == NcclActorContinuationOutcomeKind::kDeferred ||
            outcome.kind == NcclActorContinuationOutcomeKind::kHostJobPending ||
            outcome.kind == NcclActorContinuationOutcomeKind::kUncertain;
        const auto status =
            nccl_actor->complete_continuation(request.ordinary_continuation, std::move(outcome));
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) nccl_failure = status;
        if (status == CUDA_SUCCESS && !keeps_fence)
            release_continuation_fences(request.ordinary_continuation);
    }
    bool observe_continuation(Slot& slot, Slot::Request& request) {
        prepare_continuation(slot, request);
        if (request.continuation_rejection)
            return queue_reply(slot, request, *request.continuation_rejection);
        if (!request.ordinary_continuation) return false;
        if (!request.has_continuation_dependencies) {
            const auto status = bind_continuation_dependencies(slot, request);
            if (status == CUDA_ERROR_NOT_READY) return false;
            if (status != CUDA_SUCCESS) {
                const auto cancelled =
                    nccl_actor->cancel_continuation(request.ordinary_continuation);
                if (cancelled != CUDA_SUCCESS) {
                    nccl_failure = cancelled;
                    return false;
                }
                release_continuation_fences(request.ordinary_continuation);
                request.ordinary_continuation = 0;
                reject_continuation(slot, request, status);
                return queue_reply(slot, request, *request.continuation_rejection);
            }
        }
        poll_continuation_result(slot, request);
        auto observed = nccl_actor->continuation_reply(request.ordinary_continuation);
        return observed.ready && observed.reply && queue_reply(slot, request, *observed.reply);
    }
#endif
    void observe() noexcept {
        const HostScope host_scope(config.host_stage_metrics.get(), HostStage::kObserve);
        if (graph_construction) return;
        retire_foreign_events();
        for (auto& slot : slots) {
            if (!transport.has_peer(slot.connection)) retire(slot);
            if (slot.host_owner || slot.is_metadata_owner() || slot.is_nccl) continue;
            harvest_asynchronous_results(slot);
            auto snapshot = slot.session->scheduling_snapshot();
            // Idle/quarantined factories have no event to observe. In particular,
            // do not reactivate a once-failed destroy and drop its pending error
            // reply before the transport has delivered it to the caller.
            if (snapshot.has_context &&
                (has_physical_reservation(snapshot) || has_private_boundaries(snapshot) ||
                 !slot.low_epochs.empty())) {
                const auto activated = activate(slot);
                if (activated.status != CUDA_SUCCESS) {
                    if (slot.is_paired_member && activated.status == CUDA_ERROR_NOT_READY &&
                        !activated.original_status)
                        continue;
                    if (has_scheduling_policy()) policy_failed(slot, activated.status);
                    drop(slot);
                    continue;
                }
                for (auto& epoch : slot.low_epochs)
                    if (const auto progress = slot.session->persistent_progress(epoch.sequence))
                        epoch.progress = *progress;
                if (snapshot.submissions.awaiting_physical_stop ||
                    has_private_boundaries(snapshot) ||
                    std::ranges::any_of(slot.low_epochs,
                                        [](const auto& epoch) { return epoch.is_event; })) {
                    // Poll only: active turns reject observation and unrecorded
                    // private events are skipped by the dispatcher itself.
                    const auto observed = slot.session->poll();
                    if (observed.status != CUDA_SUCCESS &&
                        observed.status != CUDA_ERROR_NOT_READY && !slot.asynchronous_failure)
                        slot.asynchronous_failure = observed.status;
                    if (config.should_audit_native && audit_log) {
                        if (const auto pair = slot.session->native_kernel_pair();
                            pair && *pair != slot.last_native_pair_audit) {
                            slot.last_native_pair_audit = *pair;
                            (void)audit_log->info(
                                "Native kernel pipeline audit: session={}, first={}, second={}, "
                                "both_pending=true",
                                slot.connection, (*pair)[0], (*pair)[1]);
                        }
                    }
                }
                const auto after = slot.session->scheduling_snapshot();
                if (!slot.is_low && !slot.asynchronous_failure && !after.is_quarantined &&
                    has_physical_reservation(snapshot) && !has_physical_reservation(after))
                    audit("high_physical_done", slot, 0);
                for (std::size_t index = 0; index < slot.low_epochs.size();) {
                    auto& epoch = slot.low_epochs[index];
                    if (epoch.is_event) {
                        const auto retired = epoch.frontier.retirement();
                        if (retired.is_retired) {
                            if (retired.status != CUDA_SUCCESS && !slot.asynchronous_failure)
                                slot.asynchronous_failure = retired.status;
                            audit("event_retired", slot, epoch.sequence, retired.status);
                            audit_ordering(slot, epoch, 3, retired.status);
                            slot.low_epochs.erase(slot.low_epochs.begin() +
                                                  static_cast<std::ptrdiff_t>(index));
                            continue;
                        }
                        if (retired.status != CUDA_SUCCESS &&
                            retired.status != CUDA_ERROR_NOT_READY) {
                            if (!slot.session->owns_frontier(epoch.frontier) ||
                                after.is_quarantined)
                                policy_failed(slot, retired.status);
                            else if (!slot.asynchronous_failure)
                                slot.asynchronous_failure = retired.status;
                        }
                        ++index;
                        continue;
                    }
                    if (const auto progress = slot.session->persistent_progress(epoch.sequence))
                        epoch.progress = *progress;
                    remember_epoch_reply(slot, epoch.receipt);
                    if (epoch.host_reply &&
                        (epoch.host_reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue ||
                         (epoch.is_asynchronous &&
                          epoch.host_reply->origin == ExecutionReplyOrigin::kNativeSdk)) &&
                        epoch.host_reply->status != CUDA_SUCCESS) {
                        // A known initial launch/preparation failure stays in
                        // its exact dependency observation. It is not policy
                        // corruption, and its host result alone is not a stop.
                        const auto retired = epoch.frontier.retirement();
                        if (after.is_quarantined || slot.policy_failure) {
                            policy_failed(slot, retired.status == CUDA_SUCCESS ||
                                                        retired.status == CUDA_ERROR_NOT_READY
                                                    ? CUDA_ERROR_UNKNOWN
                                                    : retired.status);
                            ++index;
                            continue;
                        }
                        const bool matches_preparation = retired.status == CUDA_SUCCESS ||
                                                         retired.status == epoch.host_reply->status;
                        if (retired.is_retired && matches_preparation) {
                            slot.low_epochs.erase(slot.low_epochs.begin() +
                                                  static_cast<std::ptrdiff_t>(index));
                            continue;
                        }
                        if (!retired.is_retired &&
                            (matches_preparation || retired.status == CUDA_ERROR_NOT_READY)) {
                            ++index;
                            continue;  // Host rejection alone cannot release a private tail.
                        }
                        policy_failed(slot, retired.status);
                        ++index;
                        continue;
                    }
                    const auto& progress = epoch.progress;
                    if (!after.is_quarantined && !slot.policy_failure && epoch.was_successful &&
                        !epoch.is_paused && progress.physical_launches &&
                        progress.physical_stops == progress.physical_launches &&
                        progress.completed_blocks < progress.total_blocks) {
                        epoch.is_paused = true;
                        if (slot.is_low) audit("low_physical_stop", slot, epoch.sequence);
                    }
                    const auto completed = slot.session->query_frontier(epoch.frontier);
                    if (completed == CUDA_SUCCESS) {
                        if (epoch.was_successful && !slot.asynchronous_failure)
                            epoch.progress.completed_blocks = epoch.progress.total_blocks;
                        if (slot.is_low && epoch.is_issued)
                            audit("low_logical_done", slot, epoch.sequence);
                        slot.low_epochs.erase(slot.low_epochs.begin() +
                                              static_cast<std::ptrdiff_t>(index));
                        continue;
                    } else if (completed != CUDA_ERROR_NOT_READY) {
                        policy_failed(slot, completed);
                    }
                    ++index;
                }
            }
        }
        retire_foreign_events();
    }

    static core::ExecutionPriorityCommand command_description(
        const CudaSessionPendingDescription& pending) noexcept {
        return {.sequence = pending.command.sequence,
                .acceptance_order = pending.command.acceptance_order,
                .physical_generation = pending.command.physical_generation,
                .geometry_epoch = pending.command.geometry_epoch,
                .command_epoch = pending.command.command_epoch,
                .control_epoch = pending.control_epoch,
                .member_generation = pending.member_generation,
                .configuration_epoch = pending.configuration_epoch,
                .physical_configuration_epoch = pending.physical_configuration_epoch,
                .is_resume = pending.command.is_resume};
    }

    struct PriorityObservations {
        std::array<core::ExecutionPriorityPeer, core::kMaxExecutionPriorityPeers> peers{};
        std::size_t size{};
        bool is_valid{};
    };
    PriorityObservations priority_observations(std::uint64_t plan_serial = 0) noexcept {
        const HostScope host_scope(config.host_stage_metrics.get(), HostStage::kObservations);
        PriorityObservations result;
        auto& peers = result.peers;
        if (slots.size() > peers.size()) return {};
        auto& peer_count = result.size;
        peer_count = slots.size();
        for (std::size_t index = 0; index < slots.size(); ++index) {
            const auto& slot = slots[index];
            if (slot.is_metadata_owner()) {
                // CPU-only metadata never creates GPU demand or native stop authority.
                peers[index] = {slot.policy_id, slot.priority, false, false, false, false};
                continue;
            }
            if (slot.is_nccl) {
                const bool demand = !slot.is_retiring &&
                                    std::ranges::any_of(slot.requests, [](const auto& request) {
                                        return !request.nccl_request;
                                    });
                peers[index] = {slot.policy_id, slot.priority, demand,
                                false,          false,         slot.policy_failure.has_value()};
                continue;
            }
            // Observe reservation and quarantine atomically. Mixing snapshots
            // around a worker completion could miss both an active call and
            // its newly discovered uncertainty, incorrectly granting another peer.
            HostOwnerSnapshot host_state;
            CudaSessionSchedulingSnapshot state;
            {
                const HostScope snapshot_scope(config.host_stage_metrics.get(),
                                               HostStage::kSnapshot);
                host_state = slot.host_owner ? slot.host_owner->snapshot() : HostOwnerSnapshot{};
                state = slot.host_owner
                            ? static_cast<CudaSessionSchedulingSnapshot>(host_state.session)
                            : slot.session->scheduling_snapshot();
            }
            if (config.host_stage_metrics)
                config.host_stage_metrics->queue(
                    {slot.connection, slot.peer.pid, slot.peer.start_time_ticks},
                    {.requests = slot.requests.size(),
                     .queued = state.submissions.queued,
                     .issuing = state.submissions.issuing,
                     .paused = state.submissions.paused,
                     .awaiting_stop = state.submissions.awaiting_physical_stop,
                     .ready = state.has_ready_work,
                     .physical = has_physical_reservation(state) || host_state.has_reservation,
                     .low_epochs = slot.low_epochs.size(),
                     .preparation = state.has_pending_preparation});
            const bool has_preparation = !slot.host_owner && state.has_context &&
                                         !slot.is_retiring && state.has_pending_preparation;
            const bool needs_cleanup =
                slot.is_retiring &&
                (slot.host_owner ? !host_state.is_closed
                                 : state.has_context && !state.submissions.outstanding &&
                                       !slot.session_leases.has_leases());
            const bool callback_demand =
                std::ranges::any_of(slot.host_functions, [](const auto& callback) {
                    return callback.native && !callback.is_enqueued && !callback.is_uncertain;
                });
            const bool gpu_request =
                callback_demand || std::ranges::any_of(slot.requests, [&](const auto& request) {
                    return !request.continuation && !is_metadata(request.message) &&
                           !(is_execution_command(request.message.header.operation) &&
                             request.receipt);
                });
            peers[index] = {slot.policy_id,
                            slot.priority,
                            gpu_request || state.has_ready_work || has_preparation || needs_cleanup,
                            has_physical_reservation(state) || host_state.has_reservation,
                            state.submissions.paused != 0,
                            state.is_quarantined || slot.policy_failure.has_value()};
            if (!slot.host_owner && state.has_context) {
                const auto pending = [&] {
                    const HostScope pending_scope(config.host_stage_metrics.get(),
                                                  HostStage::kPending);
                    return slot.session->pending_command();
                }();
                if (pending.status == CUDA_SUCCESS && pending.command)
                    peers[index].command = command_description(pending.command.description());
                else if (pending.status != CUDA_SUCCESS && pending.status != CUDA_ERROR_NOT_READY)
                    peers[index].has_unknown_state = true;
                if (config.issue_depth_metrics) {
                    if (state.submissions.paused)
                        config.issue_depth_metrics->invalidate(
                            IssueDepthMetrics::Failure::kUnsupported);
                    config.issue_depth_metrics->observe(
                        {slot.connection, slot.peer.pid, slot.peer.start_time_ticks},
                        pending.command ? pending.command.description().command.sequence : 0,
                        state.has_ready_work, peers[index].has_unknown_state, plan_serial);
                }
                if (config.paired && config.paired->should_rebalance_on_arrival &&
                    slot.is_paired_member) {
                    for (const auto& epoch : slot.low_epochs) {
                        if (epoch.is_event || !epoch.is_issued || !epoch.was_successful ||
                            epoch.is_paused)
                            continue;
                        const auto progress = slot.session->persistent_progress(epoch.sequence);
                        if (!progress || !progress->workers ||
                            progress->physical_launches <= progress->physical_stops ||
                            (progress->kind != transform::PtxExecutionKind::kPreemptive &&
                             progress->kind != transform::PtxExecutionKind::kSliced))
                            continue;
                        if (peers[index].physical) {
                            // No single epoch can represent several physical owners.
                            peers[index].physical.reset();
                            break;
                        }
                        peers[index].physical = core::ExecutionPriorityPhysical{
                            epoch.sequence, progress->physical_launches};
                    }
                }
            }
        }
#ifdef GLIMMER_HAS_NCCL
        if (nccl_actor) {
            const auto snapshot = nccl_state();
            if (snapshot.status != CUDA_SUCCESS) return {};
            for (const auto& reservation : snapshot.reservations) {
                if (peer_count == peers.size()) return {};
                const auto policy = std::ranges::find(native_policies, reservation.reservation,
                                                      &NativePolicy::reservation);
                if (policy == native_policies.end() || policy->generation != reservation.generation)
                    return {};
                const auto ready = native_progress_grant(reservation);
                auto& peer = peers[peer_count++];
                peer = {policy->identity,
                        reservation.priority,
                        ready.has_value(),
                        reservation.has_physical_work,
                        false,
                        reservation.has_unknown_state};
                const auto owner =
                    std::ranges::find(native_peers, reservation.peer, &NativePeer::id);
                if (owner == native_peers.end() ||
                    (owner->disconnected && reservation.has_physical_work))
                    peer.has_unknown_state = true;
                peer.reservation_generation = reservation.generation;
                if (ready && reservation.has_physical_work)
                    peer.progress = core::ExecutionPriorityProgress{
                        policy->identity, reservation.generation, ready->cutoff};
            }
            if (nccl_failure)
                for (auto& peer : std::span{peers}.first(peer_count)) peer.has_unknown_state = true;
        }
#endif
        if (memory_native_health() != CUDA_SUCCESS)
            for (auto& peer : std::span{peers}.first(peer_count)) peer.has_unknown_state = true;
        result.is_valid = true;
        return result;
    }
    struct ConsideredArrivals {
        struct Entry {
            std::uint64_t policy{}, sequence{};
        };
        std::array<Entry, core::kMaxExecutionPriorityPeers> entries{};
        std::size_t size{};
    };
    ConsideredArrivals initial_arrivals(const PriorityObservations& observed,
                                        bool requires_quiescence) noexcept {
        ConsideredArrivals result;
        if (!config.paired || !config.paired->should_rebalance_on_arrival || !observed.is_valid ||
            arrival_checkpoint || is_stopping)
            return result;
        const auto peers = std::span{observed.peers}.first(observed.size);
        if (std::ranges::any_of(peers, [&](const auto& peer) {
                return peer.has_unknown_state || (requires_quiescence && peer.has_physical_work);
            }))
            return result;
        const auto plan = core::plan_execution_priority(peers, config.scheduling_mode);
        if (!plan.is_valid) return result;
        for (auto& slot : slots) {
            if (!slot.is_paired_member || !slot.session || slot.host_owner || slot.is_nccl ||
                slot.is_metadata_owner() || !slot.is_initialized || slot.is_retiring ||
                slot.is_closing || slot.policy_failure || slot.admission_fence)
                continue;
            const auto peer =
                std::ranges::find(peers, slot.policy_id, &core::ExecutionPriorityPeer::session);
            if (peer == peers.end() || peer->has_physical_work || peer->has_paused_work ||
                !peer->command || peer->command->is_resume || peer->command->physical_generation ||
                peer->command->command_epoch != 1 ||
                peer->command->sequence <= slot.considered_arrival_sequence)
                continue;
            const auto index = static_cast<std::size_t>(peer - peers.begin());
            if (plan.decisions[index].action != core::ExecutionPriorityAction::kExecute) continue;
            const auto pending = slot.session->pending_command();
            if (pending.status != CUDA_SUCCESS || !pending.command ||
                !pending.command.description().command.is_persistent ||
                !pending.command.description().command.function ||
                command_description(pending.command.description()) != *peer->command)
                continue;
            result.entries[result.size++] = {slot.policy_id, peer->command->sequence};
        }
        return result;
    }
    ConsideredArrivals issue_arrivals() noexcept {
        if (!config.paired || !config.paired->should_rebalance_on_arrival) return {};
        return initial_arrivals(priority_observations(), true);
    }
    void consume_arrivals(const ConsideredArrivals& considered) noexcept {
        for (const auto& entry : std::span{considered.entries}.first(considered.size)) {
            const auto slot = std::ranges::find(slots, entry.policy, &Slot::policy_id);
            if (slot != slots.end())
                slot->considered_arrival_sequence =
                    std::max(slot->considered_arrival_sequence, entry.sequence);
        }
    }
    void observe_arrival_checkpoint() noexcept {
        if (!arrival_checkpoint) return;
        auto& checkpoint = *arrival_checkpoint;
        for (std::size_t leg = 0; leg < checkpoint.size; ++leg) {
            if (checkpoint.is_stopped[leg]) continue;
            const auto& owner = checkpoint.sessions[leg];
            const auto slot =
                std::ranges::find(slots, checkpoint.legs[leg].session, &Slot::policy_id);
            if (slot == slots.end() || slot->policy_failure ||
                owner->health_snapshot().is_quarantined)
                continue;
            const auto progress =
                owner->persistent_progress(checkpoint.legs[leg].physical.sequence);
            const auto& frontier = checkpoint.frontiers[leg];
            const auto retired = frontier.retirement();
            checkpoint.is_stopped[leg] =
                (progress &&
                 progress->physical_stops >= checkpoint.legs[leg].physical.generation) ||
                (owner->owns_frontier(frontier) && retired.is_retired &&
                 retired.status == CUDA_SUCCESS);
        }
        if (!std::ranges::all_of(std::span{checkpoint.is_stopped}.first(checkpoint.size),
                                 [](bool stopped) { return stopped; }))
            return;
        if (serial_pair && std::ranges::any_of(std::span{checkpoint.legs}.first(checkpoint.size),
                                               [&](const auto& leg) {
                                                   return leg.session == serial_pair->policies[0] &&
                                                          leg.physical.sequence ==
                                                              serial_pair->sequence;
                                               }))
            serial_pair.reset();
        arrival_checkpoint.reset();
    }
    void begin_arrival_checkpoint() noexcept {
        if (!config.paired || !config.paired->should_rebalance_on_arrival || arrival_checkpoint ||
            profile_pair || pair_profile_failure || is_stopping)
            return;
        const auto observed = priority_observations();
        const auto considered = initial_arrivals(observed, false);
        if (!considered.size) return;
        const auto peers = std::span{observed.peers}.first(observed.size);
        const auto ordinary = core::plan_execution_priority(peers);
        ArrivalCheckpoint checkpoint;
        for (const auto& peer : peers) {
            if (!peer.has_physical_work) continue;
            if (!peer.physical || peer.priority != ordinary.selected_priority ||
                checkpoint.size == checkpoint.legs.size())
                return;
            const auto slot = std::ranges::find(slots, peer.session, &Slot::policy_id);
            if (slot == slots.end() || !slot->is_paired_member || slot->host_owner ||
                slot->is_retiring || slot->is_closing || slot->asynchronous_failure)
                return;
            const auto epoch = std::ranges::find(slot->low_epochs, peer.physical->sequence,
                                                 &Slot::LowEpoch::sequence);
            if (epoch == slot->low_epochs.end() || !slot->session->owns_frontier(epoch->frontier))
                return;
            const auto leg = checkpoint.size++;
            checkpoint.sessions[leg] = slot->session_leases.acquire();
            checkpoint.frontiers[leg] = epoch->frontier;
            checkpoint.legs[leg] = {peer.session, *peer.physical};
        }
        if (!checkpoint.size) return;
        for (const auto& entry : std::span{considered.entries}.first(considered.size)) {
            const auto slot = std::ranges::find(slots, entry.policy, &Slot::policy_id);
            if (slot == slots.end() ||
                !std::ranges::all_of(std::span{checkpoint.sessions}.first(checkpoint.size),
                                     [&](const auto& owner) {
                                         return slot->session->can_share_configuration(*owner);
                                     }))
                continue;
            const auto trigger =
                std::ranges::find(peers, entry.policy, &core::ExecutionPriorityPeer::session);
            if (trigger == peers.end() || !trigger->command) continue;
            const core::ExecutionRebalanceProposal proposal{
                core::ExecutionRebalanceTrigger{entry.policy, *trigger->command}, checkpoint.legs,
                checkpoint.size};
            const auto plan = core::plan_execution_rebalance(peers, proposal);
            if (!plan.is_valid) continue;
            checkpoint.trigger_policy = entry.policy;
            checkpoint.trigger_sequence = entry.sequence;
            checkpoint.trigger = slot->session_leases.acquire();
            arrival_checkpoint = std::move(checkpoint);
            consume_arrivals(considered);
            return;
        }
    }
    std::uint64_t priority_now_ns() const noexcept {
        if (config.priority_clock_ns) return config.priority_clock_ns();
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();
        return now > 0 ? static_cast<std::uint64_t>(now) : 0;
    }
    core::ExecutionPriorityPlan apply_priority_idle(const PriorityObservations& observed,
                                                    core::ExecutionPriorityPlan plan) noexcept {
        if (!priority_idle.enabled()) return plan;
        return priority_idle.apply(std::span{observed.peers}.first(observed.size),
                                   config.scheduling_mode, priority_now_ns(), plan);
    }
    core::ExecutionPriorityPlan priority_plan(HostStage callsite = HostStage::kPlanOther,
                                              std::uint64_t plan_serial = 0) noexcept {
        const HostScope host_scope(config.host_stage_metrics.get(), callsite);
        const auto observed = priority_observations(plan_serial);
        if (!observed.is_valid) return {};
        auto plan = [&] {
            const HostScope policy_scope(config.host_stage_metrics.get(), HostStage::kPolicy);
            return core::plan_execution_priority(std::span{observed.peers}.first(observed.size),
                                                 config.scheduling_mode);
        }();
        if (arrival_checkpoint) {
            core::ExecutionRebalanceProposal proposal;
            const auto& checkpoint = *arrival_checkpoint;
            for (std::size_t leg = 0; leg < checkpoint.size; ++leg)
                if (!checkpoint.is_stopped[leg])
                    proposal.legs[proposal.size++] = checkpoint.legs[leg];
            for (const auto& peer : std::span{observed.peers}.first(observed.size))
                if (peer.session == checkpoint.trigger_policy && peer.command &&
                    peer.command->sequence == checkpoint.trigger_sequence &&
                    !peer.command->is_resume && !peer.command->physical_generation &&
                    peer.command->command_epoch == 1 && peer.has_ready_work &&
                    !peer.has_physical_work && !peer.has_paused_work)
                    proposal.trigger = core::ExecutionRebalanceTrigger{peer.session, *peer.command};
            plan = core::plan_execution_rebalance(std::span{observed.peers}.first(observed.size),
                                                  proposal);
        }
        if (profile_pair) {
            const bool outside_physical = std::ranges::any_of(
                std::span{observed.peers}.first(observed.size), [&](const auto& peer) {
                    return peer.has_physical_work && peer.session != profile_pair->policies[0] &&
                           peer.session != profile_pair->policies[1];
                });
            for (auto& decision : std::span{plan.decisions}.first(plan.size)) {
                if (decision.action != core::ExecutionPriorityAction::kExecute) continue;
                bool selected{};
                for (std::size_t leg = 0; leg < 2; ++leg)
                    selected =
                        selected ||
                        (decision.session == profile_pair->policies[leg] && decision.command &&
                         decision.command->sequence == profile_pair->sequences[leg]);
                if (outside_physical || !selected) {
                    decision.action = core::ExecutionPriorityAction::kWait;
                    decision.command.reset();
                }
            }
        }
        if (uses_priority_space_share())
            for (std::size_t index = 0; index < observed.size; ++index) {
                auto& decision = plan.decisions[index];
                const auto slot = std::ranges::find(slots, decision.session, &Slot::policy_id);
                if (slot == slots.end() ||
                    !space_share_budget_exhausted(*slot, plan.selected_priority) ||
                    decision.action != core::ExecutionPriorityAction::kExecute)
                    continue;
                decision.command.reset();
                decision.progress = observed.peers[index].progress;
                decision.action = decision.progress
                                      ? core::ExecutionPriorityAction::kExecuteProgress
                                      : core::ExecutionPriorityAction::kWait;
            }
        return apply_priority_idle(observed, plan);
    }

    static IssueAuditKind source_issue_audit_kind(const Slot& slot,
                                                  std::uint64_t sequence) noexcept {
        const auto epoch = std::ranges::find(slot.low_epochs, sequence, &Slot::LowEpoch::sequence);
        return epoch != slot.low_epochs.end() && epoch->is_paused ? IssueAuditKind::kResumeAttempt
                                                                  : IssueAuditKind::kInitial;
    }
    void note_issue(Slot& slot, const CudaExecutionIssueResult& issued, bool records_native = true,
                    IssueAuditKind kind = IssueAuditKind::kInitial) noexcept {
        if (issued.state != CudaExecutionIssueState::kIssued) return;
        if (uses_priority_space_share()) {
            const auto pin = std::ranges::find(config.client_priorities, slot.peer,
                                               &ExecutionClientPriority::peer);
            if (pin != config.client_priorities.end())
                space_share_issued[static_cast<std::size_t>(
                    pin - config.client_priorities.begin())] = true;
        }
        if (issued.native_status == CUDA_SUCCESS && issued.library_status.value_or(0) == 0)
            priority_idle.note_issue(slot.priority);
        const auto tracked =
            std::ranges::find(slot.low_epochs, issued.sequence, &Slot::LowEpoch::sequence);
        if (tracked != slot.low_epochs.end() && tracked->is_event) {
            tracked->is_issued = true;
            tracked->was_successful = issued.native_status == CUDA_SUCCESS;
            audit("event_issued", slot, issued.sequence, issued.native_status);
            audit_ordering(slot, *tracked, 2, issued.native_status);
            return;  // An event is ordering work, never useful compute evidence.
        }
        if (slot.is_low || slot.session->persistent_progress(issued.sequence)) {
            const auto epoch =
                std::ranges::find(slot.low_epochs, issued.sequence, &Slot::LowEpoch::sequence);
            if (epoch == slot.low_epochs.end()) {
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                return;  // Never continue policy issue after an untracked physical epoch.
            }
            epoch->is_issued = true;
            epoch->was_successful =
                issued.native_status == CUDA_SUCCESS && issued.library_status.value_or(0) == 0;
            if (const auto progress = slot.session->persistent_progress(issued.sequence))
                epoch->progress = *progress;
        }
        if (kind == IssueAuditKind::kInitial)
            audit(slot.is_low ? "low_issued" : "high_issued", slot, issued.sequence,
                  issued.native_status);
        if (!config.should_audit_native || !records_native || !audit_log) return;
        // Track only a sequence, never a frontier/resource lease. Ordinary
        // collection reports actual retirement through the passive observer.
        if (!slot.retirement_audit ||
            !slot.session->set_retirement_observer(slot.retirement_audit) ||
            !slot.retirement_audit->track(issued.sequence))
            semantic_lost(5);
        const auto issued_progress = slot.session->persistent_progress(issued.sequence);
        try {
            if (issued.library_status)
                (void)audit_log->info(
                    "Native library issue audit: session={}, sequence={}, status={}",
                    slot.connection, issued.sequence, *issued.library_status);
            if (const auto& progress = issued_progress)
                (void)audit_log->info(
                    "Native persistent audit: session={}, sequence={}, total_blocks={}, "
                    "physical_launches={}, workers={}",
                    slot.connection, issued.sequence, progress->total_blocks,
                    progress->physical_launches, progress->workers);
            (void)audit_log->info("Native issue audit: session={}, sequence={}, status={}",
                                  slot.connection, issued.sequence,
                                  static_cast<int>(issued.native_status));
        } catch (...) {
            // The actual result and physical ownership do not depend on logging.
        }
    }
    CudaExecutionIssueResult issue_scheduled_command(Slot& slot, bool& is_resume) noexcept {
        const auto result = [&] {
            const HostScope attempt_scope(config.host_stage_metrics.get(),
                                          HostStage::kIssueAttempt);
            return issue_scheduled_command_impl(slot, is_resume);
        }();
        if (config.host_stage_metrics)
            config.host_stage_metrics->issue(result.state == CudaExecutionIssueState::kIssued ? 0
                                             : result.state == CudaExecutionIssueState::kIdle ? 1
                                             : result.native_status == CUDA_ERROR_NOT_READY   ? 2
                                                                                              : 3);
        return result;
    }
    CudaExecutionIssueResult issue_scheduled_command_impl(Slot& slot, bool& is_resume) noexcept {
        is_resume = false;
        if (const auto health = memory_native_health(); health != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = health};
        if (!slot.session || slot.host_owner || slot.is_nccl || slot.is_metadata_owner() ||
            slot.admission_fence)
            return {};
        std::optional<std::uint64_t> sequence;
        std::optional<std::size_t> profile_leg;
        if (profile_pair) {
            if (profile_pair->needs_initial_pair) return {.native_status = CUDA_ERROR_NOT_READY};
            for (std::size_t leg = 0; leg < 2; ++leg)
                if (slot.policy_id == profile_pair->policies[leg]) profile_leg = leg;
            if (!profile_leg) return {.native_status = CUDA_ERROR_NOT_READY};
            const auto& frontier = profile_pair->frontiers[*profile_leg];
            if (!slot.session->owns_frontier(frontier))
                return {.state = CudaExecutionIssueState::kError,
                        .native_status = CUDA_ERROR_INVALID_HANDLE};
            const auto retired = frontier.retirement();
            if (retired.is_retired) {
                if (retired.status != CUDA_SUCCESS)
                    return {.state = CudaExecutionIssueState::kError,
                            .native_status = retired.status};
                // Its partner still owns the isolated interval. This leg is
                // idle, not a stale command and never permission for queued work.
                return {};
            }
            sequence = profile_pair->sequences[*profile_leg];
        }
        if (serial_pair) {
            if (slot.policy_id == serial_pair->policies[1]) {
                if (!serial_pair->tail || !serial_pair->tail->is_ready)
                    return {.native_status = CUDA_ERROR_NOT_READY};
                sequence = serial_pair->tail->sequence;
            }
            if (slot.policy_id == serial_pair->policies[0]) {
                if (serial_pair->tail && serial_pair->tail->is_ready)
                    return {.native_status = CUDA_ERROR_NOT_READY};
                sequence = serial_pair->sequence;
            }
        }
        const auto pending = slot.session->pending_command(sequence);
        if (pending.status != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = pending.status};
        if (!pending.command) return {};
        if (pair_profile_failure && slot.is_paired_member &&
            pending.command.description().command.function) {
            (void)slot.session->reject_queued(*pair_profile_failure);
            return {};
        }
        if (profile_leg) {
            const auto progress = slot.session->persistent_progress(*sequence);
            const auto& expected = profile_pair->reservation.configuration.legs[*profile_leg];
            if (!progress || progress->workers != expected.workers ||
                progress->kind != expected.kind ||
                !profile_pair->sessions[0]->can_share_configuration(*profile_pair->sessions[1]))
                return {.state = CudaExecutionIssueState::kError,
                        .native_status = CUDA_ERROR_INVALID_HANDLE};
        } else if (config.paired && config.paired->is_isolated_profile &&
                   pending.command.description().command.is_persistent && !is_stopping &&
                   !slot.is_retiring) {
            // Trained earlier cases (for example a warmup) cannot authorize a
            // lone different command. Every new persistent input needs its
            // actual partner/sample or configured serial-original selection.
            return {.native_status = CUDA_ERROR_NOT_READY};
        }
        if (slot.is_paired_member) {
            for (const auto& peer : slots) {
                if (!peer.is_paired_member || peer.priority < slot.priority ||
                    !has_physical_reservation(peer.session->scheduling_snapshot()))
                    continue;
                if (profile_leg && (peer.policy_id == profile_pair->policies[0] ||
                                    peer.policy_id == profile_pair->policies[1]))
                    continue;
                const bool private_tail =
                    std::ranges::any_of(peer.low_epochs, [&](const auto& epoch) {
                        const auto progress = peer.session->persistent_progress(epoch.sequence);
                        return progress && progress->physical_launches > progress->physical_stops;
                    });
                if (pending.command.description().command.is_persistent || private_tail)
                    return {.native_status = CUDA_ERROR_NOT_READY};
            }
        }
        // No saved connection-wide grant survives an earlier command's native
        // issue/retirement, configuration restore, dependency or queue transition.
        const auto plan_serial =
            config.issue_depth_metrics ? config.issue_depth_metrics->begin_plan() : 0;
        const auto plan = priority_plan(HostStage::kPlanIssue, plan_serial);
        if (!plan.is_valid)
            return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
        const auto decisions = std::span{plan.decisions}.first(plan.size);
        const auto selected =
            std::ranges::find(decisions, slot.policy_id, &core::ExecutionPriorityDecision::session);
        const auto description = pending.command.description();
        const auto exact = command_description(description);
        if (selected == decisions.end() ||
            selected->action != core::ExecutionPriorityAction::kExecute ||
            selected->command != exact)
            return {.native_status = CUDA_ERROR_NOT_READY};
        is_resume = exact.is_resume;
        if (is_resume && std::ranges::find(slot.low_epochs, exact.sequence,
                                           &Slot::LowEpoch::sequence) == slot.low_epochs.end())
            return {.state = CudaExecutionIssueState::kError,
                    .sequence = exact.sequence,
                    .native_status = CUDA_ERROR_UNKNOWN};
        if (config.issue_depth_metrics)
            config.issue_depth_metrics->authorized({.connection = slot.connection,
                                                    .sequence = exact.sequence,
                                                    .plan_serial = plan_serial,
                                                    .is_resume = is_resume});
        const auto considered = issue_arrivals();
        const bool is_sgemm_target = sgemm_issue_range && sgemm_audit_captured &&
                                     config.sgemm_audit_target &&
                                     slot.connection == sgemm_audit_session &&
                                     exact.sequence == config.sgemm_audit_target->sequence;
        const bool is_first_target_issue =
            sgemm_issue_range && sgemm_issue_range->begin(is_sgemm_target, is_resume);
        const auto issued = [&] {
            const HostScope issue_scope(config.host_stage_metrics.get(), HostStage::kIssueNative);
            if (!is_resume)
                lifecycle(HostStageMetrics::Lifecycle::kIssueBegin, slot, 0, exact.sequence,
                          description.command.is_persistent ? 1 : 0);
            if (config.issue_depth_metrics)
                config.issue_depth_metrics->begin_issue(slot.connection, exact.sequence, is_resume);
            const auto result = slot.session->issue(pending.command);
            if (config.issue_depth_metrics)
                config.issue_depth_metrics->end_issue(
                    slot.connection, exact.sequence,
                    result.state == CudaExecutionIssueState::kIssued &&
                        result.sequence == exact.sequence && result.native_status == CUDA_SUCCESS &&
                        result.completion_status == CUDA_SUCCESS &&
                        result.library_status.value_or(0) == 0);
            if (!is_resume)
                lifecycle(HostStageMetrics::Lifecycle::kIssueEnd, slot, result.sequence,
                          exact.sequence,
                          result.state == CudaExecutionIssueState::kIssued &&
                                  result.native_status == CUDA_SUCCESS &&
                                  result.completion_status == CUDA_SUCCESS &&
                                  result.library_status.value_or(0) == 0
                              ? 0
                              : 1);
            return result;
        }();
        if (sgemm_issue_range)
            sgemm_issue_range->finish(is_first_target_issue,
                                      issued.state == CudaExecutionIssueState::kIssued &&
                                          issued.sequence == exact.sequence,
                                      {.native = static_cast<int>(issued.native_status),
                                       .completion = static_cast<int>(issued.completion_status)},
                                      issued.library_status.has_value(),
                                      issued.library_status.value_or(-1));
        if (issued.state == CudaExecutionIssueState::kIssued && serial_pair && serial_pair->tail &&
            serial_pair->tail->is_ready && slot.policy_id == serial_pair->policies[1] &&
            issued.sequence == serial_pair->tail->sequence)
            serial_pair.reset();
        if (issued.state == CudaExecutionIssueState::kIssued &&
            issued.native_status == CUDA_SUCCESS)
            consume_arrivals(considered);
        return issued;
    }

    void note_scheduled_issue(Slot& slot, const CudaExecutionIssueResult& issued,
                              bool was_resume) noexcept {
        const auto epoch =
            std::ranges::find(slot.low_epochs, issued.sequence, &Slot::LowEpoch::sequence);
        if (was_resume) {
            if (epoch == slot.low_epochs.end()) {
                policy_failed(slot, CUDA_ERROR_UNKNOWN);
                return;
            }
            if (issued.state == CudaExecutionIssueState::kError &&
                issue_failure(issued) == CUDA_ERROR_NOT_READY)
                return;
            if (issued.state != CudaExecutionIssueState::kIssued ||
                issued.native_status != CUDA_SUCCESS || issued.completion_status != CUDA_SUCCESS) {
                policy_failed(slot, issue_failure(issued));
                return;
            }
            epoch->retreat_requested = epoch->retreat_unsupported = epoch->is_paused = false;
            if (const auto progress = slot.session->persistent_progress(epoch->sequence))
                epoch->progress = *progress;
            if (slot.is_low) audit("low_resumed", slot, epoch->sequence);
        }
        note_issue(slot, issued, !was_resume,
                   was_resume ? IssueAuditKind::kResumeAttempt : IssueAuditKind::kInitial);
    }

    void observe_serial_pair() noexcept {
        if (!serial_pair) return;
        if (serial_pair->tail && serial_pair->tail->is_ready) {
            const auto& second = serial_pair->sessions[1];
            const auto& frontier = serial_pair->tail->frontier;
            if (!second->health_snapshot().is_quarantined && second->owns_frontier(frontier) &&
                frontier.retirement().is_retired)
                serial_pair.reset();
            return;
        }
        const auto& first = serial_pair->sessions[0];
        if (first->owns_frontier(serial_pair->first) &&
            serial_pair->first.retirement().is_retired) {
            if (serial_pair->tail) {
                if (first->health_snapshot().is_quarantined) return;
                const auto status = serial_pair->first.retirement().status;
                if (status == CUDA_SUCCESS) {
                    serial_pair->tail->is_ready = true;
                    return;
                }
                // A known failed first leg cannot authorize the selected tail.
                // Reject never-issued profile inputs; uncertain owners above
                // stay retained instead of borrowing this retirement proof.
                fail_pair_profile(status == CUDA_ERROR_NOT_READY ? CUDA_ERROR_UNKNOWN : status);
            }
            serial_pair.reset();
            return;
        }
        const auto status = first->query_frontier(serial_pair->first);
        if (status != CUDA_SUCCESS && status != CUDA_ERROR_NOT_READY) {
            for (auto& slot : slots)
                if (slot.policy_id == serial_pair->policies[0] ||
                    slot.policy_id == serial_pair->policies[1])
                    policy_failed(slot, status);
        }
    }

    void invalidate_profile_pair() noexcept {
        if (!profile_pair) return;
        for (const auto& observer : profile_pair->reservation.observations)
            if (observer) observer->finish(false);
        // Only the measurement is terminal. The exact frontiers/Sessions remain
        // retained until actual retirement; no accepted work is cancelled here.
    }

    void observe_profile_pair() noexcept {
        if (!profile_pair) return;
        std::size_t retired{};
        for (std::size_t leg = 0; leg < 2; ++leg) {
            if (profile_pair->sessions[leg]->owns_frontier(profile_pair->frontiers[leg]) &&
                profile_pair->frontiers[leg].retirement().is_retired)
                ++retired;
        }
        if (retired == 2)
            profile_pair.reset();
        else if (retired)
            profile_pair->needs_initial_pair = false;
    }

    bool permits_profile_request(const Slot& slot, const Slot::Request& request) const noexcept {
        if (!profile_pair) return true;
        const auto operation = request.message.header.operation;
        return is_host_function_observation(operation) || is_async_frontier_batch(operation) ||
               operation == ExecutionOpcode::kSnapshot ||
               operation == ExecutionOpcode::kCaptureFrontier ||
               operation == ExecutionOpcode::kQueryFrontier ||
               operation == ExecutionOpcode::kReleaseFrontier ||
               (request.receipt && slot.session && slot.session->host_result(*request.receipt));
    }

    void fail_pair_profile(CUresult status) noexcept {
        if (!config.paired || !config.paired->is_isolated_profile || pair_profile_failure ||
            profile_pair || status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY)
            return;
        pair_profile_failure = status;
        // No sample/native attempt exists here. Reject only records whose queue
        // proves they were never issued; healthy lifecycle/metadata still run.
        for (auto& slot : slots)
            if (slot.is_paired_member && slot.session) (void)slot.session->reject_queued(status);
    }

    CudaSessionPairKernelResult prepare_pair_candidate(Slot& slot) noexcept {
        const auto pending = slot.session->pending_command();
        if (pending.status != CUDA_SUCCESS) return {.status = pending.status};
        if (!pending.command || !pending.command.description().command.is_persistent)
            return {.status = CUDA_ERROR_NOT_SUPPORTED};
        auto prepared = slot.session->prepare_pair_kernel(pending.command);
        if (prepared.status != CUDA_SUCCESS) {
            if (slot.session->health_snapshot().is_quarantined)
                policy_failed(slot, prepared.status);
            else
                fail_pair_profile(prepared.status);
        }
        return prepared;
    }

    bool schedule_pending_pair() noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return true;
        if (!config.paired || !pair_profile_cache || serial_pair || pair_profile_failure ||
            arrival_checkpoint)
            return false;
        std::array<Slot*, 2> candidates{};
        if (profile_pair) {
            if (!profile_pair->needs_initial_pair) return false;
            for (std::size_t leg = 0; leg < 2; ++leg) {
                const auto found =
                    std::ranges::find(slots, profile_pair->policies[leg], &Slot::policy_id);
                if (found == slots.end()) return false;
                candidates[leg] = &*found;
            }
            return issue_selected_pair(candidates, profile_pair->selection,
                                       profile_pair->frontiers);
        }
        const auto before = priority_observations();
        if (!before.is_valid ||
            std::ranges::any_of(std::span{before.peers}.first(before.size), [](const auto& peer) {
                return peer.has_physical_work || peer.has_unknown_state;
            }))
            return false;
        const auto ordinary = apply_priority_idle(
            before, core::plan_execution_priority(std::span{before.peers}.first(before.size),
                                                  config.scheduling_mode));
        if (!ordinary.is_valid) return false;
        std::array<Slot*, core::kMaxExecutionPriorityPeers> eligible{};
        std::size_t count{};
        // Collect exact persistent commands; backend preparation distinguishes
        // public PTX catalogs from known private replacement bindings. Opaque
        // native work keeps its ordinary authorization and dependency cut paths.
        for (auto& slot : slots) {
            if (!slot.is_paired_member || !slot.is_initialized || slot.is_retiring ||
                slot.is_closing || slot.policy_failure || slot.admission_fence)
                continue;
            const auto decisions = std::span{ordinary.decisions}.first(ordinary.size);
            const auto decision = std::ranges::find(decisions, slot.policy_id,
                                                    &core::ExecutionPriorityDecision::session);
            if (decision == decisions.end() ||
                decision->action != core::ExecutionPriorityAction::kExecute)
                continue;
            const auto current = slot.session->pending_command();
            if (current.status != CUDA_SUCCESS || !current.command) continue;
            if (decision->command != command_description(current.command.description()))
                return false;
            if (!current.command.description().command.is_persistent) {
                if (config.paired->is_isolated_profile &&
                    current.command.description().command.function) {
                    fail_pair_profile(CUDA_ERROR_NOT_SUPPORTED);
                    return false;
                }
                continue;
            }
            if (count == eligible.size()) return false;
            eligible[count++] = &slot;
        }
        if (count < candidates.size()) return false;
        std::optional<CudaPairProfileChoice> best;
        if (config.paired->is_isolated_profile) {
            if (count != 2 || !eligible[0]->session->can_share_configuration(*eligible[1]->session))
                return false;
            candidates = {eligible[0], eligible[1]};
        } else {
            std::array<bool, core::kMaxExecutionPriorityPeers> grouped{};
            for (std::size_t anchor = 0; anchor < count; ++anchor) {
                if (grouped[anchor]) continue;
                std::array<Slot*, core::kMaxExecutionPriorityPeers> group{};
                std::size_t size{};
                for (std::size_t index = anchor; index < count; ++index)
                    if (!grouped[index] &&
                        (index == anchor || eligible[anchor]->session->can_share_configuration(
                                                *eligible[index]->session))) {
                        group[size++] = eligible[index];
                        grouped[index] = true;
                    }
                if (size < 2) continue;
                if (activate(*group[0]).status != CUDA_SUCCESS) return false;
                std::array<CudaSessionPairKernel, core::kMaxExecutionPriorityPeers> catalogs{};
                // SDK metadata is linear in eligible commands, not pairs.
                // All pair scoring below is CPU-only within one materialized
                // compatible configuration; another group cannot stale it yet.
                std::size_t prepared_count{};
                for (std::size_t index = 0; index < size; ++index) {
                    const auto prepared = prepare_pair_candidate(*group[index]);
                    if (prepared.disposition == CudaSessionPairKernelDisposition::kUnsupported &&
                        prepared.status == CUDA_ERROR_NOT_SUPPORTED && !prepared.kernel)
                        continue;
                    if (prepared.status != CUDA_SUCCESS || !prepared.kernel ||
                        prepared.disposition != CudaSessionPairKernelDisposition::kPrepared)
                        return false;
                    // Compact both arrays together without reordering ties.
                    // Native NOT_SUPPORTED is an error, never an exclusion.
                    group[prepared_count] = group[index];
                    catalogs[prepared_count++] = prepared.kernel;
                }
                size = prepared_count;
                for (std::size_t first = 0; first < size; ++first)
                    for (std::size_t second = first + 1; second < size; ++second) {
                        const std::array proposal{group[first], group[second]};
                        const auto evaluated = CudaExecutionSession::evaluate_pair(
                            *proposal[0]->session, catalogs[first], *proposal[1]->session,
                            catalogs[second], *pair_profile_cache, config.paired->workers);
                        if (evaluated.status != CUDA_SUCCESS) return false;
                        const auto order = cuda_pair_profile_compare_choices(
                            evaluated.choice, best.value_or(evaluated.choice));
                        if (!order) return false;
                        if (!best || *order > 0) {
                            best = evaluated.choice;
                            candidates = proposal;
                        }
                    }
            }
            if (!best) return false;
        }
        // Other compatible groups may have materialized a different native
        // configuration while evaluating. Re-capture ONLY the final exact pair;
        // losing proposals never select geometry, original fallback or observers.
        if (activate(*candidates[0]).status != CUDA_SUCCESS) return false;
        std::array<CudaSessionPairKernel, 2> kernels;
        for (std::size_t leg = 0; leg < candidates.size(); ++leg) {
            const auto prepared = prepare_pair_candidate(*candidates[leg]);
            if (prepared.status != CUDA_SUCCESS || !prepared.kernel ||
                prepared.disposition != CudaSessionPairKernelDisposition::kPrepared)
                return false;
            kernels[leg] = prepared.kernel;
        }
        std::array<CudaExecutionFrontier, 2> frontiers{};
        for (std::size_t leg = 0; leg < candidates.size(); ++leg) {
            auto& slot = *candidates[leg];
            const auto frontier = slot.session->capture_command_frontier(
                kernels[leg].description().pending.command.sequence);
            if (frontier.status != CUDA_SUCCESS) return false;
            frontiers[leg] = frontier.frontier;
        }
        CudaSessionPairSelectionResult selected;
        CudaPairProfileReservation reservation;
        if (config.paired->is_isolated_profile) {
            auto sampled = CudaExecutionSession::reserve_pair_sample(
                *candidates[0]->session, kernels[0], *candidates[1]->session, kernels[1],
                *pair_profile_cache, config.paired->workers);
            selected = {sampled.status, std::move(sampled.selection)};
            reservation = std::move(sampled.reservation);
        } else {
            const auto rechecked = CudaExecutionSession::evaluate_pair(
                *candidates[0]->session, kernels[0], *candidates[1]->session, kernels[1],
                *pair_profile_cache, config.paired->workers);
            if (rechecked.status != CUDA_SUCCESS || rechecked.choice.is_shared != best->is_shared ||
                rechecked.choice.is_trained != best->is_trained ||
                rechecked.choice.first_original != best->first_original ||
                rechecked.choice.candidate != best->candidate ||
                rechecked.choice.configuration != best->configuration ||
                rechecked.choice.original != best->original ||
                rechecked.choice.paired != best->paired)
                return false;
            selected = CudaExecutionSession::select_pair(
                *candidates[0]->session, kernels[0], *candidates[1]->session, kernels[1],
                *pair_profile_cache, config.paired->workers);
        }
        if (selected.status != CUDA_SUCCESS) {
            fail_pair_profile(selected.status);
            return false;
        }
        if (reservation) {
            const auto descriptions = selected.selection.kernels();
            profile_pair = ProfilePair{
                {candidates[0]->session_leases.acquire(), candidates[1]->session_leases.acquire()},
                {candidates[0]->policy_id, candidates[1]->policy_id},
                {descriptions[0].pending.command.sequence,
                 descriptions[1].pending.command.sequence},
                frontiers,
                selected.selection,
                std::move(reservation),
                true};
        }
        return issue_selected_pair(candidates, selected.selection, frontiers);
    }

    bool issue_selected_pair(std::array<Slot*, 2> candidates,
                             const CudaSessionPairSelection& selected,
                             const std::array<CudaExecutionFrontier, 2>& frontiers) noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return false;
        auto observed = priority_observations();
        if (!observed.is_valid) return false;
        const std::array policies{candidates[0]->policy_id, candidates[1]->policy_id};
        const auto contexts = selected.contexts();
        for (std::size_t leg = 0; leg < candidates.size(); ++leg) {
            auto peers = std::span{observed.peers}.first(observed.size);
            const auto peer =
                std::ranges::find(peers, policies[leg], &core::ExecutionPriorityPeer::session);
            if (peer == peers.end()) return false;
            peer->pair_context = contexts[leg];
        }
        const auto proposal = selected.proposal(policies);
        const auto plan = core::plan_execution_pair(std::span{observed.peers}.first(observed.size),
                                                    proposal, config.scheduling_mode);
        if (!plan.is_valid || !plan.grant_count) return false;
        const auto idle_plan = apply_priority_idle(observed, plan.priority);
        if (!idle_plan.is_valid) return false;
        for (const auto& grant : std::span{plan.grants}.first(plan.grant_count)) {
            const auto decisions = std::span{idle_plan.decisions}.first(idle_plan.size);
            const auto permitted = std::ranges::find(decisions, grant.session,
                                                     &core::ExecutionPriorityDecision::session);
            if (permitted == decisions.end() ||
                permitted->action != core::ExecutionPriorityAction::kExecute ||
                permitted->command != grant.command)
                return false;
        }
        const auto considered = initial_arrivals(observed, true);
        if (!proposal.is_shared) {
            const auto first = proposal.first_serial;
            serial_pair = SerialPair{{candidates[first]->session_leases.acquire(),
                                      candidates[1 - first]->session_leases.acquire()},
                                     {policies[first], policies[1 - first]},
                                     frontiers[first],
                                     proposal.legs[first].command.sequence};
            if (config.paired->is_isolated_profile)
                serial_pair->tail = SerialPair::Tail{proposal.legs[1 - first].command.sequence,
                                                     frontiers[1 - first], false};
        }
        // Both exact native calls run before any note/audit/poll turn. A partial
        // method failure never rolls back or substitutes the already-issued leg.
        const auto result = CudaExecutionSession::issue_pair(
            *candidates[0]->session, *candidates[1]->session, selected, plan, policies);
        for (std::size_t leg = 0; leg < candidates.size(); ++leg)
            if (result.attempted[leg] &&
                result.issues[leg].state == CudaExecutionIssueState::kIssued &&
                result.issues[leg].native_status == CUDA_SUCCESS) {
                consume_arrivals(considered);
                break;
            }
        bool attempted{};
        for (std::size_t leg = 0; leg < candidates.size(); ++leg) {
            if (!result.attempted[leg]) continue;
            attempted = true;
            note_scheduled_issue(*candidates[leg], result.issues[leg],
                                 proposal.legs[leg].command.is_resume);
        }
        if (attempted && profile_pair) {
            profile_pair->needs_initial_pair = false;
            if (result.status != CUDA_SUCCESS) invalidate_profile_pair();
        }
        if (!attempted) serial_pair.reset();
        return attempted;
    }

    void record_prepared_kernel(Slot& slot, const CudaSessionPendingCommand& command,
                                CudaSessionKernelPreparationDisposition disposition) noexcept {
        if (config.issue_depth_metrics && config.issue_depth_metrics->records_events() &&
            disposition == CudaSessionKernelPreparationDisposition::kPrepared)
            config.issue_depth_metrics->prepared(slot.connection,
                                                 command.description().command.sequence);
    }

    void prepare_asynchronous_kernel(Slot& slot) noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return;
        if (!slot.session || slot.host_owner || slot.is_nccl || slot.is_metadata_owner() ||
            slot.admission_fence || slot.is_retiring || slot.policy_failure ||
            slot.asynchronous_receipts.empty() || !slot.is_initialized ||
            slot.is_bootstrap_pending || !slot.session->health_snapshot().has_context)
            return;
        const auto pending = slot.session->pending_preparation();
        if (pending.status == CUDA_ERROR_NOT_READY ||
            (pending.status == CUDA_SUCCESS && !pending.command))
            return;
        if (pending.status != CUDA_SUCCESS) {
            policy_failed(slot, pending.status);
            return;
        }
        // Admission has no native authority. Preparation separately selects a
        // fresh Core-eligible owner, and native issue recaptures its own grant.
        const auto plan = priority_plan();
        const auto decisions = std::span{plan.decisions}.first(plan.size);
        const auto selected =
            std::ranges::find(decisions, slot.policy_id, &core::ExecutionPriorityDecision::session);
        if (!plan.is_valid || selected == decisions.end() ||
            selected->action != core::ExecutionPriorityAction::kExecute)
            return;
        const auto activated = activate(slot);
        if (activated.status == CUDA_ERROR_NOT_READY && !activated.original_status) return;
        if (activated.status != CUDA_SUCCESS) {
            policy_failed(slot, activated.status);
            return;
        }
        // Activation can restore configuration and invalidate a saved token.
        // Reacquire this same sequence, never silently prepare a successor.
        const auto current =
            slot.session->pending_preparation(pending.command.description().command.sequence);
        if (current.status != CUDA_SUCCESS || !current.command) {
            if (current.status != CUDA_ERROR_NOT_READY)
                policy_failed(slot,
                              current.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : current.status);
            return;
        }
        const auto refreshed = priority_plan();
        const auto fresh_decisions = std::span{refreshed.decisions}.first(refreshed.size);
        const auto fresh_selected = std::ranges::find(fresh_decisions, slot.policy_id,
                                                      &core::ExecutionPriorityDecision::session);
        if (!refreshed.is_valid || fresh_selected == fresh_decisions.end() ||
            fresh_selected->action != core::ExecutionPriorityAction::kExecute)
            return;
        const auto prepared = slot.session->prepare_pending_kernel(current.command);
        record_prepared_kernel(slot, current.command, prepared.disposition);
        switch (prepared.disposition) {
            case CudaSessionKernelPreparationDisposition::kPrepared:
            case CudaSessionKernelPreparationDisposition::kDeferred:
            case CudaSessionKernelPreparationDisposition::kRejected:
                break;
            case CudaSessionKernelPreparationDisposition::kUncertain:
            case CudaSessionKernelPreparationDisposition::kFailed:
                policy_failed(
                    slot, prepared.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : prepared.status);
                break;
        }
        harvest_asynchronous_results(slot);
    }

    void schedule(Slot& slot) noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return;
        if (slot.host_owner || slot.is_nccl) return;
        const auto snapshot = slot.session->scheduling_snapshot();
        if (snapshot.has_context) {
            const auto activated = activate(slot);
            if (activated.status != CUDA_SUCCESS) {
                if (activated.status != CUDA_ERROR_NOT_READY) policy_failed(slot, activated.status);
                return;
            }
        }
        const auto budget = snapshot.submissions.queued + snapshot.submissions.paused;
        for (std::size_t attempt = 0; attempt < budget; ++attempt) {
            // New-mode high draining includes admitted ASY1 work, not merely
            // the one command prepared by the old once-per-turn visit. Each
            // preparation and issue independently reacquires native/Core proof.
            if (uses_priority_space_share()) prepare_asynchronous_kernel(slot);
            bool is_resume{};
            const auto issued = issue_scheduled_command(slot, is_resume);
            if (issued.state == CudaExecutionIssueState::kIdle) break;
            if (issued.state == CudaExecutionIssueState::kError) {
                const auto status = issue_failure(issued);
                if (status == CUDA_ERROR_NOT_READY) break;
                if (!slot.asynchronous_failure) slot.asynchronous_failure = status;
                (void)slot.session->reject_queued(status);
                break;
            }
            note_scheduled_issue(slot, issued, is_resume);
        }
    }

#ifdef GLIMMER_HAS_NCCL
    NativeDependencyPath trace_native_dependency(
        NativeSourceKey key, CudaExecutionDependencyInspection inspected,
        std::optional<CudaExecutionFrontier> root_frontier = std::nullopt,
        bool is_retirement = false) noexcept {
        NativeDependencyPath path;
        path.is_retirement = is_retirement;
        const auto peer = std::ranges::find(native_peers, key.peer, &NativePeer::id);
        const auto source = std::ranges::find(slots, key.session, &Slot::connection);
        if (peer == native_peers.end() || peer->disconnected || source == slots.end() ||
            source->peer != peer->identity || !source->session || source->host_owner ||
            source->is_metadata_owner() || source->is_nccl)
            return path;
        path.source_id = key.session;
        path.source = source->session_leases.acquire();
        path.frontier = std::move(root_frontier);
        if (inspected.has_failure_observation) {
            if (inspected.status == CUDA_SUCCESS) {
                path.status = CUDA_ERROR_UNKNOWN;
            } else if (inspected.has_retired_failure) {
                path.rejection = inspected.status;
                path.status = CUDA_SUCCESS;
                path.has_progress = true;
            } else {
                path.status = CUDA_ERROR_NOT_READY;
            }
            return path;
        }
        if (inspected.status != CUDA_SUCCESS && inspected.status != CUDA_ERROR_NOT_READY &&
            !inspected.foreign_blocker) {
            path.status = inspected.status;
            return path;
        }
        try {
            path.edges.reserve(foreign_events.size());
            // An independent local member of this exact cut may be actionable
            // while another member waits on a foreign event. Do not hide that
            // local work behind a physical-only foreign leaf.
            while (inspected.foreign_blocker && !inspected.has_progress) {
                const auto edge = std::ranges::find_if(foreign_events, [&](const auto& item) {
                    return item->target_sequence && item->peer == peer->identity &&
                           item->target_id == path.source_id && item->target == path.source &&
                           item->source_frontier &&
                           item->source_frontier->same_capture(*inspected.foreign_blocker);
                });
                if (edge == foreign_events.end() || path.edges.size() == foreign_events.size())
                    return path;
                const auto& captured = *edge;
                if (!captured->source->owns_frontier(*captured->source_frontier) ||
                    !captured->target->owns_frontier(captured->target_frontier) ||
                    std::ranges::any_of(path.edges, [&](const auto& previous) {
                        return previous->source == captured->source &&
                               previous->source_frontier->same_capture(*captured->source_frontier);
                    }))
                    return path;
                const auto owner_slot =
                    std::ranges::find(slots, captured->source_id, &Slot::connection);
                if (owner_slot == slots.end() || owner_slot->peer != peer->identity ||
                    owner_slot->session != captured->source || owner_slot->host_owner ||
                    owner_slot->is_metadata_owner() || owner_slot->is_nccl)
                    return path;
                path.edges.push_back(captured);
                inspected =
                    is_retirement
                        ? captured->source->inspect_frontier_retirement(*captured->source_frontier)
                        : captured->source->inspect_frontier_dependency(*captured->source_frontier);
                if (inspected.has_retired_failure && inspected.has_failure_observation &&
                    inspected.status != CUDA_SUCCESS) {
                    // The exact failed observation is settled. Authorize the
                    // parent cut to publish its real preissue wait rejection;
                    // do not replay the failed source or claim the entire cut
                    // retired. Other source ownership remains independent.
                    path.status = CUDA_SUCCESS;
                    path.has_progress = true;
                    path.is_parent_rejection = true;
                    return path;
                }
                if (inspected.has_failure_observation) {
                    path.status = inspected.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN
                                                                   : CUDA_ERROR_NOT_READY;
                    return path;
                }
                if (inspected.status != CUDA_SUCCESS && inspected.status != CUDA_ERROR_NOT_READY &&
                    !inspected.foreign_blocker) {
                    path.status = inspected.status;
                    return path;
                }
                path.source_id = captured->source_id;
                path.source = captured->source;
                path.frontier = captured->source_frontier;
            }
            // Local preparatory failure can be published only by the same
            // exact backend progress path. Quarantine is not such permission.
            if (path.source->health_snapshot().is_quarantined) {
                path.status = CUDA_ERROR_UNKNOWN;
                return path;
            }
            path.status = CUDA_SUCCESS;
            path.has_progress = inspected.has_progress;
            path.preparation_sequence = inspected.preparation_sequence;
        } catch (const std::bad_alloc&) {
            path.status = CUDA_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            path.status = CUDA_ERROR_UNKNOWN;
        }
        return path;
    }
    NativeDependencyPath trace_nccl_dependency(NativeSourceKey key,
                                               const NcclActorRequestSnapshot& request) noexcept {
        const auto root = nccl_actor->inspect_source(request.request, request.grant, key.session);
        if (root.status != CUDA_SUCCESS || root.source_session != key.session) {
            NativeDependencyPath failed;
            failed.status = root.status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_HANDLE : root.status;
            return failed;
        }
        return trace_native_dependency(key, root.dependency);
    }
    NativeDependencyPath trace_continuation_dependency(
        NativeSourceKey key, const NcclActorContinuationSnapshot& request) noexcept {
        const auto root = nccl_actor->inspect_continuation_source(request.continuation,
                                                                  request.grant, key.session);
        const auto source = std::ranges::find(slots, key.session, &Slot::connection);
        if (root.status != CUDA_SUCCESS || root.binding.session != key.session ||
            !root.binding.owner || source == slots.end() || source->session != root.binding.owner ||
            !root.binding.owner->owns_frontier(root.binding.frontier)) {
            NativeDependencyPath failed;
            failed.status = root.status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_HANDLE : root.status;
            return failed;
        }
        return trace_native_dependency(
            key, root.dependency, root.binding.frontier,
            request.stage == NcclActorContinuationStage::kPrepared ||
                request.stage == NcclActorContinuationStage::kBlocked ||
                request.stage == NcclActorContinuationStage::kWaitingRetirement);
    }
    static bool same_dependency_path(const NativeDependencyPath& left,
                                     const NativeDependencyPath& right) noexcept {
        if (left.status != CUDA_SUCCESS || right.status != CUDA_SUCCESS ||
            left.rejection != right.rejection || left.source_id != right.source_id ||
            left.is_retirement != right.is_retirement || left.has_progress != right.has_progress ||
            left.preparation_sequence != right.preparation_sequence ||
            left.is_parent_rejection != right.is_parent_rejection || left.source != right.source ||
            left.frontier.has_value() != right.frontier.has_value() || left.edges != right.edges)
            return false;
        return !left.frontier || left.frontier->same_capture(*right.frontier);
    }
    bool permits_nccl_progress(std::uint64_t policy_id, NcclActorGrant grant,
                               core::ExecutionPriorityAction* observed_action = nullptr) noexcept {
        const auto plan = priority_plan();
        if (!plan.is_valid) return false;
        const auto decisions = std::span{plan.decisions}.first(plan.size);
        const auto selected =
            std::ranges::find(decisions, policy_id, &core::ExecutionPriorityDecision::session);
        const core::ExecutionPriorityProgress exact{policy_id, grant.generation, grant.cutoff};
        const bool permitted =
            selected != decisions.end() &&
            (selected->action == core::ExecutionPriorityAction::kExecute ||
             (selected->action == core::ExecutionPriorityAction::kExecuteProgress &&
              selected->progress == exact));
        if (permitted && observed_action) *observed_action = selected->action;
        return permitted;
    }
    std::uint64_t audit_native_grant(const NcclActorReservation& reservation, NcclActorGrant grant,
                                     std::uint64_t policy, core::ExecutionPriorityAction action,
                                     NativeAuditTarget target_kind, std::uint64_t target,
                                     NativeAuditMode mode, std::uint64_t root_source = 0,
                                     const NativeDependencyPath* path = nullptr) noexcept {
        const auto authorization = semantic(2, [&](auto& json) {
            json.add("event", 10);
            json.add("policy", policy);
            json.add("peer", reservation.peer);
            json.add("priority", reservation.priority);
            json.add("reservation", grant.reservation);
            json.add("generation", grant.generation);
            json.add("cutoff", grant.cutoff);
            json.add("target_kind", static_cast<unsigned int>(target_kind));
            json.add("target", target);
            json.add("action", static_cast<unsigned int>(action));
            json.add("mode", static_cast<unsigned int>(mode));
            json.add("root_source", root_source);
            json.add("leaf_source", path ? path->source_id : 0);
            json.add("capture", path && path->frontier ? path->frontier->diagnostic_id() : 0);
            json.add("edges", path ? path->edges.size() : 0);
            json.add("path_kind", path && path->is_parent_rejection ? 1 : 0);
        });
        if (!authorization || !path) return authorization;
        std::uint64_t index{};
        for (const auto& edge : path->edges) {
            const auto generation = edge->generation.diagnostic_identity();
            if (!generation || !edge->source_frontier || !edge->source_frontier->diagnostic_id() ||
                !edge->target_frontier.diagnostic_id()) {
                semantic_lost(4);
                return 0;
            }
            (void)semantic(2, [&](auto& json) {
                json.add("event", 11);
                json.add("grant", authorization);
                json.add("index", index++);
                json.add("source", edge->source_id);
                json.add("target", edge->target_id);
                json.add("event_object", generation->event_object);
                json.add("record_sequence", generation->record_sequence);
                json.add("source_capture", edge->source_frontier->diagnostic_id());
                json.add("target_capture", edge->target_frontier.diagnostic_id());
                json.add("target_sequence", edge->target_sequence);
            });
        }
        return authorization;
    }
    void audit_native_result(std::uint64_t authorization,
                             const NcclActorSourceProgress& result) noexcept {
        if (!authorization) return;
        (void)semantic(2, [&](auto& json) {
            json.add("event", 12);
            json.add("grant", authorization);
            json.add("status", static_cast<int>(result.status));
            json.add("issue_state", static_cast<unsigned int>(result.issue.state));
            json.add("command_sequence", result.issue.sequence);
            json.add("native_status", static_cast<int>(result.issue.native_status));
            json.add("completion_status", static_cast<int>(result.issue.completion_status));
            json.add("has_library", result.issue.library_status.has_value());
            json.add("library_status", result.issue.library_status.value_or(0));
            json.add("source", result.source_session);
        });
    }
    void audit_handler_outcome(std::uint64_t authorization, NcclActorContinuationPermit permit,
                               const NcclActorContinuationOutcome& outcome) noexcept {
        if (!authorization) return;
        (void)semantic(2, [&](auto& json) {
            json.add("event", 14);
            json.add("grant", authorization);
            json.add("continuation", permit.continuation);
            json.add("attempt", permit.attempt);
            json.add("status", static_cast<int>(outcome.status));
            json.add("outcome", static_cast<unsigned int>(outcome.kind) + 1);
            json.add("has_wire", outcome.reply.has_value());
            json.add("wire_status", outcome.reply ? outcome.reply->status : 0);
            json.add("has_original", outcome.reply && outcome.reply->original_status.has_value());
            json.add("original_status",
                     outcome.reply ? outcome.reply->original_status.value_or(0) : 0);
            json.add("receipt_sequence", outcome.binding && outcome.binding->receipt
                                             ? outcome.binding->receipt->sequence()
                                             : 0);
            json.add("mailbox_sequence", outcome.pending_sequence);
            json.add("capture", outcome.binding ? outcome.binding->diagnostic_capture : 0);
            json.add("source", outcome.binding ? outcome.binding->session : 0);
        });
    }
    NcclActorSourceProgress prepare_dependency_leaf(const NativeDependencyPath& path) noexcept {
        NcclActorSourceProgress result;
        result.source_session = path.source_id;
        if (!path.preparation_sequence) return result;
        const auto source = std::ranges::find(slots, path.source_id, &Slot::connection);
        if (source == slots.end() || source->session != path.source) return result;
        result.status = activate(*source).status;
        if (result.status != CUDA_SUCCESS) return result;
        const auto pending = path.source->pending_preparation(*path.preparation_sequence);
        if (pending.status != CUDA_SUCCESS || !pending.command) {
            result.status =
                pending.status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_HANDLE : pending.status;
            if (result.status != CUDA_ERROR_NOT_READY) {
                nccl_failure = result.status;
                return result;  // Do not repair TLS after a failed exact-owner proof.
            }
            const auto restored = nccl_entry();
            if (restored != CUDA_SUCCESS) result.status = restored;
            return result;
        }
        const auto prepared = path.source->prepare_pending_kernel(pending.command);
        record_prepared_kernel(*source, pending.command, prepared.disposition);
        switch (prepared.disposition) {
            case CudaSessionKernelPreparationDisposition::kPrepared:
            case CudaSessionKernelPreparationDisposition::kRejected:
                // Preparation changed this exact captured prefix; no native
                // command was issued. Its next fresh cut observes any rejection.
                result.status = CUDA_SUCCESS;
                break;
            case CudaSessionKernelPreparationDisposition::kDeferred:
                result.status = CUDA_ERROR_NOT_READY;
                break;
            case CudaSessionKernelPreparationDisposition::kFailed:
            case CudaSessionKernelPreparationDisposition::kUncertain:
                result.status =
                    prepared.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : prepared.status;
                nccl_failure = result.status;
                return result;  // No further SDK work after unknown native ownership.
        }
        harvest_asynchronous_results(*source);
        const auto restored = nccl_entry();
        if (restored != CUDA_SUCCESS) result.status = restored;
        return result;
    }
    NcclActorSourceProgress progress_frontier_leaf(const NativeDependencyPath& path) noexcept {
        if (path.preparation_sequence) return prepare_dependency_leaf(path);
        NcclActorSourceProgress result;
        result.source_session = path.source_id;
        if (!path.frontier) return result;
        const auto source = std::ranges::find(slots, path.source_id, &Slot::connection);
        if (source == slots.end() || source->session != path.source) return result;
        result.status = activate(*source).status;
        if (result.status != CUDA_SUCCESS) return result;
        result.issue = path.is_retirement
                           ? path.source->progress_frontier_retirement(*path.frontier)
                           : path.source->progress_frontier_dependency(*path.frontier);
        result.status =
            result.issue.completion_status != CUDA_SUCCESS          ? result.issue.completion_status
            : result.issue.state == CudaExecutionIssueState::kError ? issue_failure(result.issue)
                                                                    : CUDA_SUCCESS;
        if (result.issue.completion_status != CUDA_SUCCESS ||
            (result.status != CUDA_SUCCESS && result.status != CUDA_ERROR_NOT_READY)) {
            // A failed proof must not trigger another callback to repair TLS.
            nccl_failure = result.status;
            return result;
        }
        const auto restored = nccl_entry();
        if (restored != CUDA_SUCCESS) result.status = restored;
        return result;
    }
    NcclActorSourceProgress progress_nccl_leaf(const NativeDependencyPath& path,
                                               const NcclActorRequestSnapshot& request,
                                               std::uint64_t root_source) noexcept {
        if (path.preparation_sequence) return prepare_dependency_leaf(path);
        return path.frontier
                   ? progress_frontier_leaf(path)
                   : nccl_actor->progress_source(request.request, request.grant, root_source);
    }
    void account_continuation_progress(const NcclActorSourceProgress& progressed) noexcept {
        const auto source = std::ranges::find(slots, progressed.source_session, &Slot::connection);
        const auto audit_kind = source == slots.end()
                                    ? IssueAuditKind::kInitial
                                    : source_issue_audit_kind(*source, progressed.issue.sequence);
        if (progressed.issue.completion_status != CUDA_SUCCESS ||
            (progressed.status != CUDA_SUCCESS && progressed.status != CUDA_ERROR_NOT_READY)) {
            if (source != slots.end()) {
                note_issue(*source, progressed.issue, false, audit_kind);
                policy_failed(*source, progressed.status);
            }
            nccl_failure = progressed.status;
            return;
        }
        if (progressed.issue.state == CudaExecutionIssueState::kRejected) {
            if (source == slots.end() || !source->session ||
                progressed.issue.native_status == CUDA_SUCCESS)
                nccl_failure = CUDA_ERROR_UNKNOWN;
            return;
        }
        if (progressed.issue.state != CudaExecutionIssueState::kIssued) return;
        if (source == slots.end() || !source->session) {
            nccl_failure = CUDA_ERROR_INVALID_HANDLE;
            return;
        }
        const auto epoch = std::ranges::find(source->low_epochs, progressed.issue.sequence,
                                             &Slot::LowEpoch::sequence);
        if (epoch != source->low_epochs.end() && epoch->is_paused &&
            progressed.status == CUDA_SUCCESS && progressed.issue.native_status == CUDA_SUCCESS &&
            progressed.issue.completion_status == CUDA_SUCCESS &&
            progressed.issue.library_status.value_or(0) == 0) {
            epoch->is_paused = epoch->retreat_requested = epoch->retreat_unsupported = false;
            if (source->is_low) audit("low_resumed", *source, progressed.issue.sequence);
        }
        note_issue(*source, progressed.issue, false, audit_kind);
    }
    void advance_continuation(const NcclActorReservation& reservation,
                              const NcclActorContinuationSnapshot& continuation,
                              std::uint64_t policy_id) noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return;
        const auto slot_at = std::ranges::find(slots, continuation.connection, &Slot::connection);
        if (slot_at == slots.end() || slot_at->native_peer != reservation.peer) {
            nccl_failure = CUDA_ERROR_INVALID_HANDLE;
            return;
        }
        auto& slot = *slot_at;
        const auto pending = std::ranges::find_if(slot.requests, [&](const auto& request) {
            return request.ordinary_continuation == continuation.continuation &&
                   request.message.header.sequence == continuation.wire_sequence &&
                   request.message.header.operation == continuation.operation;
        });
        if (!continuation.is_ready && !continuation.dependencies.empty()) {
            const auto source = continuation_progress_source(reservation.peer, continuation);
            if (!source) return;
            const NativeSourceKey root{.peer = reservation.peer, .session = *source};
            const auto path = trace_continuation_dependency(root, continuation);
            if (path.status == CUDA_ERROR_NOT_READY) return;
            if (path.status != CUDA_SUCCESS) {
                nccl_failure = path.status;
                return;
            }
            const auto rechecked = trace_continuation_dependency(root, continuation);
            core::ExecutionPriorityAction observed_action{};
            if (!same_dependency_path(path, rechecked) ||
                !permits_nccl_progress(policy_id, continuation.grant, &observed_action))
                return;
            const auto authorization = audit_native_grant(
                reservation, continuation.grant, policy_id, observed_action,
                NativeAuditTarget::kContinuation, continuation.continuation,
                path.rejection != CUDA_SUCCESS && !path.is_retirement ? NativeAuditMode::kReject
                                                                      : NativeAuditMode::kSource,
                root.session, &path);
            if (path.rejection != CUDA_SUCCESS && !path.is_retirement) {
                // Only the exact still-unentered accepted target can receive
                // this authenticated preparation rejection. Never queue-wide.
                const bool rejected =
                    pending != slot.requests.end() && pending->receipt &&
                    path.source == slot.session &&
                    slot.session->reject_command(*pending->receipt, path.rejection);
                audit_native_result(authorization,
                                    {.status = rejected ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN,
                                     .source_session = path.source_id});
                if (!rejected) nccl_failure = CUDA_ERROR_UNKNOWN;
                return;
            }
            const auto entered = nccl_entry();
            if (entered != CUDA_SUCCESS) {
                audit_native_result(authorization,
                                    {.status = entered, .source_session = path.source_id});
                return;
            }
            const auto progressed = progress_frontier_leaf(path);
            audit_native_result(authorization, progressed);
            account_continuation_progress(progressed);
            return;
        }
        core::ExecutionPriorityAction observed_action{};
        if (!continuation.is_ready || pending == slot.requests.end() || slot.is_retiring ||
            !pending->has_continuation_dependencies ||
            !permits_nccl_progress(policy_id, continuation.grant, &observed_action) ||
            nccl_entry() != CUDA_SUCCESS)
            return;
        if (slot.host_owner && continuation.operation == ExecutionOpcode::kClose &&
            slot.bound_cleanup_parents && !slot.cleanup_retirement) {
            const auto prepared = begin_helper_retirement(slot, &*pending);
            if (prepared != CUDA_SUCCESS) {
                if (prepared != CUDA_ERROR_NOT_READY) policy_failed(slot, prepared);
                return;
            }
        }
        const auto authorization = audit_native_grant(
            reservation, continuation.grant, policy_id, observed_action,
            NativeAuditTarget::kContinuation, continuation.continuation, NativeAuditMode::kHandler);
        const auto admitted =
            nccl_actor->begin_continuation(continuation.continuation, continuation.grant);
        if (admitted.status != CUDA_SUCCESS) {
            audit_native_result(authorization, {.status = admitted.status});
            if (admitted.status != CUDA_ERROR_NOT_READY) nccl_failure = admitted.status;
            return;
        }
        has_continuation_attempt = true;
        // Every armed attempt finishes exactly once, even if reply construction
        // throws after an original call. No policy/actor reentry while armed.
        NcclActorContinuationOutcome outcome;
        try {
            std::optional<ExecutionWireReply> reply;
            if (slot.session && !slot.is_metadata_owner() && !slot.is_bootstrap_pending &&
                slot.session->health_snapshot().has_context) {
                const auto activated = activate(slot);
                if (activated.status != CUDA_SUCCESS) {
                    policy_failed(slot, activated.status);
                    reply = rejection(activated.status);
                }
            }
            if (!reply) reply = execute(slot, *pending, true, true);
            outcome = continuation_outcome(slot, *pending, std::move(reply));
            if (outcome.kind != NcclActorContinuationOutcomeKind::kUncertain) {
                const auto restored = nccl_entry();
                if (restored != CUDA_SUCCESS) {
                    outcome.kind = NcclActorContinuationOutcomeKind::kUncertain;
                    outcome.status = restored;
                    outcome.pending_lifetime.reset();
                    outcome.pending_sequence = 0;
                }
            }
        } catch (const std::bad_alloc&) {
            outcome.kind = NcclActorContinuationOutcomeKind::kUncertain;
            outcome.status = CUDA_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            outcome.kind = NcclActorContinuationOutcomeKind::kUncertain;
            outcome.status = CUDA_ERROR_UNKNOWN;
        }
        const bool keeps_fence =
            outcome.kind == NcclActorContinuationOutcomeKind::kDeferred ||
            outcome.kind == NcclActorContinuationOutcomeKind::kHostJobPending ||
            outcome.kind == NcclActorContinuationOutcomeKind::kUncertain;
        audit_handler_outcome(authorization, admitted.permit, outcome);
        const auto finished =
            nccl_actor->finish_continuation_attempt(admitted.permit, std::move(outcome));
        has_continuation_attempt = false;
        if (finished != CUDA_SUCCESS) {
            nccl_failure = finished;
            return;
        }
        if (!keeps_fence) release_continuation_fences(continuation.continuation);
    }
    void advance_nccl() noexcept {
        if (memory_native_health() != CUDA_SUCCESS) return;
        if (!nccl_actor || nccl_stopped) return;
        const auto snapshot = nccl_state();
        if (snapshot.status != CUDA_SUCCESS) return;
        for (const auto& reservation : snapshot.reservations) {
            const auto chosen = native_progress_grant(reservation);
            if (!chosen) continue;
            const auto continuation = std::ranges::find_if(
                reservation.continuations, [&](const auto& item) { return item.grant == *chosen; });
            if (continuation != reservation.continuations.end()) {
                const auto policy = std::ranges::find(native_policies, reservation.reservation,
                                                      &NativePolicy::reservation);
                if (policy == native_policies.end() ||
                    policy->generation != reservation.generation) {
                    nccl_failure = CUDA_ERROR_UNKNOWN;
                    return;
                }
                advance_continuation(reservation, *continuation, policy->identity);
                observe_nccl();
                continue;
            }
            const auto selected = std::ranges::find_if(
                reservation.requests, [&](const auto& item) { return item.grant == *chosen; });
            if (selected == reservation.requests.end()) continue;
            const auto policy = std::ranges::find(native_policies, reservation.reservation,
                                                  &NativePolicy::reservation);
            if (policy == native_policies.end() || policy->generation != reservation.generation) {
                nccl_failure = CUDA_ERROR_UNKNOWN;
                return;
            }
            const auto policy_id = policy->identity;
            // Rebuild after each native transition. No saved connection-wide
            // grant survives another batch's new reservation or uncertainty.
            const auto plan = priority_plan();
            if (!plan.is_valid) return;
            const auto decision =
                std::ranges::find(std::span{plan.decisions}.first(plan.size), policy_id,
                                  &core::ExecutionPriorityDecision::session);
            if (decision == std::span{plan.decisions}.first(plan.size).end()) return;
            const core::ExecutionPriorityProgress exact{policy_id, selected->grant.generation,
                                                        selected->grant.cutoff};
            if (decision->action != core::ExecutionPriorityAction::kExecute &&
                !(decision->action == core::ExecutionPriorityAction::kExecuteProgress &&
                  decision->progress == exact))
                continue;
            // CPU snapshots are stage-specific capabilities. The actor rechecks
            // the complete grant before entering either source or native lane.
            const auto current = nccl_actor->snapshot();
            if (current.status != CUDA_SUCCESS) {
                nccl_failure = current.status;
                return;
            }
            const auto batch = std::ranges::find(current.reservations, reservation.reservation,
                                                 &NcclActorReservation::reservation);
            if (batch == current.reservations.end() || batch->generation != reservation.generation)
                continue;
            const auto request = std::ranges::find(batch->requests, selected->request,
                                                   &NcclActorRequestSnapshot::request);
            if (request == batch->requests.end() || request->grant != selected->grant) continue;
            if (nccl_entry() != CUDA_SUCCESS) return;
            if (request->stage == NcclActorStage::kSourceBlocked && !request->is_ready) {
                const auto dependency = native_progress_source(reservation.peer, *request);
                if (!dependency) continue;
                const NativeSourceKey root{.peer = reservation.peer, .session = *dependency};
                const auto path = trace_nccl_dependency(root, *request);
                if (path.status == CUDA_ERROR_NOT_READY) continue;
                if (path.status != CUDA_SUCCESS) {
                    nccl_failure = path.status;
                    return;
                }
                // Both traversals are CPU-only. Reauthenticate every captured
                // edge and original stage, then rebuild Core immediately before
                // the single leaf transition. No leaf receives its own grant.
                const auto rechecked = trace_nccl_dependency(root, *request);
                core::ExecutionPriorityAction observed_action{};
                if (!same_dependency_path(path, rechecked) ||
                    !permits_nccl_progress(policy_id, request->grant, &observed_action))
                    continue;
                const auto authorization =
                    audit_native_grant(reservation, request->grant, policy_id, observed_action,
                                       NativeAuditTarget::kRequest, request->request,
                                       path.rejection != CUDA_SUCCESS ? NativeAuditMode::kReject
                                                                      : NativeAuditMode::kSource,
                                       *dependency, &path);
                if (path.rejection != CUDA_SUCCESS) {
                    // The actor repeats the exact root-source inspection. No
                    // caller-supplied code or leaf error mints clean authority.
                    const auto rejected = nccl_actor->reject_source(
                        request->request, request->grant, *dependency, path.rejection);
                    audit_native_result(authorization,
                                        {.status = rejected, .source_session = path.source_id});
                    if (rejected != CUDA_SUCCESS) {
                        nccl_failure = rejected;
                        return;
                    }
                    observe_nccl();
                    continue;
                }
                const auto progressed = progress_nccl_leaf(path, *request, *dependency);
                audit_native_result(authorization, progressed);
                const auto source =
                    std::ranges::find(slots, progressed.source_session, &Slot::connection);
                const auto audit_kind =
                    source == slots.end()
                        ? IssueAuditKind::kInitial
                        : source_issue_audit_kind(*source, progressed.issue.sequence);
                if (progressed.issue.completion_status != CUDA_SUCCESS ||
                    (progressed.status != CUDA_SUCCESS &&
                     progressed.status != CUDA_ERROR_NOT_READY)) {
                    if (source != slots.end()) {
                        note_issue(*source, progressed.issue, false, audit_kind);
                        policy_failed(*source, progressed.status);
                    }
                    // The exact issue survives a subsequent proof failure, but
                    // it never grants another native callback to repair context.
                    observe_nccl();
                    return;
                }
                if (progressed.issue.state == CudaExecutionIssueState::kRejected) {
                    if (source == slots.end() || !source->session ||
                        progressed.issue.native_status == CUDA_SUCCESS) {
                        nccl_failure = CUDA_ERROR_UNKNOWN;
                        return;
                    }
                    // The immutable root cut retains this rejected predecessor
                    // after record erasure. A fresh grant/inspection consumes its
                    // settled proof on the next pass; this result alone cannot
                    // authorize cancellation of a different source's turn.
                    observe_nccl();
                    continue;
                }
                if (progressed.issue.state == CudaExecutionIssueState::kIssued) {
                    if (source == slots.end() || !source->session) {
                        nccl_failure = CUDA_ERROR_INVALID_HANDLE;
                        return;
                    }
                    const auto epoch = std::ranges::find(
                        source->low_epochs, progressed.issue.sequence, &Slot::LowEpoch::sequence);
                    if (epoch != source->low_epochs.end() && epoch->is_paused &&
                        progressed.status == CUDA_SUCCESS &&
                        progressed.issue.native_status == CUDA_SUCCESS &&
                        progressed.issue.completion_status == CUDA_SUCCESS &&
                        progressed.issue.library_status.value_or(0) == 0) {
                        epoch->is_paused = epoch->retreat_requested = epoch->retreat_unsupported =
                            false;
                        if (source->is_low)
                            audit("low_resumed", *source, progressed.issue.sequence);
                    }
                    // Both progress paths already restored proven NULL. Epoch
                    // publication is CPU-only; normal observation owns polling.
                    note_issue(*source, progressed.issue, false, audit_kind);
                }
            } else {
                const auto authorization = audit_native_grant(
                    reservation, request->grant, policy_id, decision->action,
                    NativeAuditTarget::kRequest, request->request, NativeAuditMode::kNative);
                const auto issued = nccl_actor->issue(request->request, request->grant);
                if (issued == CUDA_SUCCESS) priority_idle.note_issue(reservation.priority);
                audit_native_result(authorization, {.status = issued});
            }
            observe_nccl();
        }
        if (is_stopping || (config.exit_when_idle && has_client && slots.empty())) {
            (void)nccl_actor->close();
            // Only a fully empty, healthy actor may execute native cleanup.
            // Live/unknown native TLS is never stopped or reset by disconnect.
            const auto state = nccl_actor->snapshot();
            if (!nccl_failure && state.status == CUDA_SUCCESS && state.reservations.empty() &&
                (!nccl_driver_ready || nccl_entry() == CUDA_SUCCESS) &&
                nccl_actor->shutdown() == CUDA_SUCCESS)
                nccl_stopped = true;
        }
    }
#endif

    bool permits_quarantined_cleanup(const Slot& target) const noexcept {
        if (target.host_owner || target.is_metadata_owner() || target.is_nccl ||
            target.policy_failure || !target.session->can_cleanup_quarantined())
            return false;
#ifdef GLIMMER_HAS_NCCL
        if (nccl_actor) {
            const auto state = nccl_actor->snapshot();
            if (nccl_failure || state.status != CUDA_SUCCESS ||
                std::ranges::any_of(state.reservations, [](const auto& item) {
                    return item.has_physical_work || item.has_unknown_state;
                }))
                return false;
        }
#endif
        for (const auto& peer : slots) {
            if (&peer == &target || peer.is_metadata_owner() || peer.is_nccl) continue;
            const auto host = peer.host_owner ? peer.host_owner->snapshot() : HostOwnerSnapshot{};
            const auto state = peer.host_owner
                                   ? static_cast<CudaSessionSchedulingSnapshot>(host.session)
                                   : peer.session->scheduling_snapshot();
            if (host.has_reservation || has_physical_reservation(state) || peer.policy_failure ||
                (state.is_quarantined &&
                 (peer.host_owner || !peer.session->can_cleanup_quarantined())))
                return false;
        }
        return true;
    }
    void request_priority_retreat(Slot& slot, const core::ExecutionPriorityPlan& plan) noexcept {
        if (!plan.is_valid || slot.is_metadata_owner() || slot.low_epochs.empty() ||
            !std::ranges::any_of(
                std::span{plan.decisions}.first(plan.size), [&](const auto& decision) {
                    return decision.session == slot.policy_id &&
                           decision.action == core::ExecutionPriorityAction::kRequestRetreat;
                }))
            return;
        const auto activated = activate(slot);
        for (auto& epoch : slot.low_epochs) {
            if (epoch.is_event || !epoch.is_issued || epoch.is_paused || epoch.retreat_requested ||
                epoch.retreat_unsupported || epoch.frontier.retirement().is_retired)
                continue;
            if (arrival_checkpoint && plan.selected_priority == slot.priority) {
                const auto& checkpoint = *arrival_checkpoint;
                const auto legs = std::span{checkpoint.legs}.first(checkpoint.size);
                const auto target =
                    std::ranges::find(legs, slot.policy_id, &core::ExecutionRebalanceLeg::session);
                const auto progress = slot.session->persistent_progress(epoch.sequence);
                if (target == legs.end() || target->physical.sequence != epoch.sequence ||
                    !progress || progress->physical_launches != target->physical.generation ||
                    progress->physical_stops >= target->physical.generation)
                    continue;
            }
            const auto status = activated.status == CUDA_SUCCESS
                                    ? slot.session->request_retreat(epoch.sequence)
                                    : activated.status;
            if (status == CUDA_SUCCESS) {
                epoch.retreat_requested = true;
                audit("retreat_requested", slot, epoch.sequence);
            } else if (status == CUDA_ERROR_NOT_SUPPORTED) {
                epoch.retreat_unsupported = true;
                audit("nonpreemptible_wait", slot, epoch.sequence);
            } else if (status == CUDA_ERROR_NOT_READY) {
                // Native source host exclusion is not a failed retreat or
                // permission to destroy/reject its queued producers.
                continue;
            } else {
                policy_failed(slot, status);
                break;
            }
        }
    }
    void request_received_priority_retreats() noexcept {
        for (auto& slot : slots) {
            // A previous control can fail a peer. Never reuse its old authority
            // for the next owner, and never infer physical stop from intake.
            if (!slot.low_epochs.empty())
                request_priority_retreat(slot, priority_plan(HostStage::kPlanReceive));
        }
    }
    HostStageMetrics::SlotPlanReason slot_plan_reason(const Slot& slot) const noexcept {
        using Reason = HostStageMetrics::SlotPlanReason;
        if (!config.should_skip_idle_helper_plans || highest_priority ||
            !receives_priority_before_observation() || priority_idle.enabled() || config.shared)
            return Reason::kDisabledPolicy;
        if (!slot.host_owner || slot.is_nccl || slot.is_metadata_owner()) return Reason::kNonHelper;
        if (!slot.requests.empty() || slot.is_retiring || slot.is_retirement_started ||
            slot.is_closing || slot.is_bootstrap_pending || slot.admission_fence ||
            slot.is_paired_member || slot.policy_failure || slot.asynchronous_failure ||
            !slot.low_epochs.empty() || !slot.host_functions.empty() ||
            !slot.asynchronous_receipts.empty() || !slot.frontiers.empty() ||
            slot.bound_cleanup_parents)
            return Reason::kLocalObligation;
        // No request, retreat or retirement consumes this helper's action, and
        // schedule(host_owner) is a no-op. Other plans still observe its coherent
        // reservation/quarantine, including outstanding external business leases.
        return Reason::kSkippedHelper;
    }
    bool advance() noexcept {
        const HostScope host_scope(config.host_stage_metrics.get(), HostStage::kAdvance);
        if (graph_construction) return advance_graph_construction();
        if (uses_priority_space_share()) space_share_issued.fill(false);
        if (const auto health = memory_native_health(); health != CUDA_SUCCESS)
            for (auto& slot : slots)
                if (slot.session && !slot.is_metadata_owner() && !slot.host_owner && !slot.is_nccl)
                    // Only CPU-known unissued entries can fail here. Issued or
                    // paused work retains its actual physical retirement proof.
                    (void)slot.session->reject_queued(health);
        observe();
        observe_arrival_checkpoint();
        observe_serial_pair();
        observe_profile_pair();
#ifdef GLIMMER_HAS_NCCL
        observe_nccl();
#endif
        for (std::size_t index = 0; index < slots.size();) {
            if (graph_construction) return true;
            auto& slot = slots[index];
#ifdef GLIMMER_HAS_NCCL
            if (slot.host_owner) finish_helper_retirement(slot);
#endif
            auto action = core::ExecutionPriorityAction::kExecute;
            bool skips_local_plan{};
            if (has_scheduling_policy() && !slot.is_metadata_owner()) {
                const auto reason = slot_plan_reason(slot);
                skips_local_plan = reason == HostStageMetrics::SlotPlanReason::kSkippedHelper;
                if (config.host_stage_metrics) config.host_stage_metrics->slot_plan(reason);
            }
            if (has_scheduling_policy() && !slot.is_metadata_owner() && !skips_local_plan) {
                // Earlier SDK calls can fail or retire a peer in this pass.
                // Rebuild from current observations instead of replaying a plan.
                const auto plan = priority_plan(HostStage::kPlanSlot);
                action = core::ExecutionPriorityAction::kQuarantined;
                if (plan.is_valid)
                    for (std::size_t item = 0; item < plan.size; ++item)
                        if (plan.decisions[item].session == slot.policy_id)
                            action = plan.decisions[item].action;
                if (action == core::ExecutionPriorityAction::kRequestRetreat)
                    request_priority_retreat(slot, plan);
            }
            if (!skips_local_plan && action == core::ExecutionPriorityAction::kExecute &&
                !slot.is_metadata_owner() && !slot.admission_fence && !slot.is_paired_member &&
                !uses_priority_space_share())
                schedule(slot);
            if (slot.is_retiring) {
#ifdef GLIMMER_HAS_NCCL
                retire_continuation_requests(slot);
#endif
#ifdef GLIMMER_HAS_NCCL
                if (slot.is_nccl) {
                    for (std::size_t pending = 0; pending < slot.requests.size();) {
                        const auto id = slot.requests[pending].nccl_request;
                        if (id && !nccl_actor->reply(id).ready) {
                            ++pending;
                            continue;
                        }
                        if (id) {
                            const auto status = nccl_actor->release_reply(id);
                            if (status != CUDA_SUCCESS) {
                                nccl_failure = status;
                                ++pending;
                                continue;
                            }
                        }
                        slot.requests.erase(slot.requests.begin() +
                                            static_cast<std::ptrdiff_t>(pending));
                    }
                    if (slot.requests.empty()) {
                        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                        continue;
                    }
                    ++index;
                    continue;
                }
#endif
                if (memory_native_health() != CUDA_SUCCESS) {
                    ++index;
                    continue;
                }
                if (slot.is_pointer_metadata) {
                    const auto stopped = slot.session->shutdown();
                    if (stopped.disposition == CudaSessionDisposition::kComplete &&
                        stopped.status == CUDA_SUCCESS) {
                        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                        continue;
                    }
                    ++index;
                    continue;
                }
#ifdef GLIMMER_HAS_CUDNN
                if (slot.dnn_metadata) {
                    const auto stopped = slot.dnn_metadata->shutdown();
                    if (stopped.disposition == CudaDnnDisposition::kComplete &&
                        stopped.status == CUDA_SUCCESS && slot.dnn_metadata->snapshot().is_closed) {
                        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                        continue;
                    }
                    ++index;
                    continue;
                }
#endif
                if (slot.lt_metadata) {
                    const auto stopped = slot.lt_metadata->shutdown();
                    if (stopped.disposition == CudaLtDisposition::kComplete &&
                        stopped.status == CUDA_SUCCESS && slot.lt_metadata->snapshot().is_closed) {
                        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                        continue;
                    }
                    ++index;
                    continue;
                }
                if (slot.host_owner) {
                    retire(slot);
                    if (!slot.is_retirement_started || (slot.host_owner_leases.has_leases()
#ifdef GLIMMER_HAS_NCCL
                                                        && !slot.cleanup_retirement
#endif
                                                        )) {
                        ++index;
                        continue;
                    }
                    if (slot.host_owner->snapshot().is_closed) {
#ifdef GLIMMER_HAS_NCCL
                        finish_helper_retirement(slot);
                        if (slot.cleanup_retirement) {
                            ++index;
                            continue;
                        }
#endif
                        slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                        continue;
                    }
                    if (action == core::ExecutionPriorityAction::kExecute &&
                        memory_native_health() == CUDA_SUCCESS)
                        slot.host_owner->permit_shutdown();
                    ++index;
                    continue;
                }
                if (slot.session_leases.has_leases()) {
                    ++index;
                    continue;
                }
                retire(slot);
                const auto snapshot = slot.session->scheduling_snapshot();
                const bool can_cleanup = permits_quarantined_cleanup(slot);
                if ((has_scheduling_policy() && !snapshot.has_context && snapshot.is_quarantined) ||
                    snapshot.submissions.outstanding ||
                    (has_scheduling_policy() && snapshot.has_context &&
                     action != core::ExecutionPriorityAction::kExecute && !can_cleanup)) {
                    ++index;
                    continue;
                }
                if (snapshot.has_context && activate(slot).status != CUDA_SUCCESS) {
                    ++index;
                    continue;
                }
                const auto stopped = slot.session->shutdown();
                if (stopped.disposition == CudaSessionDisposition::kComplete &&
                    stopped.status == CUDA_SUCCESS) {
                    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index));
                    continue;
                }
                ++index;
                continue;
            }
            bool has_unaccepted_command{};
            for (std::size_t request_index = 0; request_index < slot.requests.size();) {
                auto& request = slot.requests[request_index];
                const auto operation = request.message.header.operation;
                const bool is_profile_control = is_dnn_profile_control(slot, request);
                if (!is_profile_control && !permits_profile_request(slot, request)) {
                    has_unaccepted_command = true;
                    ++request_index;
                    continue;
                }
                const bool cleanup_only =
                    operation == ExecutionOpcode::kClose && permits_quarantined_cleanup(slot);
                const bool reply_only = (is_execution_command(operation) && request.receipt) ||
                                        (slot.is_nccl && request.nccl_request) ||
                                        request.has_host_job || request.has_host_shutdown ||
                                        (slot.host_owner && operation == ExecutionOpcode::kClose &&
                                         slot.host_owner->snapshot().is_closed);
                const bool asynchronous_admission = is_asynchronous_admission(request);
                const bool can_issue =
                    memory_native_health() == CUDA_SUCCESS &&
                    action == core::ExecutionPriorityAction::kExecute && !slot.policy_failure &&
                    (!has_scheduling_policy() || !slot.snapshot().is_quarantined) &&
                    (config.scheduling_mode != core::ExecutionSchedulingMode::kProfile || [&] {
                        const auto plan = priority_plan();
                        return plan.is_valid &&
                               std::ranges::any_of(
                                   std::span{plan.decisions}.first(plan.size),
                                   [&](const auto& decision) {
                                       return decision.session == slot.policy_id &&
                                              decision.action ==
                                                  core::ExecutionPriorityAction::kExecute;
                                   });
                    }());
                const bool is_ordered =
                    is_profile_control || !has_unaccepted_command ||
                    (!is_execution_command(operation) && !is_ordering_command(operation) &&
                     operation != ExecutionOpcode::kCaptureFrontier &&
                     operation != ExecutionOpcode::kGraphControl);
                try {
                    if (request.continuation) {
                        if (!has_unaccepted_command) {
#ifdef GLIMMER_HAS_NCCL
                            const bool replied = observe_continuation(slot, request);
#else
                            auto rejected = rejection(CUDA_ERROR_NOT_SUPPORTED);
                            if (slot.is_bootstrap_pending) {
                                slot.is_bootstrap_pending = false;
                                describe_bootstrap(rejected, ExecutionBootstrapOwnership::kNone);
                            }
                            const bool replied = queue_reply(slot, request, rejected);
#endif
                            if (slot.is_retiring) break;
                            if (replied) {
                                slot.requests.erase(slot.requests.begin() +
                                                    static_cast<std::ptrdiff_t>(request_index));
                                continue;
                            }
                        }
                        has_unaccepted_command =
                            has_unaccepted_command ||
                            ((is_execution_command(operation) || is_ordering_command(operation)) &&
                             !request.receipt);
                        ++request_index;
                        continue;
                    }
                    if (slot.admission_fence && !is_profile_control && !reply_only &&
                        operation != ExecutionOpcode::kSnapshot &&
                        !is_host_function_observation(operation)) {
                        has_unaccepted_command = has_unaccepted_command ||
                                                 is_execution_command(operation) ||
                                                 is_ordering_command(operation);
                        ++request_index;
                        continue;
                    }
                    if (is_ordered &&
                        (is_profile_control || slot.is_metadata_owner() ||
                         (slot.is_nccl && slot.is_initialized &&
                          operation == ExecutionOpcode::kNcclCall) ||
                         is_metadata(request.message) || reply_only || asynchronous_admission ||
                         can_issue || cleanup_only || memory_native_health() != CUDA_SUCCESS)) {
                        if (!is_profile_control && !slot.host_owner && !slot.is_metadata_owner() &&
                            !slot.is_nccl && !slot.is_bootstrap_pending &&
                            !is_metadata(request.message) && !asynchronous_admission &&
                            slot.session->health_snapshot().has_context &&
                            (can_issue || cleanup_only)) {
                            const auto activated = activate(slot);
                            if (activated.status != CUDA_SUCCESS) {
                                if (slot.is_paired_member &&
                                    activated.status == CUDA_ERROR_NOT_READY &&
                                    !activated.original_status) {
                                    has_unaccepted_command = true;
                                    ++request_index;
                                    continue;
                                }
                                policy_failed(slot, activated.status);
                                break;
                            }
                        }
                        const auto reply = execute(slot, request, can_issue);
                        if (slot.is_retiring) break;
                        if (reply) {
                            if (!queue_reply(slot, request, *reply)) break;
                            slot.requests.erase(slot.requests.begin() +
                                                static_cast<std::ptrdiff_t>(request_index));
                            if (graph_construction) return true;
                            continue;
                        }
                    }
                } catch (...) {
                    drop(slot);
                    break;
                }
                has_unaccepted_command =
                    has_unaccepted_command ||
                    ((is_execution_command(operation) || is_ordering_command(operation)) &&
                     !request.receipt);
                ++request_index;
            }
            ++index;
        }
        if (graph_construction) return true;
#ifdef GLIMMER_HAS_NCCL
        advance_nccl();
#endif
        if (!uses_priority_space_share())
            for (auto& slot : slots) prepare_asynchronous_kernel(slot);
        begin_arrival_checkpoint();
        // Shared candidates must see all admissions from this bounded turn,
        // instead of the first slot issuing before its partner is even visited.
        if (uses_priority_space_share()) {
            struct Candidate {
                std::uint64_t policy{};
                std::uint32_t priority{};
            };
            std::array<Candidate, core::kMaxExecutionPriorityPeers> order{};
            std::size_t count{};
            for (const auto& slot : slots) {
                if (!slot.session || slot.host_owner || slot.is_nccl || slot.is_metadata_owner() ||
                    slot.admission_fence || (slot.is_retiring && !slot.is_paired_member))
                    continue;
                if (count == order.size()) return false;
                order[count++] = {slot.policy_id, slot.priority};
            }
            auto selected = std::span(order).first(count);
            std::ranges::sort(selected, [](const auto& left, const auto& right) {
                return left.priority != right.priority ? left.priority > right.priority
                                                       : left.policy < right.policy;
            });
            for (const auto& candidate : selected) {
                const auto slot = std::ranges::find(slots, candidate.policy, &Slot::policy_id);
                if (slot != slots.end()) schedule(*slot);
            }
        } else if (!schedule_pending_pair()) {
            for (auto& slot : slots) {
                // Existing paired members may need to resume accepted work
                // while disconnect cleanup is draining their physical tails.
                if (slot.is_paired_member) {
                    if (!slot.admission_fence) schedule(slot);
                    continue;
                }
                if (slot.admission_fence || slot.host_owner || slot.is_nccl ||
                    slot.is_metadata_owner() || slot.is_retiring)
                    continue;
                // Legacy nonpaired work already uses its existing request and
                // continuation turns. The added post-admission pass must not
                // activate unrelated idle or quarantined native contexts.
                if (!std::ranges::any_of(slot.low_epochs,
                                         [](const auto& epoch) { return epoch.is_asynchronous; }))
                    continue;
                schedule(slot);  // Recaptures fresh complete Core peers per exact issue.
            }
        }
        observe_process_memory();
        if (config.exit_when_idle && has_client && slots.empty()) (void)stop_execution_domain();
        return true;
    }
};

ExecutionService::ExecutionService(ExecutionServiceConfig config, CudaSessionDispatch dispatch)
    : implementation_(std::make_unique<Implementation>(std::move(config), std::move(dispatch))) {}
ExecutionService::ExecutionService(std::shared_ptr<ExecutionSocketPollJournal> journal,
                                   ExecutionServiceConfig config, CudaSessionDispatch dispatch)
    : implementation_(std::make_unique<Implementation>(std::move(config), std::move(dispatch),
                                                       require_poll_journal(std::move(journal)))) {}
#ifdef GLIMMER_HAS_NCCL
ExecutionService::ExecutionService(ExecutionServiceConfig config, CudaSessionDispatch dispatch,
                                   ExecutionNcclDispatch nccl)
    : implementation_(std::make_unique<Implementation>(std::move(config), std::move(dispatch),
                                                       nullptr, nccl)) {}
ExecutionService::ExecutionService(std::shared_ptr<ExecutionSocketPollJournal> journal,
                                   ExecutionServiceConfig config, CudaSessionDispatch dispatch,
                                   ExecutionNcclDispatch nccl)
    : implementation_(std::make_unique<Implementation>(
          std::move(config), std::move(dispatch), require_poll_journal(std::move(journal)), nccl)) {
}
#endif
ExecutionService::~ExecutionService() {
    if (!implementation_->owns()) {
        (void)implementation_.release();
        return;
    }
    if (!request_stop()) {
        try {
            spdlog::critical("Execution service retains unproven native sessions at exit");
        } catch (...) {
        }
        (void)implementation_.release();
    }
}
bool ExecutionService::start() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_started || !random_incarnation(state.incarnation) ||
        !state.transport.start())
        return false;
    state.is_started = true;
    (void)state.semantic(2, [&](auto& json) {
        json.add("event", 1);
        json.add("pid", state.process);
        json.add("incarnation0", state.incarnation[0]);
        json.add("incarnation1", state.incarnation[1]);
        if (state.config.should_audit_native) json.add("native_audit_schema", 2);
    });
    state.start_sgemm_audit();
#ifdef GLIMMER_HAS_NCCL
    if (state.nccl_actor &&
        (state.config.should_audit_native || state.config.should_audit_priority) &&
        !state.nccl_actor->set_observer({&state, Implementation::actor_semantic}))
        state.semantic_lost(3);
#endif
    return true;
}
bool ExecutionService::step(std::uint32_t wait_ms) noexcept {
    auto& state = *implementation_;
    const HostScope host_scope(state.config.host_stage_metrics.get(), HostStage::kStep);
    if (!state.owns() || !state.is_started || state.is_stopping || state.is_active ||
        wait_ms > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        return false;
    const Implementation::ActiveScope scope(state.is_active);
    auto wait_ns = static_cast<std::uint64_t>(wait_ms) * 1000000;
    if (state.priority_idle.has_idle_window()) {
        const auto remaining = state.priority_idle.remaining_ns(state.priority_now_ns());
        wait_ns = std::min(wait_ns, remaining);
    }
    // GPU completion does not make a transport descriptor readable. Do not
    // postpone physical polling (or block publication) while this owner has
    // actual native work in flight. This CPU hint grants no execution or stop
    // authority; separately threaded host owners retain their wake contract.
    if (std::ranges::any_of(state.slots, [](const auto& slot) {
            if (!slot.session || slot.host_owner) return false;
            const auto snapshot = slot.session->scheduling_snapshot();
            return snapshot.has_context && !snapshot.is_quarantined &&
                   (snapshot.submissions.issuing || snapshot.submissions.awaiting_physical_stop);
        }))
        wait_ns = 0;
    {
        const HostScope transport_scope(state.config.host_stage_metrics.get(),
                                        HostStage::kTransport);
        if (!state.transport.poll_for_ns(wait_ns)) return false;
    }
    try {
        if (!state.graph_construction && !state.graph_deferred_intake.empty()) {
            auto deferred = std::move(state.graph_deferred_intake);
            state.graph_deferred_intake.clear();
            for (auto& request : deferred) {
                const auto connection = request.connection;
                try {
                    state.receive(std::move(request));
                } catch (...) {
                    state.transport.close_peer(connection);
                }
            }
        }
        const bool receives_priority_first = state.receives_priority_before_observation();
        if (receives_priority_first) {
            // Preserve disconnected-owner fencing before accepting new demand,
            // without spending its retreat window on physical completion queries.
            for (auto& slot : state.slots)
                if (!state.transport.has_peer(slot.connection)) state.retire(slot);
        } else {
            state.observe();
        }
        bool has_received_request{};
        if (state.config.should_receive_async_burst) state.async_intake_peers.clear();
        for (std::size_t count = 0; count < state.config.transport.max_peers; ++count) {
            auto request = state.transport.take_request();
            if (!request) break;
            has_received_request = true;
            const auto connection = request->connection;
            try {
                const auto sequence = request->message.header.sequence;
                const bool can_burst =
                    state.config.should_receive_async_burst &&
                    state.can_receive_async_burst(
                        {connection, request->peer, request->message.header,
                         request->message.payload, request->descriptor.get() >= 0});
                const HostScope receive_scope(state.config.host_stage_metrics.get(),
                                              HostStage::kReceive);
                state.receive(std::move(*request));
                if (can_burst) {
                    const auto slot = std::ranges::find(state.slots, connection,
                                                        &Implementation::Slot::connection);
                    if (slot != state.slots.end() && !slot->is_retiring &&
                        slot->requests.size() == 1 &&
                        slot->requests.front().message.header.sequence == sequence)
                        state.async_intake_peers.push_back({connection, sequence});
                }
            } catch (...) {
                state.transport.close_peer(connection);
            }
        }
        if (state.config.should_receive_async_burst && !state.receive_async_burst()) return false;
        if (receives_priority_first && has_received_request && !state.graph_construction)
            state.request_received_priority_retreats();
        // Physical observation and all high-issue fences remain in advance().
        return state.advance();
    } catch (...) {
        return false;
    }
}
bool ExecutionService::finished() const noexcept {
    const auto& state = *implementation_;
    const bool finished = state.owns() && state.config.exit_when_idle && state.has_client &&
                          state.slots.empty() && !state.has_process_memory() &&
                          (!state.execution_domain() || state.execution_domain_stopped)
#ifdef GLIMMER_HAS_NCCL
                          && (!state.nccl_actor || state.nccl_stopped)
#endif
        ;
    if (finished && state.audit_log) state.audit_log->flush();
    return finished;
}
bool ExecutionService::audit_healthy() const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.has_semantic_loss &&
           (!state.audit_log || (state.is_semantic_closed && !state.audit_log->lost()));
}
std::optional<std::uint64_t> ExecutionService::async_intake_second_takes() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.config.should_receive_async_burst) return std::nullopt;
    return state.async_intake_second_takes;
}
std::optional<ExecutionBlockCompletionObservation> ExecutionService::block_completion_observation(
    const control::TaskPeerIdentity& peer) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || state.is_active || !state.is_started || state.is_stopping ||
        !state.config.persistent.observes_block_completion)
        return std::nullopt;
    std::optional<ExecutionBlockCompletionObservation> result;
    for (const auto& slot : state.slots) {
        if (slot.peer != peer || !slot.session || slot.host_owner || slot.is_metadata_owner() ||
            slot.is_nccl || slot.policy_failure || slot.asynchronous_failure || slot.is_retiring ||
            slot.session->health_snapshot().is_quarantined)
            continue;
        for (const auto& epoch : slot.low_epochs) {
            if (epoch.is_event || !epoch.is_issued || !epoch.was_successful || epoch.is_paused ||
                epoch.retreat_requested || epoch.retreat_unsupported)
                continue;
            const auto progress = slot.session->persistent_progress(epoch.sequence);
            if (!progress || !progress->observes_block_completion ||
                !progress->has_completed_block || progress->physical_launches != 1 ||
                progress->physical_stops || progress->completed_blocks || !progress->total_blocks)
                continue;
            if (result) return std::nullopt;  // Ambiguity cannot supply another command's witness.
            result =
                ExecutionBlockCompletionObservation{slot.connection, epoch.sequence, *progress};
        }
    }
    return result;
}
bool ExecutionService::publish_block_completion_audit(
    const control::TaskPeerIdentity& peer, std::uint32_t priority,
    const ExecutionBlockCompletionObservation& observation) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_active || !state.is_started || state.is_stopping ||
        !state.audit_log || !state.config.persistent.observes_block_completion)
        return false;
    const auto pinned = std::ranges::find_if(state.config.client_priorities, [&](const auto& pin) {
        return pin.peer == peer && pin.priority == priority;
    });
    const auto current = block_completion_observation(peer);
    if (pinned == state.config.client_priorities.end() || !current ||
        current->session != observation.session || current->sequence != observation.sequence)
        return false;
    const auto& progress = current->progress;
    if (!progress.workers) return false;
    const auto recorded = state.audit_log->info(
        "Block completion audit: v=1, session={}, sequence={}, priority={}, pid={}, "
        "start={}, uid={}, total_blocks={}, physical_launches=1, physical_stops=0, "
        "stopped_completed_blocks=0, workers={}, has_completed_block=true",
        current->session, current->sequence, priority, peer.pid, peer.start_time_ticks, peer.uid,
        progress.total_blocks, progress.workers);
    if (!recorded) state.has_semantic_loss = true;
    return recorded;
}
bool ExecutionService::request_stop() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_active) return false;
    const Implementation::ActiveScope scope(state.is_active);
    state.is_stopping = true;
    state.transport.stop();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!state.slots.empty() || state.has_process_memory()
#ifdef GLIMMER_HAS_NCCL
           || (state.nccl_actor && !state.nccl_stopped)
#endif
    ) {
        (void)state.advance();
        if (state.slots.empty() && !state.has_process_memory()
#ifdef GLIMMER_HAS_NCCL
            && (!state.nccl_actor || state.nccl_stopped)
#endif
        )
            break;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    if (!state.stop_execution_domain()) return false;
    if (!state.is_semantic_closed) {
        // Drain prior diagnostics before the terminal marker so a propagated
        // sink failure cannot publish an otherwise healthy semantic closure.
        if (state.audit_log) state.audit_log->flush();
        if (const auto& target = state.config.sgemm_audit_target) {
            if (target->client && !state.sgemm_audit_session) state.has_semantic_loss = true;
            if (!state.audit_log ||
                !state.audit_log->info(
                    "SGEMM diagnostic: v=1, event=end, session={}, sequence={}, captured={}",
                    state.sgemm_audit_session, target->sequence,
                    state.sgemm_audit_captured ? 1 : 0))
                state.has_semantic_loss = true;
        }
        if (state.sgemm_issue_range) {
            const auto& result = state.sgemm_issue_range->summary();
            const bool is_supported =
                state.config.scheduling_mode == core::ExecutionSchedulingMode::kPriority &&
                !state.config.paired && !state.config.shared && !state.config.uses_nccl;
            if (!state.audit_log ||
                !state.audit_log->info(
                    "SGEMM range diagnostic: v=1, event=end, attempts={}, resumes={}, push={}, "
                    "pop={}, issued={}, native={}, completion={}, has_library={}, library={}, "
                    "saturated={}, supported={}",
                    result.attempts, result.resumes, result.push, result.pop,
                    result.first_issued ? 1 : 0, result.native_status, result.completion_status,
                    result.has_library_status ? 1 : 0, result.library_status,
                    result.saturated ? 1 : 0, is_supported ? 1 : 0))
                state.has_semantic_loss = true;
        }
        (void)state.semantic(2, [](auto& json) { json.add("event", 2); });
        state.is_semantic_closed = true;
        if (state.audit_log) state.audit_log->flush();
    }
    return true;
}
backend::cuda::CudaPtxCacheSnapshot ExecutionService::ptx_cache_snapshot() const noexcept {
    const auto& state = *implementation_;
    return state.owns() ? state.ptx_cache->snapshot() : backend::cuda::CudaPtxCacheSnapshot{};
}
backend::cuda::CudaProfileSnapshot ExecutionService::profile_snapshot() const noexcept {
    const auto& state = *implementation_;
    return state.owns() && state.profile_cache ? state.profile_cache->snapshot()
                                               : backend::cuda::CudaProfileSnapshot{};
}
backend::cuda::CudaPairProfileSnapshot ExecutionService::pair_profile_snapshot() const noexcept {
    const auto& state = *implementation_;
    return state.owns() && state.pair_profile_cache ? state.pair_profile_cache->snapshot()
                                                    : backend::cuda::CudaPairProfileSnapshot{};
}
CUresult ExecutionService::pair_profile_status() const noexcept {
    const auto& state = *implementation_;
    return state.owns() ? state.pair_profile_failure.value_or(CUDA_SUCCESS)
                        : CUDA_ERROR_INVALID_CONTEXT;
}
std::size_t ExecutionService::pending_priority_requests(std::uint32_t priority) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || state.is_active) return 0;
    std::size_t count{};
    for (const auto& slot : state.slots)
        if (!slot.is_retiring && slot.priority == priority) count += slot.requests.size();
    return count;
}
std::optional<std::size_t> ExecutionService::pending_priority_preparations(
    std::uint32_t priority) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || state.is_active || !state.is_started || state.is_stopping)
        return std::nullopt;
    std::size_t count{};
    for (const auto& slot : state.slots) {
        if (slot.priority != priority || slot.host_owner || slot.is_nccl ||
            slot.is_metadata_owner() || !slot.session || slot.is_retiring)
            continue;
        if (slot.policy_failure || slot.session->health_snapshot().is_quarantined)
            return std::nullopt;
        for (const auto& epoch : slot.low_epochs) {
            if (!epoch.is_asynchronous || epoch.is_issued) continue;
            if (!slot.session->owns_frontier(epoch.frontier)) return std::nullopt;
            const auto pending = slot.session->pending_preparation(epoch.sequence);
            if (pending.status == CUDA_SUCCESS && pending.command)
                ++count;
            else if (pending.status != CUDA_ERROR_INVALID_HANDLE)
                return std::nullopt;
        }
    }
    return count;
}
std::optional<std::size_t> ExecutionService::pending_priority_async_admissions(
    std::uint32_t priority) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || state.is_active || !state.is_started || state.is_stopping)
        return std::nullopt;
    std::size_t count{};
    for (const auto& slot : state.slots) {
        if (slot.priority != priority || slot.host_owner || slot.is_nccl ||
            slot.is_metadata_owner() || !slot.session || slot.is_retiring)
            continue;
        if (slot.policy_failure || slot.session->health_snapshot().is_quarantined)
            return std::nullopt;
        for (const auto& epoch : slot.low_epochs) {
            if (!epoch.is_asynchronous || epoch.is_issued) continue;
            if (!slot.session->owns_frontier(epoch.frontier)) return std::nullopt;
            // Highest-priority original epochs need not use low retreat
            // bookkeeping. Their exact host reply also proves admission is
            // no longer waiting for first native issue, even after receipt ACK.
            if (epoch.host_reply || slot.session->host_result(epoch.receipt)) continue;
            const auto retired = epoch.frontier.retirement();
            if (retired.is_retired) continue;
            if (retired.status != CUDA_ERROR_NOT_READY) return std::nullopt;
            ++count;
        }
    }
    return count;
}
bool ExecutionService::save_profile() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.is_stopping || !state.slots.empty()) return false;
    if (state.config.profile_output.empty()) return true;
    if (!state.profile_cache || !state.profile_cache->is_learning()) return false;
    const auto archive = state.profile_cache->encode(state.profile_identity);
    return archive && write_profile_archive(state.config.profile_output, *archive);
}
bool ExecutionService::save_ptx_cache() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.is_stopping || !state.slots.empty()) return false;
    if (state.config.ptx_cache_output.empty()) return true;
    const auto archive = state.ptx_cache->encode(state.ptx_identity);
    return archive && write_execution_archive(state.config.ptx_cache_output, *archive,
                                              backend::cuda::kCudaPtxArchiveMaxBytes);
}
bool ExecutionService::save_pair_profile() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.is_stopping || !state.slots.empty() || state.profile_pair ||
        state.pair_profile_failure || !state.config.paired ||
        !state.config.paired->is_isolated_profile || !state.execution_domain_stopped ||
        !state.pair_profile_cache)
        return false;
    const auto measured = state.pair_profile_cache->snapshot();
    if (!measured.is_valid || !measured.is_learning || measured.active || !measured.trained_pairs ||
        measured.trained_pairs != measured.pairs)
        return false;
    const auto archive = state.pair_profile_cache->encode(state.profile_identity);
    return archive && write_execution_archive(state.config.paired->profile_output, *archive,
                                              backend::cuda::kCudaPairProfileArchiveMaxBytes);
}
}  // namespace glimmer::app
