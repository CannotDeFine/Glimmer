#include "internal/cuda_dnn_registry.h"
#include "internal/cuda_dnn_legacy.h"
#include "internal/cuda_native_output_cell.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::backend::cuda {
namespace {
namespace dnn_profile = ::glimmer::internal::cudnn_profile;
thread_local std::shared_ptr<const void> dnn_owner;
CudaDnnResult rejected(CUresult status) {
    return {.status = status};
}
CudaDnnResult completed(cudnnStatus_t status) {
    return {.disposition = CudaDnnDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .original_status = status};
}
bool known_rejection(cudnnStatus_t status) {
    return dnn_profile::known_metadata_rejection(status);
}
template <typename Value>
std::uint32_t raw_bits(const Value& value) noexcept {
    std::uint32_t bits{};
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
std::uint32_t sequence_padding_bytes(std::uint32_t type, bool allows_bfloat) noexcept {
    if (type == 0) return 4;
    if (type == 1) return 8;
    if (type == 2 || (allows_bfloat && type == 9)) return 2;
    return 0;
}
struct SequenceCall {
    bool& active;
    explicit SequenceCall(bool& flag) noexcept : active(flag) {
        active = true;
    }
    ~SequenceCall() {
        active = false;
    }
};
bool has_dropout_dependencies(CudaDnnObjectKind kind) noexcept {
    return kind == CudaDnnObjectKind::kRnn || kind == CudaDnnObjectKind::kAttn;
}
struct DependencyCall {
    bool& active;
    const bool is_enabled;
    explicit DependencyCall(bool& flag, bool enabled = true) noexcept
        : active(flag), is_enabled(enabled) {
        if (is_enabled) active = true;
    }
    ~DependencyCall() {
        if (is_enabled) active = false;
    }
};
}  // namespace

struct CudaDnnDropoutMutation::State {
    CudaDnnDropoutBinding binding;
    // Retain both identities after an ambiguous mutation, without retaining
    // application storage or attempting an unsafe native rollback.
    CudaDnnDropoutBinding uncertain_candidate;
    std::atomic<bool> is_pending{}, is_uncertain{};
};
struct CudaDnnDropoutMutation::Implementation {
    std::shared_ptr<State> state;
    std::shared_ptr<const void> lifetime;
    CudaDnnDropout native{};
    CudaDnnDropoutBinding old_binding;
    const pid_t process{::getpid()};
    CudaDnnResult result;
    std::atomic<bool> is_claimed{}, is_issued{}, is_finishing{}, is_finished{};

    Implementation(std::shared_ptr<State> control, std::shared_ptr<const void> owner,
                   CudaDnnDropout descriptor)
        : state(std::move(control)),
          lifetime(std::move(owner)),
          native(descriptor),
          old_binding(state->binding) {}
};
CudaDnnDropoutMutation::CudaDnnDropoutMutation(std::shared_ptr<State> state,
                                               std::shared_ptr<const void> lifetime,
                                               CudaDnnDropout native)
    : implementation_(
          std::make_unique<Implementation>(std::move(state), std::move(lifetime), native)) {}
CudaDnnDropoutMutation::~CudaDnnDropoutMutation() {
    auto& state = *implementation_;
    if (state.process != ::getpid()) return;
    if (state.is_issued.load() && !state.is_finished.load()) state.state->is_uncertain.store(true);
    // The command owns this token through its physical trailing event. No
    // caller-native cleanup happens on the thread releasing the last lease.
    state.state->is_pending.store(false, std::memory_order_release);
}
CudaDnnDropout CudaDnnDropoutMutation::native() const noexcept {
    const auto& state = *implementation_;
    return state.process == ::getpid() && !state.state->is_uncertain.load() ? state.native
                                                                            : nullptr;
}
const CudaDnnDropoutBinding& CudaDnnDropoutMutation::old_binding() const noexcept {
    return implementation_->old_binding;
}
bool CudaDnnDropoutMutation::claim() noexcept {
    auto& state = *implementation_;
    return native() && !state.is_claimed.exchange(true);
}
bool CudaDnnDropoutMutation::mark_issued() noexcept {
    auto& state = *implementation_;
    return native() && state.is_claimed.load() && !state.is_issued.exchange(true);
}
void CudaDnnDropoutMutation::finish(cudnnStatus_t status, const CudaDnnDropoutBinding& binding,
                                    CUresult proof) noexcept {
    auto& state = *implementation_;
    if (state.process != ::getpid() || !state.is_issued.load() || state.is_finishing.exchange(true))
        return;
    if (proof != CUDA_SUCCESS || status != CUDNN_STATUS_SUCCESS) {
        // Stateful initialization has no documented error atomicity. Even
        // parameter errors cannot establish that the old descriptor survived.
        state.state->uncertain_candidate = binding;
        state.state->is_uncertain.store(true, std::memory_order_release);
    } else {
        state.state->binding = binding;
    }
    state.result = completed(status);
    if (proof != CUDA_SUCCESS) {
        state.result.disposition = CudaDnnDisposition::kRejected;
        state.result.status = proof;
    }
    state.is_finished.store(true, std::memory_order_release);
}
std::optional<CudaDnnResult> CudaDnnDropoutMutation::completion_result() const noexcept {
    const auto& state = *implementation_;
    if (state.process != ::getpid() || !state.is_finished.load(std::memory_order_acquire))
        return std::nullopt;
    return state.result;
}

struct CudaDnnRnnLease::State {
    struct BackendPointer {
        void* native{};
        std::weak_ptr<const void> lifetime;
    };
    const pid_t process{::getpid()};
    std::weak_ptr<const void> owner;
    bool is_active{}, is_quarantined{};
    bool is_folded_active{}, is_sequence_active{}, is_dependency_active{}, is_rnn_query_active{};
    std::vector<BackendPointer> backend_objects;
    bool owns() const noexcept {
        return process == ::getpid() && dnn_owner && owner.lock() == dnn_owner;
    }
};
struct CudaDnnRnnLease::Implementation {
    std::shared_ptr<State> state;
    std::shared_ptr<const void> owner;
    bool is_active{};
};
CudaDnnRnnLease::CudaDnnRnnLease(std::shared_ptr<State> state, std::shared_ptr<const void> owner)
    : implementation_(std::make_unique<Implementation>(
          Implementation{std::move(state), std::move(owner), false})) {}
CudaDnnRnnLease::~CudaDnnRnnLease() {
    if (implementation_->is_active && implementation_->state->owns()) {
        implementation_->state->is_quarantined = true;
        end_call();
    }
}
bool CudaDnnRnnLease::begin_call() noexcept {
    auto& lease = *implementation_;
    if (!lease.state->owns() || lease.state->is_quarantined || lease.state->is_active ||
        lease.state->is_folded_active || lease.state->is_sequence_active ||
        lease.state->is_dependency_active || lease.state->is_rnn_query_active || lease.is_active)
        return false;
    lease.state->is_active = lease.is_active = true;
    return true;
}
void CudaDnnRnnLease::end_call() noexcept {
    auto& lease = *implementation_;
    if (lease.state->owns() && lease.is_active) lease.state->is_active = lease.is_active = false;
}
void CudaDnnRnnLease::quarantine() noexcept {
    if (implementation_->state->owns()) implementation_->state->is_quarantined = true;
}
bool CudaDnnRnnLease::owns_backend_pointer(void* native) const noexcept {
    const auto& lease = *implementation_;
    if (!lease.state->owns() || !lease.is_active || !lease.state->is_active) return true;
    return std::ranges::any_of(lease.state->backend_objects, [native](const auto& object) {
        return object.native == native && !object.lifetime.expired();
    });
}

struct CudaDnnRegistry::Implementation {
    struct Object {
        std::uint64_t id{};
        CudaDnnObjectKind kind{};
        void* native{};
        bool is_uncertain{}, has_destroy_attempt{};
        std::shared_ptr<CudaDnnDropoutMutation::State> dropout;
        // ID edges do not affect immutable-borrow counts or own RNG storage.
        // Healthy parents use at most two slots; a failed rebind retains the
        // union of old and candidate identities without post-issue allocation.
        std::array<std::uint64_t, 4> dropout_dependencies{};
        // Algorithm descriptors are liveness edges, not dropout/RNG bindings.
        // A failed rebind retains both old and candidate before quarantine.
        std::array<std::uint64_t, 2> algorithm_dependencies{};
        std::shared_ptr<const CudaDnnBackendSnapshot> backend;
        // Inert identity addresses only: original VariantPack Set stores these
        // values without accessing payload. Never execute this public pack.
        struct PointerIdentity {
            std::uint64_t slot{};
        };
        std::vector<std::shared_ptr<PointerIdentity>> backend_pointers;
        // Historical comparison value only, never dereferenced or an owner.
        cudnnHandle_t backend_handle_value{};
    };
    struct DependencyUpdate {
        std::array<CudaDnnDropout, 2> native{};
        std::array<std::uint64_t, 4> replacement{}, retained{};
    };
    const CudaDnnDispatch dispatch;
    const std::size_t maximum;
    const pid_t process{::getpid()};
    std::weak_ptr<const void> owner;
    const std::shared_ptr<CudaDnnRnnLease::State> rnn_calls{
        std::make_shared<CudaDnnRnnLease::State>()};
    std::vector<std::shared_ptr<Object>> objects;
    std::uint64_t next_id{1};
    bool has_id_space{true}, is_closing{}, is_faulted{};
    // Existing scoped operations and queued RNN leases share one reentry
    // authority, including when a callback reaches a different CUDA session.
    bool& is_folded_active{rnn_calls->is_folded_active};
    bool& is_sequence_active{rnn_calls->is_sequence_active};
    bool& is_dependency_active{rnn_calls->is_dependency_active};
    bool& is_rnn_query_active{rnn_calls->is_rnn_query_active};

    Implementation(CudaDnnDispatch sdk, std::size_t bound) : dispatch(sdk), maximum(bound) {
        if (!bound || !sdk.tensor_create || !sdk.tensor_destroy || !sdk.tensor_set_4d ||
            !sdk.tensor_set_4d_ex || !sdk.tensor_set_nd || !sdk.tensor_set_nd_ex ||
            !sdk.tensor_get_4d || !sdk.tensor_get_nd || !sdk.tensor_size ||
            !sdk.activation_create || !sdk.activation_destroy || !sdk.activation_set ||
            !sdk.activation_get)
            throw std::invalid_argument("invalid cuDNN metadata dispatch or bound");
        objects.reserve(bound);
        rnn_calls->backend_objects.reserve(bound);
        if (!dnn_owner) dnn_owner = std::make_shared<int>(0);
        owner = dnn_owner;
        rnn_calls->owner = owner;
    }
    bool owns() const noexcept {
        return ::getpid() == process && dnn_owner && owner.lock() == dnn_owner;
    }
    CUresult gate(bool mutating = false) const noexcept {
        if (!owns() || is_quarantined()) return CUDA_ERROR_UNKNOWN;
        if (is_folded_active || is_sequence_active || is_dependency_active || is_rnn_query_active ||
            rnn_calls->is_active)
            return CUDA_ERROR_NOT_READY;
        return is_closing && mutating ? CUDA_ERROR_DEINITIALIZED : CUDA_SUCCESS;
    }
    bool is_quarantined() const noexcept {
        return is_faulted || rnn_calls->is_quarantined ||
               std::ranges::any_of(objects, [](const auto& object) {
                   return object->dropout && object->dropout->is_uncertain.load();
               });
    }
    auto find(std::uint64_t id) {
        return std::ranges::find_if(objects, [id](const auto& entry) { return entry->id == id; });
    }
    bool has_dependents(std::uint64_t id) const noexcept {
        return std::ranges::any_of(objects, [id](const auto& entry) {
            return std::ranges::find(entry->dropout_dependencies, id) !=
                       entry->dropout_dependencies.end() ||
                   std::ranges::find(entry->algorithm_dependencies, id) !=
                       entry->algorithm_dependencies.end();
        });
    }
    CUresult pin_dependencies(const Object& parent,
                              std::array<std::shared_ptr<Object>, 4>& owners) noexcept {
        std::size_t count{};
        bool is_pending{};
        for (const auto id : parent.dropout_dependencies) {
            if (!id) continue;
            const auto found = find(id);
            if (found == objects.end() || (*found)->kind != CudaDnnObjectKind::kDropout ||
                !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt ||
                !(*found)->dropout || (*found)->dropout->is_uncertain.load())
                return CUDA_ERROR_INVALID_HANDLE;
            is_pending |= (*found)->dropout->is_pending.load(std::memory_order_acquire);
            const auto end = owners.begin() + count;
            if (std::find(owners.begin(), end, *found) == end) owners[count++] = *found;
        }
        return is_pending ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    CUresult prepare_dependencies(const Object& parent,
                                  const std::array<std::uint64_t, 2>& candidates,
                                  DependencyUpdate& update) noexcept {
        update.retained = parent.dropout_dependencies;
        const auto insert = [](auto& edges, std::uint64_t id) {
            if (std::ranges::find(edges, id) != edges.end()) return true;
            const auto empty = std::ranges::find(edges, 0);
            if (empty == edges.end()) return false;
            *empty = id;
            return true;
        };
        bool is_pending{};
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto id = candidates[index];
            if (!id) continue;
            const auto found = find(id);
            if (found == objects.end() || (*found)->kind != CudaDnnObjectKind::kDropout ||
                !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt ||
                !(*found)->dropout || (*found)->dropout->is_uncertain.load())
                return CUDA_ERROR_INVALID_HANDLE;
            is_pending |= (*found)->dropout->is_pending.load(std::memory_order_acquire);
            if (!insert(update.replacement, id) || !insert(update.retained, id)) {
                is_faulted = true;
                return CUDA_ERROR_UNKNOWN;
            }
            update.native[index] = static_cast<CudaDnnDropout>((*found)->native);
        }
        return is_pending ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    CudaDnnResult commit_dependencies(Object& parent, const DependencyUpdate& update,
                                      cudnnStatus_t status) noexcept {
        parent.dropout_dependencies =
            status == CUDNN_STATUS_SUCCESS ? update.replacement : update.retained;
        if (status != CUDNN_STATUS_SUCCESS) parent.is_uncertain = is_faulted = true;
        return completed(status);
    }
    CudaDnnResult mutation(Object& object, cudnnStatus_t status) noexcept {
        if (status != CUDNN_STATUS_SUCCESS && !known_rejection(status))
            object.is_uncertain = is_faulted = true;
        return completed(status);
    }
    cudnnStatus_t destroy_native(Object& object) noexcept {
        object.has_destroy_attempt = true;
        switch (object.kind) {
            case CudaDnnObjectKind::kTensor:
                return dispatch.tensor_destroy(static_cast<cudnnTensorDescriptor_t>(object.native));
            case CudaDnnObjectKind::kActivation:
                return dispatch.activation_destroy(static_cast<CudaDnnActivation>(object.native));
            case CudaDnnObjectKind::kFilter:
                return dispatch.filter_destroy(static_cast<CudaDnnFilterHandle>(object.native));
            case CudaDnnObjectKind::kConvolution:
                return dispatch.convolution_destroy(static_cast<CudaDnnConvolution>(object.native));
            case CudaDnnObjectKind::kPooling:
                return dispatch.pooling_destroy(static_cast<CudaDnnPooling>(object.native));
            case CudaDnnObjectKind::kLrn:
                return dispatch.lrn_destroy(static_cast<CudaDnnLrn>(object.native));
            case CudaDnnObjectKind::kReduction:
                return dispatch.reduction_destroy(static_cast<CudaDnnReduction>(object.native));
            case CudaDnnObjectKind::kDropout:
                return dispatch.dropout_destroy(static_cast<CudaDnnDropout>(object.native));
            case CudaDnnObjectKind::kOpTensor:
                return dispatch.op_tensor_destroy(static_cast<CudaDnnOpTensor>(object.native));
            case CudaDnnObjectKind::kTensorTransform:
                return dispatch.tensor_transform_destroy(
                    static_cast<CudaDnnTensorTransform>(object.native));
            case CudaDnnObjectKind::kRnnData:
                return dispatch.rnn_data_destroy(static_cast<CudaDnnRnnData>(object.native));
            case CudaDnnObjectKind::kSeqData:
                return dispatch.seq_data_destroy(static_cast<CudaDnnSeqData>(object.native));
            case CudaDnnObjectKind::kRnn:
                return dispatch.rnn_destroy(static_cast<CudaDnnRnn>(object.native));
            case CudaDnnObjectKind::kAttn:
                return dispatch.attn_destroy(static_cast<CudaDnnAttn>(object.native));
            case CudaDnnObjectKind::kBackend:
                return dispatch.backend_destroy(object.native);
            case CudaDnnObjectKind::kAlgorithm:
#if CUDNN_MAJOR == 8
                return dispatch.legacy.algorithm_destroy(
                    static_cast<cudnnAlgorithmDescriptor_t>(object.native));
#else
                return CUDNN_STATUS_NOT_SUPPORTED;
#endif
        }
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
    CudaDnnResult malformed(Object& object, CudaDnnResult result) noexcept {
        object.is_uncertain = is_faulted = true;
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
};

CudaDnnRegistry::CudaDnnRegistry(CudaDnnDispatch dispatch, std::size_t max_objects)
    : implementation_(std::make_unique<Implementation>(dispatch, max_objects)) {}
CudaDnnRegistry::~CudaDnnRegistry() {
    if (implementation_->owns()) {
        const auto result = shutdown();
        if (result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
            result.original_status == CUDNN_STATUS_SUCCESS)
            return;
    }
    // Never move native destruction to a foreign owner or invalidate leases
    // whose GPU consumer has not stopped. Retain uncertain ownership once.
    try {
        spdlog::critical("Retaining unsettled native cuDNN metadata");
    } catch (...) {
    }
    (void)implementation_.release();
}
CudaDnnResult CudaDnnRegistry::library_info(CudaDnnLibraryInfoKind kind, std::uint32_t property,
                                            bool has_output) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    if (kind != CudaDnnLibraryInfoKind::kProperty && (property || has_output))
        return rejected(CUDA_ERROR_INVALID_VALUE);

    decltype(state.dispatch.version) query{};
    switch (kind) {
        case CudaDnnLibraryInfoKind::kVersion:
            query = state.dispatch.version;
            break;
        case CudaDnnLibraryInfoKind::kCudartVersion:
            query = state.dispatch.cudart_version;
            break;
        case CudaDnnLibraryInfoKind::kMaxDeviceVersion:
            query = state.dispatch.max_device_version;
            break;
        case CudaDnnLibraryInfoKind::kProperty: {
            // The public nonfixed enum represents 0..3. Its binary-only
            // selector 100 is not safely expressible through this SDK type.
            if (property > 3) return rejected(CUDA_ERROR_INVALID_VALUE);
            if (!state.dispatch.property) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            int value{};
            auto result = completed(state.dispatch.property(
                static_cast<libraryPropertyType>(property), has_output ? &value : nullptr));
            if (result.original_status == CUDNN_STATUS_SUCCESS && has_output)
                result.value = raw_bits(value);
            return result;
        }
        default:
            return rejected(CUDA_ERROR_INVALID_VALUE);
    }
    if (!query) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    // These original functions return data, not cudnnStatus_t. Do not invent
    // native status provenance or treat a native zero as a transport failure.
    return {.disposition = CudaDnnDisposition::kComplete,
            .status = CUDA_SUCCESS,
            .value = static_cast<std::uint64_t>(query())};
}
CudaDnnResult CudaDnnRegistry::create(CudaDnnObjectKind kind) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    if (kind != CudaDnnObjectKind::kTensor && kind != CudaDnnObjectKind::kActivation &&
        kind != CudaDnnObjectKind::kFilter && kind != CudaDnnObjectKind::kConvolution &&
        kind != CudaDnnObjectKind::kPooling && kind != CudaDnnObjectKind::kLrn &&
        kind != CudaDnnObjectKind::kReduction && kind != CudaDnnObjectKind::kDropout &&
        kind != CudaDnnObjectKind::kOpTensor && kind != CudaDnnObjectKind::kTensorTransform &&
        kind != CudaDnnObjectKind::kRnnData && kind != CudaDnnObjectKind::kSeqData &&
        kind != CudaDnnObjectKind::kRnn && kind != CudaDnnObjectKind::kAttn &&
        kind != CudaDnnObjectKind::kAlgorithm)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (kind == CudaDnnObjectKind::kAlgorithm) {
#if CUDNN_MAJOR == 8
        if (!state.dispatch.legacy.algorithm_create || !state.dispatch.legacy.algorithm_destroy)
            return rejected(CUDA_ERROR_NOT_SUPPORTED);
#else
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
#endif
    }
    // Creation cannot establish ownership without a corresponding destroy path.
    if ((kind == CudaDnnObjectKind::kFilter &&
         (!state.dispatch.filter_create || !state.dispatch.filter_destroy)) ||
        (kind == CudaDnnObjectKind::kConvolution &&
         (!state.dispatch.convolution_create || !state.dispatch.convolution_destroy)) ||
        (kind == CudaDnnObjectKind::kPooling &&
         (!state.dispatch.pooling_create || !state.dispatch.pooling_destroy)) ||
        (kind == CudaDnnObjectKind::kLrn &&
         (!state.dispatch.lrn_create || !state.dispatch.lrn_destroy)) ||
        (kind == CudaDnnObjectKind::kReduction &&
         (!state.dispatch.reduction_create || !state.dispatch.reduction_destroy)) ||
        (kind == CudaDnnObjectKind::kDropout &&
         (!state.dispatch.dropout_create || !state.dispatch.dropout_destroy)) ||
        (kind == CudaDnnObjectKind::kOpTensor &&
         (!state.dispatch.op_tensor_create || !state.dispatch.op_tensor_destroy)) ||
        (kind == CudaDnnObjectKind::kTensorTransform &&
         (!state.dispatch.tensor_transform_create || !state.dispatch.tensor_transform_destroy)) ||
        (kind == CudaDnnObjectKind::kRnnData &&
         (!state.dispatch.rnn_data_create || !state.dispatch.rnn_data_destroy)) ||
        (kind == CudaDnnObjectKind::kSeqData &&
         (!state.dispatch.seq_data_create || !state.dispatch.seq_data_destroy)) ||
        (kind == CudaDnnObjectKind::kRnn &&
         (!state.dispatch.rnn_create || !state.dispatch.rnn_destroy)) ||
        (kind == CudaDnnObjectKind::kAttn &&
         (!state.dispatch.attn_create || !state.dispatch.attn_destroy)))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (!state.has_id_space || state.objects.size() == state.maximum)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto object = std::make_shared<Implementation::Object>();
        object->id = state.next_id;
        object->kind = kind;
        if (kind == CudaDnnObjectKind::kDropout)
            object->dropout = std::make_shared<CudaDnnDropoutMutation::State>();
        const DependencyCall active{
            state.is_dependency_active,
            has_dropout_dependencies(kind) || kind == CudaDnnObjectKind::kAlgorithm};
        cudnnStatus_t status{};
        if (kind == CudaDnnObjectKind::kTensor) {
            cudnnTensorDescriptor_t native{};
            status = state.dispatch.tensor_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kActivation) {
            CudaDnnActivation native{};
            status = state.dispatch.activation_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kFilter) {
            CudaDnnFilterHandle native{};
            status = state.dispatch.filter_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kConvolution) {
            CudaDnnConvolution native{};
            status = state.dispatch.convolution_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kPooling) {
            CudaDnnPooling native{};
            status = state.dispatch.pooling_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kLrn) {
            CudaDnnLrn native{};
            status = state.dispatch.lrn_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kReduction) {
            CudaDnnReduction native{};
            status = state.dispatch.reduction_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kDropout) {
            CudaDnnDropout native{};
            status = state.dispatch.dropout_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kOpTensor) {
            CudaDnnOpTensor native{};
            status = state.dispatch.op_tensor_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kTensorTransform) {
            CudaDnnTensorTransform native{};
            status = state.dispatch.tensor_transform_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kRnnData) {
            CudaDnnRnnData native{};
            status = state.dispatch.rnn_data_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kSeqData) {
            CudaDnnSeqData native{};
            status = state.dispatch.seq_data_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kRnn) {
            CudaDnnRnn native{};
            status = state.dispatch.rnn_create(&native);
            object->native = native;
        } else if (kind == CudaDnnObjectKind::kAttn) {
            CudaDnnAttn native{};
            status = state.dispatch.attn_create(&native);
            object->native = native;
#if CUDNN_MAJOR == 8
        } else {
            cudnnAlgorithmDescriptor_t native{};
            status = state.dispatch.legacy.algorithm_create(&native);
            object->native = native;
#endif
        }
        auto result = completed(status);
        if (status != CUDNN_STATUS_SUCCESS && !object->native) return result;
        if (state.next_id == std::numeric_limits<std::uint64_t>::max())
            state.has_id_space = false;
        else
            ++state.next_id;
        object->is_uncertain = status != CUDNN_STATUS_SUCCESS || !object->native;
        for (auto& existing : state.objects)
            if (existing->native == object->native)
                existing->is_uncertain = object->is_uncertain = true;
        state.objects.push_back(object);  // Capacity reserved before the SDK call.
        if (object->is_uncertain) {
            state.is_faulted = true;
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        } else
            result.id = object->id;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaDnnRegistry::destroy(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    // A durable edge is not transient work. Deferring here could prevent the
    // same connection from ever submitting the dependent's destruction.
    if (state.has_dependents(id)) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const DependencyCall active{state.is_dependency_active,
                                has_dropout_dependencies((*found)->kind) ||
                                    (*found)->kind == CudaDnnObjectKind::kBackend ||
                                    (*found)->kind == CudaDnnObjectKind::kAlgorithm};
    if ((*found)->kind == CudaDnnObjectKind::kBackend &&
        (!state.dispatch.version || state.dispatch.version() != 91002))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto status = state.destroy_native(**found);
    if (status == CUDNN_STATUS_SUCCESS)
        state.objects.erase(found);
    else
        (*found)->is_uncertain = state.is_faulted = true;
    return completed(status);
}

namespace {
constexpr std::size_t kBackendMaxElements = std::size_t{1} << 16;
constexpr std::uint32_t backend_types(std::uint32_t type) noexcept {
    return std::uint32_t{1} << type;
}
struct BackendAttributeRow {
    std::uint32_t types{};
    // Zero denotes scalar storage; descriptor kind zero is encoded as kind+1.
    std::uint32_t child{};
    bool is_array{}, ignores_count{};
};
constexpr std::uint32_t kBackendOperationChild = 0xffffffffU;
bool backend_operation_kind(std::uint32_t kind) noexcept {
    return (kind >= 10 && kind <= 13) || kind == 19 || kind == 22;
}
bool backend_child_kind(const BackendAttributeRow& row, std::uint32_t kind) noexcept {
    return row.child == kBackendOperationChild ? backend_operation_kind(kind)
                                               : row.child == kind + 1;
}
bool backend_planning_kind(std::uint32_t kind) noexcept {
    return kind == 2 || kind == 3 || kind == 4 || kind == 5 || kind == 15;
}
bool backend_cpu_get(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return (snapshot.kind == 2 && (name == 1303 || name == 1304 || name == 1305 || name == 1307)) ||
           (snapshot.kind == 4 && name == 204) ||
           (snapshot.kind == 5 && (name == 400 || name == 407));
}
bool backend_property_copy(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return (snapshot.kind == 2 && name == 1307) || (snapshot.kind == 4 && name == 204) ||
           (snapshot.kind == 5 && name == 407);
}
bool backend_layout_copy(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return snapshot.kind == 2 && name == 1304;
}
bool backend_note_array(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return (snapshot.kind == 2 && (name == 1303 || name == 1305)) ||
           (snapshot.kind == 9 && name == 1101);
}
bool backend_property_json(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return snapshot.kind == 38 && name == 2702;
}
void backend_clear_metadata_observations(CudaDnnBackendSnapshot& snapshot) noexcept {
    snapshot.numerical_note_count.reset();
    snapshot.behavior_note_count.reset();
    snapshot.layout_type_count.reset();
    snapshot.json_bytes.reset();
}
void backend_invalidate_metadata(CudaDnnBackendSnapshot& snapshot) noexcept {
    backend_clear_metadata_observations(snapshot);
    snapshot.metadata_finalized.reset();
    snapshot.property_mode = CudaDnnBackendPropertyMode::kUnknown;
    snapshot.property_device_id.reset();
    snapshot.generated_property.reset();
}
CUresult backend_context_status(const CudaDnnBackendSnapshot& snapshot,
                                const CudaDnnBackendContext* context,
                                bool allows_context_free = false) noexcept {
    if (!context)
        return backend_planning_kind(snapshot.kind) && !allows_context_free
                   ? CUDA_ERROR_INVALID_CONTEXT
                   : CUDA_SUCCESS;
    if (!context->owner || context->lifetime.expired() || !context->check_context ||
        context->device_ordinal < 0)
        return CUDA_ERROR_INVALID_CONTEXT;
    // Copied device properties are reusable on the same device. Historical
    // session/handle identities do not confer permanent native context affinity.
    if (!allows_context_free && snapshot.context &&
        snapshot.context->device_ordinal != context->device_ordinal)
        return CUDA_ERROR_INVALID_CONTEXT;
    return context->check_context(context->opaque);
}
void backend_context_failure(CudaDnnResult& result, CUresult status) noexcept {
    if (status == CUDA_SUCCESS) return;
    result.status = status;
    result.disposition = CudaDnnDisposition::kRejected;
}
bool backend_kind(std::uint32_t kind) noexcept {
    if (!dnn_profile::supports_backend_kind(kind)) return false;
    switch (kind) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 21:
        case 22:
        case 24:
        case 32:
        case 38:
            return true;
        default:
            return false;
    }
}
std::size_t backend_width(std::uint32_t type) noexcept {
    if (type == 2 || type == 24) return 1;
    if (type == 0 || type == 3 || type == 5 || type == 6) return 8;
    if (type == 26) return 16;
    if (type <= 29 && type != 0 && type != 6 && type != 15) return 4;
    return 0;
}
BackendAttributeRow backend_row(const CudaDnnBackendSnapshot& snapshot,
                                std::uint32_t name) noexcept {
    if (!dnn_profile::supports_backend_kind(snapshot.kind) ||
        !dnn_profile::supports_backend_attribute(name))
        return {};
    const auto scalar = [](std::uint32_t type, bool array = false, bool ignores = false) {
        return BackendAttributeRow{backend_types(type), 0, array, ignores};
    };
    const auto child = [](std::uint32_t type, bool ignores = true) {
        return BackendAttributeRow{backend_types(15), type + 1, false, ignores};
    };
    const auto coefficient =
        BackendAttributeRow{backend_types(4) | backend_types(5), 0, false, false};
    switch (snapshot.kind) {
        case 0:
            if (name == 0) return scalar(14);
            if (name == 1) return scalar(1);
            if (name == 2) return scalar(10);
            if (name >= 3 && name <= 8)
                return {backend_types(4) | backend_types(5), 0, false, false};
            if (name == 9) return scalar(3);
            break;
        case 1:
            if (name == 100) return scalar(1);
            if (name == 101) return scalar(7);
            if (name >= 102 && name <= 105) return scalar(3, true);
            if (name == 106) return scalar(3);
            break;
        case 2:
            if (name == 1300) return child(15, false);
            if (name == 1301) return scalar(3);
            if (name == 1302) return {backend_types(15), 9, true, false};
            if (name == 1303) return scalar(11, true);
            if (name == 1304) return {backend_types(15), 10, true, false};
            if (name == 1305) return scalar(19, true);
            if (name == 1306) return scalar(23);
            if (name == 1307) return child(38, false);
            break;
        case 3:
            if (name == 300) return child(2, false);
            if (name == 302) return {backend_types(15), 8, true, false};
            if (name == 303) return scalar(3);
            if (name == 304) return scalar(23);
            break;
        case 4:
            if (name == 200) return scalar(8);
            if (name == 201) return child(15, false);
            if (name == 202) return {backend_types(15), 4, true, false};
            if (name == 203) return scalar(23);
            if (name == 204) return child(38, false);
            break;
        case 5:
            if (name == 400) return scalar(0);
            if (name == 401) return child(3, false);
            if (name == 402) return scalar(3);
            if (name == 407) return child(38, false);
            break;
        case 7:
            if (name == 600) return scalar(9);
            if (name == 601) return scalar(3);
            break;
        case 8:
            if (name == 1200) return scalar(9);
            if (name >= 1201 && name <= 1203) return scalar(3);
            break;
        case 9:
            if (name == 1100) return scalar(3);
            if (name == 1101) return scalar(12, true);
            break;
        case 10:
            if (name == 700 || name == 701)
                return {backend_types(4) | backend_types(5) | backend_types(23), 0, false, false};
            if (name == 702) return child(1, false);
            if (name >= 703 && name <= 705) return child(17, false);
            break;
        case 11:
            if (name == 712 || name == 713)
                return {backend_types(4) | backend_types(5), 0, false, false};
            if (name == 714) return child(1, false);
            if (name >= 715 && name <= 717) return child(17, false);
            break;
        case 12:
            if (name == 706 || name == 707)
                return {backend_types(4) | backend_types(5), 0, false, false};
            if (name == 708) return child(1, false);
            if (name >= 709 && name <= 711) return child(17, false);
            break;
        case 13:
            if (name == 750) return child(0, false);
            if ((name >= 751 && name <= 753) || (name >= 756 && name <= 758))
                return child(17, false);
            if (name == 754 || name == 755) return coefficient;
            break;
        case 15:
            if (name == 800) return scalar(0);
            if (name == 801) return {backend_types(15), kBackendOperationChild, true, false};
            if (name == 802) return scalar(3);
            if (name == 803) return scalar(2);
            break;
        case 16:
            if (name == 1000) return scalar(3, true);
            if (name == 1001) return scalar(6, true);
            if (name == 1003) return scalar(6);
            break;
        case 17:
            if (name == 901) return scalar(1);
            if (name == 907 || name == 908) return scalar(2);
            if (name == 909) return {backend_types(3) | backend_types(20), 0, false, false};
            if (name == 902 || name == 903) return scalar(3, true);
            if (name == 900 || (name >= 904 && name <= 906)) return scalar(3);
            // Ragged 913 has a private vector Get ABI and nontrivial reset.
            break;
        case 18:
            if (name == 1500) return scalar(1);
            if (name == 1503)
                return {backend_types(3) | backend_types(4) | backend_types(5) | backend_types(23),
                        0, false, false};
            break;
        case 19:
            if (name >= 1520 && name <= 1522) return child(17);
            if (name == 1523) return child(18);
            if (name == 1524) return scalar(3);
            if (name >= 1525 && name <= 1527) return child(17);
            break;
        case 21:
            if (name == 1600) return scalar(18);
            if (name == 1601) return scalar(1);
            break;
        case 22:
            if (name == 1610 || name == 1611) return child(17);
            if (name == 1612) return child(21);
            break;
        case 24:
            if (name == 1700) return scalar(21);
            if (name == 1701) return scalar(1);
            if (name == 1702) return scalar(3);
            if (name >= 1703 && name <= 1706)
                return {backend_types(3) | backend_types(26), 0, true, false};
            if (name == 1707) return scalar(10);
            if (name == 1708) return scalar(22);
            break;
        case 32:
            if (name == 2300) return scalar(29);
            if (name >= 2301 && name <= 2305) return scalar(5);
            break;
        case 38:
            if (name == 2700) return scalar(23);
            if (name == 2701) return scalar(0);
            if (name == 2702) return scalar(24, true);
            break;
        default:
            break;
    }
    return {};
}
bool backend_scalar_bits(std::uint32_t type, const std::vector<std::byte>& bytes) noexcept {
    const auto width = backend_width(type);
    if (!width || bytes.size() % width || bytes.size() / width > kBackendMaxElements) return false;
    std::uint32_t maximum{};
    switch (type) {
        case 1:
            maximum = dnn_profile::data_type_enum_max();
            break;
        case 2:
        case 7:
        case 10:
            maximum = 1;
            break;
        case 14:
            maximum = 1023;
            break;
        case 9:
            maximum = 63;
            break;
        case 11:
            maximum = 15;
            break;
        case 12:
            maximum = 7;
            break;
        case 19:
            maximum = dnn_profile::behavior_note_enum_max();
            break;
        case 18:
            maximum = 15;
            break;
        case 20:
        case 22:
        case 29:
            maximum = 3;
            break;
        case 21:
        case 8:
            maximum = 7;
            break;
        default:
            return true;
    }
    for (std::size_t offset = 0; offset < bytes.size(); offset += width) {
        std::uint32_t value{};
        std::memcpy(&value, bytes.data() + offset, width);
        if (value > maximum) return false;
    }
    return true;
}
const CudaDnnBackendSnapshotAttribute* backend_attribute(const CudaDnnBackendSnapshot& snapshot,
                                                         std::uint32_t name) noexcept {
    const auto found = std::ranges::find_if(
        snapshot.attributes, [name](const auto& attribute) { return attribute.name == name; });
    return found == snapshot.attributes.end() ? nullptr : &*found;
}
bool backend_generated_array(const CudaDnnBackendSnapshot& snapshot, std::uint32_t name) noexcept {
    return (snapshot.kind == 4 && name == 202) || (snapshot.kind == 2 && name == 1302) ||
           (snapshot.kind == 3 && name == 302);
}
std::shared_ptr<const CudaDnnBackendSnapshot> backend_graph_lineage(
    const CudaDnnBackendSnapshot& snapshot) noexcept {
    const auto name = snapshot.kind == 4 ? 201U : snapshot.kind == 2 ? 1300U : 300U;
    const auto* attribute = backend_attribute(snapshot, name);
    if (!attribute) return snapshot.generated_graph;
    if (attribute->descriptors.empty() || !attribute->descriptors.front()) return {};
    const auto& child = attribute->descriptors.front();
    return child->kind == 15 ? child : child->generated_graph;
}
std::shared_ptr<const CudaDnnBackendSnapshot> backend_property_lineage(
    const CudaDnnBackendSnapshot& snapshot) noexcept {
    const auto name = snapshot.kind == 4 ? 204U : snapshot.kind == 2 ? 1307U : 407U;
    if (snapshot.kind == 2 || snapshot.kind == 4 || snapshot.kind == 5) {
        if (const auto* property = backend_attribute(snapshot, name))
            return property->descriptors.empty() ? nullptr : property->descriptors.front();
    }
    if (snapshot.kind == 3 || snapshot.kind == 5) {
        if (const auto* source = backend_attribute(snapshot, snapshot.kind == 3 ? 300U : 401U)) {
            if (source->descriptors.empty() || !source->descriptors.front()) return {};
            return backend_property_lineage(*source->descriptors.front());
        }
    }
    return snapshot.generated_property;
}
void backend_observe_generated_capacity(const CudaDnnDispatch& dispatch,
                                        cudnnBackendDescriptor_t native,
                                        CudaDnnBackendSnapshot& snapshot) noexcept {
    snapshot.generated_array_capacity.reset();
    if (!dispatch.backend_get || !snapshot.is_valid || !snapshot.is_finalized ||
        (snapshot.kind != 2 && snapshot.kind != 3 && snapshot.kind != 4))
        return;
    const auto name = snapshot.kind == 4 ? 202U : snapshot.kind == 2 ? 1302U : 302U;
    // Engine uses zero for all knobs; Config instead needs the full fixed table.
    const std::int64_t requested = snapshot.kind == 3 ? 43 : 0;
    std::int64_t count{-1};
    try {
        const auto status =
            dispatch.backend_get(native, static_cast<cudnnBackendAttributeName_t>(name),
                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, requested, &count, nullptr);
        const auto maximum = snapshot.kind == 4 ? kBackendMaxElements : std::size_t{43};
        if (status == CUDNN_STATUS_SUCCESS && count >= 0 &&
            static_cast<std::uint64_t>(count) <= maximum)
            snapshot.generated_array_capacity = static_cast<std::uint32_t>(count);
    } catch (...) {
        // Injected observers may fail; the requested Finalize/copy still owns
        // its exact original result. No observation is preferable to a guess.
    }
}
struct BackendCountQuery {
    std::uint32_t name{}, type{}, maximum{};
};
std::optional<std::uint32_t> backend_observe_count(const CudaDnnDispatch& dispatch,
                                                   cudnnBackendDescriptor_t native,
                                                   BackendCountQuery query) noexcept {
    if (!dispatch.backend_get) return std::nullopt;
    try {
        std::int64_t count{-1};
        const auto status = dispatch.backend_get(
            native, static_cast<cudnnBackendAttributeName_t>(query.name),
            static_cast<cudnnBackendAttributeType_t>(query.type), 0, &count, nullptr);
        if (status == CUDNN_STATUS_SUCCESS && count >= 0 &&
            static_cast<std::uint64_t>(count) <= query.maximum)
            return static_cast<std::uint32_t>(count);
    } catch (...) {
        // Optional proof never substitutes for the requested native status.
    }
    return std::nullopt;
}
void backend_observe_metadata_header(const CudaDnnDispatch& dispatch,
                                     cudnnBackendDescriptor_t native,
                                     CudaDnnBackendSnapshot& snapshot) noexcept {
    if (!dispatch.backend_get || (snapshot.kind != 9 && snapshot.kind != 38)) return;
    try {
        std::int64_t count{-1};
        const auto status = dispatch.backend_get(
            native, static_cast<cudnnBackendAttributeName_t>(snapshot.kind == 9 ? 1100 : 2700),
            snapshot.kind == 9 ? CUDNN_TYPE_INT64 : CUDNN_TYPE_INT32, 1, &count, nullptr);
        if (status == CUDNN_STATUS_SUCCESS && count == 1) {
            snapshot.metadata_finalized = true;
            snapshot.is_finalized = true;
            if (snapshot.kind == 38)
                snapshot.property_mode = CudaDnnBackendPropertyMode::kPopulated;
        } else if (status == CUDNN_STATUS_NOT_INITIALIZED) {
            snapshot.metadata_finalized = false;
            snapshot.is_finalized = false;
        }
    } catch (...) {
        // Retain exact copied proof when available; otherwise leave it unknown.
    }
}
CUresult backend_observe_metadata(const CudaDnnDispatch& dispatch, cudnnBackendDescriptor_t native,
                                  CudaDnnBackendSnapshot& snapshot,
                                  const CudaDnnBackendContext* context,
                                  bool allows_context_free = false) noexcept {
    backend_clear_metadata_observations(snapshot);
    if (!snapshot.is_finalized || (snapshot.kind != 2 && snapshot.kind != 9 && snapshot.kind != 38))
        return CUDA_SUCCESS;
    if (snapshot.kind == 2) {
        snapshot.numerical_note_count = backend_observe_count(dispatch, native, {1303, 11, 10});
        if (const auto proof = backend_context_status(snapshot, context, allows_context_free);
            proof != CUDA_SUCCESS)
            return proof;
        snapshot.behavior_note_count = backend_observe_count(dispatch, native, {1305, 19, 4});
    } else if (snapshot.kind == 9)
        snapshot.layout_type_count = backend_observe_count(dispatch, native, {1101, 12, 4});
    else if (snapshot.kind == 38)
        snapshot.json_bytes = backend_observe_count(
            dispatch, native, {2702, 24, static_cast<std::uint32_t>(kBackendMaxElements)});
    return backend_context_status(snapshot, context, allows_context_free);
}
CUresult backend_observe_attribute(const CudaDnnDispatch& dispatch, cudnnBackendDescriptor_t native,
                                   CudaDnnBackendSnapshot& snapshot, std::uint32_t name,
                                   const CudaDnnBackendContext* context) noexcept {
    const bool is_notes = backend_note_array(snapshot, name);
    const bool is_json = backend_property_json(snapshot, name);
    if (!is_notes && !is_json) return CUDA_SUCCESS;
    const bool cpu = backend_cpu_get(snapshot, name);
    if ((snapshot.kind == 9 || snapshot.kind == 38) && !snapshot.metadata_finalized) {
        backend_observe_metadata_header(dispatch, native, snapshot);
        if (const auto proof = backend_context_status(snapshot, context, cpu);
            proof != CUDA_SUCCESS)
            return proof;
    }
    if (snapshot.kind == 2 && snapshot.is_finalized) {
        if (name == 1303)
            snapshot.numerical_note_count = backend_observe_count(dispatch, native, {name, 11, 10});
        else
            snapshot.behavior_note_count = backend_observe_count(dispatch, native, {name, 19, 4});
    } else if (snapshot.kind == 9 && snapshot.metadata_finalized != false)
        snapshot.layout_type_count = backend_observe_count(dispatch, native, {name, 12, 4});
    else if (snapshot.kind == 38 && snapshot.metadata_finalized != false)
        snapshot.json_bytes = backend_observe_count(
            dispatch, native, {name, 24, static_cast<std::uint32_t>(kBackendMaxElements)});
    return backend_context_status(snapshot, context, cpu);
}
std::optional<std::uint32_t> backend_array_length(const CudaDnnBackendSnapshot& snapshot,
                                                  std::uint32_t name) noexcept {
    if (snapshot.kind == 2 && name == 1303) return snapshot.numerical_note_count;
    if (snapshot.kind == 2 && name == 1305) return snapshot.behavior_note_count;
    if (snapshot.kind == 9 && name == 1101) return snapshot.layout_type_count;
    if (snapshot.kind == 38 && name == 2702) return snapshot.json_bytes;
    if (!snapshot.is_valid) return std::nullopt;
    if (snapshot.kind == 16) {
        if (name == 1000) return static_cast<std::uint32_t>(snapshot.variant_uids.size());
        if (name == 1001) return static_cast<std::uint32_t>(snapshot.pointer_slots.size());
    }
    if (snapshot.kind == 17) {
        const auto* dimensions = backend_attribute(snapshot, name);
        if (!dimensions) return 0;
        const auto count = dimensions->scalars.size() / sizeof(std::int64_t);
        // Backend Tensor has nine inline slots, unlike legacy CUDNN_DIM_MAX.
        if (count > 9) return std::nullopt;
        return static_cast<std::uint32_t>(count);
    }
    const auto rank_name = snapshot.kind == 1 ? 106U : 1702U;
    const auto* rank = backend_attribute(snapshot, rank_name);
    if (!rank || rank->scalars.size() != sizeof(std::int64_t)) return 0;
    std::int32_t value{};
    std::memcpy(&value, rank->scalars.data(), sizeof(value));
    if (value < 0 || value > 6) return std::nullopt;
    return static_cast<std::uint32_t>(value);
}
CudaDnnBackendAttributePreparation backend_prepare(
    const CudaDnnBackendSnapshot& snapshot, CudaDnnBackendAttributeOperation operation,
    const CudaDnnBackendAttributeHeader& header) noexcept {
    CudaDnnBackendAttributePreparation result;
    result.result = rejected(CUDA_ERROR_INVALID_VALUE);
    if ((operation != CudaDnnBackendAttributeOperation::kSet &&
         operation != CudaDnnBackendAttributeOperation::kGet) ||
        header.presence > 3 ||
        (operation == CudaDnnBackendAttributeOperation::kSet && (header.presence & 2)) ||
        header.type > 29)
        return result;
    const auto row = backend_row(snapshot, header.name);
    if (!row.types || !(row.types & backend_types(header.type))) {
        result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
        return result;
    }
    result.generation = snapshot.generation;
    result.result = {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
    const bool has_array = (header.presence & 1) != 0;
    if (operation == CudaDnnBackendAttributeOperation::kSet) {
        result.direction = CudaDnnBackendAttributeDirection::kInput;
        if (backend_property_json(snapshot, header.name) && header.count < 0) {
            result.result = rejected(CUDA_ERROR_INVALID_VALUE);
            return result;
        }
        // The original wrapper can log values before checking finalized state.
        if (!has_array) return result;
        if (header.count > static_cast<std::int64_t>(kBackendMaxElements)) {
            result.result = rejected(CUDA_ERROR_INVALID_VALUE);
            return result;
        }
        result.input_elements = header.count > 0 ? static_cast<std::uint32_t>(header.count)
                                                 : (row.ignores_count ? 1U : 0U);
        if (snapshot.kind == 18 && header.name == 1503 && header.count == 1 &&
            backend_width(header.type) == 4)
            result.input_elements = 2;
        return result;
    }
    result.count_bytes = (header.presence & 2) ? sizeof(std::int64_t) : 0;
    // Unlike other descriptor getters, Heur's graph getter dereferences a
    // NULL array for requested==1. Preserve safe original errors, not this trap.
    if (snapshot.kind == 4 && header.name == 201 && !has_array && header.count == 1 &&
        snapshot.is_finalized && (header.presence & 2)) {
        result.result = rejected(CUDA_ERROR_INVALID_VALUE);
        return result;
    }
    // These two native classes intentionally expose only NOT_SUPPORTED Get.
    if (snapshot.kind == 0 || snapshot.kind == 13 ||
        (snapshot.kind == 15 && (header.name == 800 || header.name == 801)) ||
        (snapshot.kind == 16 && header.name != 1003) ||
        (snapshot.kind == 38 && header.name == 2701))
        return result;
    result.direction = row.child          ? CudaDnnBackendAttributeDirection::kDescriptorOutput
                       : header.type == 0 ? CudaDnnBackendAttributeDirection::kHandleOutput
                       : header.type == 6 ? CudaDnnBackendAttributeDirection::kPointerOutput
                                          : CudaDnnBackendAttributeDirection::kScalarOutput;
    if ((header.type == 0 || header.type == 6) && !snapshot.is_valid) {
        // Failed rebinds do not prove which symbolic identity remains installed.
        result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
        return result;
    }
    if (backend_property_json(snapshot, header.name)) {
        // A known unfinalized object reaches the original state error without
        // touching either output. Unknown copied state needs a native proof.
        if (snapshot.metadata_finalized == false || !header.presence ||
            (has_array && header.count < 0))
            return result;
        if (header.count <= 0) {
            if (!(header.presence & 2)) result.result = rejected(CUDA_ERROR_INVALID_VALUE);
            return result;
        }
        if (!snapshot.json_bytes) {
            result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
            return result;
        }
        const auto bytes = *snapshot.json_bytes;
        const bool is_success_path = static_cast<std::uint64_t>(header.count) >= bytes;
        if (is_success_path &&
            (!(header.presence & 2) || (header.count > 0 && bytes && !has_array))) {
            result.result = rejected(CUDA_ERROR_INVALID_VALUE);
            return result;
        }
        if (is_success_path && header.count > 0 && has_array) {
            result.output_elements = bytes;
            result.output_bytes = bytes;
        }
        return result;
    }
    if (!has_array || header.count < 0) return result;
    if (row.child) {
        if (!snapshot.is_valid) {
            result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
            return result;
        }
        if (!snapshot.is_finalized) return result;
        if (backend_layout_copy(snapshot, header.name)) {
            if (header.count > static_cast<std::int64_t>(kBackendMaxElements)) {
                result.result = rejected(CUDA_ERROR_INVALID_VALUE);
                return result;
            }
            result.input_elements = static_cast<std::uint32_t>(header.count);
            return result;
        }
        if (backend_generated_array(snapshot, header.name)) {
            if (!snapshot.generated_array_capacity) {
                result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
                return result;
            }
            if (snapshot.kind == 3) {
                // Config validates every requested destination, even the
                // unwritten suffix beyond its normalized knob table.
                if (header.count > static_cast<std::int64_t>(kBackendMaxElements)) {
                    result.result = rejected(CUDA_ERROR_INVALID_VALUE);
                    return result;
                }
                result.input_elements = static_cast<std::uint32_t>(header.count);
            } else {
                const auto capacity = *snapshot.generated_array_capacity;
                result.input_elements = snapshot.kind == 2 && header.count == 0
                                            ? capacity
                                            : static_cast<std::uint32_t>(
                                                  std::min<std::int64_t>(header.count, capacity));
            }
            return result;
        }
        if (header.count == 1) result.input_elements = 1;
        return result;
    }
    if (row.is_array) {
        if (backend_note_array(snapshot, header.name) &&
            (snapshot.kind == 9 ? snapshot.metadata_finalized == false : !snapshot.is_finalized))
            return result;
        const auto size = backend_array_length(snapshot, header.name);
        if (!size) {
            result.result = rejected(CUDA_ERROR_NOT_SUPPORTED);
            return result;
        }
        result.output_elements =
            backend_note_array(snapshot, header.name) && header.count == 0
                ? *size
                : static_cast<std::uint32_t>(std::min<std::int64_t>(*size, header.count));
    } else if (header.count == 1) {
        result.output_elements = snapshot.kind == 18 && header.name == 1503
                                     ? static_cast<std::uint32_t>(8 / backend_width(header.type))
                                     : 1;
    }
    result.output_bytes =
        static_cast<std::uint32_t>(result.output_elements * backend_width(header.type));
    if (header.type == 0 || header.type == 6) result.output_bytes = 0;
    // The configured SDK declares INT64 but stores just an int32 for this
    // deprecated attribute. Transport padding is not a caller write grant.
    if (((snapshot.kind == 19 && header.name == 1524) ||
         (snapshot.kind == 17 && header.name == 909 && header.type == 3)) &&
        result.output_elements)
        result.output_bytes = 4;
    return result;
}
bool backend_next_generation(const CudaDnnBackendSnapshot& snapshot) noexcept {
    return snapshot.generation != std::numeric_limits<std::uint64_t>::max();
}
bool backend_tensor_scalar(const CudaDnnDispatch& dispatch, cudnnBackendDescriptor_t native,
                           cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type,
                           void* output, std::size_t bytes) noexcept {
    alignas(std::max_align_t) std::array<std::byte, 8> scratch{};
    std::int64_t count{-1};
    const auto status = dispatch.backend_get(native, name, type, 1, &count, scratch.data());
    if (status != CUDNN_STATUS_SUCCESS || count != 1) return false;
    std::memcpy(output, scratch.data(), bytes);
    return true;
}
bool backend_tensor_array(const CudaDnnDispatch& dispatch, cudnnBackendDescriptor_t native,
                          cudnnBackendAttributeName_t name, std::array<std::int64_t, 9>& output,
                          std::uint32_t& rank) noexcept {
    std::int64_t count{-1};
    const auto status =
        dispatch.backend_get(native, name, CUDNN_TYPE_INT64,
                             static_cast<std::int64_t>(output.size()), &count, output.data());
    if (status != CUDNN_STATUS_SUCCESS || count < 0 ||
        count > static_cast<std::int64_t>(output.size()))
        return false;
    rank = static_cast<std::uint32_t>(count);
    // The SDK writes exactly the returned prefix. Unused storage is never
    // native metadata, even if a faulty callback touched it.
    std::fill(output.begin() + count, output.end(), 0);
    return true;
}
bool backend_observe_tensor(const CudaDnnDispatch& dispatch, cudnnBackendDescriptor_t native,
                            CudaDnnBackendTensorObservation& output) noexcept {
    if (!dispatch.backend_get) return false;
    std::uint8_t is_virtual{}, is_by_value{};
    if (!backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE,
                               &output.type, sizeof(output.type)) ||
        output.type > dnn_profile::data_type_enum_max() ||
        !backend_tensor_array(dispatch, native, CUDNN_ATTR_TENSOR_DIMENSIONS, output.dimensions,
                              output.dimension_count) ||
        !backend_tensor_array(dispatch, native, CUDNN_ATTR_TENSOR_STRIDES, output.strides,
                              output.stride_count) ||
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_VECTOR_COUNT, CUDNN_TYPE_INT64,
                               &output.vector_count, sizeof(output.vector_count)) ||
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
                               CUDNN_TYPE_INT64, &output.vectorized_dimension,
                               sizeof(output.vectorized_dimension)) ||
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64,
                               &output.uid, sizeof(output.uid)) ||
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN,
                               &is_virtual, sizeof(is_virtual)) ||
        is_virtual > 1 ||
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN,
                               &is_by_value, sizeof(is_by_value)) ||
        is_by_value > 1 ||
        // 9.10.2 requires INT64 here but writes only the enum's four bytes.
        !backend_tensor_scalar(dispatch, native, CUDNN_ATTR_TENSOR_REORDERING_MODE,
                               CUDNN_TYPE_INT64, &output.reordering, sizeof(output.reordering)) ||
        output.reordering > 3)
        return false;
    output.is_virtual = is_virtual != 0;
    output.is_by_value = is_by_value != 0;
    return true;
}
}  // namespace

CudaDnnResult CudaDnnRegistry::create_backend(std::uint32_t kind) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    if (!backend_kind(kind)) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (!state.dispatch.backend_create || !state.dispatch.backend_destroy ||
        !state.dispatch.version)
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (state.objects.size() == state.maximum || !state.has_id_space)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto object = std::make_shared<Implementation::Object>();
        auto snapshot = std::make_shared<CudaDnnBackendSnapshot>();
        snapshot->kind = kind;
        snapshot->generation = 1;
        snapshot->is_valid = true;
        if (kind == 9 || kind == 38) snapshot->metadata_finalized = false;
        if (kind == 38) {
            snapshot->property_mode = CudaDnnBackendPropertyMode::kDiscovery;
            snapshot->property_device_id = 0;
        }
        if (kind == 17)
            snapshot->tensor_alignment = {1, CudaDnnBackendTensorAlignmentSource::kNativeDefault};
        object->kind = CudaDnnObjectKind::kBackend;
        object->backend = snapshot;
        std::erase_if(state.rnn_calls->backend_objects,
                      [](const auto& entry) { return entry.lifetime.expired(); });
        const DependencyCall active{state.is_dependency_active};
        if (state.dispatch.version() != 91002) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        cudnnBackendDescriptor_t native{};
        const auto status =
            state.dispatch.backend_create(static_cast<cudnnBackendDescriptorType_t>(kind), &native);
        auto result = completed(status);
        if (status != CUDNN_STATUS_SUCCESS && !native && known_rejection(status)) return result;
        object->native = native;
        object->is_uncertain = status != CUDNN_STATUS_SUCCESS || !native ||
                               std::ranges::any_of(state.objects, [native](const auto& prior) {
                                   return prior->native == native;
                               });
        object->id = state.next_id;
        if (state.next_id == std::numeric_limits<std::uint64_t>::max())
            state.has_id_space = false;
        else
            ++state.next_id;
        state.objects.push_back(object);
        state.rnn_calls->backend_objects.push_back({native, object});
        if (object->is_uncertain) {
            state.is_faulted = true;
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        } else
            result.id = object->id;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaDnnRegistry::initialize_backend(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || !(*found)->backend)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.backend_initialize || !state.dispatch.version)
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    if (!backend_next_generation(*(*found)->backend)) return rejected(CUDA_ERROR_INVALID_VALUE);
    try {
        auto next = std::make_shared<CudaDnnBackendSnapshot>();
        next->kind = (*found)->backend->kind;
        next->generation = (*found)->backend->generation + 1;
        const DependencyCall active{state.is_dependency_active};
        if (state.dispatch.version() != 91002) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        const auto status = state.dispatch.backend_initialize((*found)->native);
        next->is_valid = status == CUDNN_STATUS_SUCCESS;
        if (next->is_valid && (next->kind == 9 || next->kind == 38))
            next->metadata_finalized = false;
        if (next->is_valid && next->kind == 38) {
            next->property_mode = CudaDnnBackendPropertyMode::kDiscovery;
            next->property_device_id = 0;
        }
        if (next->is_valid && next->kind == 17)
            next->tensor_alignment = {1, CudaDnnBackendTensorAlignmentSource::kNativeDefault};
        (*found)->backend = std::move(next);
        if (status == CUDNN_STATUS_SUCCESS) {
            (*found)->backend_pointers.clear();
            (*found)->backend_handle_value = nullptr;
        }
        return completed(status);
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaDnnRegistry::finalize_backend(std::uint64_t id,
                                                const CudaDnnBackendContext* context) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || !(*found)->backend)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.backend_finalize || !state.dispatch.version)
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    if (!backend_next_generation(*(*found)->backend)) return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto& previous = *(*found)->backend;
    if (previous.kind == 38 && !context &&
        (previous.property_mode == CudaDnnBackendPropertyMode::kUnknown ||
         previous.property_mode == CudaDnnBackendPropertyMode::kDiscovery))
        return rejected(CUDA_ERROR_INVALID_CONTEXT);
    try {
        auto next = std::make_shared<CudaDnnBackendSnapshot>(*(*found)->backend);
        ++next->generation;
        next->tensor.reset();
        next->generated_array_capacity.reset();
        backend_clear_metadata_observations(*next);
        CudaDnnBackendTensorObservation observation;
        const DependencyCall active{state.is_dependency_active};
        if (const auto proof = backend_context_status(*next, context); proof != CUDA_SUCCESS)
            return rejected(proof);
        const auto version = state.dispatch.version();
        if (const auto proof = backend_context_status(*next, context); proof != CUDA_SUCCESS)
            return rejected(proof);
        if (version != 91002) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        const auto status = state.dispatch.backend_finalize((*found)->native);
        auto proof = backend_context_status(*next, context);
        next->is_finalized = status == CUDNN_STATUS_SUCCESS;
        next->is_valid &= status == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS;
        if (next->kind == 9 || next->kind == 38) {
            if (status == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS) {
                next->is_valid = true;
                next->metadata_finalized = true;
                if (next->kind == 38) next->property_mode = CudaDnnBackendPropertyMode::kPopulated;
            } else
                backend_invalidate_metadata(*next);
        }
        if (status != CUDNN_STATUS_SUCCESS)
            next->tensor_alignment.reset();
        else if (proof == CUDA_SUCCESS && next->kind == 17 && next->tensor_alignment &&
                 backend_observe_tensor(state.dispatch, (*found)->native, observation)) {
            observation.alignment = *next->tensor_alignment;
            next->tensor = observation;
        }
        if (next->is_valid && proof == CUDA_SUCCESS) {
            next->generated_graph = backend_graph_lineage(*next);
            next->generated_property = backend_property_lineage(*next);
            backend_observe_generated_capacity(state.dispatch, (*found)->native, *next);
            proof = backend_context_status(*next, context);
            if (proof == CUDA_SUCCESS)
                proof = backend_observe_metadata(state.dispatch, (*found)->native, *next, context);
            if (proof != CUDA_SUCCESS) {
                next->is_valid = false;
                next->generated_array_capacity.reset();
                backend_invalidate_metadata(*next);
            }
        }
        if (!next->is_valid) {
            next->generated_graph.reset();
            next->generated_property.reset();
        }
        (*found)->backend = std::move(next);
        auto result = completed(status);
        backend_context_failure(result, proof);
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnBackendAttributePreparation CudaDnnRegistry::prepare_backend_attribute(
    CudaDnnBackendAttributeOperation operation, const CudaDnnBackendAttributeHeader& header,
    const CudaDnnBackendContext* context) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(operation == CudaDnnBackendAttributeOperation::kSet);
        status != CUDA_SUCCESS)
        return {.result = rejected(status)};
    const auto found = state.find(header.descriptor);
    if (found == state.objects.end() || !(*found)->backend)
        return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
    if (!state.dispatch.version ||
        (operation == CudaDnnBackendAttributeOperation::kSet && !state.dispatch.backend_set) ||
        (operation == CudaDnnBackendAttributeOperation::kGet && !state.dispatch.backend_get))
        return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
    auto initial = backend_prepare(*(*found)->backend, operation, header);
    const auto row = backend_row(*(*found)->backend, header.name);
    if (header.type > 29 || !(row.types & backend_types(header.type)) || header.presence > 3 ||
        operation != CudaDnnBackendAttributeOperation::kGet ||
        (!backend_note_array(*(*found)->backend, header.name) &&
         !backend_property_json(*(*found)->backend, header.name))) {
        if (initial.result.status != CUDA_SUCCESS) return initial;
    }
    try {
        auto observed = *(*found)->backend;
        const bool cpu = operation == CudaDnnBackendAttributeOperation::kGet &&
                         backend_cpu_get(observed, header.name);
        const DependencyCall active{state.is_dependency_active};
        if (const auto proof = backend_context_status(observed, context, cpu);
            proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        const auto version = state.dispatch.version();
        if (const auto proof = backend_context_status(observed, context, cpu);
            proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        if (version != 91002) return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
        if (operation == CudaDnnBackendAttributeOperation::kGet) {
            const bool requires_observation =
                (backend_note_array(observed, header.name) && (header.presence & 1)) ||
                (backend_property_json(observed, header.name) && header.count > 0);
            if (requires_observation) {
                if (const auto proof = backend_observe_attribute(state.dispatch, (*found)->native,
                                                                 observed, header.name, context);
                    proof != CUDA_SUCCESS)
                    return {.result = rejected(proof)};
            }
        }
        auto result = backend_prepare(observed, operation, header);
        if (result.direction == CudaDnnBackendAttributeDirection::kDescriptorOutput &&
            result.input_elements && state.is_closing)
            return {.result = rejected(CUDA_ERROR_DEINITIALIZED)};
        return result;
    } catch (...) {
        return {.result = rejected(CUDA_ERROR_OUT_OF_MEMORY)};
    }
}
CudaDnnBackendAttributeResult CudaDnnRegistry::set_backend_attribute(
    const CudaDnnBackendAttributeHeader& header, std::uint64_t generation,
    const CudaDnnBackendAttributeValues& values, const CudaDnnBackendContext* context) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS)
        return {.result = rejected(status)};
    const auto found = state.find(header.descriptor);
    if (found == state.objects.end() || !(*found)->backend)
        return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
    const auto& previous = *(*found)->backend;
    if (generation != previous.generation || !backend_next_generation(previous))
        return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
    if (!state.dispatch.backend_set || !state.dispatch.version)
        return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
    const auto preparation =
        backend_prepare(previous, CudaDnnBackendAttributeOperation::kSet, header);
    if (preparation.result.status != CUDA_SUCCESS) return {.result = preparation.result};
    const auto row = backend_row(previous, header.name);
    const auto width = backend_width(header.type);
    const bool is_handle = header.type == 0;
    const bool is_pointer = header.type == 6;
    if ((row.child &&
         (!values.scalars.empty() || values.descriptors.size() != preparation.input_elements)) ||
        (!row.child && !values.descriptors.empty()) ||
        (is_handle &&
         (values.handles.size() != preparation.input_elements || !values.scalars.empty())) ||
        (!is_handle && !values.handles.empty()) ||
        (is_pointer &&
         (values.pointer_slots.size() != preparation.input_elements || !values.scalars.empty())) ||
        (!is_pointer && !values.pointer_slots.empty()) ||
        (!row.child && !is_handle && !is_pointer &&
         (values.scalars.size() != preparation.input_elements * width ||
          !backend_scalar_bits(header.type, values.scalars))))
        return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
    if (found->use_count() != 1)
        return {.result = {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS}};
    try {
        std::vector<std::shared_ptr<Implementation::Object>> pins;
        std::vector<cudnnBackendDescriptor_t> native;
        std::vector<void*> pointers;
        auto pointer_owners = (*found)->backend_pointers;
        pointers.reserve(preparation.input_elements);
        pins.reserve(values.descriptors.size() + 1);
        native.reserve(values.descriptors.size());
        CudaDnnBackendSnapshotAttribute attribute{.name = header.name,
                                                  .type = header.type,
                                                  .count = header.count,
                                                  .scalars = values.scalars,
                                                  .handles = values.handles,
                                                  .pointer_slots = values.pointer_slots};
        attribute.descriptors.reserve(values.descriptors.size());
        for (const auto id : values.descriptors) {
            if (!id) {
                native.push_back(nullptr);
                attribute.descriptors.push_back(nullptr);
                continue;
            }
            const auto child = state.find(id);
            if (child == state.objects.end() || !(*child)->backend ||
                !backend_child_kind(row, (*child)->backend->kind) || (*child)->is_uncertain ||
                (*child)->has_destroy_attempt)
                return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
            if ((*child)->backend->kind != 38 && (*child)->backend->context &&
                (!context || (*child)->backend->context->device_ordinal != context->device_ordinal))
                return {.result = rejected(CUDA_ERROR_INVALID_CONTEXT)};
            pins.push_back(*child);
            native.push_back((*child)->native);
            attribute.descriptors.push_back((*child)->backend);
        }
        // Native logging may inspect the captured caller prefix, whereas these
        // inline descriptor fields retain only their first semantic value.
        if (row.child && !row.is_array && attribute.descriptors.size() > 1)
            attribute.descriptors.resize(1);
        {
            const DependencyCall resolution{state.is_dependency_active};
            for (const auto& reference : values.handles) {
                if (!reference.owner && !reference.id) {
                    pointers.push_back(nullptr);
                    continue;
                }
                if (!context || !reference.id || reference.owner != context->owner)
                    return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
                const auto resolved = context->resolve_handle
                                          ? context->resolve_handle(reference.id, context->opaque)
                                      : reference.id == context->handle ? context->native
                                                                        : nullptr;
                if (!resolved) return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
                pointers.push_back(resolved);
            }
        }
        for (const auto slot : values.pointer_slots) {
            if (!slot) {
                pointers.push_back(nullptr);
                continue;
            }
            auto identity = std::ranges::find_if(
                pointer_owners, [slot](const auto& entry) { return entry->slot == slot; });
            if (identity == pointer_owners.end()) {
                if (pointer_owners.size() == kBackendMaxElements)
                    return {.result = rejected(CUDA_ERROR_OUT_OF_MEMORY)};
                auto owner = std::make_shared<Implementation::Object::PointerIdentity>();
                owner->slot = slot;
                pointers.push_back(owner.get());
                pointer_owners.push_back(std::move(owner));
            } else
                pointers.push_back(identity->get());
        }
        const auto retained_bytes = previous.kind == 18 && header.name == 1503 ? 8 : width;
        if (!row.child && !row.is_array && attribute.scalars.size() > retained_bytes)
            attribute.scalars.resize(retained_bytes);
        if (row.is_array && !backend_property_json(previous, header.name)) {
            const auto slots = previous.kind == 17 ? 9U : previous.kind == 16 ? 100U : 6U;
            if (attribute.scalars.size() > slots * width) attribute.scalars.resize(slots * width);
        }
        pins.push_back(*found);
        auto next = std::make_shared<CudaDnnBackendSnapshot>(previous);
        ++next->generation;
        next->tensor.reset();
        next->generated_array_capacity.reset();
        next->generated_graph.reset();
        backend_clear_metadata_observations(*next);
        if (backend_planning_kind(previous.kind) && context)
            next->context = {context->owner, context->lifetime, context->device_ordinal};
        for (const auto& child : attribute.descriptors)
            if (child && !child->is_valid) next->is_valid = false;
        const auto prior = std::ranges::find_if(
            next->attributes, [&](const auto& entry) { return entry.name == header.name; });
        if (previous.kind == 15 && header.name == 801 && prior != next->attributes.end()) {
            if (prior->descriptors.size() > kBackendMaxElements - attribute.descriptors.size())
                return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
            prior->descriptors.insert(prior->descriptors.end(), attribute.descriptors.begin(),
                                      attribute.descriptors.end());
            prior->count = static_cast<std::int64_t>(prior->descriptors.size());
        } else if (prior == next->attributes.end())
            next->attributes.push_back(std::move(attribute));
        else
            *prior = std::move(attribute);
        if (previous.kind == 16 && header.name == 1000 && header.count > 0) {
            next->variant_uids.resize(std::min<std::size_t>(preparation.input_elements, 100));
            if (!next->variant_uids.empty())
                std::memcpy(next->variant_uids.data(), values.scalars.data(),
                            next->variant_uids.size() * sizeof(std::int64_t));
        }
        if (previous.kind == 16 && header.name == 1001 && header.count > 0) {
            const auto slots = std::span{values.pointer_slots}.first(
                std::min<std::size_t>(values.pointer_slots.size(), 100));
            next->pointer_slots.assign(slots.begin(), slots.end());
        }
        if (previous.kind == 16 && header.name == 1003 && !values.pointer_slots.empty())
            next->workspace_slot = values.pointer_slots.front();
        alignas(std::max_align_t) std::array<std::byte, 16> dummy{};
        const void* array = nullptr;
        if (header.presence & 1)
            array = (is_handle || is_pointer)
                        ? (pointers.empty() ? static_cast<void*>(dummy.data())
                                            : static_cast<void*>(pointers.data()))
                    : row.child ? (native.empty() ? static_cast<void*>(dummy.data())
                                                  : static_cast<void*>(native.data()))
                                : (values.scalars.empty() ? static_cast<const void*>(dummy.data())
                                                          : values.scalars.data());
        const DependencyCall active{state.is_dependency_active};
        if (const auto proof = backend_context_status(previous, context); proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        const auto version = state.dispatch.version();
        if (const auto proof = backend_context_status(previous, context); proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        if (version != 91002) return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
        const auto status = state.dispatch.backend_set(
            (*found)->native, static_cast<cudnnBackendAttributeName_t>(header.name),
            static_cast<cudnnBackendAttributeType_t>(header.type), header.count, array);
        const auto proof = backend_context_status(previous, context);
        // Even a failed native setter may retain a prefix of pointer values.
        (*found)->backend_pointers.swap(pointer_owners);
        if (previous.kind == 5 && header.name == 400 && status == CUDNN_STATUS_SUCCESS &&
            !pointers.empty())
            (*found)->backend_handle_value = static_cast<cudnnHandle_t>(pointers.front());
        const bool was_finalized = previous.is_finalized;
        next->is_valid &= status == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS;
        bool malformed_property = false;
        if (previous.kind == 38) {
            backend_invalidate_metadata(*next);
            if (status == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS && !was_finalized) {
                if (header.name == 2700 && header.count == 1 && (header.presence & 1) &&
                    values.scalars.size() == sizeof(std::int32_t)) {
                    std::int32_t ordinal{};
                    std::memcpy(&ordinal, values.scalars.data(), sizeof(ordinal));
                    next->property_device_id = ordinal;
                    next->property_mode = CudaDnnBackendPropertyMode::kDiscovery;
                } else if (header.name == 2701 && header.count == 1 && pointers.size() == 1 &&
                           pointers.front())
                    next->property_mode = CudaDnnBackendPropertyMode::kHandle;
                else if (header.name == 2702 && header.count >= 0 &&
                         (header.count == 0 || (header.presence & 1)))
                    next->property_mode = CudaDnnBackendPropertyMode::kJson;
                else
                    malformed_property = true;
                next->metadata_finalized = false;
                next->is_valid = !malformed_property;
            }
        } else if (previous.kind == 9 && status != CUDNN_STATUS_SUCCESS)
            backend_invalidate_metadata(*next);
        if (next->is_valid) {
            next->generated_graph = backend_graph_lineage(*next);
            next->generated_property = backend_property_lineage(*next);
        } else
            next->generated_property.reset();
        if (status != CUDNN_STATUS_SUCCESS || was_finalized)
            next->tensor_alignment.reset();
        else if (previous.kind == 17 && header.name == 900) {
            // This original setter stores exactly its first INT64. Its getter
            // is unsupported, so failed or contradictory receipts confer no proof.
            next->tensor_alignment.reset();
            if (header.type == 3 && header.count == 1 && (header.presence & 1) &&
                values.scalars.size() == sizeof(std::int64_t)) {
                std::int64_t alignment{};
                std::memcpy(&alignment, values.scalars.data(), sizeof(alignment));
                next->tensor_alignment = {alignment,
                                          CudaDnnBackendTensorAlignmentSource::kNativeSet};
            }
        }
        const bool is_read_only =
            previous.kind == 9 ||
            (previous.kind == 2 && header.name >= 1303 && header.name <= 1305);
        (*found)->backend = next;
        if (status == CUDNN_STATUS_SUCCESS &&
            (was_finalized || malformed_property || is_read_only)) {
            next->is_valid = false;
            backend_invalidate_metadata(*next);
            auto failure = completed(status);
            failure.disposition = CudaDnnDisposition::kRejected;
            failure.status = CUDA_ERROR_UNKNOWN;
            backend_context_failure(failure, proof);
            return {.result = std::move(failure),
                    .generation = next->generation,
                    .has_context_loss = proof != CUDA_SUCCESS};
        }
        auto native_result = completed(status);
        backend_context_failure(native_result, proof);
        return {.result = std::move(native_result),
                .generation = next->generation,
                .is_configuration_valid = next->is_valid,
                .has_context_loss = proof != CUDA_SUCCESS};
    } catch (...) {
        return {.result = rejected(CUDA_ERROR_OUT_OF_MEMORY)};
    }
}
CudaDnnBackendAttributeResult CudaDnnRegistry::get_backend_attribute(
    const CudaDnnBackendAttributeHeader& header, std::uint64_t generation,
    const CudaDnnBackendAttributeValues& values, const CudaDnnBackendContext* context) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS)
        return {.result = rejected(status)};
    const auto found = state.find(header.descriptor);
    if (found == state.objects.end() || !(*found)->backend)
        return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
    if (generation != (*found)->backend->generation)
        return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
    if (!state.dispatch.backend_get || !state.dispatch.version)
        return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
    try {
        auto previous = *(*found)->backend;
        const auto row = backend_row(previous, header.name);
        if (header.presence > 3 || header.type > 29 || !(row.types & backend_types(header.type)))
            return {.result =
                        backend_prepare(previous, CudaDnnBackendAttributeOperation::kGet, header)
                            .result};
        const bool observes =
            (backend_note_array(previous, header.name) && (header.presence & 1)) ||
            (backend_property_json(previous, header.name) && header.count > 0);
        const bool cpu = backend_cpu_get(previous, header.name);
        const DependencyCall observation_scope{state.is_dependency_active, observes};
        if (observes) {
            if (const auto proof = backend_context_status(previous, context, cpu);
                proof != CUDA_SUCCESS)
                return {.result = rejected(proof)};
            const auto version = state.dispatch.version();
            if (const auto proof = backend_context_status(previous, context, cpu);
                proof != CUDA_SUCCESS)
                return {.result = rejected(proof)};
            if (version != 91002) return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
            if (const auto proof = backend_observe_attribute(state.dispatch, (*found)->native,
                                                             previous, header.name, context);
                proof != CUDA_SUCCESS)
                return {.result = rejected(proof)};
        }
        const auto preparation =
            backend_prepare(previous, CudaDnnBackendAttributeOperation::kGet, header);
        if (preparation.result.status != CUDA_SUCCESS) return {.result = preparation.result};
        const bool generated = backend_generated_array(previous, header.name);
        const bool layouts = backend_layout_copy(previous, header.name);
        const bool property = backend_property_copy(previous, header.name);
        if (preparation.input_elements && state.is_closing)
            return {.result = rejected(CUDA_ERROR_DEINITIALIZED)};
        if (!values.scalars.empty() || !values.handles.empty() || !values.pointer_slots.empty() ||
            values.descriptors.size() != preparation.input_elements)
            return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
        std::vector<std::shared_ptr<Implementation::Object>> destinations;
        std::vector<std::shared_ptr<CudaDnnBackendSnapshot>> candidates;
        std::vector<std::size_t> positions;
        std::vector<cudnnBackendDescriptor_t> native;
        destinations.reserve(values.descriptors.size());
        candidates.reserve(values.descriptors.size());
        positions.reserve(values.descriptors.size());
        native.reserve(values.descriptors.size());
        // Exclusivity precedes any self-held reference, including source aliases.
        for (const auto id : values.descriptors) {
            if (!id) {
                // Engine's KnobInfo writer does not validate destination
                // pointers. Other rows retain their safe original NULL errors.
                if (previous.kind == 2 && header.name == 1302)
                    return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
                native.push_back(nullptr);
                continue;
            }
            const auto destination = state.find(id);
            if (destination == state.objects.end() || !(*destination)->backend ||
                !backend_child_kind(row, (*destination)->backend->kind) ||
                (*destination)->is_uncertain || (*destination)->has_destroy_attempt)
                return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
            if (destination->use_count() != 1)
                return {.result = {.disposition = CudaDnnDisposition::kDeferred,
                                   .status = CUDA_SUCCESS}};
            if (!backend_next_generation(*(*destination)->backend))
                return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
            native.push_back((*destination)->native);
        }
        const auto* attribute = backend_attribute(previous, header.name);
        for (std::size_t index = 0; index < values.descriptors.size(); ++index) {
            if (!values.descriptors[index]) continue;
            const auto destination = state.find(values.descriptors[index]);
            auto next = std::make_shared<CudaDnnBackendSnapshot>(*(*destination)->backend);
            const auto next_generation = next->generation + 1;
            if (layouts || property) {
                *next = {};
                next->kind = row.child - 1;
                next->is_valid = true;
                const auto copied_property = backend_property_lineage(previous);
                if (property && previous.is_valid && copied_property) *next = *copied_property;
            } else if (generated ||
                       (previous.kind == 3 && header.name == 300 && previous.generated_graph)) {
                *next = {};
                next->kind = row.child - 1;
                next->is_finalized = true;
                next->context = previous.context;
                if (next->kind == 2 || next->kind == 3) {
                    next->generated_graph = backend_graph_lineage(previous);
                    next->generated_property = backend_property_lineage(previous);
                    next->is_valid = previous.is_valid && next->generated_graph &&
                                     next->generated_graph->is_valid;
                } else {
                    // Knob values remain native-only. They carry no device or
                    // graph authority and remain queryable after context loss.
                    next->is_valid = previous.is_valid;
                    next->context.reset();
                }
            } else if (previous.kind == 2 && header.name == 1300 && previous.generated_graph)
                *next = *previous.generated_graph;
            else if (previous.is_valid && attribute && index < attribute->descriptors.size() &&
                     attribute->descriptors[index])
                *next = *attribute->descriptors[index];
            else {
                next->is_valid = false;
                next->tensor.reset();
                next->tensor_alignment.reset();
                next->generated_graph.reset();
                next->generated_array_capacity.reset();
                backend_invalidate_metadata(*next);
            }
            next->generation = next_generation;
            // Keep every position until the actual written prefix is known.
            // An unused aliased suffix must not replace an earlier result.
            destinations.push_back(*destination);
            candidates.push_back(std::move(next));
            positions.push_back(index);
        }
        auto source = *found;
        CudaDnnBackendAttributeResult result;
        result.generation = previous.generation;
        result.is_configuration_valid = previous.is_valid;
        if (header.type == 0)
            result.values.handles.resize(preparation.output_elements);
        else if (header.type == 6)
            result.values.pointer_slots.resize(preparation.output_elements);
        else
            result.values.scalars.resize(preparation.output_elements * backend_width(header.type));
        // Scratch is independent of requested capacity and contains all valid
        // inline metadata. Only the exact prepared prefix may be published.
        alignas(std::max_align_t) std::array<std::byte, 256> scratch{};
        std::vector<std::byte> json_scratch;
        if (backend_property_json(previous, header.name) && previous.json_bytes)
            json_scratch.resize(*previous.json_bytes);
        void* scalar_scratch = json_scratch.empty() ? static_cast<void*>(scratch.data())
                                                    : static_cast<void*>(json_scratch.data());
        void* array = nullptr;
        if (header.presence & 1)
            array = row.child ? (native.empty() ? static_cast<void*>(scratch.data())
                                                : static_cast<void*>(native.data()))
                              : scalar_scratch;
        std::int64_t count{};
        const DependencyCall active{state.is_dependency_active, !observes};
        if (const auto proof = backend_context_status(previous, context, cpu);
            proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        const auto version = observes ? 91002 : state.dispatch.version();
        if (const auto proof = backend_context_status(previous, context, cpu);
            proof != CUDA_SUCCESS)
            return {.result = rejected(proof)};
        if (version != 91002) return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};
        const auto status = state.dispatch.backend_get(
            source->native, static_cast<cudnnBackendAttributeName_t>(header.name),
            static_cast<cudnnBackendAttributeType_t>(header.type), header.count,
            ((header.presence & 2) || ((generated || layouts) && (header.presence & 1))) ? &count
                                                                                         : nullptr,
            array);
        result.result = completed(status);
        const auto proof = backend_context_status(previous, context, cpu);
        result.has_context_loss = proof != CUDA_SUCCESS;
        backend_context_failure(result.result, proof);
        bool malformed =
            status == CUDNN_STATUS_SUCCESS &&
            ((!previous.is_finalized && !((previous.kind == 9 || previous.kind == 38) &&
                                          !previous.metadata_finalized.has_value())) ||
             preparation.direction == CudaDnnBackendAttributeDirection::kNone ||
             !(header.presence & 3) || ((header.presence & 1) && header.count < 0));
        for (std::size_t index = 0; index < native.size(); ++index) {
            const auto id = values.descriptors[index];
            const auto expected = id ? (*state.find(id))->native : nullptr;
            malformed |= native[index] != expected;
        }
        if (status != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS) {
            for (std::size_t index = 0; index < destinations.size(); ++index) {
                candidates[index]->is_valid = false;
                candidates[index]->tensor.reset();
                candidates[index]->tensor_alignment.reset();
                candidates[index]->generated_graph.reset();
                candidates[index]->generated_array_capacity.reset();
                backend_invalidate_metadata(*candidates[index]);
                destinations[index]->backend = candidates[index];
            }
            result.values = {};
            return result;
        }
        std::size_t written = native.size();
        if (layouts && ((header.presence & 2) || (header.presence & 1))) {
            if (count < 0 || count > std::max<std::int64_t>(0, header.count)) malformed = true;
            if (header.presence & 1) {
                if (static_cast<std::uint64_t>(count) > native.size()) malformed = true;
                if (!malformed) written = static_cast<std::size_t>(count);
            }
        } else if (generated && ((header.presence & 2) || (header.presence & 1))) {
            const auto capacity = previous.generated_array_capacity;
            if (count < 0 || (previous.kind != 4 && count > 43)) malformed = true;
            if (header.presence & 1) {
                if (header.count < 0 || !capacity ||
                    static_cast<std::uint64_t>(count) > preparation.input_elements)
                    malformed = true;
                else if (previous.kind != 4) {
                    const auto expected = previous.kind == 2 && header.count == 0
                                              ? *capacity
                                              : static_cast<std::uint32_t>(std::min<std::int64_t>(
                                                    *capacity, header.count));
                    malformed |= count != expected;
                }
                if (!malformed) written = static_cast<std::size_t>(count);
            } else if (capacity) {
                const auto expected =
                    previous.kind == 4 || (previous.kind == 2 && header.count == 0)
                        ? static_cast<std::int64_t>(*capacity)
                        : std::max<std::int64_t>(0,
                                                 std::min<std::int64_t>(*capacity, header.count));
                malformed |= count != expected;
            }
        } else if ((header.presence & 2) && backend_note_array(previous, header.name)) {
            const auto maximum = header.name == 1303 ? 10 : 4;
            malformed |= count < 0 || count > maximum || (header.count < 0 && count != 0) ||
                         (header.count > 0 && count > header.count);
            if (header.presence & 1)
                malformed |= count != preparation.output_elements;
            else if (const auto length = backend_array_length(previous, header.name)) {
                const auto expected =
                    header.count == 0
                        ? static_cast<std::int64_t>(*length)
                        : std::max<std::int64_t>(0, std::min<std::int64_t>(*length, header.count));
                malformed |= count != expected;
            }
        } else if ((header.presence & 2) && backend_property_json(previous, header.name)) {
            malformed |= count < 0;
            if (previous.json_bytes) malformed |= count != *previous.json_bytes;
            if (header.count > 0)
                malformed |= count != preparation.output_bytes ||
                             static_cast<std::uint64_t>(count) > kBackendMaxElements;
        } else if (header.presence & 2) {
            std::optional<std::int64_t> expected_count{1};
            if (row.is_array) {
                const auto length = backend_array_length(previous, header.name);
                if (length)
                    expected_count = (header.presence & 1)
                                         ? static_cast<std::int64_t>(preparation.output_elements)
                                         : *length;
                else
                    expected_count.reset();
            }
            if (!expected_count || count != *expected_count) malformed = true;
        }
        if (preparation.output_bytes)
            std::memcpy(result.values.scalars.data(), scalar_scratch, preparation.output_bytes);
        if (layouts || property)
            for (std::size_t index = 0; index < written; ++index)
                if (!native[index]) malformed = true;
        if (!row.child && header.type != 0 && header.type != 6 &&
            !backend_scalar_bits(header.type, result.values.scalars))
            malformed = true;
        if (backend_note_array(previous, header.name)) {
            std::uint32_t prior{};
            for (std::uint32_t index = 0; index < preparation.output_elements; ++index) {
                std::uint32_t value{};
                std::memcpy(&value, result.values.scalars.data() + index * sizeof(value),
                            sizeof(value));
                malformed |= value >= (header.name == 1303 ? 10U : 4U) || (index && value <= prior);
                prior = value;
            }
        }
        if (header.type == 0 && preparation.output_elements) {
            std::array<cudnnHandle_t, 1> observed{};
            std::memcpy(observed.data(), scratch.data(), sizeof(observed));
            if (observed.front() != source->backend_handle_value) malformed = true;
            if (attribute && !attribute->handles.empty())
                result.values.handles.front() = attribute->handles.front();
            else if (observed.front())
                malformed = true;
        }
        if (header.type == 6 && preparation.output_elements) {
            void* observed{};
            std::memcpy(&observed, scratch.data(), sizeof(observed));
            if (observed) {
                const auto identity = std::ranges::find_if(
                    source->backend_pointers,
                    [observed](const auto& entry) { return entry.get() == observed; });
                if (identity == source->backend_pointers.end())
                    malformed = true;
                else
                    result.values.pointer_slots.front() = (*identity)->slot;
            }
            if (result.values.pointer_slots.front() != previous.workspace_slot) malformed = true;
        }
        if (previous.kind == 17 && header.name == 909) {
            // This getter's INT64 transport is not its enum representation.
            std::uint32_t reordering{};
            if (preparation.output_bytes)
                std::memcpy(&reordering, scratch.data(), sizeof(reordering));
            if (header.type != 3 || reordering > 3) malformed = true;
        }
        for (std::size_t index = 0; index < destinations.size(); ++index) {
            if (!malformed && positions[index] >= written) continue;
            if (malformed) {
                candidates[index]->is_valid = false;
                candidates[index]->tensor.reset();
                candidates[index]->tensor_alignment.reset();
                candidates[index]->generated_graph.reset();
                candidates[index]->generated_array_capacity.reset();
                backend_invalidate_metadata(*candidates[index]);
            } else if (!result.has_context_loss) {
                // Private capacity queries never supply public Get provenance.
                // All candidate storage was allocated before native mutation.
                backend_observe_generated_capacity(state.dispatch, destinations[index]->native,
                                                   *candidates[index]);
                auto copy_proof = backend_context_status(previous, context, cpu);
                if (copy_proof == CUDA_SUCCESS && (layouts || property)) {
                    backend_observe_metadata_header(state.dispatch, destinations[index]->native,
                                                    *candidates[index]);
                    copy_proof = backend_context_status(previous, context, cpu);
                }
                if (copy_proof == CUDA_SUCCESS)
                    copy_proof =
                        backend_observe_metadata(state.dispatch, destinations[index]->native,
                                                 *candidates[index], context, cpu);
                if (copy_proof != CUDA_SUCCESS) {
                    result.has_context_loss = true;
                    backend_context_failure(result.result, copy_proof);
                    candidates[index]->is_valid = false;
                    candidates[index]->generated_graph.reset();
                    candidates[index]->generated_array_capacity.reset();
                    backend_invalidate_metadata(*candidates[index]);
                }
            }
            destinations[index]->backend = candidates[index];
        }
        if (result.has_context_loss) {
            // Loss during a private observation invalidates every written
            // destination, including earlier aliases; ownership remains known.
            for (std::size_t index = 0; index < destinations.size(); ++index) {
                if (positions[index] >= written) continue;
                candidates[index]->is_valid = false;
                candidates[index]->generated_graph.reset();
                candidates[index]->generated_array_capacity.reset();
                backend_invalidate_metadata(*candidates[index]);
                destinations[index]->backend = candidates[index];
            }
            result.values = {};
            return result;
        }
        if (malformed) {
            result.values = {};
            result.result.disposition = CudaDnnDisposition::kRejected;
            result.result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        result.count_bytes = preparation.count_bytes;
        result.count = result.count_bytes ? count : 0;
        return result;
    } catch (...) {
        return {.result = rejected(CUDA_ERROR_OUT_OF_MEMORY)};
    }
}
std::optional<CudaDnnBackendView> CudaDnnRegistry::borrow_backend(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (state.gate(true) != CUDA_SUCCESS) return std::nullopt;
    const auto found = state.find(id);
    if (found == state.objects.end() || !(*found)->backend || (*found)->is_uncertain ||
        (*found)->has_destroy_attempt)
        return std::nullopt;
    return CudaDnnBackendView{{CudaDnnObjectKind::kBackend, (*found)->native, *found},
                              (*found)->backend};
}
std::optional<CudaDnnBackendCallView> CudaDnnRegistry::borrow_backend_call(
    std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (state.gate(true) != CUDA_SUCCESS) return std::nullopt;
    const auto found = state.find(id);
    if (found == state.objects.end() || !(*found)->backend || (*found)->is_uncertain ||
        (*found)->has_destroy_attempt)
        return std::nullopt;
    try {
        auto call = std::shared_ptr<CudaDnnRnnLease>(new CudaDnnRnnLease(state.rnn_calls, *found));
        return CudaDnnBackendCallView{
            {{CudaDnnObjectKind::kBackend, (*found)->native, *found}, (*found)->backend},
            std::move(call)};
    } catch (...) {
        return std::nullopt;
    }
}
CudaDnnResult CudaDnnRegistry::set_tensor(std::uint64_t id, CudaDnnTensorSetter setter,
                                          const CudaDnnTensor& tensor) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    // These are C++ enum representability and bounded-array checks, not vendor
    // shape/type support decisions. Representable unknown values reach cuDNN.
    const bool is_packed =
        setter == CudaDnnTensorSetter::kFour || setter == CudaDnnTensorSetter::kNdPacked;
    const bool is_four =
        setter == CudaDnnTensorSetter::kFour || setter == CudaDnnTensorSetter::kFourStrided;
    if ((setter != CudaDnnTensorSetter::kFour && setter != CudaDnnTensorSetter::kFourStrided &&
         setter != CudaDnnTensorSetter::kNd && setter != CudaDnnTensorSetter::kNdPacked) ||
        tensor.type > dnn_profile::data_type_enum_max() || (is_packed && tensor.format > 3) ||
        tensor.dimensions < 0 || tensor.dimensions > CUDNN_DIM_MAX ||
        (is_four && tensor.dimensions != 4))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const auto native = static_cast<cudnnTensorDescriptor_t>((*found)->native);
    const auto type = static_cast<cudnnDataType_t>(tensor.type);
    const auto& d = tensor.sizes;
    const auto& s = tensor.strides;
    cudnnStatus_t status{};
    switch (setter) {
        case CudaDnnTensorSetter::kFour:
            status = state.dispatch.tensor_set_4d(native,
                                                  static_cast<cudnnTensorFormat_t>(tensor.format),
                                                  type, d[0], d[1], d[2], d[3]);
            break;
        case CudaDnnTensorSetter::kFourStrided:
            status = state.dispatch.tensor_set_4d_ex(native, type, d[0], d[1], d[2], d[3], s[0],
                                                     s[1], s[2], s[3]);
            break;
        case CudaDnnTensorSetter::kNd:
            status =
                state.dispatch.tensor_set_nd(native, type, tensor.dimensions, d.data(), s.data());
            break;
        case CudaDnnTensorSetter::kNdPacked:
            status = state.dispatch.tensor_set_nd_ex(
                native, static_cast<cudnnTensorFormat_t>(tensor.format), type, tensor.dimensions,
                d.data());
            break;
        default:
            return rejected(CUDA_ERROR_INVALID_VALUE);
    }
    return state.mutation(**found, status);
}
CudaDnnResult CudaDnnRegistry::get_tensor(std::uint64_t id, int dimensions, bool is_four) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (dimensions < 0 || dimensions > CUDNN_DIM_MAX || (is_four && dimensions != 4))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    CudaDnnTensor tensor;
    cudnnDataType_t type{};
    const auto native = static_cast<cudnnTensorDescriptor_t>((*found)->native);
    auto& d = tensor.sizes;
    auto& s = tensor.strides;
    cudnnStatus_t status{};
    if (is_four) {
        tensor.dimensions = 4;
        status = state.dispatch.tensor_get_4d(native, &type, &d[0], &d[1], &d[2], &d[3], &s[0],
                                              &s[1], &s[2], &s[3]);
    } else
        status = state.dispatch.tensor_get_nd(native, dimensions, &type, &tensor.dimensions,
                                              d.data(), s.data());
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) return result;
    static_assert(sizeof(type) == sizeof(tensor.type));
    std::memcpy(&tensor.type, &type, sizeof(type));
    // nbDims is the actual rank, not the number of entries copied. A caller may
    // request a prefix (including zero entries) without changing that rank.
    if (tensor.dimensions < 0 || tensor.dimensions > CUDNN_DIM_MAX ||
        tensor.type > dnn_profile::data_type_enum_max()) {
        (*found)->is_uncertain = state.is_faulted = true;
        result.disposition = CudaDnnDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    const auto copied = static_cast<std::size_t>(std::min(dimensions, tensor.dimensions));
    std::fill(tensor.sizes.begin() + copied, tensor.sizes.end(), 0);
    std::fill(tensor.strides.begin() + copied, tensor.strides.end(), 0);
    result.tensor = tensor;
    return result;
}
CudaDnnResult CudaDnnRegistry::tensor_size(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    std::size_t bytes{};
    auto result = completed(
        state.dispatch.tensor_size(static_cast<cudnnTensorDescriptor_t>((*found)->native), &bytes));
    if (result.original_status == CUDNN_STATUS_SUCCESS) result.value = bytes;
    return result;
}
CudaDnnResult CudaDnnRegistry::set_activation(std::uint64_t id,
                                              CudaDnnActivationState activation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kActivation)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (activation.mode > 7 || activation.nan > 1) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    return state.mutation(
        **found, state.dispatch.activation_set(static_cast<CudaDnnActivation>((*found)->native),
                                               static_cast<cudnnActivationMode_t>(activation.mode),
                                               static_cast<cudnnNanPropagation_t>(activation.nan),
                                               activation.coefficient));
}
CudaDnnResult CudaDnnRegistry::get_activation(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kActivation)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    cudnnActivationMode_t mode{};
    cudnnNanPropagation_t nan{};
    double coefficient{};
    auto result = completed(state.dispatch.activation_get(
        static_cast<CudaDnnActivation>((*found)->native), &mode, &nan, &coefficient));
    if (result.original_status == CUDNN_STATUS_SUCCESS) {
        std::uint32_t mode_bits{}, nan_bits{};
        static_assert(sizeof(mode) == sizeof(mode_bits) && sizeof(nan) == sizeof(nan_bits));
        std::memcpy(&mode_bits, &mode, sizeof(mode_bits));
        std::memcpy(&nan_bits, &nan, sizeof(nan_bits));
        if (mode_bits > 7 || nan_bits > 1) {
            (*found)->is_uncertain = state.is_faulted = true;
            result.disposition = CudaDnnDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        result.activation = {mode_bits, nan_bits, coefficient};
    }
    return result;
}
CudaDnnResult CudaDnnRegistry::set_filter(std::uint64_t id, const CudaDnnFilter& filter,
                                          bool is_four) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kFilter)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (filter.type > dnn_profile::data_type_enum_max() || filter.format > 3 ||
        filter.dimensions < 0 || filter.dimensions > CUDNN_DIM_MAX ||
        (is_four && filter.dimensions != 4))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((is_four && !state.dispatch.filter_set_4d) || (!is_four && !state.dispatch.filter_set_nd))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const auto native = static_cast<CudaDnnFilterHandle>((*found)->native);
    const auto type = static_cast<cudnnDataType_t>(filter.type);
    const auto format = static_cast<cudnnTensorFormat_t>(filter.format);
    const auto& sizes = filter.sizes;
    const auto status = is_four ? state.dispatch.filter_set_4d(native, type, format, sizes[0],
                                                               sizes[1], sizes[2], sizes[3])
                                : state.dispatch.filter_set_nd(native, type, format,
                                                               filter.dimensions, sizes.data());
    return state.mutation(**found, status);
}
CudaDnnResult CudaDnnRegistry::get_filter(std::uint64_t id, int requested, bool is_four) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kFilter)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (is_four && requested != 4) return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((is_four && !state.dispatch.filter_get_4d) || (!is_four && !state.dispatch.filter_get_nd))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    CudaDnnFilter filter;
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    const auto native = static_cast<CudaDnnFilterHandle>((*found)->native);
    auto& sizes = filter.sizes;
    cudnnStatus_t status{};
    if (is_four) {
        filter.dimensions = 4;
        status = state.dispatch.filter_get_4d(native, &type, &format, &sizes[0], &sizes[1],
                                              &sizes[2], &sizes[3]);
    } else {
        // Capacity does not determine allocation: the descriptor's actual rank
        // is SDK-bounded to eight. Preserve native rejection of negative requests.
        status = state.dispatch.filter_get_nd(native, requested, &type, &format, &filter.dimensions,
                                              sizes.data());
    }
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) return result;
    filter.type = raw_bits(type);
    filter.format = raw_bits(format);
    if (requested < 0 || filter.dimensions < 0 || filter.dimensions > CUDNN_DIM_MAX ||
        filter.type > dnn_profile::data_type_enum_max() || filter.format > 3)
        return state.malformed(**found, result);
    const auto copied = static_cast<std::size_t>(std::min(requested, filter.dimensions));
    std::fill(sizes.begin() + copied, sizes.end(), 0);
    result.filter = filter;
    return result;
}
CudaDnnResult CudaDnnRegistry::filter_size(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kFilter)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.filter_size) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    std::size_t bytes{};
    auto result = completed(
        state.dispatch.filter_size(static_cast<CudaDnnFilterHandle>((*found)->native), &bytes));
    if (result.original_status == CUDNN_STATUS_SUCCESS) result.value = bytes;
    return result;
}
CudaDnnResult CudaDnnRegistry::set_convolution(std::uint64_t id,
                                               const CudaDnnConvolutionState& convolution,
                                               bool is_two) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kConvolution)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (convolution.mode > 1 || convolution.type > dnn_profile::data_type_enum_max() ||
        convolution.dimensions < 0 || convolution.dimensions > CUDNN_DIM_MAX ||
        (is_two && convolution.dimensions != 2))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((is_two && !state.dispatch.convolution_set_2d) ||
        (!is_two && !state.dispatch.convolution_set_nd))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const auto native = static_cast<CudaDnnConvolution>((*found)->native);
    const auto mode = static_cast<cudnnConvolutionMode_t>(convolution.mode);
    const auto type = static_cast<cudnnDataType_t>(convolution.type);
    const auto& padding = convolution.padding;
    const auto& strides = convolution.strides;
    const auto& dilation = convolution.dilation;
    const auto status =
        is_two ? state.dispatch.convolution_set_2d(native, padding[0], padding[1], strides[0],
                                                   strides[1], dilation[0], dilation[1], mode, type)
               : state.dispatch.convolution_set_nd(native, convolution.dimensions, padding.data(),
                                                   strides.data(), dilation.data(), mode, type);
    return state.mutation(**found, status);
}
CudaDnnResult CudaDnnRegistry::get_convolution(std::uint64_t id, int requested,
                                               bool is_two) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kConvolution)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (is_two && requested != 2) return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((is_two && !state.dispatch.convolution_get_2d) ||
        (!is_two && !state.dispatch.convolution_get_nd))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    CudaDnnConvolutionState convolution;
    cudnnConvolutionMode_t mode{};
    cudnnDataType_t type{};
    const auto native = static_cast<CudaDnnConvolution>((*found)->native);
    auto& padding = convolution.padding;
    auto& strides = convolution.strides;
    auto& dilation = convolution.dilation;
    cudnnStatus_t status{};
    if (is_two) {
        convolution.dimensions = 2;
        status = state.dispatch.convolution_get_2d(native, &padding[0], &padding[1], &strides[0],
                                                   &strides[1], &dilation[0], &dilation[1], &mode,
                                                   &type);
    } else {
        // Unlike tensor queries, a request above six is a native NOT_SUPPORTED,
        // not a request to silently clip. Fixed storage already holds all ranks.
        status = state.dispatch.convolution_get_nd(native, requested, &convolution.dimensions,
                                                   padding.data(), strides.data(), dilation.data(),
                                                   &mode, &type);
    }
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) return result;
    convolution.mode = raw_bits(mode);
    convolution.type = raw_bits(type);
    if (requested < 0 || requested > CUDNN_DIM_MAX - 2 || convolution.dimensions < 0 ||
        convolution.dimensions > CUDNN_DIM_MAX - 2 || convolution.mode > 1 ||
        convolution.type > dnn_profile::data_type_enum_max())
        return state.malformed(**found, result);
    // The native getter copies the requested extent, including retained higher-rank entries.
    const auto copied = static_cast<std::size_t>(requested);
    std::fill(padding.begin() + copied, padding.end(), 0);
    std::fill(strides.begin() + copied, strides.end(), 0);
    std::fill(dilation.begin() + copied, dilation.end(), 0);
    result.convolution = convolution;
    return result;
}
CudaDnnResult CudaDnnRegistry::set_convolution_attribute(std::uint64_t id,
                                                         CudaDnnConvolutionAttribute attribute,
                                                         std::uint32_t value) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kConvolution)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if ((attribute != CudaDnnConvolutionAttribute::kGroup &&
         attribute != CudaDnnConvolutionAttribute::kMath &&
         attribute != CudaDnnConvolutionAttribute::kReorder) ||
        (attribute == CudaDnnConvolutionAttribute::kMath && value > 3) ||
        (attribute == CudaDnnConvolutionAttribute::kReorder && value > 1))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((attribute == CudaDnnConvolutionAttribute::kGroup &&
         !state.dispatch.convolution_group_set) ||
        (attribute == CudaDnnConvolutionAttribute::kMath && !state.dispatch.convolution_math_set) ||
        (attribute == CudaDnnConvolutionAttribute::kReorder &&
         !state.dispatch.convolution_reorder_set))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const auto native = static_cast<CudaDnnConvolution>((*found)->native);
    cudnnStatus_t status{};
    switch (attribute) {
        case CudaDnnConvolutionAttribute::kGroup:
            status =
                state.dispatch.convolution_group_set(native, std::bit_cast<std::int32_t>(value));
            break;
        case CudaDnnConvolutionAttribute::kMath:
            status =
                state.dispatch.convolution_math_set(native, static_cast<cudnnMathType_t>(value));
            break;
        case CudaDnnConvolutionAttribute::kReorder:
            status = state.dispatch.convolution_reorder_set(
                native, static_cast<decltype(CUDNN_DEFAULT_REORDER)>(value));
            break;
    }
    return state.mutation(**found, status);
}
CudaDnnResult CudaDnnRegistry::get_convolution_attribute(
    std::uint64_t id, CudaDnnConvolutionAttribute attribute) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kConvolution)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    const auto native = static_cast<CudaDnnConvolution>((*found)->native);
    std::uint32_t bits{};
    cudnnStatus_t status{};
    switch (attribute) {
        case CudaDnnConvolutionAttribute::kGroup: {
            if (!state.dispatch.convolution_group_get) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            int value{};
            status = state.dispatch.convolution_group_get(native, &value);
            bits = raw_bits(value);
            break;
        }
        case CudaDnnConvolutionAttribute::kMath: {
            if (!state.dispatch.convolution_math_get) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            cudnnMathType_t value{};
            status = state.dispatch.convolution_math_get(native, &value);
            bits = raw_bits(value);
            break;
        }
        case CudaDnnConvolutionAttribute::kReorder: {
            if (!state.dispatch.convolution_reorder_get) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            decltype(CUDNN_DEFAULT_REORDER) value{};
            status = state.dispatch.convolution_reorder_get(native, &value);
            bits = raw_bits(value);
            break;
        }
        default:
            return rejected(CUDA_ERROR_INVALID_VALUE);
    }
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) return result;
    if ((attribute == CudaDnnConvolutionAttribute::kMath && bits > 3) ||
        (attribute == CudaDnnConvolutionAttribute::kReorder && bits > 1))
        return state.malformed(**found, result);
    result.value = bits;
    return result;
}
CudaDnnResult CudaDnnRegistry::convolution_output(std::uint64_t convolution, std::uint64_t tensor,
                                                  std::uint64_t filter, int dimensions,
                                                  bool is_two) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto operation = state.find(convolution), input = state.find(tensor),
               weights = state.find(filter);
    if (operation == state.objects.end() || input == state.objects.end() ||
        weights == state.objects.end() || (*operation)->kind != CudaDnnObjectKind::kConvolution ||
        (*input)->kind != CudaDnnObjectKind::kTensor ||
        (*weights)->kind != CudaDnnObjectKind::kFilter)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (dimensions < 0 || dimensions > CUDNN_DIM_MAX || (is_two && dimensions != 4))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if ((is_two && !state.dispatch.convolution_output_2d) ||
        (!is_two && !state.dispatch.convolution_output_nd))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto native = static_cast<CudaDnnConvolution>((*operation)->native);
    const auto x = static_cast<cudnnTensorDescriptor_t>((*input)->native);
    const auto w = static_cast<CudaDnnFilterHandle>((*weights)->native);
    CudaDnnTensor output;
    output.dimensions = dimensions;
    auto& sizes = output.sizes;
    const auto status =
        is_two ? state.dispatch.convolution_output_2d(native, x, w, &sizes[0], &sizes[1], &sizes[2],
                                                      &sizes[3])
               : state.dispatch.convolution_output_nd(native, x, w, dimensions, sizes.data());
    auto result = completed(status);
    if (status == CUDNN_STATUS_SUCCESS) result.tensor = output;
    return result;
}
CudaDnnResult CudaDnnRegistry::set_pooling(std::uint64_t id,
                                           const CudaDnnPoolingState& pooling) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kPooling)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (pooling.mode > 3 || pooling.nan > 1 || pooling.dimensions < 0 ||
        pooling.dimensions > CUDNN_DIM_MAX)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.dispatch.pooling_set_nd) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    // Native cuDNN, not a convolution-derived policy, decides shape legality.
    // In particular negative padding is valid, and ranks seven/eight must
    // reach the original setter's NOT_SUPPORTED result.
    return state.mutation(
        **found, state.dispatch.pooling_set_nd(
                     static_cast<CudaDnnPooling>((*found)->native),
                     static_cast<decltype(CUDNN_POOLING_MAX)>(pooling.mode),
                     static_cast<cudnnNanPropagation_t>(pooling.nan), pooling.dimensions,
                     pooling.window.data(), pooling.padding.data(), pooling.strides.data()));
}
CudaDnnResult CudaDnnRegistry::get_pooling(std::uint64_t id, int requested) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kPooling)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.pooling_get_nd) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    CudaDnnPoolingState pooling;
    decltype(CUDNN_POOLING_MAX) mode{};
    cudnnNanPropagation_t nan{};
    // Unlike convolution getters, pooling permits capacities seven/eight.
    // Preserve larger/negative requests for their original native errors.
    const auto status = state.dispatch.pooling_get_nd(
        static_cast<CudaDnnPooling>((*found)->native), requested, &mode, &nan, &pooling.dimensions,
        pooling.window.data(), pooling.padding.data(), pooling.strides.data());
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) return result;
    pooling.mode = raw_bits(mode);
    pooling.nan = raw_bits(nan);
    // Pooling queries require a previously initialized descriptor; unlike a
    // zero requested prefix, a zero actual spatial rank is not initialized.
    if (requested < 0 || requested > CUDNN_DIM_MAX || pooling.dimensions <= 0 ||
        pooling.dimensions > CUDNN_DIM_MAX - 2 || pooling.mode > 3 || pooling.nan > 1)
        return state.malformed(**found, result);
    const auto copied = static_cast<std::size_t>(std::min(requested, pooling.dimensions));
    std::fill(pooling.window.begin() + copied, pooling.window.end(), 0);
    std::fill(pooling.padding.begin() + copied, pooling.padding.end(), 0);
    std::fill(pooling.strides.begin() + copied, pooling.strides.end(), 0);
    result.pooling = pooling;
    return result;
}
CudaDnnResult CudaDnnRegistry::pooling_output(std::uint64_t pooling, std::uint64_t tensor,
                                              int dimensions) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto operation = state.find(pooling), input = state.find(tensor);
    if (operation == state.objects.end() || input == state.objects.end() ||
        (*operation)->kind != CudaDnnObjectKind::kPooling ||
        (*input)->kind != CudaDnnObjectKind::kTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.pooling_output_nd) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    CudaDnnTensor output;
    output.dimensions = dimensions;
    const auto status = state.dispatch.pooling_output_nd(
        static_cast<CudaDnnPooling>((*operation)->native),
        static_cast<cudnnTensorDescriptor_t>((*input)->native), dimensions, output.sizes.data());
    auto result = completed(status);
    if (status == CUDNN_STATUS_SUCCESS) {
        // The native API rejects a request inconsistent with the descriptor.
        // Forward the raw request, but never publish an unbounded success.
        if (dimensions < 0 || dimensions > CUDNN_DIM_MAX)
            return state.malformed(**operation, result);
        result.tensor = output;
    }
    return result;
}
CudaDnnResult CudaDnnRegistry::set_lrn(std::uint64_t id, CudaDnnLrnState lrn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kLrn)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.lrn_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    // Window and floating-point validity are native SDK decisions. Do not
    // narrow the unsigned window or normalize any floating-point bit pattern.
    return state.mutation(**found, state.dispatch.lrn_set(static_cast<CudaDnnLrn>((*found)->native),
                                                          lrn.n, lrn.alpha, lrn.beta, lrn.k));
}
CudaDnnResult CudaDnnRegistry::set_reduction(std::uint64_t id,
                                             const CudaDnnReductionState& reduction) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kReduction)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.reduction_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    // These bounds describe C++ enum representation, not supported SDK
    // combinations. In particular, unnamed operations 9..15 and indices
    // formats other than unsigned 32-bit remain native validation decisions.
    if (reduction.operation > 15 || reduction.type > dnn_profile::data_type_enum_max() ||
        reduction.nan > 1 || reduction.indices > 1 || reduction.indices_type > 3)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    return state.mutation(
        **found, state.dispatch.reduction_set(
                     static_cast<CudaDnnReduction>((*found)->native),
                     static_cast<cudnnReduceTensorOp_t>(reduction.operation),
                     static_cast<cudnnDataType_t>(reduction.type),
                     static_cast<cudnnNanPropagation_t>(reduction.nan),
                     static_cast<decltype(CUDNN_REDUCE_TENSOR_NO_INDICES)>(reduction.indices),
                     static_cast<decltype(CUDNN_32BIT_INDICES)>(reduction.indices_type)));
}
CudaDnnResult CudaDnnRegistry::set_op_tensor(std::uint64_t id,
                                             CudaDnnOpTensorState operation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kOpTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.op_tensor_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    // Representation bounds are distinct from the SDK's supported operations
    // and computation types. Unnamed representable enum values reach cuDNN.
    if (operation.operation > 7 || operation.type > dnn_profile::data_type_enum_max() ||
        operation.nan > 1)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    return state.mutation(
        **found, state.dispatch.op_tensor_set(static_cast<CudaDnnOpTensor>((*found)->native),
                                              static_cast<cudnnOpTensorOp_t>(operation.operation),
                                              static_cast<cudnnDataType_t>(operation.type),
                                              static_cast<cudnnNanPropagation_t>(operation.nan)));
}
CudaDnnResult CudaDnnRegistry::get_op_tensor(std::uint64_t id, std::uint32_t output_mask) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kOpTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.op_tensor_get) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (output_mask > 7) return rejected(CUDA_ERROR_INVALID_VALUE);
    cudnnOpTensorOp_t operation{};
    cudnnDataType_t type{};
    cudnnNanPropagation_t nan{};
    auto result = completed(state.dispatch.op_tensor_get(
        static_cast<CudaDnnOpTensor>((*found)->native), (output_mask & 1U) ? &operation : nullptr,
        (output_mask & 2U) ? &type : nullptr, (output_mask & 4U) ? &nan : nullptr));
    if (result.original_status != CUDNN_STATUS_SUCCESS) return result;
    CudaDnnOpTensorState output;
    if (output_mask & 1U) output.operation = raw_bits(operation);
    if (output_mask & 2U) output.type = raw_bits(type);
    if (output_mask & 4U) output.nan = raw_bits(nan);
    // Never load malformed native enum representations or expose partial
    // results. Unrequested fields remain zero and confer no query authority.
    if (output.operation > 7 || output.type > dnn_profile::data_type_enum_max() || output.nan > 1)
        return state.malformed(**found, result);
    result.op_tensor = output;
    return result;
}
CudaDnnResult CudaDnnRegistry::set_tensor_transform(
    std::uint64_t id, const CudaDnnTensorTransformState& transform) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kTensorTransform)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.tensor_transform_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (transform.dimensions > CUDNN_DIM_MAX || transform.format > 3 || transform.direction > 1 ||
        transform.presence > 7)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    // Null arrays are not equivalent to supplied zero arrays. Preserve raw
    // padding/fold bits and ranks 0..2 for the original SDK's validation.
    return state.mutation(
        **found, state.dispatch.tensor_transform_set(
                     static_cast<CudaDnnTensorTransform>((*found)->native), transform.dimensions,
                     static_cast<cudnnTensorFormat_t>(transform.format),
                     (transform.presence & 1U) ? transform.before.data() : nullptr,
                     (transform.presence & 2U) ? transform.after.data() : nullptr,
                     (transform.presence & 4U) ? transform.fold.data() : nullptr,
                     static_cast<cudnnFoldingDirection_t>(transform.direction)));
}
CudaDnnResult CudaDnnRegistry::init_transform_dest(std::uint64_t transform, std::uint64_t source,
                                                   std::uint64_t destination,
                                                   bool has_size) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto operation = state.find(transform), input = state.find(source),
               output = state.find(destination);
    if (operation == state.objects.end() || input == state.objects.end() ||
        output == state.objects.end() ||
        (*operation)->kind != CudaDnnObjectKind::kTensorTransform ||
        (*input)->kind != CudaDnnObjectKind::kTensor ||
        (*output)->kind != CudaDnnObjectKind::kTensor)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.init_transform_dest) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    // Do not acquire our own source lease before this check: a legitimate
    // source/destination alias must not manufacture an immutable-borrow conflict.
    if (output->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    std::size_t bytes{};
    // This ABI mutates existing descriptor storage, not a device allocation.
    // Production callback attachment requires independent SDK work-classification
    // evidence; the handle-free signature alone does not prove CPU-only work.
    const auto status = state.dispatch.init_transform_dest(
        static_cast<CudaDnnTensorTransform>((*operation)->native),
        static_cast<cudnnTensorDescriptor_t>((*input)->native),
        static_cast<cudnnTensorDescriptor_t>((*output)->native), has_size ? &bytes : nullptr);
    auto result = completed(status);
    if (status != CUDNN_STATUS_SUCCESS) {
        // Cross-descriptor initialization has no established error atomicity.
        // Preserve original failure but never assume the destination rolled back.
        (*output)->is_uncertain = state.is_faulted = true;
    } else if (has_size) {
        result.value = bytes;
    }
    return result;
}
CudaDnnResult CudaDnnRegistry::set_rnn_data(std::uint64_t id,
                                            const CudaDnnRnnDataState& data) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kRnnData)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.rnn_data_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (data.type > dnn_profile::data_type_enum_max() || data.layout > 3 || data.presence > 3 ||
        data.lengths.size() > kCudaDnnMaxSequenceLengths)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const bool has_lengths = data.presence & 1U, has_padding = data.presence & 2U;
    // Original 9.10.2 checks null lengths before type/layout/sizes, and reads
    // nonnull lengths only after all these scalar checks have succeeded.
    const bool reads_lengths = has_lengths && data.type <= 2 && data.layout <= 2 &&
                               data.max_sequence_length > 0 && data.max_sequence_length <= 65535 &&
                               data.batch_size > 0 && data.vector_size > 0;
    const auto length_count = reads_lengths ? static_cast<std::size_t>(data.batch_size) : 0;
    const auto padding_bytes =
        reads_lengths && has_padding ? sequence_padding_bytes(data.type, false) : 0;
    if (length_count > kCudaDnnMaxSequenceLengths || data.lengths.size() != length_count ||
        data.padding_bytes != padding_bytes)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const SequenceCall active{state.is_sequence_active};
    const int dummy{};
    const int* lengths =
        has_lengths ? (data.lengths.empty() ? &dummy : data.lengths.data()) : nullptr;
    alignas(double) auto padding = data.padding;
    const auto status = state.dispatch.rnn_data_set(
        static_cast<CudaDnnRnnData>((*found)->native), static_cast<cudnnDataType_t>(data.type),
        static_cast<cudnnRNNDataLayout_t>(data.layout), data.max_sequence_length, data.batch_size,
        data.vector_size, lengths, has_padding ? padding.data() : nullptr);
    // Array replacement can delete old storage before a later native error.
    // Neither a status category nor a hidden query proves rollback.
    if (status != CUDNN_STATUS_SUCCESS) (*found)->is_uncertain = state.is_faulted = true;
    return completed(status);
}
CudaDnnResult CudaDnnRegistry::set_seq_data(std::uint64_t id,
                                            const CudaDnnSeqDataState& data) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kSeqData)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.seq_data_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (data.type > dnn_profile::data_type_enum_max() || data.presence > 15 ||
        data.lengths.size() > kCudaDnnMaxSequenceLengths ||
        data.sequence_count > std::numeric_limits<std::size_t>::max())
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const bool has_sizes = data.presence & 1U, has_axes = data.presence & 2U,
               has_lengths = data.presence & 4U, has_padding = data.presence & 8U;
    if (has_axes && std::ranges::any_of(data.axes, [](auto axis) { return axis > 3; }))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const bool reads_sizes = data.type <= 16 && data.dimensions == 4;
    if (reads_sizes && !has_sizes) return rejected(CUDA_ERROR_INVALID_VALUE);
    bool reads_lengths{};
    if (reads_sizes && std::ranges::all_of(data.sizes, [](int value) { return value > 0; })) {
        // Native dimensions are indexed by semantic axis; axes[] describes
        // layout. Prevent its signed32 batch*beam multiplication from wrapping.
        const auto count =
            static_cast<std::uint64_t>(data.sizes[1]) * static_cast<std::uint64_t>(data.sizes[2]);
        if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            return rejected(CUDA_ERROR_INVALID_VALUE);
        reads_lengths = count == data.sequence_count;
        if (reads_lengths && (count > kCudaDnnMaxSequenceLengths || !has_lengths || !has_axes))
            return rejected(CUDA_ERROR_INVALID_VALUE);
    }
    const auto length_count = reads_lengths ? static_cast<std::size_t>(data.sequence_count) : 0;
    const auto padding_bytes =
        reads_lengths && has_padding ? sequence_padding_bytes(data.type, true) : 0;
    if (data.lengths.size() != length_count || data.padding_bytes != padding_bytes)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const SequenceCall active{state.is_sequence_active};
    std::array<cudnnSeqDataAxis_t, 4> axes{};
    if (has_axes)
        std::transform(data.axes.begin(), data.axes.end(), axes.begin(),
                       [](auto axis) { return static_cast<cudnnSeqDataAxis_t>(axis); });
    const int dummy{};
    const int* lengths =
        has_lengths ? (data.lengths.empty() ? &dummy : data.lengths.data()) : nullptr;
    alignas(double) auto padding = data.padding;
    const auto status = state.dispatch.seq_data_set(
        static_cast<CudaDnnSeqData>((*found)->native), static_cast<cudnnDataType_t>(data.type),
        data.dimensions, has_sizes ? data.sizes.data() : nullptr, has_axes ? axes.data() : nullptr,
        static_cast<std::size_t>(data.sequence_count), lengths,
        has_padding ? padding.data() : nullptr);
    if (status != CUDNN_STATUS_SUCCESS) (*found)->is_uncertain = state.is_faulted = true;
    return completed(status);
}
CudaDnnResult CudaDnnRegistry::get_seq_data(std::uint64_t id, CudaDnnSeqDataQuery query) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kSeqData)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.seq_data_get) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (query.outputs > 127 || query.lengths_requested > std::numeric_limits<std::size_t>::max())
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const SequenceCall active{state.is_sequence_active};
    bool is_issued{};
    try {
        CudaDnnSeqDataState data;
        data.presence = ((query.outputs & 4U) ? 1U : 0U) | ((query.outputs & 8U) ? 2U : 0U) |
                        ((query.outputs & 32U) ? 4U : 0U) | ((query.outputs & 64U) ? 8U : 0U);
        const bool has_sizes = data.presence & 1U, has_axes = data.presence & 2U,
                   has_lengths = data.presence & 4U, has_padding = data.presence & 8U;
        // A healthy owned setter never admits a count above this bound. Keep
        // the raw capacity (including SIZE_MAX), not a guessed clamped request.
        if (has_lengths && query.dimensions_requested >= 0)
            data.lengths.resize(static_cast<std::size_t>(
                std::min<std::uint64_t>(query.lengths_requested, kCudaDnnMaxSequenceLengths)));
        cudnnDataType_t type{};
        std::array<cudnnSeqDataAxis_t, 4> axes{};
        std::size_t count{};
        int dummy{};
        int* lengths =
            has_lengths ? (data.lengths.empty() ? &dummy : data.lengths.data()) : nullptr;
        // Same-call private scalars establish type/rank/count without a cached
        // setter or an extra SDK query. Other output nullness remains exact.
        is_issued = true;
        const auto status = state.dispatch.seq_data_get(
            static_cast<CudaDnnSeqData>((*found)->native), &type, &data.dimensions,
            query.dimensions_requested, has_sizes ? data.sizes.data() : nullptr,
            has_axes ? axes.data() : nullptr, &count,
            static_cast<std::size_t>(query.lengths_requested), lengths,
            has_padding ? data.padding.data() : nullptr);
        auto result = completed(status);
        if (status != CUDNN_STATUS_SUCCESS) return result;
        data.type = raw_bits(type);
        data.sequence_count = count;
        if (query.dimensions_requested < 0 || data.dimensions != 4 ||
            data.type > dnn_profile::data_type_enum_max() || count > kCudaDnnMaxSequenceLengths)
            return state.malformed(**found, std::move(result));
        const auto dimensions = static_cast<std::size_t>(std::min(query.dimensions_requested, 4));
        if (has_axes)
            for (std::size_t index = 0; index < dimensions; ++index) {
                data.axes[index] = raw_bits(axes[index]);
                if (data.axes[index] > 3) return state.malformed(**found, std::move(result));
            }
        std::fill(data.sizes.begin() + dimensions, data.sizes.end(), 0);
        const auto copied =
            has_lengths
                ? static_cast<std::size_t>(std::min<std::uint64_t>(query.lengths_requested, count))
                : 0;
        data.lengths.resize(copied);
        if (has_padding) {
            // Configured BF16 Set succeeds, but its Get with padding must
            // return INTERNAL_ERROR (4000). Never fabricate successful BF16 output.
            data.padding_bytes = sequence_padding_bytes(data.type, false);
            if (!data.padding_bytes) return state.malformed(**found, std::move(result));
        }
        std::fill(data.padding.begin() + data.padding_bytes, data.padding.end(), std::byte{});
        result.seq_data = std::move(data);
        return result;
    } catch (...) {
        if (is_issued) {
            (*found)->is_uncertain = state.is_faulted = true;
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnResult CudaDnnRegistry::set_rnn(std::uint64_t id, const CudaDnnRnnState& rnn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kRnn)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.rnn_set_v8) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (rnn.algorithm > 7 || rnn.cell_mode > 3 || rnn.bias_mode > 3 || rnn.direction > 1 ||
        rnn.input_mode > 1 || rnn.type > dnn_profile::data_type_enum_max() ||
        rnn.math_precision > dnn_profile::data_type_enum_max() || rnn.math_type > 3)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    Implementation::DependencyUpdate dependencies;
    const auto prepared = state.prepare_dependencies(**found, {rnn.dropout, 0}, dependencies);
    if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_NOT_READY) return rejected(prepared);
    if (prepared == CUDA_ERROR_NOT_READY || found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const DependencyCall active{state.is_dependency_active};
    // Configuration does not populate dynamic libraries. Build and physical
    // computation borrows govern their independent, context-spanning lifetime.
    const auto status = state.dispatch.rnn_set_v8(
        static_cast<CudaDnnRnn>((*found)->native), static_cast<cudnnRNNAlgo_t>(rnn.algorithm),
        static_cast<cudnnRNNMode_t>(rnn.cell_mode), static_cast<cudnnRNNBiasMode_t>(rnn.bias_mode),
        static_cast<cudnnDirectionMode_t>(rnn.direction),
        static_cast<cudnnRNNInputMode_t>(rnn.input_mode), static_cast<cudnnDataType_t>(rnn.type),
        static_cast<cudnnDataType_t>(rnn.math_precision),
        static_cast<cudnnMathType_t>(rnn.math_type), rnn.input_size, rnn.hidden_size,
        rnn.projection_size, rnn.layers, dependencies.native[0], rnn.flags);
    return state.commit_dependencies(**found, dependencies, status);
}
CudaDnnResult CudaDnnRegistry::set_attn(std::uint64_t id, const CudaDnnAttnState& attn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kAttn)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.attn_set) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (attn.type > dnn_profile::data_type_enum_max() ||
        attn.precision > dnn_profile::data_type_enum_max() || attn.math_type > 3)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    Implementation::DependencyUpdate dependencies;
    const auto prepared = state.prepare_dependencies(**found, attn.dropouts, dependencies);
    if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_NOT_READY) return rejected(prepared);
    if (prepared == CUDA_ERROR_NOT_READY || found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    const DependencyCall active{state.is_dependency_active};
    const auto status = state.dispatch.attn_set(
        static_cast<CudaDnnAttn>((*found)->native), attn.mode, attn.heads, attn.scale,
        static_cast<cudnnDataType_t>(attn.type), static_cast<cudnnDataType_t>(attn.precision),
        static_cast<cudnnMathType_t>(attn.math_type), dependencies.native[0],
        dependencies.native[1], attn.query_size, attn.key_size, attn.value_size,
        attn.query_projection, attn.key_projection, attn.value_projection, attn.output_projection,
        attn.max_query_sequence, attn.max_key_sequence, attn.max_batch, attn.max_beam);
    return state.commit_dependencies(**found, dependencies, status);
}
CudaDnnResult CudaDnnRegistry::set_rnn_clip(std::uint64_t id, CudaDnnRnnClipState clip) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kRnn ||
        !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.rnn_set_clip_v8) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (clip.mode > 1 || clip.nan > 1) return rejected(CUDA_ERROR_INVALID_VALUE);
    // Test exclusive parent ownership before acquiring this call's own lease.
    if (found->use_count() != 1)
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return {.disposition = status == CUDA_ERROR_NOT_READY ? CudaDnnDisposition::kDeferred
                                                              : CudaDnnDisposition::kRejected,
                .status = status};
    const auto parent = *found;
    const DependencyCall active{state.rnn_calls->is_active};
    // Native logging reads current dropout fields. Temporary immutable leases
    // protect those fields without converting liveness edges to durable borrows.
    // Keep all double bits for native validation, including NaN and infinities.
    const auto status = state.dispatch.rnn_set_clip_v8(
        static_cast<CudaDnnRnn>(parent->native), static_cast<cudnnRNNClipMode_t>(clip.mode),
        static_cast<cudnnNanPropagation_t>(clip.nan), clip.lower, clip.upper);
    if (status != CUDNN_STATUS_SUCCESS) parent->is_uncertain = state.is_faulted = true;
    return completed(status);
}
CudaDnnRegistry::RnnQueryResult CudaDnnRegistry::query_rnn_sizes(
    cudnnHandle_t handle, const CudaDnnRnnQuery& query,
    CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        RnnQueryResult output;
        output.result.result = rejected(status);
        if (status == CUDA_ERROR_NOT_READY)
            output.result.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return reject(status);
    if (!handle || !check_context || !session_state) return reject(CUDA_ERROR_INVALID_VALUE);
    const bool is_temp = query.kind == CudaDnnRnnQueryKind::kTempSpace;
    const bool is_attention = query.kind == CudaDnnRnnQueryKind::kAttentionBuffers;
    if ((!is_temp && !is_attention && query.kind != CudaDnnRnnQueryKind::kWeightSpace) ||
        (!is_temp && (query.input || query.mode)) || (is_temp && query.mode > 1) ||
        (query.outputs & ~(is_attention ? 7U
                           : is_temp    ? 6U
                                        : 1U)))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto found = state.find(query.descriptor);
    const auto kind = is_attention ? CudaDnnObjectKind::kAttn : CudaDnnObjectKind::kRnn;
    if (found == state.objects.end() || (*found)->kind != kind || !(*found)->native ||
        (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    std::shared_ptr<Implementation::Object> input;
    if (is_temp) {
        const auto data = state.find(query.input);
        if (data == state.objects.end() || (*data)->kind != CudaDnnObjectKind::kRnnData ||
            !(*data)->native || (*data)->is_uncertain || (*data)->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        input = *data;
    }
    const bool needs_preflight = is_temp && (query.outputs & 2U);
    const auto& sdk = state.dispatch;
    if ((is_attention && !sdk.attn_buffers) || (is_temp && !sdk.rnn_temp_space_sizes) ||
        (!is_attention && !is_temp && !sdk.rnn_weight_space_size) ||
        (needs_preflight && !sdk.rnn_data_get))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return reject(status);
    const auto parent = *found;
    const DependencyCall active{state.rnn_calls->is_active};
    RnnQueryResult output;
    if (needs_preflight) {
        // Configured 9.10.2 workspace sizing can divide by a zero fresh-state
        // sequence length. The original getter requires all scalar outputs and
        // padding even at zero array capacity; it never reads lengths here.
        cudnnDataType_t type{};
        cudnnRNNDataLayout_t layout{};
        int sequence_length{}, batch{}, vector{};
        alignas(double) std::array<std::byte, 8> padding{};
        output.preflight_status =
            sdk.rnn_data_get(static_cast<CudaDnnRnnData>(input->native), &type, &layout,
                             &sequence_length, &batch, &vector, 0, nullptr, padding.data());
        if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS) {
            output.result.result = rejected(proof);
            output.has_context_loss = true;
            return output;
        }
        // Never attribute the private getter's error to the requested query.
        if (output.preflight_status != CUDNN_STATUS_SUCCESS) return output;
        if (raw_bits(type) > dnn_profile::data_type_enum_max() || raw_bits(layout) > 3) {
            input->is_uncertain = state.is_faulted = true;
            return output;
        }
        if (sequence_length <= 0) {
            output.result.result = rejected(CUDA_ERROR_INVALID_VALUE);
            return output;
        }
    }
    std::array<std::size_t, 3> sizes{};
    const auto slot = [&](std::size_t index) {
        return (query.outputs & (1U << index)) ? &sizes[index] : nullptr;
    };
    cudnnStatus_t status{};
    if (is_attention) {
        status = sdk.attn_buffers(handle, static_cast<CudaDnnAttn>(parent->native), slot(0),
                                  slot(1), slot(2));
    } else if (is_temp) {
        status =
            sdk.rnn_temp_space_sizes(handle, static_cast<CudaDnnRnn>(parent->native),
                                     static_cast<cudnnForwardMode_t>(query.mode),
                                     static_cast<CudaDnnRnnData>(input->native), slot(1), slot(2));
    } else {
        status =
            sdk.rnn_weight_space_size(handle, static_cast<CudaDnnRnn>(parent->native), slot(0));
    }
    output.result.result = completed(status);
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS) {
        output.result.result.disposition = CudaDnnDisposition::kRejected;
        output.result.result.status = proof;
        output.has_context_loss = true;
        return output;
    }
    // Transactional publication: native error and context loss expose no sizes.
    if (status == CUDNN_STATUS_SUCCESS) {
        for (std::size_t index = 0; index < sizes.size(); ++index)
            if (query.outputs & (1U << index)) output.result.sizes[index] = sizes[index];
    }
    return output;
}
CudaDnnRegistry::FoldedResult CudaDnnRegistry::folded_backward_data(
    cudnnHandle_t handle, const CudaDnnFoldedDescriptors& descriptors,
    CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS)
        return {.result = rejected(status)};
    if (!handle || !check_context || !session_state || descriptors.format > 3)
        return {.result = rejected(CUDA_ERROR_INVALID_VALUE)};
    if (!state.dispatch.folded_backward_data || !state.dispatch.convolution_get_nd)
        return {.result = rejected(CUDA_ERROR_NOT_SUPPORTED)};

    constexpr std::array kinds{
        CudaDnnObjectKind::kFilter,          CudaDnnObjectKind::kTensor,
        CudaDnnObjectKind::kConvolution,     CudaDnnObjectKind::kTensor,
        CudaDnnObjectKind::kFilter,          CudaDnnObjectKind::kTensor,
        CudaDnnObjectKind::kConvolution,     CudaDnnObjectKind::kTensor,
        CudaDnnObjectKind::kTensorTransform, CudaDnnObjectKind::kTensorTransform,
        CudaDnnObjectKind::kTensorTransform, CudaDnnObjectKind::kTensorTransform};
    std::array<std::shared_ptr<Implementation::Object>*, 12> objects{};
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const auto id = index < descriptors.inputs.size()
                            ? descriptors.inputs[index]
                            : descriptors.outputs[index - descriptors.inputs.size()];
        if (index == objects.size() - 1 && !id) continue;
        const auto found = state.find(id);
        if (found == state.objects.end() || (*found)->kind != kinds[index] || !(*found)->native ||
            (*found)->is_uncertain || (*found)->has_destroy_attempt)
            return {.result = rejected(CUDA_ERROR_INVALID_HANDLE)};
        objects[index] = &*found;
    }

    // Check each unique destination before copying any owner. An input/output
    // or repeated-output alias must not create its own immutable-borrow conflict.
    std::array<Implementation::Object*, 8> destinations{};
    std::size_t destination_count{};
    for (std::size_t index = descriptors.inputs.size(); index < objects.size(); ++index) {
        if (!objects[index]) continue;
        auto* object = objects[index]->get();
        const auto end = destinations.begin() + destination_count;
        if (std::find(destinations.begin(), end, object) != end) continue;
        if (objects[index]->use_count() != 1)
            return {
                .result = {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS}};
        destinations[destination_count++] = object;
    }
    std::array<std::shared_ptr<Implementation::Object>, 12> owners{};
    std::size_t owner_count{};
    for (const auto* object : objects) {
        if (!object) continue;
        const auto end = owners.begin() + owner_count;
        if (std::none_of(owners.begin(), end,
                         [&](const auto& owner) { return owner.get() == object->get(); }))
            owners[owner_count++] = *object;
    }
    struct ActiveCall {
        bool& active;
        ~ActiveCall() {
            active = false;
        }
    };
    state.is_folded_active = true;
    const ActiveCall active{state.is_folded_active};
    const auto native = [&](std::size_t index) {
        return objects[index] ? (*objects[index])->native : nullptr;
    };

    // Fresh convolutions have zero strides, and the 2D getter does not prove
    // rank. The original Nd query safely establishes the two divisors used by
    // the configured Folded implementation. This is a crash-containment gate,
    // not an emulation of native Folded validation or descriptor initialization.
    CudaDnnConvolutionState convolution;
    cudnnConvolutionMode_t mode{};
    cudnnDataType_t type{};
    FoldedResult output;
    output.preflight_status = state.dispatch.convolution_get_nd(
        static_cast<CudaDnnConvolution>(native(2)), 2, &convolution.dimensions,
        convolution.padding.data(), convolution.strides.data(), convolution.dilation.data(), &mode,
        &type);
    if (const auto status = check_context(session_state); status != CUDA_SUCCESS) {
        output.result = rejected(status);
        output.has_context_loss = true;
        return output;
    }
    if (output.preflight_status != CUDNN_STATUS_SUCCESS) return output;
    convolution.mode = raw_bits(mode);
    convolution.type = raw_bits(type);
    if (convolution.dimensions < 0 || convolution.dimensions > CUDNN_DIM_MAX - 2 ||
        convolution.mode > 1 || convolution.type > dnn_profile::data_type_enum_max()) {
        (*objects[2])->is_uncertain = state.is_faulted = true;
        return output;  // Getter status is diagnostic, never original Folded status.
    }
    if (convolution.dimensions != 2 || convolution.strides[0] <= 0 || convolution.strides[1] <= 0) {
        output.result = rejected(CUDA_ERROR_INVALID_VALUE);
        return output;
    }

    const auto status = state.dispatch.folded_backward_data(
        handle, static_cast<CudaDnnFilterHandle>(native(0)),
        static_cast<cudnnTensorDescriptor_t>(native(1)), static_cast<CudaDnnConvolution>(native(2)),
        static_cast<cudnnTensorDescriptor_t>(native(3)),
        static_cast<cudnnTensorFormat_t>(descriptors.format),
        static_cast<CudaDnnFilterHandle>(native(4)),
        static_cast<cudnnTensorDescriptor_t>(native(5)), static_cast<CudaDnnConvolution>(native(6)),
        static_cast<cudnnTensorDescriptor_t>(native(7)),
        static_cast<CudaDnnTensorTransform>(native(8)),
        static_cast<CudaDnnTensorTransform>(native(9)),
        static_cast<CudaDnnTensorTransform>(native(10)),
        static_cast<CudaDnnTensorTransform>(native(11)));
    output.result = completed(status);
    const auto proof = check_context(session_state);
    if (status != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS) {
        // Original Folded can partially mutate before any error. Preserve its
        // status but do not retry, roll back, or claim known output contents.
        for (std::size_t index = 0; index < destination_count; ++index)
            destinations[index]->is_uncertain = true;
        state.is_faulted = true;
    }
    if (proof != CUDA_SUCCESS) {
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = proof;
        output.has_context_loss = true;
    }
    return output;
}
CudaDnnRnnWeightParamsResult CudaDnnRegistry::rnn_weight_params(
    cudnnHandle_t handle, const CudaDnnRnnWeightParams& parameters, CUdeviceptr weights,
    std::size_t weight_bytes, CUresult (*check_context)(const void*) noexcept,
    const void* session_state) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnWeightParamsResult output{rejected(status)};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return reject(status);
    if (!handle || !check_context || !session_state || (parameters.outputs & ~3U) ||
        weight_bytes > std::numeric_limits<CUdeviceptr>::max() - weights)
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto found = state.find(parameters.descriptor);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kRnn ||
        !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);

    std::array<std::shared_ptr<Implementation::Object>*, 2> objects{};
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const auto id = parameters.destinations[index];
        if (!id) continue;
        const auto destination = state.find(id);
        if (destination == state.objects.end() ||
            (*destination)->kind != CudaDnnObjectKind::kTensor || !(*destination)->native ||
            (*destination)->is_uncertain || (*destination)->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        // Configured SDK originals write through this output without checking
        // for null. The reverse pair safely reaches the original BAD_PARAM.
        if (!(parameters.outputs & (1U << index))) return reject(CUDA_ERROR_INVALID_VALUE);
        objects[index] = &*destination;
    }
    if (!state.dispatch.rnn_weight_params) return reject(CUDA_ERROR_NOT_SUPPORTED);

    // Check unique mutable outputs before retaining our own references. A
    // shared matrix/bias identity is one destination, not an external borrower.
    std::array<Implementation::Object*, 2> destinations{};
    std::size_t destination_count{};
    for (const auto* object : objects) {
        if (!object) continue;
        const auto end = destinations.begin() + destination_count;
        if (std::find(destinations.begin(), end, object->get()) != end) continue;
        if (object->use_count() != 1) return reject(CUDA_ERROR_NOT_READY);
        destinations[destination_count++] = object->get();
    }
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return reject(status);
    const auto parent = *found;
    std::array<std::shared_ptr<Implementation::Object>, 2> destination_owners{};
    std::size_t owner_count{};
    for (const auto* object : objects) {
        if (!object) continue;
        const auto end = destination_owners.begin() + owner_count;
        if (std::none_of(destination_owners.begin(), end,
                         [&](const auto& owner) { return owner.get() == object->get(); }))
            destination_owners[owner_count++] = *object;
    }
    const DependencyCall active{state.rnn_calls->is_active};
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS) {
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    }
    const auto native = [&](std::size_t index) {
        return objects[index] ? static_cast<cudnnTensorDescriptor_t>((*objects[index])->native)
                              : nullptr;
    };
    std::array<void*, 2> addresses{};
    const auto status = state.dispatch.rnn_weight_params(
        handle, static_cast<CudaDnnRnn>(parent->native), parameters.pseudo_layer, weight_bytes,
        reinterpret_cast<const void*>(weights), parameters.linear_layer, native(0),
        (parameters.outputs & 1U) ? &addresses[0] : nullptr, native(1),
        (parameters.outputs & 2U) ? &addresses[1] : nullptr);
    CudaDnnRnnWeightParamsResult output{completed(status)};
    const auto quarantine = [&] {
        for (std::size_t index = 0; index < destination_count; ++index)
            destinations[index]->is_uncertain = true;
        state.is_faulted = true;
    };
    const auto proof = check_context(session_state);
    if (status != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS) quarantine();
    if (proof != CUDA_SUCCESS) {
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = proof;
        output.has_context_loss = true;
        return output;
    }
    if (status != CUDNN_STATUS_SUCCESS) return output;

    std::uint32_t non_null{};
    std::array<std::uint64_t, 2> offsets{};
    for (std::size_t index = 0; index < addresses.size(); ++index) {
        if (!(parameters.outputs & (1U << index)) || !addresses[index]) continue;
        const auto address = reinterpret_cast<std::uintptr_t>(addresses[index]);
        // Null-base queries intentionally return numerical offsets. A zero
        // may denote the first valid matrix or an absent parameter; preserve
        // that native ambiguity rather than guessing from cached metadata.
        if (address < weights || address - weights >= weight_bytes) {
            quarantine();
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status = CUDA_ERROR_UNKNOWN;
            return output;
        }
        non_null |= 1U << index;
        offsets[index] = address - weights;
    }
    // Mapping and transactional publication remain under the same native-call
    // exclusion and output ownership as the mutation. These offsets prove no
    // tensor extent and grant no persistent allocation ownership.
    output.non_null = non_null;
    output.offsets = offsets;
    return output;
}
CudaDnnAttnWeightsResult CudaDnnRegistry::attn_weights(
    cudnnHandle_t handle, const CudaDnnAttnWeights& parameters,
    CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnAttnWeightsResult output{rejected(status)};
        if (status == CUDA_ERROR_NOT_READY)
            output.result.disposition = CudaDnnDisposition::kDeferred;
        return output;
    };
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return reject(status);
    if (!handle || !check_context || !session_state || parameters.kind > 7 ||
        parameters.has_base > 1 || parameters.has_output > 1 ||
        (parameters.has_base &&
         parameters.weight_bytes > std::numeric_limits<std::uintptr_t>::max() - 1))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto found = state.find(parameters.descriptor);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kAttn ||
        !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    std::shared_ptr<Implementation::Object>* destination{};
    if (parameters.destination) {
        const auto object = state.find(parameters.destination);
        if (object == state.objects.end() || (*object)->kind != CudaDnnObjectKind::kTensor ||
            !(*object)->native || (*object)->is_uncertain || (*object)->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_HANDLE);
        destination = &*object;
    }
    if (!state.dispatch.version || !state.dispatch.attn_weights)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // Check external destination readers before adding any local owner. The
    // parent and its dropout dependencies are immutable for this host query.
    if (destination && destination->use_count() != 1) return reject(CUDA_ERROR_NOT_READY);
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return reject(status);
    const auto parent = *found;
    const auto destination_owner = destination ? *destination : nullptr;
    const DependencyCall active{state.rnn_calls->is_active};
    const auto context_failure = [&](CUresult proof) {
        auto output = reject(proof);
        output.has_context_loss = true;
        return output;
    };
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS)
        return context_failure(proof);
    // Only this audited implementation treats the input as numerical address
    // arithmetic, including logging. Never generalize the sentinel to another
    // SDK version without reviewing its pointer-access behavior.
    const auto version = state.dispatch.version();
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS)
        return context_failure(proof);
    if (version != 91002) return reject(CUDA_ERROR_NOT_SUPPORTED);

    constexpr std::uintptr_t base = 1;
    void* address{};
    const auto status = state.dispatch.attn_weights(
        handle, static_cast<CudaDnnAttn>(parent->native),
        static_cast<cudnnMultiHeadAttnWeightKind_t>(parameters.kind), parameters.weight_bytes,
        parameters.has_base ? reinterpret_cast<const void*>(base) : nullptr,
        destination_owner ? static_cast<cudnnTensorDescriptor_t>(destination_owner->native)
                          : nullptr,
        parameters.has_output ? &address : nullptr);
    CudaDnnAttnWeightsResult output{completed(status)};
    const auto quarantine = [&] {
        if (destination_owner) destination_owner->is_uncertain = true;
        state.is_faulted = true;
    };
    const auto proof = check_context(session_state);
    if (status != CUDNN_STATUS_SUCCESS || proof != CUDA_SUCCESS) quarantine();
    if (proof != CUDA_SUCCESS) {
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = proof;
        output.has_context_loss = true;
        return output;
    }
    if (status != CUDNN_STATUS_SUCCESS) return output;
    // These safe nullable calls must reach the original, but its successful
    // receipt may not contradict the audited mandatory-output/null-base gates.
    if (!destination_owner || !parameters.has_output ||
        (!parameters.has_base && parameters.weight_bytes)) {
        quarantine();
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = CUDA_ERROR_UNKNOWN;
        return output;
    }
    if (!address) return output;

    const auto value = reinterpret_cast<std::uintptr_t>(address);
    // Unlike RNN WeightParams, this original has no null-base offset mode.
    // A successful nonnull result must be strictly within the declared range.
    if (!parameters.has_base || value < base || value - base >= parameters.weight_bytes) {
        quarantine();
        output.result.disposition = CudaDnnDisposition::kRejected;
        output.result.status = CUDA_ERROR_UNKNOWN;
        return output;
    }
    output.non_null = 1;
    output.offset = value - base;
    return output;
}
std::optional<CudaDnnObjectView> CudaDnnRegistry::borrow(std::uint64_t id,
                                                         CudaDnnObjectKind kind) noexcept {
    auto& state = *implementation_;
    if (state.gate(true) != CUDA_SUCCESS) return std::nullopt;
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != kind) return std::nullopt;
    if ((*found)->dropout && (*found)->dropout->is_pending.load(std::memory_order_acquire))
        return std::nullopt;
    return CudaDnnObjectView{kind, (*found)->native, *found};
}
CudaDnnDropoutMutationResult CudaDnnRegistry::begin_dropout_mutation(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS)
        return {.result = rejected(status), .mutation = {}};
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kDropout)
        return {.result = rejected(CUDA_ERROR_INVALID_HANDLE), .mutation = {}};
    if (found->use_count() != 1 || (*found)->dropout->is_pending.load(std::memory_order_acquire))
        return {.result = {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS},
                .mutation = {}};
    try {
        auto mutation = std::shared_ptr<CudaDnnDropoutMutation>(new CudaDnnDropoutMutation(
            (*found)->dropout, *found, static_cast<CudaDnnDropout>((*found)->native)));
        (*found)->dropout->is_pending.store(true, std::memory_order_release);
        return {.result = {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS},
                .mutation = std::move(mutation)};
    } catch (...) {
        return {.result = rejected(CUDA_ERROR_OUT_OF_MEMORY), .mutation = {}};
    }
}
std::optional<CudaDnnDropoutView> CudaDnnRegistry::borrow_dropout(std::uint64_t id) noexcept {
    auto object = borrow(id, CudaDnnObjectKind::kDropout);
    if (!object) return std::nullopt;
    const auto found = implementation_->find(id);
    return CudaDnnDropoutView{std::move(*object), (*found)->dropout->binding};
}
CudaDnnRnnBorrowResult CudaDnnRegistry::borrow_rnn(std::uint64_t id) noexcept {
    return borrow_composite(id, CudaDnnObjectKind::kRnn);
}
CudaDnnRnnBorrowResult CudaDnnRegistry::borrow_attn(std::uint64_t id) noexcept {
    return borrow_composite(id, CudaDnnObjectKind::kAttn);
}
CudaDnnRnnBorrowResult CudaDnnRegistry::borrow_composite(std::uint64_t id,
                                                         CudaDnnObjectKind kind) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnBorrowResult result;
        result.result = rejected(status);
        if (status == CUDA_ERROR_NOT_READY)
            result.result.disposition = CudaDnnDisposition::kDeferred;
        return result;
    };
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return reject(status);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != kind || !(*found)->native ||
        (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return reject(status);
    try {
        CudaDnnRnnBorrowResult result;
        result.view.object = {kind, (*found)->native, *found};
        for (std::size_t index = 0; index < (*found)->algorithm_dependencies.size(); ++index) {
            const auto id = (*found)->algorithm_dependencies[index];
            if (!id) continue;
            const auto algorithm = state.find(id);
            if (algorithm == state.objects.end() ||
                (*algorithm)->kind != CudaDnnObjectKind::kAlgorithm || !(*algorithm)->native ||
                (*algorithm)->is_uncertain || (*algorithm)->has_destroy_attempt)
                return reject(CUDA_ERROR_INVALID_HANDLE);
            result.view.algorithms[index] = {CudaDnnObjectKind::kAlgorithm, (*algorithm)->native,
                                             *algorithm};
        }
        for (std::size_t index = 0; index < dependencies.size(); ++index) {
            const auto& dependency = dependencies[index];
            if (dependency)
                result.view.dropouts[index] = {
                    {CudaDnnObjectKind::kDropout, dependency->native, dependency},
                    dependency->dropout->binding};
        }
        result.view.lease =
            std::shared_ptr<CudaDnnRnnLease>(new CudaDnnRnnLease(state.rnn_calls, *found));
        result.result.disposition = CudaDnnDisposition::kComplete;
        result.result.status = CUDA_SUCCESS;
        return result;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaDnnRnnBuildResult CudaDnnRegistry::build_rnn_dynamic(
    cudnnHandle_t handle, std::uint64_t id, int mini_batch,
    CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult status) {
        CudaDnnRnnBuildResult result{rejected(status)};
        if (status == CUDA_ERROR_NOT_READY)
            result.result.disposition = CudaDnnDisposition::kDeferred;
        return result;
    };
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return reject(status);
    if (!handle || !check_context || !session_state) return reject(CUDA_ERROR_INVALID_VALUE);
    const auto found = state.find(id);
    if (found == state.objects.end() || (*found)->kind != CudaDnnObjectKind::kRnn ||
        !(*found)->native || (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if (!state.dispatch.rnn_build_dynamic) return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (found->use_count() != 1) return reject(CUDA_ERROR_NOT_READY);
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (const auto status = state.pin_dependencies(**found, dependencies); status != CUDA_SUCCESS)
        return reject(status);
    const auto parent = *found;
    const DependencyCall active{state.rnn_calls->is_active};
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS)
        return {rejected(proof), true};
    const auto status = state.dispatch.rnn_build_dynamic(
        handle, static_cast<CudaDnnRnn>(parent->native), mini_batch);
    CudaDnnRnnBuildResult result{completed(status)};
    // Population may have installed some libraries before reporting failure.
    // No rollback or context-affine destruction is inferred from that status.
    if (status != CUDNN_STATUS_SUCCESS) parent->is_uncertain = state.is_faulted = true;
    if (const auto proof = check_context(session_state); proof != CUDA_SUCCESS) {
        parent->is_uncertain = state.is_faulted = true;
        result.result.disposition = CudaDnnDisposition::kRejected;
        result.result.status = proof;
        result.has_context_loss = true;
    }
    return result;
}
void CudaDnnRegistry::close() noexcept {
    if (implementation_->owns()) implementation_->is_closing = true;
}
CudaDnnResult CudaDnnRegistry::shutdown() noexcept {
    auto& state = *implementation_;
    if (state.owns() &&
        (state.is_folded_active || state.is_sequence_active || state.is_dependency_active ||
         state.is_rnn_query_active || state.rnn_calls->is_active))
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_ERROR_NOT_READY};
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    state.is_closing = true;
    if (std::ranges::any_of(state.objects,
                            [](const auto& entry) { return entry.use_count() != 1; }))
        return {.disposition = CudaDnnDisposition::kDeferred, .status = CUDA_SUCCESS};
    while (!state.objects.empty()) {
        // IDs, not shared_ptr borrows, represent durable dependencies. Select
        // parents before their dropouts even when creation order was inverted.
        const auto next =
            std::find_if(state.objects.rbegin(), state.objects.rend(),
                         [&](const auto& entry) { return !state.has_dependents(entry->id); });
        if (next == state.objects.rend()) {
            state.is_faulted = true;
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        const auto found = std::prev(next.base());
        auto& object = **found;
        if (object.is_uncertain || object.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        const DependencyCall active{state.is_dependency_active,
                                    has_dropout_dependencies(object.kind) ||
                                        object.kind == CudaDnnObjectKind::kBackend ||
                                        object.kind == CudaDnnObjectKind::kAlgorithm};
        if (object.kind == CudaDnnObjectKind::kBackend &&
            (!state.dispatch.version || state.dispatch.version() != 91002))
            return rejected(CUDA_ERROR_NOT_SUPPORTED);
        const auto status = state.destroy_native(object);
        if (status != CUDNN_STATUS_SUCCESS) {
            object.is_uncertain = state.is_faulted = true;
            return completed(status);
        }
        // A later failure must not cause already destroyed objects to be retried.
        state.objects.erase(found);
    }
    return completed(CUDNN_STATUS_SUCCESS);
}
CudaDnnRegistrySnapshot CudaDnnRegistry::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    return {state.objects.size(),
            static_cast<std::size_t>(std::ranges::count_if(
                state.objects, [](const auto& entry) { return entry.use_count() > 1; })),
            state.is_closing, state.is_closing && state.objects.empty(), state.is_quarantined()};
}
#include "internal/cuda_dnn_legacy_registry.inc"
}  // namespace glimmer::backend::cuda
