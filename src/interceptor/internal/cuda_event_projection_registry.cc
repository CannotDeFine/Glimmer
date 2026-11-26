#include "cuda_event_projection_registry.h"
#include "cuda_operation_queue.h"

#include <limits>
#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <condition_variable>
#include <thread>

namespace glimmer::interceptor {

namespace {
struct EventMetadataGate {
    mutable std::mutex mutex;
    mutable std::condition_variable condition;
    mutable std::thread::id caller;
    mutable bool is_busy{};
};
class EventMetadataPermit final {
   public:
    explicit EventMetadataPermit(std::shared_ptr<const EventMetadataGate> gate)
        : gate_(std::move(gate)) {
        if (!gate_) {
            return;
        }
        std::unique_lock lock(gate_->mutex);
        if (gate_->is_busy && gate_->caller == std::this_thread::get_id()) {
            status = CUDA_ERROR_INVALID_VALUE;
            return;
        }
        gate_->condition.wait(lock, [&] { return !gate_->is_busy; });
        gate_->is_busy = true;
        gate_->caller = std::this_thread::get_id();
    }
    ~EventMetadataPermit() {
        if (gate_ && status == CUDA_SUCCESS) {
            {
                const std::lock_guard lock(gate_->mutex);
                gate_->is_busy = false;
                gate_->caller = {};
            }
            gate_->condition.notify_all();
        }
    }
    EventMetadataPermit(const EventMetadataPermit&) = delete;
    EventMetadataPermit& operator=(const EventMetadataPermit&) = delete;
    CUresult status{CUDA_SUCCESS};

   private:
    std::shared_ptr<const EventMetadataGate> gate_;
};
}  // namespace

struct CudaEventEntry {
    std::shared_ptr<const void> identity;
    CUcontext context{};
    std::uint64_t native_epoch{};
    bool is_logical{};
    bool has_native_completion{};
    CUresult registration_status{CUDA_SUCCESS};
    std::shared_ptr<const CudaOperationToken> native_prefix{};
    std::weak_ptr<const void> capture_owner{};
    bool allows_logical_binding{true};
};
using EventMap = std::unordered_map<CUevent, CudaEventEntry>;
struct CudaEventPreparedRegistration {
    EventMap::node_type node;
    CUevent event{};
    bool was_published{};
    bool is_quarantined{};
};
struct CudaEventPreparationState {
    std::mutex mutex;
    EventMap entries;
    std::array<std::shared_ptr<CudaEventPreparedRegistration>, 256> pending;
    CUresult status{CUDA_SUCCESS};
    std::size_t pending_count() const noexcept {
        return static_cast<std::size_t>(std::ranges::count_if(
            pending, [](const auto& slot) { return static_cast<bool>(slot); }));
    }
};

void CudaEventRegistrationReservation::cancel() noexcept {
    if (owner_ && registration_) {
        const std::lock_guard lock(owner_->mutex);
        if (!registration_->is_quarantined) {
            const auto slot = std::ranges::find(owner_->pending, registration_);
            if (slot != owner_->pending.end()) {
                slot->reset();
            }
        }
    }
    registration_.reset();
    owner_.reset();
}
CudaEventRegistrationReservation::~CudaEventRegistrationReservation() {
    cancel();
}
CudaEventRegistrationReservation& CudaEventRegistrationReservation::operator=(
    CudaEventRegistrationReservation&& other) noexcept {
    if (this != &other) {
        cancel();
        owner_ = std::move(other.owner_);
        registration_ = std::move(other.registration_);
    }
    return *this;
}
CudaEventProjectionRegistry::CudaEventProjectionRegistry()
    : state(std::make_shared<CudaEventPreparationState>()) {}

CUresult CudaEventProjectionRegistry::prepare_native_event(
    CUcontext context, CudaEventRegistrationReservation& output) noexcept {
    if (!context) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return prepare_event(context, output);
}
CUresult CudaEventProjectionRegistry::prepare_runtime_event(
    CudaEventRegistrationReservation& output) noexcept {
    return prepare_event(nullptr, output);
}
CUresult CudaEventProjectionRegistry::prepare_unrecorded_runtime_event(
    CudaEventRegistrationReservation& output) noexcept {
    return prepare_event(nullptr, output, true, true);
}
CUresult CudaEventProjectionRegistry::status() const noexcept {
    const std::lock_guard lock(state->mutex);
    return state->status;
}
CUresult CudaEventProjectionRegistry::admit_metadata(
    std::span<const CudaEventProjection> owned,
    const std::function<CUresult()>& operation) noexcept {
    if (!operation || owned.size() > 2) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        std::array<std::shared_ptr<const EventMetadataGate>, 2> gates;
        const auto validate = [&] {
            const std::lock_guard lock(state->mutex);
            if (state->status != CUDA_SUCCESS) {
                return state->status;
            }
            for (const auto& snapshot : owned) {
                const auto found = state->entries.find(snapshot.event_);
                if (!snapshot || snapshot.registry_identity_ != identity ||
                    found == state->entries.end() || found->second.identity != snapshot.identity_) {
                    return CUDA_ERROR_INVALID_HANDLE;
                }
            }
            return CUDA_SUCCESS;
        };
        auto result = validate();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        for (std::size_t index = 0; index < owned.size(); ++index) {
            gates[index] =
                std::static_pointer_cast<const EventMetadataGate>(owned[index].identity_);
        }
        if (gates[0] == gates[1]) {
            gates[1].reset();
        } else if (gates[1] && std::less<const void*>{}(gates[1].get(), gates[0].get())) {
            std::swap(gates[0], gates[1]);
        }
        const EventMetadataPermit first(gates[0]);
        if (first.status != CUDA_SUCCESS) {
            return first.status;
        }
        const EventMetadataPermit second(gates[1]);
        if (second.status != CUDA_SUCCESS) {
            return second.status;
        }
        result = validate();
        return result == CUDA_SUCCESS ? operation() : result;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult CudaEventProjectionRegistry::prepare_record_event(
    CUcontext context, CudaEventRegistrationReservation& output) noexcept {
    return context ? prepare_event(context, output, false) : CUDA_ERROR_INVALID_VALUE;
}
CUresult CudaEventProjectionRegistry::prepare_runtime_record_event(
    CudaEventRegistrationReservation& output) noexcept {
    return prepare_event(nullptr, output, false);
}
CUresult CudaEventProjectionRegistry::prepare_event(CUcontext context,
                                                    CudaEventRegistrationReservation& output,
                                                    bool allows_logical_binding,
                                                    bool is_unrecorded) noexcept {
    if (output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        auto registration = std::make_shared<CudaEventPreparedRegistration>();
        EventMap prepared;
        prepared.emplace(nullptr, CudaEventEntry{std::make_shared<EventMetadataGate>(), context});
        registration->node = prepared.extract(nullptr);
        registration->node.mapped().allows_logical_binding = allows_logical_binding;
        registration->node.mapped().has_native_completion = is_unrecorded;
        const std::lock_guard lock(state->mutex);
        if (state->status != CUDA_SUCCESS) {
            return state->status;
        }
        const auto slot = std::ranges::find(state->pending, nullptr);
        if (slot == state->pending.end()) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        const auto pending = state->pending_count();
        const auto maximum = state->entries.max_size();
        if (pending >= maximum || state->entries.size() >= maximum - pending) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        state->entries.reserve(state->entries.size() + pending + 1);
        *slot = registration;
        output.owner_ = state;
        output.registration_ = std::move(registration);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult CudaEventProjectionRegistry::publish_native_event(
    CudaEventRegistrationReservation& reservation, CUevent event, CUresult native_status) noexcept {
    return publish_prepared_event(reservation, event, native_status, nullptr, false);
}
CUresult CudaEventProjectionRegistry::publish_runtime_event(
    CudaEventRegistrationReservation& reservation, CUevent event, CUcontext context) noexcept {
    return publish_prepared_event(reservation, event, CUDA_SUCCESS, context, true);
}
CUresult CudaEventProjectionRegistry::publish_prepared_event(
    CudaEventRegistrationReservation& reservation, CUevent event, CUresult native_status,
    CUcontext runtime_context, bool is_runtime) noexcept {
    if (reservation.owner_ != state || !reservation.registration_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const std::lock_guard lock(state->mutex);
    auto& registration = *reservation.registration_;
    const auto slot = std::ranges::find(state->pending, reservation.registration_);
    if (registration.was_published || registration.node.empty() || slot == state->pending.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (is_runtime == static_cast<bool>(registration.node.mapped().context)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    registration.was_published = true;
    registration.event = event;
    if (is_runtime) {
        if (!runtime_context) {
            registration.is_quarantined = true;
            state->status = CUDA_ERROR_UNKNOWN;
            return state->status;
        }
        registration.node.mapped().context = runtime_context;
    }
    if (!event) {
        if (native_status == CUDA_SUCCESS) {
            registration.is_quarantined = true;
            state->status = CUDA_ERROR_UNKNOWN;
            return state->status;
        }
        slot->reset();
        return native_status;
    }
    if (state->status != CUDA_SUCCESS || state->entries.contains(event)) {
        registration.is_quarantined = true;
        state->status = CUDA_ERROR_UNKNOWN;
        return native_status == CUDA_SUCCESS ? state->status : native_status;
    }
    registration.node.key() = event;
    registration.node.mapped().registration_status = native_status;
    // Preparation reserves capacity for this node and every concurrent ticket.
    // Pointer hashing/equality and equal-allocator node transfer allocate nothing.
    try {
        auto inserted = state->entries.insert(std::move(registration.node));
        if (!inserted.inserted) {
            registration.node = std::move(inserted.node);
            registration.is_quarantined = true;
            state->status = CUDA_ERROR_UNKNOWN;
            return native_status == CUDA_SUCCESS ? state->status : native_status;
        }
        slot->reset();
        return native_status;
    } catch (...) {
        registration.is_quarantined = true;
        state->status = CUDA_ERROR_UNKNOWN;
        return native_status == CUDA_SUCCESS ? state->status : native_status;
    }
}

CUresult CudaEventProjectionRegistry::register_event(CUcontext context, CUevent event,
                                                     bool is_logical) noexcept {
    if (!context || !event) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        // Allocate before the lock; the unique marker contains no CUDA owner.
        const auto identity = std::make_shared<EventMetadataGate>();
        const std::lock_guard lock(state->mutex);
        if (state->status != CUDA_SUCCESS) {
            return state->status;
        }
        if (state->entries.contains(event)) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        const auto pending = state->pending_count();
        const auto maximum = state->entries.max_size();
        if (pending >= maximum || state->entries.size() >= maximum - pending) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        state->entries.reserve(state->entries.size() + pending + 1);
        return state->entries
                       .emplace(event, CudaEventEntry{identity, context, 0, is_logical, is_logical})
                       .second
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_INVALID_HANDLE;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaEventProjectionRegistry::lookup(CUevent event,
                                             CudaEventProjection& output) const noexcept {
    if (output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    const auto found = state->entries.find(event);
    if (found == state->entries.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const auto& entry = found->second;
    if (entry.registration_status != CUDA_SUCCESS) {
        return entry.registration_status;
    }
    output.identity_ = entry.identity;
    output.registry_identity_ = identity;
    output.event_ = event;
    output.context_ = entry.context;
    output.native_epoch_ = entry.native_epoch;
    output.is_logical_ = entry.is_logical;
    output.has_native_completion_ = entry.has_native_completion;
    output.native_prefix_ = entry.native_prefix;
    output.capture_owner_ = entry.capture_owner;
    output.allows_logical_binding_ = entry.allows_logical_binding;
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::lookup_owned(const CudaEventProjection& owned,
                                                   CudaEventProjection& output) const noexcept {
    if (!owned || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    const auto found = state->entries.find(owned.event_);
    if (found == state->entries.end() || found->second.identity != owned.identity_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const auto& entry = found->second;
    if (entry.registration_status != CUDA_SUCCESS) {
        return entry.registration_status;
    }
    output.identity_ = entry.identity;
    output.registry_identity_ = identity;
    output.event_ = owned.event_;
    output.context_ = entry.context;
    output.native_epoch_ = entry.native_epoch;
    output.is_logical_ = entry.is_logical;
    output.has_native_completion_ = entry.has_native_completion;
    output.native_prefix_ = entry.native_prefix;
    output.capture_owner_ = entry.capture_owner;
    output.allows_logical_binding_ = entry.allows_logical_binding;
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::publish_capture_record(
    const CudaEventProjection& owned, const std::shared_ptr<const void>& capture_owner) noexcept {
    if (!owned || owned.registry_identity_ != identity) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    const auto found = state->entries.find(owned.event_);
    if (found == state->entries.end() || found->second.identity != owned.identity_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (found->second.native_epoch != owned.native_epoch_) {
        return CUDA_ERROR_NOT_READY;
    }
    found->second.capture_owner = capture_owner;
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::publish_logical(
    const CudaEventProjection& accepted,
    const CudaOperationToken* accepted_native_prefix) noexcept {
    if (!accepted) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    const auto found = state->entries.find(accepted.event_);
    if (found == state->entries.end() || found->second.identity != accepted.identity_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (found->second.native_epoch != accepted.native_epoch_) {
        return CUDA_ERROR_NOT_READY;
    }
    if (!found->second.allows_logical_binding) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!found->second.is_logical && !found->second.has_native_completion) {
        const auto& prefix = found->second.native_prefix;
        if (!prefix || !accepted_native_prefix || !*accepted_native_prefix ||
            !accepted.native_prefix_ || *prefix != *accepted_native_prefix ||
            *accepted.native_prefix_ != *accepted_native_prefix) {
            return CUDA_ERROR_NOT_READY;
        }
    }
    found->second.is_logical = true;
    found->second.native_prefix.reset();
    found->second.capture_owner.reset();
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::matches_native_epoch(const CudaEventProjection& accepted,
                                                           bool& output) const noexcept {
    output = false;
    if (!accepted) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    const auto found = state->entries.find(accepted.event_);
    if (found == state->entries.end() || found->second.identity != accepted.identity_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    output = found->second.native_epoch == accepted.native_epoch_;
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::invalidate_native_event(
    const CudaEventProjection& owned) noexcept {
    auto snapshot = owned;
    return invalidate_native_events(std::span{&snapshot, 1});
}

CUresult CudaEventProjectionRegistry::invalidate_native_events(
    std::span<CudaEventProjection> owned,
    const std::shared_ptr<const CudaOperationToken>& issued_prefix) noexcept {
    if (issued_prefix && !*issued_prefix) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (std::size_t index = 0; index < owned.size(); ++index) {
        const auto& snapshot = owned[index];
        if (!snapshot) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const auto found = state->entries.find(snapshot.event_);
        if (snapshot.registry_identity_ != identity || found == state->entries.end() ||
            found->second.identity != snapshot.identity_) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        if (found->second.native_epoch != snapshot.native_epoch_) {
            return CUDA_ERROR_NOT_READY;
        }
        for (const auto& previous : owned.first(index)) {
            if (previous.event_ == snapshot.event_) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
        if (found->second.native_epoch == std::numeric_limits<std::uint64_t>::max()) {
            state->status = CUDA_ERROR_UNKNOWN;
            return state->status;
        }
    }
    for (auto& snapshot : owned) {
        auto& entry = state->entries.find(snapshot.event_)->second;
        ++entry.native_epoch;
        entry.is_logical = false;
        entry.has_native_completion = false;
        entry.native_prefix = issued_prefix;
        snapshot.native_epoch_ = entry.native_epoch;
        snapshot.is_logical_ = false;
        snapshot.has_native_completion_ = false;
        snapshot.native_prefix_ = issued_prefix;
    }
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::complete_native_events(
    std::span<const CudaEventProjection> proven) noexcept {
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (const auto& snapshot : proven) {
        if (!snapshot) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (snapshot.registry_identity_ != identity) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        if (snapshot.is_logical_) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    for (const auto& snapshot : proven) {
        const auto found = state->entries.find(snapshot.event_);
        if (found != state->entries.end() && found->second.identity == snapshot.identity_ &&
            found->second.native_epoch == snapshot.native_epoch_) {
            found->second.has_native_completion = true;
        }
    }
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::invalidate_native(CUcontext context) noexcept {
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (const auto& [event, entry] : state->entries) {
        (void)event;
        if ((!context || context == entry.context) &&
            entry.native_epoch == std::numeric_limits<std::uint64_t>::max()) {
            state->status = CUDA_ERROR_UNKNOWN;
            return state->status;
        }
    }
    for (auto& [event, entry] : state->entries) {
        (void)event;
        if (!context || context == entry.context) {
            ++entry.native_epoch;
            entry.is_logical = false;
            entry.has_native_completion = false;
            entry.native_prefix.reset();
        }
    }
    return CUDA_SUCCESS;
}

CUresult CudaEventProjectionRegistry::complete_native(CUcontext context) noexcept {
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (auto& [event, entry] : state->entries) {
        (void)event;
        if ((!context || context == entry.context) && entry.registration_status == CUDA_SUCCESS) {
            entry.has_native_completion = true;
        }
    }
    return CUDA_SUCCESS;
}

void CudaEventProjectionRegistry::forget(CUevent event) noexcept {
    const std::lock_guard lock(state->mutex);
    state->entries.erase(event);
}

CUresult CudaEventProjectionRegistry::forget_owned(const CudaEventProjection& owned) noexcept {
    const std::lock_guard lock(state->mutex);
    if (!owned || owned.registry_identity_ != identity) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto found = state->entries.find(owned.event_);
    if (found == state->entries.end() || found->second.identity != owned.identity_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    state->entries.erase(found);
    return CUDA_SUCCESS;
}

void CudaEventProjectionRegistry::forget_context(CUcontext context) noexcept {
    const std::lock_guard lock(state->mutex);
    std::erase_if(state->entries,
                  [&](const auto& value) { return value.second.context == context; });
}

}  // namespace glimmer::interceptor
