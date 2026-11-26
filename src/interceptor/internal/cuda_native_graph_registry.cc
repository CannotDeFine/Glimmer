#include "cuda_native_graph_registry.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <utility>
#include <vector>

namespace glimmer::interceptor {
struct CudaNativeGraphRegistration {
    CUcontext context{};
    CUgraphExec executable{};
    CudaNativeGraphEffects effects;
    std::vector<CudaNativeGraphBoundEvent> events;
    bool is_prepared{};
    // Constructor success, supported effects and confirmed context; immutable
    // across mutation generations, unlike current replay readiness.
    bool has_supported_origin{};
    bool is_quarantined{};
    bool needs_context_publication{};
};
struct CudaNativeGraphRegistryState {
    std::mutex mutex;
    std::array<std::shared_ptr<CudaNativeGraphRegistration>, 256> entries;
    CUresult status{CUDA_SUCCESS};
};

CUcontext CudaNativeGraphExecutable::context() const noexcept {
    return registration_ ? registration_->context : nullptr;
}
const CudaNativeGraphEffects& CudaNativeGraphExecutable::effects() const noexcept {
    static const CudaNativeGraphEffects empty;
    return registration_ ? registration_->effects : empty;
}
std::span<const CudaNativeGraphBoundEvent> CudaNativeGraphExecutable::events() const noexcept {
    return registration_ ? std::span<const CudaNativeGraphBoundEvent>{registration_->events}
                         : std::span<const CudaNativeGraphBoundEvent>{};
}

void CudaNativeGraphReservation::cancel() noexcept {
    if (owner_ && registration_) {
        const std::lock_guard lock(owner_->mutex);
        if (!registration_->executable && !registration_->is_quarantined) {
            for (auto& entry : owner_->entries) {
                if (entry == registration_) {
                    entry.reset();
                    break;
                }
            }
        }
    }
    registration_.reset();
    previous_.reset();
    owner_.reset();
}
CudaNativeGraphReservation::~CudaNativeGraphReservation() {
    cancel();
}
CudaNativeGraphReservation& CudaNativeGraphReservation::operator=(
    CudaNativeGraphReservation&& other) noexcept {
    if (this != &other) {
        cancel();
        owner_ = std::move(other.owner_);
        registration_ = std::move(other.registration_);
        previous_ = std::move(other.previous_);
    }
    return *this;
}
CudaNativeGraphRegistry::CudaNativeGraphRegistry()
    : state(std::make_shared<CudaNativeGraphRegistryState>()) {}

CUresult CudaNativeGraphRegistry::prepare(CUcontext context, const CudaNativeGraphEffects& effects,
                                          const CudaEventProjectionRegistry& events,
                                          CudaNativeGraphReservation& output) noexcept {
    if (!context || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return prepare_impl(context, effects, events, output, false);
}

CUresult CudaNativeGraphRegistry::prepare_unbound(const CudaNativeGraphEffects& effects,
                                                  const CudaEventProjectionRegistry& events,
                                                  CudaNativeGraphReservation& output) noexcept {
    if (output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return prepare_impl(nullptr, effects, events, output, true);
}

CUresult CudaNativeGraphRegistry::prepare_impl(CUcontext context,
                                               const CudaNativeGraphEffects& effects,
                                               const CudaEventProjectionRegistry& events,
                                               CudaNativeGraphReservation& output,
                                               bool needs_context_publication) noexcept {
    try {
        auto registration = std::make_shared<CudaNativeGraphRegistration>();
        registration->context = context;
        registration->effects = effects;
        registration->is_prepared = static_cast<bool>(effects);
        registration->needs_context_publication = needs_context_publication;
        registration->events.reserve(effects.events().size());
        for (const auto& effect : effects.events()) {
            CudaEventProjection projection;
            const auto result = events.lookup(effect.event, projection);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            if (!needs_context_publication && projection.context() != context) {
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            registration->events.push_back({effect, std::move(projection)});
        }
        const std::lock_guard lock(state->mutex);
        if (state->status != CUDA_SUCCESS) {
            return state->status;
        }
        const auto available = std::ranges::find(state->entries, nullptr);
        if (available == state->entries.end()) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        *available = registration;
        output.owner_ = state;
        output.registration_ = std::move(registration);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeGraphRegistry::publish(CudaNativeGraphReservation& reservation,
                                          CUgraphExec executable, CUresult native_status,
                                          std::optional<CUcontext> actual_context) noexcept {
    if (reservation.owner_ != state || !reservation.registration_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const std::lock_guard lock(state->mutex);
    auto& registration = *reservation.registration_;
    if (registration.executable || registration.is_quarantined ||
        std::ranges::find(state->entries, reservation.registration_) == state->entries.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!executable) {
        if (native_status == CUDA_SUCCESS) {
            registration.is_quarantined = true;
            state->status = CUDA_ERROR_UNKNOWN;
            return state->status;
        }
        // Scope exit cancels the known unpublished slot.
        return native_status;
    }
    const bool is_duplicate = std::ranges::any_of(state->entries, [&](const auto& entry) {
        return entry && entry->executable == executable;
    });
    registration.executable = executable;
    const bool has_unknown_context = registration.needs_context_publication && !actual_context;
    const bool has_changed_context = !registration.needs_context_publication && actual_context &&
                                     *actual_context != registration.context;
    if (is_duplicate || state->status != CUDA_SUCCESS || has_unknown_context ||
        has_changed_context) {
        registration.is_prepared = false;
        registration.is_quarantined = true;
        state->status = CUDA_ERROR_UNKNOWN;
        return native_status == CUDA_SUCCESS ? state->status : native_status;
    }
    if (registration.needs_context_publication) {
        registration.context = *actual_context;
        registration.needs_context_publication = false;
        if (!registration.context ||
            std::ranges::any_of(registration.events, [&](const auto& bound) {
                return bound.registration.context() != registration.context;
            })) {
            // Native constructor success is not single-context replay proof.
            registration.is_prepared = false;
        }
    }
    if (native_status != CUDA_SUCCESS) {
        registration.is_prepared = false;
    }
    registration.has_supported_origin = registration.is_prepared;
    return native_status;
}

CUresult CudaNativeGraphRegistry::lookup(CUgraphExec executable,
                                         CudaNativeGraphExecutable& output) const noexcept {
    if (!executable || output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (const auto& entry : state->entries) {
        if (entry && entry->executable == executable) {
            if (!entry->is_prepared) {
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            output.registration_ = entry;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}
CUresult CudaNativeGraphRegistry::is_current(
    const CudaNativeGraphExecutable& owned) const noexcept {
    if (!owned) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::lock_guard lock(state->mutex);
    if (state->status != CUDA_SUCCESS) {
        return state->status;
    }
    for (const auto& entry : state->entries) {
        if (entry == owned.registration_) {
            return entry->is_prepared ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}
CUresult CudaNativeGraphRegistry::prepare_event_update(
    CUgraphExec executable, CUgraphNode node, CUevent event, CudaNativeGraphEventKind kind,
    const CudaEventProjectionRegistry& events, CudaNativeGraphReservation& output) noexcept {
    return prepare_update(executable, node, event, kind, std::nullopt, &events, output);
}

CUresult CudaNativeGraphRegistry::prepare_node_update(CUgraphExec executable, CUgraphNode node,
                                                      std::optional<CUgraphNodeType> expected_type,
                                                      CudaNativeGraphReservation& output) noexcept {
    return prepare_update(executable, node, nullptr, std::nullopt, expected_type, nullptr, output);
}

CUresult CudaNativeGraphRegistry::prepare_graph_update(
    CUgraphExec executable, const CudaNativeGraphEffects& effects,
    std::optional<CUgraphNode> child_node, const CudaEventProjectionRegistry& events,
    CudaNativeGraphReservation& output) noexcept {
    return prepare_update(executable, nullptr, nullptr, std::nullopt, std::nullopt, &events, output,
                          &effects, child_node);
}

CUresult CudaNativeGraphRegistry::prepare_update(CUgraphExec executable, CUgraphNode node,
                                                 CUevent event,
                                                 std::optional<CudaNativeGraphEventKind> event_kind,
                                                 std::optional<CUgraphNodeType> expected_type,
                                                 const CudaEventProjectionRegistry* events,
                                                 CudaNativeGraphReservation& output,
                                                 const CudaNativeGraphEffects* graph_replacement,
                                                 std::optional<CUgraphNode> child_node) noexcept {
    if (!executable || output) return CUDA_ERROR_INVALID_VALUE;
    try {
        std::shared_ptr<CudaNativeGraphRegistration> previous;
        bool was_prepared{};
        {
            const std::lock_guard lock(state->mutex);
            if (state->status != CUDA_SUCCESS) return state->status;
            for (const auto& entry : state->entries) {
                if (entry && entry->executable == executable) {
                    previous = entry;
                    was_prepared = entry->is_prepared;
                    break;
                }
            }
        }
        if (!previous) return CUDA_ERROR_INVALID_HANDLE;
        auto replacement = std::make_shared<CudaNativeGraphRegistration>();
        replacement->context = previous->context;
        replacement->executable = executable;
        replacement->effects = previous->effects;
        replacement->events = previous->events;
        replacement->has_supported_origin = previous->has_supported_origin;
        CudaNativeGraphEffects effects;
        const bool can_prepare = graph_replacement
                                     ? previous->has_supported_origin && previous->effects &&
                                           (!child_node || was_prepared)
                                     : was_prepared;
        const auto updated =
            can_prepare
                ? graph_replacement
                      ? child_node ? previous->effects.replace_child(*child_node,
                                                                     *graph_replacement, effects)
                                   : previous->effects.replace_graph(*graph_replacement, effects)
                  : event_kind ? previous->effects.replace_event(node, event, *event_kind, effects)
                               : previous->effects.preserve_node(node, expected_type, effects)
                : CUDA_ERROR_NOT_SUPPORTED;
        if (updated == CUDA_ERROR_OUT_OF_MEMORY || updated == CUDA_ERROR_UNKNOWN) return updated;
        if (updated == CUDA_SUCCESS) {
            replacement->effects = std::move(effects);
            replacement->is_prepared = true;
        }
        if (updated == CUDA_SUCCESS && graph_replacement) {
            replacement->events.clear();
            replacement->events.reserve(replacement->effects.events().size());
            for (const auto& effect : replacement->effects.events()) {
                CudaEventProjection projection;
                const auto bound = events->lookup(effect.event, projection);
                if (bound != CUDA_SUCCESS || projection.context() != previous->context)
                    replacement->is_prepared = false;
                replacement->events.push_back({effect, std::move(projection)});
            }
        }
        if (updated == CUDA_SUCCESS && event_kind) {
            CudaEventProjection projection;
            const auto bound = events->lookup(event, projection);
            replacement->is_prepared =
                bound == CUDA_SUCCESS && projection.context() == previous->context;
            for (auto& target : replacement->events) {
                if (target.effect.node == node && target.effect.kind == *event_kind) {
                    target.effect.event = event;
                    if (replacement->is_prepared) target.registration = projection;
                }
            }
        }
        {
            const std::lock_guard lock(state->mutex);
            if (state->status != CUDA_SUCCESS) return state->status;
            if (std::ranges::find(state->entries, previous) == state->entries.end() ||
                previous->is_prepared != was_prepared)
                return CUDA_ERROR_NOT_READY;
            previous->is_prepared = false;
            output.owner_ = state;
            output.previous_ = std::move(previous);
            output.registration_ = std::move(replacement);
        }
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeGraphRegistry::publish_event_update(CudaNativeGraphReservation& reservation,
                                                       CUresult native_status) noexcept {
    if (reservation.owner_ != state || !reservation.previous_ || !reservation.registration_) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUresult result = native_status;
    {
        const std::lock_guard lock(state->mutex);
        const auto previous = std::ranges::find(state->entries, reservation.previous_);
        if (previous == state->entries.end() || state->status != CUDA_SUCCESS) {
            state->status = CUDA_ERROR_UNKNOWN;
            result = native_status == CUDA_SUCCESS ? state->status : native_status;
        } else if (native_status == CUDA_SUCCESS) {
            *previous = reservation.registration_;
        }
    }
    // Release staged CPU ownership outside the registry lock; never undo issue.
    reservation.cancel();
    return result;
}

void CudaNativeGraphRegistry::invalidate(CUgraphExec executable) noexcept {
    if (!executable) {
        return;
    }
    const std::lock_guard lock(state->mutex);
    for (const auto& entry : state->entries) {
        if (entry && entry->executable == executable) {
            entry->is_prepared = false;
        }
    }
}
void CudaNativeGraphRegistry::forget(CUgraphExec executable) noexcept {
    if (!executable) {
        return;
    }
    const std::lock_guard lock(state->mutex);
    for (auto& entry : state->entries) {
        if (entry && entry->executable == executable && !entry->is_quarantined) {
            entry.reset();
        }
    }
}
void CudaNativeGraphRegistry::forget_context(CUcontext context) noexcept {
    const std::lock_guard lock(state->mutex);
    for (auto& entry : state->entries) {
        if (entry && entry->context == context && !entry->is_quarantined) {
            entry.reset();
        }
    }
}
}  // namespace glimmer::interceptor
