#include "internal/cuda_operation_queue.h"

#include "internal/diagnostics.h"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace glimmer::interceptor {

struct CudaQueuedResult {
    OrderedOperationToken operation;
    CUresult error{CUDA_SUCCESS};
};

struct CudaQueueEventState {
    CUevent native{};
    std::shared_ptr<void> resources;
    CudaOperationToken latest;
    bool is_open{true};
};

struct CudaOperationQueue::Implementation {
    struct Stream {
        OrderedStreamToken token;
        CUstream native{};
        std::shared_ptr<void> resources;
        CudaOperationToken tail;
        OrderedStreamKind kind{};
        bool is_open{true};
    };
    struct Entry {
        CudaOperationToken token;
        CUstream stream{};
        std::shared_ptr<void> stream_resources;
        CudaQueuedOperation operation;
        std::optional<PhysicalExecutionTicket> execution;
        CUevent event{};
        bool is_physically_stopped{true};
        bool has_free_submission{};
        bool has_ambiguous_free{};
        bool has_native_prefix_proof{true};
        bool has_ambiguous_event_cleanup{};
        std::shared_ptr<void> native_reservation_resources{};
    };

    Implementation(CUcontext supplied_context, CudaQueueDispatch supplied_dispatch,
                   OperationOrderLimits limits)
        : context(supplied_context),
          dispatch(supplied_dispatch),
          order(limits),
          owner_thread(std::this_thread::get_id()),
          pending_limit(limits.pending_operations) {
        if (context == nullptr || !dispatch.context_get_current || !dispatch.event_create ||
            !dispatch.event_record || !dispatch.event_query || !dispatch.event_destroy ||
            !dispatch.stream_synchronize || limits.pending_operations == 0 ||
            limits.pending_operations > 65536 || limits.streams == 0 || limits.streams > 4096) {
            throw std::invalid_argument("invalid CUDA operation queue configuration");
        }
        entries.reserve(limits.pending_operations);
        streams.reserve(limits.streams);
        events.reserve(limits.pending_operations);
    }

    [[nodiscard]] CUresult check_context() const noexcept {
        if (owner_thread != std::this_thread::get_id()) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        CUcontext current = nullptr;
        const auto result = dispatch.context_get_current(&current);
        return result == CUDA_SUCCESS && current != context ? CUDA_ERROR_INVALID_CONTEXT : result;
    }
    [[nodiscard]] Stream* find_stream(const OrderedStreamToken& token) noexcept {
        const auto found = std::ranges::find(streams, token, &Stream::token);
        return found == streams.end() ? nullptr : &*found;
    }
    [[nodiscard]] Entry* find_entry(const CudaOperationToken& token) noexcept {
        const auto found =
            std::ranges::find_if(entries, [&](const auto& entry) { return entry->token == token; });
        return found == entries.end() ? nullptr : found->get();
    }
    [[nodiscard]] CudaQueueEventState* find_event(const CudaQueueEventToken& token) const noexcept {
        const auto found = std::ranges::find(events, token.state_);
        return found == events.end() || !(*found)->is_open ? nullptr : found->get();
    }
    [[nodiscard]] bool validates(const CudaQueuedOperation& operation) const noexcept {
        if (const auto* kernel = std::get_if<CudaQueuedKernel>(&operation)) {
            return dispatch.launch_kernel && kernel->function && kernel->resources &&
                   std::ranges::all_of(kernel->grid, [](auto extent) { return extent != 0; }) &&
                   std::ranges::all_of(kernel->block, [](auto extent) { return extent != 0; });
        }
        if (const auto* copy = std::get_if<CudaQueuedCopy>(&operation)) {
            if (!copy->resources || copy->memory_bytes == 0) {
                return false;
            }
            switch (copy->kind) {
                case CudaQueuedCopyKind::kHostToDevice:
                    return dispatch.copy_host_to_device && copy->destination && copy->host_pointer;
                case CudaQueuedCopyKind::kDeviceToHost:
                    return dispatch.copy_device_to_host && copy->source && copy->host_pointer;
                case CudaQueuedCopyKind::kDeviceToDevice:
                    return dispatch.copy_device_to_device && copy->source && copy->destination;
            }
            return false;
        }
        if (const auto* free = std::get_if<CudaQueuedFree>(&operation)) {
            return dispatch.free_async && free->pointer && free->resources &&
                   free->release_observer;
        }
        if (const auto* record = std::get_if<CudaQueuedEventRecord>(&operation)) {
            return find_event(record->event) != nullptr;
        }
        if (const auto* fence = std::get_if<CudaQueuedNativeFence>(&operation)) {
            return dispatch.stream_wait_event && dispatch.event_synchronize && fence->event &&
                   fence->resources;
        }
        if (const auto* reservation = std::get_if<CudaQueuedNativeReservation>(&operation)) {
            return reservation->resources != nullptr;
        }
        return true;
    }
    [[nodiscard]] CUresult issue(Entry& entry) noexcept {
        if (auto* kernel = std::get_if<CudaQueuedKernel>(&entry.operation)) {
            auto grid = kernel->grid;
            if (kernel->continuation) {
                const auto result = kernel->continuation->prepare_launch(grid, kernel->arguments);
                if (result != CUDA_SUCCESS) {
                    return result;
                }
            }
            if (std::ranges::any_of(grid, [](auto extent) { return extent == 0; })) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            // CUDA consumes these owned addresses during the native call.
            const auto status = dispatch.launch_kernel(
                kernel->function, grid[0], grid[1], grid[2], kernel->block[0], kernel->block[1],
                kernel->block[2], kernel->shared_memory_bytes, entry.stream,
                const_cast<void**>(kernel->arguments.addresses().data()), nullptr);
            return status == CUDA_SUCCESS && kernel->continuation
                       ? kernel->continuation->stage_completion(entry.stream)
                       : status;
        }
        if (const auto* copy = std::get_if<CudaQueuedCopy>(&entry.operation)) {
            switch (copy->kind) {
                case CudaQueuedCopyKind::kHostToDevice:
                    return dispatch.copy_host_to_device(copy->destination, copy->host_pointer,
                                                        copy->memory_bytes, entry.stream);
                case CudaQueuedCopyKind::kDeviceToHost:
                    return dispatch.copy_device_to_host(copy->host_pointer, copy->source,
                                                        copy->memory_bytes, entry.stream);
                case CudaQueuedCopyKind::kDeviceToDevice:
                    return dispatch.copy_device_to_device(copy->destination, copy->source,
                                                          copy->memory_bytes, entry.stream);
            }
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (const auto* free = std::get_if<CudaQueuedFree>(&entry.operation)) {
            const auto status = dispatch.free_async(free->pointer, entry.stream);
            entry.has_free_submission = status == CUDA_SUCCESS;
            entry.has_ambiguous_free = status != CUDA_SUCCESS;
            return status;
        }
        if (const auto* record = std::get_if<CudaQueuedEventRecord>(&entry.operation)) {
            if (record->guard) {
                bool should_write = false;
                const auto checked = record->guard->should_write(should_write);
                if (checked != CUDA_SUCCESS || !should_write) {
                    return checked;
                }
            }
            return dispatch.event_record(record->event.state_->native, entry.stream);
        }
        if (const auto* fence = std::get_if<CudaQueuedNativeFence>(&entry.operation)) {
            return dispatch.stream_wait_event(entry.stream, fence->event, 0);
        }
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult destroy_event(Entry& entry) noexcept {
        if (entry.has_ambiguous_event_cleanup) {
            return entry.token.result_->error == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN
                                                              : entry.token.result_->error;
        }
        if (entry.event == nullptr) {
            return CUDA_SUCCESS;
        }
        const auto result = dispatch.event_destroy(entry.event);
        if (result == CUDA_SUCCESS) {
            entry.event = nullptr;
        } else {
            // An error can follow actual destruction. Do not touch this handle
            // again: another native allocation could already have recycled it.
            entry.has_ambiguous_event_cleanup = true;
        }
        return result;
    }
    [[nodiscard]] CUresult confirm_release(Entry& entry) noexcept {
        if (entry.has_free_submission) {
            const auto* free = std::get_if<CudaQueuedFree>(&entry.operation);
            if (!free || !free->release_observer) {
                return poison(entry, CUDA_ERROR_UNKNOWN);
            }
            free->release_observer->confirmed_release(free->pointer);
            entry.has_free_submission = false;
        }
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult confirm_native_prefix(Entry& entry) noexcept {
        entry.has_native_prefix_proof = true;
        auto* fence = std::get_if<CudaQueuedNativeFence>(&entry.operation);
        if (fence && fence->completion_observer) {
            // Consume before callback: even a metadata failure must not replay.
            const auto observer = std::move(fence->completion_observer);
            const auto status = observer->confirmed_native_completion();
            if (status != CUDA_SUCCESS) {
                return poison(entry, status);
            }
        }
        return CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult poison(Entry& entry, CUresult result) noexcept {
        if (entry.token.result_->error == CUDA_SUCCESS) {
            entry.token.result_->error = result;
        }
        if (fault == CUDA_SUCCESS) {
            fault = result;
        }
        return result;
    }
    void collect() noexcept {
        std::erase_if(entries, [&](const auto& entry) {
            const auto state = order.state(entry->token.result_->operation);
            return entry->event == nullptr && entry->is_physically_stopped &&
                   entry->has_native_prefix_proof &&
                   (state == OrderedOperationState::kSucceeded ||
                    state == OrderedOperationState::kFailed);
        });
        std::erase_if(streams, [&](const auto& stream) {
            return !stream.is_open && order.state(stream.tail ? stream.tail.result_->operation
                                                              : OrderedOperationToken{}) ==
                                          OrderedOperationState::kSucceeded;
        });
        std::erase_if(events, [&](const auto& event) {
            const auto state = event->latest ? order.state(event->latest.result_->operation)
                                             : std::optional(OrderedOperationState::kSucceeded);
            return !event->is_open && (state == OrderedOperationState::kSucceeded ||
                                       state == OrderedOperationState::kFailed);
        });
    }

    CUcontext context;
    CudaQueueDispatch dispatch;
    OperationOrder order;
    std::thread::id owner_thread;
    const std::size_t pending_limit;
    std::vector<Stream> streams;
    std::vector<std::unique_ptr<Entry>> entries;
    std::vector<std::shared_ptr<CudaQueueEventState>> events;
    CUresult fault{CUDA_SUCCESS};
    bool is_closed{};
};

CudaOperationQueue::CudaOperationQueue(CUcontext context, CudaQueueDispatch dispatch,
                                       OperationOrderLimits limits)
    : implementation_(std::make_unique<Implementation>(context, dispatch, limits)) {}
CudaOperationQueue::~CudaOperationQueue() {
    if (shutdown() != CUDA_SUCCESS) {
        // An unknown physical stop cannot authorize releasing CUDA inputs.
        // Intentionally retain this bounded owner until process termination.
        (void)implementation_.release();
        report_diagnostic("CUDA operation queue shutdown unproven; resources quarantined");
    }
}

CUresult CudaOperationQueue::adopt_drained_owner() noexcept {
    auto& owner = *implementation_;
    if (owner.is_closed || owner.fault != CUDA_SUCCESS || !owner.entries.empty()) {
        return CUDA_ERROR_NOT_READY;
    }
    CUcontext current{};
    const auto status = owner.dispatch.context_get_current(&current);
    if (status != CUDA_SUCCESS || current != owner.context) {
        return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
    }
    owner.owner_thread = std::this_thread::get_id();
    return CUDA_SUCCESS;
}

OrderedStreamToken CudaOperationQueue::bind_stream(CUstream stream, OrderedStreamKind kind,
                                                   std::shared_ptr<void> resources) noexcept {
    auto& owner = *implementation_;
    if (owner.is_closed || owner.fault != CUDA_SUCCESS || !resources ||
        (kind == OrderedStreamKind::kLegacyDefault && stream != nullptr &&
         stream != CU_STREAM_LEGACY) ||
        (kind == OrderedStreamKind::kPerThreadDefault &&
         (stream == nullptr || stream == CU_STREAM_LEGACY)) ||
        ((kind == OrderedStreamKind::kBlocking || kind == OrderedStreamKind::kNonBlocking) &&
         (stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)) ||
        std::ranges::any_of(owner.streams,
                            [&](const auto& binding) { return binding.native == stream; })) {
        return {};
    }
    auto token = owner.order.create_stream(kind);
    if (token) {
        owner.streams.push_back({token, stream, std::move(resources), {}, kind, true});
    }
    return token;
}
bool CudaOperationQueue::close_stream(const OrderedStreamToken& stream) noexcept {
    auto& owner = *implementation_;
    auto* binding = owner.find_stream(stream);
    if (!binding || !binding->is_open || !owner.order.close_stream(stream)) {
        return false;
    }
    binding->is_open = false;
    owner.collect();
    return true;
}
CudaQueueEventToken CudaOperationQueue::bind_event(CUevent event,
                                                   std::shared_ptr<void> resources) noexcept {
    auto& owner = *implementation_;
    if (owner.is_closed || owner.fault != CUDA_SUCCESS || !event || !resources ||
        owner.events.size() >= owner.pending_limit ||
        std::ranges::any_of(owner.events,
                            [&](const auto& binding) { return binding->native == event; })) {
        return {};
    }
    try {
        CudaQueueEventToken token;
        token.state_ = std::make_shared<CudaQueueEventState>(
            CudaQueueEventState{event, std::move(resources), {}, true});
        owner.events.push_back(token.state_);
        return token;
    } catch (...) {
        return {};
    }
}
bool CudaOperationQueue::close_event(const CudaQueueEventToken& event) noexcept {
    auto& owner = *implementation_;
    auto* binding = owner.find_event(event);
    if (!binding) {
        return false;
    }
    binding->is_open = false;
    owner.collect();
    return true;
}
std::optional<CudaOperationToken> CudaOperationQueue::event_snapshot(
    const CudaQueueEventToken& event) const noexcept {
    const auto* binding = implementation_->find_event(event);
    return binding ? std::optional(binding->latest) : std::nullopt;
}
CUresult CudaOperationQueue::query_event(const CudaQueueEventToken& event) const noexcept {
    const auto snapshot = event_snapshot(event);
    return !snapshot ? CUDA_ERROR_INVALID_HANDLE : *snapshot ? query(*snapshot) : CUDA_SUCCESS;
}
CudaOperationToken CudaOperationQueue::wait_event(const OrderedStreamToken& stream,
                                                  const CudaQueueEventToken& event) noexcept {
    const auto snapshot = event_snapshot(event);
    if (!snapshot) {
        return {};
    }
    const std::array dependencies{*snapshot};
    return submit(stream, CudaQueuedMarker{},
                  *snapshot ? std::span<const CudaOperationToken>(dependencies)
                            : std::span<const CudaOperationToken>{});
}
CudaOperationToken CudaOperationQueue::submit(
    const OrderedStreamToken& stream, CudaQueuedOperation operation,
    std::span<const CudaOperationToken> dependencies) noexcept {
    auto& owner = *implementation_;
    auto* binding = owner.find_stream(stream);
    if (owner.is_closed || owner.fault != CUDA_SUCCESS || !binding || !binding->is_open ||
        owner.entries.size() >= owner.pending_limit || dependencies.size() > owner.pending_limit ||
        !owner.validates(operation)) {
        return {};
    }
    try {
        std::vector<OrderedOperationToken> ordered_dependencies;
        ordered_dependencies.reserve(dependencies.size());
        for (const auto& dependency : dependencies) {
            if (!state(dependency).has_value()) {
                return {};
            }
            ordered_dependencies.push_back(dependency ? dependency.result_->operation
                                                      : OrderedOperationToken{});
        }
        const auto* record = std::get_if<CudaQueuedEventRecord>(&operation);
        auto* event = record ? owner.find_event(record->event) : nullptr;
        if (event && event->latest && record->should_order_previous) {
            ordered_dependencies.push_back(event->latest.result_->operation);
        }
        auto result = std::make_shared<CudaQueuedResult>();
        CudaOperationToken token;
        token.result_ = result;
        auto entry =
            std::make_unique<Implementation::Entry>(Implementation::Entry{token,
                                                                          binding->native,
                                                                          binding->resources,
                                                                          std::move(operation),
                                                                          {},
                                                                          nullptr,
                                                                          true,
                                                                          false,
                                                                          false});
        const auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
        entry->has_native_prefix_proof =
            !std::holds_alternative<CudaQueuedNativeFence>(entry->operation);
        result->operation = owner.order.submit(stream, ordered_dependencies,
                                               kernel && kernel->continuation != nullptr);
        if (!result->operation) {
            return {};
        }
        // Reserved vectors and noexcept moves: no failure after publishing tail.
        owner.entries.push_back(std::move(entry));
        binding->tail = token;
        if (event) {
            event->latest = token;
        }
        owner.collect();
        return token;
    } catch (...) {
        return {};
    }
}

CUresult CudaOperationQueue::pump() noexcept {
    return pump_selected(std::nullopt);
}
CUresult CudaOperationQueue::pump_admitted(std::span<const CudaOperationToken> admitted) noexcept {
    return pump_selected(admitted);
}
std::vector<CudaKernelCandidate> CudaOperationQueue::kernel_candidates() const {
    const auto& owner = *implementation_;
    std::vector<CudaKernelCandidate> candidates;
    if (owner.is_closed || owner.fault != CUDA_SUCCESS) {
        return candidates;
    }
    candidates.reserve(owner.entries.size());
    for (const auto& entry : owner.entries) {
        const auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
        if (!kernel && !std::holds_alternative<CudaQueuedNativeReservation>(entry->operation)) {
            continue;
        }
        const auto current = state(entry->token);
        if (current == OrderedOperationState::kRunning ||
            current == OrderedOperationState::kSuspended ||
            (current == OrderedOperationState::kQueued &&
             owner.order.is_ready(entry->token.result_->operation))) {
            candidates.push_back(
                {entry->token, *current, kernel && kernel->continuation != nullptr});
        }
    }
    return candidates;
}
CUresult CudaOperationQueue::pump_selected(
    std::optional<std::span<const CudaOperationToken>> admitted) noexcept {
    auto& owner = *implementation_;
    if (owner.is_closed) {
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    }
    if (owner.fault != CUDA_SUCCESS) {
        return owner.fault;
    }
    if (admitted) {
        if (admitted->size() > owner.pending_limit) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (std::size_t index = 0; index < admitted->size(); ++index) {
            const auto* entry = owner.find_entry((*admitted)[index]);
            if (!entry || !std::holds_alternative<CudaQueuedKernel>(entry->operation) ||
                !owner.order.is_ready(entry->token.result_->operation) ||
                std::ranges::find(admitted->first(index), entry->token) !=
                    admitted->first(index).end()) {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    }
    const auto context_status = owner.check_context();
    if (context_status != CUDA_SUCCESS) {
        return context_status;
    }
    for (auto& entry : owner.entries) {
        if (!entry->execution) {
            continue;
        }
        auto status = owner.dispatch.event_query(entry->event);
        if (status == CUDA_ERROR_NOT_READY) {
            if (auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
                kernel && kernel->continuation)
                kernel->continuation->observe_pending();
            continue;
        }
        if (status != CUDA_SUCCESS) {
            return owner.poison(*entry, status);
        }
        entry->is_physically_stopped = true;
        // This private event follows a successfully issued native stream wait.
        status = owner.confirm_native_prefix(*entry);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        status = owner.confirm_release(*entry);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        status = owner.destroy_event(*entry);
        if (status != CUDA_SUCCESS) {
            return owner.poison(*entry, status);
        }
        bool is_complete = true;
        if (auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
            kernel && kernel->continuation) {
            status = kernel->continuation->inspect_stopped(is_complete);
        }
        entry->token.result_->error = status;
        const bool transitioned =
            status == CUDA_SUCCESS && !is_complete
                ? owner.order.suspend(*entry->execution)
                : owner.order.complete(*entry->execution, status == CUDA_SUCCESS);
        if (!transitioned) {
            return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
        }
        entry->execution.reset();
    }
    owner.collect();
    for (auto& entry : owner.entries) {
        if (std::holds_alternative<CudaQueuedNativeReservation>(entry->operation)) {
            continue;
        }
        if (!owner.order.is_ready(entry->token.result_->operation)) {
            continue;
        }
        if (admitted && std::holds_alternative<CudaQueuedKernel>(entry->operation) &&
            std::ranges::find(*admitted, entry->token) == admitted->end()) {
            continue;
        }
        auto status = owner.dispatch.event_create(&entry->event, CU_EVENT_DISABLE_TIMING);
        const auto has_invalid_success = status == CUDA_SUCCESS && entry->event == nullptr;
        if (has_invalid_success) {
            status = CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS) {
            entry->has_ambiguous_event_cleanup = entry->event != nullptr;
            entry->token.result_->error = status;
            if (!owner.order.fail_stopped(entry->token.result_->operation)) {
                return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
            }
            if (!entry->has_native_prefix_proof || entry->has_ambiguous_event_cleanup ||
                has_invalid_success) {
                // The imported prefix was issued BEFORE queue submission.
                return owner.poison(*entry, status);
            }
            continue;
        }
        entry->execution = owner.order.start(entry->token.result_->operation);
        if (!entry->execution) {
            return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
        }
        // Even an issue error may be asynchronous. Retain physical ownership
        // until shutdown obtains a native stream-drain proof; never retry.
        entry->is_physically_stopped = false;
        status = owner.issue(*entry);
        if (status == CUDA_SUCCESS) {
            status = owner.dispatch.event_record(entry->event, entry->stream);
        }
        if (status != CUDA_SUCCESS) {
            return owner.poison(*entry, status);
        }
    }
    owner.collect();
    return CUDA_SUCCESS;
}
bool CudaOperationQueue::native_fence_is_ready(const CudaOperationToken& operation) const noexcept {
    const auto& owner = *implementation_;
    if (owner.is_closed || owner.fault != CUDA_SUCCESS) {
        return false;
    }
    const auto found = std::ranges::find_if(
        owner.entries, [&](const auto& entry) { return entry->token == operation; });
    return found != owner.entries.end() &&
           std::holds_alternative<CudaQueuedNativeFence>((*found)->operation) &&
           owner.order.is_ready(operation.result_->operation);
}
bool CudaOperationQueue::native_reservation_is_ready(
    const CudaOperationToken& operation) const noexcept {
    return is_native_reservation(operation) &&
           implementation_->order.is_ready(operation.result_->operation);
}
bool CudaOperationQueue::is_native_reservation(const CudaOperationToken& operation) const noexcept {
    const auto& owner = *implementation_;
    const auto found = std::ranges::find_if(
        owner.entries, [&](const auto& entry) { return entry->token == operation; });
    return !owner.is_closed && owner.fault == CUDA_SUCCESS && found != owner.entries.end() &&
           std::holds_alternative<CudaQueuedNativeReservation>((*found)->operation);
}
CUresult CudaOperationQueue::query_native_reservation_boundary(
    const CudaOperationToken& operation) const noexcept {
    const auto& owner = *implementation_;
    if (owner.is_closed) {
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    }
    if (owner.fault != CUDA_SUCCESS) {
        return owner.fault;
    }
    if (!native_reservation_is_ready(operation)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return std::ranges::any_of(owner.entries,
                               [](const auto& entry) {
                                   return !entry->is_physically_stopped ||
                                          !entry->has_native_prefix_proof;
                               })
               ? CUDA_ERROR_NOT_READY
               : CUDA_SUCCESS;
}
bool CudaOperationQueue::replace_native_reservation(const CudaOperationToken& operation,
                                                    CudaQueuedNativeFence fence) noexcept {
    auto& owner = *implementation_;
    if (query_native_reservation_boundary(operation) != CUDA_SUCCESS ||
        !owner.validates(CudaQueuedOperation{fence})) {
        return false;
    }
    auto* entry = owner.find_entry(operation);
    auto* reservation = std::get_if<CudaQueuedNativeReservation>(&entry->operation);
    if (!reservation) {
        return false;
    }
    static_assert(std::is_nothrow_constructible_v<CudaQueuedOperation, CudaQueuedNativeFence> &&
                  std::is_nothrow_swappable_v<CudaQueuedOperation>);
    try {
        CudaQueuedOperation replacement{std::move(fence)};
        entry->native_reservation_resources = std::move(reservation->resources);
        entry->operation.swap(replacement);
    } catch (...) {
        // The checked alternatives and nothrow operations above cannot throw;
        // still contain the standard library's variant exception boundary.
        return false;
    }
    entry->has_native_prefix_proof = false;
    return true;
}
std::optional<OrderedOperationState> CudaOperationQueue::state(
    const CudaOperationToken& operation) const noexcept {
    return implementation_->order.state(operation ? operation.result_->operation
                                                  : OrderedOperationToken{});
}
CUresult CudaOperationQueue::query(const CudaOperationToken& operation) const noexcept {
    const auto current = state(operation);
    if (!current) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (operation && operation.result_->error != CUDA_SUCCESS) {
        return operation.result_->error;
    }
    if (*current == OrderedOperationState::kSucceeded) {
        return CUDA_SUCCESS;
    }
    return *current == OrderedOperationState::kFailed ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_NOT_READY;
}
void CudaOperationQueue::execution_granted(const CudaOperationToken& operation,
                                           CudaQueueExecutionIdentity identity) noexcept {
    auto* entry = implementation_->find_entry(operation);
    if (!entry) return;
    if (auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
        kernel && kernel->continuation)
        kernel->continuation->execution_granted(identity);
}
CUresult CudaOperationQueue::request_retreat(const CudaOperationToken& operation,
                                             bool is_priority) noexcept {
    auto* entry = implementation_->find_entry(operation);
    if (!entry || state(operation) != OrderedOperationState::kRunning) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
    if (is_priority && kernel && kernel->continuation)
        kernel->continuation->observe_priority_retreat();
    return kernel && kernel->continuation ? kernel->continuation->request_retreat()
                                          : CUDA_ERROR_NOT_SUPPORTED;
}
bool CudaOperationQueue::resume(const CudaOperationToken& operation) noexcept {
    auto& owner = *implementation_;
    return !owner.is_closed && owner.fault == CUDA_SUCCESS && operation &&
           owner.order.resume(operation.result_->operation);
}
bool CudaOperationQueue::cancel_stopped(const CudaOperationToken& operation) noexcept {
    auto& owner = *implementation_;
    const auto* entry = owner.find_entry(operation);
    if (entry && !entry->has_native_prefix_proof) {
        return false;
    }
    if (!operation || !owner.order.fail_stopped(operation.result_->operation)) {
        return false;
    }
    operation.result_->error = CUDA_ERROR_UNKNOWN;
    owner.collect();
    return true;
}
std::optional<CudaOperationToken> CudaOperationQueue::stream_snapshot(
    const OrderedStreamToken& stream) const noexcept {
    const auto* binding = implementation_->find_stream(stream);
    return binding && binding->is_open ? std::optional(binding->tail) : std::nullopt;
}
std::vector<CudaOperationToken> CudaOperationQueue::context_snapshot() const {
    std::vector<CudaOperationToken> snapshot;
    snapshot.reserve(implementation_->streams.size());
    for (const auto& stream : implementation_->streams) {
        if (stream.tail) {
            snapshot.push_back(stream.tail);
        }
    }
    return snapshot;
}
std::vector<CudaOperationToken> CudaOperationQueue::implicit_completion_snapshot(
    bool is_legacy) const {
    std::vector<CudaOperationToken> snapshot;
    snapshot.reserve(implementation_->streams.size());
    for (const auto& stream : implementation_->streams) {
        if (stream.tail && (stream.kind == OrderedStreamKind::kLegacyDefault ||
                            (is_legacy && stream.kind != OrderedStreamKind::kNonBlocking))) {
            snapshot.push_back(stream.tail);
        }
    }
    return snapshot;
}
std::optional<std::vector<CudaOperationToken>> CudaOperationQueue::completion_snapshot(
    const OrderedStreamToken& stream) const {
    const auto* binding = implementation_->find_stream(stream);
    if (!binding || !binding->is_open) {
        return std::nullopt;
    }
    auto snapshot =
        binding->kind == OrderedStreamKind::kNonBlocking
            ? std::vector<CudaOperationToken>{}
            : implicit_completion_snapshot(binding->kind == OrderedStreamKind::kLegacyDefault);
    if (binding->tail && std::ranges::find(snapshot, binding->tail) == snapshot.end()) {
        snapshot.push_back(binding->tail);
    }
    return snapshot;
}

CUresult CudaOperationQueue::query_native_boundary(
    std::span<const CudaOperationToken> snapshot) const noexcept {
    const auto& owner = *implementation_;
    if (owner.is_closed) {
        return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    }
    if (owner.fault != CUDA_SUCCESS) {
        return owner.fault;
    }
    const auto context_status = owner.check_context();
    if (context_status != CUDA_SUCCESS) {
        return context_status;
    }
    bool is_pending = false;
    for (const auto& token : snapshot) {
        if (!token) {
            continue;
        }
        const auto status = query(token);
        if (status == CUDA_ERROR_NOT_READY) {
            is_pending = true;
        } else if (status != CUDA_SUCCESS) {
            return status;
        }
    }
    return is_pending ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}
std::size_t CudaOperationQueue::pending_count() const noexcept {
    return implementation_->entries.size();
}

CUresult CudaOperationQueue::shutdown() noexcept {
    auto& owner = *implementation_;
    if (owner.is_closed) {
        return CUDA_SUCCESS;
    }
    auto status = owner.check_context();
    if (status != CUDA_SUCCESS) {
        return status;
    }
    for (auto& entry : owner.entries) {
        if (entry->execution && !entry->is_physically_stopped) {
            if (auto* kernel = std::get_if<CudaQueuedKernel>(&entry->operation);
                kernel && kernel->continuation) {
                status = kernel->continuation->request_retreat();
                if (status != CUDA_SUCCESS) {
                    return owner.poison(*entry, status);
                }
            }
        }
    }
    for (auto& entry : owner.entries) {
        if (!entry->has_native_prefix_proof) {
            const auto* fence = std::get_if<CudaQueuedNativeFence>(&entry->operation);
            if (!fence || !owner.dispatch.event_synchronize) {
                return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
            }
            status = owner.dispatch.event_synchronize(fence->event);
            if (status != CUDA_SUCCESS) {
                return owner.poison(*entry, status);
            }
            status = owner.confirm_native_prefix(*entry);
            if (status != CUDA_SUCCESS) {
                return status;
            }
        }
        if (entry->execution) {
            if (!entry->is_physically_stopped) {
                status = owner.dispatch.stream_synchronize(entry->stream);
                if (status != CUDA_SUCCESS) {
                    return owner.poison(*entry, status);
                }
                entry->is_physically_stopped = true;
            }
            status = owner.confirm_release(*entry);
            if (status != CUDA_SUCCESS) {
                return status;
            }
            if (entry->has_ambiguous_free) {
                // Stop does not tell us whether a failed native free actually
                // released its allocation. Do not retry it through RAII cleanup.
                // Keep the owner until an explicit context recovery boundary.
                return owner.poison(*entry, entry->token.result_->error);
            }
            if (!owner.order.complete(*entry->execution, false)) {
                return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
            }
            entry->execution.reset();
        } else if (const auto current = state(entry->token);
                   current == OrderedOperationState::kQueued ||
                   current == OrderedOperationState::kSuspended) {
            if (!owner.order.fail_stopped(entry->token.result_->operation)) {
                return owner.poison(*entry, CUDA_ERROR_UNKNOWN);
            }
        }
        status = owner.destroy_event(*entry);
        if (status != CUDA_SUCCESS) {
            return owner.poison(*entry, status);
        }
    }
    owner.entries.clear();
    owner.streams.clear();
    owner.events.clear();
    owner.is_closed = true;
    return CUDA_SUCCESS;
}
void CudaOperationQueue::retire_after_context_destroy() noexcept {
    auto& owner = *implementation_;
    for (auto& entry : owner.entries) {
        if (entry->token.result_->error == CUDA_SUCCESS) {
            entry->token.result_->error = CUDA_ERROR_CONTEXT_IS_DESTROYED;
        }
        if (const auto* free = std::get_if<CudaQueuedFree>(&entry->operation);
            free && (entry->has_free_submission || entry->has_ambiguous_free)) {
            free->release_observer->context_destroyed(free->pointer);
        }
        if (entry->execution) {
            (void)owner.order.complete(*entry->execution, false);
        } else {
            (void)owner.order.fail_stopped(entry->token.result_->operation);
        }
    }
    owner.entries.clear();
    owner.streams.clear();
    owner.events.clear();
    owner.is_closed = true;
}

}  // namespace glimmer::interceptor
