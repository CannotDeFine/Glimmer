#include "internal/operation_order.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace glimmer::interceptor {

struct OperationOrderIdentity {};
struct OrderedOperation {
    std::shared_ptr<OperationOrderIdentity> owner;
    std::vector<std::shared_ptr<OrderedOperation>> dependencies;
    OrderedOperationState state{OrderedOperationState::kQueued};
    std::uint64_t generation{};
    bool can_suspend{};
};
struct OrderedStream {
    std::shared_ptr<OperationOrderIdentity> owner;
    OrderedStreamKind kind{};
    std::shared_ptr<OrderedOperation> tail;
    bool is_open{true};
};

namespace {
bool is_terminal(const OrderedOperation& operation) noexcept {
    return operation.state == OrderedOperationState::kSucceeded ||
           operation.state == OrderedOperationState::kFailed;
}
void append_dependency(std::vector<std::shared_ptr<OrderedOperation>>& dependencies,
                       const std::shared_ptr<OrderedOperation>& operation) {
    if (operation && std::ranges::find(dependencies, operation) == dependencies.end()) {
        dependencies.push_back(operation);
    }
}
}  // namespace

OrderedStreamToken::OrderedStreamToken(std::shared_ptr<OrderedStream> stream) noexcept
    : stream_(std::move(stream)) {}
OrderedOperationToken::OrderedOperationToken(std::shared_ptr<OrderedOperation> operation) noexcept
    : operation_(std::move(operation)) {}
OperationOrder::OperationOrder(OperationOrderLimits limits)
    : pending_limit(limits.pending_operations),
      stream_limit(limits.streams),
      identity_(std::make_shared<OperationOrderIdentity>()) {}
OperationOrder::~OperationOrder() = default;

bool OperationOrder::owns(const OrderedStreamToken& stream) const noexcept {
    return stream.stream_ && stream.stream_->owner == identity_;
}
bool OperationOrder::owns(const OrderedOperationToken& operation) const noexcept {
    return operation.operation_ && operation.operation_->owner == identity_;
}
OrderedStreamToken OperationOrder::create_stream(OrderedStreamKind kind) noexcept {
    switch (kind) {
        case OrderedStreamKind::kLegacyDefault:
        case OrderedStreamKind::kBlocking:
        case OrderedStreamKind::kNonBlocking:
        case OrderedStreamKind::kPerThreadDefault:
            break;
        default:
            return {};
    }
    if (streams_.size() >= stream_limit || (kind == OrderedStreamKind::kLegacyDefault &&
                                            std::ranges::any_of(streams_, [](const auto& stream) {
                                                return stream->kind ==
                                                       OrderedStreamKind::kLegacyDefault;
                                            }))) {
        return {};
    }
    try {
        auto stream = std::make_shared<OrderedStream>();
        stream->owner = identity_;
        stream->kind = kind;
        streams_.push_back(stream);
        return OrderedStreamToken(std::move(stream));
    } catch (...) {
        return {};
    }
}
bool OperationOrder::close_stream(const OrderedStreamToken& stream) noexcept {
    if (!owns(stream) || !stream.stream_->is_open) {
        return false;
    }
    stream.stream_->is_open = false;
    collect();
    return true;
}
OrderedOperationToken OperationOrder::submit(const OrderedStreamToken& stream,
                                             std::span<const OrderedOperationToken> dependencies,
                                             bool can_suspend) noexcept {
    if (!owns(stream) || !stream.stream_->is_open || pending_.size() >= pending_limit ||
        dependencies.size() > pending_limit) {
        return {};
    }
    for (const auto& dependency : dependencies) {
        if (dependency && !owns(dependency)) {
            return {};
        }
    }
    try {
        auto operation = std::make_shared<OrderedOperation>();
        operation->owner = identity_;
        operation->can_suspend = can_suspend;
        append_dependency(operation->dependencies, stream.stream_->tail);
        for (const auto& dependency : dependencies) {
            append_dependency(operation->dependencies, dependency.operation_);
        }
        for (const auto& other : streams_) {
            const bool is_legacy_submission =
                stream.stream_->kind == OrderedStreamKind::kLegacyDefault;
            if ((is_legacy_submission && other->kind != OrderedStreamKind::kNonBlocking) ||
                (stream.stream_->kind != OrderedStreamKind::kNonBlocking &&
                 other->kind == OrderedStreamKind::kLegacyDefault)) {
                append_dependency(operation->dependencies, other->tail);
            }
        }
        // All potentially throwing preparation precedes tail publication.
        pending_.push_back(operation);
        stream.stream_->tail = operation;
        collect();
        return OrderedOperationToken(std::move(operation));
    } catch (...) {
        return {};
    }
}
bool OperationOrder::is_ready(const OrderedOperationToken& operation) const noexcept {
    return owns(operation) && operation.operation_->state == OrderedOperationState::kQueued &&
           operation.operation_->generation != std::numeric_limits<std::uint64_t>::max() &&
           std::ranges::all_of(operation.operation_->dependencies, [](const auto& dependency) {
               return dependency->state == OrderedOperationState::kSucceeded;
           });
}
std::optional<PhysicalExecutionTicket> OperationOrder::start(
    const OrderedOperationToken& operation) noexcept {
    if (!is_ready(operation)) {
        return std::nullopt;
    }
    auto& node = *operation.operation_;
    ++node.generation;
    node.state = OrderedOperationState::kRunning;
    return PhysicalExecutionTicket{operation, node.generation};
}
bool OperationOrder::matches_running(const PhysicalExecutionTicket& execution) const noexcept {
    return owns(execution.operation) && execution.generation != 0 &&
           execution.operation.operation_->state == OrderedOperationState::kRunning &&
           execution.operation.operation_->generation == execution.generation;
}
bool OperationOrder::suspend(const PhysicalExecutionTicket& execution) noexcept {
    if (!matches_running(execution) || !execution.operation.operation_->can_suspend ||
        execution.generation == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    execution.operation.operation_->state = OrderedOperationState::kSuspended;
    return true;
}
bool OperationOrder::resume(const OrderedOperationToken& operation) noexcept {
    if (!owns(operation) || operation.operation_->state != OrderedOperationState::kSuspended ||
        operation.operation_->generation == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    operation.operation_->state = OrderedOperationState::kQueued;
    return true;
}
bool OperationOrder::complete(const PhysicalExecutionTicket& execution, bool succeeded) noexcept {
    if (!matches_running(execution)) {
        return false;
    }
    execution.operation.operation_->state =
        succeeded ? OrderedOperationState::kSucceeded : OrderedOperationState::kFailed;
    collect();
    return true;
}
bool OperationOrder::fail_stopped(const OrderedOperationToken& operation) noexcept {
    if (!owns(operation) || is_terminal(*operation.operation_) ||
        operation.operation_->state == OrderedOperationState::kRunning) {
        return false;
    }
    operation.operation_->state = OrderedOperationState::kFailed;
    collect();
    return true;
}
std::optional<OrderedOperationState> OperationOrder::state(
    const OrderedOperationToken& operation) const noexcept {
    if (!operation) {
        return OrderedOperationState::kSucceeded;
    }
    return owns(operation) ? std::optional(operation.operation_->state) : std::nullopt;
}
std::optional<OrderedOperationToken> OperationOrder::stream_snapshot(
    const OrderedStreamToken& stream) const noexcept {
    if (!owns(stream) || !stream.stream_->is_open) {
        return std::nullopt;
    }
    return OrderedOperationToken(stream.stream_->tail);
}
std::vector<OrderedOperationToken> OperationOrder::context_snapshot() const {
    std::vector<OrderedOperationToken> snapshot;
    snapshot.reserve(streams_.size());
    for (const auto& stream : streams_) {
        if (stream->tail) {
            snapshot.push_back(OrderedOperationToken(stream->tail));
        }
    }
    return snapshot;
}
void OperationOrder::collect() noexcept {
    // Edges point only to earlier submissions, so failure propagation needs
    // one insertion-order pass. Terminal snapshots retain state, not history.
    for (const auto& operation : pending_) {
        if (operation->state == OrderedOperationState::kQueued &&
            std::ranges::any_of(operation->dependencies, [](const auto& dependency) {
                return dependency->state == OrderedOperationState::kFailed;
            })) {
            operation->state = OrderedOperationState::kFailed;
        }
        if (is_terminal(*operation)) {
            operation->dependencies.clear();
        }
    }
    std::erase_if(pending_, [](const auto& operation) { return is_terminal(*operation); });
    std::erase_if(streams_, [](const auto& stream) {
        // A closed stream's asynchronous error must remain visible to a later
        // context/default-stream boundary, not disappear with its handle.
        return !stream->is_open &&
               (!stream->tail || stream->tail->state == OrderedOperationState::kSucceeded);
    });
}

}  // namespace glimmer::interceptor
