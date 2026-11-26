#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::interceptor {

struct OperationOrderIdentity;
struct OrderedStream;
struct OrderedOperation;

// Preserve the existing enum representation used by the execution components.
// NOLINTBEGIN(performance-enum-size)
enum class OrderedStreamKind : std::uint8_t {
    kLegacyDefault,
    kBlocking,
    kNonBlocking,
    kPerThreadDefault
};
enum class OrderedOperationState : std::uint8_t {
    kQueued,
    kRunning,
    kSuspended,
    kSucceeded,
    kFailed
};
// NOLINTEND(performance-enum-size)

class OrderedStreamToken {
   public:
    OrderedStreamToken() = default;
    [[nodiscard]] explicit operator bool() const noexcept {
        return stream_ != nullptr;
    }
    bool operator==(const OrderedStreamToken&) const = default;

   private:
    friend class OperationOrder;
    explicit OrderedStreamToken(std::shared_ptr<OrderedStream> stream) noexcept;
    std::shared_ptr<OrderedStream> stream_;
};

// A record/synchronization snapshot retains this token, not a mutable event
// handle or stream tail. An empty token denotes no preceding work.
class OrderedOperationToken {
   public:
    OrderedOperationToken() = default;
    [[nodiscard]] explicit operator bool() const noexcept {
        return operation_ != nullptr;
    }
    bool operator==(const OrderedOperationToken&) const = default;

   private:
    friend class OperationOrder;
    explicit OrderedOperationToken(std::shared_ptr<OrderedOperation> operation) noexcept;
    std::shared_ptr<OrderedOperation> operation_;
};

struct PhysicalExecutionTicket {
    OrderedOperationToken operation;
    std::uint64_t generation{};
};

struct OperationOrderLimits {
    std::size_t pending_operations{1024};
    std::size_t streams{256};
};

// Private dependency model for one context. All calls, including snapshot
// queries, must be externally serialized by one owner. No callbacks or CUDA
// calls occur here. The backend must prove physical stop before suspend/fail.
class OperationOrder {
   public:
    explicit OperationOrder(OperationOrderLimits limits = {});
    OperationOrder(const OperationOrder&) = delete;
    OperationOrder& operator=(const OperationOrder&) = delete;
    OperationOrder(OperationOrder&&) = delete;
    OperationOrder& operator=(OperationOrder&&) = delete;
    ~OperationOrder();

    [[nodiscard]] OrderedStreamToken create_stream(OrderedStreamKind kind) noexcept;
    [[nodiscard]] bool close_stream(const OrderedStreamToken& stream) noexcept;
    [[nodiscard]] OrderedOperationToken submit(
        const OrderedStreamToken& stream, std::span<const OrderedOperationToken> dependencies = {},
        bool can_suspend = false) noexcept;
    [[nodiscard]] bool is_ready(const OrderedOperationToken& operation) const noexcept;
    [[nodiscard]] std::optional<PhysicalExecutionTicket> start(
        const OrderedOperationToken& operation) noexcept;
    [[nodiscard]] bool suspend(const PhysicalExecutionTicket& execution) noexcept;
    [[nodiscard]] bool resume(const OrderedOperationToken& operation) noexcept;
    [[nodiscard]] bool complete(const PhysicalExecutionTicket& execution, bool succeeded) noexcept;
    [[nodiscard]] bool fail_stopped(const OrderedOperationToken& operation) noexcept;
    [[nodiscard]] std::optional<OrderedOperationState> state(
        const OrderedOperationToken& operation) const noexcept;
    [[nodiscard]] std::optional<OrderedOperationToken> stream_snapshot(
        const OrderedStreamToken& stream) const noexcept;
    [[nodiscard]] std::vector<OrderedOperationToken> context_snapshot() const;
    [[nodiscard]] std::size_t pending_count() const noexcept {
        return pending_.size();
    }

   private:
    [[nodiscard]] bool owns(const OrderedStreamToken& stream) const noexcept;
    [[nodiscard]] bool owns(const OrderedOperationToken& operation) const noexcept;
    [[nodiscard]] bool matches_running(const PhysicalExecutionTicket& execution) const noexcept;
    void collect() noexcept;

    const std::size_t pending_limit;
    const std::size_t stream_limit;
    std::shared_ptr<OperationOrderIdentity> identity_;
    std::vector<std::shared_ptr<OrderedStream>> streams_;
    std::vector<std::shared_ptr<OrderedOperation>> pending_;
};

}  // namespace glimmer::interceptor
