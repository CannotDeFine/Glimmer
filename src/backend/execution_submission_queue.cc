#include "internal/execution_submission_queue.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::backend {
namespace {
// Only bind_executor initializes this identity. Its weak observers cannot
// mistake a recycled std::thread::id for the lifetime of the original owner.
thread_local std::shared_ptr<const void> executor_lifetime;
}  // namespace

struct ExecutionHostStorage::Counter {
    std::atomic<std::size_t> bytes{};
};
struct ExecutionHostStorage::Implementation {
    struct Charge {
        std::shared_ptr<Counter> counter;
        std::size_t bytes{};
        ~Charge() {
            if (bytes) counter->bytes.fetch_sub(bytes);
        }
    } charge;  // Destroy last, after the resource and its backing allocation.
    const std::size_t capacity;
    std::unique_ptr<std::byte[]> buffer;
    std::pmr::monotonic_buffer_resource arena;
    Implementation(std::shared_ptr<Counter> counter, std::size_t bytes)
        : charge{std::move(counter), 0},
          capacity(bytes),
          buffer(std::make_unique_for_overwrite<std::byte[]>(bytes)),
          arena(buffer.get(), bytes, std::pmr::null_memory_resource()) {
        charge.counter->bytes.fetch_add(bytes);
        charge.bytes = bytes;
    }
};
ExecutionHostStorage::ExecutionHostStorage(std::shared_ptr<Counter> counter, std::size_t bytes)
    : implementation_(std::make_unique<Implementation>(std::move(counter), bytes)) {}
ExecutionHostStorage::~ExecutionHostStorage() = default;
std::pmr::memory_resource* ExecutionHostStorage::resource() noexcept {
    return &implementation_->arena;
}
std::size_t ExecutionHostStorage::bytes() const noexcept {
    return implementation_->capacity;
}

struct ExecutionSubmissionQueue::Implementation {
    enum class Phase : std::uint8_t { kQueued, kIssuing, kHostReturned, kPaused, kResuming };
    struct Payload {
        std::vector<std::byte> parameters;
        std::shared_ptr<const void> lifetime;
    };
    struct Entry {
        std::uint64_t sequence{};
        std::uint64_t generation{};
        std::uint32_t operation{};
        std::uint64_t context{};
        std::uint64_t stream{};
        Phase phase{Phase::kQueued};
        std::optional<ExecutionHostReply> reply;
        std::unique_ptr<Payload> payload;
        std::size_t issue_parameter_bytes{};
        bool is_acknowledged{};
        bool is_physically_stopped{};
        bool is_releasing_payload{};
    };

    ExecutionSessionIdentity session;
    ExecutionSubmissionLimits limits;
    const pid_t process_id{getpid()};
    const std::shared_ptr<const void> identity{std::make_shared<int>(0)};
    mutable std::mutex mutex;
    std::vector<std::shared_ptr<Entry>> entries;
    std::weak_ptr<const void> executor;
    std::uint64_t next_sequence;
    std::size_t parameter_bytes{};
    const std::shared_ptr<ExecutionHostStorage::Counter> host_storage{
        std::make_shared<ExecutionHostStorage::Counter>()};
    bool has_sequence_space{true};
    bool has_bound_executor{};
    bool is_closing{};
    bool is_quarantined{};

    Implementation(ExecutionSessionIdentity session_value, ExecutionSubmissionLimits limits_value)
        : session(session_value), limits(limits_value), next_sequence(session.first_sequence) {
        if (session.client_id == 0 || session.server_incarnation == 0 ||
            session.first_sequence == 0 || limits.max_commands == 0 ||
            limits.max_parameter_bytes == 0 || limits.max_retained_host_bytes == 0)
            throw std::invalid_argument("invalid execution submission domain or limits");
        entries.reserve(limits.max_commands);
    }

    [[nodiscard]] bool is_executor() const noexcept {
        return executor_lifetime && executor.lock() == executor_lifetime;
    }

    [[nodiscard]] bool is_same_process() const noexcept {
        return getpid() == process_id;
    }

    [[nodiscard]] std::shared_ptr<Entry> find(const std::weak_ptr<const void>& owner,
                                              std::uint64_t sequence) const noexcept {
        if (owner.lock() != identity || sequence == 0) return {};
        const auto found = std::ranges::find_if(
            entries, [sequence](const auto& entry) { return entry->sequence == sequence; });
        return found == entries.end() ? nullptr : *found;
    }

    void collect(const std::shared_ptr<Entry>& entry) noexcept {
        if (!entry->is_acknowledged || !entry->is_physically_stopped || entry->is_releasing_payload)
            return;
        const auto found = std::ranges::find(entries, entry);
        if (found != entries.end()) entries.erase(found);
    }

    [[nodiscard]] std::shared_ptr<Entry> find_issue(
        const ExecutionIssueToken& token) const noexcept {
        const auto entry = find(token.owner_, token.sequence_);
        return entry && entry->generation != 0 && entry->generation == token.generation_ ? entry
                                                                                         : nullptr;
    }

    [[nodiscard]] bool is_issuing() const noexcept {
        return std::ranges::any_of(entries, [](const auto& entry) {
            return entry->phase == Phase::kIssuing || entry->phase == Phase::kResuming;
        });
    }

    [[nodiscard]] ExecutionIssue issue(const Entry& entry) const noexcept {
        ExecutionIssue result;
        result.receipt.owner_ = identity;
        result.receipt.sequence_ = entry.sequence;
        result.token.owner_ = identity;
        result.token.sequence_ = entry.sequence;
        result.token.generation_ = entry.generation;
        result.operation = entry.operation;
        result.context = entry.context;
        result.stream = entry.stream;
        result.parameters = std::span{entry.payload->parameters}.first(entry.issue_parameter_bytes);
        return result;
    }

    // The caller marks retirement under the mutex before invoking this helper.
    // Keep bytes charged until all outside-lock destruction has actually ended.
    void release_payload(const std::shared_ptr<Entry>& entry) noexcept {
        std::unique_ptr<Payload> released;
        std::size_t bytes{};
        {
            const std::lock_guard lock(mutex);
            assert(entry->is_releasing_payload && entry->payload);
            bytes = entry->payload->parameters.size();
            released = std::move(entry->payload);
        }
        released.reset();
        {
            const std::lock_guard lock(mutex);
            assert(parameter_bytes >= bytes);
            parameter_bytes -= bytes;
            entry->is_releasing_payload = false;
            collect(entry);
        }
    }
};

ExecutionReceipt::operator bool() const noexcept {
    return sequence_ != 0 && !owner_.expired();
}
std::uint64_t ExecutionReceipt::sequence() const noexcept {
    return sequence_;
}
ExecutionIssueToken::operator bool() const noexcept {
    return sequence_ != 0 && generation_ != 0 && !owner_.expired();
}
std::uint64_t ExecutionIssueToken::sequence() const noexcept {
    return sequence_;
}
std::uint64_t ExecutionIssueToken::generation() const noexcept {
    return generation_;
}
bool ExecutionAcceptance::accepted() const noexcept {
    return status == ExecutionAcceptStatus::kAccepted && static_cast<bool>(receipt);
}

ExecutionSubmissionQueue::ExecutionSubmissionQueue(ExecutionSessionIdentity identity,
                                                   ExecutionSubmissionLimits limits)
    : implementation_(std::make_unique<Implementation>(identity, limits)) {}

ExecutionSubmissionQueue::~ExecutionSubmissionQueue() {
    if (!implementation_->is_same_process()) {
        // Never lock inherited state or release parent-owned leases after fork.
        (void)implementation_.release();
        return;
    }
    if (!shutdown()) {
        const auto* retained = implementation_.release();
        try {
            spdlog::critical("Retaining unfinished execution queue: client={}, incarnation={}",
                             retained->session.client_id, retained->session.server_incarnation);
        } catch (...) {
            // Diagnostic failure cannot release potentially live GPU inputs.
        }
    }
}

bool ExecutionSubmissionQueue::bind_executor() noexcept {
    if (!implementation_->is_same_process()) return false;
    try {
        if (!executor_lifetime) executor_lifetime = std::make_shared<int>(0);
        auto& state = *implementation_;
        const std::lock_guard lock(state.mutex);
        if (state.is_closing && state.entries.empty()) return false;
        if (state.has_bound_executor) {
            if (state.executor.expired()) {
                state.is_quarantined = true;
                state.is_closing = true;
            }
            return state.is_executor();
        }
        state.executor = executor_lifetime;
        state.has_bound_executor = true;
        return true;
    } catch (...) {
        return false;
    }
}

bool ExecutionSubmissionQueue::owns_executor() const noexcept {
    if (!implementation_->is_same_process()) return false;
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    return state.is_executor();
}

bool ExecutionSubmissionQueue::executor_is_quarantined() const noexcept {
    if (!implementation_->is_same_process()) return true;
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    return !state.is_executor() || state.is_quarantined;
}

ExecutionHostAllocation ExecutionSubmissionQueue::allocate_host_storage(
    std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (!state.is_same_process()) return {ExecutionAcceptStatus::kWrongProcess, {}};
    const std::lock_guard lock(state.mutex);
    if (!state.is_executor() || !bytes) return {};
    if (state.is_quarantined) return {ExecutionAcceptStatus::kQuarantined, {}};
    if (state.is_closing) return {ExecutionAcceptStatus::kClosed, {}};
    if (bytes > state.limits.max_retained_host_bytes)
        return {ExecutionAcceptStatus::kOutOfMemory, {}};
    if (bytes > state.limits.max_retained_host_bytes - state.host_storage->bytes.load())
        return {ExecutionAcceptStatus::kFull, {}};
    try {
        return {ExecutionAcceptStatus::kAccepted,
                std::shared_ptr<ExecutionHostStorage>(
                    new ExecutionHostStorage(state.host_storage, bytes))};
    } catch (...) {
        return {ExecutionAcceptStatus::kOutOfMemory, {}};
    }
}

ExecutionAcceptance ExecutionSubmissionQueue::accept(std::uint64_t sequence,
                                                     const ExecutionRequest& request) noexcept {
    if (!implementation_->is_same_process())
        return {.status = ExecutionAcceptStatus::kWrongProcess, .receipt = {}};
    const auto capacity = request.parameter_capacity_bytes ? request.parameter_capacity_bytes
                                                           : request.parameters.size();
    if (sequence == 0 || request.operation == 0 || request.context == 0 || !request.lifetime ||
        capacity < request.parameters.size() ||
        (!request.parameters.empty() && !request.parameters.data()))
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    try {
        auto& state = *implementation_;
        const std::lock_guard lock(state.mutex);
        if (state.has_bound_executor && state.executor.expired()) {
            state.is_quarantined = true;
            state.is_closing = true;
        }
        if (state.is_quarantined)
            return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
        if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
        if (!state.has_sequence_space || sequence != state.next_sequence)
            return {.status = ExecutionAcceptStatus::kWrongSequence, .receipt = {}};
        if (capacity > state.limits.max_parameter_bytes)
            return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
        if (state.entries.size() >= state.limits.max_commands ||
            capacity > state.limits.max_parameter_bytes - state.parameter_bytes)
            return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};

        auto entry = std::make_shared<Implementation::Entry>();
        entry->sequence = sequence;
        entry->operation = request.operation;
        entry->context = request.context;
        entry->stream = request.stream;
        entry->payload = std::make_unique<Implementation::Payload>();
        entry->payload->parameters.resize(capacity);
        if (!request.parameters.empty())
            std::ranges::copy(request.parameters, entry->payload->parameters.begin());
        entry->payload->lifetime = request.lifetime;
        entry->issue_parameter_bytes = request.parameters.size();
        // reserve() bounds the vector at construction. Caller request ownership
        // also prevents rollback from running a last lifetime callback here.
        state.entries.push_back(std::move(entry));
        state.parameter_bytes += capacity;
        if (sequence == std::numeric_limits<std::uint64_t>::max())
            state.has_sequence_space = false;
        else
            ++state.next_sequence;
        ExecutionReceipt receipt;
        receipt.owner_ = state.identity;
        receipt.sequence_ = sequence;
        return {.status = ExecutionAcceptStatus::kAccepted, .receipt = std::move(receipt)};
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

std::optional<ExecutionIssue> ExecutionSubmissionQueue::take_next() noexcept {
    return take_next_impl(nullptr);
}
std::optional<ExecutionIssue> ExecutionSubmissionQueue::take_next(
    const ExecutionReceipt& receipt) noexcept {
    return take_next_impl(&receipt);
}
std::optional<ExecutionIssue> ExecutionSubmissionQueue::take_next_impl(
    const ExecutionReceipt* receipt) noexcept {
    if (!implementation_->is_same_process()) return std::nullopt;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    if (!state.is_executor() || state.is_quarantined || state.is_issuing()) return std::nullopt;
    const auto selected = receipt ? state.find(receipt->owner_, receipt->sequence_) : nullptr;
    if (receipt && !selected) return std::nullopt;
    const auto found = std::ranges::find_if(state.entries, [&](const auto& entry) {
        return entry->phase == Implementation::Phase::kQueued && (!receipt || entry == selected);
    });
    if (found == state.entries.end()) return std::nullopt;
    auto& entry = **found;
    entry.phase = Implementation::Phase::kIssuing;
    entry.generation = 1;
    return state.issue(entry);
}

bool ExecutionSubmissionQueue::limit_unissued_parameters(const ExecutionReceipt& receipt,
                                                         std::size_t bytes) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find(receipt.owner_, receipt.sequence_);
    if (!state.is_executor() || state.is_quarantined || !entry || !entry->payload ||
        entry->phase != Implementation::Phase::kQueued || entry->generation || entry->reply ||
        entry->is_releasing_payload || bytes > entry->issue_parameter_bytes)
        return false;
    entry->issue_parameter_bytes = bytes;
    return true;
}

bool ExecutionSubmissionQueue::copy_unissued_parameters(
    const ExecutionReceipt& receipt, std::span<std::byte> output) const noexcept {
    if (!implementation_->is_same_process()) return false;
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find(receipt.owner_, receipt.sequence_);
    if (!state.is_executor() || state.is_quarantined || !entry || !entry->payload ||
        entry->phase != Implementation::Phase::kQueued || entry->generation || entry->reply ||
        entry->is_releasing_payload || output.size() != entry->issue_parameter_bytes ||
        (!output.empty() && !output.data()))
        return false;
    if (!output.empty())
        std::memcpy(output.data(), entry->payload->parameters.data(), output.size());
    return true;
}

bool ExecutionSubmissionQueue::replace_unissued_parameters(
    const ExecutionReceipt& receipt, std::span<const std::byte> input) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find(receipt.owner_, receipt.sequence_);
    if (!state.is_executor() || state.is_quarantined || !entry || !entry->payload ||
        entry->phase != Implementation::Phase::kQueued || entry->generation || entry->reply ||
        entry->is_releasing_payload || input.size() > entry->payload->parameters.size() ||
        (!input.empty() && !input.data()))
        return false;
    if (!input.empty()) std::memmove(entry->payload->parameters.data(), input.data(), input.size());
    entry->issue_parameter_bytes = input.size();
    return true;
}

bool ExecutionSubmissionQueue::patch_issue_parameters(const ExecutionIssueToken& token,
                                                      std::size_t offset,
                                                      std::span<const std::byte> bytes) noexcept {
    if (!implementation_->is_same_process() || bytes.empty() || !bytes.data()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || state.is_quarantined || !entry || !entry->payload ||
        (entry->phase != Implementation::Phase::kIssuing &&
         entry->phase != Implementation::Phase::kResuming))
        return false;
    auto& parameters = entry->payload->parameters;
    if (offset > entry->issue_parameter_bytes ||
        bytes.size() > entry->issue_parameter_bytes - offset)
        return false;
    std::memmove(parameters.data() + offset, bytes.data(), bytes.size());
    return true;
}

bool ExecutionSubmissionQueue::publish_host_result(const ExecutionIssueToken& token,
                                                   std::int32_t native_status) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kIssuing)
        return false;
    entry->reply = {.status = native_status, .origin = ExecutionReplyOrigin::kNativeSdk};
    entry->phase = Implementation::Phase::kHostReturned;
    return true;
}

bool ExecutionSubmissionQueue::fail_uncertain_issue(const ExecutionIssueToken& token,
                                                    std::int32_t failure_status) noexcept {
    if (!implementation_->is_same_process()) return false;
    if (failure_status == 0) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || !entry ||
        (entry->phase != Implementation::Phase::kIssuing &&
         entry->phase != Implementation::Phase::kResuming))
        return false;
    if (entry->phase == Implementation::Phase::kIssuing)
        entry->reply = {.status = failure_status, .origin = ExecutionReplyOrigin::kUncertainIssue};
    entry->phase = Implementation::Phase::kHostReturned;
    state.is_quarantined = true;
    state.is_closing = true;
    return true;
}

bool ExecutionSubmissionQueue::publish_preparation_failure(const ExecutionIssueToken& token,
                                                           std::int32_t failure_status) noexcept {
    if (!implementation_->is_same_process() || failure_status == 0) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kIssuing)
        return false;
    entry->reply = {.status = failure_status, .origin = ExecutionReplyOrigin::kRejectedBeforeIssue};
    entry->phase = Implementation::Phase::kHostReturned;
    return true;
}

bool ExecutionSubmissionQueue::reject_unissued(const ExecutionReceipt& receipt,
                                               std::int32_t failure_status) noexcept {
    if (!implementation_->is_same_process()) return false;
    if (failure_status == 0) return false;
    auto& state = *implementation_;
    std::shared_ptr<Implementation::Entry> entry;
    {
        const std::lock_guard lock(state.mutex);
        entry = state.find(receipt.owner_, receipt.sequence_);
        if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kQueued)
            return false;
        entry->reply = {.status = failure_status,
                        .origin = ExecutionReplyOrigin::kRejectedBeforeIssue};
        entry->phase = Implementation::Phase::kHostReturned;
        entry->is_physically_stopped = true;
        entry->is_releasing_payload = true;
    }
    state.release_payload(entry);
    return true;
}

bool ExecutionSubmissionQueue::reject_before_native_issue(const ExecutionIssueToken& token,
                                                          std::int32_t failure_status) noexcept {
    if (!implementation_->is_same_process() || failure_status == 0) return false;
    auto& state = *implementation_;
    std::shared_ptr<Implementation::Entry> entry;
    {
        const std::lock_guard lock(state.mutex);
        entry = state.find_issue(token);
        if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kIssuing)
            return false;
        entry->reply = {.status = failure_status,
                        .origin = ExecutionReplyOrigin::kRejectedBeforeIssue};
        entry->phase = Implementation::Phase::kHostReturned;
        entry->is_physically_stopped = true;
        entry->is_releasing_payload = true;
    }
    state.release_payload(entry);
    return true;
}

std::optional<ExecutionHostReply> ExecutionSubmissionQueue::host_result(
    const ExecutionReceipt& receipt) const noexcept {
    if (!implementation_->is_same_process()) return std::nullopt;
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find(receipt.owner_, receipt.sequence_);
    return entry && !entry->is_acknowledged ? entry->reply : std::nullopt;
}

bool ExecutionSubmissionQueue::acknowledge_result(const ExecutionReceipt& receipt) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find(receipt.owner_, receipt.sequence_);
    if (!entry || !entry->reply || entry->is_acknowledged) return false;
    entry->is_acknowledged = true;
    state.collect(entry);
    return true;
}

bool ExecutionSubmissionQueue::complete_after_physical_stop(
    const ExecutionIssueToken& token) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    std::shared_ptr<Implementation::Entry> entry;
    {
        const std::lock_guard lock(state.mutex);
        entry = state.find_issue(token);
        if (!state.is_executor() || !entry ||
            entry->phase != Implementation::Phase::kHostReturned || entry->is_physically_stopped)
            return false;
        entry->is_physically_stopped = true;
        entry->is_releasing_payload = true;
    }
    state.release_payload(entry);
    return true;
}

bool ExecutionSubmissionQueue::pause_after_physical_stop(
    const ExecutionIssueToken& token) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kHostReturned ||
        entry->is_physically_stopped || !entry->reply || entry->reply->status != 0 ||
        entry->reply->origin != ExecutionReplyOrigin::kNativeSdk)
        return false;
    entry->phase = Implementation::Phase::kPaused;
    return true;
}

std::optional<ExecutionIssue> ExecutionSubmissionQueue::resume_after_physical_stop(
    const ExecutionIssueToken& token) noexcept {
    if (!implementation_->is_same_process()) return std::nullopt;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || state.is_quarantined || !entry ||
        entry->phase != Implementation::Phase::kPaused || state.is_issuing())
        return std::nullopt;
    if (entry->generation == std::numeric_limits<std::uint64_t>::max()) {
        state.is_quarantined = true;
        state.is_closing = true;
        return std::nullopt;
    }
    ++entry->generation;
    entry->phase = Implementation::Phase::kResuming;
    return state.issue(*entry);
}

bool ExecutionSubmissionQueue::finish_continuation_issue(
    const ExecutionIssueToken& token) noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    const auto entry = state.find_issue(token);
    if (!state.is_executor() || !entry || entry->phase != Implementation::Phase::kResuming)
        return false;
    entry->phase = Implementation::Phase::kHostReturned;
    return true;
}

void ExecutionSubmissionQueue::close() noexcept {
    if (!implementation_->is_same_process()) return;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    state.is_closing = true;
}

void ExecutionSubmissionQueue::quarantine() noexcept {
    if (!implementation_->is_same_process()) return;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    state.is_closing = true;
    state.is_quarantined = true;
}

bool ExecutionSubmissionQueue::shutdown() noexcept {
    if (!implementation_->is_same_process()) return false;
    auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    state.is_closing = true;
    return state.entries.empty();
}

ExecutionSubmissionSnapshot ExecutionSubmissionQueue::snapshot() const noexcept {
    if (!implementation_->is_same_process()) return {.is_closing = true, .is_quarantined = true};
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    ExecutionSubmissionSnapshot result;
    result.outstanding = state.entries.size();
    result.retained_parameter_bytes = state.parameter_bytes;
    result.retained_host_bytes = state.host_storage->bytes.load();
    result.has_executor = !state.executor.expired();
    result.is_closing = state.is_closing;
    result.is_quarantined = state.is_quarantined;
    for (const auto& entry : state.entries) {
        if (entry->phase == Implementation::Phase::kQueued) ++result.queued;
        if (entry->phase == Implementation::Phase::kIssuing ||
            entry->phase == Implementation::Phase::kResuming)
            ++result.issuing;
        if (entry->phase == Implementation::Phase::kPaused) ++result.paused;
        if (entry->phase == Implementation::Phase::kHostReturned && !entry->is_physically_stopped)
            ++result.awaiting_physical_stop;
        if (entry->reply && !entry->is_acknowledged) ++result.unacknowledged_replies;
    }
    return result;
}

ExecutionSessionIdentity ExecutionSubmissionQueue::identity() const noexcept {
    return implementation_->is_same_process() ? implementation_->session
                                              : ExecutionSessionIdentity{};
}

}  // namespace glimmer::backend
