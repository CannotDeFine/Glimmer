#include "host_stage_metrics.h"

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <locale>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace glimmer::app {
namespace {
constexpr std::size_t kOutputMax = 512 * 1024 * 1024;
static_assert(sizeof(HostStageMetrics::Bucket) * HostStageMetrics::bucket_count <
              256 * 1024 * 1024);
constexpr std::array names{"step",       "transport",     "receive",     "observe",
                           "advance",    "plan_receive",  "plan_slot",   "plan_issue",
                           "plan_other", "observations",  "snapshot",    "pending",
                           "policy",     "issue_attempt", "issue_native"};
static_assert(names.size() == HostStageMetrics::stage_count);
class Descriptor {
   public:
    explicit Descriptor(int value) : value_(value) {}
    ~Descriptor() {
        if (value_ >= 0) (void)::close(value_);
    }
    int get() const noexcept {
        return value_;
    }
    bool close() noexcept {
        const auto value = value_;
        value_ = -1;
        return value >= 0 && ::close(value) == 0;
    }

   private:
    int value_;
};
bool output_parent(const std::filesystem::path& path) {
    if (!path.is_absolute() || path.filename().empty() || path.filename() == "." ||
        path.filename() == ".." || path.lexically_normal() != path)
        return false;
    const auto parent = path.parent_path();
    if (std::filesystem::canonical(parent) != parent) return false;
    struct stat status {};
    return ::lstat(parent.c_str(), &status) == 0 && S_ISDIR(status.st_mode) &&
           status.st_uid == ::geteuid();
}
bool publish_document(const std::string& output, const std::string& bytes) {
    if (bytes.size() > kOutputMax || !HostStageMetrics::fresh_output(output)) return false;
    const std::filesystem::path path{output};
    Descriptor parent(
        ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    struct stat parent_status {};
    if (parent.get() < 0 || ::fstat(parent.get(), &parent_status) != 0 ||
        parent_status.st_uid != ::geteuid())
        return false;
    const auto name = path.filename().string(),
               temporary = name + ".pending-" + std::to_string(::getpid());
    Descriptor file(::openat(parent.get(), temporary.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (file.get() < 0) return false;
    std::size_t written{};
    bool ok = true;
    while (written < bytes.size()) {
        const auto count = ::write(file.get(), bytes.data() + written, bytes.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            ok = false;
            break;
        }
        written += static_cast<std::size_t>(count);
    }
    ok = (::fsync(file.get()) == 0) && ok;
    ok = file.close() && ok;
    if (ok) ok = ::linkat(parent.get(), temporary.c_str(), parent.get(), name.c_str(), 0) == 0;
    const bool removed = ::unlinkat(parent.get(), temporary.c_str(), 0) == 0;
    return ::fsync(parent.get()) == 0 && ok && removed;
}
}  // namespace

HostStageMetrics::HostStageMetrics(Clock clock, std::size_t lifecycle_capacity)
    : clock_(clock),
      buckets_(bucket_count),
      events_(lifecycle_capacity <= lifecycle_capacity_limit ? lifecycle_capacity : 0),
      intakes_(events_.size()) {
    if (lifecycle_capacity > lifecycle_capacity_limit) valid_ = false;
    Stamp now;
    if (!clock_ || !read(now)) {
        valid_ = false;
        return;
    }
    origin_ = now.wall - now.wall % bucket_ns;
}
bool HostStageMetrics::system_clock(Stamp& stamp) noexcept {
    timespec wall{}, cpu{};
    if (::clock_gettime(CLOCK_MONOTONIC, &wall) != 0 ||
        ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) != 0 || wall.tv_sec < 0 || cpu.tv_sec < 0)
        return false;
    constexpr auto max_seconds = std::numeric_limits<std::uint64_t>::max() / 1000000000 - 1;
    if (static_cast<std::uint64_t>(wall.tv_sec) > max_seconds ||
        static_cast<std::uint64_t>(cpu.tv_sec) > max_seconds)
        return false;
    stamp = {static_cast<std::uint64_t>(wall.tv_sec) * 1000000000 +
                 static_cast<std::uint64_t>(wall.tv_nsec),
             static_cast<std::uint64_t>(cpu.tv_sec) * 1000000000 +
                 static_cast<std::uint64_t>(cpu.tv_nsec)};
    return true;
}
bool HostStageMetrics::read(Stamp& stamp) noexcept {
    if (!valid_ || !clock_(stamp) || stamp.wall < last_wall_ || stamp.cpu < last_cpu_) {
        valid_ = false;
        return false;
    }
    last_wall_ = stamp.wall;
    last_cpu_ = stamp.cpu;
    add(clock_reads_, 1);
    return valid_;
}
void HostStageMetrics::add(std::uint64_t& destination, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - destination)
        valid_ = false;
    else
        destination += value;
}
HostStageMetrics::Bucket* HostStageMetrics::current(std::uint64_t wall) noexcept {
    if (!valid_ || wall < origin_ || (wall - origin_) / bucket_ns >= bucket_count) {
        valid_ = false;
        return nullptr;
    }
    const auto index = static_cast<std::size_t>((wall - origin_) / bucket_ns);
    used_buckets_ = std::max(used_buckets_, index + 1);
    return &buckets_[index];
}
HostStageMetrics::Scope::Scope(HostStageMetrics* metrics, Stage stage) noexcept
    : metrics_(metrics) {
    if (metrics_) depth_ = metrics_->enter(stage);
}
HostStageMetrics::Scope::~Scope() {
    if (metrics_) metrics_->leave(depth_);
}
std::size_t HostStageMetrics::enter(Stage stage) noexcept {
    Stamp now;
    if (!valid_ || depth_ == depth_limit || static_cast<std::size_t>(stage) >= stage_count ||
        !read(now)) {
        valid_ = false;
        return depth_limit;
    }
    const auto depth = depth_++;
    stack_[depth] = {stage, now, 0, 0};
    return depth;
}
void HostStageMetrics::leave(std::size_t depth) noexcept {
    if (!valid_) return;
    Stamp end;
    if (!depth_ || depth + 1 != depth_ || !read(end)) {
        valid_ = false;
        return;
    }
    const auto frame = stack_[--depth_];
    const auto wall = end.wall - frame.begin.wall, cpu = end.cpu - frame.begin.cpu;
    if (wall < frame.child_wall || cpu < frame.child_cpu) {
        valid_ = false;
        return;
    }
    if (depth_) {
        add(stack_[depth_ - 1].child_wall, wall);
        add(stack_[depth_ - 1].child_cpu, cpu);
    }
    auto* bucket = current(end.wall);
    if (!bucket) return;
    if (!bucket->minimum_span_start || frame.begin.wall < bucket->minimum_span_start)
        bucket->minimum_span_start = frame.begin.wall;
    bucket->maximum_span_end = std::max(bucket->maximum_span_end, end.wall);
    auto& counter = bucket->stages[static_cast<std::size_t>(frame.stage)];
    add(counter.calls, 1);
    add(counter.wall, wall);
    add(counter.cpu, cpu);
    add(counter.exclusive_wall, wall - frame.child_wall);
    add(counter.exclusive_cpu, cpu - frame.child_cpu);
}
void HostStageMetrics::queue(Slot slot, const Queue& observation) noexcept {
    if (!valid_) return;
    // The containing snapshot scope has just read the clock on exit; no extra
    // clock or backend observation is necessary for this copied queue sample.
    auto* bucket = current(last_wall_);
    if (!bucket) return;
    const auto index = slot_index(slot);
    if (!valid_) return;
    auto& value = bucket->queues[index];
    add(value.samples, 1);
    add(value.requests, observation.requests);
    add(value.queued, observation.queued);
    add(value.issuing, observation.issuing);
    add(value.paused, observation.paused);
    add(value.awaiting_stop, observation.awaiting_stop);
    add(value.ready, observation.ready);
    add(value.physical, observation.physical);
    add(value.low_epochs, observation.low_epochs);
    add(value.preparation, observation.preparation);
    value.max_requests = std::max(value.max_requests, observation.requests);
    value.max_queued = std::max(value.max_queued, observation.queued);
}
std::size_t HostStageMetrics::slot_index(Slot slot) noexcept {
    if (!valid_ || !slot.connection || !slot.pid || !slot.start_ticks) {
        valid_ = false;
        return slot_count;
    }
    std::size_t index{};
    while (index < used_slots_ && slots_[index].connection != slot.connection) ++index;
    if (index == used_slots_) {
        if (used_slots_ == slot_count) {
            valid_ = false;
            return slot_count;
        }
        slots_[used_slots_++] = slot;
    } else if (slots_[index].pid != slot.pid || slots_[index].start_ticks != slot.start_ticks) {
        valid_ = false;
        return slot_count;
    }
    return index;
}
void HostStageMetrics::lifecycle(Lifecycle kind, Slot slot, std::uint64_t wire,
                                 std::uint64_t command, std::uint64_t detail,
                                 std::uint64_t frame_received_ns, std::uint64_t owner_enter_ns,
                                 std::uint64_t first_read_ns) noexcept {
    if (events_.empty() || !valid_) return;
    Stamp now;
    const auto index = slot_index(slot);
    if (!valid_) return;
    if (used_events_ == events_.size() || kind > Lifecycle::kRetire || !read(now)) {
        valid_ = false;
        return;
    }
    const bool receives = kind == Lifecycle::kReceive;
    if ((receives && (!wire || !first_read_ns || first_read_ns > frame_received_ns ||
                      !frame_received_ns || !owner_enter_ns || frame_received_ns > owner_enter_ns ||
                      owner_enter_ns > now.wall || used_intakes_ == intakes_.size())) ||
        (!receives && (frame_received_ns || owner_enter_ns || first_read_ns))) {
        valid_ = false;
        return;
    }
    // Validate both bounded records before appending either. Missing intake
    // evidence invalidates diagnostics, never request execution or ownership.
    events_[used_events_++] = {now.wall, index,   static_cast<std::uint64_t>(kind),
                               wire,     command, detail};
    if (receives)
        intakes_[used_intakes_++] = {frame_received_ns,      owner_enter_ns, now.wall, index, wire,
                                     detail & 0xffffffffULL, first_read_ns};
}
void HostStageMetrics::issue(std::size_t outcome) noexcept {
    if (!valid_) return;
    if (outcome >= 4) {
        valid_ = false;
        return;
    }
    if (auto* bucket = current(last_wall_)) add(bucket->issue_outcomes[outcome], 1);
}
void HostStageMetrics::slot_plan(SlotPlanReason reason) noexcept {
    if (!valid_) return;
    const auto index = static_cast<std::size_t>(reason);
    if (!depth_ || index >= slot_plan_reason_count) {
        valid_ = false;
        return;
    }
    if (auto* bucket = current(last_wall_)) add(bucket->slot_plans[index], 1);
}
void HostStageMetrics::reply_at(std::uint64_t wall, Slot slot, std::uint64_t wire,
                                std::uint64_t command, std::uint64_t detail) noexcept {
    if (events_.empty() || !valid_) return;
    Stamp now;
    const auto index = slot_index(slot);
    if (!valid_) return;
    if (used_events_ == events_.size() || !read(now) || !wall || wall < origin_ ||
        wall > now.wall || (used_events_ && wall < events_[used_events_ - 1].wall)) {
        valid_ = false;
        return;
    }
    events_[used_events_++] = {wall, index,   static_cast<std::uint64_t>(Lifecycle::kReply),
                               wire, command, detail};
}
bool HostStageMetrics::fresh_output(const std::string& output) noexcept {
    try {
        const std::filesystem::path path{output};
        if (output.size() > 4096 || !output_parent(path)) return false;
        struct stat status {};
        return ::lstat(path.c_str(), &status) != 0 && errno == ENOENT;
    } catch (...) {
        return false;
    }
}
bool HostStageMetrics::publish(const std::string& output) const noexcept {
    try {
        if (!fresh_output(output)) return false;
        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << "{\"schema\":1,\"diagnostic_only\":true,\"valid\":"
             << (valid_ && !depth_ ? "true" : "false") << ",\"pid\":" << ::getpid()
             << ",\"origin_ns\":" << origin_ << ",\"bucket_ns\":" << bucket_ns
             << ",\"clock_reads\":" << clock_reads_
             << ",\"idle_enabled\":" << (idle_enabled_ ? "true" : "false")
             << ",\"attribution\":\"span_end_bucket\",\"stages\":[";
        for (std::size_t i = 0; i < names.size(); ++i)
            json << (i ? ",\"" : "\"") << names[i] << '"';
        json << "],\"counter_fields\":[\"calls\",\"wall_ns\",\"cpu_ns\",\"exclusive_wall_ns\","
                "\"exclusive_cpu_ns\"],"
                "\"queue_fields\":[\"samples\",\"requests_sum\",\"queued_sum\",\"issuing_sum\","
                "\"paused_sum\","
                "\"awaiting_stop_sum\",\"ready_samples\",\"physical_samples\",\"low_epochs_sum\","
                "\"preparation_samples\",\"requests_max\",\"queued_max\"],\"slots\":[";
        for (std::size_t i = 0; i < used_slots_; ++i) {
            const auto& slot = slots_[i];
            json << (i ? ",[" : "[") << slot.connection << ',' << slot.pid << ','
                 << slot.start_ticks << ']';
        }
        json << "],\"slot_plan_schema\":1,\"slot_plan_attribution\":\"last_scope_stamp\","
                "\"slot_plan_reasons\":[\"disabled_policy\",\"non_helper\",\"local_obligation\","
                "\"skipped_helper\"],\"buckets\":[";
        for (std::size_t i = 0; i < used_buckets_; ++i) {
            const auto& b = buckets_[i];
            json << (i ? ",{" : "{") << "\"begin_ns\":" << origin_ + i * bucket_ns
                 << ",\"minimum_span_start_ns\":" << b.minimum_span_start
                 << ",\"maximum_span_end_ns\":" << b.maximum_span_end << ",\"stages\":[";
            for (std::size_t j = 0; j < stage_count; ++j) {
                const auto& c = b.stages[j];
                json << (j ? ",[" : "[") << c.calls << ',' << c.wall << ',' << c.cpu << ','
                     << c.exclusive_wall << ',' << c.exclusive_cpu << ']';
            }
            json << "],\"queues\":[";
            bool first = true;
            for (std::size_t j = 0; j < used_slots_; ++j) {
                const auto& q = b.queues[j];
                if (!q.samples) continue;
                json << (first ? "[" : ",[") << j << ',' << q.samples << ',' << q.requests << ','
                     << q.queued << ',' << q.issuing << ',' << q.paused << ',' << q.awaiting_stop
                     << ',' << q.ready << ',' << q.physical << ',' << q.low_epochs << ','
                     << q.preparation << ',' << q.max_requests << ',' << q.max_queued << ']';
                first = false;
            }
            json << "],\"issue_outcomes\":[" << b.issue_outcomes[0] << ',' << b.issue_outcomes[1]
                 << ',' << b.issue_outcomes[2] << ',' << b.issue_outcomes[3]
                 << "],\"slot_plans\":[";
            for (std::size_t j = 0; j < slot_plan_reason_count; ++j)
                json << (j ? "," : "") << b.slot_plans[j];
            json << "]}";
            if (!json || json.tellp() < 0 || static_cast<std::size_t>(json.tellp()) > kOutputMax)
                return false;
        }
        json << "],\"lifecycle\":{\"capacity\":" << events_.size()
             << ",\"fields\":[\"wall_ns\",\"slot\",\"kind\",\"wire\",\"command\",\"detail\"],"
                "\"kinds\":[\"receive\",\"admit\",\"issue_begin\",\"issue_end\",\"reply\","
                "\"retire\"],"
                "\"events\":[";
        for (std::size_t i = 0; i < used_events_; ++i) {
            const auto& e = events_[i];
            json << (i ? ",[" : "[") << e.wall << ',' << e.slot << ',' << e.kind << ',' << e.wire
                 << ',' << e.command << ',' << e.detail << ']';
            if (!json || json.tellp() < 0 || static_cast<std::size_t>(json.tellp()) > kOutputMax)
                return false;
        }
        json << "]},\"intake\":{\"schema\":2,\"capacity\":" << intakes_.size()
             << ",\"fields\":[\"frame_received_ns\",\"owner_enter_ns\",\"retained_ns\","
                "\"slot\",\"wire\",\"opcode\",\"first_read_ns\"],\"records\":[";
        for (std::size_t i = 0; i < used_intakes_; ++i) {
            const auto& entry = intakes_[i];
            json << (i ? ",[" : "[") << entry.frame << ',' << entry.owner << ',' << entry.retained
                 << ',' << entry.slot << ',' << entry.wire << ',' << entry.opcode << ','
                 << entry.first_read << ']';
            if (!json || json.tellp() < 0 || static_cast<std::size_t>(json.tellp()) > kOutputMax)
                return false;
        }
        json << "]}}\n";
        return publish_document(output, json.str());
    } catch (...) {
        return false;
    }
}

IssueDepthMetrics::IssueDepthMetrics(Clock clock, std::size_t event_capacity)
    : IssueDepthMetrics(clock, event_capacity, SelectionConfig{}) {}
IssueDepthMetrics::IssueDepthMetrics(Clock clock, std::size_t event_capacity,
                                     SelectionConfig selections)
    : clock_(clock),
      buckets_(bucket_capacity),
      event_capacity_(event_capacity),
      selection_capacity_(selections.capacity) {
    entries_.reserve(entry_capacity);
    if (event_capacity_ > event_capacity_limit || selection_capacity_ > selection_capacity_limit ||
        (selection_capacity_ && !event_capacity_)) {
        invalidate(Failure::kCapacity);
        return;
    }
    if (event_capacity_) {
        active_events_.reserve(entry_capacity);
        events_.reserve(event_capacity_);
        if (event_storage_bytes() >= std::size_t{80} * 1024 * 1024) {
            invalidate(Failure::kCapacity);
            return;
        }
    }
    if (selection_capacity_) {
        selections_.reserve(selection_capacity_);
        if (event_storage_bytes() + selection_storage_bytes() >= std::size_t{96} * 1024 * 1024) {
            invalidate(Failure::kCapacity);
            return;
        }
    }
    if (!clock_ || !clock_(last_) || !last_ || last_ > UINT64_MAX - bucket_capacity * bucket_ns) {
        invalidate(Failure::kClock);
        return;
    }
    clock_reads_ = 1;
    begin_ = last_;
    origin_ = begin_ - begin_ % bucket_ns;
}
bool IssueDepthMetrics::system_clock(std::uint64_t& now) noexcept {
    timespec stamp{};
    if (::clock_gettime(CLOCK_MONOTONIC, &stamp) != 0 || stamp.tv_sec < 0 ||
        static_cast<std::uint64_t>(stamp.tv_sec) > UINT64_MAX / 1000000000 - 1)
        return false;
    now = static_cast<std::uint64_t>(stamp.tv_sec) * 1000000000 +
          static_cast<std::uint64_t>(stamp.tv_nsec);
    return true;
}
void IssueDepthMetrics::invalidate(Failure failure) noexcept {
    if (valid()) failure_ = failure;
}
IssueDepthMetrics::Entry* IssueDepthMetrics::find(std::uint64_t connection,
                                                  std::uint64_t sequence) noexcept {
    const auto found = std::ranges::find_if(entries_, [&](const auto& entry) {
        return entry.connection == connection && entry.sequence == sequence;
    });
    return found == entries_.end() ? nullptr : &*found;
}
IssueDepthMetrics::Event* IssueDepthMetrics::active_event(const Entry* entry) noexcept {
    if (!event_capacity_) return nullptr;
    if (active_events_.size() != entries_.size()) {
        invalidate(Failure::kTransition);
        return nullptr;
    }
    auto& event = active_events_[static_cast<std::size_t>(entry - entries_.data())];
    if (event.connection != entry->connection || event.sequence != entry->sequence ||
        event.opcode != entry->opcode) {
        invalidate(Failure::kIdentity);
        return nullptr;
    }
    return &event;
}
std::uint64_t IssueDepthMetrics::begin_plan() noexcept {
    if (!event_capacity_ || !valid()) return 0;
    if (is_finished_ || plan_serial_ == UINT64_MAX) {
        invalidate(Failure::kCapacity);
        return 0;
    }
    return ++plan_serial_;
}
void IssueDepthMetrics::authorized(Authorization authorization) noexcept {
    if (!event_capacity_ || !valid()) return;
    const auto [connection, sequence, plan_serial, is_resume] = authorization;
    auto* entry = find(connection, sequence);
    if (!entry) return;  // Non-compute commands do not enter this ledger.
    auto* event = active_event(entry);
    if (!event) return;
    if (is_resume || entry->state != State::kAdmitted || !plan_serial ||
        plan_serial != plan_serial_ || event->plan_serial != plan_serial ||
        !event->matched_observed_ns || event->authorized_ns) {
        invalidate(is_resume ? Failure::kUnsupported : Failure::kTransition);
        return;
    }
    if (!advance()) return;
    event->authorized_ns = last_;
}
IssueDepthMetrics::Identity* IssueDepthMetrics::identity(std::uint64_t connection) noexcept {
    for (std::size_t index = 0; index < used_slots_; ++index)
        if (identities_[index].slot.connection == connection) return &identities_[index];
    return nullptr;
}
IssueDepthMetrics::Identity* IssueDepthMetrics::identity(Slot slot) noexcept {
    if (!slot.connection || !slot.pid || !slot.start_ticks) {
        invalidate(Failure::kIdentity);
        return nullptr;
    }
    if (auto* found = identity(slot.connection)) {
        if (found->slot.pid != slot.pid || found->slot.start_ticks != slot.start_ticks) {
            invalidate(Failure::kIdentity);
            return nullptr;
        }
        return found;
    }
    if (used_slots_ == slot_capacity) {
        invalidate(Failure::kCapacity);
        return nullptr;
    }
    auto& added = identities_[used_slots_++];
    added.slot = slot;
    return &added;
}
IssueDepthMetrics::Bucket* IssueDepthMetrics::advance() noexcept {
    if (!valid()) return nullptr;
    std::uint64_t now{};
    if (is_finished_ || !clock_(now) || now < last_ || clock_reads_ == UINT64_MAX) {
        invalidate(Failure::kClock);
        return nullptr;
    }
    ++clock_reads_;
    if (now - origin_ >= bucket_capacity * bucket_ns) {
        invalidate(Failure::kCapacity);
        return nullptr;
    }
    while (last_ < now) {
        const auto index = static_cast<std::size_t>((last_ - origin_) / bucket_ns);
        const auto end = std::min(now, origin_ + (index + 1) * bucket_ns);
        const auto elapsed = end - last_;
        auto& bucket = buckets_[index];
        // Depth and bucket duration are bounded, so their products cannot overflow.
        bucket.covered_ns += elapsed;
        bucket.admitted_not_submitted_ns += elapsed * unsubmitted_;
        bucket.submitted_not_retired_ns += elapsed * submitted_;
        bucket.issue_call_ns += elapsed * issuing_;
        if (!unsubmitted_ && !submitted_) bucket.no_admitted_compute_ns += elapsed;
        if (!submitted_ && !issuing_) bucket.no_inflight_compute_ns += elapsed;
        last_ = end;
    }
    const auto index = static_cast<std::size_t>((now - origin_) / bucket_ns);
    used_buckets_ = std::max(used_buckets_, index + 1);
    return &buckets_[index];
}
void IssueDepthMetrics::admit(Slot slot, std::uint64_t sequence, std::uint32_t opcode,
                              bool requires_preparation) noexcept {
    if (!valid()) return;
    auto* owner = identity(slot);
    if (!owner) return;
    if (!sequence || sequence <= owner->last_admitted || !opcode ||
        (requires_preparation && opcode != 11)) {
        invalidate(Failure::kTransition);
        return;
    }
    if (entries_.size() == entry_capacity) {
        invalidate(Failure::kCapacity);
        return;
    }
    if (event_capacity_ && active_events_.size() != entries_.size()) {
        invalidate(Failure::kTransition);
        return;
    }
    const auto operation = std::ranges::find_if(
        opcodes_, [opcode](const auto& item) { return !item.opcode || item.opcode == opcode; });
    if (operation == opcodes_.end()) {
        invalidate(Failure::kCapacity);
        return;
    }
    auto* bucket = advance();
    if (!bucket) return;
    entries_.push_back({slot.connection, sequence, opcode, State::kAdmitted});
    if (event_capacity_)
        active_events_.push_back({.connection = slot.connection,
                                  .sequence = sequence,
                                  .opcode = opcode,
                                  .admitted_ns = last_,
                                  .admission_observation_generation = owner->observation_generation,
                                  .preparation_required = requires_preparation ? 1U : 0U});
    operation->opcode = opcode;
    ++operation->admitted;
    owner->last_admitted = sequence;
    ++owner->admitted;
    ++bucket->admitted;
    ++unsubmitted_;
}
void IssueDepthMetrics::prepared(std::uint64_t connection, std::uint64_t sequence) noexcept {
    if (!event_capacity_ || !valid()) return;
    const auto* entry = find(connection, sequence);
    auto* event = entry ? active_event(entry) : nullptr;
    if (!entry || entry->state != State::kAdmitted || !event || !event->preparation_required ||
        event->service_observed_prepared_ns || event->first_observed_ns) {
        invalidate(Failure::kTransition);
        return;
    }
    if (advance()) event->service_observed_prepared_ns = last_;
}
void IssueDepthMetrics::begin_issue(std::uint64_t connection, std::uint64_t sequence,
                                    bool is_resume) noexcept {
    if (!valid()) return;
    auto* entry = find(connection, sequence);
    if (!entry) return;
    if (is_resume || entry->state != State::kAdmitted) {
        invalidate(is_resume ? Failure::kUnsupported : Failure::kTransition);
        return;
    }
    auto* event = active_event(entry);
    if (!valid()) return;
    if (event && (!event->authorized_ns || event->issue_begin_ns)) {
        invalidate(Failure::kTransition);
        return;
    }
    if (!advance()) return;
    if (event) event->issue_begin_ns = last_;
    entry->state = State::kIssuing;
    ++issuing_;
}
void IssueDepthMetrics::end_issue(std::uint64_t connection, std::uint64_t sequence,
                                  bool is_successful) noexcept {
    if (!valid()) return;
    auto* entry = find(connection, sequence);
    if (!entry) return;
    if (!is_successful || entry->state != State::kIssuing) {
        invalidate(Failure::kTransition);
        return;
    }
    auto* event = active_event(entry);
    if (!valid()) return;
    auto* bucket = advance();
    if (!bucket) return;
    if (event) event->issue_end_ns = last_;
    entry->state = State::kSubmitted;
    --issuing_;
    --unsubmitted_;
    ++submitted_;
    ++bucket->submitted;
    ++identity(connection)->submitted;
}
void IssueDepthMetrics::retired(std::uint64_t connection, std::uint64_t sequence,
                                bool is_successful) noexcept {
    if (!valid()) return;
    auto* entry = find(connection, sequence);
    if (!entry) return;  // Ordinary non-compute retirement is outside this ledger.
    if (!is_successful || entry->state != State::kSubmitted) {
        invalidate(Failure::kTransition);
        return;
    }
    auto* event = active_event(entry);
    if (!valid()) return;
    if (event && events_.size() == event_capacity_) {
        invalidate(Failure::kCapacity);
        return;
    }
    auto* bucket = advance();
    if (!bucket) return;
    if (event) {
        event->retired_ns = last_;
        events_.push_back(*event);
        *event = active_events_.back();
        active_events_.pop_back();
    }
    --submitted_;
    ++bucket->retired;
    ++identity(connection)->retired;
    *entry = entries_.back();
    entries_.pop_back();
}
bool IssueDepthMetrics::record_selection(const Identity& owner, std::uint64_t pending,
                                         SelectionClass classification) noexcept {
    if (!selection_capacity_ || !pending) return true;
    if (selections_.size() == selection_capacity_) {
        invalidate(Failure::kCapacity);
        return false;
    }
    // Reuse the existing observation's clock and generation. Zero pending is
    // omitted, not treated as a duration or evidence of an empty backend queue.
    selections_.push_back(
        {owner.slot.connection, owner.observation_generation, last_, pending, classification});
    return true;
}
void IssueDepthMetrics::observe(Slot slot, std::uint64_t pending, bool has_ready_work,
                                bool has_unknown_state, std::uint64_t plan_serial) noexcept {
    if (!valid()) return;
    auto* owner = identity(slot);
    if (!owner || !owner->admitted) return;
    auto* bucket = advance();
    if (!bucket) return;
    if (event_capacity_) {
        if (owner->observation_generation == UINT64_MAX) {
            invalidate(Failure::kCapacity);
            return;
        }
        ++owner->observation_generation;
    }
    ++bucket->observations;
    if (owner->last_observation)
        bucket->maximum_observation_gap_ns =
            std::max(bucket->maximum_observation_gap_ns, last_ - owner->last_observation);
    owner->last_observation = last_;
    if (has_unknown_state || (has_ready_work && !pending)) {
        if (!record_selection(*owner, pending, SelectionClass::kUnknown)) return;
        ++bucket->unknown;
        return;
    }
    const auto* entry = find(slot.connection, pending);
    if (!entry) {
        if (!record_selection(*owner, pending, SelectionClass::kScopeExternal)) return;
        if (pending)
            ++bucket->other_pending;
        else
            ++bucket->no_selected_command;
        return;
    }
    if (entry->state != State::kAdmitted) {
        if (!record_selection(*owner, pending, SelectionClass::kUnknown)) return;
        ++bucket->unknown;
        return;
    }
    if (!has_ready_work) {
        if (!record_selection(*owner, pending, SelectionClass::kTrackedNotReady)) return;
        ++bucket->not_ready_compute;
        return;
    }
    if (!record_selection(*owner, pending, SelectionClass::kTrackedReady)) return;
    ++bucket->ready_compute;
    if (!submitted_ && !issuing_) ++bucket->ready_without_inflight;
    if (auto* event = active_event(entry)) {
        if ((event->preparation_required && !event->service_observed_prepared_ns) ||
            event->positive_observations == UINT64_MAX ||
            (plan_serial && (plan_serial != plan_serial_ || event->authorized_ns))) {
            invalidate(Failure::kTransition);
            return;
        }
        if (!event->first_observed_ns) {
            if (owner->observation_generation <= event->admission_observation_generation) {
                invalidate(Failure::kTransition);
                return;
            }
            event->first_observed_ns = last_;
            event->first_positive_observation_generation = owner->observation_generation;
        }
        ++event->positive_observations;
        if (plan_serial) {
            if (event->plan_serial == plan_serial) {
                invalidate(Failure::kTransition);
                return;
            }
            event->plan_serial = plan_serial;
            event->matched_observed_ns = last_;
            ++event->plan_positive_observations;
        }
    }
}
void IssueDepthMetrics::finish() noexcept {
    if (is_finished_) return;
    (void)advance();
    if (!entries_.empty()) invalidate(Failure::kUnclosed);
    if (event_capacity_ && valid()) {
        std::uint64_t retired_count{};
        for (std::size_t index = 0; index < used_slots_; ++index) {
            const auto count = identities_[index].retired;
            if (count > event_capacity_ - retired_count) {
                invalidate(Failure::kTransition);
                break;
            }
            retired_count += count;
        }
        if (!active_events_.empty() || retired_count != events_.size())
            invalidate(Failure::kTransition);
    }
    is_finished_ = true;
}
bool IssueDepthMetrics::publish(const std::string& path) const noexcept {
    try {
        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << "{\"schema\":"
             << (selection_capacity_ ? 5
                 : event_capacity_   ? 4
                                     : 1)
             << ",\"diagnostic_only\":true,\"valid\":"
             << (valid() && is_finished_ ? "true" : "false")
             << ",\"failure\":" << static_cast<unsigned int>(failure_) << ",\"pid\":" << ::getpid()
             << ",\"origin_ns\":" << origin_ << ",\"begin_ns\":" << begin_
             << ",\"end_ns\":" << last_ << ",\"bucket_ns\":" << bucket_ns
             << ",\"clock_reads\":" << clock_reads_ << ",\"unclosed\":" << entries_.size()
             << ",\"readiness_is_instantaneous\":true,\"outstanding_is_gpu_busy\":false,"
                "\"slots\":[";
        for (std::size_t index = 0; index < used_slots_; ++index) {
            const auto& item = identities_[index];
            json << (index ? "," : "") << "{\"connection\":" << item.slot.connection
                 << ",\"pid\":" << item.slot.pid << ",\"start_ticks\":" << item.slot.start_ticks
                 << ",\"admitted\":" << item.admitted << ",\"submitted\":" << item.submitted
                 << ",\"retired\":" << item.retired;
            if (selection_capacity_)
                json << ",\"observation_generation\":" << item.observation_generation;
            json << '}';
        }
        json << "],\"opcodes\":[";
        bool is_first = true;
        for (const auto& operation : opcodes_) {
            if (!operation.opcode) continue;
            json << (is_first ? "" : ",") << "{\"opcode\":" << operation.opcode
                 << ",\"admitted\":" << operation.admitted << '}';
            is_first = false;
        }
        json << "],\"buckets\":[";
        for (std::size_t index = 0; index < used_buckets_; ++index) {
            const auto& b = buckets_[index];
            json << (index ? "," : "") << "{\"begin_ns\":" << origin_ + index * bucket_ns
                 << ",\"covered_ns\":" << b.covered_ns
                 << ",\"admitted_not_submitted_ns\":" << b.admitted_not_submitted_ns
                 << ",\"submitted_not_retired_ns\":" << b.submitted_not_retired_ns
                 << ",\"issue_call_ns\":" << b.issue_call_ns
                 << ",\"no_admitted_compute_ns\":" << b.no_admitted_compute_ns
                 << ",\"no_inflight_compute_ns\":" << b.no_inflight_compute_ns
                 << ",\"admitted\":" << b.admitted << ",\"submitted\":" << b.submitted
                 << ",\"retired\":" << b.retired << ",\"observations\":" << b.observations
                 << ",\"ready_compute\":" << b.ready_compute
                 << ",\"not_ready_compute\":" << b.not_ready_compute
                 << ",\"other_pending\":" << b.other_pending << ",\"unknown\":" << b.unknown
                 << ",\"no_selected_command\":" << b.no_selected_command
                 << ",\"ready_without_inflight\":" << b.ready_without_inflight
                 << ",\"maximum_observation_gap_ns\":" << b.maximum_observation_gap_ns << '}';
        }
        json << ']';
        if (event_capacity_) {
            // Seventeen uint64 fields require at most 360 bytes per JSON row.
            // Keep metadata/buckets a separate bounded allowance below the publisher limit.
            static_assert(event_capacity_limit * 360 + std::size_t{16} * 1024 * 1024 < kOutputMax);
            json << ",\"events\":{\"capacity\":" << event_capacity_
                 << ",\"complete\":" << (valid() && is_finished_ ? "true" : "false")
                 << ",\"ready_duration\":false,\"gpu_timestamps\":false,\"fields\":["
                    "\"connection\",\"sequence\",\"opcode\",\"admitted_ns\","
                    "\"first_observed_ns\",\"matched_observed_ns\",\"authorized_ns\","
                    "\"issue_begin_ns\",\"issue_end_ns\",\"retired_ns\",\"plan_serial\","
                    "\"positive_observations\",\"plan_positive_observations\","
                    "\"admission_observation_generation\",\"first_positive_observation_"
                    "generation\",\"preparation_required\",\"service_observed_prepared_ns\"],"
                    "\"records\":[";
            for (std::size_t index = 0; index < events_.size(); ++index) {
                const auto& e = events_[index];
                json << (index ? ",[" : "[") << e.connection << ',' << e.sequence << ',' << e.opcode
                     << ',' << e.admitted_ns << ',' << e.first_observed_ns << ','
                     << e.matched_observed_ns << ',' << e.authorized_ns << ',' << e.issue_begin_ns
                     << ',' << e.issue_end_ns << ',' << e.retired_ns << ',' << e.plan_serial << ','
                     << e.positive_observations << ',' << e.plan_positive_observations << ','
                     << e.admission_observation_generation << ','
                     << e.first_positive_observation_generation << ',' << e.preparation_required
                     << ',' << e.service_observed_prepared_ns << ']';
                if (!json || json.tellp() < 0 ||
                    static_cast<std::size_t>(json.tellp()) > kOutputMax)
                    return false;
            }
            json << "]}";
        }
        if (selection_capacity_) {
            static_assert(event_capacity_limit * 360 + selection_capacity_limit * 110 +
                              std::size_t{16} * 1024 * 1024 <
                          kOutputMax);
            json << ",\"selections\":{\"capacity\":" << selection_capacity_
                 << ",\"complete\":" << (valid() && is_finished_ ? "true" : "false")
                 << ",\"fields\":[\"connection\",\"observation_generation\",\"observed_ns\","
                    "\"pending_sequence\",\"classification\"],\"classifications\":["
                    "\"tracked_ready\",\"tracked_not_ready\",\"scope_external\",\"unknown\"],"
                    "\"records\":[";
            for (std::size_t index = 0; index < selections_.size(); ++index) {
                const auto& selection = selections_[index];
                json << (index ? ",[" : "[") << selection.connection << ','
                     << selection.observation_generation << ',' << selection.observed_ns << ','
                     << selection.pending_sequence << ','
                     << static_cast<std::uint64_t>(selection.classification) << ']';
                if (!json || json.tellp() < 0 ||
                    static_cast<std::size_t>(json.tellp()) > kOutputMax)
                    return false;
            }
            json << "]}";
        }
        json << "}\n";
        return publish_document(path, json.str());
    } catch (...) {
        return false;
    }
}
}  // namespace glimmer::app
