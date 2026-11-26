#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace glimmer::app {

// Diagnostic-only, constructing-thread ownership. No CUDA, locks or hot-path I/O.
// Counters cannot authorize execution; any observation failure only invalidates
// the diagnostic. Spans are assigned to their END bucket. A consumer must also
// check each selected bucket's minimum span start against its analysis window.
class HostStageMetrics final {
   public:
    enum class Stage : std::size_t {
        kStep,
        kTransport,
        kReceive,
        kObserve,
        kAdvance,
        kPlanReceive,
        kPlanSlot,
        kPlanIssue,
        kPlanOther,
        kObservations,
        kSnapshot,
        kPending,
        kPolicy,
        kIssueAttempt,
        kIssueNative,
        kCount
    };
    struct Stamp {
        std::uint64_t wall{}, cpu{};
    };
    using Clock = bool (*)(Stamp&) noexcept;
    static constexpr std::uint64_t bucket_ns = 50000000;
    static constexpr std::size_t bucket_count = 16384, depth_limit = 32, slot_count = 128;
    static constexpr std::size_t stage_count = static_cast<std::size_t>(Stage::kCount);
    struct Counter {
        std::uint64_t calls{}, wall{}, cpu{}, exclusive_wall{}, exclusive_cpu{};
    };
    enum class SlotPlanReason : std::size_t {
        kDisabledPolicy,
        kNonHelper,
        kLocalObligation,
        kSkippedHelper,
        kCount
    };
    static constexpr std::size_t slot_plan_reason_count =
        static_cast<std::size_t>(SlotPlanReason::kCount);
    struct Queue {
        std::uint64_t samples{}, requests{}, queued{}, issuing{}, paused{}, awaiting_stop{};
        std::uint64_t ready{}, physical{}, low_epochs{}, preparation{}, max_requests{},
            max_queued{};
    };
    struct Slot {
        std::uint64_t connection{}, pid{}, start_ticks{};
    };
    // Service-visible boundaries only: these are not GPU timestamps. Admission
    // and the transport-owned reply may both precede native issue.
    enum class Lifecycle : std::uint64_t {
        kReceive,
        kAdmit,
        kIssueBegin,
        kIssueEnd,
        kReply,
        kRetire
    };
    struct Event {
        std::uint64_t wall{}, slot{}, kind{}, wire{}, command{}, detail{};
    };
    static_assert(sizeof(Event) == 48);
    // Earlier transport/owner stamps stay outside the ordered lifecycle stream.
    // Retention shares the exact kReceive stamp, slot and wire of that stream.
    struct Intake {
        std::uint64_t frame{}, owner{}, retained{}, slot{}, wire{}, opcode{};
        std::uint64_t first_read{};
    };
    static_assert(sizeof(Intake) == 56);
    static constexpr std::size_t lifecycle_capacity_limit = 4194304;
    struct Bucket {
        std::uint64_t minimum_span_start{}, maximum_span_end{};
        std::array<Counter, stage_count> stages{};
        std::array<Queue, slot_count> queues{};
        // issued, idle, not-ready error, other error; actual returned outcomes.
        std::array<std::uint64_t, 4> issue_outcomes{};
        std::array<std::uint64_t, slot_plan_reason_count> slot_plans{};
    };
    class Scope final {
       public:
        Scope(HostStageMetrics* metrics, Stage stage) noexcept;
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

       private:
        HostStageMetrics* metrics_{};
        std::size_t depth_{};
    };

    explicit HostStageMetrics(Clock clock = system_clock, std::size_t lifecycle_capacity = 0);
    static bool system_clock(Stamp& stamp) noexcept;
    void queue(Slot slot, const Queue& observation) noexcept;
    void issue(std::size_t outcome) noexcept;
    // Counts at the last existing scope timestamp; never reads another clock.
    void slot_plan(SlotPlanReason reason) noexcept;
    void lifecycle(Lifecycle kind, Slot slot, std::uint64_t wire = 0, std::uint64_t command = 0,
                   std::uint64_t detail = 0, std::uint64_t frame_received_ns = 0,
                   std::uint64_t owner_enter_ns = 0, std::uint64_t first_read_ns = 0) noexcept;
    // A trusted transport timestamp captured before an eager reply flush.
    // This cannot rewind stage clocks or convey execution authority.
    void reply_at(std::uint64_t wall, Slot slot, std::uint64_t wire, std::uint64_t command,
                  std::uint64_t detail) noexcept;
    [[nodiscard]] std::size_t event_count() const noexcept {
        return used_events_;
    }
    [[nodiscard]] bool records_intake() const noexcept {
        return !intakes_.empty();
    }
    [[nodiscard]] std::size_t intake_count() const noexcept {
        return used_intakes_;
    }
    void set_idle_enabled(bool enabled) noexcept {
        idle_enabled_ = enabled;
    }
    [[nodiscard]] bool valid() const noexcept {
        return valid_;
    }
    [[nodiscard]] bool publish(const std::string& path) const noexcept;
    [[nodiscard]] static bool fresh_output(const std::string& path) noexcept;
    [[nodiscard]] std::uint64_t origin_ns() const noexcept {
        return origin_;
    }
    [[nodiscard]] const Bucket& bucket(std::size_t index) const {
        return buckets_.at(index);
    }

   private:
    struct Frame {
        Stage stage{};
        Stamp begin{};
        std::uint64_t child_wall{}, child_cpu{};
    };
    friend class Scope;
    bool read(Stamp& stamp) noexcept;
    std::size_t enter(Stage stage) noexcept;
    void leave(std::size_t depth) noexcept;
    Bucket* current(std::uint64_t wall) noexcept;
    std::size_t slot_index(Slot slot) noexcept;
    void add(std::uint64_t& destination, std::uint64_t value) noexcept;
    Clock clock_;
    std::vector<Bucket> buckets_;
    std::vector<Event> events_;
    std::vector<Intake> intakes_;
    std::array<Frame, depth_limit> stack_{};
    std::array<Slot, slot_count> slots_{};
    std::size_t depth_{}, used_buckets_{}, used_slots_{};
    std::size_t used_events_{};
    std::size_t used_intakes_{};
    std::uint64_t origin_{}, last_wall_{}, last_cpu_{}, clock_reads_{};
    bool valid_{true}, idle_enabled_{};
};

// Owner-thread, bounded observation of ordinary compute admissions. All times
// are host CLOCK_MONOTONIC times, not GPU start/stop timestamps. A positive
// submitted depth includes completed work not yet collected by the owner.
class IssueDepthMetrics final {
   public:
    using Clock = bool (*)(std::uint64_t&) noexcept;
    using Slot = HostStageMetrics::Slot;
    static constexpr std::uint64_t bucket_ns = 200000000;
    static constexpr std::size_t bucket_capacity = 4096, entry_capacity = 32768,
                                 slot_capacity = 128;
    static constexpr std::size_t event_capacity_limit = 524288;
    static constexpr std::size_t selection_capacity_limit = 524288;
    enum class SelectionClass : std::uint64_t {
        kTrackedReady,
        kTrackedNotReady,
        kScopeExternal,
        kUnknown
    };
    struct Selection {
        std::uint64_t connection{}, observation_generation{}, observed_ns{}, pending_sequence{};
        SelectionClass classification{};
    };
    struct SelectionConfig {
        std::size_t capacity{};
    };
    struct Event {
        std::uint64_t connection{}, sequence{}, opcode{}, admitted_ns{};
        std::uint64_t first_observed_ns{}, matched_observed_ns{}, authorized_ns{};
        std::uint64_t issue_begin_ns{}, issue_end_ns{}, retired_ns{}, plan_serial{};
        std::uint64_t positive_observations{}, plan_positive_observations{};
        // Per-slot call generations, not owner steps or readiness transitions.
        std::uint64_t admission_observation_generation{}, first_positive_observation_generation{};
        std::uint64_t preparation_required{}, service_observed_prepared_ns{};
    };
    static_assert(sizeof(Event) * (event_capacity_limit + entry_capacity) <
                  std::size_t{80} * 1024 * 1024);
    static_assert(sizeof(Event) * (event_capacity_limit + entry_capacity) +
                      sizeof(Selection) * selection_capacity_limit <
                  std::size_t{96} * 1024 * 1024);
    struct Authorization {
        std::uint64_t connection{}, sequence{}, plan_serial{};
        bool is_resume{};
    };
    struct Bucket {
        std::uint64_t covered_ns{}, admitted_not_submitted_ns{}, submitted_not_retired_ns{};
        std::uint64_t issue_call_ns{}, no_admitted_compute_ns{}, no_inflight_compute_ns{};
        std::uint64_t admitted{}, submitted{}, retired{};
        std::uint64_t observations{}, ready_compute{}, not_ready_compute{}, other_pending{};
        std::uint64_t no_selected_command{}, unknown{};
        std::uint64_t ready_without_inflight{}, maximum_observation_gap_ns{};
    };
    enum class Failure {
        kNone,
        kClock,
        kCapacity,
        kIdentity,
        kTransition,
        kUnsupported,
        kUnclosed
    };
    explicit IssueDepthMetrics(Clock clock = system_clock, std::size_t event_capacity = 0);
    IssueDepthMetrics(Clock clock, std::size_t event_capacity, SelectionConfig selections);
    static bool system_clock(std::uint64_t& now) noexcept;
    void admit(Slot slot, std::uint64_t sequence, std::uint32_t opcode,
               bool requires_preparation = false) noexcept;
    // Called only after successful preparation of the exact deferred admission.
    // This owner observation does not assert that the command is dependency-ready.
    void prepared(std::uint64_t connection, std::uint64_t sequence) noexcept;
    // Untracked commands are non-compute; they never contribute to depth.
    void begin_issue(std::uint64_t connection, std::uint64_t sequence, bool is_resume) noexcept;
    void end_issue(std::uint64_t connection, std::uint64_t sequence, bool is_successful) noexcept;
    void retired(std::uint64_t connection, std::uint64_t sequence, bool is_successful) noexcept;
    // Reuses an existing snapshot/pending result. Counts are instantaneous
    // observations, never readiness duration or an interpolated lower bound.
    void observe(Slot slot, std::uint64_t pending, bool has_ready_work, bool has_unknown_state,
                 std::uint64_t plan_serial = 0) noexcept;
    // Events are opt-in. The serial is passed explicitly through one issue plan;
    // authorization consumes that plan's positive observation, never a cached grant.
    std::uint64_t begin_plan() noexcept;
    void authorized(Authorization authorization) noexcept;
    void invalidate(Failure failure) noexcept;
    void finish() noexcept;
    [[nodiscard]] bool valid() const noexcept {
        return failure_ == Failure::kNone;
    }
    [[nodiscard]] Failure failure() const noexcept {
        return failure_;
    }
    [[nodiscard]] std::uint64_t origin_ns() const noexcept {
        return origin_;
    }
    [[nodiscard]] const Bucket& bucket(std::size_t index) const {
        return buckets_.at(index);
    }
    [[nodiscard]] std::size_t event_count() const noexcept {
        return events_.size();
    }
    [[nodiscard]] bool records_events() const noexcept {
        return event_capacity_ != 0;
    }
    [[nodiscard]] const Event& event(std::size_t index) const {
        return events_.at(index);
    }
    [[nodiscard]] std::size_t event_storage_bytes() const noexcept {
        return (events_.capacity() + active_events_.capacity()) * sizeof(Event);
    }
    [[nodiscard]] std::size_t selection_count() const noexcept {
        return selections_.size();
    }
    [[nodiscard]] const Selection& selection(std::size_t index) const {
        return selections_.at(index);
    }
    [[nodiscard]] std::size_t selection_storage_bytes() const noexcept {
        return selections_.capacity() * sizeof(Selection);
    }
    [[nodiscard]] bool publish(const std::string& path) const noexcept;

   private:
    enum class State { kAdmitted, kIssuing, kSubmitted };
    struct Entry {
        std::uint64_t connection{}, sequence{};
        std::uint32_t opcode{};
        State state{};
    };
    struct Identity {
        Slot slot;
        std::uint64_t last_admitted{}, admitted{}, submitted{}, retired{}, last_observation{};
        std::uint64_t observation_generation{};
    };
    struct OpcodeCount {
        std::uint32_t opcode{};
        std::uint64_t admitted{};
    };
    Entry* find(std::uint64_t connection, std::uint64_t sequence) noexcept;
    Event* active_event(const Entry* entry) noexcept;
    Identity* identity(Slot slot) noexcept;
    Identity* identity(std::uint64_t connection) noexcept;
    bool record_selection(const Identity& owner, std::uint64_t pending,
                          SelectionClass classification) noexcept;
    Bucket* advance() noexcept;
    Clock clock_;
    std::vector<Bucket> buckets_;
    std::vector<Entry> entries_;
    std::vector<Event> active_events_, events_;
    std::vector<Selection> selections_;
    std::array<Identity, slot_capacity> identities_{};
    std::array<OpcodeCount, 16> opcodes_{};
    std::size_t used_slots_{}, used_buckets_{};
    std::uint64_t origin_{}, begin_{}, last_{}, clock_reads_{};
    std::uint64_t unsubmitted_{}, submitted_{}, issuing_{};
    std::uint64_t plan_serial_{};
    std::size_t event_capacity_{};
    std::size_t selection_capacity_{};
    Failure failure_{Failure::kNone};
    bool is_finished_{};
};

}  // namespace glimmer::app
