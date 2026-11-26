#include "host_stage_metrics.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
using Metrics = glimmer::app::HostStageMetrics;
using Stage = Metrics::Stage;
std::size_t position{};
bool fails{};
std::array<Metrics::Stamp, 5> stamps{
    {{100000000, 0}, {100000100, 100}, {100000200, 200}, {100000300, 250}, {100000500, 350}}};
bool clock_value(Metrics::Stamp& stamp) noexcept {
    if (fails || position == stamps.size()) return false;
    stamp = stamps[position++];
    return true;
}
void expect(bool value) {
    if (!value) throw std::runtime_error("Host-stage metric assertion failed");
}
template <typename Metric>
std::string published(const Metric& metrics) {
    char name[] = "/tmp/glimmer-host-intake-test-XXXXXX";
    const auto* directory = ::mkdtemp(name);
    expect(directory != nullptr);
    const auto output = std::string{directory} + "/metrics.json";
    expect(metrics.publish(output));
    std::ifstream file(output);
    std::string text{std::istreambuf_iterator<char>{file}, {}};
    file.close();
    expect(::unlink(output.c_str()) == 0 && ::rmdir(directory) == 0);
    return text;
}
void cases() {
    { const Metrics::Scope disabled(nullptr, Stage::kStep); }
    expect(position == 0);
    Metrics metrics(clock_value);
    {
        const Metrics::Scope outer(&metrics, Stage::kStep);
        { const Metrics::Scope inner(&metrics, Stage::kSnapshot); }
        metrics.queue({1, 20, 30}, {.requests = 2, .queued = 1, .ready = 1});
        metrics.issue(0);
        const auto clock_reads = position;
        metrics.slot_plan(Metrics::SlotPlanReason::kSkippedHelper);
        metrics.slot_plan(Metrics::SlotPlanReason::kLocalObligation);
        expect(position == clock_reads);
    }
    expect(metrics.valid());
    const auto& bucket = metrics.bucket(0);
    const auto& parent = bucket.stages[static_cast<std::size_t>(Stage::kStep)];
    const auto& child = bucket.stages[static_cast<std::size_t>(Stage::kSnapshot)];
    expect(parent.calls == 1 && parent.wall == 400 && parent.cpu == 250 &&
           parent.exclusive_wall == 300 && parent.exclusive_cpu == 200);
    expect(child.calls == 1 && child.wall == 100 && child.cpu == 50 && child.exclusive_wall == 100);
    expect(bucket.queues[0].samples == 1 && bucket.queues[0].max_requests == 2 &&
           bucket.issue_outcomes[0] == 1);
    expect(bucket.minimum_span_start == 100000100 && bucket.maximum_span_end == 100000500);
    expect(bucket.slot_plans == std::array<std::uint64_t, 4>{0, 0, 1, 1});
    char name[] = "/tmp/glimmer-host-stage-test-XXXXXX";
    const auto* directory = ::mkdtemp(name);
    expect(directory != nullptr);
    const auto output = std::string{directory} + "/metrics.json";
    expect(Metrics::fresh_output(output));
    expect(metrics.publish(output));
    expect(!metrics.publish(output) && !Metrics::fresh_output(output));
    std::ifstream file(output);
    const std::string text{std::istreambuf_iterator<char>{file}, {}};
    expect(text.find("\"valid\":true") != std::string::npos &&
           text.find("\"span_end_bucket\"") != std::string::npos &&
           text.find("\"slot_plan_schema\":1") != std::string::npos &&
           text.find("\"slot_plans\":[0,0,1,1]") != std::string::npos);
    file.close();
    expect(::unlink(output.c_str()) == 0 && ::rmdir(directory) == 0);
    fails = true;
    { const Metrics::Scope invalid(&metrics, Stage::kStep); }
    expect(!metrics.valid());
    const auto previous = position;
    { const Metrics::Scope ignored(&metrics, Stage::kStep); }
    metrics.queue({2, 20, 30}, {});
    expect(position == previous);
}
void lifecycle_cases() {
    using Kind = Metrics::Lifecycle;
    position = 0;
    fails = false;
    {
        Metrics disabled(clock_value);
        const auto before = position;
        disabled.lifecycle(Kind::kReceive, {1, 2, 3}, 1);
        expect(position == before && disabled.event_count() == 0 && disabled.valid() &&
               !disabled.records_intake() && disabled.intake_count() == 0);
    }
    position = 0;
    {
        Metrics bounded(clock_value, 2);
        bounded.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, 100000001);
        bounded.lifecycle(Kind::kReply, {1, 2, 3}, 1);
        expect(bounded.valid() && bounded.event_count() == 2);
        bounded.lifecycle(Kind::kRetire, {1, 2, 3});
        expect(!bounded.valid() && bounded.event_count() == 2 && bounded.intake_count() == 1);
    }
    position = 0;
    {
        Metrics identity(clock_value, 2);
        identity.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, 100000001);
        identity.lifecycle(Kind::kReply, {1, 2, 4}, 1);
        expect(!identity.valid() && identity.event_count() == 1);
    }
    position = 0;
    {
        Metrics regression(clock_value, 2);
        regression.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, 100000001);
        position = 0;
        regression.lifecycle(Kind::kReply, {1, 2, 3}, 1);
        expect(!regression.valid() && regression.event_count() == 1);
    }
    for (const auto wall : {std::uint64_t{0}, std::uint64_t{99999999}, std::uint64_t{100000050},
                            std::uint64_t{100000150}, std::uint64_t{100000300}}) {
        position = 0;
        Metrics timestamp(clock_value, 2);
        timestamp.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, 100000001);
        timestamp.reply_at(wall, {1, 2, 3}, 1, 0, 1);
        expect(timestamp.valid() == (wall == 100000150));
    }
}
void intake_cases() {
    using Kind = Metrics::Lifecycle;
    position = 0;
    fails = false;
    {
        Metrics metrics(clock_value, 2);
        metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 7, 0, (std::uint64_t{2} << 32) | 29, 100000010,
                          100000050, 100000001);
        expect(position == 2 && metrics.valid() && metrics.records_intake() &&
               metrics.event_count() == 1 && metrics.intake_count() == 1);
        // The second frame/owner can precede the first retention: no backwards
        // timestamp is appended to the lifecycle stream, and no extra clock runs.
        metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 8, 0, 30, 100000020, 100000060, 100000002);
        expect(position == 3 && metrics.valid() && metrics.event_count() == 2 &&
               metrics.intake_count() == 2);
        const auto text = published(metrics);
        expect(text.find("[100000100,0,0,7,0,8589934621]") != std::string::npos &&
               text.find("[100000200,0,0,8,0,30]") != std::string::npos &&
               text.find("\"intake\":{\"schema\":2,") != std::string::npos &&
               text.find("[100000010,100000050,100000100,0,7,29,100000001]") != std::string::npos &&
               text.find("[100000020,100000060,100000200,0,8,30,100000002]") != std::string::npos);
    }
    constexpr std::array<std::array<std::uint64_t, 2>, 6> boundaries{{{0, 0},
                                                                      {0, 100000050},
                                                                      {100000010, 0},
                                                                      {100000060, 100000050},
                                                                      {100000010, 100000101},
                                                                      {100000100, 100000100}}};
    for (const auto& pair : boundaries) {
        position = 0;
        Metrics metrics(clock_value, 1);
        metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, pair[0], pair[1], 100000001);
        const bool expected = pair[0] == 100000100;
        expect(metrics.valid() == expected && metrics.event_count() == (expected ? 1 : 0) &&
               metrics.intake_count() == (expected ? 1 : 0));
        if (!expected) {
            const auto before = position;
            metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 2, 0, 29, 100000010, 100000050, 100000001);
            expect(position == before && metrics.event_count() == 0 && metrics.intake_count() == 0);
        }
    }
    for (const auto kind :
         {Kind::kAdmit, Kind::kIssueBegin, Kind::kIssueEnd, Kind::kReply, Kind::kRetire}) {
        position = 0;
        Metrics metrics(clock_value, 1);
        metrics.lifecycle(kind, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, 100000001);
        expect(!metrics.valid() && metrics.event_count() == 0 && metrics.intake_count() == 0);
    }
    position = 0;
    {
        Metrics metrics(clock_value, 1);
        metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 0, 0, 29, 100000010, 100000050, 100000001);
        expect(!metrics.valid() && metrics.event_count() == 0 && metrics.intake_count() == 0);
        expect(published(metrics).find("\"valid\":false") != std::string::npos);
    }
    for (const auto first :
         {std::uint64_t{0}, std::uint64_t{100000011}, std::uint64_t{100000010}}) {
        position = 0;
        Metrics metrics(clock_value, 1);
        metrics.lifecycle(Kind::kReceive, {1, 2, 3}, 1, 0, 29, 100000010, 100000050, first);
        const bool expected = first == 100000010;
        expect(position == 2 && metrics.valid() == expected &&
               metrics.event_count() == (expected ? 1 : 0) &&
               metrics.intake_count() == (expected ? 1 : 0));
    }
    position = 0;
    {
        Metrics metrics(clock_value, 1);
        metrics.lifecycle(Kind::kAdmit, {1, 2, 3}, 1, 0, 29, 0, 0, 100000001);
        expect(!metrics.valid() && metrics.event_count() == 0 && metrics.intake_count() == 0);
    }
}
using Depth = glimmer::app::IssueDepthMetrics;
std::uint64_t depth_now{100000000};
std::uint64_t depth_reads{};
bool depth_clock(std::uint64_t& now) noexcept {
    ++depth_reads;
    now = depth_now;
    return true;
}
void depth_cases() {
    constexpr Depth::Slot first{1, 20, 30}, second{2, 21, 31};
    depth_now = 100000000;
    Depth metrics(depth_clock);
    depth_now = 120000000;
    metrics.admit(first, 1, 1);
    depth_now = 130000000;
    metrics.observe(first, 1, false, false);
    depth_now = 140000000;
    metrics.observe(first, 1, true, false);
    metrics.begin_issue(1, 1, false);
    depth_now = 160000000;
    metrics.end_issue(1, 1, true);
    depth_now = 180000000;
    metrics.admit(second, 1, 2);  // Sequence numbers are connection-local.
    depth_now = 200000000;
    metrics.observe(second, 1, true, false);
    metrics.begin_issue(2, 1, false);
    depth_now = 220000000;
    metrics.end_issue(2, 1, true);
    depth_now = 240000000;
    metrics.retired(1, 1, true);
    depth_now = 260000000;
    metrics.retired(2, 1, true);
    depth_now = 300000000;
    metrics.observe(first, 77, true, false);  // Non-compute pending work is not ready compute.
    metrics.observe(first, 0, true, false);   // Snapshot/pending mismatch is explicitly unknown.
    metrics.observe(first, 0, false, false);
    metrics.finish();
    const auto& a = metrics.bucket(0);
    const auto& b = metrics.bucket(1);
    expect(metrics.valid() && a.covered_ns == 100000000 && b.covered_ns == 100000000);
    expect(a.admitted_not_submitted_ns == 60000000 && b.admitted_not_submitted_ns == 20000000);
    expect(a.submitted_not_retired_ns == 40000000 && b.submitted_not_retired_ns == 80000000);
    expect(a.issue_call_ns == 20000000 && b.issue_call_ns == 20000000);
    expect(a.no_admitted_compute_ns == 20000000 && b.no_admitted_compute_ns == 40000000);
    expect(a.no_inflight_compute_ns == 40000000 && b.no_inflight_compute_ns == 40000000);
    expect(a.admitted == 2 && a.submitted == 1 && b.submitted == 1 && b.retired == 2);
    expect(a.ready_compute == 1 && a.not_ready_compute == 1 && a.observations == 2 &&
           a.ready_without_inflight == 1 && b.ready_compute == 1 && b.ready_without_inflight == 0 &&
           b.other_pending == 1 && b.unknown == 1 && b.observations == 4 &&
           b.maximum_observation_gap_ns == 160000000);
    for (const auto* bucket : {&a, &b})
        expect(bucket->observations == bucket->ready_compute + bucket->not_ready_compute +
                                           bucket->other_pending + bucket->unknown +
                                           bucket->no_selected_command);
    expect(published(metrics).find("\"valid\":true") != std::string::npos);
    for (unsigned int failure = 0; failure < 8; ++failure) {
        depth_now = 100000000;
        Depth invalid(depth_clock);
        invalid.admit(first, 1, 1);
        if (failure == 0) invalid.admit(first, 1, 1);
        if (failure == 1) invalid.admit({1, 21, 30}, 2, 1);
        if (failure == 2) invalid.end_issue(1, 1, true);
        if (failure == 3) invalid.begin_issue(1, 1, true);
        if (failure == 4) invalid.retired(1, 1, true);
        if (failure == 5) depth_now -= 1;
        if (failure == 6) depth_now += Depth::bucket_capacity * Depth::bucket_ns;
        invalid.finish();  // Case 7 has an unclosed admission.
        expect(!invalid.valid() && published(invalid).find("\"valid\":false") != std::string::npos);
    }
    depth_now = 100000000;
    Depth bounded(depth_clock);
    for (std::uint64_t sequence = 1; sequence <= Depth::entry_capacity + 1; ++sequence)
        bounded.admit(first, sequence, 1);
    expect(bounded.failure() == Depth::Failure::kCapacity);
    Depth excluded(depth_clock);
    excluded.begin_issue(1, 55, false);
    excluded.end_issue(1, 55, true);
    excluded.retired(1, 55, true);
    excluded.finish();
    expect(excluded.valid() && excluded.bucket(0).submitted == 0);
}
void issue_event_cases() {
    constexpr Depth::Slot first{1, 20, 30}, second{2, 21, 31};
    depth_now = 100000000;
    Depth disabled(depth_clock);
    const auto reads = depth_reads;
    expect(disabled.begin_plan() == 0);
    disabled.authorized({1, 1, 1, false});
    disabled.prepared(1, 1);
    expect(depth_reads == reads && !disabled.records_events() && disabled.event_count() == 0 &&
           disabled.event_storage_bytes() == 0);
    disabled.finish();
    const auto disabled_output = published(disabled);
    expect(disabled_output.find("\"schema\":1") != std::string::npos &&
           disabled_output.find("\"events\"") == std::string::npos);

    Depth metrics(depth_clock, 3);
    depth_now = 110000000;
    metrics.admit(first, 1, 1);
    depth_now = 120000000;
    metrics.admit(second, 1, 2);
    depth_now = 125000000;
    metrics.observe(first, 1, false, false);
    depth_now = 130000000;
    metrics.observe(first, 1, true, false);  // Generic observation cannot authorize.
    depth_now = 140000000;
    auto serial = metrics.begin_plan();
    metrics.observe(first, 1, true, false, serial);
    depth_now = 145000000;
    serial = metrics.begin_plan();
    metrics.observe(first, 1, true, false, serial);
    depth_now = 150000000;
    metrics.authorized({1, 1, serial, false});
    depth_now = 155000000;
    metrics.begin_issue(1, 1, false);
    depth_now = 160000000;
    metrics.end_issue(1, 1, true);
    const auto submit = [&](Depth::Slot slot, std::uint64_t sequence) {
        const auto current = metrics.begin_plan();
        metrics.observe(slot, sequence, true, false, current);
        depth_now += 1000000;
        metrics.authorized({slot.connection, sequence, current, false});
        depth_now += 1000000;
        metrics.begin_issue(slot.connection, sequence, false);
        depth_now += 1000000;
        metrics.end_issue(slot.connection, sequence, true);
    };
    depth_now = 170000000;
    submit(second, 1);
    depth_now = 200000000;
    metrics.retired(1, 1, true);  // Swap the last active row into the first position.
    depth_now = 210000000;
    metrics.admit(first, 2, 1);
    submit(first, 2);
    depth_now = 250000000;
    metrics.retired(2, 1, true);
    depth_now = 260000000;
    metrics.retired(1, 2, true);
    metrics.finish();
    expect(metrics.valid() && metrics.records_events() && metrics.event_count() == 3);
    const auto& event = metrics.event(0);
    expect(event.connection == 1 && event.sequence == 1 && event.opcode == 1 &&
           event.admitted_ns == 110000000 && event.first_observed_ns == 130000000 &&
           event.matched_observed_ns == 145000000 && event.authorized_ns == 150000000 &&
           event.issue_begin_ns == 155000000 && event.issue_end_ns == 160000000 &&
           event.retired_ns == 200000000 && event.plan_serial == 2 &&
           event.positive_observations == 3 && event.plan_positive_observations == 2 &&
           event.admission_observation_generation == 0 &&
           event.first_positive_observation_generation == 2 && !event.preparation_required &&
           !event.service_observed_prepared_ns);
    expect(metrics.event(1).connection == 2 && metrics.event(1).sequence == 1 &&
           metrics.event(2).connection == 1 && metrics.event(2).sequence == 2 &&
           metrics.event(1).admission_observation_generation == 0 &&
           metrics.event(1).first_positive_observation_generation == 1 &&
           metrics.event(2).admission_observation_generation == 4 &&
           metrics.event(2).first_positive_observation_generation == 5);
    const auto document = published(metrics);
    expect(document.find("\"schema\":4") != std::string::npos &&
           document.find("\"complete\":true") != std::string::npos &&
           document.find("\"ready_duration\":false,\"gpu_timestamps\":false") !=
               std::string::npos &&
           document.find(
               "\"admission_observation_generation\",\"first_positive_observation_generation\"") !=
               std::string::npos &&
           document.find("\"preparation_required\",\"service_observed_prepared_ns\"") !=
               std::string::npos);

    // Every existing observation category advances only its own slot generation.
    // Multiple admissions can share a generation; observations are not owner steps.
    Depth generations(depth_clock, 3);
    generations.admit(first, 1, 1);
    generations.admit(first, 2, 1);
    generations.admit(second, 1, 2);
    const auto before_observations = depth_reads;
    generations.observe(first, 0, false, false);
    generations.observe(first, 99, true, false);
    generations.observe(first, 0, false, true);
    generations.observe(first, 1, false, false);
    expect(depth_reads == before_observations + 4);
    const auto complete = [&](Depth::Slot slot, std::uint64_t sequence) {
        const auto current = generations.begin_plan();
        generations.observe(slot, sequence, true, false, current);
        generations.authorized({slot.connection, sequence, current, false});
        generations.begin_issue(slot.connection, sequence, false);
        generations.end_issue(slot.connection, sequence, true);
        generations.retired(slot.connection, sequence, true);
    };
    complete(first, 1);  // Swaps the still-unobserved second slot's active row.
    complete(second, 1);
    complete(first, 2);
    generations.finish();
    expect(generations.valid() && generations.event_count() == 3 &&
           generations.event(0).first_positive_observation_generation == 5 &&
           generations.event(1).first_positive_observation_generation == 1 &&
           generations.event(2).first_positive_observation_generation == 6 &&
           generations.event(2).admission_observation_generation == 0);

    for (unsigned int failure = 0; failure < 14; ++failure) {
        depth_now = 100000000;
        Depth invalid(depth_clock, 1);
        invalid.admit(first, 1, 1);
        const auto plan = invalid.begin_plan();
        invalid.observe(first, 1, true, false, failure == 0 ? 0 : plan);
        if (failure == 1) (void)invalid.begin_plan();                    // Stale positive plan.
        if (failure == 2) invalid.admit(first, 2, 1);                    // Wrong tracked sequence.
        if (failure == 3) invalid.begin_issue(1, 1, false);              // Missing authorization.
        if (failure == 4) invalid.observe(first, 1, true, false, plan);  // Duplicate plan.
        invalid.authorized({1, failure == 2 ? 2U : 1U, failure == 5 ? 0 : plan, failure == 6});
        if (failure == 7) invalid.authorized({1, 1, plan, false});
        if (failure == 8) depth_now -= 1;
        invalid.begin_issue(1, 1, false);
        invalid.end_issue(1, 1, failure != 9);
        if (failure != 10) invalid.retired(1, 1, failure != 11);
        if (failure == 12) {
            invalid.admit(second, 1, 2);
            const auto next = invalid.begin_plan();
            invalid.observe(second, 1, true, false, next);
            invalid.authorized({2, 1, next, false});
            invalid.begin_issue(2, 1, false);
            invalid.end_issue(2, 1, true);
            invalid.retired(2, 1, true);  // Completed capacity is one, not rolling.
        }
        if (failure == 13) invalid.admit({1, 99, 30}, 2, 1);
        invalid.finish();
        expect(!invalid.valid() &&
               published(invalid).find("\"complete\":false") != std::string::npos);
        expect(invalid.event_count() == (failure >= 12 ? 1U : 0U));
    }
    depth_now = 100000000;
    Depth oversized(depth_clock, Depth::event_capacity_limit + 1);
    expect(oversized.failure() == Depth::Failure::kCapacity &&
           oversized.event_storage_bytes() == 0);
    Depth production(depth_clock, Depth::event_capacity_limit);
    expect(production.valid() && production.event_storage_bytes() < std::size_t{80} * 1024 * 1024);
    production.authorized({1, 77, production.begin_plan(), false});  // Non-compute is excluded.
    production.finish();
    expect(production.valid() && production.event_count() == 0);
}
void preparation_event_cases() {
    constexpr Depth::Slot first{1, 20, 30}, second{2, 21, 31};
    depth_now = 100000000;
    Depth metrics(depth_clock, 2);
    metrics.admit(first, 1, 11, true);
    metrics.admit(second, 1, 11);
    metrics.observe(first, 0, false, false);
    metrics.observe(first, 1, false, false);
    depth_now += 100;
    const auto before_prepared = depth_reads;
    metrics.prepared(1, 1);
    expect(depth_reads == before_prepared + 1);
    const auto prepared_ns = depth_now;
    depth_now += 100;
    for (const auto slot : {second, first}) {
        const auto plan = metrics.begin_plan();
        metrics.observe(slot, 1, true, false, plan);
        metrics.authorized({slot.connection, 1, plan, false});
        metrics.begin_issue(slot.connection, 1, false);
        metrics.end_issue(slot.connection, 1, true);
        metrics.retired(slot.connection, 1, true);
    }
    metrics.finish();
    expect(metrics.valid() && metrics.event_count() == 2 &&
           !metrics.event(0).preparation_required &&
           !metrics.event(0).service_observed_prepared_ns &&
           metrics.event(1).preparation_required == 1 &&
           metrics.event(1).service_observed_prepared_ns == prepared_ns &&
           metrics.event(1).admitted_ns < prepared_ns &&
           prepared_ns < metrics.event(1).first_observed_ns &&
           metrics.event(1).first_positive_observation_generation == 3);
    for (unsigned int failure = 0; failure < 8; ++failure) {
        depth_now = 100000000;
        Depth invalid(depth_clock, 2);
        invalid.admit(first, 1, failure == 7 ? 30 : 11, failure != 4);
        invalid.admit(second, 1, 11);  // Same sequence in another slot is not this admission.
        if (failure == 0) invalid.observe(first, 1, true, false);  // Missing marker.
        if (failure == 1) invalid.prepared(1, 1);                  // Duplicate marker.
        if (failure == 2) invalid.prepared(99, 1);                 // Foreign slot.
        if (failure == 3) invalid.prepared(2, 1);                  // Nondeferred other slot.
        if (failure == 4) invalid.observe(first, 1, true, false);  // Late nondeferred marker.
        if (failure == 5) --depth_now;                             // Backwards clock.
        if (failure == 6) invalid.prepared(1, 2);                  // Wrong sequence.
        invalid.prepared(1, 1);
        expect(!invalid.valid());
        invalid.finish();
        expect(published(invalid).find("\"complete\":false") != std::string::npos);
    }
}
void selection_cases() {
    using Class = Depth::SelectionClass;
    constexpr Depth::Slot first{1, 20, 30}, second{2, 21, 31};
    for (const bool enabled : {false, true}) {
        depth_now = 100000000;
        Depth metrics(depth_clock, 2, Depth::SelectionConfig{enabled ? 5U : 0U});
        metrics.admit(first, 1, 11);
        metrics.admit(second, 1, 11);
        const auto observe = [&](Depth::Slot slot, std::uint64_t pending, bool ready,
                                 std::uint64_t plan = 0) {
            ++depth_now;
            const auto reads = depth_reads;
            metrics.observe(slot, pending, ready, false, plan);
            expect(depth_reads == reads + 1);  // No journal-specific clock read.
        };
        observe(first, 0, false);
        observe(first, 77, true);
        observe(first, 1, false);
        observe(first, 1, false);  // Preserve repeated samples, not a duration span.
        observe(second, 0, false);
        for (const auto slot : {second, first}) {
            const auto plan = metrics.begin_plan();
            observe(slot, 1, true, plan);
            metrics.authorized({slot.connection, 1, plan, false});
            metrics.begin_issue(slot.connection, 1, false);
            metrics.end_issue(slot.connection, 1, true);
            metrics.retired(slot.connection, 1, true);
        }
        metrics.finish();
        expect(metrics.valid() && metrics.event_count() == 2 &&
               metrics.event(0).first_positive_observation_generation == 2 &&
               metrics.event(1).first_positive_observation_generation == 5 &&
               metrics.selection_count() == (enabled ? 5U : 0U));
        const auto& bucket = metrics.bucket(0);
        expect(bucket.observations == 7 && bucket.no_selected_command == 2 &&
               bucket.ready_compute == 2 && bucket.not_ready_compute == 2 &&
               bucket.other_pending == 1 && bucket.unknown == 0);
        const auto text = published(metrics);
        if (!enabled) {
            expect(metrics.selection_storage_bytes() == 0 &&
                   text.find("\"schema\":4") != std::string::npos &&
                   text.find("\"selections\"") == std::string::npos &&
                   text.find("\"observation_generation\":") == std::string::npos);
            continue;
        }
        const auto& external = metrics.selection(0);
        expect(external.connection == 1 && external.observation_generation == 2 &&
               external.observed_ns == 100000002 && external.pending_sequence == 77 &&
               external.classification == Class::kScopeExternal);
        for (const auto index : {1U, 2U}) {
            const auto& row = metrics.selection(index);
            expect(row.connection == 1 && row.observation_generation == index + 2 &&
                   row.observed_ns == 100000002 + index && row.pending_sequence == 1 &&
                   row.classification == Class::kTrackedNotReady);
        }
        for (const auto index : {0U, 1U}) {
            const auto& event = metrics.event(index);
            const auto& selected = metrics.selection(index + 3);
            expect(selected.connection == event.connection &&
                   selected.pending_sequence == event.sequence &&
                   selected.observed_ns == event.first_observed_ns &&
                   selected.observation_generation == event.first_positive_observation_generation &&
                   selected.classification == Class::kTrackedReady);
        }
        expect(
            text.find("\"schema\":5") != std::string::npos &&
            text.find("\"fields\":[\"connection\",\"observation_generation\",\"observed_ns\","
                      "\"pending_sequence\",\"classification\"],\"classifications\":["
                      "\"tracked_ready\",\"tracked_not_ready\",\"scope_external\",\"unknown\"]") !=
                std::string::npos &&
            text.find("\"selections\":{\"capacity\":5,\"complete\":true") != std::string::npos &&
            text.find("\"retired\":1,\"observation_generation\":5") != std::string::npos &&
            text.find("\"retired\":1,\"observation_generation\":2") != std::string::npos &&
            text.find("[1,2,100000002,77,2]") != std::string::npos);
    }
    Depth unknown(depth_clock, 1, Depth::SelectionConfig{3});
    unknown.admit(first, 1, 11);
    unknown.observe(first, 77, true, true);
    unknown.observe(first, 0, true, false);
    const auto plan = unknown.begin_plan();
    unknown.observe(first, 1, true, false, plan);
    unknown.authorized({1, 1, plan, false});
    unknown.begin_issue(1, 1, false);
    unknown.observe(first, 1, true, false);  // Pending no longer denotes an admitted entry.
    unknown.end_issue(1, 1, true);
    unknown.retired(1, 1, true);
    unknown.finish();
    expect(unknown.valid() && unknown.selection_count() == 3 &&
           unknown.selection(0).classification == Class::kUnknown &&
           unknown.selection(1).observation_generation == 3 &&
           unknown.selection(2).classification == Class::kUnknown &&
           unknown.bucket(0).unknown == 3);
    Depth bounded(depth_clock, 1, Depth::SelectionConfig{1});
    bounded.admit(first, 1, 11);
    bounded.observe(first, 77, true, false);
    bounded.observe(first, 0, false, false);  // Omitted zeros consume no storage.
    expect(bounded.valid());
    bounded.observe(first, 78, true, false);
    const auto reads = depth_reads;
    bounded.observe(first, 1, true, false);
    expect(bounded.failure() == Depth::Failure::kCapacity && bounded.selection_count() == 1 &&
           depth_reads == reads);
    bounded.finish();
    expect(published(bounded).find("\"selections\":{\"capacity\":1,\"complete\":false") !=
           std::string::npos);
    Depth foreign(depth_clock, 1, Depth::SelectionConfig{2});
    foreign.admit(first, 1, 11);
    foreign.observe({1, 99, 30}, 1, true, false);
    expect(foreign.failure() == Depth::Failure::kIdentity && !foreign.selection_count());
    for (const auto capacities :
         {std::array<std::size_t, 2>{0, 1},
          std::array<std::size_t, 2>{1, Depth::selection_capacity_limit + 1}}) {
        Depth invalid(depth_clock, capacities[0], Depth::SelectionConfig{capacities[1]});
        expect(invalid.failure() == Depth::Failure::kCapacity && !invalid.event_storage_bytes() &&
               !invalid.selection_storage_bytes());
    }
    Depth maximum(depth_clock, Depth::event_capacity_limit,
                  Depth::SelectionConfig{Depth::selection_capacity_limit});
    expect(maximum.valid() && maximum.selection_storage_bytes() == 20971520 &&
           maximum.event_storage_bytes() + maximum.selection_storage_bytes() <
               std::size_t{96} * 1024 * 1024);
    maximum.finish();
    expect(maximum.valid());
}
}  // namespace
int main() {
    try {
        cases();
        lifecycle_cases();
        intake_cases();
        depth_cases();
        issue_event_cases();
        preparation_event_cases();
        selection_cases();
        return EXIT_SUCCESS;
    } catch (...) {
        return EXIT_FAILURE;
    }
}
