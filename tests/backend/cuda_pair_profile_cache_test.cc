#include "internal/cuda_pair_profile_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
constexpr std::string_view kRuntime = "independent-pair-test-runtime";
constexpr std::uint64_t kMaximumDuration = 3600ULL * 1000 * 1000 * 1000;
std::atomic<std::uint64_t> now_ns{1};
std::atomic<std::uint64_t> clock_calls{};
CudaPairProfileCache* observed_pair{};
std::atomic<bool> clock_observed_unlocked{};

std::uint64_t clock_ns() noexcept {
    ++clock_calls;
    if (observed_pair) clock_observed_unlocked = observed_pair->snapshot().is_valid;
    return now_ns.load();
}

struct ClockLatch {
    const std::uint64_t value;
    std::promise<void> entered_promise, release_promise;
    std::future<void> entered{entered_promise.get_future()};
    std::future<void> released{release_promise.get_future()};

    explicit ClockLatch(std::uint64_t timestamp) : value(timestamp) {}
};

thread_local ClockLatch* held_clock{};

std::uint64_t controlled_clock() noexcept {
    if (!held_clock) return clock_ns();
    ++clock_calls;
    held_clock->entered_promise.set_value();
    held_clock->released.wait();
    return held_clock->value;
}

enum class ObserverCallback { kBegin, kFinish };

std::future<void> hold_callback(ClockLatch& latch,
                                std::shared_ptr<CudaExecutionObserver> observation,
                                ObserverCallback callback) {
    return std::async(std::launch::async, [&latch, observation = std::move(observation), callback] {
        held_clock = &latch;
        if (callback == ObserverCallback::kBegin)
            observation->begin();
        else
            observation->finish(true);
        held_clock = nullptr;
    });
}

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

CudaProfilePlatform platform() {
    return {{std::byte{1}, std::byte{2}}, 12060, 12060, 9};
}

CudaProfileGeometry geometry() {
    return {{128, 1, 1}, {64, 1, 1}, 128, Kind::kPreemptive};
}

struct Original {
    CudaProfilePlatform platform{::platform()};
    std::string source{"first source"};
    std::vector<std::byte> binary{std::byte{1}, std::byte{0}, std::byte{2}};
    std::string name{"first"};
    CudaProfileGeometry launch{geometry()};
    std::uint64_t duration{100};
};

std::array<Original, 2> originals() {
    Original second;
    second.source = "second source";
    second.name = "second";
    second.duration = 300;
    return {Original{}, std::move(second)};
}

CudaProfileConfig single_config(std::uint32_t samples = 2) {
    return {128, std::size_t{8} * 1024 * 1024, 0, samples, {2}, {Kind::kPreemptive, Kind::kSliced}};
}

CudaProfileFunction bind_single(CudaProfileCache& cache, const Original& input) {
    return cache.bind(input.platform, input.source, input.binary, input.name);
}

void train_single(CudaProfileCache& cache, const Original& input,
                  const CudaProfileConfig& config = single_config()) {
    const auto function = bind_single(cache, input);
    expect(static_cast<bool>(function), "original fixture binds exact full code identity");
    const auto calls = (1 + config.workers.size() * config.kinds.size()) *
                       (config.warmup_calls + config.samples_per_candidate);
    for (std::size_t attempt = 0; attempt <= calls; ++attempt) {
        auto choice = cache.choose(function, input.launch, 2, config.kinds);
        expect(choice.was_selected, "single training reserves a real observer");
        if (choice.is_trained) return;
        expect(static_cast<bool>(choice.observation), "single training cannot inject totals");
        now_ns = 10;
        choice.observation->begin();
        now_ns = 10 + (choice.workers ? 1000 : input.duration);
        choice.observation->finish(true);
    }
    throw std::runtime_error("single fixture failed to finish its bounded candidate set");
}

std::shared_ptr<CudaProfileCache> qualified_singles(std::span<const Original> inputs,
                                                    std::uint32_t samples = 2) {
    const auto config = single_config(samples);
    CudaProfileCache learning(config, clock_ns);
    for (const auto& input : inputs) train_single(learning, input, config);
    const auto archive = learning.encode(kRuntime);
    expect(archive.has_value(), "encode genuine completed original and private measurements");
    auto result = std::make_shared<CudaProfileCache>(config, clock_ns, false);
    expect(result->restore(*archive, kRuntime), "restore complete singles into readonly cache");
    for (const auto& input : inputs) {
        const auto measured =
            result->original_measurement(bind_single(*result, input), input.launch);
        expect(measured && measured->samples == samples &&
                   measured->total_ns == samples * input.duration && measured->identity &&
                   measured->identity->source == input.source &&
                   measured->identity->name == input.name && measured->geometry == input.launch,
               "independent original totals and full identity are exact readonly evidence");
    }
    return result;
}

void resign(std::vector<std::byte>& bytes) {
    expect(bytes.size() >= 8, "archive checksum has a complete slot");
    std::uint64_t digest = 14695981039346656037ULL;
    for (const auto byte : std::span{bytes}.first(bytes.size() - 8))
        digest = (digest ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
    for (std::size_t index = 0; index != 8; ++index)
        bytes[bytes.size() - 8 + index] = static_cast<std::byte>((digest >> (8 * index)) & 255U);
}

void write_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 4, "literal u32 field bounds");
    for (std::size_t index = 0; index != 4; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
}

struct U64Patch {
    std::size_t offset;
    std::uint64_t value;
};

void write_u64(std::vector<std::byte>& bytes, U64Patch field) {
    expect(field.offset <= bytes.size() && bytes.size() - field.offset >= 8,
           "literal u64 field bounds");
    for (std::size_t index = 0; index != 8; ++index)
        bytes[field.offset + index] = static_cast<std::byte>((field.value >> (8 * index)) & 255U);
}

void original_measurement_qualification() {
    const auto inputs = originals();
    CudaProfileCache learning(single_config(), clock_ns);
    const auto learning_key = bind_single(learning, inputs[0]);
    expect(!learning.original_measurement(learning_key, inputs[0].launch),
           "learning original is unavailable before training");
    train_single(learning, inputs[0]);
    expect(!learning.original_measurement(learning_key, inputs[0].launch),
           "even a fully trained learning cache cannot authorize paired profiling");
    auto pending_input = inputs[1];
    const auto pending = bind_single(learning, pending_input);
    auto lease = learning.choose(pending, pending_input.launch, 2, single_config().kinds);
    const auto partial = learning.encode(kRuntime);
    expect(lease.observation && partial, "retain one incomplete original beside completed data");
    CudaProfileCache partial_restore(single_config(), clock_ns, false);
    expect(partial_restore.restore(*partial, kRuntime), "complete-only single archive restores");
    expect(!partial_restore.original_measurement(bind_single(partial_restore, pending_input),
                                                 pending_input.launch),
           "incomplete original cannot become a qualified pair leg through restoration");

    const auto singles = qualified_singles(inputs);
    const auto key = bind_single(*singles, inputs[0]);
    const auto measured = singles->original_measurement(key, inputs[0].launch);
    expect(measured && measured->identity->platform == inputs[0].platform &&
               measured->identity->binary ==
                   std::string(reinterpret_cast<const char*>(inputs[0].binary.data()),
                               inputs[0].binary.size()),
           "original view retains full platform and binary including embedded zero bytes");
    const auto before = singles->encode(kRuntime);
    const auto calls = clock_calls.load();
    for (unsigned int change = 0; change != 7; ++change) {
        auto identity = *measured->identity;
        if (change == 0) identity.platform.uuid[15] = std::byte{9};
        if (change == 1) ++identity.platform.driver_version;
        if (change == 2) ++identity.platform.toolkit_version;
        if (change == 3) ++identity.platform.generator_abi;
        if (change == 4) identity.source += 'x';
        if (change == 5) identity.binary[1] = 'x';
        if (change == 6) identity.name += 'x';
        expect(!singles->original_measurement(identity, inputs[0].launch),
               "archive resolver cannot invent or bind a differing original identity");
    }
    for (unsigned int change = 0; change != 4; ++change) {
        auto launch = inputs[0].launch;
        if (change == 0) ++launch.grid[2];
        if (change == 1) ++launch.block[1];
        if (change == 2) ++launch.shared_memory_bytes;
        if (change == 3) launch.kind = Kind::kSliced;
        expect(!singles->original_measurement(key, launch),
               "every original launch field remains part of qualified identity");
    }
    expect(!singles->original_measurement(learning_key, inputs[0].launch) &&
               clock_calls.load() == calls && singles->encode(kRuntime) == before,
           "foreign token and readonly misses cannot call clocks or change the singles archive");
    auto worker = std::async(std::launch::async, [&] {
        for (unsigned int repeat = 0; repeat != 32; ++repeat) {
            const auto view = singles->original_measurement(key, inputs[0].launch);
            if (!view || view->total_ns != 200 || view->samples != 2) return false;
        }
        return true;
    });
    expect(worker.get(), "qualified immutable originals are safe on another CPU thread");
    const auto child = fork();
    expect(child >= 0, "fork original measurement ownership control");
    if (child == 0)
        _exit(!singles->original_measurement(key, inputs[0].launch) &&
                      !singles->original_measurement(*measured->identity, inputs[0].launch)
                  ? 0
                  : 1);
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "both original view overloads reject inherited process state before locking");
    expect(singles->encode(kRuntime) == before,
           "identity and thread/fork controls preserve the existing singles archive format");
}

CudaPairProfileConfig pair_config() {
    return {16, 8, std::size_t{8} * 1024 * 1024, 0, 2};
}

CudaPairProfileDomain domain() {
    return {CudaPairProfileDomainKind::kDistinctContexts, {7, 9}};
}

CudaPairProfileLimits limits() {
    return {2, 8, 1024, 65536, 16, 65536};
}

std::array<CudaPairProfileCandidate, 2> catalog() {
    const CudaPairProfileVariant first{Kind::kPreemptive, 2, 16, 4, 256};
    const CudaPairProfileVariant second{Kind::kSliced, 4, 24, 4, 512};
    auto alternate_first = first;
    auto alternate_second = second;
    alternate_first.workers = 4;
    alternate_second.workers = 2;
    return {CudaPairProfileCandidate{{first, second}},
            CudaPairProfileCandidate{{alternate_first, alternate_second}}};
}

std::array<CudaPairProfileKernel, 2> bind_pair(CudaPairProfileCache& pairs,
                                               CudaProfileCache& singles,
                                               const std::array<Original, 2>& inputs) {
    std::array<CudaPairProfileKernel, 2> result;
    for (std::size_t index = 0; index != result.size(); ++index)
        result[index] = pairs.bind(bind_single(singles, inputs[index]), inputs[index].launch);
    expect(result[0] && result[1], "pair tokens come from qualified original observations");
    return result;
}

void observe(CudaPairProfileReservation& reserved, std::array<std::uint64_t, 2> durations) {
    expect(static_cast<bool>(reserved), "pair training reserves exactly two observer leases");
    now_ns = 100;
    reserved.observations[0]->begin();
    now_ns = 110;
    reserved.observations[1]->begin();
    // Deliver finish callbacks in clock order, independent of which leg is faster.
    const std::array ends{100 + durations[0], 110 + durations[1]};
    const auto first = ends[0] <= ends[1] ? std::size_t{0} : std::size_t{1};
    now_ns = ends[first];
    reserved.observations[first]->finish(true);
    now_ns = ends[1 - first];
    reserved.observations[1 - first]->finish(true);
}

template <typename Durations>
void train_pair(CudaPairProfileCache& pairs, const std::array<CudaPairProfileKernel, 2>& kernels,
                std::span<const CudaPairProfileCandidate> candidates, Durations durations,
                CudaPairProfileDomain arrangement = domain(),
                CudaPairProfileLimits capacity = limits()) {
    for (unsigned int attempt = 0; attempt != 64; ++attempt) {
        if (pairs.choose(kernels, arrangement, capacity, candidates).is_trained) return;
        auto reserved = pairs.reserve(kernels, arrangement, capacity, candidates);
        expect(reserved && reserved.candidate < candidates.size() &&
                   reserved.configuration == candidates[reserved.candidate],
               "reservation identifies the exact first incomplete catalog member");
        observe(reserved, durations(reserved.candidate, attempt));
    }
    throw std::runtime_error("bounded pair fixture did not finish its catalog");
}

void checked_sampling_outcomes() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    CudaPairProfileCache pairs(singles, config, clock_ns);
    const auto key = bind_single(*singles, inputs[0]);
    const auto calls = clock_calls.load();
    expect(pairs.bind_checked({}, inputs[0].launch).status == CudaPairProfileBindStatus::kInvalid,
           "invalid binding is not storage exhaustion or qualified data");
    auto missing = inputs[0].launch;
    ++missing.grid[0];
    expect(pairs.bind_checked(key, missing).status == CudaPairProfileBindStatus::kMissingOriginal,
           "absent qualified geometry is not a transient active reservation");
    auto tiny = config;
    tiny.max_bytes = 1;
    CudaPairProfileCache no_identity_room(singles, tiny, clock_ns);
    expect(no_identity_room.bind_checked(key, inputs[0].launch).status ==
               CudaPairProfileBindStatus::kCapacity,
           "known identity storage exhaustion is explicit");
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    auto exact_identity_budget = config;
    exact_identity_budget.max_bytes = pairs.snapshot().retained_bytes;
    CudaPairProfileCache no_pair_room(singles, exact_identity_budget, clock_ns);
    const auto limited = bind_pair(no_pair_room, *singles, inputs);
    expect(no_pair_room.reserve_checked(limited, domain(), limits(), candidates).status ==
                   CudaPairProfileReserveStatus::kCapacity &&
               !no_pair_room.snapshot().active && !no_pair_room.snapshot().pairs,
           "pair metadata exhaustion cannot masquerade as busy sampling");
    expect(pairs.reserve_checked({}, domain(), limits(), candidates).status ==
                   CudaPairProfileReserveStatus::kInvalid &&
               pairs.reserve_checked(kernels, domain(), limits(), {}).status ==
                   CudaPairProfileReserveStatus::kInvalid &&
               clock_calls.load() == calls,
           "checked validation and capacity results never invoke a measurement clock");
    auto first = pairs.reserve_checked(kernels, domain(), limits(), candidates);
    expect(first.status == CudaPairProfileReserveStatus::kReserved && first.reservation &&
               pairs.reserve_checked(kernels, domain(), limits(), candidates).status ==
                   CudaPairProfileReserveStatus::kBusy &&
               pairs.snapshot().active == 1,
           "live joint reservation is a retryable busy outcome, never another sample");
    observe(first.reservation, {50, 70});
    expect(pairs.reserve_checked(kernels, domain(), limits(), candidates).status ==
                   CudaPairProfileReserveStatus::kBusy &&
               !pairs.snapshot().active,
           "finished but retained observers remain transient bounded-storage owners");
    first.reservation = {};
    train_pair(pairs, kernels, candidates,
               [](auto, auto) { return std::array<std::uint64_t, 2>{50, 70}; });
    const auto trained = pairs.reserve_checked(kernels, domain(), limits(), candidates);
    expect(trained.status == CudaPairProfileReserveStatus::kTrained && !trained.reservation &&
               pairs.snapshot().trained_pairs == 1 && !pairs.snapshot().active,
           "fully trained data is explicit, not a failed allocation or tuning permission");
}

void expect_serial(const CudaPairProfileChoice& choice, std::size_t first, bool is_trained,
                   const char* message) {
    expect(!choice.is_shared && choice.first_original == first && choice.is_trained == is_trained,
           message);
}

void exact_scores_reversal_and_readonly() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    const auto singles_before = singles->encode(kRuntime);
    auto config = pair_config();
    config.warmup_calls = 1;
    CudaPairProfileCache pairs(singles, config, clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    const auto calls = clock_calls.load();
    const auto cold = pairs.choose(kernels, domain(), limits(), candidates);
    expect_serial(cold, 0, false, "cold choose is original serial shorter-first, never learning");
    expect(cold.original[0] == CudaPairProfileMeasurement{200, 2} &&
               cold.original[1] == CudaPairProfileMeasurement{600, 2} &&
               pairs.snapshot().pairs == 0 && clock_calls.load() == calls,
           "cold original oracle is exact and does not reserve work or consult clocks");
    observed_pair = &pairs;
    for (unsigned int attempt = 0; attempt != 6; ++attempt) {
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(reserved && reserved.candidate == attempt / 3,
               "warmup and sample reservations complete one candidate before the next");
        const std::array<std::uint64_t, 2> durations =
            attempt % 3 == 0 ? std::array<std::uint64_t, 2>{3000, 4000}
            : attempt < 3    ? std::array<std::uint64_t, 2>{80, 240}
                             : std::array<std::uint64_t, 2>{200, 100};
        observe(reserved, durations);
        if (attempt != 5)
            expect_serial(pairs.choose(kernels, domain(), limits(), candidates), 0, false,
                          "partial catalog cannot borrow a previously completed candidate");
    }
    observed_pair = nullptr;
    const auto choice = pairs.choose(kernels, domain(), limits(), candidates);
    expect(choice.is_shared && choice.is_trained && choice.candidate == 1 &&
               choice.configuration == candidates[1] && choice.normalized_speed == 3.5L &&
               choice.original[0] == CudaPairProfileMeasurement{200, 2} &&
               choice.original[1] == CudaPairProfileMeasurement{600, 2} &&
               choice.paired[0] == CudaPairProfileMeasurement{400, 2} &&
               choice.paired[1] == CudaPairProfileMeasurement{200, 2} &&
               pairs.snapshot().samples == 4 && pairs.snapshot().trained_pairs == 1 &&
               clock_observed_unlocked,
           "independent sum 100/200 + 300/100 wins; warmups are not measured samples");
    expect(!pairs.reserve(kernels, domain(), limits(), candidates),
           "completed training never reserves or replays an application operation");

    auto reversed_catalog = candidates;
    for (auto& candidate : reversed_catalog) std::swap(candidate.legs[0], candidate.legs[1]);
    auto reversed_domain = domain();
    std::swap(reversed_domain.labels[0], reversed_domain.labels[1]);
    const std::array reversed_kernels{kernels[1], kernels[0]};
    const auto reversed =
        pairs.choose(reversed_kernels, reversed_domain, limits(), reversed_catalog);
    expect(reversed.is_shared && reversed.candidate == 1 && reversed.first_original == 1 &&
               reversed.configuration == reversed_catalog[1] &&
               reversed.original[0] == choice.original[1] &&
               reversed.original[1] == choice.original[0] &&
               reversed.paired[0] == choice.paired[1] && reversed.paired[1] == choice.paired[0] &&
               reversed.normalized_speed == choice.normalized_speed && pairs.snapshot().pairs == 1,
           "reverse lookup swaps every identity, geometry, configuration, baseline and result leg");
    expect_serial(pairs.choose(reversed_kernels, domain(), limits(), reversed_catalog), 1, false,
                  "reversing functions without their exact domain labels cannot reuse timings");

    const auto archive = pairs.encode(kRuntime);
    expect(archive.has_value(), "completed pair catalog encodes without code execution");
    CudaPairProfileCache restored(singles, config, clock_ns, false);
    expect(restored.restore(*archive, kRuntime), "qualified pair archive restores readonly");
    const auto restored_kernels = bind_pair(restored, *singles, inputs);
    const auto loaded = restored.choose(restored_kernels, domain(), limits(), candidates);
    expect(loaded.is_shared && loaded.configuration == choice.configuration &&
               loaded.original == choice.original && loaded.paired == choice.paired &&
               loaded.normalized_speed == choice.normalized_speed &&
               !restored.reserve(restored_kernels, domain(), limits(), candidates) &&
               restored.snapshot().loaded_pairs == 1 && !restored.snapshot().is_learning &&
               singles->encode(kRuntime) == singles_before,
           "readonly exact pair reuse has no observers and does not mutate single profiles");
}

void thresholds_ties_and_identical_legs() {
    auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    const auto candidates = catalog();
    for (const auto durations :
         {std::array<std::uint64_t, 2>{200, 600}, std::array<std::uint64_t, 2>{400, 1200}}) {
        CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        train_pair(pairs, kernels, std::span{candidates}.first(1),
                   [&](std::size_t, unsigned int) { return durations; });
        expect_serial(pairs.choose(kernels, domain(), limits(), std::span{candidates}.first(1)), 0,
                      true, "sum equal to or below one preserves trained original serial work");
    }
    CudaPairProfileCache tied(singles, pair_config(), clock_ns);
    const auto tied_kernels = bind_pair(tied, *singles, inputs);
    train_pair(tied, tied_kernels, candidates,
               [](std::size_t, unsigned int) { return std::array<std::uint64_t, 2>{200, 100}; });
    expect_serial(tied.choose(tied_kernels, domain(), limits(), candidates), 0, true,
                  "equal best normalized scores do not invent a preferred sharing candidate");

    inputs[1] = inputs[0];
    CudaPairProfileCache identical(singles, pair_config(), clock_ns);
    const auto identical_kernels = bind_pair(identical, *singles, inputs);
    const CudaPairProfileDomain same{CudaPairProfileDomainKind::kSameContext, {7, 7}};
    train_pair(
        identical, identical_kernels, std::span{candidates}.first(1),
        [](std::size_t, unsigned int) { return std::array<std::uint64_t, 2>{100, 50}; }, same);
    const auto exact =
        identical.choose(identical_kernels, same, limits(), std::span{candidates}.first(1));
    expect(exact.is_shared && exact.first_original == 0 &&
               exact.paired[0] == CudaPairProfileMeasurement{200, 2} &&
               exact.paired[1] == CudaPairProfileMeasurement{100, 2},
           "identical leg identities prefer exact catalog orientation and stable serial ties");
    auto swapped = candidates[0];
    std::swap(swapped.legs[0], swapped.legs[1]);
    const std::array reverse{swapped};
    const auto reverse_choice = identical.choose(identical_kernels, same, limits(), reverse);
    expect(reverse_choice.is_shared && reverse_choice.configuration == swapped &&
               reverse_choice.paired[0] == exact.paired[1] &&
               reverse_choice.paired[1] == exact.paired[0],
           "identical kernels still restore asymmetric resource and measurement orientation");
}

void exact_identity_and_catalog_misses() {
    const auto base = originals();
    std::vector<Original> inputs(base.begin(), base.end());
    for (unsigned int change = 0; change != 11; ++change) {
        auto input = base[0];
        if (change == 0) input.platform.uuid[15] = std::byte{8};
        if (change == 1) ++input.platform.driver_version;
        if (change == 2) ++input.platform.toolkit_version;
        if (change == 3) ++input.platform.generator_abi;
        if (change == 4) input.source += 'x';
        if (change == 5) input.binary[1] = std::byte{3};
        if (change == 6) input.name += 'x';
        if (change == 7) ++input.launch.grid[2];
        if (change == 8) ++input.launch.block[1];
        if (change == 9) ++input.launch.shared_memory_bytes;
        if (change == 10) input.launch.kind = Kind::kSliced;
        inputs.push_back(std::move(input));
    }
    const auto singles = qualified_singles(inputs);
    const auto before = singles->encode(kRuntime);
    CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
    const auto kernels = bind_pair(pairs, *singles, base);
    const auto candidates = catalog();
    train_pair(pairs, kernels, candidates, [](std::size_t candidate, unsigned int) {
        return candidate == 0 ? std::array<std::uint64_t, 2>{100, 100}
                              : std::array<std::uint64_t, 2>{200, 100};
    });
    for (std::size_t index = 2; index != inputs.size(); ++index) {
        const auto changed = pairs.bind(bind_single(*singles, inputs[index]), inputs[index].launch);
        expect(static_cast<bool>(changed), "each changed identity has its own real original proof");
        const std::array changed_pair{changed, kernels[1]};
        expect(!pairs.choose(changed_pair, domain(), limits(), candidates).is_trained,
               "exact source/binary/name/platform/geometry drift cannot borrow paired durations");
    }
    for (unsigned int change = 0; change != 6; ++change) {
        auto capacity = limits();
        if (change == 0) capacity.sm_count = 1;
        if (change == 1) ++capacity.blocks_per_sm;
        if (change == 2) ++capacity.threads_per_sm;
        if (change == 3) ++capacity.registers_per_sm;
        if (change == 4) ++capacity.max_workers;
        if (change == 5) ++capacity.shared_memory_bytes_per_sm;
        expect(!pairs.choose(kernels, domain(), capacity, candidates).is_trained,
               "hardware capacity is exact pair identity, not only a permissive fit filter");
    }
    for (unsigned int change = 0; change != 7; ++change) {
        auto changed = candidates;
        if (change == 0) std::swap(changed[0], changed[1]);
        if (change == 1) changed[0].legs[0].kind = Kind::kSliced;
        if (change == 2) changed[0].legs[0].workers = 6;
        if (change == 3) ++changed[0].legs[0].registers_per_thread;
        if (change == 4) ++changed[0].legs[0].occupancy_blocks_per_sm;
        if (change == 5) ++changed[0].legs[0].static_shared_memory_bytes;
        if (change == 6) std::swap(changed[0].legs[0], changed[0].legs[1]);
        expect(!pairs.choose(kernels, domain(), limits(), changed).is_trained,
               "ordered complete resource catalog cannot alias a trained or partial catalog");
    }
    auto changed_domain = domain();
    ++changed_domain.labels[0];
    expect(!pairs.choose(kernels, changed_domain, limits(), candidates).is_trained,
           "logical domain arrangement labels separate otherwise identical pairs");
    changed_domain = {CudaPairProfileDomainKind::kSameContext, {7, 7}};
    expect(!pairs.choose(kernels, changed_domain, limits(), candidates).is_trained,
           "same-context and distinct-context execution arrangements never share measurements");
    expect(!pairs.choose(kernels, domain(), limits(), std::span{candidates}.first(1)).is_trained,
           "a strict subset catalog cannot borrow a complete larger catalog");
    CudaPairProfileCache foreign(singles, pair_config(), clock_ns);
    expect(!foreign.reserve(kernels, domain(), limits(), candidates) &&
               !foreign.choose(kernels, domain(), limits(), candidates).is_trained &&
               foreign.snapshot().kernels == 0 && foreign.snapshot().pairs == 0,
           "opaque kernel capabilities are owned by their exact pair cache");
    expect(singles->encode(kRuntime) == before,
           "all pair lookup identities and misses leave genuine singles unchanged");
}

void joint_resources_and_input_bounds() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const std::array candidates{catalog()[0]};
    // Independent arithmetic: blocks=1+2, threads=64*(1+2),
    // registers=64*16+2*64*24, shared=(256+128)+2*(512+128).
    const CudaPairProfileLimits exact{2, 3, 192, 4096, 4, 1664};
    const std::array launches{inputs[0].launch, inputs[1].launch};
    expect(cuda_pair_profile_candidate_fits(launches, exact, candidates[0]),
           "catalog construction and cache validation use the same exact feasibility check");
    for (const auto dimension : {0U, 1U, 2U}) {
        auto invalid = launches;
        invalid[0].block[dimension] = 0;
        expect(!cuda_pair_profile_candidate_fits(invalid, exact, candidates[0]),
               "pure catalog helper rejects every zero block dimension");
    }
    auto invalid = launches;
    invalid[0].grid = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    expect(!cuda_pair_profile_candidate_fits(invalid, exact, candidates[0]),
           "pure catalog helper checks logical grid volume overflow");
    {
        const auto fit = pairs.reserve(kernels, domain(), exact, candidates);
        expect(static_cast<bool>(fit), "every joint per-SM resource equality fits exactly");
    }
    const auto before = pairs.snapshot();
    const auto reject = [&](CudaPairProfileDomain arrangement, CudaPairProfileLimits capacity,
                            std::span<const CudaPairProfileCandidate> choices) {
        const auto state = pairs.snapshot();
        expect(!pairs.reserve(kernels, arrangement, capacity, choices) &&
                   !pairs.choose(kernels, arrangement, capacity, choices).is_shared &&
                   pairs.snapshot().active == state.active &&
                   pairs.snapshot().pairs == state.pairs &&
                   pairs.snapshot().samples == state.samples,
               "invalid joint input rejects before creating a case or reserving either leg");
    };
    for (unsigned int resource = 0; resource != 5; ++resource) {
        auto capacity = exact;
        if (resource == 0) --capacity.blocks_per_sm;
        if (resource == 1) --capacity.threads_per_sm;
        if (resource == 2) --capacity.registers_per_sm;
        if (resource == 3) --capacity.shared_memory_bytes_per_sm;
        if (resource == 4) --capacity.max_workers;
        expect(!cuda_pair_profile_candidate_fits(launches, capacity, candidates[0]),
               "pure selector rejects each independent one-unit resource deficit");
        reject(domain(), capacity, candidates);
    }
    for (unsigned int field = 0; field != 6; ++field) {
        auto capacity = limits();
        if (field == 0) capacity.sm_count = 0;
        if (field == 1) capacity.blocks_per_sm = 0;
        if (field == 2) capacity.threads_per_sm = 0;
        if (field == 3) capacity.registers_per_sm = 0;
        if (field == 4) capacity.max_workers = 0;
        if (field == 5) capacity.shared_memory_bytes_per_sm = 0;
        reject(domain(), capacity, candidates);
    }
    for (const auto kind :
         {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, static_cast<Kind>(99)}) {
        auto changed = candidates;
        changed[0].legs[0].kind = kind;
        reject(domain(), limits(), changed);
    }
    for (const auto workers : {0U, 3U, 18U, std::numeric_limits<unsigned int>::max()}) {
        auto changed = candidates;
        changed[0].legs[0].workers = workers;
        reject(domain(), limits(), changed);
    }
    for (const auto occupancy : {0U, 1U}) {
        auto changed = candidates;
        changed[0].legs[1].occupancy_blocks_per_sm = occupancy;
        reject(domain(), limits(), changed);
    }
    auto overflow = candidates;
    overflow[0].legs[0].static_shared_memory_bytes = std::numeric_limits<std::uint64_t>::max();
    auto huge = limits();
    huge.shared_memory_bytes_per_sm = std::numeric_limits<std::uint64_t>::max();
    reject(domain(), huge, overflow);
    overflow[0].legs[0].workers = 4;
    overflow[0].legs[0].static_shared_memory_bytes = UINT64_MAX / 2;
    reject(domain(), huge, overflow);
    overflow = candidates;
    for (auto& leg : overflow[0].legs) {
        leg.workers = std::numeric_limits<unsigned int>::max();
        leg.registers_per_thread = std::numeric_limits<unsigned int>::max();
        leg.occupancy_blocks_per_sm = std::numeric_limits<unsigned int>::max();
    }
    huge = {1, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT64_MAX};
    reject(domain(), huge, overflow);
    huge.max_workers = static_cast<unsigned int>(INT32_MAX);
    for (auto& leg : overflow[0].legs) leg.workers = static_cast<unsigned int>(INT32_MAX);
    reject(domain(), huge, overflow);
    reject({static_cast<CudaPairProfileDomainKind>(0), {7, 9}}, limits(), candidates);
    reject({CudaPairProfileDomainKind::kSameContext, {7, 9}}, limits(), candidates);
    reject({CudaPairProfileDomainKind::kDistinctContexts, {7, 7}}, limits(), candidates);
    reject({CudaPairProfileDomainKind::kDistinctContexts, {0, 9}}, limits(), candidates);
    reject(domain(), limits(), {});
    const std::array duplicate{candidates[0], candidates[0]};
    reject(domain(), limits(), duplicate);
    std::vector<CudaPairProfileCandidate> over_catalog(9, candidates[0]);
    for (std::size_t index = 0; index != over_catalog.size(); ++index)
        over_catalog[index].legs[0].static_shared_memory_bytes += index;
    reject(domain(), limits(), over_catalog);
    auto launch = inputs[0].launch;
    launch.grid = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    expect(!pairs.bind(bind_single(*singles, inputs[0]), launch),
           "unqualified overflowing launch volume cannot create a pair kernel");
    launch = inputs[0].launch;
    launch.block[1] = 0;
    expect(!pairs.bind(bind_single(*singles, inputs[0]), launch) &&
               pairs.snapshot().samples == before.samples,
           "zero launch volume cannot reserve any joint sample");
}

void invalid_observation_sequences() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    const std::array candidates{catalog()[0]};
    auto config = pair_config();
    config.samples_per_candidate = 1;
    for (unsigned int scenario = 0; scenario != 14; ++scenario) {
        CudaPairProfileCache pairs(singles, config, clock_ns);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "invalid-observation control starts with a real lease");
        auto& first = reserved.observations[0];
        auto& second = reserved.observations[1];
        now_ns = scenario == 12 ? 0 : 100;
        if (scenario != 1 && scenario != 11) first->begin();
        if (scenario == 2) first->begin();
        if (scenario == 7 || scenario == 8) {
            now_ns = 200;
            first->finish(true);
            now_ns = scenario == 7 ? 201 : 200;
            second->begin();
            now_ns = 300;
            second->finish(true);
        } else if (scenario == 10 || scenario == 11) {
            first.reset();
            now_ns = 110;
            second->begin();
            now_ns = 200;
            second->finish(true);
        } else {
            now_ns = scenario == 13 ? 90 : 110;
            if (scenario != 9) second->begin();
            now_ns = scenario == 4   ? 100
                     : scenario == 5 ? 99
                     : scenario == 6 ? 100 + kMaximumDuration + 1
                                     : 200;
            first->finish(scenario != 0);
            if (scenario == 3) first->finish(true);
            now_ns = scenario == 6 ? 110 + kMaximumDuration + 1 : 250;
            second->finish(true);
        }
        reserved.observations = {};
        const auto failed = pairs.snapshot();
        expect(
            failed.samples == 0 && failed.active == 0 && failed.trained_pairs == 0 &&
                failed.cancelled + failed.rejected_samples == 1,
            "failure, duplicate, missing, abandoned or invalid-time observations settle once only");
        expect_serial(pairs.choose(kernels, domain(), limits(), candidates), 0, false,
                      "invalid host intervals cannot become trained pair or GPU-overlap evidence");
        auto retry = pairs.reserve(kernels, domain(), limits(), candidates);
        observe(retry, {200, 100});
        const auto trained = pairs.snapshot();
        expect(
            trained.samples == 1 && trained.trained_pairs == 1 && trained.active == 0,
            "failed sample frees only its measurement reservation and does not advance training");
        retry.observations[0]->begin();
        retry.observations[0]->finish(false);
        retry.observations[1]->finish(false);
        expect(pairs.snapshot().samples == trained.samples &&
                   pairs.snapshot().rejected_samples == trained.rejected_samples &&
                   pairs.snapshot().cancelled == trained.cancelled,
               "callbacks after final sample commit cannot duplicate or retroactively alter it");
    }
    CudaPairProfileCache maximum(singles, config, clock_ns);
    const auto keys = bind_pair(maximum, *singles, inputs);
    auto exact = maximum.reserve(keys, domain(), limits(), candidates);
    observe(exact, {kMaximumDuration, kMaximumDuration});
    expect(maximum.snapshot().samples == 1 && maximum.snapshot().trained_pairs == 1,
           "inclusive one-hour bound accepts two overlapping final-retirement observations");
}

void concurrent_observers_and_process_guards() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.samples_per_candidate = 1;
    CudaPairProfileCache pairs(singles, config, clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const std::array candidates{catalog()[0]};
    auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
    expect(reserved && !pairs.reserve(kernels, domain(), limits(), candidates) &&
               pairs.snapshot().active == 1,
           "a concurrent reservation cannot overwrite either active leg");
    const auto child = fork();
    expect(child >= 0, "fork pair observer ownership control");
    if (child == 0) {
        const auto calls = clock_calls.load();
        reserved.observations[0]->begin();
        reserved.observations[1]->finish(true);
        _exit(!pairs.snapshot().is_valid &&
                      !pairs.reserve(kernels, domain(), limits(), candidates) &&
                      !pairs.choose(kernels, domain(), limits(), candidates).is_trained &&
                      !pairs.encode(kRuntime) && clock_calls.load() == calls
                  ? 0
                  : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited pair methods and observers reject before locks and external clocks");
    observed_pair = &pairs;
    now_ns = 100;
    std::promise<void> first_started;
    auto first_ready = first_started.get_future();
    std::promise<void> second_started;
    auto second_ready = second_started.get_future();
    std::promise<void> first_finished;
    auto first_done = first_finished.get_future();
    auto first_worker = std::async(std::launch::async, [&] {
        reserved.observations[0]->begin();
        first_started.set_value();
        second_ready.wait();
        now_ns = 300;
        reserved.observations[0]->finish(true);
        first_finished.set_value();
    });
    auto second_worker = std::async(std::launch::async, [&] {
        first_ready.wait();
        now_ns = 110;
        reserved.observations[1]->begin();
        second_started.set_value();
        first_done.wait();
        now_ns = 310;
        reserved.observations[1]->finish(true);
    });
    first_worker.get();
    second_worker.get();
    observed_pair = nullptr;
    const auto chosen = pairs.choose(kernels, domain(), limits(), candidates);
    expect(chosen.is_shared && chosen.paired[0] == CudaPairProfileMeasurement{200, 1} &&
               chosen.paired[1] == CudaPairProfileMeasurement{200, 1} && clock_observed_unlocked,
           "two actual host observer lifetimes overlap across threads with clocks outside locks");
    // The CPU model deliberately provides no physical GPU co-residency proof.
    std::array<std::future<bool>, 4> readers;
    for (auto& reader : readers)
        reader = std::async(std::launch::async, [&] {
            for (unsigned int repeat = 0; repeat != 32; ++repeat) {
                const auto value = pairs.choose(kernels, domain(), limits(), candidates);
                if (!value.is_shared || value.paired != chosen.paired) return false;
            }
            return true;
        });
    for (auto& reader : readers)
        expect(reader.get(), "parallel readonly choices preserve exact results");
    auto abandon_domain = domain();
    ++abandon_domain.labels[1];
    auto abandoned = pairs.reserve(kernels, abandon_domain, limits(), candidates);
    expect(static_cast<bool>(abandoned), "reserve separate cancellation control");
    auto destructor =
        std::async(std::launch::async,
                   [lease = std::move(abandoned.observations[0])]() mutable { lease.reset(); });
    destructor.get();
    abandoned.observations[1].reset();
    expect(pairs.snapshot().cancelled == 1 && pairs.snapshot().active == 0,
           "off-thread observer destruction cancels exactly once without SDK cleanup authority");
}

void in_flight_clocks_commit_and_cancel() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.samples_per_candidate = 1;
    const std::array candidates{catalog()[0]};
    {
        CudaPairProfileCache pairs(singles, config, controlled_clock);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "reserve held-begin clock control");
        const std::weak_ptr first_observer = reserved.observations[0];
        const std::weak_ptr second_observer = reserved.observations[1];
        ClockLatch begin{100};
        auto worker = hold_callback(begin, reserved.observations[0], ObserverCallback::kBegin);
        begin.entered.get();
        // Leg zero's clock is still inside user code. Leg one must acquire cache
        // exclusion and finish, but cannot commit against an unpublished start.
        now_ns = 110;
        reserved.observations[1]->begin();
        now_ns = 210;
        reserved.observations[1]->finish(true);
        const auto pending = pairs.snapshot();
        begin.release_promise.set_value();
        worker.get();
        now_ns = 300;
        reserved.observations[0]->finish(true);
        const auto choice = pairs.choose(kernels, domain(), limits(), candidates);
        expect(pending.active == 1 && pending.samples == 0 && pending.rejected_samples == 0 &&
                   choice.is_shared && choice.paired[0] == CudaPairProfileMeasurement{200, 1} &&
                   choice.paired[1] == CudaPairProfileMeasurement{100, 1} &&
                   pairs.snapshot().samples == 1 && pairs.snapshot().active == 0,
               "held begin permits the other leg's callbacks but publishes exactly one complete "
               "sample");
        reserved.observations = {};
        expect(first_observer.expired() && second_observer.expired(),
               "held-begin control releases both observer objects after terminal commit");
    }
    for (const std::size_t first_return : {std::size_t{0}, std::size_t{1}}) {
        CudaPairProfileCache pairs(singles, config, controlled_clock);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "reserve simultaneous-finish clock control");
        now_ns = 100;
        reserved.observations[0]->begin();
        now_ns = 110;
        reserved.observations[1]->begin();
        std::array<ClockLatch, 2> clocks{ClockLatch{300}, ClockLatch{310}};
        auto first_worker =
            hold_callback(clocks[0], reserved.observations[0], ObserverCallback::kFinish);
        clocks[0].entered.get();
        auto second_worker =
            hold_callback(clocks[1], reserved.observations[1], ObserverCallback::kFinish);
        clocks[1].entered.get();
        const auto both_held = pairs.snapshot();
        clocks[first_return].release_promise.set_value();
        if (first_return == 0)
            first_worker.get();
        else
            second_worker.get();
        const auto one_held = pairs.snapshot();
        clocks[1 - first_return].release_promise.set_value();
        if (first_return == 0)
            second_worker.get();
        else
            first_worker.get();
        const auto choice = pairs.choose(kernels, domain(), limits(), candidates);
        expect(both_held.samples == 0 && both_held.active == 1 && one_held.samples == 0 &&
                   one_held.active == 1 && choice.is_shared &&
                   choice.paired[0] == CudaPairProfileMeasurement{200, 1} &&
                   choice.paired[1] == CudaPairProfileMeasurement{200, 1} &&
                   pairs.snapshot().samples == 1 && pairs.snapshot().rejected_samples == 0,
               "both clock-return orders wait for the last in-flight result and commit once");
        reserved.observations[0]->finish(false);
        reserved.observations[1]->finish(false);
        reserved.observations = {};
        expect(
            pairs.snapshot().samples == 1 && pairs.snapshot().cancelled == 0 &&
                pairs.snapshot().rejected_samples == 0,
            "terminal clock callbacks and handle destruction cannot alter the committed outcome");
    }
    for (const auto callback : {ObserverCallback::kBegin, ObserverCallback::kFinish}) {
        CudaPairProfileCache pairs(singles, config, controlled_clock);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "reserve cancellation during external clock");
        const std::weak_ptr first_observer = reserved.observations[0];
        const std::weak_ptr second_observer = reserved.observations[1];
        if (callback == ObserverCallback::kFinish) {
            now_ns = 100;
            reserved.observations[0]->begin();
            now_ns = 110;
            reserved.observations[1]->begin();
        }
        ClockLatch clock{callback == ObserverCallback::kBegin ? 100U : 300U};
        auto worker = hold_callback(clock, reserved.observations[0], callback);
        clock.entered.get();
        reserved.observations[1].reset();
        const auto cancelled = pairs.snapshot();
        clock.release_promise.set_value();
        worker.get();
        reserved.observations[0]->finish(true);
        reserved.observations[0].reset();
        expect(cancelled.cancelled == 1 && cancelled.active == 0 && cancelled.samples == 0 &&
                   pairs.snapshot().cancelled == 1 && pairs.snapshot().rejected_samples == 0 &&
                   pairs.snapshot().samples == 0 && first_observer.expired() &&
                   second_observer.expired(),
               "unfinished other-leg abandonment wins once while a begin or finish clock is held");
        auto retry = pairs.reserve(kernels, domain(), limits(), candidates);
        observe(retry, {200, 100});
        expect(pairs.snapshot().samples == 1 && pairs.snapshot().cancelled == 1,
               "cancelled held-clock storage releases completely for a new valid sample");
    }
    {
        CudaPairProfileCache pairs(singles, config, controlled_clock);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "reserve same-leg in-flight callback control");
        ClockLatch begin{100};
        auto worker = hold_callback(begin, reserved.observations[0], ObserverCallback::kBegin);
        begin.entered.get();
        reserved.observations[0]->finish(true);
        const auto rejected = pairs.snapshot();
        begin.release_promise.set_value();
        worker.get();
        reserved.observations = {};
        expect(rejected.rejected_samples == 1 && rejected.active == 0 &&
                   pairs.snapshot().rejected_samples == 1 && pairs.snapshot().samples == 0 &&
                   pairs.snapshot().cancelled == 0,
               "same-leg finish cannot consume a still-in-flight begin or double-settle rejection");
    }
}

void bounds_eviction_and_pins() {
    const auto base = originals();
    std::vector<Original> inputs(base.begin(), base.end());
    auto third = base[0];
    third.name = "third";
    inputs.push_back(third);
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.max_pairs = 1;
    config.samples_per_candidate = 1;
    CudaPairProfileCache pairs(singles, config, clock_ns);
    auto kernels = bind_pair(pairs, *singles, base);
    expect(!pairs.bind(bind_single(*singles, third), third.launch),
           "two live kernel tokens pin the exact 2*max_pairs kernel quota");
    const std::array candidates{catalog()[0]};
    auto active = pairs.reserve(kernels, domain(), limits(), candidates);
    auto changed = domain();
    ++changed.labels[1];
    expect(active && !pairs.reserve(kernels, changed, limits(), candidates) &&
               pairs.snapshot().pairs == 1 && pairs.snapshot().active == 1,
           "an active pair cannot be overwritten or evicted at capacity");
    observe(active, {200, 100});
    expect(!pairs.reserve(kernels, changed, limits(), candidates),
           "terminal but externally live observers continue pinning their exact pair");
    active.observations = {};
    auto replacement = pairs.reserve(kernels, changed, limits(), candidates);
    expect(replacement && pairs.snapshot().evictions >= 1 && pairs.snapshot().pairs == 1,
           "released inactive pair is evictable without exceeding metadata quota");
    replacement.observations = {};
    kernels = {};
    expect(static_cast<bool>(pairs.bind(bind_single(*singles, third), third.launch)),
           "unborrowed kernel identities can evict after all measurement leases release");
    auto tiny = pair_config();
    tiny.max_bytes = 1;
    CudaPairProfileCache bytes(singles, tiny, clock_ns);
    expect(!bytes.bind(bind_single(*singles, base[0]), base[0].launch) &&
               bytes.snapshot().retained_bytes == 0 && bytes.snapshot().kernels == 0,
           "byte quota rejects before retaining code identities or pair metadata");
    const auto rejects = [&](std::shared_ptr<const CudaProfileCache> original_cache,
                             CudaPairProfileConfig invalid) {
        bool rejected{};
        try {
            CudaPairProfileCache value(std::move(original_cache), invalid, clock_ns);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "invalid pair cache configuration rejects at cold construction");
    };
    rejects({}, pair_config());
    rejects(std::make_shared<CudaProfileCache>(single_config(), clock_ns), pair_config());
    auto cold_singles = std::make_shared<CudaProfileCache>(single_config(), clock_ns, false);
    CudaPairProfileCache cold(cold_singles, pair_config(), clock_ns, false);
    expect(!cold.bind(bind_single(*cold_singles, base[0]), base[0].launch) &&
               !cold.bind(bind_single(*singles, base[0]), base[0].launch) &&
               !cold.reserve({}, domain(), limits(), candidates) &&
               !cold.choose({}, domain(), limits(), candidates).is_shared &&
               cold.snapshot().kernels == 0 && cold.snapshot().pairs == 0 &&
               cold.snapshot().samples == 0,
           "cold or foreign singles cannot seed a readonly pair or fabricate original evidence");
    for (unsigned int field = 0; field != 5; ++field) {
        auto invalid = pair_config();
        if (field == 0) invalid.max_pairs = 0;
        if (field == 1) invalid.max_candidates = 0;
        if (field == 2) invalid.max_bytes = 0;
        if (field == 3) invalid.samples_per_candidate = 0;
        if (field == 4) invalid.warmup_calls = 1001;
        rejects(singles, invalid);
    }
    auto invalid = pair_config();
    invalid.max_pairs = std::numeric_limits<std::size_t>::max();
    rejects(singles, invalid);
    invalid = pair_config();
    invalid.samples_per_candidate = 1001;
    rejects(singles, invalid);
    invalid = pair_config();
    invalid.max_candidates = 65;
    rejects(singles, invalid);
    invalid = pair_config();
    invalid.max_bytes = kCudaPairProfileArchiveMaxBytes + 1;
    rejects(singles, invalid);
}

void maximum_totals_and_exact_score_boundary() {
    auto inputs = originals();
    for (auto& input : inputs) input.duration = kMaximumDuration / 2;
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.samples_per_candidate = 1000;
    const std::array candidates{catalog()[0]};
    for (const bool improves_by_one_ns : {false, true}) {
        CudaPairProfileCache pairs(singles, config, clock_ns);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        for (unsigned int sample = 0; sample != 1000; ++sample) {
            auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
            observe(reserved, {kMaximumDuration - (improves_by_one_ns && sample == 999 ? 1 : 0),
                               kMaximumDuration});
        }
        const auto choice = pairs.choose(kernels, domain(), limits(), candidates);
        expect(choice.is_trained && choice.is_shared == improves_by_one_ns &&
                   pairs.snapshot().samples == 1000 && pairs.snapshot().rejected_samples == 0,
               "bounded large integer totals preserve exact strict score threshold, not rounded "
               "ratios");
        if (improves_by_one_ns)
            expect(
                choice.original[0] == CudaPairProfileMeasurement{kMaximumDuration, 2} &&
                    choice.original[1] == choice.original[0] &&
                    choice.paired[0] ==
                        CudaPairProfileMeasurement{1000 * kMaximumDuration - 1, 1000} &&
                    choice.paired[1] == CudaPairProfileMeasurement{1000 * kMaximumDuration, 1000},
                "maximum-sized totals and one nanosecond improvement survive explicit reporting");
    }
}

void large_cross_candidate_rationals() {
    constexpr std::uint32_t samples = 1000;
    constexpr std::uint64_t original_total = samples * kMaximumDuration;
    constexpr std::uint64_t paired_duration = kMaximumDuration - 1;
    constexpr std::uint64_t paired_total = samples * paired_duration;
    auto inputs = originals();
    for (auto& input : inputs) input.duration = kMaximumDuration;
    // These original facts are trained through all three actual single-cache
    // candidates and restored, not injected into a public measurement record.
    const auto singles = qualified_singles(inputs, samples);
    const auto singles_before = singles->encode(kRuntime);
    auto config = pair_config();
    config.samples_per_candidate = samples;
    const auto candidates = catalog();
    for (unsigned int scenario = 0; scenario != 3; ++scenario) {
        CudaPairProfileCache pairs(singles, config, clock_ns);
        const auto kernels = bind_pair(pairs, *singles, inputs);
        for (unsigned int attempt = 0; attempt != 2 * samples; ++attempt) {
            auto reserved = pairs.reserve(kernels, domain(), limits(), candidates);
            const auto candidate = attempt / samples;
            expect(reserved && reserved.candidate == candidate,
                   "large comparison genuinely measures both complete candidate catalogs");
            std::array<std::uint64_t, 2> durations{paired_duration, paired_duration};
            if (attempt % samples == samples - 1) {
                // Scenario zero places the mathematically stronger candidate
                // second, scenario two first. Scenario one reverses its two
                // denominators, giving exactly equal rational sums.
                const bool skew = scenario == 1 || candidate == (scenario == 0 ? 1U : 0U);
                if (skew) {
                    const auto smaller =
                        scenario == 1 && candidate == 1 ? std::size_t{1} : std::size_t{0};
                    --durations[smaller];
                    ++durations[1 - smaller];
                }
            }
            observe(reserved, durations);
        }
        const auto choice = pairs.choose(kernels, domain(), limits(), candidates);
        expect(choice.is_trained &&
                   choice.original[0] == CudaPairProfileMeasurement{original_total, samples} &&
                   choice.original[1] == choice.original[0] &&
                   pairs.snapshot().samples == std::uint64_t{2} * samples &&
                   pairs.snapshot().rejected_samples == 0,
               "all four original/paired facts carry one thousand real bounded samples");
        // For O>0,T>1, O/(T-1)+O/(T+1)-2O/T equals
        // 2O/[T*(T*T-1)] > 0. This independent algebra avoids reproducing the
        // implementation's 256-bit multiplication or using rounded float scores.
        // Swapping T-1 and T+1 preserves the exact sum, so scenario one is a tie.
        if (scenario == 1) {
            expect_serial(choice, 0, true,
                          "large asymmetric but swapped denominators are an exact candidate tie");
        } else {
            const auto expected_candidate = scenario == 0 ? std::size_t{1} : std::size_t{0};
            expect(choice.is_shared && choice.candidate == expected_candidate &&
                       choice.configuration == candidates[expected_candidate] &&
                       choice.paired[0] == CudaPairProfileMeasurement{paired_total - 1, samples} &&
                       choice.paired[1] == CudaPairProfileMeasurement{paired_total + 1, samples},
                   "near-equal large scores select the algebraic winner in either catalog order");
        }
    }
    expect(singles->encode(kRuntime) == singles_before,
           "large pair comparisons leave the independently trained original archive unchanged");
}

void observers_outlive_cache_owner() {
    const auto inputs = originals();
    const std::array candidates{catalog()[0]};
    auto config = pair_config();
    config.samples_per_candidate = 1;
    for (unsigned int scenario = 0; scenario != 3; ++scenario) {
        auto singles = qualified_singles(inputs);
        const std::weak_ptr original_owner = singles;
        auto pairs = std::make_unique<CudaPairProfileCache>(singles, config, clock_ns);
        auto kernels = bind_pair(*pairs, *singles, inputs);
        auto reserved = pairs->reserve(kernels, domain(), limits(), candidates);
        expect(static_cast<bool>(reserved), "reserve independently retained observer lifetime");
        const std::weak_ptr first_observer = reserved.observations[0];
        const std::weak_ptr second_observer = reserved.observations[1];
        if (scenario != 2) {
            now_ns = 100;
            reserved.observations[0]->begin();
            now_ns = 110;
            reserved.observations[1]->begin();
        }
        pairs.reset();
        singles.reset();
        expect(!original_owner.expired(),
               "live observers retain the cache implementation and readonly original owner");
        const auto foreign_singles = qualified_singles(inputs);
        CudaPairProfileCache foreign(foreign_singles, config, clock_ns);
        expect(!foreign.reserve(kernels, domain(), limits(), candidates) &&
                   !foreign.choose(kernels, domain(), limits(), candidates).is_trained &&
                   foreign.snapshot().kernels == 0 && foreign.snapshot().pairs == 0,
               "surviving kernel state never becomes a capability for another cache owner");
        const auto before = clock_calls.load();
        if (scenario == 0) {
            now_ns = 200;
            reserved.observations[0]->finish(true);
            reserved.observations[0].reset();
            expect(!original_owner.expired() && first_observer.expired(),
                   "one finished leg can release while the other retains the destroyed cache");
            now_ns = 250;
            reserved.observations[1]->finish(true);
            expect(clock_calls.load() == before + 2,
                   "both final-retirement callbacks still execute their clocks after owner "
                   "destruction");
        } else {
            reserved.observations[0].reset();
            reserved.observations[1]->begin();
            reserved.observations[1]->finish(true);
            expect(
                clock_calls.load() == before && !original_owner.expired(),
                "abandonment after owner destruction cancels once without later clock callbacks");
        }
        reserved.observations = {};
        expect(original_owner.expired() && first_observer.expired() && second_observer.expired(),
               "last observer destruction releases all original-owner and observer objects without "
               "a cycle");
        expect(
            kernels[0] && kernels[1] && !foreign.reserve(kernels, domain(), limits(), candidates) &&
                !foreign.choose(kernels, domain(), limits(), candidates).is_trained,
            "expired owner tokens remain foreign even while their immutable code state survives");
        kernels = {};
        const auto fresh = bind_pair(foreign, *foreign_singles, inputs);
        auto healthy = foreign.reserve(fresh, domain(), limits(), candidates);
        observe(healthy, {200, 100});
        expect(foreign.snapshot().samples == 1 && foreign.snapshot().cancelled == 0,
               "stale-owner controls cannot poison a new cache's valid once-only measurement");
    }
}

void finished_leg_release_and_reservation_pins() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.samples_per_candidate = 1;
    CudaPairProfileCache pairs(singles, config, clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    auto reservation = pairs.reserve(kernels, domain(), limits(), candidates);
    now_ns = 100;
    reservation.observations[0]->begin();
    now_ns = 110;
    reservation.observations[1]->begin();
    now_ns = 200;
    reservation.observations[0]->finish(true);
    reservation.observations[0].reset();
    expect(pairs.snapshot().active == 1 && pairs.snapshot().cancelled == 0,
           "retired leg storage may release while the other real host interval remains live");
    now_ns = 250;
    reservation.observations[1]->finish(true);
    expect(pairs.snapshot().samples == 1 && pairs.snapshot().cancelled == 0 &&
               !pairs.reserve(kernels, domain(), limits(), candidates),
           "finished observer still pins its reservation storage until both handles release");
    reservation.observations = {};
    auto next = pairs.reserve(kernels, domain(), limits(), candidates);
    expect(next && next.candidate == 1,
           "dropping finished handles refunds only bounded observer storage, not another sample");
}

enum class LiteralWidth : unsigned int { kByte = 1, kWord = 4, kWide = 8 };

struct LiteralArchive {
    std::vector<std::byte> bytes;
    std::size_t version{}, metric{}, runtime{}, config{}, count{}, pair_begin{}, domain{}, limits{};
    std::array<std::size_t, 2> original_begin{}, original_end{}, original_grid{}, original_total{};
    std::array<std::size_t, 2> candidate{};
    std::size_t catalog_count{};

    std::size_t number(std::uint64_t value, LiteralWidth width = LiteralWidth::kWord) {
        const auto offset = bytes.size();
        for (unsigned int index = 0; index != static_cast<unsigned int>(width); ++index)
            bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 255U));
        return offset;
    }
    std::size_t text(std::string_view value) {
        const auto offset = number(value.size());
        for (const auto byte : value) bytes.push_back(static_cast<std::byte>(byte));
        return offset;
    }
};

LiteralArchive literal_archive(bool reversed = false,
                               const CudaProfileContextConfiguration* configuration = nullptr,
                               const CudaProfileFunctionConfiguration* function = nullptr) {
    LiteralArchive result;
    result.text("Glimmer pair profile archive");
    result.version = result.number(function ? 3 : configuration ? 2 : 1);
    result.metric = result.number(1);
    result.runtime = result.text(kRuntime);
    result.config = result.number(16);
    result.number(8);
    result.number(std::uint64_t{8} * 1024 * 1024, LiteralWidth::kWide);
    result.number(0);
    result.number(2);
    result.count = result.number(1);
    result.pair_begin = result.bytes.size();
    for (std::size_t leg = 0; leg != 2; ++leg) {
        const auto original = reversed ? 1 - leg : leg;
        result.original_begin[leg] = result.bytes.size();
        result.number(1, LiteralWidth::kByte);
        result.number(2, LiteralWidth::kByte);
        for (unsigned int byte = 2; byte != 16; ++byte) result.number(0, LiteralWidth::kByte);
        result.number(12060);
        result.number(12060);
        result.number(9);
        result.text(original == 0 ? "first source" : "second source");
        result.text(std::string_view{"\1\0\2", 3});
        result.text(original == 0 ? "first" : "second");
        result.original_grid[leg] = result.number(128);
        result.number(1);
        result.number(1);
        result.number(64);
        result.number(1);
        result.number(1);
        result.number(128);
        result.number(3);  // Preemptive, independent wire literal.
        if (configuration) {
            result.number(configuration->available);
            for (const auto value : configuration->values)
                result.number(value, LiteralWidth::kWide);
        } else if (function) {
            result.number(0);
            for (unsigned int field = 0; field < 19; ++field) result.number(0, LiteralWidth::kWide);
        }
        if (function) {
            result.number(function->functions);
            result.number(function->cache_preference);
            for (std::size_t role = 0; role < 6; ++role) {
                result.number(function->available[role]);
                for (const auto value : function->values[role])
                    result.number(static_cast<std::uint32_t>(value));
            }
        }
        result.original_total[leg] = result.number(original == 0 ? 200 : 600, LiteralWidth::kWide);
        result.number(2);
        result.original_end[leg] = result.bytes.size();
    }
    result.domain = result.number(2);  // Distinct contexts.
    result.number(reversed ? 9 : 7, LiteralWidth::kWide);
    result.number(reversed ? 7 : 9, LiteralWidth::kWide);
    result.limits = result.number(2);
    result.number(8);
    result.number(1024);
    result.number(65536);
    result.number(16);
    result.number(65536, LiteralWidth::kWide);
    result.catalog_count = result.number(2);
    for (std::size_t candidate = 0; candidate != 2; ++candidate) {
        result.candidate[candidate] = result.bytes.size();
        for (std::size_t leg = 0; leg != 2; ++leg) {
            const auto original = reversed ? 1 - leg : leg;
            result.number(original == 0 ? 3 : 4);  // Preemptive / Sliced.
            result.number((candidate == 0) == (original == 0) ? 2 : 4);
            result.number(original == 0 ? 16 : 24);
            result.number(4);
            result.number(original == 0 ? 256 : 512, LiteralWidth::kWide);
        }
        result.number(0);
        result.number(2);
        result.number(candidate == 1 && !reversed ? 400 : 200, LiteralWidth::kWide);
        result.number(candidate == 1 && reversed ? 400 : 200, LiteralWidth::kWide);
    }
    result.number(0, LiteralWidth::kWide);
    resign(result.bytes);
    return result;
}

void archive_literals_and_hostile_transactions() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    const auto before_singles = singles->encode(kRuntime);
    CudaPairProfileCache learned(singles, pair_config(), clock_ns);
    const auto kernels = bind_pair(learned, *singles, inputs);
    const auto candidates = catalog();
    train_pair(learned, kernels, candidates, [](std::size_t candidate, unsigned int) {
        return candidate == 0 ? std::array<std::uint64_t, 2>{100, 100}
                              : std::array<std::uint64_t, 2>{200, 100};
    });
    const auto archive = learned.encode(kRuntime);
    const auto literal = literal_archive();
    expect(archive && *archive == literal.bytes,
           "independent literal v1 archive binds complete identities, resources, counters and LE "
           "checksum");
    const auto reject = [&](std::span<const std::byte> bytes, std::string_view runtime = kRuntime) {
        CudaPairProfileCache target(singles, pair_config(), clock_ns, false);
        expect(!target.restore(bytes, runtime) && target.snapshot().pairs == 0 &&
                   target.snapshot().kernels == 0 && target.snapshot().retained_bytes == 0 &&
                   target.snapshot().loaded_pairs == 0 &&
                   singles->encode(kRuntime) == before_singles,
               "hostile pair restore is all-or-none and never mutates qualified singles");
        expect(target.restore(literal.bytes, kRuntime),
               "a rejected restore leaves the empty destination reusable for valid exact evidence");
    };
    reject(*archive, "other-runtime");
    reject(*archive, {});
    reject(*archive, std::string_view{"a\0b", 3});
    for (std::size_t size = 0; size != archive->size(); ++size)
        reject(std::span{*archive}.first(size));
    auto corrupt = *archive;
    corrupt[corrupt.size() / 2] ^= std::byte{1};
    reject(corrupt);
    const auto bad32 = [&](std::size_t position, std::uint32_t value) {
        auto bytes = literal.bytes;
        write_u32(bytes, position, value);
        resign(bytes);
        reject(bytes);
    };
    const auto bad64 = [&](U64Patch field) {
        auto bytes = literal.bytes;
        write_u64(bytes, field);
        resign(bytes);
        reject(bytes);
    };
    bad32(0, UINT32_MAX);
    bad32(literal.version, 0);
    bad32(literal.version, 2);
    bad32(literal.metric, 2);
    bad32(literal.runtime, UINT32_MAX);
    bad32(literal.config, 15);
    bad32(literal.config + 4, 7);
    bad64({literal.config + 8, UINT64_MAX});
    bad32(literal.config + 16, 1);
    bad32(literal.config + 20, 1);
    bad32(literal.count, UINT32_MAX);
    bad32(literal.original_begin[0] + 16, 12061);
    bad32(literal.original_begin[0] + 20, 12061);
    bad32(literal.original_begin[0] + 24, 8);
    bad32(literal.original_begin[0] + 28, UINT32_MAX);
    bad32(literal.original_grid[0], 0);
    bad32(literal.original_grid[0] + 28, 99);
    bad64({literal.original_total[0], 201});
    bad32(literal.original_total[0] + 8, 1);
    bad32(literal.domain, 0);
    bad64({literal.domain + 4, 0});
    bad64({literal.domain + 12, 7});
    bad32(literal.limits, 0);
    bad32(literal.limits + 4, 2);
    bad32(literal.limits + 8, 191);
    bad32(literal.limits + 12, 4095);
    bad64({literal.limits + 20, 1663});
    bad32(literal.catalog_count, 0);
    bad32(literal.catalog_count, 9);
    bad32(literal.candidate[0], 0);
    bad32(literal.candidate[0] + 4, 3);
    bad32(literal.candidate[0] + 12, 0);
    bad64({literal.candidate[0] + 16, UINT64_MAX});
    bad32(literal.candidate[0] + 48, 1);
    bad32(literal.candidate[0] + 52, 1);
    bad64({literal.candidate[0] + 56, 0});
    bad64({literal.candidate[0] + 56, 2 * kMaximumDuration + 1});
    corrupt = literal.bytes;
    std::copy_n(corrupt.begin() + static_cast<std::ptrdiff_t>(literal.candidate[0]), 48,
                corrupt.begin() + static_cast<std::ptrdiff_t>(literal.candidate[1]));
    resign(corrupt);
    reject(corrupt);
    for (const bool reverse_duplicate : {false, true}) {
        const auto second = literal_archive(reverse_duplicate);
        auto doubled = literal.bytes;
        doubled.insert(doubled.end() - 8,
                       second.bytes.begin() + static_cast<std::ptrdiff_t>(second.pair_begin),
                       second.bytes.end() - 8);
        write_u32(doubled, literal.count, 2);
        resign(doubled);
        reject(doubled);
        if (!reverse_duplicate) {
            const auto second_start = literal.bytes.size() - 8;
            write_u64(doubled, {second_start + second.domain - second.pair_begin + 12, 10});
            write_u64(doubled, {second_start + second.candidate[1] - second.pair_begin + 56, 0});
            resign(doubled);
            reject(doubled);
        }
    }
    corrupt = literal.bytes;
    corrupt.insert(corrupt.end() - 8, std::byte{0});
    resign(corrupt);
    reject(corrupt);
    std::vector<std::byte> oversized(kCudaPairProfileArchiveMaxBytes + 1);
    reject(oversized);
    CudaPairProfileCache restored(singles, pair_config(), clock_ns, false);
    expect(restored.restore(literal.bytes, kRuntime), "literal archive independently restores");
    const auto retained = restored.encode(kRuntime);
    expect(!restored.restore(literal.bytes, kRuntime) && restored.encode(kRuntime) == retained,
           "nonempty destination cannot be overwritten, even with another valid archive");
    CudaPairProfileCache bound(singles, pair_config(), clock_ns, false);
    const auto pinned = bind_pair(bound, *singles, inputs);
    expect(!bound.restore(literal.bytes, kRuntime) && bound.snapshot().kernels == 2 && pinned[0] &&
               pinned[1],
           "even bound-only destination is nonempty and cannot change token identity");
    const auto reversed_literal = literal_archive(true);
    CudaPairProfileCache reverse(singles, pair_config(), clock_ns, false);
    expect(reverse.restore(reversed_literal.bytes, kRuntime),
           "literal reversed catalog is valid alone");
    const auto reversed_keys = bind_pair(reverse, *singles, inputs);
    expect(reverse.choose(reversed_keys, domain(), limits(), candidates).is_shared,
           "reversed archive restores original caller orientation without another measurement");
    auto different_inputs = inputs;
    ++different_inputs[0].duration;
    const auto changed_singles = qualified_singles(different_inputs);
    CudaPairProfileCache changed(changed_singles, pair_config(), clock_ns, false);
    expect(!changed.restore(literal.bytes, kRuntime) && changed.snapshot().kernels == 0,
           "same source identity with changed actual original totals cannot authorize old paired "
           "scores");
    CudaProfileCache old_format(single_config(), clock_ns, false);
    expect(!old_format.restore(literal.bytes, kRuntime),
           "pair v1 cannot masquerade as single archive");
    CudaPairProfileCache pair_format(singles, pair_config(), clock_ns, false);
    expect(
        before_singles && !pair_format.restore(*before_singles, kRuntime) &&
            singles->encode(kRuntime) == before_singles,
        "existing single archive stays byte-compatible and cannot be reinterpreted as paired data");
}

void context_configuration_archive_identity() {
    auto inputs = originals();
    for (auto& input : inputs) {
        input.launch.context.available = 15;
        input.launch.context.values[0] = 8;
        input.launch.context.values[1] = 1;
        input.launch.context.values[3] = 1024;
    }
    const auto singles = qualified_singles(inputs);
    const auto single_archive = singles->encode(kRuntime);
    const auto literal = literal_archive(false, &inputs[0].launch.context);
    CudaPairProfileCache literal_restored(singles, pair_config(), clock_ns, false);
    expect(literal_restored.restore(literal.bytes, kRuntime) &&
               literal_restored.snapshot().loaded_pairs == 1,
           "independent literal V2 schema resolves canonical configured original fields");
    expect(single_archive &&
               (*single_archive)[4 + std::string_view{"Glimmer profile archive"}.size()] ==
                   std::byte{2},
           "canonical context measurements emit explicit V2 single archives");
    CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    train_pair(pairs, kernels, candidates,
               [](auto, auto) { return std::array<std::uint64_t, 2>{50, 70}; });
    const auto archive = pairs.encode(kRuntime);
    CudaPairProfileCache restored(singles, pair_config(), clock_ns, false);
    expect(archive &&
               (*archive)[4 + std::string_view{"Glimmer pair profile archive"}.size()] ==
                   std::byte{2} &&
               restored.restore(*archive, kRuntime) && restored.snapshot().loaded_pairs == 1,
           "V2 paired archive resolves the same complete configured original measurements");
    auto changed = inputs[0].launch;
    changed.context.values[1] = 2;
    expect(pairs.bind_checked(bind_single(*singles, inputs[0]), changed).status ==
               CudaPairProfileBindStatus::kMissingOriginal,
           "changing an actual context value cannot inherit an older measurement");
    const auto legacy_singles = qualified_singles(originals());
    CudaPairProfileCache legacy(legacy_singles, pair_config(), clock_ns, false);
    expect(!legacy.restore(*archive, kRuntime) && !legacy.snapshot().kernels &&
               !legacy.snapshot().pairs,
           "initial-only legacy originals cannot silently qualify V2 configured paired data");
    std::vector<std::byte> pattern(4 + 8 * inputs[0].launch.context.values.size());
    write_u32(pattern, 0, inputs[0].launch.context.available);
    for (std::size_t index = 0; index < inputs[0].launch.context.values.size(); ++index)
        write_u64(pattern, {4 + 8 * index, inputs[0].launch.context.values[index]});
    // Explicit bind_single above may update code LRU order, not measurements.
    // Freeze after that permitted bookkeeping; archive resolution is readonly.
    const auto before_hostile = singles->encode(kRuntime);
    for (unsigned int field = 0; field < 5; ++field) {
        for (const bool paired : {false, true}) {
            auto corrupted = paired ? *archive : *single_archive;
            const auto found =
                std::search(corrupted.begin(), corrupted.end(), pattern.begin(), pattern.end());
            expect(found != corrupted.end(),
                   "independently locate complete literal context fields");
            const auto offset = static_cast<std::size_t>(found - corrupted.begin());
            if (field == 0) write_u32(corrupted, offset, std::uint32_t{1} << 31);
            if (field == 1) write_u32(corrupted, offset, 3);
            if (field == 2) write_u32(corrupted, offset, 0);
            if (field == 3) write_u64(corrupted, {offset + 4 + 8, 99});
            if (field == 4) write_u64(corrupted, {offset + 4 + std::size_t{8} * 4, 1});
            resign(corrupted);
            CudaPairProfileCache rejected_pair(singles, pair_config(), clock_ns, false);
            CudaProfileCache rejected_single(single_config(), clock_ns, false);
            expect(paired ? !rejected_pair.restore(corrupted, kRuntime) &&
                                !rejected_pair.snapshot().pairs && !rejected_pair.snapshot().kernels
                          : !rejected_single.restore(corrupted, kRuntime) &&
                                !rejected_single.snapshot().functions,
                   "resigned unknown/partial/noncanonical context fields reject transactionally");
        }
    }
    expect(singles->encode(kRuntime) == before_hostile,
           "hostile pair restoration preserves qualified originals and current code ordering");
    auto mixed = inputs;
    mixed[1].launch.context.values[1] = 2;
    const auto mixed_singles = qualified_singles(mixed);
    CudaPairProfileCache mixed_pairs(mixed_singles, pair_config(), clock_ns);
    const auto mixed_kernels = bind_pair(mixed_pairs, *mixed_singles, mixed);
    expect(mixed_pairs.reserve_checked(mixed_kernels,
                                       {CudaPairProfileDomainKind::kSameContext, {7, 7}}, limits(),
                                       candidates)
                       .status == CudaPairProfileReserveStatus::kInvalid &&
               !mixed_pairs.snapshot().pairs,
           "one materialized context cannot claim two unequal complete configurations");
}

void function_configuration_archive_identity() {
    auto inputs = originals();
    CudaProfileFunctionConfiguration function;
    function.functions = 17;        // Original and preemptive, independent role bits.
    function.cache_preference = 2;  // Explicit shared preference, not CACHE_MODE_CA.
    for (const std::size_t role : {std::size_t{0}, std::size_t{4}}) {
        function.available[role] = 1023;
        function.values[role] = {1024, 128, 0, 0, 17, 70, 70, 0, 49152, -1};
    }
    for (auto& input : inputs) input.launch.function = function;
    const auto singles = qualified_singles(inputs);
    const auto single_archive = singles->encode(kRuntime);
    const auto literal = literal_archive(false, nullptr, &function);
    LiteralArchive single_literal;
    single_literal.text("Glimmer profile archive");
    for (const auto word : {3U, 1U}) single_literal.number(word);
    single_literal.text(kRuntime);
    // Independent single configuration: warmup0, samples2, workers[2],
    // families[preemptive3,sliced4], two exact original code identities.
    for (const auto word : {0U, 2U, 1U, 2U, 2U, 3U, 4U, 2U}) single_literal.number(word);
    for (std::size_t leg = 0; leg < 2; ++leg) {
        const auto append = [&](std::size_t first, std::size_t last) {
            const auto bytes = std::span{literal.bytes}.subspan(first, last - first);
            single_literal.bytes.insert(single_literal.bytes.end(), bytes.begin(), bytes.end());
        };
        append(literal.original_begin[leg], literal.original_grid[leg]);
        single_literal.number(1);  // One geometry on this code identity.
        append(literal.original_grid[leg], literal.original_total[leg]);
        single_literal.number(3);  // Original and two private candidates.
        for (unsigned int candidate = 0; candidate < 3; ++candidate) {
            single_literal.number(candidate ? 2 : 0);
            single_literal.number(candidate == 2 ? 4 : 3);
            single_literal.number(candidate ? 2000 : leg ? 600 : 200, LiteralWidth::kWide);
        }
    }
    single_literal.number(0, LiteralWidth::kWide);
    resign(single_literal.bytes);
    CudaProfileCache independent_singles(single_config(), clock_ns, false);
    expect(independent_singles.restore(single_literal.bytes, kRuntime) &&
               independent_singles.snapshot().loaded_cases == 2,
           "independent V3 single literal parses original and private configuration fields");
    CudaPairProfileCache independent(singles, pair_config(), clock_ns, false);
    expect(independent.restore(literal.bytes, kRuntime) && independent.snapshot().loaded_pairs == 1,
           "independent V3 literal resolves signed attributes and separate native roles");
    CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
    const auto kernels = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    train_pair(pairs, kernels, candidates,
               [](auto, auto) { return std::array<std::uint64_t, 2>{50, 70}; });
    const auto archive = pairs.encode(kRuntime);
    CudaPairProfileCache restored(singles, pair_config(), clock_ns, false);
    CudaProfileCache restored_singles(single_config(), clock_ns, false);
    expect(single_archive && archive &&
               (*single_archive)[4 + std::string_view{"Glimmer profile archive"}.size()] ==
                   std::byte{3} &&
               (*archive)[4 + std::string_view{"Glimmer pair profile archive"}.size()] ==
                   std::byte{3} &&
               restored_singles.restore(*single_archive, kRuntime) &&
               restored.restore(*archive, kRuntime),
           "actual V3 single/pair codecs preserve complete function identity");
    auto changed = inputs[0].launch;
    changed.function.cache_preference = 3;
    expect(pairs.bind_checked(bind_single(*singles, inputs[0]), changed).status ==
               CudaPairProfileBindStatus::kMissingOriginal,
           "different successful cache preference cannot inherit another original measurement");
    changed = inputs[0].launch;
    changed.function.values[4][8] += 1024;
    expect(pairs.bind_checked(bind_single(*singles, inputs[0]), changed).status ==
               CudaPairProfileBindStatus::kMissingOriginal,
           "private-family actual settings are not replaced with original-function values");
    const auto legacy_singles = qualified_singles(originals());
    CudaPairProfileCache legacy(legacy_singles, pair_config(), clock_ns, false);
    expect(!legacy.restore(*archive, kRuntime) && !legacy.snapshot().pairs,
           "V1 initial-only function originals cannot qualify V3 configured data");
    std::vector<std::byte> pattern(8 + std::size_t{6} * (4 + 16 * 4));
    write_u32(pattern, 0, function.functions);
    write_u32(pattern, 4, function.cache_preference);
    for (std::size_t role = 0; role < 6; ++role) {
        const auto base = 8 + role * (4 + 16 * 4);
        write_u32(pattern, base, function.available[role]);
        for (std::size_t attribute = 0; attribute < 16; ++attribute)
            write_u32(pattern, base + 4 + attribute * 4,
                      static_cast<std::uint32_t>(function.values[role][attribute]));
    }
    const auto before = singles->encode(kRuntime);
    for (unsigned int field = 0; field < 8; ++field) {
        for (const bool paired : {false, true}) {
            auto corrupted = paired ? *archive : *single_archive;
            const auto found =
                std::search(corrupted.begin(), corrupted.end(), pattern.begin(), pattern.end());
            expect(found != corrupted.end(), "independently locate literal V3 configuration bytes");
            const auto offset = static_cast<std::size_t>(found - corrupted.begin());
            if (field == 0) write_u32(corrupted, offset, 1U << 31);
            if (field == 1) write_u32(corrupted, offset, 16);
            if (field == 2) write_u32(corrupted, offset + 4, 5);
            if (field == 3) write_u32(corrupted, offset + 8, 1022);
            if (field == 4) write_u32(corrupted, offset + 8, 65536);
            if (field == 5) write_u32(corrupted, offset + 12 + std::size_t{9} * 4, 101);
            if (field == 6) write_u32(corrupted, offset + 12 + std::size_t{7} * 4, 2);
            if (field == 7) write_u32(corrupted, offset + 8 + (4 + 16 * 4) + 4, 1);
            resign(corrupted);
            CudaPairProfileCache rejected_pair(singles, pair_config(), clock_ns, false);
            CudaProfileCache rejected_single(single_config(), clock_ns, false);
            expect(paired ? !rejected_pair.restore(corrupted, kRuntime) &&
                                !rejected_pair.snapshot().pairs
                          : !rejected_single.restore(corrupted, kRuntime) &&
                                !rejected_single.snapshot().functions,
                   "resigned unknown/partial/noncanonical V3 state rejects transactionally");
        }
    }
    expect(singles->encode(kRuntime) == before,
           "hostile V3 restoration preserves qualified originals");
}

void incomplete_archive_and_readonly_miss() {
    const auto inputs = originals();
    const auto singles = qualified_singles(inputs);
    const auto before = singles->encode(kRuntime);
    CudaPairProfileCache pairs(singles, pair_config(), clock_ns);
    const auto keys = bind_pair(pairs, *singles, inputs);
    const auto candidates = catalog();
    auto active = pairs.reserve(keys, domain(), limits(), candidates);
    const auto archive = pairs.encode(kRuntime);
    expect(archive.has_value(), "active catalog is omitted, not serialized as a completed pair");
    CudaPairProfileCache readonly(singles, pair_config(), clock_ns, false);
    expect(readonly.restore(*archive, kRuntime) && readonly.snapshot().pairs == 0 &&
               readonly.snapshot().kernels == 0 && readonly.snapshot().loaded_pairs == 0,
           "no active or partial pair facts survive into readonly selection");
    const auto restored_keys = bind_pair(readonly, *singles, inputs);
    expect_serial(readonly.choose(restored_keys, domain(), limits(), candidates), 0, false,
                  "cold restored catalog chooses serial originals rather than private fallback");
    expect(!readonly.reserve(restored_keys, domain(), limits(), candidates) &&
               readonly.snapshot().pairs == 0 && readonly.snapshot().samples == 0,
           "readonly catalog miss cannot begin isolated or application replay");
    observe(active, {200, 100});
    active.observations = {};
    const auto partial = pairs.encode(kRuntime);
    CudaPairProfileCache missing(singles, pair_config(), clock_ns, false);
    expect(partial && missing.restore(*partial, kRuntime) && missing.snapshot().pairs == 0 &&
               singles->encode(kRuntime) == before,
           "a valid partial sample is not a completed catalog and singles remain unchanged");
    expect(!pairs.encode({}) && !pairs.encode(std::string_view{"x\0y", 3}) &&
               !pairs.encode(std::string(4097, 'r')),
           "runtime archive identity is bounded, nonempty and free of embedded terminators");
}
}  // namespace

namespace {
void noninserting_global_lookup() {
    const auto original_pair = originals();
    std::array<Original, 3> inputs{original_pair[0], original_pair[1], original_pair[0]};
    inputs[2].source = "third independent original source";
    inputs[2].name = "third";
    inputs[2].duration = 50;
    auto singles = qualified_singles(inputs);
    auto config = pair_config();
    config.max_pairs = 1;
    CudaPairProfileCache learning(singles, config, clock_ns);
    const auto bound = bind_pair(learning, *singles, original_pair);
    const auto candidates = catalog();
    train_pair(learning, bound, candidates, [](std::size_t index, auto) {
        return index ? std::array<std::uint64_t, 2>{50, 70}
                     : std::array<std::uint64_t, 2>{100, 200};
    });
    const auto archive = learning.encode(kRuntime);
    CudaPairProfileCache readonly(singles, config, clock_ns, false);
    expect(archive && readonly.restore(*archive, kRuntime),
           "one retained measured pair fills its capacity");
    std::array<CudaProfileFunction, 3> functions;
    for (std::size_t index = 0; index < functions.size(); ++index)
        functions[index] = bind_single(*singles, inputs[index]);
    const auto before = readonly.snapshot();
    const auto clocks = clock_calls.load();
    for (unsigned int repeat = 0; repeat < 16; ++repeat) {
        const auto missing =
            readonly.lookup({functions[0], functions[2]}, {inputs[0].launch, inputs[2].launch},
                            domain(), limits(), candidates);
        expect(!missing.is_shared && !missing.is_trained && missing.first_original == 1 &&
                   missing.original[0].total_ns == 200 && missing.original[1].total_ns == 100,
               "cold global lookup retains qualified original means without inserting the third "
               "identity");
        const auto retained =
            readonly.lookup({functions[0], functions[1]}, {inputs[0].launch, inputs[1].launch},
                            domain(), limits(), candidates);
        expect(retained.is_shared && retained.is_trained && retained.candidate == 1,
               "speculative cold lookup cannot evict a measured winning pair");
    }
    const auto after = readonly.snapshot();
    expect(before.kernels == after.kernels && before.pairs == after.pairs &&
               before.retained_bytes == after.retained_bytes &&
               before.evictions == after.evictions && readonly.encode(kRuntime) == archive &&
               clock_calls.load() == clocks,
           "global readonly lookup changes no metadata/archives/reservations and calls no clock");
}

void global_choice_comparisons() {
    const auto clocks = clock_calls.load();
    CudaPairProfileChoice first{.is_shared = true, .is_trained = true, .candidate = 0};
    first.original = {CudaPairProfileMeasurement{100, 1}, CudaPairProfileMeasurement{100, 1}};
    first.paired = first.original;
    auto second = first;
    first.paired[1] = {50, 1};
    first.normalized_speed = -1;
    second.normalized_speed = std::numeric_limits<long double>::infinity();
    expect(cuda_pair_profile_compare_choices(first, second) == 1 &&
               cuda_pair_profile_compare_choices(second, first) == -1 &&
               cuda_pair_profile_compare_choices(first, first) == 0,
           "global choices compare exact facts, not rounded normalized-speed reports");
    constexpr auto maximum = kCudaPairProfileMaxSampleNs * 1000;
    first.original = {CudaPairProfileMeasurement{maximum - 1, 1000},
                      CudaPairProfileMeasurement{maximum - 1, 1000}};
    first.paired = {CudaPairProfileMeasurement{maximum - 2, 1000},
                    CudaPairProfileMeasurement{maximum, 1000}};
    second.original = first.original;
    second.paired = first.original;
    first.normalized_speed = second.normalized_speed = 2;
    expect(cuda_pair_profile_compare_choices(first, second) == 1 &&
               cuda_pair_profile_compare_choices(second, first) == -1,
           "near-cancelling wide rational scores distinguish differences below floating precision");
    std::swap(first.paired[0], first.paired[1]);
    expect(cuda_pair_profile_compare_choices(first, second) == 1,
           "score ordering is independent of leg orientation");
    CudaPairProfileChoice serial;
    serial.original = {CudaPairProfileMeasurement{50, 1}, CudaPairProfileMeasurement{300, 1}};
    expect(cuda_pair_profile_compare_choices(first, serial) == 1 &&
               cuda_pair_profile_compare_choices(serial, first) == -1,
           "a qualified shared choice outranks serial fallback without forging a grant");
    auto slower = serial;
    slower.original = {CudaPairProfileMeasurement{200, 1}, CudaPairProfileMeasurement{100, 1}};
    slower.first_original = 1;
    expect(cuda_pair_profile_compare_choices(serial, slower) == 1 &&
               cuda_pair_profile_compare_choices(slower, serial) == -1 &&
               cuda_pair_profile_compare_choices(serial, {}) == 0,
           "serial global choice prefers exact shortest known mean and keeps missing data "
           "conservative");
    for (unsigned int scenario = 0; scenario < 7; ++scenario) {
        auto invalid = first;
        if (scenario == 0) invalid.first_original = 2;
        if (scenario == 1) invalid.original[0].samples = 0;
        if (scenario == 2) invalid.paired[0].samples = 1001;
        if (scenario == 3) invalid.paired[1].total_ns = maximum + 1;
        if (scenario == 4) invalid.is_trained = false;
        if (scenario == 5) invalid.candidate = 64;
        if (scenario == 6) invalid.paired[0].total_ns = UINT64_MAX;
        expect(!cuda_pair_profile_compare_choices(invalid, second) &&
                   !cuda_pair_profile_compare_choices(second, invalid),
               "invalid score facts reject symmetrically before arithmetic");
    }
    serial.first_original = 1;
    expect(!cuda_pair_profile_compare_choices(serial, slower) && clock_calls.load() == clocks,
           "contradictory shortest-original metadata rejects without any clock callback");
}
}  // namespace

int main() {
    try {
        original_measurement_qualification();
        checked_sampling_outcomes();
        context_configuration_archive_identity();
        function_configuration_archive_identity();
        exact_scores_reversal_and_readonly();
        thresholds_ties_and_identical_legs();
        exact_identity_and_catalog_misses();
        joint_resources_and_input_bounds();
        invalid_observation_sequences();
        concurrent_observers_and_process_guards();
        in_flight_clocks_commit_and_cancel();
        bounds_eviction_and_pins();
        maximum_totals_and_exact_score_boundary();
        large_cross_candidate_rationals();
        observers_outlive_cache_owner();
        finished_leg_release_and_reservation_pins();
        archive_literals_and_hostile_transactions();
        incomplete_archive_and_readonly_miss();
        global_choice_comparisons();
        noninserting_global_lookup();
        std::cout << "Twenty independent CPU paired-profile groups passed; no GPU-overlap "
                     "qualification\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Paired profile cache: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
