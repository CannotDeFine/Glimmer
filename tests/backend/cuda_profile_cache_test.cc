#include "internal/cuda_profile_cache.h"
#include "internal/cuda_ptx_cache.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
std::uint64_t now_ns = 1;
CudaProfileCache* observed_cache{};
bool observed_unlocked{};
std::uint64_t clock_ns() noexcept {
    if (observed_cache) observed_unlocked = observed_cache->snapshot().is_valid;
    return now_ns;
}
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
CudaProfilePlatform platform() {
    return {{std::byte{1}}, 12060, 12060, 1};
}
CudaProfileGeometry geometry() {
    return {{20, 1, 1}, {32, 1, 1}, 0, Kind::kDynamic};
}
void sample(CudaProfileChoice& choice, std::uint64_t duration, bool success = true) {
    expect(choice.was_selected && choice.observation && !choice.is_trained,
           "untrained candidate carries a once-only observation");
    now_ns = 100;
    choice.observation->begin();
    now_ns += duration;
    choice.observation->finish(success);
    choice.observation->finish(success);
    choice.observation.reset();
}
void exact_identity_and_selection() {
    CudaProfileCache cache({128, std::size_t{64} * 1024 * 1024, 1, 2, {2, 4}}, clock_ns);
    observed_cache = &cache;
    const std::array binary{std::byte{1}, std::byte{0}};
    auto key = cache.bind(platform(), "original code", binary, "kernel");
    auto same = cache.bind(platform(), "original code", binary, "kernel");
    expect(key && same && cache.snapshot().functions == 1, "reuse exact immutable code identity");
    for (const auto workers : {0U, 2U, 4U})
        for (unsigned int call = 0; call < 3; ++call) {
            auto choice = cache.choose(key, geometry(), 4, true);
            expect(choice.workers == workers, "rotate candidates over distinct natural requests");
            sample(choice, workers == 0 ? 1000 : workers == 2 ? 50 : 100);
        }
    auto chosen = cache.choose(same, geometry(), 4, true);
    expect(chosen.was_selected && chosen.is_trained && chosen.workers == 2 && !chosen.observation &&
               cache.snapshot().samples == 6 && cache.snapshot().trained_cases == 1 &&
               observed_unlocked,
           "ignore warmups, choose measured minimum and call clocks outside exclusion");
    auto changed = platform();
    ++changed.driver_version;
    auto driver = cache.bind(changed, "original code", binary, "kernel");
    changed = platform();
    changed.uuid[0] = std::byte{2};
    auto device = cache.bind(changed, "original code", binary, "kernel");
    auto source = cache.bind(platform(), "changed code", binary, "kernel");
    auto name = cache.bind(platform(), "original code", binary, "other");
    const std::array different_binary{std::byte{2}, std::byte{0}};
    auto image = cache.bind(platform(), "original code", different_binary, "kernel");
    expect(cache.snapshot().functions == 6,
           "GPU, driver, full code, binary and name separate keys");
    for (const auto* different : {&driver, &device, &source, &name, &image}) {
        auto choice = cache.choose(*different, geometry(), 4, true);
        expect(choice.was_selected && !choice.is_trained && choice.workers == 0,
               "changed code/platform cannot reuse timing");
    }
    for (unsigned int field = 0; field < 4; ++field) {
        auto launch = geometry();
        if (field == 0) ++launch.grid[1];
        if (field == 1) ++launch.block[0];
        if (field == 2) ++launch.shared_memory_bytes;
        if (field == 3) launch.kind = Kind::kSliced;
        auto choice = cache.choose(key, launch, 4, true);
        expect(choice.was_selected && !choice.is_trained,
               "complete launch identity separates cases");
    }
    auto native_only = cache.choose(key, geometry(), 4, false);
    expect(native_only.was_selected && !native_only.is_trained,
           "native fallback cannot masquerade as a transformed timing sample");
    observed_cache = nullptr;
}
void failed_abandoned_and_foreign_observations() {
    CudaProfileCache cache({1, std::size_t{1024} * 1024, 0, 1, {2}}, clock_ns);
    auto key = cache.bind(platform(), "code", {}, "kernel");
    {
        auto choice = cache.choose(key, geometry(), 2, true);
        expect(!cache.choose(key, geometry(), 2, true).was_selected,
               "concurrent reservation cannot overwrite a live sample");
    }
    expect(cache.snapshot().cancelled == 1, "unissued sample cancellation releases reservation");
    auto choice = cache.choose(key, geometry(), 2, true);
    sample(choice, 10, false);
    choice = cache.choose(key, geometry(), 2, true);
    sample(choice, 0);
    expect(cache.snapshot().samples == 0 && cache.snapshot().rejected_samples == 2,
           "failed SDK work and invalid clock are never timings");
    choice = cache.choose(key, geometry(), 2, true);
    const auto child = fork();
    expect(child >= 0, "fork profile ownership control");
    if (child == 0) {
        choice.observation->begin();
        choice.observation->finish(true);
        _exit(!cache.snapshot().is_valid && !cache.choose(key, geometry(), 2, true).was_selected
                  ? 0
                  : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited profile state rejects before locks and clocks");
    sample(choice, 100);
    choice = cache.choose(key, geometry(), 2, true);
    expect(choice.workers == 2, "failed observations did not advance the candidate");
    sample(choice, 200);
    expect(cache.choose(key, geometry(), 2, true).workers == 0,
           "original remains selectable when it is faster");
    CudaProfileCache foreign({}, clock_ns);
    expect(!foreign.choose(key, geometry(), 2, true).was_selected,
           "a foreign cache cannot use another identity capability");
    auto invalid = geometry();
    invalid.kind = static_cast<Kind>(7);
    expect(!cache.choose(key, invalid, 2, true).was_selected, "invalid kind rejects");
}
void bounds_pins_and_case_eviction() {
    CudaProfileCache cache({1, std::size_t{1024} * 1024, 0, 1, {2, 4}}, clock_ns);
    auto key = cache.bind(platform(), "code", {}, "kernel");
    expect(key && !cache.bind(platform(), "other", {}, "kernel"),
           "function lease pins exact code at bounded capacity");
    auto pinned = cache.choose(key, geometry(), 4, true);
    for (unsigned int index = 1; index <= 20; ++index) {
        auto changed = geometry();
        changed.shared_memory_bytes = index;
        auto choice = cache.choose(key, changed, 4, true);
        sample(choice, 20);
    }
    expect(cache.snapshot().cases == 16 && cache.snapshot().evictions >= 5,
           "shape cache remains bounded while retaining its active observation");
    sample(pinned, 10);
    auto next = cache.choose(key, geometry(), 4, true);
    expect(next.workers == 2, "pinned generation survives LRU pressure");
    auto cancellation = std::async(
        std::launch::async, [lease = std::move(next.observation)]() mutable { lease.reset(); });
    cancellation.get();
    key = {};
    expect(static_cast<bool>(cache.bind(platform(), "other", {}, "kernel")),
           "unborrowed code evicts after off-thread CPU-only observation release");
    CudaProfileCache tiny({1, 1, 0, 1, {2}}, clock_ns);
    expect(!tiny.bind(platform(), "code", {}, "kernel") && tiny.snapshot().retained_bytes == 0,
           "too-small byte budget bypasses without retaining over-budget metadata");
    bool rejected{};
    try {
        CudaProfileCache invalid({1, 1024, 0, 1, {2, 2}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "duplicate worker candidates reject at construction");
}
void complete_family_worker_candidates() {
    const std::vector kinds{Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive,
                            Kind::kSliced};
    CudaProfileCache cache({4, std::size_t{1024} * 1024, 0, 1, {1, 2, 3, 4, 5, 6, 7}, kinds},
                           clock_ns);
    const auto key = cache.bind(platform(), "code", {}, "kernel");
    auto choice = cache.choose(key, geometry(), 7, kinds);
    expect(choice.workers == 0, "original remains first in the complete candidate set");
    sample(choice, 1000);
    for (const auto kind : kinds)
        for (unsigned int workers = 1; workers <= 7; ++workers) {
            choice = cache.choose(key, geometry(), 7, kinds);
            expect(choice.was_selected && choice.kind == kind && choice.workers == workers,
                   "bounded family/worker pairs remain distinct actual candidates");
            sample(choice, kind == Kind::kSliced && workers == 3 ? 10 : 100);
        }
    choice = cache.choose(key, geometry(), 7, kinds);
    expect(choice.is_trained && choice.kind == Kind::kSliced && choice.workers == 3 &&
               cache.snapshot().samples == 36 && cache.snapshot().trained_cases == 1,
           "choose the measured family and workers from all 36 candidates");
    auto different = kinds;
    std::swap(different[0], different[1]);
    choice = cache.choose(key, geometry(), 7, different);
    expect(choice.was_selected && !choice.is_trained && choice.workers == 0,
           "ordered candidate identity cannot alias another preparation policy");
    different = {Kind::kDynamic, Kind::kDynamic};
    expect(!cache.choose(key, geometry(), 7, different).was_selected,
           "duplicate eligibility rejects before creating an observation");
    auto launch = geometry();
    launch.grid = {2, 2, 2};
    choice = cache.choose(key, launch, 7, kinds);
    sample(choice, 1000);
    choice = cache.choose(key, launch, 7, kinds);
    expect(choice.kind == Kind::kBlockStatic && choice.workers == 1,
           "independent-thread static cannot sample a multidimensional logical grid");
    CudaProfileCache limited({4, std::size_t{1024} * 1024, 0, 1, {2}, {Kind::kSliced}}, clock_ns);
    const auto limited_key = limited.bind(platform(), "code", {}, "kernel");
    expect(!limited.choose(limited_key, geometry(), 2, kinds).was_selected,
           "eligible families cannot expand the trusted configured family set");
    bool rejected{};
    try {
        CudaProfileCache invalid(
            {4, std::size_t{1024} * 1024, 0, 1, {2}, {Kind::kDynamic, Kind::kDynamic}}, clock_ns);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "duplicate configured families reject at construction");
}
void profile_archive_roundtrip_and_rejections() {
    const CudaProfileConfig config{
        4, std::size_t{1024} * 1024, 0, 1, {2}, {Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}};
    CudaProfileCache measured(config, clock_ns);
    auto key = measured.bind(platform(), "code", {}, "kernel");
    for (const auto duration : {100U, 10U, 50U, 20U}) {
        auto choice = measured.choose(key, geometry(), 2, config.kinds);
        sample(choice, duration);
    }
    auto incomplete_key = measured.bind(platform(), "pending", {}, "kernel");
    auto pending = measured.choose(incomplete_key, geometry(), 2, config.kinds);
    const auto archive = measured.encode("runtime-A");
    expect(archive && !archive->empty(), "encode only completed measurements without live leases");
    CudaProfileCache restored(config, clock_ns, false);
    expect(restored.restore(*archive, "runtime-A") && restored.snapshot().functions == 1 &&
               restored.snapshot().loaded_cases == 1 && !restored.is_learning(),
           "transactional complete-only restore into read-only selection");
    auto restored_key = restored.bind(platform(), "code", {}, "kernel");
    auto choice = restored.choose(restored_key, geometry(), 2, config.kinds);
    expect(choice.is_trained && choice.kind == Kind::kDynamic && !choice.observation,
           "ordinary learned selection uses the fastest prepared candidate without observation");
    choice = restored.choose(restored_key, geometry(), 2, config.kinds, true);
    expect(choice.is_trained && choice.kind == Kind::kSliced && !choice.observation,
           "priority selection excludes a faster nonpreemptible family and original");
    auto changed = geometry();
    ++changed.block[0];
    expect(!restored.choose(restored_key, changed, 2, config.kinds).was_selected &&
               restored.snapshot().cases == 1 && restored.snapshot().samples == 0,
           "read-only misses cannot create a sample or a training case");
    auto changed_gpu = platform();
    changed_gpu.uuid[0] = std::byte{7};
    const auto other_gpu = restored.bind(changed_gpu, "code", {}, "kernel");
    expect(!restored.choose(other_gpu, geometry(), 2, config.kinds).was_selected &&
               !restored.restore(*archive, "runtime-A") &&
               restored.choose(restored_key, geometry(), 2, config.kinds).is_trained,
           "platform mismatch misses and a live cache cannot be overwritten by restore");

    const auto reject = [&](std::span<const std::byte> bytes,
                            std::string_view identity = "runtime-A") {
        CudaProfileCache cache(config, clock_ns, false);
        expect(!cache.restore(bytes, identity) && cache.snapshot().functions == 0 &&
                   cache.snapshot().loaded_cases == 0,
               "invalid archive rejects atomically without publishing any code/case");
    };
    reject(*archive, "runtime-B");
    for (std::size_t length = 0; length < archive->size(); ++length)
        reject(std::span{*archive}.first(length));
    auto corrupt = *archive;
    corrupt[corrupt.size() / 2] ^= std::byte{1};
    reject(corrupt);
    const auto resign = [](std::vector<std::byte>& bytes) {
        std::uint64_t digest = 14695981039346656037ULL;
        for (const auto byte : std::span{bytes}.first(bytes.size() - 8))
            digest = (digest ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
        for (std::size_t index = 0; index < 8; ++index)
            bytes[bytes.size() - 8 + index] =
                static_cast<std::byte>((digest >> (8 * index)) & 255U);
    };
    const auto overwrite = [&](std::size_t offset, std::uint32_t value) {
        auto invalid = *archive;
        for (std::size_t index = 0; index < 4; ++index)
            invalid[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
        resign(invalid);
        reject(invalid);
    };
    std::size_t cursor{};
    const auto take = [&] {
        std::uint32_t value{};
        expect(cursor + 4 <= archive->size(), "test archive field offset");
        for (std::size_t index = 0; index < 4; ++index)
            value |= std::to_integer<std::uint32_t>((*archive)[cursor++]) << (8 * index);
        return value;
    };
    const auto skip_text = [&] {
        const auto length = take();
        cursor += length;
        expect(cursor <= archive->size(), "test archive string offset");
    };
    overwrite(0, UINT32_MAX);
    skip_text();
    overwrite(cursor, 99);  // Version.
    (void)take();
    overwrite(cursor, 99);  // Metric.
    (void)take();
    skip_text();           // Runtime identity.
    overwrite(cursor, 1);  // Warmup mismatch.
    (void)take();
    (void)take();  // Samples.
    const auto workers = take();
    cursor += static_cast<std::size_t>(workers) * 4;
    const auto kinds = take();
    cursor += static_cast<std::size_t>(kinds) * 4;
    overwrite(cursor, UINT32_MAX);  // Code count.
    expect(take() == 1, "one complete exported function");
    cursor += 16;
    overwrite(cursor, 0);  // Driver identity.
    cursor += 12;
    skip_text();
    skip_text();
    skip_text();
    overwrite(cursor, 17);  // Shape count.
    expect(take() == 1, "one complete exported case");
    overwrite(cursor, 0);  // Grid axis.
    cursor += std::size_t{7} * 4;
    overwrite(cursor, 99);  // Geometry kind.
    (void)take();
    overwrite(cursor, 37);  // Candidate count.
    expect(take() == 4, "original plus three private candidates");
    overwrite(cursor, 2);  // Original workers must be zero.
    cursor += 8;
    overwrite(cursor, 0);  // Zero original elapsed time.
    cursor += 8;
    overwrite(cursor, 3);       // Unconfigured worker.
    overwrite(cursor + 4, 99);  // Invalid candidate enum.
    corrupt = *archive;
    corrupt.insert(corrupt.end() - 8, std::byte{0});
    resign(corrupt);
    reject(corrupt);
    auto tiny_config = config;
    tiny_config.max_bytes = 1;
    CudaProfileCache tiny(tiny_config, clock_ns, false);
    expect(!tiny.restore(*archive, "runtime-A") && tiny.snapshot().functions == 0,
           "retained metadata must fit the receiving cache budget");
}
void generator_abi_separates_learned_and_restored_profiles(std::uint32_t prior_abi) {
    const CudaProfileConfig config{4, std::size_t{1024} * 1024, 0, 1, {2}, {Kind::kDynamic}};
    auto prior_platform = platform();
    // Literal identities are independent of the current generator constant.
    // Prior rows may deserialize, but ABI7/8/9 cannot authorize ABI10 reuse.
    prior_platform.generator_abi = prior_abi;
    auto current_platform = prior_platform;
    current_platform.generator_abi = 10;
    expect(kCudaPtxGeneratorAbi == 10, "literal profile epoch matches the current generator");
    constexpr std::string_view source = "identical opaque-parameter source";
    constexpr std::string_view name = "opaque";
    constexpr std::string_view runtime = "generator-identity-runtime";
    const std::array binary{std::byte{9}, std::byte{3}};

    CudaProfileCache measured(config, clock_ns);
    const auto prior = measured.bind(prior_platform, source, binary, name);
    expect(static_cast<bool>(prior), "bind the explicit prior generator epoch");
    for (const auto duration : {100U, 10U}) {
        auto choice = measured.choose(prior, geometry(), 2, config.kinds);
        sample(choice, duration);
    }
    const auto learned = measured.choose(prior, geometry(), 2, config.kinds);
    expect(learned.was_selected && learned.is_trained && learned.workers == 2 &&
               learned.kind == Kind::kDynamic && !learned.observation &&
               measured.snapshot().samples == 2 && measured.snapshot().trained_cases == 1,
           "prior ABI learns a complete independent private-worker choice");
    const auto archive = measured.encode(runtime);
    expect(archive.has_value(), "serialize genuine learned prior rows without rewriting identity");

    const auto current = measured.bind(current_platform, source, binary, name);
    expect(current && measured.snapshot().functions == 2,
           "ABI 10 cannot bind the existing prior code record with otherwise identical identity");
    auto fresh = measured.choose(current, geometry(), 2, config.kinds);
    expect(fresh.was_selected && !fresh.is_trained && fresh.workers == 0 && fresh.observation &&
               measured.snapshot().samples == 2 && measured.snapshot().trained_cases == 1,
           "live ABI 10 starts at an untrained original candidate instead of prior private timing");
    fresh.observation.reset();
    const auto prior_again = measured.bind(prior_platform, source, binary, name);
    const auto still_learned = measured.choose(prior_again, geometry(), 2, config.kinds);
    expect(still_learned.was_selected && still_learned.is_trained && still_learned.workers == 2 &&
               !still_learned.observation && measured.snapshot().functions == 2 &&
               measured.snapshot().samples == 2,
           "ABI 10 miss and cancelled observation preserve exact prior-platform measurements");

    CudaProfileCache restored(config, clock_ns, false);
    expect(restored.restore(*archive, runtime) && restored.snapshot().functions == 1 &&
               restored.snapshot().loaded_cases == 1 && !restored.is_learning(),
           "prior profile rows may deserialize into the read-only profile cache");
    const auto restored_current = restored.bind(current_platform, source, binary, name);
    expect(restored_current && restored.snapshot().functions == 2,
           "current platform binds a distinct code identity after old rows deserialize");
    const auto miss = restored.choose(restored_current, geometry(), 2, config.kinds);
    expect(!miss.was_selected && !miss.is_trained && !miss.observation &&
               restored.snapshot().cases == 1 && restored.snapshot().loaded_cases == 1 &&
               restored.snapshot().samples == 0,
           "restored prior timing cannot authorize any ABI 10 choice or create a sample");
    const auto restored_prior = restored.bind(prior_platform, source, binary, name);
    const auto old_choice = restored.choose(restored_prior, geometry(), 2, config.kinds);
    expect(old_choice.was_selected && old_choice.is_trained && old_choice.workers == 2 &&
               old_choice.kind == Kind::kDynamic && !old_choice.observation &&
               restored.snapshot().functions == 2 && restored.snapshot().loaded_cases == 1 &&
               restored.snapshot().samples == 0,
           "restored measurements remain usable only under their exact prior platform");
}
CudaProfileGeometry sharing_geometry() {
    return {{64, 1, 1}, {64, 1, 1}, 0, Kind::kDynamic};
}
template <typename Duration>
void train_sharing_profile(CudaProfileCache& cache, const CudaProfileFunction& key,
                           CudaProfileGeometry launch, const CudaProfileConfig& config,
                           unsigned int maximum, Duration duration) {
    for (std::size_t index = 0; index < 36; ++index) {
        auto choice = cache.choose(key, launch, maximum, config.kinds);
        expect(choice.was_selected, "training fixture receives the next original/private choice");
        if (choice.is_trained) return;
        sample(choice, duration(choice));
    }
    expect(cache.choose(key, launch, maximum, config.kinds).is_trained,
           "bounded fixture completes the whole actual candidate set");
}
void sharing_threshold_and_legacy_selection() {
    const CudaProfileConfig config{4, std::size_t{1024} * 1024, 0, 1, {2, 4}, {Kind::kDynamic}};
    CudaProfileCache cache(config, clock_ns);
    const auto key = cache.bind(platform(), "threshold code", {}, "kernel");
    train_sharing_profile(cache, key, sharing_geometry(), config, 4, [](const auto& choice) {
        return choice.workers == 0 ? 900U : choice.workers == 4 ? 1000U : 1100U;
    });
    expect(cache.choose(key, sharing_geometry(), 4, config.kinds).workers == 0,
           "legacy global-minimum selection still chooses the faster original");
    const std::array candidates{CudaProfileSharingCandidate{Kind::kDynamic, 4}};
    CudaProfileSharingLimits limits{2, 128, 4, 90};
    const auto boundary = cache.choose_sharing(key, sharing_geometry(), limits, candidates);
    expect(boundary.was_selected && boundary.is_trained && !boundary.observation &&
               boundary.kind == Kind::kDynamic && boundary.workers == 4,
           "sharing accepts an exactly 90-percent private choice although original is faster");
    for (const auto percent : {91U, 100U, 10000U}) {
        limits.minimum_speed_percent = percent;
        const auto original = cache.choose_sharing(key, sharing_geometry(), limits, candidates);
        expect(original.was_selected && original.is_trained && original.workers == 0 &&
                   !original.observation,
               "a below-threshold private measurement explicitly selects original");
    }
    limits.minimum_speed_percent = 89;
    expect(cache.choose_sharing(key, sharing_geometry(), limits, candidates).workers == 4,
           "a weaker explicit threshold selects the same measured private candidate");
    expect(cache.snapshot().samples == 3 && cache.snapshot().cases == 1 &&
               cache.snapshot().cancelled == 0,
           "sharing never adds samples, cases or hidden observations to a learning cache");
}
void sharing_resource_filtered_superset() {
    const CudaProfileConfig config{
        8,
        std::size_t{1024} * 1024,
        0,
        1,
        {16, 2, 8, 4, 3},
        {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}};
    CudaProfileCache measured(config, clock_ns);
    const std::array binary{std::byte{4}, std::byte{2}};
    const auto key = measured.bind(platform(), "superset code", binary, "kernel");
    train_sharing_profile(measured, key, sharing_geometry(), config, 16, [](const auto& choice) {
        if (!choice.workers) return 900U;
        if (choice.kind == Kind::kPreemptive || choice.workers == 16 || choice.workers == 2)
            return 1U;
        if (choice.kind == Kind::kDynamic && choice.workers == 3) return 700U;
        if (choice.kind == Kind::kDynamic && choice.workers == 4) return 810U;
        if (choice.kind == Kind::kSliced && choice.workers == 4) return 820U;
        if (choice.kind == Kind::kDynamic && choice.workers == 8) return 830U;
        if (choice.kind == Kind::kBlockStatic && choice.workers == 4) return 840U;
        if (choice.kind == Kind::kStatic && choice.workers == 8) return 850U;
        return 1200U;
    });
    const auto archive = measured.encode("sharing-runtime");
    expect(archive.has_value(), "complete broad measurements serialize without a new schema");
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(*archive, "sharing-runtime"), "restore the unchanged broad profile");
    const auto restored = cache.bind(platform(), "superset code", binary, "kernel");
    std::array candidates{CudaProfileSharingCandidate{Kind::kSliced, 7},
                          CudaProfileSharingCandidate{Kind::kDynamic, 16},
                          CudaProfileSharingCandidate{Kind::kStatic, 16},
                          CudaProfileSharingCandidate{Kind::kBlockStatic, 4}};
    CudaProfileSharingLimits limits{2, 256, 16, 90};
    auto chosen = cache.choose_sharing(restored, sharing_geometry(), limits, candidates);
    expect(chosen.is_trained && !chosen.observation && chosen.kind == Kind::kSliced &&
               chosen.workers == 4,
           "reuse a trained superset with reordered families and per-family resource caps");
    const std::array filtered{Kind::kSliced, Kind::kDynamic, Kind::kStatic, Kind::kBlockStatic};
    expect(!cache.choose(restored, sharing_geometry(), 8, filtered).was_selected,
           "legacy exact-vector identity still misses the resource-filtered candidate set");
    limits.max_workers = 4;
    chosen = cache.choose_sharing(restored, sharing_geometry(), limits, candidates);
    expect(chosen.kind == Kind::kDynamic && chosen.workers == 4,
           "global cap selects the largest configured bounded geometry, not faster smaller two");
    limits.max_workers = 16;
    limits.threads_per_sm = 128;
    expect(cache.choose_sharing(restored, sharing_geometry(), limits, candidates).workers == 4,
           "thread budget independently bounds workers");
    limits.threads_per_sm = 256;
    candidates[0].max_workers = 0;
    chosen = cache.choose_sharing(restored, sharing_geometry(), limits, candidates);
    expect(chosen.kind == Kind::kDynamic && chosen.workers == 8,
           "zero occupancy excludes only that family and cannot select a faster excluded worker");
    const std::array dynamic{CudaProfileSharingCandidate{Kind::kDynamic, 16}};
    limits.sm_count = 3;
    expect(cache.choose_sharing(restored, sharing_geometry(), limits, dynamic).workers == 3,
           "only explicitly measured SM multiples may be selected, never rounded workers");
    expect(cache.snapshot().samples == 0 && cache.snapshot().cases == 1 &&
               cache.snapshot().loaded_cases == 1 && cache.encode("sharing-runtime") == archive,
           "resource-filtered inference changes neither training nor archive contents");
}
void sharing_static_and_input_bounds() {
    const CudaProfileConfig config{8,      std::size_t{1024} * 1024,
                                   0,      1,
                                   {2, 4}, {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic}};
    CudaProfileCache cache(config, clock_ns);
    const std::array both{CudaProfileSharingCandidate{Kind::kStatic, 4},
                          CudaProfileSharingCandidate{Kind::kDynamic, 4}};
    const std::array only_static{both[0]};
    const std::array block_or_dynamic{CudaProfileSharingCandidate{Kind::kBlockStatic, 4}, both[1]};
    const std::array only_block_static{block_or_dynamic[0]};
    const CudaProfileSharingLimits limits{2, 128, 4, 90};
    const auto duration = [](const auto& choice) {
        if (!choice.workers) return 1000U;
        if (choice.kind == Kind::kStatic) return 1U;
        return choice.kind == Kind::kBlockStatic ? 2U : 100U;
    };
    for (const auto name : {"prefixDeviceSelectSweepKernel<int>", "prefixDeviceScanKernel<int>"}) {
        const auto key = cache.bind(platform(), "unsafe-static code", {}, name);
        train_sharing_profile(cache, key, sharing_geometry(), config, 4, duration);
        const auto choice = cache.choose_sharing(key, sharing_geometry(), limits, both);
        expect(choice.kind == Kind::kDynamic && choice.workers == 4 &&
                   !cache.choose_sharing(key, sharing_geometry(), limits, only_static).was_selected,
               "both upstream unsafe-name fragments exclude independent-thread static");
        const auto block_choice =
            cache.choose_sharing(key, sharing_geometry(), limits, block_or_dynamic);
        expect(block_choice.kind == Kind::kDynamic && block_choice.workers == 4 &&
                   !cache.choose_sharing(key, sharing_geometry(), limits, only_block_static)
                        .was_selected,
               "unsafe names also exclude faster BlockStatic while retaining safe Dynamic");
    }
    auto launch = sharing_geometry();
    launch.grid = {8, 4, 2};
    const auto key = cache.bind(platform(), "bounds code", {}, "ordinary");
    train_sharing_profile(cache, key, launch, config, 4, duration);
    expect(cache.choose_sharing(key, launch, limits, both).kind == Kind::kDynamic &&
               !cache.choose_sharing(key, launch, limits, only_static).was_selected,
           "multidimensional static is excluded while the cooperative family remains eligible");
    const auto multidimensional = cache.choose_sharing(key, launch, limits, block_or_dynamic);
    expect(multidimensional.kind == Kind::kBlockStatic && multidimensional.workers == 4,
           "ordinary BlockStatic remains eligible for multidimensional logical input");
    launch = sharing_geometry();
    train_sharing_profile(cache, key, launch, config, 4, duration);
    expect(cache.choose_sharing(key, launch, limits, both).kind == Kind::kStatic,
           "ordinary one-dimensional static remains eligible");
    for (const auto invalid_limits :
         {CudaProfileSharingLimits{0, 128, 4, 90}, CudaProfileSharingLimits{2, 0, 4, 90},
          CudaProfileSharingLimits{2, 128, 0, 90}, CudaProfileSharingLimits{2, 128, 4, 0},
          CudaProfileSharingLimits{2, 128, 4, 10001}, CudaProfileSharingLimits{2, 63, 4, 90},
          CudaProfileSharingLimits{2, 128, 1, 90}})
        expect(!cache.choose_sharing(key, launch, invalid_limits, both).was_selected,
               "invalid or insufficient resource/threshold bounds preserve original");
    for (const auto kind : {Kind::kPreemptive, Kind::kSliced, static_cast<Kind>(99)}) {
        const std::array invalid{CudaProfileSharingCandidate{kind, 4}};
        expect(!cache.choose_sharing(key, launch, limits, invalid).was_selected,
               "preemptive, unconfigured and invalid families cannot grant sharing choices");
    }
    const std::array duplicate{both[0], both[0]};
    const std::array too_many{both[0], both[1], both[0], both[1], both[0]};
    expect(!cache.choose_sharing(key, launch, limits, duplicate).was_selected &&
               !cache.choose_sharing(key, launch, limits, too_many).was_selected &&
               !cache.choose_sharing(key, launch, limits, {}).was_selected,
           "duplicate, oversized and empty eligibility cannot select a private family");
    for (unsigned int field = 0; field < 5; ++field) {
        auto invalid = launch;
        if (field == 0) invalid.grid[0] = 0;
        if (field == 1) invalid.block[1] = 0;
        if (field == 2) invalid.grid.fill(std::numeric_limits<unsigned int>::max());
        if (field == 3) invalid.block.fill(std::numeric_limits<unsigned int>::max());
        if (field == 4) invalid.kind = static_cast<Kind>(99);
        expect(!cache.choose_sharing(key, invalid, limits, both).was_selected,
               "invalid geometry and overflowing grid/block products reject before selection");
    }
    for (const auto grid : {8U, 9U}) {
        auto boundary = launch;
        boundary.grid[0] = grid;
        const CudaProfileSharingLimits boundary_limits{2, 256, 4, 90};
        train_sharing_profile(cache, key, boundary, config, 4, duration);
        expect(
            cache.choose_sharing(key, boundary, boundary_limits, both).was_selected == (grid == 9),
            "ceil(logical CTAs per SM) fitting the thread budget bypasses private selection");
    }
}
void sharing_incomplete_missing_and_identity() {
    const CudaProfileConfig config{16,     std::size_t{1024} * 1024,       0, 1,
                                   {2, 4}, {Kind::kDynamic, Kind::kSliced}};
    const CudaProfileSharingLimits limits{2, 128, 4, 90};
    const std::array both{CudaProfileSharingCandidate{Kind::kDynamic, 4},
                          CudaProfileSharingCandidate{Kind::kSliced, 4}};
    const std::array dynamic{both[0]};
    CudaProfileCache measured(config, clock_ns);
    const auto key = measured.bind(platform(), "exact code", {}, "kernel");
    const auto cold = measured.choose_sharing(key, sharing_geometry(), limits, both);
    expect(!cold.was_selected && !cold.observation && measured.snapshot().cases == 0,
           "cold sharing creates no sample or case even in a learning cache");
    auto active = measured.choose(key, sharing_geometry(), 4, config.kinds);
    expect(!measured.choose_sharing(key, sharing_geometry(), limits, both).was_selected,
           "a live observation is never reused by sharing");
    sample(active, 1000);
    for (unsigned int index = 0; index < 2; ++index) {
        auto choice = measured.choose(key, sharing_geometry(), 4, config.kinds);
        sample(choice, 100);
    }
    expect(!measured.choose_sharing(key, sharing_geometry(), limits, dynamic).was_selected,
           "completed requested entries cannot hide an incomplete wider record");
    train_sharing_profile(measured, key, sharing_geometry(), config, 4,
                          [](const auto&) { return 100U; });
    expect(measured.choose_sharing(key, sharing_geometry(), limits, both).is_trained,
           "the completed inactive record becomes reusable without new learning");
    const auto archive = measured.encode("identity-runtime");
    expect(archive.has_value(), "complete identity fixture serializes");
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(*archive, "identity-runtime"), "restore read-only identity fixture");
    const auto restored = cache.bind(platform(), "exact code", {}, "kernel");
    expect(!cache.choose_sharing(key, sharing_geometry(), limits, both).was_selected &&
               !cache.choose_sharing({}, sharing_geometry(), limits, both).was_selected,
           "foreign and absent capabilities cannot supply cache identity");
    for (unsigned int field = 0; field < 7; ++field) {
        auto changed = platform();
        if (field == 0) changed.uuid[0] = std::byte{7};
        if (field == 1) ++changed.driver_version;
        if (field == 2) ++changed.toolkit_version;
        if (field == 3) ++changed.generator_abi;
        const std::array binary{std::byte{1}};
        const auto other = cache.bind(
            changed, field == 4 ? "other code" : "exact code",
            field == 5 ? std::span<const std::byte>{binary} : std::span<const std::byte>{},
            field == 6 ? "other kernel" : "kernel");
        expect(other && !cache.choose_sharing(other, sharing_geometry(), limits, both).was_selected,
               "every GPU/code/binary/name identity component must match a trained record");
    }
    for (unsigned int field = 0; field < 4; ++field) {
        auto changed = sharing_geometry();
        if (field == 0) ++changed.grid[0];
        if (field == 1) ++changed.block[0];
        if (field == 2) ++changed.shared_memory_bytes;
        if (field == 3) changed.kind = Kind::kSliced;
        expect(!cache.choose_sharing(restored, changed, limits, both).was_selected,
               "sharing retains full geometry, dynamic shared bytes and preparation-kind identity");
    }
    const auto child = fork();
    expect(child >= 0, "fork the read-only sharing fixture");
    if (!child)
        _exit(!cache.choose_sharing(restored, sharing_geometry(), limits, both).was_selected ? 0
                                                                                             : 1);
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited sharing state rejects before locking or observing");
    expect(cache.snapshot().samples == 0 && cache.snapshot().cases == 1 &&
               cache.snapshot().cancelled == 0 && cache.snapshot().rejected_samples == 0,
           "all inference misses and fork rejection leave training untouched");
}
void sharing_requires_one_complete_superset() {
    const CudaProfileConfig config{4,      std::size_t{1024} * 1024,       0, 1,
                                   {2, 4}, {Kind::kDynamic, Kind::kSliced}};
    CudaProfileCache cache(config, clock_ns);
    const auto key = cache.bind(platform(), "separate records", {}, "kernel");
    for (const auto kind : config.kinds) {
        const std::array eligible{kind};
        for (unsigned int call = 0; call < 3; ++call) {
            auto choice = cache.choose(key, sharing_geometry(), 4, eligible);
            sample(choice, choice.workers ? 100 : 1000);
        }
    }
    const std::array both{CudaProfileSharingCandidate{Kind::kDynamic, 4},
                          CudaProfileSharingCandidate{Kind::kSliced, 4}};
    const CudaProfileSharingLimits limits{2, 128, 4, 90};
    expect(!cache.choose_sharing(key, sharing_geometry(), limits, both).was_selected &&
               cache.snapshot().trained_cases == 2,
           "separate fully trained family records cannot be mixed into an invented superset");
    const auto limited = cache.bind(platform(), "smaller worker only", {}, "kernel");
    train_sharing_profile(cache, limited, sharing_geometry(), config, 2,
                          [](const auto& choice) { return choice.workers ? 100 : 1000; });
    expect(!cache.choose_sharing(limited, sharing_geometry(), limits, both).was_selected,
           "a measured smaller worker cannot replace the requested largest configured candidate");
    CudaProfileConfig mismatch = config;
    mismatch.workers = {4, 2};
    CudaProfileCache incompatible(mismatch, clock_ns, false);
    const auto archive = cache.encode("configuration-runtime");
    expect(archive && !incompatible.restore(*archive, "configuration-runtime"),
           "sharing does not relax archive configuration identity to obtain reuse");
}
void sharing_large_totals_and_default_family() {
    const CudaProfileConfig config{2, std::size_t{1024} * 1024, 0, 1000, {2}};
    CudaProfileCache cache(config, clock_ns);
    const auto key = cache.bind(platform(), "large exact timing totals", {}, "kernel");
    for (unsigned int index = 0; index < 2000; ++index) {
        auto choice = cache.choose(key, sharing_geometry(), 2, true);
        sample(choice, 3600ULL * 1000 * 1000 * 1000);
    }
    const std::array candidates{CudaProfileSharingCandidate{Kind::kDynamic, 2}};
    CudaProfileSharingLimits limits{2, 64, 2, 10000};
    const auto original = cache.choose_sharing(key, sharing_geometry(), limits, candidates);
    expect(original.is_trained && original.workers == 0 && !original.observation,
           "maximum valid sample totals and percentage cannot overflow into private acceptance");
    limits.minimum_speed_percent = 100;
    const auto equal = cache.choose_sharing(key, sharing_geometry(), limits, candidates);
    expect(equal.is_trained && equal.kind == Kind::kDynamic && equal.workers == 2,
           "empty configured families retain the exact geometry family at an equality boundary");
    const std::array unavailable{CudaProfileSharingCandidate{Kind::kDynamic, 0}};
    expect(!cache.choose_sharing(key, sharing_geometry(), limits, unavailable).was_selected &&
               cache.snapshot().samples == 2000 && cache.snapshot().cases == 1,
           "zero native occupancy cannot authorize a candidate or start another observation");
}
void sharing_selection() {
    sharing_threshold_and_legacy_selection();
    sharing_resource_filtered_superset();
    sharing_static_and_input_bounds();
    sharing_incomplete_missing_and_identity();
    sharing_requires_one_complete_superset();
    sharing_large_totals_and_default_family();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--sharing") {
            sharing_selection();
            std::cout << "Workload-agnostic resource-constrained cache checks passed\n";
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "only the focused --sharing test mode is supported");
        exact_identity_and_selection();
        failed_abandoned_and_foreign_observations();
        bounds_pins_and_case_eviction();
        complete_family_worker_candidates();
        profile_archive_roundtrip_and_rejections();
        for (const auto prior_abi : {7U, 8U, 9U})
            generator_abi_separates_learned_and_restored_profiles(prior_abi);
        sharing_selection();
        std::cout << "Bounded once-only profile and configuration checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
