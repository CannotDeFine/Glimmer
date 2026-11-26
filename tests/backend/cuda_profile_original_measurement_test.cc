#include "internal/cuda_profile_cache.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;
using Kind = glimmer::transform::PtxExecutionKind;
using namespace std::string_literals;

constexpr std::string_view kRuntimeIdentity = "original-measurement-readonly-v1";
constexpr std::uint64_t kOriginalFirst = (std::uint64_t{1} << 32) + 101;
constexpr std::uint64_t kOriginalSecond = (std::uint64_t{1} << 32) + 303;
constexpr std::uint64_t kOriginalTotal = kOriginalFirst + kOriginalSecond;
std::atomic<std::uint64_t> now_ns{1}, clock_calls{0};

static_assert(std::is_same_v<decltype(CudaProfileOriginalMeasurement::identity),
                             std::shared_ptr<const CudaProfileCodeIdentity>>);

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::uint64_t clock_ns() noexcept {
    clock_calls.fetch_add(1, std::memory_order_relaxed);
    return now_ns.load(std::memory_order_relaxed);
}
CudaProfileConfig configuration() {
    return {4, std::size_t{8} * 1024 * 1024, 1, 2, {80, 160}, {Kind::kPreemptive, Kind::kSliced}};
}
CudaProfileCodeIdentity identity() {
    CudaProfileCodeIdentity result;
    for (std::size_t index = 0; index < result.platform.uuid.size(); ++index)
        result.platform.uuid[index] = static_cast<std::byte>(index + 1);
    result.platform.driver_version = 12060;
    result.platform.toolkit_version = 12090;
    result.platform.generator_abi = 9;
    result.source = "exact source\0retained suffix\xff"s;
    result.binary =
        "\0\x7f\xff"
        "binary\0suffix"s;
    result.name = "qualified_original_kernel";
    return result;
}
CudaProfileGeometry geometry() {
    return {{640, 2, 1}, {128, 1, 1}, 64, Kind::kPreemptive};
}
CudaProfileFunction bind(CudaProfileCache& cache, const CudaProfileCodeIdentity& code) {
    return cache.bind(code.platform, code.source,
                      std::as_bytes(std::span{code.binary.data(), code.binary.size()}), code.name);
}
void sample(CudaProfileChoice& choice, std::uint64_t duration) {
    expect(choice.was_selected && choice.observation && !choice.is_trained,
           "training owns one actual candidate observation");
    now_ns.store(1000, std::memory_order_relaxed);
    choice.observation->begin();
    now_ns.store(1000 + duration, std::memory_order_relaxed);
    choice.observation->finish(true);
    choice.observation.reset();
}
void train(CudaProfileCache& cache, const CudaProfileFunction& function, CudaProfileGeometry shape,
           const CudaProfileConfig& config) {
    unsigned int original_calls{};
    for (unsigned int call = 0; call < 64; ++call) {
        auto choice = cache.choose(function, shape, 160, config.kinds);
        if (choice.is_trained) {
            expect(
                choice.was_selected && choice.workers && !choice.observation && original_calls == 3,
                "complete catalog selects a faster private candidate, not original");
            return;
        }
        std::uint64_t duration = 100 + choice.workers;
        if (!choice.workers) {
            expect(original_calls < 3, "original has exactly one warmup and two samples");
            const std::array durations{std::uint64_t{37}, kOriginalFirst, kOriginalSecond};
            duration = durations[original_calls++];
        }
        sample(choice, duration);
    }
    throw std::runtime_error("bounded natural observations finish the complete catalog");
}
std::vector<std::byte> trained_archive(const CudaProfileConfig& config,
                                       const std::vector<CudaProfileCodeIdentity>& codes,
                                       CudaProfileGeometry shape) {
    CudaProfileCache learning(config, clock_ns);
    for (const auto& code : codes) {
        const auto function = bind(learning, code);
        expect(static_cast<bool>(function), "bind complete exact training identity");
        train(learning, function, shape, config);
    }
    const auto archive = learning.encode(kRuntimeIdentity);
    expect(archive.has_value(), "encode complete inactive original measurements");
    return *archive;
}
auto snapshot_fields(const CudaProfileSnapshot& value) {
    return std::tuple{value.functions,     value.retained_bytes, value.cases,
                      value.trained_cases, value.samples,        value.rejected_samples,
                      value.cancelled,     value.hits,           value.bypasses,
                      value.evictions,     value.is_valid,       value.loaded_cases,
                      value.is_learning};
}
struct Evidence {
    CudaProfileSnapshot snapshot;
    std::vector<std::byte> archive;
    std::uint64_t clock_count{};
};
Evidence capture(const CudaProfileCache& cache) {
    const auto snapshot = cache.snapshot();
    const auto archive = cache.encode(kRuntimeIdentity);
    expect(snapshot.is_valid && archive, "capture healthy readonly metadata and archive");
    return {snapshot, *archive, clock_calls.load(std::memory_order_relaxed)};
}
void unchanged(const CudaProfileCache& cache, const Evidence& before) {
    const auto archive = cache.encode(kRuntimeIdentity);
    expect(snapshot_fields(cache.snapshot()) == snapshot_fields(before.snapshot),
           "original lookup changes no cache counters, cases, bounds or learning state");
    expect(archive && *archive == before.archive,
           "original lookup preserves every encoded byte, including code LRU order");
    expect(clock_calls.load(std::memory_order_relaxed) == before.clock_count,
           "original lookup never invokes the sampling clock");
}
void measurement(const std::optional<CudaProfileOriginalMeasurement>& value,
                 const CudaProfileCodeIdentity& code, CudaProfileGeometry shape) {
    expect(value && value->identity && *value->identity == code && value->geometry == shape,
           "view preserves full immutable code, binary, name, platform and geometry");
    expect(value->total_ns == kOriginalTotal && value->samples == 2 &&
               value->total_ns / value->samples > std::numeric_limits<std::uint32_t>::max(),
           "view exposes exact original totals and sample count without warmup or narrowing");
}

void complete_exact_readonly_views() {
    const auto config = configuration();
    const auto code = identity();
    auto other = code;
    other.source += " second identity";
    other.name += "_second";
    const auto shape = geometry();
    const auto archive = trained_archive(config, {code, other}, shape);
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(archive, kRuntimeIdentity), "restore complete readonly catalog");
    const auto function = bind(cache, code);
    const auto second = bind(cache, other);
    expect(function && second && !cache.is_learning() && cache.snapshot().loaded_cases == 2,
           "restored identities resolve to their actual readonly owner");
    const CudaProfileCache& readonly = cache;
    const auto before = capture(readonly);
    const auto by_function = readonly.original_measurement(function, shape);
    const auto by_identity = readonly.original_measurement(code, shape);
    measurement(by_function, code, shape);
    measurement(by_identity, code, shape);
    expect(by_function->identity == by_identity->identity,
           "both overloads alias the same existing immutable code allocation");
    for (unsigned int repeat = 0; repeat < 8; ++repeat) {
        measurement(readonly.original_measurement(second, shape), other, shape);
        measurement(readonly.original_measurement(code, shape), code, shape);
    }
    unchanged(readonly, before);
}

void every_identity_and_geometry_component_is_exact() {
    const auto config = configuration();
    const auto code = identity();
    const auto shape = geometry();
    const auto archive = trained_archive(config, {code}, shape);
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(archive, kRuntimeIdentity), "restore identity mutation control");
    const auto function = bind(cache, code);
    const auto before = capture(cache);
    const auto miss = [&](const CudaProfileCodeIdentity& changed) {
        expect(!cache.original_measurement(changed, shape),
               "any exact full-identity change misses without creating or binding code");
    };
    for (const auto member : {&CudaProfileCodeIdentity::source, &CudaProfileCodeIdentity::binary,
                              &CudaProfileCodeIdentity::name}) {
        const auto& original = code.*member;
        for (std::size_t index = 0; index < original.size(); ++index) {
            auto changed = code;
            (changed.*member)[index] = original[index] == 'x' ? 'y' : 'x';
            miss(changed);
        }
        auto changed = code;
        (changed.*member).push_back('x');
        miss(changed);
        changed = code;
        (changed.*member).pop_back();
        miss(changed);
        changed = code;
        (changed.*member).clear();
        miss(changed);
    }
    for (std::size_t index = 0; index < code.platform.uuid.size(); ++index) {
        auto changed = code;
        changed.platform.uuid[index] ^= std::byte{1};
        miss(changed);
    }
    for (const auto member :
         {&CudaProfilePlatform::driver_version, &CudaProfilePlatform::toolkit_version,
          &CudaProfilePlatform::generator_abi}) {
        auto changed = code;
        ++(changed.platform.*member);
        miss(changed);
    }
    auto changed = code;
    changed.platform = {};
    miss(changed);
    changed = code;
    changed.name.insert(3, 1, '\0');
    miss(changed);
    changed.name.assign(4097, 'x');
    miss(changed);
    changed = code;
    changed.source.assign(config.max_bytes + 1, 'x');
    miss(changed);
    changed = code;
    changed.binary.assign(config.max_bytes + 1, 'x');
    miss(changed);

    const auto shape_miss = [&](CudaProfileGeometry different) {
        expect(!cache.original_measurement(code, different) &&
                   !cache.original_measurement(function, different),
               "both overloads require the exact valid complete launch shape");
    };
    for (std::size_t index = 0; index < shape.grid.size(); ++index) {
        auto different = shape;
        ++different.grid[index];
        shape_miss(different);
        different = shape;
        different.grid[index] = 0;
        shape_miss(different);
        different = shape;
        ++different.block[index];
        shape_miss(different);
        different = shape;
        different.block[index] = 0;
        shape_miss(different);
    }
    auto different = shape;
    ++different.shared_memory_bytes;
    shape_miss(different);
    different = shape;
    different.kind = Kind::kSliced;
    shape_miss(different);
    different.kind = static_cast<Kind>(99);
    shape_miss(different);
    different = shape;
    different.grid.fill(std::numeric_limits<unsigned int>::max());
    shape_miss(different);
    different = shape;
    different.block.fill(std::numeric_limits<unsigned int>::max());
    shape_miss(different);
    measurement(cache.original_measurement(code, shape), code, shape);
    unchanged(cache, before);
}

void trained_overflowing_launch_volumes_still_miss() {
    const auto config = configuration();
    const auto code = identity();
    for (const bool is_grid : {true, false}) {
        auto shape = geometry();
        auto& dimensions = is_grid ? shape.grid : shape.block;
        dimensions.fill(std::numeric_limits<unsigned int>::max());
        // The existing single-cache codec accepts these nonzero dimensions.
        // Its retained measurements must not bypass the view's volume check.
        const auto archive = trained_archive(config, {code}, shape);
        CudaProfileCache cache(config, clock_ns, false);
        expect(cache.restore(archive, kRuntimeIdentity),
               "restore existing complete catalog with overflowing launch volume");
        const auto function = bind(cache, code);
        expect(cache.snapshot().trained_cases == 1,
               "overflow rejection is not merely a missing or incomplete shape");
        const auto before = capture(cache);
        expect(!cache.original_measurement(function, shape) &&
                   !cache.original_measurement(code, shape),
               "matching trained launch still needs a checked representable volume");
        unchanged(cache, before);
    }
}

void archive_resolution_does_not_bind_or_borrow_authority() {
    const auto config = configuration();
    const auto code = identity();
    const auto shape = geometry();
    const auto archive = trained_archive(config, {code}, shape);
    CudaProfileCache cache(config, clock_ns, false);
    const auto empty = capture(cache);
    for (unsigned int index = 0; index < 32; ++index) {
        auto unknown = code;
        unknown.name += std::to_string(index);
        expect(!cache.original_measurement(unknown, shape) &&
                   !cache.original_measurement(code, shape) &&
                   !cache.original_measurement(CudaProfileFunction{}, shape),
               "archive resolution miss is not an implicit bind or case creation");
    }
    unchanged(cache, empty);
    expect(cache.restore(archive, kRuntimeIdentity),
           "unbound lookup misses leave the cache empty and eligible for actual restore");
    const auto function = bind(cache, code);
    CudaProfileCache other(config, clock_ns, false);
    expect(other.restore(archive, kRuntimeIdentity), "restore independent readonly authority");
    const auto other_function = bind(other, code);
    const auto before = capture(cache);
    const auto other_before = capture(other);
    expect(!cache.original_measurement(other_function, shape) &&
               !other.original_measurement(function, shape),
           "equal-value function capabilities cannot cross their original cache owner");
    const auto first = cache.original_measurement(code, shape);
    const auto second = other.original_measurement(code, shape);
    measurement(first, code, shape);
    measurement(second, code, shape);
    expect(first->identity != second->identity,
           "independent archives do not share an invented global code authority");
    unchanged(cache, before);
    unchanged(other, other_before);

    CudaProfileFunction expired;
    {
        CudaProfileCache temporary(config, clock_ns, false);
        expect(temporary.restore(archive, kRuntimeIdentity), "restore expiring function owner");
        expired = bind(temporary, code);
    }
    expect(!cache.original_measurement(expired, shape),
           "retained code bytes do not revive an expired capability owner");
    unchanged(cache, before);

    auto unknown = code;
    unknown.name += "_untrained";
    const auto untrained = bind(cache, unknown);
    expect(static_cast<bool>(untrained), "bind cold code explicitly outside measurement lookup");
    const auto cold = capture(cache);
    expect(!cache.original_measurement(untrained, shape) &&
               !cache.original_measurement(unknown, shape),
           "explicitly bound cold code does not supply a trained original");
    unchanged(cache, cold);
}

void learning_partial_and_active_cases_never_escape() {
    const auto config = configuration();
    const auto code = identity();
    const auto shape = geometry();
    CudaProfileCache learning(config, clock_ns);
    const auto function = bind(learning, code);
    train(learning, function, shape, config);
    const auto trained = capture(learning);
    expect(!learning.original_measurement(function, shape) &&
               !learning.original_measurement(code, shape),
           "complete data in a learning cache is not a readonly original measurement");
    unchanged(learning, trained);

    CudaProfileCache partial(config, clock_ns);
    const auto incomplete = bind(partial, code);
    for (const auto duration : {std::uint64_t{37}, kOriginalFirst, kOriginalSecond}) {
        auto choice = partial.choose(incomplete, shape, 160, config.kinds);
        expect(!choice.workers, "complete only the original leg of an incomplete catalog");
        sample(choice, duration);
    }
    expect(partial.snapshot().samples == 2 && !partial.snapshot().trained_cases,
           "original measurements alone do not complete private candidate qualification");
    const auto before = capture(partial);
    expect(!partial.original_measurement(incomplete, shape) &&
               !partial.original_measurement(code, shape),
           "partial catalog cannot supply an original view");
    unchanged(partial, before);
    CudaProfileCache restored_partial(config, clock_ns, false);
    expect(restored_partial.restore(before.archive, kRuntimeIdentity) &&
               restored_partial.snapshot().functions == 0 &&
               !restored_partial.original_measurement(code, shape),
           "incomplete catalog cannot enter a readonly cache through an archive");

    auto active = partial.choose(incomplete, shape, 160, config.kinds);
    expect(active.was_selected && active.workers && active.observation,
           "actual next private candidate holds its once-only active reservation");
    now_ns.store(1000, std::memory_order_relaxed);
    active.observation->begin();
    const auto held = capture(partial);
    expect(!partial.original_measurement(incomplete, shape) &&
               !partial.original_measurement(code, shape),
           "an active reservation cannot expose an original view");
    unchanged(partial, held);
    CudaProfileCache restored_active(config, clock_ns, false);
    expect(restored_active.restore(held.archive, kRuntimeIdentity) &&
               restored_active.snapshot().functions == 0 &&
               !restored_active.original_measurement(code, shape),
           "active cases are omitted rather than reopened as readonly measurements");
    // Readonly choose cannot reserve a sample, so there is no public way to
    // manufacture an active readonly case. Preserve that invariant, not a seam.
    const auto readonly_cold = capture(restored_active);
    expect(!restored_active.original_measurement(incomplete, shape),
           "learning reservation capability has no authority in a readonly owner");
    unchanged(restored_active, readonly_cold);
    active.observation.reset();
    expect(partial.snapshot().cancelled == 1,
           "abandoned active sample keeps its actual once-only cancellation accounting");
}

void const_views_pin_existing_identity_and_outlive_the_cache() {
    auto config = configuration();
    config.max_functions = 1;
    const auto code = identity();
    const auto shape = geometry();
    const auto archive = trained_archive(config, {code}, shape);
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(archive, kRuntimeIdentity), "restore single-capacity identity owner");
    auto view = cache.original_measurement(code, shape);
    measurement(view, code, shape);
    auto other = code;
    other.name += "_replacement";
    expect(
        !bind(cache, other) && cache.snapshot().functions == 1 && cache.snapshot().evictions == 0,
        "const measurement aliases and pins the existing code, not a detached copy");
    measurement(cache.original_measurement(code, shape), code, shape);
    view.reset();
    const auto replacement = bind(cache, other);
    expect(
        replacement && cache.snapshot().evictions == 1 && !cache.original_measurement(code, shape),
        "dropping the last measurement permits actual bounded code eviction");

    std::optional<CudaProfileOriginalMeasurement> retained;
    std::weak_ptr<const CudaProfileCodeIdentity> weak;
    {
        CudaProfileCache temporary(config, clock_ns, false);
        expect(temporary.restore(archive, kRuntimeIdentity), "restore scoped code owner");
        retained = temporary.original_measurement(code, shape);
        measurement(retained, code, shape);
        weak = retained->identity;
    }
    expect(!weak.expired(), "immutable measurement keeps exact bytes after cache destruction");
    measurement(retained, code, shape);
    retained.reset();
    expect(weak.expired(), "last immutable measurement releases its sole surviving code pin");
}

void concurrent_readonly_calls_and_fork_guards() {
    const auto config = configuration();
    const auto code = identity();
    auto unknown = code;
    unknown.binary += "not the same binary";
    const auto shape = geometry();
    const auto archive = trained_archive(config, {code}, shape);
    CudaProfileCache cache(config, clock_ns, false);
    expect(cache.restore(archive, kRuntimeIdentity), "restore concurrent immutable catalog");
    const auto function = bind(cache, code);
    const auto expected = cache.original_measurement(code, shape);
    measurement(expected, code, shape);
    const auto before = capture(cache);
    std::vector<std::future<bool>> readers;
    for (unsigned int thread = 0; thread < 8; ++thread) {
        readers.push_back(std::async(std::launch::async, [&] {
            const CudaProfileCache& readonly = cache;
            for (unsigned int repeat = 0; repeat < 256; ++repeat) {
                const auto by_function = readonly.original_measurement(function, shape);
                const auto by_identity = readonly.original_measurement(code, shape);
                if (!by_function || !by_identity || by_function->identity != expected->identity ||
                    by_identity->identity != expected->identity || by_function->geometry != shape ||
                    by_identity->geometry != shape || by_function->total_ns != kOriginalTotal ||
                    by_identity->total_ns != kOriginalTotal || by_function->samples != 2 ||
                    by_identity->samples != 2 || readonly.original_measurement(unknown, shape))
                    return false;
            }
            return true;
        }));
    }
    for (auto& reader : readers)
        expect(reader.get(), "concurrent const readers keep exact aliases and complete totals");
    unchanged(cache, before);

    const auto child = fork();
    expect(child >= 0, "fork inherited readonly measurement control");
    if (child == 0) {
        const auto calls = clock_calls.load(std::memory_order_relaxed);
        const bool rejected = !cache.original_measurement(function, shape) &&
                              !cache.original_measurement(code, shape) &&
                              !cache.snapshot().is_valid &&
                              clock_calls.load(std::memory_order_relaxed) == calls;
        // Inherited cache ownership is intentionally invalid in this child.
        _exit(rejected ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "forked lookups reject before inherited locks, allocations or clock calls");
    measurement(cache.original_measurement(function, shape), code, shape);
    unchanged(cache, before);
}
}  // namespace

int main() {
    try {
        complete_exact_readonly_views();
        every_identity_and_geometry_component_is_exact();
        trained_overflowing_launch_volumes_still_miss();
        archive_resolution_does_not_bind_or_borrow_authority();
        learning_partial_and_active_cases_never_escape();
        const_views_pin_existing_identity_and_outlive_the_cache();
        concurrent_readonly_calls_and_fork_guards();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
