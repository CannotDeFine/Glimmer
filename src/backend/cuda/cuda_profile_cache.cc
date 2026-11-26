#include "internal/cuda_profile_cache.h"
#include "internal/cache_archive.h"
#include "internal/profile_configuration_archive.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
constexpr std::uint64_t kMaxSampleNs = 3600ULL * 1000 * 1000 * 1000;
void increment(std::uint64_t& value) noexcept {
    if (value != UINT64_MAX) ++value;
}
std::uint64_t monotonic_ns() noexcept {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}
bool valid_kind(transform::PtxExecutionKind kind) noexcept {
    unsigned int bits{};
    static_assert(sizeof(bits) == sizeof(kind));
    std::memcpy(&bits, &kind, sizeof(bits));
    return bits <= static_cast<unsigned int>(transform::PtxExecutionKind::kSliced);
}
struct Candidate {
    unsigned int workers{};
    transform::PtxExecutionKind kind{};
    std::uint32_t warmups{}, samples{};
    std::uint64_t total_ns{};
};
struct ProfileCase {
    CudaProfileGeometry geometry;
    std::array<Candidate, 36> candidates{};
    std::size_t count{};
    std::uint64_t generation{}, last_use{};
    bool is_active{};
};
struct SampleCoordinates {
    std::size_t slot{}, candidate{};
    std::uint64_t generation{};
};
bool trained(const ProfileCase& record, const CudaProfileConfig& config) noexcept {
    return record.generation && !record.is_active && record.count &&
           std::ranges::all_of(
               std::span{record.candidates}.first(record.count),
               [&](const auto& value) { return value.samples == config.samples_per_candidate; });
}
std::optional<std::uint64_t> launch_volume(const std::array<unsigned int, 3>& dimensions) noexcept {
    std::uint64_t result{1};
    for (const auto value : dimensions) {
        if (!value || result > UINT64_MAX / value) return std::nullopt;
        result *= value;
    }
    return result;
}
struct SharingCandidates {
    std::array<Candidate, 4> values{};
    std::size_t count{};
};
std::optional<SharingCandidates> sharing_candidates(
    const CudaProfileConfig& config, std::string_view name, CudaProfileGeometry geometry,
    CudaProfileSharingLimits limits,
    std::span<const CudaProfileSharingCandidate> eligible) noexcept {
    if (!valid_kind(geometry.kind) || !cuda_profile_context_configuration_valid(geometry.context) ||
        !cuda_profile_function_configuration_valid(geometry.function) || !limits.sm_count ||
        !limits.threads_per_sm || !limits.max_workers || !limits.minimum_speed_percent ||
        limits.minimum_speed_percent > 10000 || eligible.size() > 4)
        return std::nullopt;
    for (std::size_t index = 0; index < eligible.size(); ++index) {
        const auto kind = eligible[index].kind;
        if (!valid_kind(kind) || kind == transform::PtxExecutionKind::kPreemptive ||
            std::ranges::any_of(eligible.first(index),
                                [&](const auto& prior) { return prior.kind == kind; }) ||
            (config.kinds.empty() ? kind != geometry.kind
                                  : std::ranges::find(config.kinds, kind) == config.kinds.end()))
            return std::nullopt;
    }
    const auto blocks = launch_volume(geometry.grid), threads = launch_volume(geometry.block);
    if (!blocks || !threads) return std::nullopt;
    SharingCandidates result;
    const auto blocks_per_sm = limits.threads_per_sm / *threads;
    // Division avoids overflowing ceil(logical CTAs / SMs) * threads per CTA.
    const auto logical_blocks_per_sm = *blocks / limits.sm_count + (*blocks % limits.sm_count != 0);
    if (!blocks_per_sm || logical_blocks_per_sm <= blocks_per_sm) return result;
    const auto thread_workers = blocks_per_sm * limits.sm_count;
    const bool has_unsafe_static_name =
        name.find("DeviceSelectSweepKernel") != std::string_view::npos ||
        name.find("DeviceScanKernel") != std::string_view::npos;
    for (const auto& candidate : eligible) {
        if (candidate.kind == transform::PtxExecutionKind::kStatic &&
            (geometry.grid[1] != 1 || geometry.grid[2] != 1))
            continue;
        if (has_unsafe_static_name && (candidate.kind == transform::PtxExecutionKind::kStatic ||
                                       candidate.kind == transform::PtxExecutionKind::kBlockStatic))
            continue;
        unsigned int workers{};
        for (const auto configured : config.workers)
            if (configured % limits.sm_count == 0 && configured <= limits.max_workers &&
                configured <= candidate.max_workers && configured <= thread_workers)
                workers = std::max(workers, configured);
        if (workers) result.values[result.count++] = {.workers = workers, .kind = candidate.kind};
    }
    return result;
}
bool contains_candidates(const ProfileCase& record, std::span<const Candidate> requested) noexcept {
    const auto candidates = std::span{record.candidates}.first(record.count);
    return std::ranges::all_of(requested, [&](const auto& requested_candidate) {
        return std::ranges::any_of(candidates, [&](const auto& candidate) {
            return candidate.workers == requested_candidate.workers &&
                   candidate.kind == requested_candidate.kind;
        });
    });
}
using internal::ArchiveReader;
using internal::ArchiveWriter;
using internal::checksum;
}  // namespace

bool cuda_profile_context_configuration_valid(
    const CudaProfileContextConfiguration& configuration) noexcept {
    constexpr auto mask = (std::uint32_t{1} << 19) - 1;
    if ((configuration.available & ~mask) ||
        (configuration.available && (configuration.available & 7U) != 7U) ||
        configuration.values[0] > UINT32_MAX || configuration.values[1] > 3 ||
        configuration.values[2] > 2)
        return false;
    for (std::size_t index = 0; index < configuration.values.size(); ++index)
        if (!(configuration.available & (std::uint32_t{1} << index)) && configuration.values[index])
            return false;
    return true;
}

bool cuda_profile_function_configuration_valid(
    const CudaProfileFunctionConfiguration& configuration) noexcept {
    if ((configuration.functions & ~63U) ||
        (configuration.functions && !(configuration.functions & 1U)) ||
        configuration.cache_preference > 4 ||
        (!configuration.functions && configuration.cache_preference))
        return false;
    for (std::size_t role = 0; role < configuration.available.size(); ++role) {
        const auto available = configuration.available[role];
        const bool present = (configuration.functions & (1U << role)) != 0;
        if ((available & ~65535U) || (present ? (available & 1023U) != 1023U : available != 0))
            return false;
        for (std::size_t attribute = 0; attribute < configuration.values[role].size();
             ++attribute) {
            const auto value = configuration.values[role][attribute];
            if (!(available & (1U << attribute))) {
                if (value) return false;
                continue;
            }
            if (attribute == 9 ? value < -1 || value > 100 : value < 0) return false;
            if ((attribute == 7 || attribute == 10 || attribute == 14) && value > 1) return false;
            if (attribute == 15 && value > 2) return false;
            if (attribute == 0 && !value) return false;
        }
    }
    return true;
}

struct CudaProfileCode : CudaProfileCodeIdentity {
    std::array<ProfileCase, 16> cases{};
    std::size_t bytes{};
};

namespace {
std::optional<CudaProfileOriginalMeasurement> qualified_original(
    const std::shared_ptr<CudaProfileCode>& code, CudaProfileGeometry geometry,
    const CudaProfileConfig& config) noexcept {
    if (!valid_kind(geometry.kind) || !cuda_profile_context_configuration_valid(geometry.context) ||
        !cuda_profile_function_configuration_valid(geometry.function) ||
        !launch_volume(geometry.grid) || !launch_volume(geometry.block))
        return std::nullopt;
    const ProfileCase* found{};
    for (const auto& record : code->cases)
        if (record.geometry == geometry && trained(record, config) &&
            (!found || record.last_use > found->last_use))
            found = &record;
    if (!found) return std::nullopt;
    const auto& original = found->candidates[0];
    if (original.workers || !original.samples || !original.total_ns) return std::nullopt;
    return CudaProfileOriginalMeasurement{
        std::static_pointer_cast<const CudaProfileCodeIdentity>(code), geometry, original.total_ns,
        original.samples};
}
}  // namespace

struct CudaProfileCache::Implementation {
    const CudaProfileConfig config;
    const CudaProfileClock clock;
    const bool is_learning;
    const pid_t process{::getpid()};
    mutable std::mutex mutex;
    std::list<std::shared_ptr<CudaProfileCode>> codes;
    std::size_t retained_bytes{};
    std::uint64_t generation{}, samples{}, rejected{}, cancelled{}, hits{}, bypasses{}, evictions{};
    std::uint64_t loaded_cases{};

    Implementation(CudaProfileConfig value, CudaProfileClock timer, bool learns)
        : config(std::move(value)), clock(timer ? timer : monotonic_ns), is_learning(learns) {
        if (!config.max_functions || config.max_functions > 1024 || !config.max_bytes ||
            !config.samples_per_candidate || config.samples_per_candidate > 1000 ||
            config.warmup_calls > 1000 || config.workers.empty() || config.workers.size() > 7 ||
            config.kinds.size() > 5)
            throw std::invalid_argument("invalid profile cache bounds");
        for (std::size_t index = 0; index < config.workers.size(); ++index) {
            const auto workers = config.workers[index];
            if (!workers || workers > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
                std::ranges::find(std::span{config.workers}.first(index), workers) !=
                    std::span{config.workers}.first(index).end())
                throw std::invalid_argument("invalid or duplicate profile worker candidate");
        }
        for (std::size_t index = 0; index < config.kinds.size(); ++index) {
            const auto prior = std::span{config.kinds}.first(index);
            if (!valid_kind(config.kinds[index]) ||
                std::ranges::find(prior, config.kinds[index]) != prior.end())
                throw std::invalid_argument("invalid or duplicate profile family");
        }
    }
    bool evict() noexcept {
        const auto found = std::find_if(codes.rbegin(), codes.rend(),
                                        [](const auto& code) { return code.use_count() == 1; });
        if (found == codes.rend()) return false;
        retained_bytes -= (*found)->bytes;
        codes.erase(std::next(found).base());
        increment(evictions);
        return true;
    }
    class Sample final : public CudaExecutionObserver {
       public:
        Sample(std::shared_ptr<Implementation> owner, std::shared_ptr<CudaProfileCode> code,
               SampleCoordinates coordinates)
            : owner_(std::move(owner)),
              code_(std::move(code)),
              slot_(coordinates.slot),
              candidate_(coordinates.candidate),
              generation_(coordinates.generation) {}
        ~Sample() override {
            if (is_finished_ || ::getpid() != owner_->process) return;
            const std::lock_guard lock(owner_->mutex);
            auto& record = code_->cases[slot_];
            if (record.generation == generation_ && record.is_active) {
                record.is_active = false;
                increment(owner_->cancelled);
            }
        }
        void begin() noexcept override {
            if (has_started_ || is_finished_ || ::getpid() != owner_->process) return;
            has_started_ = true;
            start_ns_ = owner_->clock();
        }
        void finish(bool is_successful) noexcept override {
            if (is_finished_ || ::getpid() != owner_->process) return;
            is_finished_ = true;
            const auto end_ns = owner_->clock();
            const std::lock_guard lock(owner_->mutex);
            auto& record = code_->cases[slot_];
            if (record.generation != generation_ || !record.is_active) return;
            record.is_active = false;
            // Bound invalid/backwards clocks and implausibly long stalled inputs.
            if (!is_successful || !has_started_ || !start_ns_ || end_ns <= start_ns_ ||
                end_ns - start_ns_ > kMaxSampleNs) {
                increment(owner_->rejected);
                return;
            }
            auto& candidate = record.candidates[candidate_];
            if (candidate.warmups < owner_->config.warmup_calls) {
                ++candidate.warmups;
            } else if (candidate.samples < owner_->config.samples_per_candidate) {
                ++candidate.samples;
                candidate.total_ns += end_ns - start_ns_;
                increment(owner_->samples);
            }
        }

       private:
        std::shared_ptr<Implementation> owner_;
        std::shared_ptr<CudaProfileCode> code_;
        std::size_t slot_, candidate_;
        std::uint64_t generation_, start_ns_{};
        bool has_started_{}, is_finished_{};
    };
};

CudaProfileCache::CudaProfileCache(CudaProfileConfig config, CudaProfileClock clock, bool learns)
    : implementation_(std::make_shared<Implementation>(std::move(config), clock, learns)) {}
CudaProfileCache::~CudaProfileCache() = default;
CudaProfileFunction CudaProfileCache::bind(const CudaProfilePlatform& platform,
                                           std::string_view source,
                                           std::span<const std::byte> binary,
                                           std::string_view name) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !platform.driver_version || !platform.toolkit_version ||
        !platform.generator_abi ||
        std::ranges::all_of(platform.uuid, [](auto value) { return value == std::byte{}; }) ||
        (source.empty() && binary.empty()) || (!source.empty() && !source.data()) ||
        (!binary.empty() && !binary.data()) || name.empty() || !name.data() || name.size() > 4096 ||
        name.find('\0') != std::string_view::npos || source.size() > state.config.max_bytes ||
        binary.size() > state.config.max_bytes - source.size())
        return {};
    try {
        const std::lock_guard lock(state.mutex);
        const auto binary_text =
            binary.empty()
                ? std::string_view{}
                : std::string_view{reinterpret_cast<const char*>(binary.data()), binary.size()};
        const auto found = std::ranges::find_if(state.codes, [&](const auto& code) {
            return code->platform == platform && code->name == name && code->source == source &&
                   code->binary == binary_text;
        });
        CudaProfileFunction result;
        result.owner_ = implementation_;
        if (found != state.codes.end()) {
            result.code_ = *found;
            state.codes.splice(state.codes.begin(), state.codes, found);
            return result;
        }
        auto code = std::make_shared<CudaProfileCode>();
        code->platform = platform;
        code->source = source;
        code->binary = binary_text;
        code->name = name;
        auto bytes = sizeof(CudaProfileCode);
        for (const auto* text : {&code->source, &code->binary, &code->name}) {
            if (text->capacity() >= SIZE_MAX - bytes) return {};
            bytes += text->capacity() + 1;
        }
        code->bytes = bytes;
        if (bytes > state.config.max_bytes) {
            increment(state.bypasses);
            return {};
        }
        while (state.codes.size() >= state.config.max_functions ||
               bytes > state.config.max_bytes - state.retained_bytes) {
            if (!state.evict()) {
                increment(state.bypasses);
                return {};
            }
        }
        state.codes.push_front(code);
        state.retained_bytes += bytes;
        result.code_ = std::move(code);
        return result;
    } catch (...) {
        return {};
    }
}

CudaProfileChoice CudaProfileCache::choose(const CudaProfileFunction& function,
                                           CudaProfileGeometry geometry, unsigned int max_workers,
                                           bool can_transform) noexcept {
    const std::array kinds{geometry.kind};
    return choose(
        function, geometry, max_workers,
        can_transform ? std::span{kinds} : std::span<const transform::PtxExecutionKind>{});
}
std::span<const transform::PtxExecutionKind> CudaProfileCache::kinds() const noexcept {
    return ::getpid() == implementation_->process
               ? std::span<const transform::PtxExecutionKind>{implementation_->config.kinds}
               : std::span<const transform::PtxExecutionKind>{};
}
bool CudaProfileCache::is_learning() const noexcept {
    return ::getpid() == implementation_->process && implementation_->is_learning;
}
CudaProfileChoice CudaProfileCache::choose(
    const CudaProfileFunction& function, CudaProfileGeometry geometry, unsigned int max_workers,
    std::span<const transform::PtxExecutionKind> eligible_kinds,
    bool requires_preemption) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || function.owner_.lock() != implementation_ ||
        !function.code_ || !valid_kind(geometry.kind) ||
        !cuda_profile_context_configuration_valid(geometry.context) ||
        !cuda_profile_function_configuration_valid(geometry.function) ||
        eligible_kinds.size() > 5 ||
        std::ranges::any_of(geometry.grid, [](auto value) { return !value; }) ||
        std::ranges::any_of(geometry.block, [](auto value) { return !value; }))
        return {};
    for (std::size_t index = 0; index < eligible_kinds.size(); ++index) {
        const auto kind = eligible_kinds[index];
        const auto prior = eligible_kinds.first(index);
        if (!valid_kind(kind) || std::ranges::find(prior, kind) != prior.end() ||
            (!state.config.kinds.empty() &&
             std::ranges::find(state.config.kinds, kind) == state.config.kinds.end()) ||
            (state.config.kinds.empty() && kind != geometry.kind))
            return {};
    }
    try {
        std::array<Candidate, 36> candidates{};
        candidates[0].kind = geometry.kind;
        std::size_t count = 1;
        for (const auto kind : eligible_kinds) {
            if (kind == transform::PtxExecutionKind::kStatic &&
                (geometry.grid[1] != 1 || geometry.grid[2] != 1))
                continue;
            for (auto value : state.config.workers) {
                if (value > max_workers) continue;
                if (kind == transform::PtxExecutionKind::kStatic)
                    value = std::min(value, geometry.grid[0]);
                if (std::ranges::none_of(std::span{candidates}.first(count),
                                         [&](const auto& entry) {
                                             return entry.workers == value && entry.kind == kind;
                                         }))
                    candidates[count++] = {.workers = value, .kind = kind};
            }
        }
        const std::lock_guard lock(state.mutex);
        if (state.generation >= UINT64_MAX - 1) return {};
        auto& cases = function.code_->cases;
        auto found = std::ranges::find_if(cases, [&](const auto& entry) {
            if (!entry.generation || entry.geometry != geometry || entry.count != count)
                return false;
            for (std::size_t index = 0; index < count; ++index)
                if (entry.candidates[index].workers != candidates[index].workers ||
                    entry.candidates[index].kind != candidates[index].kind)
                    return false;
            return true;
        });
        if (found == cases.end()) {
            if (!state.is_learning) {
                increment(state.bypasses);
                return {};
            }
            found = cases.end();
            for (auto entry = cases.begin(); entry != cases.end(); ++entry)
                if (!entry->is_active &&
                    (found == cases.end() || entry->last_use < found->last_use))
                    found = entry;
            if (found == cases.end()) {
                increment(state.bypasses);
                return {};
            }
            if (found->generation) increment(state.evictions);
            *found = {};
            found->geometry = geometry;
            found->generation = ++state.generation;
            found->count = count;
            found->candidates = candidates;
        }
        found->last_use = ++state.generation;
        if (found->is_active) {
            increment(state.bypasses);
            return {};
        }
        for (std::size_t index = 0; index < count; ++index) {
            const auto& candidate = found->candidates[index];
            if (candidate.samples >= state.config.samples_per_candidate) continue;
            if (!state.is_learning || requires_preemption) {
                increment(state.bypasses);
                return {};
            }
            auto sample = std::make_shared<Implementation::Sample>(
                implementation_, function.code_,
                SampleCoordinates{static_cast<std::size_t>(found - cases.begin()), index,
                                  found->generation});
            found->is_active = true;
            return {candidate.workers, std::move(sample), true, false, candidate.kind};
        }
        const Candidate* best{};
        for (const auto& candidate : std::span{found->candidates}.first(count)) {
            if (requires_preemption &&
                (!candidate.workers ||
                 (candidate.kind != transform::PtxExecutionKind::kPreemptive &&
                  candidate.kind != transform::PtxExecutionKind::kSliced)))
                continue;
            if (!best || static_cast<long double>(candidate.total_ns) / candidate.samples <
                             static_cast<long double>(best->total_ns) / best->samples)
                best = &candidate;
        }
        if (!best) {
            increment(state.bypasses);
            return {};
        }
        increment(state.hits);
        return {best->workers, {}, true, true, best->kind};
    } catch (...) {
        return {};
    }
}

CudaProfileChoice CudaProfileCache::choose_sharing(
    const CudaProfileFunction& function, CudaProfileGeometry geometry,
    CudaProfileSharingLimits limits,
    std::span<const CudaProfileSharingCandidate> eligible_candidates) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || function.owner_.lock() != implementation_ || !function.code_)
        return {};
    const auto requested = sharing_candidates(state.config, function.code_->name, geometry, limits,
                                              eligible_candidates);
    if (!requested || !requested->count) return {};
    try {
        const std::lock_guard lock(state.mutex);
        const auto candidates = std::span{requested->values}.first(requested->count);
        ProfileCase* found{};
        for (auto& record : function.code_->cases)
            if (record.geometry == geometry && trained(record, state.config) &&
                contains_candidates(record, candidates) &&
                (!found || record.last_use > found->last_use))
                found = &record;
        if (!found) {
            increment(state.bypasses);
            return {};
        }
        const Candidate* best{};
        for (const auto& candidate : std::span{found->candidates}.first(found->count)) {
            if (!std::ranges::any_of(candidates, [&](const auto& requested_candidate) {
                    return candidate.workers == requested_candidate.workers &&
                           candidate.kind == requested_candidate.kind;
                }))
                continue;
            // Every candidate has the same fully trained sample count.
            if (!best || candidate.total_ns < best->total_ns) best = &candidate;
        }
        if (!best) return {};
        const auto& original = found->candidates[0];
        // total_ns <= kMaxSampleNs * 1000; multiplying only by 100 is safe.
        // Integer division preserves the exact inclusive percentage boundary.
        if (original.total_ns * 100 / limits.minimum_speed_percent < best->total_ns)
            best = &original;
        increment(state.generation);
        found->last_use = state.generation;
        increment(state.hits);
        return {best->workers, {}, true, true, best->kind};
    } catch (...) {
        return {};
    }
}

std::optional<CudaProfileOriginalMeasurement> CudaProfileCache::original_measurement(
    const CudaProfileFunction& function, CudaProfileGeometry geometry) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || state.is_learning ||
        function.owner_.lock() != implementation_ || !function.code_)
        return std::nullopt;
    try {
        const std::lock_guard lock(state.mutex);
        return qualified_original(function.code_, geometry, state.config);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<CudaProfileOriginalMeasurement> CudaProfileCache::original_measurement(
    const CudaProfileCodeIdentity& identity, CudaProfileGeometry geometry) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || state.is_learning || identity.name.empty() ||
        identity.name.size() > 4096 || identity.name.find('\0') != std::string::npos ||
        identity.source.size() > state.config.max_bytes ||
        identity.binary.size() > state.config.max_bytes - identity.source.size())
        return std::nullopt;
    try {
        const std::lock_guard lock(state.mutex);
        const auto code = std::ranges::find_if(state.codes, [&](const auto& candidate) {
            return static_cast<const CudaProfileCodeIdentity&>(*candidate) == identity;
        });
        return code == state.codes.end() ? std::nullopt
                                         : qualified_original(*code, geometry, state.config);
    } catch (...) {
        return std::nullopt;
    }
}

CudaProfileSnapshot CudaProfileCache::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process) return {};
    const std::lock_guard lock(state.mutex);
    CudaProfileSnapshot result{state.codes.size(),
                               state.retained_bytes,
                               0,
                               0,
                               state.samples,
                               state.rejected,
                               state.cancelled,
                               state.hits,
                               state.bypasses,
                               state.evictions,
                               true};
    result.loaded_cases = state.loaded_cases;
    result.is_learning = state.is_learning;
    for (const auto& code : state.codes)
        for (const auto& entry : code->cases) {
            if (!entry.generation) continue;
            ++result.cases;
            if (std::ranges::all_of(
                    std::span{entry.candidates}.first(entry.count), [&](const auto& candidate) {
                        return candidate.samples >= state.config.samples_per_candidate;
                    }))
                ++result.trained_cases;
        }
    return result;
}

std::optional<std::vector<std::byte>> CudaProfileCache::encode(
    std::string_view runtime_identity) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || runtime_identity.empty() || runtime_identity.size() > 4096 ||
        runtime_identity.find('\0') != std::string_view::npos)
        return std::nullopt;
    try {
        const std::lock_guard lock(state.mutex);
        const bool has_configuration = std::ranges::any_of(state.codes, [&](const auto& code) {
            return std::ranges::any_of(code->cases, [&](const auto& entry) {
                return trained(entry, state.config) && entry.geometry.context.available;
            });
        });
        const bool has_function_configuration =
            std::ranges::any_of(state.codes, [&](const auto& code) {
                return std::ranges::any_of(code->cases, [&](const auto& entry) {
                    return trained(entry, state.config) && entry.geometry.function.functions;
                });
            });
        const unsigned int version = has_function_configuration ? 3 : has_configuration ? 2 : 1;
        ArchiveWriter output{kCudaProfileArchiveMaxBytes};
        output.text("Glimmer profile archive");
        output.number(version);
        output.number(1);  // Final-retirement host-observed nanoseconds, not GPU cycles.
        output.text(runtime_identity);
        const auto& config = state.config;
        output.number(config.warmup_calls);
        output.number(config.samples_per_candidate);
        output.number(config.workers.size());
        for (const auto workers : config.workers) output.number(workers);
        output.number(config.kinds.size());
        for (const auto kind : config.kinds) output.number(static_cast<unsigned int>(kind));
        {
            const auto complete_count = [&](const auto& code) {
                return std::ranges::count_if(
                    code->cases, [&](const auto& record) { return trained(record, config); });
            };
            output.number(static_cast<std::size_t>(std::ranges::count_if(
                state.codes, [&](const auto& code) { return complete_count(code) != 0; })));
            for (const auto& code : state.codes) {
                const auto count = complete_count(code);
                if (!count) continue;
                for (const auto byte : code->platform.uuid)
                    output.number(std::to_integer<unsigned int>(byte), 1);
                output.number(code->platform.driver_version);
                output.number(code->platform.toolkit_version);
                output.number(code->platform.generator_abi);
                output.text(code->source);
                output.text(code->binary);
                output.text(code->name);
                output.number(static_cast<std::size_t>(count));
                for (const auto& record : code->cases) {
                    if (!trained(record, config)) continue;
                    for (const auto value : record.geometry.grid) output.number(value);
                    for (const auto value : record.geometry.block) output.number(value);
                    output.number(record.geometry.shared_memory_bytes);
                    output.number(static_cast<unsigned int>(record.geometry.kind));
                    if (version >= 2)
                        internal::write_profile_configuration(output, record.geometry.context);
                    if (version >= 3)
                        internal::write_function_configuration(output, record.geometry.function);
                    output.number(record.count);
                    for (const auto& candidate : std::span{record.candidates}.first(record.count)) {
                        output.number(candidate.workers);
                        output.number(static_cast<unsigned int>(candidate.kind));
                        output.number(candidate.total_ns, 8);
                    }
                }
            }
        }
        output.number(checksum(output.bytes), 8);
        return std::move(output.bytes);
    } catch (...) {
        return std::nullopt;
    }
}

bool CudaProfileCache::restore(std::span<const std::byte> archive,
                               std::string_view runtime_identity) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !archive.data() || archive.size() < 8 ||
        archive.size() > kCudaProfileArchiveMaxBytes || runtime_identity.empty() ||
        runtime_identity.size() > 4096 || runtime_identity.find('\0') != std::string_view::npos)
        return false;
    try {
        ArchiveReader digest{archive.last(8)}, input{archive.first(archive.size() - 8)};
        if (checksum(input.bytes) != digest.number(8) || input.text() != "Glimmer profile archive")
            return false;
        const auto version = input.number();
        if ((version != 1 && version != 2 && version != 3) || input.number() != 1 ||
            input.text() != runtime_identity)
            return false;
        const auto& config = state.config;
        if (input.number() != config.warmup_calls ||
            input.number() != config.samples_per_candidate ||
            input.number() != config.workers.size())
            return false;
        for (const auto workers : config.workers)
            if (input.number() != workers) return false;
        if (input.number() != config.kinds.size()) return false;
        for (const auto kind : config.kinds)
            if (input.kind() != kind) return false;
        const auto count = input.number();
        if (count > config.max_functions) return false;
        std::list<std::shared_ptr<CudaProfileCode>> codes;
        std::size_t retained{};
        std::uint64_t generation{};
        for (std::uint64_t index = 0; index < count; ++index) {
            auto code = std::make_shared<CudaProfileCode>();
            for (auto& byte : code->platform.uuid) byte = static_cast<std::byte>(input.number(1));
            code->platform.driver_version = static_cast<unsigned int>(input.number());
            code->platform.toolkit_version = static_cast<unsigned int>(input.number());
            code->platform.generator_abi = static_cast<unsigned int>(input.number());
            const auto source = input.text(), binary = input.text(), name = input.text();
            if (!code->platform.driver_version || !code->platform.toolkit_version ||
                !code->platform.generator_abi ||
                std::ranges::all_of(code->platform.uuid,
                                    [](auto byte) { return byte == std::byte{}; }) ||
                (source.empty() && binary.empty()) || name.empty() || name.size() > 4096 ||
                name.find('\0') != std::string_view::npos || source.size() > config.max_bytes ||
                binary.size() > config.max_bytes - source.size())
                return false;
            code->source = source;
            code->binary = binary;
            code->name = name;
            code->bytes = sizeof(CudaProfileCode);
            for (const auto* text : {&code->source, &code->binary, &code->name}) {
                if (text->capacity() >= SIZE_MAX - code->bytes) return false;
                code->bytes += text->capacity() + 1;
            }
            if (code->bytes > config.max_bytes - retained) return false;
            retained += code->bytes;
            if (std::ranges::any_of(codes, [&](const auto& previous) {
                    return previous->platform == code->platform && previous->source == source &&
                           previous->binary == binary && previous->name == name;
                }))
                return false;
            const auto cases = input.number();
            if (!cases || cases > code->cases.size()) return false;
            for (std::size_t slot = 0; slot < cases; ++slot) {
                auto& record = code->cases[slot];
                for (auto& value : record.geometry.grid)
                    value = static_cast<unsigned int>(input.number());
                for (auto& value : record.geometry.block)
                    value = static_cast<unsigned int>(input.number());
                record.geometry.shared_memory_bytes = static_cast<unsigned int>(input.number());
                record.geometry.kind = input.kind();
                if (version >= 2)
                    record.geometry.context = internal::read_profile_configuration(input);
                if (version >= 3)
                    record.geometry.function = internal::read_function_configuration(input);
                record.count = static_cast<std::size_t>(input.number());
                if (!record.count || record.count > record.candidates.size() ||
                    !cuda_profile_context_configuration_valid(record.geometry.context) ||
                    !cuda_profile_function_configuration_valid(record.geometry.function) ||
                    std::ranges::any_of(record.geometry.grid, [](auto value) { return !value; }) ||
                    std::ranges::any_of(record.geometry.block, [](auto value) { return !value; }))
                    return false;
                for (std::size_t entry = 0; entry < record.count; ++entry) {
                    auto& candidate = record.candidates[entry];
                    candidate.workers = static_cast<unsigned int>(input.number());
                    candidate.kind = input.kind();
                    candidate.total_ns = input.number(8);
                    candidate.samples = config.samples_per_candidate;
                    candidate.warmups = config.warmup_calls;
                    if (candidate.total_ns < candidate.samples ||
                        candidate.total_ns > kMaxSampleNs * candidate.samples)
                        return false;
                    if (!entry) {
                        if (candidate.workers || candidate.kind != record.geometry.kind)
                            return false;
                        continue;
                    }
                    if (!candidate.workers ||
                        (config.kinds.empty() ? candidate.kind != record.geometry.kind
                                              : std::ranges::find(config.kinds, candidate.kind) ==
                                                    config.kinds.end()) ||
                        (candidate.kind == transform::PtxExecutionKind::kStatic &&
                         (record.geometry.grid[1] != 1 || record.geometry.grid[2] != 1)) ||
                        std::ranges::none_of(
                            config.workers,
                            [&](auto workers) {
                                return candidate.workers ==
                                       (candidate.kind == transform::PtxExecutionKind::kStatic
                                            ? std::min(workers, record.geometry.grid[0])
                                            : workers);
                            }) ||
                        std::ranges::any_of(std::span{record.candidates}.first(entry),
                                            [&](const auto& prior) {
                                                return prior.workers == candidate.workers &&
                                                       prior.kind == candidate.kind;
                                            }))
                        return false;
                }
                if (std::ranges::any_of(
                        std::span{code->cases}.first(slot), [&](const auto& previous) {
                            if (previous.geometry != record.geometry ||
                                previous.count != record.count)
                                return false;
                            for (std::size_t entry = 0; entry < record.count; ++entry)
                                if (previous.candidates[entry].kind !=
                                        record.candidates[entry].kind ||
                                    previous.candidates[entry].workers !=
                                        record.candidates[entry].workers)
                                    return false;
                            return true;
                        }))
                    return false;
                record.generation = record.last_use = ++generation;
            }
            codes.push_back(std::move(code));
        }
        if (!input.bytes.empty()) return false;
        const std::lock_guard lock(state.mutex);
        if (!state.codes.empty() || state.generation || state.loaded_cases) return false;
        state.codes = std::move(codes);
        state.retained_bytes = retained;
        state.generation = generation;
        state.loaded_cases = generation;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::backend::cuda
