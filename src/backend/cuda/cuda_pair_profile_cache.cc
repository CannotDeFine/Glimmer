#include "internal/cuda_pair_profile_cache.h"

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
#include <utility>

namespace glimmer::backend::cuda {
namespace {

using internal::ArchiveReader;
using internal::ArchiveWriter;
using internal::checksum;
using transform::PtxExecutionKind;
constexpr std::size_t kMaxCandidates = 64;
constexpr std::size_t kAllocationOverhead = 64;

void increment(std::uint64_t& value) noexcept {
    if (value != UINT64_MAX) ++value;
}

std::uint64_t monotonic_ns() noexcept {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

template <typename Enum>
std::uint32_t enum_bits(Enum value) noexcept {
    static_assert(sizeof(value) == sizeof(std::uint32_t));
    std::uint32_t result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool add(std::uint64_t& total, std::uint64_t value) noexcept {
    if (value > UINT64_MAX - total) return false;
    total += value;
    return true;
}

std::optional<std::uint64_t> product(std::uint64_t first, std::uint64_t second) noexcept {
    if (second && first > UINT64_MAX / second) return std::nullopt;
    return first * second;
}

std::optional<std::uint64_t> volume(const std::array<unsigned int, 3>& dimensions) noexcept {
    std::uint64_t result = 1;
    for (const auto value : dimensions) {
        if (!value) return std::nullopt;
        const auto next = product(result, value);
        if (!next) return std::nullopt;
        result = *next;
    }
    return result;
}

bool valid_measurement(const CudaProfileOriginalMeasurement& value) noexcept {
    return value.identity && value.samples && value.samples <= 1000 &&
           cuda_profile_context_configuration_valid(value.geometry.context) &&
           cuda_profile_function_configuration_valid(value.geometry.function) &&
           value.total_ns >= value.samples &&
           value.total_ns <= kCudaPairProfileMaxSampleNs * value.samples &&
           enum_bits(value.geometry.kind) <= enum_bits(PtxExecutionKind::kSliced) &&
           volume(value.geometry.grid).has_value() && volume(value.geometry.block).has_value();
}

bool same_measurement(const CudaProfileOriginalMeasurement& first,
                      const CudaProfileOriginalMeasurement& second) noexcept {
    return first.geometry == second.geometry && first.total_ns == second.total_ns &&
           first.samples == second.samples && *first.identity == *second.identity;
}

std::optional<std::size_t> identity_bytes(const CudaProfileOriginalMeasurement& value) noexcept {
    std::uint64_t total = sizeof(CudaProfileCodeIdentity) + 2 * kAllocationOverhead;
    for (const auto* text :
         {&value.identity->source, &value.identity->binary, &value.identity->name})
        if (!add(total, text->capacity()) || !add(total, 1)) return std::nullopt;
    if (total > SIZE_MAX) return std::nullopt;
    return static_cast<std::size_t>(total);
}

bool valid_catalog(const std::array<CudaProfileOriginalMeasurement, 2>& originals,
                   CudaPairProfileDomain domain, CudaPairProfileLimits limits,
                   std::span<const CudaPairProfileCandidate> candidates,
                   const CudaPairProfileConfig& config) noexcept {
    const auto domain_kind = enum_bits(domain.kind);
    if (domain_kind == 1 && originals[0].geometry.context.available &&
        originals[1].geometry.context.available &&
        originals[0].geometry.context != originals[1].geometry.context)
        return false;
    if (candidates.empty() || candidates.size() > config.max_candidates ||
        (domain_kind != 1 && domain_kind != 2) || !domain.labels[0] || !domain.labels[1] ||
        ((domain.labels[0] == domain.labels[1]) != (domain_kind == 1)) || !limits.sm_count ||
        !limits.blocks_per_sm || !limits.threads_per_sm || !limits.registers_per_sm ||
        !limits.shared_memory_bytes_per_sm || !limits.max_workers ||
        limits.max_workers > static_cast<unsigned int>(INT32_MAX) ||
        originals[0].identity->platform != originals[1].identity->platform)
        return false;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!cuda_pair_profile_candidate_fits({originals[0].geometry, originals[1].geometry},
                                              limits, candidate) ||
            std::ranges::find(candidates.first(index), candidate) != candidates.first(index).end())
            return false;
    }
    return true;
}

// Exact rational comparisons, not floating-point qualification. Each factor is
// at most 3.6e18 (1000 samples of at most one hour). A comparison of two summed
// ratios therefore needs fewer than 256 bits. The fixed limbs never allocate.
using Wide = std::array<std::uint32_t, 8>;
Wide wide(std::uint64_t value) noexcept {
    Wide result{};
    result[0] = static_cast<std::uint32_t>(value);
    result[1] = static_cast<std::uint32_t>(value >> 32);
    return result;
}
Wide multiply(const Wide& first, const Wide& second) noexcept {
    Wide result{};
    for (std::size_t left = 0; left < first.size(); ++left) {
        std::uint64_t carry{};
        for (std::size_t right = 0; right + left < first.size(); ++right) {
            const auto value =
                std::uint64_t{first[left]} * second[right] + result[left + right] + carry;
            result[left + right] = static_cast<std::uint32_t>(value);
            carry = value >> 32;
        }
    }
    return result;
}
Wide sum(const Wide& first, const Wide& second) noexcept {
    Wide result{};
    std::uint64_t carry{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        const auto value = std::uint64_t{first[index]} + second[index] + carry;
        result[index] = static_cast<std::uint32_t>(value);
        carry = value >> 32;
    }
    return result;
}
int compare(const Wide& first, const Wide& second) noexcept {
    for (std::size_t index = first.size(); index; --index) {
        if (first[index - 1] < second[index - 1]) return -1;
        if (first[index - 1] > second[index - 1]) return 1;
    }
    return 0;
}
struct Score {
    Wide numerator{}, denominator{};
};
int compare(const Score& first, const Score& second) noexcept {
    return compare(multiply(first.numerator, second.denominator),
                   multiply(second.numerator, first.denominator));
}

struct CandidateState {
    CudaPairProfileCandidate configuration;
    std::uint32_t warmups{}, samples{};
    std::array<std::uint64_t, 2> total_ns{};
};

bool valid_runtime(std::string_view value) noexcept {
    return !value.empty() && value.size() <= 4096 && value.find('\0') == std::string_view::npos;
}

}  // namespace

std::optional<int> cuda_pair_profile_compare_choices(const CudaPairProfileChoice& first,
                                                     const CudaPairProfileChoice& second) noexcept {
    const auto valid = [](const CudaPairProfileChoice& choice) {
        if (choice.first_original > 1 ||
            (choice.is_shared &&
             (!choice.is_trained || !choice.candidate || *choice.candidate >= kMaxCandidates)) ||
            (!choice.is_shared && choice.candidate))
            return false;
        for (const auto* measurements : {&choice.original, &choice.paired})
            for (const auto& value : *measurements) {
                if (!value.samples && !value.total_ns && !choice.is_shared) continue;
                if (!value.samples || value.samples > 1000 || value.total_ns < value.samples ||
                    value.total_ns > kCudaPairProfileMaxSampleNs * value.samples)
                    return false;
            }
        if (choice.original[0].samples && choice.original[1].samples) {
            const bool second_is_shorter =
                choice.original[1].total_ns * choice.original[0].samples <
                choice.original[0].total_ns * choice.original[1].samples;
            if (choice.first_original != (second_is_shorter ? 1U : 0U)) return false;
        }
        return true;
    };
    if (!valid(first) || !valid(second)) return std::nullopt;
    if (first.is_shared != second.is_shared) return first.is_shared ? 1 : -1;
    if (!first.is_shared) {
        const auto left = first.original[first.first_original];
        const auto right = second.original[second.first_original];
        if (!left.samples || !right.samples) return 0;
        return compare(wide(right.total_ns * left.samples), wide(left.total_ns * right.samples));
    }
    const auto score = [](const CudaPairProfileChoice& choice) {
        std::array<Wide, 2> numerators, denominators;
        for (std::size_t leg = 0; leg < 2; ++leg) {
            numerators[leg] = wide(choice.original[leg].total_ns * choice.paired[leg].samples);
            denominators[leg] = wide(choice.original[leg].samples * choice.paired[leg].total_ns);
        }
        return Score{
            sum(multiply(numerators[0], denominators[1]), multiply(numerators[1], denominators[0])),
            multiply(denominators[0], denominators[1])};
    };
    return compare(score(first), score(second));
}

bool cuda_pair_profile_candidate_fits(const std::array<CudaProfileGeometry, 2>& geometry,
                                      CudaPairProfileLimits limits,
                                      const CudaPairProfileCandidate& candidate) noexcept {
    if (!limits.sm_count || !limits.blocks_per_sm || !limits.threads_per_sm ||
        !limits.registers_per_sm || !limits.shared_memory_bytes_per_sm || !limits.max_workers ||
        limits.max_workers > static_cast<unsigned int>(INT32_MAX))
        return false;
    std::uint64_t blocks{}, threads{}, registers{}, shared{};
    for (std::size_t leg = 0; leg < 2; ++leg) {
        const auto& variant = candidate.legs[leg];
        const auto kind = enum_bits(variant.kind);
        const auto block_threads = volume(geometry[leg].block);
        if (!cuda_profile_context_configuration_valid(geometry[leg].context) ||
            !cuda_profile_function_configuration_valid(geometry[leg].function) ||
            (kind != enum_bits(PtxExecutionKind::kPreemptive) &&
             kind != enum_bits(PtxExecutionKind::kSliced)) ||
            !volume(geometry[leg].grid) || !variant.workers ||
            variant.workers > limits.max_workers || variant.workers % limits.sm_count ||
            !variant.occupancy_blocks_per_sm || !block_threads ||
            *block_threads > limits.threads_per_sm)
            return false;
        const std::uint64_t per_sm = variant.workers / limits.sm_count;
        if (per_sm > variant.occupancy_blocks_per_sm) return false;
        const auto leg_threads = product(per_sm, *block_threads);
        const auto leg_registers =
            leg_threads ? product(*leg_threads, variant.registers_per_thread) : std::nullopt;
        auto block_shared = variant.static_shared_memory_bytes;
        if (!add(block_shared, geometry[leg].shared_memory_bytes)) return false;
        const auto leg_shared = product(per_sm, block_shared);
        if (!leg_threads || !leg_registers || !leg_shared || !add(blocks, per_sm) ||
            !add(threads, *leg_threads) || !add(registers, *leg_registers) ||
            !add(shared, *leg_shared))
            return false;
    }
    return blocks <= limits.blocks_per_sm && threads <= limits.threads_per_sm &&
           registers <= limits.registers_per_sm && shared <= limits.shared_memory_bytes_per_sm;
}

struct CudaPairProfileKernelState {
    CudaProfileOriginalMeasurement measurement;
    std::size_t bytes{};
};

struct CudaPairProfileCache::Implementation {
    struct Sample;
    struct Pair {
        std::array<std::shared_ptr<CudaPairProfileKernelState>, 2> kernels;
        CudaPairProfileDomain domain;
        CudaPairProfileLimits limits;
        std::array<CandidateState, kMaxCandidates> candidates{};
        std::size_t count{};
        bool is_active{};
        std::weak_ptr<Sample> reservation;
    };
    struct Observer final : CudaExecutionObserver {
        std::shared_ptr<Sample> sample;
        const std::size_t leg;
        bool is_armed{};
        Observer(std::shared_ptr<Sample> value, std::size_t index)
            : sample(std::move(value)), leg(index) {}
        ~Observer() override;
        void begin() noexcept override;
        void finish(bool is_successful) noexcept override;
    };
    struct Leg {
        std::uint64_t begin_ns{}, end_ns{};
        bool has_begin{}, has_finish{}, is_clock_active{};
    };
    struct Sample {
        std::shared_ptr<Implementation> owner;
        std::shared_ptr<Pair> pair;
        std::size_t candidate{};
        std::array<Leg, 2> legs{};
        std::uint64_t last_clock_ns{};
        bool is_terminal{};
        Sample(std::shared_ptr<Implementation> value, std::shared_ptr<Pair> record,
               std::size_t index)
            : owner(std::move(value)), pair(std::move(record)), candidate(index) {}
        void reject(bool is_cancelled) noexcept {
            if (is_terminal) return;
            is_terminal = true;
            pair->is_active = false;
            increment(is_cancelled ? owner->cancelled : owner->rejected_samples);
        }
        void commit() noexcept {
            if (!legs[0].has_finish || !legs[1].has_finish) return;
            if (std::max(legs[0].begin_ns, legs[1].begin_ns) >=
                std::min(legs[0].end_ns, legs[1].end_ns)) {
                reject(false);
                return;
            }
            auto& entry = pair->candidates[candidate];
            if (entry.warmups < owner->config.warmup_calls) {
                ++entry.warmups;
            } else {
                for (std::size_t index = 0; index < 2; ++index)
                    entry.total_ns[index] += legs[index].end_ns - legs[index].begin_ns;
                ++entry.samples;
                increment(owner->samples);
            }
            is_terminal = true;
            pair->is_active = false;
        }
    };

    const std::shared_ptr<const CudaProfileCache> originals;
    const CudaPairProfileConfig config;
    const CudaProfileClock clock;
    const bool is_learning;
    const pid_t process{::getpid()};
    mutable std::mutex mutex;
    std::list<std::shared_ptr<CudaPairProfileKernelState>> kernels;
    std::list<std::shared_ptr<Pair>> pairs;
    std::size_t retained_bytes{};
    std::uint64_t samples{}, rejected_samples{}, cancelled{}, evictions{}, loaded_pairs{};

    Implementation(std::shared_ptr<const CudaProfileCache> source, CudaPairProfileConfig value,
                   CudaProfileClock timer, bool learns)
        : originals(std::move(source)),
          config(value),
          clock(timer ? timer : monotonic_ns),
          is_learning(learns) {
        if (!originals || !originals->snapshot().is_valid || originals->is_learning() ||
            !config.max_pairs || config.max_pairs > 1024 || !config.max_candidates ||
            config.max_candidates > kMaxCandidates || !config.max_bytes ||
            config.max_bytes > kCudaPairProfileArchiveMaxBytes || !config.samples_per_candidate ||
            config.samples_per_candidate > 1000 || config.warmup_calls > 1000)
            throw std::invalid_argument("invalid paired profile configuration");
    }
    static constexpr std::size_t pair_bytes() noexcept {
        return sizeof(Pair) + sizeof(Sample) + 2 * sizeof(Observer) + 6 * kAllocationOverhead;
    }
    bool trained(const Pair& pair) const noexcept {
        return !pair.is_active && pair.count &&
               std::ranges::all_of(std::span{pair.candidates}.first(pair.count),
                                   [&](const auto& entry) {
                                       return entry.warmups == config.warmup_calls &&
                                              entry.samples == config.samples_per_candidate;
                                   });
    }
    void prune_kernels() noexcept {
        for (auto iterator = kernels.begin(); iterator != kernels.end();) {
            if (iterator->use_count() == 1) {
                retained_bytes -= (*iterator)->bytes;
                iterator = kernels.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }
    bool evict() noexcept {
        for (auto iterator = pairs.end(); iterator != pairs.begin();) {
            --iterator;
            if (!(*iterator)->is_active && iterator->use_count() == 1) {
                retained_bytes -= pair_bytes();
                pairs.erase(iterator);
                increment(evictions);
                prune_kernels();
                return true;
            }
        }
        return false;
    }
    bool room(std::size_t bytes, bool needs_pair, bool needs_kernel) noexcept {
        prune_kernels();
        if (bytes > config.max_bytes) return false;
        while (bytes > config.max_bytes - retained_bytes ||
               (needs_pair && pairs.size() >= config.max_pairs) ||
               (needs_kernel && kernels.size() >= 2 * config.max_pairs))
            if (!evict()) return false;
        return true;
    }
    std::shared_ptr<CudaPairProfileKernelState> find_kernel(
        const CudaProfileOriginalMeasurement& measurement) const noexcept {
        for (const auto& kernel : kernels)
            if (same_measurement(kernel->measurement, measurement)) return kernel;
        return {};
    }
    bool matches(const Pair& pair,
                 const std::array<std::shared_ptr<CudaPairProfileKernelState>, 2>& inputs,
                 CudaPairProfileDomain domain, CudaPairProfileLimits limits,
                 std::span<const CudaPairProfileCandidate> catalog, bool reversed) const noexcept {
        if (pair.limits != limits || pair.domain.kind != domain.kind ||
            pair.count != catalog.size())
            return false;
        for (std::size_t leg = 0; leg < 2; ++leg) {
            const auto mapped = reversed ? 1 - leg : leg;
            if (pair.domain.labels[mapped] != domain.labels[leg] ||
                !same_measurement(pair.kernels[mapped]->measurement, inputs[leg]->measurement))
                return false;
            for (std::size_t index = 0; index < catalog.size(); ++index)
                if (pair.candidates[index].configuration.legs[mapped] != catalog[index].legs[leg])
                    return false;
        }
        return true;
    }
    std::pair<std::shared_ptr<Pair>, bool> find_pair(
        const std::array<std::shared_ptr<CudaPairProfileKernelState>, 2>& inputs,
        CudaPairProfileDomain domain, CudaPairProfileLimits limits,
        std::span<const CudaPairProfileCandidate> catalog) const noexcept {
        // Exact orientation wins, including identical identities with asymmetric
        // catalogs. Only then consider reversal of the complete key.
        for (const bool reversed : {false, true})
            for (const auto& pair : pairs)
                if (matches(*pair, inputs, domain, limits, catalog, reversed))
                    return {pair, reversed};
        return {};
    }
};

CudaPairProfileCache::Implementation::Observer::~Observer() {
    if (!is_armed || ::getpid() != sample->owner->process) return;
    const std::lock_guard lock(sample->owner->mutex);
    // A physically retired leg may release its lease before the other leg
    // retires. Only abandoning an unfinished leg cancels the joint sample.
    if (!sample->legs[leg].has_finish) sample->reject(true);
}

void CudaPairProfileCache::Implementation::Observer::begin() noexcept {
    auto& value = *sample;
    auto& state = *value.owner;
    if (::getpid() != state.process) return;
    std::uint64_t minimum_ns{};
    {
        const std::lock_guard lock(state.mutex);
        if (value.is_terminal) return;
        if (value.legs[leg].has_begin || value.legs[leg].has_finish) {
            value.reject(false);
            return;
        }
        value.legs[leg].has_begin = true;
        value.legs[leg].is_clock_active = true;
        minimum_ns = value.last_clock_ns;
    }
    const auto now = state.clock();
    const std::lock_guard lock(state.mutex);
    if (value.is_terminal) return;
    value.legs[leg].is_clock_active = false;
    value.legs[leg].begin_ns = now;
    if (!now || now < minimum_ns) {
        value.reject(false);
    } else {
        value.last_clock_ns = std::max(value.last_clock_ns, now);
    }
}

void CudaPairProfileCache::Implementation::Observer::finish(bool is_successful) noexcept {
    auto& value = *sample;
    auto& state = *value.owner;
    if (::getpid() != state.process) return;
    std::uint64_t minimum_ns{};
    {
        const std::lock_guard lock(state.mutex);
        if (value.is_terminal) return;
        const auto& current = value.legs[leg];
        if (!is_successful || !current.has_begin || current.has_finish || current.is_clock_active) {
            value.reject(false);
            return;
        }
        value.legs[leg].has_finish = true;
        value.legs[leg].is_clock_active = true;
        minimum_ns = value.last_clock_ns;
    }
    const auto now = state.clock();
    const std::lock_guard lock(state.mutex);
    if (value.is_terminal) return;
    auto& current = value.legs[leg];
    current.is_clock_active = false;
    current.end_ns = now;
    if (now < minimum_ns || now <= current.begin_ns ||
        now - current.begin_ns > kCudaPairProfileMaxSampleNs) {
        value.reject(false);
        return;
    }
    value.last_clock_ns = std::max(value.last_clock_ns, now);
    if (!value.legs[1 - leg].is_clock_active) value.commit();
}

CudaPairProfileCache::CudaPairProfileCache(std::shared_ptr<const CudaProfileCache> originals,
                                           CudaPairProfileConfig config, CudaProfileClock clock,
                                           bool is_learning)
    : implementation_(
          std::make_shared<Implementation>(std::move(originals), config, clock, is_learning)) {}
CudaPairProfileCache::~CudaPairProfileCache() = default;

CudaPairProfileKernel CudaPairProfileCache::bind(const CudaProfileFunction& function,
                                                 CudaProfileGeometry geometry) noexcept {
    return bind_checked(function, geometry).kernel;
}
CudaPairProfileBindResult CudaPairProfileCache::bind_checked(
    const CudaProfileFunction& function, CudaProfileGeometry geometry) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !function ||
        !cuda_profile_context_configuration_valid(geometry.context) ||
        !cuda_profile_function_configuration_valid(geometry.function) ||
        enum_bits(geometry.kind) > enum_bits(PtxExecutionKind::kSliced) || !volume(geometry.grid) ||
        !volume(geometry.block))
        return {};
    const auto measurement = state.originals->original_measurement(function, geometry);
    if (!measurement || !valid_measurement(*measurement))
        return {.status = CudaPairProfileBindStatus::kMissingOriginal};
    try {
        const std::lock_guard lock(state.mutex);
        auto kernel = state.find_kernel(*measurement);
        if (!kernel) {
            const auto bytes = identity_bytes(*measurement);
            if (!bytes || *bytes > SIZE_MAX - sizeof(CudaPairProfileKernelState))
                return {.status = CudaPairProfileBindStatus::kCapacity};
            kernel = std::make_shared<CudaPairProfileKernelState>();
            kernel->measurement = *measurement;
            kernel->bytes = *bytes + sizeof(CudaPairProfileKernelState);
            if (!state.room(kernel->bytes, false, true))
                return {.status = CudaPairProfileBindStatus::kCapacity};
            state.kernels.push_front(kernel);
            state.retained_bytes += kernel->bytes;
        }
        CudaPairProfileKernel result;
        result.owner_ = implementation_;
        result.state_ = std::move(kernel);
        return {CudaPairProfileBindStatus::kBound, std::move(result)};
    } catch (...) {
        return {.status = CudaPairProfileBindStatus::kCapacity};
    }
}

CudaPairProfileChoice CudaPairProfileCache::lookup(
    const std::array<CudaProfileFunction, 2>& functions,
    const std::array<CudaProfileGeometry, 2>& geometry, CudaPairProfileDomain domain,
    CudaPairProfileLimits limits,
    std::span<const CudaPairProfileCandidate> candidates) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process) return {};
    std::array<CudaProfileOriginalMeasurement, 2> originals;
    CudaPairProfileChoice result;
    for (std::size_t leg = 0; leg < originals.size(); ++leg) {
        const auto measured = state.originals->original_measurement(functions[leg], geometry[leg]);
        if (!measured || !valid_measurement(*measured)) return {};
        originals[leg] = *measured;
        result.original[leg] = {measured->total_ns, measured->samples};
    }
    if (originals[1].total_ns * originals[0].samples < originals[0].total_ns * originals[1].samples)
        result.first_original = 1;
    if (!valid_catalog(originals, domain, limits, candidates, state.config)) return result;
    std::array<CudaPairProfileKernel, 2> kernels;
    try {
        const std::lock_guard lock(state.mutex);
        for (std::size_t leg = 0; leg < kernels.size(); ++leg) {
            auto retained = state.find_kernel(originals[leg]);
            if (!retained) return result;
            kernels[leg].owner_ = implementation_;
            kernels[leg].state_ = std::move(retained);
        }
    } catch (...) {
        return result;
    }
    return choose(kernels, domain, limits, candidates);
}

CudaPairProfileChoice CudaPairProfileCache::choose(
    const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
    CudaPairProfileLimits limits,
    std::span<const CudaPairProfileCandidate> candidates) const noexcept {
    const auto& state = *implementation_;
    CudaPairProfileChoice result;
    if (::getpid() != state.process) return result;
    std::array<std::shared_ptr<CudaPairProfileKernelState>, 2> inputs;
    std::array<CudaProfileOriginalMeasurement, 2> originals;
    for (std::size_t leg = 0; leg < 2; ++leg) {
        if (!kernels[leg].state_ || kernels[leg].owner_.lock() != implementation_) return result;
        inputs[leg] = kernels[leg].state_;
        originals[leg] = inputs[leg]->measurement;
        result.original[leg] = {originals[leg].total_ns, originals[leg].samples};
    }
    if (originals[1].total_ns * originals[0].samples < originals[0].total_ns * originals[1].samples)
        result.first_original = 1;
    if (!valid_catalog(originals, domain, limits, candidates, state.config)) return result;
    try {
        const std::lock_guard lock(state.mutex);
        const auto [pair, reversed] = state.find_pair(inputs, domain, limits, candidates);
        if (!pair || !state.trained(*pair)) return result;
        result.is_trained = true;
        std::optional<std::size_t> best;
        Score best_score;
        bool has_tie{};
        for (std::size_t index = 0; index < pair->count; ++index) {
            const auto& entry = pair->candidates[index];
            std::array<Wide, 2> numerators, denominators;
            for (std::size_t leg = 0; leg < 2; ++leg) {
                numerators[leg] = wide(originals[leg].total_ns * entry.samples);
                denominators[leg] =
                    wide(originals[leg].samples * entry.total_ns[reversed ? 1 - leg : leg]);
            }
            const Score score{sum(multiply(numerators[0], denominators[1]),
                                  multiply(numerators[1], denominators[0])),
                              multiply(denominators[0], denominators[1])};
            if (!best || compare(score, best_score) > 0) {
                best = index;
                best_score = score;
                has_tie = false;
            } else if (compare(score, best_score) == 0) {
                has_tie = true;
            }
        }
        if (!best || has_tie || compare(best_score.numerator, best_score.denominator) <= 0)
            return result;
        result.is_shared = true;
        result.candidate = best;
        result.configuration = candidates[*best];
        for (std::size_t leg = 0; leg < 2; ++leg) {
            const auto& entry = pair->candidates[*best];
            result.paired[leg] = {entry.total_ns[reversed ? 1 - leg : leg], entry.samples};
            result.normalized_speed +=
                (static_cast<long double>(originals[leg].total_ns) / originals[leg].samples) /
                (static_cast<long double>(result.paired[leg].total_ns) / entry.samples);
        }
        return result;
    } catch (...) {
        return result;
    }
}

CudaPairProfileReservation CudaPairProfileCache::reserve(
    const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
    CudaPairProfileLimits limits, std::span<const CudaPairProfileCandidate> candidates) noexcept {
    return reserve_checked(kernels, domain, limits, candidates).reservation;
}
CudaPairProfileReserveResult CudaPairProfileCache::reserve_checked(
    const std::array<CudaPairProfileKernel, 2>& kernels, CudaPairProfileDomain domain,
    CudaPairProfileLimits limits, std::span<const CudaPairProfileCandidate> candidates) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !state.is_learning) return {};
    std::array<std::shared_ptr<CudaPairProfileKernelState>, 2> inputs;
    std::array<CudaProfileOriginalMeasurement, 2> originals;
    for (std::size_t leg = 0; leg < 2; ++leg) {
        if (!kernels[leg].state_ || kernels[leg].owner_.lock() != implementation_) return {};
        inputs[leg] = kernels[leg].state_;
        originals[leg] = inputs[leg]->measurement;
    }
    if (!valid_catalog(originals, domain, limits, candidates, state.config)) return {};
    try {
        const std::lock_guard lock(state.mutex);
        auto [pair, reversed] = state.find_pair(inputs, domain, limits, candidates);
        const bool is_new = !pair;
        if (pair && state.trained(*pair)) return {.status = CudaPairProfileReserveStatus::kTrained};
        if (pair && (pair->is_active || !pair->reservation.expired()))
            return {.status = CudaPairProfileReserveStatus::kBusy};
        if (!pair) {
            pair = std::make_shared<Implementation::Pair>();
            pair->kernels = inputs;
            pair->domain = domain;
            pair->limits = limits;
            pair->count = candidates.size();
            for (std::size_t index = 0; index < candidates.size(); ++index)
                pair->candidates[index].configuration = candidates[index];
        }
        std::size_t index{};
        while (index < pair->count &&
               pair->candidates[index].samples == state.config.samples_per_candidate)
            ++index;
        if (index == pair->count) return {.status = CudaPairProfileReserveStatus::kTrained};
        auto sample = std::make_shared<Implementation::Sample>(implementation_, pair, index);
        // Allocate both callbacks before publishing an active reservation. A
        // failed allocation must not run a cancelling destructor under this lock.
        auto first = std::make_shared<Implementation::Observer>(sample, reversed ? 1 : 0);
        auto second = std::make_shared<Implementation::Observer>(sample, reversed ? 0 : 1);
        if (is_new) {
            if (!state.room(Implementation::pair_bytes(), true, false))
                return {.status = CudaPairProfileReserveStatus::kCapacity};
            state.pairs.push_front(pair);
            state.retained_bytes += Implementation::pair_bytes();
        }
        pair->reservation = sample;
        pair->is_active = true;
        first->is_armed = true;
        second->is_armed = true;
        CudaPairProfileReservation result;
        result.candidate = index;
        result.configuration = candidates[index];
        result.observations = {std::move(first), std::move(second)};
        return {CudaPairProfileReserveStatus::kReserved, std::move(result)};
    } catch (...) {
        return {.status = CudaPairProfileReserveStatus::kCapacity};
    }
}

CudaPairProfileSnapshot CudaPairProfileCache::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process) return {};
    const std::lock_guard lock(state.mutex);
    CudaPairProfileSnapshot result;
    result.kernels = state.kernels.size();
    result.pairs = state.pairs.size();
    result.retained_bytes = state.retained_bytes;
    for (const auto& pair : state.pairs) {
        if (pair->is_active) ++result.active;
        if (state.trained(*pair)) ++result.trained_pairs;
    }
    result.samples = state.samples;
    result.rejected_samples = state.rejected_samples;
    result.cancelled = state.cancelled;
    result.evictions = state.evictions;
    result.loaded_pairs = state.loaded_pairs;
    result.is_valid = true;
    result.is_learning = state.is_learning;
    return result;
}

namespace {

void write_original(ArchiveWriter& output, const CudaProfileOriginalMeasurement& original,
                    unsigned int version) {
    const auto& identity = *original.identity;
    for (const auto byte : identity.platform.uuid)
        output.number(std::to_integer<unsigned int>(byte), 1);
    output.number(identity.platform.driver_version);
    output.number(identity.platform.toolkit_version);
    output.number(identity.platform.generator_abi);
    output.text(identity.source);
    output.text(identity.binary);
    output.text(identity.name);
    for (const auto value : original.geometry.grid) output.number(value);
    for (const auto value : original.geometry.block) output.number(value);
    output.number(original.geometry.shared_memory_bytes);
    output.number(enum_bits(original.geometry.kind));
    if (version >= 2) internal::write_profile_configuration(output, original.geometry.context);
    if (version >= 3) internal::write_function_configuration(output, original.geometry.function);
    output.number(original.total_ns, 8);
    output.number(original.samples);
}

std::optional<CudaProfileOriginalMeasurement> read_original(ArchiveReader& input,
                                                            unsigned int version,
                                                            const CudaProfileCache& originals,
                                                            std::size_t maximum_bytes) {
    CudaProfileCodeIdentity identity;
    for (auto& byte : identity.platform.uuid) byte = static_cast<std::byte>(input.number(1));
    identity.platform.driver_version = static_cast<unsigned int>(input.number());
    identity.platform.toolkit_version = static_cast<unsigned int>(input.number());
    identity.platform.generator_abi = static_cast<unsigned int>(input.number());
    const auto source = input.text(maximum_bytes);
    const auto binary = input.text(maximum_bytes);
    const auto name = input.text(4096);
    if (source.size() > maximum_bytes || binary.size() > maximum_bytes - source.size() ||
        name.size() > maximum_bytes - source.size() - binary.size() || name.empty() ||
        name.find('\0') != std::string_view::npos)
        return std::nullopt;
    identity.source = source;
    identity.binary = binary;
    identity.name = name;
    CudaProfileGeometry geometry;
    for (auto& value : geometry.grid) value = static_cast<unsigned int>(input.number());
    for (auto& value : geometry.block) value = static_cast<unsigned int>(input.number());
    geometry.shared_memory_bytes = static_cast<unsigned int>(input.number());
    geometry.kind = input.kind();
    if (version >= 2) geometry.context = internal::read_profile_configuration(input);
    if (version >= 3) geometry.function = internal::read_function_configuration(input);
    if (!cuda_profile_context_configuration_valid(geometry.context) ||
        !cuda_profile_function_configuration_valid(geometry.function))
        return std::nullopt;
    const auto total_ns = input.number(8);
    const auto samples = input.number();
    auto result = originals.original_measurement(identity, geometry);
    if (!result || !valid_measurement(*result) || result->total_ns != total_ns ||
        result->samples != samples)
        return std::nullopt;
    return result;
}

void write_limits(ArchiveWriter& output, CudaPairProfileLimits limits) {
    output.number(limits.sm_count);
    output.number(limits.blocks_per_sm);
    output.number(limits.threads_per_sm);
    output.number(limits.registers_per_sm);
    output.number(limits.max_workers);
    output.number(limits.shared_memory_bytes_per_sm, 8);
}

CudaPairProfileLimits read_limits(ArchiveReader& input) {
    CudaPairProfileLimits result;
    result.sm_count = static_cast<unsigned int>(input.number());
    result.blocks_per_sm = static_cast<unsigned int>(input.number());
    result.threads_per_sm = static_cast<unsigned int>(input.number());
    result.registers_per_sm = static_cast<unsigned int>(input.number());
    result.max_workers = static_cast<unsigned int>(input.number());
    result.shared_memory_bytes_per_sm = input.number(8);
    return result;
}

void write_candidate(ArchiveWriter& output, const CandidateState& candidate) {
    for (const auto& variant : candidate.configuration.legs) {
        output.number(enum_bits(variant.kind));
        output.number(variant.workers);
        output.number(variant.registers_per_thread);
        output.number(variant.occupancy_blocks_per_sm);
        output.number(variant.static_shared_memory_bytes, 8);
    }
    output.number(candidate.warmups);
    output.number(candidate.samples);
    for (const auto total : candidate.total_ns) output.number(total, 8);
}

CandidateState read_candidate(ArchiveReader& input) {
    CandidateState result;
    for (auto& variant : result.configuration.legs) {
        variant.kind = input.kind();
        variant.workers = static_cast<unsigned int>(input.number());
        variant.registers_per_thread = static_cast<unsigned int>(input.number());
        variant.occupancy_blocks_per_sm = static_cast<unsigned int>(input.number());
        variant.static_shared_memory_bytes = input.number(8);
    }
    result.warmups = static_cast<std::uint32_t>(input.number());
    result.samples = static_cast<std::uint32_t>(input.number());
    for (auto& total : result.total_ns) total = input.number(8);
    return result;
}

}  // namespace

std::optional<std::vector<std::byte>> CudaPairProfileCache::encode(
    std::string_view runtime_identity) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || !valid_runtime(runtime_identity)) return std::nullopt;
    try {
        const std::lock_guard lock(state.mutex);
        const bool has_configuration = std::ranges::any_of(state.pairs, [&](const auto& pair) {
            return state.trained(*pair) &&
                   std::ranges::any_of(pair->kernels, [](const auto& kernel) {
                       return kernel->measurement.geometry.context.available != 0;
                   });
        });
        const bool has_function_configuration =
            std::ranges::any_of(state.pairs, [&](const auto& pair) {
                return state.trained(*pair) &&
                       std::ranges::any_of(pair->kernels, [](const auto& kernel) {
                           return kernel->measurement.geometry.function.functions != 0;
                       });
            });
        const unsigned int version = has_function_configuration ? 3 : has_configuration ? 2 : 1;
        ArchiveWriter output{kCudaPairProfileArchiveMaxBytes};
        output.text("Glimmer pair profile archive");
        output.number(version);
        output.number(1);  // Overlapping host lifetime nanoseconds, not GPU timing.
        output.text(runtime_identity);
        output.number(state.config.max_pairs);
        output.number(state.config.max_candidates);
        output.number(state.config.max_bytes, 8);
        output.number(state.config.warmup_calls);
        output.number(state.config.samples_per_candidate);
        {
            const auto count = std::ranges::count_if(
                state.pairs, [&](const auto& pair) { return state.trained(*pair); });
            output.number(static_cast<std::uint64_t>(count));
            for (const auto& pair : state.pairs) {
                if (!state.trained(*pair)) continue;
                for (const auto& kernel : pair->kernels)
                    write_original(output, kernel->measurement, version);
                output.number(enum_bits(pair->domain.kind));
                for (const auto label : pair->domain.labels) output.number(label, 8);
                write_limits(output, pair->limits);
                output.number(pair->count);
                for (const auto& entry : std::span{pair->candidates}.first(pair->count))
                    write_candidate(output, entry);
            }
        }
        output.number(checksum(output.bytes), 8);
        return std::move(output.bytes);
    } catch (...) {
        return std::nullopt;
    }
}

bool CudaPairProfileCache::restore(std::span<const std::byte> archive,
                                   std::string_view runtime_identity) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !valid_runtime(runtime_identity) || !archive.data() ||
        archive.size() < 8 || archive.size() > kCudaPairProfileArchiveMaxBytes)
        return false;
    try {
        {
            const std::lock_guard lock(state.mutex);
            if (!state.pairs.empty() || !state.kernels.empty()) return false;
        }
        ArchiveReader digest{archive.last(8)}, input{archive.first(archive.size() - 8)};
        if (checksum(input.bytes) != digest.number(8) ||
            input.text(64) != "Glimmer pair profile archive")
            return false;
        const auto version = input.number();
        if ((version != 1 && version != 2 && version != 3) || input.number() != 1 ||
            input.text(4096) != runtime_identity || input.number() != state.config.max_pairs ||
            input.number() != state.config.max_candidates ||
            input.number(8) != state.config.max_bytes ||
            input.number() != state.config.warmup_calls ||
            input.number() != state.config.samples_per_candidate)
            return false;
        const auto count = input.number();
        if (count > state.config.max_pairs) return false;
        std::list<std::shared_ptr<CudaPairProfileKernelState>> kernels;
        std::list<std::shared_ptr<Implementation::Pair>> pairs;
        std::size_t retained{};
        for (std::uint64_t index = 0; index < count; ++index) {
            if (Implementation::pair_bytes() > state.config.max_bytes - retained) return false;
            retained += Implementation::pair_bytes();
            auto pair = std::make_shared<Implementation::Pair>();
            std::array<CudaProfileOriginalMeasurement, 2> measurements;
            for (std::size_t leg = 0; leg < 2; ++leg) {
                auto measurement = read_original(input, static_cast<unsigned int>(version),
                                                 *state.originals, state.config.max_bytes);
                if (!measurement) return false;
                measurements[leg] = *measurement;
                for (const auto& kernel : kernels)
                    if (same_measurement(kernel->measurement, *measurement)) {
                        pair->kernels[leg] = kernel;
                        break;
                    }
                if (pair->kernels[leg]) continue;
                const auto bytes = identity_bytes(*measurement);
                if (!bytes || *bytes > SIZE_MAX - sizeof(CudaPairProfileKernelState) ||
                    *bytes + sizeof(CudaPairProfileKernelState) >
                        state.config.max_bytes - retained ||
                    kernels.size() >= 2 * state.config.max_pairs)
                    return false;
                auto kernel = std::make_shared<CudaPairProfileKernelState>();
                kernel->measurement = std::move(*measurement);
                kernel->bytes = *bytes + sizeof(CudaPairProfileKernelState);
                retained += kernel->bytes;
                pair->kernels[leg] = kernel;
                kernels.push_back(std::move(kernel));
            }
            const auto domain_kind = input.number();
            if (domain_kind != 1 && domain_kind != 2) return false;
            pair->domain.kind = static_cast<CudaPairProfileDomainKind>(domain_kind);
            for (auto& label : pair->domain.labels) label = input.number(8);
            pair->limits = read_limits(input);
            pair->count = static_cast<std::size_t>(input.number());
            if (!pair->count || pair->count > state.config.max_candidates) return false;
            std::array<CudaPairProfileCandidate, kMaxCandidates> catalog{};
            for (std::size_t candidate = 0; candidate < pair->count; ++candidate) {
                auto& entry = pair->candidates[candidate];
                entry = read_candidate(input);
                if (entry.warmups != state.config.warmup_calls ||
                    entry.samples != state.config.samples_per_candidate ||
                    std::ranges::any_of(entry.total_ns, [&](auto total) {
                        return total < entry.samples ||
                               total > kCudaPairProfileMaxSampleNs * entry.samples;
                    }))
                    return false;
                catalog[candidate] = entry.configuration;
            }
            const auto candidates = std::span{catalog}.first(pair->count);
            if (!valid_catalog(measurements, pair->domain, pair->limits, candidates, state.config))
                return false;
            for (const auto& previous : pairs)
                if (state.matches(*previous, pair->kernels, pair->domain, pair->limits, candidates,
                                  false) ||
                    state.matches(*previous, pair->kernels, pair->domain, pair->limits, candidates,
                                  true))
                    return false;
            pairs.push_back(std::move(pair));
        }
        if (!input.bytes.empty()) return false;
        const std::lock_guard lock(state.mutex);
        if (!state.pairs.empty() || !state.kernels.empty()) return false;
        state.pairs.swap(pairs);
        state.kernels.swap(kernels);
        state.retained_bytes = retained;
        state.loaded_pairs = count;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::backend::cuda
