#include "internal/cuda_ptx_cache.h"
#include "internal/cache_archive.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
void increment(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
}
std::size_t source_hash(std::string_view source) noexcept {
    return std::hash<std::string_view>{}(source);
}
bool add_bytes(std::size_t& total, std::size_t count, std::size_t width = 1) noexcept {
    if (width && count > (std::numeric_limits<std::size_t>::max() - total) / width) return false;
    total += count * width;
    return true;
}
bool valid_kind(transform::PtxExecutionKind kind) noexcept {
    unsigned int bits{};
    static_assert(sizeof(kind) == sizeof(bits));
    std::memcpy(&bits, &kind, sizeof(bits));
    return bits <= static_cast<unsigned int>(transform::PtxExecutionKind::kSliced);
}
bool valid_text(std::string_view value, std::size_t maximum) noexcept {
    return !value.empty() && value.size() <= maximum && value.find('\0') == std::string_view::npos;
}
bool valid_image(std::string_view source, transform::PtxExecutionKind kind, bool shares,
                 transform::PtxTransformLimits limits, const transform::PersistentPtx& image) {
    if (!transform::valid_ptx_transform_limits(limits) ||
        !valid_text(source, limits.source_bytes) || !valid_text(image.image, limits.output_bytes) ||
        !image.image.starts_with(source) || image.image.size() == source.size() ||
        image.kernels.empty() || (image.has_module_variables && !shares))
        return false;
    constexpr std::array<std::string_view, 5> suffixes{"", "_block_static", "_dynamic",
                                                       "_preemptive", "_sliced"};
    std::set<std::string_view> names;
    for (const auto& kernel : image.kernels) {
        const auto& name = kernel.original_name;
        if (name.empty() || name.find("__glimmer_ptb") != std::string::npos ||
            (!std::isalpha(static_cast<unsigned char>(name.front())) && name.front() != '_' &&
             name.front() != '$') ||
            !std::ranges::all_of(name,
                                 [](unsigned char value) {
                                     return std::isalnum(value) || value == '_' || value == '$';
                                 }) ||
            !names.insert(name).second ||
            kernel.persistent_name !=
                name + "__glimmer_ptb" + std::string(suffixes[static_cast<unsigned int>(kind)]) ||
            kernel.execution_kind != kind ||
            kernel.observes_block_completion != limits.observes_block_completion ||
            (kernel.observes_block_completion &&
             kind != transform::PtxExecutionKind::kPreemptive) ||
            kernel.parameter_count != kernel.parameter_layout.size() ||
            kernel.parameter_count > 256)
            return false;
        std::size_t total{};
        for (const auto& parameter : kernel.parameter_layout) {
            if (!parameter.size_bytes || parameter.size_bytes > 65536 - total ||
                !parameter.alignment_bytes || parameter.alignment_bytes > 128 ||
                (parameter.alignment_bytes & (parameter.alignment_bytes - 1)))
                return false;
            total += parameter.size_bytes;
        }
    }
    return true;
}
}  // namespace
struct CudaPtxCache::Implementation {
    struct Entry {
        std::string source;
        transform::PtxExecutionKind kind{};
        bool shares_original_module{};
        transform::PtxTransformLimits preparation_limits{};
        std::uint32_t generator_abi{kCudaPtxGeneratorAbi};
        std::size_t hash{}, bytes{};
        std::shared_ptr<const transform::PersistentPtx> image;
    };
    const CudaPtxCacheLimits limits;
    const CudaPtxCacheHash hash;
    const pid_t process{::getpid()};
    mutable std::mutex mutex;
    std::list<Entry> entries;
    CudaPtxCacheSnapshot counters;
    bool has_restored{};
    Implementation(CudaPtxCacheLimits value, CudaPtxCacheHash hasher)
        : limits(value), hash(hasher ? hasher : source_hash) {
        if (limits.max_entries > 4096)
            throw std::invalid_argument("excessive PTX cache entry bound");
    }
    bool evict_one() noexcept {
        const auto found = std::find_if(entries.rbegin(), entries.rend(), [](const auto& entry) {
            return entry.image.use_count() <= 1;
        });
        if (found == entries.rend()) return false;
        counters.retained_bytes -= found->bytes;
        entries.erase(std::prev(found.base()));
        increment(counters.evictions);
        return true;
    }
    static bool charge(Entry& entry) noexcept {
        auto bytes = sizeof(Entry);
        if (!add_bytes(bytes, entry.source.capacity()) || !add_bytes(bytes, 1)) return false;
        if (entry.image) {
            const auto& image = *entry.image;
            if (!add_bytes(bytes, sizeof(image)) || !add_bytes(bytes, image.image.capacity()) ||
                !add_bytes(bytes, 1) ||
                !add_bytes(bytes, image.kernels.capacity(), sizeof(transform::PtxKernelVariant)))
                return false;
            for (const auto& kernel : image.kernels)
                if (!add_bytes(bytes, kernel.original_name.capacity()) || !add_bytes(bytes, 1) ||
                    !add_bytes(bytes, kernel.persistent_name.capacity()) || !add_bytes(bytes, 1) ||
                    !add_bytes(bytes, kernel.parameter_layout.capacity(),
                               sizeof(transform::PtxParameterLayout)))
                    return false;
        }
        entry.bytes = bytes;
        return true;
    }
};
CudaPtxCache::CudaPtxCache(CudaPtxCacheLimits limits, CudaPtxCacheHash hash)
    : implementation_(std::make_unique<Implementation>(limits, hash)) {}
CudaPtxCache::~CudaPtxCache() = default;
CudaPtxCacheResult CudaPtxCache::lookup(std::string_view source, transform::PtxExecutionKind kind,
                                        bool shares_original_module) noexcept {
    return lookup(source, kind, shares_original_module, {});
}
CudaPtxCacheResult CudaPtxCache::lookup(std::string_view source, transform::PtxExecutionKind kind,
                                        bool shares_original_module,
                                        transform::PtxTransformLimits limits) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process) return {CudaPtxCacheStatus::kWrongProcess, {}, false};
    if (!valid_kind(kind) || source.empty() || !transform::valid_ptx_transform_limits(limits))
        return {CudaPtxCacheStatus::kInvalid, {}, false};
    try {
        if (!state.limits.max_entries || !state.limits.max_bytes ||
            source.size() > limits.source_bytes || source.size() > state.limits.max_bytes) {
            std::lock_guard lock(state.mutex);
            increment(state.counters.lookups);
            increment(state.counters.bypasses);
            return {};
        }
        if (source.find('\0') != std::string_view::npos)
            return {CudaPtxCacheStatus::kInvalid, {}, false};
        const auto hash = state.hash(source);  // No external callback under exclusion.
        std::lock_guard lock(state.mutex);
        increment(state.counters.lookups);
        const auto found = std::ranges::find_if(state.entries, [&](const auto& entry) {
            return entry.hash == hash && entry.kind == kind &&
                   entry.shares_original_module == shares_original_module &&
                   entry.preparation_limits == limits &&
                   entry.generator_abi == kCudaPtxGeneratorAbi && entry.source == source;
        });
        if (found != state.entries.end()) {
            increment(state.counters.hits);
            auto image = found->image;
            state.entries.splice(state.entries.begin(), state.entries, found);
            return {image ? CudaPtxCacheStatus::kReady : CudaPtxCacheStatus::kUnsupported,
                    std::move(image), true};
        }
        Implementation::Entry entry;
        entry.source = source;
        entry.kind = kind;
        entry.shares_original_module = shares_original_module;
        entry.preparation_limits = limits;
        entry.hash = hash;
        increment(state.counters.transforms);
        if (auto transformed = transform::transform_persistent_ptx(entry.source, kind,
                                                                   shares_original_module, limits))
            entry.image = std::make_shared<const transform::PersistentPtx>(std::move(*transformed));
        CudaPtxCacheResult result{
            entry.image ? CudaPtxCacheStatus::kReady : CudaPtxCacheStatus::kUnsupported,
            entry.image, false};
        if (!Implementation::charge(entry) || entry.bytes > state.limits.max_bytes) {
            increment(state.counters.bypasses);
            return result;
        }
        while (state.entries.size() >= state.limits.max_entries ||
               entry.bytes > state.limits.max_bytes - state.counters.retained_bytes) {
            if (!state.evict_one()) {
                increment(state.counters.bypasses);
                return result;
            }
        }
        state.entries.push_front(std::move(entry));
        state.counters.retained_bytes += state.entries.front().bytes;
        return result;
    } catch (...) {
        // Cache/preparation allocation failure never becomes a negative entry.
        // The caller may still use its original uncached preparation path.
        return {};
    }
}
CudaPtxCacheSnapshot CudaPtxCache::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process) return {};
    try {
        std::lock_guard lock(state.mutex);
        auto result = state.counters;
        result.entries = state.entries.size();
        result.is_valid = true;
        return result;
    } catch (...) {
        return {};
    }
}
bool CudaPtxCache::prune() noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process) return false;
    try {
        std::lock_guard lock(state.mutex);
        while (state.evict_one()) {
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<std::vector<std::byte>> CudaPtxCache::encode(
    std::string_view runtime_identity) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || !valid_text(runtime_identity, 4096)) return std::nullopt;
    try {
        internal::ArchiveWriter output{kCudaPtxArchiveMaxBytes};
        output.text("Glimmer PTX preparation archive");
        output.number(2);
        output.number(kCudaPtxGeneratorAbi);
        output.text(runtime_identity);
        {
            const std::lock_guard lock(state.mutex);
            output.number(state.entries.size());
            for (const auto& entry : state.entries) {
                if (!transform::valid_ptx_transform_limits(entry.preparation_limits) ||
                    !valid_text(entry.source, entry.preparation_limits.source_bytes))
                    return std::nullopt;
                output.number(static_cast<unsigned int>(entry.kind));
                output.number(entry.shares_original_module);
                output.number(entry.preparation_limits.source_bytes);
                output.number(entry.preparation_limits.output_bytes);
                output.number(entry.preparation_limits.token_count);
                output.number(entry.preparation_limits.observes_block_completion);
                output.text(entry.source);
                output.number(entry.image != nullptr);
                if (!entry.image) continue;
                const auto& image = *entry.image;
                if (!valid_image(entry.source, entry.kind, entry.shares_original_module,
                                 entry.preparation_limits, image))
                    return std::nullopt;
                output.number(image.has_module_variables);
                output.text(image.image);
                output.number(image.kernels.size());
                for (const auto& kernel : image.kernels) {
                    output.text(kernel.original_name);
                    output.text(kernel.persistent_name);
                    output.number(kernel.parameter_count);
                    output.number(kernel.observes_block_completion);
                    for (const auto& parameter : kernel.parameter_layout) {
                        output.number(parameter.size_bytes);
                        output.number(parameter.alignment_bytes);
                    }
                }
            }
        }
        output.number(internal::checksum(output.bytes), 8);
        return std::move(output.bytes);
    } catch (...) {
        return std::nullopt;
    }
}

bool CudaPtxCache::restore(std::span<const std::byte> archive,
                           std::string_view runtime_identity) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !archive.data() || archive.size() < 8 ||
        archive.size() > kCudaPtxArchiveMaxBytes || !valid_text(runtime_identity, 4096))
        return false;
    try {
        internal::ArchiveReader digest{archive.last(8)}, input{archive.first(archive.size() - 8)};
        if (internal::checksum(input.bytes) != digest.number(8) ||
            input.text(64) != "Glimmer PTX preparation archive" || input.number() != 2 ||
            input.number() != kCudaPtxGeneratorAbi || input.text(4096) != runtime_identity)
            return false;
        const auto count = input.number();
        if (count > state.limits.max_entries) return false;
        std::list<Implementation::Entry> entries;
        std::size_t retained{};
        for (std::size_t index = 0; index < count; ++index) {
            Implementation::Entry entry;
            entry.kind = input.kind();
            entry.shares_original_module = input.boolean();
            entry.preparation_limits.source_bytes = static_cast<std::size_t>(input.number());
            entry.preparation_limits.output_bytes = static_cast<std::size_t>(input.number());
            entry.preparation_limits.token_count = static_cast<std::size_t>(input.number());
            entry.preparation_limits.observes_block_completion = input.boolean();
            if (!transform::valid_ptx_transform_limits(entry.preparation_limits)) return false;
            const auto source = input.text(entry.preparation_limits.source_bytes);
            if (!valid_text(source, entry.preparation_limits.source_bytes)) return false;
            std::size_t minimum = sizeof(entry);
            const auto budget = state.limits.max_bytes - retained;
            if (!add_bytes(minimum, source.size() + 1) || minimum > budget) return false;
            entry.source = source;
            // Hash callbacks, parsing and allocation never run under cache exclusion.
            entry.hash = state.hash(source);
            if (std::ranges::any_of(entries, [&](const auto& prior) {
                    return prior.kind == entry.kind &&
                           prior.shares_original_module == entry.shares_original_module &&
                           prior.preparation_limits == entry.preparation_limits &&
                           prior.source == source;
                }))
                return false;
            if (input.boolean()) {
                auto image = std::make_shared<transform::PersistentPtx>();
                image->has_module_variables = input.boolean();
                const auto code = input.text(entry.preparation_limits.output_bytes);
                const auto kernels = input.number();
                if (!kernels || kernels > code.size() || !add_bytes(minimum, sizeof(*image)) ||
                    !add_bytes(minimum, code.size() + 1) ||
                    !add_bytes(minimum, static_cast<std::size_t>(kernels),
                               sizeof(transform::PtxKernelVariant)) ||
                    minimum > budget)
                    return false;
                image->image = code;
                image->kernels.reserve(static_cast<std::size_t>(kernels));
                for (std::size_t kernel_index = 0; kernel_index < kernels; ++kernel_index) {
                    const auto original = input.text(transform::kMaxPersistentPtxBytes);
                    const auto generated = input.text(transform::kMaxPersistentPtxBytes);
                    const auto parameters = input.number();
                    const auto observes_block_completion = input.boolean();
                    if (parameters > 256 || !add_bytes(minimum, original.size() + 1) ||
                        !add_bytes(minimum, generated.size() + 1) ||
                        !add_bytes(minimum, static_cast<std::size_t>(parameters),
                                   sizeof(transform::PtxParameterLayout)) ||
                        minimum > budget)
                        return false;
                    transform::PtxKernelVariant kernel{std::string(original),
                                                       std::string(generated),
                                                       static_cast<std::size_t>(parameters),
                                                       {},
                                                       entry.kind};
                    kernel.observes_block_completion = observes_block_completion;
                    kernel.parameter_layout.reserve(static_cast<std::size_t>(parameters));
                    for (std::size_t parameter = 0; parameter < parameters; ++parameter)
                        kernel.parameter_layout.push_back(
                            {static_cast<std::size_t>(input.number()),
                             static_cast<std::size_t>(input.number())});
                    image->kernels.push_back(std::move(kernel));
                }
                if (!valid_image(source, entry.kind, entry.shares_original_module,
                                 entry.preparation_limits, *image))
                    return false;
                entry.image = std::move(image);
            }
            if (!Implementation::charge(entry) || entry.bytes > budget) return false;
            retained += entry.bytes;
            entries.push_back(std::move(entry));
        }
        if (!input.bytes.empty()) return false;
        const std::lock_guard lock(state.mutex);
        if (!state.entries.empty() || state.counters.lookups || state.has_restored) return false;
        state.entries = std::move(entries);
        state.counters.retained_bytes = retained;
        state.counters.loaded_entries = count;
        state.has_restored = true;
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace glimmer::backend::cuda
