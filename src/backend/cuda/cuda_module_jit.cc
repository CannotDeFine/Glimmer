#include "internal/cuda_module_jit.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace glimmer::backend::cuda {
namespace {
constexpr std::size_t kMaxOptions = 34;
constexpr std::size_t kMaxLogBytes = 65536;
constexpr std::size_t kMaxNames = 32, kMaxNameBytes = 1025;
enum class Storage { kUnsupported, kScalar, kFloat, kLog, kNames };
struct NamesLayout {
    std::uint32_t count{};
    std::array<std::size_t, kMaxNames + 1> offsets{};
};
std::optional<NamesLayout> names_layout(const CudaModuleJitOption& option,
                                        std::size_t maximum_names = kMaxNames) noexcept {
    if (maximum_names > kMaxNames + 1 || option.value > 1 ||
        option.data.size() > 4 + maximum_names * (4 + kMaxNameBytes))
        return std::nullopt;
    std::size_t offset{};
    const auto word = [&](std::uint32_t& value) {
        if (option.data.size() - offset < 4) return false;
        std::memcpy(&value, option.data.data() + offset, 4);
        offset += 4;
        return true;
    };
    NamesLayout result;
    if (!word(result.count) || result.count > maximum_names || (!option.value && result.count))
        return std::nullopt;
    for (std::size_t index = 0; index < result.count; ++index) {
        std::uint32_t bytes{};
        if (!word(bytes) || !bytes || bytes > kMaxNameBytes || bytes > option.data.size() - offset)
            return std::nullopt;
        const auto name = std::span(option.data).subspan(offset, bytes);
        if (name.back() != std::byte{} ||
            std::ranges::find(name.first(bytes - 1), std::byte{}) != name.end() - 1)
            return std::nullopt;
        result.offsets[index] = offset;
        offset += bytes;
    }
    return offset == option.data.size() ? std::optional{result} : std::nullopt;
}
Storage storage(std::uint32_t option) noexcept {
    switch (static_cast<CUjit_option>(option)) {
        case CU_JIT_WALL_TIME:
            return Storage::kFloat;
        case CU_JIT_INFO_LOG_BUFFER:
        case CU_JIT_ERROR_LOG_BUFFER:
            return Storage::kLog;
        case CU_JIT_REFERENCED_KERNEL_NAMES:
        case CU_JIT_REFERENCED_VARIABLE_NAMES:
            return Storage::kNames;
        case CU_JIT_MAX_REGISTERS:
        case CU_JIT_THREADS_PER_BLOCK:
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_OPTIMIZATION_LEVEL:
        case CU_JIT_TARGET_FROM_CUCONTEXT:
        case CU_JIT_TARGET:
        case CU_JIT_FALLBACK_STRATEGY:
        case CU_JIT_GENERATE_DEBUG_INFO:
        case CU_JIT_LOG_VERBOSE:
        case CU_JIT_GENERATE_LINE_INFO:
        case CU_JIT_CACHE_MODE:
        case CU_JIT_LTO:
        case CU_JIT_FTZ:
        case CU_JIT_PREC_DIV:
        case CU_JIT_PREC_SQRT:
        case CU_JIT_FMA:
        case CU_JIT_REFERENCED_KERNEL_COUNT:
        case CU_JIT_REFERENCED_VARIABLE_COUNT:
        case CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES:
        case CU_JIT_POSITION_INDEPENDENT_CODE:
        case CU_JIT_MIN_CTA_PER_SM:
        case CU_JIT_MAX_THREADS_PER_BLOCK:
        case CU_JIT_OVERRIDE_DIRECTIVE_VALUES:
            return Storage::kScalar;
        default:
            return Storage::kUnsupported;
    }
}
bool valid(std::span<const CudaModuleJitOption> options) noexcept {
    if (options.size() > kMaxOptions) return false;
    std::array<bool, kMaxOptions> seen{};
    std::array<std::size_t, 2> log_bytes{};
    std::array<std::optional<std::uint64_t>, 2> capacities{};
    std::array<std::uint32_t, 2> names{};
    std::array<std::uint64_t, 2> name_counts{};
    for (const auto& option : options) {
        if (option.option >= seen.size() || seen[option.option]) return false;
        seen[option.option] = true;
        switch (storage(option.option)) {
            case Storage::kUnsupported:
                return false;
            case Storage::kScalar:
                if (!option.data.empty()) return false;
                if (option.option == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
                    capacities[0] = option.value;
                if (option.option == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
                    capacities[1] = option.value;
                if (option.option == CU_JIT_REFERENCED_KERNEL_COUNT) name_counts[0] = option.value;
                if (option.option == CU_JIT_REFERENCED_VARIABLE_COUNT)
                    name_counts[1] = option.value;
                break;
            case Storage::kFloat:
                if (option.value || option.data.size() != sizeof(float)) return false;
                break;
            case Storage::kLog:
                if (option.value || option.data.empty() || option.data.size() > kMaxLogBytes)
                    return false;
                log_bytes[option.option == CU_JIT_INFO_LOG_BUFFER ? 0 : 1] = option.data.size();
                break;
            case Storage::kNames: {
                const auto shape = names_layout(option);
                if (!shape) return false;
                names[option.option == CU_JIT_REFERENCED_KERNEL_NAMES ? 0 : 1] = shape->count;
                break;
            }
        }
    }
    for (std::size_t index = 0; index < log_bytes.size(); ++index)
        if ((log_bytes[index] != 0) != capacities[index].has_value() ||
            (capacities[index] && *capacities[index] != log_bytes[index]))
            return false;
    if (!std::ranges::equal(names, name_counts)) return false;
    return true;
}
static_assert(sizeof(float) == 4);
static_assert(sizeof(CUjit_option) == sizeof(std::uint32_t));
static_assert(sizeof(std::uintptr_t) == sizeof(std::uint64_t));
static_assert(std::endian::native == std::endian::little);
}  // namespace

std::unique_ptr<CudaModuleJitArguments> CudaModuleJitArguments::create(
    std::span<const CudaModuleJitOption> options) noexcept {
    if (!valid(options)) return {};
    try {
        auto result = std::unique_ptr<CudaModuleJitArguments>(new CudaModuleJitArguments);
        result->inputs_.assign(options.begin(), options.end());
        for (const auto key : {CU_JIT_REFERENCED_KERNEL_NAMES, CU_JIT_REFERENCED_VARIABLE_NAMES}) {
            const auto names_key = static_cast<std::uint32_t>(key), count_key = names_key + 1;
            const auto requested =
                std::ranges::find(result->inputs_, names_key, &CudaModuleJitOption::option);
            if (requested == result->inputs_.end()) continue;
            const auto layout = names_layout(*requested).value();
            // Zero means no supplied filter. Do not replace it with a restrictive list.
            if (!layout.count) continue;
            bool contains_pattern{};
            for (std::size_t index = 0; index < layout.count; ++index)
                contains_pattern = contains_pattern ||
                                   std::strcmp(reinterpret_cast<const char*>(
                                                   requested->data.data() + layout.offsets[index]),
                                               kCudaModuleJitPrivateReferencePattern) == 0;
            if (!contains_pattern) {
                if (result->private_inputs_.empty()) result->private_inputs_ = result->inputs_;
                auto& names = *std::ranges::find(result->private_inputs_, names_key,
                                                 &CudaModuleJitOption::option);
                auto& count = *std::ranges::find(result->private_inputs_, count_key,
                                                 &CudaModuleJitOption::option);
                const auto length =
                    static_cast<std::uint32_t>(sizeof(kCudaModuleJitPrivateReferencePattern));
                const auto word = std::bit_cast<std::array<std::byte, 4>>(length);
                names.data.insert(names.data.end(), word.begin(), word.end());
                const auto pattern =
                    std::as_bytes(std::span{kCudaModuleJitPrivateReferencePattern});
                names.data.insert(names.data.end(), pattern.begin(), pattern.end());
                const auto updated = layout.count + 1;
                std::memcpy(names.data.data(), &updated, 4);
                names.value = 1;
                count.value = updated;
            }
        }
        result->outputs_ =
            result->private_inputs_.empty() ? result->inputs_ : result->private_inputs_;
        for (std::size_t index = 0; index < result->inputs_.size(); ++index)
            result->outputs_[index].data.reserve(
                std::max(result->outputs_[index].data.size(), result->inputs_[index].data.size()));
        result->reset();
        return result;
    } catch (...) {
        return {};
    }
}
bool CudaModuleJitArguments::accepts(std::span<const CudaModuleJitOption> options) noexcept {
    return valid(options);
}
void CudaModuleJitArguments::reset() noexcept {
    is_private_ = false;
    reset_active();
}
void CudaModuleJitArguments::reset_for_private() noexcept {
    is_private_ = !private_inputs_.empty();
    reset_active();
}
bool CudaModuleJitArguments::has_private_references() const noexcept {
    return !private_inputs_.empty();
}
std::span<const CudaModuleJitOption> CudaModuleJitArguments::active_inputs() const noexcept {
    return is_private_ ? private_inputs_ : inputs_;
}
void CudaModuleJitArguments::reset_active() noexcept {
    for (auto& names : names_) names.fill(nullptr);
    const auto inputs = active_inputs();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        auto& output = outputs_[index];
        output.option = input.option;
        output.value = input.value;
        output.data.resize(input.data.size());  // Maximum cold capacity is already reserved.
        std::ranges::copy(input.data, output.data.begin());
        options_[index] = static_cast<CUjit_option>(input.option);
        switch (storage(input.option)) {
            case Storage::kScalar:
                // Native CUDA ABI: scalar bits live in the void* slot, not a
                // client pointer to an integer. Never dereference this value.
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                values_[index] = reinterpret_cast<void*>(static_cast<std::uintptr_t>(input.value));
                break;
            case Storage::kFloat: {
                // DataEx writes float bytes into the optionValues slot, not
                // through its value. Keep unused high bytes canonical/local.
                std::uint32_t bits{};
                std::memcpy(&bits, input.data.data(), sizeof(bits));
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                values_[index] = reinterpret_cast<void*>(static_cast<std::uintptr_t>(bits));
                break;
            }
            case Storage::kLog:
                values_[index] = output.data.data();
                break;
            case Storage::kNames: {
                auto& names = names_[input.option == CU_JIT_REFERENCED_KERNEL_NAMES ? 0 : 1];
                const auto shape = names_layout(input, kMaxNames + (is_private_ ? 1 : 0)).value();
                for (std::size_t name = 0; name < shape.count; ++name)
                    names[name] =
                        reinterpret_cast<const char*>(output.data.data() + shape.offsets[name]);
                values_[index] = input.value ? names.data() : nullptr;
                break;
            }
            case Storage::kUnsupported:
                break;  // create() rejects this shape.
        }
    }
}
std::span<CUjit_option> CudaModuleJitArguments::native_options() noexcept {
    return std::span(options_).first(active_inputs().size());
}
std::span<void*> CudaModuleJitArguments::native_values() noexcept {
    return std::span(values_).first(active_inputs().size());
}
std::span<const CudaModuleJitOption> CudaModuleJitArguments::inputs() const noexcept {
    return inputs_;
}
std::optional<std::span<const CudaModuleJitOption>> CudaModuleJitArguments::capture() noexcept {
    const auto inputs = active_inputs();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        auto& output = outputs_[index];
        std::uint32_t native_key{};
        std::memcpy(&native_key, &options_[index], sizeof(native_key));
        if (native_key != input.option) return std::nullopt;
        switch (storage(input.option)) {
            case Storage::kScalar: {
                const auto value =
                    static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(values_[index]));
                const bool is_output = input.option == CU_JIT_THREADS_PER_BLOCK ||
                                       input.option == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES ||
                                       input.option == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
                if ((!is_output && value != input.value) ||
                    (is_output && value != input.value && value > UINT32_MAX))
                    return std::nullopt;
                output.value = value;
                break;
            }
            case Storage::kFloat:
                if (reinterpret_cast<std::uintptr_t>(values_[index]) > UINT32_MAX)
                    return std::nullopt;
                std::memcpy(output.data.data(), &values_[index], sizeof(float));
                break;
            case Storage::kLog:
                if (values_[index] != output.data.data()) return std::nullopt;
                break;
            case Storage::kNames: {
                const auto& names = names_[input.option == CU_JIT_REFERENCED_KERNEL_NAMES ? 0 : 1];
                const auto* expected = input.value ? names.data() : nullptr;
                if (values_[index] != expected || output.data != input.data) return std::nullopt;
                const auto shape = names_layout(input, kMaxNames + (is_private_ ? 1 : 0)).value();
                for (std::size_t name = 0; name < names.size(); ++name) {
                    const auto* pointer = name < shape.count
                                              ? reinterpret_cast<const char*>(output.data.data() +
                                                                              shape.offsets[name])
                                              : nullptr;
                    if (names[name] != pointer) return std::nullopt;
                }
                break;
            }
            case Storage::kUnsupported:
                return std::nullopt;
        }
    }
    return std::span<const CudaModuleJitOption>{outputs_}.first(inputs.size());
}
std::size_t CudaModuleJitArguments::storage_bytes() const noexcept {
    auto bytes = sizeof(*this);
    for (const auto* options : {&inputs_, &private_inputs_, &outputs_}) {
        bytes += options->capacity() * sizeof(CudaModuleJitOption);
        for (const auto& option : *options) bytes += option.data.capacity();
    }
    return bytes;
}

}  // namespace glimmer::backend::cuda
