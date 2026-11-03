#include "glimmer/control/execution_jit_protocol.h"

#include <array>

namespace glimmer::control {
namespace {
constexpr std::uint32_t kMagic = 0x3154494a;  // JIT1.
std::optional<std::uint32_t> referenced_name_count(const ExecutionJitOption& option) noexcept {
    if (option.value > 1 || option.data.size() > kExecutionMaxJitNamesBytes) return std::nullopt;
    ExecutionPayloadReader input(option.data);
    std::uint32_t count{};
    if (!input.read_u32(count) || count > kExecutionMaxJitNames || (!option.value && count))
        return std::nullopt;
    for (std::uint32_t index = 0; index < count; ++index) {
        std::span<const std::byte> name;
        if (!input.read_blob(name) || name.empty() || name.size() > kExecutionMaxJitNameBytes ||
            name.back() != std::byte{})
            return std::nullopt;
        for (const auto byte : name.first(name.size() - 1))
            if (byte == std::byte{}) return std::nullopt;
    }
    return input.empty() ? std::optional{count} : std::nullopt;
}
bool valid_records(std::span<const ExecutionJitOption> options) noexcept {
    if (options.size() > kExecutionMaxJitOptions) return false;
    std::array<bool, kExecutionMaxJitOptions> seen{};
    std::size_t storage_bytes{};
    for (const auto& option : options) {
        if (option.option >= seen.size() || seen[option.option]) return false;
        seen[option.option] = true;
        switch (execution_jit_storage(option.option)) {
            case ExecutionJitStorage::kScalar:
                if (!option.data.empty()) return false;
                break;
            case ExecutionJitStorage::kFloat:
                if (option.value || option.data.size() != 4) return false;
                break;
            case ExecutionJitStorage::kLog:
                if (option.value || option.data.empty() ||
                    option.data.size() > kExecutionMaxJitLogBytes)
                    return false;
                break;
            case ExecutionJitStorage::kNames:
                if (!referenced_name_count(option)) return false;
                break;
            case ExecutionJitStorage::kUnsupported:
                return false;
        }
        storage_bytes += option.data.size();  // At most34 independently bounded records.
        if (storage_bytes > 2 * kExecutionMaxJitLogBytes + 4 + 2 * kExecutionMaxJitNamesBytes)
            return false;
    }
    return true;
}
}  // namespace

ExecutionJitStorage execution_jit_storage(std::uint32_t option) noexcept {
    if (option == 2) return ExecutionJitStorage::kFloat;
    if (option == 3 || option == 5) return ExecutionJitStorage::kLog;
    if (option == 25 || option == 27) return ExecutionJitStorage::kNames;
    if (option <= 14 || (option >= 20 && option <= 24) || option == 26 || option == 28 ||
        (option >= 29 && option <= 33))
        return ExecutionJitStorage::kScalar;
    return ExecutionJitStorage::kUnsupported;
}

std::optional<std::vector<std::byte>> encode_execution_jit_options(
    std::span<const ExecutionJitOption> options) noexcept {
    if (!valid_records(options)) return std::nullopt;
    ExecutionPayloadWriter output(kExecutionMaxJitWireBytes);
    if (!output.write_u32(kMagic) || !output.write_u32(static_cast<std::uint32_t>(options.size())))
        return std::nullopt;
    for (const auto& option : options)
        if (!output.write_u32(option.option) || !output.write_u64(option.value) ||
            !output.write_blob(option.data))
            return std::nullopt;
    return output.take();
}

std::optional<std::vector<ExecutionJitOption>> decode_execution_jit_options(
    std::span<const std::byte> bytes) noexcept {
    if (bytes.size() > kExecutionMaxJitWireBytes) return std::nullopt;
    ExecutionPayloadReader input(bytes);
    std::uint32_t magic{}, count{};
    if (!input.read_u32(magic) || magic != kMagic || !input.read_u32(count) ||
        count > kExecutionMaxJitOptions)
        return std::nullopt;
    try {
        std::vector<ExecutionJitOption> result;
        result.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            ExecutionJitOption option;
            std::span<const std::byte> data;
            if (!input.read_u32(option.option) || !input.read_u64(option.value) ||
                !input.read_blob(data) || data.size() > kExecutionMaxJitLogBytes)
                return std::nullopt;
            option.data.assign(data.begin(), data.end());
            result.push_back(std::move(option));
        }
        if (!input.empty() || !valid_records(result)) return std::nullopt;
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

bool valid_execution_jit_input(std::span<const ExecutionJitOption> options) noexcept {
    if (!valid_records(options)) return false;
    std::array<std::size_t, 2> logs{};
    std::array<std::uint64_t, 2> capacities{};
    std::array<bool, 2> has_capacity{};
    std::array<std::uint32_t, 2> names{};
    std::array<std::uint64_t, 2> name_counts{};
    for (const auto& option : options) {
        if (option.option == 3 || option.option == 5)
            logs[(option.option - 3) / 2] = option.data.size();
        if (option.option == 4 || option.option == 6) {
            const auto index = (option.option - 4) / 2;
            capacities[index] = option.value;
            has_capacity[index] = true;
        }
        if (option.option == 25 || option.option == 27) {
            const auto count = referenced_name_count(option);
            if (!count) return false;
            names[(option.option - 25) / 2] = *count;
        }
        if (option.option == 26 || option.option == 28)
            name_counts[(option.option - 26) / 2] = option.value;
    }
    for (std::size_t index = 0; index < logs.size(); ++index)
        if ((logs[index] != 0) != has_capacity[index] ||
            (has_capacity[index] && capacities[index] != logs[index]))
            return false;
    for (std::size_t index = 0; index < names.size(); ++index)
        if (names[index] != name_counts[index]) return false;
    return true;
}

bool execution_jit_output_matches(std::span<const ExecutionJitOption> input,
                                  std::span<const ExecutionJitOption> output) noexcept {
    if (!valid_execution_jit_input(input) || !valid_records(output) ||
        input.size() != output.size())
        return false;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto& requested = input[index];
        const auto& actual = output[index];
        if (requested.option != actual.option || requested.data.size() != actual.data.size())
            return false;
        const bool is_scalar_output =
            requested.option == 1 || requested.option == 4 || requested.option == 6;
        if ((!is_scalar_output && actual.value != requested.value) ||
            (is_scalar_output && actual.value != requested.value && actual.value > UINT32_MAX))
            return false;
        if (execution_jit_storage(requested.option) == ExecutionJitStorage::kNames &&
            actual.data != requested.data)
            return false;
    }
    return true;
}

}  // namespace glimmer::control
