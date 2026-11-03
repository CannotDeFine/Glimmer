#include "glimmer/control/execution_dnn_legacy_protocol.h"

#include <bit>
#include <utility>

namespace glimmer::control {
namespace {
bool valid_array(const ExecutionDnnLegacyDescriptorArray& array, std::int32_t count) noexcept {
    if (array.has_descriptors > 1 || array.descriptors.size() > kExecutionDnnMaxSequenceLengths)
        return false;
    const auto captured = array.has_descriptors && count > 0 ? static_cast<std::size_t>(count) : 0;
    return array.descriptors.size() == captured;
}
bool write_array(ExecutionPayloadWriter& output,
                 const ExecutionDnnLegacyDescriptorArray& array) noexcept {
    ExecutionPayloadWriter elements;
    for (auto id : array.descriptors)
        if (!elements.write_u64(id)) return false;
    return output.write_u32(array.has_descriptors) && output.write_blob(elements.take());
}
bool read_array(ExecutionPayloadReader& input, ExecutionDnnLegacyDescriptorArray& output,
                std::int32_t count) {
    std::span<const std::byte> bytes;
    if (!input.read_u32(output.has_descriptors) || output.has_descriptors > 1 ||
        !input.read_blob(bytes) || bytes.size() % 8 ||
        bytes.size() / 8 > kExecutionDnnMaxSequenceLengths)
        return false;
    const auto captured = output.has_descriptors && count > 0 ? static_cast<std::size_t>(count) : 0;
    if (bytes.size() / 8 != captured) return false;
    output.descriptors.resize(captured);
    ExecutionPayloadReader elements(bytes);
    for (auto& id : output.descriptors)
        if (!elements.read_u64(id)) return false;
    return true;
}
bool write_signed(ExecutionPayloadWriter& output, std::int32_t value) noexcept {
    return output.write_u32(std::bit_cast<std::uint32_t>(value));
}
bool read_signed(ExecutionPayloadReader& input, std::int32_t& output) noexcept {
    std::uint32_t bits{};
    if (!input.read_u32(bits)) return false;
    output = std::bit_cast<std::int32_t>(bits);
    return true;
}
bool valid_metadata(const ExecutionDnnLegacyMetadata& value) noexcept {
    if (value.has_output > 1 || (!value.has_output && value.seed)) return false;
    switch (value.kind) {
        case ExecutionDnnLegacyMetadataKind::kVersionCheck:
            return !value.object && !value.has_output && value.value >= 1 && value.value <= 6;
        case ExecutionDnnLegacyMetadataKind::kSetAlgorithm:
        case ExecutionDnnLegacyMetadataKind::kSetMath:
        case ExecutionDnnLegacyMetadataKind::kSetBias:
            return !value.has_output;
        case ExecutionDnnLegacyMetadataKind::kGetMath:
        case ExecutionDnnLegacyMetadataKind::kGetBias:
            return !value.value;
    }
    return false;
}
bool valid_control(const ExecutionDnnLegacyControl& value) noexcept {
    if (value.has_output > 1 || (!value.has_output && value.seed)) return false;
    switch (value.kind) {
        case ExecutionDnnLegacyControlKind::kSetup:
        case ExecutionDnnLegacyControlKind::kClip:
        case ExecutionDnnLegacyControlKind::kSetAlgorithm:
            return !value.has_output;
        case ExecutionDnnLegacyControlKind::kWorkspaceSize:
        case ExecutionDnnLegacyControlKind::kReserveSize:
            return valid_array(value.sequence, value.sequence_length);
        case ExecutionDnnLegacyControlKind::kParameterSize:
            return true;
        case ExecutionDnnLegacyControlKind::kMatrixParameter:
        case ExecutionDnnLegacyControlKind::kBiasParameter:
            return !value.seed &&
                   (value.parameters.weights.allocation || !value.parameters.weights.offset);
        case ExecutionDnnLegacyControlKind::kInferenceMaxCount:
            return value.seed <= UINT32_MAX;
    }
    return false;
}
bool valid_output(const ExecutionDnnLegacyOutput& value) noexcept {
    return value.written <= 3 && ((value.written & 1U) || !value.scalar) && value.non_null <= 1 &&
           ((value.written & 2U) || !value.non_null) &&
           (value.non_null || (!value.offset && !value.extent)) &&
           value.extent <= UINT64_MAX - value.offset;
}
bool valid_compute(const ExecutionDnnLegacyCompute& value) noexcept {
    std::uint32_t unused{};
    switch (value.kind) {
        case ExecutionDnnLegacyComputeKind::kForwardTraining:
            unused = (1U << 2) | (1U << 3) | (1U << 8) | (1U << 9);
            if (value.states[4] || value.states[5]) return false;
            break;
        case ExecutionDnnLegacyComputeKind::kBackwardData:
            unused = 1U;
            break;
        case ExecutionDnnLegacyComputeKind::kBackwardWeights:
            unused =
                (1U << 2) | (1U << 3) | (1U << 5) | (1U << 6) | (1U << 7) | (1U << 8) | (1U << 9);
            for (std::size_t index = 1; index != value.states.size(); ++index)
                if (value.states[index]) return false;
            break;
        default:
            return false;
    }
    for (std::size_t index = 0; index != value.sequences.size(); ++index) {
        const auto& array = value.sequences[index];
        if (index == 2 && value.kind != ExecutionDnnLegacyComputeKind::kBackwardData) {
            if (array.has_descriptors || !array.descriptors.empty()) return false;
        } else if (!valid_array(array, value.sequence_length)) {
            return false;
        }
    }
    for (std::size_t index = 0; index != value.buffers.size(); ++index) {
        const auto& buffer = value.buffers[index];
        if ((!buffer.allocation && buffer.offset) ||
            ((unused & (1U << index)) && buffer.allocation))
            return false;
    }
    return true;
}
}  // namespace

bool write_execution_dnn_legacy_metadata(ExecutionPayloadWriter& output,
                                         const ExecutionDnnLegacyMetadata& value) noexcept {
    return valid_metadata(value) && output.write_u32(static_cast<std::uint32_t>(value.kind)) &&
           output.write_u64(value.object) && output.write_u32(value.value) &&
           output.write_u32(value.has_output) && output.write_u32(value.seed);
}
bool read_execution_dnn_legacy_metadata(ExecutionPayloadReader& input,
                                        ExecutionDnnLegacyMetadata& output) noexcept {
    ExecutionDnnLegacyMetadata value;
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u64(value.object) || !input.read_u32(value.value) ||
        !input.read_u32(value.has_output) || !input.read_u32(value.seed))
        return false;
    value.kind = static_cast<ExecutionDnnLegacyMetadataKind>(kind);
    if (!valid_metadata(value)) return false;
    output = value;
    return true;
}

bool write_execution_dnn_legacy_control(ExecutionPayloadWriter& output,
                                        const ExecutionDnnLegacyControl& value) noexcept {
    if (!valid_control(value) || !output.write_u32(static_cast<std::uint32_t>(value.kind)) ||
        !output.write_u64(value.handle) || !output.write_u64(value.metadata) ||
        !output.write_u64(value.descriptor) || !output.write_u32(value.has_output) ||
        !output.write_u64(value.seed))
        return false;
    switch (value.kind) {
        case ExecutionDnnLegacyControlKind::kSetup: {
            const auto& setup = value.setup;
            return write_signed(output, setup.hidden_size) && write_signed(output, setup.layers) &&
                   output.write_u64(setup.dropout) && output.write_u32(setup.input_mode) &&
                   output.write_u32(setup.direction) && output.write_u32(setup.cell_mode) &&
                   output.write_u32(setup.algorithm) && output.write_u32(setup.math_precision);
        }
        case ExecutionDnnLegacyControlKind::kClip:
            return write_execution_dnn_rnn_clip(output, value.clip);
        case ExecutionDnnLegacyControlKind::kWorkspaceSize:
        case ExecutionDnnLegacyControlKind::kReserveSize:
            return write_signed(output, value.sequence_length) &&
                   write_array(output, value.sequence);
        case ExecutionDnnLegacyControlKind::kParameterSize:
            return output.write_u64(value.input) && output.write_u32(value.data_type);
        case ExecutionDnnLegacyControlKind::kMatrixParameter:
        case ExecutionDnnLegacyControlKind::kBiasParameter: {
            const auto& parameters = value.parameters;
            return output.write_u64(parameters.input) && output.write_u64(parameters.filter) &&
                   output.write_u64(parameters.destination) &&
                   write_signed(output, parameters.pseudo_layer) &&
                   write_signed(output, parameters.linear_layer) &&
                   output.write_u64(parameters.weights.allocation) &&
                   output.write_u64(parameters.weights.offset);
        }
        case ExecutionDnnLegacyControlKind::kSetAlgorithm:
            return output.write_u64(value.algorithm);
        case ExecutionDnnLegacyControlKind::kInferenceMaxCount:
            return true;
    }
    return false;
}
bool read_execution_dnn_legacy_control(ExecutionPayloadReader& input,
                                       ExecutionDnnLegacyControl& output) noexcept {
    try {
        ExecutionDnnLegacyControl value;
        std::uint32_t kind{};
        if (!input.read_u32(kind) || !input.read_u64(value.handle) ||
            !input.read_u64(value.metadata) || !input.read_u64(value.descriptor) ||
            !input.read_u32(value.has_output) || !input.read_u64(value.seed))
            return false;
        value.kind = static_cast<ExecutionDnnLegacyControlKind>(kind);
        bool is_decoded{};
        switch (value.kind) {
            case ExecutionDnnLegacyControlKind::kSetup: {
                auto& setup = value.setup;
                is_decoded = read_signed(input, setup.hidden_size) &&
                             read_signed(input, setup.layers) && input.read_u64(setup.dropout) &&
                             input.read_u32(setup.input_mode) && input.read_u32(setup.direction) &&
                             input.read_u32(setup.cell_mode) && input.read_u32(setup.algorithm) &&
                             input.read_u32(setup.math_precision);
                break;
            }
            case ExecutionDnnLegacyControlKind::kClip:
                is_decoded = read_execution_dnn_rnn_clip(input, value.clip);
                break;
            case ExecutionDnnLegacyControlKind::kWorkspaceSize:
            case ExecutionDnnLegacyControlKind::kReserveSize:
                is_decoded = read_signed(input, value.sequence_length) &&
                             read_array(input, value.sequence, value.sequence_length);
                break;
            case ExecutionDnnLegacyControlKind::kParameterSize:
                is_decoded = input.read_u64(value.input) && input.read_u32(value.data_type);
                break;
            case ExecutionDnnLegacyControlKind::kMatrixParameter:
            case ExecutionDnnLegacyControlKind::kBiasParameter: {
                auto& parameters = value.parameters;
                is_decoded = input.read_u64(parameters.input) &&
                             input.read_u64(parameters.filter) &&
                             input.read_u64(parameters.destination) &&
                             read_signed(input, parameters.pseudo_layer) &&
                             read_signed(input, parameters.linear_layer) &&
                             input.read_u64(parameters.weights.allocation) &&
                             input.read_u64(parameters.weights.offset);
                break;
            }
            case ExecutionDnnLegacyControlKind::kSetAlgorithm:
                is_decoded = input.read_u64(value.algorithm);
                break;
            case ExecutionDnnLegacyControlKind::kInferenceMaxCount:
                is_decoded = true;
                break;
        }
        if (!is_decoded || !valid_control(value)) return false;
        output = std::move(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool write_execution_dnn_legacy_output(ExecutionPayloadWriter& output,
                                       const ExecutionDnnLegacyOutput& value) noexcept {
    return valid_output(value) && output.write_u32(value.written) &&
           output.write_u64(value.scalar) && output.write_u32(value.non_null) &&
           output.write_u64(value.offset) && output.write_u64(value.extent);
}
bool read_execution_dnn_legacy_output(ExecutionPayloadReader& input,
                                      ExecutionDnnLegacyOutput& output) noexcept {
    ExecutionDnnLegacyOutput value;
    if (!input.read_u32(value.written) || !input.read_u64(value.scalar) ||
        !input.read_u32(value.non_null) || !input.read_u64(value.offset) ||
        !input.read_u64(value.extent) || !valid_output(value))
        return false;
    output = value;
    return true;
}

bool write_execution_dnn_legacy_compute(ExecutionPayloadWriter& output,
                                        const ExecutionDnnLegacyCompute& value) noexcept {
    if (!valid_compute(value) || !output.write_u32(static_cast<std::uint32_t>(value.kind)) ||
        !output.write_u64(value.handle) || !output.write_u64(value.metadata) ||
        !output.write_u64(value.descriptor) || !write_signed(output, value.sequence_length))
        return false;
    for (const auto& sequence : value.sequences)
        if (!write_array(output, sequence)) return false;
    for (auto id : value.states)
        if (!output.write_u64(id)) return false;
    if (!output.write_u64(value.filter)) return false;
    for (const auto& buffer : value.buffers)
        if (!output.write_u64(buffer.allocation) || !output.write_u64(buffer.offset)) return false;
    return output.write_u64(value.workspace_bytes) && output.write_u64(value.reserve_bytes);
}
bool read_execution_dnn_legacy_compute(ExecutionPayloadReader& input,
                                       ExecutionDnnLegacyCompute& output) noexcept {
    try {
        ExecutionDnnLegacyCompute value;
        std::uint32_t kind{};
        if (!input.read_u32(kind) || !input.read_u64(value.handle) ||
            !input.read_u64(value.metadata) || !input.read_u64(value.descriptor) ||
            !read_signed(input, value.sequence_length))
            return false;
        value.kind = static_cast<ExecutionDnnLegacyComputeKind>(kind);
        for (auto& sequence : value.sequences)
            if (!read_array(input, sequence, value.sequence_length)) return false;
        for (auto& id : value.states)
            if (!input.read_u64(id)) return false;
        if (!input.read_u64(value.filter)) return false;
        for (auto& buffer : value.buffers)
            if (!input.read_u64(buffer.allocation) || !input.read_u64(buffer.offset)) return false;
        if (!input.read_u64(value.workspace_bytes) || !input.read_u64(value.reserve_bytes) ||
            !valid_compute(value))
            return false;
        output = std::move(value);
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace glimmer::control
