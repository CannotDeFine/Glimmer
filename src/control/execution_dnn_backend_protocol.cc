#include "glimmer/control/execution_dnn_backend_protocol.h"

#include <algorithm>
#include <array>
#include <bit>
#include <utility>

namespace glimmer::control {
namespace {

constexpr std::size_t kReferenceBytes = 20;

bool valid_scalar_blob(std::uint32_t type, std::span<const std::byte> bytes) noexcept {
    const auto width = execution_dnn_backend_scalar_width(type);
    if (!width || bytes.size() > kExecutionDnnBackendMaxValueBytes || bytes.size() % width ||
        bytes.size() / width > kExecutionDnnBackendMaxValueElements)
        return false;
    return type != 2 || std::ranges::all_of(bytes, [](std::byte value) {
               return value == std::byte{0} || value == std::byte{1};
           });
}

bool valid_reference(const ExecutionDnnBackendReference& reference,
                     ExecutionDnnBackendValueClass value_class) noexcept {
    if (reference.kind == ExecutionDnnBackendReferenceKind::kNull)
        return reference.owner == 0 && reference.id == 0;
    if (!reference.owner || !reference.id) return false;
    switch (value_class) {
        case ExecutionDnnBackendValueClass::kHandle:
            return reference.kind == ExecutionDnnBackendReferenceKind::kHandle;
        case ExecutionDnnBackendValueClass::kDescriptor:
            return reference.kind == ExecutionDnnBackendReferenceKind::kDescriptor;
        case ExecutionDnnBackendValueClass::kPointerSlot:
            return reference.kind == ExecutionDnnBackendReferenceKind::kPointerSlot;
        default:
            return false;
    }
}

bool read_reference(ExecutionPayloadReader& input, ExecutionDnnBackendReference& reference,
                    ExecutionDnnBackendValueClass value_class) noexcept {
    std::uint32_t kind{};
    if (!input.read_u32(kind) || !input.read_u64(reference.owner) || !input.read_u64(reference.id))
        return false;
    reference.kind = static_cast<ExecutionDnnBackendReferenceKind>(kind);
    return valid_reference(reference, value_class);
}

bool valid_reference_blob(std::span<const std::byte> bytes,
                          ExecutionDnnBackendValueClass value_class) noexcept {
    if (bytes.size() > kExecutionDnnBackendMaxValueBytes || bytes.size() % kReferenceBytes ||
        bytes.size() / kReferenceBytes > kExecutionDnnBackendMaxValueElements)
        return false;
    ExecutionPayloadReader input(bytes);
    while (!input.empty()) {
        ExecutionDnnBackendReference reference;
        if (!read_reference(input, reference, value_class)) return false;
    }
    return true;
}

bool valid_values(const ExecutionDnnBackendAttributeValues& value,
                  ExecutionDnnBackendValueClass value_class) noexcept {
    if (value_class == ExecutionDnnBackendValueClass::kInvalid) return false;
    if (value_class == ExecutionDnnBackendValueClass::kScalar)
        return value.references.empty() && valid_scalar_blob(value.type, value.scalars);
    return value.scalars.empty() &&
           value.references.size() <= kExecutionDnnBackendMaxValueElements &&
           value.references.size() <= kExecutionDnnBackendMaxValueBytes / kReferenceBytes &&
           std::ranges::all_of(value.references, [value_class](const auto& reference) {
               return valid_reference(reference, value_class);
           });
}

std::size_t value_body_bytes(const ExecutionDnnBackendAttributeValues& value) noexcept {
    return execution_dnn_backend_value_class(value.type) == ExecutionDnnBackendValueClass::kScalar
               ? value.scalars.size()
               : value.references.size() * kReferenceBytes;
}

bool valid_prepare(const ExecutionDnnBackendPrepare& value) noexcept {
    return (value.operation == ExecutionDnnBackendAttributeOperation::kSet &&
            value.header.presence <= 1) ||
           (value.operation == ExecutionDnnBackendAttributeOperation::kGet &&
            value.header.presence <= 3);
}

bool valid_count_bytes(std::uint32_t bytes) noexcept {
    return bytes == 0 || bytes == 1 || bytes == 8;
}

bool valid_preparation(const ExecutionDnnBackendAttributePreparation& value) noexcept {
    if (!value.generation || value.input_elements > kExecutionDnnBackendMaxValueElements ||
        value.output_elements > kExecutionDnnBackendMaxValueElements ||
        value.output_bytes > kExecutionDnnBackendMaxValueBytes ||
        !valid_count_bytes(value.count_bytes))
        return false;
    switch (value.direction) {
        case ExecutionDnnBackendAttributeDirection::kNone:
            return !value.input_elements && !value.output_elements && !value.output_bytes;
        case ExecutionDnnBackendAttributeDirection::kInput:
        case ExecutionDnnBackendAttributeDirection::kDescriptorOutput:
            return !value.output_elements && !value.output_bytes;
        case ExecutionDnnBackendAttributeDirection::kScalarOutput:
            return !value.input_elements &&
                   ((value.output_elements == 0) == (value.output_bytes == 0));
        case ExecutionDnnBackendAttributeDirection::kHandleOutput:
        case ExecutionDnnBackendAttributeDirection::kPointerOutput:
            return !value.input_elements && !value.output_bytes;
        default:
            return false;
    }
}

bool valid_attribute(const ExecutionDnnBackendAttribute& value) noexcept {
    return value.header.presence <= 3 && value.generation &&
           value.header.type == value.values.type &&
           ((value.header.presence & 1U) ||
            (value.values.scalars.empty() && value.values.references.empty())) &&
           valid_values(value.values, execution_dnn_backend_value_class(value.values.type));
}

bool valid_get_result(const ExecutionDnnBackendGetResult& value) noexcept {
    return valid_count_bytes(value.count_bytes) && (value.count_bytes || value.count_bits == 0) &&
           (value.count_bytes != 1 || value.count_bits <= UINT8_MAX) &&
           valid_values(value.values, execution_dnn_backend_value_class(value.values.type));
}

bool valid_context_operation(ExecutionOpcode operation) noexcept {
    switch (operation) {
        case ExecutionOpcode::kDnnBackendPrepare:
        case ExecutionOpcode::kDnnBackendSet:
        case ExecutionOpcode::kDnnBackendGet:
        case ExecutionOpcode::kDnnBackendFinalize:
        case ExecutionOpcode::kDnnBackendInitialize:
        case ExecutionOpcode::kDnnBackendDestroy:
            return true;
        default:
            return false;
    }
}

struct BindingOrder {
    std::uint32_t last{};
    bool has_tensor{}, has_workspace{};
    bool accept(std::uint32_t index) noexcept {
        if (index == kExecutionDnnBackendWorkspaceIndex) {
            if (has_workspace) return false;
            has_workspace = true;
            return true;
        }
        if (index >= kExecutionDnnBackendMaxTensorBindings || (has_tensor && index <= last))
            return false;
        has_tensor = true;
        last = index;
        return true;
    }
};

bool valid_requirement(const ExecutionDnnBackendExecuteRequirement& value,
                       std::size_t& host_bytes) noexcept {
    if (value.role == ExecutionDnnBackendBindingRole::kWorkspace) {
        if (value.index != kExecutionDnnBackendWorkspaceIndex || value.uid ||
            value.value_class == ExecutionDnnBackendBindingClass::kByValue)
            return false;
    } else if (value.role != ExecutionDnnBackendBindingRole::kTensor ||
               value.index >= kExecutionDnnBackendMaxTensorBindings) {
        return false;
    }
    if (value.value_class == ExecutionDnnBackendBindingClass::kIgnored)
        return !value.bytes && !value.alignment &&
               value.access == ExecutionDnnBackendBindingAccess::kNone;
    if (!value.alignment || value.access < ExecutionDnnBackendBindingAccess::kRead ||
        value.access > ExecutionDnnBackendBindingAccess::kReadWrite)
        return false;
    if (value.value_class == ExecutionDnnBackendBindingClass::kDevice) return true;
    if (value.value_class != ExecutionDnnBackendBindingClass::kByValue ||
        value.access != ExecutionDnnBackendBindingAccess::kRead)
        return false;
    // A native NULL remains NULL, even if the required tensor extent is large.
    // Only actual Execute-time captures consume the bounded host payload quota.
    if (value.slot) {
        if (value.bytes > kExecutionDnnBackendMaxValueBytes - host_bytes) return false;
        host_bytes += static_cast<std::size_t>(value.bytes);
    }
    return true;
}

bool read_requirement(ExecutionPayloadReader& input,
                      ExecutionDnnBackendExecuteRequirement& value) noexcept {
    std::uint32_t role{}, value_class{}, access{};
    std::uint64_t uid{};
    if (!input.read_u32(value.index) || !input.read_u32(role) || !input.read_u32(value_class) ||
        !input.read_u32(access) || !input.read_u64(uid) || !input.read_u64(value.slot) ||
        !input.read_u64(value.bytes) || !input.read_u64(value.alignment))
        return false;
    value.role = static_cast<ExecutionDnnBackendBindingRole>(role);
    value.value_class = static_cast<ExecutionDnnBackendBindingClass>(value_class);
    value.access = static_cast<ExecutionDnnBackendBindingAccess>(access);
    value.uid = std::bit_cast<std::int64_t>(uid);
    return true;
}

struct BindingView {
    std::uint32_t index{};
    ExecutionDnnBackendBindingClass value_class{};
    std::uint64_t slot{}, allocation{}, offset{};
    std::span<const std::byte> bytes;
};

bool valid_binding(const BindingView& value, std::size_t& host_bytes) noexcept {
    if (value.value_class > ExecutionDnnBackendBindingClass::kByValue ||
        (value.index == kExecutionDnnBackendWorkspaceIndex &&
         value.value_class == ExecutionDnnBackendBindingClass::kByValue) ||
        value.bytes.size() > kExecutionDnnBackendMaxValueBytes - host_bytes)
        return false;
    if (!value.slot) return !value.allocation && !value.offset && value.bytes.empty();
    switch (value.value_class) {
        case ExecutionDnnBackendBindingClass::kIgnored:
            return !value.allocation && !value.offset && value.bytes.empty();
        case ExecutionDnnBackendBindingClass::kDevice:
            return value.allocation && value.bytes.empty();
        case ExecutionDnnBackendBindingClass::kByValue:
            if (value.allocation || value.offset) return false;
            host_bytes += value.bytes.size();
            return true;
    }
    return false;
}

}  // namespace

ExecutionDnnBackendValueClass execution_dnn_backend_value_class(std::uint32_t type) noexcept {
    switch (type) {
        case 0:
            return ExecutionDnnBackendValueClass::kHandle;
        case 6:
            return ExecutionDnnBackendValueClass::kPointerSlot;
        case 15:
            return ExecutionDnnBackendValueClass::kDescriptor;
        default:
            return type <= 29 ? ExecutionDnnBackendValueClass::kScalar
                              : ExecutionDnnBackendValueClass::kInvalid;
    }
}

std::size_t execution_dnn_backend_scalar_width(std::uint32_t type) noexcept {
    if (execution_dnn_backend_value_class(type) != ExecutionDnnBackendValueClass::kScalar) return 0;
    switch (type) {
        case 2:
        case 24:
            return 1;
        case 3:
        case 5:
            return 8;
        case 26:
            return 16;
        default:
            return 4;
    }
}

bool write_execution_dnn_backend_values(ExecutionPayloadWriter& output,
                                        const ExecutionDnnBackendAttributeValues& value) noexcept {
    const auto value_class = execution_dnn_backend_value_class(value.type);
    if (!valid_values(value, value_class) ||
        !output.reserve_additional_capacity(8 + value_body_bytes(value)))
        return false;
    if (value_class == ExecutionDnnBackendValueClass::kScalar)
        return output.write_u32(value.type) && output.write_blob(value.scalars);
    if (!output.write_u32(value.type) ||
        !output.write_u32(static_cast<std::uint32_t>(value_body_bytes(value))))
        return false;
    for (const auto& reference : value.references)
        if (!output.write_u32(static_cast<std::uint32_t>(reference.kind)) ||
            !output.write_u64(reference.owner) || !output.write_u64(reference.id))
            return false;
    return true;
}

bool read_execution_dnn_backend_values(ExecutionPayloadReader& input,
                                       ExecutionDnnBackendAttributeValues& value) noexcept {
    try {
        auto cursor = input;
        ExecutionDnnBackendAttributeValues decoded;
        std::span<const std::byte> bytes;
        if (!cursor.read_u32(decoded.type) || !cursor.read_blob(bytes)) return false;
        const auto value_class = execution_dnn_backend_value_class(decoded.type);
        if (value_class == ExecutionDnnBackendValueClass::kInvalid) return false;
        if (value_class == ExecutionDnnBackendValueClass::kScalar) {
            if (!valid_scalar_blob(decoded.type, bytes)) return false;
            decoded.scalars.assign(bytes.begin(), bytes.end());
        } else {
            // Validate the entire bounded span and every reference before any
            // vector allocation. A malformed later element cannot escape early.
            if (!valid_reference_blob(bytes, value_class)) return false;
            decoded.references.resize(bytes.size() / kReferenceBytes);
            ExecutionPayloadReader elements(bytes);
            for (auto& reference : decoded.references)
                if (!read_reference(elements, reference, value_class)) return false;
        }
        value = std::move(decoded);
        input = cursor;
        return true;
    } catch (...) {
        return false;
    }
}

bool write_execution_dnn_backend_create(ExecutionPayloadWriter& output,
                                        const ExecutionDnnBackendCreate& value) noexcept {
    return value.has_output <= 1 && output.reserve_additional_capacity(8) &&
           output.write_u32(value.kind) && output.write_u32(value.has_output);
}

bool read_execution_dnn_backend_create(ExecutionPayloadReader& input,
                                       ExecutionDnnBackendCreate& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendCreate decoded;
    if (!cursor.read_u32(decoded.kind) || !cursor.read_u32(decoded.has_output) ||
        decoded.has_output > 1)
        return false;
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_attribute_header(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttributeHeader& value) noexcept {
    return value.presence <= 3 && output.reserve_additional_capacity(28) &&
           output.write_u64(value.descriptor) && output.write_u32(value.name) &&
           output.write_u32(value.type) &&
           output.write_u64(std::bit_cast<std::uint64_t>(value.count)) &&
           output.write_u32(value.presence);
}

bool read_execution_dnn_backend_attribute_header(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttributeHeader& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendAttributeHeader decoded;
    std::uint64_t count{};
    if (!cursor.read_u64(decoded.descriptor) || !cursor.read_u32(decoded.name) ||
        !cursor.read_u32(decoded.type) || !cursor.read_u64(count) ||
        !cursor.read_u32(decoded.presence) || decoded.presence > 3)
        return false;
    decoded.count = std::bit_cast<std::int64_t>(count);
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_prepare(ExecutionPayloadWriter& output,
                                         const ExecutionDnnBackendPrepare& value) noexcept {
    return valid_prepare(value) && output.reserve_additional_capacity(32) &&
           output.write_u32(static_cast<std::uint32_t>(value.operation)) &&
           write_execution_dnn_backend_attribute_header(output, value.header);
}

bool read_execution_dnn_backend_prepare(ExecutionPayloadReader& input,
                                        ExecutionDnnBackendPrepare& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendPrepare decoded;
    std::uint32_t operation{};
    if (!cursor.read_u32(operation) ||
        !read_execution_dnn_backend_attribute_header(cursor, decoded.header))
        return false;
    decoded.operation = static_cast<ExecutionDnnBackendAttributeOperation>(operation);
    if (!valid_prepare(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_preparation(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttributePreparation& value) noexcept {
    return valid_preparation(value) && output.reserve_additional_capacity(28) &&
           output.write_u64(value.generation) &&
           output.write_u32(static_cast<std::uint32_t>(value.direction)) &&
           output.write_u32(value.input_elements) && output.write_u32(value.output_elements) &&
           output.write_u32(value.count_bytes) && output.write_u32(value.output_bytes);
}

bool read_execution_dnn_backend_preparation(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttributePreparation& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendAttributePreparation decoded;
    std::uint32_t direction{};
    if (!cursor.read_u64(decoded.generation) || !cursor.read_u32(direction) ||
        !cursor.read_u32(decoded.input_elements) || !cursor.read_u32(decoded.output_elements) ||
        !cursor.read_u32(decoded.count_bytes) || !cursor.read_u32(decoded.output_bytes))
        return false;
    decoded.direction = static_cast<ExecutionDnnBackendAttributeDirection>(direction);
    if (!valid_preparation(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_attribute(ExecutionPayloadWriter& output,
                                           const ExecutionDnnBackendAttribute& value) noexcept {
    return valid_attribute(value) &&
           output.reserve_additional_capacity(44 + value_body_bytes(value.values)) &&
           write_execution_dnn_backend_attribute_header(output, value.header) &&
           output.write_u64(value.generation) &&
           write_execution_dnn_backend_values(output, value.values);
}

bool read_execution_dnn_backend_attribute(ExecutionPayloadReader& input,
                                          ExecutionDnnBackendAttribute& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendAttribute decoded;
    if (!read_execution_dnn_backend_attribute_header(cursor, decoded.header) ||
        !cursor.read_u64(decoded.generation) ||
        !read_execution_dnn_backend_values(cursor, decoded.values) || !valid_attribute(decoded))
        return false;
    value = std::move(decoded);
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_get_result(ExecutionPayloadWriter& output,
                                            const ExecutionDnnBackendGetResult& value) noexcept {
    return valid_get_result(value) &&
           output.reserve_additional_capacity(20 + value_body_bytes(value.values)) &&
           output.write_u32(value.count_bytes) && output.write_u64(value.count_bits) &&
           write_execution_dnn_backend_values(output, value.values);
}

bool read_execution_dnn_backend_get_result(ExecutionPayloadReader& input,
                                           ExecutionDnnBackendGetResult& value) noexcept {
    auto cursor = input;
    ExecutionDnnBackendGetResult decoded;
    if (!cursor.read_u32(decoded.count_bytes) || !cursor.read_u64(decoded.count_bits) ||
        !read_execution_dnn_backend_values(cursor, decoded.values) || !valid_get_result(decoded))
        return false;
    value = std::move(decoded);
    input = cursor;
    return true;
}

bool write_execution_dnn_backend_context(ExecutionPayloadWriter& output,
                                         const ExecutionDnnBackendContext& value) noexcept {
    return valid_context_operation(value.operation) &&
           value.payload.size() <= kExecutionDnnBackendMaxValueBytes &&
           output.reserve_additional_capacity(24 + value.payload.size()) &&
           output.write_u64(value.metadata) && output.write_u64(value.handle) &&
           output.write_u32(static_cast<std::uint32_t>(value.operation)) &&
           output.write_blob(value.payload);
}

bool read_execution_dnn_backend_context(ExecutionPayloadReader& input,
                                        ExecutionDnnBackendContext& value) noexcept {
    try {
        auto reader = input;
        ExecutionDnnBackendContext decoded;
        std::uint32_t operation{};
        std::span<const std::byte> payload;
        if (!reader.read_u64(decoded.metadata) || !reader.read_u64(decoded.handle) ||
            !reader.read_u32(operation) || !reader.read_blob(payload) ||
            payload.size() > kExecutionDnnBackendMaxValueBytes)
            return false;
        decoded.operation = static_cast<ExecutionOpcode>(operation);
        if (!valid_context_operation(decoded.operation)) return false;
        decoded.payload.assign(payload.begin(), payload.end());
        value = std::move(decoded);
        input = reader;
        return true;
    } catch (...) {
        return false;
    }
}

bool write_execution_dnn_backend_execute_prepare(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendExecutePrepare& value) noexcept {
    return output.reserve_additional_capacity(32) && output.write_u64(value.handle) &&
           output.write_u64(value.metadata) && output.write_u64(value.plan) &&
           output.write_u64(value.variant);
}

bool read_execution_dnn_backend_execute_prepare(ExecutionPayloadReader& input,
                                                ExecutionDnnBackendExecutePrepare& value) noexcept {
    auto reader = input;
    ExecutionDnnBackendExecutePrepare decoded;
    if (!reader.read_u64(decoded.handle) || !reader.read_u64(decoded.metadata) ||
        !reader.read_u64(decoded.plan) || !reader.read_u64(decoded.variant))
        return false;
    value = decoded;
    input = reader;
    return true;
}

bool write_execution_dnn_backend_execute_preparation(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendExecutePreparation& value) noexcept {
    if (!value.plan_generation || !value.variant_generation ||
        value.requirements.size() > kExecutionDnnBackendMaxBindings)
        return false;
    BindingOrder order;
    std::size_t host_bytes{};
    for (const auto& requirement : value.requirements)
        if (!order.accept(requirement.index) || !valid_requirement(requirement, host_bytes))
            return false;
    if (!output.reserve_additional_capacity(20 + 48 * value.requirements.size())) return false;
    if (!output.write_u64(value.plan_generation) || !output.write_u64(value.variant_generation) ||
        !output.write_u32(static_cast<std::uint32_t>(value.requirements.size())))
        return false;
    for (const auto& requirement : value.requirements)
        if (!output.write_u32(requirement.index) ||
            !output.write_u32(static_cast<std::uint32_t>(requirement.role)) ||
            !output.write_u32(static_cast<std::uint32_t>(requirement.value_class)) ||
            !output.write_u32(static_cast<std::uint32_t>(requirement.access)) ||
            !output.write_u64(std::bit_cast<std::uint64_t>(requirement.uid)) ||
            !output.write_u64(requirement.slot) || !output.write_u64(requirement.bytes) ||
            !output.write_u64(requirement.alignment))
            return false;
    return true;
}

bool read_execution_dnn_backend_execute_preparation(
    ExecutionPayloadReader& input, ExecutionDnnBackendExecutePreparation& value) noexcept {
    try {
        auto reader = input;
        ExecutionDnnBackendExecutePreparation decoded;
        std::uint32_t count{};
        if (!reader.read_u64(decoded.plan_generation) ||
            !reader.read_u64(decoded.variant_generation) || !reader.read_u32(count) ||
            !decoded.plan_generation || !decoded.variant_generation ||
            count > kExecutionDnnBackendMaxBindings)
            return false;
        std::array<ExecutionDnnBackendExecuteRequirement, kExecutionDnnBackendMaxBindings> values;
        BindingOrder order;
        std::size_t host_bytes{};
        for (auto& requirement : std::span{values}.first(count))
            if (!read_requirement(reader, requirement) || !order.accept(requirement.index) ||
                !valid_requirement(requirement, host_bytes))
                return false;
        decoded.requirements.assign(values.begin(), values.begin() + count);
        value = std::move(decoded);
        input = reader;
        return true;
    } catch (...) {
        return false;
    }
}

bool write_execution_dnn_backend_execute(ExecutionPayloadWriter& output,
                                         const ExecutionDnnBackendExecute& value) noexcept {
    if (!value.plan_generation || !value.variant_generation ||
        value.bindings.size() > kExecutionDnnBackendMaxBindings)
        return false;
    BindingOrder order;
    std::size_t host_bytes{};
    for (const auto& binding : value.bindings)
        if (!order.accept(binding.index) ||
            !valid_binding({binding.index, binding.value_class, binding.slot, binding.allocation,
                            binding.offset, binding.host_bytes},
                           host_bytes))
            return false;
    if (!output.reserve_additional_capacity(52 + 36 * value.bindings.size() + host_bytes))
        return false;
    if (!write_execution_dnn_backend_execute_prepare(
            output, {value.handle, value.metadata, value.plan, value.variant}) ||
        !output.write_u64(value.plan_generation) || !output.write_u64(value.variant_generation) ||
        !output.write_u32(static_cast<std::uint32_t>(value.bindings.size())))
        return false;
    for (const auto& binding : value.bindings)
        if (!output.write_u32(binding.index) ||
            !output.write_u32(static_cast<std::uint32_t>(binding.value_class)) ||
            !output.write_u64(binding.slot) || !output.write_u64(binding.allocation) ||
            !output.write_u64(binding.offset) || !output.write_blob(binding.host_bytes))
            return false;
    return true;
}

bool read_execution_dnn_backend_execute(ExecutionPayloadReader& input,
                                        ExecutionDnnBackendExecute& value) noexcept {
    try {
        auto reader = input;
        ExecutionDnnBackendExecute decoded;
        std::uint32_t count{};
        if (!reader.read_u64(decoded.handle) || !reader.read_u64(decoded.metadata) ||
            !reader.read_u64(decoded.plan) || !reader.read_u64(decoded.variant) ||
            !reader.read_u64(decoded.plan_generation) ||
            !reader.read_u64(decoded.variant_generation) || !reader.read_u32(count) ||
            !decoded.plan_generation || !decoded.variant_generation ||
            count > kExecutionDnnBackendMaxBindings)
            return false;
        std::array<BindingView, kExecutionDnnBackendMaxBindings> values;
        BindingOrder order;
        std::size_t host_bytes{};
        for (auto& binding : std::span{values}.first(count)) {
            std::uint32_t value_class{};
            if (!reader.read_u32(binding.index) || !reader.read_u32(value_class) ||
                !reader.read_u64(binding.slot) || !reader.read_u64(binding.allocation) ||
                !reader.read_u64(binding.offset) || !reader.read_blob(binding.bytes))
                return false;
            binding.value_class = static_cast<ExecutionDnnBackendBindingClass>(value_class);
            if (!order.accept(binding.index) || !valid_binding(binding, host_bytes)) return false;
        }
        // Every span/class/index is validated before the first heap allocation.
        decoded.bindings.reserve(count);
        for (const auto& binding : std::span{values}.first(count)) {
            ExecutionDnnBackendExecuteBinding owned{binding.index,  binding.value_class,
                                                    binding.slot,   binding.allocation,
                                                    binding.offset, {}};
            owned.host_bytes.assign(binding.bytes.begin(), binding.bytes.end());
            decoded.bindings.push_back(std::move(owned));
        }
        value = std::move(decoded);
        input = reader;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace glimmer::control
