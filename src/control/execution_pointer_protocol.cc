#include "glimmer/control/execution_pointer_protocol.h"

#include "glimmer/control/execution_process_memory_protocol.h"

#include <bit>

namespace glimmer::control {
namespace {
bool valid(const ExecutionPointerQuery& value) noexcept {
    if (value.domain == ExecutionPointerDomain::kDriver) {
        if (value.selector != 3 && value.selector != 9 &&
            !(value.selector == 1 && value.kind == ExecutionPointerKind::kProcessDevice))
            return false;
    } else if (value.domain != ExecutionPointerDomain::kRuntime || value.selector != 0) {
        return false;
    }
    switch (value.kind) {
        case ExecutionPointerKind::kNull:
        case ExecutionPointerKind::kUnregistered:
            return !value.source_session && !value.allocation && !value.offset;
        case ExecutionPointerKind::kDevice:
        case ExecutionPointerKind::kHost:
            return value.source_session && value.allocation;
        case ExecutionPointerKind::kProcessDevice:
            return value.source_session && execution_is_process_allocation(value.allocation);
    }
    return false;
}

bool valid(const ExecutionPointerObservation& value) noexcept {
    if (value.effects > 15 || static_cast<std::uint32_t>(value.device_alias) > 1 ||
        static_cast<std::uint32_t>(value.host_alias) > 1)
        return false;
    if (!(value.effects & 1U) && value.type != 0) return false;
    if (!(value.effects & 2U) && value.device != 0) return false;
    return ((value.effects & 4U) || value.device_alias == ExecutionPointerAlias::kNull) &&
           ((value.effects & 8U) || value.host_alias == ExecutionPointerAlias::kNull);
}
}  // namespace

bool write_execution_pointer_query(ExecutionPayloadWriter& output,
                                   const ExecutionPointerQuery& value) noexcept {
    if (!valid(value) || !output.reserve_additional_capacity(kExecutionPointerQueryBytes))
        return false;
    return output.write_u32(static_cast<std::uint32_t>(value.domain)) &&
           output.write_u32(value.selector) &&
           output.write_u32(static_cast<std::uint32_t>(value.kind)) && output.write_u32(0) &&
           output.write_u64(value.query_session) && output.write_u64(value.source_session) &&
           output.write_u64(value.allocation) && output.write_u64(value.offset);
}

bool read_execution_pointer_query(ExecutionPayloadReader& input,
                                  ExecutionPointerQuery& value) noexcept {
    auto cursor = input;
    ExecutionPointerQuery decoded;
    std::uint32_t domain{}, kind{}, reserved{};
    if (!cursor.read_u32(domain) || !cursor.read_u32(decoded.selector) || !cursor.read_u32(kind) ||
        !cursor.read_u32(reserved) || reserved || !cursor.read_u64(decoded.query_session) ||
        !cursor.read_u64(decoded.source_session) || !cursor.read_u64(decoded.allocation) ||
        !cursor.read_u64(decoded.offset))
        return false;
    decoded.domain = static_cast<ExecutionPointerDomain>(domain);
    decoded.kind = static_cast<ExecutionPointerKind>(kind);
    if (!valid(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}

bool write_execution_pointer_observation(ExecutionPayloadWriter& output,
                                         const ExecutionPointerObservation& value) noexcept {
    if (!valid(value) || !output.reserve_additional_capacity(kExecutionPointerObservationBytes))
        return false;
    return output.write_u32(value.effects) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.type)) &&
           output.write_u32(std::bit_cast<std::uint32_t>(value.device)) &&
           output.write_u32(static_cast<std::uint32_t>(value.device_alias)) &&
           output.write_u32(static_cast<std::uint32_t>(value.host_alias));
}

bool read_execution_pointer_observation(ExecutionPayloadReader& input,
                                        ExecutionPointerObservation& value) noexcept {
    auto cursor = input;
    ExecutionPointerObservation decoded;
    std::uint32_t type{}, device{}, device_alias{}, host_alias{};
    if (!cursor.read_u32(decoded.effects) || !cursor.read_u32(type) || !cursor.read_u32(device) ||
        !cursor.read_u32(device_alias) || !cursor.read_u32(host_alias))
        return false;
    decoded.type = std::bit_cast<std::int32_t>(type);
    decoded.device = std::bit_cast<std::int32_t>(device);
    decoded.device_alias = static_cast<ExecutionPointerAlias>(device_alias);
    decoded.host_alias = static_cast<ExecutionPointerAlias>(host_alias);
    if (!valid(decoded)) return false;
    value = decoded;
    input = cursor;
    return true;
}

}  // namespace glimmer::control
