#pragma once

#include "glimmer/control/execution_protocol.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glimmer::control {

inline constexpr std::size_t kExecutionDnnBackendMaxValueElements = std::size_t{1} << 16;
inline constexpr std::size_t kExecutionDnnBackendMaxValueBytes = std::size_t{2} << 20;
inline constexpr std::size_t kExecutionDnnBackendMaxTensorBindings = 100;
inline constexpr std::size_t kExecutionDnnBackendMaxBindings = 101;
inline constexpr std::uint32_t kExecutionDnnBackendWorkspaceIndex = UINT32_MAX;

// Fixed-width enum types and values are part of the protocol contract.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendValueClass : std::uint32_t {
    kInvalid = 0,
    kScalar = 1,
    kHandle = 2,
    kDescriptor = 3,
    kPointerSlot = 4
};

// These helpers classify the 30 installed attribute type numbers without SDK
// headers. Unknown types are invalid, not an untyped byte transport.
[[nodiscard]] ExecutionDnnBackendValueClass execution_dnn_backend_value_class(
    std::uint32_t type) noexcept;
[[nodiscard]] std::size_t execution_dnn_backend_scalar_width(std::uint32_t type) noexcept;

// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendReferenceKind : std::uint32_t {
    kNull = 0,
    kHandle = 1,
    kDescriptor = 2,
    kPointerSlot = 3
};

struct ExecutionDnnBackendReference {
    ExecutionDnnBackendReferenceKind kind{};
    std::uint64_t owner{}, id{};
};

struct ExecutionDnnBackendAttributeValues {
    std::uint32_t type{};
    // Canonical little-endian scalar fields, including raw float/enum bits and
    // the two signed 64-bit fields of a fraction. Boolean bytes are only 0/1.
    std::vector<std::byte> scalars;
    // Symbolic identities, never native pointers or allocation grants. A
    // PointerSlot keeps an address binding local until execution determines
    // whether to capture host values or acquire a device-allocation lease.
    std::vector<ExecutionDnnBackendReference> references;
};

struct ExecutionDnnBackendCreate {
    std::uint32_t kind{}, has_output{};
};
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendAttributeOperation : std::uint32_t { kSet = 1, kGet = 2 };
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendAttributeDirection : std::uint32_t {
    kNone = 0,
    kInput = 1,
    kScalarOutput = 2,
    kDescriptorOutput = 3,
    kHandleOutput = 4,
    kPointerOutput = 5
};
struct ExecutionDnnBackendAttributeHeader {
    std::uint64_t descriptor{};
    std::uint32_t name{}, type{};
    std::int64_t count{};
    // Array=1 and count-output=2 are independent. Set forbids count-output.
    std::uint32_t presence{};
};
struct ExecutionDnnBackendPrepare {
    ExecutionDnnBackendAttributeOperation operation{};
    ExecutionDnnBackendAttributeHeader header;
};
struct ExecutionDnnBackendAttributePreparation {
    std::uint64_t generation{};
    ExecutionDnnBackendAttributeDirection direction{};
    // Typed transport storage is distinct from the exact caller write prefix.
    // Native one-element calls may access twice or half the declared width.
    std::uint32_t input_elements{}, output_elements{}, count_bytes{}, output_bytes{};
};
struct ExecutionDnnBackendAttribute {
    ExecutionDnnBackendAttributeHeader header;
    std::uint64_t generation{};
    ExecutionDnnBackendAttributeValues values;
};
struct ExecutionDnnBackendGetResult {
    std::uint32_t count_bytes{};
    std::uint64_t count_bits{};
    ExecutionDnnBackendAttributeValues values;
};

// Execution-channel envelope; metadata continues to identify the original
// DNM1 owner. The service authenticates both owners and resolves the handle.
struct ExecutionDnnBackendContext {
    std::uint64_t metadata{}, handle{};
    ExecutionOpcode operation{};
    std::vector<std::byte> payload;
};
struct ExecutionDnnBackendExecutePrepare {
    std::uint64_t handle{}, metadata{}, plan{}, variant{};
};
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendBindingRole : std::uint32_t { kTensor = 1, kWorkspace = 2 };
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendBindingClass : std::uint32_t {
    kIgnored = 0,
    kDevice = 1,
    kByValue = 2
};
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionDnnBackendBindingAccess : std::uint32_t {
    kNone = 0,
    kRead = 1,
    kWrite = 2,
    kReadWrite = 3
};
struct ExecutionDnnBackendExecuteRequirement {
    std::uint32_t index{};
    ExecutionDnnBackendBindingRole role{};
    ExecutionDnnBackendBindingClass value_class{};
    ExecutionDnnBackendBindingAccess access{};
    std::int64_t uid{};
    std::uint64_t slot{}, bytes{}, alignment{};
};
struct ExecutionDnnBackendExecutePreparation {
    std::uint64_t plan_generation{}, variant_generation{};
    std::vector<ExecutionDnnBackendExecuteRequirement> requirements;
};
struct ExecutionDnnBackendExecuteBinding {
    std::uint32_t index{};
    ExecutionDnnBackendBindingClass value_class{};
    std::uint64_t slot{}, allocation{}, offset{};
    std::vector<std::byte> host_bytes;
};
struct ExecutionDnnBackendExecute {
    std::uint64_t handle{}, metadata{}, plan{}, variant{};
    std::uint64_t plan_generation{}, variant_generation{};
    std::vector<ExecutionDnnBackendExecuteBinding> bindings;
};

// Context: two u64 + operation u32 + blob. Execute preparation request: 32B.
// Preparation reply: two generations + count u32 + ordered 48B requirements.
// Execute: six u64 + count u32 + index/class/slot/allocation/offset + host blob.
// Tensor indices are unique, increasing 0..99; a workspace index may occur once.
// Duplicate UIDs are deliberately preserved. Null slots never acquire storage.
// Positive alignment need not be a power of two. Cross-message correspondence,
// owner identities and actual ranges remain server admission responsibilities.
[[nodiscard]] bool write_execution_dnn_backend_context(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendContext& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_context(ExecutionPayloadReader& input,
                                                      ExecutionDnnBackendContext& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_execute_prepare(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendExecutePrepare& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_execute_prepare(
    ExecutionPayloadReader& input, ExecutionDnnBackendExecutePrepare& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_execute_preparation(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendExecutePreparation& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_execute_preparation(
    ExecutionPayloadReader& input, ExecutionDnnBackendExecutePreparation& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_execute(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendExecute& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_execute(ExecutionPayloadReader& input,
                                                      ExecutionDnnBackendExecute& value) noexcept;

// Encode u32 type followed by one byte-length-prefixed blob. Reference elements
// are 20 bytes (u32 kind/u64 owner/u64 id); scalar widths derive from type.
// Signed native counts and nullable arrays belong to the enclosing operation.
// Failure preserves writer prefix or reader cursor and destination. Successful
// reads consume only this value, leaving trailing-byte checks to the caller.
[[nodiscard]] bool write_execution_dnn_backend_values(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttributeValues& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_values(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttributeValues& value) noexcept;

[[nodiscard]] bool write_execution_dnn_backend_create(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendCreate& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_create(ExecutionPayloadReader& input,
                                                     ExecutionDnnBackendCreate& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_attribute_header(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttributeHeader& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_attribute_header(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttributeHeader& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_prepare(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendPrepare& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_prepare(ExecutionPayloadReader& input,
                                                      ExecutionDnnBackendPrepare& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_preparation(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttributePreparation& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_preparation(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttributePreparation& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_attribute(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendAttribute& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_attribute(
    ExecutionPayloadReader& input, ExecutionDnnBackendAttribute& value) noexcept;
[[nodiscard]] bool write_execution_dnn_backend_get_result(
    ExecutionPayloadWriter& output, const ExecutionDnnBackendGetResult& value) noexcept;
[[nodiscard]] bool read_execution_dnn_backend_get_result(
    ExecutionPayloadReader& input, ExecutionDnnBackendGetResult& value) noexcept;

}  // namespace glimmer::control
