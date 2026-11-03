#include "glimmer/control/execution_dnn_backend_protocol.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::control;
using Bytes = std::vector<std::byte>;
using Context = ExecutionDnnBackendContext;
using Prepare = ExecutionDnnBackendExecutePrepare;
using Preparation = ExecutionDnnBackendExecutePreparation;
using Requirement = ExecutionDnnBackendExecuteRequirement;
using Execute = ExecutionDnnBackendExecute;
using Binding = ExecutionDnnBackendExecuteBinding;
using Role = ExecutionDnnBackendBindingRole;
using Class = ExecutionDnnBackendBindingClass;
using Access = ExecutionDnnBackendBindingAccess;
constexpr std::size_t kHostLimit = 2097152;
constexpr std::uint32_t kWorkspace = UINT32_MAX;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Independent little-endian fixtures, never the production writer or sizeof(struct).
template <std::size_t Width>
void append(Bytes& bytes, std::uint64_t value) {
    static_assert(Width == 4 || Width == 8);
    for (std::size_t index = 0; index < Width; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 255));
}

void blob(Bytes& bytes, std::span<const std::byte> value) {
    expect(value.size() <= UINT32_MAX, "fixture blob length is representable");
    append<4>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

Bytes wire(const Context& value) {
    Bytes bytes;
    append<8>(bytes, value.metadata);
    append<8>(bytes, value.handle);
    append<4>(bytes, static_cast<std::uint32_t>(value.operation));
    blob(bytes, value.payload);
    return bytes;
}
Bytes wire(const Prepare& value) {
    Bytes bytes;
    append<8>(bytes, value.handle);
    append<8>(bytes, value.metadata);
    append<8>(bytes, value.plan);
    append<8>(bytes, value.variant);
    return bytes;
}
Bytes wire(const Preparation& value) {
    Bytes bytes;
    append<8>(bytes, value.plan_generation);
    append<8>(bytes, value.variant_generation);
    append<4>(bytes, value.requirements.size());
    for (const auto& entry : value.requirements) {
        append<4>(bytes, entry.index);
        append<4>(bytes, static_cast<std::uint32_t>(entry.role));
        append<4>(bytes, static_cast<std::uint32_t>(entry.value_class));
        append<4>(bytes, static_cast<std::uint32_t>(entry.access));
        append<8>(bytes, std::bit_cast<std::uint64_t>(entry.uid));
        append<8>(bytes, entry.slot);
        append<8>(bytes, entry.bytes);
        append<8>(bytes, entry.alignment);
    }
    return bytes;
}
Bytes wire(const Execute& value) {
    auto bytes = wire(Prepare{value.handle, value.metadata, value.plan, value.variant});
    append<8>(bytes, value.plan_generation);
    append<8>(bytes, value.variant_generation);
    append<4>(bytes, value.bindings.size());
    for (const auto& entry : value.bindings) {
        append<4>(bytes, entry.index);
        append<4>(bytes, static_cast<std::uint32_t>(entry.value_class));
        append<8>(bytes, entry.slot);
        append<8>(bytes, entry.allocation);
        append<8>(bytes, entry.offset);
        blob(bytes, entry.host_bytes);
    }
    return bytes;
}

bool write(ExecutionPayloadWriter& output, const Context& value) {
    return write_execution_dnn_backend_context(output, value);
}
bool write(ExecutionPayloadWriter& output, const Prepare& value) {
    return write_execution_dnn_backend_execute_prepare(output, value);
}
bool write(ExecutionPayloadWriter& output, const Preparation& value) {
    return write_execution_dnn_backend_execute_preparation(output, value);
}
bool write(ExecutionPayloadWriter& output, const Execute& value) {
    return write_execution_dnn_backend_execute(output, value);
}
bool read(ExecutionPayloadReader& input, Context& value) {
    return read_execution_dnn_backend_context(input, value);
}
bool read(ExecutionPayloadReader& input, Prepare& value) {
    return read_execution_dnn_backend_execute_prepare(input, value);
}
bool read(ExecutionPayloadReader& input, Preparation& value) {
    return read_execution_dnn_backend_execute_preparation(input, value);
}
bool read(ExecutionPayloadReader& input, Execute& value) {
    return read_execution_dnn_backend_execute(input, value);
}

template <typename Value>
Value sentinel() {
    if constexpr (std::is_same_v<Value, Context>) {
        return {UINT64_MAX,
                0x12345678,
                ExecutionOpcode::kDnnBackendGet,
                {std::byte{0x9a}, std::byte{0xbc}}};
    } else if constexpr (std::is_same_v<Value, Prepare>) {
        return {UINT64_MAX, 0x12345678, 0xabcdef, UINT64_C(0xfedcba9876543210)};
    } else if constexpr (std::is_same_v<Value, Preparation>) {
        return {0xfedcba,
                UINT64_MAX,
                {{19, Role::kTensor, Class::kDevice, Access::kWrite, INT64_MIN, 77, 41, 24}}};
    } else {
        static_assert(std::is_same_v<Value, Execute>);
        return {UINT64_MAX,
                0x12345678,
                0xabcdef,
                0xfedcba,
                17,
                31,
                {{27, Class::kByValue, UINT64_MAX, 0, 0, {std::byte{0xaa}, std::byte{0xbb}}}}};
    }
}

void same_reader(ExecutionPayloadReader actual, ExecutionPayloadReader expected) {
    for (;;) {
        std::uint32_t left = 0x12345678, right = 0x12345678;
        const auto has_left = actual.read_u32(left);
        const auto has_right = expected.read_u32(right);
        expect(has_left == has_right && left == right && actual.empty() == expected.empty(),
               "failed decoding preserves the reader cursor as well as output");
        if (!has_left) break;
    }
}

template <typename Value>
void reject_wire(std::span<const std::byte> bytes) {
    Bytes framed;
    append<4>(framed, 0x5a5aa5a5);
    framed.insert(framed.end(), bytes.begin(), bytes.end());
    ExecutionPayloadReader reader(framed);
    std::uint32_t prefix{};
    expect(reader.read_u32(prefix) && prefix == 0x5a5aa5a5, "consume unrelated prefix");
    const auto before_reader = reader;
    auto value = sentinel<Value>();
    const auto before = wire(value);
    expect(!read(reader, value), "malformed or incomplete packet rejects");
    expect(wire(value) == before, "failed decoding preserves every destination field");
    same_reader(reader, before_reader);
}

template <typename Value>
void reject_value(const Value& value) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(0x12345678), "seed writer with unrelated prefix");
    expect(!write(writer, value), "invalid structured input rejects");
    const Bytes prefix{std::byte{0x78}, std::byte{0x56}, std::byte{0x34}, std::byte{0x12}};
    expect(writer.take() == prefix, "invalid input rejects before any field is written");
    reject_wire<Value>(wire(value));
}

template <typename Value>
void valid(const Value& original, bool all_short = true) {
    const auto expected = wire(original);
    ExecutionPayloadWriter output;
    expect(write(output, original) && output.take() == expected,
           "encoded fields exactly match independent little-endian wire");
    auto decoded = sentinel<Value>();
    ExecutionPayloadReader input(expected);
    expect(read(input, decoded) && input.empty() && wire(decoded) == expected,
           "independent wire decodes without normalization or lost ordering");
    ExecutionPayloadWriter exact(expected.size());
    expect(write(exact, original) && exact.take() == expected, "exact writer bound succeeds");
    if (all_short) {
        for (std::size_t size = 0; size < expected.size(); ++size) {
            reject_wire<Value>(std::span{expected}.first(size));
            ExecutionPayloadWriter small(size + 4);
            expect(small.write_u32(0x12345678) && !write(small, original) &&
                       small.take() == Bytes{std::byte{0x78}, std::byte{0x56}, std::byte{0x34},
                                             std::byte{0x12}},
                   "every insufficient writer bound preserves its prefix");
        }
    } else {
        for (const auto size : {std::size_t{0}, expected.size() / 2, expected.size() - 1})
            reject_wire<Value>(std::span{expected}.first(size));
    }
    Bytes framed;
    append<4>(framed, 0xaabbccdd);
    framed.insert(framed.end(), expected.begin(), expected.end());
    append<4>(framed, 0x12345678);
    ExecutionPayloadReader composed(framed);
    std::uint32_t prefix{}, suffix{};
    expect(composed.read_u32(prefix) && prefix == 0xaabbccdd && read(composed, decoded) &&
               wire(decoded) == expected && !composed.empty() && composed.read_u32(suffix) &&
               suffix == 0x12345678 && composed.empty(),
           "successful decoder consumes only its fields, leaving enclosing trailing check");
}

Requirement device_requirement() {
    return {0, Role::kTensor, Class::kDevice, Access::kReadWrite, INT64_MIN, UINT64_MAX, 48, 24};
}
Binding device_binding() {
    return {0, Class::kDevice, UINT64_MAX, 0x123456789abcdef, UINT64_MAX, {}};
}

void literals_and_layouts() {
    expect(kExecutionDnnBackendMaxTensorBindings == 100 && kExecutionDnnBackendMaxBindings == 101 &&
               kExecutionDnnBackendWorkspaceIndex == UINT32_MAX &&
               kExecutionDnnBackendMaxValueBytes == kHostLimit,
           "frozen structural limits are independently fixed");
    const Prepare request{UINT64_C(0x0706050403020100), UINT64_C(0x0f0e0d0c0b0a0908),
                          UINT64_C(0x1716151413121110), UINT64_C(0x1f1e1d1c1b1a1918)};
    Bytes literal;
    for (unsigned value = 0; value < 32; ++value) literal.push_back(static_cast<std::byte>(value));
    expect(wire(request) == literal, "prepare is exactly four ordered u64 fields, 32 bytes");
    valid(request);
    valid(Prepare{});  // IDs are not a structural substitute for service authorization.

    const Context context{1,
                          2,
                          ExecutionOpcode::kDnnBackendPrepare,
                          {std::byte{0xab}, std::byte{0x00}, std::byte{0xff}}};
    const Bytes context_literal{
        std::byte{1},    std::byte{0}, std::byte{0},   std::byte{0}, std::byte{0},   std::byte{0},
        std::byte{0},    std::byte{0}, std::byte{2},   std::byte{0}, std::byte{0},   std::byte{0},
        std::byte{0},    std::byte{0}, std::byte{0},   std::byte{0}, std::byte{122}, std::byte{0},
        std::byte{0},    std::byte{0}, std::byte{3},   std::byte{0}, std::byte{0},   std::byte{0},
        std::byte{0xab}, std::byte{0}, std::byte{0xff}};
    expect(wire(context) == context_literal,
           "context literal fixes IDs, operation and byte length");
    valid(context);

    const Preparation prepared{
        1, 2, {{0, Role::kTensor, Class::kDevice, Access::kRead, -2, 3, 0, 24}}};
    Bytes prepared_literal(68, std::byte{0});
    prepared_literal[0] = std::byte{1};
    prepared_literal[8] = std::byte{2};
    prepared_literal[16] = std::byte{1};
    prepared_literal[24] = std::byte{1};
    prepared_literal[28] = std::byte{1};
    prepared_literal[32] = std::byte{1};
    std::fill(prepared_literal.begin() + 36, prepared_literal.begin() + 44, std::byte{0xff});
    prepared_literal[36] = std::byte{0xfe};
    prepared_literal[44] = std::byte{3};
    prepared_literal[60] = std::byte{24};
    expect(wire(prepared) == prepared_literal,
           "preparation is 20 + 48N, including signed UID bits");
    valid(prepared);

    const Execute command{1, 2, 3, 4, 5, 6, {{0, Class::kIgnored, 0, 0, 0, {}}}};
    Bytes command_literal(88, std::byte{0});
    for (std::size_t index = 0; index < 6; ++index)
        command_literal[index * 8] = static_cast<std::byte>(index + 1);
    command_literal[48] = std::byte{1};
    expect(wire(command) == command_literal, "final request is 52 + 36N + host payload bytes");
    valid(command);
    auto mixed = command;
    mixed.bindings = {device_binding(),
                      {1, Class::kByValue, 8, 0, 0, context.payload},
                      {kWorkspace, Class::kDevice, 19, 21, 0, {}}};
    expect(wire(mixed).size() == 52 + 36 * 3 + 3, "each host blob adds only its actual byte count");
    valid(mixed);
}

void context_boundaries() {
    const std::array allowed{119U, 120U, 121U, 122U, 123U, 124U};
    for (std::uint32_t operation = 0; operation <= 130; ++operation) {
        Context value{0, 0, static_cast<ExecutionOpcode>(operation), {std::byte{0x34}}};
        if (std::ranges::find(allowed, operation) != allowed.end())
            valid(value);
        else
            reject_value(value);
    }
    reject_value(Context{1, 2, static_cast<ExecutionOpcode>(UINT32_MAX), {}});
    Context maximum{UINT64_MAX, UINT64_MAX, ExecutionOpcode::kDnnBackendSet,
                    Bytes(kHostLimit, std::byte{0xa5})};
    valid(maximum, false);
    maximum.payload.push_back(std::byte{0});
    reject_value(maximum);
    auto huge = wire(Context{1, 2, ExecutionOpcode::kDnnBackendGet, {}});
    std::fill(huge.begin() + 20, huge.end(), std::byte{0xff});
    reject_wire<Context>(huge);
}

void requirement_boundaries() {
    const Preparation base{1, UINT64_MAX, {device_requirement()}};
    valid(base);
    valid(Preparation{1, 2, {}});
    for (const auto access : {Access::kRead, Access::kWrite, Access::kReadWrite}) {
        auto value = base;
        value.requirements[0].access = access;
        value.requirements[0].bytes = 0;
        valid(value);
    }
    const auto reject = [&](const auto& change) {
        auto value = base;
        change(value.requirements[0]);
        reject_value(value);
    };
    for (const auto raw : {0U, 3U, UINT32_MAX})
        reject([&](auto& entry) { entry.role = static_cast<Role>(raw); });
    for (const auto raw : {3U, 4U, UINT32_MAX})
        reject([&](auto& entry) { entry.value_class = static_cast<Class>(raw); });
    for (const auto raw : {0U, 4U, UINT32_MAX})
        reject([&](auto& entry) { entry.access = static_cast<Access>(raw); });
    reject([](auto& entry) { entry.alignment = 0; });
    reject([](auto& entry) { entry.index = 100; });
    reject([](auto& entry) { entry.index = kWorkspace; });
    reject([](auto& entry) { entry.role = Role::kWorkspace; });

    auto ignored = base;
    ignored.requirements[0] = {0, Role::kTensor, Class::kIgnored, Access::kNone, -7, 3, 0, 0};
    valid(ignored);
    for (unsigned field = 0; field < 3; ++field) {
        auto value = ignored;
        auto& entry = value.requirements[0];
        if (field == 0) entry.bytes = 1;
        if (field == 1) entry.alignment = 1;
        if (field == 2) entry.access = Access::kRead;
        reject_value(value);
    }
    auto by_value = base;
    by_value.requirements[0].value_class = Class::kByValue;
    by_value.requirements[0].access = Access::kRead;
    valid(by_value);
    for (const auto access : {Access::kNone, Access::kWrite, Access::kReadWrite}) {
        auto value = by_value;
        value.requirements[0].access = access;
        reject_value(value);
    }
    auto workspace = base;
    workspace.requirements[0] = {
        kWorkspace, Role::kWorkspace, Class::kDevice, Access::kReadWrite, 0, 7, UINT64_MAX, 24};
    valid(workspace);
    workspace.requirements[0].uid = 1;
    reject_value(workspace);
    workspace.requirements[0].uid = 0;
    workspace.requirements[0].value_class = Class::kByValue;
    workspace.requirements[0].access = Access::kRead;
    reject_value(workspace);

    for (const auto value_class : {Class::kDevice, Class::kByValue}) {
        auto null_value = base;
        null_value.requirements[0].value_class = value_class;
        null_value.requirements[0].access = Access::kRead;
        null_value.requirements[0].slot = 0;
        null_value.requirements[0].bytes = UINT64_MAX;
        valid(null_value);  // Required extent does not fabricate a capture for native NULL.
    }
    for (unsigned field = 0; field < 2; ++field) {
        auto value = base;
        (field ? value.variant_generation : value.plan_generation) = 0;
        reject_value(value);
    }
    auto huge = wire(base);
    std::fill(huge.begin() + 16, huge.begin() + 20, std::byte{0xff});
    reject_wire<Preparation>(huge);
}

void binding_boundaries() {
    const Execute base{0, 0, 0, 0, 1, UINT64_MAX, {device_binding()}};
    valid(base);
    auto empty = base;
    empty.bindings.clear();
    valid(empty);
    for (const auto raw : {3U, 4U, UINT32_MAX}) {
        auto value = base;
        value.bindings[0].value_class = static_cast<Class>(raw);
        reject_value(value);
    }
    auto missing_allocation = base;
    missing_allocation.bindings[0].allocation = 0;
    reject_value(missing_allocation);
    auto has_device_bytes = base;
    has_device_bytes.bindings[0].host_bytes = {std::byte{1}};
    reject_value(has_device_bytes);
    for (const auto value_class : {Class::kIgnored, Class::kDevice, Class::kByValue}) {
        auto null_value = base;
        null_value.bindings[0] = {0, value_class, 0, 0, 0, {}};
        valid(null_value);
        for (unsigned field = 0; field < 3; ++field) {
            auto malformed = null_value;
            auto& entry = malformed.bindings[0];
            if (field == 0) entry.allocation = 1;
            if (field == 1) entry.offset = 1;
            if (field == 2) entry.host_bytes = {std::byte{0}};
            reject_value(malformed);
        }
    }
    for (const auto value_class : {Class::kIgnored, Class::kByValue}) {
        auto value = base;
        value.bindings[0] = {0, value_class, 7, 0, 0, {}};
        valid(value);  // Exact nonnull by-value length is checked against preparation later.
        for (unsigned field = 0; field < 2; ++field) {
            auto malformed = value;
            (field ? malformed.bindings[0].offset : malformed.bindings[0].allocation) = 1;
            reject_value(malformed);
        }
        if (value_class == Class::kIgnored) {
            value.bindings[0].host_bytes = {std::byte{0}};
            reject_value(value);
        }
    }
    auto workspace = base;
    workspace.bindings[0] = {kWorkspace, Class::kDevice, 4, 5, 0, {}};
    valid(workspace);
    for (const auto slot : {UINT64_C(0), UINT64_C(7)}) {
        workspace.bindings[0] = {kWorkspace, Class::kByValue, slot, 0, 0, {}};
        reject_value(workspace);
    }
    for (unsigned field = 0; field < 2; ++field) {
        auto value = base;
        (field ? value.variant_generation : value.plan_generation) = 0;
        reject_value(value);
    }
    auto huge_count = wire(base);
    std::fill(huge_count.begin() + 48, huge_count.begin() + 52, std::byte{0xff});
    reject_wire<Execute>(huge_count);
    auto huge_blob = wire(base);
    std::fill(huge_blob.begin() + 84, huge_blob.begin() + 88, std::byte{0xff});
    reject_wire<Execute>(huge_blob);
}

void ordered_limits_and_aggregates() {
    Preparation preparation{11, 22, {}};
    Execute execution{1, 2, 3, 4, 11, 22, {}};
    for (std::uint32_t index = 0; index < 100; ++index) {
        preparation.requirements.push_back(
            {index, Role::kTensor, Class::kDevice, Access::kRead, -9, index + 1, 0, 24});
        execution.bindings.push_back({index, Class::kDevice, index + 1, 1, 0, {}});
    }
    // Workspace may occur between tensor entries. Duplicate UIDs and slot values are legal.
    preparation.requirements.insert(
        preparation.requirements.begin() + 37,
        {kWorkspace, Role::kWorkspace, Class::kIgnored, Access::kNone, 0, 0, 0, 0});
    execution.bindings.insert(execution.bindings.begin() + 37,
                              {kWorkspace, Class::kIgnored, 0, 0, 0, {}});
    valid(preparation, false);
    valid(execution, false);
    for (unsigned control = 0; control < 4; ++control) {
        auto bad_preparation = preparation;
        auto bad_execution = execution;
        if (control == 0) {
            bad_preparation.requirements.push_back(bad_preparation.requirements.back());
            bad_execution.bindings.push_back(bad_execution.bindings.back());
        } else if (control == 1) {
            bad_preparation.requirements[1].index = 0;
            bad_execution.bindings[1].index = 0;
        } else if (control == 2) {
            std::swap(bad_preparation.requirements[1], bad_preparation.requirements[2]);
            std::swap(bad_execution.bindings[1], bad_execution.bindings[2]);
        } else {
            bad_preparation.requirements[1] = bad_preparation.requirements[37];
            bad_execution.bindings[1] = bad_execution.bindings[37];
        }
        reject_value(bad_preparation);
        reject_value(bad_execution);
    }
    auto invalid_index = execution;
    invalid_index.bindings.back().index = 100;
    reject_value(invalid_index);
    // Sparse but ordered indices are structural; final complete correspondence is server-owned.
    valid(Preparation{1,
                      2,
                      {{3, Role::kTensor, Class::kIgnored, Access::kNone, 4, 8, 0, 0},
                       {99, Role::kTensor, Class::kIgnored, Access::kNone, 4, 8, 0, 0}}});

    Preparation captures{
        1,
        2,
        {{0, Role::kTensor, Class::kByValue, Access::kRead, 1, 7, kHostLimit - 1, 24},
         {1, Role::kTensor, Class::kByValue, Access::kRead, 1, 7, 1, 1}}};
    valid(captures);
    captures.requirements[1].bytes = 2;
    reject_value(captures);
    captures.requirements[0].slot = 0;
    captures.requirements[0].bytes = UINT64_MAX;
    valid(captures);
    captures.requirements[0].slot = 1;
    reject_value(captures);

    Execute captured{1,
                     2,
                     3,
                     4,
                     5,
                     6,
                     {{0, Class::kByValue, 7, 0, 0, Bytes(kHostLimit - 1, std::byte{0xa5})},
                      {1, Class::kByValue, 7, 0, 0, {std::byte{0xff}}}}};
    valid(captured, false);
    captured.bindings[1].host_bytes.push_back(std::byte{0});
    reject_value(captured);
    captured.bindings.resize(1);
    captured.bindings[0].host_bytes.resize(kHostLimit);
    valid(captured, false);
    captured.bindings[0].host_bytes.push_back(std::byte{0});
    reject_value(captured);
}

void reference_output_directions() {
    using Direction = ExecutionDnnBackendAttributeDirection;
    using Value = ExecutionDnnBackendAttributePreparation;
    const std::array all{Direction::kNone,         Direction::kInput,
                         Direction::kScalarOutput, Direction::kDescriptorOutput,
                         Direction::kHandleOutput, Direction::kPointerOutput};
    for (std::size_t index = 0; index < all.size(); ++index)
        expect(static_cast<std::uint32_t>(all[index]) == index,
               "reference output directions append without renumbering old directions");
    const auto bytes = [](const Value& value) {
        Bytes result;
        append<8>(result, value.generation);
        append<4>(result, static_cast<std::uint32_t>(value.direction));
        append<4>(result, value.input_elements);
        append<4>(result, value.output_elements);
        append<4>(result, value.count_bytes);
        append<4>(result, value.output_bytes);
        return result;
    };
    const auto reject = [&](const Value& value) {
        ExecutionPayloadWriter output;
        expect(output.write_u32(0x12345678), "seed attribute preparation writer");
        expect(!write_execution_dnn_backend_preparation(output, value) &&
                   output.take() ==
                       Bytes{std::byte{0x78}, std::byte{0x56}, std::byte{0x34}, std::byte{0x12}},
               "invalid reference direction preparation writes no partial fields");
        const auto packet = bytes(value);
        ExecutionPayloadReader reader(packet);
        Value decoded{77, Direction::kInput, 19, 0, 8, 0};
        const auto original = bytes(decoded);
        expect(
            !read_execution_dnn_backend_preparation(reader, decoded) && bytes(decoded) == original,
            "invalid reference direction preparation preserves output");
    };
    for (const auto direction : {Direction::kHandleOutput, Direction::kPointerOutput}) {
        for (const auto elements : {0U, 1U, 65536U}) {
            for (const auto count_bytes : {0U, 1U, 8U}) {
                const Value value{UINT64_MAX, direction, 0, elements, count_bytes, 0};
                const auto packet = bytes(value);
                expect(packet.size() == 28, "appended direction preserves 28-byte preparation");
                ExecutionPayloadWriter writer;
                expect(write_execution_dnn_backend_preparation(writer, value) &&
                           writer.take() == packet,
                       "reference output count is symbolic and has no scalar byte span");
                Value decoded;
                ExecutionPayloadReader reader(packet);
                expect(read_execution_dnn_backend_preparation(reader, decoded) && reader.empty() &&
                           bytes(decoded) == packet,
                       "reference output direction round trips exact fields");
                for (std::size_t length = 0; length < packet.size(); ++length) {
                    Value unchanged{43, Direction::kInput, 51, 0, 1, 0};
                    const auto before = bytes(unchanged);
                    ExecutionPayloadReader truncated{std::span{packet}.first(length)};
                    expect(!read_execution_dnn_backend_preparation(truncated, unchanged) &&
                               bytes(unchanged) == before,
                           "every short new-direction packet preserves output");
                    ExecutionPayloadWriter small(length);
                    expect(!write_execution_dnn_backend_preparation(small, value),
                           "every short new-direction writer rejects");
                }
            }
        }
        reject({1, direction, 1, 0, 8, 0});
        reject({1, direction, 0, 1, 8, 1});
        reject({1, direction, 0, 65537, 8, 0});
        reject({0, direction, 0, 1, 8, 0});
        for (const auto count : {2U, 4U, UINT32_MAX}) reject({1, direction, 0, 1, count, 0});
    }
    for (const auto raw : {6U, 7U, UINT32_MAX})
        reject({1, static_cast<Direction>(raw), 0, 0, 0, 0});
}

void opcode_compatibility() {
    const std::array old{
        ExecutionOpcode::kDnnAttnCompute,     ExecutionOpcode::kDnnBackendCreate,
        ExecutionOpcode::kDnnBackendDestroy,  ExecutionOpcode::kDnnBackendInitialize,
        ExecutionOpcode::kDnnBackendFinalize, ExecutionOpcode::kDnnBackendPrepare,
        ExecutionOpcode::kDnnBackendSet,      ExecutionOpcode::kDnnBackendGet};
    for (std::size_t index = 0; index < old.size(); ++index)
        expect(static_cast<std::uint32_t>(old[index]) == 117 + index,
               "previous append-only opcode numbers are unchanged");
    const std::array added{ExecutionOpcode::kDnnBackendContext,
                           ExecutionOpcode::kPrepareDnnBackendExecute,
                           ExecutionOpcode::kDnnBackendExecute};
    for (std::size_t index = 0; index < added.size(); ++index) {
        expect(static_cast<std::uint32_t>(added[index]) == 125 + index,
               "new opcode numeric ABI is appended exactly");
        for (const auto kind : {ExecutionMessageKind::kRequest, ExecutionMessageKind::kReply}) {
            ExecutionWireMessage message{{kind, added[index], 0, 7, 9, {11, 13}},
                                         {std::byte{0xde}, std::byte{0xad}}};
            Bytes literal;
            append<4>(literal, 0x31584547);
            append<4>(literal, kind == ExecutionMessageKind::kRequest ? 0x00010001 : 0x00020001);
            append<4>(literal, 125 + index);
            append<4>(literal, 2);
            for (const auto value : {7U, 9U, 11U, 13U}) append<8>(literal, value);
            literal.insert(literal.end(), message.payload.begin(), message.payload.end());
            const auto encoded = encode_execution_message(message);
            expect(encoded && *encoded == literal, "new opcode envelope matches literal header");
            const auto decoded = decode_execution_header(std::span{literal}.first(48));
            expect(decoded && decoded->operation == added[index] && decoded->kind == kind &&
                       decoded->session == 7 && decoded->sequence == 9 &&
                       decoded->incarnation == std::array<std::uint64_t, 2>{11, 13} &&
                       decoded->payload_bytes == 2,
                   "new opcode header decodes full identities and kind");
            for (std::size_t length = 0; length < 48; ++length)
                expect(!decode_execution_header(std::span{literal}.first(length)),
                       "short enclosing headers reject");
        }
    }
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kFunctionControl) == 128U,
           "public function control is appended after unchanged backend opcodes");
    ExecutionWireMessage function_control;
    function_control.header.operation = ExecutionOpcode::kFunctionControl;
    const auto encoded_function_control = encode_execution_message(function_control);
    expect(encoded_function_control.has_value(), "appended public function control encodes");
    const auto decoded_function_control = decode_execution_header(*encoded_function_control);
    expect(decoded_function_control.has_value() &&
               decoded_function_control->operation == ExecutionOpcode::kFunctionControl,
           "appended public function control decodes with unchanged identity");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kGeam) == 129U,
           "GEAM is appended without changing existing backend or function-control opcodes");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kPointerMetadata) == 130U,
           "pointer metadata is appended without changing prior opcodes");
    ExecutionWireMessage pointer_metadata;
    pointer_metadata.header.operation = ExecutionOpcode::kPointerMetadata;
    expect(encode_execution_message(pointer_metadata).has_value(),
           "new metadata opcode is valid general framing, not a backend execution operation");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kNcclCall) == 131U,
           "NCCL is appended without changing prior opcodes");
    ExecutionWireMessage nccl;
    nccl.header.operation = ExecutionOpcode::kNcclCall;
    const auto encoded_nccl = encode_execution_message(nccl);
    expect(encoded_nccl.has_value(), "appended NCCL opcode encodes");
    const auto decoded_nccl = decode_execution_header(*encoded_nccl);
    expect(decoded_nccl && decoded_nccl->operation == ExecutionOpcode::kNcclCall,
           "appended NCCL opcode decodes without changing its identity");
    const auto first_invalid = static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute) + 1U;
    for (const auto raw : {0U, first_invalid, UINT32_MAX}) {
        ExecutionWireMessage invalid;
        invalid.header.operation = static_cast<ExecutionOpcode>(raw);
        expect(!encode_execution_message(invalid), "out-of-range opcode cannot be encoded");
    }
}
}  // namespace

int main() {
    try {
        literals_and_layouts();
        context_boundaries();
        requirement_boundaries();
        binding_boundaries();
        ordered_limits_and_aggregates();
        reference_output_directions();
        opcode_compatibility();
        std::cout << "Backend execution protocol tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
