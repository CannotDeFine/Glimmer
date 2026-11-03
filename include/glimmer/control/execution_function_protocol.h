#pragma once

#include "glimmer/control/execution_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace glimmer::control {

inline constexpr std::size_t kExecutionFunctionAttributesBytes = 76;
struct ExecutionFunctionAttributes {
    // sharedSizeBytes, constSizeBytes, localSizeBytes.
    std::array<std::uint64_t, 3> sizes{};
    // maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA,
    // maxDynamicSharedSizeBytes, preferredShmemCarveout, clusterDimMustBeSet,
    // requiredClusterWidth/Height/Depth, clusterSchedulingPolicyPreference,
    // nonPortableClusterSizeAllowed. No native padding or reserved fields.
    std::array<std::int32_t, 13> values{};
};

// Explicit little-endian fields. Failure preserves the writer's entire prefix.
[[nodiscard]] bool write_execution_function_attributes(
    ExecutionPayloadWriter& output, const ExecutionFunctionAttributes& value) noexcept;
// Failure preserves both reader cursor and destination. The enclosing message
// is responsible for rejecting trailing bytes; this codec permits composition.
[[nodiscard]] bool read_execution_function_attributes(ExecutionPayloadReader& input,
                                                      ExecutionFunctionAttributes& value) noexcept;

}  // namespace glimmer::control
