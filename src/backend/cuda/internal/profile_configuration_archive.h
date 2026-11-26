#pragma once

#include "cache_archive.h"
#include "cuda_profile_cache.h"

#include <bit>

namespace glimmer::backend::cuda::internal {

inline void write_profile_configuration(ArchiveWriter& output,
                                        const CudaProfileContextConfiguration& configuration) {
    output.number(configuration.available);
    for (const auto value : configuration.values) output.number(value, 8);
}
inline CudaProfileContextConfiguration read_profile_configuration(ArchiveReader& input) {
    CudaProfileContextConfiguration result;
    result.available = static_cast<std::uint32_t>(input.number());
    for (auto& value : result.values) value = input.number(8);
    return result;
}

inline void write_function_configuration(ArchiveWriter& output,
                                         const CudaProfileFunctionConfiguration& configuration) {
    output.number(configuration.functions);
    output.number(configuration.cache_preference);
    for (std::size_t role = 0; role < configuration.available.size(); ++role) {
        output.number(configuration.available[role]);
        for (const auto value : configuration.values[role])
            output.number(std::bit_cast<std::uint32_t>(value));
    }
}
inline CudaProfileFunctionConfiguration read_function_configuration(ArchiveReader& input) {
    CudaProfileFunctionConfiguration result;
    result.functions = static_cast<std::uint32_t>(input.number());
    result.cache_preference = static_cast<std::uint32_t>(input.number());
    for (std::size_t role = 0; role < result.available.size(); ++role) {
        result.available[role] = static_cast<std::uint32_t>(input.number());
        for (auto& value : result.values[role])
            value = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(input.number()));
    }
    return result;
}

}  // namespace glimmer::backend::cuda::internal
