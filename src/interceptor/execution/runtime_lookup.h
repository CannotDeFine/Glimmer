#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>

// Exact compiler registration ABI, not application-owned C++ identifiers.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** CUDARTAPI __cudaRegisterFatBinary(void*);
extern "C" void CUDARTAPI __cudaRegisterFatBinaryEnd(void**);
extern "C" void CUDARTAPI __cudaUnregisterFatBinary(void**);
extern "C" void CUDARTAPI __cudaRegisterFunction(void**, const char*, char*, const char*, int,
                                                 uint3*, uint3*, dim3*, dim3*, int*);
extern "C" void CUDARTAPI __cudaRegisterVar(void**, char*, char*, const char*, int, std::size_t,
                                            int, int);
extern "C" void CUDARTAPI __cudaRegisterManagedVar(void**, void**, char*, const char*, int,
                                                   std::size_t, int, int);
extern "C" void CUDARTAPI __cudaRegisterTexture(void**, const struct textureReference*,
                                                const void**, const char*, int, int, int);
extern "C" void CUDARTAPI __cudaRegisterSurface(void**, const struct surfaceReference*,
                                                const void**, const char*, int, int);
extern "C" char CUDARTAPI __cudaInitModule(void**);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(const char*, void**,
                                                              unsigned long long,
                                                              cudaDriverEntryPointQueryResult*);
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char*, void**, unsigned int, unsigned long long, cudaDriverEntryPointQueryResult*);
namespace glimmer::interceptor::execution {
struct RuntimeEntryQuery {
    std::string_view symbol;
    unsigned int version{};
    unsigned long long flags{};
    bool is_versioned{}, is_ptds{}, has_symbol_status{};
    int symbol_status{cudaDriverEntryPointSymbolNotFound};
};
struct RuntimeEntryResult {
    cudaError_t status{cudaErrorInitializationError};
    void* address{};
    int symbol_status{cudaDriverEntryPointSymbolNotFound};
};
// Cold SDK metadata only. No frontend resource lock or native resource issue.
[[nodiscard]] RuntimeEntryResult runtime_entry(const RuntimeEntryQuery& query) noexcept;
}  // namespace glimmer::interceptor::execution
