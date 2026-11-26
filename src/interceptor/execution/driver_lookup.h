#pragma once

#include <cuda.h>

#include <string_view>

namespace glimmer::interceptor::execution {

// Cold loader/SDK metadata only; never invoke a virtual resource operation or
// hold frontend resource exclusion. Native resolution bypasses public routing.
[[nodiscard]] void* native_driver_symbol(const char* name) noexcept;
// Runtime device-count metadata may run before the caller loaded a Driver DSO.
// Anchors the original provider and initializes metadata only, without a context.
[[nodiscard]] CUresult initialize_driver_metadata() noexcept;
// Pure loaded-ELF profile proof. Call before frontend resource exclusion; no
// vendor SDK call, dlopen, disk read or resource publication is permitted here.
[[nodiscard]] bool dnn_application_profile_matches() noexcept;
struct DriverEntryQuery {
    std::string_view symbol;
    int version{};
    cuuint64_t flags{};
    bool is_v2{}, has_symbol_status{};
    int symbol_status{CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND};
};
struct DriverEntryResult {
    CUresult status{CUDA_ERROR_NOT_INITIALIZED};
    void* address{};
    int symbol_status{CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND};
};
[[nodiscard]] DriverEntryResult driver_entry(const DriverEntryQuery& query) noexcept;

}  // namespace glimmer::interceptor::execution
