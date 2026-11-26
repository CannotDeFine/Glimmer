#pragma once

#include <cudnn.h>

#include <cstdint>

// Private SDK conversion policy shared by the optional frontend, backend and
// service composition. Wire integers remain SDK-independent. These constants
// prove enum representability, not native mutation/output or compute behavior.
#if !((CUDNN_MAJOR == 8 && CUDNN_VERSION == 8907) || (CUDNN_MAJOR == 9 && CUDNN_VERSION >= 91002))
#error "The owned cuDNN adapter requires the 8.9.7 or >=9.10.2 major-9 profile"
#endif

namespace glimmer::internal::cudnn_profile {
inline constexpr std::uint32_t kHeaderVersion = CUDNN_VERSION;
inline constexpr std::uint32_t kSdkMajor = CUDNN_MAJOR;

constexpr cudnnStatus_t host_allocation_status() noexcept {
#if CUDNN_MAJOR == 8
    return CUDNN_STATUS_ALLOC_FAILED;
#else
    return CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
#endif
}
constexpr std::uint32_t status_enum_max() noexcept {
    return kSdkMajor == 8 ? 15U : 8191U;
}
constexpr std::uint32_t data_type_enum_max() noexcept {
    return kSdkMajor == 8 ? 15U : 31U;
}
constexpr std::uint32_t behavior_note_enum_max() noexcept {
    return kSdkMajor == 8 ? 3U : 7U;
}
constexpr bool status_is_representable(std::int32_t value) noexcept {
    return value >= 0 && static_cast<std::uint32_t>(value) <= status_enum_max();
}
constexpr cudnnStatus_t profile_mismatch_status() noexcept {
    // A policy error, never native provenance. Integer1 is representable in
    // both source ABIs; major9's normal errors may be invalid enum objects in8.
    return static_cast<cudnnStatus_t>(1);
}
constexpr bool known_metadata_rejection(cudnnStatus_t status) noexcept {
#if CUDNN_MAJOR == 8
    return status == CUDNN_STATUS_BAD_PARAM || status == CUDNN_STATUS_INVALID_VALUE ||
           status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_ARCH_MISMATCH;
#else
    const auto category = static_cast<int>(status) / 1000;
    return category == 2 || category == 3;
#endif
}
constexpr bool supports_backend_kind(std::uint32_t kind) noexcept {
    return kind <= (kSdkMajor == 8 ? 33U : 40U);
}
constexpr bool supports_backend_attribute(std::uint32_t attribute) noexcept {
    if constexpr (kSdkMajor == 8) {
        switch (attribute) {
            case 204:
            case 303:
            case 304:
            case 407:
            case 803:
            case 1307:
            case 2700:
            case 2701:
            case 2702:
                return false;
            default:
                break;
        }
    }
    // Existing typed schema validation still decides whether a row is known.
    return true;
}
}  // namespace glimmer::internal::cudnn_profile
