#include "dnn_library_info_fixture.h"

#include <bit>
#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_library_info;
using Kind = CudaDnnLibraryInfoKind;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.value,
           "policy rejection carries no native result or published output");
}
void scalar(const CudaDnnResult& result, std::size_t value) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               !result.original_status && result.value == value,
           "direct-return native value retains full width without invented SDK status");
}
void property(const CudaDnnResult& result, cudnnStatus_t status, std::uint32_t value = 0) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && result.value == value,
           "property preserves original status and publishes only successful requested int bits");
}
void ordinary() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch(), 8);
    const std::array kinds{Kind::kVersion, Kind::kCudartVersion, Kind::kMaxDeviceVersion};
    const std::array<std::size_t, 5> values{0, 1, 91002, (std::size_t{1} << 40) + 71,
                                            std::numeric_limits<std::size_t>::max()};
    for (std::size_t index = 0; index != kinds.size(); ++index) {
        for (const auto value : values) {
            fixture::control.values[index].store(value);
            scalar(registry.library_info(kinds[index]), value);
        }
    }
    for (const int value : {INT32_MIN, -1, 0, INT32_MAX}) {
        fixture::control.property_value.store(value);
        for (std::uint32_t selector = 0; selector != 3; ++selector) {
            property(registry.library_info(Kind::kProperty, selector, true), CUDNN_STATUS_SUCCESS,
                     std::bit_cast<std::uint32_t>(value));
            expect(fixture::control.property_selector.load() == selector,
                   "all public selectors reach native unchanged");
        }
    }
    property(registry.library_info(Kind::kProperty, 1, false), CUDNN_STATUS_BAD_PARAM);
    expect(fixture::control.null_outputs.load() == 1, "absent output is native nullptr");
    property(registry.library_info(Kind::kProperty, 3, true), CUDNN_STATUS_BAD_PARAM);
    fixture::control.property_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    property(registry.library_info(Kind::kProperty, 0, true), CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    fixture::control.property_status.store(CUDNN_STATUS_SUCCESS);
    property(registry.library_info(Kind::kProperty, 0, true), CUDNN_STATUS_SUCCESS, INT32_MAX);

    const auto before = fixture::calls();
    for (const auto kind : kinds) {
        rejected(registry.library_info(kind, 1, false), CUDA_ERROR_INVALID_VALUE);
        rejected(registry.library_info(kind, 0, true), CUDA_ERROR_INVALID_VALUE);
    }
    rejected(registry.library_info(static_cast<Kind>(0)), CUDA_ERROR_INVALID_VALUE);
    rejected(registry.library_info(static_cast<Kind>(5)), CUDA_ERROR_INVALID_VALUE);
    rejected(registry.library_info(Kind::kProperty, 4, true), CUDA_ERROR_INVALID_VALUE);
    rejected(registry.library_info(Kind::kProperty, UINT32_MAX, true), CUDA_ERROR_INVALID_VALUE);
    auto foreign =
        std::async(std::launch::async, [&] { return registry.library_info(Kind::kVersion); });
    rejected(foreign.get(), CUDA_ERROR_UNKNOWN);
    const auto child = ::fork();
    expect(child >= 0, "fork inherited registry authority");
    if (!child) {
        const auto result = registry.library_info(Kind::kProperty, 0, true);
        ::_exit(result.disposition == CudaDnnDisposition::kRejected &&
                        result.status == CUDA_ERROR_UNKNOWN && !result.original_status &&
                        !result.value && fixture::calls() == before
                    ? 0
                    : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               fixture::calls() == before,
           "foreign thread/fork and invalid requests never invoke callbacks");
    expect(registry.shutdown().original_status == CUDNN_STATUS_SUCCESS &&
               registry.snapshot().is_closed,
           "resource-free metadata owner closes");
    rejected(registry.library_info(Kind::kVersion), CUDA_ERROR_DEINITIALIZED);
    rejected(registry.library_info(Kind::kProperty, 0, true), CUDA_ERROR_DEINITIALIZED);
    expect(fixture::calls() == before && fixture::is_resource_free(),
           "closed owner never dispatches or creates CUDA/descriptor resources");
}
void missing_callbacks() {
    for (unsigned int missing = 1; missing != 5; ++missing) {
        fixture::reset();
        CudaDnnRegistry registry(fixture::dispatch(missing), 8);
        const auto kind = static_cast<Kind>(missing);
        rejected(registry.library_info(kind, 0, kind == Kind::kProperty), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture::calls(), "missing callback never claims native invocation");
        if (missing == 4)
            scalar(registry.library_info(Kind::kVersion), 91002);
        else
            property(registry.library_info(Kind::kProperty, 0, true), CUDNN_STATUS_SUCCESS, 9);
        expect(registry.shutdown().original_status == CUDNN_STATUS_SUCCESS &&
                   registry.snapshot().is_closed && fixture::is_resource_free(),
               "clean unsupported query leaves owner usable and exactly closable");
    }
}
}  // namespace
int main() {
    try {
        ordinary();
        missing_callbacks();
        std::cout << "Original library-info CPU registry checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
