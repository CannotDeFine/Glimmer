#include <cudnn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Ordinary SDK metadata input only, with no explicit Driver/device API calls.
// Compile now; later compare native/owned observations under matched library
// identities. Querying a maximum known SM is not a current-device observation.
namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
constexpr int kPrefix = 0x12345678;
constexpr int kSentinel = -7919;
constexpr int kSuffix = 0x23456789;
constexpr std::array properties{MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL};
using Guarded = std::array<int, 3>;
Guarded guarded() {
    return {kPrefix, kSentinel, kSuffix};
}
void guards(const Guarded& values) {
    expect(values.front() == kPrefix && values.back() == kSuffix,
           "both complete host guard integers remain unchanged");
}
}  // namespace
int main() {
    try {
        std::array<std::size_t, 3> baseline{};
        std::array<int, 3> baseline_properties{};
        unsigned int scalar_calls{}, property_calls{};
        for (unsigned int repetition = 0; repetition != 4; ++repetition) {
            const std::array values{cudnnGetVersion(), cudnnGetCudartVersion(),
                                    cudnnGetMaxDeviceVersion()};
            scalar_calls += 3;
            expect(values[0] && values[1] && values[2],
                   "configured native library reports nonzero version observations");
            std::array<int, 3> parts{};
            for (std::size_t index = 0; index != properties.size(); ++index) {
                auto output = guarded();
                checked(cudnnGetProperty(properties[index], &output[1]),
                        "query original version component");
                ++property_calls;
                guards(output);
                parts[index] = output[1];
                expect(cudnnGetProperty(properties[index], nullptr) == CUDNN_STATUS_BAD_PARAM,
                       "configured SDK rejects a null Property output without a store");
                ++property_calls;
            }
            expect(parts[0] >= 9 && parts[1] >= 0 && parts[2] >= 0,
                   "configured major-nine-or-newer version components are nonnegative");
            const auto composed = static_cast<std::uint64_t>(parts[0]) * 10000 +
                                  static_cast<std::uint64_t>(parts[1]) * 100 +
                                  static_cast<std::uint64_t>(parts[2]);
            expect(composed == values[0],
                   "original components agree with the original full version, not build constants");
            auto invalid = guarded();
            // Selector 3 is within this unfixed C++ enum's representation but
            // is not one of its three named public properties.
            const auto unsupported = static_cast<libraryPropertyType>(3);
            const auto invalid_status = cudnnGetProperty(unsupported, &invalid[1]);
            ++property_calls;
            expect(invalid_status != CUDNN_STATUS_SUCCESS && invalid == guarded(),
                   "unsupported Property preserves every host output and guard byte");
            const auto null_status = cudnnGetProperty(unsupported, nullptr);
            ++property_calls;
            expect(null_status != CUDNN_STATUS_SUCCESS, "unsupported null Property is not success");
            if (repetition == 0) {
                baseline = values;
                baseline_properties = parts;
            } else {
                expect(values == baseline && parts == baseline_properties,
                       "repeated original library observations remain stable");
            }
            std::cout << "repetition=" << repetition << " version=" << values[0]
                      << " library_build_cudart_version=" << values[1]
                      << " library_max_known_sm_version=" << values[2] << " major=" << parts[0]
                      << " minor=" << parts[1] << " patch=" << parts[2]
                      << " invalid_property_status=" << static_cast<int>(invalid_status)
                      << " invalid_null_status=" << static_cast<int>(null_status) << '\n';
        }
        expect(scalar_calls == 12 && property_calls == 32, "complete four-query repetition matrix");
        std::cout << "complete scalar_calls=" << scalar_calls
                  << " property_calls=" << property_calls
                  << " explicit_device_calls=0 native_owned_comparison_required=true"
                     " full_device_qualified=false\n";
        expect(static_cast<bool>(std::cout), "complete library-information receipt is written");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
