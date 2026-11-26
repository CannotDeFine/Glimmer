#include <cudnn.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <dlfcn.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Unchanged ordinary SDK input for explicit native/owned metadata pairs. No
// Glimmer API, Driver call, GPU computation or scheduling assertion is involved.
namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("SDK status " + std::to_string(static_cast<int>(status)));
}
template <typename Function>
Function symbol(void* library, const char* name) {
    auto* address = ::dlsym(library, name);
    expect(address != nullptr, "resolve original public SDK entry");
    Function result{};
    static_assert(sizeof(result) == sizeof(address));
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
struct Api {
    decltype(&cudnnGetVersion) version;
    decltype(&cudnnCreateTensorDescriptor) create;
    decltype(&cudnnDestroyTensorDescriptor) destroy;
    decltype(&cudnnSetTensorNdDescriptor) set;
    decltype(&cudnnGetTensorNdDescriptor) get;
    decltype(&cudnnGetTensorSizeInBytes) size;
};
void exercise(const Api& api) {
    expect(api.version() == CUDNN_VERSION, "actual native version equals selected source headers");
    cudnnTensorDescriptor_t descriptor{};
    checked(api.create(&descriptor));
    expect(descriptor != nullptr, "SDK publishes one live descriptor");
    try {
        constexpr std::array dimensions{2, 3, 4, 5};
        constexpr std::array strides{60, 20, 5, 1};
        checked(api.set(descriptor, CUDNN_DATA_FLOAT, 4, dimensions.data(), strides.data()));
        std::array<int, 8> observed_dimensions{}, observed_strides{};
        observed_dimensions.fill(-7919);
        observed_strides.fill(-7919);
        int rank = -1;
        auto type = CUDNN_DATA_INT8;
        checked(api.get(descriptor, 6, &type, &rank, observed_dimensions.data() + 1,
                        observed_strides.data() + 1));
        expect(rank == 4 && type == CUDNN_DATA_FLOAT, "original rank and type are preserved");
        for (std::size_t index = 0; index < observed_dimensions.size(); ++index) {
            const bool is_value = index >= 1 && index <= 4;
            expect(observed_dimensions[index] == (is_value ? dimensions[index - 1] : -7919) &&
                       observed_strides[index] == (is_value ? strides[index - 1] : -7919),
                   "all dimensions, strides, unused capacity and outer guards match");
        }
        std::size_t bytes{};
        checked(api.size(descriptor, &bytes));
        expect(bytes == std::size_t{2} * 3U * 4U * 5U * sizeof(float),
               "complete tensor storage is exact");
    } catch (...) {
        checked(api.destroy(descriptor));
        throw;
    }
    checked(api.destroy(descriptor));
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "usage: test exact-sdk-library");
        void* library = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        expect(library != nullptr, "load selected original SDK library");
        exercise({cudnnGetVersion, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor,
                  cudnnSetTensorNdDescriptor, cudnnGetTensorNdDescriptor,
                  cudnnGetTensorSizeInBytes});
        exercise(
            {symbol<decltype(&cudnnGetVersion)>(library, "cudnnGetVersion"),
             symbol<decltype(&cudnnCreateTensorDescriptor)>(library, "cudnnCreateTensorDescriptor"),
             symbol<decltype(&cudnnDestroyTensorDescriptor)>(library,
                                                             "cudnnDestroyTensorDescriptor"),
             symbol<decltype(&cudnnSetTensorNdDescriptor)>(library, "cudnnSetTensorNdDescriptor"),
             symbol<decltype(&cudnnGetTensorNdDescriptor)>(library, "cudnnGetTensorNdDescriptor"),
             symbol<decltype(&cudnnGetTensorSizeInBytes)>(library, "cudnnGetTensorSizeInBytes")});
        expect(::dlclose(library) == 0, "retire explicit SDK lookup reference normally");
        std::cout << "profile_metadata_complete;version=" << CUDNN_VERSION
                  << ";routes=2;calls=12;tensor_elements=120;all_guards_preserved=1"
                     ";explicit_driver_calls=0;compute_qualified=0;full_goal_qualified=0\n";
        expect(static_cast<bool>(std::cout), "write complete metadata receipt");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#pragma GCC diagnostic pop
