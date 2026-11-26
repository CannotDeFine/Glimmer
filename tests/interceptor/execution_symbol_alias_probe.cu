#include <cuda_runtime.h>

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// SDK-semantics diagnostic only. The ordinary numerical symbol application is
// separate and has no callback interposition or modified oracle.
extern "C" void glimmer_symbol_alias_second();
__device__ unsigned int alias_first_value = 41;
__device__ unsigned int alias_extra_first_value = 97;
__global__ void alias_first_kernel(unsigned int* output) {
    *output = 41;
}
__global__ void alias_extra_first_kernel(unsigned int* output) {
    *output = 97;
}
namespace {
using RegisterVariable = void (*)(void**, char*, char*, const char*, int, std::size_t, int, int);
using RegisterFunction = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                  dim3*, dim3*, int*);
struct Binding {
    void** handle{};
    char* device{};
    const char* name{};
    int external{}, constant{}, global{};
    std::size_t bytes{};
};
std::array<Binding, 3> bindings;
std::array<char, 6> alias_identities{};
std::array<void**, 2> retired_handles{};
struct FunctionBinding {
    void** handle{};
    char* device{};
    const char* name{};
};
std::array<FunctionBinding, 3> functions;
std::array<char, 6> function_identities{};
RegisterVariable original() {
    static const auto value =
        reinterpret_cast<RegisterVariable>(::dlsym(RTLD_NEXT, "__cudaRegisterVar"));
    if (!value) std::abort();
    return value;
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
void associate(std::size_t index, std::size_t identity) {
    const auto& binding = bindings.at(index);
    if (!binding.handle) throw std::runtime_error("missing original compiler registration");
    original()(binding.handle, &alias_identities.at(identity), binding.device, binding.name,
               binding.external, binding.bytes, binding.constant, binding.global);
}
unsigned int read(std::size_t identity) {
    unsigned int value{};
    using Copy =
        cudaError_t(CUDARTAPI*)(void*, const void*, std::size_t, std::size_t, cudaMemcpyKind);
    const auto copy = static_cast<Copy>(&cudaMemcpyFromSymbol);
    check(copy(&value, &alias_identities.at(identity), sizeof(value), 0, cudaMemcpyDeviceToHost),
          "read associated alias");
    return value;
}
RegisterFunction original_function() {
    static const auto value =
        reinterpret_cast<RegisterFunction>(::dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
    if (!value) std::abort();
    return value;
}
void associate_function(std::size_t function, std::size_t identity) {
    const auto& binding = functions.at(function);
    if (!binding.handle) throw std::runtime_error("missing original compiler kernel registration");
    original_function()(binding.handle, &function_identities.at(identity), binding.device,
                        binding.name, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
}
cudaError_t invoke(std::size_t identity, unsigned int* device, unsigned int* output) {
    void* parameters[]{&device};
    const auto launched = cudaLaunchKernel(&function_identities.at(identity), dim3{1}, dim3{1},
                                           parameters, 0, nullptr);
    return launched == cudaSuccess
               ? cudaMemcpy(output, device, sizeof(*output), cudaMemcpyDeviceToHost)
               : launched;
}
}  // namespace
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters): exact original compiler callback ABI.
extern "C" void __cudaRegisterVar(void** handle, char* host, char* device, const char* name,
                                  int external, std::size_t bytes, int constant, int global) {
    if (std::strcmp(name, "alias_first_value") == 0)
        bindings[0] = {handle, device, name, external, constant, global, bytes};
    else if (std::strcmp(name, "alias_second_value") == 0)
        bindings[1] = {handle, device, name, external, constant, global, bytes};
    else if (std::strcmp(name, "alias_extra_first_value") == 0)
        bindings[2] = {handle, device, name, external, constant, global, bytes};
    original()(handle, host, device, name, external, bytes, constant, global);
}
extern "C" void __cudaUnregisterFatBinary(void** handle) {
    for (auto retired : retired_handles)
        if (retired == handle) return;
    using Unregister = void (*)(void**);
    static const auto native =
        reinterpret_cast<Unregister>(::dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary"));
    if (!native) std::abort();
    native(handle);
    for (auto& retired : retired_handles)
        if (!retired) {
            retired = handle;
            return;
        }
    std::abort();
}
extern "C" void __cudaRegisterFunction(void** handle, const char* host, char* device,
                                       const char* name, int limit, uint3* tid, uint3* bid,
                                       dim3* bdim, dim3* gdim, int* warp) {
    if (std::strstr(device, "alias_extra_first_kernel"))
        functions[2] = {handle, device, name};
    else if (std::strstr(device, "alias_first_kernel"))
        functions[0] = {handle, device, name};
    else if (std::strstr(device, "alias_second_kernel"))
        functions[1] = {handle, device, name};
    original_function()(handle, host, device, name, limit, tid, bid, bdim, gdim, warp);
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming,
// bugprone-easily-swappable-parameters)
int main() {
    try {
        glimmer_symbol_alias_second();
        if (bindings[0].handle == bindings[1].handle)
            throw std::runtime_error("probe requires two distinct original module registrations");
        // All registrations precede the first CUDA lookup, as real compiler
        // constructors do. A lazy lookup cache must not choose this experiment's
        // registration order for us.
        associate(0, 0);
        associate(1, 0);
        associate(1, 1);
        associate(0, 1);
        associate(0, 2);
        associate(1, 2);
        associate(0, 2);
        associate(0, 3);
        associate(2, 3);
        associate(2, 4);
        associate(0, 4);
        associate(0, 5);
        associate(1, 5);
        for (std::size_t index = 0; index < 3; ++index)
            if (!functions[index].handle)
                throw std::runtime_error("incomplete kernel registration probe");
        associate_function(0, 0);
        associate_function(1, 0);
        associate_function(1, 1);
        associate_function(0, 1);
        associate_function(0, 2);
        associate_function(1, 2);
        associate_function(0, 2);
        associate_function(0, 3);
        associate_function(2, 3);
        associate_function(2, 4);
        associate_function(0, 4);
        associate_function(0, 5);
        associate_function(1, 5);
        const auto first = read(0);
        const auto second = read(1);
        const auto repeated = read(2);
        const auto within_first = read(3), within_reverse = read(4);
        unsigned int* function_output{};
        check(cudaMalloc(reinterpret_cast<void**>(&function_output), sizeof(*function_output)),
              "kernel alias output");
        std::array<unsigned int, 6> function_values{};
        for (std::size_t index = 0; index < 5; ++index)
            check(invoke(index, function_output, &function_values[index]),
                  "kernel alias selection");
        __cudaUnregisterFatBinary(bindings[0].handle);
        using Copy =
            cudaError_t(CUDARTAPI*)(void*, const void*, std::size_t, std::size_t, cudaMemcpyKind);
        const auto copy = static_cast<Copy>(&cudaMemcpyFromSymbol);
        std::array<unsigned int, 6> after{};
        std::array<int, 6> statuses{};
        for (std::size_t index = 0; index < after.size(); ++index) {
            (void)cudaGetLastError();
            statuses[index] =
                static_cast<int>(copy(&after[index], &alias_identities[index], sizeof(after[index]),
                                      0, cudaMemcpyDeviceToHost));
        }
        std::array<int, 6> function_retired_statuses{};
        std::array<unsigned int, 6> function_retired_values{};
        for (std::size_t index = 0; index < function_values.size(); ++index) {
            (void)cudaGetLastError();
            function_retired_statuses[index] =
                static_cast<int>(invoke(index, function_output, &function_retired_values[index]));
        }
        check(cudaFree(function_output), "retired kernel alias output cleanup");
        (void)cudaGetLastError();
        check(cudaDeviceReset(), "probe cleanup");
        std::array<unsigned int, 6> reset_values{};
        std::array<int, 6> reset_statuses{};
        check(cudaMalloc(reinterpret_cast<void**>(&function_output), sizeof(*function_output)),
              "fresh kernel alias output");
        for (std::size_t index = 0; index < after.size(); ++index) {
            (void)cudaGetLastError();
            reset_statuses[index] =
                static_cast<int>(copy(&reset_values[index], &alias_identities[index],
                                      sizeof(reset_values[index]), 0, cudaMemcpyDeviceToHost));
        }
        std::array<int, 6> function_reset_statuses{};
        std::array<unsigned int, 6> function_reset_values{};
        for (std::size_t index = 0; index < function_values.size(); ++index) {
            (void)cudaGetLastError();
            function_reset_statuses[index] =
                static_cast<int>(invoke(index, function_output, &function_reset_values[index]));
        }
        (void)cudaGetLastError();
        check(cudaFree(function_output), "fresh kernel alias output cleanup");
        (void)cudaGetLastError();
        check(cudaDeviceReset(), "probe reset cleanup");
        std::cout << "Original SDK kernel alias:";
        for (auto value : function_values) std::cout << ' ' << value;
        std::cout << "; retired:";
        for (std::size_t index = 0; index < function_values.size(); ++index)
            std::cout << ' ' << function_retired_statuses[index] << ':'
                      << function_retired_values[index];
        std::cout << "; reset:";
        for (std::size_t index = 0; index < function_values.size(); ++index)
            std::cout << ' ' << function_reset_statuses[index] << ':'
                      << function_reset_values[index];
        std::cout << '\n';
        std::cout << "Original SDK variable alias lifecycle: within_module=" << within_first << ','
                  << within_reverse << "; after_reset_statuses=" << reset_statuses[0] << ','
                  << reset_statuses[1] << ',' << reset_statuses[2] << ',' << reset_statuses[3]
                  << ',' << reset_statuses[4] << "; after_reset_values=" << reset_values[0] << ','
                  << reset_values[1] << ',' << reset_values[2] << ',' << reset_values[3] << ','
                  << reset_values[4] << '\n';
        std::cout << "Original SDK variable alias untouched: after_retire=" << statuses[5] << ','
                  << after[5] << "; after_reset=" << reset_statuses[5] << ',' << reset_values[5]
                  << '\n';
        std::cout << "Original SDK variable alias: first=" << first << "; second=" << second
                  << "; repeated=" << repeated << "; distinct_modules=true"
                  << "; after_retire_statuses=" << statuses[0] << ',' << statuses[1] << ','
                  << statuses[2] << "; after_retire_values=" << after[0] << ',' << after[1] << ','
                  << after[2] << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
