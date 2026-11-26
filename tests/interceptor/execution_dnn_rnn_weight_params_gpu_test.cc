#include "rnn_lstm_oracle.h"

#include <cuda.h>
#include <cudnn.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Ordinary SDK input, compiled only unless a separate device run is authorized.
// Every logical parameter is discovered through original descriptors/addresses.
// The independent model has no knowledge of vendor offsets or opaque padding.
namespace {
namespace oracle = glimmer::tests::rnn_lstm_oracle;
constexpr std::size_t kGuard = 256, kMaxWeightBytes = std::size_t{64} * 1024;
constexpr std::size_t kMaxTemporaryBytes = std::size_t{64} * 1024 * 1024;
bool has_failure{};
unsigned int parameter_queries{}, forward_calls{}, data_calls{}, weight_calls{}, cases{};
unsigned int configuration{}, iteration{}, resource_outputs{}, resources_destroyed{};
unsigned int device_allocations{}, device_releases{}, live_libraries{};

void expect(bool condition, const char* message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    std::cout << "{\"kind\":\"status\",\"operation\":" << std::quoted(operation)
              << ",\"status\":" << static_cast<int>(status) << "}\n"
              << std::flush;
    if (static_cast<int>(status)) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Single-threaded test dispatch. Providers outlive every SDK resource and the
// cached pointers; this does not import a Glimmer API or change SDK call order.
class Library final {
   public:
    explicit Library(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        expect(handle_ != nullptr, "open selected SDK provider");
        ++live_libraries;
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() {
        if (!has_failure) {
            if (dlclose(handle_) != 0)
                has_failure = true;
            else
                --live_libraries;
        }
    }
    template <typename Function>
    Function get(const char* name) const {
        dlerror();
        void* address = dlsym(handle_, name);
        const char* error = dlerror();
        expect(address != nullptr && error == nullptr, "resolve selected SDK symbol");
        static_assert(sizeof(Function) == sizeof(address));
        Function function{};
        std::memcpy(&function, &address, sizeof(function));
        return function;
    }

   private:
    void* handle_{};
};
#define DNN_FUNCTIONS(F)             \
    F(cudnnGetVersion)               \
    F(cudnnCreate)                   \
    F(cudnnDestroy)                  \
    F(cudnnSetStream)                \
    F(cudnnCreateRNNDescriptor)      \
    F(cudnnDestroyRNNDescriptor)     \
    F(cudnnSetRNNDescriptor_v8)      \
    F(cudnnCreateRNNDataDescriptor)  \
    F(cudnnDestroyRNNDataDescriptor) \
    F(cudnnSetRNNDataDescriptor)     \
    F(cudnnCreateTensorDescriptor)   \
    F(cudnnDestroyTensorDescriptor)  \
    F(cudnnSetTensorNdDescriptor)    \
    F(cudnnGetTensorNdDescriptor)    \
    F(cudnnGetRNNWeightSpaceSize)    \
    F(cudnnGetRNNTempSpaceSizes)     \
    F(cudnnGetRNNWeightParams)       \
    F(cudnnRNNForward)               \
    F(cudnnRNNBackwardData_v8)       \
    F(cudnnRNNBackwardWeights_v8)
#define DRIVER_FUNCTIONS(F) \
    F(cuInit)               \
    F(cuDeviceGet)          \
    F(cuCtxCreate)          \
    F(cuCtxDestroy)         \
    F(cuCtxSynchronize)     \
    F(cuStreamCreate)       \
    F(cuStreamDestroy)      \
    F(cuMemAlloc)           \
    F(cuMemFree)            \
    F(cuMemcpyHtoD)         \
    F(cuMemcpyDtoH)
#define SYMBOL_NAME_IMPL(NAME) #NAME
#define SYMBOL_NAME(NAME) SYMBOL_NAME_IMPL(NAME)
struct Api {
// NOLINTNEXTLINE(bugprone-macro-parentheses): NAME declares a member.
#define DECLARE(NAME) decltype(&::NAME) NAME{&::NAME};
    DNN_FUNCTIONS(DECLARE)
    DRIVER_FUNCTIONS(DECLARE)
#undef DECLARE
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): distinct named SDK providers.
    void select(const Library& dnn, const Library& driver, bool cached) {
#define RESOLVE_DNN(NAME) NAME = cached ? dnn.get<decltype(&::NAME)>(SYMBOL_NAME(NAME)) : &::NAME;
        DNN_FUNCTIONS(RESOLVE_DNN)
#undef RESOLVE_DNN
#define RESOLVE_DRIVER(NAME) \
    NAME = cached ? driver.get<decltype(&::NAME)>(SYMBOL_NAME(NAME)) : &::NAME;
        DRIVER_FUNCTIONS(RESOLVE_DRIVER)
#undef RESOLVE_DRIVER
    }
    template <typename Function>
    static void provider(const char* name, Function function) {
        void* address{};
        static_assert(sizeof(function) == sizeof(address));
        std::memcpy(&address, &function, sizeof(address));
        Dl_info info{};
        expect(dladdr(address, &info) != 0 && info.dli_fname != nullptr,
               "identify actual SDK function provider");
        std::cout << "{\"kind\":\"provider\",\"symbol\":" << std::quoted(name)
                  << ",\"path\":" << std::quoted(info.dli_fname) << "}\n"
                  << std::flush;
    }
    void providers() const {
#define PROVIDER(NAME) provider(SYMBOL_NAME(NAME), NAME);
        DNN_FUNCTIONS(PROVIDER)
        DRIVER_FUNCTIONS(PROVIDER)
#undef PROVIDER
    }
} api;
#undef SYMBOL_NAME
#undef SYMBOL_NAME_IMPL
#undef DRIVER_FUNCTIONS
#undef DNN_FUNCTIONS

void hex(std::span<const std::byte> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 4096> encoded{};
    std::size_t cursor{};
    for (const auto byte : bytes) {
        const auto value = std::to_integer<unsigned int>(byte);
        encoded[cursor++] = digits[value >> 4];
        encoded[cursor++] = digits[value & 15];
        if (cursor == encoded.size()) {
            std::cout.write(encoded.data(), static_cast<std::streamsize>(cursor));
            cursor = 0;
        }
    }
    std::cout.write(encoded.data(), static_cast<std::streamsize>(cursor));
}
void mapped_providers() {
    std::ifstream maps("/proc/self/maps");
    expect(maps.is_open(), "open own provider mappings");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("libcudnn") != std::string::npos ||
            line.find("libcuda.so") != std::string::npos ||
            line.find("libglimmer_execution_interceptor") != std::string::npos)
            std::cout << "{\"kind\":\"mapping\",\"line\":" << std::quoted(line) << "}\n";
    }
    expect(maps.eof() && !maps.bad(), "read complete own provider mappings");
    std::cout.flush();
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    explicit Resource(decltype(Destroy) destroy) : destroy_(destroy) {}
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (value_ && !has_failure) {
            const auto status = destroy_(std::exchange(value_, {}));
            std::cout << "{\"kind\":\"resource_destroy\",\"status\":" << static_cast<int>(status)
                      << "}\n"
                      << std::flush;
            if (static_cast<int>(status))
                has_failure = true;
            else
                ++resources_destroyed;
        }
    }
    Handle* output() {
        ++resource_outputs;
        return &value_;
    }
    Handle get() const {
        return value_;
    }

   private:
    Handle value_{};
    decltype(Destroy) destroy_;
};
using Context = Resource<CUcontext, cuCtxDestroy>;
using Handle = Resource<cudnnHandle_t, cudnnDestroy>;
using Rnn = Resource<cudnnRNNDescriptor_t, cudnnDestroyRNNDescriptor>;
using Data = Resource<cudnnRNNDataDescriptor_t, cudnnDestroyRNNDataDescriptor>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Stream = Resource<CUstream, cuStreamDestroy>;

class Device final {
   public:
    Device(const char* name, std::size_t bytes)
        : name_(name), bytes_(bytes), initial_(bytes + 2 * kGuard, std::byte{0xa7}) {
        expect(bytes <= kMaxTemporaryBytes, "bound device input allocation");
        checked(api.cuMemAlloc(&base_, initial_.size()), "allocate guarded input");
        ++device_allocations;
        reset(std::vector<std::byte>(bytes));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() {
        if (base_ && !has_failure) {
            const auto status = api.cuMemFree(base_);
            std::cout << "{\"kind\":\"device_release\",\"configuration\":" << configuration
                      << ",\"name\":" << std::quoted(name_)
                      << ",\"status\":" << static_cast<int>(status) << "}\n"
                      << std::flush;
            if (status != CUDA_SUCCESS)
                has_failure = true;
            else
                ++device_releases;
        }
    }
    void* get() const {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): Driver device address, never a host pointer.
        return reinterpret_cast<void*>(base_ + kGuard);
    }
    const void* allocation_end() const {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): original device endpoint control.
        return reinterpret_cast<const void*>(base_ + initial_.size());
    }
    std::size_t size() const {
        return bytes_;
    }
    void reset(std::span<const std::byte> bytes) {
        expect(bytes.size() == bytes_, "complete input reset");
        std::copy(bytes.begin(), bytes.end(), initial_.begin() + kGuard);
        record("reset", initial_);
        checked(api.cuMemcpyHtoD(base_, initial_.data(), initial_.size()), "upload complete input");
    }
    template <typename T>
    void values(std::span<const double> values) {
        std::vector<T> encoded;
        encoded.reserve(values.size());
        for (const auto value : values) encoded.push_back(static_cast<T>(value));
        reset(std::as_bytes(std::span{encoded}));
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> bytes(initial_.size());
        const auto status = api.cuMemcpyDtoH(bytes.data(), base_, bytes.size());
        record("read", bytes);
        checked(status, "download complete guarded output");
        expect(std::equal(bytes.begin(), bytes.begin() + kGuard, initial_.begin()) &&
                   std::equal(bytes.end() - kGuard, bytes.end(), initial_.end() - kGuard),
               "every allocation prefix/suffix guard remains intact");
        return {bytes.begin() + kGuard, bytes.end() - kGuard};
    }
    void unchanged() const {
        const auto bytes = read();
        expect(std::equal(bytes.begin(), bytes.end(), initial_.begin() + kGuard),
               "complete native input payload is unchanged");
    }

   private:
    void record(const char* action, std::span<const std::byte> bytes) const {
        std::cout << "{\"kind\":\"buffer\",\"configuration\":" << configuration
                  << ",\"iteration\":" << iteration << ",\"action\":" << std::quoted(action)
                  << ",\"name\":" << std::quoted(name_) << ",\"base\":" << base_
                  << ",\"payload_bytes\":" << bytes_ << ",\"guard_bytes\":" << kGuard
                  << ",\"guarded_hex\":\"";
        hex(bytes);
        std::cout << "\"}\n" << std::flush;
    }
    const char* name_;
    CUdeviceptr base_{};
    std::size_t bytes_{};
    std::vector<std::byte> initial_;
};
struct Shape {
    cudnnDataType_t type{};
    int rank{};
    std::array<int, 3> dimensions{}, strides{};
};
Shape shape(cudnnTensorDescriptor_t descriptor) {
    struct Guarded {
        std::uint64_t before{0x123456789abcdef0};
        std::array<int, 8> values;
        std::uint64_t after{0xfedcba9876543210};
        Guarded() {
            values.fill(-8123);
        }
    } dimensions, strides;
    Shape result;
    const auto status = api.cudnnGetTensorNdDescriptor(
        descriptor, 3, &result.type, &result.rank, dimensions.values.data(), strides.values.data());
    std::cout << "{\"kind\":\"shape\",\"configuration\":" << configuration
              << ",\"query\":" << parameter_queries
              << ",\"descriptor\":" << reinterpret_cast<std::uintptr_t>(descriptor)
              << ",\"status\":" << static_cast<int>(status)
              << ",\"type\":" << static_cast<int>(result.type) << ",\"rank\":" << result.rank;
    for (const auto& named :
         {std::pair{"dimensions", &dimensions}, std::pair{"strides", &strides}}) {
        std::cout << ",\"" << named.first << "\":{\"before\":" << named.second->before
                  << ",\"after\":" << named.second->after << ",\"values\":[";
        for (std::size_t i = 0; i != named.second->values.size(); ++i)
            std::cout << (i ? "," : "") << named.second->values[i];
        std::cout << "]}";
    }
    std::cout << "}\n" << std::flush;
    checked(status, "observe authoritative parameter shape");
    expect(result.rank == 0 || result.rank == 3, "bounded original parameter rank");
    for (const auto* array : {&dimensions, &strides}) {
        expect(array->before == 0x123456789abcdef0 && array->after == 0xfedcba9876543210,
               "parameter getter preserves outer guards");
        for (std::size_t i = static_cast<std::size_t>(result.rank); i != array->values.size(); ++i)
            expect(array->values[i] == -8123, "parameter getter preserves unrequested tail");
    }
    std::copy_n(dimensions.values.begin(), 3, result.dimensions.begin());
    std::copy_n(strides.values.begin(), 3, result.strides.begin());
    return result;
}
struct Mapping {
    std::array<std::array<std::size_t, 4>, 8> matrices{};
    std::array<std::array<std::size_t, 2>, 8> biases{};
    bool operator==(const Mapping&) const = default;
};
void query(cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, std::size_t bytes, const void* weight,
           int linear, cudnnTensorDescriptor_t matrix, void** matrix_address,
           cudnnTensorDescriptor_t bias, void** bias_address) {
    const auto status = api.cudnnGetRNNWeightParams(handle, rnn, 0, bytes, weight, linear, matrix,
                                                    matrix_address, bias, bias_address);
    std::cout << "{\"kind\":\"query\",\"configuration\":" << configuration
              << ",\"query\":" << parameter_queries + 1 << ",\"linear\":" << linear
              << ",\"bytes\":" << bytes << ",\"base\":" << reinterpret_cast<std::uintptr_t>(weight)
              << ",\"status\":" << static_cast<int>(status)
              << ",\"matrix_descriptor\":" << reinterpret_cast<std::uintptr_t>(matrix)
              << ",\"bias_descriptor\":" << reinterpret_cast<std::uintptr_t>(bias)
              << ",\"matrix_output\":" << reinterpret_cast<std::uintptr_t>(matrix_address)
              << ",\"bias_output\":" << reinterpret_cast<std::uintptr_t>(bias_address)
              << ",\"matrix_address\":"
              << reinterpret_cast<std::uintptr_t>(matrix_address ? *matrix_address : nullptr)
              << ",\"bias_address\":"
              << reinterpret_cast<std::uintptr_t>(bias_address ? *bias_address : nullptr) << "}\n"
              << std::flush;
    checked(status, "query original RNN parameter identity");
    ++parameter_queries;
}
template <typename T>
Mapping mapping(cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, std::size_t bytes,
                const void* weight) {
    Tensor matrix(api.cudnnDestroyTensorDescriptor), bias(api.cudnnDestroyTensorDescriptor);
    checked(api.cudnnCreateTensorDescriptor(matrix.output()), "create parameter matrix output");
    checked(api.cudnnCreateTensorDescriptor(bias.output()), "create parameter bias output");
    std::vector<bool> observed(bytes);
    Mapping result;
    for (std::size_t linear = 0; linear != 8; ++linear) {
        std::array<void*, 2> addresses{};
        query(handle, rnn, bytes, weight, static_cast<int>(linear), matrix.get(), &addresses[0],
              bias.get(), &addresses[1]);
        for (std::size_t part = 0; part != 2; ++part) {
            const auto metadata = shape(part ? bias.get() : matrix.get());
            expect(metadata.rank == 3 &&
                       metadata.type == (sizeof(T) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT) &&
                       metadata.dimensions == std::array{1, 2, part ? 1 : 2} &&
                       std::ranges::all_of(metadata.strides, [](int value) { return value > 0; }),
                   "logical double-bias LSTM parameter shapes are complete and typed");
            const auto base = reinterpret_cast<std::uintptr_t>(weight);
            const auto address = reinterpret_cast<std::uintptr_t>(addresses[part]);
            expect(address >= base && address - base < bytes,
                   "parameter start is in the declared owned range or raw-offset interval");
            const auto offset = address - base;
            const auto columns = part ? std::size_t{1} : std::size_t{2};
            for (std::size_t row = 0; row != 2; ++row)
                for (std::size_t col = 0; col != columns; ++col) {
                    const auto element = row * static_cast<std::size_t>(metadata.strides[1]) +
                                         col * static_cast<std::size_t>(metadata.strides[2]);
                    expect(element <= (bytes - offset) / sizeof(T),
                           "bounded descriptor stride product");
                    const auto location = offset + element * sizeof(T);
                    expect(location <= bytes && sizeof(T) <= bytes - location &&
                               location % sizeof(T) == 0,
                           "each trainable scalar is completely inside the original weight space");
                    for (std::size_t i = 0; i != sizeof(T); ++i) {
                        expect(!observed[location + i], "trainable scalar mappings do not overlap");
                        observed[location + i] = true;
                    }
                    if (part)
                        result.biases[linear][row] = location;
                    else
                        result.matrices[linear][2 * row + col] = location;
                    std::cout << "{\"kind\":\"scalar_location\",\"configuration\":" << configuration
                              << ",\"query\":" << parameter_queries << ",\"linear\":" << linear
                              << ",\"part\":" << part << ",\"row\":" << row << ",\"column\":" << col
                              << ",\"offset\":" << location << "}\n"
                              << std::flush;
                }
        }
    }
    expect(std::ranges::count(observed, true) == static_cast<std::ptrdiff_t>(48 * sizeof(T)),
           "all 48 logical trainable parameters are discovered exactly once");
    return result;
}
void mapping_controls(cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, const Device& weight,
                      const Mapping& mapped) {
    Tensor matrix(api.cudnnDestroyTensorDescriptor), bias(api.cudnnDestroyTensorDescriptor),
        alias(api.cudnnDestroyTensorDescriptor);
    for (auto* tensor : {&matrix, &bias, &alias})
        checked(api.cudnnCreateTensorDescriptor(tensor->output()),
                "create optional mapping output");
    const auto expected = [&](std::size_t offset) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): compare original device-address bits.
        return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(weight.get()) + offset);
    };
    void* m{};
    void* b{};
    query(handle, rnn, weight.size(), weight.get(), 0, matrix.get(), &m, nullptr, nullptr);
    expect(m == expected(mapped.matrices[0][0]), "matrix-only original output");
    query(handle, rnn, weight.size(), weight.get(), 0, nullptr, nullptr, bias.get(), &b);
    expect(b == expected(mapped.biases[0][0]), "bias-only original output");
    query(handle, rnn, weight.size(), weight.get(), 0, alias.get(), &m, alias.get(), &b);
    expect(m == expected(mapped.matrices[0][0]) && b == expected(mapped.biases[0][0]) &&
               shape(alias.get()).dimensions == std::array{1, 2, 1},
           "aliased descriptor retains final bias shape without changing original matrix address");
    query(handle, rnn, weight.size(), weight.get(), 0, matrix.get(), &m, bias.get(), &m);
    expect(m == expected(mapped.biases[0][0]), "aliased host address output publishes bias last");
    // Projection is disabled. Both outputs are natively absent, independent of
    // capacity; a nonnull zero-byte allocation endpoint is not invented storage.
    for (const auto* pointer :
         std::array<const void*, 3>{weight.get(), weight.allocation_end(), nullptr}) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): deliberately nonnull output sentinel.
        m = b = reinterpret_cast<void*>(std::uintptr_t{0x9abc});
        query(handle, rnn, 0, pointer, 8, matrix.get(), &m, bias.get(), &b);
        expect(!m && !b && shape(matrix.get()).rank == 0 && shape(bias.get()).rank == 0,
               "absent projection/bias preserve null outputs at zero capacity");
    }
    query(handle, rnn, 0, nullptr, 8, nullptr, nullptr, nullptr, nullptr);
}
template <typename Function>
void parameter_elements(const Mapping& map, const oracle::Parameters& parameters,
                        Function&& function) {
    for (std::size_t linear = 0; linear != 8; ++linear) {
        for (std::size_t i = 0; i != 4; ++i)
            function(map.matrices[linear][i], parameters.matrices[linear][i]);
        for (std::size_t i = 0; i != 2; ++i)
            function(map.biases[linear][i], parameters.biases[linear][i]);
    }
}
std::vector<double> sequence(cudnnRNNDataLayout_t layout, const std::array<int, 2>& lengths,
                             std::span<const double> logical) {
    std::vector<double> result;
    if (layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED) {
        for (std::size_t batch = 0; batch != 2; ++batch)
            for (std::size_t time = 0; time != 3; ++time)
                for (std::size_t channel = 0; channel != 2; ++channel)
                    result.push_back(logical[(time * 2 + batch) * 2 + channel]);
    } else {
        for (std::size_t time = 0; time != 3; ++time)
            for (std::size_t batch = 0; batch != 2; ++batch) {
                if (layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED &&
                    time >= static_cast<std::size_t>(lengths[batch]))
                    continue;
                for (std::size_t channel = 0; channel != 2; ++channel)
                    result.push_back(logical[(time * 2 + batch) * 2 + channel]);
            }
    }
    return result;
}
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): byte offset and mathematical reference.
double check_value(const std::vector<std::byte>& bytes, std::size_t offset, double expected) {
    expect(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset, "complete numerical read");
    T actual{};
    std::memcpy(&actual, bytes.data() + offset, sizeof(T));
    const auto error = std::abs(static_cast<double>(actual) - expected);
    const auto tolerance =
        sizeof(T) == 8 ? 2e-9 + 2e-8 * std::abs(expected) : 3e-5 + 3e-4 * std::abs(expected);
    expect(std::isfinite(actual) && error <= tolerance,
           "full original output matches independent LSTM oracle");
    return error;
}
template <typename T>
double check_values(const Device& device, std::span<const double> expected) {
    const auto bytes = device.read();
    expect(bytes.size() == expected.size() * sizeof(T), "complete numerical tensor extent");
    double error{};
    for (std::size_t i = 0; i != expected.size(); ++i)
        error = std::max(error, check_value<T>(bytes, i * sizeof(T), expected[i]));
    return error;
}
void snapshot(const char* name, const Device& device) {
    const auto bytes = device.read();
    std::cout << "{\"kind\":\"snapshot\",\"configuration\":" << configuration
              << ",\"iteration\":" << iteration << ",\"name\":" << std::quoted(name)
              << ",\"payload_hex\":\"";
    hex(bytes);
    std::cout << "\"}\n" << std::flush;
}
template <typename T>
void run_case(cudnnHandle_t handle, cudnnRNNDataLayout_t layout) {
    constexpr auto type = sizeof(T) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    // SDK9 requires padded I/O enabled for either unpacked data layout.
    const auto aux_flags = layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
                               ? CUDNN_RNN_PADDED_IO_DISABLED
                               : CUDNN_RNN_PADDED_IO_ENABLED;
    ++configuration;
    iteration = 0;
    std::cout << "{\"kind\":\"configuration\",\"configuration\":" << configuration
              << ",\"type\":" << static_cast<int>(type)
              << ",\"layout\":" << static_cast<int>(layout) << ",\"aux_flags\":" << aux_flags
              << "}\n"
              << std::flush;
    auto input = oracle::seeded_inputs();
    if (layout != CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED) input.lengths = {3, 3};
    const auto parameters = oracle::seeded_parameters();
    const auto upstream = oracle::seeded_upstream();
    const auto reference = oracle::forward(parameters, input);
    const auto gradient = oracle::backward(parameters, input, upstream);
    const auto difference = oracle::max_finite_difference_error(parameters, input, upstream);
    expect(difference < 2e-9,
           "all logical parameter/input/state gradients pass independent central differences");
    Rnn rnn(api.cudnnDestroyRNNDescriptor);
    Data x_descriptor(api.cudnnDestroyRNNDataDescriptor),
        y_descriptor(api.cudnnDestroyRNNDataDescriptor);
    Tensor state(api.cudnnDestroyTensorDescriptor);
    checked(api.cudnnCreateRNNDescriptor(rnn.output()), "create original RNN");
    checked(api.cudnnSetRNNDescriptor_v8(rnn.get(), CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                         CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                         CUDNN_LINEAR_INPUT, type, type, CUDNN_DEFAULT_MATH, 2, 2,
                                         2, 1, nullptr, aux_flags),
            "configure nonzero tiny standard LSTM");
    T padding{};
    for (auto* descriptor : {&x_descriptor, &y_descriptor}) {
        checked(api.cudnnCreateRNNDataDescriptor(descriptor->output()), "create sequence metadata");
        checked(api.cudnnSetRNNDataDescriptor(descriptor->get(), type, layout, 3, 2, 2,
                                              input.lengths.data(), &padding),
                "configure sequence metadata");
    }
    checked(api.cudnnCreateTensorDescriptor(state.output()), "create hidden/cell metadata");
    const std::array dimensions{1, 2, 2}, strides{4, 2, 1};
    checked(api.cudnnSetTensorNdDescriptor(state.get(), type, 3, dimensions.data(), strides.data()),
            "configure complete hidden/cell shape");
    std::size_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
    checked(api.cudnnGetRNNWeightSpaceSize(handle, rnn.get(), &weight_bytes), "query weight size");
    checked(api.cudnnGetRNNTempSpaceSizes(handle, rnn.get(), CUDNN_FWD_MODE_TRAINING,
                                          x_descriptor.get(), &workspace_bytes, &reserve_bytes),
            "query training workspace/reserve sizes");
    std::cout << "{\"kind\":\"capacities\",\"configuration\":" << configuration
              << ",\"weight_bytes\":" << weight_bytes << ",\"workspace_bytes\":" << workspace_bytes
              << ",\"reserve_bytes\":" << reserve_bytes << "}\n"
              << std::flush;
    expect(weight_bytes && weight_bytes <= kMaxWeightBytes &&
               workspace_bytes <= kMaxTemporaryBytes && reserve_bytes <= kMaxTemporaryBytes,
           "bound original capacities before device allocation");
    Device weight("weight", weight_bytes), dweight("dweight", weight_bytes),
        workspace("workspace", workspace_bytes), reserve("reserve", reserve_bytes);
    const auto mapped = mapping<T>(handle, rnn.get(), weight_bytes, weight.get());
    expect(mapping<T>(handle, rnn.get(), weight_bytes, dweight.get()) == mapped &&
               mapping<T>(handle, rnn.get(), weight_bytes, nullptr) == mapped,
           "owned gradient and raw-offset mappings agree with original weight allocation");
    mapping_controls(handle, rnn.get(), weight, mapped);
    std::vector<std::byte> weight_payload(weight_bytes), gradient_payload(weight_bytes);
    parameter_elements(mapped, parameters, [&](std::size_t offset, double value) {
        const auto encoded = static_cast<T>(value);
        const auto initial = static_cast<T>(value * .25 + .125);
        std::memcpy(weight_payload.data() + offset, &encoded, sizeof(T));
        std::memcpy(gradient_payload.data() + offset, &initial, sizeof(T));
    });
    weight.reset(weight_payload);
    dweight.reset(gradient_payload);
    const auto x_values = sequence(layout, input.lengths, input.x);
    const auto dy_values = sequence(layout, input.lengths, upstream.dy);
    Device x("x", x_values.size() * sizeof(T)), dy("dy", dy_values.size() * sizeof(T)),
        y("y", x.size()), dx("dx", x.size());
    Device hx("hx", 4 * sizeof(T)), cx("cx", 4 * sizeof(T)), dhy("dhy", 4 * sizeof(T)),
        dcy("dcy", 4 * sizeof(T)), hy("hy", 4 * sizeof(T)), cy("cy", 4 * sizeof(T)),
        dhx("dhx", 4 * sizeof(T)), dcx("dcx", 4 * sizeof(T)),
        lengths("lengths", sizeof(input.lengths));
    x.values<T>(x_values);
    dy.values<T>(dy_values);
    hx.values<T>(input.hx);
    cx.values<T>(input.cx);
    dhy.values<T>(upstream.dhy);
    dcy.values<T>(upstream.dcy);
    lengths.reset(std::as_bytes(std::span{input.lengths}));
    const auto* device_lengths = layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
                                     ? static_cast<const std::int32_t*>(lengths.get())
                                     : nullptr;
    for (unsigned int repeat = 1; repeat != 3; ++repeat) {
        iteration = repeat;
        for (auto* device : {&y, &dx, &hy, &cy, &dhx, &dcx})
            device->reset(std::vector<std::byte>(device->size(), std::byte{0xc3}));
        // Pageable uploads may return after staging; prove every initial input
        // and output reset ready before the nonblocking diagnostic stream runs.
        checked(api.cuCtxSynchronize(), "prove diagnostic input initialization complete");
        checked(api.cudnnRNNForward(handle, rnn.get(), CUDNN_FWD_MODE_TRAINING, device_lengths,
                                    x_descriptor.get(), x.get(), y_descriptor.get(), y.get(),
                                    state.get(), hx.get(), hy.get(), state.get(), cx.get(),
                                    cy.get(), weight_bytes, weight.get(), workspace_bytes,
                                    workspace.get(), reserve_bytes, reserve.get()),
                "original training forward");
        ++forward_calls;
        checked(api.cudnnRNNBackwardData_v8(
                    handle, rnn.get(), device_lengths, y_descriptor.get(), y.get(), dy.get(),
                    x_descriptor.get(), dx.get(), state.get(), hx.get(), dhy.get(), dhx.get(),
                    state.get(), cx.get(), dcy.get(), dcx.get(), weight_bytes, weight.get(),
                    workspace_bytes, workspace.get(), reserve_bytes, reserve.get()),
                "original backward data");
        ++data_calls;
        checked(api.cudnnRNNBackwardWeights_v8(
                    handle, rnn.get(), CUDNN_WGRAD_MODE_ADD, device_lengths, x_descriptor.get(),
                    x.get(), state.get(), hx.get(), y_descriptor.get(), y.get(), weight_bytes,
                    dweight.get(), workspace_bytes, workspace.get(), reserve_bytes, reserve.get()),
                "original ADD weight gradients after matching backward data");
        ++weight_calls;
        checked(api.cuCtxSynchronize(), "prove ordinary diagnostic computation complete");
        double error{};
        error = std::max(error, check_values<T>(y, sequence(layout, input.lengths, reference.y)));
        error = std::max(error, check_values<T>(dx, sequence(layout, input.lengths, gradient.dx)));
        error = std::max(error, check_values<T>(hy, reference.hy));
        error = std::max(error, check_values<T>(cy, reference.cy));
        error = std::max(error, check_values<T>(dhx, gradient.dhx));
        error = std::max(error, check_values<T>(dcx, gradient.dcx));
        const auto actual = dweight.read();
        parameter_elements(mapped, gradient.parameters, [&](std::size_t offset, double derivative) {
            T initial{};
            std::memcpy(&initial, gradient_payload.data() + offset, sizeof(T));
            error = std::max(error, check_value<T>(actual, offset, initial + repeat * derivative));
        });
        for (const auto* device : {&x, &dy, &hx, &cx, &dhy, &dcy, &weight, &lengths})
            device->unchanged();
        (void)workspace.read();
        (void)reserve.read();
        std::cout << "{\"kind\":\"iteration\",\"case\":" << ++cases
                  << ",\"configuration\":" << configuration << ",\"iteration\":" << repeat
                  << ",\"mapped_parameters\":48,\"max_absolute_error\":" << std::setprecision(17)
                  << error << ",\"cpu_finite_difference_error\":" << difference
                  << ",\"complete_oracle_passed\":true}\n"
                  << std::flush;
        for (const auto& named : {std::pair{"y", &y},
                                  {"hy", &hy},
                                  {"cy", &cy},
                                  {"dx", &dx},
                                  {"dhx", &dhx},
                                  {"dcx", &dcx},
                                  {"dweight_including_diagnostic_padding", &dweight}})
            snapshot(named.first, *named.second);
        mapped_providers();
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--cached"),
               "usage: execution_dnn_rnn_weight_params_gpu_test EXACT_CUDNN_TOP_PATH [--cached]");
        {
            Library dnn(argv[1]), driver("libcuda.so.1");
            api.select(dnn, driver, argc == 3);
            api.providers();
            const auto version = api.cudnnGetVersion();
            std::cout << "{\"kind\":\"sdk\",\"runtime_version\":" << version
                      << ",\"header_version\":" << CUDNN_VERSION
                      << ",\"cached\":" << (argc == 3 ? "true" : "false") << "}\n"
                      << std::flush;
            expect(version == CUDNN_VERSION, "selected SDK runtime and header agree");
            checked(api.cuInit(0), "initialize original Driver");
            CUdevice device{};
            checked(api.cuDeviceGet(&device, 0), "select visible diagnostic device");
            Context context(api.cuCtxDestroy);
            checked(api.cuCtxCreate(context.output(), 0, device), "create diagnostic context");
            Handle handle(api.cudnnDestroy);
            checked(api.cudnnCreate(handle.output()), "create original cuDNN handle");
            Stream stream(api.cuStreamDestroy);
            checked(api.cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING),
                    "create explicit stream");
            for (const auto layout :
                 {CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                  CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED}) {
                checked(api.cudnnSetStream(handle.get(), nullptr),
                        "select original default stream");
                run_case<float>(handle.get(), layout);
                checked(
                    api.cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                    "select original explicit stream");
                run_case<double>(handle.get(), layout);
            }
            checked(api.cudnnSetStream(handle.get(), nullptr),
                    "detach explicit stream before cleanup");
            expect(parameter_queries == 192 && cases == 12 && forward_calls == 12 &&
                       data_calls == 12 && weight_calls == 12,
                   "complete six-configuration query and two-cycle training matrix");
        }
        expect(!has_failure && resource_outputs == 81 && resources_destroyed == resource_outputs &&
                   device_allocations == 102 && device_releases == device_allocations &&
                   live_libraries == 0,
               "all original owners clean up successfully");
        std::cout << "{\"kind\":\"rnn_v8_result\",\"status\":\"passed\",\"parameter_queries\":"
                  << parameter_queries << ",\"numerical_cases\":" << cases
                  << ",\"forward_calls\":" << forward_calls
                  << ",\"backward_data_calls\":" << data_calls
                  << ",\"backward_weight_calls\":" << weight_calls
                  << ",\"resources_destroyed\":" << resources_destroyed
                  << ",\"device_releases\":" << device_releases
                  << ",\"live_libraries\":" << live_libraries
                  << ",\"normal_cleanup\":true,\"logical_parameter_gradient_elements\":576,"
                     "\"logical_output_and_gradient_elements\":1040,"
                     "\"opaque_padding_gradient_qualified\":false,\"temporary_mathematics_"
                     "qualified\":false,"
                     "\"all_rnn_models_qualified\":false,\"full_device_qualified\":false,"
                     "\"saturation_qualified\":false,\"persistent_qualified\":false,"
                     "\"priority_qualified\":false,\"performance_qualified\":false,"
                     "\"full_goal_qualified\":false}\n"
                  << std::flush;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
