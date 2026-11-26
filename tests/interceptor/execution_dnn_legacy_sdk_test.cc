#include "rnn_lstm_oracle.h"

#include <cuda.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// This ordinary application uses only SDK entry points and an independent CPU
// model. Its direct and explicit-DSO lookup modes share every input and oracle.
// Legacy declarations are intentionally used with their actual SDK8 ABI.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static_assert(CUDNN_MAJOR == 8);
namespace {
namespace oracle = glimmer::tests::rnn_lstm_oracle;
constexpr std::size_t kGuard = 256;
constexpr std::size_t kMaxWeightBytes = std::size_t{64} * 1024;
constexpr std::size_t kMaxTemporaryBytes = std::size_t{64} * 1024 * 1024;
constexpr std::array<const char*, 25> kLegacyNames{"cudnnOpsInferVersionCheck",
                                                   "cudnnOpsTrainVersionCheck",
                                                   "cudnnCnnInferVersionCheck",
                                                   "cudnnCnnTrainVersionCheck",
                                                   "cudnnAdvInferVersionCheck",
                                                   "cudnnAdvTrainVersionCheck",
                                                   "cudnnCreateAlgorithmDescriptor",
                                                   "cudnnSetAlgorithmDescriptor",
                                                   "cudnnDestroyAlgorithmDescriptor",
                                                   "cudnnSetRNNDescriptor_v6",
                                                   "cudnnSetRNNMatrixMathType",
                                                   "cudnnGetRNNMatrixMathType",
                                                   "cudnnSetRNNBiasMode",
                                                   "cudnnGetRNNBiasMode",
                                                   "cudnnRNNSetClip",
                                                   "cudnnGetRNNWorkspaceSize",
                                                   "cudnnGetRNNTrainingReserveSize",
                                                   "cudnnGetRNNParamsSize",
                                                   "cudnnGetRNNLinLayerMatrixParams",
                                                   "cudnnGetRNNLinLayerBiasParams",
                                                   "cudnnSetRNNAlgorithmDescriptor",
                                                   "cudnnGetRNNForwardInferenceAlgorithmMaxCount",
                                                   "cudnnRNNForwardTraining",
                                                   "cudnnRNNBackwardData",
                                                   "cudnnRNNBackwardWeights"};
std::array<unsigned int, kLegacyNames.size()> legacy_calls{};
bool has_failure{};
std::size_t live_resources{};

void expect(bool condition, const std::string& message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
void record_success(std::string_view operation) {
    for (std::size_t index = 0; index != kLegacyNames.size(); ++index)
        if (operation == kLegacyNames[index]) ++legacy_calls[index];
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
    record_success(operation);
}

class Library final {
   public:
    explicit Library(const char* path) : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        expect(handle_ != nullptr, "open the explicitly selected cuDNN top-level DSO");
        ++live_resources;
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() {
        if (!has_failure && handle_) {
            if (dlclose(handle_) != 0)
                has_failure = true;
            else
                --live_resources;
        }
    }
    template <typename Function>
    Function get(const char* name) const {
        dlerror();
        void* address = dlsym(handle_, name);
        const char* error = dlerror();
        expect(address != nullptr && error == nullptr, std::string("resolve SDK symbol: ") + name);
        static_assert(sizeof(Function) == sizeof(address));
        Function function{};
        std::memcpy(&function, &address, sizeof(function));
        return function;
    }

   private:
    void* handle_{};
};

// Cached mode resolves every cuDNN operation, including constructors and
// destructors. No native pointer equality is required: interposition is valid.
#define SDK_FUNCTIONS(F)                            \
    F(cudnnGetVersion)                              \
    F(cudnnCreate)                                  \
    F(cudnnDestroy)                                 \
    F(cudnnSetStream)                               \
    F(cudnnCreateRNNDescriptor)                     \
    F(cudnnDestroyRNNDescriptor)                    \
    F(cudnnCreateTensorDescriptor)                  \
    F(cudnnDestroyTensorDescriptor)                 \
    F(cudnnSetTensorNdDescriptor)                   \
    F(cudnnCreateFilterDescriptor)                  \
    F(cudnnDestroyFilterDescriptor)                 \
    F(cudnnSetFilterNdDescriptor)                   \
    F(cudnnGetFilterNdDescriptor)                   \
    F(cudnnGetFilterSizeInBytes)                    \
    F(cudnnCreateDropoutDescriptor)                 \
    F(cudnnDestroyDropoutDescriptor)                \
    F(cudnnDropoutGetStatesSize)                    \
    F(cudnnSetDropoutDescriptor)                    \
    F(cudnnOpsInferVersionCheck)                    \
    F(cudnnOpsTrainVersionCheck)                    \
    F(cudnnCnnInferVersionCheck)                    \
    F(cudnnCnnTrainVersionCheck)                    \
    F(cudnnAdvInferVersionCheck)                    \
    F(cudnnAdvTrainVersionCheck)                    \
    F(cudnnCreateAlgorithmDescriptor)               \
    F(cudnnSetAlgorithmDescriptor)                  \
    F(cudnnDestroyAlgorithmDescriptor)              \
    F(cudnnSetRNNDescriptor_v6)                     \
    F(cudnnSetRNNMatrixMathType)                    \
    F(cudnnGetRNNMatrixMathType)                    \
    F(cudnnSetRNNBiasMode)                          \
    F(cudnnGetRNNBiasMode)                          \
    F(cudnnRNNSetClip)                              \
    F(cudnnGetRNNWorkspaceSize)                     \
    F(cudnnGetRNNTrainingReserveSize)               \
    F(cudnnGetRNNParamsSize)                        \
    F(cudnnGetRNNLinLayerMatrixParams)              \
    F(cudnnGetRNNLinLayerBiasParams)                \
    F(cudnnSetRNNAlgorithmDescriptor)               \
    F(cudnnGetRNNForwardInferenceAlgorithmMaxCount) \
    F(cudnnRNNForwardTraining)                      \
    F(cudnnRNNBackwardData)                         \
    F(cudnnRNNBackwardWeights)
struct Api {
// NOLINTNEXTLINE(bugprone-macro-parentheses): NAME declares a member, not an expression.
#define DECLARE(NAME) decltype(&::NAME) NAME{};
    SDK_FUNCTIONS(DECLARE)
#undef DECLARE
    Api(const Library& library, bool cached) {
#define RESOLVE(NAME) NAME = cached ? library.get<decltype(&::NAME)>(#NAME) : &::NAME;
        SDK_FUNCTIONS(RESOLVE)
#undef RESOLVE
    }
};
#undef SDK_FUNCTIONS
#define DNN(NAME, ...) checked(api.NAME(__VA_ARGS__), #NAME)

// Any uncertain SDK failure is fail-stop. In particular, unwinding must not
// invent successful cleanup or call into a context with unproved physical work.
template <typename Handle, typename Status>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (!value_ || has_failure) return;
        const auto result = destroy_(value_);
        if (static_cast<int>(result) != 0) {
            has_failure = true;
            std::cerr << destroy_name_ << ": " << static_cast<int>(result) << '\n';
        } else {
            record_success(destroy_name_);
            --live_resources;
        }
    }
    template <typename Create, typename... Args>
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): paired SDK lifecycle labels.
    void create(Create create, Status (*destroy)(Handle), const char* create_name,
                const char* destroy_name, Args... args) {
        expect(value_ == nullptr, "create each SDK resource only once");
        destroy_ = destroy;
        destroy_name_ = destroy_name;
        checked(create(&value_, args...), create_name);
        expect(value_ != nullptr, std::string(create_name) + " returned a live handle");
        ++live_resources;
    }
    Handle get() const {
        return value_;
    }

   private:
    Handle value_{};
    Status (*destroy_)(Handle){};
    const char* destroy_name_{};
};
using Context = Resource<CUcontext, CUresult>;
using Stream = Resource<CUstream, CUresult>;
using Handle = Resource<cudnnHandle_t, cudnnStatus_t>;
using Rnn = Resource<cudnnRNNDescriptor_t, cudnnStatus_t>;
using Algorithm = Resource<cudnnAlgorithmDescriptor_t, cudnnStatus_t>;
using Dropout = Resource<cudnnDropoutDescriptor_t, cudnnStatus_t>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnStatus_t>;
using Filter = Resource<cudnnFilterDescriptor_t, cudnnStatus_t>;

void hex(std::span<const std::byte> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    for (const auto byte : bytes) {
        const auto value = std::to_integer<unsigned int>(byte);
        std::cout.put(digits[value >> 4]);
        std::cout.put(digits[value & 15]);
    }
}
void json_string(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::cout.put('"');
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (character == '"' || character == '\\') {
            std::cout.put('\\');
            std::cout.put(character);
        } else if (byte < 0x20) {
            std::cout << "\\u00" << digits[byte >> 4] << digits[byte & 15];
        } else {
            std::cout.put(character);
        }
    }
    std::cout.put('"');
}
void providers() {
    struct Collection {
        std::vector<std::string> paths;
        bool failed{};
    } collection;
    const auto callback = [](dl_phdr_info* info, std::size_t, void* opaque) noexcept {
        auto& collected = *static_cast<Collection*>(opaque);
        try {
            if (!info->dlpi_name) return 0;
            const std::string_view path(info->dlpi_name);
            const auto slash = path.rfind('/');
            const auto name = path.substr(slash == std::string_view::npos ? 0 : slash + 1);
            if (name.starts_with("libcudnn") && name.find(".so") != std::string_view::npos)
                collected.paths.emplace_back(path);
            return 0;
        } catch (...) {
            collected.failed = true;
            return 1;
        }
    };
    const auto result = dl_iterate_phdr(callback, &collection);
    expect(result == 0 && !collection.failed, "collect actual loaded cuDNN provider paths");
    std::ranges::sort(collection.paths);
    std::cout << "{\"kind\":\"providers\",\"paths\":[";
    for (std::size_t index = 0; index != collection.paths.size(); ++index) {
        if (index) std::cout.put(',');
        json_string(collection.paths[index]);
    }
    std::cout << "]}\n";
    std::cout.flush();
}
std::uint64_t fingerprint(std::span<const std::byte> bytes) {
    // FNV-1a is only a content fingerprint, not a cryptographic evidence seal.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto byte : bytes) {
        hash ^= std::to_integer<unsigned int>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

class Device final {
   public:
    Device(const char* name, std::size_t bytes)
        : name_(name), bytes_(bytes), initial_(bounded_size(bytes), std::byte{0xa7}) {
        checked(cuMemAlloc(&base_, initial_.size()), "cuMemAlloc");
        ++live_resources;
        reset(std::vector<std::byte>(bytes));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() {
        if (!base_ || has_failure) return;
        if (cuMemFree(base_) != CUDA_SUCCESS)
            has_failure = true;
        else
            --live_resources;
    }
    void* get() const {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): actual guarded CUDA allocation address.
        return reinterpret_cast<void*>(base_ + kGuard);
    }
    std::size_t size() const {
        return bytes_;
    }
    const char* name() const {
        return name_;
    }
    void reset(std::span<const std::byte> bytes) {
        expect(bytes.size() == bytes_, std::string(name_) + ": complete payload initialization");
        std::copy(bytes.begin(), bytes.end(), initial_.begin() + kGuard);
        checked(cuMemcpyHtoD(base_, initial_.data(), initial_.size()), "cuMemcpyHtoD");
    }
    void values(std::span<const double> values) {
        std::vector<float> encoded;
        encoded.reserve(values.size());
        for (const auto value : values) encoded.push_back(static_cast<float>(value));
        reset(std::as_bytes(std::span{encoded}));
    }
    std::vector<std::byte> snapshot(unsigned int iteration, bool opaque = false,
                                    bool unchanged = false) const {
        std::vector<std::byte> bytes(initial_.size());
        checked(cuMemcpyDtoH(bytes.data(), base_, bytes.size()), "cuMemcpyDtoH");
        const auto whole = std::span<const std::byte>{bytes};
        const auto payload = whole.subspan(kGuard, bytes_);
        // Publish raw data before validating it, preserving failed-oracle evidence.
        std::cout << "{\"kind\":\"buffer\",\"name\":\"" << name_ << "\",\"iteration\":" << iteration
                  << ",\"payload_bytes\":" << bytes_ << ",\"guard_bytes\":" << kGuard
                  << ",\"opaque\":" << (opaque ? "true" : "false");
        if (opaque) {
            std::cout << ",\"payload_fnv1a64\":\"" << std::hex << std::setfill('0') << std::setw(16)
                      << fingerprint(payload) << std::dec << "\",\"prefix_hex\":\"";
            hex(whole.first(kGuard));
            std::cout << "\",\"suffix_hex\":\"";
            hex(whole.last(kGuard));
        } else {
            std::cout << ",\"guarded_hex\":\"";
            hex(whole);
        }
        std::cout << "\"}\n";
        std::cout.flush();
        expect(std::equal(bytes.begin(), bytes.begin() + kGuard, initial_.begin()) &&
                   std::equal(bytes.end() - kGuard, bytes.end(), initial_.end() - kGuard),
               std::string(name_) + ": complete allocation prefix/suffix guards");
        if (unchanged)
            expect(bytes == initial_, std::string(name_) + ": every input byte remains unchanged");
        return {payload.begin(), payload.end()};
    }

   private:
    static std::size_t bounded_size(std::size_t bytes) {
        expect(bytes <= kMaxTemporaryBytes,
               "bound device allocation before host or SDK allocation");
        return bytes + 2 * kGuard;
    }
    const char* name_{};
    std::size_t bytes_{};
    std::vector<std::byte> initial_;
    CUdeviceptr base_{};
};

using Sequence = std::array<Tensor, oracle::kSteps>;
using DescriptorArray = std::array<cudnnTensorDescriptor_t, oracle::kSteps>;
DescriptorArray configure_sequence(const Api& api, Sequence& descriptors,
                                   const std::array<int, 2>& lengths) {
    DescriptorArray result{};
    for (std::size_t time = 0; time != descriptors.size(); ++time) {
        auto& tensor = descriptors[time];
        tensor.create(api.cudnnCreateTensorDescriptor, api.cudnnDestroyTensorDescriptor,
                      "cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor");
        const auto batch = static_cast<int>(std::ranges::count_if(
            lengths, [time](int length) { return time < static_cast<std::size_t>(length); }));
        const std::array dimensions{batch, 2, 1}, strides{2, 1, 1};
        DNN(cudnnSetTensorNdDescriptor, tensor.get(), CUDNN_DATA_FLOAT, 3, dimensions.data(),
            strides.data());
        result[time] = tensor.get();
    }
    return result;
}
void configure_states(const Api& api, std::array<Tensor, 6>& states) {
    const std::array dimensions{1, 2, 2}, strides{4, 2, 1};
    for (auto& state : states) {
        state.create(api.cudnnCreateTensorDescriptor, api.cudnnDestroyTensorDescriptor,
                     "cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor");
        DNN(cudnnSetTensorNdDescriptor, state.get(), CUDNN_DATA_FLOAT, 3, dimensions.data(),
            strides.data());
    }
}
std::vector<double> packed(const std::array<int, 2>& lengths, std::span<const double> dense) {
    expect(dense.size() == 12 && lengths == std::array{3, 2}, "fixed packed sequence contract");
    std::vector<double> result;
    for (std::size_t time = 0; time != 3; ++time)
        for (std::size_t batch = 0; batch != 2; ++batch) {
            if (time >= static_cast<std::size_t>(lengths[batch])) continue;
            for (std::size_t channel = 0; channel != 2; ++channel)
                result.push_back(dense[(time * 2 + batch) * 2 + channel]);
        }
    expect(result.size() == 10, "all ten packed scalar inputs are represented");
    return result;
}

struct Mapping {
    std::array<std::array<std::size_t, 4>, 8> matrices{};
    std::array<std::array<std::size_t, 2>, 8> biases{};
    bool operator==(const Mapping&) const = default;
};
struct Shape {
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    int rank{};
    std::array<int, 3> dimensions{};
    std::size_t bytes{};
};
Shape filter_shape(const Api& api, cudnnFilterDescriptor_t descriptor) {
    struct Guarded {
        std::uint64_t before{0x123456789abcdef0ULL};
        std::array<int, 8> dimensions{-8123, -8123, -8123, -8123, -8123, -8123, -8123, -8123};
        std::uint64_t after{0xfedcba9876543210ULL};
    } guarded;
    Shape shape;
    DNN(cudnnGetFilterNdDescriptor, descriptor, 3, &shape.type, &shape.format, &shape.rank,
        guarded.dimensions.data());
    DNN(cudnnGetFilterSizeInBytes, descriptor, &shape.bytes);
    expect(guarded.before == 0x123456789abcdef0ULL && guarded.after == 0xfedcba9876543210ULL &&
               std::ranges::all_of(std::span{guarded.dimensions}.subspan(3),
                                   [](int value) { return value == -8123; }),
           "original filter getter preserves every unrequested host output");
    std::copy_n(guarded.dimensions.begin(), 3, shape.dimensions.begin());
    return shape;
}
void create_weight_filter(const Api& api, Filter& descriptor, std::size_t bytes) {
    descriptor.create(api.cudnnCreateFilterDescriptor, api.cudnnDestroyFilterDescriptor,
                      "cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor");
    expect(bytes > 0 && bytes <= kMaxWeightBytes && bytes % sizeof(float) == 0,
           "bounded exact float weight extent");
    const std::array dimensions{static_cast<int>(bytes / sizeof(float)), 1, 1};
    DNN(cudnnSetFilterNdDescriptor, descriptor.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3,
        dimensions.data());
}
Mapping mapping(const Api& api, cudnnHandle_t handle, cudnnRNNDescriptor_t rnn,
                cudnnTensorDescriptor_t x, cudnnFilterDescriptor_t weights, const Device& device) {
    Filter matrix, bias;
    for (auto* filter : {&matrix, &bias})
        filter->create(api.cudnnCreateFilterDescriptor, api.cudnnDestroyFilterDescriptor,
                       "cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor");
    Mapping result;
    std::vector<bool> observed(device.size());
    for (std::size_t linear = 0; linear != 8; ++linear) {
        std::array<void*, 2> addresses{};
        DNN(cudnnGetRNNLinLayerMatrixParams, handle, rnn, 0, x, weights, device.get(),
            static_cast<int>(linear), matrix.get(), &addresses[0]);
        DNN(cudnnGetRNNLinLayerBiasParams, handle, rnn, 0, x, weights, device.get(),
            static_cast<int>(linear), bias.get(), &addresses[1]);
        for (std::size_t part = 0; part != 2; ++part) {
            const auto shape = filter_shape(api, part ? bias.get() : matrix.get());
            const std::size_t elements = part ? 2 : 4;
            std::size_t product = 1;
            for (const auto dimension : shape.dimensions) {
                expect(dimension > 0 && static_cast<std::size_t>(dimension) <= elements / product,
                       "bounded original legacy parameter dimensions");
                product *= static_cast<std::size_t>(dimension);
            }
            expect(shape.rank == 3 && shape.type == CUDNN_DATA_FLOAT &&
                       shape.format == CUDNN_TENSOR_NCHW && product == elements &&
                       shape.bytes == elements * sizeof(float),
                   "original filter extent agrees with complete logical parameter extent");
            const auto base = reinterpret_cast<std::uintptr_t>(device.get());
            const auto address = reinterpret_cast<std::uintptr_t>(addresses[part]);
            expect(address >= base && address - base <= device.size(),
                   "actual legacy parameter pointer lies in its own allocation");
            const auto offset = static_cast<std::size_t>(address - base);
            expect(offset % sizeof(float) == 0 && shape.bytes <= device.size() - offset,
                   "entire observed filter subregion lies in the declared parent weight range");
            std::cout << "{\"kind\":\"layout\",\"name\":\"" << device.name()
                      << "\",\"linear\":" << linear << ",\"part\":\"" << (part ? "bias" : "matrix")
                      << "\",\"offset\":" << offset << ",\"extent\":" << shape.bytes
                      << ",\"dimensions\":[" << shape.dimensions[0] << ',' << shape.dimensions[1]
                      << ',' << shape.dimensions[2] << "],\"logical_offsets\":[";
            // Legacy filters are contiguous. Logical matrices have row-major
            // hidden/output rows and input/previous-hidden columns; no result-
            // dependent transpose or copied v8 stride interpretation is used.
            for (std::size_t element = 0; element != elements; ++element) {
                const auto location = offset + element * sizeof(float);
                for (std::size_t byte = 0; byte != sizeof(float); ++byte) {
                    expect(!observed[location + byte], "all mapped parameter bytes are disjoint");
                    observed[location + byte] = true;
                }
                if (part)
                    result.biases[linear][element] = location;
                else
                    result.matrices[linear][element] = location;
                std::cout << (element ? "," : "") << location;
            }
            std::cout << "]}\n";
        }
    }
    expect(std::ranges::count(observed, true) == 48 * static_cast<std::ptrdiff_t>(sizeof(float)),
           "discover all 48 scalar parameter positions exactly once");
    return result;
}
template <typename Function>
void parameters_at(const Mapping& mapping, const oracle::Parameters& parameters, Function visit) {
    for (std::size_t linear = 0; linear != 8; ++linear) {
        for (std::size_t index = 0; index != 4; ++index)
            visit(mapping.matrices[linear][index], parameters.matrices[linear][index]);
        for (std::size_t index = 0; index != 2; ++index)
            visit(mapping.biases[linear][index], parameters.biases[linear][index]);
    }
}
double check_value(std::span<const std::byte> bytes, std::size_t offset, double expected,
                   std::string_view name) {
    expect(offset <= bytes.size() && sizeof(float) <= bytes.size() - offset,
           "complete numerical scalar read");
    float actual{};
    std::memcpy(&actual, bytes.data() + offset, sizeof(actual));
    const auto error = std::abs(static_cast<double>(actual) - expected);
    expect(std::isfinite(actual) && error <= 3e-5 + 3e-4 * std::abs(expected),
           std::string(name) + ": independent LSTM oracle mismatch at byte " +
               std::to_string(offset) + " actual=" + std::to_string(actual) +
               " expected=" + std::to_string(expected));
    return error;
}
double check_values(std::span<const std::byte> bytes, std::span<const double> expected,
                    std::string_view name) {
    expect(bytes.size() == expected.size() * sizeof(float), "complete numerical tensor extent");
    double error{};
    for (std::size_t index = 0; index != expected.size(); ++index)
        error = std::max(error, check_value(bytes, index * sizeof(float), expected[index], name));
    return error;
}

void run(const Api& api, cudnnHandle_t handle) {
    const auto input = oracle::seeded_inputs();
    const auto parameters = oracle::seeded_parameters();
    const auto upstream = oracle::seeded_upstream();
    const auto reference = oracle::forward(parameters, input);
    const auto gradient = oracle::backward(parameters, input, upstream);
    const auto difference = oracle::max_finite_difference_error(parameters, input, upstream);
    expect(difference < 2e-9, "complete independent central-difference gradient validation");

    DNN(cudnnOpsInferVersionCheck);
    DNN(cudnnOpsTrainVersionCheck);
    DNN(cudnnCnnInferVersionCheck);
    DNN(cudnnCnnTrainVersionCheck);
    DNN(cudnnAdvInferVersionCheck);
    DNN(cudnnAdvTrainVersionCheck);
    providers();
    std::size_t dropout_bytes{};
    DNN(cudnnDropoutGetStatesSize, handle, &dropout_bytes);
    expect(dropout_bytes > 0 && dropout_bytes <= kMaxTemporaryBytes,
           "bound real owned dropout state storage");
    Device dropout_state("dropout_state", dropout_bytes);
    Dropout dropout;
    dropout.create(api.cudnnCreateDropoutDescriptor, api.cudnnDestroyDropoutDescriptor,
                   "cudnnCreateDropoutDescriptor", "cudnnDestroyDropoutDescriptor");
    DNN(cudnnSetDropoutDescriptor, dropout.get(), handle, 0.0F, dropout_state.get(),
        dropout_state.size(), 17ULL);
    Algorithm algorithm;
    algorithm.create(api.cudnnCreateAlgorithmDescriptor, api.cudnnDestroyAlgorithmDescriptor,
                     "cudnnCreateAlgorithmDescriptor", "cudnnDestroyAlgorithmDescriptor");
    cudnnAlgorithm_t algorithm_value{};
    algorithm_value.algo.RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
    DNN(cudnnSetAlgorithmDescriptor, algorithm.get(), algorithm_value);
    Rnn rnn;
    rnn.create(api.cudnnCreateRNNDescriptor, api.cudnnDestroyRNNDescriptor,
               "cudnnCreateRNNDescriptor", "cudnnDestroyRNNDescriptor");
    DNN(cudnnSetRNNDescriptor_v6, handle, rnn.get(), 2, 1, dropout.get(), CUDNN_LINEAR_INPUT,
        CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT);
    DNN(cudnnSetRNNMatrixMathType, rnn.get(), CUDNN_DEFAULT_MATH);
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    DNN(cudnnGetRNNMatrixMathType, rnn.get(), &math_type);
    expect(math_type == CUDNN_DEFAULT_MATH, "original math type getter round trip");
    DNN(cudnnSetRNNBiasMode, rnn.get(), CUDNN_RNN_DOUBLE_BIAS);
    cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_NO_BIAS;
    DNN(cudnnGetRNNBiasMode, rnn.get(), &bias_mode);
    expect(bias_mode == CUDNN_RNN_DOUBLE_BIAS, "original double bias getter round trip");
    DNN(cudnnRNNSetClip, handle, rnn.get(), CUDNN_RNN_CLIP_NONE, CUDNN_NOT_PROPAGATE_NAN, 0.0, 0.0);
    DNN(cudnnSetRNNAlgorithmDescriptor, handle, rnn.get(), algorithm.get());
    int algorithm_count{};
    DNN(cudnnGetRNNForwardInferenceAlgorithmMaxCount, handle, rnn.get(), &algorithm_count);
    expect(algorithm_count > 0, "actual original inference algorithm count is positive");

    Sequence x_descriptors, y_descriptors, dy_descriptors, dx_descriptors;
    const auto x_array = configure_sequence(api, x_descriptors, input.lengths);
    const auto y_array = configure_sequence(api, y_descriptors, input.lengths);
    const auto dy_array = configure_sequence(api, dy_descriptors, input.lengths);
    const auto dx_array = configure_sequence(api, dx_descriptors, input.lengths);
    std::array<Tensor, 6> states;
    configure_states(api, states);
    std::vector<cudnnTensorDescriptor_t> identities;
    for (const auto* descriptors : {&x_array, &y_array, &dy_array, &dx_array})
        identities.insert(identities.end(), descriptors->begin(), descriptors->end());
    for (const auto& state : states) identities.push_back(state.get());
    for (std::size_t left = 0; left != identities.size(); ++left)
        for (std::size_t right = left + 1; right != identities.size(); ++right)
            expect(identities[left] != identities[right],
                   "18 independent tensor descriptor identities");

    std::size_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
    DNN(cudnnGetRNNParamsSize, handle, rnn.get(), x_array[0], &weight_bytes, CUDNN_DATA_FLOAT);
    DNN(cudnnGetRNNWorkspaceSize, handle, rnn.get(), 3, x_array.data(), &workspace_bytes);
    DNN(cudnnGetRNNTrainingReserveSize, handle, rnn.get(), 3, x_array.data(), &reserve_bytes);
    expect(weight_bytes > 0 && weight_bytes <= kMaxWeightBytes &&
               weight_bytes % sizeof(float) == 0 && workspace_bytes <= kMaxTemporaryBytes &&
               reserve_bytes <= kMaxTemporaryBytes,
           "bound original parameter and temporary capacities before allocation");
    Filter weight_descriptor, gradient_descriptor;
    create_weight_filter(api, weight_descriptor, weight_bytes);
    create_weight_filter(api, gradient_descriptor, weight_bytes);
    Device weight("weight", weight_bytes), dweight("dweight", weight_bytes),
        workspace("workspace", workspace_bytes), reserve("reserve", reserve_bytes);
    const auto mapped =
        mapping(api, handle, rnn.get(), x_array[0], weight_descriptor.get(), weight);
    expect(
        mapping(api, handle, rnn.get(), x_array[0], gradient_descriptor.get(), dweight) == mapped,
        "independently queried weight and gradient layouts agree for all 48 parameters");
    std::vector<std::byte> weight_payload(weight_bytes), initial_gradient(weight_bytes);
    parameters_at(mapped, parameters, [&](std::size_t offset, double value) {
        const auto encoded = static_cast<float>(value);
        const auto initial = static_cast<float>(value * .25 + .125);
        std::memcpy(weight_payload.data() + offset, &encoded, sizeof(encoded));
        std::memcpy(initial_gradient.data() + offset, &initial, sizeof(initial));
    });
    weight.reset(weight_payload);
    dweight.reset(initial_gradient);
    const auto x_values = packed(input.lengths, input.x);
    const auto dy_values = packed(input.lengths, upstream.dy);
    const auto y_reference = packed(input.lengths, reference.y);
    const auto dx_reference = packed(input.lengths, gradient.dx);
    Device x("x", 10 * sizeof(float)), dy("dy", 10 * sizeof(float)), y("y", x.size()),
        dx("dx", x.size());
    Device hx("hx", 4 * sizeof(float)), cx("cx", 4 * sizeof(float)), dhy("dhy", 4 * sizeof(float)),
        dcy("dcy", 4 * sizeof(float)), hy("hy", 4 * sizeof(float)), cy("cy", 4 * sizeof(float)),
        dhx("dhx", 4 * sizeof(float)), dcx("dcx", 4 * sizeof(float));
    x.values(x_values);
    dy.values(dy_values);
    hx.values(input.hx);
    cx.values(input.cx);
    dhy.values(upstream.dhy);
    dcy.values(upstream.dcy);
    checked(cuCtxSynchronize(), "cuCtxSynchronize");
    std::cout << std::setprecision(17)
              << "{\"kind\":\"model\",\"layers\":1,\"input\":2,\"hidden\":2,\"steps\":3,"
                 "\"batch\":2,\"lengths\":[3,2],\"packed_floats\":10,\"seed\":17,"
                 "\"dropout\":0,\"algorithm\":\"standard\",\"type\":\"float\","
                 "\"bias\":\"double\",\"direction\":\"uni\",\"input_mode\":\"linear\","
                 "\"math\":\"default\",\"clip\":\"none\",\"tensor_descriptors\":18,"
                 "\"weight_bytes\":"
              << weight_bytes << ",\"workspace_bytes\":" << workspace_bytes
              << ",\"reserve_bytes\":" << reserve_bytes << ",\"dropout_bytes\":" << dropout_bytes
              << ",\"algorithm_count\":" << algorithm_count
              << ",\"central_difference_max_error\":" << difference << "}\n";
    for (const auto* device : {&x, &dy, &hx, &cx, &dhy, &dcy, &weight, &dweight})
        device->snapshot(0, false, true);
    for (const auto* device : {&workspace, &reserve, &dropout_state}) device->snapshot(0, true);

    for (unsigned int iteration = 1; iteration != 3; ++iteration) {
        for (auto* device : {&y, &dx, &hy, &cy, &dhx, &dcx})
            device->reset(std::vector<std::byte>(device->size(), std::byte{0xc3}));
        // Pageable uploads may finish staging before device execution. Explicit
        // context completion orders all initialization with the nonblocking stream.
        checked(cuCtxSynchronize(), "cuCtxSynchronize");
        DNN(cudnnRNNForwardTraining, handle, rnn.get(), 3, x_array.data(), x.get(), states[0].get(),
            hx.get(), states[1].get(), cx.get(), weight_descriptor.get(), weight.get(),
            y_array.data(), y.get(), states[2].get(), hy.get(), states[3].get(), cy.get(),
            workspace.get(), workspace.size(), reserve.get(), reserve.size());
        DNN(cudnnRNNBackwardData, handle, rnn.get(), 3, y_array.data(), y.get(), dy_array.data(),
            dy.get(), states[2].get(), dhy.get(), states[3].get(), dcy.get(),
            weight_descriptor.get(), weight.get(), states[0].get(), hx.get(), states[1].get(),
            cx.get(), dx_array.data(), dx.get(), states[4].get(), dhx.get(), states[5].get(),
            dcx.get(), workspace.get(), workspace.size(), reserve.get(), reserve.size());
        // The legacy ABI has no WgradMode: it adds to the retained initial
        // gradient. Iteration two must retain both previous and current terms.
        DNN(cudnnRNNBackwardWeights, handle, rnn.get(), 3, x_array.data(), x.get(), states[0].get(),
            hx.get(), y_array.data(), y.get(), workspace.get(), workspace.size(),
            gradient_descriptor.get(), dweight.get(), reserve.get(), reserve.size());
        checked(cuCtxSynchronize(), "cuCtxSynchronize");
        const auto y_bytes = y.snapshot(iteration);
        const auto dx_bytes = dx.snapshot(iteration);
        const auto hy_bytes = hy.snapshot(iteration);
        const auto cy_bytes = cy.snapshot(iteration);
        const auto dhx_bytes = dhx.snapshot(iteration);
        const auto dcx_bytes = dcx.snapshot(iteration);
        const auto gradient_bytes = dweight.snapshot(iteration);
        for (const auto* device : {&x, &dy, &hx, &cx, &dhy, &dcy, &weight})
            device->snapshot(iteration, false, true);
        for (const auto* device : {&workspace, &reserve, &dropout_state})
            device->snapshot(iteration, true);
        double error = std::max(
            {check_values(y_bytes, y_reference, "y"), check_values(dx_bytes, dx_reference, "dx"),
             check_values(hy_bytes, reference.hy, "hy"), check_values(cy_bytes, reference.cy, "cy"),
             check_values(dhx_bytes, gradient.dhx, "dhx"),
             check_values(dcx_bytes, gradient.dcx, "dcx")});
        parameters_at(mapped, gradient.parameters, [&](std::size_t offset, double value) {
            float initial{};
            std::memcpy(&initial, initial_gradient.data() + offset, sizeof(initial));
            error = std::max(
                error, check_value(gradient_bytes, offset,
                                   static_cast<double>(initial) + iteration * value, "dweight"));
        });
        std::cout << "{\"kind\":\"iteration\",\"iteration\":" << iteration
                  << ",\"tensor_scalars_checked\":36,\"parameter_gradients_checked\":48,"
                     "\"guarded_allocations_checked\":17,\"unchanged_inputs_checked\":7,"
                     "\"max_absolute_error\":"
                  << error << ",\"complete_oracle_passed\":true}\n";
        std::cout.flush();
    }
}
#undef DNN

void execute(const char* path, bool cached) {
    Library library(path);
    const Api api(library, cached);
    expect(api.cudnnGetVersion() == CUDNN_VERSION, "exact selected SDK8 runtime/header version");
    std::cout << "{\"kind\":\"sdk\",\"header_version\":" << CUDNN_VERSION
              << ",\"cached\":" << (cached ? "true" : "false") << "}\n";
    checked(cuInit(0), "cuInit");
    CUdevice device{};
    checked(cuDeviceGet(&device, 0), "cuDeviceGet");
    Context context;
    context.create(cuCtxCreate, cuCtxDestroy, "cuCtxCreate", "cuCtxDestroy", 0U, device);
    Stream stream;
    stream.create(cuStreamCreate, cuStreamDestroy, "cuStreamCreate", "cuStreamDestroy",
                  static_cast<unsigned int>(CU_STREAM_NON_BLOCKING));
    Handle handle;
    handle.create(api.cudnnCreate, api.cudnnDestroy, "cudnnCreate", "cudnnDestroy");
    checked(api.cudnnSetStream(handle.get(), stream.get()), "cudnnSetStream");
    run(api, handle.get());
    checked(cuCtxSynchronize(), "cuCtxSynchronize");
    checked(api.cudnnSetStream(handle.get(), nullptr), "cudnnSetStream");
}
}  // namespace
#pragma GCC diagnostic pop

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--cached"),
               "usage: execution_dnn_legacy_sdk_test EXACT_CUDNN_TOP_PATH [--cached]");
        execute(argv[1], argc == 3);
        expect(!has_failure && live_resources == 0, "complete normal SDK resource cleanup");
        expect(std::ranges::all_of(legacy_calls, [](auto count) { return count != 0; }),
               "all 25 legacy and algorithm prerequisite functions succeeded");
        expect(legacy_calls[18] == 16 && legacy_calls[19] == 16 && legacy_calls[22] == 2 &&
                   legacy_calls[23] == 2 && legacy_calls[24] == 2,
               "exact native parameter-query and non-idempotent compute inventory");
        std::cout << "{\"kind\":\"legacy_sdk_result\",\"status\":\"passed\",\"iterations\":2,"
                     "\"mapped_parameters\":48,\"normal_cleanup\":true,\"live_resources\":0,"
                     "\"opaque_padding_mathematics_qualified\":false,"
                     "\"temporary_mathematics_qualified\":false,\"full_device_qualified\":false,"
                     "\"saturation_qualified\":false,\"persistent_qualified\":false,"
                     "\"priority_qualified\":false,\"performance_qualified\":false,"
                     "\"full_goal_qualified\":false,\"legacy_calls\":{";
        for (std::size_t index = 0; index != kLegacyNames.size(); ++index)
            std::cout << (index ? "," : "") << '"' << kLegacyNames[index]
                      << "\":" << legacy_calls[index];
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "legacy SDK input failed: " << error.what() << '\n';
        return 1;
    }
}
