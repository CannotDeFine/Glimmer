#include "attention_oracle.h"

#include <cuda.h>
#include <cudnn.h>

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
#include <utility>
#include <vector>

// Ordinary SDK input: compile-only until a separate device run is authorized.
// The pure model knows logical parameters, never opaque native weight offsets.
namespace {
namespace oracle = glimmer::tests::attention_oracle;
constexpr std::size_t kGuard = 256, kMaxWeights = 64 * 1024;
constexpr std::size_t kMaxTemporary = 64 * 1024 * 1024;
bool has_failure{};
unsigned int configurations{}, parameter_queries{}, forward_calls{}, data_calls{}, weight_calls{};
void expect(bool value, const char* message) {
    if (!value) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status)) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (value_ && !has_failure && static_cast<int>(Destroy(std::exchange(value_, {}))))
            has_failure = true;
    }
    Handle get() const {
        return value_;
    }
    Handle* output() {
        return &value_;
    }

   private:
    Handle value_{};
};
using Context = Resource<CUcontext, cuCtxDestroy>;
using Handle = Resource<cudnnHandle_t, cudnnDestroy>;
using Attention = Resource<cudnnAttnDescriptor_t, cudnnDestroyAttnDescriptor>;
using Sequence = Resource<cudnnSeqDataDescriptor_t, cudnnDestroySeqDataDescriptor>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Stream = Resource<CUstream, cuStreamDestroy>;

class Device final {
   public:
    explicit Device(std::size_t bytes) : bytes_(bytes) {
        expect(bytes <= kMaxTemporary, "bound every guarded device allocation");
        initial_.assign(bytes + 2 * kGuard, std::byte{0xa7});
        checked(cuMemAlloc(&base_, initial_.size()), "allocate guarded device storage");
        reset(std::vector<std::byte>(bytes));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() {
        if (base_ && !has_failure && cuMemFree(base_) != CUDA_SUCCESS) has_failure = true;
    }
    void* get() const {
        return reinterpret_cast<void*>(base_ + kGuard);
    }
    std::size_t size() const {
        return bytes_;
    }
    void reset(std::span<const std::byte> bytes) {
        expect(bytes.size() == bytes_, "reset complete device payload");
        std::copy(bytes.begin(), bytes.end(), initial_.begin() + kGuard);
        checked(cuMemcpyHtoD(base_, initial_.data(), initial_.size()), "upload guarded payload");
    }
    template <typename T>
    void values(std::span<const double> values) {
        std::vector<T> typed;
        typed.reserve(values.size());
        for (const auto value : values) typed.push_back(static_cast<T>(value));
        reset(std::as_bytes(std::span{typed}));
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> observed(initial_.size());
        checked(cuMemcpyDtoH(observed.data(), base_, observed.size()),
                "read guarded device payload");
        expect(std::equal(observed.begin(), observed.begin() + kGuard, initial_.begin()) &&
                   std::equal(observed.end() - kGuard, observed.end(), initial_.end() - kGuard),
               "complete prefix and suffix guards remain unchanged");
        return {observed.begin() + kGuard, observed.end() - kGuard};
    }
    void unchanged() const {
        const auto observed = read();
        expect(std::equal(observed.begin(), observed.end(), initial_.begin() + kGuard),
               "every byte of a read-only native input remains unchanged");
    }

   private:
    CUdeviceptr base_{};
    std::size_t bytes_{};
    std::vector<std::byte> initial_;
};
struct Mapping {
    std::array<std::array<std::size_t, 4>, 4> matrices{};
    std::array<std::array<std::size_t, 2>, 4> biases{};
    bool operator==(const Mapping&) const = default;
};
template <typename T>
Mapping mapping(cudnnHandle_t handle, cudnnAttnDescriptor_t attention, const Device& weight) {
    Tensor tensor;
    checked(cudnnCreateTensorDescriptor(tensor.output()), "create original layout output");
    Mapping result;
    std::vector<bool> covered(weight.size());
    for (unsigned int kind = 0; kind != 8; ++kind) {
        struct Address {
            std::uint64_t before{0x123456789abcdef0ULL};
            void* value{};
            std::uint64_t after{0xfedcba9876543210ULL};
        } address;
        checked(cudnnGetMultiHeadAttnWeights(
                    handle, attention, static_cast<cudnnMultiHeadAttnWeightKind_t>(kind),
                    weight.size(), weight.get(), tensor.get(), &address.value),
                "map original projected attention parameter");
        ++parameter_queries;
        expect(address.before == 0x123456789abcdef0ULL && address.after == 0xfedcba9876543210ULL,
               "parameter address preserves adjacent guards");
        const auto base = reinterpret_cast<std::uintptr_t>(weight.get());
        const auto pointer = reinterpret_cast<std::uintptr_t>(address.value);
        expect(pointer >= base && pointer - base < weight.size(),
               "mapped original parameter start");
        const auto offset = pointer - base;
        cudnnDataType_t type{};
        int rank{};
        std::array<int, 5> sizes{-17, -17, -17, -17, -17};
        std::array<int, 5> strides{-19, -19, -19, -19, -19};
        checked(cudnnGetTensorNdDescriptor(tensor.get(), 3, &type, &rank, sizes.data() + 1,
                                           strides.data() + 1),
                "observe original parameter dimensions and strides");
        const bool bias = kind >= 4;
        expect(type == (sizeof(T) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT) && rank == 3 &&
                   sizes == std::array<int, 5>{-17, 1, 2, bias ? 1 : 2, -17} &&
                   strides.front() == -19 && strides.back() == -19 && strides[1] >= 0 &&
                   strides[2] > 0 && strides[3] >= 0 && (bias || strides[3] > 0),
               "complete typed logical parameter shape with guarded original strides");
        for (std::size_t row = 0; row != 2; ++row)
            for (std::size_t column = 0; column != (bias ? 1U : 2U); ++column) {
                const auto index = row * static_cast<std::size_t>(strides[2]) +
                                   column * static_cast<std::size_t>(strides[3]);
                expect(index <= (weight.size() - offset) / sizeof(T),
                       "bound original stride product");
                const auto location = offset + index * sizeof(T);
                expect(location <= weight.size() && sizeof(T) <= weight.size() - location &&
                           location % sizeof(T) == 0,
                       "entire mapped scalar lies inside original weight capacity");
                for (std::size_t byte = 0; byte != sizeof(T); ++byte) {
                    expect(!covered[location + byte],
                           "all trainable scalar intervals are disjoint");
                    covered[location + byte] = true;
                }
                if (bias)
                    result.biases[kind - 4][row] = location;
                else
                    result.matrices[kind][row * 2 + column] = location;
            }
    }
    expect(std::ranges::count(covered, true) == static_cast<std::ptrdiff_t>(24 * sizeof(T)),
           "all 24 projected parameters are mapped, including key bias with zero gradient");
    return result;
}
template <typename T>
std::vector<std::byte> encode(const oracle::Parameters& values, const Mapping& mapping,
                              std::size_t bytes) {
    std::vector<std::byte> output(bytes);
    for (std::size_t kind = 0; kind != 4; ++kind) {
        for (std::size_t index = 0; index != 4; ++index) {
            const auto value = static_cast<T>(values.matrices[kind][index]);
            std::memcpy(output.data() + mapping.matrices[kind][index], &value, sizeof(T));
        }
        for (std::size_t index = 0; index != 2; ++index) {
            const auto value = static_cast<T>(values.biases[kind][index]);
            std::memcpy(output.data() + mapping.biases[kind][index], &value, sizeof(T));
        }
    }
    return output;
}
template <typename T>
double scalar(std::span<const std::byte> bytes, std::size_t offset) {
    expect(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset,
           "bounded scalar observation");
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return static_cast<double>(value);
}
template <typename T>
void compare(double observed, double expected) {
    const double absolute = sizeof(T) == 8 ? 2e-9 : 2e-5;
    const double relative = sizeof(T) == 8 ? 2e-8 : 4e-4;
    expect(std::isfinite(observed) && std::isfinite(expected) &&
               std::abs(observed - expected) <= absolute + relative * std::abs(expected),
           "every observed native scalar agrees with the independent attention oracle");
}
template <typename T>
void compare_tensor(const char* label, const Device& device, std::span<const double> expected) {
    const auto observed = device.read();
    expect(observed.size() == expected.size() * sizeof(T), "complete numerical output span");
    std::cout << ';' << label << '=';
    for (std::size_t index = 0; index != expected.size(); ++index) {
        const auto value = scalar<T>(observed, index * sizeof(T));
        compare<T>(value, expected[index]);
        std::cout << value << ',';
    }
}
template <typename T>
void compare_gradients(const Device& device, const Mapping& mapping,
                       const oracle::Parameters& gradient, const oracle::Parameters& initial,
                       unsigned int repeats, bool is_set) {
    const auto observed = device.read();
    std::cout << ";parameter_gradients=";
    for (std::size_t kind = 0; kind != 4; ++kind) {
        for (std::size_t index = 0; index != 4; ++index) {
            const auto value = scalar<T>(observed, mapping.matrices[kind][index]);
            compare<T>(value, (is_set ? 0 : initial.matrices[kind][index]) +
                                  repeats * gradient.matrices[kind][index]);
            std::cout << value << ',';
        }
        for (std::size_t index = 0; index != 2; ++index) {
            const auto value = scalar<T>(observed, mapping.biases[kind][index]);
            compare<T>(value, (is_set ? 0 : initial.biases[kind][index]) +
                                  repeats * gradient.biases[kind][index]);
            std::cout << value << ',';
        }
    }
    // Unmapped padding is opaque diagnostic data, never a gradient oracle.
    std::uint64_t digest = 1469598103934665603ULL;
    for (const auto byte : observed)
        digest = (digest ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
    std::cout << ";opaque_weight_snapshot_fnv1a=" << digest;
}
void configure_sequence(cudnnSeqDataDescriptor_t descriptor, cudnnDataType_t type, int time) {
    const std::array sizes{time, 1, 1, 2};
    constexpr std::array axes{CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
                              CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
    alignas(double) const std::array<std::byte, 8> padding{};
    checked(cudnnSetSeqDataDescriptor(descriptor, type, 4, sizes.data(), axes.data(), 1, &time,
                                      const_cast<std::byte*>(padding.data())),
            "configure time-major one-batch one-beam sequence");
}
template <typename T>
void run_case(cudnnHandle_t handle, bool residual, bool restricted) {
    ++configurations;
    constexpr auto type = sizeof(T) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    auto inputs = oracle::seeded_inputs();
    inputs.has_residual = residual;
    if (restricted) {
        inputs.lower = {0, 1};
        inputs.upper = {2, 3};
    }
    const auto parameters = oracle::seeded_parameters();
    const auto upstream = oracle::seeded_upstream();
    expect(oracle::max_finite_difference_error(parameters, inputs, upstream) < 1e-8,
           "independent analytic derivatives agree with central differences");
    const auto output = oracle::forward(parameters, inputs);
    const auto gradient = oracle::backward(parameters, inputs, upstream);
    Attention attention;
    checked(cudnnCreateAttnDescriptor(attention.output()), "create projected attention");
    checked(cudnnSetAttnDescriptor(attention.get(), CUDNN_ATTN_ENABLE_PROJ_BIASES, 1, inputs.scale,
                                   type, type, CUDNN_DEFAULT_MATH, nullptr, nullptr, 2, 2, 2, 2, 2,
                                   2, 2, 3, 3, 1, 1),
            "configure projected attention without dropout segments");
    std::array<Sequence, 4> descriptors;
    for (std::size_t index = 0; index != descriptors.size(); ++index) {
        checked(cudnnCreateSeqDataDescriptor(descriptors[index].output()),
                "create sequence metadata");
        configure_sequence(descriptors[index].get(), type, index == 0 || index == 3 ? 2 : 3);
    }
    std::size_t weight_bytes{}, training_workspace{}, reserve_bytes{};
    checked(cudnnGetMultiHeadAttnBuffers(handle, attention.get(), &weight_bytes,
                                         &training_workspace, &reserve_bytes),
            "query original training capacities");
    std::size_t inference_weights{}, inference_workspace{};
    checked(cudnnGetMultiHeadAttnBuffers(handle, attention.get(), &inference_weights,
                                         &inference_workspace, nullptr),
            "query separate inference workspace capacity");
    expect(weight_bytes && weight_bytes <= kMaxWeights && inference_weights == weight_bytes &&
               reserve_bytes && reserve_bytes <= kMaxTemporary &&
               training_workspace <= kMaxTemporary && inference_workspace <= kMaxTemporary,
           "bound original opaque weight, workspace and reserve storage");
    Device weights(weight_bytes), dweights(weight_bytes);
    const auto mapped = mapping<T>(handle, attention.get(), weights);
    expect(mapping<T>(handle, attention.get(), dweights) == mapped,
           "original gradient-space layout matches parameter-space layout");
    auto initial = oracle::seeded_parameters();
    for (auto& matrix : initial.matrices)
        for (auto& value : matrix) value *= .125;
    for (auto& bias : initial.biases)
        for (auto& value : bias) value *= .125;
    weights.reset(encode<T>(parameters, mapped, weight_bytes));
    dweights.reset(encode<T>(initial, mapped, weight_bytes));
    Device queries(4 * sizeof(T)), keys(6 * sizeof(T)), values(6 * sizeof(T));
    Device skip(4 * sizeof(T)), y(4 * sizeof(T)), dy(4 * sizeof(T));
    Device dq(4 * sizeof(T)), dk(6 * sizeof(T)), dv(6 * sizeof(T));
    Device qo_lengths(sizeof(int)), kv_lengths(sizeof(int));
    Device workspace(std::max(training_workspace, inference_workspace)), reserve(reserve_bytes);
    queries.values<T>(inputs.queries);
    keys.values<T>(inputs.keys);
    values.values<T>(inputs.values);
    skip.values<T>(inputs.residual);
    dy.values<T>(upstream.out);
    const std::array qlengths{2}, klengths{3};
    qo_lengths.reset(std::as_bytes(std::span{qlengths}));
    kv_lengths.reset(std::as_bytes(std::span{klengths}));
    // Full native paths inspect parent maxQ=3, not merely actual Q.TIME=2.
    const std::array lower{inputs.lower[0], inputs.lower[1], 0};
    const std::array upper{inputs.upper[0], inputs.upper[1], 3};
    const auto forward = [&](int current, std::size_t work_bytes, std::size_t saved_bytes,
                             void* saved) {
        checked(cudnnMultiHeadAttnForward(
                    handle, attention.get(), current, lower.data(), upper.data(),
                    static_cast<const int*>(qo_lengths.get()),
                    static_cast<const int*>(kv_lengths.get()), descriptors[0].get(), queries.get(),
                    residual ? skip.get() : nullptr, descriptors[1].get(), keys.get(),
                    descriptors[2].get(), values.get(), descriptors[3].get(), y.get(), weight_bytes,
                    weights.get(), work_bytes, workspace.get(), saved_bytes, saved),
                "original projected attention forward");
        ++forward_calls;
    };
    for (unsigned int iteration = 0; iteration != 3; ++iteration) {
        const bool is_set = iteration == 2;
        if (is_set) dweights.reset(encode<T>(initial, mapped, weight_bytes));
        for (auto* result : {&y, &dq, &dk, &dv})
            result->reset(std::vector<std::byte>(result->size(), std::byte{0x37}));
        reserve.reset(std::vector<std::byte>(reserve.size()));
        // Pageable HtoD may finish staging before DMA. The explicit barrier
        // orders initialization before either default or nonblocking streams.
        checked(cuCtxSynchronize(), "prove all input/reset uploads complete before compute");
        forward(-1, training_workspace, reserve_bytes, reserve.get());
        checked(
            cudnnMultiHeadAttnBackwardData(
                handle, attention.get(), lower.data(), upper.data(),
                static_cast<const int*>(qo_lengths.get()),
                static_cast<const int*>(kv_lengths.get()), descriptors[3].get(), dy.get(),
                descriptors[0].get(), dq.get(), queries.get(), descriptors[1].get(), dk.get(),
                keys.get(), descriptors[2].get(), dv.get(), values.get(), weight_bytes,
                weights.get(), training_workspace, workspace.get(), reserve_bytes, reserve.get()),
            "original attention data gradients after matching forward");
        ++data_calls;
        checked(cudnnMultiHeadAttnBackwardWeights(
                    handle, attention.get(), is_set ? CUDNN_WGRAD_MODE_SET : CUDNN_WGRAD_MODE_ADD,
                    descriptors[0].get(), queries.get(), descriptors[1].get(), keys.get(),
                    descriptors[2].get(), values.get(), descriptors[3].get(), dy.get(),
                    weight_bytes, weights.get(), dweights.get(), training_workspace,
                    workspace.get(), reserve_bytes, reserve.get()),
                "original attention complete parameter gradients");
        ++weight_calls;
        checked(cuCtxSynchronize(), "complete native forward/backward sequence");
        std::cout << "attention_case=" << configurations << ";iteration=" << iteration
                  << ";type=" << static_cast<int>(type) << ";residual=" << residual
                  << ";restricted=" << restricted << ";mode=" << (is_set ? "SET" : "ADD");
        compare_tensor<T>("out", y, output.out);
        compare_tensor<T>("dq", dq, gradient.queries);
        compare_tensor<T>("dk", dk, gradient.keys);
        compare_tensor<T>("dv", dv, gradient.values);
        compare_gradients<T>(dweights, mapped, gradient.parameters, initial,
                             is_set ? 1 : iteration + 1, is_set);
        for (const auto* input :
             {&queries, &keys, &values, &skip, &dy, &weights, &qo_lengths, &kv_lengths})
            input->unchanged();
        (void)workspace.read();
        (void)reserve.read();
        std::cout << ";guards=passed;residual_gradient_separate=1\n";
    }
    y.reset(std::vector<std::byte>(y.size(), std::byte{0x39}));
    checked(cuCtxSynchronize(), "order incremental output initialization");
    forward(0, inference_workspace, 0, nullptr);
    forward(1, inference_workspace, 0, nullptr);
    checked(cuCtxSynchronize(), "complete incremental inference");
    std::cout << "attention_incremental=" << configurations;
    compare_tensor<T>("out", y, output.out);
    for (const auto* input : {&queries, &keys, &values, &skip, &weights, &qo_lengths, &kv_lengths})
        input->unchanged();
    (void)workspace.read();
    std::cout << ";guards=passed\n";
}
}  // namespace

int main() {
    try {
        expect(cudnnGetVersion() == 91002, "input requires the statically audited cuDNN 9.10.2");
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select one device");
        {
            Context context;
            checked(cuCtxCreate(context.output(), 0, device), "create diagnostic context");
            Handle handle;
            checked(cudnnCreate(handle.output()), "create original cuDNN handle");
            std::cout << std::setprecision(17);
            for (const bool residual : {false, true})
                for (const bool restricted : {false, true})
                    run_case<float>(handle.get(), residual, restricted);
            Stream stream;
            checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING),
                    "create explicit stream");
            checked(cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                    "bind explicit original handle stream");
            for (const bool residual : {false, true})
                for (const bool restricted : {false, true})
                    run_case<double>(handle.get(), residual, restricted);
            checked(cudnnSetStream(handle.get(), nullptr), "restore default handle stream");
        }
        expect(!has_failure && configurations == 8 && parameter_queries == 128 &&
                   forward_calls == 40 && data_calls == 24 && weight_calls == 24,
               "complete bounded SDK input and exact call receipt");
        std::cout << "cuDNN attention SDK input passed: configurations=8;parameter_queries=128;"
                     "forward_calls=40;data_calls=24;weight_calls=24;logical_gradient_checks=576;"
                     "opaque_padding_is_diagnostic=1;full_device=0;priority=0\n";
        return 0;
    } catch (const std::exception& error) {
        has_failure = true;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
