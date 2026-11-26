#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Ordinary SDK input only. No injected dispatch, automatic GPU CTest, timing,
// full-device or persistent-kernel claim is part of compilation.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool cleanup_failed{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (release() != 0) cleanup_failed = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(release(), "native cleanup");
    }

   private:
    int release() noexcept {
        if (!value_) return 0;
        return static_cast<int>(Destroy(std::exchange(value_, Handle{})));
    }
    Handle value_{};
};
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::array kShape{4, 3, 2, 2};
constexpr std::array kStrides{16, 4, 2, 1};
constexpr std::size_t kExtent = 60, kGuard = 32, kAuxiliaryGuard = 256;
constexpr std::size_t kMaximumAuxiliaryBytes = 64 * 1024 * 1024;
constexpr double kEpsilon = 0.03125;
constexpr std::array kModes{CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_PER_ACTIVATION};
constexpr std::size_t position(int n, int c, int h, int w) {
    return static_cast<std::size_t>(n * 16 + c * 4 + h * 2 + w);
}
template <typename Function>
void each(Function function) {
    for (int n = 0; n != 4; ++n)
        for (int c = 0; c != 3; ++c)
            for (int h = 0; h != 2; ++h)
                for (int w = 0; w != 2; ++w) function(n, c, h, w);
}
template <typename Value>
struct Descriptors {
    Tensor tensor;
    std::array<Tensor, 2> parameters;
    Descriptors() {
        constexpr auto type = std::is_same_v<Value, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
        checked(cudnnCreateTensorDescriptor(tensor.output()), "create data tensor");
        checked(cudnnSetTensorNdDescriptor(tensor.get(), type, 4, kShape.data(), kStrides.data()),
                "set padded HW-packed data tensor");
        std::size_t bytes{};
        checked(cudnnGetTensorSizeInBytes(tensor.get(), &bytes), "query data extent");
        expect(bytes == kExtent * sizeof(Value), "independent strided data byte hull");
        for (std::size_t i = 0; i != parameters.size(); ++i) {
            checked(cudnnCreateTensorDescriptor(parameters[i].output()), "create parameter tensor");
            // Deliberately do not call the reference-unimplemented DeriveBN.
            checked(cudnnSetTensor4dDescriptor(parameters[i].get(), CUDNN_TENSOR_NCHW, type, 1, 3,
                                               i == 0 ? 1 : 2, i == 0 ? 1 : 2),
                    "set explicit BN parameter tensor");
        }
    }
    void close() {
        for (auto& descriptor : parameters) descriptor.close();
        tensor.close();
    }
};
template <typename Value>
struct Buffer {
    Memory memory;
    std::vector<Value> expected;
    std::vector<bool> logical;
    explicit Buffer(std::size_t count)
        : expected(count + 2 * kGuard, Value{97}), logical(expected.size(), false) {
        checked(cuMemAlloc(memory.output(), expected.size() * sizeof(Value)),
                "allocate guarded tensor");
    }
    Value& at(std::size_t index) {
        return expected[kGuard + index];
    }
    Value at(std::size_t index) const {
        return expected[kGuard + index];
    }
    void initialize(std::size_t index, Value value) {
        at(index) = value;
        logical[kGuard + index] = true;
    }
    void* data() const {
        return reinterpret_cast<void*>(memory.get() + kGuard * sizeof(Value));
    }
    void upload() const {
        checked(cuMemcpyHtoD(memory.get(), expected.data(), expected.size() * sizeof(Value)),
                "upload complete guarded tensor");
    }
    std::vector<Value> snapshot() const {
        auto actual = expected;
        checked(cuMemcpyDtoH(actual.data(), memory.get(), actual.size() * sizeof(Value)),
                "download complete guarded tensor");
        return actual;
    }
    void verify_unchanged(const std::vector<Value>& before) const {
        expect(snapshot() == before, "native read-only or omitted state remains byte-exact");
    }
    void verify(bool is_output) const {
        const auto actual = snapshot();
        for (std::size_t i = 0; i != actual.size(); ++i) {
            if (!is_output || !logical[i]) {
                expect(actual[i] == expected[i],
                       "read-only values, guards and stride holes are exact");
                continue;
            }
            const long double tolerance = std::is_same_v<Value, double> ? 5.0e-10L : 8.0e-4L;
            expect(std::isfinite(actual[i]) &&
                       std::abs(static_cast<long double>(actual[i]) - expected[i]) <=
                           tolerance * (1 + std::abs(static_cast<long double>(expected[i]))),
                   "complete BN output or state matches the independent oracle");
        }
    }
};
struct Auxiliary {
    Memory memory;
    std::size_t bytes;
    std::vector<std::byte> initial;
    static std::size_t allocation_size(std::size_t count) {
        expect(count <= kMaximumAuxiliaryBytes, "bounded workspace/reserve size");
        return count + 2 * kAuxiliaryGuard;
    }
    explicit Auxiliary(std::size_t count)
        : bytes(count), initial(allocation_size(count), std::byte{0xa7}) {
        checked(cuMemAlloc(memory.output(), initial.size()), "allocate guarded auxiliary");
        checked(cuMemcpyHtoD(memory.get(), initial.data(), initial.size()),
                "initialize auxiliary once");
    }
    void* data() const {
        return reinterpret_cast<void*>(memory.get() + kAuxiliaryGuard);
    }
    void verify_guards() const {
        auto actual = initial;
        checked(cuMemcpyDtoH(actual.data(), memory.get(), actual.size()), "read auxiliary guards");
        for (std::size_t i = 0; i != actual.size(); ++i)
            if (i < kAuxiliaryGuard || i >= kAuxiliaryGuard + bytes)
                expect(actual[i] == initial[i], "workspace/reserve guard remains unchanged");
    }
};
struct Statistics {
    long double mean{}, variance{}, inverse{}, sum_dy{}, sum_dy_normalized{};
};
template <typename Value>
void run(cudnnHandle_t handle, CUstream stream, const Descriptors<Value>& descriptors,
         std::size_t mode_index, unsigned& forwards, unsigned& backwards) {
    const auto mode = kModes[mode_index];
    const auto tensor = descriptors.tensor.get();
    const auto parameters = descriptors.parameters[mode_index].get();
    const std::size_t groups = mode_index == 0 ? 3 : 12;
    Buffer<Value> x(kExtent), y(kExtent), dy(kExtent), dx(kExtent);
    Buffer<Value> scale(groups), bias(groups), running_mean(groups), running_variance(groups),
        saved_mean(groups), saved_inverse(groups), dscale(groups), dbias(groups);
    std::array<std::vector<std::size_t>, 12> members;
    each([&](int n, int c, int h, int w) {
        const auto index = position(n, c, h, w);
        const auto group = static_cast<std::size_t>(mode_index == 0 ? c : c * 4 + h * 2 + w);
        members[group].push_back(index);
        x.initialize(index, static_cast<Value>(n * 13 + c * 7 + h * 3 + w * 5 - 18) * Value{0.125});
        y.initialize(index, static_cast<Value>(n - c + h - w) * Value{0.0625});
        dy.initialize(
            index, static_cast<Value>((n * 7 + c * 3 + h * 5 + w * 11) % 17 - 8) * Value{0.0625});
        dx.initialize(index, static_cast<Value>(c - h + w - n) * Value{0.03125});
    });
    for (std::size_t group = 0; group != groups; ++group) {
        const auto g = static_cast<Value>(group);
        scale.initialize(group, Value{0.75} + g * Value{0.0625});
        bias.initialize(group, Value{-0.125} + g * Value{0.03125});
        running_mean.initialize(group, Value{-0.25});
        running_variance.initialize(group, Value{1.5});
        saved_mean.initialize(group, Value{-7});
        saved_inverse.initialize(group, Value{-8});
        dscale.initialize(group, Value{0.125});
        dbias.initialize(group, Value{-0.0625});
    }
    const std::array buffers{&x,
                             &y,
                             &dy,
                             &dx,
                             &scale,
                             &bias,
                             &running_mean,
                             &running_variance,
                             &saved_mean,
                             &saved_inverse,
                             &dscale,
                             &dbias};
    for (const auto* buffer : buffers) buffer->upload();
    std::size_t forward_bytes{}, backward_bytes{}, reserve_bytes{};
    checked(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                handle, mode, CUDNN_BATCHNORM_OPS_BN, tensor, nullptr, tensor, parameters, nullptr,
                &forward_bytes),
            "query forward workspace");
    checked(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                handle, mode, CUDNN_BATCHNORM_OPS_BN, tensor, nullptr, tensor, nullptr, tensor,
                parameters, nullptr, &backward_bytes),
            "query backward workspace");
    checked(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                handle, mode, CUDNN_BATCHNORM_OPS_BN, nullptr, tensor, &reserve_bytes),
            "query training reserve");
    Auxiliary workspace(std::max(forward_bytes, backward_bytes)), reserve(reserve_bytes);
    constexpr Value alpha{0.75}, beta{-0.25}, alpha_data{1.25}, beta_data{0.25},
        alpha_parameter{0.5}, beta_parameter{-0.125};
    for (unsigned repetition = 0; repetition != 3; ++repetition) {
        const bool use_statistics = repetition != 2;
        const double average_factor = repetition == 0 ? 1.0 : 0.375;
        std::array<std::vector<Value>, 12> before;
        for (std::size_t i = 0; i != buffers.size(); ++i) before[i] = buffers[i]->snapshot();
        std::array<Statistics, 12> expected;
        for (std::size_t group = 0; group != groups; ++group) {
            auto& stats = expected[group];
            const auto count = static_cast<long double>(members[group].size());
            for (const auto index : members[group]) stats.mean += x.at(index);
            stats.mean /= count;
            for (const auto index : members[group]) {
                const auto centered = static_cast<long double>(x.at(index)) - stats.mean;
                stats.variance += centered * centered;
            }
            stats.variance /= count;
            stats.inverse = 1 / std::sqrt(stats.variance + kEpsilon);
            for (const auto index : members[group]) {
                const auto normalized =
                    (static_cast<long double>(x.at(index)) - stats.mean) * stats.inverse;
                y.at(index) = static_cast<Value>(
                    alpha * (scale.at(group) * normalized + bias.at(group)) + beta * y.at(index));
                stats.sum_dy += dy.at(index);
                stats.sum_dy_normalized += dy.at(index) * normalized;
            }
            if (use_statistics) {
                running_mean.at(group) = static_cast<Value>(
                    (1 - average_factor) * running_mean.at(group) + average_factor * stats.mean);
                running_variance.at(group) =
                    static_cast<Value>((1 - average_factor) * running_variance.at(group) +
                                       average_factor * stats.variance * count / (count - 1));
                saved_mean.at(group) = static_cast<Value>(stats.mean);
                saved_inverse.at(group) = static_cast<Value>(stats.inverse);
            }
        }
        checked(cudnnBatchNormalizationForwardTrainingEx(
                    handle, mode, CUDNN_BATCHNORM_OPS_BN, &alpha, &beta, tensor, x.data(), nullptr,
                    nullptr, tensor, y.data(), parameters, scale.data(), bias.data(),
                    average_factor, use_statistics ? running_mean.data() : nullptr,
                    use_statistics ? running_variance.data() : nullptr, kEpsilon,
                    use_statistics ? saved_mean.data() : nullptr,
                    use_statistics ? saved_inverse.data() : nullptr, nullptr, workspace.data(),
                    workspace.bytes, reserve.data(), reserve.bytes),
                "original BN Ex forward");
        checked(cuStreamSynchronize(stream), "complete forward and saved state");
        ++forwards;
        for (std::size_t i = 0; i != buffers.size(); ++i) {
            const auto* buffer = buffers[i];
            const bool may_write =
                buffer == &y ||
                (use_statistics && (buffer == &running_mean || buffer == &running_variance ||
                                    buffer == &saved_mean || buffer == &saved_inverse));
            if (!may_write) buffer->verify_unchanged(before[i]);
            buffer->verify(buffer != &x && buffer != &dy && buffer != &scale && buffer != &bias);
            before[i] = buffer->snapshot();
        }
        workspace.verify_guards();
        reserve.verify_guards();
        // Do not clear/recreate reserve or rerun forward before backward. The
        // same application allocations carry the actual original SDK state.
        for (std::size_t group = 0; group != groups; ++group) {
            const auto& stats = expected[group];
            const auto count = static_cast<long double>(members[group].size());
            for (const auto index : members[group]) {
                const auto normalized =
                    (static_cast<long double>(x.at(index)) - stats.mean) * stats.inverse;
                const auto derivative =
                    scale.at(group) * stats.inverse / count *
                    (count * dy.at(index) - stats.sum_dy - normalized * stats.sum_dy_normalized);
                dx.at(index) =
                    static_cast<Value>(alpha_data * derivative + beta_data * dx.at(index));
            }
            dscale.at(group) = static_cast<Value>(alpha_parameter * stats.sum_dy_normalized +
                                                  beta_parameter * dscale.at(group));
            dbias.at(group) = static_cast<Value>(alpha_parameter * stats.sum_dy +
                                                 beta_parameter * dbias.at(group));
        }
        checked(cudnnBatchNormalizationBackwardEx(
                    handle, mode, CUDNN_BATCHNORM_OPS_BN, &alpha_data, &beta_data, &alpha_parameter,
                    &beta_parameter, tensor, x.data(), nullptr, nullptr, tensor, dy.data(), nullptr,
                    nullptr, tensor, dx.data(), parameters, scale.data(), nullptr, dscale.data(),
                    dbias.data(), kEpsilon, use_statistics ? saved_mean.data() : nullptr,
                    use_statistics ? saved_inverse.data() : nullptr, nullptr, workspace.data(),
                    workspace.bytes, reserve.data(), reserve.bytes),
                "original BN Ex backward");
        checked(cuStreamSynchronize(stream), "complete backward");
        ++backwards;
        for (std::size_t i = 0; i != buffers.size(); ++i) {
            const auto* buffer = buffers[i];
            if (buffer != &dx && buffer != &dscale && buffer != &dbias)
                buffer->verify_unchanged(before[i]);
            buffer->verify(buffer != &x && buffer != &dy && buffer != &scale && buffer != &bias);
        }
        workspace.verify_guards();
        reserve.verify_guards();
    }
    reserve.memory.close();
    workspace.memory.close();
    for (auto* buffer : buffers) buffer->memory.close();
}
}  // namespace

int main() {
    try {
        Descriptors<float> single;
        Descriptors<double> dual;
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create cuDNN handle");
        checked(cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                "bind explicit stream");
        unsigned forwards{}, backwards{};
        for (std::size_t mode = 0; mode != kModes.size(); ++mode) {
            run(handle.get(), stream.get(), single, mode, forwards, backwards);
            run(handle.get(), stream.get(), dual, mode, forwards, backwards);
        }
        expect(forwards == 12 && backwards == 12, "complete type/mode/repeated-state matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        dual.close();
        single.close();
        expect(!cleanup_failed, "checked once-only cleanup");
        std::cout
            << "cuDNN BN Ex input passed: forward_calls=" << forwards
            << "; backward_calls=" << backwards
            << "; fused_qualified=false; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
