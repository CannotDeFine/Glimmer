#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Ordinary SDK input, compiled but not automatically executed by CTest.
// Zero FP32 weights give a layout-independent LSTM forward/data-gradient oracle.
// Weight gradients are complete guarded snapshots ONLY: opaque padding and
// parameter offsets require the separate WeightParams group before numerical
// qualification. This is neither full RNN correctness nor a performance test.
namespace {
bool has_failure{};
unsigned int forward_calls{}, backward_data_calls{}, backward_weight_calls{}, build_calls{};
constexpr std::size_t kGuard = 256;
constexpr std::size_t kMaxTemporaryBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaxWeightBytes = 64U * 1024U;
constexpr int kWidth = 2, kBatch = 2, kSequence = 3;

void expect(bool condition, const char* message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0) {
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
        if (value_ && !has_failure && static_cast<int>(Destroy(std::exchange(value_, {}))) != 0)
            has_failure = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }

   private:
    Handle value_{};
};
using Context = Resource<CUcontext, cuCtxDestroy>;
using Handle = Resource<cudnnHandle_t, cudnnDestroy>;
using Rnn = Resource<cudnnRNNDescriptor_t, cudnnDestroyRNNDescriptor>;
using Data = Resource<cudnnRNNDataDescriptor_t, cudnnDestroyRNNDataDescriptor>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Stream = Resource<CUstream, cuStreamDestroy>;

class Device final {
   public:
    explicit Device(std::size_t bytes)
        : bytes_(bytes), initial_(bytes + 2 * kGuard, std::byte{0xa7}) {
        expect(bytes <= kMaxTemporaryBytes, "bounded ordinary device allocation");
        checked(cuMemAlloc(&base_, initial_.size()), "allocate guarded device storage");
        reset(std::vector<std::byte>(bytes, std::byte{}));
    }
    explicit Device(const std::vector<float>& values) : Device(values.size() * sizeof(float)) {
        reset(std::as_bytes(std::span{values}));
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
    void reset(std::span<const std::byte> value) {
        expect(value.size() == bytes_, "complete payload reset");
        std::copy(value.begin(), value.end(), initial_.begin() + kGuard);
        checked(cuMemcpyHtoD(base_, initial_.data(), initial_.size()), "upload guarded payload");
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> value(initial_.size());
        checked(cuMemcpyDtoH(value.data(), base_, value.size()),
                "download complete guarded payload");
        expect(std::equal(value.begin(), value.begin() + kGuard, initial_.begin()) &&
                   std::equal(value.end() - kGuard, value.end(), initial_.end() - kGuard),
               "native operation preserves both complete guard regions");
        return {value.begin() + kGuard, value.end() - kGuard};
    }
    void unchanged() const {
        const auto value = read();
        expect(std::equal(value.begin(), value.end(), initial_.begin() + kGuard),
               "original input payload remains byte-identical");
    }
    void near(const std::vector<float>& expected) const {
        const auto value = read();
        expect(value.size() == expected.size() * sizeof(float), "complete numerical output extent");
        for (std::size_t i = 0; i != expected.size(); ++i) {
            float actual{};
            std::memcpy(&actual, value.data() + i * sizeof(float), sizeof(actual));
            expect(std::isfinite(actual) &&
                       std::abs(actual - expected[i]) <= 2e-5F + 2e-4F * std::abs(expected[i]),
                   "every output element matches independent zero-weight LSTM oracle");
        }
    }

   private:
    CUdeviceptr base_{};
    std::size_t bytes_{};
    std::vector<std::byte> initial_;
};

std::vector<float> values(std::size_t count, float offset) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i != count; ++i) result[i] = offset + 0.03125F * static_cast<float>(i);
    return result;
}
void snapshot(const char* name, const Device& buffer) {
    const auto bytes = buffer.read();
    std::cout << name << "=";
    for (auto byte : bytes)
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << std::to_integer<unsigned>(byte);
    std::cout << std::dec << ";";
}
std::size_t offset(cudnnRNNDataLayout_t layout, const std::array<int, 2>& lengths, int time,
                   int batch, int channel) {
    if (layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
        return static_cast<std::size_t>((batch * kSequence + time) * kWidth + channel);
    if (layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED)
        return static_cast<std::size_t>((time * kBatch + batch) * kWidth + channel);
    std::size_t previous{};
    for (int t = 0; t < time; ++t)
        for (auto length : lengths) previous += t < length;
    return (previous + static_cast<std::size_t>(batch)) * kWidth + channel;
}

void compute_case(cudnnHandle_t handle, cudnnRNNDataLayout_t layout, unsigned int state_case,
                  bool device_lengths) {
    Rnn rnn;
    Data x_descriptor, y_descriptor;
    Tensor state_descriptor;
    checked(cudnnCreateRNNDescriptor(rnn.output()), "create original RNN descriptor");
    checked(cudnnCreateRNNDataDescriptor(x_descriptor.output()), "create original input metadata");
    checked(cudnnCreateRNNDataDescriptor(y_descriptor.output()), "create original output metadata");
    checked(cudnnCreateTensorDescriptor(state_descriptor.output()),
            "create original state metadata");
    checked(cudnnSetRNNDescriptor_v8(rnn.get(), CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                     CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                     CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                     CUDNN_DEFAULT_MATH, kWidth, kWidth, kWidth, 1, nullptr, 0),
            "configure tiny FP32 LSTM");
    const bool packed = layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
    const std::array lengths{kSequence, packed ? kSequence - 1 : kSequence};
    float padding{};
    for (auto descriptor : {x_descriptor.get(), y_descriptor.get()})
        checked(cudnnSetRNNDataDescriptor(descriptor, CUDNN_DATA_FLOAT, layout, kSequence, kBatch,
                                          kWidth, lengths.data(), &padding),
                "configure sequence layout without opaque weight assumptions");
    const std::array dimensions{1, kBatch, kWidth};
    const std::array strides{kBatch * kWidth, kWidth, 1};
    checked(cudnnSetTensorNdDescriptor(state_descriptor.get(), CUDNN_DATA_FLOAT, 3,
                                       dimensions.data(), strides.data()),
            "configure contiguous hidden and cell states");
    std::size_t weight_bytes{}, workspace_bytes{}, reserve_bytes{};
    checked(cudnnGetRNNWeightSpaceSize(handle, rnn.get(), &weight_bytes), "query weight capacity");
    checked(cudnnGetRNNTempSpaceSizes(handle, rnn.get(), CUDNN_FWD_MODE_TRAINING,
                                      x_descriptor.get(), &workspace_bytes, &reserve_bytes),
            "query training temporary capacities");
    expect(weight_bytes && weight_bytes <= kMaxWeightBytes && weight_bytes % sizeof(float) == 0 &&
               workspace_bytes <= kMaxTemporaryBytes && reserve_bytes <= kMaxTemporaryBytes,
           "bound native-reported allocation sizes before allocation");
    const auto count = static_cast<std::size_t>(lengths[0] + lengths[1]) * kWidth;
    const auto input = values(count, -0.2F);
    const auto gradient = values(count, 0.1F);
    const auto initial_hidden = values(kBatch * kWidth, -0.1F);
    const auto initial_cell = values(kBatch * kWidth, 0.3F);
    const auto hidden_gradient = values(kBatch * kWidth, 0.15F);
    const auto cell_gradient = values(kBatch * kWidth, -0.05F);
    const std::vector<float> output_sentinel(count, -77.0F),
        state_sentinel(kBatch * kWidth, -77.0F);
    Device x(input), dy(gradient), hx(initial_hidden), cx(initial_cell), dhy(hidden_gradient),
        dcy(cell_gradient), y(output_sentinel), dx(output_sentinel), hy(state_sentinel),
        cy(state_sentinel), dhx(state_sentinel), dcx(state_sentinel), weight(weight_bytes),
        dweight(weight_bytes), workspace(workspace_bytes), reserve(reserve_bytes),
        sequence_lengths(sizeof(lengths));
    sequence_lengths.reset(std::as_bytes(std::span{lengths}));
    const auto* lengths_pointer =
        device_lengths ? static_cast<const std::int32_t*>(sequence_lengths.get()) : nullptr;
    const bool has_initial = state_case != 1 && state_case != 3;
    const bool has_output = state_case != 2 && state_case != 3;
    const bool has_gradient = state_case != 2 && state_case != 3;
    const bool has_final_gradient = state_case != 3;
    auto state = state_case == 3 ? nullptr : state_descriptor.get();
    std::vector<float> expected_y(count), expected_hidden(kBatch * kWidth),
        expected_cell(kBatch * kWidth), expected_cell_gradient(kBatch * kWidth);
    for (int batch = 0; batch != kBatch; ++batch) {
        for (int channel = 0; channel != kWidth; ++channel) {
            const auto index = static_cast<std::size_t>(batch * kWidth + channel);
            float cell = has_initial ? initial_cell[index] : 0.0F;
            float factor = 1.0F;
            float derivative{};
            for (int time = 0; time != lengths[batch]; ++time) {
                cell *= 0.5F;
                factor *= 0.5F;
                const auto element = offset(layout, lengths, time, batch, channel);
                const auto tanh_cell = std::tanh(cell);
                expected_y[element] = 0.5F * tanh_cell;
                derivative += gradient[element] * 0.5F * (1.0F - tanh_cell * tanh_cell) * factor;
            }
            expected_cell[index] = cell;
            expected_hidden[index] = 0.5F * std::tanh(cell);
            if (has_gradient)
                derivative += hidden_gradient[index] * 0.5F *
                                  (1.0F - std::tanh(cell) * std::tanh(cell)) * factor +
                              cell_gradient[index] * factor;
            expected_cell_gradient[index] = derivative;
        }
    }
    for (unsigned int repeat = 0; repeat != 2; ++repeat) {
        y.reset(std::as_bytes(std::span{output_sentinel}));
        dx.reset(std::as_bytes(std::span{output_sentinel}));
        for (auto* output : {&hy, &cy, &dhx, &dcx})
            output->reset(std::as_bytes(std::span{state_sentinel}));
        checked(cudnnRNNForward(handle, rnn.get(), CUDNN_FWD_MODE_TRAINING, lengths_pointer,
                                x_descriptor.get(), x.get(), y_descriptor.get(), y.get(), state,
                                has_initial ? hx.get() : nullptr, has_output ? hy.get() : nullptr,
                                state, has_initial ? cx.get() : nullptr,
                                has_output ? cy.get() : nullptr, weight_bytes, weight.get(),
                                workspace_bytes, workspace.get(), reserve_bytes, reserve.get()),
                "execute original training forward");
        ++forward_calls;
        checked(cudnnRNNBackwardData_v8(
                    handle, rnn.get(), lengths_pointer, y_descriptor.get(), y.get(), dy.get(),
                    x_descriptor.get(), dx.get(), state, has_initial ? hx.get() : nullptr,
                    has_gradient ? dhy.get() : nullptr, has_final_gradient ? dhx.get() : nullptr,
                    state, has_initial ? cx.get() : nullptr, has_gradient ? dcy.get() : nullptr,
                    has_final_gradient ? dcx.get() : nullptr, weight_bytes, weight.get(),
                    workspace_bytes, workspace.get(), reserve_bytes, reserve.get()),
                "execute original backward data after matching training forward");
        ++backward_data_calls;
        // Do not reset dweight between repeats: preserve actual native ADD input.
        checked(cudnnRNNBackwardWeights_v8(handle, rnn.get(), CUDNN_WGRAD_MODE_ADD, lengths_pointer,
                                           x_descriptor.get(), x.get(), state,
                                           has_initial ? hx.get() : nullptr, y_descriptor.get(),
                                           y.get(), weight_bytes, dweight.get(), workspace_bytes,
                                           workspace.get(), reserve_bytes, reserve.get()),
                "execute original ADD backward weights after matching backward data");
        ++backward_weight_calls;
        checked(cuCtxSynchronize(), "complete explicit diagnostic computation");
        y.near(expected_y);
        dx.near(std::vector<float>(count, 0.0F));
        if (has_output) {
            hy.near(expected_hidden);
            cy.near(expected_cell);
        } else {
            hy.unchanged();
            cy.unchanged();
        }
        if (has_final_gradient) {
            dhx.near(std::vector<float>(kBatch * kWidth, 0.0F));
            dcx.near(expected_cell_gradient);
        } else {
            dhx.unchanged();
            dcx.unchanged();
        }
        for (const auto* input_buffer : {&x, &dy, &hx, &cx, &dhy, &dcy, &weight, &sequence_lengths})
            input_buffer->unchanged();
        (void)workspace.read();
        (void)reserve.read();
        std::cout << "rnn_compute_case=" << backward_weight_calls
                  << "; layout=" << static_cast<int>(layout) << "; state_case=" << state_case
                  << "; device_lengths=" << device_lengths << "; repeat=" << repeat << ";";
        for (const auto& named : {std::pair{"y", &y},
                                  {"hy", &hy},
                                  {"cy", &cy},
                                  {"dx", &dx},
                                  {"dhx", &dhx},
                                  {"dcx", &dcx},
                                  {"dweight_diagnostic_only", &dweight}})
            snapshot(named.first, *named.second);
        std::cout << " dweight_numerically_qualified=false\n";
    }
}

void build_dynamic(cudnnHandle_t handle) {
    Rnn rnn;
    checked(cudnnCreateRNNDescriptor(rnn.output()), "create dynamic RNN descriptor");
    checked(cudnnSetRNNDescriptor_v8(rnn.get(), CUDNN_RNN_ALGO_PERSIST_DYNAMIC, CUDNN_LSTM,
                                     CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                     CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                     CUDNN_DEFAULT_MATH, 32, 32, 32, 1, nullptr, 0),
            "configure standalone dynamic build input");
    const auto status = cudnnBuildRNNDynamic(handle, rnn.get(), kBatch);
    ++build_calls;
    std::cout << "rnn_build_case=1; status=" << static_cast<int>(status)
              << "; mini_batch=" << kBatch << "; numerical_qualified=false\n";
    checked(status, "build original device-specific RNN library");
    checked(cuCtxSynchronize(), "finish standalone dynamic build input");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool is_build = argc == 2 && std::string_view(argv[1]) == "--build-dynamic";
        expect(argc == 1 || is_build,
               "usage: execution_dnn_rnn_compute_gpu_test [--build-dynamic]");
        {
            Context context;
            Handle handle;
            Stream stream;
            checked(cuInit(0), "initialize Driver");
            CUdevice device{};
            checked(cuDeviceGet(&device, 0), "select diagnostic device");
            checked(cuCtxCreate(context.output(), 0, device), "create owned diagnostic context");
            checked(cudnnCreate(handle.output()), "create original library handle");
            if (is_build) {
                build_dynamic(handle.get());
            } else {
                checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING),
                        "create explicit stream");
                for (unsigned int stream_case = 0; stream_case != 2; ++stream_case) {
                    checked(cudnnSetStream(handle.get(), stream_case ? stream.get() : nullptr),
                            "select original stream");
                    for (auto layout : {CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                                        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED})
                        for (unsigned int state_case = 0; state_case != 4; ++state_case)
                            compute_case(handle.get(), layout, state_case, stream_case != 0);
                }
                checked(cudnnSetStream(handle.get(), nullptr), "release handle stream dependency");
                expect(
                    forward_calls == 48 && backward_data_calls == 48 && backward_weight_calls == 48,
                    "complete bounded numerical input call receipt");
            }
        }
        expect(!has_failure, "all original cleanup operations succeeded");
        std::cout
            << "cuDNN RNN computation input passed: forward_calls=" << forward_calls
            << "; backward_data_calls=" << backward_data_calls
            << "; backward_weight_calls=" << backward_weight_calls
            << "; build_calls=" << build_calls
            << "; zero_weight_forward_data_oracle=" << (is_build ? "false" : "true")
            << "; dweight_numerically_qualified=false;"
               " full_rnn_qualified=false; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
