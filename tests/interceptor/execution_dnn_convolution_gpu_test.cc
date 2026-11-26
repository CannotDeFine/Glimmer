#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// An ordinary installed-SDK application, not a Glimmer fixture or an automatic
// GPU CTest. Compile now; execute compute modes only with the bounded grouped
// GPU collector. The separate metadata-only mode makes no explicit Driver call.
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
        throw std::runtime_error(std::string(operation) +
                                 " failed: " + std::to_string(static_cast<int>(status)));
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        const auto status = release();
        if (status != 0) {
            cleanup_failed = true;
            try {
                std::cerr << "Native cleanup failed without retry: " << status << '\n';
            } catch (...) {
            }
        }
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
        // An uncertain failed native destruction is never attempted again.
        const auto value = std::exchange(value_, Handle{});
        return static_cast<int>(Destroy(value));
    }
    Handle value_{};
};
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Filter = Resource<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor>;
using Convolution = Resource<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor>;
using DeviceMemory = Resource<CUdeviceptr, cuMemFree>;

struct ConvolutionMetadata {
    int rank{};
    cudnnConvolutionMode_t mode{CUDNN_CONVOLUTION};
    cudnnDataType_t type{CUDNN_DATA_FLOAT};
    std::array<int, 6> padding{}, strides{}, dilation{};
};
struct GuardedConvolutionMetadata {
    std::array<int, 3> rank{-771, -772, -773};
    std::array<cudnnConvolutionMode_t, 3> mode{CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION,
                                               CUDNN_CONVOLUTION};
    std::array<cudnnDataType_t, 3> type{CUDNN_DATA_DOUBLE, CUDNN_DATA_INT32, CUDNN_DATA_HALF};
    std::array<int, 10> padding{}, strides{}, dilation{};
    GuardedConvolutionMetadata() {
        for (std::size_t i = 0; i != padding.size(); ++i) {
            padding[i] = -1000 - static_cast<int>(i);
            strides[i] = -2000 - static_cast<int>(i);
            dilation[i] = -3000 - static_cast<int>(i);
        }
    }
    bool operator==(const GuardedConvolutionMetadata&) const = default;
};

void inspect_convolution_metadata(cudnnConvolutionDescriptor_t descriptor,
                                  const ConvolutionMetadata& expected, std::string_view label,
                                  unsigned int& queries) {
    // Configured cuDNN 9.10.2 copies requested entries, not min(requested, rank).
    // These independent expected arrays include native initial zeros and the
    // rank-three history retained after either successful rank-two setter.
    for (const int requested : {0, 1, 2, 3, 6}) {
        GuardedConvolutionMetadata observed;
        auto wanted = observed;
        wanted.rank[1] = expected.rank;
        wanted.mode[1] = expected.mode;
        wanted.type[1] = expected.type;
        std::copy_n(expected.padding.begin(), requested, wanted.padding.begin() + 1);
        std::copy_n(expected.strides.begin(), requested, wanted.strides.begin() + 1);
        std::copy_n(expected.dilation.begin(), requested, wanted.dilation.begin() + 1);
        checked(cudnnGetConvolutionNdDescriptor(
                    descriptor, requested, &observed.rank[1], observed.padding.data() + 1,
                    observed.strides.data() + 1, observed.dilation.data() + 1, &observed.mode[1],
                    &observed.type[1]),
                "requested-count convolution metadata");
        ++queries;
        expect(observed == wanted,
               "requested values, scalar guards and all unrequested array guards match");
        std::cout << "metadata_case=" << label << "; requested=" << requested
                  << "; rank=" << observed.rank[1]
                  << "; mode=" << static_cast<int>(observed.mode[1])
                  << "; type=" << static_cast<int>(observed.type[1]);
        const auto print = [&](std::string_view name, const auto& values) {
            std::cout << "; " << name << '=';
            for (std::size_t i = 0; i != values.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << values[i];
            }
        };
        print("guarded_padding", observed.padding);
        print("guarded_strides", observed.strides);
        print("guarded_dilation", observed.dilation);
        std::cout << '\n';
    }
    for (const int requested :
         {-1, std::numeric_limits<int>::min(), 7, std::numeric_limits<int>::max()}) {
        GuardedConvolutionMetadata observed;
        const auto unchanged = observed;
        const auto status = cudnnGetConvolutionNdDescriptor(
            descriptor, requested, &observed.rank[1], observed.padding.data() + 1,
            observed.strides.data() + 1, observed.dilation.data() + 1, &observed.mode[1],
            &observed.type[1]);
        ++queries;
        expect(status == (requested < 0 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_NOT_SUPPORTED),
               "original signed capacity failure is preserved");
        expect(observed == unchanged, "failed metadata query publishes no scalar or array");
        std::cout << "metadata_case=" << label << "; requested=" << requested
                  << "; native_status=" << static_cast<int>(status) << "; outputs_unchanged=true\n";
    }
}

void run_metadata_only() {
    unsigned int queries{};
    {
        Convolution descriptor;
        checked(cudnnCreateConvolutionDescriptor(descriptor.output()), "create fresh convolution");
        inspect_convolution_metadata(descriptor.get(), {}, "fresh-rank-zero", queries);
        descriptor.close();
    }
    for (const bool has_history : {false, true}) {
        for (const bool use_two_dimensional_setter : {false, true}) {
            Convolution descriptor;
            checked(cudnnCreateConvolutionDescriptor(descriptor.output()), "create convolution");
            if (has_history) {
                constexpr std::array padding{1, 2, 7}, strides{2, 3, 8}, dilation{3, 4, 9};
                checked(cudnnSetConvolutionNdDescriptor(descriptor.get(), 3, padding.data(),
                                                        strides.data(), dilation.data(),
                                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                        "set distinctive rank-three history");
            }
            ConvolutionMetadata expected;
            expected.rank = 2;
            expected.mode = CUDNN_CROSS_CORRELATION;
            expected.type = CUDNN_DATA_DOUBLE;
            expected.padding = {3, 4, has_history ? 7 : 0, 0, 0, 0};
            expected.strides = {5, 6, has_history ? 8 : 0, 0, 0, 0};
            expected.dilation = {2, 3, has_history ? 9 : 0, 0, 0, 0};
            if (use_two_dimensional_setter) {
                checked(cudnnSetConvolution2dDescriptor(
                            descriptor.get(), expected.padding[0], expected.padding[1],
                            expected.strides[0], expected.strides[1], expected.dilation[0],
                            expected.dilation[1], expected.mode, expected.type),
                        "set rank two through 2D setter");
            } else {
                checked(cudnnSetConvolutionNdDescriptor(
                            descriptor.get(), 2, expected.padding.data(), expected.strides.data(),
                            expected.dilation.data(), expected.mode, expected.type),
                        "set rank two through Nd setter");
            }
            const auto label =
                has_history
                    ? (use_two_dimensional_setter ? "rank-three-to-two-2D" : "rank-three-to-two-Nd")
                    : (use_two_dimensional_setter ? "fresh-rank-two-2D" : "fresh-rank-two-Nd");
            inspect_convolution_metadata(descriptor.get(), expected, label, queries);
            descriptor.close();
        }
    }
    expect(queries == 45 && !cleanup_failed, "complete metadata matrix and checked cleanup");
    std::cout << "cuDNN convolution metadata input passed: cases=5; getter_calls=" << queries
              << "; explicit_driver_calls=0; explicit_device_calls=0; compute_calls=0; "
                 "matched_native_owned_required=true; full_device_qualified=false; "
                 "priority_qualified=false\n";
}

constexpr int kN = 1, kC = 2, kH = 4, kW = 5, kK = 2, kR = 2, kS = 3;
constexpr int kPadH = 1, kPadW = 0, kStrideH = 2, kStrideW = 1;
constexpr int kOutH = (kH + 2 * kPadH - kR) / kStrideH + 1;
constexpr int kOutW = (kW + 2 * kPadW - kS) / kStrideW + 1;
constexpr std::size_t kInputCount = kN * kC * kH * kW;
constexpr std::size_t kFilterCount = kK * kC * kR * kS;
constexpr std::size_t kOutputCount = kN * kK * kOutH * kOutW;
constexpr std::size_t kGuard = 8, kWorkspaceGuard = 256;
constexpr std::size_t kMaximumWorkspace = 64 * 1024 * 1024;
constexpr std::array<int, 4> kInputShape{kN, kC, kH, kW};
constexpr std::array<int, 4> kFilterShape{kK, kC, kR, kS};
constexpr std::array<int, 4> kOutputShape{kN, kK, kOutH, kOutW};
struct Evidence {
    unsigned int compute_calls{}, zero_workspace_candidates{}, nonzero_workspace_candidates{};
    unsigned int null_workspace_calls{}, nonnull_zero_workspace_calls{};
    unsigned int find_calls{};
};
template <typename Value>
struct Descriptors {
    static constexpr auto kType =
        std::is_same_v<Value, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    Tensor input, output;
    Filter filter;
    Convolution convolution;
    Descriptors() {
        checked(cudnnCreateTensorDescriptor(input.output()), "create input descriptor");
        checked(cudnnCreateTensorDescriptor(output.output()), "create output descriptor");
        checked(cudnnCreateFilterDescriptor(filter.output()), "create filter descriptor");
        checked(cudnnCreateConvolutionDescriptor(convolution.output()),
                "create convolution descriptor");
        checked(cudnnSetTensor4dDescriptor(input.get(), CUDNN_TENSOR_NCHW, kType, kN, kC, kH, kW),
                "set input tensor");
        checked(cudnnSetFilter4dDescriptor(filter.get(), kType, CUDNN_TENSOR_NCHW, kK, kC, kR, kS),
                "set four-dimensional filter");
        cudnnDataType_t type{};
        cudnnTensorFormat_t format{};
        std::array<int, 4> shape{};
        checked(cudnnGetFilter4dDescriptor(filter.get(), &type, &format, &shape[0], &shape[1],
                                           &shape[2], &shape[3]),
                "get four-dimensional filter");
        expect(type == kType && format == CUDNN_TENSOR_NCHW && shape == kFilterShape,
               "original four-dimensional filter metadata");
        checked(cudnnSetFilterNdDescriptor(filter.get(), kType, CUDNN_TENSOR_NCHW, 4,
                                           kFilterShape.data()),
                "set Nd filter");
        int rank{};
        checked(cudnnGetFilterNdDescriptor(filter.get(), 4, &type, &format, &rank, shape.data()),
                "get Nd filter");
        expect(rank == 4 && type == kType && format == CUDNN_TENSOR_NCHW && shape == kFilterShape,
               "original Nd filter metadata");
        std::size_t filter_bytes{};
        checked(cudnnGetFilterSizeInBytes(filter.get(), &filter_bytes), "filter byte size");
        expect(filter_bytes == kFilterCount * sizeof(Value), "native filter extent matches input");
        checked(cudnnSetConvolution2dDescriptor(convolution.get(), kPadH, kPadW, kStrideH, kStrideW,
                                                1, 1, CUDNN_CROSS_CORRELATION, kType),
                "set two-dimensional convolution");
        std::array<int, 2> padding{}, strides{}, dilation{};
        cudnnConvolutionMode_t mode{};
        checked(cudnnGetConvolution2dDescriptor(convolution.get(), &padding[0], &padding[1],
                                                &strides[0], &strides[1], &dilation[0],
                                                &dilation[1], &mode, &type),
                "get two-dimensional convolution");
        expect(padding == std::array{kPadH, kPadW} && strides == std::array{kStrideH, kStrideW} &&
                   dilation == std::array{1, 1} && mode == CUDNN_CROSS_CORRELATION && type == kType,
               "native two-dimensional convolution metadata");
        checked(cudnnSetConvolutionNdDescriptor(convolution.get(), 2, padding.data(),
                                                strides.data(), dilation.data(), mode, kType),
                "set Nd convolution");
        padding = {};
        strides = {};
        dilation = {};
        checked(cudnnGetConvolutionNdDescriptor(convolution.get(), 2, &rank, padding.data(),
                                                strides.data(), dilation.data(), &mode, &type),
                "get Nd convolution");
        expect(rank == 2 && padding == std::array{kPadH, kPadW} &&
                   strides == std::array{kStrideH, kStrideW} && dilation == std::array{1, 1} &&
                   mode == CUDNN_CROSS_CORRELATION && type == kType,
               "native Nd convolution metadata");
        checked(cudnnSetConvolutionGroupCount(convolution.get(), 1), "set group count");
        int groups{};
        checked(cudnnGetConvolutionGroupCount(convolution.get(), &groups), "get group count");
        expect(groups == 1, "native group count");
        checked(cudnnGetConvolution2dForwardOutputDim(convolution.get(), input.get(), filter.get(),
                                                      &shape[0], &shape[1], &shape[2], &shape[3]),
                "two-dimensional output shape");
        expect(shape == kOutputShape, "native output shape agrees with independent formula");
        shape = {};
        checked(cudnnGetConvolutionNdForwardOutputDim(convolution.get(), input.get(), filter.get(),
                                                      4, shape.data()),
                "Nd output shape");
        expect(shape == kOutputShape, "native Nd output shape agrees with independent formula");
        constexpr std::array<int, 4> output_strides{kK * kOutH * kOutW, kOutH * kOutW, kOutW, 1};
        checked(
            cudnnSetTensorNdDescriptor(output.get(), kType, 4, shape.data(), output_strides.data()),
            "set output tensor");
    }
    void close() {
        convolution.close();
        filter.close();
        output.close();
        input.close();
    }
};
template <typename Value>
std::array<Value, kOutputCount> reference(const std::vector<Value>& input,
                                          const std::vector<Value>& weights) {
    std::array<Value, kOutputCount> result{};
    for (int n = 0; n < kN; ++n)
        for (int k = 0; k < kK; ++k)
            for (int oh = 0; oh < kOutH; ++oh)
                for (int ow = 0; ow < kOutW; ++ow) {
                    long double sum{};
                    for (int c = 0; c < kC; ++c)
                        for (int r = 0; r < kR; ++r)
                            for (int s = 0; s < kS; ++s) {
                                const int ih = oh * kStrideH - kPadH + r;
                                const int iw = ow * kStrideW - kPadW + s;
                                if (ih < 0 || ih >= kH || iw < 0 || iw >= kW) continue;
                                const auto x =
                                    static_cast<std::size_t>(((n * kC + c) * kH + ih) * kW + iw);
                                const auto w =
                                    static_cast<std::size_t>(((k * kC + c) * kR + r) * kS + s);
                                sum += static_cast<long double>(input[kGuard + x]) *
                                       weights[kGuard + w];
                            }
                    const auto y =
                        static_cast<std::size_t>(((n * kK + k) * kOutH + oh) * kOutW + ow);
                    result[y] = static_cast<Value>(sum);
                }
    return result;
}
struct Candidate {
    cudnnConvolutionFwdAlgo_t algorithm{};
    cudnnMathType_t math{};
    std::size_t workspace{};
};
template <typename Value>
void execute(cudnnHandle_t handle, CUstream stream, const Descriptors<Value>& descriptors,
             Evidence& evidence, bool should_find) {
    std::vector<Value> input(kInputCount + 2 * kGuard, Value{-1234});
    std::vector<Value> weights(kFilterCount + 2 * kGuard, Value{-2345});
    std::vector<Value> initial(kOutputCount + 2 * kGuard, Value{-3456});
    for (std::size_t i = 0; i < kInputCount; ++i)
        input[kGuard + i] = static_cast<Value>(static_cast<int>(i % 13) - 6) * Value{0.25};
    for (std::size_t i = 0; i < kFilterCount; ++i)
        weights[kGuard + i] = static_cast<Value>(static_cast<int>(i % 7) - 3) * Value{0.125};
    for (std::size_t i = 0; i < kOutputCount; ++i)
        initial[kGuard + i] = static_cast<Value>(static_cast<int>(i) - 8) * Value{0.0625};
    const auto convolution = reference(input, weights);
    DeviceMemory x, w, y;
    checked(cuMemAlloc(x.output(), input.size() * sizeof(Value)), "allocate input");
    checked(cuMemAlloc(w.output(), weights.size() * sizeof(Value)), "allocate filter");
    checked(cuMemAlloc(y.output(), initial.size() * sizeof(Value)), "allocate output");
    checked(cuMemcpyHtoD(x.get(), input.data(), input.size() * sizeof(Value)), "upload input");
    checked(cuMemcpyHtoD(w.get(), weights.data(), weights.size() * sizeof(Value)), "upload filter");
    int maximum{};
    checked(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maximum), "algorithm maximum");
    expect(maximum > 0 && maximum <= 64, "bounded actual SDK algorithm maximum");
    std::vector<cudnnConvolutionFwdAlgoPerf_t> performance(static_cast<std::size_t>(maximum));
    int returned{};
    if (should_find) {
        checked(cudnnFindConvolutionForwardAlgorithm(
                    handle, descriptors.input.get(), descriptors.filter.get(),
                    descriptors.convolution.get(), descriptors.output.get(), maximum, &returned,
                    performance.data()),
                "actual forward algorithm benchmark");
        ++evidence.find_calls;
    } else {
        checked(cudnnGetConvolutionForwardAlgorithm_v7(
                    handle, descriptors.input.get(), descriptors.filter.get(),
                    descriptors.convolution.get(), descriptors.output.get(), maximum, &returned,
                    performance.data()),
                "forward heuristic");
    }
    expect(returned > 0 && returned <= maximum, "bounded native algorithm count");
    std::array<std::optional<Candidate>, 2> selected;
    for (int index = 0; index < returned; ++index) {
        const auto& entry = performance[static_cast<std::size_t>(index)];
        if (entry.status != CUDNN_STATUS_SUCCESS) continue;
        checked(cudnnSetConvolutionMathType(descriptors.convolution.get(), entry.mathType),
                "set candidate math");
        std::size_t bytes{};
        checked(cudnnGetConvolutionForwardWorkspaceSize(
                    handle, descriptors.input.get(), descriptors.filter.get(),
                    descriptors.convolution.get(), descriptors.output.get(), entry.algo, &bytes),
                "candidate workspace size");
        const std::size_t kind = bytes ? 1 : 0;
        if (!selected[kind] && bytes <= kMaximumWorkspace)
            selected[kind] = Candidate{entry.algo, entry.mathType, bytes};
        if (selected[0] && selected[1]) break;
    }
    expect(selected[0] || selected[1], "at least one original successful bounded candidate");
    const Value alpha{1.25}, beta{-0.5};
    for (const auto& choice : selected) {
        if (!choice) continue;
        const auto candidate = *choice;
        checked(cudnnSetConvolutionMathType(descriptors.convolution.get(), candidate.math),
                "select candidate math");
        cudnnMathType_t actual_math{};
        checked(cudnnGetConvolutionMathType(descriptors.convolution.get(), &actual_math),
                "query candidate math");
        expect(actual_math == candidate.math, "original selected math type");
        std::vector<std::byte> workspace(candidate.workspace + 2 * kWorkspaceGuard,
                                         std::byte{0xa5});
        DeviceMemory scratch;
        checked(cuMemAlloc(scratch.output(), workspace.size()), "allocate guarded workspace");
        checked(cuMemcpyHtoD(scratch.get(), workspace.data(), workspace.size()),
                "upload workspace guards");
        checked(cuMemcpyHtoD(y.get(), initial.data(), initial.size() * sizeof(Value)),
                "upload initial output");
        auto expected = initial;
        std::vector<Value> output(initial.size()), observed_input(input.size()),
            observed_weights(weights.size());
        if (candidate.workspace)
            ++evidence.nonzero_workspace_candidates;
        else
            ++evidence.zero_workspace_candidates;
        for (unsigned int iteration = 0; iteration < 3; ++iteration) {
            // A zero-workspace algorithm exercises both exact nonnull identity
            // and nullptr, without fabricating a zero requirement for other algorithms.
            const bool is_null_workspace = !candidate.workspace && iteration == 1;
            // NOLINTBEGIN(performance-no-int-to-ptr): original CUDA address ABI.
            auto* workspace_pointer =
                is_null_workspace ? nullptr
                                  : reinterpret_cast<void*>(scratch.get() + kWorkspaceGuard);
            checked(cudnnConvolutionForward(
                        handle, &alpha, descriptors.input.get(),
                        reinterpret_cast<const void*>(x.get() + kGuard * sizeof(Value)),
                        descriptors.filter.get(),
                        reinterpret_cast<const void*>(w.get() + kGuard * sizeof(Value)),
                        descriptors.convolution.get(), candidate.algorithm, workspace_pointer,
                        candidate.workspace, &beta, descriptors.output.get(),
                        reinterpret_cast<void*>(y.get() + kGuard * sizeof(Value))),
                    "original convolution forward");
            // NOLINTEND(performance-no-int-to-ptr)
            ++evidence.compute_calls;
            if (is_null_workspace)
                ++evidence.null_workspace_calls;
            else if (!candidate.workspace)
                ++evidence.nonnull_zero_workspace_calls;
            for (std::size_t i = 0; i < kOutputCount; ++i)
                expected[kGuard + i] = alpha * convolution[i] + beta * expected[kGuard + i];
            checked(cuStreamSynchronize(stream), "physical convolution completion");
            checked(cuMemcpyDtoH(output.data(), y.get(), output.size() * sizeof(Value)),
                    "read full output");
            for (std::size_t i = 0; i < output.size(); ++i) {
                if (i < kGuard || i >= kGuard + kOutputCount) {
                    expect(output[i] == expected[i], "output guard remains byte-value exact");
                    continue;
                }
                const double tolerance = std::is_same_v<Value, double> ? 1.0e-11 : 1.0e-4;
                expect(
                    std::isfinite(output[i]) &&
                        std::abs(static_cast<double>(output[i] - expected[i])) <=
                            tolerance * std::max(1.0, std::abs(static_cast<double>(expected[i]))),
                    "every convolution value matches repeated non-idempotent CPU oracle");
            }
            checked(cuMemcpyDtoH(observed_input.data(), x.get(), input.size() * sizeof(Value)),
                    "verify input");
            checked(cuMemcpyDtoH(observed_weights.data(), w.get(), weights.size() * sizeof(Value)),
                    "verify filter");
            expect(observed_input == input && observed_weights == weights,
                   "input/filter and both guards are unchanged");
            checked(cuMemcpyDtoH(workspace.data(), scratch.get(), workspace.size()),
                    "read workspace guards");
            for (std::size_t i = 0; i < kWorkspaceGuard; ++i)
                expect(workspace[i] == std::byte{0xa5} &&
                           workspace[workspace.size() - 1 - i] == std::byte{0xa5},
                       "native workspace writes remain within declared range");
        }
        scratch.close();
    }
    y.close();
    w.close();
    x.close();
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const bool should_find = argc == 2 && std::string_view(argv[1]) == "--find";
        const bool metadata_only = argc == 2 && std::string_view(argv[1]) == "--metadata-only";
        expect(argc == 1 || should_find || metadata_only,
               "usage: convolution input [--find | --metadata-only]");
        if (metadata_only) {
            run_metadata_only();
            return EXIT_SUCCESS;
        }
        // Both complete descriptor graphs are configured before any explicit
        // Driver initialization or GPU context creation.
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
        cudaStream_t observed{};
        checked(cudnnGetStream(handle.get(), &observed), "query explicit stream");
        expect(observed == reinterpret_cast<cudaStream_t>(stream.get()),
               "exact public stream identity");
        Evidence evidence;
        execute(handle.get(), stream.get(), single, evidence, should_find);
        execute(handle.get(), stream.get(), dual, evidence, should_find);
        expect(evidence.find_calls == (should_find ? 2U : 0U), "exact requested search mode");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        dual.close();
        single.close();
        expect(!cleanup_failed, "all native resources have checked cleanup");
        std::cout << "cuDNN convolution input passed: types=2; compute_calls="
                  << evidence.compute_calls
                  << "; zero_workspace_candidates=" << evidence.zero_workspace_candidates
                  << "; nonzero_workspace_candidates=" << evidence.nonzero_workspace_candidates
                  << "; null_workspace_calls=" << evidence.null_workspace_calls
                  << "; nonnull_zero_workspace_calls=" << evidence.nonnull_zero_workspace_calls
                  << "; find_calls=" << evidence.find_calls
                  << "; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
