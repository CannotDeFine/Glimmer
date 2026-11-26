#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_convolution_fixture.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <source_location>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_convolution_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public convolution SDK status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
void driver_check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public convolution Driver status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct, global and explicit convolution lookup selects the controlled ABI");
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void lookup(void* library) {
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreateFilterDescriptor);
    CHECK(cudnnDestroyFilterDescriptor);
    CHECK(cudnnSetFilter4dDescriptor);
    CHECK(cudnnGetFilter4dDescriptor);
    CHECK(cudnnSetFilterNdDescriptor);
    CHECK(cudnnGetFilterNdDescriptor);
    CHECK(cudnnGetFilterSizeInBytes);
    CHECK(cudnnCreateConvolutionDescriptor);
    CHECK(cudnnDestroyConvolutionDescriptor);
    CHECK(cudnnSetConvolution2dDescriptor);
    CHECK(cudnnGetConvolution2dDescriptor);
    CHECK(cudnnSetConvolutionNdDescriptor);
    CHECK(cudnnGetConvolutionNdDescriptor);
    CHECK(cudnnSetConvolutionGroupCount);
    CHECK(cudnnGetConvolutionGroupCount);
    CHECK(cudnnSetConvolutionMathType);
    CHECK(cudnnGetConvolutionMathType);
    CHECK(cudnnSetConvolutionReorderType);
    CHECK(cudnnGetConvolutionReorderType);
    CHECK(cudnnGetConvolution2dForwardOutputDim);
    CHECK(cudnnGetConvolutionNdForwardOutputDim);
    CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount);
    CHECK(cudnnGetConvolutionForwardAlgorithm_v7);
    CHECK(cudnnFindConvolutionForwardAlgorithm);
    CHECK(cudnnGetConvolutionForwardWorkspaceSize);
    CHECK(cudnnConvolutionForward);
#undef CHECK
}
struct Descriptors {
    cudnnTensorDescriptor_t x{}, y{};
    cudnnFilterDescriptor_t w{};
    cudnnConvolutionDescriptor_t convolution{};
};
struct ConvolutionOutput {
    std::uint64_t before{0x123456789abcdef0ULL};
    int rank{99};
    cudnnConvolutionMode_t mode{CUDNN_CONVOLUTION};
    cudnnDataType_t type{CUDNN_DATA_DOUBLE};
    std::array<int, 8> padding, strides, dilation;
    std::uint64_t after{0xfedcba9876543210ULL};
    ConvolutionOutput() {
        padding.fill(91);
        strides.fill(92);
        dilation.fill(93);
    }
    bool operator==(const ConvolutionOutput&) const = default;
};
void convolution_requested_spans(cudnnConvolutionDescriptor_t descriptor) {
    // The first two ints are writable, but an actual requested-six publication
    // crosses into PROT_NONE. Validate every entire span before any scalar store.
    const long page_bytes = ::sysconf(_SC_PAGESIZE);
    expect(page_bytes > 32, "valid system page size for exact output boundary");
    const auto page = static_cast<std::size_t>(page_bytes);
    auto* mapping =
        ::mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(mapping != MAP_FAILED, "allocate two-page output validation boundary");
    auto* boundary = static_cast<std::byte*>(mapping) + page;
    expect(::mprotect(boundary, page, PROT_NONE) == 0, "protect suffix of requested output span");
    auto* short_output = reinterpret_cast<int*>(boundary - 2 * sizeof(int));
    bool valid = true;
    for (std::size_t target = 0; target != 3; ++target) {
        short_output[0] = 701;
        short_output[1] = 702;
        ConvolutionOutput output;
        const auto expected = output;
        const std::array<int*, 3> pointers{output.padding.data(), output.strides.data(),
                                           output.dilation.data()};
        auto selected = pointers;
        selected[target] = short_output;
        const auto status =
            cudnnGetConvolutionNdDescriptor(descriptor, 6, &output.rank, selected[0], selected[1],
                                            selected[2], &output.mode, &output.type);
        valid &= status == CUDNN_STATUS_BAD_PARAM && output == expected && short_output[0] == 701 &&
                 short_output[1] == 702;
    }
    expect(::munmap(mapping, page * 2) == 0, "release full guarded mapping");
    expect(valid,
           "all three requested spans must be writable before any scalar or array publication");
}
void convolution_requested_tails(cudnnConvolutionDescriptor_t descriptor) {
    using glimmer::backend::cuda::CudaDnnConvolutionState;
    const auto verify = [&](const CudaDnnConvolutionState& expected) {
        for (const int requested : {0, 1, 2, 3, 6}) {
            ConvolutionOutput output;
            auto wanted = output;
            wanted.rank = expected.dimensions;
            wanted.mode = static_cast<cudnnConvolutionMode_t>(expected.mode);
            wanted.type = static_cast<cudnnDataType_t>(expected.type);
            std::copy_n(expected.padding.begin(), requested, wanted.padding.begin());
            std::copy_n(expected.strides.begin(), requested, wanted.strides.begin());
            std::copy_n(expected.dilation.begin(), requested, wanted.dilation.begin());
            check(cudnnGetConvolutionNdDescriptor(
                descriptor, requested, &output.rank, output.padding.data(), output.strides.data(),
                output.dilation.data(), &output.mode, &output.type));
            expect(output == wanted,
                   "Nd getter writes requested entries independently of actual rank and preserves "
                   "every later guard");
        }
    };
    verify({});
    CudaDnnConvolutionState lower{
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE, 2, {1, 2}, {3, 4}, {5, 6}};
    const auto set_nd = [&](const CudaDnnConvolutionState& value) {
        check(cudnnSetConvolutionNdDescriptor(descriptor, value.dimensions, value.padding.data(),
                                              value.strides.data(), value.dilation.data(),
                                              static_cast<cudnnConvolutionMode_t>(value.mode),
                                              static_cast<cudnnDataType_t>(value.type)));
    };
    set_nd(lower);
    verify(lower);
    for (const bool is_two : {false, true}) {
        auto higher = lower;
        higher.dimensions = 3;
        higher.padding[2] = 7;
        higher.strides[2] = 8;
        higher.dilation[2] = 9;
        set_nd(higher);
        verify(higher);
        if (is_two)
            check(cudnnSetConvolution2dDescriptor(descriptor, 1, 2, 3, 4, 5, 6,
                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
        else
            set_nd(lower);
        higher.dimensions = 2;
        verify(higher);
    }
    convolution_requested_spans(descriptor);
    for (const int requested : {-1, INT32_MIN, 7, 12, INT32_MAX}) {
        ConvolutionOutput output;
        const auto expected = output;
        expect(cudnnGetConvolutionNdDescriptor(
                   descriptor, requested, &output.rank, output.padding.data(),
                   output.strides.data(), output.dilation.data(), &output.mode, &output.type) ==
                       (requested < 0 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_NOT_SUPPORTED) &&
                   output == expected,
               "unclipped signed capacity preserves exact original rejection without any caller "
               "write");
    }
    fixture::control.convolution_get_status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    ConvolutionOutput output;
    const auto expected = output;
    const auto result = cudnnGetConvolutionNdDescriptor(
        descriptor, 6, &output.rank, output.padding.data(), output.strides.data(),
        output.dilation.data(), &output.mode, &output.type);
    fixture::control.convolution_get_status = CUDNN_STATUS_SUCCESS;
    expect(result == CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH && output == expected,
           "native failure after private requested-tail writes publishes no scalar or array");
}
void configure(const Descriptors& d, cudnnDataType_t type) {
    check(cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NCHW, type, 1, 2, 4, 5));
    check(cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NCHW, type, 1, 2, 3, 3));
    check(cudnnSetFilter4dDescriptor(d.w, type, CUDNN_TENSOR_NCHW, 2, 2, 2, 3));
    check(cudnnSetConvolution2dDescriptor(d.convolution, 1, 0, 2, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                          type));
}
Descriptors metadata() {
    Descriptors d;
    check(cudnnCreateTensorDescriptor(&d.x));
    check(cudnnCreateTensorDescriptor(&d.y));
    check(cudnnCreateFilterDescriptor(&d.w));
    check(cudnnCreateConvolutionDescriptor(&d.convolution));
    convolution_requested_tails(d.convolution);
    configure(d, CUDNN_DATA_FLOAT);
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    std::array<int, 4> sizes{};
    check(cudnnGetFilter4dDescriptor(d.w, &type, &format, &sizes[0], &sizes[1], &sizes[2],
                                     &sizes[3]));
    expect(type == CUDNN_DATA_FLOAT && format == CUDNN_TENSOR_NCHW &&
               sizes == std::array<int, 4>{2, 2, 2, 3},
           "exact original 4D filter metadata");
    check(cudnnSetFilterNdDescriptor(d.w, type, format, 4, sizes.data()));
    std::array<int, 12> dimensions;
    dimensions.fill(91);
    int rank{};
    check(cudnnGetFilterNdDescriptor(d.w, 12, &type, &format, &rank, dimensions.data()));
    expect(rank == 4 && std::equal(sizes.begin(), sizes.end(), dimensions.begin()) &&
               std::ranges::all_of(std::span(dimensions).subspan(4), [](int i) { return i == 91; }),
           "oversized filter capacity preserves unused caller suffix");
    std::size_t bytes{};
    check(cudnnGetFilterSizeInBytes(d.w, &bytes));
    expect(bytes == 24 * sizeof(float), "native filter size reaches caller unchanged");
    std::array<int, 2> padding{}, strides{}, dilation{};
    cudnnConvolutionMode_t mode{};
    check(cudnnGetConvolution2dDescriptor(d.convolution, &padding[0], &padding[1], &strides[0],
                                          &strides[1], &dilation[0], &dilation[1], &mode, &type));
    expect(padding == std::array<int, 2>{1, 0} && strides == std::array<int, 2>{2, 1} &&
               dilation == std::array<int, 2>{1, 1} && mode == CUDNN_CROSS_CORRELATION,
           "exact original 2D convolution metadata");
    check(cudnnSetConvolutionNdDescriptor(d.convolution, 2, padding.data(), strides.data(),
                                          dilation.data(), mode, type));
    std::array<int, 8> p, s, v;
    p.fill(91);
    s.fill(92);
    v.fill(93);
    rank = 99;
    expect(cudnnGetConvolutionNdDescriptor(d.convolution, 7, &rank, p.data(), s.data(), v.data(),
                                           &mode, &type) == CUDNN_STATUS_NOT_SUPPORTED &&
               rank == 99 && p[0] == 91 && s[0] == 92 && v[0] == 93,
           "convolution query capacity above six preserves native rejection and outputs");
    check(cudnnGetConvolutionNdDescriptor(d.convolution, 6, &rank, p.data(), s.data(), v.data(),
                                          &mode, &type));
    expect(rank == 2 && p == std::array<int, 8>{1, 0, 7, 0, 0, 0, 91, 91} &&
               s == std::array<int, 8>{2, 1, 8, 0, 0, 0, 92, 92} &&
               v == std::array<int, 8>{1, 1, 9, 0, 0, 0, 93, 93},
           "convolution requested-six getter returns retained third entries and guards beyond six");
    check(cudnnSetConvolutionGroupCount(d.convolution, 2));
    int group{};
    check(cudnnGetConvolutionGroupCount(d.convolution, &group));
    expect(group == 2, "original group attribute round trip");
    // Injected SDK success preserves every signed int bit; this does not
    // claim that negative groups are valid for actual vendor computation.
    for (const auto original :
         {std::numeric_limits<int>::min(), -1, 0, std::numeric_limits<int>::max()}) {
        check(cudnnSetConvolutionGroupCount(d.convolution, original));
        check(cudnnGetConvolutionGroupCount(d.convolution, &group));
        expect(group == original, "signed group query preserves original SDK value");
    }
    check(cudnnSetConvolutionGroupCount(d.convolution, 1));
    check(cudnnSetConvolutionMathType(d.convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    cudnnMathType_t math{};
    check(cudnnGetConvolutionMathType(d.convolution, &math));
    expect(math == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, "original math attribute round trip");
    check(cudnnSetConvolutionReorderType(d.convolution, CUDNN_NO_REORDER));
    cudnnReorderType_t reorder{};
    check(cudnnGetConvolutionReorderType(d.convolution, &reorder));
    expect(reorder == CUDNN_NO_REORDER, "original reorder attribute round trip");
    check(cudnnGetConvolution2dForwardOutputDim(d.convolution, d.x, d.w, &sizes[0], &sizes[1],
                                                &sizes[2], &sizes[3]));
    expect(sizes == std::array<int, 4>{1, 2, 3, 3}, "native 2D output shape");
    std::array<int, 5> shape{0, 0, 0, 0, 991};
    check(cudnnGetConvolutionNdForwardOutputDim(d.convolution, d.x, d.w, 4, shape.data()));
    expect(std::equal(sizes.begin(), sizes.end(), shape.begin()) && shape[4] == 991 &&
               !fixture::control.context_creates.load(),
           "all new descriptor operations are context independent");
    return d;
}
void queries(cudnnHandle_t handle, const Descriptors& d) {
    int maximum{};
    check(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maximum));
    expect(maximum == 12, "native max-count can exceed algorithm-enum count");
    std::size_t workspace{};
    check(cudnnGetConvolutionForwardWorkspaceSize(handle, d.x, d.w, d.convolution, d.y,
                                                  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                                  &workspace));
    expect(workspace == 32, "native workspace size query");
    fixture::control.fail_workspace.store(true);
    workspace = 991;
    expect(cudnnGetConvolutionForwardWorkspaceSize(handle, d.x, d.w, d.convolution, d.y,
                                                   CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                                   &workspace) == CUDNN_STATUS_BAD_PARAM &&
               workspace == 991,
           "failed native workspace query suppresses output");
    std::array<cudnnConvolutionFwdAlgoPerfStruct, 20> performance{};
    for (auto& item : performance) item.memory = 77;
    int count{};
    check(cudnnGetConvolutionForwardAlgorithm_v7(handle, d.x, d.w, d.convolution, d.y, 20, &count,
                                                 performance.data()));
    expect(count == 10 && performance[10].memory == 77,
           "heuristic returns more than eight variants without clearing caller suffix");
    for (int i = 0; i < count; ++i) {
        const auto& item = performance[static_cast<std::size_t>(i)];
        expect(item.algo == i % 8 &&
                   item.status == (i == 2 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_SUCCESS) &&
                   std::bit_cast<std::uint32_t>(item.time) ==
                       0x3f000000U + static_cast<std::uint32_t>(i) * 0x1000U &&
                   item.memory == (std::size_t{1} << 33) + static_cast<std::size_t>(i) &&
                   item.determinism == i % 2 && item.mathType == i % 4 && item.reserved[0] == -7 &&
                   item.reserved[1] == i && item.reserved[2] == 31337,
               "every native performance field reaches the public ABI exactly");
    }
    fixture::control.fail_heuristic.store(true);
    performance[0].memory = 887;
    count = 991;
    expect(
        cudnnGetConvolutionForwardAlgorithm_v7(handle, d.x, d.w, d.convolution, d.y, 20, &count,
                                               performance.data()) == CUDNN_STATUS_NOT_SUPPORTED &&
            count == 991 && performance[0].memory == 887,
        "failed native heuristic leaves both count and output records unchanged");
}
using FindFunction = decltype(&cudnnFindConvolutionForwardAlgorithm);
void find_queries(FindFunction find, cudnnHandle_t handle, const Descriptors& d, CUcontext context,
                  CUstream stream) {
    std::array<cudnnConvolutionFwdAlgoPerfStruct, 20> records{};
    int untouched = 991;
    expect(find(handle, d.x, d.w, d.convolution, d.y, 20, &untouched, nullptr) ==
                   CUDNN_STATUS_BAD_PARAM &&
               untouched == 991 && fixture::control.context_synchronizes.load() == 0,
           "invalid result storage rejects before benchmark or physical proof");
    for (const auto requested : {20, 5, std::numeric_limits<int>::max(), 0, -1}) {
        for (auto& item : records) item.memory = 77;
        std::array<std::byte, sizeof(records)> previous{};
        std::memcpy(previous.data(), records.data(), previous.size());
        int count = 991;
        const auto status =
            find(handle, d.x, d.w, d.convolution, d.y, requested, &count, records.data());
        if (requested <= 0) {
            expect(status == CUDNN_STATUS_BAD_PARAM && count == 991 &&
                       std::memcmp(previous.data(), records.data(), previous.size()) == 0,
                   "zero/negative Find request preserves native error and all caller outputs");
            continue;
        }
        check(status);
        expect(count == std::min(requested, 10) &&
                   records[static_cast<std::size_t>(count)].memory == 77,
               "actual Find returns bounded variants and preserves caller suffix");
        for (int i = 0; i != count; ++i) {
            const auto& item = records[static_cast<std::size_t>(i)];
            expect(
                item.algo == i % 8 &&
                    item.status == (i == 2 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_SUCCESS) &&
                    std::bit_cast<std::uint32_t>(item.time) ==
                        0x3f000000U + static_cast<std::uint32_t>(i) * 0x1000U &&
                    item.memory == (std::size_t{1} << 33) + static_cast<std::size_t>(i) &&
                    item.determinism == i % 2 && item.mathType == i % 4 && item.reserved[0] == -7 &&
                    item.reserved[1] == i && item.reserved[2] == 31337,
                "every original Find performance field crosses actual IPC unchanged");
        }
    }
    fixture::control.fail_find.store(true);
    std::array<std::byte, sizeof(records)> previous{};
    std::memcpy(previous.data(), records.data(), previous.size());
    int count = 991;
    expect(find(handle, d.x, d.w, d.convolution, d.y, 20, &count, records.data()) ==
                   CUDNN_STATUS_EXECUTION_FAILED &&
               count == 991 && std::memcmp(previous.data(), records.data(), previous.size()) == 0,
           "native Find failure never publishes touched count or partially filled records");
    fixture::control.can_stop.store(false);
    fixture::control.should_block_find.store(true);
    fixture::control.has_find_entered.store(false);
    fixture::control.can_finish_find.store(false);
    auto invocation = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(context));
        const auto status = find(handle, d.x, d.w, d.convolution, d.y, 20, &count, records.data());
        driver_check(cuCtxSetCurrent(nullptr));
        return status;
    });
    const auto entry_deadline = Clock::now() + std::chrono::seconds{5};
    while (!fixture::control.has_find_entered.load() && Clock::now() < entry_deadline &&
           invocation.wait_for(std::chrono::milliseconds{1}) != std::future_status::ready) {
    }
    const bool has_entered_find = fixture::control.has_find_entered.load();
    const bool has_pending_call =
        invocation.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    fixture::control.can_finish_find.store(true);
    fixture::control.can_finish_find.notify_all();
    check(invocation.get());
    fixture::control.should_block_find.store(false);
    expect(has_entered_find && has_pending_call && count == 10,
           "public result waits for actual host-blocking native Find execution");
    std::array mutations{std::async(std::launch::async,
                                    [&] {
                                        return cudnnSetTensor4dDescriptor(
                                            d.x, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4, 5);
                                    }),
                         std::async(std::launch::async,
                                    [&] {
                                        return cudnnSetTensor4dDescriptor(
                                            d.y, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 3, 3);
                                    }),
                         std::async(std::launch::async,
                                    [&] {
                                        return cudnnSetFilter4dDescriptor(
                                            d.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 2, 2, 3);
                                    }),
                         std::async(std::launch::async, [&] {
                             return cudnnSetConvolution2dDescriptor(d.convolution, 1, 0, 2, 1, 1, 1,
                                                                    CUDNN_CROSS_CORRELATION,
                                                                    CUDNN_DATA_FLOAT);
                         })};
    auto stream_mutation = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(context));
        const auto status = cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream));
        driver_check(cuCtxSetCurrent(nullptr));
        return status;
    });
    bool has_pending_metadata = true;
    for (auto& mutation : mutations)
        has_pending_metadata &=
            mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool has_pending_handle =
        stream_mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.can_stop.store(true);
    for (auto& mutation : mutations) check(mutation.get());
    check(stream_mutation.get());
    expect(has_pending_metadata && has_pending_handle,
           "Find host reply retains all four descriptors and handle until physical event");
}
template <typename Scalar>
void arithmetic(cudnnHandle_t handle, const Descriptors& d, unsigned int workspace_mode) {
    constexpr auto type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    configure(d, type);
    const auto values = fixture::values<Scalar>();
    const std::array<std::size_t, 4> extents{sizeof(values.x), sizeof(values.w), sizeof(values.y),
                                             47};
    std::array<CUdeviceptr, 4> allocations{};
    for (std::size_t i = 0; i < allocations.size(); ++i)
        driver_check(cuMemAlloc(&allocations[i], extents[i] + 32));
    driver_check(cuMemcpyHtoD(allocations[0] + 16, values.x.data(), sizeof(values.x)));
    driver_check(cuMemcpyHtoD(allocations[1] + 16, values.w.data(), sizeof(values.w)));
    driver_check(cuMemcpyHtoD(allocations[2] + 16, values.y.data(), sizeof(values.y)));
    const Scalar alpha{1.5}, beta{-0.25};
    fixture::control.zero_workspace.store(workspace_mode != 0);
    void* workspace = workspace_mode == 2
                          ? nullptr
                          : reinterpret_cast<void*>(allocations[3] +
                                                    (workspace_mode == 3 ? extents[3] + 32 : 16));
    check(cudnnConvolutionForward(handle, &alpha, d.x, reinterpret_cast<void*>(allocations[0] + 16),
                                  d.w, reinterpret_cast<void*>(allocations[1] + 16), d.convolution,
                                  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, workspace,
                                  workspace_mode == 0 ? 47 : 0, &beta, d.y,
                                  reinterpret_cast<void*>(allocations[2] + 16)));
    for (std::size_t allocation = 0; allocation < allocations.size(); ++allocation) {
        std::vector<std::byte> output(extents[allocation] + 32);
        driver_check(cuMemcpyDtoH(output.data(), allocations[allocation], output.size()));
        for (std::size_t i = 0; i < output.size(); ++i)
            if (i < 16 || i >= extents[allocation] + 16)
                expect(output[i] == std::byte{0xcd}, "both public allocation guards unchanged");
        if (allocation == 0)
            expect(std::memcmp(output.data() + 16, values.x.data(), sizeof(values.x)) == 0,
                   "public convolution input stays unchanged");
        if (allocation == 1)
            expect(std::memcmp(output.data() + 16, values.w.data(), sizeof(values.w)) == 0,
                   "public convolution filter stays unchanged");
        if (allocation == 2)
            for (std::size_t i = 0; i < values.y.size(); ++i) {
                Scalar actual{};
                std::memcpy(&actual, output.data() + 16 + i * sizeof(Scalar), sizeof(Scalar));
                expect(actual ==
                           alpha * static_cast<Scalar>(fixture::kProduct[i]) + beta * values.y[i],
                       "all public CPU oracle outputs include original nonzero beta");
            }
        if (allocation == 3 && workspace_mode == 0)
            expect(output[16] == std::byte{0xa5} && output[62] == std::byte{0xa5},
                   "native SDK receives and can use complete declared workspace interval");
        driver_check(cuMemFree(allocations[allocation]));
    }
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent convolution caller provider");
    lookup(library);
    const auto cached_size = reinterpret_cast<decltype(&cudnnGetFilterSizeInBytes)>(
        ::dlsym(library, "cudnnGetFilterSizeInBytes"));
    const auto cached_find =
        reinterpret_cast<FindFunction>(::dlsym(library, "cudnnFindConvolutionForwardAlgorithm"));
    const auto d = metadata();
    expect(::dlclose(library) == 0, "close explicit convolution provider reference");
    std::size_t bytes{};
    check(cached_size(d.w, &bytes));
    expect(bytes == 96, "cached controlled wrapper survives provider close");
    CUcontext context{};
    driver_check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    driver_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    cudaStream_t selected{};
    check(cudnnGetStream(handle, &selected));
    expect(selected == reinterpret_cast<cudaStream_t>(stream), "explicit public stream identity");
    queries(handle, d);
    find_queries(cached_find, handle, d, context, stream);
    arithmetic<float>(handle, d, 0);
    arithmetic<double>(handle, d, 0);
    check(cudnnSetStream(handle, nullptr));
    driver_check(cuStreamDestroy(stream));
    arithmetic<double>(handle, d, 3);
    arithmetic<float>(handle, d, 2);
    fixture::control.zero_workspace.store(false);
    std::array<CUdeviceptr, 4> buffers{};
    constexpr std::array<std::size_t, 4> sizes{256, 128, 128, 64};
    for (std::size_t i = 0; i < buffers.size(); ++i)
        driver_check(cuMemAlloc(&buffers[i], sizes[i]));
    fixture::control.can_stop.store(false);
    fixture::control.fail_forward.store(true);
    const float alpha = 1.5F, beta = -0.25F;
    expect(cudnnConvolutionForward(handle, &alpha, d.x, reinterpret_cast<void*>(buffers[0]), d.w,
                                   reinterpret_cast<void*>(buffers[1]), d.convolution,
                                   CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                   reinterpret_cast<void*>(buffers[3]), 64, &beta, d.y,
                                   reinterpret_cast<void*>(buffers[2])) ==
               CUDNN_STATUS_EXECUTION_FAILED,
           "native forward failure is not host admission success");
    auto mutation = std::async(std::launch::async, [&] {
        return cudnnSetFilter4dDescriptor(d.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 2, 2, 3);
    });
    auto descriptor_release = std::async(
        std::launch::async, [&] { return cudnnDestroyConvolutionDescriptor(d.convolution); });
    auto handle_release = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(context));
        const auto result = cudnnDestroy(handle);
        driver_check(cuCtxSetCurrent(nullptr));
        return result;
    });
    auto workspace_release = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(context));
        const auto result = cuMemFree(buffers[3]);
        driver_check(cuCtxSetCurrent(nullptr));
        return result;
    });
    const bool mutation_pending =
        mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool descriptor_pending =
        descriptor_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool handle_pending =
        handle_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool workspace_pending =
        workspace_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.can_stop.store(true);
    check(mutation.get());
    check(descriptor_release.get());
    check(handle_release.get());
    driver_check(workspace_release.get());
    expect(mutation_pending && descriptor_pending && handle_pending && workspace_pending,
           "native error retains metadata mutation/destruction, handle and workspace until stop");
    expect(cudnnDestroyConvolutionDescriptor(d.convolution) == CUDNN_STATUS_NOT_INITIALIZED,
           "stale convolution token cannot repeat original destruction");
    for (std::size_t i = 0; i < 3; ++i) driver_check(cuMemFree(buffers[i]));
    check(cudnnDestroyTensorDescriptor(d.x));
    check(cudnnDestroyTensorDescriptor(d.y));
    check(cudnnDestroyFilterDescriptor(d.w));
    driver_check(cuCtxDestroy(context));
    expect(glimmer_dnn_convolution_native_calls() == 0, "caller-native provider never executes");
}
#pragma GCC diagnostic pop
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual convolution service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-convolution-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure actual authenticated frontend/service IPC");
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start original-owned convolution service");
    client = std::async(std::launch::async, [=] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public convolution progress");
    client.get();
    expect(service.request_stop() && service.finished(), "all physical/native owners retire");
    expect(fixture::state.forwards == 5 && fixture::state.finds == 7 &&
               fixture::control.context_synchronizes.load() == 7 && fixture::state.creates == 2 &&
               fixture::state.destroys == 2 && base::state.creates == 2 &&
               base::state.destroys == 2 && base::state.handle_creates == 1 &&
               base::state.handle_destroys == 1 && fixture::driver.frees == 20 &&
               fixture::driver.stream_destroys == 1 && fixture::driver.memory.empty() &&
               fixture::driver.streams.empty() && fixture::driver.events.empty() &&
               fixture::driver.contexts.empty() && fixture::control.context_creates.load() == 2 &&
               fixture::control.context_destroys.load() == 2 &&
               fixture::control.host_registers.load() &&
               fixture::control.host_registers.load() == fixture::control.host_unregisters.load() &&
               glimmer_dnn_convolution_native_calls() == 0,
           "service owns every numerical call and once-only cleanup, including pageable staging");
}
constexpr unsigned int kConvolutionReplyCases = 15;
void malformed_convolution_reply(const char* provider, unsigned int index) {
    using namespace glimmer::control;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual hostile convolution metadata peer");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-convolution-tail-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticate hostile requested-tail endpoint");
    const int requested = index == 0                   ? 7
                          : index == 1                 ? INT32_MAX
                          : index == 2                 ? -1
                          : index == 10 || index == 12 ? 0
                                                       : 6;
    const bool clean = index >= 8 && index <= 10;
    const auto expected_status = index == 8 ? CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH
                                 : clean    ? CUDNN_STATUS_SUCCESS
                                            : CUDNN_STATUS_INTERNAL_ERROR;
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile convolution query endpoint");
    client = std::async(std::launch::async, [=] {
        auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
        expect(library, "load original convolution query provider");
        symbol(library, "cudnnGetConvolutionNdDescriptor", &cudnnGetConvolutionNdDescriptor);
        const auto cached = reinterpret_cast<decltype(&cudnnGetConvolutionNdDescriptor)>(
            ::dlsym(library, "cudnnGetConvolutionNdDescriptor"));
        expect(::dlclose(library) == 0, "cache controlled Nd getter after explicit provider close");
        cudnnConvolutionDescriptor_t descriptor{};
        check(cudnnCreateConvolutionDescriptor(&descriptor));
        ConvolutionOutput output;
        auto expected = output;
        if (clean && index != 8) {
            expected.rank = index == 10 ? 0 : 2;
            expected.mode = CUDNN_CROSS_CORRELATION;
            expected.type = CUDNN_DATA_FLOAT;
            if (requested) {
                expected.padding = {1, 2, 7, 0, 0, 0, 91, 91};
                expected.strides = {3, 4, 8, 0, 0, 0, 92, 92};
                expected.dilation = {5, 6, 9, 0, 0, 0, 93, 93};
            }
        }
        expect(cached(descriptor, requested, &output.rank, output.padding.data(),
                      output.strides.data(), output.dilation.data(), &output.mode,
                      &output.type) == expected_status &&
                   output == expected,
               "invalid successful capacity/rank/receipt cannot publish any scalar or array");
        if (clean) {
            ConvolutionOutput follow;
            auto wanted = follow;
            wanted.rank = 2;
            wanted.mode = CUDNN_CROSS_CORRELATION;
            wanted.type = CUDNN_DATA_FLOAT;
            wanted.padding[0] = 1;
            wanted.strides[0] = 3;
            wanted.dilation[0] = 5;
            check(cached(descriptor, 1, &follow.rank, follow.padding.data(), follow.strides.data(),
                         follow.dilation.data(), &follow.mode, &follow.type));
            expect(follow == wanted,
                   "clean original error or success preserves subsequent owner usability");
        } else {
            expect(cached(descriptor, 1, &output.rank, output.padding.data(), output.strides.data(),
                          output.dilation.data(), &output.mode,
                          &output.type) == CUDNN_STATUS_NOT_INITIALIZED &&
                       output == expected,
                   "malformed success quarantines metadata locally without another RPC or write");
        }
        expect(!glimmer_dnn_convolution_native_calls(),
               "hostile metadata receipt never invokes caller-native SDK");
    });
    unsigned int hellos{}, creates{}, queries{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1),
               "bounded hostile convolution query progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "retain actual peer authentication in hostile test");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty() && !hellos++,
                       "query uses exact DNM1 bootstrap only");
                const auto encoded = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(encoded.has_value(), "encode actual metadata retention");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && kind == 4 && input.empty() && !creates++,
                       "create only exact convolution metadata kind");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = 17;
            } else {
                std::uint64_t id{};
                std::uint32_t count{}, is_two{};
                expect(request->message.header.operation == ExecutionOpcode::kGetDnnConvolution &&
                           input.read_u64(id) && id == 17 && input.read_u32(count) &&
                           std::bit_cast<std::int32_t>(count) == (queries ? 1 : requested) &&
                           input.read_u32(is_two) && !is_two && input.empty(),
                       "exact native signed requested bits reach Nd query unchanged");
                ++queries;
                expect(queries == 1 || (clean && queries == 2),
                       "only clean receipts permit another query");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                if (queries == 1 && index == 8) {
                    reply.status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
                    reply.original_status = reply.status;
                } else {
                    const bool first = queries == 1;
                    const int rank = first && index == 3    ? 7
                                     : first && index == 4  ? -1
                                     : first && index == 5  ? 9
                                     : first && index == 10 ? 0
                                                            : 2;
                    const std::uint32_t mode =
                        first && index == 6 ? 2U
                                            : static_cast<std::uint32_t>(CUDNN_CROSS_CORRELATION);
                    const std::uint32_t type =
                        first && index == 7 ? 32U : static_cast<std::uint32_t>(CUDNN_DATA_FLOAT);
                    // Encode the documented field order directly to inject
                    // an invalid rank without asking the validated value codec
                    // to accept it or making any fixture native copy unbounded.
                    ExecutionPayloadWriter output;
                    expect(output.write_u32(mode) && output.write_u32(type) &&
                               output.write_u32(std::bit_cast<std::uint32_t>(rank)),
                           "encode raw native metadata scalars");
                    constexpr std::array<std::array<int, 8>, 3> arrays{
                        {{1, 2, 7}, {3, 4, 8}, {5, 6, 9}}};
                    for (const auto& values : arrays)
                        for (const int value : values)
                            expect(output.write_u32(static_cast<std::uint32_t>(value)),
                                   "encode bounded retained metadata slots");
                    reply.data = output.take();
                    if (first && (index == 11 || index == 12)) reply.parameter_bytes = 1;
                    if (first && index == 13) reply.id = 19;
                    if (first && index == 14) reply.address = 19;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "envelope reaches query-level validation through real wire codec");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *encoded}),
                   "send hostile native metadata result");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == 1 && queries == (clean ? 2U : 1U),
           "no hidden getter retry, native fallback or extra bootstrap after quarantine");
    server.stop();
}
void isolated_convolution_replies(const char* executable, const char* provider) {
    for (unsigned int index = 0; index != kConvolutionReplyCases; ++index) {
        const auto text = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated convolution receipt");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, "--malformed-case", text.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("convolution requested-tail hostile case failed: " + text);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4,
               "independent convolution provider path and optional hostile mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies") {
            expect(argc == 3, "hostile parent has no case index");
            isolated_convolution_replies(argv[0], argv[1]);
            return 0;
        }
        if (mode == "--malformed-case") {
            expect(argc == 4, "isolated hostile case needs index");
            const std::string_view text{argv[3]};
            expect(!text.empty() && text.size() <= 2, "bounded decimal case index");
            unsigned int index{};
            for (const char digit : text) {
                expect(digit >= '0' && digit <= '9', "decimal case index");
                index = 10 * index + static_cast<unsigned int>(digit - '0');
            }
            expect(index < kConvolutionReplyCases, "known hostile case");
            malformed_convolution_reply(argv[1], index);
            ::_exit(0);
        }
        expect(argc == 2, "known public convolution test mode");
        run(argv[1]);
        std::cout << "Public convolution CPU IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.can_finish_find.store(true);
        fixture::control.can_finish_find.notify_all();
        fixture::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
