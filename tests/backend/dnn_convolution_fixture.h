#pragma once

#include "dnn_fixture.h"
#include "internal/cuda_dnn_execution.h"

#include <atomic>
#include <bit>
#include <map>
#include <vector>

namespace glimmer::tests::dnn_convolution {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;

struct Descriptor {
    bool alive{}, configured{};
    CudaDnnObjectKind kind{};
    CudaDnnFilter filter;
    CudaDnnConvolutionState convolution;
    int group{1};
    cudnnMathType_t math{CUDNN_DEFAULT_MATH};
    cudnnReorderType_t reorder{CUDNN_DEFAULT_REORDER};
};
struct Invocation {
    cudnnHandle_t handle{};
    std::array<void*, 4> descriptors{};
    const void* x{};
    const void* w{};
    void* y{};
    void* workspace{};
    std::size_t workspace_bytes{};
    std::uint32_t algorithm{};
    double alpha{}, beta{};
    cudaStream_t stream{};
};
struct State {
    std::array<Descriptor, 16> descriptors{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, outputs{}, forwards{};
    unsigned int workspace_queries{}, heuristics{}, max_queries{};
    unsigned int finds{};
    int find_requested{};
    int maximum{12}, returned{10};
    bool should_overshoot_count{}, should_corrupt_performance{};
    std::size_t workspace_required{32};
    cudnnStatus_t query_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t heuristic_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t forward_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t find_status{CUDNN_STATUS_SUCCESS};
    Invocation last;
};
inline State state;
struct Control {
    std::atomic<bool> can_stop{true};
    std::atomic<bool> fail_workspace{}, fail_heuristic{}, fail_forward{}, zero_workspace{};
    std::atomic<bool> fail_find{}, should_block_find{}, has_find_entered{}, can_finish_find{true};
    std::atomic<bool> should_change_find_context{}, should_change_sync_context{};
    std::atomic<int> synchronize_status{CUDA_SUCCESS};
    std::atomic<int> convolution_get_status{CUDNN_STATUS_SUCCESS};
    std::atomic<unsigned int> context_synchronizes{};
    std::atomic<unsigned int> context_creates{}, context_destroys{}, host_registers{},
        host_unregisters{};
};
inline Control control;
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
struct ContextResource {
    CUcontext context{};
    std::unique_ptr<int> identity;
};
struct Driver {
    CUcontext current{};
    bool has_library_ownership{};
    std::map<CUcontext, std::unique_ptr<int>> contexts;
    std::map<CUdeviceptr, Memory> memory;
    std::map<CUstream, ContextResource> streams;
    std::map<CUevent, ContextResource> events;
    std::map<void*, std::size_t> host_mappings;
    unsigned int frees{}, stream_destroys{}, event_destroys{};
};
inline thread_local Driver driver;
inline CUcontext current() {
    return driver.current;
}
inline void reset() {
    driver = {};
    state = {};
    base::reset(current);
    control.can_stop.store(true);
    control.fail_workspace.store(false);
    control.fail_heuristic.store(false);
    control.fail_forward.store(false);
    control.zero_workspace.store(false);
    control.fail_find.store(false);
    control.should_block_find.store(false);
    control.has_find_entered.store(false);
    control.can_finish_find.store(true);
    control.should_change_find_context.store(false);
    control.should_change_sync_context.store(false);
    control.synchronize_status.store(CUDA_SUCCESS);
    control.convolution_get_status.store(CUDNN_STATUS_SUCCESS);
    control.context_synchronizes.store(0);
    control.context_creates.store(0);
    control.context_destroys.store(0);
    control.host_registers.store(0);
    control.host_unregisters.store(0);
}
inline Descriptor& descriptor(void* pointer, CudaDnnObjectKind kind) {
    base::owner();
    for (auto& item : state.descriptors)
        if (&item == pointer) {
            base::require(item.alive && item.kind == kind);
            return item;
        }
    throw std::runtime_error("unknown independent convolution descriptor");
}
inline void* create(CudaDnnObjectKind kind) {
    base::owner();
    for (auto& item : state.descriptors)
        if (!item.alive) {
            item = {};
            item.alive = true;
            item.kind = kind;
            ++state.creates;
            return &item;
        }
    throw std::runtime_error("bounded independent convolution descriptor capacity");
}
inline cudnnStatus_t destroy(void* pointer, CudaDnnObjectKind kind) {
    descriptor(pointer, kind).alive = false;
    ++state.destroys;
    return CUDNN_STATUS_SUCCESS;
}
inline std::size_t tensor_bytes(const CudaDnnTensor& tensor) {
    std::size_t elements{1};
    for (int i = 0; i < tensor.dimensions; ++i) {
        const auto index = static_cast<std::size_t>(i);
        base::require(tensor.sizes[index] > 0 && tensor.strides[index] > 0);
        elements += static_cast<std::size_t>(tensor.sizes[index] - 1) *
                    static_cast<std::size_t>(tensor.strides[index]);
    }
    return elements * (tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float));
}
inline cudnnStatus_t set_filter(CudaDnnFilterHandle pointer, cudnnDataType_t type,
                                cudnnTensorFormat_t format, int rank, const int* sizes) {
    auto& item = descriptor(pointer, CudaDnnObjectKind::kFilter);
    ++state.sets;
    if (rank <= 0 || rank > 8 || format != CUDNN_TENSOR_NCHW) return CUDNN_STATUS_BAD_PARAM;
    item.filter = {static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(format), rank, {}};
    std::copy_n(sizes, rank, item.filter.sizes.begin());
    item.configured = true;
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t get_filter(CudaDnnFilterHandle pointer, int requested, cudnnDataType_t* type,
                                cudnnTensorFormat_t* format, int* rank, int* sizes) {
    auto& item = descriptor(pointer, CudaDnnObjectKind::kFilter);
    ++state.gets;
    *type = static_cast<cudnnDataType_t>(item.filter.type);
    *format = static_cast<cudnnTensorFormat_t>(item.filter.format);
    *rank = item.filter.dimensions;
    if (requested < 0 || requested > 8 || !item.configured) return CUDNN_STATUS_BAD_PARAM;
    std::copy_n(item.filter.sizes.begin(), std::min(requested, *rank), sizes);
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t set_convolution(CudaDnnConvolution pointer, int rank, const int* padding,
                                     const int* strides, const int* dilation,
                                     cudnnConvolutionMode_t mode, cudnnDataType_t type) {
    auto& item = descriptor(pointer, CudaDnnObjectKind::kConvolution);
    ++state.sets;
    if (rank <= 0) return CUDNN_STATUS_BAD_PARAM;
    if (rank > 6) return CUDNN_STATUS_NOT_SUPPORTED;
    // Audited 9.10.2 Nd and 2D setters overwrite only their supplied prefix.
    // A smaller subsequent rank does not erase previous native tail entries.
    item.convolution.mode = static_cast<std::uint32_t>(mode);
    item.convolution.type = static_cast<std::uint32_t>(type);
    item.convolution.dimensions = rank;
    std::copy_n(padding, rank, item.convolution.padding.begin());
    std::copy_n(strides, rank, item.convolution.strides.begin());
    std::copy_n(dilation, rank, item.convolution.dilation.begin());
    item.configured = true;
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t get_convolution(CudaDnnConvolution pointer, int requested, int* rank,
                                     int* padding, int* strides, int* dilation,
                                     cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
    auto& item = descriptor(pointer, CudaDnnObjectKind::kConvolution);
    ++state.gets;
    if (requested < 0) return CUDNN_STATUS_BAD_PARAM;
    if (requested > 6) return CUDNN_STATUS_NOT_SUPPORTED;
    *rank = item.convolution.dimensions;
    *mode = static_cast<cudnnConvolutionMode_t>(item.convolution.mode);
    *type = static_cast<cudnnDataType_t>(item.convolution.type);
    // Actual rank is a separate output; even a fresh rank-zero object exposes
    // requested initialized slots. Injected failure deliberately touches only
    // the owner's private results so public nonpublication can be verified.
    std::copy_n(item.convolution.padding.begin(), requested, padding);
    std::copy_n(item.convolution.strides.begin(), requested, strides);
    std::copy_n(item.convolution.dilation.begin(), requested, dilation);
    return static_cast<cudnnStatus_t>(control.convolution_get_status.load());
}
inline std::array<int, 4> output_shape(CudaDnnConvolution convolution, cudnnTensorDescriptor_t x,
                                       CudaDnnFilterHandle filter) {
    const auto& c = descriptor(convolution, CudaDnnObjectKind::kConvolution).convolution;
    const auto& f = descriptor(filter, CudaDnnObjectKind::kFilter).filter;
    const auto& t = base::object(x, CudaDnnObjectKind::kTensor).tensor;
    base::require(c.dimensions == 2 && f.dimensions == 4 && t.dimensions == 4);
    return {
        t.sizes[0], f.sizes[0],
        (t.sizes[2] + 2 * c.padding[0] - c.dilation[0] * (f.sizes[2] - 1) - 1) / c.strides[0] + 1,
        (t.sizes[3] + 2 * c.padding[1] - c.dilation[1] * (f.sizes[3] - 1) - 1) / c.strides[1] + 1};
}
inline void query_descriptors(cudnnHandle_t handle, cudnnTensorDescriptor_t x,
                              CudaDnnFilterHandle filter, CudaDnnConvolution convolution,
                              cudnnTensorDescriptor_t y) {
    (void)base::handle(handle);
    base::require(base::object(x, CudaDnnObjectKind::kTensor).configured &&
                  base::object(y, CudaDnnObjectKind::kTensor).configured &&
                  descriptor(filter, CudaDnnObjectKind::kFilter).configured &&
                  descriptor(convolution, CudaDnnObjectKind::kConvolution).configured);
}
// This is an injected CPU arithmetic implementation, not cuDNN/GPU evidence.
// Expected values in consumers are a separate fixed 18-element oracle.
template <typename Scalar>
inline void forward_values(const Invocation& call) {
    const auto& x = base::object(call.descriptors[0], CudaDnnObjectKind::kTensor).tensor;
    const auto& w = descriptor(call.descriptors[1], CudaDnnObjectKind::kFilter).filter;
    const auto& c = descriptor(call.descriptors[2], CudaDnnObjectKind::kConvolution);
    const auto& y = base::object(call.descriptors[3], CudaDnnObjectKind::kTensor).tensor;
    base::require(x.dimensions == 4 && w.dimensions == 4 && y.dimensions == 4 &&
                  c.convolution.dimensions == 2 && c.group == 1);
    const auto* input = static_cast<const Scalar*>(call.x);
    const auto* filter = static_cast<const Scalar*>(call.w);
    auto* output = static_cast<Scalar*>(call.y);
    for (int n = 0; n < y.sizes[0]; ++n)
        for (int k = 0; k < y.sizes[1]; ++k)
            for (int h = 0; h < y.sizes[2]; ++h)
                for (int v = 0; v < y.sizes[3]; ++v) {
                    Scalar sum{};
                    for (int channel = 0; channel < x.sizes[1]; ++channel)
                        for (int r = 0; r < w.sizes[2]; ++r)
                            for (int s = 0; s < w.sizes[3]; ++s) {
                                const int ih = h * c.convolution.strides[0] -
                                               c.convolution.padding[0] +
                                               r * c.convolution.dilation[0];
                                const int iw = v * c.convolution.strides[1] -
                                               c.convolution.padding[1] +
                                               s * c.convolution.dilation[1];
                                if (ih < 0 || ih >= x.sizes[2] || iw < 0 || iw >= x.sizes[3])
                                    continue;
                                const int fr = c.convolution.mode == CUDNN_CONVOLUTION
                                                   ? w.sizes[2] - 1 - r
                                                   : r;
                                const int fs = c.convolution.mode == CUDNN_CONVOLUTION
                                                   ? w.sizes[3] - 1 - s
                                                   : s;
                                const auto input_index = n * x.strides[0] + channel * x.strides[1] +
                                                         ih * x.strides[2] + iw * x.strides[3];
                                const auto filter_index =
                                    ((k * w.sizes[1] + channel) * w.sizes[2] + fr) * w.sizes[3] +
                                    fs;
                                sum += input[input_index] * filter[filter_index];
                            }
                    const auto output_index =
                        n * y.strides[0] + k * y.strides[1] + h * y.strides[2] + v * y.strides[3];
                    output[output_index] = static_cast<Scalar>(call.alpha) * sum +
                                           static_cast<Scalar>(call.beta) * output[output_index];
                }
}
inline constexpr std::array<double, 18> kProduct{1.125,  -0.625, -0.625, -1.25, -1.625, 1.5,
                                                 0.875,  0.5,    0.125,  0.375, 0.875,  0.5,
                                                 -0.375, 0.625,  -1,     -0.75, -0.25,  0.25};
template <typename Scalar>
inline auto values() {
    struct Values {
        std::array<Scalar, 40> x;
        std::array<Scalar, 24> w;
        std::array<Scalar, 18> y;
    } result;
    for (std::size_t i = 0; i < result.x.size(); ++i)
        result.x[i] = static_cast<Scalar>(static_cast<int>(i % 7) - 3) / Scalar{2};
    for (std::size_t i = 0; i < result.w.size(); ++i)
        result.w[i] = static_cast<Scalar>(static_cast<int>(i % 5) - 2) / Scalar{4};
    for (std::size_t i = 0; i < result.y.size(); ++i)
        result.y[i] = Scalar{1} + static_cast<Scalar>(i) / Scalar{8};
    return result;
}
inline cudnnStatus_t fill_performances(int requested, int* count,
                                       cudnnConvolutionFwdAlgoPerfStruct* results) {
    if (requested <= 0) {
        *count = 991;
        return CUDNN_STATUS_BAD_PARAM;
    }
    *count = std::min(requested, state.returned);
    for (int i = 0; i < *count; ++i) {
        auto& item = results[i];
        item.algo = static_cast<cudnnConvolutionFwdAlgo_t>(i % 8);
        item.status = i == 2 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_SUCCESS;
        item.time = std::bit_cast<float>(0x3f000000U + static_cast<std::uint32_t>(i) * 0x1000U);
        item.memory = (std::size_t{1} << 33) + static_cast<std::size_t>(i);
        item.determinism = static_cast<cudnnDeterminism_t>(i % 2);
        item.mathType = static_cast<cudnnMathType_t>(i % 4);
        item.reserved[0] = -7;
        item.reserved[1] = i;
        item.reserved[2] = 31337;
    }
    if (state.should_corrupt_performance && *count > 1) {
        const std::uint32_t bits = 16;
        std::memcpy(&results[1].algo, &bits, sizeof(bits));
    }
    if (state.should_overshoot_count) *count = requested + 1;
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.create = [](cudnnHandle_t* output) {
        driver.has_library_ownership = true;
        return base::dispatch().create(output);
    };
    sdk.tensor_size = [](cudnnTensorDescriptor_t pointer, std::size_t* output) {
        ++base::state.sizes;
        const auto& tensor = base::object(pointer, CudaDnnObjectKind::kTensor);
        if (!tensor.configured) return CUDNN_STATUS_BAD_PARAM;
        *output = tensor_bytes(tensor.tensor);
        return base::state.size_status;
    };
    sdk.filter_create = [](CudaDnnFilterHandle* output) {
        *output = static_cast<CudaDnnFilterHandle>(create(CudaDnnObjectKind::kFilter));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_destroy = [](CudaDnnFilterHandle pointer) {
        return destroy(pointer, CudaDnnObjectKind::kFilter);
    };
    sdk.filter_set_4d = [](CudaDnnFilterHandle pointer, cudnnDataType_t type,
                           cudnnTensorFormat_t format, int k, int c, int h, int w) {
        const std::array sizes{k, c, h, w};
        return set_filter(pointer, type, format, 4, sizes.data());
    };
    sdk.filter_set_nd = set_filter;
    sdk.filter_get_nd = get_filter;
    sdk.filter_get_4d = [](CudaDnnFilterHandle pointer, cudnnDataType_t* type,
                           cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w) {
        int rank{};
        std::array<int, 4> sizes{};
        const auto status = get_filter(pointer, 4, type, format, &rank, sizes.data());
        *k = sizes[0];
        *c = sizes[1];
        *h = sizes[2];
        *w = sizes[3];
        return status;
    };
    sdk.filter_size = [](CudaDnnFilterHandle pointer, std::size_t* output) {
        const auto& item = descriptor(pointer, CudaDnnObjectKind::kFilter);
        if (!item.configured) return CUDNN_STATUS_BAD_PARAM;
        std::size_t bytes = item.filter.type == CUDNN_DATA_DOUBLE ? 8 : 4;
        for (int i = 0; i < item.filter.dimensions; ++i)
            bytes *= static_cast<std::size_t>(item.filter.sizes[static_cast<std::size_t>(i)]);
        *output = bytes;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_create = [](CudaDnnConvolution* output) {
        *output = static_cast<CudaDnnConvolution>(create(CudaDnnObjectKind::kConvolution));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_destroy = [](CudaDnnConvolution pointer) {
        return destroy(pointer, CudaDnnObjectKind::kConvolution);
    };
    sdk.convolution_set_nd = set_convolution;
    sdk.convolution_get_nd = get_convolution;
    sdk.convolution_set_2d = [](CudaDnnConvolution pointer, int ph, int pw, int sh, int sw, int dh,
                                int dw, cudnnConvolutionMode_t mode, cudnnDataType_t type) {
        const std::array padding{ph, pw}, strides{sh, sw}, dilation{dh, dw};
        return set_convolution(pointer, 2, padding.data(), strides.data(), dilation.data(), mode,
                               type);
    };
    sdk.convolution_get_2d = [](CudaDnnConvolution pointer, int* ph, int* pw, int* sh, int* sw,
                                int* dh, int* dw, cudnnConvolutionMode_t* mode,
                                cudnnDataType_t* type) {
        int rank{};
        std::array<int, 2> padding{}, strides{}, dilation{};
        const auto status = get_convolution(pointer, 2, &rank, padding.data(), strides.data(),
                                            dilation.data(), mode, type);
        *ph = padding[0];
        *pw = padding[1];
        *sh = strides[0];
        *sw = strides[1];
        *dh = dilation[0];
        *dw = dilation[1];
        return status;
    };
    sdk.convolution_group_set = [](CudaDnnConvolution pointer, int group) {
        descriptor(pointer, CudaDnnObjectKind::kConvolution).group = group;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_group_get = [](CudaDnnConvolution pointer, int* output) {
        *output = descriptor(pointer, CudaDnnObjectKind::kConvolution).group;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_math_set = [](CudaDnnConvolution pointer, cudnnMathType_t math) {
        descriptor(pointer, CudaDnnObjectKind::kConvolution).math = math;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_math_get = [](CudaDnnConvolution pointer, cudnnMathType_t* output) {
        *output = descriptor(pointer, CudaDnnObjectKind::kConvolution).math;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_reorder_set = [](CudaDnnConvolution pointer, cudnnReorderType_t reorder) {
        descriptor(pointer, CudaDnnObjectKind::kConvolution).reorder = reorder;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_reorder_get = [](CudaDnnConvolution pointer, cudnnReorderType_t* output) {
        *output = descriptor(pointer, CudaDnnObjectKind::kConvolution).reorder;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_output_nd = [](CudaDnnConvolution c, cudnnTensorDescriptor_t x,
                                   CudaDnnFilterHandle w, int rank, int* output) {
        ++state.outputs;
        if (rank != 4) return CUDNN_STATUS_BAD_PARAM;
        const auto sizes = output_shape(c, x, w);
        std::copy(sizes.begin(), sizes.end(), output);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_output_2d = [](CudaDnnConvolution c, cudnnTensorDescriptor_t x,
                                   CudaDnnFilterHandle w, int* n, int* k, int* h, int* v) {
        ++state.outputs;
        const auto sizes = output_shape(c, x, w);
        *n = sizes[0];
        *k = sizes[1];
        *h = sizes[2];
        *v = sizes[3];
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_forward_max_count = [](cudnnHandle_t handle, int* count) {
        (void)base::handle(handle);
        ++state.max_queries;
        *count = state.maximum;
        return state.query_status;
    };
    sdk.convolution_forward_workspace = [](cudnnHandle_t handle, cudnnTensorDescriptor_t x,
                                           CudaDnnFilterHandle w, CudaDnnConvolution c,
                                           cudnnTensorDescriptor_t y, cudnnConvolutionFwdAlgo_t,
                                           std::size_t* bytes) {
        query_descriptors(handle, x, w, c, y);
        ++state.workspace_queries;
        *bytes = control.zero_workspace.load() ? 0 : state.workspace_required;
        if (control.fail_workspace.exchange(false)) return CUDNN_STATUS_BAD_PARAM;
        return state.query_status;
    };
    sdk.convolution_forward_heuristic = [](cudnnHandle_t handle, cudnnTensorDescriptor_t x,
                                           CudaDnnFilterHandle w, CudaDnnConvolution c,
                                           cudnnTensorDescriptor_t y, int requested, int* count,
                                           cudnnConvolutionFwdAlgoPerfStruct* results) {
        query_descriptors(handle, x, w, c, y);
        ++state.heuristics;
        if (const auto status = fill_performances(requested, count, results);
            status != CUDNN_STATUS_SUCCESS)
            return status;
        if (control.fail_heuristic.exchange(false)) return CUDNN_STATUS_NOT_SUPPORTED;
        return state.heuristic_status;
    };
    sdk.convolution_forward_find = [](cudnnHandle_t handle, cudnnTensorDescriptor_t x,
                                      CudaDnnFilterHandle w, CudaDnnConvolution c,
                                      cudnnTensorDescriptor_t y, int requested, int* count,
                                      cudnnConvolutionFwdAlgoPerfStruct* results) {
        query_descriptors(handle, x, w, c, y);
        ++state.finds;
        state.find_requested = requested;
        state.last.handle = handle;
        state.last.descriptors = {x, w, c, y};
        state.last.stream = base::handle(handle).stream;
        control.has_find_entered.store(true);
        control.has_find_entered.notify_all();
        if (control.should_block_find.load())
            while (!control.can_finish_find.load()) control.can_finish_find.wait(false);
        const auto filled = fill_performances(requested, count, results);
        if (control.should_change_find_context.load()) driver.current = nullptr;
        if (control.fail_find.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
        return filled == CUDNN_STATUS_SUCCESS ? state.find_status : filled;
    };
    sdk.convolution_forward = [](cudnnHandle_t handle, const void* alpha,
                                 cudnnTensorDescriptor_t xd, const void* x, CudaDnnFilterHandle wd,
                                 const void* w, CudaDnnConvolution cd,
                                 cudnnConvolutionFwdAlgo_t algo, void* workspace, std::size_t bytes,
                                 const void* beta, cudnnTensorDescriptor_t yd, void* y) {
        query_descriptors(handle, xd, wd, cd, yd);
        ++state.forwards;
        Invocation call{handle,
                        {xd, wd, cd, yd},
                        x,
                        w,
                        y,
                        workspace,
                        bytes,
                        static_cast<std::uint32_t>(algo),
                        0,
                        0,
                        base::handle(handle).stream};
        const bool use_double =
            base::object(xd, CudaDnnObjectKind::kTensor).tensor.type == CUDNN_DATA_DOUBLE;
        base::require(alpha && beta && x && w && y);
        if (use_double) {
            base::require(reinterpret_cast<std::uintptr_t>(alpha) % alignof(double) == 0 &&
                          reinterpret_cast<std::uintptr_t>(beta) % alignof(double) == 0);
            std::memcpy(&call.alpha, alpha, sizeof(double));
            std::memcpy(&call.beta, beta, sizeof(double));
        } else {
            float a{}, b{};
            std::memcpy(&a, alpha, sizeof(float));
            std::memcpy(&b, beta, sizeof(float));
            call.alpha = a;
            call.beta = b;
        }
        state.last = call;
        if (control.fail_forward.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
        if (state.forward_status != CUDNN_STATUS_SUCCESS) return state.forward_status;
        if (bytes < (control.zero_workspace.load() ? 0 : state.workspace_required) ||
            (bytes && !workspace))
            return CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT;
        if (algo > CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) return CUDNN_STATUS_NOT_SUPPORTED;
        if (use_double)
            forward_values<double>(call);
        else
            forward_values<float>(call);
        if (bytes) {
            auto* memory = static_cast<std::byte*>(workspace);
            memory[0] = memory[bytes - 1] = std::byte{0xa5};
        }
        return CUDNN_STATUS_SUCCESS;
    };
    return sdk;
}

inline CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* value, int ordinal) {
        *value = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        auto value = std::make_unique<int>(0);
        *output = reinterpret_cast<CUcontext>(value.get());
        driver.contexts.emplace(*output, std::move(value));
        driver.current = *output;
        ++control.context_creates;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext value) {
        base::require(!value || driver.contexts.contains(value));
        driver.current = value;
        return CUDA_SUCCESS;
    };
    sdk.context_synchronize = [] {
        base::require(driver.current && driver.contexts.contains(driver.current));
        ++control.context_synchronizes;
        if (control.should_change_sync_context.load()) driver.current = nullptr;
        return static_cast<CUresult>(control.synchronize_status.load());
    };
    sdk.context_destroy = [](CUcontext value) {
        base::require(value && value == driver.current && driver.contexts.contains(value) &&
                      driver.host_mappings.empty());
        if (driver.has_library_ownership)
            for (const auto& handle : base::state.handles)
                base::require(!handle.alive || handle.context != value);
        std::erase_if(driver.memory,
                      [=](const auto& item) { return item.second.context == value; });
        // cuCtxDestroy also releases resources that still belong to that
        // context; it must not consume another owned context's identities.
        std::erase_if(driver.streams,
                      [=](const auto& item) { return item.second.context == value; });
        std::erase_if(driver.events,
                      [=](const auto& item) { return item.second.context == value; });
        driver.contexts.erase(value);
        driver.current = nullptr;
        ++control.context_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        *output = driver.current;
        return CUDA_SUCCESS;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        Memory allocation{driver.current, std::vector<std::byte>(bytes, std::byte{0xcd})};
        *output = reinterpret_cast<CUdeviceptr>(allocation.bytes.data());
        base::require(driver.current && *output);
        driver.memory.emplace(*output, std::move(allocation));
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr pointer) {
        base::require(driver.memory.at(pointer).context == driver.current);
        driver.memory.erase(pointer);
        ++driver.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                          CUstream) {
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download_async = [](void* destination, CUdeviceptr source, std::size_t bytes, CUstream) {
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* output, void* pointer, unsigned int) {
        base::require(driver.current);
        *output =
            driver.host_mappings.contains(pointer) ? reinterpret_cast<CUdeviceptr>(pointer) : 0;
        return *output ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    };
    sdk.host_register = [](void* pointer, std::size_t bytes, unsigned int) {
        base::require(driver.current && driver.host_mappings.emplace(pointer, bytes).second);
        ++control.host_registers;
        return CUDA_SUCCESS;
    };
    sdk.host_unregister = [](void* pointer) {
        base::require(driver.current && driver.host_mappings.erase(pointer) == 1);
        ++control.host_unregisters;
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        base::require(driver.current);
        auto value = std::make_unique<int>(0);
        *output = reinterpret_cast<CUstream>(value.get());
        driver.streams.emplace(*output, ContextResource{driver.current, std::move(value)});
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream stream) {
        base::require(driver.streams.at(stream).context == driver.current);
        base::require(driver.streams.erase(stream) == 1);
        ++driver.stream_destroys;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        base::require(driver.current);
        auto value = std::make_unique<int>(0);
        *output = reinterpret_cast<CUevent>(value.get());
        driver.events.emplace(*output, ContextResource{driver.current, std::move(value)});
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        base::require(driver.events.contains(event));
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent event) {
        base::require(driver.events.contains(event));
        return control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        base::require(driver.events.at(event).context == driver.current);
        base::require(driver.events.erase(event) == 1);
        ++driver.event_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        return control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_convolution
