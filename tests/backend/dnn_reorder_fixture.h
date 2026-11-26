#pragma once

#include "dnn_convolution_fixture.h"

// Synthetic byte permutation/copy provider: validates the original-call ABI and
// ownership only. It is not an implementation of NVIDIA's DEFAULT permutation.
namespace glimmer::tests::dnn_reorder {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace driver = glimmer::tests::dnn_convolution;
inline constexpr std::array<std::size_t, 17> kWidths{4, 8, 2, 1, 4, 4, 1, 4, 32,
                                                     2, 8, 1, 1, 1, 4, 1, 1};
struct Descriptor {
    bool alive{}, configured{};
    CudaDnnFilter filter;
    unsigned int destroy_attempts{};
};
struct Invocation {
    cudnnHandle_t handle{};
    cudaStream_t stream{};
    CudaDnnFilterHandle filter{};
    std::uint32_t reorder{};
    int reorder_bias{};
    std::array<const void*, 4> buffers{};
    std::size_t filter_bytes{}, bias_bytes{};
};
struct State {
    std::array<Descriptor, 16> descriptors{};
    unsigned int creates{}, destroys{}, sets{}, nd_queries{}, four_queries{}, size_queries{},
        version_queries{}, calls{};
    Invocation last;
};
struct Control {
    std::atomic<std::size_t> version{91002}, reported_size{};
    std::atomic<int> query_status{CUDNN_STATUS_SUCCESS}, size_status{CUDNN_STATUS_SUCCESS},
        compute_status{CUDNN_STATUS_SUCCESS}, reported_type{-1}, reported_rank{-1};
    std::atomic<unsigned int> calls{}, last_reorder{};
    std::atomic<int> last_flag{};
};
inline State state;
inline Control control;
inline void reset() {
    driver::reset();
    state = {};
    control.version.store(91002);
    control.reported_size.store(0);
    control.query_status.store(CUDNN_STATUS_SUCCESS);
    control.size_status.store(CUDNN_STATUS_SUCCESS);
    control.compute_status.store(CUDNN_STATUS_SUCCESS);
    control.reported_type.store(-1);
    control.reported_rank.store(-1);
    control.calls.store(0);
    control.last_reorder.store(0);
    control.last_flag.store(0);
}
inline Descriptor& descriptor(CudaDnnFilterHandle native) {
    base::owner();
    for (auto& value : state.descriptors)
        if (reinterpret_cast<CudaDnnFilterHandle>(&value) == native) {
            base::require(value.alive);
            return value;
        }
    throw std::runtime_error("unknown reorder filter descriptor");
}
inline std::size_t filter_bytes(const CudaDnnFilter& filter) {
    base::require(filter.type < kWidths.size());
    const std::size_t vector = filter.type == 8 ? 32 : filter.type == 5 || filter.type == 7 ? 4 : 1;
    std::size_t elements{1};
    for (int i = 0; i != filter.dimensions; ++i) {
        base::require(filter.sizes[static_cast<std::size_t>(i)] > 0);
        elements *= static_cast<unsigned int>(filter.sizes[static_cast<std::size_t>(i)]);
    }
    return elements * kWidths[filter.type] / vector;
}
inline std::size_t bias_bytes(const CudaDnnFilter& filter, std::uint32_t reorder) {
    return static_cast<unsigned int>(filter.sizes[0]) * (reorder ? kWidths.at(filter.type) : 4U);
}
inline std::span<std::byte> memory(const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<CUdeviceptr>(pointer);
    for (auto& [start, allocation] : driver::driver.memory)
        if (address >= start && address - start <= allocation.bytes.size() &&
            bytes <= allocation.bytes.size() - static_cast<std::size_t>(address - start)) {
            base::require(allocation.context == driver::current());
            return std::span(allocation.bytes)
                .subspan(static_cast<std::size_t>(address - start), bytes);
        }
    throw std::runtime_error("original reorder touched an unowned or short native range");
}
inline cudnnStatus_t set_filter(CudaDnnFilterHandle native, cudnnDataType_t type,
                                cudnnTensorFormat_t format, int rank, const int* sizes) {
    auto& value = descriptor(native);
    ++state.sets;
    if (rank < 1 || rank > 8) return CUDNN_STATUS_BAD_PARAM;
    value.filter = {static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(format), rank, {}};
    std::copy_n(sizes, rank, value.filter.sizes.begin());
    value.configured = true;
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.filter_create = [](CudaDnnFilterHandle* output) {
        base::owner();
        for (auto& value : state.descriptors)
            if (!value.alive) {
                value = {};
                value.alive = true;
                ++state.creates;
                *output = reinterpret_cast<CudaDnnFilterHandle>(&value);
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.filter_destroy = [](CudaDnnFilterHandle native) {
        auto& value = descriptor(native);
        base::require(++value.destroy_attempts == 1);
        value.alive = false;
        ++state.destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_set_nd = &set_filter;
    sdk.filter_set_4d = [](CudaDnnFilterHandle native, cudnnDataType_t type,
                           cudnnTensorFormat_t format, int k, int c, int h, int w) {
        const std::array sizes{k, c, h, w};
        return set_filter(native, type, format, 4, sizes.data());
    };
    sdk.filter_get_nd = [](CudaDnnFilterHandle native, int requested, cudnnDataType_t* type,
                           cudnnTensorFormat_t* format, int* rank, int* sizes) {
        const auto& value = descriptor(native);
        ++state.nd_queries;
        base::require(value.configured && requested == 8);
        const auto raw_type = control.reported_type.load() < 0
                                  ? value.filter.type
                                  : static_cast<unsigned int>(control.reported_type.load());
        std::memcpy(type, &raw_type, sizeof(*type));
        *format = static_cast<cudnnTensorFormat_t>(value.filter.format);
        *rank = control.reported_rank.load() < 0 ? value.filter.dimensions
                                                 : control.reported_rank.load();
        std::copy_n(value.filter.sizes.begin(), value.filter.dimensions, sizes);
        return static_cast<cudnnStatus_t>(control.query_status.load());
    };
    sdk.filter_get_4d = [](CudaDnnFilterHandle, cudnnDataType_t*, cudnnTensorFormat_t*, int*, int*,
                           int*, int*) {
        ++state.four_queries;
        return CUDNN_STATUS_NOT_SUPPORTED;
    };
    sdk.filter_size = [](CudaDnnFilterHandle native, std::size_t* bytes) {
        const auto& value = descriptor(native);
        ++state.size_queries;
        *bytes = control.reported_size.load() ? control.reported_size.load()
                                              : filter_bytes(value.filter);
        return static_cast<cudnnStatus_t>(control.size_status.load());
    };
    sdk.version = [] {
        base::owner();
        ++state.version_queries;
        return control.version.load();
    };
    sdk.reorder_filter_bias = [](cudnnHandle_t handle, CudaDnnFilterHandle native,
                                 decltype(CUDNN_DEFAULT_REORDER) reorder, const void* input,
                                 void* output, int flag, const void* bias, void* bias_output) {
        const auto& owned = base::handle(handle);
        const auto filter = descriptor(native).filter;
        ++state.calls;
        control.last_reorder.store(static_cast<std::uint32_t>(reorder));
        control.last_flag.store(flag);
        ++control.calls;
        state.last = {handle, owned.stream,
                      native, static_cast<std::uint32_t>(reorder),
                      flag,   {input, output, bias, bias_output},
                      0,      0};
        if (filter.dimensions != 4) return CUDNN_STATUS_BAD_PARAM;
        if (reorder == CUDNN_DEFAULT_REORDER &&
            (filter.type != 8 || filter.format != CUDNN_TENSOR_NCHW_VECT_C))
            return CUDNN_STATUS_NOT_SUPPORTED;
        const auto bytes = filter_bytes(filter);
        state.last.filter_bytes = bytes;
        auto source = memory(input, bytes), destination = memory(output, bytes);
        if (reorder == CUDNN_DEFAULT_REORDER)
            std::reverse_copy(source.begin(), source.end(), destination.begin());
        else
            std::copy(source.begin(), source.end(), destination.begin());
        // A real filter launch can precede a bias failure. Deliberately
        // write the full filter then fail before touching the bias.
        const auto status = static_cast<cudnnStatus_t>(control.compute_status.load());
        if (status != CUDNN_STATUS_SUCCESS) return status;
        if (flag) {
            const auto count = bias_bytes(filter, static_cast<std::uint32_t>(reorder));
            state.last.bias_bytes = count;
            auto bias_input = memory(bias, count), bias_destination = memory(bias_output, count);
            if (reorder == CUDNN_DEFAULT_REORDER)
                std::reverse_copy(bias_input.begin(), bias_input.end(), bias_destination.begin());
            else
                std::copy(bias_input.begin(), bias_input.end(), bias_destination.begin());
        } else {
            base::require(!bias && !bias_output);
        }
        return CUDNN_STATUS_SUCCESS;
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = driver::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
inline CudaDnnFilter filter(std::uint32_t type = 8) {
    return {type,
            type == 5 || type == 7 || type == 8
                ? static_cast<std::uint32_t>(CUDNN_TENSOR_NCHW_VECT_C)
                : static_cast<std::uint32_t>(CUDNN_TENSOR_NCHW),
            4,
            {32, 32, 1, 1}};
}
}  // namespace glimmer::tests::dnn_reorder
