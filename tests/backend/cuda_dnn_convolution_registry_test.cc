#include "dnn_fixture.h"

#include <bit>
#include <future>
#include <iostream>
#include <optional>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace sdk = glimmer::tests::dnn;
using Attribute = CudaDnnConvolutionAttribute;
using Kind = CudaDnnObjectKind;
using Reorder = decltype(CUDNN_DEFAULT_REORDER);
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "successful native completion retains its status");
}
void original(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && !result.id,
           "native error is not a policy error");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.id,
           "pre-SDK rejection has no invented native status");
}
struct Object {
    Kind kind{};
    bool alive{};
    unsigned int destroys{};
    CudaDnnFilter filter;
    CudaDnnConvolutionState convolution;
    std::array<std::uint32_t, 3> attributes{1, 0, 0};
};
struct Fixture {
    std::array<Object, 16> objects;
    unsigned int creates{}, destroys{}, sets{}, gets{}, sizes{}, shapes{};
    std::array<unsigned int, 4> setter_calls{}, getter_calls{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS},
        set_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS},
        size_status{CUDNN_STATUS_SUCCESS}, shape_status{CUDNN_STATUS_SUCCESS};
    void* duplicate{};
    bool partial{}, null_success{}, invalid_capacity_success{};
    std::optional<std::uint32_t> reported_type, reported_format, reported_mode, reported_attribute;
    std::optional<int> reported_rank;
    int last_requested{};
    unsigned int fail_destroy_call{};
} fixture;
void reset() {
    sdk::reset();
    fixture = {};
}
Object& object(void* pointer, Kind kind) {
    sdk::owner();
    for (auto& entry : fixture.objects)
        if (&entry == pointer) {
            expect(entry.alive && entry.kind == kind, "fake native object has exact live kind");
            return entry;
        }
    throw std::runtime_error("unknown native convolution descriptor");
}
void* create(Kind kind) {
    sdk::owner();
    ++fixture.creates;
    if (fixture.null_success || (fixture.create_status != CUDNN_STATUS_SUCCESS && !fixture.partial))
        return nullptr;
    if (fixture.duplicate) return fixture.duplicate;
    for (auto& entry : fixture.objects)
        if (!entry.alive) {
            entry = {};
            entry.kind = kind;
            entry.alive = true;
            return &entry;
        }
    throw std::runtime_error("fake convolution object capacity");
}
cudnnStatus_t destroy(void* pointer, Kind kind) {
    auto& entry = object(pointer, kind);
    expect(++entry.destroys == 1, "native destruction is never retried");
    ++fixture.destroys;
    const auto status = fixture.fail_destroy_call == fixture.destroys ? CUDNN_STATUS_INTERNAL_ERROR
                                                                      : fixture.destroy_status;
    if (status == CUDNN_STATUS_SUCCESS) entry.alive = false;
    return status;
}
template <typename Value>
void store_bits(Value* output, std::uint32_t bits) {
    static_assert(sizeof(Value) == sizeof(bits));
    std::memcpy(output, &bits, sizeof(bits));
}
cudnnStatus_t set_filter(CudaDnnFilterHandle pointer, CudaDnnFilter value, unsigned int index) {
    auto& entry = object(pointer, Kind::kFilter);
    ++fixture.sets;
    ++fixture.setter_calls[index];
    if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
    if (value.type > CUDNN_DATA_FP4_E2M1 || value.format > CUDNN_TENSOR_NCHW_VECT_C)
        return CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
    if (!value.dimensions || value.sizes[0] <= 0) return CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    entry.filter = value;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t get_filter(CudaDnnFilterHandle pointer, int requested, cudnnDataType_t* type,
                         cudnnTensorFormat_t* format, int* rank, int* sizes, unsigned int index) {
    const auto& value = object(pointer, Kind::kFilter).filter;
    ++fixture.gets;
    ++fixture.getter_calls[index];
    fixture.last_requested = requested;
    store_bits(type, fixture.reported_type.value_or(value.type));
    store_bits(format, fixture.reported_format.value_or(value.format));
    *rank = fixture.reported_rank.value_or(value.dimensions);
    const auto count = std::min(std::max(requested, 0), value.dimensions);
    std::copy_n(value.sizes.begin(), count, sizes);
    if (fixture.get_status != CUDNN_STATUS_SUCCESS) return fixture.get_status;
    return requested < 0 && !fixture.invalid_capacity_success ? CUDNN_STATUS_BAD_PARAM
                                                              : CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t set_convolution(CudaDnnConvolution pointer, CudaDnnConvolutionState value,
                              unsigned int index) {
    auto& entry = object(pointer, Kind::kConvolution);
    ++fixture.sets;
    ++fixture.setter_calls[index];
    if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
    if (value.dimensions > CUDNN_DIM_MAX - 2) return CUDNN_STATUS_NOT_SUPPORTED;
    if (!value.dimensions || value.padding[0] < 0) return CUDNN_STATUS_BAD_PARAM;
    entry.convolution.mode = value.mode;
    entry.convolution.type = value.type;
    entry.convolution.dimensions = value.dimensions;
    std::copy_n(value.padding.begin(), value.dimensions, entry.convolution.padding.begin());
    std::copy_n(value.strides.begin(), value.dimensions, entry.convolution.strides.begin());
    std::copy_n(value.dilation.begin(), value.dimensions, entry.convolution.dilation.begin());
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t get_convolution(CudaDnnConvolution pointer, int requested, int* rank, int* padding,
                              int* strides, int* dilation, cudnnConvolutionMode_t* mode,
                              cudnnDataType_t* type, unsigned int index) {
    const auto& value = object(pointer, Kind::kConvolution).convolution;
    ++fixture.gets;
    ++fixture.getter_calls[index];
    fixture.last_requested = requested;
    store_bits(mode, fixture.reported_mode.value_or(value.mode));
    store_bits(type, fixture.reported_type.value_or(value.type));
    *rank = fixture.reported_rank.value_or(value.dimensions);
    // A hostile successful request must not make this test provider overrun
    // fixed native arrays; legal native getters copy requested, not actual rank.
    const auto count = std::clamp(requested, 0, CUDNN_DIM_MAX - 2);
    std::copy_n(value.padding.begin(), count, padding);
    std::copy_n(value.strides.begin(), count, strides);
    std::copy_n(value.dilation.begin(), count, dilation);
    if (fixture.get_status != CUDNN_STATUS_SUCCESS) return fixture.get_status;
    if (!fixture.invalid_capacity_success) {
        if (requested < 0) return CUDNN_STATUS_BAD_PARAM;
        if (requested > CUDNN_DIM_MAX - 2) return CUDNN_STATUS_NOT_SUPPORTED;
    }
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t set_attribute(CudaDnnConvolution pointer, unsigned int index, std::uint32_t bits) {
    auto& entry = object(pointer, Kind::kConvolution);
    ++fixture.sets;
    if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
    if (!index && std::bit_cast<std::int32_t>(bits) <= 0) return CUDNN_STATUS_BAD_PARAM;
    entry.attributes[index] = bits;
    return CUDNN_STATUS_SUCCESS;
}
template <typename Value>
cudnnStatus_t get_attribute(CudaDnnConvolution pointer, unsigned int index, Value* output) {
    const auto& entry = object(pointer, Kind::kConvolution);
    ++fixture.gets;
    store_bits(output, fixture.reported_attribute.value_or(entry.attributes[index]));
    return fixture.get_status;
}
cudnnStatus_t shape(CudaDnnConvolution convolution, cudnnTensorDescriptor_t tensor,
                    CudaDnnFilterHandle filter, int dimensions, int* output) {
    (void)object(convolution, Kind::kConvolution);
    const auto& input = sdk::object(tensor, Kind::kTensor).tensor;
    const auto& weights = object(filter, Kind::kFilter).filter;
    ++fixture.shapes;
    for (int index = 0; index < dimensions; ++index)
        output[index] =
            index == 1 ? weights.sizes[0] : input.sizes[static_cast<std::size_t>(index)];
    return fixture.shape_status;
}
CudaDnnDispatch dispatch() {
    auto result = sdk::dispatch();
    result.filter_create = [](CudaDnnFilterHandle* output) {
        *output = static_cast<CudaDnnFilterHandle>(create(Kind::kFilter));
        return fixture.create_status;
    };
    result.filter_destroy = [](CudaDnnFilterHandle value) { return destroy(value, Kind::kFilter); };
    result.convolution_create = [](CudaDnnConvolution* output) {
        *output = static_cast<CudaDnnConvolution>(create(Kind::kConvolution));
        return fixture.create_status;
    };
    result.convolution_destroy = [](CudaDnnConvolution value) {
        return destroy(value, Kind::kConvolution);
    };
    result.filter_set_4d = [](CudaDnnFilterHandle pointer, cudnnDataType_t type,
                              cudnnTensorFormat_t format, int k, int c, int h, int w) {
        return set_filter(
            pointer,
            {static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(format), 4, {k, c, h, w}},
            0);
    };
    result.filter_set_nd = [](CudaDnnFilterHandle pointer, cudnnDataType_t type,
                              cudnnTensorFormat_t format, int rank, const int* sizes) {
        CudaDnnFilter value{
            static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(format), rank, {}};
        std::copy_n(sizes, rank, value.sizes.begin());
        return set_filter(pointer, value, 1);
    };
    result.filter_get_4d = [](CudaDnnFilterHandle pointer, cudnnDataType_t* type,
                              cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w) {
        int rank{};
        std::array<int, CUDNN_DIM_MAX> sizes{};
        const auto status = get_filter(pointer, 4, type, format, &rank, sizes.data(), 0);
        *k = sizes[0];
        *c = sizes[1];
        *h = sizes[2];
        *w = sizes[3];
        return status;
    };
    result.filter_get_nd = [](CudaDnnFilterHandle pointer, int requested, cudnnDataType_t* type,
                              cudnnTensorFormat_t* format, int* rank, int* sizes) {
        return get_filter(pointer, requested, type, format, rank, sizes, 1);
    };
    result.filter_size = [](CudaDnnFilterHandle pointer, std::size_t* output) {
        (void)object(pointer, Kind::kFilter);
        ++fixture.sizes;
        *output = (std::size_t{1} << 35) + 93;
        return fixture.size_status;
    };
    result.convolution_set_2d = [](CudaDnnConvolution pointer, int ph, int pw, int sh, int sw,
                                   int dh, int dw, cudnnConvolutionMode_t mode,
                                   cudnnDataType_t type) {
        return set_convolution(pointer,
                               {static_cast<std::uint32_t>(mode),
                                static_cast<std::uint32_t>(type),
                                2,
                                {ph, pw},
                                {sh, sw},
                                {dh, dw}},
                               2);
    };
    result.convolution_set_nd = [](CudaDnnConvolution pointer, int rank, const int* padding,
                                   const int* strides, const int* dilation,
                                   cudnnConvolutionMode_t mode, cudnnDataType_t type) {
        CudaDnnConvolutionState value{
            static_cast<std::uint32_t>(mode), static_cast<std::uint32_t>(type), rank, {}, {}, {}};
        std::copy_n(padding, rank, value.padding.begin());
        std::copy_n(strides, rank, value.strides.begin());
        std::copy_n(dilation, rank, value.dilation.begin());
        return set_convolution(pointer, value, 3);
    };
    result.convolution_get_2d = [](CudaDnnConvolution pointer, int* ph, int* pw, int* sh, int* sw,
                                   int* dh, int* dw, cudnnConvolutionMode_t* mode,
                                   cudnnDataType_t* type) {
        int rank{};
        std::array<int, CUDNN_DIM_MAX> padding{}, strides{}, dilation{};
        const auto status = get_convolution(pointer, 2, &rank, padding.data(), strides.data(),
                                            dilation.data(), mode, type, 2);
        *ph = padding[0];
        *pw = padding[1];
        *sh = strides[0];
        *sw = strides[1];
        *dh = dilation[0];
        *dw = dilation[1];
        return status;
    };
    result.convolution_get_nd = [](CudaDnnConvolution pointer, int requested, int* rank,
                                   int* padding, int* strides, int* dilation,
                                   cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
        return get_convolution(pointer, requested, rank, padding, strides, dilation, mode, type, 3);
    };
    result.convolution_group_set = [](CudaDnnConvolution pointer, int value) {
        return set_attribute(pointer, 0, std::bit_cast<std::uint32_t>(value));
    };
    result.convolution_group_get = [](CudaDnnConvolution pointer, int* output) {
        return get_attribute(pointer, 0, output);
    };
    result.convolution_math_set = [](CudaDnnConvolution pointer, cudnnMathType_t value) {
        return set_attribute(pointer, 1, static_cast<std::uint32_t>(value));
    };
    result.convolution_math_get = [](CudaDnnConvolution pointer, cudnnMathType_t* output) {
        return get_attribute(pointer, 1, output);
    };
    result.convolution_reorder_set = [](CudaDnnConvolution pointer, Reorder value) {
        return set_attribute(pointer, 2, static_cast<std::uint32_t>(value));
    };
    result.convolution_reorder_get = [](CudaDnnConvolution pointer, Reorder* output) {
        return get_attribute(pointer, 2, output);
    };
    result.convolution_output_nd = shape;
    result.convolution_output_2d = [](CudaDnnConvolution convolution,
                                      cudnnTensorDescriptor_t tensor, CudaDnnFilterHandle filter,
                                      int* n, int* c, int* h, int* w) {
        std::array<int, 4> output{};
        const auto status = shape(convolution, tensor, filter, 4, output.data());
        *n = output[0];
        *c = output[1];
        *h = output[2];
        *w = output[3];
        return status;
    };
    return result;
}
CudaDnnFilter filter_value() {
    return {CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NHWC, 4, {7, 3, 5, 9}};
}
CudaDnnConvolutionState convolution_value() {
    return {CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE, 2, {1, 2}, {3, 4}, {5, 6}};
}
void metadata_round_trip() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto filter = registry.create(Kind::kFilter).id;
    const auto convolution = registry.create(Kind::kConvolution).id;
    expect(filter && convolution && filter != convolution, "new kinds have distinct owned IDs");
    auto value = filter_value();
    success(registry.set_filter(filter, value, true));
    auto result = registry.get_filter(filter, 4, true);
    success(result);
    expect(result.filter.sizes == value.sizes && result.filter.format == value.format &&
               result.filter.type == value.type,
           "four-dimensional filter preserves every field");
    value.dimensions = 5;
    value.sizes[4] = 11;
    success(registry.set_filter(filter, value));
    result = registry.get_filter(filter, 12);
    success(result);
    expect(result.filter.dimensions == 5 && result.filter.sizes == value.sizes &&
               fixture.last_requested == 12,
           "oversized filter request preserves native capacity");
    result = registry.get_filter(filter, 2);
    success(result);
    expect(result.filter.dimensions == 5 && result.filter.sizes[1] == 3 && !result.filter.sizes[2],
           "filter query publishes actual rank and only requested prefix");
    result = registry.get_filter(filter, 0);
    success(result);
    expect(result.filter.sizes == std::array<int, CUDNN_DIM_MAX>{},
           "zero filter capacity publishes no elements");
    original(registry.get_filter(filter, -3), CUDNN_STATUS_BAD_PARAM);
    expect(fixture.last_requested == -3, "negative filter capacity reaches original SDK");
    result = registry.filter_size(filter);
    success(result);
    expect(result.value == (std::uint64_t{1} << 35) + 93, "filter byte count remains full width");
    auto operation = convolution_value();
    success(registry.set_convolution(convolution, operation, true));
    result = registry.get_convolution(convolution, 2, true);
    success(result);
    expect(result.convolution.padding == operation.padding &&
               result.convolution.strides == operation.strides &&
               result.convolution.dilation == operation.dilation &&
               result.convolution.mode == operation.mode &&
               result.convolution.type == operation.type,
           "two-dimensional convolution retains all scalar fields");
    operation.dimensions = 3;
    operation.padding[2] = 7;
    operation.strides[2] = 8;
    operation.dilation[2] = 9;
    success(registry.set_convolution(convolution, operation));
    result = registry.get_convolution(convolution, 6);
    success(result);
    expect(result.convolution.dimensions == 3 && result.convolution.padding == operation.padding,
           "Nd convolution retains every native spatial dimension");
    result = registry.get_convolution(convolution, 1);
    success(result);
    expect(result.convolution.dimensions == 3 && !result.convolution.padding[1],
           "convolution prefix query preserves actual rank");
    for (int capacity : {-1, 7, 12, std::numeric_limits<int>::max()}) {
        original(registry.get_convolution(convolution, capacity),
                 capacity < 0 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_NOT_SUPPORTED);
        expect(fixture.last_requested == capacity,
               "convolution capacity is never silently clipped");
    }
    for (auto attribute : {Attribute::kGroup, Attribute::kMath, Attribute::kReorder}) {
        const auto expected = attribute == Attribute::kGroup  ? 7U
                              : attribute == Attribute::kMath ? 3U
                                                              : 1U;
        success(registry.set_convolution_attribute(convolution, attribute, expected));
        result = registry.get_convolution_attribute(convolution, attribute);
        success(result);
        expect(result.value == expected, "native convolution attribute round trip");
    }
    original(registry.set_convolution_attribute(convolution, Attribute::kGroup, UINT32_MAX),
             CUDNN_STATUS_BAD_PARAM);
    expect(fixture.setter_calls == std::array<unsigned int, 4>{1, 1, 1, 1} &&
               fixture.getter_calls[0] == 1 && fixture.getter_calls[2] == 1,
           "four/two and Nd entry points remain distinct");
    const auto tensor = registry.create(Kind::kTensor).id;
    success(registry.set_tensor(tensor, CudaDnnTensorSetter::kFour,
                                {CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 4, {2, 3, 13, 17}, {}}));
    for (bool is_two : {false, true}) {
        result = registry.convolution_output(convolution, tensor, filter, 4, is_two);
        success(result);
        expect(result.tensor.dimensions == 4 && result.tensor.sizes[0] == 2 &&
                   result.tensor.sizes[1] == 7 && result.tensor.sizes[3] == 17,
               "output-shape query returns original same-registry descriptor result");
    }
    success(registry.shutdown());
    expect(fixture.destroys == 2 && sdk::state.destroys == 1,
           "all exact native kinds destroyed once");
    rejected(registry.get_filter(filter, 4), CUDA_ERROR_INVALID_HANDLE);
}
void requested_convolution_tails() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto id = registry.create(Kind::kConvolution).id;
    const auto verify = [&](const CudaDnnConvolutionState& expected) {
        for (const int requested : {0, 1, 2, 3, 6}) {
            const auto result = registry.get_convolution(id, requested);
            success(result);
            expect(result.convolution.dimensions == expected.dimensions &&
                       result.convolution.mode == expected.mode &&
                       result.convolution.type == expected.type &&
                       fixture.last_requested == requested,
                   "actual rank and enums stay independent of requested native copy extent");
            for (std::size_t i = 0; i != CUDNN_DIM_MAX; ++i) {
                const bool returned = i < static_cast<std::size_t>(requested);
                expect(
                    result.convolution.padding[i] == (returned ? expected.padding[i] : 0) &&
                        result.convolution.strides[i] == (returned ? expected.strides[i] : 0) &&
                        result.convolution.dilation[i] == (returned ? expected.dilation[i] : 0),
                    "all requested original slots survive, only beyond-request slots canonicalize");
            }
        }
    };
    verify({});  // Native creation has rank zero and all initialized array slots zero.
    auto expected = convolution_value();
    success(registry.set_convolution(id, expected));
    verify(expected);
    for (const bool two_dimensional : {false, true}) {
        auto higher = convolution_value();
        higher.dimensions = 3;
        higher.padding[2] = 7;
        higher.strides[2] = 8;
        higher.dilation[2] = 9;
        success(registry.set_convolution(id, higher));
        verify(higher);
        success(registry.set_convolution(id, convolution_value(), two_dimensional));
        expected = higher;
        expected.dimensions = 2;
        verify(expected);
        const auto result = registry.get_convolution(id, 2, true);
        success(result);
        expect(result.convolution.padding[2] == 0 && result.convolution.strides[2] == 0 &&
                   result.convolution.dilation[2] == 0,
               "2D getter still publishes exactly two entries, not retained native tail");
    }
    fixture.get_status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    const auto failure = registry.get_convolution(id, 6);
    original(failure, fixture.get_status);
    expect(failure.convolution.dimensions == 0 &&
               failure.convolution.padding == std::array<int, CUDNN_DIM_MAX>{} &&
               failure.convolution.strides == std::array<int, CUDNN_DIM_MAX>{} &&
               failure.convolution.dilation == std::array<int, CUDNN_DIM_MAX>{} &&
               !registry.snapshot().is_quarantined,
           "native failure after private writes publishes no retained descriptor fields");
    fixture.get_status = CUDNN_STATUS_SUCCESS;
    verify(expected);
    success(registry.shutdown());
    expect(fixture.creates == 1 && fixture.destroys == 1,
           "requested-tail owner retires exactly once");
}
void validation_and_failed_outputs() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto filter = registry.create(Kind::kFilter).id;
    const auto convolution = registry.create(Kind::kConvolution).id;
    auto value = filter_value();
    auto operation = convolution_value();
    success(registry.set_filter(filter, value));
    success(registry.set_convolution(convolution, operation));
    const auto sets = fixture.sets;
    for (unsigned int mode = 0; mode != 4; ++mode) {
        auto invalid = value;
        if (!mode) invalid.type = 32;
        if (mode == 1) invalid.format = 4;
        if (mode == 2) invalid.dimensions = -1;
        if (mode == 3) invalid.dimensions = 9;
        rejected(registry.set_filter(filter, invalid), CUDA_ERROR_INVALID_VALUE);
    }
    for (unsigned int mode = 0; mode != 4; ++mode) {
        auto invalid = operation;
        if (!mode) invalid.type = 32;
        if (mode == 1) invalid.mode = 2;
        if (mode == 2) invalid.dimensions = -1;
        if (mode == 3) invalid.dimensions = 9;
        rejected(registry.set_convolution(convolution, invalid), CUDA_ERROR_INVALID_VALUE);
    }
    rejected(registry.set_filter(convolution, value), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.set_convolution_attribute(convolution, Attribute::kMath, 4),
             CUDA_ERROR_INVALID_VALUE);
    rejected(registry.set_convolution_attribute(convolution, Attribute::kReorder, 2),
             CUDA_ERROR_INVALID_VALUE);
    rejected(registry.get_convolution_attribute(convolution, static_cast<Attribute>(99)),
             CUDA_ERROR_INVALID_VALUE);
    expect(fixture.sets == sets, "unrepresentable fields rejected before SDK entry");
    value.type = 31;
    original(registry.set_filter(filter, value), CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    operation.dimensions = 8;
    original(registry.set_convolution(convolution, operation), CUDNN_STATUS_NOT_SUPPORTED);
    fixture.get_status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    auto result = registry.get_filter(filter, 4);
    original(result, fixture.get_status);
    expect(result.filter.dimensions == 0 && result.filter.sizes[0] == 0,
           "failed filter query publishes no touched SDK output");
    result = registry.get_convolution(convolution, 2);
    original(result, fixture.get_status);
    expect(result.convolution.dimensions == 0 && result.convolution.strides[0] == 0,
           "failed convolution query publishes no touched SDK output");
    result = registry.get_convolution_attribute(convolution, Attribute::kGroup);
    original(result, fixture.get_status);
    expect(result.value == 0, "failed native attribute output remains unpublished");
    fixture.size_status = fixture.get_status;
    result = registry.filter_size(filter);
    original(result, fixture.size_status);
    expect(result.value == 0, "failed native size is unpublished");
    const auto tensor = registry.create(Kind::kTensor).id;
    fixture.shape_status = fixture.get_status;
    result = registry.convolution_output(convolution, tensor, filter, 4);
    original(result, fixture.shape_status);
    expect(result.tensor.dimensions == 0 && result.tensor.sizes[0] == 0,
           "failed shape output is unpublished");
    const auto calls = fixture.shapes;
    rejected(registry.convolution_output(filter, tensor, convolution, 4),
             CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.convolution_output(convolution, tensor, 999, 4), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.convolution_output(convolution, tensor, filter, 9), CUDA_ERROR_INVALID_VALUE);
    expect(fixture.shapes == calls, "unknown kinds and unbounded output never enter SDK");
    success(registry.shutdown());
}
void optional_callbacks_and_leases() {
    reset();
    {
        CudaDnnRegistry old(sdk::dispatch());
        rejected(old.create(Kind::kFilter), CUDA_ERROR_NOT_SUPPORTED);
        rejected(old.create(Kind::kConvolution), CUDA_ERROR_NOT_SUPPORTED);
        success(old.shutdown());
    }
    auto sdk_dispatch = dispatch();
    sdk_dispatch.filter_destroy = nullptr;
    {
        CudaDnnRegistry incomplete(sdk_dispatch);
        rejected(incomplete.create(Kind::kFilter), CUDA_ERROR_NOT_SUPPORTED);
        expect(fixture.creates == 0, "missing cleanup callback prevents native creation");
    }
    sdk_dispatch = dispatch();
    sdk_dispatch.filter_get_nd = nullptr;
    sdk_dispatch.convolution_group_set = nullptr;
    CudaDnnRegistry registry(sdk_dispatch, 2);
    const auto filter = registry.create(Kind::kFilter).id;
    const auto convolution = registry.create(Kind::kConvolution).id;
    rejected(registry.create(Kind::kFilter), CUDA_ERROR_OUT_OF_MEMORY);
    expect(fixture.creates == 2, "capacity bound is checked before new native ownership");
    rejected(registry.get_filter(filter, 4), CUDA_ERROR_NOT_SUPPORTED);
    rejected(registry.set_convolution_attribute(convolution, Attribute::kGroup, 2),
             CUDA_ERROR_NOT_SUPPORTED);
    auto filter_lease = registry.borrow(filter, Kind::kFilter);
    auto convolution_lease = registry.borrow(convolution, Kind::kConvolution);
    expect(filter_lease && convolution_lease && !registry.borrow(filter, Kind::kConvolution),
           "new kinds produce only matching immutable views");
    expect(
        registry.set_filter(filter, filter_value()).disposition == CudaDnnDisposition::kDeferred &&
            registry.set_convolution(convolution, convolution_value()).disposition ==
                CudaDnnDisposition::kDeferred &&
            registry.set_convolution_attribute(convolution, Attribute::kMath, 1).disposition ==
                CudaDnnDisposition::kDeferred &&
            registry.destroy(filter).disposition == CudaDnnDisposition::kDeferred,
        "immutable leases defer all mutation and native destruction");
    expect(fixture.sets == 0 && fixture.destroys == 0, "deferral does not enter native SDK");
    const auto foreign =
        std::async(std::launch::async, [&] { return registry.filter_size(filter); }).get();
    rejected(foreign, CUDA_ERROR_UNKNOWN);
    expect(fixture.sizes == 0, "foreign thread never invokes native SDK");
    const auto child = ::fork();
    expect(child >= 0, "fork ownership test");
    if (!child)
        ::_exit(registry.get_convolution(convolution, 2).status == CUDA_ERROR_UNKNOWN ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork cannot use inherited descriptor ownership");
    registry.close();
    rejected(registry.set_filter(filter, filter_value()), CUDA_ERROR_DEINITIALIZED);
    expect(!registry.borrow(filter, Kind::kFilter), "close seals new borrows");
    expect(registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
           "close waits both immutable leases");
    filter_lease.reset();
    convolution_lease.reset();
    success(registry.shutdown());
    success(registry.shutdown());
    expect(fixture.destroys == 2, "closed registry cleanup is idempotent");
}
void isolated_failures() {
    for (unsigned int mode = 0; mode != 17; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "isolated uncertain ownership fork");
        if (!child) {
            try {
                reset();
                CudaDnnRegistry registry(dispatch());
                const auto first = registry.create(mode == 2 ? Kind::kTensor : Kind::kFilter);
                success(first);
                const auto convolution = registry.create(Kind::kConvolution);
                success(convolution);
                CudaDnnResult result;
                if (mode < 4) {
                    fixture.null_success = mode == 0;
                    fixture.partial = mode == 1 || mode == 3;
                    if (fixture.partial) fixture.create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    if (mode >= 2) {
                        auto view =
                            registry.borrow(first.id, mode == 2 ? Kind::kTensor : Kind::kFilter);
                        fixture.duplicate = view->native;
                    }
                    result = registry.create(mode == 2 ? Kind::kConvolution : Kind::kFilter);
                    expect(result.disposition == CudaDnnDisposition::kRejected && !result.id &&
                               result.original_status == fixture.create_status,
                           "partial/null/duplicate native ownership cannot publish a new ID");
                } else if (mode == 4) {
                    fixture.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
                    original(registry.destroy(convolution.id), fixture.destroy_status);
                } else if (mode == 5) {
                    fixture.set_status = CUDNN_STATUS_EXECUTION_FAILED;
                    original(registry.set_filter(first.id, filter_value()), fixture.set_status);
                } else if (mode == 6) {
                    fixture.fail_destroy_call = 2;
                    original(registry.shutdown(), CUDNN_STATUS_INTERNAL_ERROR);
                    expect(registry.snapshot().objects == 1 && fixture.destroys == 2,
                           "partial shutdown removes already destroyed new-kind metadata");
                } else {
                    success(registry.set_filter(first.id, filter_value()));
                    success(registry.set_convolution(convolution.id, convolution_value()));
                    if (mode == 7) fixture.reported_type = UINT32_MAX;
                    if (mode == 8) fixture.reported_format = 4;
                    if (mode == 9) fixture.reported_rank = 9;
                    if (mode == 10) fixture.reported_mode = 2;
                    if (mode == 11) fixture.reported_rank = 7;
                    if (mode == 12) fixture.reported_attribute = 4;
                    if (mode == 13 || mode == 14 || mode == 16)
                        fixture.invalid_capacity_success = true;
                    if (mode == 15) fixture.reported_rank = -1;
                    const int requested = mode == 13   ? 12
                                          : mode == 14 ? -1
                                          : mode == 16 ? INT32_MAX
                                                       : 2;
                    result = mode < 10    ? registry.get_filter(first.id, 4)
                             : mode == 12 ? registry.get_convolution_attribute(convolution.id,
                                                                               Attribute::kMath)
                                          : registry.get_convolution(convolution.id, requested);
                    expect(result.disposition == CudaDnnDisposition::kRejected &&
                               result.status == CUDA_ERROR_UNKNOWN &&
                               result.original_status == CUDNN_STATUS_SUCCESS &&
                               !result.filter.dimensions && !result.convolution.dimensions &&
                               !result.value,
                           "malformed successful native output quarantines without publishing");
                }
                expect(registry.snapshot().is_quarantined, "uncertain state is explicit");
                const auto destroys = fixture.destroys;
                rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
                rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
                expect(fixture.destroys == destroys, "uncertain cleanup is not retried");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "convolution registry isolated mode " << mode << ": " << error.what()
                          << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated new-kind ownership control passes");
    }
}
}  // namespace
int main() {
    try {
        metadata_round_trip();
        requested_convolution_tails();
        validation_and_failed_outputs();
        optional_callbacks_and_leases();
        isolated_failures();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
