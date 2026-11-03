#include "glimmer/control/execution_dnn_protocol.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionDnnLibraryInfoKind::kVersion) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLibraryInfoKind::kCudartVersion) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLibraryInfoKind::kMaxDeviceVersion) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnLibraryInfoKind::kProperty) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kTensor) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kActivation) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kFilter) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kConvolution) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kPooling) == 5);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kLrn) == 6);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kReduction) == 7);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kDropout) == 8);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kOpTensor) == 9);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kTensorTransform) == 10);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kRnnData) == 11);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kSeqData) == 12);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kRnn) == 13);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kAttn) == 14);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kBackend) == 15);
static_assert(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kAlgorithm) == 16);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnQueryKind::kWeightSpace) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnQueryKind::kTempSpace) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnQueryKind::kAttentionBuffers) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kFour) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kFourStrided) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kNd) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kNdPacked) == 4);
static_assert(static_cast<std::uint32_t>(ExecutionDnnComputeKind::kActivationForward) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnComputeKind::kAddTensor) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnAttnComputeKind::kForward) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnAttnComputeKind::kBackwardData) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnAttnComputeKind::kBackwardWeights) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnComputeKind::kForward) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnComputeKind::kBackwardData) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnRnnComputeKind::kBackwardWeights) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnDropoutKind::kSet) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnDropoutKind::kRestore) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionAttribute::kGroup) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionAttribute::kMath) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionAttribute::kReorder) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionQueryKind::kWorkspace) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionQueryKind::kHeuristic) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnConvolutionQueryKind::kMaxCount) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount) ==
              1);
static_assert(
    static_cast<std::uint32_t>(ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount) == 2);
static_assert(
    static_cast<std::uint32_t>(ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnReductionQueryKind::kIndices) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnReductionQueryKind::kWorkspace) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBatchNormKind::kForward) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBatchNormKind::kBackward) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnBatchNormKind::kReserve) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kPoolingForward) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kSoftmaxForward) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kLrnForward) == 3);
static_assert(static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kTransformTensor) == 4);

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

bool same(const ExecutionDnnTensor& left, const ExecutionDnnTensor& right) {
    return left.type == right.type && left.format == right.format &&
           left.dimensions == right.dimensions && left.sizes == right.sizes &&
           left.strides == right.strides;
}
bool same(const ExecutionDnnActivation& left, const ExecutionDnnActivation& right) {
    return left.mode == right.mode && left.nan == right.nan &&
           left.coefficient_bits == right.coefficient_bits;
}
bool same(const ExecutionDnnCompute& left, const ExecutionDnnCompute& right) {
    if (left.kind != right.kind || left.handle != right.handle || left.metadata != right.metadata ||
        left.activation != right.activation || left.tensors != right.tensors ||
        left.scalar_bytes != right.scalar_bytes || left.alpha != right.alpha ||
        left.beta != right.beta)
        return false;
    for (std::size_t index = 0; index < left.buffers.size(); ++index)
        if (left.buffers[index].allocation != right.buffers[index].allocation ||
            left.buffers[index].offset != right.buffers[index].offset)
            return false;
    return true;
}
void set_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 4, "bounded test mutation");
    for (std::size_t index = 0; index < 4; ++index)
        bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
}

ExecutionDnnTensor tensor_fixture() {
    ExecutionDnnTensor result;
    result.type = std::numeric_limits<std::uint32_t>::max();
    result.format = 0x87654321;
    result.dimensions = 8;
    result.sizes = {1, -1, 0, 4, INT32_MIN, INT32_MAX, 7, 8};
    result.strides = {INT32_MAX, INT32_MIN, 0, -1, 80, 40, 20, 1};
    return result;
}
ExecutionDnnCompute compute_fixture(ExecutionDnnComputeKind kind, std::uint32_t bytes) {
    ExecutionDnnCompute result;
    result.kind = kind;
    result.handle = 0x0102030405060708;
    result.metadata = 0xf1e2d3c4b5a69788;
    result.activation = kind == ExecutionDnnComputeKind::kActivationForward ? UINT64_MAX : 0;
    result.tensors = {0x1122334455667788, 0x8877665544332211};
    result.buffers = {{{0x8000000000000001, UINT64_MAX}, {0x7200000000000034, 0xabcdef0123456789}}};
    result.scalar_bytes = bytes;
    for (std::size_t index = 0; index < result.alpha.size(); ++index) {
        result.alpha[index] = static_cast<std::byte>(index + 0x20);
        result.beta[index] = static_cast<std::byte>(index + 0xa0);
    }
    return result;
}

void tensor_codecs() {
    const auto original = tensor_fixture();
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_tensor(writer, original), "encode fixed tensor fields");
    const auto encoded = writer.take();
    expect(encoded.size() == 76 && encoded[4] == std::byte{0x21} && encoded[7] == std::byte{0x87} &&
               encoded[8] == std::byte{8} && encoded[16] == std::byte{0xff} &&
               encoded[31] == std::byte{0x80} && encoded[44] == std::byte{0xff} &&
               encoded[47] == std::byte{0x7f},
           "tensor type/format/rank, all sizes then strides have explicit endian layout");
    ExecutionDnnTensor decoded;
    ExecutionPayloadReader reader(encoded);
    expect(read_execution_dnn_tensor(reader, decoded) && reader.empty() && same(decoded, original),
           "raw vendor enums and signed dimension/stride bits are not normalized");
    ExecutionDnnTensor sentinel;
    sentinel.dimensions = 3;
    sentinel.type = 2;
    sentinel.sizes.fill(19);
    const auto reject = [&](std::span<const std::byte> bytes) {
        auto output = sentinel;
        ExecutionPayloadReader input(bytes);
        expect(!read_execution_dnn_tensor(input, output) && same(output, sentinel),
               "failed tensor decoding cannot publish partial output");
    };
    for (std::size_t length = 0; length < encoded.size(); ++length)
        reject(std::span{encoded}.first(length));
    for (const auto rank : {std::int32_t{-1}, std::int32_t{9}, INT32_MIN, INT32_MAX}) {
        auto invalid = original;
        invalid.dimensions = rank;
        ExecutionPayloadWriter output;
        expect(!write_execution_dnn_tensor(output, invalid) && output.take().empty(),
               "unbounded tensor rank rejects before encoding fields");
        auto bytes = encoded;
        set_u32(bytes, 8, std::bit_cast<std::uint32_t>(rank));
        reject(bytes);
    }
    for (const auto rank : {0, 8}) {
        auto valid = original;
        valid.dimensions = rank;
        ExecutionPayloadWriter output;
        expect(write_execution_dnn_tensor(output, valid), "bounded rank boundary can reach SDK");
        const auto bytes = output.take();
        ExecutionPayloadReader input(bytes);
        expect(read_execution_dnn_tensor(input, decoded) && same(decoded, valid),
               "bounded rank retains all fixed signed fields");
    }
    for (std::size_t limit = 0; limit < encoded.size(); ++limit) {
        ExecutionPayloadWriter output(limit);
        expect(!write_execution_dnn_tensor(output, original),
               "tensor encoding observes negotiated payload bound");
    }
}

void activation_codecs() {
    const ExecutionDnnActivation original{UINT32_MAX, 0x12345678, 0x7ff8123456789abc};
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_activation(writer, original), "encode exact activation bits");
    const auto encoded = writer.take();
    expect(encoded.size() == 16 && encoded[4] == std::byte{0x78} && encoded[7] == std::byte{0x12} &&
               encoded[8] == std::byte{0xbc} && encoded[15] == std::byte{0x7f},
           "activation mode/nan/double-bit fields use explicit little endian");
    ExecutionDnnActivation decoded;
    ExecutionPayloadReader reader(encoded);
    expect(
        read_execution_dnn_activation(reader, decoded) && reader.empty() && same(decoded, original),
        "vendor validation receives unmodified enums and NaN payload bits");
    const ExecutionDnnActivation sentinel{4, 1, 0x8000000000000000};
    for (std::size_t length = 0; length < encoded.size(); ++length) {
        auto output = sentinel;
        ExecutionPayloadReader input{std::span{encoded}.first(length)};
        expect(!read_execution_dnn_activation(input, output) && same(output, sentinel),
               "every truncated activation leaves output untouched");
        ExecutionPayloadWriter bounded(length);
        expect(!write_execution_dnn_activation(bounded, original),
               "activation writes cannot exceed negotiated size");
    }
}

void compute_codecs() {
    constexpr std::array kinds{ExecutionDnnComputeKind::kActivationForward,
                               ExecutionDnnComputeKind::kAddTensor};
    for (const auto kind : kinds) {
        for (const std::uint32_t width : {4U, 8U}) {
            const auto original = compute_fixture(kind, width);
            ExecutionPayloadWriter writer;
            expect(write_execution_dnn_compute(writer, original), "encode owned compute fields");
            const auto encoded = writer.take();
            expect(encoded.size() == 88 + 2 * width && encoded[0] == static_cast<std::byte>(kind) &&
                       encoded[4] == std::byte{8} && encoded[11] == std::byte{1} &&
                       encoded[12] == std::byte{0x88} && encoded[19] == std::byte{0xf1} &&
                       encoded[28] == std::byte{0x88} && encoded[35] == std::byte{0x11} &&
                       encoded[44] == std::byte{1} && encoded[51] == std::byte{0x80} &&
                       encoded[52] == std::byte{0xff} &&
                       encoded[76] == static_cast<std::byte>(width) &&
                       encoded[80] == static_cast<std::byte>(width) &&
                       encoded[84] == original.alpha[0] &&
                       encoded[84 + width] == static_cast<std::byte>(width) &&
                       encoded[88 + width] == original.beta[0],
                   "compute IDs, intervals, scalar width and two blobs match frozen wire order");
            auto expected = original;
            std::fill(expected.alpha.begin() + width, expected.alpha.end(), std::byte{});
            std::fill(expected.beta.begin() + width, expected.beta.end(), std::byte{});
            ExecutionDnnCompute decoded;
            ExecutionPayloadReader reader(encoded);
            expect(read_execution_dnn_compute(reader, decoded) && reader.empty() &&
                       same(decoded, expected),
                   "compute roundtrip copies only actual coefficient bytes and zeroes unused tail");
            const auto sentinel = compute_fixture(kinds[0], 8);
            const auto reject = [&](std::span<const std::byte> bytes) {
                auto output = sentinel;
                ExecutionPayloadReader input(bytes);
                expect(!read_execution_dnn_compute(input, output) && same(output, sentinel),
                       "malformed compute cannot publish partial ownership or coefficients");
            };
            for (std::size_t length = 0; length < encoded.size(); ++length) {
                reject(std::span{encoded}.first(length));
                ExecutionPayloadWriter bounded(length);
                expect(!write_execution_dnn_compute(bounded, original),
                       "compute encoding observes negotiated payload bound");
            }
            for (const auto invalid_kind : {0U, 3U, UINT32_MAX}) {
                auto invalid = original;
                invalid.kind = static_cast<ExecutionDnnComputeKind>(invalid_kind);
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_compute(output, invalid) && output.take().empty(),
                       "unknown compute kind rejects before payload creation");
                auto bytes = encoded;
                set_u32(bytes, 0, invalid_kind);
                reject(bytes);
            }
            for (const auto invalid_width : {0U, 1U, 3U, 5U, 7U, 9U, UINT32_MAX}) {
                auto invalid = original;
                invalid.scalar_bytes = invalid_width;
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_compute(output, invalid) && output.take().empty(),
                       "unknown scalar width rejects before array slicing");
                auto bytes = encoded;
                set_u32(bytes, 76, invalid_width);
                reject(bytes);
            }
            for (const auto offset : {std::size_t{80}, std::size_t{84} + width})
                for (const auto length : {0U, width - 1, width + 1, UINT32_MAX}) {
                    auto bytes = encoded;
                    set_u32(bytes, offset, length);
                    reject(bytes);
                }
            if (kind == ExecutionDnnComputeKind::kAddTensor) {
                auto invalid = original;
                invalid.activation = 1;
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_compute(output, invalid) && output.take().empty(),
                       "add-tensor must not smuggle an unused activation descriptor");
                auto bytes = encoded;
                bytes[20] = std::byte{1};
                reject(bytes);
            }
        }
    }
    // Decoder success is shape evidence only; IDs and allocation intervals still
    // require authenticated owner-side validation, including zero/stale values.
    ExecutionDnnCompute unowned;
    unowned.kind = ExecutionDnnComputeKind::kActivationForward;
    unowned.scalar_bytes = 4;
    ExecutionPayloadWriter output;
    expect(write_execution_dnn_compute(output, unowned), "codec does not mint ownership from IDs");
    const auto bytes = output.take();
    ExecutionPayloadReader input(bytes);
    ExecutionDnnCompute decoded;
    expect(read_execution_dnn_compute(input, decoded) && same(decoded, unowned),
           "structural decoder preserves zero IDs for independent ownership rejection");
}

void composition_and_constants() {
    expect(kExecutionDnnMetadataRole == 0x314d4e44 &&
               kExecutionDnnMetadataRole != kExecutionLtMetadataRole &&
               kExecutionDnnMetadataRole != kExecutionHostOwnerRole &&
               kExecutionDnnMetadataRole != kExecutionStagingOwnerRole &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kTensor) == 1 &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kActivation) == 2 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kFour) == 1 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kFourStrided) == 2 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kNd) == 3 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorSetter::kNdPacked) == 4,
           "metadata role and operation enums have stable CUDA-free identities");
    const auto tensor = tensor_fixture();
    const ExecutionDnnActivation activation{3, 1, UINT64_MAX};
    auto compute = compute_fixture(ExecutionDnnComputeKind::kAddTensor, 8);
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(kExecutionDnnMetadataRole) &&
               write_execution_dnn_tensor(writer, tensor) &&
               write_execution_dnn_activation(writer, activation) &&
               write_execution_dnn_compute(writer, compute) && writer.write_u32(0x12345678),
           "field codecs compose without owning the entire payload");
    const auto bytes = writer.take();
    ExecutionPayloadReader reader(bytes);
    std::uint32_t value{};
    ExecutionDnnTensor decoded_tensor;
    ExecutionDnnActivation decoded_activation;
    ExecutionDnnCompute decoded_compute;
    expect(reader.read_u32(value) && value == kExecutionDnnMetadataRole &&
               read_execution_dnn_tensor(reader, decoded_tensor) && !reader.empty() &&
               read_execution_dnn_activation(reader, decoded_activation) && !reader.empty() &&
               read_execution_dnn_compute(reader, decoded_compute) && !reader.empty() &&
               same(decoded_tensor, tensor) && same(decoded_activation, activation) &&
               same(decoded_compute, compute) && reader.read_u32(value) && value == 0x12345678 &&
               reader.empty(),
           "successful codecs consume their own fields only; caller checks trailing data");
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnCompute),
           "original DNN computation can never claim private Driver replacement status");
}

template <typename Value, typename Write, typename Read>
std::vector<std::byte> checked_codec(const Value& original, std::size_t size, Write write,
                                     Read read) {
    const auto encoded_value = [&](const Value& value) {
        ExecutionPayloadWriter output;
        expect(write(output, value), "encode bounded convolution value");
        return output.take();
    };
    auto bytes = encoded_value(original);
    expect(bytes.size() == size, "convolution wire size excludes native struct padding");
    Value decoded{};
    ExecutionPayloadReader reader(bytes);
    expect(read(reader, decoded) && reader.empty() && encoded_value(decoded) == bytes,
           "convolution value roundtrips all explicit raw fields");
    for (std::size_t length = 0; length < bytes.size(); ++length) {
        auto sentinel = original;
        ExecutionPayloadReader input{std::span{bytes}.first(length)};
        expect(!read(input, sentinel) && encoded_value(sentinel) == bytes,
               "every truncated convolution value preserves the destination");
        ExecutionPayloadWriter bounded(length);
        expect(!write(bounded, original), "convolution codec honors bounded payload capacity");
    }
    auto trailing = bytes;
    trailing.push_back(std::byte{0xab});
    ExecutionPayloadReader composed(trailing);
    expect(read(composed, decoded) && !composed.empty(),
           "convolution codecs leave trailing framing validation to caller");
    return bytes;
}
template <typename Value, typename Write, typename Read>
void malformed_convolution(const Value& original, const std::vector<std::byte>& bytes, Write write,
                           Read read) {
    auto output = original;
    ExecutionPayloadReader input(bytes);
    expect(!read(input, output), "malformed convolution value is rejected");
    ExecutionPayloadWriter before, after;
    expect(write(before, original) && write(after, output) && before.take() == after.take(),
           "malformed convolution value cannot publish partial output");
}
void convolution_codecs() {
    ExecutionDnnFilter filter{UINT32_MAX, 0x12345678, 8, {1, -1, 0, INT32_MAX, INT32_MIN, 6, 7, 8}};
    const auto filter_bytes =
        checked_codec(filter, 44, write_execution_dnn_filter, read_execution_dnn_filter);
    expect(filter_bytes[4] == std::byte{0x78} && filter_bytes[7] == std::byte{0x12} &&
               filter_bytes[8] == std::byte{8} && filter_bytes[31] == std::byte{0x80},
           "filter encoding preserves raw enums and signed dimension bits");
    ExecutionDnnConvolutionState convolution{0xfedcba98,
                                             UINT32_MAX,
                                             8,
                                             {0, -1, INT32_MIN, INT32_MAX, 4, 5, 6, 7},
                                             {1, 2, 3, 4, 5, 6, 7, 8},
                                             {8, 7, 6, 5, 4, 3, 2, 1}};
    const auto convolution_bytes = checked_codec(convolution, 108, write_execution_dnn_convolution,
                                                 read_execution_dnn_convolution);
    expect(convolution_bytes[0] == std::byte{0x98} && convolution_bytes[3] == std::byte{0xfe} &&
               convolution_bytes[44] == std::byte{1} && convolution_bytes[76] == std::byte{8},
           "convolution ordering is mode/type/rank then padding/strides/dilation");
    for (const auto rank : {UINT32_MAX, 9U, 0x80000000U, 0x7fffffffU}) {
        auto bytes = filter_bytes;
        set_u32(bytes, 8, rank);
        malformed_convolution(filter, bytes, write_execution_dnn_filter, read_execution_dnn_filter);
        bytes = convolution_bytes;
        set_u32(bytes, 8, rank);
        malformed_convolution(convolution, bytes, write_execution_dnn_convolution,
                              read_execution_dnn_convolution);
        filter.dimensions = std::bit_cast<std::int32_t>(rank);
        convolution.dimensions = filter.dimensions;
        ExecutionPayloadWriter a, b;
        expect(!write_execution_dnn_filter(a, filter) && a.take().empty() &&
                   !write_execution_dnn_convolution(b, convolution) && b.take().empty(),
               "out-of-bounds convolution ranks fail before partial encoding");
        filter.dimensions = convolution.dimensions = 8;
    }
    filter.dimensions = convolution.dimensions = 0;
    (void)checked_codec(filter, 44, write_execution_dnn_filter, read_execution_dnn_filter);
    (void)checked_codec(convolution, 108, write_execution_dnn_convolution,
                        read_execution_dnn_convolution);

    for (const auto kind : {ExecutionDnnConvolutionQueryKind::kWorkspace,
                            ExecutionDnnConvolutionQueryKind::kHeuristic,
                            ExecutionDnnConvolutionQueryKind::kMaxCount}) {
        ExecutionDnnConvolutionQuery query{kind,         0x0102030405060708, UINT64_MAX,
                                           {1, 2, 3, 4}, UINT32_MAX,         0x80000000};
        const auto bytes = checked_codec(query, 60, write_execution_dnn_convolution_query,
                                         read_execution_dnn_convolution_query);
        expect(bytes[0] == static_cast<std::byte>(kind) && bytes[4] == std::byte{8} &&
                   bytes[11] == std::byte{1} && bytes[59] == std::byte{0x80},
               "forward query preserves signed requested count and exact identities");
        for (const auto bad_kind : {0U, 4U, UINT32_MAX}) {
            auto invalid = bytes;
            set_u32(invalid, 0, bad_kind);
            malformed_convolution(query, invalid, write_execution_dnn_convolution_query,
                                  read_execution_dnn_convolution_query);
            auto malformed = query;
            malformed.kind = static_cast<ExecutionDnnConvolutionQueryKind>(bad_kind);
            ExecutionPayloadWriter output;
            expect(
                !write_execution_dnn_convolution_query(output, malformed) && output.take().empty(),
                "unknown forward query kind does not enter wire stream");
        }
    }
    for (const std::uint32_t width : {4U, 8U}) {
        ExecutionDnnConvolutionForward forward;
        forward.handle = 0x8877665544332211;
        forward.metadata = UINT64_MAX;
        forward.descriptors = {1, 2, 3, 4};
        forward.buffers = {{{9, 10}, {11, 12}, {13, 14}, {15, UINT64_MAX}}};
        forward.algorithm = UINT32_MAX;
        forward.workspace_bytes = 0;
        forward.scalar_bytes = width;
        forward.alpha.fill(std::byte{0xaa});
        forward.beta.fill(std::byte{0xbb});
        const auto bytes =
            checked_codec(forward, 136 + 2 * width, write_execution_dnn_convolution_forward,
                          read_execution_dnn_convolution_forward);
        expect(bytes[0] == std::byte{0x11} && bytes[7] == std::byte{0x88} &&
                   bytes[96] == std::byte{15} && bytes[111] == std::byte{0xff} &&
                   bytes[112] == std::byte{0xff} && bytes[116] == std::byte{0} &&
                   bytes[124] == static_cast<std::byte>(width) && bytes[132] == std::byte{0xaa} &&
                   bytes[136 + width] == std::byte{0xbb},
               "zero workspace size preserves nonzero owned pointer and exact scalar blobs");
        for (const auto invalid_width : {0U, 1U, 16U, UINT32_MAX}) {
            auto invalid = bytes;
            set_u32(invalid, 124, invalid_width);
            malformed_convolution(forward, invalid, write_execution_dnn_convolution_forward,
                                  read_execution_dnn_convolution_forward);
            auto value = forward;
            value.scalar_bytes = invalid_width;
            ExecutionPayloadWriter output;
            expect(!write_execution_dnn_convolution_forward(output, value) && output.take().empty(),
                   "invalid scalar width cannot produce a partial forward message");
        }
        for (const auto length_offset : {std::size_t{128}, std::size_t{132 + width}}) {
            for (const auto invalid_length : {0U, width - 1, width + 1, UINT32_MAX}) {
                auto invalid = bytes;
                set_u32(invalid, length_offset, invalid_length);
                malformed_convolution(forward, invalid, write_execution_dnn_convolution_forward,
                                      read_execution_dnn_convolution_forward);
            }
        }
    }
    for (const std::uint32_t count : {0U, 1U, 9U, 64U}) {
        ExecutionDnnConvolutionPerformances performance;
        performance.count = count;
        for (std::size_t i = 0; i < count; ++i)
            performance.values[i] = {UINT32_MAX,
                                     0x80000000,
                                     0x7fc12345,
                                     UINT64_MAX,
                                     0x12345678,
                                     0x87654321,
                                     {INT32_MIN, -1, INT32_MAX}};
        const auto bytes =
            checked_codec(performance, 4 + count * 40, write_execution_dnn_convolution_performances,
                          read_execution_dnn_convolution_performances);
        if (count)
            expect(bytes[12] == std::byte{0x45} && bytes[15] == std::byte{0x7f} &&
                       bytes[16] == std::byte{0xff} && bytes[35] == std::byte{0x80},
                   "performance NaN time/memory/reserved bits have explicit stable layout");
        for (const auto invalid_count : {65U, UINT32_MAX}) {
            auto invalid = bytes;
            set_u32(invalid, 0, invalid_count);
            malformed_convolution(performance, invalid,
                                  write_execution_dnn_convolution_performances,
                                  read_execution_dnn_convolution_performances);
            auto value = performance;
            value.count = invalid_count;
            ExecutionPayloadWriter output;
            expect(!write_execution_dnn_convolution_performances(output, value) &&
                       output.take().empty(),
                   "performance batch stays bounded independently of native requested count");
        }
    }
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kFilter) == 3 &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kConvolution) == 4 &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnConvolutionForward),
           "convolution is an append-only original SDK route, never private replacement");
    for (const auto operation :
         {ExecutionOpcode::kSetDnnFilter, ExecutionOpcode::kGetDnnFilter,
          ExecutionOpcode::kDnnFilterSize, ExecutionOpcode::kSetDnnConvolution,
          ExecutionOpcode::kGetDnnConvolution, ExecutionOpcode::kSetDnnConvolutionAttribute,
          ExecutionOpcode::kGetDnnConvolutionAttribute, ExecutionOpcode::kDnnConvolutionOutput,
          ExecutionOpcode::kDnnConvolutionQuery, ExecutionOpcode::kDnnConvolutionForward}) {
        ExecutionWireMessage message;
        message.header.operation = operation;
        const auto bytes = encode_execution_message(message);
        expect(bytes.has_value(), "new convolution opcode encodes");
        const auto decoded =
            decode_execution_header(std::span{*bytes}.first(kExecutionHeaderBytes));
        expect(decoded && decoded->operation == operation,
               "every appended convolution opcode is accepted by header decoder");
    }
}

void convolution_find_codec() {
    for (const auto requested : {0U, 1U, 64U, 0x7fffffffU, 0x80000000U, UINT32_MAX}) {
        const ExecutionDnnConvolutionFind original{
            0x0102030405060708,
            0xfedcba9876543210,
            {1, 0x8877665544332211, 0x8000000000000000, UINT64_MAX},
            requested};
        const auto bytes = checked_codec(original, 52, write_execution_dnn_convolution_find,
                                         read_execution_dnn_convolution_find);
        expect(bytes[0] == std::byte{8} && bytes[7] == std::byte{1} &&
                   bytes[8] == std::byte{0x10} && bytes[15] == std::byte{0xfe} &&
                   bytes[16] == std::byte{1} && bytes[24] == std::byte{0x11} &&
                   bytes[31] == std::byte{0x88} && bytes[39] == std::byte{0x80} &&
                   bytes[40] == std::byte{0xff} && bytes[47] == std::byte{0xff},
               "Find layout is handle/metadata/x/filter/convolution/y/requested without padding");
        ExecutionPayloadReader tail{std::span{bytes}.subspan(48)};
        std::uint32_t raw{};
        expect(tail.read_u32(raw) && tail.empty() && raw == requested,
               "Find preserves raw signed zero/negative/oversized counts for original SDK status");
    }
    // A structural codec cannot confer ownership. Zero identities must reach
    // independent authenticated service checks, without inventing resources.
    const ExecutionDnnConvolutionFind unowned{};
    (void)checked_codec(unowned, 52, write_execution_dnn_convolution_find,
                        read_execution_dnn_convolution_find);
    expect(!execution_requires_descriptor(ExecutionOpcode::kDnnConvolutionFind) &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnConvolutionFind),
           "Find carries owned IDs, not file descriptors or compute-replacement provenance");
}

void pooling_codecs() {
    ExecutionDnnPooling pooling{UINT32_MAX,
                                0x80000000,
                                8,
                                {1, -1, 0, INT32_MIN, INT32_MAX, 6, 7, 8},
                                {-3, -2, -1, 0, 1, 2, INT32_MIN, INT32_MAX},
                                {8, 7, 6, 5, 4, 3, 2, 1}};
    const auto bytes =
        checked_codec(pooling, 108, write_execution_dnn_pooling, read_execution_dnn_pooling);
    expect(bytes[7] == std::byte{0x80} && bytes[8] == std::byte{8} &&
               bytes[44] == std::byte{0xfd} && bytes[47] == std::byte{0xff} &&
               bytes[76] == std::byte{8} && bytes[104] == std::byte{1},
           "pooling preserves raw enums, negative padding and window/padding/stride field order");
    for (const auto rank : {-1, 9, INT32_MIN, INT32_MAX}) {
        auto malformed = bytes;
        set_u32(malformed, 8, std::bit_cast<std::uint32_t>(rank));
        malformed_convolution(pooling, malformed, write_execution_dnn_pooling,
                              read_execution_dnn_pooling);
        auto invalid = pooling;
        invalid.dimensions = rank;
        ExecutionPayloadWriter output;
        expect(!write_execution_dnn_pooling(output, invalid) && output.take().empty(),
               "pooling unbounded rank rejects before output");
    }
    for (const auto rank : {0, 1, 6, 7, 8}) {
        pooling.dimensions = rank;
        (void)checked_codec(pooling, 108, write_execution_dnn_pooling, read_execution_dnn_pooling);
    }
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kPooling) == 5,
           "pooling kind appends without changing previous descriptor kinds");
}

void tensor_forward_codecs() {
    for (const auto kind : {ExecutionDnnTensorForwardKind::kPoolingForward,
                            ExecutionDnnTensorForwardKind::kSoftmaxForward,
                            ExecutionDnnTensorForwardKind::kLrnForward,
                            ExecutionDnnTensorForwardKind::kTransformTensor}) {
        for (const auto width : {4U, 8U}) {
            ExecutionDnnTensorForward forward;
            forward.kind = kind;
            forward.handle = 0x0102030405060708;
            forward.metadata = 0xfedcba9876543210;
            const bool is_pooling = kind == ExecutionDnnTensorForwardKind::kPoolingForward;
            const bool is_lrn = kind == ExecutionDnnTensorForwardKind::kLrnForward;
            const bool is_softmax = kind == ExecutionDnnTensorForwardKind::kSoftmaxForward;
            const bool has_descriptor = is_pooling || is_lrn;
            forward.descriptor = has_descriptor ? UINT64_MAX : 0;
            forward.tensors = {5, 5};
            forward.buffers = {{{9, 16}, {9, 16}}};
            forward.algorithm = is_softmax ? UINT32_MAX : 0;
            forward.mode = is_softmax || is_lrn ? 0x80000000 : 0;
            forward.scalar_bytes = width;
            forward.alpha.fill(std::byte{0xa5});
            forward.beta.fill(std::byte{0x5a});
            const auto bytes =
                checked_codec(forward, 96 + 2 * width, write_execution_dnn_tensor_forward,
                              read_execution_dnn_tensor_forward);
            expect(
                bytes[0] == static_cast<std::byte>(kind) && bytes[4] == std::byte{8} &&
                    bytes[11] == std::byte{1} && bytes[28] == std::byte{5} &&
                    bytes[36] == std::byte{5} && bytes[44] == std::byte{9} &&
                    bytes[60] == std::byte{9} && bytes[84] == static_cast<std::byte>(width) &&
                    bytes[92] == std::byte{0xa5} && bytes[96 + width] == std::byte{0x5a},
                "forward layout retains repeated in-place identities and exact coefficient bits");
            for (const auto bad_kind : {0U, 5U, UINT32_MAX}) {
                auto invalid = bytes;
                set_u32(invalid, 0, bad_kind);
                malformed_convolution(forward, invalid, write_execution_dnn_tensor_forward,
                                      read_execution_dnn_tensor_forward);
                auto value = forward;
                value.kind = static_cast<ExecutionDnnTensorForwardKind>(bad_kind);
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_tensor_forward(output, value) && output.take().empty(),
                       "invalid forward kind rejects before encoding");
            }
            for (const auto invalid_width : {0U, 1U, 16U, UINT32_MAX}) {
                auto invalid = bytes;
                set_u32(invalid, 84, invalid_width);
                malformed_convolution(forward, invalid, write_execution_dnn_tensor_forward,
                                      read_execution_dnn_tensor_forward);
                auto value = forward;
                value.scalar_bytes = invalid_width;
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_tensor_forward(output, value) && output.take().empty(),
                       "invalid forward width rejects before encoding");
            }
            for (const auto offset : {std::size_t{88}, std::size_t{92 + width}}) {
                for (const auto length : {0U, width - 1, width + 1, UINT32_MAX}) {
                    auto invalid = bytes;
                    set_u32(invalid, offset, length);
                    malformed_convolution(forward, invalid, write_execution_dnn_tensor_forward,
                                          read_execution_dnn_tensor_forward);
                }
            }
            std::array<ExecutionDnnTensorForward, 3> invalid{forward, forward, forward};
            invalid[0].descriptor = has_descriptor ? 0 : 1;
            invalid[1].algorithm = 1;
            invalid[2].mode = 1;
            const std::size_t invalid_count = is_softmax ? 1 : is_lrn ? 2 : 3;
            for (std::size_t i = 0; i < invalid_count; ++i) {
                ExecutionPayloadWriter output;
                expect(!write_execution_dnn_tensor_forward(output, invalid[i]) &&
                           output.take().empty(),
                       "kind-specific fields reject inconsistent ownership shapes");
                auto malformed = bytes;
                if (i == 0) {
                    set_u32(malformed, 20, has_descriptor ? 0 : 1);
                    set_u32(malformed, 24, 0);
                } else
                    set_u32(malformed, i == 1 ? 76 : 80, 1);
                malformed_convolution(forward, malformed, write_execution_dnn_tensor_forward,
                                      read_execution_dnn_tensor_forward);
            }
        }
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnTensorForward) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnTensorForward),
           "tensor forward never carries replacement provenance or native file capabilities");
}

void lrn_codecs() {
    for (const auto n : {0U, 1U, 16U, UINT32_MAX}) {
        const ExecutionDnnLrn value{n, 0x7ff8123456789abc, 0x8000000000000000, 0xfff0000000000000};
        const auto bytes =
            checked_codec(value, 28, write_execution_dnn_lrn, read_execution_dnn_lrn);
        expect(bytes[4] == std::byte{0xbc} && bytes[11] == std::byte{0x7f} &&
                   bytes[19] == std::byte{0x80} && bytes[26] == std::byte{0xf0} &&
                   bytes[27] == std::byte{0xff},
               "LRN preserves native unsigned window and every double NaN/sign/infinity bit");
        ExecutionPayloadWriter writer;
        expect(writer.write_u64(UINT64_MAX) && write_execution_dnn_lrn(writer, value),
               "LRN setter composes owned ID and exact state");
        const auto payload = writer.take();
        expect(payload.size() == 36, "LRN metadata setter has no native padding");
        ExecutionPayloadReader reader(payload);
        std::uint64_t id{};
        ExecutionDnnLrn output;
        expect(reader.read_u64(id) && id == UINT64_MAX && read_execution_dnn_lrn(reader, output) &&
                   reader.empty() && output.n == n && output.alpha_bits == value.alpha_bits &&
                   output.beta_bits == value.beta_bits && output.k_bits == value.k_bits,
               "SDK validation receives raw setter fields without a hidden normalization");
    }
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kLrn) == 6 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kPoolingForward) == 1 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kSoftmaxForward) == 2 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kLrnForward) == 3 &&
               static_cast<std::uint32_t>(ExecutionDnnTensorForwardKind::kTransformTensor) == 4,
           "LRN and transform extend existing kinds without renumbering or new wire layout");
}

void reduction_codecs() {
    const ExecutionDnnReduction reduction{UINT32_MAX, 0x80000000, 0x12345678, 0, 3};
    const auto state =
        checked_codec(reduction, 20, write_execution_dnn_reduction, read_execution_dnn_reduction);
    expect(state[0] == std::byte{0xff} && state[7] == std::byte{0x80} &&
               state[8] == std::byte{0x78} && state[12] == std::byte{0} &&
               state[16] == std::byte{3},
           "reduction preserves all five raw vendor enums in declared order");
    for (const auto kind :
         {ExecutionDnnReductionQueryKind::kIndices, ExecutionDnnReductionQueryKind::kWorkspace}) {
        const ExecutionDnnReductionQuery query{
            kind, 0x0102030405060708, UINT64_MAX, {1, 0x8877665544332211, 3}};
        const auto bytes = checked_codec(query, 44, write_execution_dnn_reduction_query,
                                         read_execution_dnn_reduction_query);
        expect(bytes[0] == static_cast<std::byte>(kind) && bytes[4] == std::byte{8} &&
                   bytes[20] == std::byte{1} && bytes[28] == std::byte{0x11} &&
                   bytes[35] == std::byte{0x88} && bytes[36] == std::byte{3},
               "size query retains handle/owner/input/reduction/output identities");
        for (const auto invalid_kind : {0U, 3U, UINT32_MAX}) {
            auto malformed = bytes;
            set_u32(malformed, 0, invalid_kind);
            malformed_convolution(query, malformed, write_execution_dnn_reduction_query,
                                  read_execution_dnn_reduction_query);
            auto invalid = query;
            invalid.kind = static_cast<ExecutionDnnReductionQueryKind>(invalid_kind);
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_reduction_query(writer, invalid) && writer.take().empty(),
                   "unknown reduction query kind rejects before writing");
        }
    }
    for (const auto width : {4U, 8U}) {
        for (const bool null_optional : {false, true}) {
            ExecutionDnnReductionCompute compute;
            compute.handle = 0x0102030405060708;
            compute.metadata = UINT64_MAX;
            compute.descriptors = {1, 2, 3};
            compute.buffers = {{{4, 16}, {5, 32}, {6, UINT64_MAX}, {7, 64}}};
            if (null_optional) {
                compute.buffers[2] = {};
                compute.buffers[3] = {};
            }
            compute.indices_bytes = null_optional ? UINT64_MAX : 0;
            compute.workspace_bytes = null_optional ? 0x8877665544332211 : 0;
            compute.scalar_bytes = width;
            compute.alpha.fill(std::byte{0xa5});
            compute.beta.fill(std::byte{0x5a});
            const auto bytes =
                checked_codec(compute, 132 + 2 * width, write_execution_dnn_reduction_compute,
                              read_execution_dnn_reduction_compute);
            expect(bytes[16] == std::byte{1} && bytes[24] == std::byte{2} &&
                       bytes[32] == std::byte{3} && bytes[40] == std::byte{4} &&
                       bytes[56] == std::byte{5} &&
                       bytes[72] == (null_optional ? std::byte{0} : std::byte{6}) &&
                       bytes[88] == (null_optional ? std::byte{0} : std::byte{7}) &&
                       bytes[104] == (null_optional ? std::byte{0xff} : std::byte{0}) &&
                       bytes[112] == (null_optional ? std::byte{0x11} : std::byte{0}) &&
                       bytes[120] == static_cast<std::byte>(width) &&
                       bytes[128] == std::byte{0xa5} && bytes[132 + width] == std::byte{0x5a},
                   "null optional sizes and nonnull zero/end pointers are distinct wire values");
            for (const auto invalid_width : {0U, 1U, 16U, UINT32_MAX}) {
                auto malformed = bytes;
                set_u32(malformed, 120, invalid_width);
                malformed_convolution(compute, malformed, write_execution_dnn_reduction_compute,
                                      read_execution_dnn_reduction_compute);
                auto invalid = compute;
                invalid.scalar_bytes = invalid_width;
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_reduction_compute(writer, invalid) &&
                           writer.take().empty(),
                       "reduction scalar width is bounded before blob access");
            }
            for (const auto offset : {std::size_t{124}, std::size_t{128 + width}}) {
                for (const auto length : {0U, width - 1, width + 1, UINT32_MAX}) {
                    auto malformed = bytes;
                    set_u32(malformed, offset, length);
                    malformed_convolution(compute, malformed, write_execution_dnn_reduction_compute,
                                          read_execution_dnn_reduction_compute);
                }
            }
            for (std::size_t i = 0; i != compute.buffers.size(); ++i) {
                auto invalid = compute;
                invalid.buffers[i] = {0, 1};
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_reduction_compute(writer, invalid) &&
                           writer.take().empty(),
                       "absent allocation cannot carry a forged offset");
                auto malformed = bytes;
                set_u32(malformed, 40 + 16 * i, 0);
                set_u32(malformed, 44 + 16 * i, 0);
                set_u32(malformed, 48 + 16 * i, 1);
                set_u32(malformed, 52 + 16 * i, 0);
                malformed_convolution(compute, malformed, write_execution_dnn_reduction_compute,
                                      read_execution_dnn_reduction_compute);
            }
        }
    }
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kReduction) == 7 &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnReduction) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnReduction),
           "reduction appends an owned original route, not replacement or host indices");
}

void batch_norm_codecs() {
    for (const auto kind :
         {ExecutionDnnBatchNormKind::kForward, ExecutionDnnBatchNormKind::kBackward,
          ExecutionDnnBatchNormKind::kReserve}) {
        ExecutionDnnBatchNormQuery query{
            kind, UINT32_MAX, 0x80000000, 0x0102030405060708, UINT64_MAX, {1, 2, 3, 4, 5, 6, 7}};
        if (kind == ExecutionDnnBatchNormKind::kForward)
            query.descriptors[2] = query.descriptors[4] = 0;
        if (kind == ExecutionDnnBatchNormKind::kReserve)
            for (std::size_t i = 1; i != 6; ++i) query.descriptors[i] = 0;
        const auto bytes = checked_codec(query, 84, write_execution_dnn_batch_norm_query,
                                         read_execution_dnn_batch_norm_query);
        expect(bytes[0] == static_cast<std::byte>(kind) && bytes[4] == std::byte{0xff} &&
                   bytes[11] == std::byte{0x80} && bytes[12] == std::byte{8} &&
                   bytes[19] == std::byte{1} && bytes[28] == std::byte{1} &&
                   bytes[76] == std::byte{7},
               "BN query keeps raw enums and all seven descriptor positions");
        for (const auto bad_kind : {0U, 4U, UINT32_MAX}) {
            auto malformed = bytes;
            set_u32(malformed, 0, bad_kind);
            malformed_convolution(query, malformed, write_execution_dnn_batch_norm_query,
                                  read_execution_dnn_batch_norm_query);
            auto invalid = query;
            invalid.kind = static_cast<ExecutionDnnBatchNormKind>(bad_kind);
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_batch_norm_query(writer, invalid) && writer.take().empty(),
                   "unknown BN query kind rejects before publication");
        }
        for (std::size_t i = 0; i != query.descriptors.size(); ++i) {
            if (query.descriptors[i]) continue;
            auto invalid = query;
            invalid.descriptors[i] = 1;
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_batch_norm_query(writer, invalid) && writer.take().empty(),
                   "query-only and forward-unused descriptors cannot be smuggled");
            auto malformed = bytes;
            set_u32(malformed, 28 + 8 * i, 1);
            malformed_convolution(query, malformed, write_execution_dnn_batch_norm_query,
                                  read_execution_dnn_batch_norm_query);
        }
        if (kind == ExecutionDnnBatchNormKind::kReserve) {
            ExecutionDnnBatchNormCompute invalid;
            invalid.query = query;
            invalid.scalar_bytes = 4;
            ExecutionPayloadWriter writer;
            expect(
                !write_execution_dnn_batch_norm_compute(writer, invalid) && writer.take().empty(),
                "reserve kind never enters the compute queue");
            continue;
        }
        for (const auto width : {4U, 8U}) {
            ExecutionDnnBatchNormCompute compute;
            compute.query = query;
            compute.workspace_bytes = UINT64_MAX;
            compute.reserve_bytes = 0;
            compute.epsilon_bits = 0x7ff8123456789abc;
            compute.average_factor_bits = 0x8000000000000000;
            compute.scalar_bytes = width;
            for (std::size_t i = 0; i != compute.buffers.size(); ++i)
                compute.buffers[i] = {11 + i, 19 + i};
            // Missing running mean and present running variance stay distinct.
            compute.buffers[7] = {};
            compute.buffers[11] = {};
            compute.buffers[12] = {UINT64_MAX, UINT64_MAX};
            if (kind == ExecutionDnnBatchNormKind::kForward)
                compute.buffers[2] = compute.buffers[4] = {};
            const std::size_t count = kind == ExecutionDnnBatchNormKind::kForward ? 2 : 4;
            for (std::size_t i = 0; i != count; ++i)
                for (std::size_t byte = 0; byte != width; ++byte)
                    compute.coefficients[i][byte] = static_cast<std::byte>(0x80 + 8 * i + byte);
            const auto encoded =
                checked_codec(compute, 328 + count * width, write_execution_dnn_batch_norm_compute,
                              read_execution_dnn_batch_norm_compute);
            expect(encoded[84] == std::byte{11} && encoded[292] == std::byte{0xff} &&
                       encoded[300] == std::byte{0} && encoded[308] == std::byte{0xbc} &&
                       encoded[315] == std::byte{0x7f} && encoded[323] == std::byte{0x80} &&
                       encoded[324] == static_cast<std::byte>(width) &&
                       encoded[328] == std::byte{0x80} && encoded[328 + width] == std::byte{0x88},
                   "BN exact doubles and raw fixed coefficient bytes have no blob prefixes");
            for (const auto bad_width : {0U, 1U, 16U, UINT32_MAX}) {
                auto malformed = encoded;
                set_u32(malformed, 324, bad_width);
                malformed_convolution(compute, malformed, write_execution_dnn_batch_norm_compute,
                                      read_execution_dnn_batch_norm_compute);
                auto invalid = compute;
                invalid.scalar_bytes = bad_width;
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_batch_norm_compute(writer, invalid) &&
                           writer.take().empty(),
                       "BN coefficient access is bounded before writing");
            }
            for (std::size_t i = 0; i != compute.buffers.size(); ++i) {
                auto invalid = compute;
                invalid.buffers[i] = {0, 1};
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_batch_norm_compute(writer, invalid) &&
                           writer.take().empty(),
                       "absent BN pointer cannot retain an offset");
                auto malformed = encoded;
                set_u32(malformed, 84 + 16 * i, 0);
                set_u32(malformed, 88 + 16 * i, 0);
                set_u32(malformed, 92 + 16 * i, 1);
                set_u32(malformed, 96 + 16 * i, 0);
                malformed_convolution(compute, malformed, write_execution_dnn_batch_norm_compute,
                                      read_execution_dnn_batch_norm_compute);
            }
            if (kind == ExecutionDnnBatchNormKind::kForward) {
                for (const std::size_t i : {2U, 4U}) {
                    auto malformed = encoded;
                    set_u32(malformed, 84 + 16 * i, 1);
                    malformed_convolution(compute, malformed,
                                          write_execution_dnn_batch_norm_compute,
                                          read_execution_dnn_batch_norm_compute);
                }
                auto invalid = compute;
                invalid.coefficients[3][0] = std::byte{1};
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_batch_norm_compute(writer, invalid) &&
                           writer.take().empty(),
                       "forward cannot hide extra backward coefficients");
            }
        }
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnBatchNorm) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnBatchNorm),
           "BN Ex remains an owned original SDK route");
}

void op_tensor_codecs() {
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kOpTensor) == 9 &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kTensorTransform) == 10,
           "pointwise and transform kinds append without changing previous descriptors");
    const ExecutionDnnOpTensor operation{UINT32_MAX, 0x12345678, 0x80000000};
    const auto encoded =
        checked_codec(operation, 12, write_execution_dnn_op_tensor, read_execution_dnn_op_tensor);
    expect(encoded[0] == std::byte{0xff} && encoded[4] == std::byte{0x78} &&
               encoded[7] == std::byte{0x12} && encoded[11] == std::byte{0x80},
           "pointwise codec preserves raw enum fields for the backend validation boundary");
    for (std::uint32_t mask = 0; mask != 8; ++mask) {
        ExecutionPayloadWriter query;
        expect(query.write_u64(UINT64_MAX) && query.write_u32(mask),
               "pointwise getter encodes every independent optional-output mask");
        const auto message = query.take();
        std::uint64_t id{};
        std::uint32_t decoded_mask{};
        ExecutionPayloadReader input(message);
        expect(message.size() == 12 && input.read_u64(id) && id == UINT64_MAX &&
                   input.read_u32(decoded_mask) && decoded_mask == mask && input.empty(),
               "pointwise getter has one full identity and a three-bit output mask");
        for (std::uint32_t rank = 0; rank <= kExecutionDnnMaxDimensions; ++rank) {
            const ExecutionDnnTensorTransform transform{
                rank,
                UINT32_MAX,
                0x89abcdef,
                mask,
                {INT32_MIN, INT32_MAX, -1, 0, 1, -8, 18, 29},
                {INT32_MAX, INT32_MIN, 0, -1, -18, 12, -4, 3},
                {0, 1, UINT32_MAX, 0x80000000, 13, 24, 35, 46}};
            const auto bytes = checked_codec(transform, 112, write_execution_dnn_tensor_transform,
                                             read_execution_dnn_tensor_transform);
            expect(bytes[0] == static_cast<std::byte>(rank) && bytes[4] == std::byte{0xff} &&
                       bytes[8] == std::byte{0xef} && bytes[11] == std::byte{0x89} &&
                       bytes[12] == static_cast<std::byte>(mask) && bytes[19] == std::byte{0x80} &&
                       bytes[23] == std::byte{0x7f} && bytes[51] == std::byte{0x7f} &&
                       bytes[55] == std::byte{0x80} && bytes[80] == std::byte{0} &&
                       bytes[84] == std::byte{1} && bytes[91] == std::byte{0xff} &&
                       bytes[95] == std::byte{0x80},
                   "transform order is rank/format/direction/presence then before/after/fold");
            for (const auto invalid_rank : {9U, 0x80000000U, UINT32_MAX}) {
                auto invalid = transform;
                invalid.dimensions = invalid_rank;
                ExecutionPayloadWriter output;
                expect(
                    !write_execution_dnn_tensor_transform(output, invalid) && output.take().empty(),
                    "unbounded unsigned transform ranks reject before writing");
                auto malformed = bytes;
                set_u32(malformed, 0, invalid_rank);
                malformed_convolution(transform, malformed, write_execution_dnn_tensor_transform,
                                      read_execution_dnn_tensor_transform);
            }
            for (const auto invalid_mask : {8U, 0x80000000U, UINT32_MAX}) {
                auto invalid = transform;
                invalid.presence = invalid_mask;
                ExecutionPayloadWriter output;
                expect(
                    !write_execution_dnn_tensor_transform(output, invalid) && output.take().empty(),
                    "unknown transform pointer-presence bits reject before writing");
                auto malformed = bytes;
                set_u32(malformed, 12, invalid_mask);
                malformed_convolution(transform, malformed, write_execution_dnn_tensor_transform,
                                      read_execution_dnn_tensor_transform);
            }
        }
    }
    for (const std::uint32_t has_size : {0U, 1U}) {
        ExecutionPayloadWriter writer;
        expect(writer.write_u64(0x1020304050607080) && writer.write_u64(UINT64_MAX) &&
                   writer.write_u64(UINT64_MAX) && writer.write_u32(has_size),
               "InitTransformDest preserves alias identities and optional byte output");
        const auto bytes = writer.take();
        expect(bytes.size() == 28 && bytes[0] == std::byte{0x80} && bytes[8] == std::byte{0xff} &&
                   bytes[16] == std::byte{0xff} && bytes[24] == static_cast<std::byte>(has_size),
               "destination initialization carries three existing IDs, not a pointer allocation");
    }
}

void dropout_codecs() {
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kDropout) == 8 &&
               static_cast<std::uint32_t>(ExecutionDnnDropoutKind::kSet) == 1 &&
               static_cast<std::uint32_t>(ExecutionDnnDropoutKind::kRestore) == 2,
           "dropout kind and operation identities append without changing old metadata");
    for (const auto kind : {ExecutionDnnDropoutKind::kSet, ExecutionDnnDropoutKind::kRestore}) {
        for (const auto probability : {0U, 0x80000000U, 0x3f000000U, 0x7fc12345U, UINT32_MAX}) {
            ExecutionDnnDropout original{
                0x0102030405060708, 0x1020304050607080,
                0x123456789abcdef0, {0x8899aabbccddeeff, 0xffeeddccbbaa9988},
                0x80000000ffffffff, UINT64_MAX,
                probability,        kind};
            const auto encoded = checked_codec(original, 64, write_execution_dnn_dropout,
                                               read_execution_dnn_dropout);
            const std::array fields{original.metadata,      original.descriptor,
                                    original.handle,        original.states.allocation,
                                    original.states.offset, original.state_bytes,
                                    original.seed};
            for (std::size_t field = 0; field != fields.size(); ++field)
                for (std::size_t byte = 0; byte != 8; ++byte)
                    expect(encoded[field * 8 + byte] ==
                               static_cast<std::byte>(fields[field] >> (8 * byte)),
                           "dropout seven u64 fields have exact independent little-endian order");
            for (std::size_t byte = 0; byte != 4; ++byte)
                expect(
                    encoded[56 + byte] == static_cast<std::byte>(probability >> (8 * byte)) &&
                        encoded[60 + byte] ==
                            static_cast<std::byte>(static_cast<std::uint32_t>(kind) >> (8 * byte)),
                    "dropout probability bits precede its final operation kind");
            for (const auto invalid_kind : {0U, 3U, 0x80000000U, UINT32_MAX}) {
                auto malformed = encoded;
                set_u32(malformed, 60, invalid_kind);
                malformed_convolution(original, malformed, write_execution_dnn_dropout,
                                      read_execution_dnn_dropout);
                auto invalid = original;
                invalid.kind = static_cast<ExecutionDnnDropoutKind>(invalid_kind);
                ExecutionPayloadWriter rejected;
                expect(!write_execution_dnn_dropout(rejected, invalid) && rejected.take().empty(),
                       "unknown dropout operation rejects before partial output");
            }
            auto invalid = original;
            invalid.states.allocation = 0;
            ExecutionPayloadWriter rejected;
            expect(!write_execution_dnn_dropout(rejected, invalid) && rejected.take().empty(),
                   "absent state allocation cannot carry a hidden pointer offset");
            auto malformed = encoded;
            std::fill(malformed.begin() + 24, malformed.begin() + 32, std::byte{0});
            malformed_convolution(original, malformed, write_execution_dnn_dropout,
                                  read_execution_dnn_dropout);
            // Null storage with a full-width declared count is deliberately
            // different from an owned nonnull zero-byte/end pointer.
            original.states = {};
            original.state_bytes = UINT64_MAX;
            (void)checked_codec(original, 64, write_execution_dnn_dropout,
                                read_execution_dnn_dropout);
            original.states = {UINT64_MAX, UINT64_MAX};
            original.state_bytes = 0;
            (void)checked_codec(original, 64, write_execution_dnn_dropout,
                                read_execution_dnn_dropout);
        }
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnDropout) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnDropout) &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnDropoutStatesSize),
           "dropout original state initialization cannot advertise a private replacement");
}

void reorder_filter_codecs() {
    for (const auto flag :
         {std::int32_t{0}, std::int32_t{1}, std::int32_t{-1}, INT32_MIN, INT32_MAX}) {
        for (const auto mode : {0U, 1U, UINT32_MAX}) {
            ExecutionDnnReorderFilterAndBias value;
            value.handle = 0x0102030405060708;
            value.metadata = UINT64_MAX;
            value.filter = 0xfedcba9876543210;
            value.reorder = mode;
            value.reorder_bias = flag;
            for (std::size_t i = 0; i != value.buffers.size(); ++i)
                if (i < 2 || flag != 0)
                    value.buffers[i] = {0x8000000000000000ULL + i, UINT64_MAX - i};
            const auto encoded =
                checked_codec(value, 96, write_execution_dnn_reorder_filter_and_bias,
                              read_execution_dnn_reorder_filter_and_bias);
            const std::array identities{value.handle, value.metadata, value.filter};
            for (std::size_t i = 0; i != identities.size(); ++i)
                for (std::size_t byte = 0; byte != 8; ++byte)
                    expect(encoded[i * 8 + byte] ==
                               static_cast<std::byte>(identities[i] >> (8 * byte)),
                           "reorder handle/metadata/filter identities have exact endian order");
            for (std::size_t byte = 0; byte != 4; ++byte)
                expect(encoded[24 + byte] == static_cast<std::byte>(mode >> (8 * byte)) &&
                           encoded[28 + byte] ==
                               static_cast<std::byte>(std::bit_cast<std::uint32_t>(flag) >>
                                                      (8 * byte)),
                       "raw reorder enum and signed non-boolean flag preserve every bit");
            for (std::size_t i = 0; i != value.buffers.size(); ++i) {
                for (std::size_t byte = 0; byte != 8; ++byte)
                    expect(
                        encoded[32 + i * 16 + byte] ==
                                static_cast<std::byte>(value.buffers[i].allocation >> (8 * byte)) &&
                            encoded[40 + i * 16 + byte] ==
                                static_cast<std::byte>(value.buffers[i].offset >> (8 * byte)),
                        "all four allocation/offset pairs keep their exact wire positions");
                auto malformed = encoded;
                std::fill_n(malformed.begin() + static_cast<std::ptrdiff_t>(32 + i * 16), 8,
                            std::byte{0});
                malformed[40 + i * 16] = std::byte{1};
                malformed_convolution(value, malformed, write_execution_dnn_reorder_filter_and_bias,
                                      read_execution_dnn_reorder_filter_and_bias);
                auto invalid = value;
                invalid.buffers[i] = {0, 1};
                ExecutionPayloadWriter rejected;
                expect(!write_execution_dnn_reorder_filter_and_bias(rejected, invalid) &&
                           rejected.take().empty(),
                       "absent allocation with a hidden offset rejects before writing");
                if (flag == 0 && i >= 2) {
                    malformed = encoded;
                    malformed[32 + i * 16] = std::byte{1};
                    malformed_convolution(value, malformed,
                                          write_execution_dnn_reorder_filter_and_bias,
                                          read_execution_dnn_reorder_filter_and_bias);
                    invalid = value;
                    invalid.buffers[i] = {1, 0};
                    ExecutionPayloadWriter ignored;
                    expect(!write_execution_dnn_reorder_filter_and_bias(ignored, invalid) &&
                               ignored.take().empty(),
                           "ignored bias cannot carry unrelated allocation authority");
                }
            }
            // Range and required-identity checks belong to admission, not a
            // CUDA-free value codec; even negative flags retain that separation.
            value.buffers = {};
            (void)checked_codec(value, 96, write_execution_dnn_reorder_filter_and_bias,
                                read_execution_dnn_reorder_filter_and_bias);
        }
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnReorderFilterAndBias) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnReorderFilterAndBias),
           "original reorder cannot advertise replacement status or a backing descriptor");
}

void library_info_codecs() {
    constexpr std::array kinds{
        ExecutionDnnLibraryInfoKind::kVersion, ExecutionDnnLibraryInfoKind::kCudartVersion,
        ExecutionDnnLibraryInfoKind::kMaxDeviceVersion, ExecutionDnnLibraryInfoKind::kProperty};
    for (const auto kind : kinds) {
        ExecutionDnnLibraryInfo query{kind, 0, 0};
        const auto encoded = checked_codec(query, 12, write_execution_dnn_library_info,
                                           read_execution_dnn_library_info);
        expect(encoded[0] == static_cast<std::byte>(kind) &&
                   std::all_of(encoded.begin() + 1, encoded.end(),
                               [](auto value) { return value == std::byte{}; }),
               "library-info kind precedes canonical property/output fields");
        for (const auto invalid_kind : {0U, 5U, UINT32_MAX}) {
            auto invalid = query;
            invalid.kind = static_cast<ExecutionDnnLibraryInfoKind>(invalid_kind);
            ExecutionPayloadWriter rejected;
            expect(!write_execution_dnn_library_info(rejected, invalid) && rejected.take().empty(),
                   "unknown library-info kind rejects before writing");
            auto malformed = encoded;
            set_u32(malformed, 0, invalid_kind);
            malformed_convolution(query, malformed, write_execution_dnn_library_info,
                                  read_execution_dnn_library_info);
        }
        if (kind != ExecutionDnnLibraryInfoKind::kProperty) {
            for (const std::size_t field : {4U, 8U}) {
                auto invalid = query;
                if (field == 4)
                    invalid.property = 1;
                else
                    invalid.has_output = 1;
                ExecutionPayloadWriter rejected;
                expect(
                    !write_execution_dnn_library_info(rejected, invalid) && rejected.take().empty(),
                    "direct-return query rejects unused selector/output authority");
                auto malformed = encoded;
                set_u32(malformed, field, 1);
                malformed_convolution(query, malformed, write_execution_dnn_library_info,
                                      read_execution_dnn_library_info);
            }
            continue;
        }
        for (const auto selector : {0U, 1U, 2U, 3U, 0x80000000U, UINT32_MAX})
            for (const auto present : {0U, 1U}) {
                query.property = selector;
                query.has_output = present;
                const auto bytes = checked_codec(query, 12, write_execution_dnn_library_info,
                                                 read_execution_dnn_library_info);
                for (std::size_t byte = 0; byte != 4; ++byte)
                    expect(bytes[4 + byte] == static_cast<std::byte>(selector >> (8 * byte)) &&
                               bytes[8 + byte] == static_cast<std::byte>(present >> (8 * byte)),
                           "raw selectors and independent output presence preserve wire bits");
                for (const auto invalid_presence : {2U, UINT32_MAX}) {
                    auto invalid = query;
                    invalid.has_output = invalid_presence;
                    ExecutionPayloadWriter rejected;
                    expect(!write_execution_dnn_library_info(rejected, invalid) &&
                               rejected.take().empty(),
                           "Property output presence must be a canonical boolean");
                    auto malformed = bytes;
                    set_u32(malformed, 8, invalid_presence);
                    malformed_convolution(query, malformed, write_execution_dnn_library_info,
                                          read_execution_dnn_library_info);
                }
            }
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnLibraryInfo) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnLibraryInfo),
           "library observations neither replace computation nor import capabilities");
}

void backward_convolution_query_codecs() {
    constexpr std::array kinds{ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount,
                               ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount,
                               ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace};
    for (const auto kind : kinds)
        for (const auto presence : {0U, 1U}) {
            ExecutionDnnBackwardConvolutionQuery query;
            query.kind = kind;
            query.handle = 0x0102030405060708;
            query.has_output = presence;
            const bool is_workspace = kind == kinds.back();
            if (is_workspace) {
                query.metadata = 0x8877665544332211;
                query.descriptors = {0x1122334455667788, 0x2132435465768798, 0x31425364758697a8,
                                     0x415263748596a7b8};
                query.algorithm = UINT32_MAX;
            }
            const auto bytes =
                checked_codec(query, 60, write_execution_dnn_backward_convolution_query,
                              read_execution_dnn_backward_convolution_query);
            expect(bytes[0] == static_cast<std::byte>(kind) && bytes[4] == std::byte{8} &&
                       bytes[11] == std::byte{1} && bytes[56] == static_cast<std::byte>(presence),
                   "backward query encodes kind/handle/presence at frozen offsets");
            if (is_workspace)
                expect(
                    bytes[12] == std::byte{0x11} && bytes[19] == std::byte{0x88} &&
                        bytes[20] == std::byte{0x88} && bytes[28] == std::byte{0x98} &&
                        bytes[36] == std::byte{0xa8} && bytes[44] == std::byte{0xb8} &&
                        bytes[52] == std::byte{0xff} && bytes[55] == std::byte{0xff},
                    "workspace preserves public filter/dy/convolution/dx order and raw algorithm");
            else
                expect(std::all_of(bytes.begin() + 12, bytes.begin() + 56,
                                   [](auto value) { return value == std::byte{}; }),
                       "MaxCount has no implicit metadata or algorithm authority");
            const auto reject = [&](const ExecutionDnnBackwardConvolutionQuery& invalid,
                                    std::size_t offset, std::uint32_t value) {
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_backward_convolution_query(writer, invalid) &&
                           writer.take().empty(),
                       "invalid backward query rejects before writing");
                auto malformed = bytes;
                set_u32(malformed, offset, value);
                malformed_convolution(query, malformed,
                                      write_execution_dnn_backward_convolution_query,
                                      read_execution_dnn_backward_convolution_query);
            };
            for (const auto unknown : {0U, 4U, UINT32_MAX}) {
                auto invalid = query;
                invalid.kind = static_cast<ExecutionDnnBackwardConvolutionQueryKind>(unknown);
                reject(invalid, 0, unknown);
            }
            for (const auto invalid_presence : {2U, UINT32_MAX}) {
                auto invalid = query;
                invalid.has_output = invalid_presence;
                reject(invalid, 56, invalid_presence);
            }
            if (!is_workspace) {
                auto invalid = query;
                invalid.metadata = 1;
                reject(invalid, 12, 1);
                invalid = query;
                invalid.algorithm = 1;
                reject(invalid, 52, 1);
                for (std::size_t index = 0; index != query.descriptors.size(); ++index) {
                    invalid = query;
                    invalid.descriptors[index] = 1;
                    reject(invalid, 20 + 8 * index, 1);
                }
            }
        }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnBackwardConvolutionQuery) &&
               !execution_requires_descriptor(ExecutionOpcode::kDnnBackwardConvolutionQuery),
           "backward scalar queries neither replace computation nor import backing descriptors");
}

void folded_convolution_codecs() {
    ExecutionDnnFoldedConvBackwardDataDescriptors descriptors;
    descriptors.handle = 0x0807060504030201;
    descriptors.metadata = 0x1817161514131211;
    descriptors.inputs = {0x2827262524232221, 0x3837363534333231, 0x4847464544434241,
                          0x5857565554535251};
    descriptors.outputs = {0x6867666564636261, 0x7877767574737271, 0x8887868584838281,
                           0x9897969594939291, 0xa8a7a6a5a4a3a2a1, 0xb8b7b6b5b4b3b2b1,
                           0xc8c7c6c5c4c3c2c1, 0xd8d7d6d5d4d3d2d1};
    const auto roundtrip = [](const auto& value) {
        return checked_codec(value, 116, write_execution_dnn_folded_conv_backward_data_descriptors,
                             read_execution_dnn_folded_conv_backward_data_descriptors);
    };
    for (const auto format : {0U, 1U, 2U, 3U, 4U, UINT32_MAX}) {
        descriptors.format = format;
        const auto bytes = roundtrip(descriptors);
        expect(bytes[0] == std::byte{1} && bytes[7] == std::byte{8} &&
                   bytes[8] == std::byte{0x11} && bytes[15] == std::byte{0x18} &&
                   bytes[16] == std::byte{0x21} && bytes[24] == std::byte{0x31} &&
                   bytes[32] == std::byte{0x41} && bytes[40] == std::byte{0x51} &&
                   bytes[48] == static_cast<std::byte>(format & 0xffU) &&
                   bytes[51] == static_cast<std::byte>(format >> 24) &&
                   bytes[52] == std::byte{0x61} && bytes[59] == std::byte{0x68} &&
                   bytes[108] == std::byte{0xd1} && bytes[115] == std::byte{0xd8},
               "Folded codec preserves full-width identities, raw format and packed ABI offsets");
    }
    auto aliases = descriptors;
    aliases.outputs[0] = aliases.inputs[0];
    aliases.outputs[1] = aliases.inputs[1];
    aliases.outputs[2] = aliases.inputs[2];
    aliases.outputs[3] = aliases.inputs[1];
    aliases.outputs[5] = aliases.outputs[6] = aliases.outputs[7] = aliases.outputs[4];
    (void)roundtrip(aliases);
    aliases.outputs[7] = 0;
    (void)roundtrip(aliases);
    // The codec retains absent mandatory identities for independent service/
    // registry validation; it cannot establish ownership or descriptor kinds.
    (void)roundtrip(ExecutionDnnFoldedConvBackwardDataDescriptors{});
    expect(
        !execution_allows_compute_replacement(
            ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors) &&
            !execution_requires_descriptor(ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors),
        "Folded mutates metadata without a replacement command or backing descriptor");
}

void sequence_metadata_codecs() {
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kRnnData) == 11 &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kSeqData) == 12,
           "sequence kinds append to existing opaque metadata identities");
    ExecutionDnnRnnDataState rnn;
    rnn.type = UINT32_MAX;
    rnn.layout = 0x12345678;
    rnn.max_sequence_length = INT32_MIN;
    rnn.batch_size = INT32_MAX;
    rnn.vector_size = -1;
    rnn.presence = 3;
    rnn.lengths = {INT32_MIN, -1, 0, INT32_MAX};
    ExecutionDnnSeqDataState seq;
    seq.type = UINT32_MAX;
    seq.dimensions = INT32_MIN;
    seq.presence = 15;
    seq.sequence_count = UINT64_MAX;
    seq.sizes = {INT32_MIN, -1, 0, INT32_MAX};
    seq.axes = {0, 3, 4, UINT32_MAX};
    seq.lengths = rnn.lengths;
    for (std::size_t i = 0; i < 8; ++i) rnn.padding[i] = seq.padding[i] = std::byte(0xa0 + i);
    for (const auto width : {0U, 2U, 4U, 8U}) {
        rnn.padding_bytes = seq.padding_bytes = width;
        const auto a = checked_codec(rnn, 48 + width, write_execution_dnn_rnn_data,
                                     read_execution_dnn_rnn_data);
        const auto b = checked_codec(seq, 76 + width, write_execution_dnn_seq_data,
                                     read_execution_dnn_seq_data);
        expect(a[8] == std::byte{0} && a[11] == std::byte{0x80} && a[24] == std::byte{16} &&
                   a[28] == std::byte{0} && a[31] == std::byte{0x80} && b[12] == std::byte{0xff} &&
                   b[19] == std::byte{0xff} && b[52] == std::byte{16} && b[56] == std::byte{0} &&
                   b[59] == std::byte{0x80},
               "sequence wire uses raw signed bits, full u64 count and byte-length LE32 blobs");
    }
    for (std::uint32_t presence = 0; presence != 16; ++presence) {
        auto value = seq;
        value.presence = presence;
        if (!(presence & 4)) value.lengths.clear();
        if (!(presence & 8)) value.padding_bytes = 0;
        (void)checked_codec(value, 60 + 4 * value.lengths.size() + value.padding_bytes,
                            write_execution_dnn_seq_data, read_execution_dnn_seq_data);
        if (presence < 4) {
            auto input = rnn;
            input.presence = presence;
            if (!(presence & 1)) input.lengths.clear();
            if (!(presence & 2)) input.padding_bytes = 0;
            (void)checked_codec(input, 32 + 4 * input.lengths.size() + input.padding_bytes,
                                write_execution_dnn_rnn_data, read_execution_dnn_rnn_data);
        }
    }
    ExecutionPayloadWriter a, b;
    expect(write_execution_dnn_rnn_data(a, rnn) && write_execution_dnn_seq_data(b, seq),
           "encode malformed-input references");
    const auto rnn_bytes = a.take(), seq_bytes = b.take();
    for (const auto count :
         {1U, 3U, UINT32_MAX,
          4U * static_cast<std::uint32_t>(kExecutionDnnMaxSequenceLengths + 1)}) {
        auto bad_rnn = rnn_bytes, bad_seq = seq_bytes;
        set_u32(bad_rnn, 24, static_cast<std::uint32_t>(count));
        set_u32(bad_seq, 52, static_cast<std::uint32_t>(count));
        malformed_convolution(rnn, bad_rnn, write_execution_dnn_rnn_data,
                              read_execution_dnn_rnn_data);
        malformed_convolution(seq, bad_seq, write_execution_dnn_seq_data,
                              read_execution_dnn_seq_data);
    }
    for (auto width : {1U, 3U, 5U, 9U, UINT32_MAX}) {
        auto bad_rnn = rnn_bytes, bad_seq = seq_bytes;
        set_u32(bad_rnn, 44, width);
        set_u32(bad_seq, 72, width);
        malformed_convolution(rnn, bad_rnn, write_execution_dnn_rnn_data,
                              read_execution_dnn_rnn_data);
        malformed_convolution(seq, bad_seq, write_execution_dnn_seq_data,
                              read_execution_dnn_seq_data);
    }
    for (const auto mask : {0U, 128U, UINT32_MAX}) {
        auto bad_rnn = rnn_bytes, bad_seq = seq_bytes;
        set_u32(bad_rnn, 20, mask);
        set_u32(bad_seq, 8, mask);
        malformed_convolution(rnn, bad_rnn, write_execution_dnn_rnn_data,
                              read_execution_dnn_rnn_data);
        malformed_convolution(seq, bad_seq, write_execution_dnn_seq_data,
                              read_execution_dnn_seq_data);
    }
    for (std::uint32_t outputs = 0; outputs != 128; ++outputs) {
        const ExecutionDnnSeqDataQuery query{outputs % 2 ? INT32_MIN : INT32_MAX, UINT64_MAX,
                                             outputs};
        (void)checked_codec(query, 16, write_execution_dnn_seq_data_query,
                            read_execution_dnn_seq_data_query);
    }
    ExecutionDnnSeqDataQuery query{4, UINT64_MAX, 128};
    ExecutionPayloadWriter rejected;
    expect(!write_execution_dnn_seq_data_query(rejected, query), "unknown nullable mask rejects");
    query.outputs = 127;
    auto query_bytes = checked_codec(query, 16, write_execution_dnn_seq_data_query,
                                     read_execution_dnn_seq_data_query);
    set_u32(query_bytes, 12, 128);
    malformed_convolution(query, query_bytes, write_execution_dnn_seq_data_query,
                          read_execution_dnn_seq_data_query);

    // Do not apply the exhaustive every-byte truncation helper to a 4 MiB value.
    rnn.lengths.assign(kExecutionDnnMaxSequenceLengths, -17);
    ExecutionPayloadWriter maximum;
    expect(write_execution_dnn_rnn_data(maximum, rnn), "maximum admitted sequence blob encodes");
    const auto maximum_bytes = maximum.take();
    ExecutionPayloadReader maximum_reader(maximum_bytes);
    ExecutionDnnRnnDataState decoded;
    expect(read_execution_dnn_rnn_data(maximum_reader, decoded) && maximum_reader.empty() &&
               decoded.lengths == rnn.lengths,
           "maximum admitted sequence blob decodes completely");
    auto oversized = maximum_bytes;
    oversized.insert(oversized.begin() + 28, 4, std::byte{});
    set_u32(oversized, 24, 4U * static_cast<std::uint32_t>(kExecutionDnnMaxSequenceLengths + 1));
    malformed_convolution(rnn, oversized, write_execution_dnn_rnn_data,
                          read_execution_dnn_rnn_data);
    rnn.lengths.push_back(1);
    ExecutionPayloadWriter too_large;
    expect(!write_execution_dnn_rnn_data(too_large, rnn), "length quota is checked on write too");
}

void rnn_attention_codecs() {
    expect(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kRnn) == 13 &&
               static_cast<std::uint32_t>(ExecutionDnnObjectKind::kAttn) == 14,
           "RNN and attention kinds append after SeqData");
    const ExecutionDnnRnnState rnn{
        0x04030201, 0x14131211, 0x24232221, 0x34333231, 0x44434241, 0x54535251,         0x64636261,
        0x74737271, INT32_MIN,  INT32_MAX,  -1,         0,          0x8887868584838281, UINT32_MAX};
    const auto rnn_bytes = checked_codec(rnn, 60, write_execution_dnn_rnn, read_execution_dnn_rnn);
    expect(rnn_bytes[0] == std::byte{1} && rnn_bytes[31] == std::byte{0x74} &&
               rnn_bytes[35] == std::byte{0x80} && rnn_bytes[39] == std::byte{0x7f} &&
               rnn_bytes[48] == std::byte{0x81} && rnn_bytes[55] == std::byte{0x88} &&
               rnn_bytes[56] == std::byte{0xff},
           "RNN wire preserves enum fields, signed sizes, full identity and unsigned flags");
    auto nullable_rnn = rnn;
    nullable_rnn.dropout = 0;
    (void)checked_codec(nullable_rnn, 60, write_execution_dnn_rnn, read_execution_dnn_rnn);

    ExecutionDnnAttnState attn;
    attn.mode = UINT32_MAX;
    attn.heads = INT32_MIN;
    attn.type = 0x04030201;
    attn.precision = 0x14131211;
    attn.math_type = 0x24232221;
    attn.dropouts = {0x3837363534333231, 0x4847464544434241};
    attn.query_size = INT32_MIN;
    attn.key_size = INT32_MAX;
    attn.value_size = -1;
    attn.query_projection = 0;
    attn.key_projection = 0x01020304;
    attn.value_projection = 0x11121314;
    attn.output_projection = 0x21222324;
    attn.max_query_sequence = 0x31323334;
    attn.max_key_sequence = 0x41424344;
    attn.max_batch = 0x51525354;
    attn.max_beam = 0x61626364;
    for (const std::uint64_t bits :
         {UINT64_C(0), UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000000),
          UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_MAX}) {
        attn.scale_bits = bits;
        const auto bytes =
            checked_codec(attn, 88, write_execution_dnn_attn, read_execution_dnn_attn);
        expect(bytes[0] == std::byte{0xff} && bytes[7] == std::byte{0x80} &&
                   bytes[8] == static_cast<std::byte>(bits & 0xffU) &&
                   bytes[15] == static_cast<std::byte>(bits >> 56) && bytes[16] == std::byte{1} &&
                   bytes[28] == std::byte{0x31} && bytes[36] == std::byte{0x41} &&
                   bytes[47] == std::byte{0x80} && bytes[84] == std::byte{0x64} &&
                   bytes[87] == std::byte{0x61},
               "attention wire keeps raw double payloads, positional identities and eleven sizes");
    }
    for (const std::array<std::uint64_t, 2> aliases : {std::array<std::uint64_t, 2>{0, 0},
                                                       {0, UINT64_MAX},
                                                       {UINT64_MAX, 0},
                                                       {UINT64_MAX, UINT64_MAX}}) {
        attn.dropouts = aliases;
        (void)checked_codec(attn, 88, write_execution_dnn_attn, read_execution_dnn_attn);
    }
    ExecutionPayloadWriter composed;
    expect(composed.write_u64(UINT64_MAX) && write_execution_dnn_rnn(composed, rnn) &&
               write_execution_dnn_attn(composed, attn) && composed.write_u32(17),
           "fixed state codecs compose after descriptor identity");
    const auto bytes = composed.take();
    ExecutionPayloadReader reader(bytes);
    std::uint64_t id{};
    std::uint32_t sentinel{};
    ExecutionDnnRnnState first;
    ExecutionDnnAttnState second;
    expect(reader.read_u64(id) && id == UINT64_MAX && read_execution_dnn_rnn(reader, first) &&
               read_execution_dnn_attn(reader, second) && reader.read_u32(sentinel) &&
               sentinel == 17 && reader.empty(),
           "RNN/attention readers consume only their own wire fields");
}

void rnn_query_clip_codecs() {
    constexpr std::array kinds{ExecutionDnnRnnQueryKind::kWeightSpace,
                               ExecutionDnnRnnQueryKind::kTempSpace,
                               ExecutionDnnRnnQueryKind::kAttentionBuffers};
    for (std::size_t index = 0; index != kinds.size(); ++index) {
        const bool is_temp = kinds[index] == ExecutionDnnRnnQueryKind::kTempSpace;
        const std::uint32_t allowed = index == 0 ? 1U : index == 1 ? 6U : 7U;
        ExecutionDnnRnnQuery query{kinds[index],
                                   0x1817161514131211,
                                   0x2827262524232221,
                                   0x3837363534333231,
                                   is_temp ? UINT64_MAX : 0,
                                   is_temp ? UINT32_MAX : 0,
                                   allowed};
        const auto bytes =
            checked_codec(query, 44, write_execution_dnn_rnn_query, read_execution_dnn_rnn_query);
        expect(bytes[0] == std::byte(index + 1) && bytes[4] == std::byte{0x11} &&
                   bytes[12] == std::byte{0x21} && bytes[20] == std::byte{0x31} &&
                   bytes[40] == std::byte(allowed),
               "RNN query exact field order excludes host alignment padding");
        for (std::uint32_t mask = 0; mask != 8; ++mask) {
            auto masked = query;
            masked.outputs = mask;
            if (!(mask & ~allowed)) {
                (void)checked_codec(masked, 44, write_execution_dnn_rnn_query,
                                    read_execution_dnn_rnn_query);
            } else {
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_rnn_query(writer, masked),
                       "per-kind output masks reject unrelated slots");
                auto bad = bytes;
                set_u32(bad, 40, mask);
                malformed_convolution(query, bad, write_execution_dnn_rnn_query,
                                      read_execution_dnn_rnn_query);
            }
        }
        for (auto mask : {8U, UINT32_MAX}) {
            auto bad = bytes;
            set_u32(bad, 40, mask);
            malformed_convolution(query, bad, write_execution_dnn_rnn_query,
                                  read_execution_dnn_rnn_query);
        }
        for (auto kind : {0U, 4U, UINT32_MAX}) {
            auto bad = bytes;
            set_u32(bad, 0, kind);
            malformed_convolution(query, bad, write_execution_dnn_rnn_query,
                                  read_execution_dnn_rnn_query);
        }
        if (!is_temp) {
            for (auto offset : {28U, 36U}) {
                auto bad = bytes;
                set_u32(bad, offset, 1);
                malformed_convolution(query, bad, write_execution_dnn_rnn_query,
                                      read_execution_dnn_rnn_query);
            }
            query.input = 1;
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_rnn_query(writer, query),
                   "unused input identity must be canonical zero");
            query.input = 0;
            query.mode = 1;
            expect(!write_execution_dnn_rnn_query(writer, query),
                   "unused forward mode must be canonical zero");
        }
    }
    for (auto mode : {0U, 1U, 2U, UINT32_MAX}) {
        ExecutionDnnRnnQuery query{ExecutionDnnRnnQueryKind::kTempSpace, 0, 0, 0, 0, mode, 6};
        (void)checked_codec(query, 44, write_execution_dnn_rnn_query, read_execution_dnn_rnn_query);
    }
    const ExecutionDnnRnnQueryResult result{{UINT64_MAX, 0, 0x8877665544332211}};
    const auto bytes = checked_codec(result, 24, write_execution_dnn_rnn_query_result,
                                     read_execution_dnn_rnn_query_result);
    expect(bytes[7] == std::byte{0xff} && bytes[8] == std::byte{0} &&
               bytes[16] == std::byte{0x11} && bytes[23] == std::byte{0x88},
           "result slots preserve all size bits without native size_t narrowing");
    for (auto bits : {UINT64_C(0), UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000000),
                      UINT64_C(0xfff0000000000000), UINT64_C(0x7ff8123456789abc), UINT64_MAX}) {
        ExecutionDnnRnnClip clip{UINT32_MAX, 0x11223344, bits, ~bits};
        const auto encoded =
            checked_codec(clip, 24, write_execution_dnn_rnn_clip, read_execution_dnn_rnn_clip);
        expect(encoded[0] == std::byte{0xff} && encoded[4] == std::byte{0x44},
               "Clip preserves raw enums and special floating representations");
        ExecutionPayloadWriter composed;
        expect(composed.write_u64(UINT64_MAX) && write_execution_dnn_rnn_clip(composed, clip) &&
                   write_execution_dnn_rnn_query_result(composed, result),
               "Clip request composes with descriptor identity and following values");
        const auto compound = composed.take();
        ExecutionPayloadReader reader(compound);
        std::uint64_t id{};
        ExecutionDnnRnnClip decoded;
        ExecutionDnnRnnQueryResult sizes;
        expect(compound.size() == 56 && reader.read_u64(id) && id == UINT64_MAX &&
                   read_execution_dnn_rnn_clip(reader, decoded) && decoded.lower_bits == bits &&
                   decoded.upper_bits == ~bits &&
                   read_execution_dnn_rnn_query_result(reader, sizes) &&
                   sizes.sizes == result.sizes && reader.empty(),
               "Clip and result codecs consume only their exact framed fields");
    }
}

void rnn_compute_codecs() {
    constexpr std::array kinds{ExecutionDnnRnnComputeKind::kForward,
                               ExecutionDnnRnnComputeKind::kBackwardData,
                               ExecutionDnnRnnComputeKind::kBackwardWeights};
    // Explicit independent ABI slot lists, rather than the production bit masks.
    const std::array<std::vector<std::size_t>, 3> unused{
        {{3, 4, 7, 8, 11, 12}, {1, 6, 10}, {3, 4, 6, 7, 8, 9, 10, 11, 12}}};
    for (std::size_t kind_index = 0; kind_index < kinds.size(); ++kind_index) {
        ExecutionDnnRnnCompute value;
        value.kind = kinds[kind_index];
        value.mode = kind_index == 1 ? 0 : UINT32_MAX;
        value.handle = 0x1817161514131211;
        value.metadata = 0x2827262524232221;
        value.descriptors = {0x3837363534333231, UINT64_MAX, UINT64_MAX, 0, 0};
        if (kind_index != 2) value.descriptors[4] = 0x4847464544434241;
        for (std::size_t slot = 0; slot < value.buffers.size(); ++slot)
            value.buffers[slot] = {0x8877665544332200 + slot, UINT64_MAX - slot};
        for (const auto slot : unused[kind_index]) value.buffers[slot] = {};
        value.weight_bytes = UINT64_MAX;
        value.workspace_bytes = 0;
        value.reserve_bytes = 0xf8f7f6f5f4f3f2f1;
        const auto bytes = checked_codec(value, 344, write_execution_dnn_rnn_compute,
                                         read_execution_dnn_rnn_compute);
        expect(bytes[0] == std::byte(kind_index + 1) && bytes[8] == std::byte{0x11} &&
                   bytes[16] == std::byte{0x21} && bytes[24] == std::byte{0x31} &&
                   bytes[64] == std::byte{0} && bytes[71] == std::byte{0x88} &&
                   bytes[72] == std::byte{0xff} && bytes[320] == std::byte{0xff} &&
                   bytes[328] == std::byte{0} && bytes[336] == std::byte{0xf1} &&
                   bytes[343] == std::byte{0xf8},
               "RNN wire offsets preserve kind, authority, positional IDs, ranges and all counts");
        ExecutionPayloadReader fields(bytes);
        std::uint32_t small{};
        std::uint64_t large{};
        expect(fields.read_u32(small) && small == kind_index + 1 && fields.read_u32(small) &&
                   small == value.mode && fields.read_u64(large) && large == value.handle &&
                   fields.read_u64(large) && large == value.metadata,
               "RNN header fields follow the independently decoded frozen order");
        for (const auto descriptor : value.descriptors)
            expect(fields.read_u64(large) && large == descriptor,
                   "every positional descriptor remains in its original slot");
        for (const auto& buffer : value.buffers)
            expect(fields.read_u64(large) && large == buffer.allocation && fields.read_u64(large) &&
                       large == buffer.offset,
                   "every positional buffer retains allocation and offset order");
        expect(fields.read_u64(large) && large == value.weight_bytes && fields.read_u64(large) &&
                   large == value.workspace_bytes && fields.read_u64(large) &&
                   large == value.reserve_bytes && fields.empty(),
               "three independent sizes terminate the fixed compute value");
        const auto reject_value = [&](const ExecutionDnnRnnCompute& invalid) {
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_rnn_compute(writer, invalid) && writer.take().empty(),
                   "invalid RNN shape rejects before writing any field");
        };
        for (const auto kind : {0U, 4U, UINT32_MAX}) {
            auto invalid = value;
            invalid.kind = static_cast<ExecutionDnnRnnComputeKind>(kind);
            reject_value(invalid);
            auto bad = bytes;
            set_u32(bad, 0, kind);
            malformed_convolution(value, bad, write_execution_dnn_rnn_compute,
                                  read_execution_dnn_rnn_compute);
        }
        for (const auto slot : unused[kind_index]) {
            auto invalid = value;
            invalid.buffers[slot].allocation = 1;
            reject_value(invalid);
            auto bad = bytes;
            set_u32(bad, 64 + slot * 16, 1);
            malformed_convolution(value, bad, write_execution_dnn_rnn_compute,
                                  read_execution_dnn_rnn_compute);
        }
        for (std::size_t slot = 0; slot < value.buffers.size(); ++slot) {
            auto invalid = value;
            invalid.buffers[slot] = {0, UINT64_MAX};
            reject_value(invalid);
            auto bad = bytes;
            set_u32(bad, 64 + slot * 16, 0);
            set_u32(bad, 68 + slot * 16, 0);
            set_u32(bad, 72 + slot * 16, 1);
            malformed_convolution(value, bad, write_execution_dnn_rnn_compute,
                                  read_execution_dnn_rnn_compute);
            // An absent pointer does not zero its independent native size count.
            auto absent = value;
            absent.buffers[slot] = {};
            (void)checked_codec(absent, 344, write_execution_dnn_rnn_compute,
                                read_execution_dnn_rnn_compute);
        }
        for (const auto mode : {0U, 1U, 2U, UINT32_MAX}) {
            auto changed = value;
            changed.mode = mode;
            if (kind_index != 1 || mode == 0) {
                (void)checked_codec(changed, 344, write_execution_dnn_rnn_compute,
                                    read_execution_dnn_rnn_compute);
            } else {
                reject_value(changed);
                auto bad = bytes;
                set_u32(bad, 4, mode);
                malformed_convolution(value, bad, write_execution_dnn_rnn_compute,
                                      read_execution_dnn_rnn_compute);
            }
        }
        if (kind_index == 2) {
            auto invalid = value;
            invalid.descriptors[4] = 1;
            reject_value(invalid);
            auto bad = bytes;
            set_u32(bad, 56, 1);
            malformed_convolution(value, bad, write_execution_dnn_rnn_compute,
                                  read_execution_dnn_rnn_compute);
        }
        auto aliases = value;
        aliases.descriptors.fill(UINT64_MAX);
        if (kind_index == 2) aliases.descriptors[4] = 0;
        for (std::size_t slot = 0; slot < aliases.buffers.size(); ++slot)
            if (std::ranges::find(unused[kind_index], slot) == unused[kind_index].end())
                aliases.buffers[slot] = {UINT64_MAX, UINT64_MAX};
        (void)checked_codec(aliases, 344, write_execution_dnn_rnn_compute,
                            read_execution_dnn_rnn_compute);
        ExecutionDnnRnnCompute empty;
        empty.kind = value.kind;
        empty.weight_bytes = empty.workspace_bytes = empty.reserve_bytes = UINT64_MAX;
        (void)checked_codec(empty, 344, write_execution_dnn_rnn_compute,
                            read_execution_dnn_rnn_compute);
    }
    for (const auto batch : {INT32_MIN, -1, 0, 1, INT32_MAX}) {
        const ExecutionDnnBuildRnnDynamic build{0x1817161514131211, 0x2827262524232221, UINT64_MAX,
                                                batch};
        const auto bytes = checked_codec(build, 28, write_execution_dnn_build_rnn_dynamic,
                                         read_execution_dnn_build_rnn_dynamic);
        expect(bytes[0] == std::byte{0x11} && bytes[8] == std::byte{0x21} &&
                   bytes[16] == std::byte{0xff} &&
                   bytes[24] == std::byte(std::bit_cast<std::uint32_t>(batch) & 0xffU) &&
                   bytes[27] == std::byte(std::bit_cast<std::uint32_t>(batch) >> 24),
               "Build preserves every signed mini-batch bit after three full-width identities");
    }
    const ExecutionDnnBuildRnnDynamic empty_build;
    (void)checked_codec(empty_build, 28, write_execution_dnn_build_rnn_dynamic,
                        read_execution_dnn_build_rnn_dynamic);
    expect(static_cast<std::uint32_t>(kinds[0]) == 1 && static_cast<std::uint32_t>(kinds[1]) == 2 &&
               static_cast<std::uint32_t>(kinds[2]) == 3 &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnRnnCompute) &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnBuildRnnDynamic),
           "RNN kinds are stable and do not authorize replacement or prove native issue");
}

void rnn_weight_params_codecs() {
    for (const auto index : {INT32_MIN, -1, 0, 8, INT32_MAX}) {
        for (std::uint32_t outputs = 0; outputs != 4; ++outputs) {
            ExecutionDnnRnnWeightParams value{0x0807060504030201,
                                              0x1817161514131211,
                                              0x2827262524232221,
                                              index,
                                              index,
                                              {0x3837363534333231, 0x4847464544434241},
                                              UINT64_MAX,
                                              {0x5857565554535251, 0x6867666564636261},
                                              outputs};
            const auto bytes = checked_codec(value, 76, write_execution_dnn_rnn_weight_params,
                                             read_execution_dnn_rnn_weight_params);
            expect(bytes[0] == std::byte{1} && bytes[8] == std::byte{0x11} &&
                       bytes[16] == std::byte{0x21} &&
                       bytes[24] == std::byte(std::bit_cast<std::uint32_t>(index) & 0xffU) &&
                       bytes[31] == std::byte(std::bit_cast<std::uint32_t>(index) >> 24) &&
                       bytes[32] == std::byte{0x31} && bytes[40] == std::byte{0x41} &&
                       bytes[48] == std::byte{0xff} && bytes[56] == std::byte{0x51} &&
                       bytes[64] == std::byte{0x61} && bytes[72] == std::byte(outputs),
                   "WeightParams exact wire positions preserve signed indices and full widths");
            for (unsigned int absent = 0; absent != 4; ++absent) {
                auto optional = value;
                optional.weights = {};
                optional.weight_bytes = absent ? UINT64_MAX : 0;
                for (std::size_t i = 0; i != optional.destinations.size(); ++i)
                    if (absent & (1U << i)) optional.destinations[i] = 0;
                (void)checked_codec(optional, 76, write_execution_dnn_rnn_weight_params,
                                    read_execution_dnn_rnn_weight_params);
            }
            value.destinations[1] = value.destinations[0];
            (void)checked_codec(value, 76, write_execution_dnn_rnn_weight_params,
                                read_execution_dnn_rnn_weight_params);
            for (const auto mask : {4U, UINT32_MAX}) {
                auto bad = bytes;
                set_u32(bad, 72, mask);
                malformed_convolution(value, bad, write_execution_dnn_rnn_weight_params,
                                      read_execution_dnn_rnn_weight_params);
                auto invalid = value;
                invalid.outputs = mask;
                ExecutionPayloadWriter writer;
                expect(!write_execution_dnn_rnn_weight_params(writer, invalid) &&
                           writer.take().empty(),
                       "invalid output mask rejects before partial encoding");
            }
            auto bad = bytes;
            std::fill_n(bad.begin() + 32, 8, std::byte{});
            malformed_convolution(value, bad, write_execution_dnn_rnn_weight_params,
                                  read_execution_dnn_rnn_weight_params);
            value.weights.allocation = 0;
            ExecutionPayloadWriter invalid;
            expect(!write_execution_dnn_rnn_weight_params(invalid, value) && invalid.take().empty(),
                   "null input cannot carry a forged allocation-relative offset");
        }
    }
    for (std::uint32_t mask = 0; mask != 4; ++mask) {
        for (const auto offset : {std::uint64_t{0}, std::uint64_t{1}, UINT64_MAX}) {
            ExecutionDnnRnnWeightParamsResult value{mask, {}};
            for (std::size_t i = 0; i != value.offsets.size(); ++i)
                if (mask & (1U << i)) value.offsets[i] = offset;
            const auto bytes =
                checked_codec(value, 20, write_execution_dnn_rnn_weight_params_result,
                              read_execution_dnn_rnn_weight_params_result);
            auto bad = bytes;
            set_u32(bad, 0, 4);
            malformed_convolution(value, bad, write_execution_dnn_rnn_weight_params_result,
                                  read_execution_dnn_rnn_weight_params_result);
            value.non_null = 4;
            ExecutionPayloadWriter invalid;
            expect(!write_execution_dnn_rnn_weight_params_result(invalid, value) &&
                       invalid.take().empty(),
                   "response mask is bounded without assuming request ownership");
        }
    }
    for (std::size_t slot = 0; slot != 2; ++slot) {
        ExecutionDnnRnnWeightParamsResult value;
        const auto bytes = checked_codec(value, 20, write_execution_dnn_rnn_weight_params_result,
                                         read_execution_dnn_rnn_weight_params_result);
        auto bad = bytes;
        bad[4 + 8 * slot] = std::byte{1};
        malformed_convolution(value, bad, write_execution_dnn_rnn_weight_params_result,
                              read_execution_dnn_rnn_weight_params_result);
        value.offsets[slot] = 1;
        ExecutionPayloadWriter invalid;
        expect(
            !write_execution_dnn_rnn_weight_params_result(invalid, value) && invalid.take().empty(),
            "absent response slots cannot carry noncanonical offsets");
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kDnnRnnWeightParams),
           "descriptor observation never claims replacement computation");
}
void attn_weights_codecs() {
    for (std::uint32_t kind = 0; kind != 8; ++kind) {
        for (std::uint32_t presence = 0; presence != 8; ++presence) {
            ExecutionDnnAttnWeights value{0x0807060504030201,
                                          0x1817161514131211,
                                          0x2827262524232221,
                                          kind,
                                          presence & 1U,
                                          UINT64_MAX,
                                          (presence & 2U) ? 0x3837363534333231ULL : 0,
                                          (presence >> 2) & 1U};
            const auto bytes = checked_codec(value, 52, write_execution_dnn_attn_weights,
                                             read_execution_dnn_attn_weights);
            expect(
                bytes[0] == std::byte{1} && bytes[8] == std::byte{0x11} &&
                    bytes[16] == std::byte{0x21} && bytes[24] == std::byte(kind) &&
                    bytes[28] == std::byte(value.has_base) && bytes[32] == std::byte{0xff} &&
                    bytes[40] == std::byte((presence & 2U) ? 0x31 : 0) &&
                    bytes[48] == std::byte(value.has_output),
                "attention layout fields are exactly placed without transmitting a base address");
            for (const auto size : {std::uint64_t{0}, std::uint64_t{1}}) {
                auto zero = value;
                zero.weight_bytes = size;
                (void)checked_codec(zero, 52, write_execution_dnn_attn_weights,
                                    read_execution_dnn_attn_weights);
            }
            for (const auto offset : {24U, 28U, 48U}) {
                for (const auto invalid : {offset == 24 ? 8U : 2U, UINT32_MAX}) {
                    auto bad = bytes;
                    set_u32(bad, offset, invalid);
                    malformed_convolution(value, bad, write_execution_dnn_attn_weights,
                                          read_execution_dnn_attn_weights);
                    auto malformed = value;
                    if (offset == 24)
                        malformed.kind = invalid;
                    else if (offset == 28)
                        malformed.has_base = invalid;
                    else
                        malformed.has_output = invalid;
                    ExecutionPayloadWriter output;
                    expect(!write_execution_dnn_attn_weights(output, malformed) &&
                               output.take().empty(),
                           "invalid attention kind/presence rejects before partial encoding");
                }
            }
        }
    }
    for (const auto offset : {std::uint64_t{0}, std::uint64_t{1}, UINT64_MAX}) {
        ExecutionDnnAttnWeightsResult value{1, offset};
        const auto bytes = checked_codec(value, 12, write_execution_dnn_attn_weights_result,
                                         read_execution_dnn_attn_weights_result);
        for (const auto invalid : {2U, UINT32_MAX}) {
            auto bad = bytes;
            set_u32(bad, 0, invalid);
            malformed_convolution(value, bad, write_execution_dnn_attn_weights_result,
                                  read_execution_dnn_attn_weights_result);
            auto malformed = value;
            malformed.non_null = invalid;
            ExecutionPayloadWriter output;
            expect(!write_execution_dnn_attn_weights_result(output, malformed) &&
                       output.take().empty(),
                   "response presence has no hidden address-mode bits");
        }
    }
    ExecutionDnnAttnWeightsResult absent;
    auto bad = checked_codec(absent, 12, write_execution_dnn_attn_weights_result,
                             read_execution_dnn_attn_weights_result);
    bad[4] = std::byte{1};
    malformed_convolution(absent, bad, write_execution_dnn_attn_weights_result,
                          read_execution_dnn_attn_weights_result);
    absent.offset = 1;
    ExecutionPayloadWriter invalid;
    expect(!write_execution_dnn_attn_weights_result(invalid, absent) && invalid.take().empty() &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnAttnWeights),
           "absent mapping is canonical and never grants replacement computation");
}
void attn_compute_codecs() {
    const std::array<std::vector<std::size_t>, 3> unused{
        {{7, 8, 9, 10, 12}, {5, 6, 12}, {0, 1, 5, 6, 8, 9, 10}}};
    for (std::uint32_t kind = 1; kind <= 3; ++kind) {
        ExecutionDnnAttnCompute value;
        value.kind = static_cast<ExecutionDnnAttnComputeKind>(kind);
        value.mode = kind == 3 ? 1 : 0;
        value.current = kind == 1 ? INT32_MIN : 0;
        value.window_presence = kind == 3 ? 0 : 3;
        value.handle = 0x1817161514131211;
        value.metadata = 0x2827262524232221;
        value.descriptors = {UINT64_MAX, 17, 17, 0, 19};
        for (std::size_t index = 0; index != value.buffers.size(); ++index)
            value.buffers[index] = {0x8877665544332200 + index, UINT64_MAX - index};
        for (const auto index : unused[kind - 1]) value.buffers[index] = {};
        value.weight_bytes = UINT64_MAX;
        value.workspace_bytes = 0;
        value.reserve_bytes = 0xf8f7f6f5f4f3f2f1;
        const auto empty = checked_codec(value, 344, write_execution_dnn_attn_compute,
                                         read_execution_dnn_attn_compute);
        expect(empty[0] == std::byte(kind) && empty[16] == std::byte{0x11} &&
                   empty[24] == std::byte{0x21} && empty[32] == std::byte{0xff} &&
                   empty[312] == std::byte{0xff} && empty[320] == std::byte{0} &&
                   empty[328] == std::byte{0xf1} && empty[335] == std::byte{0xf8} &&
                   empty[336] == std::byte{0} && empty[340] == std::byte{0},
               "attention preparation preserves the exact 344-byte canonical record");
        ExecutionPayloadReader fields(empty);
        std::uint32_t small{};
        std::uint64_t large{};
        expect(fields.read_u32(small) && small == kind && fields.read_u32(small) &&
                   small == value.mode && fields.read_u32(small) &&
                   small == std::bit_cast<std::uint32_t>(value.current) && fields.read_u32(small) &&
                   small == value.window_presence && fields.read_u64(large) &&
                   large == value.handle && fields.read_u64(large) && large == value.metadata,
               "attention header fields retain signed bits and independent window presence");
        for (const auto id : value.descriptors)
            expect(fields.read_u64(large) && large == id, "all five descriptor positions survive");
        for (const auto& buffer : value.buffers)
            expect(fields.read_u64(large) && large == buffer.allocation && fields.read_u64(large) &&
                       large == buffer.offset,
                   "all fifteen range positions survive without deduplication");
        const auto invalid = [&](const ExecutionDnnAttnCompute& malformed) {
            ExecutionPayloadWriter writer;
            expect(!write_execution_dnn_attn_compute(writer, malformed) && writer.take().empty(),
                   "invalid compute shape fails before partial encoding");
        };
        for (const auto bad_kind : {0U, 4U, UINT32_MAX}) {
            auto bad = empty;
            set_u32(bad, 0, bad_kind);
            malformed_convolution(value, bad, write_execution_dnn_attn_compute,
                                  read_execution_dnn_attn_compute);
            auto bad_value = value;
            bad_value.kind = static_cast<ExecutionDnnAttnComputeKind>(bad_kind);
            invalid(bad_value);
        }
        for (const auto index : unused[kind - 1]) {
            auto bad = empty;
            set_u32(bad, 72 + index * 16, 1);
            malformed_convolution(value, bad, write_execution_dnn_attn_compute,
                                  read_execution_dnn_attn_compute);
            auto bad_value = value;
            bad_value.buffers[index].allocation = 1;
            invalid(bad_value);
        }
        for (std::size_t index = 0; index != value.buffers.size(); ++index) {
            auto bad_value = value;
            bad_value.buffers[index] = {0, 1};
            invalid(bad_value);
            auto bad = empty;
            set_u32(bad, 72 + index * 16, 0);
            set_u32(bad, 76 + index * 16, 0);
            set_u32(bad, 80 + index * 16, 1);
            malformed_convolution(value, bad, write_execution_dnn_attn_compute,
                                  read_execution_dnn_attn_compute);
        }
        for (const auto mode : {0U, 1U, 2U, UINT32_MAX}) {
            auto changed = value;
            changed.mode = mode;
            if (mode <= (kind == 3 ? 1U : 0U))
                (void)checked_codec(changed, 344, write_execution_dnn_attn_compute,
                                    read_execution_dnn_attn_compute);
            else {
                invalid(changed);
                auto bad = empty;
                set_u32(bad, 4, mode);
                malformed_convolution(value, bad, write_execution_dnn_attn_compute,
                                      read_execution_dnn_attn_compute);
            }
        }
        for (const auto current : {INT32_MIN, -1, 0, 1, INT32_MAX}) {
            auto changed = value;
            changed.current = current;
            if (kind == 1 || !current)
                (void)checked_codec(changed, 344, write_execution_dnn_attn_compute,
                                    read_execution_dnn_attn_compute);
            else
                invalid(changed);
        }
        for (std::uint32_t presence = 0; presence != 5; ++presence) {
            auto changed = value;
            changed.window_presence = presence;
            if (presence > 3 || (kind == 3 && presence)) {
                invalid(changed);
                auto bad = empty;
                set_u32(bad, 12, presence);
                malformed_convolution(value, bad, write_execution_dnn_attn_compute,
                                      read_execution_dnn_attn_compute);
                continue;
            }
            (void)checked_codec(changed, 344, write_execution_dnn_attn_compute,
                                read_execution_dnn_attn_compute);
            if (presence & 1) changed.lower = {INT32_MIN, -1, 0, INT32_MAX};
            if (presence & 2) changed.upper = {INT32_MAX, 1, 0, INT32_MIN};
            const auto bytes =
                checked_codec(changed, 344 + 4 * (changed.lower.size() + changed.upper.size()),
                              write_execution_dnn_attn_compute, read_execution_dnn_attn_compute);
            for (const auto offset :
                 {336U, static_cast<unsigned int>(340 + 4 * changed.lower.size())})
                for (const auto length :
                     {1U, 3U, UINT32_MAX,
                      4U * static_cast<unsigned int>(kExecutionDnnMaxAttentionWindows + 1)}) {
                    auto bad = bytes;
                    set_u32(bad, offset, length);
                    malformed_convolution(changed, bad, write_execution_dnn_attn_compute,
                                          read_execution_dnn_attn_compute);
                }
            if (presence == 3) {
                changed.upper.pop_back();
                invalid(changed);
            } else {
                if (!(presence & 1))
                    changed.lower.push_back(7);
                else
                    changed.upper.push_back(8);
                invalid(changed);
            }
        }
        value.descriptors.fill(0);
        value.buffers.fill({});
        (void)checked_codec(value, 344, write_execution_dnn_attn_compute,
                            read_execution_dnn_attn_compute);
    }
    ExecutionDnnAttnCompute maximum;
    maximum.kind = ExecutionDnnAttnComputeKind::kForward;
    maximum.window_presence = 3;
    maximum.lower.assign(kExecutionDnnMaxAttentionWindows, INT32_MIN);
    maximum.upper.assign(kExecutionDnnMaxAttentionWindows, INT32_MAX);
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_attn_compute(writer, maximum), "maximum bounded snapshots encode");
    const auto bytes = writer.take();
    ExecutionPayloadReader reader(bytes);
    ExecutionDnnAttnCompute decoded;
    expect(bytes.size() == 344 + 8 * kExecutionDnnMaxAttentionWindows &&
               read_execution_dnn_attn_compute(reader, decoded) && reader.empty() &&
               decoded.lower == maximum.lower && decoded.upper == maximum.upper,
           "maximum snapshots fit one frame and retain all signed values");
    maximum.lower.push_back(0);
    maximum.upper.push_back(0);
    ExecutionPayloadWriter oversized;
    expect(!write_execution_dnn_attn_compute(oversized, maximum) && oversized.take().empty(),
           "snapshot quota fails before fields are written");
    for (const ExecutionDnnAttnWindowRange range :
         {ExecutionDnnAttnWindowRange{}, {0, 1}, {0, 1U << 20}, {(1U << 20) - 1, 1}})
        (void)checked_codec(range, 8, write_execution_dnn_attn_window_range,
                            read_execution_dnn_attn_window_range);
    for (const ExecutionDnnAttnWindowRange range : {ExecutionDnnAttnWindowRange{1, 0},
                                                    {1U << 20, 1},
                                                    {UINT32_MAX, 1},
                                                    {1, UINT32_MAX},
                                                    {0, UINT32_MAX}}) {
        ExecutionPayloadWriter invalid;
        expect(!write_execution_dnn_attn_window_range(invalid, range) && invalid.take().empty(),
               "range rejects zero-count nonzero-first, overflow and oversized prefix");
        ExecutionPayloadWriter raw;
        expect(raw.write_u32(range.first) && raw.write_u32(range.count), "encode hostile range");
        malformed_convolution(ExecutionDnnAttnWindowRange{}, raw.take(),
                              write_execution_dnn_attn_window_range,
                              read_execution_dnn_attn_window_range);
    }
    expect(!execution_allows_compute_replacement(ExecutionOpcode::kPrepareDnnAttn) &&
               !execution_allows_compute_replacement(ExecutionOpcode::kDnnAttnCompute),
           "attention original compute/preparation never claim persistent replacement");
}
}  // namespace

int main() {
    try {
        tensor_codecs();
        activation_codecs();
        compute_codecs();
        composition_and_constants();
        convolution_codecs();
        convolution_find_codec();
        pooling_codecs();
        tensor_forward_codecs();
        lrn_codecs();
        reduction_codecs();
        batch_norm_codecs();
        dropout_codecs();
        op_tensor_codecs();
        reorder_filter_codecs();
        library_info_codecs();
        backward_convolution_query_codecs();
        folded_convolution_codecs();
        sequence_metadata_codecs();
        rnn_attention_codecs();
        rnn_query_clip_codecs();
        rnn_compute_codecs();
        rnn_weight_params_codecs();
        attn_weights_codecs();
        attn_compute_codecs();
        std::cout << "CUDA-free DNN wire codec checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
