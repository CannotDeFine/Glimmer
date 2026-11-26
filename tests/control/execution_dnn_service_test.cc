#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_fixture.h"

#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using glimmer::core::ExecutionSchedulingMode;
namespace dnn = glimmer::tests::dnn;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Native {
    char context_token{};
    CUcontext current{};
    unsigned int initializations{}, context_creates{}, context_sets{}, context_destroys{};
    unsigned int identity_queries{}, unexpected_calls{};
    unsigned int library_queries{}, property_queries{};
    std::size_t version{0x100000000ULL + 91002};
    std::array<unsigned int, 3> backward_queries{};
    unsigned int folded_calls{}, folded_preflights{}, sequence_calls{};
    unsigned int rnn_attention_creates{}, rnn_attention_destroys{}, rnn_attention_sets{};
    unsigned int rnn_data_creates{}, rnn_data_destroys{}, rnn_queries{}, rnn_preflights{},
        rnn_clips{}, rnn_computations{}, rnn_builds{}, rnn_weight_params{}, attn_weights{};
    unsigned int seq_creates{}, seq_destroys{}, attn_computations{}, attn_preflights{};
} native;

CUcontext current_context() {
    return native.current;
}
CUresult unexpected() {
    ++native.unexpected_calls;
    return CUDA_ERROR_NOT_SUPPORTED;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result;
    result.initialize = [](unsigned int) {
        ++native.initializations;
        return CUDA_SUCCESS;
    };
    result.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    result.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(!native.current, "regular session creates its own empty-current native context");
        ++native.context_creates;
        *output = native.current = reinterpret_cast<CUcontext>(&native.context_token);
        return CUDA_SUCCESS;
    };
    result.context_set_current = [](CUcontext context) {
        expect(!context || context == reinterpret_cast<CUcontext>(&native.context_token),
               "activate only the fake owned context");
        ++native.context_sets;
        native.current = context;
        return CUDA_SUCCESS;
    };
    result.context_destroy = [](CUcontext context) {
        expect(context && context == native.current, "destroy only the current owned context");
        ++native.context_destroys;
        native.current = nullptr;
        return CUDA_SUCCESS;
    };
    result.execution.context_get_current = [](CUcontext* output) {
        *output = native.current;
        return CUDA_SUCCESS;
    };
    result.module_load = [](CUmodule*, const void*) { return unexpected(); };
    result.module_unload = [](CUmodule) { return unexpected(); };
    result.function_get = [](CUfunction*, CUmodule, const char*) { return unexpected(); };
    result.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return unexpected();
    };
    result.allocate = [](CUdeviceptr*, std::size_t) { return unexpected(); };
    result.release = [](CUdeviceptr) { return unexpected(); };
    result.upload = [](CUdeviceptr, const void*, std::size_t) { return unexpected(); };
    result.download = [](void*, CUdeviceptr, std::size_t) { return unexpected(); };
    result.stream_create = [](CUstream*, unsigned int) { return unexpected(); };
    result.stream_destroy = [](CUstream) { return unexpected(); };
    result.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**) { return unexpected(); };
    result.execution.event_create = [](CUevent*, unsigned int) { return unexpected(); };
    result.execution.event_record = [](CUevent, CUstream) { return unexpected(); };
    result.execution.event_query = [](CUevent) { return unexpected(); };
    result.execution.event_destroy = [](CUevent) { return unexpected(); };
    result.execution.stream_synchronize = [](CUstream) { return unexpected(); };
    result.device_uuid = [](CUuuid* output, CUdevice) {
        ++native.identity_queries;
        *output = {};
        output->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    result.driver_version = [](int* output) {
        ++native.identity_queries;
        *output = 12060;
        return CUDA_SUCCESS;
    };
    auto sdk = dnn::dispatch();
    sdk.version = [] {
        dnn::owner();
        ++native.library_queries;
        return native.version;
    };
    sdk.cudart_version = [] {
        dnn::owner();
        ++native.library_queries;
        return std::numeric_limits<std::size_t>::max();
    };
    sdk.max_device_version = [] {
        dnn::owner();
        ++native.library_queries;
        return std::size_t{0};
    };
    sdk.property = [](libraryPropertyType property, int* value) {
        dnn::owner();
        ++native.property_queries;
        if (value) *value = INT_MIN;
        return property <= PATCH_LEVEL && value ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
    };
    sdk.convolution_backward_data_max_count = [](cudnnHandle_t handle, int* output) {
        dnn::owner();
        expect(handle && output && native.current, "owned data count output and context");
        ++native.backward_queries[0];
        *output = 65;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_backward_filter_max_count = [](cudnnHandle_t handle, int* output) {
        dnn::owner();
        expect(handle && output && native.current, "owned filter count output and context");
        ++native.backward_queries[1];
        *output = INT_MAX;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.convolution_backward_data_workspace =
        [](cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
           cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t,
           std::size_t*) {
            ++native.backward_queries[2];
            return CUDNN_STATUS_BAD_PARAM;
        };
    sdk.convolution_get_nd = [](CudaDnnConvolution, int, int*, int*, int*, int*,
                                cudnnConvolutionMode_t*, cudnnDataType_t*) {
        dnn::owner();
        ++native.folded_preflights;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.folded_backward_data = [](cudnnHandle_t, CudaDnnFilterHandle, cudnnTensorDescriptor_t,
                                  CudaDnnConvolution, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
                                  CudaDnnFilterHandle, cudnnTensorDescriptor_t, CudaDnnConvolution,
                                  cudnnTensorDescriptor_t, CudaDnnTensorTransform,
                                  CudaDnnTensorTransform, CudaDnnTensorTransform,
                                  CudaDnnTensorTransform) {
        dnn::owner();
        ++native.folded_calls;
        return CUDNN_STATUS_BAD_PARAM;
    };
    // Present callbacks make wrong-role/kind rejection observable rather than
    // accidentally succeeding because this fixture lacks the capability.
    sdk.rnn_data_create = [](CudaDnnRnnData* output) {
        dnn::owner();
        *output = reinterpret_cast<CudaDnnRnnData>(new char{});
        ++native.rnn_data_creates;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_data_destroy = [](CudaDnnRnnData descriptor) {
        dnn::owner();
        delete reinterpret_cast<char*>(descriptor);
        ++native.rnn_data_destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_data_set = [](CudaDnnRnnData, cudnnDataType_t, cudnnRNNDataLayout_t, int, int, int,
                          const int*, void*) {
        ++native.sequence_calls;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.seq_data_create = [](CudaDnnSeqData* output) {
        dnn::owner();
        *output = reinterpret_cast<CudaDnnSeqData>(new char{});
        ++native.seq_creates;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.seq_data_destroy = [](CudaDnnSeqData descriptor) {
        dnn::owner();
        delete reinterpret_cast<char*>(descriptor);
        ++native.seq_destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.seq_data_set = [](CudaDnnSeqData, cudnnDataType_t, int, const int*,
                          const cudnnSeqDataAxis_t*, std::size_t, const int*, void*) {
        ++native.sequence_calls;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.seq_data_get = [](CudaDnnSeqData, cudnnDataType_t*, int*, int, int*, cudnnSeqDataAxis_t*,
                          std::size_t*, std::size_t, int*, void*) {
        ++native.sequence_calls;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_create = [](CudaDnnRnn* output) {
        dnn::owner();
        *output = reinterpret_cast<CudaDnnRnn>(new char{});
        ++native.rnn_attention_creates;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_destroy = [](CudaDnnRnn descriptor) {
        dnn::owner();
        delete reinterpret_cast<char*>(descriptor);
        ++native.rnn_attention_destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_set_v8 = [](CudaDnnRnn, cudnnRNNAlgo_t, cudnnRNNMode_t, cudnnRNNBiasMode_t,
                        cudnnDirectionMode_t, cudnnRNNInputMode_t, cudnnDataType_t, cudnnDataType_t,
                        cudnnMathType_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                        CudaDnnDropout, std::uint32_t) {
        ++native.rnn_attention_sets;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_create = [](CudaDnnAttn* output) {
        dnn::owner();
        *output = reinterpret_cast<CudaDnnAttn>(new char{});
        ++native.rnn_attention_creates;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.attn_destroy = [](CudaDnnAttn descriptor) {
        dnn::owner();
        delete reinterpret_cast<char*>(descriptor);
        ++native.rnn_attention_destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.attn_set = [](CudaDnnAttn, unsigned int, int, double, cudnnDataType_t, cudnnDataType_t,
                      cudnnMathType_t, CudaDnnDropout, CudaDnnDropout, int, int, int, int, int, int,
                      int, int, int, int, int) {
        ++native.rnn_attention_sets;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_weight_space_size = [](cudnnHandle_t, CudaDnnRnn, std::size_t*) {
        ++native.rnn_queries;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_temp_space_sizes = [](cudnnHandle_t, CudaDnnRnn, cudnnForwardMode_t, CudaDnnRnnData,
                                  std::size_t*, std::size_t*) {
        ++native.rnn_queries;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_buffers = [](cudnnHandle_t, CudaDnnAttn, std::size_t*, std::size_t*, std::size_t*) {
        ++native.rnn_queries;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_set_clip_v8 = [](CudaDnnRnn, cudnnRNNClipMode_t, cudnnNanPropagation_t, double,
                             double) {
        ++native.rnn_clips;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_data_get = [](CudaDnnRnnData, cudnnDataType_t*, cudnnRNNDataLayout_t*, int*, int*, int*,
                          int, int*, void*) {
        ++native.rnn_preflights;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_forward = [](cudnnHandle_t, CudaDnnRnn, cudnnForwardMode_t, const std::int32_t*,
                         CudaDnnRnnData, const void*, CudaDnnRnnData, void*,
                         cudnnTensorDescriptor_t, const void*, void*, cudnnTensorDescriptor_t,
                         const void*, void*, std::size_t, const void*, std::size_t, void*,
                         std::size_t, void*) {
        ++native.rnn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_backward_data = [](cudnnHandle_t, CudaDnnRnn, const std::int32_t*, CudaDnnRnnData,
                               const void*, const void*, CudaDnnRnnData, void*,
                               cudnnTensorDescriptor_t, const void*, const void*, void*,
                               cudnnTensorDescriptor_t, const void*, const void*, void*,
                               std::size_t, const void*, std::size_t, void*, std::size_t, void*) {
        ++native.rnn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_backward_weights = [](cudnnHandle_t, CudaDnnRnn, cudnnWgradMode_t, const std::int32_t*,
                                  CudaDnnRnnData, const void*, cudnnTensorDescriptor_t, const void*,
                                  CudaDnnRnnData, const void*, std::size_t, void*, std::size_t,
                                  void*, std::size_t, void*) {
        ++native.rnn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_build_dynamic = [](cudnnHandle_t, CudaDnnRnn, int) {
        ++native.rnn_builds;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_weight_params = [](cudnnHandle_t, CudaDnnRnn, int, std::size_t, const void*, int,
                               cudnnTensorDescriptor_t, void**, cudnnTensorDescriptor_t, void**) {
        ++native.rnn_weight_params;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_weights = [](cudnnHandle_t, CudaDnnAttn, cudnnMultiHeadAttnWeightKind_t, std::size_t,
                          const void*, cudnnTensorDescriptor_t, void**) {
        ++native.attn_weights;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_forward = [](cudnnHandle_t, CudaDnnAttn, int, const int*, const int*, const int*,
                          const int*, CudaDnnSeqData, const void*, const void*, CudaDnnSeqData,
                          const void*, CudaDnnSeqData, const void*, CudaDnnSeqData, void*,
                          std::size_t, const void*, std::size_t, void*, std::size_t, void*) {
        ++native.attn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_backward_data = [](cudnnHandle_t, CudaDnnAttn, const int*, const int*, const int*,
                                const int*, CudaDnnSeqData, const void*, CudaDnnSeqData, void*,
                                const void*, CudaDnnSeqData, void*, const void*, CudaDnnSeqData,
                                void*, const void*, std::size_t, const void*, std::size_t, void*,
                                std::size_t, void*) {
        ++native.attn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_backward_weights = [](cudnnHandle_t, CudaDnnAttn, cudnnWgradMode_t, CudaDnnSeqData,
                                   const void*, CudaDnnSeqData, const void*, CudaDnnSeqData,
                                   const void*, CudaDnnSeqData, const void*, std::size_t,
                                   const void*, void*, std::size_t, void*, std::size_t, void*) {
        ++native.attn_computations;
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.attn_get = [](CudaDnnAttn, unsigned int*, int*, double*, cudnnDataType_t*, cudnnDataType_t*,
                      cudnnMathType_t*, CudaDnnDropout*, CudaDnnDropout*, int*, int*, int*, int*,
                      int*, int*, int*, int*, int*, int*, int*) {
        ++native.attn_preflights;
        return CUDNN_STATUS_BAD_PARAM;
    };
    result.dnn = std::make_shared<const CudaDnnDispatch>(sdk);
    result.verified_dnn_header_version = CUDNN_VERSION;
    return result;
}

ExecutionWireReply complete(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kComplete && !reply->status,
           "actual service operation must complete successfully");
    return std::move(*reply);
}
ExecutionWireReply sdk_success(std::optional<ExecutionWireReply> reply) {
    auto result = complete(std::move(reply));
    expect(result.original_status == 0, "original SDK success must retain its status domain");
    return result;
}
void rejected(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected && reply->status &&
               !reply->original_status && !reply->id,
           "authority rejection must occur before any claimed native SDK operation");
}
std::vector<std::byte> object_id(std::uint64_t id) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id), "encode one owned object identity");
    return writer.take();
}
std::vector<std::byte> role(std::uint32_t value, std::uint32_t flags = 0) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(value) && writer.write_u32(flags), "encode role bootstrap");
    return writer.take();
}
void bind_dnn_profile(ExecutionSocketClient& client) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_profile(writer, CUDNN_VERSION), "encode selected DNN profile");
    const auto reply = complete(client.exchange(ExecutionOpcode::kBindDnnProfile, writer.take()));
    ExecutionPayloadReader reader(reply.data);
    std::uint32_t version{};
    expect(!reply.original_status && !reply.id && !reply.address && !reply.parameter_bytes &&
               reply.arguments.empty() && read_execution_dnn_profile(reader, version) &&
               version == CUDNN_VERSION && reader.empty(),
           "profile binding echoes exact headers without SDK provenance or object authority");
}
void open_metadata(ExecutionSocketClient& client) {
    const auto reply = complete(client.open(role(kExecutionDnnMetadataRole)));
    expect(!reply.original_status && decode_execution_bootstrap_ownership(reply.data) ==
                                         ExecutionBootstrapOwnership::kMetadataRetained,
           "DNM1 Hello retains CPU ownership without claiming CUDA initialization");
    bind_dnn_profile(client);
}
std::uint64_t create_descriptor(ExecutionSocketClient& client, ExecutionDnnObjectKind kind) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(static_cast<std::uint32_t>(kind)), "encode descriptor kind");
    const auto reply =
        sdk_success(client.exchange(ExecutionOpcode::kCreateDnnMetadata, writer.take()));
    expect(reply.id && !reply.parameter_bytes, "created descriptor has one nonquarantined ID");
    return reply.id;
}
ExecutionDnnCompute problem(std::uint64_t metadata) {
    ExecutionDnnCompute result;
    result.kind = ExecutionDnnComputeKind::kAddTensor;
    result.handle = 1;
    result.metadata = metadata;
    result.tensors = {1, 2};
    result.buffers = {{{1, 0}, {2, 0}}};
    result.scalar_bytes = 4;
    return result;
}
std::vector<std::byte> computation(const ExecutionDnnCompute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_compute(writer, value), "encode structurally valid DNN computation");
    return writer.take();
}
ExecutionDnnTensorForward tensor_forward(std::uint64_t metadata) {
    ExecutionDnnTensorForward result;
    result.kind = ExecutionDnnTensorForwardKind::kSoftmaxForward;
    result.handle = 1;
    result.metadata = metadata;
    result.tensors = {1, 2};
    result.buffers = {{{1, 0}, {2, 0}}};
    result.scalar_bytes = 4;
    return result;
}
std::vector<std::byte> computation(const ExecutionDnnTensorForward& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_tensor_forward(writer, value),
           "encode structurally valid pooling/softmax computation");
    return writer.take();
}
ExecutionDnnReductionCompute reduction(std::uint64_t metadata) {
    ExecutionDnnReductionCompute value;
    value.handle = 1;
    value.metadata = metadata;
    value.descriptors = {1, 2, 3};
    value.buffers = {{{1, 0}, {2, 0}, {}, {}}};
    value.scalar_bytes = 4;
    return value;
}
std::vector<std::byte> computation(const ExecutionDnnReductionCompute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_reduction_compute(writer, value), "encode reduction computation");
    return writer.take();
}
std::vector<std::byte> reduction_query(const ExecutionDnnReductionCompute& value,
                                       ExecutionDnnReductionQueryKind kind) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_reduction_query(
               writer, {kind, value.handle, value.metadata, value.descriptors}),
           "encode reduction size query");
    return writer.take();
}
ExecutionDnnBatchNormCompute batch_norm(std::uint64_t metadata) {
    ExecutionDnnBatchNormCompute value;
    value.query.kind = ExecutionDnnBatchNormKind::kForward;
    value.query.handle = 1;
    value.query.metadata = metadata;
    value.query.descriptors = {1, 1, 0, 0, 0, 1, 0};
    for (const auto index : {0U, 1U, 5U, 6U}) value.buffers[index] = {1, 0};
    value.scalar_bytes = 4;
    return value;
}
std::vector<std::byte> computation(const ExecutionDnnBatchNormCompute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_batch_norm_compute(writer, value), "encode BN Ex computation");
    return writer.take();
}
std::vector<std::byte> batch_norm_query(const ExecutionDnnBatchNormCompute& value,
                                        ExecutionDnnBatchNormKind kind) {
    auto query = value.query;
    query.kind = kind;
    if (kind == ExecutionDnnBatchNormKind::kReserve)
        for (std::size_t index = 1; index != 6; ++index) query.descriptors[index] = 0;
    else if (kind == ExecutionDnnBatchNormKind::kBackward)
        query.descriptors[2] = query.descriptors[4] = query.descriptors[0];
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_batch_norm_query(writer, query), "encode BN Ex size query");
    return writer.take();
}
std::vector<std::byte> computation(const ExecutionDnnDropout& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_dropout(writer, value), "encode queued dropout configuration");
    return writer.take();
}
std::vector<std::byte> computation(const ExecutionDnnReorderFilterAndBias& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_reorder_filter_and_bias(writer, value),
           "encode queued filter/bias reorder");
    return writer.take();
}
std::vector<std::byte> library_info(ExecutionDnnLibraryInfoKind kind, std::uint32_t property = 0,
                                    std::uint32_t has_output = 0) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_library_info(writer, {kind, property, has_output}),
           "encode original library information query");
    return writer.take();
}
std::vector<std::byte> backward_query(const ExecutionDnnBackwardConvolutionQuery& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backward_convolution_query(writer, value),
           "encode original backward-convolution query");
    return writer.take();
}
std::vector<std::byte> folded_request(const ExecutionDnnFoldedConvBackwardDataDescriptors& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_folded_conv_backward_data_descriptors(writer, value),
           "encode original twelve-descriptor Folded request");
    auto bytes = writer.take();
    expect(bytes.size() == 116, "Folded authority travels in exactly 116 bytes");
    return bytes;
}
void reject_folded(ExecutionSocketClient& client, std::span<const std::byte> bytes,
                   CUresult status) {
    const auto reply =
        client.exchange(ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors, bytes);
    rejected(reply);
    expect(reply->status == status && !reply->address && reply->data.empty() &&
               !reply->parameter_bytes,
           "clean Folded rejection preserves exact Driver status and empty nonfaulted receipt");
}
void query_library_info(ExecutionSocketClient& client) {
    constexpr std::array kinds{ExecutionDnnLibraryInfoKind::kVersion,
                               ExecutionDnnLibraryInfoKind::kCudartVersion,
                               ExecutionDnnLibraryInfoKind::kMaxDeviceVersion};
    constexpr std::array<std::uint64_t, 3> values{0x100000000ULL + 91002,
                                                  std::numeric_limits<std::size_t>::max(), 0};
    for (std::size_t index = 0; index != kinds.size(); ++index) {
        const auto reply =
            complete(client.exchange(ExecutionOpcode::kDnnLibraryInfo, library_info(kinds[index])));
        expect(!reply.original_status && !reply.id && !reply.parameter_bytes &&
                   reply.data.empty() && reply.address == values[index],
               "original scalar result is full-width and has no fabricated SDK status");
    }
    const auto property =
        sdk_success(client.exchange(ExecutionOpcode::kDnnLibraryInfo,
                                    library_info(ExecutionDnnLibraryInfoKind::kProperty, 0, 1)));
    expect(property.address == 0x80000000ULL && !property.id && !property.parameter_bytes &&
               property.data.empty(),
           "original signed property output uses zero-extended raw i32 bits");
    for (const auto query :
         {ExecutionDnnLibraryInfo{ExecutionDnnLibraryInfoKind::kProperty, 3, 1},
          ExecutionDnnLibraryInfo{ExecutionDnnLibraryInfoKind::kProperty, 0, 0}}) {
        const auto reply =
            client.exchange(ExecutionOpcode::kDnnLibraryInfo,
                            library_info(query.kind, query.property, query.has_output));
        expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                   reply->original_status == CUDNN_STATUS_BAD_PARAM &&
                   reply->status == CUDNN_STATUS_BAD_PARAM && !reply->address &&
                   !reply->parameter_bytes && reply->data.empty(),
               "original error preserves status but suppresses touched temporary output");
    }
    for (const auto fields :
         {std::array<std::uint32_t, 3>{0, 0, 0}, std::array<std::uint32_t, 3>{5, 0, 0},
          std::array<std::uint32_t, 3>{1, 1, 0}, std::array<std::uint32_t, 3>{2, 0, 1},
          std::array<std::uint32_t, 3>{4, 0, 2}, std::array<std::uint32_t, 3>{4, 4, 1},
          std::array<std::uint32_t, 3>{4, UINT32_MAX, 1}}) {
        ExecutionPayloadWriter writer;
        for (auto field : fields) expect(writer.write_u32(field), "encode malformed raw query");
        rejected(client.exchange(ExecutionOpcode::kDnnLibraryInfo, writer.take()));
    }
    auto truncated = library_info(ExecutionDnnLibraryInfoKind::kVersion);
    truncated.pop_back();
    rejected(client.exchange(ExecutionOpcode::kDnnLibraryInfo, truncated));
    auto extra = library_info(ExecutionDnnLibraryInfoKind::kVersion);
    extra.push_back(std::byte{0});
    rejected(client.exchange(ExecutionOpcode::kDnnLibraryInfo, extra));
}
void reject_operation_descriptor_requests(ExecutionSocketClient& client, std::uint64_t tensor) {
    for (const auto wrong : {tensor, UINT64_MAX}) {
        ExecutionPayloadWriter setter;
        expect(setter.write_u64(wrong) && write_execution_dnn_op_tensor(setter, {}),
               "encode OpTensor set on unknown or wrong-kind authority");
        rejected(client.exchange(ExecutionOpcode::kSetDnnOpTensor, setter.take()));
        ExecutionPayloadWriter getter;
        expect(getter.write_u64(wrong) && getter.write_u32(7),
               "encode OpTensor get on unknown or wrong-kind authority");
        rejected(client.exchange(ExecutionOpcode::kGetDnnOpTensor, getter.take()));
        ExecutionPayloadWriter transform;
        expect(transform.write_u64(wrong) && write_execution_dnn_tensor_transform(transform, {}),
               "encode TensorTransform set on unknown or wrong-kind authority");
        rejected(client.exchange(ExecutionOpcode::kSetDnnTensorTransform, transform.take()));
        for (const auto has_size : {0U, 1U}) {
            ExecutionPayloadWriter init;
            expect(init.write_u64(wrong) && init.write_u64(tensor) && init.write_u64(tensor) &&
                       init.write_u32(has_size),
                   "encode destination initialization without transform authority");
            rejected(client.exchange(ExecutionOpcode::kDnnInitTransformDest, init.take()));
        }
    }
    ExecutionPayloadWriter invalid_mask;
    expect(invalid_mask.write_u64(tensor) && invalid_mask.write_u32(8),
           "encode invalid OpTensor output mask");
    rejected(client.exchange(ExecutionOpcode::kGetDnnOpTensor, invalid_mask.take()));
    ExecutionPayloadWriter invalid_flag;
    expect(invalid_flag.write_u64(tensor) && invalid_flag.write_u64(tensor) &&
               invalid_flag.write_u64(tensor) && invalid_flag.write_u32(2),
           "encode invalid destination byte-output presence");
    rejected(client.exchange(ExecutionOpcode::kDnnInitTransformDest, invalid_flag.take()));
}
void reject_sequence_descriptor_requests(ExecutionSocketClient& client, std::uint64_t tensor) {
    for (const auto id : {tensor, UINT64_MAX}) {
        for (const auto operation :
             {ExecutionOpcode::kSetDnnRnnData, ExecutionOpcode::kSetDnnSeqData,
              ExecutionOpcode::kGetDnnSeqData}) {
            ExecutionPayloadWriter writer;
            expect(writer.write_u64(id), "encode unknown or wrong-kind sequence owner");
            if (operation == ExecutionOpcode::kSetDnnRnnData)
                expect(write_execution_dnn_rnn_data(writer, {}), "encode RNNData payload");
            else if (operation == ExecutionOpcode::kSetDnnSeqData)
                expect(write_execution_dnn_seq_data(writer, {}), "encode SeqData payload");
            else
                expect(write_execution_dnn_seq_data_query(writer, {INT_MAX, UINT64_MAX, 127}),
                       "encode raw full-width sequence query");
            auto bytes = writer.take();
            rejected(client.exchange(operation, bytes));
            bytes.push_back(std::byte{1});
            rejected(client.exchange(operation, bytes));
            bytes.resize(bytes.size() - 2);
            rejected(client.exchange(operation, bytes));
        }
    }
    ExecutionPayloadWriter mask;
    expect(mask.write_u64(tensor) && mask.write_u32(4) && mask.write_u64(UINT64_MAX) &&
               mask.write_u32(128),
           "encode invalid sequence output mask without semantic codec");
    rejected(client.exchange(ExecutionOpcode::kGetDnnSeqData, mask.take()));
    for (const auto blob_bytes : {UINT32_MAX, std::uint32_t{(1U << 20) * 4 + 4}}) {
        ExecutionPayloadWriter truncated;
        expect(truncated.write_u64(tensor), "encode sequence descriptor identity");
        // RNN fixed prefix, then a claimed lengths blob with no backing bytes.
        for (const auto value : {0U, 0U, 1U, 1U, 1U, 1U, blob_bytes})
            expect(truncated.write_u32(value), "encode truncated sequence blob prefix");
        rejected(client.exchange(ExecutionOpcode::kSetDnnRnnData, truncated.take()));
    }
}
void reject_rnn_attention_requests(ExecutionSocketClient& client, std::uint64_t tensor,
                                   bool is_metadata) {
    for (const auto operation : {ExecutionOpcode::kSetDnnRnn, ExecutionOpcode::kSetDnnAttn}) {
        const bool is_rnn = operation == ExecutionOpcode::kSetDnnRnn;
        const auto descriptor =
            is_metadata ? create_descriptor(client, is_rnn ? ExecutionDnnObjectKind::kRnn
                                                           : ExecutionDnnObjectKind::kAttn)
                        : tensor;
        const auto encode = [&](std::uint64_t id, std::uint64_t dropout, bool bad_enum) {
            ExecutionPayloadWriter writer;
            expect(writer.write_u64(id), "encode RNN/attention identity");
            if (is_rnn) {
                ExecutionDnnRnnState state;
                state.algorithm = bad_enum ? 8U : 0U;
                state.dropout = dropout;
                expect(write_execution_dnn_rnn(writer, state), "encode RNN scalar payload");
            } else {
                ExecutionDnnAttnState state;
                state.type = bad_enum ? 32U : 0U;
                state.dropouts = {dropout, dropout};
                expect(write_execution_dnn_attn(writer, state), "encode attention scalar payload");
            }
            return writer.take();
        };
        for (const auto wrong : {tensor, UINT64_MAX}) {
            rejected(client.exchange(operation, encode(wrong, 0, false)));
            rejected(client.exchange(operation, encode(descriptor, wrong, false)));
        }
        rejected(client.exchange(operation, encode(descriptor, 0, true)));
        // Use actual new-kind authority for format failures: a wrong parent
        // kind must not accidentally supply the only reason for rejection.
        auto bytes = encode(descriptor, 0, false);
        bytes.push_back(std::byte{1});
        rejected(client.exchange(operation, bytes));
        bytes.resize(bytes.size() - 2);
        rejected(client.exchange(operation, bytes));
        if (is_metadata)
            sdk_success(
                client.exchange(ExecutionOpcode::kDestroyDnnMetadata, object_id(descriptor)));
        else
            rejected(client.exchange(operation, encode(descriptor, 0, false)));
    }
}
std::vector<std::byte> rnn_query(const ExecutionDnnRnnQuery& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_rnn_query(writer, value), "encode RNN query authority");
    auto bytes = writer.take();
    expect(bytes.size() == 44, "RNN query has exact scalar wire extent");
    return bytes;
}
std::array<ExecutionDnnRnnQuery, 3> reject_rnn_queries(ExecutionSocketClient& client,
                                                       ExecutionSocketClient& metadata,
                                                       std::uint64_t handle, std::uint64_t tensor,
                                                       bool is_metadata) {
    const auto rnn = create_descriptor(metadata, ExecutionDnnObjectKind::kRnn);
    const auto attn = create_descriptor(metadata, ExecutionDnnObjectKind::kAttn);
    const auto input = create_descriptor(metadata, ExecutionDnnObjectKind::kRnnData);
    std::array<ExecutionDnnRnnQuery, 3> queries{
        {{ExecutionDnnRnnQueryKind::kWeightSpace, handle, metadata.session(), rnn, 0, 0, 1},
         {ExecutionDnnRnnQueryKind::kTempSpace, handle, metadata.session(), rnn, input, 0, 6},
         {ExecutionDnnRnnQueryKind::kAttentionBuffers, handle, metadata.session(), attn, 0, 0, 7}}};
    for (const auto& query : queries) {
        if (is_metadata) {
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(query)));
            continue;
        }
        // These requests otherwise have real correctly typed metadata and handles.
        // Malformed wire must fail independently of capability or identity checks.
        auto bytes = rnn_query(query);
        bytes.push_back(std::byte{1});
        rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, bytes));
        bytes.resize(bytes.size() - 2);
        rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, bytes));
        for (const auto offset : {0U, 40U}) {
            bytes = rnn_query(query);
            bytes[offset] = std::byte{8};
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, bytes));
        }
        for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
            auto bad = query;
            bad.metadata = owner;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(bad)));
        }
        for (const auto descriptor : {tensor, UINT64_MAX}) {
            auto bad = query;
            bad.descriptor = descriptor;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(bad)));
        }
        auto bad = query;
        bad.handle = UINT64_MAX;
        rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(bad)));
        if (query.kind == ExecutionDnnRnnQueryKind::kTempSpace) {
            bad = query;
            bad.input = rnn;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(bad)));
            bad = query;
            bad.mode = 2;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(bad)));
        } else {
            for (const auto offset : {28U, 36U}) {
                bytes = rnn_query(query);
                bytes[offset] = std::byte{1};
                rejected(client.exchange(ExecutionOpcode::kDnnRnnQuery, bytes));
            }
        }
    }
    const auto clip = [](std::uint64_t id, std::uint32_t mode, std::uint32_t nan) {
        ExecutionPayloadWriter writer;
        expect(writer.write_u64(id) && write_execution_dnn_rnn_clip(writer, {mode, nan, 0, 0}),
               "encode RNN clip mutation");
        return writer.take();
    };
    for (const auto id : {tensor, UINT64_MAX})
        rejected(metadata.exchange(ExecutionOpcode::kSetDnnRnnClip, clip(id, 0, 0)));
    for (const auto offset : {8U, 12U}) {
        auto bytes = clip(rnn, 0, 0);
        bytes[offset] = std::byte{2};
        rejected(metadata.exchange(ExecutionOpcode::kSetDnnRnnClip, bytes));
    }
    auto bytes = clip(rnn, 0, 0);
    bytes.push_back(std::byte{1});
    rejected(metadata.exchange(ExecutionOpcode::kSetDnnRnnClip, bytes));
    bytes.resize(bytes.size() - 2);
    rejected(metadata.exchange(ExecutionOpcode::kSetDnnRnnClip, bytes));
    if (!is_metadata) rejected(client.exchange(ExecutionOpcode::kSetDnnRnnClip, clip(rnn, 0, 0)));
    return queries;
}
std::vector<std::byte> rnn_computation(const ExecutionDnnRnnCompute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_rnn_compute(writer, value), "encode RNN compute authority");
    auto bytes = writer.take();
    expect(bytes.size() == 344, "RNN computation has exact canonical extent");
    return bytes;
}
std::vector<std::byte> rnn_build(const ExecutionDnnBuildRnnDynamic& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_build_rnn_dynamic(writer, value), "encode dynamic RNN authority");
    auto bytes = writer.take();
    expect(bytes.size() == 28, "dynamic RNN request has exact canonical extent");
    return bytes;
}
std::vector<std::byte> rnn_weight_params(const ExecutionDnnRnnWeightParams& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_rnn_weight_params(writer, value), "encode RNN weight authority");
    auto bytes = writer.take();
    expect(bytes.size() == 76, "RNN weight request has exact canonical extent");
    return bytes;
}
ExecutionDnnRnnWeightParams reject_rnn_weight_params_requests(ExecutionSocketClient& client,
                                                              ExecutionSocketClient& metadata,
                                                              std::uint64_t handle,
                                                              std::uint64_t tensor,
                                                              bool is_metadata) {
    const auto rnn = create_descriptor(metadata, ExecutionDnnObjectKind::kRnn);
    const ExecutionDnnRnnWeightParams value{.handle = handle,
                                            .metadata = metadata.session(),
                                            .descriptor = rnn,
                                            .destinations = {tensor, tensor},
                                            .outputs = 3};
    const auto reject = [&](const auto& invalid) {
        rejected(client.exchange(ExecutionOpcode::kDnnRnnWeightParams, rnn_weight_params(invalid)));
    };
    if (is_metadata) {
        reject(value);
        return value;
    }
    auto bytes = rnn_weight_params(value);
    bytes.push_back(std::byte{1});
    rejected(client.exchange(ExecutionOpcode::kDnnRnnWeightParams, bytes));
    bytes.resize(bytes.size() - 2);
    rejected(client.exchange(ExecutionOpcode::kDnnRnnWeightParams, bytes));
    for (const auto offset : {40U, 72U}) {
        bytes = rnn_weight_params(value);
        bytes[offset] = std::byte{4};
        rejected(client.exchange(ExecutionOpcode::kDnnRnnWeightParams, bytes));
    }
    for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
        auto invalid = value;
        invalid.metadata = owner;
        reject(invalid);
    }
    for (const auto descriptor : {tensor, UINT64_MAX}) {
        auto invalid = value;
        invalid.descriptor = descriptor;
        reject(invalid);
    }
    for (std::size_t index = 0; index < value.destinations.size(); ++index) {
        for (const auto destination : {rnn, UINT64_MAX}) {
            auto invalid = value;
            invalid.destinations[index] = destination;
            reject(invalid);
        }
    }
    for (const auto outputs : {0U, 1U, 2U}) {
        auto invalid = value;
        invalid.outputs = outputs;
        reject(invalid);  // Present descriptors require writable native output slots.
    }
    auto invalid = value;
    invalid.handle = UINT64_MAX;
    reject(invalid);
    invalid = value;
    invalid.weights.allocation = UINT64_MAX;
    invalid.weight_bytes = 8;
    reject(invalid);
    return value;
}
std::vector<std::byte> attn_weights(const ExecutionDnnAttnWeights& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_attn_weights(writer, value), "encode attention layout authority");
    auto bytes = writer.take();
    expect(bytes.size() == 52, "attention layout request has exact canonical extent");
    return bytes;
}
ExecutionDnnAttnWeights reject_attn_weights_requests(ExecutionSocketClient& client,
                                                     ExecutionSocketClient& metadata,
                                                     std::uint64_t handle, std::uint64_t tensor,
                                                     bool is_metadata) {
    const auto attention = create_descriptor(metadata, ExecutionDnnObjectKind::kAttn);
    const ExecutionDnnAttnWeights value{.handle = handle,
                                        .metadata = metadata.session(),
                                        .descriptor = attention,
                                        .has_base = 1,
                                        .weight_bytes = 128,
                                        .destination = tensor,
                                        .has_output = 1};
    const auto reject = [&](const auto& invalid) {
        rejected(client.exchange(ExecutionOpcode::kDnnAttnWeights, attn_weights(invalid)));
    };
    if (is_metadata) {
        reject(value);
        return value;
    }
    auto bytes = attn_weights(value);
    bytes.push_back(std::byte{1});
    rejected(client.exchange(ExecutionOpcode::kDnnAttnWeights, bytes));
    bytes.resize(bytes.size() - 2);
    rejected(client.exchange(ExecutionOpcode::kDnnAttnWeights, bytes));
    for (const auto offset : {24U, 28U, 48U}) {
        bytes = attn_weights(value);
        bytes[offset] = std::byte{8};
        rejected(client.exchange(ExecutionOpcode::kDnnAttnWeights, bytes));
    }
    for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
        auto invalid = value;
        invalid.metadata = owner;
        reject(invalid);
    }
    for (const auto descriptor : {std::uint64_t{0}, tensor, UINT64_MAX}) {
        auto invalid = value;
        invalid.descriptor = descriptor;
        reject(invalid);
    }
    for (const auto destination : {attention, UINT64_MAX}) {
        auto invalid = value;
        invalid.destination = destination;
        reject(invalid);
    }
    for (const auto handle_id : {std::uint64_t{0}, UINT64_MAX}) {
        auto invalid = value;
        invalid.handle = handle_id;
        reject(invalid);
    }
    auto overflowing = value;
    overflowing.weight_bytes = UINT64_MAX;
    reject(overflowing);  // The private nonnull sentinel also needs backend arithmetic bounds.
    return value;
}
std::vector<std::byte> attn_computation(const ExecutionDnnAttnCompute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_attn_compute(writer, value), "encode attention compute authority");
    auto bytes = writer.take();
    expect(bytes.size() == 344 + 4 * (value.lower.size() + value.upper.size()),
           "attention computation has exact canonical extent");
    return bytes;
}
std::array<ExecutionDnnAttnCompute, 3> reject_attn_compute_requests(ExecutionSocketClient& client,
                                                                    ExecutionSocketClient& metadata,
                                                                    std::uint64_t handle,
                                                                    std::uint64_t tensor,
                                                                    bool is_metadata) {
    const auto attn = create_descriptor(metadata, ExecutionDnnObjectKind::kAttn);
    const auto sequence = create_descriptor(metadata, ExecutionDnnObjectKind::kSeqData);
    std::array<ExecutionDnnAttnCompute, 3> values{};
    for (std::size_t index = 0; index != values.size(); ++index) {
        auto& value = values[index];
        value.kind = static_cast<ExecutionDnnAttnComputeKind>(index + 1);
        value.handle = handle;
        value.metadata = metadata.session();
        value.descriptors = {attn, sequence, sequence, sequence, sequence};
        for (const auto opcode :
             {ExecutionOpcode::kPrepareDnnAttn, ExecutionOpcode::kDnnAttnCompute}) {
            const auto reject = [&](const auto& invalid) {
                rejected(client.exchange(opcode, attn_computation(invalid)));
            };
            if (is_metadata) {
                reject(value);
                continue;
            }
            auto bytes = attn_computation(value);
            bytes.push_back(std::byte{1});
            rejected(client.exchange(opcode, bytes));
            bytes.resize(bytes.size() - 2);
            rejected(client.exchange(opcode, bytes));
            // Invalid kind/mode/presence and a null allocation with an offset.
            for (const auto offset : {0U, 4U, 12U, 80U}) {
                bytes = attn_computation(value);
                bytes[offset] = std::byte{8};
                rejected(client.exchange(opcode, bytes));
            }
            // Every route has its own inactive buffer positions.
            const auto unused = index == 0 ? 7U : (index == 1 ? 5U : 0U);
            bytes = attn_computation(value);
            bytes[72 + 16 * unused] = std::byte{1};
            rejected(client.exchange(opcode, bytes));
            for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
                auto invalid = value;
                invalid.metadata = owner;
                reject(invalid);
            }
            for (std::size_t slot = 0; slot != value.descriptors.size(); ++slot) {
                auto invalid = value;
                invalid.descriptors[slot] = slot == 0 ? tensor : attn;
                reject(invalid);
                invalid.descriptors[slot] = UINT64_MAX;
                reject(invalid);
            }
            auto invalid = value;
            invalid.descriptors[0] = 0;
            reject(invalid);
            invalid = value;
            invalid.handle = UINT64_MAX;
            reject(invalid);
            if (opcode == ExecutionOpcode::kPrepareDnnAttn && index < 2) {
                invalid = value;
                invalid.window_presence = 3;
                invalid.lower = {0};
                invalid.upper = {1};
                reject(invalid);  // Preparation never accepts captured caller data.
            }
        }
    }
    return values;
}
std::array<ExecutionDnnRnnCompute, 3> reject_rnn_compute_requests(ExecutionSocketClient& client,
                                                                  ExecutionSocketClient& metadata,
                                                                  std::uint64_t handle,
                                                                  std::uint64_t tensor,
                                                                  bool is_metadata) {
    const auto rnn = create_descriptor(metadata, ExecutionDnnObjectKind::kRnn);
    const auto data = create_descriptor(metadata, ExecutionDnnObjectKind::kRnnData);
    std::array<ExecutionDnnRnnCompute, 3> values{};
    for (std::size_t index = 0; index != values.size(); ++index) {
        auto& value = values[index];
        value.kind = static_cast<ExecutionDnnRnnComputeKind>(index + 1);
        value.handle = handle;
        value.metadata = metadata.session();
        value.descriptors = {rnn, data, data, tensor, 0};
        if (is_metadata) {
            rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, rnn_computation(value)));
            continue;
        }
        auto bytes = rnn_computation(value);
        bytes.push_back(std::byte{1});
        rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, bytes));
        bytes.resize(bytes.size() - 2);
        rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, bytes));
        // Real typed identities and present callbacks separate wire rejection
        // from absent capability. Slot zero also proves null-offset canonicality.
        for (const auto offset : {std::size_t{0}, std::size_t{72}}) {
            bytes = rnn_computation(value);
            bytes[offset] = std::byte{8};
            rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, bytes));
        }
        const auto unused = index == 1 ? 1U : 3U;
        bytes = rnn_computation(value);
        bytes[64 + unused * 16] = std::byte{1};
        rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, bytes));
        for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
            auto invalid = value;
            invalid.metadata = owner;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, rnn_computation(invalid)));
        }
        for (std::size_t slot = 0; slot != 4; ++slot) {
            auto invalid = value;
            invalid.descriptors[slot] = slot == 0 ? tensor : rnn;
            rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, rnn_computation(invalid)));
        }
        auto invalid = value;
        invalid.handle = UINT64_MAX;
        rejected(client.exchange(ExecutionOpcode::kDnnRnnCompute, rnn_computation(invalid)));
    }
    const ExecutionDnnBuildRnnDynamic build{handle, metadata.session(), rnn, 1};
    if (is_metadata) {
        rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, rnn_build(build)));
    } else {
        auto bytes = rnn_build(build);
        bytes.push_back(std::byte{1});
        rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, bytes));
        bytes.resize(bytes.size() - 2);
        rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, bytes));
        for (const auto owner : {std::uint64_t{0}, client.session(), UINT64_MAX}) {
            auto invalid = build;
            invalid.metadata = owner;
            rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, rnn_build(invalid)));
        }
        for (const auto descriptor : {tensor, UINT64_MAX}) {
            auto invalid = build;
            invalid.descriptor = descriptor;
            rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, rnn_build(invalid)));
        }
        auto invalid = build;
        invalid.handle = UINT64_MAX;
        rejected(client.exchange(ExecutionOpcode::kDnnBuildRnnDynamic, rnn_build(invalid)));
    }
    return values;
}
template <class Function>
void drive(ExecutionService& service, Function function) {
    auto client = std::async(std::launch::async, std::move(function));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded raw-service fixture progress");
    client.get();
}
ExecutionServiceConfig config(const std::string& path, ExecutionSchedulingMode mode,
                              TaskPeerIdentity identity) {
    ExecutionServiceConfig result;
    result.transport.socket_path = path;
    result.scheduling_mode = mode;
    result.uses_original_kernels = true;
    if (mode == ExecutionSchedulingMode::kPriority) result.client_priorities = {{identity, 7}};
    return result;
}
void reset() {
    native = {};
    dnn::reset(&current_context);
}
void no_gpu_ownership() {
    expect(!native.initializations && !native.context_creates && !native.context_sets &&
               !native.context_destroys && !native.identity_queries && !native.unexpected_calls &&
               !native.sequence_calls && !native.rnn_attention_sets &&
               native.rnn_attention_creates == native.rnn_attention_destroys &&
               native.rnn_data_creates == native.rnn_data_destroys && !native.rnn_queries &&
               !native.rnn_preflights && !native.rnn_clips && !native.rnn_computations &&
               !native.rnn_builds && !native.rnn_weight_params && !native.attn_weights &&
               !native.attn_computations && !native.attn_preflights &&
               native.seq_creates == native.seq_destroys && !dnn::state.handle_creates &&
               !dnn::state.activation_calls && !dnn::state.add_calls,
           "CPU metadata cannot initialize, profile, create or execute GPU resources");
}

void metadata_in_mode(const std::string& path, ExecutionSchedulingMode mode,
                      TaskPeerIdentity identity) {
    reset();
    ExecutionService service(config(path, mode, identity), dispatch());
    expect(service.start(), "start actual DNN metadata service");
    drive(service, [&] {
        ExecutionSocketClient client({path, identity, 3000});
        open_metadata(client);
        query_library_info(client);
        const auto tensor = create_descriptor(client, ExecutionDnnObjectKind::kTensor);
        const auto activation = create_descriptor(client, ExecutionDnnObjectKind::kActivation);
        ExecutionDnnTensor value;
        value.dimensions = 4;
        value.sizes = {1, 2, 3, 4};
        value.strides = {24, 12, 4, 1};
        for (const auto setter :
             {ExecutionDnnTensorSetter::kFour, ExecutionDnnTensorSetter::kFourStrided,
              ExecutionDnnTensorSetter::kNd, ExecutionDnnTensorSetter::kNdPacked}) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(tensor) && input.write_u32(static_cast<std::uint32_t>(setter)) &&
                       write_execution_dnn_tensor(input, value),
                   "encode original tensor metadata");
            sdk_success(client.exchange(ExecutionOpcode::kSetDnnTensor, input.take()));
        }
        ExecutionPayloadWriter query;
        expect(query.write_u64(tensor) && query.write_u32(8) && query.write_u32(0),
               "request bounded full tensor metadata");
        const auto tensor_reply =
            sdk_success(client.exchange(ExecutionOpcode::kGetDnnTensor, query.take()));
        ExecutionPayloadReader tensor_reader(tensor_reply.data);
        ExecutionDnnTensor actual;
        expect(read_execution_dnn_tensor(tensor_reader, actual) && tensor_reader.empty() &&
                   actual.dimensions == value.dimensions && actual.sizes == value.sizes &&
                   actual.strides == value.strides,
               "original metadata crosses real authenticated transport without GPU policy");
        expect(sdk_success(client.exchange(ExecutionOpcode::kDnnTensorSize, object_id(tensor)))
                       .address == 4096,
               "native byte-size query stays a CPU-owned descriptor operation");
        const ExecutionDnnActivation activation_value{1, 1, std::bit_cast<std::uint64_t>(7.5)};
        ExecutionPayloadWriter activation_input;
        expect(activation_input.write_u64(activation) &&
                   write_execution_dnn_activation(activation_input, activation_value),
               "encode original activation metadata");
        sdk_success(client.exchange(ExecutionOpcode::kSetDnnActivation, activation_input.take()));
        const auto activation_reply =
            sdk_success(client.exchange(ExecutionOpcode::kGetDnnActivation, object_id(activation)));
        ExecutionPayloadReader activation_reader(activation_reply.data);
        ExecutionDnnActivation actual_activation;
        expect(read_execution_dnn_activation(activation_reader, actual_activation) &&
                   activation_reader.empty() && actual_activation.mode == activation_value.mode &&
                   actual_activation.nan == activation_value.nan &&
                   actual_activation.coefficient_bits == activation_value.coefficient_bits,
               "activation metadata and coefficient bits survive policy-independent IPC");

        rejected(client.exchange(ExecutionOpcode::kAllocate, object_id(4096)));
        rejected(client.exchange(ExecutionOpcode::kCreateDnn));
        rejected(
            client.exchange(ExecutionOpcode::kDnnCompute, computation(problem(client.session()))));
        rejected(client.exchange(ExecutionOpcode::kDnnTensorForward,
                                 computation(tensor_forward(client.session()))));
        ExecutionPayloadWriter lrn_input;
        expect(lrn_input.write_u64(tensor) && write_execution_dnn_lrn(lrn_input, {5, 0, 0, 0}),
               "encode LRN mutation with an owned but incorrect descriptor kind");
        rejected(client.exchange(ExecutionOpcode::kSetDnnLrn, lrn_input.take()));
        const auto reduction_value = reduction(client.session());
        rejected(client.exchange(ExecutionOpcode::kDnnReduction, computation(reduction_value)));
        for (const auto kind :
             {ExecutionDnnReductionQueryKind::kIndices, ExecutionDnnReductionQueryKind::kWorkspace})
            rejected(client.exchange(ExecutionOpcode::kDnnReductionQuery,
                                     reduction_query(reduction_value, kind)));
        ExecutionPayloadWriter reduction_input;
        expect(
            reduction_input.write_u64(tensor) && write_execution_dnn_reduction(reduction_input, {}),
            "encode reduction mutation with incorrect metadata kind");
        rejected(client.exchange(ExecutionOpcode::kSetDnnReduction, reduction_input.take()));
        const auto bn = batch_norm(client.session());
        rejected(client.exchange(ExecutionOpcode::kDnnBatchNorm, computation(bn)));
        for (const auto kind :
             {ExecutionDnnBatchNormKind::kForward, ExecutionDnnBatchNormKind::kBackward,
              ExecutionDnnBatchNormKind::kReserve})
            rejected(
                client.exchange(ExecutionOpcode::kDnnBatchNormQuery, batch_norm_query(bn, kind)));
        const ExecutionDnnDropout dropout{.metadata = client.session(),
                                          .descriptor = tensor,
                                          .handle = 1,
                                          .states = {},
                                          .kind = ExecutionDnnDropoutKind::kSet};
        rejected(client.exchange(ExecutionOpcode::kDnnDropout, computation(dropout)));
        rejected(client.exchange(ExecutionOpcode::kDnnDropoutStatesSize, object_id(1)));
        for (const auto kind : {ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount,
                                ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount,
                                ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace}) {
            ExecutionDnnBackwardConvolutionQuery query{.kind = kind, .handle = 1, .has_output = 1};
            if (kind == ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace) {
                query.metadata = client.session();
                query.descriptors = {tensor, tensor, activation, tensor};
            }
            rejected(client.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                     backward_query(query)));
        }
        const ExecutionDnnReorderFilterAndBias reorder{.handle = 1,
                                                       .metadata = client.session(),
                                                       .filter = tensor,
                                                       .buffers = {{{1, 0}, {2, 0}, {}, {}}}};
        rejected(client.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, computation(reorder)));
        const ExecutionDnnFoldedConvBackwardDataDescriptors folded{
            .handle = 1,
            .metadata = client.session(),
            .inputs = {tensor, tensor, activation, tensor},
            .outputs = {tensor, tensor, activation, tensor, tensor, tensor, tensor, 0}};
        reject_folded(client, folded_request(folded), CUDA_ERROR_NOT_SUPPORTED);
        reject_operation_descriptor_requests(client, tensor);
        reject_sequence_descriptor_requests(client, tensor);
        reject_rnn_attention_requests(client, tensor, true);
        (void)reject_rnn_queries(client, client, 1, tensor, true);
        (void)reject_rnn_compute_requests(client, client, 1, tensor, true);
        (void)reject_rnn_weight_params_requests(client, client, 1, tensor, true);
        (void)reject_attn_weights_requests(client, client, 1, tensor, true);
        (void)reject_attn_compute_requests(client, client, 1, tensor, true);
        sdk_success(client.exchange(ExecutionOpcode::kDestroyDnnMetadata, object_id(tensor)));
        rejected(client.exchange(ExecutionOpcode::kDestroyDnnMetadata, object_id(tensor)));
        sdk_success(client.exchange(ExecutionOpcode::kClose));
    });
    expect(service.request_stop(), "metadata shutdown proves exact native-descriptor cleanup");
    no_gpu_ownership();
    expect(dnn::state.creates == 2 && dnn::state.destroys == 2 && dnn::state.sets == 5 &&
               dnn::state.gets == 2 && dnn::state.sizes == 1 && native.library_queries == 3 &&
               native.property_queries == 3 &&
               native.backward_queries == std::array<unsigned int, 3>{} && !native.folded_calls &&
               !native.folded_preflights &&
               dnn::state.tensor_setters == std::array<unsigned int, 4>{1, 1, 1, 1},
           "all metadata effects occur exactly once in each scheduler mode");
}

void malformed_bootstrap(const std::string& path, TaskPeerIdentity identity) {
    reset();
    ExecutionService service(config(path, ExecutionSchedulingMode::kNaive, identity), dispatch());
    expect(service.start(), "start bootstrap authority fixture");
    drive(service, [&] {
        for (const auto& payload : {role(0xdeadbeef), role(kExecutionDnnMetadataRole, 1)}) {
            ExecutionSocketClient client({path, identity, 3000});
            const auto reply = client.open(payload);
            expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                       reply->status == CUDA_ERROR_INVALID_VALUE && !reply->original_status &&
                       decode_execution_bootstrap_ownership(reply->data) ==
                           ExecutionBootstrapOwnership::kNone,
                   "unknown role or metadata context flags fail with proven empty ownership");
        }
    });
    expect(service.request_stop(), "failed role bootstraps retain no resource owner");
    no_gpu_ownership();
    expect(!dnn::state.creates && !dnn::state.destroys,
           "invalid Hello cannot create a hidden CPU descriptor");
}

void compute_authority(const std::string& path, TaskPeerIdentity identity) {
    reset();
    native.version = 91002;  // Authority rejection must not rely on an unavailable SDK version.
    ExecutionService service(config(path, ExecutionSchedulingMode::kNaive, identity), dispatch());
    expect(service.start(), "start independent DNN compute authority fixture");
    drive(service, [&] {
        ExecutionSocketClient metadata({path, identity, 3000});
        open_metadata(metadata);
        const auto tensor = create_descriptor(metadata, ExecutionDnnObjectKind::kTensor);
        const auto activation = create_descriptor(metadata, ExecutionDnnObjectKind::kActivation);
        ExecutionSocketClient context({path, identity, 3000});
        sdk_success(context.open());
        bind_dnn_profile(context);
        for (const auto kind :
             {ExecutionDnnLibraryInfoKind::kVersion, ExecutionDnnLibraryInfoKind::kCudartVersion,
              ExecutionDnnLibraryInfoKind::kMaxDeviceVersion,
              ExecutionDnnLibraryInfoKind::kProperty})
            rejected(context.exchange(ExecutionOpcode::kDnnLibraryInfo, library_info(kind)));
        const auto handle = sdk_success(context.exchange(ExecutionOpcode::kCreateDnn)).id;
        expect(handle, "regular context owns a real injected native DNN handle");
        const auto rnn_queries = reject_rnn_queries(context, metadata, handle, tensor, false);
        const auto rnn_computations =
            reject_rnn_compute_requests(context, metadata, handle, tensor, false);
        const auto rnn_weights =
            reject_rnn_weight_params_requests(context, metadata, handle, tensor, false);
        const auto attention_weights =
            reject_attn_weights_requests(context, metadata, handle, tensor, false);
        const auto attention_computations =
            reject_attn_compute_requests(context, metadata, handle, tensor, false);
        for (const auto kind : {ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount,
                                ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount}) {
            ExecutionDnnBackwardConvolutionQuery query{
                .kind = kind, .handle = handle, .has_output = 1};
            const auto reply = sdk_success(context.exchange(
                ExecutionOpcode::kDnnBackwardConvolutionQuery, backward_query(query)));
            expect(reply.address == (kind == ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount
                                         ? 65ULL
                                         : static_cast<std::uint64_t>(INT_MAX)) &&
                       !reply.id && !reply.parameter_bytes && reply.data.empty(),
                   "distinct native count results have no Find transport-capacity clamp");
            auto raw = backward_query(query);
            for (const auto offset : {0U, 12U, 20U, 52U, 56U}) {
                auto bad = raw;
                bad[offset] = offset == 0 ? std::byte{4} : std::byte{2};
                rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery, bad));
            }
            raw.pop_back();
            rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery, raw));
            raw = backward_query(query);
            raw.push_back(std::byte{});
            rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery, raw));
            query.has_output = 0;
            rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                      backward_query(query)));
            query.has_output = 1;
            query.handle = UINT64_MAX;
            rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                      backward_query(query)));
        }
        const ExecutionDnnBackwardConvolutionQuery backward{
            .kind = ExecutionDnnBackwardConvolutionQueryKind::kDataWorkspace,
            .handle = handle,
            .metadata = metadata.session(),
            .descriptors = {tensor, tensor, activation, tensor},
            .has_output = 1};
        for (const auto owner : {metadata.session(), context.session(), UINT64_MAX}) {
            auto bad = backward;
            bad.metadata = owner;
            rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                      backward_query(bad)));
        }
        const ExecutionDnnFoldedConvBackwardDataDescriptors folded{
            .handle = handle,
            .metadata = metadata.session(),
            .inputs = {tensor, tensor, activation, tensor},
            .outputs = {tensor, tensor, activation, tensor, tensor, tensor, tensor, 0}};
        // These are genuine live tensor/activation IDs, deliberately supplied
        // where filter/convolution/transform ownership is required.
        for (const auto owner : {metadata.session(), context.session(), UINT64_MAX}) {
            auto bad = folded;
            bad.metadata = owner;
            reject_folded(context, folded_request(bad), CUDA_ERROR_INVALID_HANDLE);
        }
        auto unknown_handle = folded;
        unknown_handle.handle = UINT64_MAX;
        reject_folded(context, folded_request(unknown_handle), CUDA_ERROR_INVALID_HANDLE);
        auto short_folded = folded_request(folded);
        short_folded.pop_back();
        reject_folded(context, short_folded, CUDA_ERROR_INVALID_VALUE);
        auto extra_folded = folded_request(folded);
        extra_folded.push_back(std::byte{});
        reject_folded(context, extra_folded, CUDA_ERROR_INVALID_VALUE);
        auto value = problem(metadata.session());
        value.handle = handle;
        value.tensors = {tensor, tensor};
        auto invalid = value;
        invalid.metadata = UINT64_MAX;
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(invalid)));
        invalid.metadata = context.session();
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(invalid)));
        invalid = value;
        invalid.tensors[0] = activation;
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(invalid)));
        invalid.tensors[0] = UINT64_MAX;
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(invalid)));
        invalid = value;
        invalid.kind = ExecutionDnnComputeKind::kActivationForward;
        invalid.activation = tensor;
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(invalid)));
        auto forward = tensor_forward(metadata.session());
        forward.handle = handle;
        forward.tensors = {tensor, tensor};
        for (const auto unknown : {UINT64_MAX, context.session()}) {
            auto bad_forward = forward;
            bad_forward.metadata = unknown;
            rejected(
                context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_forward)));
        }
        for (const auto wrong_tensor : {activation, UINT64_MAX}) {
            auto bad_forward = forward;
            bad_forward.tensors[0] = wrong_tensor;
            rejected(
                context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_forward)));
        }
        auto bad_pool = forward;
        bad_pool.kind = ExecutionDnnTensorForwardKind::kPoolingForward;
        bad_pool.descriptor = activation;
        rejected(context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_pool)));
        bad_pool.descriptor = UINT64_MAX;
        rejected(context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_pool)));
        for (const auto kind : {ExecutionDnnTensorForwardKind::kLrnForward,
                                ExecutionDnnTensorForwardKind::kTransformTensor}) {
            auto new_forward = forward;
            new_forward.kind = kind;
            new_forward.descriptor =
                kind == ExecutionDnnTensorForwardKind::kLrnForward ? activation : 0;
            for (const auto unknown : {UINT64_MAX, context.session()}) {
                auto bad_forward = new_forward;
                bad_forward.metadata = unknown;
                rejected(
                    context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_forward)));
            }
            auto bad_forward = new_forward;
            bad_forward.tensors[0] = activation;
            rejected(
                context.exchange(ExecutionOpcode::kDnnTensorForward, computation(bad_forward)));
            if (kind == ExecutionDnnTensorForwardKind::kLrnForward) {
                // LRN cannot consume a tensor/activation as its operation descriptor.
                rejected(
                    context.exchange(ExecutionOpcode::kDnnTensorForward, computation(new_forward)));
                new_forward.descriptor = tensor;
                rejected(
                    context.exchange(ExecutionOpcode::kDnnTensorForward, computation(new_forward)));
            }
        }
        ExecutionPayloadWriter lrn_input;
        expect(lrn_input.write_u64(tensor) && write_execution_dnn_lrn(lrn_input, {5, 0, 0, 0}),
               "encode context-role attempt to mutate CPU-only LRN metadata");
        rejected(context.exchange(ExecutionOpcode::kSetDnnLrn, lrn_input.take()));
        auto reduce = reduction(metadata.session());
        reduce.handle = handle;
        reduce.descriptors = {tensor, activation, tensor};
        const auto reject_reduction = [&](const auto& candidate) {
            rejected(context.exchange(ExecutionOpcode::kDnnReduction, computation(candidate)));
            for (const auto kind : {ExecutionDnnReductionQueryKind::kIndices,
                                    ExecutionDnnReductionQueryKind::kWorkspace})
                rejected(context.exchange(ExecutionOpcode::kDnnReductionQuery,
                                          reduction_query(candidate, kind)));
        };
        for (const auto owner : {UINT64_MAX, context.session()}) {
            auto bad = reduce;
            bad.metadata = owner;
            reject_reduction(bad);
        }
        reject_reduction(reduce);  // Activation cannot authorize reduction.
        for (const auto wrong : {activation, UINT64_MAX}) {
            auto bad = reduce;
            bad.descriptors[0] = wrong;
            reject_reduction(bad);
        }
        ExecutionPayloadWriter reduction_input;
        expect(
            reduction_input.write_u64(tensor) && write_execution_dnn_reduction(reduction_input, {}),
            "encode reduction mutation on nonmetadata connection");
        rejected(context.exchange(ExecutionOpcode::kSetDnnReduction, reduction_input.take()));
        auto bn = batch_norm(metadata.session());
        bn.query.handle = handle;
        bn.query.descriptors = {tensor, tensor, 0, 0, 0, tensor, activation};
        const auto reject_batch_norm = [&](const auto& candidate) {
            rejected(context.exchange(ExecutionOpcode::kDnnBatchNorm, computation(candidate)));
            for (const auto query_kind :
                 {ExecutionDnnBatchNormKind::kForward, ExecutionDnnBatchNormKind::kBackward,
                  ExecutionDnnBatchNormKind::kReserve})
                rejected(context.exchange(ExecutionOpcode::kDnnBatchNormQuery,
                                          batch_norm_query(candidate, query_kind)));
        };
        for (const auto owner : {UINT64_MAX, context.session()}) {
            auto bad = bn;
            bad.query.metadata = owner;
            reject_batch_norm(bad);
        }
        for (const auto wrong : {activation, UINT64_MAX}) {
            auto bad = bn;
            bad.query.descriptors[0] = wrong;
            reject_batch_norm(bad);
        }
        for (const auto wrong : {tensor, UINT64_MAX}) {
            auto bad = bn;
            bad.query.descriptors[6] = wrong;
            reject_batch_norm(bad);
        }
        ExecutionDnnDropout dropout{.metadata = metadata.session(),
                                    .descriptor = tensor,
                                    .handle = handle,
                                    .states = {},
                                    .kind = ExecutionDnnDropoutKind::kSet};
        for (const auto operation :
             {ExecutionDnnDropoutKind::kSet, ExecutionDnnDropoutKind::kRestore}) {
            dropout.kind = operation;
            for (const auto owner : {UINT64_MAX, context.session()}) {
                auto bad = dropout;
                bad.metadata = owner;
                rejected(context.exchange(ExecutionOpcode::kDnnDropout, computation(bad)));
            }
            for (const auto wrong : {tensor, activation, UINT64_MAX}) {
                auto bad = dropout;
                bad.descriptor = wrong;
                rejected(context.exchange(ExecutionOpcode::kDnnDropout, computation(bad)));
            }
        }
        rejected(context.exchange(ExecutionOpcode::kDnnDropoutStatesSize, object_id(UINT64_MAX)));
        const ExecutionDnnReorderFilterAndBias reorder{.handle = handle,
                                                       .metadata = metadata.session(),
                                                       .filter = tensor,
                                                       .buffers = {{{1, 0}, {2, 0}, {}, {}}}};
        for (const auto owner : {UINT64_MAX, context.session()}) {
            auto bad = reorder;
            bad.metadata = owner;
            rejected(context.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, computation(bad)));
        }
        for (const auto wrong : {tensor, activation, UINT64_MAX}) {
            auto bad = reorder;
            bad.filter = wrong;
            rejected(context.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, computation(bad)));
        }
        auto noncanonical = computation(reorder);
        noncanonical[64] = std::byte{1};  // A hidden bias identity with a zero flag.
        rejected(context.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, noncanonical));
        auto short_reorder = computation(reorder);
        short_reorder.pop_back();
        rejected(context.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, short_reorder));
        reject_operation_descriptor_requests(context, tensor);
        reject_sequence_descriptor_requests(context, tensor);
        reject_rnn_attention_requests(context, tensor, false);
        for (const auto sequence_kind :
             {ExecutionDnnObjectKind::kRnnData, ExecutionDnnObjectKind::kSeqData}) {
            ExecutionPayloadWriter create;
            expect(create.write_u32(static_cast<std::uint32_t>(sequence_kind)),
                   "encode new sequence lifecycle on wrong authority");
            rejected(context.exchange(ExecutionOpcode::kCreateDnnMetadata, create.take()));
        }
        ExecutionPayloadWriter kind;
        expect(kind.write_u32(1), "encode descriptor request on wrong role");
        rejected(context.exchange(ExecutionOpcode::kCreateDnnMetadata, kind.take()));
        sdk_success(metadata.exchange(ExecutionOpcode::kClose));
        for (const auto& query : rnn_queries)
            rejected(context.exchange(ExecutionOpcode::kDnnRnnQuery, rnn_query(query)));
        for (const auto& value : rnn_computations)
            rejected(context.exchange(ExecutionOpcode::kDnnRnnCompute, rnn_computation(value)));
        rejected(context.exchange(
            ExecutionOpcode::kDnnBuildRnnDynamic,
            rnn_build({handle, metadata.session(), rnn_computations[0].descriptors[0], 1})));
        rejected(
            context.exchange(ExecutionOpcode::kDnnRnnWeightParams, rnn_weight_params(rnn_weights)));
        rejected(
            context.exchange(ExecutionOpcode::kDnnAttnWeights, attn_weights(attention_weights)));
        for (const auto& attention : attention_computations)
            for (const auto opcode :
                 {ExecutionOpcode::kPrepareDnnAttn, ExecutionOpcode::kDnnAttnCompute})
                rejected(context.exchange(opcode, attn_computation(attention)));
        // Keep the socket alive: the metadata slot is closing, not a guessed
        // disconnected owner. Its descriptors cannot authorize a fresh command.
        rejected(context.exchange(ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                  backward_query(backward)));
        reject_folded(context, folded_request(folded), CUDA_ERROR_INVALID_HANDLE);
        rejected(context.exchange(ExecutionOpcode::kDnnCompute, computation(value)));
        rejected(context.exchange(ExecutionOpcode::kDnnTensorForward, computation(forward)));
        reject_reduction(reduce);  // Closing metadata cannot authorize new work or queries.
        reject_batch_norm(bn);
        rejected(context.exchange(ExecutionOpcode::kDnnDropout, computation(dropout)));
        rejected(context.exchange(ExecutionOpcode::kDnnReorderFilterAndBias, computation(reorder)));
        for (const auto kind : {ExecutionDnnTensorForwardKind::kLrnForward,
                                ExecutionDnnTensorForwardKind::kTransformTensor}) {
            auto closed = forward;
            closed.kind = kind;
            closed.descriptor = kind == ExecutionDnnTensorForwardKind::kLrnForward ? activation : 0;
            rejected(context.exchange(ExecutionOpcode::kDnnTensorForward, computation(closed)));
        }
        sdk_success(context.exchange(ExecutionOpcode::kClose));
    });
    expect(service.request_stop(), "rejected computations preserve clean once-only ownership");
    expect(native.context_creates == 1 && native.context_destroys == 1 &&
               !native.identity_queries && !native.unexpected_calls && !native.library_queries &&
               !native.property_queries && dnn::state.handle_creates == 1 &&
               native.backward_queries == std::array<unsigned int, 3>{1, 1, 0} &&
               !native.folded_calls && !native.folded_preflights && !native.sequence_calls &&
               !native.rnn_queries && !native.rnn_preflights && !native.rnn_clips &&
               !native.rnn_computations && !native.rnn_builds && !native.rnn_weight_params &&
               !native.attn_weights && !native.attn_computations && !native.attn_preflights &&
               native.seq_creates == native.seq_destroys &&
               native.rnn_data_creates == native.rnn_data_destroys &&
               native.rnn_attention_creates == native.rnn_attention_destroys &&
               dnn::state.handle_destroys == 1 && dnn::state.stream_gets == 1 && !dnn::state.gets &&
               !dnn::state.sizes && !dnn::state.activation_calls && !dnn::state.add_calls &&
               dnn::state.creates == 2 && dnn::state.destroys == 2,
           "bad authority rejects before SDK compute preparation and never mints GPU resources");
}

void failed_disconnect_retains_owner(const std::string& path) {
    const auto child = ::fork();
    expect(child >= 0, "fork isolated uncertain descriptor ownership fixture");
    if (!child) {
        // Keep the quarantined service alive through _exit on success or failure:
        // no destructor gets to guess a second native cleanup attempt.
        std::unique_ptr<ExecutionService> service;
        try {
            reset();
            dnn::state.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
            const auto identity = current_process_identity();
            expect(identity.has_value(), "isolated service has its own authenticated identity");
            auto options = config(path, ExecutionSchedulingMode::kNaive, *identity);
            options.exit_when_idle = true;
            service = std::make_unique<ExecutionService>(std::move(options), dispatch());
            expect(service->start(), "start isolated uncertain metadata owner");
            drive(*service, [&] {
                ExecutionSocketClient client({path, *identity, 3000});
                open_metadata(client);
                (void)create_descriptor(client, ExecutionDnnObjectKind::kTensor);
                // Disconnect with a live descriptor; there is deliberately no Close.
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
            while (!dnn::state.destroys)
                expect(service->step(1) && std::chrono::steady_clock::now() < deadline,
                       "observe the actual failed destruction after metadata disconnect");
            expect(service->step(1) && service->step(1) && !service->finished(),
                   "failed original descriptor destruction cannot retire the service owner");
            expect(!service->request_stop() && !service->finished() && dnn::state.creates == 1 &&
                       dnn::state.destroys == 1,
                   "bounded shutdown retains uncertainty and never retries descriptor destruction");
            no_gpu_ownership();
            ::_exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated failed metadata destruction preserves uncertain ownership");
    // request_stop normally unlinks it; a diagnostic failure may bypass that call.
    if (::unlink(path.c_str()) != 0)
        expect(errno == ENOENT, "remove only the isolated fixture's owned socket path");
}

}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-dnn-service-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        const auto identity = current_process_identity();
        expect(identity.has_value(), "actual Unix peer process identity");
        for (const auto mode : {ExecutionSchedulingMode::kNaive, ExecutionSchedulingMode::kProfile,
                                ExecutionSchedulingMode::kPriority})
            metadata_in_mode(std::string(directory) + "/metadata.sock", mode, *identity);
        malformed_bootstrap(std::string(directory) + "/bootstrap.sock", *identity);
        compute_authority(std::string(directory) + "/authority.sock", *identity);
        failed_disconnect_retains_owner(std::string(directory) + "/uncertain.sock");
        std::cout << "Raw DNN service role and metadata authority checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) {
        std::cerr << "DNN service fixture private directory was not empty\n";
        result = EXIT_FAILURE;
    }
    return result;
}
