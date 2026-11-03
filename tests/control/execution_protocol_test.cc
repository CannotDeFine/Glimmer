#include "glimmer/control/execution_protocol.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace glimmer::control;
static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kHello) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kSgemm) == 30);
static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kSourceContinuation) == 132);
static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kGraphLaunch) == 146);
static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute) == 150);
static_assert(static_cast<std::uint32_t>(ExecutionCopyEndpointKind::kNull) == 0);
static_assert(static_cast<std::uint32_t>(ExecutionCopyEndpointKind::kDevice) == 1);
static_assert(static_cast<std::uint32_t>(ExecutionCopyEndpointKind::kHost) == 2);
static_assert(static_cast<std::uint32_t>(ExecutionCopyEndpointKind::kPageable) == 3);
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check_dnn_descriptor_observations() {
    for (const auto operation :
         {ExecutionOpcode::kGetDnnTensor, ExecutionOpcode::kDnnTensorSize,
          ExecutionOpcode::kGetDnnActivation, ExecutionOpcode::kGetDnnFilter,
          ExecutionOpcode::kDnnFilterSize, ExecutionOpcode::kGetDnnConvolution,
          ExecutionOpcode::kGetDnnConvolutionAttribute, ExecutionOpcode::kDnnConvolutionOutput,
          ExecutionOpcode::kGetDnnPooling, ExecutionOpcode::kDnnPoolingOutput,
          ExecutionOpcode::kGetDnnOpTensor, ExecutionOpcode::kGetDnnSeqData})
        expect(execution_is_dnn_descriptor_observation(operation),
               "read-only descriptor observations do not extend the context mutation phase");
    for (const auto operation :
         {ExecutionOpcode::kSetDnnTensor, ExecutionOpcode::kSetDnnFilter,
          ExecutionOpcode::kDnnConvolutionQuery, ExecutionOpcode::kDnnBackendGet,
          ExecutionOpcode::kDnnCompute, static_cast<ExecutionOpcode>(UINT32_MAX)})
        expect(!execution_is_dnn_descriptor_observation(operation),
               "setters, SDK operations and unknown values remain conservatively classified");
}
void check_unified_copy() {
    const std::array endpoints{
        ExecutionCopyEndpoint{ExecutionCopyEndpointKind::kDevice, 0, 17, 32},
        ExecutionCopyEndpoint{ExecutionCopyEndpointKind::kHost, 19, 23, 64},
        ExecutionCopyEndpoint{ExecutionCopyEndpointKind::kPageable, 29, 31, 96}};
    for (const auto& destination : endpoints)
        for (const auto& source : endpoints)
            for (const bool is_asynchronous : {false, true}) {
                const ExecutionUnifiedCopy expected{destination, source, 64,
                                                    is_asynchronous ? 37U : 0U, is_asynchronous};
                ExecutionPayloadWriter output;
                expect(write_execution_unified_copy(output, expected), "typed UVA packet encodes");
                const auto bytes = output.take();
                ExecutionPayloadReader input(bytes);
                ExecutionUnifiedCopy actual;
                expect(bytes.size() == 76 && read_execution_unified_copy(input, actual) &&
                           actual == expected,
                       "fixed dual-endpoint UVA packet round trips");
                for (std::size_t size = 0; size < bytes.size(); ++size) {
                    ExecutionPayloadReader truncated{std::span{bytes}.first(size)};
                    actual = expected;
                    expect(!read_execution_unified_copy(truncated, actual) && actual == expected,
                           "truncation leaves output unchanged");
                }
                auto trailing = bytes;
                trailing.push_back(std::byte{});
                ExecutionPayloadReader extra(trailing);
                expect(!read_execution_unified_copy(extra, actual), "trailing UVA bytes reject");
            }
    const ExecutionUnifiedCopy valid{endpoints[0], endpoints[1], 64, 0, false};
    for (unsigned int index = 0; index < 8; ++index) {
        auto invalid = valid;
        switch (index) {
            case 0:
                invalid.destination.kind = static_cast<ExecutionCopyEndpointKind>(4);
                break;
            case 1:
                invalid.destination.connection = 9;
                break;
            case 2:
                invalid.destination.allocation = 0;
                break;
            case 3:
                invalid.source.connection = 0;
                break;
            case 4:
                invalid.source.allocation = 0;
                break;
            case 5:
                invalid.source.offset = UINT64_MAX;
                break;
            case 6:
                invalid.stream = 7;
                break;
            default:
                invalid.destination = {};
                break;
        }
        ExecutionPayloadWriter output;
        expect(output.write_u32(7) && !write_execution_unified_copy(output, invalid) &&
                   output.take().size() == 4,
               "invalid UVA producer never publishes partial bytes");
    }
    ExecutionPayloadWriter zero;
    expect(write_execution_unified_copy(zero, {}), "canonical null zero packet encodes");
    const auto empty = zero.take();
    ExecutionPayloadReader input(empty);
    ExecutionUnifiedCopy decoded = valid;
    expect(read_execution_unified_copy(input, decoded) && decoded == ExecutionUnifiedCopy{},
           "canonical null zero is distinct from unresolved non-null pointers");
    for (const std::size_t offset : {0U, 4U, 12U, 20U, 28U, 32U, 40U, 48U, 56U, 64U, 72U}) {
        auto malformed = empty;
        malformed[offset] = std::byte{255};
        ExecutionPayloadReader invalid(malformed);
        expect(!read_execution_unified_copy(invalid, decoded), "noncanonical zero packet rejects");
    }
    ExecutionPayloadWriter bounded(kExecutionUnifiedCopyBytes - 1);
    expect(!write_execution_unified_copy(bounded, valid) && bounded.take().empty(),
           "capacity failure leaves packet empty");
}
void check_async_frontier_batches() {
    const auto put_u32 = [](std::vector<std::byte>& bytes, std::size_t offset,
                            std::uint32_t value) {
        for (unsigned int index = 0; index < 4; ++index)
            bytes.at(offset + index) = std::byte{static_cast<unsigned char>(value >> (index * 8))};
    };
    for (const auto count : {std::size_t{1}, kExecutionMaxAsyncFrontiers}) {
        std::vector<ExecutionAsyncFrontierKey> keys;
        std::vector<ExecutionAsyncFrontierObservation> observations;
        for (std::size_t index = 0; index < count; ++index) {
            const ExecutionAsyncFrontierKey key{UINT64_MAX - index, 1000 + index};
            keys.push_back(key);
            observations.push_back(
                {key, index % 3 == 0 ? 600 : (index % 3 == 1 ? 719 : 0), index % 3 != 0});
        }
        const auto batch = encode_execution_async_frontier_batch(keys);
        const auto results = encode_execution_async_frontier_observations(observations);
        expect(batch && results && batch->size() == 8 + 16 * count &&
                   results->size() == 8 + 24 * count &&
                   decode_execution_async_frontier_batch(*batch) == keys &&
                   decode_execution_async_frontier_observations(*results) == observations,
               "one/full bounded batch preserves exact uint64 identities and independent status");
    }
    std::vector<ExecutionAsyncFrontierKey> keys{{7, 17}, {8, 18}};
    std::vector<ExecutionAsyncFrontierObservation> observations{{keys[0], 719, false},
                                                                {keys[1], 0, true}};
    const auto batch = encode_execution_async_frontier_batch(keys);
    const auto results = encode_execution_async_frontier_observations(observations);
    expect(batch && results, "mixed unretired error and retired success encode");
    expect(!encode_execution_async_frontier_batch({}) &&
               !encode_execution_async_frontier_observations({}),
           "empty batches reject before allocation");
    const std::vector<ExecutionAsyncFrontierKey> too_many(kExecutionMaxAsyncFrontiers + 1);
    const std::vector<ExecutionAsyncFrontierObservation> too_many_results(
        kExecutionMaxAsyncFrontiers + 1);
    expect(!encode_execution_async_frontier_batch(too_many) &&
               !encode_execution_async_frontier_observations(too_many_results),
           "producer cannot exceed the fixed batch bound");
    for (std::size_t size = 0; size < batch->size(); ++size)
        expect(!decode_execution_async_frontier_batch(std::span{*batch}.first(size)),
               "every truncated request rejects without partial output");
    for (std::size_t size = 0; size < results->size(); ++size)
        expect(!decode_execution_async_frontier_observations(std::span{*results}.first(size)),
               "every truncated result rejects without partial output");
    for (const auto value : {0U, 257U, UINT32_MAX}) {
        auto invalid_batch = *batch;
        auto invalid_results = *results;
        put_u32(invalid_batch, 4, value);
        put_u32(invalid_results, 4, value);
        expect(!decode_execution_async_frontier_batch(invalid_batch) &&
                   !decode_execution_async_frontier_observations(invalid_results),
               "invalid count cannot cause an unchecked allocation or size overflow");
    }
    for (const auto offset : {0U, 8U, 16U, 24U, 32U}) {
        auto invalid = *batch;
        if (offset == 0)
            put_u32(invalid, offset, 2);
        else
            std::fill_n(invalid.begin() + offset, 8, std::byte{});
        expect(!decode_execution_async_frontier_batch(invalid),
               "unknown version and either zero identity reject");
    }
    for (const auto offset : {0U, 8U, 16U, 32U, 40U}) {
        auto invalid = *results;
        if (offset == 0)
            put_u32(invalid, offset, 2);
        else
            std::fill_n(invalid.begin() + offset, 8, std::byte{});
        expect(!decode_execution_async_frontier_observations(invalid),
               "result version and zero identities reject too");
    }
    for (const auto offset : {8U, 16U}) {
        auto invalid_batch = *batch;
        auto invalid_results = *results;
        std::copy_n(invalid_batch.begin() + offset, 8, invalid_batch.begin() + offset + 16);
        std::copy_n(invalid_results.begin() + offset, 8, invalid_results.begin() + offset + 24);
        expect(!decode_execution_async_frontier_batch(invalid_batch) &&
                   !decode_execution_async_frontier_observations(invalid_results),
               "duplicate frontier OR command identity rejects the entire batch");
    }
    auto extra_batch = *batch;
    auto extra_results = *results;
    extra_batch.push_back(std::byte{});
    extra_results.push_back(std::byte{});
    expect(!decode_execution_async_frontier_batch(extra_batch) &&
               !decode_execution_async_frontier_observations(extra_results),
           "trailing bytes cannot be ignored");
    for (const auto status : {-1, 1000, 600}) {
        auto invalid = observations;
        invalid[0].status = status;
        invalid[0].is_retired = true;
        expect(!encode_execution_async_frontier_observations(invalid),
               "encoder rejects out-of-domain status and NOT_READY with retirement");
        auto bytes = *results;
        put_u32(bytes, 24, std::bit_cast<std::uint32_t>(status));
        put_u32(bytes, 28, 1);
        expect(!decode_execution_async_frontier_observations(bytes),
               "decoder independently rejects invalid status and retirement combinations");
    }
    auto invalid_retired = *results;
    put_u32(invalid_retired, 28, 2);
    expect(!decode_execution_async_frontier_observations(invalid_retired),
           "physical-retirement flag must be exactly zero or one");
    for (const auto invalid_key :
         {ExecutionAsyncFrontierKey{0, 17}, ExecutionAsyncFrontierKey{7, 0},
          ExecutionAsyncFrontierKey{8, 17}, ExecutionAsyncFrontierKey{7, 18}}) {
        auto invalid_keys = keys;
        auto invalid_observations = observations;
        invalid_keys[0] = invalid_key;
        invalid_observations[0].key = invalid_key;
        expect(!encode_execution_async_frontier_batch(invalid_keys) &&
                   !encode_execution_async_frontier_observations(invalid_observations),
               "producer independently rejects zero and duplicate identities");
    }
    const ExecutionSourceContinuation scope{.caller = {1, 2}, .group = 3, .ordinary_sequence = 4};
    for (const auto operation :
         {ExecutionOpcode::kQueryAsyncFrontiers, ExecutionOpcode::kReleaseAsyncFrontiers}) {
        expect(normalize_execution_request(operation, *batch).has_value() &&
                   !encode_execution_source_continuation(scope, operation, *batch),
               "ordinary metadata cannot borrow source-continuation authority");
        auto forged = encode_execution_source_continuation(scope, ExecutionOpcode::kLaunch, *batch);
        expect(forged.has_value(), "construct an otherwise valid source envelope");
        put_u32(*forged, 56, static_cast<std::uint32_t>(operation));
        expect(!normalize_execution_request(ExecutionOpcode::kSourceContinuation, *forged),
               "forged SCN1 batch is rejected independently of the producer");
    }
}
void check_frames() {
    ExecutionWireMessage message{
        {ExecutionMessageKind::kRequest, ExecutionOpcode::kLaunch, 0, 17, 33, {41, 53}},
        {std::byte{3}, std::byte{7}}};
    auto encoded = encode_execution_message(message);
    expect(encoded && encoded->size() == kExecutionHeaderBytes + 2,
           "encode fixed explicit header and body");
    const auto header = decode_execution_header(std::span{*encoded}.first(kExecutionHeaderBytes));
    expect(header && header->session == 17 && header->sequence == 33 &&
               header->incarnation[1] == 53 && header->payload_bytes == 2 &&
               header->operation == ExecutionOpcode::kLaunch,
           "round-trip identity and exact sizes");
    expect(!decode_execution_header(std::span{*encoded}.first(kExecutionHeaderBytes - 1)) &&
               !decode_execution_header(std::span{*encoded}.first(kExecutionHeaderBytes), 1),
           "partial or oversized header rejects");
    for (std::size_t offset : {std::size_t{0}, std::size_t{4}, std::size_t{6}, std::size_t{8}}) {
        auto invalid = *encoded;
        invalid[offset] = std::byte{255};
        expect(!decode_execution_header(std::span{invalid}.first(kExecutionHeaderBytes)),
               "bad magic/version/kind/opcode rejects");
    }
    expect(!encode_execution_message(message, 1),
           "producer cannot exceed the negotiated payload limit");
    message.header.operation = ExecutionOpcode::kLoadBinary;
    encoded = encode_execution_message(message);
    expect(
        encoded &&
            decode_execution_header(std::span{*encoded}.first(kExecutionHeaderBytes))->operation ==
                ExecutionOpcode::kLoadBinary,
        "owned binary intake has a distinct explicit operation, never a process pointer");
    for (const auto operation : {ExecutionOpcode::kGetContextFlags,
                                 ExecutionOpcode::kSetContextFlags,
                                 ExecutionOpcode::kMemory,
                                 ExecutionOpcode::kImportHostMemory,
                                 ExecutionOpcode::kHostTransfer,
                                 ExecutionOpcode::kReleaseHostMemory,
                                 ExecutionOpcode::kStagedHostTransfer,
                                 ExecutionOpcode::kQueryStagingIdle,
                                 ExecutionOpcode::kEnqueueHostFunction,
                                 ExecutionOpcode::kQueryHostFunction,
                                 ExecutionOpcode::kFinishHostFunction,
                                 ExecutionOpcode::kQueryAsyncFrontiers,
                                 ExecutionOpcode::kReleaseAsyncFrontiers,
                                 ExecutionOpcode::kCreateBlas,
                                 ExecutionOpcode::kDestroyBlas,
                                 ExecutionOpcode::kSetBlasAttribute,
                                 ExecutionOpcode::kGetBlasAttribute,
                                 ExecutionOpcode::kSgemm,
                                 ExecutionOpcode::kGemmEx,
                                 ExecutionOpcode::kStridedGemmEx,
                                 ExecutionOpcode::kCreateLtMetadata,
                                 ExecutionOpcode::kDestroyLtMetadata,
                                 ExecutionOpcode::kSetLtAttribute,
                                 ExecutionOpcode::kGetLtAttribute,
                                 ExecutionOpcode::kLtVersion,
                                 ExecutionOpcode::kCreateLt,
                                 ExecutionOpcode::kDestroyLt,
                                 ExecutionOpcode::kLtHeuristic,
                                 ExecutionOpcode::kLtAlgorithmCheck,
                                 ExecutionOpcode::kLtMatmul,
                                 ExecutionOpcode::kRuntimeMetadata,
                                 ExecutionOpcode::kCreatePriorityStream,
                                 ExecutionOpcode::kGetGlobal,
                                 ExecutionOpcode::kLoadModuleFile,
                                 ExecutionOpcode::kLoadModuleJit,
                                 ExecutionOpcode::kCreateEvent,
                                 ExecutionOpcode::kRecordEvent,
                                 ExecutionOpcode::kQueryEvent,
                                 ExecutionOpcode::kSynchronizeEvent,
                                 ExecutionOpcode::kDestroyEvent,
                                 ExecutionOpcode::kStreamWaitEvent,
                                 ExecutionOpcode::kEventElapsedTime,
                                 ExecutionOpcode::kWaitForeignEvent,
                                 ExecutionOpcode::kSetBlasWorkspace,
                                 ExecutionOpcode::kStridedSgemm,
                                 ExecutionOpcode::kCreateRand,
                                 ExecutionOpcode::kSetRandSeed,
                                 ExecutionOpcode::kDestroyRand,
                                 ExecutionOpcode::kDriverContextControl,
                                 ExecutionOpcode::kRuntimeContextControl,
                                 ExecutionOpcode::kCreateDnnMetadata,
                                 ExecutionOpcode::kDestroyDnnMetadata,
                                 ExecutionOpcode::kSetDnnTensor,
                                 ExecutionOpcode::kGetDnnTensor,
                                 ExecutionOpcode::kDnnTensorSize,
                                 ExecutionOpcode::kSetDnnActivation,
                                 ExecutionOpcode::kGetDnnActivation,
                                 ExecutionOpcode::kCreateDnn,
                                 ExecutionOpcode::kDestroyDnn,
                                 ExecutionOpcode::kSetDnnStream,
                                 ExecutionOpcode::kGetDnnStream,
                                 ExecutionOpcode::kDnnCompute,
                                 ExecutionOpcode::kDnnConvolutionFind,
                                 ExecutionOpcode::kSetDnnPooling,
                                 ExecutionOpcode::kGetDnnPooling,
                                 ExecutionOpcode::kDnnPoolingOutput,
                                 ExecutionOpcode::kDnnTensorForward,
                                 ExecutionOpcode::kSetDnnLrn,
                                 ExecutionOpcode::kSetDnnReduction,
                                 ExecutionOpcode::kDnnReductionQuery,
                                 ExecutionOpcode::kDnnReduction,
                                 ExecutionOpcode::kDnnBatchNormQuery,
                                 ExecutionOpcode::kDnnBatchNorm,
                                 ExecutionOpcode::kDnnDropoutStatesSize,
                                 ExecutionOpcode::kDnnDropout,
                                 ExecutionOpcode::kSetDnnOpTensor,
                                 ExecutionOpcode::kGetDnnOpTensor,
                                 ExecutionOpcode::kSetDnnTensorTransform,
                                 ExecutionOpcode::kDnnInitTransformDest,
                                 ExecutionOpcode::kDnnReorderFilterAndBias,
                                 ExecutionOpcode::kDnnLibraryInfo,
                                 ExecutionOpcode::kDnnBackwardConvolutionQuery,
                                 ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors,
                                 ExecutionOpcode::kSetDnnRnnData,
                                 ExecutionOpcode::kSetDnnSeqData,
                                 ExecutionOpcode::kGetDnnSeqData,
                                 ExecutionOpcode::kSetDnnRnn,
                                 ExecutionOpcode::kSetDnnAttn,
                                 ExecutionOpcode::kDnnRnnQuery,
                                 ExecutionOpcode::kSetDnnRnnClip,
                                 ExecutionOpcode::kDnnRnnCompute,
                                 ExecutionOpcode::kDnnBuildRnnDynamic,
                                 ExecutionOpcode::kDnnRnnWeightParams,
                                 ExecutionOpcode::kDnnAttnWeights,
                                 ExecutionOpcode::kPrepareDnnAttn,
                                 ExecutionOpcode::kDnnAttnCompute,
                                 ExecutionOpcode::kDnnBackendCreate,
                                 ExecutionOpcode::kDnnBackendDestroy,
                                 ExecutionOpcode::kDnnBackendInitialize,
                                 ExecutionOpcode::kDnnBackendFinalize,
                                 ExecutionOpcode::kDnnBackendPrepare,
                                 ExecutionOpcode::kDnnBackendSet,
                                 ExecutionOpcode::kDnnBackendGet,
                                 ExecutionOpcode::kDnnBackendContext,
                                 ExecutionOpcode::kPrepareDnnBackendExecute,
                                 ExecutionOpcode::kDnnBackendExecute,
                                 ExecutionOpcode::kFunctionControl,
                                 ExecutionOpcode::kGeam,
                                 ExecutionOpcode::kSgemv,
                                 ExecutionOpcode::kSgemmEx,
                                 ExecutionOpcode::kPointerMetadata,
                                 ExecutionOpcode::kNcclCall}) {
        message.header.operation = operation;
        const bool is_library =
            operation == ExecutionOpcode::kSgemm || operation == ExecutionOpcode::kGemmEx ||
            operation == ExecutionOpcode::kStridedGemmEx ||
            operation == ExecutionOpcode::kStridedSgemm || operation == ExecutionOpcode::kLtMatmul;
        expect(execution_allows_compute_replacement(operation) == is_library,
               "only integrated BLAS/Lt replacements accept private Driver status replies");
        expect(execution_requires_descriptor(operation) ==
                   (operation == ExecutionOpcode::kImportHostMemory ||
                    operation == ExecutionOpcode::kLoadModuleFile ||
                    operation == ExecutionOpcode::kLoadModuleJit),
               "only explicitly tagged backing capabilities require a descriptor");
        const auto frame = encode_execution_message(message);
        expect(frame && decode_execution_header(std::span{*frame}.first(kExecutionHeaderBytes))
                                ->operation == operation,
               "context metadata operations keep existing opcode identities intact");
    }
    expect(
        static_cast<std::uint32_t>(ExecutionOpcode::kCreateDnnMetadata) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kRuntimeContextControl) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnCompute) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kCreateDnnMetadata) + 11 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnConvolutionFind) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnConvolutionForward) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnPooling) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnConvolutionFind) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnTensorForward) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnPooling) + 3 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnLrn) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnTensorForward) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnReduction) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnLrn) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnReductionQuery) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnReduction) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnReduction) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnReductionQuery) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBatchNormQuery) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnReduction) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBatchNorm) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBatchNormQuery) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnDropoutStatesSize) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBatchNorm) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnDropout) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnDropoutStatesSize) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnOpTensor) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnDropout) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kGetDnnOpTensor) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnOpTensor) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnTensorTransform) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kGetDnnOpTensor) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnInitTransformDest) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnTensorTransform) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnReorderFilterAndBias) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnInitTransformDest) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnLibraryInfo) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnReorderFilterAndBias) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackwardConvolutionQuery) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnLibraryInfo) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackwardConvolutionQuery) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnnData) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors) +
                    1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kGetDnnSeqData) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnnData) + 2 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnn) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kGetDnnSeqData) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnAttn) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnn) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnQuery) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnAttn) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnnClip) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnQuery) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnCompute) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kSetDnnRnnClip) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBuildRnnDynamic) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnCompute) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnWeightParams) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBuildRnnDynamic) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnAttnWeights) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnRnnWeightParams) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kPrepareDnnAttn) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnAttnWeights) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnAttnCompute) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kPrepareDnnAttn) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendCreate) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnAttnCompute) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendDestroy) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendCreate) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendInitialize) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendDestroy) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendFinalize) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendInitialize) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendPrepare) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendFinalize) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendSet) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendPrepare) + 1 &&
            static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendGet) ==
                static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendSet) + 1,
        "DNN operations append without renumbering earlier wire identities");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kFunctionControl) ==
               static_cast<std::uint32_t>(ExecutionOpcode::kDnnBackendExecute) + 1,
           "function controls append without renumbering earlier opcodes");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kGeam) == 129 &&
               static_cast<std::uint32_t>(ExecutionOpcode::kGeam) ==
                   static_cast<std::uint32_t>(ExecutionOpcode::kFunctionControl) + 1,
           "GEAM appends without renumbering existing wire identities");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kPointerMetadata) == 130,
           "pointer metadata appends without renumbering GEAM");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kNcclCall) == 131 &&
               kExecutionNcclRole == 0x314c434eU,
           "NCCL carrier appends without renumbering pointer metadata or reusing a metadata role");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kSourceContinuation) == 132,
           "SCN1 appends one exact opcode without renumbering NCCL or ordinary operations");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kQueryStagingIdle) == 133,
           "private idle query appends without renumbering prior operations");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kEnqueueHostFunction) == 134 &&
               static_cast<std::uint32_t>(ExecutionOpcode::kQueryHostFunction) == 135 &&
               static_cast<std::uint32_t>(ExecutionOpcode::kFinishHostFunction) == 136,
           "host function CPU boundaries append without renumbering prior operations");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kQueryAsyncFrontiers) == 137 &&
               static_cast<std::uint32_t>(ExecutionOpcode::kReleaseAsyncFrontiers) == 138,
           "async observation batches append without renumbering prior operations");
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kSgemv) == 139 &&
               static_cast<std::uint32_t>(ExecutionOpcode::kSgemmEx) == 140,
           "exact original BLAS routes append without renumbering async observation batches");
    const ExecutionSourceContinuation scope{.caller = {1, 2}, .group = 3, .ordinary_sequence = 4};
    for (const auto operation : {ExecutionOpcode::kSgemv, ExecutionOpcode::kSgemmEx}) {
        const auto continuation = encode_execution_source_continuation(scope, operation, {});
        expect(continuation.has_value(), "original BLAS commands retain exact source continuation");
        const auto decoded =
            normalize_execution_request(ExecutionOpcode::kSourceContinuation, *continuation);
        expect(decoded && decoded->operation == operation && decoded->continuation &&
                   !execution_allows_compute_replacement(operation),
               "source continuation does not grant original library compute replacement");
    }
    expect(static_cast<std::uint32_t>(ExecutionOpcode::kUnifiedCopy) == 141,
           "generic UVA copies append without renumbering original BLAS operations");
    message.header.operation = ExecutionOpcode::kUnifiedCopy;
    expect(encode_execution_message(message).has_value(), "generic copy header accepted");
    expect(
        encode_execution_source_continuation(scope, ExecutionOpcode::kUnifiedCopy, {}).has_value(),
        "generic copy retains source continuation without admission-only success");
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kLoadModuleJit) == 142);
    message.header.operation = ExecutionOpcode::kLoadModuleJit;
    expect(encode_execution_message(message).has_value() &&
               execution_requires_descriptor(message.header.operation),
           "JIT module factory appends with immutable descriptor capability");
    expect(encode_execution_source_continuation(scope, ExecutionOpcode::kLoadModuleJit, {})
               .has_value(),
           "JIT module factory retains ordinary source continuation semantics");
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kLtAlgorithmCapability) == 143);
    message.header.operation = ExecutionOpcode::kLtAlgorithmCapability;
    expect(
        encode_execution_message(message).has_value() &&
            !execution_requires_descriptor(message.header.operation) &&
            encode_execution_source_continuation(scope, message.header.operation, {}).has_value(),
        "capability metadata appends without a descriptor or GPU grant");
    message.header.operation = static_cast<ExecutionOpcode>(144);
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kProcessMemory) == 144);
    expect(
        encode_execution_message(message).has_value() &&
            !execution_requires_descriptor(message.header.operation) &&
            !execution_allows_compute_replacement(message.header.operation) &&
            encode_execution_source_continuation(scope, message.header.operation, {}).has_value(),
        "process memory appends with ordinary continuation, not replacement or admission");
    auto cleanup = scope;
    cleanup.flags = kExecutionSourceContinuationCleanup;
    cleanup.cleanup_parent = 7;
    expect(!encode_execution_source_continuation(cleanup, message.header.operation, {}),
           "process memory cannot borrow host-cleanup-only authority");
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kGraphControl) == 145);
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kGraphLaunch) == 146);
    for (const auto operation : {ExecutionOpcode::kGraphControl, ExecutionOpcode::kGraphLaunch}) {
        message.header.operation = operation;
        expect(encode_execution_message(message).has_value() &&
                   !execution_requires_descriptor(operation) &&
                   !execution_allows_compute_replacement(operation) &&
                   !encode_execution_source_continuation(scope, operation, {}),
               "graph carriers append without source-continuation or replacement authority");
    }
    static_assert(static_cast<std::uint32_t>(ExecutionOpcode::kBindDnnProfile) == 147);
    message.header.operation = ExecutionOpcode::kBindDnnProfile;
    expect(encode_execution_message(message).has_value() &&
               !execution_requires_descriptor(message.header.operation) &&
               !execution_allows_compute_replacement(message.header.operation) &&
               !encode_execution_source_continuation(scope, message.header.operation, {}),
           "profile binding has no descriptor, replacement or source-continuation authority");
    message.header.operation = static_cast<ExecutionOpcode>(
        static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute) + 1U);
    expect(!encode_execution_message(message), "operation above profile boundary rejects");
}
void check_payloads() {
    expect(
        execution_context_control_is_query(ExecutionContextControl::kGetLimit) &&
            execution_context_control_is_query(ExecutionContextControl::kGetApiVersion) &&
            execution_context_control_is_query(ExecutionContextControl::kGetAffinity) &&
            !execution_context_control_is_query(ExecutionContextControl::kSetLimit) &&
            !execution_context_control_is_query(ExecutionContextControl::kResetPersistingCache) &&
            !execution_context_control_is_query(static_cast<ExecutionContextControl>(UINT32_MAX)),
        "read-only context queries do not grant mutation priority or GPU-stop authority");
    ExecutionPayloadWriter writer(64);
    expect(writer.write_u32(0x78563412) &&
               writer.write_u64(std::numeric_limits<std::uint64_t>::max()) &&
               writer.write_string("owned") && writer.write_blob({}),
           "encode bounded scalar/text/empty data");
    auto bytes = writer.take();
    expect(bytes[0] == std::byte{0x12} && bytes[3] == std::byte{0x78},
           "wire order is explicitly little endian");
    ExecutionPayloadReader reader(bytes);
    std::uint32_t small{};
    std::uint64_t large{};
    std::string_view text;
    std::span<const std::byte> empty;
    expect(reader.read_u32(small) && small == 0x78563412 && reader.read_u64(large) &&
               large == std::numeric_limits<std::uint64_t>::max() && reader.read_string(text) &&
               text == "owned" && reader.read_blob(empty) && empty.empty() && reader.empty() &&
               !reader.read_u32(small),
           "bounded complete decoding");
    ExecutionPayloadWriter tiny(4);
    expect(!tiny.write_u64(9) && tiny.write_u32(9) && !tiny.write_u32(0),
           "failed append does not consume capacity");
    std::array invalid{std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}};
    ExecutionPayloadReader truncated(invalid);
    expect(!truncated.read_blob(empty), "untrusted blob lengths cannot escape the owned frame");
}
void check_capacity_reservation() {
    ExecutionPayloadWriter empty(20);
    expect(empty.reserve_capacity(20), "reserve complete fixed response before original mutation");
    const auto reserved = empty.take();
    expect(reserved.empty() && reserved.capacity() >= 20,
           "reservation creates storage but no wire bytes");
    ExecutionPayloadWriter writer(20);
    expect(writer.write_u32(0x87654321) && writer.reserve_capacity(20) &&
               writer.reserve_capacity(0) && writer.reserve_capacity(4) &&
               !writer.reserve_capacity(21) && !writer.reserve_capacity(SIZE_MAX) &&
               writer.write_u64(0x0102030405060708) && writer.write_u64(UINT64_MAX) &&
               !writer.write_u32(0),
           "absolute reservation preserves prefix, honors limit, and leaves existing writes "
           "unchanged");
    const auto bytes = writer.take();
    ExecutionPayloadReader reader(bytes);
    std::uint32_t prefix{};
    std::uint64_t first{}, second{};
    expect(bytes.size() == 20 && bytes.capacity() >= 20 && reader.read_u32(prefix) &&
               prefix == 0x87654321 && reader.read_u64(first) && first == 0x0102030405060708 &&
               reader.read_u64(second) && second == UINT64_MAX && reader.empty(),
           "failed and shrinking reservations cannot alter encoded contents");
    ExecutionPayloadWriter zero(0);
    expect(zero.reserve_capacity(0) && !zero.reserve_capacity(1) && zero.take().empty(),
           "zero payload bound rejects nonzero capacity without content");
    ExecutionPayloadWriter composed(12);
    expect(composed.write_u32(17) && composed.reserve_additional_capacity(8) &&
               composed.reserve_additional_capacity(0) &&
               !composed.reserve_additional_capacity(9) &&
               !composed.reserve_additional_capacity(SIZE_MAX) && composed.write_u64(29) &&
               !composed.reserve_additional_capacity(1),
           "additional reservation checks total bounds and overflow without publishing bytes");
    const auto combined = composed.take();
    ExecutionPayloadReader composed_reader(combined);
    expect(composed_reader.read_u32(prefix) && prefix == 17 && composed_reader.read_u64(first) &&
               first == 29 && composed_reader.empty(),
           "failed additional reservations preserve the entire composition");
}
void check_context_bootstrap() {
    expect(decode_execution_context_bootstrap({}) == ExecutionContextBootstrap{},
           "empty legacy request remains isolated without generation proof");
    ExecutionPayloadWriter legacy;
    expect(legacy.write_u32(0xfedcba98), "encode unchanged legacy flags");
    expect(decode_execution_context_bootstrap(legacy.take()) ==
               ExecutionContextBootstrap{ExecutionContextPurpose::kLegacyIsolated, 0xfedcba98, 0},
           "flags-only input never opts into a primary cohort");
    for (const auto purpose :
         {ExecutionContextPurpose::kExplicitUser, ExecutionContextPurpose::kProcessPrimary,
          ExecutionContextPurpose::kRuntimePrimary}) {
        for (const auto generation : {std::uint64_t{1}, std::uint64_t{0x1020304050607080},
                                      std::numeric_limits<std::uint64_t>::max()}) {
            const ExecutionContextBootstrap descriptor{purpose, 0x76543210, generation};
            const auto bytes = encode_execution_context_bootstrap(descriptor);
            expect(bytes && bytes->size() == 24 &&
                       decode_execution_context_bootstrap(*bytes) == descriptor,
                   "fixed explicit little-endian purpose/flags/full-width generation round trip");
            expect((*bytes)[0] == std::byte{'C'} && (*bytes)[1] == std::byte{'T'} &&
                       (*bytes)[2] == std::byte{'X'} && (*bytes)[3] == std::byte{'1'} &&
                       (*bytes)[8] == std::byte{0x10} && (*bytes)[11] == std::byte{0x76},
                   "record layout is independent of native struct padding and host ABI");
            for (std::size_t length = 1; length < bytes->size(); ++length)
                if (length != 4)
                    expect(!decode_execution_context_bootstrap(std::span{*bytes}.first(length)),
                           "truncated typed request cannot publish a context purpose");
            auto invalid = *bytes;
            invalid[12] = std::byte{1};
            expect(!decode_execution_context_bootstrap(invalid), "reserved bits reject");
            for (const auto bad_purpose : {0U, 4U, UINT32_MAX}) {
                invalid = *bytes;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    invalid[4 + byte] =
                        std::byte{static_cast<unsigned char>(bad_purpose >> (byte * 8))};
                expect(!decode_execution_context_bootstrap(invalid) &&
                           !encode_execution_context_bootstrap(
                               {static_cast<ExecutionContextPurpose>(bad_purpose), 0, generation}),
                       "unknown/legacy purpose is not a valid typed generation");
            }
            invalid = *bytes;
            std::fill(invalid.begin() + 16, invalid.end(), std::byte{});
            expect(!decode_execution_context_bootstrap(invalid) &&
                       !encode_execution_context_bootstrap({purpose, 0, 0}),
                   "zero generation is not a typed factory identity");
            std::vector<std::byte> trailing(bytes->begin(), bytes->end());
            trailing.push_back(std::byte{});
            expect(!decode_execution_context_bootstrap(trailing), "trailing bytes reject");
            for (const auto role :
                 {kExecutionHostOwnerRole, kExecutionStagingOwnerRole, kExecutionLtMetadataRole,
                  kExecutionDnnMetadataRole, kExecutionPointerMetadataRole,
                  kExecutionProcessMemoryRole, kExecutionNcclRole}) {
                invalid = *bytes;
                for (std::size_t byte = 0; byte < 4; ++byte)
                    invalid[byte] = std::byte{static_cast<unsigned char>(role >> (byte * 8))};
                expect(!decode_execution_context_bootstrap(invalid),
                       "helper roles cannot masquerade as typed ordinary contexts");
            }
            const auto wrapped = encode_execution_source_continuation(
                {0, {17, 19}, 23, 1, 0, 0}, ExecutionOpcode::kHello, *bytes);
            const auto normalized = wrapped ? normalize_execution_request(
                                                  ExecutionOpcode::kSourceContinuation, *wrapped)
                                            : std::nullopt;
            expect(normalized && normalized->continuation &&
                       decode_execution_context_bootstrap(normalized->payload) == descriptor,
                   "cold source continuation retains exact typed bootstrap without rewriting it");
        }
    }
    expect(!decode_execution_context_bootstrap(std::array<std::byte, 8>{}),
           "eight-byte helper/unknown role is never ordinary context metadata");
}

void check_bootstrap_ownership() {
    for (const auto ownership :
         {ExecutionBootstrapOwnership::kUnproven, ExecutionBootstrapOwnership::kNone,
          ExecutionBootstrapOwnership::kRetained, ExecutionBootstrapOwnership::kMetadataRetained}) {
        const auto encoded = encode_execution_bootstrap_ownership(ownership);
        if (!encoded) throw std::runtime_error("bootstrap ownership must encode");
        expect(encoded && decode_execution_bootstrap_ownership(*encoded) == ownership,
               "versioned bootstrap ownership round trip");
        for (std::size_t bytes = 0; bytes < encoded->size(); ++bytes)
            expect(!decode_execution_bootstrap_ownership(std::span{*encoded}.first(bytes)),
                   "missing/truncated metadata never proves empty ownership");
        auto invalid = *encoded;
        invalid[3] = std::byte{'2'};
        expect(!decode_execution_bootstrap_ownership(invalid), "unknown record version rejects");
        invalid = *encoded;
        invalid[7] = std::byte{255};
        expect(!decode_execution_bootstrap_ownership(invalid), "unknown ownership rejects");
    }
    expect(
        !encode_execution_bootstrap_ownership(static_cast<ExecutionBootstrapOwnership>(UINT32_MAX)),
        "producer cannot invent an ownership state");
}
void check_source_continuation() {
    constexpr std::size_t limit = 128;
    const ExecutionSourceContinuation scope{0, {17, 19}, 23, 29, 31, 0};
    const std::vector<std::byte> body(limit, std::byte{0x5a});
    const auto encoded =
        encode_execution_source_continuation(scope, ExecutionOpcode::kUpload, body, limit);
    expect(encoded && encoded->size() == limit + 64,
           "SCN1 prefix is exactly 64 bytes without reducing ordinary payload capacity");
    ExecutionPayloadReader literal(*encoded);
    std::uint32_t magic{}, flags{}, operation{};
    std::array<std::uint64_t, 6> fields{};
    std::span<const std::byte> decoded;
    expect(literal.read_u32(magic) && magic == 0x314e4353 && literal.read_u32(flags) && !flags,
           "independent literal continuation magic and flags");
    for (auto& field : fields) expect(literal.read_u64(field), "literal continuation field");
    expect(fields == std::array<std::uint64_t, 6>{17, 19, 23, 29, 31, 0} &&
               literal.read_u32(operation) && operation == 7 && literal.read_blob(decoded) &&
               literal.empty() && std::ranges::equal(decoded, body),
           "literal caller/group/ordinary/origin/parent/inner fields preserve unchanged bytes");
    const auto view =
        normalize_execution_request(ExecutionOpcode::kSourceContinuation, *encoded, limit);
    expect(view && view->operation == ExecutionOpcode::kUpload && view->continuation &&
               view->continuation->caller == scope.caller &&
               view->continuation->ordinary_sequence == 29 &&
               std::ranges::equal(view->payload, body),
           "normalization returns only a borrowing effective view, never native authority");
    ExecutionWireMessage message{
        {ExecutionMessageKind::kRequest, ExecutionOpcode::kSourceContinuation, 0, 37, 41, {43, 47}},
        *encoded};
    const auto frame = encode_execution_message(message, limit);
    expect(frame && decode_execution_header(std::span{*frame}.first(kExecutionHeaderBytes), limit),
           "request continuation has exactly the additional bounded wire overhead");
    message.header.kind = ExecutionMessageKind::kReply;
    expect(!encode_execution_message(message, limit),
           "reply capacity never gains wrapper overhead");
    auto exact_reply = message;
    exact_reply.payload.resize(limit);
    const auto reply_frame = encode_execution_message(exact_reply, limit);
    expect(reply_frame &&
               decode_execution_header(std::span{*reply_frame}.first(kExecutionHeaderBytes), limit),
           "outer SCN1 reply correlation preserves the entire unchanged ordinary reply capacity");
    message.header.kind = ExecutionMessageKind::kRequest;
    message.header.operation = ExecutionOpcode::kUpload;
    expect(!encode_execution_message(message, limit), "plain requests keep their original cap");
    for (std::size_t size = 0; size < encoded->size(); ++size)
        expect(!normalize_execution_request(ExecutionOpcode::kSourceContinuation,
                                            std::span{*encoded}.first(size), limit),
               "every short prefix/body rejects without exposing an inner operation");
    for (const auto offset :
         {std::size_t{0}, std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{24},
          std::size_t{32}, std::size_t{48}, std::size_t{56}, std::size_t{60}}) {
        auto invalid = *encoded;
        invalid[offset] = offset == 0 || offset == 4 || offset == 48 || offset == 56 || offset == 60
                              ? std::byte{0xff}
                              : std::byte{0};
        expect(!normalize_execution_request(ExecutionOpcode::kSourceContinuation, invalid, limit),
               "noncanonical scope, flags, operation, parent and lengths reject");
    }
    auto trailing = *encoded;
    trailing.push_back(std::byte{});
    expect(
        !normalize_execution_request(ExecutionOpcode::kSourceContinuation, trailing, limit) &&
            !normalize_execution_request(ExecutionOpcode::kSourceContinuation, *encoded,
                                         limit - 1) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kNcclCall, {}) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kQueryStagingIdle, {}) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kQueryHostFunction, {}) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kFinishHostFunction,
                                                  {}) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kSourceContinuation,
                                                  *encoded) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kUpload, {},
                                                  UINT32_MAX) &&
            !encode_execution_source_continuation(scope, ExecutionOpcode::kUpload, {}, SIZE_MAX),
        "recursion/native calls, trailing data and overflowing envelope limits reject");
    const auto callback = encode_execution_source_continuation(
        scope, ExecutionOpcode::kEnqueueHostFunction, body, limit);
    expect(callback &&
               normalize_execution_request(ExecutionOpcode::kSourceContinuation, *callback, limit)
                       ->operation == ExecutionOpcode::kEnqueueHostFunction,
           "only source host-function admission joins an ordinary continuation");
    ExecutionPayloadWriter native_bootstrap;
    expect(native_bootstrap.write_u32(kExecutionNcclRole) && native_bootstrap.write_u32(0) &&
               !encode_execution_source_continuation(scope, ExecutionOpcode::kHello,
                                                     native_bootstrap.take()),
           "ordinary continuation cannot bootstrap another native NCCL carrier");
    auto cleanup = scope;
    cleanup.flags = kExecutionSourceContinuationCleanup;
    cleanup.cleanup_parent = scope.ordinary_sequence;
    cleanup.ordinary_sequence = 1;
    cleanup.origin_source = 0;
    ExecutionPayloadWriter resource;
    expect(resource.write_u64(53), "exact cleanup resource");
    const auto resource_bytes = resource.take();
    expect(
        encode_execution_source_continuation(cleanup, ExecutionOpcode::kReleaseHostMemory,
                                             resource_bytes)
                .has_value() &&
            !encode_execution_source_continuation(cleanup, ExecutionOpcode::kAllocate,
                                                  resource_bytes) &&
            !encode_execution_source_continuation(cleanup, ExecutionOpcode::kReleaseHostMemory, {}),
        "independent cleanup sequence permits only exact nonzero host-release payload syntax");
    cleanup.cleanup_parent = 0;
    expect(!encode_execution_source_continuation(cleanup, ExecutionOpcode::kReleaseHostMemory,
                                                 resource_bytes),
           "cleanup without an import parent cannot encode");
    const auto plain = normalize_execution_request(ExecutionOpcode::kUpload, body, limit);
    expect(plain && !plain->continuation && std::ranges::equal(plain->payload, body),
           "unwrapped ordinary operations retain their original view and capacity");
}
void check_replies() {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.status = -99;
    reply.original_status = -99;
    reply.id = 31;
    reply.address = 0x1000;
    reply.parameter_bytes = 24;
    reply.arguments = {{0, 8}, {8, 8}, {16, 4}, {20, 4}};
    reply.data = {std::byte{7}};
    auto encoded = encode_execution_reply(reply);
    expect(encoded.has_value(), "encode native reply");
    auto parsed = decode_execution_reply(*encoded);
    expect(parsed && parsed->status == -99 && parsed->original_status == -99 &&
               parsed->arguments.size() == 4 && parsed->data == reply.data && parsed->id == 31,
           "preserve signed opaque SDK result and owned metadata/data");
    const std::array invalid_arguments{
        std::vector<ExecutionWireArgument>{{0, 0}},
        std::vector<ExecutionWireArgument>{{24, 1}},
        std::vector<ExecutionWireArgument>{{23, 2}},
        std::vector<ExecutionWireArgument>{{UINT64_MAX, 1}},
        std::vector<ExecutionWireArgument>{{0, 8}, {7, 4}},
        std::vector<ExecutionWireArgument>{{8, 4}, {0, 4}},
    };
    for (const auto& arguments : invalid_arguments) {
        auto malformed = reply;
        malformed.arguments = arguments;
        expect(!encode_execution_reply(malformed),
               "reply encoder rejects empty, out-of-range, overlapping or unordered arguments");
    }
    auto empty_extent = reply;
    empty_extent.parameter_bytes = 0;
    empty_extent.arguments = {{0, 8}};
    expect(!encode_execution_reply(empty_extent), "zero extent cannot contain an argument");
    auto maximum_extent = reply;
    maximum_extent.parameter_bytes = UINT64_MAX;
    maximum_extent.arguments = {{UINT64_MAX - 1, 1}};
    const auto maximum_wire = encode_execution_reply(maximum_extent);
    expect(maximum_wire && decode_execution_reply(*maximum_wire),
           "a final argument ending at the maximum extent remains valid");
    encoded->push_back(std::byte{0});
    expect(!decode_execution_reply(*encoded), "trailing bytes reject");
    for (std::size_t length = 0; length < 44; ++length)
        expect(!decode_execution_reply(std::span{*encoded}.first(length)),
               "partial reply cannot publish success");
    reply.disposition = ExecutionWireDisposition::kRejected;
    reply.status = 1;
    reply.original_status.reset();
    reply.arguments.clear();
    reply.parameter_bytes = 0;
    reply.data.clear();
    encoded = encode_execution_reply(reply);
    parsed = decode_execution_reply(*encoded);
    expect(parsed && !parsed->original_status,
           "preparation rejection is not an invented original SDK status");
    reply.disposition = ExecutionWireDisposition::kComputeReplacement;
    encoded = encode_execution_reply(reply);
    expect(encoded.has_value(), "encode distinct actual compute replacement provenance");
    parsed = decode_execution_reply(*encoded);
    expect(parsed && parsed->disposition == ExecutionWireDisposition::kComputeReplacement &&
               !parsed->original_status && parsed->id == 31 && parsed->status == 1,
           "replacement preserves actual Driver result without invented cuBLAS call");
    reply.original_status = 0;
    expect(!encode_execution_reply(reply), "replacement cannot claim original vendor status");
    (*encoded)[8] = std::byte{1};
    expect(!decode_execution_reply(*encoded), "reject forged original status on replacement");
    reply.original_status.reset();
    reply.id = 0;
    expect(!encode_execution_reply(reply), "replacement requires actual command identity");

    reply = {};
    reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
    reply.id = 37;
    reply.address = 53;
    encoded = encode_execution_reply(reply);
    expect(encoded.has_value(), "encode copied admission with its exact command/frontier");
    parsed = decode_execution_reply(*encoded);
    expect(parsed && parsed->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
               parsed->status == 0 && !parsed->original_status && parsed->id == 37 &&
               parsed->address == 53 && !parsed->parameter_bytes && parsed->arguments.empty() &&
               parsed->data.empty(),
           "admission never claims an original SDK result or a physical completion");
    for (unsigned int scenario = 0; scenario < 7; ++scenario) {
        auto invalid = reply;
        switch (scenario) {
            case 0:
                invalid.status = 1;
                break;
            case 1:
                invalid.original_status = 0;
                break;
            case 2:
                invalid.id = 0;
                break;
            case 3:
                invalid.address = 0;
                break;
            case 4:
                invalid.parameter_bytes = 8;
                break;
            case 5:
                invalid.arguments = {{0, 8}};
                break;
            default:
                invalid.data = {std::byte{1}};
                break;
        }
        expect(!encode_execution_reply(invalid), "invalid admission cannot encode");
    }
    for (const auto offset : {4U, 8U, 32U}) {
        auto invalid = *encoded;
        invalid[offset] = std::byte{1};
        expect(!decode_execution_reply(invalid), "forged admission status/provenance/extent");
    }
    for (const auto offset : {16U, 24U}) {
        auto invalid = *encoded;
        for (unsigned int byte = 0; byte < 8; ++byte) invalid[offset + byte] = std::byte{};
        expect(!decode_execution_reply(invalid), "zero command/frontier admission rejects");
    }
    auto invalid = *encoded;
    invalid[44] = std::byte{1};
    invalid.push_back(std::byte{1});
    expect(!decode_execution_reply(invalid), "admission cannot carry hidden response data");
    reply = {};
    reply.disposition = ExecutionWireDisposition::kAdmissionBusy;
    encoded = encode_execution_reply(reply);
    std::vector<std::byte> busy_literal(48);
    busy_literal[0] = std::byte{5};
    expect(encoded && *encoded == busy_literal,
           "appended busy identity has exactly the empty carrier");
    parsed = decode_execution_reply(*encoded);
    expect(parsed && parsed->disposition == ExecutionWireDisposition::kAdmissionBusy &&
               !parsed->status && !parsed->original_status && !parsed->id && !parsed->address &&
               !parsed->parameter_bytes && parsed->arguments.empty() && parsed->data.empty(),
           "busy transfers no SDK, admission, identity or completion authority");
    for (unsigned int scenario = 0; scenario < 7; ++scenario) {
        auto forged = reply;
        switch (scenario) {
            case 0:
                forged.status = 1;
                break;
            case 1:
                forged.original_status = 0;
                break;
            case 2:
                forged.id = 1;
                break;
            case 3:
                forged.address = 1;
                break;
            case 4:
                forged.parameter_bytes = 8;
                break;
            case 5:
                forged.parameter_bytes = 8;
                forged.arguments = {{0, 8}};
                break;
            default:
                forged.data = {std::byte{1}};
                break;
        }
        expect(!encode_execution_reply(forged), "busy cannot encode fabricated authority");
        forged.disposition = ExecutionWireDisposition::kComplete;
        auto wire = encode_execution_reply(forged);
        expect(wire.has_value(), "malformed busy test retains otherwise valid framing");
        (*wire)[0] = std::byte{5};
        expect(!decode_execution_reply(*wire), "busy decode rejects fabricated fields before use");
    }
    ExecutionPayloadWriter tail;
    expect(tail.write_u32(kExecutionAsyncLaunchMagic) &&
               tail.take() == std::vector<std::byte>{std::byte{'A'}, std::byte{'S'}, std::byte{'Y'},
                                                     std::byte{'1'}},
           "ordinary async opt-in is exactly the versioned four-byte ASY1 tail");
}
}  // namespace
int main() {
    try {
        check_dnn_descriptor_observations();
        check_frames();
        check_unified_copy();
        check_async_frontier_batches();
        check_payloads();
        check_capacity_reservation();
        check_context_bootstrap();
        check_bootstrap_ownership();
        check_source_continuation();
        check_replies();
        std::cout << "Execution wire protocol checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
