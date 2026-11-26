#include "execution_dnn_profile_fixture.h"
#include "glimmer/control/execution_dnn_legacy_protocol.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cudnn.h>

#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_legacy_native_calls();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define LEGACY_APIS(X)                              \
    X(cudnnOpsInferVersionCheck)                    \
    X(cudnnOpsTrainVersionCheck)                    \
    X(cudnnCnnInferVersionCheck)                    \
    X(cudnnCnnTrainVersionCheck)                    \
    X(cudnnAdvInferVersionCheck)                    \
    X(cudnnAdvTrainVersionCheck)                    \
    X(cudnnCreateAlgorithmDescriptor)               \
    X(cudnnSetAlgorithmDescriptor)                  \
    X(cudnnDestroyAlgorithmDescriptor)              \
    X(cudnnSetRNNDescriptor_v6)                     \
    X(cudnnSetRNNMatrixMathType)                    \
    X(cudnnGetRNNMatrixMathType)                    \
    X(cudnnSetRNNBiasMode)                          \
    X(cudnnGetRNNBiasMode)                          \
    X(cudnnRNNSetClip)                              \
    X(cudnnGetRNNWorkspaceSize)                     \
    X(cudnnGetRNNTrainingReserveSize)               \
    X(cudnnGetRNNParamsSize)                        \
    X(cudnnGetRNNLinLayerMatrixParams)              \
    X(cudnnGetRNNLinLayerBiasParams)                \
    X(cudnnSetRNNAlgorithmDescriptor)               \
    X(cudnnGetRNNForwardInferenceAlgorithmMaxCount) \
    X(cudnnRNNForwardTraining)                      \
    X(cudnnRNNBackwardData)                         \
    X(cudnnRNNBackwardWeights)

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr CUdeviceptr kAddress = 0x100000;
constexpr std::size_t kBytes = 4096;
constexpr unsigned int kCases = 14;
static_assert(CUDNN_MAJOR == 8);

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "exact public cuDNN success");
}
void driver_success(CUresult status) {
    expect(status == CUDA_SUCCESS, "exact public Driver success");
}
void* address(std::uintptr_t offset) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): owned logical address, never CPU-dereferenced.
    return reinterpret_cast<void*>(kAddress + offset);
}
struct Api {
// NOLINTNEXTLINE(bugprone-macro-parentheses): name declares a member, not an expression.
#define FIELD(name) decltype(&::name) name{&::name};
    LEGACY_APIS(FIELD)
#undef FIELD
};
Api lookup(const char* path, bool is_cached) {
    auto* library = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
    expect(library != nullptr, "load exact approved independent CPU provider");
    Api api;
#define LOOKUP(name)                                                                              \
    do {                                                                                          \
        auto* symbol = ::dlsym(library, #name);                                                   \
        expect(                                                                                   \
            symbol == reinterpret_cast<void*>(&::name) && symbol == ::dlsym(RTLD_DEFAULT, #name), \
            "every legacy explicit/global symbol is the owned wrapper");                          \
        if (is_cached) std::memcpy(&api.name, &symbol, sizeof(symbol));                           \
    } while (false);
    LEGACY_APIS(LOOKUP)
#undef LOOKUP
    expect(::dlclose(library) == 0, "release explicit lookup reference");
    return api;
}
struct Resources {
    CUcontext context{};
    CUdeviceptr memory{};
    cudnnHandle_t handle{};
    cudnnRNNDescriptor_t rnn{};
    std::array<cudnnTensorDescriptor_t, 2> tensors{};
    std::array<cudnnFilterDescriptor_t, 2> filters{};
    cudnnAlgorithmDescriptor_t algorithm{};

    explicit Resources(const Api& api) {
        driver_success(cuCtxCreate(&context, 0, 0));
        success(cudnnCreate(&handle));
        success(cudnnCreateRNNDescriptor(&rnn));
        for (auto& tensor : tensors) success(cudnnCreateTensorDescriptor(&tensor));
        for (auto& filter : filters) success(cudnnCreateFilterDescriptor(&filter));
        success(api.cudnnCreateAlgorithmDescriptor(&algorithm));
        driver_success(cuMemAlloc(&memory, kBytes));
        expect(memory == kAddress, "owned allocation preserves synthetic device address");
    }
    void close(const Api& api) const {
        // Parent edges retire before their shallow AlgorithmDescriptor target.
        success(cudnnDestroyRNNDescriptor(rnn));
        success(api.cudnnDestroyAlgorithmDescriptor(algorithm));
        for (auto filter : filters) success(cudnnDestroyFilterDescriptor(filter));
        for (auto tensor : tensors) success(cudnnDestroyTensorDescriptor(tensor));
        success(cudnnDestroy(handle));
        driver_success(cuMemFree(memory));
        driver_success(cuCtxDestroy(context));
    }
    std::array<cudnnTensorDescriptor_t, 3> x() const {
        return {tensors[0], tensors[1], tensors[0]};
    }
    std::array<cudnnTensorDescriptor_t, 3> y() const {
        return {tensors[1], tensors[0], tensors[1]};
    }
};
cudnnStatus_t forward(const Api& api, const Resources& values) {
    const auto x = values.x();
    const auto y = values.y();
    return api.cudnnRNNForwardTraining(
        values.handle, values.rnn, 3, x.data(), address(0), values.tensors[0], address(128),
        values.tensors[1], address(192), values.filters[0], address(512), y.data(), address(64),
        values.tensors[1], address(256), values.tensors[0], address(320), address(2048), 192,
        address(3072), 256);
}
cudnnStatus_t parameter(const Api& api, const Resources& values, void** output,
                        bool is_bias = false) {
    // The matrix query intentionally aliases the source and mutated filter.
    const auto function =
        is_bias ? api.cudnnGetRNNLinLayerBiasParams : api.cudnnGetRNNLinLayerMatrixParams;
    return function(values.handle, values.rnn, 1, values.tensors[0], values.filters[0],
                    address(512), 3, values.filters[is_bias ? 1 : 0], output);
}

void normal_application(const Api& api) {
    for (auto check : {api.cudnnOpsInferVersionCheck, api.cudnnOpsTrainVersionCheck,
                       api.cudnnCnnInferVersionCheck, api.cudnnCnnTrainVersionCheck,
                       api.cudnnAdvInferVersionCheck, api.cudnnAdvTrainVersionCheck})
        success(check());
    Resources values(api);
    cudnnAlgorithm_t algorithm{};
    algorithm.algo.RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
    success(api.cudnnSetAlgorithmDescriptor(values.algorithm, algorithm));
    success(api.cudnnSetRNNDescriptor_v6(values.handle, values.rnn, 4, 2, nullptr,
                                         CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
                                         CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
    success(api.cudnnSetRNNMatrixMathType(values.rnn, CUDNN_TENSOR_OP_MATH));
    cudnnMathType_t math = CUDNN_FMA_MATH;
    success(api.cudnnGetRNNMatrixMathType(values.rnn, &math));
    expect(math == CUDNN_TENSOR_OP_MATH, "original math observation is published");
    success(api.cudnnSetRNNBiasMode(values.rnn, CUDNN_RNN_DOUBLE_BIAS));
    cudnnRNNBiasMode_t bias = CUDNN_RNN_NO_BIAS;
    success(api.cudnnGetRNNBiasMode(values.rnn, &bias));
    expect(bias == CUDNN_RNN_DOUBLE_BIAS, "original bias observation is published");
    success(api.cudnnRNNSetClip(values.handle, values.rnn, CUDNN_RNN_CLIP_MINMAX,
                                CUDNN_PROPAGATE_NAN, -2.0, 3.0));
    const auto x = values.x();
    const auto y = values.y();
    std::size_t bytes = 0x123456789abcdef0ULL;
    success(api.cudnnGetRNNWorkspaceSize(values.handle, values.rnn, 3, x.data(), &bytes));
    expect(bytes == 192, "workspace output is independent from containing capacity");
    bytes = 0x123456789abcdef0ULL;
    success(api.cudnnGetRNNTrainingReserveSize(values.handle, values.rnn, 3, x.data(), &bytes));
    expect(bytes == 256, "reserve output is independent from containing capacity");
    bytes = 0x123456789abcdef0ULL;
    success(api.cudnnGetRNNParamsSize(values.handle, values.rnn, values.tensors[0], &bytes,
                                      CUDNN_DATA_FLOAT));
    expect(bytes == 128, "legacy parameter-size scalar is exact");
    void* pointer = address(4000);
    success(parameter(api, values, &pointer));
    expect(pointer == address(528), "matrix subregion uses original input base plus offset");
    pointer = address(4000);
    success(parameter(api, values, &pointer, true));
    expect(pointer == address(544), "bias subregion retains independent destination identity");
    success(api.cudnnSetRNNAlgorithmDescriptor(values.handle, values.rnn, values.algorithm));
    int count = -719;
    success(api.cudnnGetRNNForwardInferenceAlgorithmMaxCount(values.handle, values.rnn, &count));
    expect(count == 4, "signed count storage publishes original result");
    success(forward(api, values));
    const std::array dy{values.tensors[0], values.tensors[0], values.tensors[1]};
    const std::array dx{values.tensors[1], values.tensors[1], values.tensors[0]};
    success(api.cudnnRNNBackwardData(
        values.handle, values.rnn, 3, y.data(), address(64), dy.data(), address(384),
        values.tensors[1], address(256), values.tensors[0], address(320), values.filters[0],
        address(512), values.tensors[0], address(128), values.tensors[1], address(192), dx.data(),
        address(448), values.tensors[0], address(640), values.tensors[1], address(704),
        address(2048), 192, address(3072), 256));
    success(api.cudnnRNNBackwardWeights(values.handle, values.rnn, 3, x.data(), address(0),
                                        values.tensors[0], address(128), y.data(), address(64),
                                        address(2048), 192, values.filters[0], address(512),
                                        address(3072), 256));
    expect(x == values.x() && y == values.y(), "descriptor input arrays were never mutated");
    values.close(api);
}

void hostile_application(const Api& api, unsigned int scenario) {
    Resources values(api);
    if (scenario == 1 || scenario == 2 || scenario == 3 || scenario == 10) {
        cudnnMathType_t math = CUDNN_FMA_MATH;
        const auto result = api.cudnnGetRNNMatrixMathType(values.rnn, &math);
        expect(result == (scenario == 1 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_INTERNAL_ERROR),
               "scalar native failure and malformed/proof failure remain distinct");
        expect(math == (scenario <= 2 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH),
               "only a validated actual scalar writeback changes caller bytes");
        if (scenario == 1) {
            success(api.cudnnOpsInferVersionCheck());
            values.close(api);
        } else {
            expect(api.cudnnGetRNNMatrixMathType(values.rnn, &math) == CUDNN_STATUS_NOT_INITIALIZED,
                   "quarantined scalar metadata cannot send another RPC");
        }
    } else if (scenario >= 4 && scenario <= 7) {
        void* pointer = address(4000);
        expect(parameter(api, values, &pointer) ==
                   (scenario == 4 ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_BAD_PARAM),
               "pointer proof failure differs from exact original failure");
        expect(pointer == (scenario == 5   ? address(528)
                           : scenario == 7 ? nullptr
                                           : address(4000)),
               "bounded write/null/no-write effects are independent from native status");
        expect(parameter(api, values, &pointer) == CUDNN_STATUS_NOT_INITIALIZED,
               "uncertain filter owner does not replay a mutation");
    } else if (scenario == 8 || scenario == 9) {
        expect(forward(api, values) ==
                   (scenario == 8 ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_INTERNAL_ERROR),
               "queued original error differs from malformed completion");
        if (scenario == 8) {
            success(forward(api, values));
            values.close(api);
        } else {
            expect(forward(api, values) == CUDNN_STATUS_NOT_INITIALIZED,
                   "uncertain queued completion never reissues");
        }
    } else if (scenario == 11 || scenario == 13) {
        expect(api.cudnnSetRNNMatrixMathType(values.rnn, CUDNN_TENSOR_OP_MATH) ==
                   (scenario == 11 ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_INTERNAL_ERROR),
               "issued mutation error requires explicit partial-state quarantine");
        cudnnMathType_t math = CUDNN_FMA_MATH;
        expect(api.cudnnGetRNNMatrixMathType(values.rnn, &math) == CUDNN_STATUS_NOT_INITIALIZED &&
                   math == CUDNN_FMA_MATH,
               "failed mutation quarantines without guessed rollback");
    } else {
        expect(scenario == 12, "known guarded-array scenario");
        const auto page = ::sysconf(_SC_PAGESIZE);
        expect(page > 0, "read system page size");
        auto* memory = ::mmap(nullptr, static_cast<std::size_t>(page), PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory != MAP_FAILED, "reserve unreadable descriptor-array page");
        const auto* guarded = static_cast<const cudnnTensorDescriptor_t*>(memory);
        std::size_t bytes = 0x123456789abcdef0ULL;
        expect(api.cudnnGetRNNWorkspaceSize(values.handle, values.rnn, 1, guarded, &bytes) ==
                       CUDNN_STATUS_BAD_PARAM &&
                   bytes == 0x123456789abcdef0ULL,
               "unreadable positive array rejects without a semantic RPC or output change");
        expect(api.cudnnGetRNNWorkspaceSize(values.handle, values.rnn, -1, guarded, &bytes) ==
                       CUDNN_STATUS_BAD_PARAM &&
                   bytes == 0x123456789abcdef0ULL,
               "negative count preserves nonnull presence without dereferencing ignored array");
        expect(api.cudnnGetRNNWorkspaceSize(values.handle, values.rnn, 3, nullptr, &bytes) ==
                       CUDNN_STATUS_BAD_PARAM &&
                   bytes == 0x123456789abcdef0ULL,
               "null array retains original count independently from captured length");
        expect(::munmap(memory, static_cast<std::size_t>(page)) == 0, "release owned guard page");
        const auto x = values.x();
        success(api.cudnnGetRNNWorkspaceSize(values.handle, values.rnn, 3, x.data(), &bytes));
        expect(bytes == 192, "local pointer rejection does not poison the next valid request");
        values.close(api);
    }
}

ExecutionWireReply complete(bool has_original = true) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    if (has_original) reply.original_status = CUDNN_STATUS_SUCCESS;
    return reply;
}
void output(ExecutionWireReply& reply, ExecutionDnnLegacyOutput value = {}) {
    ExecutionPayloadWriter payload;
    expect(write_execution_dnn_legacy_output(payload, value), "encode exact legacy output proof");
    reply.data = payload.take();
    expect(reply.data.size() == 32, "fixed 32-byte original-issued observation");
}

struct Peer {
    unsigned int scenario{}, metadata_calls{}, controls{}, computes{}, cleanups{}, contexts{},
        allocations{}, handles{}, destroys{}, next_id{100};
    std::uint64_t metadata{}, execution{};
    std::map<std::uint64_t, std::uint32_t> objects;
    std::vector<std::uint64_t> created;
    std::array<unsigned int, 6> version_checks{};

    void common(std::uint64_t handle, std::uint64_t owner, std::uint64_t descriptor) const {
        expect(handle == 41 && owner == metadata && created.size() == 6 &&
                   descriptor == created[0] && objects.at(descriptor) == 13,
               "legacy operation retains exact handle/metadata/RNN authority");
    }
    std::vector<std::uint64_t> x() const {
        return {created[1], created[2], created[1]};
    }
    std::vector<std::uint64_t> y() const {
        return {created[2], created[1], created[2]};
    }

    ExecutionWireReply metadata_reply(ExecutionPayloadReader& input) {
        ExecutionDnnLegacyMetadata value;
        expect(read_execution_dnn_legacy_metadata(input, value) && input.empty(),
               "exact typed legacy metadata payload");
        ++metadata_calls;
        auto reply = complete();
        ExecutionDnnLegacyOutput observed;
        if (value.kind == ExecutionDnnLegacyMetadataKind::kVersionCheck) {
            expect(!value.object && value.value >= 1 && value.value <= 6 && !value.has_output,
                   "six original version-check selectors are distinct from GetVersion");
            ++version_checks[value.value - 1];
        } else {
            expect(created.size() == 6 &&
                       value.object ==
                           created[value.kind == ExecutionDnnLegacyMetadataKind::kSetAlgorithm ? 5
                                                                                               : 0],
                   "metadata setter/getter uses the correct original descriptor kind");
            switch (value.kind) {
                case ExecutionDnnLegacyMetadataKind::kSetAlgorithm:
                    expect(value.value == 0 && !value.has_output,
                           "original by-value algorithm bits");
                    break;
                case ExecutionDnnLegacyMetadataKind::kSetMath:
                    expect(value.value == 1 && !value.has_output, "original math setter bits");
                    break;
                case ExecutionDnnLegacyMetadataKind::kSetBias:
                    expect(value.value == 2 && !value.has_output, "original bias setter bits");
                    break;
                case ExecutionDnnLegacyMetadataKind::kGetMath:
                    expect(value.has_output && value.seed == 3, "original scalar seed bytes");
                    observed = {1, 1};
                    break;
                case ExecutionDnnLegacyMetadataKind::kGetBias:
                    expect(value.has_output && !value.seed, "independent bias seed");
                    observed = {1, 2};
                    break;
                default:
                    throw std::runtime_error("unexpected metadata selector");
            }
        }
        output(reply, observed);
        if (scenario == 1 && value.kind == ExecutionDnnLegacyMetadataKind::kGetMath)
            reply.status = *reply.original_status = CUDNN_STATUS_BAD_PARAM;
        if (scenario == 2) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.original_status = CUDNN_STATUS_EXECUTION_FAILED;
            reply.parameter_bytes = 1;
        }
        if (scenario == 3) reply.id = 991;
        if (scenario == 10) output(reply, {2, 0, 0, 0, 0});
        if (scenario == 11 || scenario == 13) {
            reply.status = *reply.original_status = CUDNN_STATUS_EXECUTION_FAILED;
            reply.parameter_bytes = scenario == 11 ? 1 : 0;
        }
        return reply;
    }

    ExecutionWireReply control_reply(ExecutionPayloadReader& input) {
        ExecutionDnnLegacyControl value;
        expect(read_execution_dnn_legacy_control(input, value) && input.empty(),
               "exact selected legacy control arm");
        common(value.handle, value.metadata, value.descriptor);
        ++controls;
        auto reply = complete();
        ExecutionDnnLegacyOutput observed;
        switch (value.kind) {
            case ExecutionDnnLegacyControlKind::kSetup:
                expect(value.setup.hidden_size == 4 && value.setup.layers == 2 &&
                           !value.setup.dropout && !value.setup.input_mode &&
                           !value.setup.direction && value.setup.cell_mode == 2 &&
                           !value.setup.algorithm && !value.setup.math_precision,
                       "exact v6 setup ABI, not a v8 signature cast");
                break;
            case ExecutionDnnLegacyControlKind::kClip:
                expect(value.clip.mode == 1 && value.clip.nan == 1 &&
                           value.clip.lower_bits == std::bit_cast<std::uint64_t>(-2.0) &&
                           value.clip.upper_bits == std::bit_cast<std::uint64_t>(3.0),
                       "legacy clip preserves its handle and exact double bits");
                break;
            case ExecutionDnnLegacyControlKind::kWorkspaceSize:
            case ExecutionDnnLegacyControlKind::kReserveSize:
                expect(value.has_output && value.seed == 0x123456789abcdef0ULL,
                       "size output carries actual caller seed");
                if (scenario == 12 && controls <= 2) {
                    expect(value.sequence.descriptors.empty() &&
                               value.sequence.has_descriptors == (controls == 1 ? 1U : 0U) &&
                               value.sequence_length == (controls == 1 ? -1 : 3),
                           "invalid count/null preserves presence without guessed array values");
                    reply.status = *reply.original_status = CUDNN_STATUS_BAD_PARAM;
                    observed = {1, value.seed};
                } else {
                    expect(value.sequence_length == 3 && value.sequence.has_descriptors &&
                               value.sequence.descriptors == x(),
                           "ordered repeated sequence IDs");
                    observed = {1, value.kind == ExecutionDnnLegacyControlKind::kWorkspaceSize
                                       ? 192U
                                       : 256U};
                }
                break;
            case ExecutionDnnLegacyControlKind::kParameterSize:
                expect(value.input == created[1] && !value.data_type && value.has_output &&
                           value.seed == 0x123456789abcdef0ULL,
                       "original input/type/size ABI");
                observed = {1, 128};
                break;
            case ExecutionDnnLegacyControlKind::kMatrixParameter:
            case ExecutionDnnLegacyControlKind::kBiasParameter: {
                const auto& query = value.parameters;
                const bool is_bias = value.kind == ExecutionDnnLegacyControlKind::kBiasParameter;
                expect(query.input == created[1] && query.filter == created[3] &&
                           query.destination == created[is_bias ? 4 : 3] &&
                           query.pseudo_layer == 1 && query.linear_layer == 3 &&
                           query.weights.allocation == 81 && query.weights.offset == 512 &&
                           value.has_output && !value.seed,
                       "filter output alias and relative owned weight region, never raw pointer");
                observed = {2, 0, 1, is_bias ? 32U : 16U, 32};
                if (scenario >= 5 && scenario <= 7) {
                    reply.status = *reply.original_status = CUDNN_STATUS_BAD_PARAM;
                    reply.parameter_bytes = 1;
                    if (scenario == 6) observed = {};
                    if (scenario == 7) observed = {2};
                }
                break;
            }
            case ExecutionDnnLegacyControlKind::kSetAlgorithm:
                expect(value.algorithm == created[5], "genuine owned AlgorithmDescriptor binding");
                break;
            case ExecutionDnnLegacyControlKind::kInferenceMaxCount:
                expect(value.has_output && value.seed == std::bit_cast<std::uint32_t>(-719),
                       "signed inference count seed is copied as low32 bytes");
                observed = {1, 4};
                break;
        }
        output(reply, observed);
        if (scenario == 4) {
            // Corrupt the untrusted wire after canonical encoding: the writer
            // correctly refuses this overflowing offset/extent combination.
            expect(reply.data.size() == 32, "fixed legacy output wire size");
            for (std::size_t index = 24; index != 32; ++index) reply.data[index] = std::byte{0xff};
        }
        return reply;
    }

    ExecutionWireReply compute_reply(ExecutionPayloadReader& input) {
        ExecutionDnnLegacyCompute value;
        expect(read_execution_dnn_legacy_compute(input, value) && input.empty(),
               "complete canonical legacy compute record");
        common(value.handle, value.metadata, value.descriptor);
        ++computes;
        expect(value.sequence_length == 3 && value.filter == created[3] &&
                   value.workspace_bytes == 192 && value.reserve_bytes == 256,
               "exact legacy filter and raw workspace/reserve sizes");
        std::array<std::uint64_t, 13> offsets{};
        std::array<bool, 13> present{};
        std::array<std::uint64_t, 6> states{};
        if (value.kind == ExecutionDnnLegacyComputeKind::kForwardTraining) {
            expect(value.sequences[0].descriptors == x() && value.sequences[1].descriptors == y(),
                   "forward sequence arrays preserve source order and repeated identities");
            offsets = {0, 64, 0, 0, 128, 192, 256, 320, 0, 0, 512, 2048, 3072};
            present = {true, true,  false, false, true, true, true,
                       true, false, false, true,  true, true};
            states = {created[1], created[2], created[2], created[1], 0, 0};
        } else if (value.kind == ExecutionDnnLegacyComputeKind::kBackwardData) {
            expect(value.sequences[0].descriptors == y() &&
                       value.sequences[1].descriptors ==
                           std::vector<std::uint64_t>{created[1], created[1], created[2]} &&
                       value.sequences[2].descriptors ==
                           std::vector<std::uint64_t>{created[2], created[2], created[1]},
                   "backward-data keeps independent y/dy/dx arrays");
            offsets = {0, 64, 384, 448, 128, 192, 256, 320, 640, 704, 512, 2048, 3072};
            present.fill(true);
            present[0] = false;
            states = {created[1], created[2], created[2], created[1], created[1], created[2]};
        } else {
            expect(value.kind == ExecutionDnnLegacyComputeKind::kBackwardWeights &&
                       value.sequences[0].descriptors == x() &&
                       value.sequences[1].descriptors == y(),
                   "legacy weight gradient has no v8 WgradMode or sequence-length device array");
            offsets = {0, 64, 0, 0, 128, 0, 0, 0, 0, 0, 512, 2048, 3072};
            present = {true,  true,  false, false, true, false, false,
                       false, false, false, true,  true, true};
            states = {created[1], 0, 0, 0, 0, 0};
        }
        expect(value.states == states, "all six legacy state descriptor positions are exact");
        for (std::size_t index = 0; index != value.buffers.size(); ++index)
            expect(value.buffers[index].allocation == (present[index] ? 81U : 0U) &&
                       value.buffers[index].offset == offsets[index],
                   "all13 original nullable buffer positions retain exact allocation/offset");
        for (std::size_t index = 0; index != value.sequences.size(); ++index)
            expect(
                value.sequences[index].has_descriptors ==
                    (index < 2 || value.kind == ExecutionDnnLegacyComputeKind::kBackwardData ? 1U
                                                                                             : 0U),
                "array presence is independent and unused slots remain zero");
        auto reply = complete();
        reply.id = 500 + computes;
        if (scenario == 8 && computes == 1)
            reply.status = *reply.original_status = CUDNN_STATUS_EXECUTION_FAILED;
        if (scenario == 9) reply.data.push_back(std::byte{1});
        return reply;
    }

    ExecutionWireReply handle(const ExecutionReceivedRequest& request) {
        const auto operation = request.message.header.operation;
        ExecutionPayloadReader input(request.message.payload);
        if (operation == ExecutionOpcode::kHello) {
            auto ownership = ExecutionBootstrapOwnership::kRetained;
            if (decode_execution_context_bootstrap(request.message.payload)) {
                execution = request.connection;
                ++contexts;
            } else {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty() && !metadata,
                       "exact single authenticated DNM1 channel");
                metadata = request.connection;
                ownership = ExecutionBootstrapOwnership::kMetadataRetained;
            }
            auto reply = complete(false);
            const auto encoded = encode_execution_bootstrap_ownership(ownership);
            expect(encoded.has_value(), "encode exact ownership bootstrap");
            reply.data.assign(encoded->begin(), encoded->end());
            return reply;
        }
        if (operation == ExecutionOpcode::kDnnLegacyMetadata) {
            expect(request.connection == metadata, "host metadata uses DNM1 channel");
            return metadata_reply(input);
        }
        if (operation == ExecutionOpcode::kDnnLegacyControl ||
            operation == ExecutionOpcode::kDnnLegacyCompute) {
            expect(request.connection == execution,
                   "legacy context authority is not metadata authority");
            return operation == ExecutionOpcode::kDnnLegacyControl ? control_reply(input)
                                                                   : compute_reply(input);
        }
        if (operation == ExecutionOpcode::kCreateDnnMetadata) {
            std::uint32_t kind{};
            constexpr std::array<std::uint32_t, 6> kinds{13, 1, 1, 3, 3, 16};
            expect(request.connection == metadata && created.size() < kinds.size() &&
                       input.read_u32(kind) && kind == kinds[created.size()] && input.empty(),
                   "independent genuine RNN/tensor/filter/algorithm lifecycle types");
            const auto id = ++next_id;
            objects.emplace(id, kind);
            created.push_back(id);
            auto reply = complete();
            reply.id = id;
            return reply;
        }
        if (operation == ExecutionOpcode::kCreateDnn) {
            expect(request.connection == execution && input.empty(), "owned legacy handle");
            ++handles;
            auto reply = complete();
            reply.id = 41;
            return reply;
        }
        if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(request.connection == execution && input.read_u64(bytes) && bytes == kBytes &&
                       input.empty(),
                   "exact owned allocation capacity");
            ++allocations;
            auto reply = complete(false);
            reply.id = 81;
            reply.address = kAddress;
            return reply;
        }
        if (operation == ExecutionOpcode::kClose) {
            expect(request.connection == execution && input.empty() && objects.empty(),
                   "normal context close follows explicit descriptor retirement");
            ++cleanups;
            created.clear();
            return complete(false);
        }
        std::uint64_t id{};
        expect(input.read_u64(id) && input.empty(), "exact original lifecycle identity");
        if (operation == ExecutionOpcode::kDestroyDnnMetadata) {
            expect(request.connection == metadata && objects.erase(id) == 1,
                   "every known metadata owner destroyed exactly once");
            ++destroys;
            return complete();
        }
        if (operation == ExecutionOpcode::kDestroyDnn) {
            expect(request.connection == execution && id == 41, "exact owned handle destroy");
            ++destroys;
            return complete();
        }
        expect(
            operation == ExecutionOpcode::kRelease && request.connection == execution && id == 81,
            "no unrecognized API, hidden native call or recovery request");
        ++destroys;
        return complete(false);
    }
};

void run(const char* provider, unsigned int scenario) {
    expect(scenario < kCases, "bounded isolated scenario");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual authenticated endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-legacy-front-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated endpoint before frontend initialization");
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start real local socket endpoint");
    auto client = std::async(std::launch::async, [=] {
        if (!scenario) {
            normal_application(lookup(provider, false));
            normal_application(lookup(provider, true));
        } else {
            hostile_application(lookup(provider, scenario % 2 != 0), scenario);
        }
        expect(!glimmer_dnn_legacy_native_calls(), "zero caller-native SDK execution/fallback");
    });
    struct Stop {
        ExecutionSocketServer& server;
        ~Stop() {
            server.stop();
        }
    } stop{server};
    Peer peer;
    peer.scenario = scenario;
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded fake endpoint progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "exact peer retained on every request");
            if (profiles.respond(server, *request)) continue;
            const auto reply = peer.handle(*request);
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "hostile receipt remains wire-valid");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {139, 149};
            expect(server.reply(request->connection, {header, *encoded}),
                   "send exact matching receipt");
        }
    }
    client.get();
    profiles.verify();
    if (!scenario) {
        for (const auto calls : peer.version_checks)
            expect(calls == 2, "each of six version-check selectors reached both symbol routes");
        expect(peer.metadata_calls == 22 && peer.controls == 18 && peer.computes == 6 &&
                   peer.contexts == 2 && peer.handles == 2 && peer.allocations == 2 &&
                   peer.cleanups == 2 && peer.destroys == 16 && peer.objects.empty(),
               "all25 direct/cached routes preserve exact semantic counts and normal cleanup");
    } else {
        const bool is_healthy = scenario == 1 || scenario == 8 || scenario == 12;
        expect(peer.contexts == 1 && peer.handles == 1 && peer.allocations == 1 &&
                   peer.cleanups == (is_healthy ? 1U : 0U) &&
                   peer.destroys == (is_healthy ? 8U : 0U),
               "healthy cases clean normally; uncertainty never invents native cleanup");
        expect(peer.metadata_calls == (scenario == 1
                                           ? 2U
                                           : (scenario == 2 || scenario == 3 || scenario == 10 ||
                                                      scenario == 11 || scenario == 13
                                                  ? 1U
                                                  : 0U)) &&
                   peer.controls == (scenario >= 4 && scenario <= 7 ? 1U
                                     : scenario == 12               ? 3U
                                                                    : 0U) &&
                   peer.computes == (scenario == 8   ? 2U
                                     : scenario == 9 ? 1U
                                                     : 0U),
               "no hidden metadata probe, compute replay or uncertain second request");
    }
}
unsigned int number(const char* raw) {
    const std::string_view text(raw);
    unsigned int result{};
    const auto converted = std::from_chars(text.data(), text.data() + text.size(), result);
    expect(converted.ec == std::errc{} && converted.ptr == text.data() + text.size() &&
               result < kCases,
           "bounded decimal case index");
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || (argc == 4 && std::string_view(argv[2]) == "--case"),
               "usage: execution_dnn_legacy_frontend_test PROVIDER [--case INDEX]");
        if (argc == 4) {
            run(argv[1], number(argv[3]));
        } else {
            for (unsigned int scenario = 0; scenario != kCases; ++scenario) {
                const auto index = std::to_string(scenario);
                const auto child = ::fork();
                expect(child >= 0, "fork isolated frontend scenario");
                if (!child) {
                    ::execl("/proc/self/exe", argv[0], argv[1], "--case", index.c_str(),
                            static_cast<char*>(nullptr));
                    ::_exit(127);
                }
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           WEXITSTATUS(status) == 0,
                       "isolated scenario exits normally and unsignalled");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#undef LEGACY_APIS
#pragma GCC diagnostic pop
