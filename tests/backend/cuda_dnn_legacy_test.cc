#include "dnn_legacy_fixture.h"
#include "internal/cuda_dnn_legacy.h"

#include <iostream>
#include <cstdlib>
#include <new>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>

#if CUDNN_MAJOR == 8
namespace {
thread_local int allocation_countdown = -1;
void* legacy_test_allocate(std::size_t bytes, std::size_t alignment = 0) {
    namespace fixture = glimmer::tests::dnn_legacy;
    if (allocation_countdown == 0) {
        allocation_countdown = -1;
        ++fixture::allocation_failures;
        throw std::bad_alloc();
    }
    if (allocation_countdown > 0) --allocation_countdown;
    if (std::exchange(fixture::fail_next_allocation, false)) {
        ++fixture::allocation_failures;
        throw std::bad_alloc();
    }
    void* pointer{};
    if (alignment) {
        if (::posix_memalign(&pointer, alignment, std::max(bytes, std::size_t{1})) != 0)
            throw std::bad_alloc();
    } else
        pointer = std::malloc(std::max(bytes, std::size_t{1}));
    if (!pointer) throw std::bad_alloc();
    return pointer;
}
}  // namespace
// Complete replacement pairs avoid allocator-family mismatches under ASan.
// Only one explicitly armed allocation fails, after the actual stream getter.
void* operator new(std::size_t bytes) {
    return legacy_test_allocate(bytes);
}
void* operator new[](std::size_t bytes) {
    return legacy_test_allocate(bytes);
}
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    return legacy_test_allocate(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
    return legacy_test_allocate(bytes, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(bytes, alignment);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](bytes, alignment);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
#endif

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_legacy;
namespace foundation = fixture::foundation;
using Metadata = CudaDnnLegacyMetadataKind;
using Control = CudaDnnLegacyControlKind;
using Compute = CudaDnnLegacyComputeKind;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void native(const CudaDnnResult& result, cudnnStatus_t expected = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == expected,
           "requested original SDK result remains exact");
}
#if CUDNN_MAJOR == 8
void driver(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
struct Algorithm {
    bool alive{};
    std::uint32_t bits{};
};
struct Calls {
    std::array<unsigned int, 25> calls{};
    std::array<Algorithm, 4> algorithms{};
    std::uint32_t math{}, bias{};
    cudnnStatus_t getter_status{CUDNN_STATUS_SUCCESS}, compute_status{CUDNN_STATUS_SUCCESS};
    bool writes{true};
    bool bad_pointer{}, failed_bind{}, failed_sync{};
    int sequence{};
    std::array<std::vector<void*>, 3> arrays;
    std::array<void*, 6> states{};
    void* filter{};
    std::array<const void*, 13> buffers{};
    std::array<std::size_t, 2> sizes{};
    unsigned int synchronizes{};
};
Calls calls;
template <unsigned int Index>
cudnnStatus_t version() {
    ++calls.calls[Index];
    return CUDNN_STATUS_SUCCESS;
}
enum class ComputeCallback : unsigned int { kForward = 22, kBackwardData, kBackwardWeights };
void capture(ComputeCallback operation, int count,
             std::array<const cudnnTensorDescriptor_t*, 3> arrays,
             std::array<cudnnTensorDescriptor_t, 6> states, cudnnFilterDescriptor_t filter,
             std::array<const void*, 13> buffers, std::size_t work, std::size_t reserve) {
    ++calls.calls[static_cast<unsigned int>(operation)];
    calls.sequence = count;
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        calls.arrays[i].clear();
        if (arrays[i])
            for (int j = 0; j < count; ++j) calls.arrays[i].push_back(arrays[i][j]);
    }
    std::copy(states.begin(), states.end(), calls.states.begin());
    calls.filter = filter;
    calls.buffers = buffers;
    calls.sizes = {work, reserve};
}
CudaDnnDispatch dispatch() {
    auto sdk = fixture::dispatch();
    auto& old = sdk.legacy;
    old.ops_infer_version = version<0>;
    old.ops_train_version = version<1>;
    old.cnn_infer_version = version<2>;
    old.cnn_train_version = version<3>;
    old.adv_infer_version = version<4>;
    old.adv_train_version = version<5>;
    old.algorithm_create = [](cudnnAlgorithmDescriptor_t* output) {
        ++calls.calls[6];
        for (auto& entry : calls.algorithms)
            if (!entry.alive) {
                entry.alive = true;
                *output = reinterpret_cast<cudnnAlgorithmDescriptor_t>(&entry);
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    old.algorithm_set = [](cudnnAlgorithmDescriptor_t descriptor, cudnnAlgorithm_t algorithm) {
        ++calls.calls[7];
        auto* entry = reinterpret_cast<Algorithm*>(descriptor);
        expect(entry && entry->alive, "actual owned Algorithm descriptor");
        std::memcpy(&entry->bits, &algorithm, sizeof(entry->bits));
        return CUDNN_STATUS_SUCCESS;
    };
    old.algorithm_destroy = [](cudnnAlgorithmDescriptor_t descriptor) {
        ++calls.calls[8];
        auto* entry = reinterpret_cast<Algorithm*>(descriptor);
        expect(entry && entry->alive, "Algorithm destroyed exactly once");
        entry->alive = false;
        return CUDNN_STATUS_SUCCESS;
    };
    old.math_set = [](cudnnRNNDescriptor_t, cudnnMathType_t value) {
        ++calls.calls[9];
        calls.math = static_cast<std::uint32_t>(value);
        return CUDNN_STATUS_SUCCESS;
    };
    old.math_get = [](cudnnRNNDescriptor_t, cudnnMathType_t* output) {
        ++calls.calls[10];
        if (output && calls.writes) *output = static_cast<cudnnMathType_t>(calls.math);
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    old.bias_set = [](cudnnRNNDescriptor_t, cudnnRNNBiasMode_t value) {
        ++calls.calls[11];
        calls.bias = static_cast<std::uint32_t>(value);
        return CUDNN_STATUS_SUCCESS;
    };
    old.bias_get = [](cudnnRNNDescriptor_t, cudnnRNNBiasMode_t* output) {
        ++calls.calls[12];
        if (output && calls.writes) *output = static_cast<cudnnRNNBiasMode_t>(calls.bias);
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original v6 setup ABI.
    old.setup = [](cudnnHandle_t handle, cudnnRNNDescriptor_t rnn, int hidden, int layers,
                   cudnnDropoutDescriptor_t dropout, cudnnRNNInputMode_t input,
                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algorithm,
                   cudnnDataType_t type) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        ++calls.calls[13];
        (void)fixture::base::handle(handle);
        expect(hidden == 2 && layers == 1 && input == CUDNN_LINEAR_INPUT &&
                   direction == CUDNN_UNIDIRECTIONAL && mode == CUDNN_LSTM &&
                   algorithm == CUDNN_RNN_ALGO_STANDARD && type == CUDNN_DATA_FLOAT,
               "old setup signature and each enum remain independent");
        fixture::metadata::object(rnn, CudaDnnObjectKind::kRnn).rnn.dropout = dropout;
        return CUDNN_STATUS_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original lower/upper Clip ABI.
    old.clip = [](cudnnHandle_t handle, cudnnRNNDescriptor_t, cudnnRNNClipMode_t mode,
                  cudnnNanPropagation_t nan, double lower, double upper) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        ++calls.calls[14];
        (void)fixture::base::handle(handle);
        expect(mode == CUDNN_RNN_CLIP_MINMAX && nan == CUDNN_PROPAGATE_NAN && lower == -2 &&
                   upper == 3,
               "legacy Clip preserves its real handle and scalar arguments");
        return CUDNN_STATUS_SUCCESS;
    };
    old.workspace_size = [](cudnnHandle_t, cudnnRNNDescriptor_t, int count,
                            const cudnnTensorDescriptor_t* array, std::size_t* output) {
        ++calls.calls[15];
        expect(count == 2 && array && array[0] == array[1], "query array aliases retained");
        if (output && calls.writes) *output = 32;
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    old.reserve_size = [](cudnnHandle_t, cudnnRNNDescriptor_t, int count,
                          const cudnnTensorDescriptor_t* array, std::size_t* output) {
        ++calls.calls[16];
        expect(count == 2 && array && array[0] == array[1], "reserve array aliases retained");
        if (output && calls.writes) *output = 48;
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    old.parameter_size = [](cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t x,
                            std::size_t* output, cudnnDataType_t type) {
        ++calls.calls[17];
        expect(x && type == CUDNN_DATA_FLOAT, "parameter sizing input and dtype");
        if (output && calls.writes) *output = 64;
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    old.matrix_parameter = [](cudnnHandle_t, cudnnRNNDescriptor_t, int layer,
                              cudnnTensorDescriptor_t x, cudnnFilterDescriptor_t filter,
                              const void* weights, int linear, cudnnFilterDescriptor_t destination,
                              void** output) {
        ++calls.calls[18];
        expect(layer == 0 && linear == 1 && x && filter && weights && destination,
               "matrix parameter exact ABI inputs");
        auto& descriptor = foundation::descriptor(destination, CudaDnnObjectKind::kFilter);
        descriptor.filter = {CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, {1, 1, 1}};
        descriptor.configured = true;
        if (output) {
            const auto address =
                reinterpret_cast<std::uintptr_t>(weights) + (calls.bad_pointer ? 4096U : 4U);
            // Inject an owned-address or out-of-range observation without dereferencing it.
            // NOLINTNEXTLINE(performance-no-int-to-ptr): synthetic ABI address.
            *output = reinterpret_cast<void*>(address);
        }
        return CUDNN_STATUS_SUCCESS;
    };
    old.bias_parameter = [](cudnnHandle_t, cudnnRNNDescriptor_t, int, cudnnTensorDescriptor_t,
                            cudnnFilterDescriptor_t, const void*, int,
                            cudnnFilterDescriptor_t destination, void** output) {
        ++calls.calls[19];
        expect(destination, "bias output descriptor retained");
        if (output) *output = nullptr;
        return CUDNN_STATUS_SUCCESS;
    };
    old.algorithm_bind = [](cudnnHandle_t handle, cudnnRNNDescriptor_t,
                            cudnnAlgorithmDescriptor_t descriptor) {
        ++calls.calls[20];
        (void)fixture::base::handle(handle);
        expect(reinterpret_cast<Algorithm*>(descriptor)->alive,
               "binding actual Algorithm lifecycle");
        return calls.failed_bind ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_SUCCESS;
    };
    old.inference_max_count = [](cudnnHandle_t, cudnnRNNDescriptor_t, int* output) {
        ++calls.calls[21];
        if (output && calls.writes) *output = 7;
        return output ? calls.getter_status : CUDNN_STATUS_BAD_PARAM;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original ForwardTraining positional
    // ABI.
    old.forward = [](cudnnHandle_t, cudnnRNNDescriptor_t, int n,
                     const cudnnTensorDescriptor_t* xdesc, const void* x,
                     cudnnTensorDescriptor_t hxdesc, const void* hx, cudnnTensorDescriptor_t cxdesc,
                     const void* cx, cudnnFilterDescriptor_t filter, const void* w,
                     const cudnnTensorDescriptor_t* ydesc, void* y, cudnnTensorDescriptor_t hydesc,
                     void* hy, cudnnTensorDescriptor_t cydesc, void* cy, void* work,
                     std::size_t work_bytes, void* reserve, std::size_t reserve_bytes) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        capture(ComputeCallback::kForward, n, {xdesc, ydesc, nullptr},
                {hxdesc, cxdesc, hydesc, cydesc, nullptr, nullptr}, filter,
                {x, y, nullptr, nullptr, hx, cx, hy, cy, nullptr, nullptr, w, work, reserve},
                work_bytes, reserve_bytes);
        return calls.compute_status;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original BackwardData positional
    // ABI.
    old.backward_data =
        [](cudnnHandle_t, cudnnRNNDescriptor_t, int n, const cudnnTensorDescriptor_t* ydesc,
           const void* y, const cudnnTensorDescriptor_t* dydesc, const void* dy,
           cudnnTensorDescriptor_t dhydesc, const void* dhy, cudnnTensorDescriptor_t dcydesc,
           const void* dcy, cudnnFilterDescriptor_t filter, const void* w,
           cudnnTensorDescriptor_t hxdesc, const void* hx, cudnnTensorDescriptor_t cxdesc,
           const void* cx, const cudnnTensorDescriptor_t* dxdesc, void* dx,
           cudnnTensorDescriptor_t dhxdesc, void* dhx, cudnnTensorDescriptor_t dcxdesc, void* dcx,
           void* work, std::size_t work_bytes, void* reserve, std::size_t reserve_bytes) {
            // NOLINTEND(bugprone-easily-swappable-parameters)
            capture(ComputeCallback::kBackwardData, n, {ydesc, dydesc, dxdesc},
                    {hxdesc, cxdesc, dhydesc, dcydesc, dhxdesc, dcxdesc}, filter,
                    {nullptr, y, dy, dx, hx, cx, dhy, dcy, dhx, dcx, w, work, reserve}, work_bytes,
                    reserve_bytes);
            return calls.compute_status;
        };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original BackwardWeights positional
    // const workspace/reserve ABI.
    old.backward_weights = [](cudnnHandle_t, cudnnRNNDescriptor_t, int n,
                              const cudnnTensorDescriptor_t* xdesc, const void* x,
                              cudnnTensorDescriptor_t hxdesc, const void* hx,
                              const cudnnTensorDescriptor_t* ydesc, const void* y, const void* work,
                              std::size_t work_bytes, cudnnFilterDescriptor_t filter, void* dw,
                              const void* reserve, std::size_t reserve_bytes) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        capture(ComputeCallback::kBackwardWeights, n, {xdesc, ydesc, nullptr},
                {hxdesc, nullptr, nullptr, nullptr, nullptr, nullptr}, filter,
                {x, y, nullptr, nullptr, hx, nullptr, nullptr, nullptr, nullptr, nullptr, dw, work,
                 reserve},
                work_bytes, reserve_bytes);
        return calls.compute_status;
    };
    return sdk;
}
struct Owner {
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t rnn{}, algorithm{}, tensor{}, filter{}, dropout{}, handle{}, sequence{1};
    std::array<CudaAllocationRange, 13> buffers{};
    explicit Owner(CudaSessionLimits limits = {}) {
        fixture::reset();
        calls = {};
        auto sdk = fixture::session_dispatch();
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
        // Fake synchronization is an independent physical proof from held tail
        // events: an original error must synchronize, then still retain its tail.
        sdk.context_synchronize = [] {
            ++calls.synchronizes;
            return calls.failed_sync ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        };
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        rnn = create(CudaDnnObjectKind::kRnn);
        algorithm = create(CudaDnnObjectKind::kAlgorithm);
        tensor = create(CudaDnnObjectKind::kTensor);
        filter = create(CudaDnnObjectKind::kFilter);
        dropout = create(CudaDnnObjectKind::kDropout);
        native(metadata->set_tensor(tensor, CudaDnnTensorSetter::kNd,
                                    {CUDNN_DATA_FLOAT, 0, 3, {1, 2, 1}, {2, 1, 1}}));
        native(metadata->set_filter(filter, {CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, {16, 1, 1}}));
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk, limits);
        driver(session->initialize(0));
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
        for (auto& range : buffers) {
            const auto allocated = session->allocate(96);
            driver(allocated);
            range = {allocated.id, 16};
        }
        auto setup = control(Control::kSetup);
        setup.setup = {2,
                       1,
                       dropout,
                       CUDNN_LINEAR_INPUT,
                       CUDNN_UNIDIRECTIONAL,
                       CUDNN_LSTM,
                       CUDNN_RNN_ALGO_STANDARD,
                       CUDNN_DATA_FLOAT};
        native(session->dnn_legacy_control(handle, *metadata, setup).result);
    }
    std::uint64_t create(CudaDnnObjectKind kind) {
        const auto result = metadata->create(kind);
        native(result);
        return result.id;
    }
    CudaDnnLegacyControl control(Control kind) const {
        CudaDnnLegacyControl result;
        result.kind = kind;
        result.handle = handle;
        result.descriptor = rnn;
        return result;
    }
    void close() {
        for (const auto& range : buffers) driver(session->release(range.allocation));
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(foundation::driver.contexts.empty() && foundation::driver.streams.empty() &&
                   foundation::driver.events.empty() && foundation::driver.memory.empty() &&
                   std::ranges::none_of(calls.algorithms,
                                        [](const auto& entry) { return entry.alive; }),
               "normal cleanup destroys every known resource without retained ownership");
    }
};
void metadata_and_control(Owner& owner) {
    for (std::uint32_t i = 1; i <= 6; ++i)
        native(owner.metadata->legacy_metadata({Metadata::kVersionCheck, 0, i, 0, 0}).result);
    native(owner.metadata->legacy_metadata({Metadata::kSetAlgorithm, owner.algorithm, 2, 0, 0})
               .result);
    auto bind = owner.control(Control::kSetAlgorithm);
    bind.algorithm = owner.algorithm;
    native(owner.session->dnn_legacy_control(owner.handle, *owner.metadata, bind).result);
    expect(owner.metadata->destroy(owner.algorithm).status == CUDA_ERROR_INVALID_VALUE,
           "algorithm liveness edge prevents premature destruction");
    native(owner.metadata->legacy_metadata({Metadata::kSetMath, owner.rnn, 1, 0, 0}).result);
    native(owner.metadata->legacy_metadata({Metadata::kSetBias, owner.rnn, 2, 0, 0}).result);
    for (const auto kind : {Metadata::kGetMath, Metadata::kGetBias}) {
        const auto expected = kind == Metadata::kGetMath ? 1U : 2U;
        for (const bool writes : {false, true}) {
            calls.writes = writes;
            calls.getter_status = CUDNN_STATUS_BAD_PARAM;
            const auto result = owner.metadata->legacy_metadata({kind, owner.rnn, 0, 1, expected});
            native(result.result, CUDNN_STATUS_BAD_PARAM);
            expect(result.output.written == (writes ? 1U : 0U) &&
                       result.output.scalar == (writes ? expected : 0U),
                   "failed getter distinguishes no write from an equal-value native store");
        }
    }
    calls.writes = true;
    calls.getter_status = CUDNN_STATUS_SUCCESS;
    auto clip = owner.control(Control::kClip);
    clip.clip = {CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -2, 3};
    native(owner.session->dnn_legacy_control(owner.handle, *owner.metadata, clip).result);
    for (const auto kind : {Control::kWorkspaceSize, Control::kReserveSize, Control::kParameterSize,
                            Control::kInferenceMaxCount}) {
        auto query = owner.control(kind);
        query.has_output = 1;
        query.seed = 123;
        if (kind == Control::kWorkspaceSize || kind == Control::kReserveSize) {
            query.sequence_length = 2;
            query.sequence = {1, {owner.tensor, owner.tensor}};
        } else if (kind == Control::kParameterSize) {
            query.input = owner.tensor;
            query.data_type = CUDNN_DATA_FLOAT;
        }
        const auto result = owner.session->dnn_legacy_control(owner.handle, *owner.metadata, query);
        native(result.result);
        const auto expected = kind == Control::kWorkspaceSize   ? 32U
                              : kind == Control::kReserveSize   ? 48U
                              : kind == Control::kParameterSize ? 64U
                                                                : 7U;
        expect(result.output.written == 1 && result.output.scalar == expected,
               "exact original scalar width and value");
    }
    auto parameter = owner.control(Control::kMatrixParameter);
    parameter.has_output = 1;
    // Source/destination alias is intentional: exclusivity must exclude only
    // foreign borrowers, not the transaction's own immutable input pin.
    parameter.parameters = {owner.tensor, owner.filter, owner.filter, 0, 1, owner.buffers[10]};
    const auto matrix = owner.session->dnn_legacy_control(owner.handle, *owner.metadata, parameter);
    native(matrix.result);
    expect(matrix.output.written == 2 && matrix.output.non_null == 1 && matrix.output.offset == 4 &&
               matrix.output.extent == 4,
           "aliased destination uses pre-call source range and actual post-call filter extent");
    parameter.kind = Control::kBiasParameter;
    const auto bias = owner.session->dnn_legacy_control(owner.handle, *owner.metadata, parameter);
    native(bias.result);
    expect(bias.output.written == 2 && !bias.output.non_null && !bias.output.offset &&
               !bias.output.extent,
           "native-written NULL is distinct from unchanged pointer output");
    native(owner.metadata->set_filter(owner.filter,
                                      {CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, {16, 1, 1}}));
}
void compute(Owner& owner, Compute kind, cudnnStatus_t status) {
    CudaSessionDnnLegacyRnn problem;
    problem.kind = kind;
    problem.handle = owner.handle;
    problem.descriptor = owner.rnn;
    problem.sequence_length = 2;
    problem.filter = owner.filter;
    problem.sequences[0] = problem.sequences[1] = {1, {owner.tensor, owner.tensor}};
    problem.states.fill(owner.tensor);
    problem.buffers = owner.buffers;
    problem.workspace_bytes = 32;
    problem.reserve_bytes = 48;
    if (kind == Compute::kForwardTraining) {
        for (const auto i : {2, 3, 8, 9}) problem.buffers[i] = {};
        problem.states[4] = problem.states[5] = 0;
    } else if (kind == Compute::kBackwardData) {
        problem.buffers[0] = {};
        problem.sequences[2] = problem.sequences[0];
    } else {
        for (const auto i : {2, 3, 5, 6, 7, 8, 9}) problem.buffers[i] = {};
        std::fill(problem.states.begin() + 1, problem.states.end(), 0);
    }
    std::array<const void*, 13> expected{};
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (problem.buffers[i].allocation) {
            const auto memory = owner.session->device_memory(problem.buffers[i], 0);
            expect(memory.has_value(), "owned buffer for exact native ABI comparison");
            // NOLINTNEXTLINE(performance-no-int-to-ptr): fixture-owned device address.
            expected[i] = reinterpret_cast<const void*>(memory->address);
        }
    calls.compute_status = status;
    const auto prior_calls = calls.calls;
    if (kind == Compute::kForwardTraining) {
        auto oversized = problem;
        oversized.sequence_length = 4;
        for (auto& array : oversized.sequences)
            if (array.has_descriptors) array.descriptors.resize(4, owner.tensor);
        const auto refused =
            owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, oversized);
        expect(refused.native.acceptance.status ==
                       glimmer::backend::ExecutionAcceptStatus::kOutOfMemory &&
                   !refused.result && !refused.native.acceptance.receipt &&
                   !owner.session->snapshot().submissions.retained_host_bytes &&
                   calls.calls == prior_calls,
               "a single oversize legacy capsule is terminal OOM before SDK entry or borrowing");
        {
            glimmer::backend::ExecutionSubmissionQueue storage_owner({91, 92});
            expect(storage_owner.bind_executor(), "bind deliberate arena exhaustion control");
            auto tiny = storage_owner.allocate_host_storage(1);
            const auto exhausted = owner.metadata->borrow_legacy_rnn(problem, tiny.storage);
            expect(
                exhausted.result.status == CUDA_ERROR_OUT_OF_MEMORY &&
                    !owner.metadata->snapshot().borrowed && calls.calls == prior_calls,
                "borrow-vector arena exhaustion rolls back dependencies and cannot use the heap");
            tiny.storage.reset();
            expect(!storage_owner.snapshot().retained_host_bytes,
                   "failed borrow refunds its storage");
        }
        // Four arena-owner allocations precede descriptor borrowing; the next
        // two create its RNN lease. Test both early and later allocation paths.
        for (int point = 0; point < 6; ++point) {
            allocation_countdown = point;
            const auto failed =
                owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, problem);
            const bool consumed = allocation_countdown == -1;
            allocation_countdown = -1;
            expect(consumed &&
                       failed.native.acceptance.status ==
                           glimmer::backend::ExecutionAcceptStatus::kOutOfMemory &&
                       !failed.result && !failed.native.acceptance.receipt &&
                       !owner.session->snapshot().submissions.retained_host_bytes &&
                       !owner.metadata->snapshot().borrowed && calls.calls == prior_calls,
                   "early storage/descriptor allocation failure is terminal OOM with complete "
                   "rollback");
        }
        const auto prior_failures = fixture::allocation_failures;
        fixture::arm_allocation_failure = true;
        const auto failed =
            owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, problem);
        const bool was_consumed = !std::exchange(fixture::fail_next_allocation, false);
        const auto after = owner.session->snapshot();
        expect(!fixture::arm_allocation_failure && was_consumed &&
                   fixture::allocation_failures == prior_failures + 1 &&
                   failed.native.acceptance.status ==
                       glimmer::backend::ExecutionAcceptStatus::kOutOfMemory &&
                   !failed.native.acceptance.receipt && !failed.native.original_status &&
                   !failed.result && failed.native.rejection_status == CUDA_ERROR_OUT_OF_MEMORY &&
                   !after.submissions.outstanding && !after.submissions.retained_host_bytes &&
                   !after.is_quarantined && calls.calls == prior_calls,
               "capsule allocation failure is terminal OOM, never a retriable queue conflict or "
               "native result");
        native(owner.metadata->set_tensor(owner.tensor, CudaDnnTensorSetter::kNd,
                                          {CUDNN_DATA_FLOAT, 0, 3, {1, 2, 1}, {2, 1, 1}}));
        const auto recreated = owner.create(CudaDnnObjectKind::kTensor);
        native(owner.metadata->destroy(recreated));
        expect(!owner.metadata->snapshot().borrowed,
               "allocation-failure rollback releases every descriptor/dependency borrow");
    }
    auto invalid = problem;
    invalid.sequences[1].descriptors.back() = UINT64_MAX;
    const auto missing =
        owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, invalid);
    expect(!missing.native.acceptance.accepted() && !missing.result && calls.calls == prior_calls &&
               !owner.session->snapshot().submissions.retained_host_bytes,
           "a late invalid descriptor rolls back the entire atomic capture before native issue");
    native(owner.metadata->set_tensor(owner.tensor, CudaDnnTensorSetter::kNd,
                                      {CUDNN_DATA_FLOAT, 0, 3, {1, 2, 1}, {2, 1, 1}}));
    invalid = problem;
    invalid.buffers[11].offset = 95;
    const auto range =
        owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, invalid);
    expect(!range.native.acceptance.accepted() && !range.result && calls.calls == prior_calls &&
               !owner.session->snapshot().submissions.retained_host_bytes,
           "exact workspace size must fit its allocation, without any computation callback");
    foundation::control.can_stop.store(false);
    const auto accepted =
        owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, problem);
    if (accepted.native.acceptance.accepted()) {
        expect(accepted.native.acceptance.receipt.sequence() == owner.sequence,
               "rejected attempts leave the exact next queue sequence unconsumed");
        ++owner.sequence;
    } else {
        std::cerr << "legacy acceptance status="
                  << static_cast<unsigned int>(accepted.native.acceptance.status)
                  << " rejection=" << static_cast<int>(accepted.native.rejection_status)
                  << " preflight=";
        if (accepted.preflight_status)
            std::cerr << static_cast<int>(*accepted.preflight_status);
        else
            std::cerr << "none";
        std::cerr << '\n';
    }
    expect(accepted.native.acceptance.accepted() && accepted.result && !accepted.result->is_ready,
           "legacy command atomically accepts copied arrays and resource leases before issue");
    const auto charged = cuda_dnn_legacy_storage_bytes(problem);
    expect(charged && owner.session->snapshot().submissions.retained_host_bytes == *charged,
           "accepted legacy arrays live in the independently charged real arena");
    const auto pressure =
        owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, problem);
    expect(pressure.native.acceptance.status == glimmer::backend::ExecutionAcceptStatus::kFull &&
               !pressure.result && !pressure.native.acceptance.receipt &&
               calls.calls == prior_calls &&
               owner.session->snapshot().submissions.retained_host_bytes == *charged,
           "occupied host budget rejects before native compute without consuming a sequence");
    expect(owner.metadata->legacy_metadata({Metadata::kSetAlgorithm, owner.algorithm, 1, 0, 0})
                       .result.disposition == CudaDnnDisposition::kDeferred &&
               owner.metadata->destroy(owner.tensor).disposition == CudaDnnDisposition::kDeferred &&
               owner.session->release(owner.buffers[1].allocation).disposition ==
                   CudaSessionDisposition::kDeferred,
           "accepted immutable descriptor, algorithm and allocation leases exclude mutation/free");
    const auto synchronizes = calls.synchronizes;
    const auto issued = owner.session->issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued && issued.library_status == status &&
               accepted.result->is_ready,
           "one requested legacy SDK invocation yields its original host result");
    native(accepted.result->result, status);
    expect(calls.buffers == expected && calls.sequence == 2 &&
               calls.sizes == std::array<std::size_t, 2>{32, 48} &&
               calls.synchronizes == synchronizes + (status == CUDNN_STATUS_SUCCESS ? 0U : 1U),
           "all13 positional pointers and exact capacities survive; errors join internal streams");
    expect(calls.arrays[0].size() == 2 && calls.arrays[0][0] == calls.arrays[0][1] &&
               calls.arrays[1] == calls.arrays[0] &&
               (kind == Compute::kBackwardData ? calls.arrays[2] == calls.arrays[0]
                                               : calls.arrays[2].empty()),
           "independent descriptor arrays retain original order and aliases");
    const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
    expect(host && host->status == status &&
               owner.session->acknowledge_result(accepted.native.acceptance.receipt),
           "exact native host status is acknowledged once");
    expect(!owner.session->poll().retired &&
               owner.session->snapshot().submissions.retained_host_bytes == *charged &&
               owner.metadata->destroy(owner.tensor).disposition == CudaDnnDisposition::kDeferred,
           "host error and ACK do not release physical descriptor leases");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1 &&
               !owner.session->snapshot().submissions.retained_host_bytes,
           "real trailing event releases one command without replay");
}
void uncertainty(unsigned int mode) {
    const auto child = ::fork();
    expect(child >= 0, "bounded quarantine child is created");
    if (!child) {
        try {
            Owner owner;
            if (mode == 0) {
                auto bind = owner.control(Control::kSetAlgorithm);
                bind.algorithm = owner.algorithm;
                native(
                    owner.session->dnn_legacy_control(owner.handle, *owner.metadata, bind).result);
                const auto candidate = owner.create(CudaDnnObjectKind::kAlgorithm);
                bind.algorithm = candidate;
                calls.failed_bind = true;
                const auto failed =
                    owner.session->dnn_legacy_control(owner.handle, *owner.metadata, bind);
                native(failed.result, CUDNN_STATUS_BAD_PARAM);
                expect(failed.has_metadata_quarantine &&
                           owner.metadata->snapshot().is_quarantined && calls.algorithms[0].alive &&
                           calls.algorithms[1].alive,
                       "failed rebind preserves both possible Algorithm owners and exact native "
                       "failure");
                const auto destroyed = calls.calls[8];
                expect(owner.metadata->shutdown().disposition == CudaDnnDisposition::kRejected &&
                           calls.calls[8] == destroyed,
                       "unknown selection has no speculative native cleanup");
            } else if (mode == 1) {
                auto query = owner.control(Control::kMatrixParameter);
                query.has_output = 1;
                query.parameters = {owner.tensor,     owner.filter, owner.filter, 0, 1,
                                    owner.buffers[10]};
                calls.bad_pointer = true;
                const auto failed =
                    owner.session->dnn_legacy_control(owner.handle, *owner.metadata, query);
                expect(failed.result.disposition == CudaDnnDisposition::kRejected &&
                           failed.result.original_status == CUDNN_STATUS_SUCCESS &&
                           failed.has_metadata_quarantine && !failed.output.written &&
                           !failed.output.non_null,
                       "an original pointer outside its leased source never becomes frontend "
                       "authority");
            } else {
                CudaSessionDnnLegacyRnn problem;
                problem.kind = Compute::kBackwardWeights;
                problem.handle = owner.handle;
                problem.descriptor = owner.rnn;
                problem.sequence_length = 2;
                problem.filter = owner.filter;
                problem.sequences[0] = problem.sequences[1] = {1, {owner.tensor, owner.tensor}};
                problem.states[0] = owner.tensor;
                for (const auto i : {0, 1, 4, 10, 11, 12}) problem.buffers[i] = owner.buffers[i];
                problem.workspace_bytes = 32;
                problem.reserve_bytes = 48;
                calls.compute_status = CUDNN_STATUS_BAD_PARAM;
                calls.failed_sync = true;
                const auto accepted =
                    owner.session->accept_dnn_legacy_rnn(owner.sequence, *owner.metadata, problem);
                if (accepted.native.acceptance.accepted()) ++owner.sequence;
                expect(accepted.native.acceptance.accepted(),
                       "uncertainty case first owns a real command");
                (void)owner.session->issue_next();
                expect(accepted.result->is_ready &&
                           accepted.result->result.original_status == CUDNN_STATUS_BAD_PARAM &&
                           accepted.result->is_cleanup_uncertain && calls.calls[24] == 1 &&
                           calls.synchronizes == 1,
                       "failed physical proof retains the original error and all native resources");
                (void)owner.session->issue_next();
                expect(calls.calls[24] == 1 && foundation::driver.frees == 0 &&
                           owner.session->snapshot().submissions.retained_host_bytes > 0,
                       "quarantined computation is neither replayed nor speculatively freed");
            }
            // Deliberate quarantine retains unknown native resources. This
            // child validates containment, not normal cleanup; the parent above
            // independently requires normal retirement of every healthy owner.
            ::_exit(0);
        } catch (...) {
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "quarantine child satisfies its exact failure-containment oracle");
}
#endif
}  // namespace
int main() {
    try {
#if CUDNN_MAJOR == 8
        CudaSessionLimits limits;
        limits.submissions.max_retained_host_bytes =
            9 * (alignof(std::max_align_t) - 1) +
            6 * (sizeof(CudaDnnObjectView) + sizeof(CudaDnnLegacyExtent) +
                 sizeof(cudnnTensorDescriptor_t));
        Owner owner(limits);
        metadata_and_control(owner);
        compute(owner, Compute::kForwardTraining, CUDNN_STATUS_SUCCESS);
        compute(owner, Compute::kBackwardData, CUDNN_STATUS_BAD_PARAM);
        compute(owner, Compute::kBackwardWeights, CUDNN_STATUS_SUCCESS);
        owner.close();
        for (const auto count : calls.calls)
            expect(count > 0, "all25 actual typed callbacks were reached");
        expect(calls.calls[22] == 1 && calls.calls[23] == 1 && calls.calls[24] == 1,
               "no requested computation was replayed");
        {
            CudaSessionLimits one_command;
            one_command.submissions.max_commands = 1;
            Owner constrained(one_command);
            metadata_and_control(constrained);
            // Both arenas fit, but the second call is destroyed at dispatcher
            // admission while its temporary preflight vectors still exist.
            compute(constrained, Compute::kBackwardWeights, CUDNN_STATUS_SUCCESS);
            constrained.close();
        }
        for (unsigned int mode = 0; mode < 3; ++mode) uncertainty(mode);
#else
        fixture::reset();
        CudaDnnRegistry registry(fixture::dispatch());
        const auto result = registry.legacy_metadata({Metadata::kVersionCheck, 0, 1, 0, 0});
        expect(result.result.status == CUDA_ERROR_NOT_SUPPORTED && !result.result.original_status &&
                   !result.output.written && !registry.create(CudaDnnObjectKind::kAlgorithm).id,
               "SDK9 cannot manufacture legacy native provenance or Algorithm resources");
        native(registry.shutdown());
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
