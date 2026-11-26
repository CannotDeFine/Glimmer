#include "dnn_pooling_fixture.h"

#include <iostream>
#include <source_location>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_pooling;
namespace base = glimmer::tests::dnn;
namespace foundation = fixture::foundation;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("pooling Driver failure at line " + std::to_string(where.line()));
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("pooling SDK failure at line " + std::to_string(where.line()));
}
template <typename Scalar>
void exercise(fixture::Case value, cudnnStatus_t native_status = CUDNN_STATUS_SUCCESS,
              bool cancel = false) {
    fixture::reset();
    const auto sdk = fixture::session_dispatch();
    CudaDnnRegistry metadata(*sdk.dnn);
    constexpr auto type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    value.x.type = value.y.type = type;
    std::array<std::uint64_t, 2> tensors{};
    for (auto& tensor : tensors) {
        const auto created = metadata.create(CudaDnnObjectKind::kTensor);
        success(created);
        tensor = created.id;
    }
    success(metadata.set_tensor(tensors[0], CudaDnnTensorSetter::kNd, value.x));
    success(metadata.set_tensor(tensors[1], CudaDnnTensorSetter::kNd, value.y));
    std::uint64_t pooling{};
    if (value.is_pooling) {
        const auto created = metadata.create(CudaDnnObjectKind::kPooling);
        success(created);
        pooling = created.id;
        success(metadata.set_pooling(pooling, value.pooling));
        const auto shape = metadata.pooling_output(pooling, tensors[0], 4);
        success(shape);
        if (value.y.sizes[2] == 3)
            expect(shape.tensor.sizes[2] == 4,
                   "smaller native pooling output is not forced to query");
        const auto oversized = metadata.get_pooling(pooling, 9);
        expect(oversized.original_status == CUDNN_STATUS_NOT_SUPPORTED &&
                   !metadata.snapshot().is_quarantined,
               "pooling oversized getter preserves original error rather than clamping");
    }
    expect(!foundation::control.context_creates.load(), "pooling metadata is context independent");
    CudaExecutionSession session({71, 72}, 73, sdk);
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    if (!value.is_inplace) {
        const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        success(session.set_dnn_stream(handle.id, stream.id));
    }
    const auto input = fixture::image<Scalar>(value.x, value.input);
    const auto initial = fixture::image<Scalar>(value.y, value.initial);
    const auto x = session.allocate(input.size());
    success(x);
    const auto y = value.is_inplace ? x : session.allocate(initial.size());
    success(y);
    success(session.upload({x.id, 0}, input));
    if (!value.is_inplace) success(session.upload({y.id, 0}, initial));
    CudaSessionDnnCompute problem;
    problem.kind = value.is_pooling ? CudaDnnComputeKind::kPoolingForward
                                    : CudaDnnComputeKind::kSoftmaxForward;
    problem.handle = handle.id;
    problem.tensors = {*metadata.borrow(tensors[0], CudaDnnObjectKind::kTensor),
                       *metadata.borrow(tensors[1], CudaDnnObjectKind::kTensor)};
    if (pooling) problem.pooling = *metadata.borrow(pooling, CudaDnnObjectKind::kPooling);
    problem.buffers = {{{x.id, 16}, {y.id, 16}}};
    problem.algorithm = value.algorithm;
    problem.mode = value.mode;
    problem.scalar_bytes = sizeof(Scalar);
    const Scalar alpha{1.25}, beta{-0.5};
    std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
    std::memcpy(problem.beta.data(), &beta, sizeof(beta));
    auto invalid = problem;
    invalid.buffers[0].offset += 17;
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "strided input's complete native byte hull must fit the owned range");
    invalid = problem;
    invalid.buffers[1].offset += 17;
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "strided output's complete native byte hull must fit the owned range");
    invalid = problem;
    invalid.tensors[0].lifetime.reset();
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "tensor native pointer without lease cannot authorize computation");
    invalid = problem;
    if (pooling)
        invalid.pooling.lifetime.reset();
    else
        invalid.algorithm = 4;
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "missing pooling lease or unrepresentable softmax enum rejects before native call");
    invalid = {};
    base::state.get_status = CUDNN_STATUS_BAD_PARAM;
    const auto failed_prepare = session.accept_dnn_compute(1, problem);
    expect(!failed_prepare.acceptance.accepted() &&
               failed_prepare.original_status == CUDNN_STATUS_BAD_PARAM,
           "native metadata failure retains original status and does not issue computation");
    base::state.get_status = CUDNN_STATUS_SUCCESS;
    const auto accepted = session.accept_dnn_compute(1, problem);
    expect(accepted.acceptance.accepted() && !fixture::state.pools && !fixture::state.softmaxes,
           "admission retains immutable arguments without executing native arithmetic");
    problem = {};
    expect(metadata.set_tensor(tensors[0], CudaDnnTensorSetter::kNd, value.x).disposition ==
                   CudaDnnDisposition::kDeferred &&
               metadata.destroy(tensors[1]).disposition == CudaDnnDisposition::kDeferred &&
               session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
           "queued computation retains tensors and native handle");
    if (pooling)
        expect(metadata.set_pooling(pooling, value.pooling).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   metadata.destroy(pooling).disposition == CudaDnnDisposition::kDeferred,
               "queued pooling keeps its operation metadata immutable");
    if (cancel) {
        expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
               "cancel one accepted computation before original issue");
        const auto host = session.host_result(accepted.acceptance.receipt);
        expect(host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   host->status == CUDA_ERROR_INVALID_VALUE &&
                   session.acknowledge_result(accepted.acceptance.receipt) &&
                   !fixture::state.pools && !fixture::state.softmaxes,
               "cancelled pooling/softmax never manufactures an original native result");
    } else {
        foundation::control.can_stop.store(false);
        fixture::state.compute_status = native_status;
        const auto issued = session.issue_next();
        const auto host = session.host_result(accepted.acceptance.receipt);
        const auto expected_status = value.algorithm == 3 ? CUDNN_STATUS_BAD_PARAM : native_status;
        expect(issued.state == CudaExecutionIssueState::kIssued &&
                   issued.library_status == expected_status && host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                   host->status == expected_status &&
                   fixture::state.pools + fixture::state.softmaxes == 1 &&
                   fixture::state.last.alpha == alpha && fixture::state.last.beta == beta &&
                   fixture::state.last.algorithm == value.algorithm &&
                   fixture::state.last.mode == value.mode &&
                   fixture::state.last.x == reinterpret_cast<const void*>(x.address + 16) &&
                   fixture::state.last.y == reinterpret_cast<void*>(y.address + 16) &&
                   (value.is_inplace ? fixture::state.last.stream == nullptr
                                     : fixture::state.last.stream != nullptr),
               "original arithmetic receives exact scalars, enums, addresses, alias and stream");
        expect(session.acknowledge_result(accepted.acceptance.receipt) && !session.poll().retired &&
                   session.release(x.id).disposition == CudaSessionDisposition::kDeferred &&
                   session.release(y.id).disposition == CudaSessionDisposition::kDeferred &&
                   metadata.destroy(tensors[0]).disposition == CudaDnnDisposition::kDeferred &&
                   metadata.destroy(tensors[1]).disposition == CudaDnnDisposition::kDeferred,
               "host status/ACK cannot release tensor or strided allocation lifetimes");
        if (pooling)
            expect(metadata.destroy(pooling).disposition == CudaDnnDisposition::kDeferred,
                   "pooling metadata remains retained after host ACK");
        foundation::control.can_stop.store(true);
        expect(session.poll().retired == 1,
               "physical event independently retires original compute");
    }
    std::vector<std::byte> actual(initial.size());
    success(session.download({y.id, 0}, actual));
    fixture::verify<Scalar>(
        value, actual, value.is_inplace ? input : initial,
        cancel || native_status != CUDNN_STATUS_SUCCESS || value.algorithm == 3);
    if (!value.is_inplace) {
        std::vector<std::byte> unchanged(input.size());
        success(session.download({x.id, 0}, unchanged));
        expect(unchanged == input, "out-of-place input, guards and stride holes stay unchanged");
    }
    success(session.release(x.id));
    if (!value.is_inplace) success(session.release(y.id));
    success(metadata.shutdown());
    success(session.shutdown());
    expect(fixture::state.creates == static_cast<unsigned int>(value.is_pooling) &&
               fixture::state.destroys == fixture::state.creates && base::state.destroys == 2 &&
               base::state.handle_destroys == 1 &&
               foundation::driver.frees == (value.is_inplace ? 1U : 2U) &&
               foundation::driver.contexts.empty() && foundation::driver.events.empty() &&
               foundation::driver.streams.empty(),
           "all known tensor-forward ownership releases exactly once");
}
template <typename Scalar>
void numerical_matrix() {
    for (std::uint32_t mode = 0; mode != 4; ++mode) exercise<Scalar>(fixture::pool_case(mode));
    exercise<Scalar>(fixture::pool_case(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, true));
    exercise<Scalar>(fixture::pool_case(CUDNN_POOLING_MAX, false, true));
    exercise<Scalar>(fixture::pool_case(CUDNN_POOLING_MAX, false, false, true));
    for (std::uint32_t algorithm = 0; algorithm != 3; ++algorithm)
        for (std::uint32_t mode = 0; mode != 2; ++mode)
            exercise<Scalar>(fixture::softmax_case(algorithm, mode));
    exercise<Scalar>(
        fixture::softmax_case(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, true));
    exercise<Scalar>(fixture::softmax_case(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, true));
}
}  // namespace
int main() {
    try {
        numerical_matrix<float>();
        numerical_matrix<double>();
        exercise<float>(fixture::pool_case(CUDNN_POOLING_MAX), CUDNN_STATUS_EXECUTION_FAILED);
        exercise<double>(fixture::softmax_case(1, 1), CUDNN_STATUS_EXECUTION_FAILED);
        exercise<float>(fixture::pool_case(CUDNN_POOLING_MAX), CUDNN_STATUS_SUCCESS, true);
        exercise<double>(fixture::softmax_case(1, 1), CUDNN_STATUS_SUCCESS, true);
        exercise<float>(fixture::softmax_case(3, 0));
        std::cout << "Pooling/softmax CPU numerical, alias and physical-lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
