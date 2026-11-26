#include "dnn_lrn_fixture.h"

#include <iostream>
#include <source_location>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_lrn;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("LRN Driver failure at line " + std::to_string(where.line()));
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("LRN SDK failure at line " + std::to_string(where.line()));
}
template <typename Scalar>
void exercise(fixture::Case value, bool fails = false, bool cancel = false,
              bool missing_callback = false) {
    fixture::reset();
    auto sdk = fixture::session_dispatch();
    if (missing_callback) {
        auto incomplete = *sdk.dnn;
        if (value.is_transform)
            incomplete.transform_tensor = nullptr;
        else
            incomplete.lrn_forward = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(incomplete);
    }
    CudaDnnRegistry metadata(*sdk.dnn);
    value.x.type = value.y.type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    std::array<std::uint64_t, 2> tensors{};
    for (auto& tensor : tensors) {
        const auto created = metadata.create(CudaDnnObjectKind::kTensor);
        success(created);
        tensor = created.id;
    }
    success(metadata.set_tensor(tensors[0], CudaDnnTensorSetter::kNd, value.x));
    success(metadata.set_tensor(tensors[1], CudaDnnTensorSetter::kNd, value.y));
    std::uint64_t lrn{};
    if (!value.is_transform) {
        const auto created = metadata.create(CudaDnnObjectKind::kLrn);
        success(created);
        lrn = created.id;
        expect(!fixture::state.sets, "original LRN creation has no framework-invented setter");
        if (!value.use_defaults) success(metadata.set_lrn(lrn, value.parameters));
    }
    expect(!foundation::control.context_creates.load(),
           "LRN descriptor operations are CPU metadata");
    CudaExecutionSession session({61, 62}, 63, sdk);
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    if (!value.use_defaults) {
        const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        success(session.set_dnn_stream(handle.id, stream.id));
    }
    const auto input = fixture::image<Scalar>(value, false);
    const auto initial = fixture::image<Scalar>(value, true);
    const auto x = session.allocate(input.size() + (value.shares_allocation ? initial.size() : 0));
    success(x);
    const auto y = value.shares_allocation ? x : session.allocate(initial.size());
    success(y);
    const auto output_offset = value.shares_allocation ? input.size() : 0;
    success(session.upload({x.id, 0}, input));
    success(session.upload({y.id, output_offset}, initial));
    CudaSessionDnnCompute problem;
    problem.kind =
        value.is_transform ? CudaDnnComputeKind::kTransformTensor : CudaDnnComputeKind::kLrnForward;
    problem.handle = handle.id;
    problem.tensors = {*metadata.borrow(tensors[0], CudaDnnObjectKind::kTensor),
                       *metadata.borrow(tensors[1], CudaDnnObjectKind::kTensor)};
    if (lrn) problem.lrn = *metadata.borrow(lrn, CudaDnnObjectKind::kLrn);
    problem.buffers = {{{x.id, 16}, {y.id, output_offset + 16}}};
    problem.mode = value.mode;
    problem.scalar_bytes = sizeof(Scalar);
    const Scalar alpha{1.5}, beta{-0.25};
    std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
    std::memcpy(problem.beta.data(), &beta, sizeof(beta));
    auto invalid = problem;
    invalid.buffers[1].offset += 17;
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "complete strided destination hull is checked before issue");
    invalid = problem;
    invalid.tensors[0].lifetime.reset();
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "tensor native pointer alone cannot authorize LRN or transform");
    invalid = problem;
    if (lrn)
        invalid.lrn.lifetime.reset();
    else
        invalid.algorithm = 1;
    expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
           "operation lease or unused algorithm fields are independently validated");
    if (lrn) {
        invalid = problem;
        invalid.mode = 2;
        expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
               "unrepresentable LRN mode rejects before original SDK invocation");
    }
    if (value.is_transform && value.shares_allocation) {
        invalid = problem;
        invalid.buffers[1].offset = 16;
        expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
               "exact shared-allocation hull overlap rejects before native transform");
        invalid.buffers[1].offset = 24;
        expect(!session.accept_dnn_compute(1, invalid).acceptance.accepted(),
               "partial shared-allocation hull overlap also rejects before native transform");
    }
    invalid = {};
    const auto accepted = session.accept_dnn_compute(1, problem);
    if (missing_callback) {
        expect(
            !accepted.acceptance.accepted() && !fixture::state.lrns && !fixture::state.transforms,
            "missing optional native compute callback fails without any original call");
        problem = {};
    } else {
        expect(accepted.acceptance.accepted() && !fixture::state.lrns && !fixture::state.transforms,
               "owned admission captures arguments without executing the native function");
        problem.alpha.fill(std::byte{0});
        problem.beta.fill(std::byte{0});
        problem = {};
        expect(metadata.destroy(tensors[0]).disposition == CudaDnnDisposition::kDeferred &&
                   metadata.set_tensor(tensors[1], CudaDnnTensorSetter::kNd, value.y).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
               "queued LRN/transform retains both tensor descriptors and native handle");
        if (lrn)
            expect(metadata.set_lrn(lrn, value.parameters).disposition ==
                           CudaDnnDisposition::kDeferred &&
                       metadata.destroy(lrn).disposition == CudaDnnDisposition::kDeferred,
                   "queued LRN metadata cannot change before original issue");
        if (cancel) {
            expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
                   "cancel exactly one unissued LRN/transform command");
            const auto host = session.host_result(accepted.acceptance.receipt);
            expect(
                host &&
                    host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                    host->status == CUDA_ERROR_INVALID_VALUE &&
                    session.acknowledge_result(accepted.acceptance.receipt) &&
                    !fixture::state.lrns && !fixture::state.transforms,
                "queued cancellation never manufactures a native computation status");
        } else {
            foundation::control.can_stop.store(false);
            fixture::fail_compute.store(fails);
            const auto issued = session.issue_next();
            const auto host = session.host_result(accepted.acceptance.receipt);
            const auto expected_status = fails        ? CUDNN_STATUS_EXECUTION_FAILED
                                         : value.mode ? CUDNN_STATUS_BAD_PARAM
                                                      : CUDNN_STATUS_SUCCESS;
            const auto& call = fixture::state.last;
            expect(issued.state == CudaExecutionIssueState::kIssued &&
                       issued.library_status == expected_status && host &&
                       host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                       host->status == expected_status && call.alpha == alpha &&
                       call.beta == beta && call.mode == value.mode &&
                       call.x == reinterpret_cast<const void*>(x.address + 16) &&
                       call.y == reinterpret_cast<void*>(y.address + output_offset + 16) &&
                       fixture::state.lrns + fixture::state.transforms == 1,
                   "native original receives captured scalars, exact views and error domain once");
            if (lrn)
                expect(call.parameters.n == value.parameters.n &&
                           std::bit_cast<std::uint64_t>(call.parameters.alpha) ==
                               std::bit_cast<std::uint64_t>(value.parameters.alpha) &&
                           std::bit_cast<std::uint64_t>(call.parameters.beta) ==
                               std::bit_cast<std::uint64_t>(value.parameters.beta) &&
                           std::bit_cast<std::uint64_t>(call.parameters.k) ==
                               std::bit_cast<std::uint64_t>(value.parameters.k) &&
                           fixture::state.sets == static_cast<unsigned int>(!value.use_defaults),
                       "original default or exact double LRN parameters survive without "
                       "normalization");
            expect(session.acknowledge_result(accepted.acceptance.receipt) &&
                       !session.poll().retired &&
                       session.release(x.id).disposition == CudaSessionDisposition::kDeferred &&
                       session.release(y.id).disposition == CudaSessionDisposition::kDeferred &&
                       metadata.destroy(tensors[0]).disposition == CudaDnnDisposition::kDeferred,
                   "native return and ACK do not release any allocation or metadata lease");
            if (lrn)
                expect(metadata.destroy(lrn).disposition == CudaDnnDisposition::kDeferred,
                       "LRN operation descriptor remains owned through physical completion");
            foundation::control.can_stop.store(true);
            expect(session.poll().retired == 1,
                   "independent event proves physical retirement once");
        }
    }
    std::vector<std::byte> actual(initial.size()), unchanged(input.size());
    success(session.download({y.id, output_offset}, actual));
    success(session.download({x.id, 0}, unchanged));
    expect(unchanged == input, "source, guards and every input stride hole remain unchanged");
    fixture::verify<Scalar>(value, actual, initial,
                            fails || cancel || missing_callback || value.mode);
    success(session.release(x.id));
    if (!value.shares_allocation) success(session.release(y.id));
    success(metadata.shutdown());
    success(session.shutdown());
    expect(fixture::state.creates == static_cast<unsigned int>(!value.is_transform) &&
               fixture::state.destroys == fixture::state.creates && base::state.destroys == 2 &&
               base::state.handle_destroys == 1 &&
               foundation::driver.frees == (value.shares_allocation ? 1U : 2U) &&
               foundation::driver.contexts.empty() && foundation::driver.events.empty() &&
               foundation::driver.streams.empty(),
           "all proven LRN/transform resources release exactly once");
}
template <typename Scalar>
void matrix() {
    for (const auto n : {5U, 3U, 4U}) exercise<Scalar>(fixture::lrn_case(n));
    exercise<Scalar>(fixture::transform_case(false));
    exercise<Scalar>(fixture::transform_case(true));
}
}  // namespace
int main() {
    try {
        matrix<float>();
        matrix<double>();
        exercise<float>(fixture::lrn_case(3), true);
        exercise<double>(fixture::transform_case(true), true);
        exercise<float>(fixture::lrn_case(), false, true);
        exercise<double>(fixture::transform_case(true), false, true);
        exercise<float>(fixture::lrn_case(), false, false, true);
        exercise<double>(fixture::transform_case(false), false, false, true);
        auto invalid_mode = fixture::lrn_case();
        invalid_mode.mode = 1;
        exercise<float>(invalid_mode);
        std::cout << "LRN/transform CPU numerical, alias and physical-lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
