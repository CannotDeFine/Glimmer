#include "dnn_batch_norm_fixture.h"

#include <iostream>
#include <source_location>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_batch_norm;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("BN Driver failure at line " + std::to_string(where.line()));
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("BN SDK failure at line " + std::to_string(where.line()));
}
enum class Fault {
    kNone,
    kForwardError,
    kBackwardError,
    kForwardCancel,
    kBackwardCancel,
    kShortWorkspace,
    kShortReserve,
    kPartialRunning,
    kPartialSaved,
    kNullAuxiliary,
    kZeroEnd,
    kZeroInterior
};
void exercise(fixture::Case value, Fault fault = Fault::kNone, int missing = 0) {
    fixture::reset();
    const bool zero =
        fault == Fault::kNullAuxiliary || fault == Fault::kZeroEnd || fault == Fault::kZeroInterior;
    if (zero) {
        fixture::required_workspace.store(0);
        fixture::required_reserve.store(0);
    }
    auto sdk = fixture::session_dispatch();
    if (missing) {
        auto incomplete = *sdk.dnn;
        if (missing == 1) incomplete.batch_norm_forward = nullptr;
        if (missing == 2) incomplete.batch_norm_backward = nullptr;
        if (missing == 3) incomplete.batch_norm_forward_workspace = nullptr;
        if (missing == 4) incomplete.batch_norm_backward_workspace = nullptr;
        if (missing == 5) incomplete.batch_norm_reserve_size = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(incomplete);
    }
    CudaDnnRegistry metadata(*sdk.dnn);
    const auto descriptors = fixture::tensors(value);
    std::array<std::uint64_t, 7> ids{};
    for (std::size_t i = 0; i != ids.size(); ++i) {
        const auto created =
            metadata.create(i == 6 ? CudaDnnObjectKind::kActivation : CudaDnnObjectKind::kTensor);
        success(created);
        ids[i] = created.id;
        if (i == 6)
            success(metadata.set_activation(ids[i],
                                            {CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0}));
        else
            success(metadata.set_tensor(ids[i], CudaDnnTensorSetter::kNd, descriptors[i]));
    }
    expect(!foundation::control.context_creates.load(), "BN descriptors need no CUDA context");
    CudaExecutionSession session({81, 82}, 83, sdk);
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    if (value.is_double) {
        const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        success(session.set_dnn_stream(handle.id, stream.id));
    }
    auto problem_for = [&](CudaDnnBatchNormKind kind) {
        CudaSessionDnnBatchNorm problem;
        problem.kind = kind;
        problem.handle = handle.id;
        problem.mode = value.mode;
        problem.operation = value.operation;
        const auto mask = fixture::descriptor_mask(value, kind);
        for (std::size_t i = 0; i != mask.size(); ++i)
            if (mask[i])
                problem.descriptors[i] = *metadata.borrow(
                    ids[i], i == 6 ? CudaDnnObjectKind::kActivation : CudaDnnObjectKind::kTensor);
        return problem;
    };
    auto queries = [&](bool detailed) {
        for (auto kind : {CudaDnnBatchNormKind::kForward, CudaDnnBatchNormKind::kBackward,
                          CudaDnnBatchNormKind::kReserve}) {
            auto problem = problem_for(kind);
            const bool absent = missing == static_cast<int>(kind) + 2;
            const auto result = session.query_dnn_batch_norm(problem);
            if (absent) {
                expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status,
                       "missing optional BN query rejects without original status");
                continue;
            }
            if (value.mode == 3 || value.operation == 3) {
                expect(result.original_status == CUDNN_STATUS_NOT_SUPPORTED,
                       "representable unnamed BN enum reaches native query");
                continue;
            }
            success(result);
            expect(result.value == (kind == CudaDnnBatchNormKind::kReserve
                                        ? fixture::required_reserve.load()
                                        : fixture::required_workspace.load()),
                   "independent BN workspace and reserve queries retain native sizes");
            if (detailed) {
                fixture::override_query.store(true);
                fixture::query_value.store(SIZE_MAX - 9);
                const auto wide = session.query_dnn_batch_norm(problem);
                success(wide);
                expect(wide.value == SIZE_MAX - 9, "BN size result retains every size_t bit");
                fixture::override_query.store(false);
                fixture::fail_query.store(true);
                const auto failed = session.query_dnn_batch_norm(problem);
                expect(
                    failed.original_status == CUDNN_STATUS_NOT_SUPPORTED_SHAPE && failed.value == 0,
                    "failed native BN query cannot publish touched output");
            }
        }
        expect(!foundation::control.context_synchronizes.load(),
               "BN queries never drain a context");
    };
    queries(true);
    {
        auto invalid = problem_for(CudaDnnBatchNormKind::kForward);
        const auto calls = fixture::state.queries;
        invalid.mode = UINT32_MAX;
        expect(session.query_dnn_batch_norm(invalid).status == CUDA_ERROR_INVALID_VALUE &&
                   !session.accept_dnn_batch_norm(1, invalid).acceptance.accepted() &&
                   fixture::state.queries == calls,
               "unrepresentable enum is rejected before query or compute conversion");
    }
    const auto initial = fixture::images(value);
    auto expected = initial;
    std::array<CudaSessionResult, 13> allocations;
    for (std::size_t i = 0; i != allocations.size(); ++i) {
        allocations[i] = session.allocate(initial[i].size());
        success(allocations[i]);
        success(session.upload({allocations[i].id, 0}, initial[i]));
    }
    std::uint64_t sequence{1};
    for (auto kind : {CudaDnnBatchNormKind::kForward, CudaDnnBatchNormKind::kBackward}) {
        const bool forward = kind == CudaDnnBatchNormKind::kForward;
        auto problem = problem_for(kind);
        const auto buffers = fixture::buffer_mask(value, kind);
        for (std::size_t i = 0; i != buffers.size(); ++i)
            if (buffers[i]) problem.buffers[i] = {allocations[i].id, 16};
        problem.workspace_bytes = fixture::required_workspace.load();
        problem.reserve_bytes = fixture::required_reserve.load();
        problem.epsilon = value.epsilon;
        problem.average_factor = forward ? value.factor : 0;
        problem.scalar_bytes = value.is_double ? 8 : 4;
        const std::array coefficients =
            forward ? std::array{value.forward[0], value.forward[1], 0., 0.} : value.backward;
        for (std::size_t i = 0; i != (forward ? 2U : 4U); ++i)
            if (value.is_double)
                std::memcpy(problem.coefficients[i].data(), &coefficients[i], 8);
            else {
                const auto coefficient = static_cast<float>(coefficients[i]);
                std::memcpy(problem.coefficients[i].data(), &coefficient, 4);
            }
        if (fault == Fault::kNullAuxiliary) {
            problem.buffers[11] = problem.buffers[12] = {};
            problem.workspace_bytes = SIZE_MAX - 11;
            problem.reserve_bytes = SIZE_MAX - 13;
        } else if (fault == Fault::kZeroEnd) {
            problem.buffers[11].offset = initial[11].size();
            problem.buffers[12].offset = initial[12].size();
        } else if (fault == Fault::kShortWorkspace)
            problem.workspace_bytes = 31;
        else if (fault == Fault::kShortReserve)
            problem.reserve_bytes = 31;
        else if (fault == Fault::kPartialRunning && forward)
            problem.buffers[8] = {};
        else if (fault == Fault::kPartialSaved)
            problem.buffers[10] = {};
        auto invalid = problem;
        invalid.descriptors[5].lifetime.reset();
        expect(!session.accept_dnn_batch_norm(sequence, invalid).acceptance.accepted(),
               "native BN parameter descriptor alone is not an immutable lease");
        invalid = problem;
        invalid.buffers[5].offset = initial[5].size();
        expect(!session.accept_dnn_batch_norm(sequence, invalid).acceptance.accepted(),
               "parameter extents are independently checked");
        invalid = problem;
        invalid.buffers[11] = {0, 1};
        expect(!session.accept_dnn_batch_norm(sequence, invalid).acceptance.accepted(),
               "null auxiliary identity cannot carry an offset");
        invalid = {};
        const auto captured_buffers = problem.buffers;
        const auto workspace_bytes = problem.workspace_bytes, reserve_bytes = problem.reserve_bytes;
        const auto accepted = session.accept_dnn_batch_norm(sequence++, problem);
        const bool absent = missing == (forward ? 1 : 2);
        const bool cancel = fault == (forward ? Fault::kForwardCancel : Fault::kBackwardCancel);
        const bool fails = fault == (forward ? Fault::kForwardError : Fault::kBackwardError);
        if (absent)
            expect(!accepted.acceptance.accepted(), "missing optional BN compute rejects");
        else
            expect(accepted.acceptance.accepted(),
                   "owned BN command admitted without native issue");
        for (auto& coefficient : problem.coefficients) coefficient.fill(std::byte{0});
        problem = {};
        if (!absent) {
            expect(metadata.destroy(ids[0]).disposition == CudaDnnDisposition::kDeferred &&
                       metadata.set_tensor(ids[5], CudaDnnTensorSetter::kNd, descriptors[5])
                               .disposition == CudaDnnDisposition::kDeferred &&
                       session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
                   "queued BN holds native handle and immutable descriptors");
            if (cancel) {
                expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
                       "cancel one queued BN");
                const auto host = session.host_result(accepted.acceptance.receipt);
                expect(host &&
                           host->origin ==
                               glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                           host->status == CUDA_ERROR_INVALID_VALUE &&
                           session.acknowledge_result(accepted.acceptance.receipt),
                       "BN cancellation reports no fabricated native status");
            } else {
                foundation::control.can_stop.store(false);
                fixture::fail_compute.store(fails);
                const auto issued = session.issue_next();
                const auto native_status =
                    fails                                     ? CUDNN_STATUS_EXECUTION_FAILED
                    : value.mode == 3 || value.operation == 3 ? CUDNN_STATUS_NOT_SUPPORTED
                    : fault == Fault::kPartialRunning || fault == Fault::kPartialSaved
                        ? CUDNN_STATUS_BAD_PARAM
                    : fault == Fault::kShortWorkspace || fault == Fault::kShortReserve
                        ? CUDNN_STATUS_INVALID_VALUE
                        : CUDNN_STATUS_SUCCESS;
                const auto host = session.host_result(accepted.acceptance.receipt);
                expect(issued.state == CudaExecutionIssueState::kIssued &&
                           issued.library_status == native_status && host &&
                           host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                           host->status == native_status,
                       "BN preserves original SDK result after actual execution");
                const auto& call = fixture::state.last;
                expect(call.coefficients == coefficients && call.epsilon == value.epsilon &&
                           call.average_factor == (forward ? value.factor : 0) &&
                           call.workspace_bytes == workspace_bytes &&
                           call.reserve_bytes == reserve_bytes,
                       "all BN coefficient and double-value snapshots survive unchanged");
                for (std::size_t i = 0; i != captured_buffers.size(); ++i)
                    expect(call.buffers[i] ==
                               (captured_buffers[i].allocation
                                    ? reinterpret_cast<void*>(allocations[i].address +
                                                              captured_buffers[i].offset)
                                    : nullptr),
                           "every optional/native BN pointer keeps its own identity");
                expect(session.acknowledge_result(accepted.acceptance.receipt) &&
                           !session.poll().retired,
                       "BN host ACK is not physical completion");
                const auto native_calls = fixture::state.forwards + fixture::state.backwards;
                queries(false);
                expect(!foundation::control.can_stop.load() && !session.poll().retired &&
                           fixture::state.forwards + fixture::state.backwards == native_calls,
                       "all three BN queries return while original event is NOT_READY");
                for (const auto& allocation : allocations)
                    expect(session.release(allocation.id).disposition ==
                               CudaSessionDisposition::kDeferred,
                           "BN retains every allocation until physical proof");
                foundation::control.can_stop.store(true);
                expect(session.poll().retired == 1, "physical BN retirement is once-only");
                if (native_status == CUDNN_STATUS_SUCCESS)
                    expected = forward ? fixture::oracle_forward(value, expected)
                                       : fixture::oracle_backward(value, expected);
            }
        }
        auto actual = initial;
        for (std::size_t i = 0; i != actual.size(); ++i)
            success(session.download({allocations[i].id, 0}, actual[i]));
        fixture::verify(value, actual, expected);
        if (forward && (absent || cancel || fails || value.mode == 3 || value.operation == 3 ||
                        fault == Fault::kPartialRunning || fault == Fault::kPartialSaved ||
                        fault == Fault::kShortWorkspace || fault == Fault::kShortReserve))
            break;
    }
    for (const auto& allocation : allocations) success(session.release(allocation.id));
    success(metadata.shutdown());
    success(session.shutdown());
    expect(base::state.creates == 7 && base::state.destroys == 7 &&
               base::state.handle_destroys == 1 && foundation::driver.frees == 13 &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               foundation::driver.streams.empty() && foundation::driver.contexts.empty(),
           "all BN resources retire once after proof");
}
}  // namespace
int main() {
    try {
        for (bool is_double : {false, true})
            for (std::uint32_t operation = 0; operation != 3; ++operation)
                exercise({.is_double = is_double, .operation = operation});
        for (bool is_double : {false, true})
            exercise({.is_double = is_double, .mode = CUDNN_BATCHNORM_PER_ACTIVATION});
        exercise({.mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT});
        exercise({.omit_statistics = true});
        exercise({.ignored_pointers = true});
        for (auto fault : {Fault::kForwardError, Fault::kBackwardError, Fault::kForwardCancel,
                           Fault::kBackwardCancel, Fault::kShortWorkspace, Fault::kShortReserve,
                           Fault::kPartialRunning, Fault::kPartialSaved, Fault::kNullAuxiliary,
                           Fault::kZeroEnd, Fault::kZeroInterior})
            exercise({}, fault);
        for (int missing = 1; missing != 6; ++missing) exercise({}, Fault::kNone, missing);
        exercise({.mode = 3});
        exercise({.operation = 3});
        std::cout
            << "BN Ex CPU forward/backward, native query and physical-lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
