#include "dnn_reduction_fixture.h"

#include <iostream>
#include <source_location>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_reduction;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("reduction Driver failure at line " +
                                 std::to_string(where.line()));
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("reduction SDK failure at line " + std::to_string(where.line()));
}
enum class Buffers {
    kPositive,
    kNullIgnored,
    kZeroInterior,
    kZeroEnd,
    kShortIndices,
    kShortWorkspace,
    kNullWorkspace
};
void exercise(fixture::Case value, Buffers mode = Buffers::kPositive, bool fails = false,
              bool cancel = false, int missing = 0) {
    fixture::reset();
    if (mode == Buffers::kNullIgnored || mode == Buffers::kZeroInterior ||
        mode == Buffers::kZeroEnd)
        fixture::required_workspace.store(0);
    auto sdk = fixture::session_dispatch();
    if (missing) {
        auto incomplete = *sdk.dnn;
        if (missing == 1) incomplete.reduce_tensor = nullptr;
        if (missing == 2) incomplete.reduction_indices_size = nullptr;
        if (missing == 3) incomplete.reduction_workspace_size = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(incomplete);
    }
    CudaDnnRegistry metadata(*sdk.dnn);
    std::array<std::uint64_t, 3> ids{};
    constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kReduction,
                               CudaDnnObjectKind::kTensor};
    for (std::size_t i = 0; i != ids.size(); ++i) {
        const auto created = metadata.create(kinds[i]);
        success(created);
        ids[i] = created.id;
    }
    expect(!fixture::state.sets && !foundation::control.context_creates.load(),
           "native descriptor creation has no hidden initialization or CUDA context");
    success(metadata.set_tensor(ids[0], CudaDnnTensorSetter::kNd, value.x));
    success(metadata.set_tensor(ids[2], CudaDnnTensorSetter::kNd, value.y));
    success(metadata.set_reduction(ids[1], value.reduction));
    CudaExecutionSession session({71, 72}, 73, sdk);
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    if (value.reduction.operation % 2) {
        const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        success(session.set_dnn_stream(handle.id, stream.id));
    }
    CudaSessionDnnReduction problem;
    problem.handle = handle.id;
    for (std::size_t i = 0; i != ids.size(); ++i)
        problem.descriptors[i] = *metadata.borrow(ids[i], kinds[i]);
    for (const auto kind :
         {CudaDnnReductionQueryKind::kIndices, CudaDnnReductionQueryKind::kWorkspace}) {
        const bool is_missing = (missing == 2 && kind == CudaDnnReductionQueryKind::kIndices) ||
                                (missing == 3 && kind == CudaDnnReductionQueryKind::kWorkspace);
        const auto queried = session.query_dnn_reduction(problem, kind);
        if (is_missing) {
            expect(queried.status == CUDA_ERROR_NOT_SUPPORTED && !queried.original_status,
                   "missing size callback rejects without inventing a native result");
            continue;
        }
        success(queried);
        expect(queried.value == (kind == CudaDnnReductionQueryKind::kWorkspace
                                     ? fixture::required_workspace.load()
                                 : fixture::needs_indices(value.reduction) ? 16U
                                                                           : 0U),
               "original size query retains independent indices/workspace semantics");
        fixture::override_query.store(true);
        fixture::query_value.store(SIZE_MAX - 7);
        const auto wide = session.query_dnn_reduction(problem, kind);
        success(wide);
        expect(wide.value == SIZE_MAX - 7,
               "size query does not truncate to a signed or 32-bit value");
        fixture::override_query.store(false);
        fixture::fail_query.store(true);
        const auto failed = session.query_dnn_reduction(problem, kind);
        expect(failed.original_status == CUDNN_STATUS_NOT_SUPPORTED_SHAPE &&
                   failed.status == CUDA_SUCCESS && failed.value == 0,
               "native size-query error retains exact status without publishing its output");
    }
    expect(!fixture::state.computes && !foundation::control.context_synchronizes.load() &&
               foundation::driver.events.empty(),
           "metadata size queries never issue computation or a GPU drain");
    const auto input = fixture::image(value, false), initial = fixture::image(value, true);
    const std::vector<std::byte> indices(48, std::byte{0xcd}), workspace(64, std::byte{0xcd});
    const std::array images{input, initial, indices, workspace};
    std::array<CudaSessionResult, 4> allocations{};
    for (std::size_t i = 0; i != allocations.size(); ++i) {
        allocations[i] = session.allocate(images[i].size());
        success(allocations[i]);
        success(session.upload({allocations[i].id, 0}, images[i]));
        problem.buffers[i] = {allocations[i].id, 16};
    }
    problem.indices_bytes = fixture::needs_indices(value.reduction) ? 16 : 0;
    problem.workspace_bytes = fixture::required_workspace.load();
    if (mode == Buffers::kNullIgnored) {
        problem.buffers[2] = problem.buffers[3] = {};
        problem.indices_bytes = SIZE_MAX - 13;
        problem.workspace_bytes = SIZE_MAX - 17;
    } else if (mode == Buffers::kZeroEnd) {
        problem.buffers[2].offset = indices.size();
        problem.buffers[3].offset = workspace.size();
    } else if (mode == Buffers::kShortIndices)
        problem.indices_bytes = 15;
    else if (mode == Buffers::kShortWorkspace)
        problem.workspace_bytes = 31;
    else if (mode == Buffers::kNullWorkspace)
        problem.buffers[3] = {};
    problem.scalar_bytes = value.x.type == CUDNN_DATA_DOUBLE ? 8 : 4;
    if (problem.scalar_bytes == 8) {
        std::memcpy(problem.alpha.data(), &value.alpha, 8);
        std::memcpy(problem.beta.data(), &value.beta, 8);
    } else {
        const float alpha = static_cast<float>(value.alpha), beta = static_cast<float>(value.beta);
        std::memcpy(problem.alpha.data(), &alpha, 4);
        std::memcpy(problem.beta.data(), &beta, 4);
    }
    auto invalid = problem;
    invalid.descriptors[1].lifetime.reset();
    expect(!session.accept_dnn_reduction(1, invalid).acceptance.accepted(),
           "native reduction pointer without its immutable lease cannot execute");
    invalid = problem;
    invalid.descriptors[1] = problem.descriptors[0];
    expect(!session.accept_dnn_reduction(1, invalid).acceptance.accepted(),
           "descriptor kind cannot be substituted");
    for (std::size_t i = 0; i != 4; ++i) {
        invalid = problem;
        invalid.buffers[i] = {allocations[i].id, images[i].size()};
        if (i == 2) invalid.indices_bytes = 1;
        if (i == 3) invalid.workspace_bytes = 1;
        expect(!session.accept_dnn_reduction(1, invalid).acceptance.accepted(),
               "complete tensor and optional positive extents are range checked");
    }
    invalid = problem;
    invalid.buffers[2] = {0, 1};
    expect(!session.accept_dnn_reduction(1, invalid).acceptance.accepted(),
           "null optional identity may not carry a nonzero offset");
    invalid = {};
    const auto native_buffers = problem.buffers;
    const auto native_indices_bytes = problem.indices_bytes;
    const auto native_workspace_bytes = problem.workspace_bytes;
    const auto accepted = session.accept_dnn_reduction(1, problem);
    const bool skipped = missing == 1;
    if (skipped)
        expect(!accepted.acceptance.accepted(), "missing reduction callback rejects admission");
    else
        expect(accepted.acceptance.accepted() && !fixture::state.computes,
               "admission snapshots arguments without native computation");
    // Expectations retain only buffer identities, never descriptor leases.
    problem.alpha.fill(std::byte{0});
    problem.beta.fill(std::byte{0});
    problem = {};
    // The snapshot below intentionally carries only buffer identities, never leases.
    const auto native_status = fails ? CUDNN_STATUS_EXECUTION_FAILED
                               : mode == Buffers::kShortIndices ||
                                       mode == Buffers::kShortWorkspace ||
                                       mode == Buffers::kNullWorkspace
                                   ? CUDNN_STATUS_INVALID_VALUE
                                   : CUDNN_STATUS_SUCCESS;
    if (!skipped) {
        expect(metadata.set_reduction(ids[1], value.reduction).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   metadata.destroy(ids[0]).disposition == CudaDnnDisposition::kDeferred &&
                   session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
               "accepted reduction keeps descriptors immutable and handle owned");
        if (cancel) {
            expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
                   "cancel one queued reduction");
            const auto host = session.host_result(accepted.acceptance.receipt);
            expect(
                host &&
                    host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                    host->status == CUDA_ERROR_INVALID_VALUE && !fixture::state.computes &&
                    session.acknowledge_result(accepted.acceptance.receipt),
                "cancelled reduction never reports a fabricated native status");
        } else {
            foundation::control.can_stop.store(false);
            fixture::fail_compute.store(fails);
            const auto issued = session.issue_next();
            const auto host = session.host_result(accepted.acceptance.receipt);
            expect(issued.state == CudaExecutionIssueState::kIssued &&
                       issued.library_status == native_status && host &&
                       host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                       host->status == native_status && fixture::state.computes == 1,
                   "original reduction executes once and preserves actual native status");
            const auto& call = fixture::state.last;
            expect(call.alpha == value.alpha && call.beta == value.beta &&
                       call.indices_bytes == native_indices_bytes &&
                       call.workspace_bytes == native_workspace_bytes &&
                       call.indices == (native_buffers[2].allocation
                                            ? reinterpret_cast<void*>(allocations[2].address +
                                                                      native_buffers[2].offset)
                                            : nullptr) &&
                       call.workspace == (native_buffers[3].allocation
                                              ? reinterpret_cast<void*>(allocations[3].address +
                                                                        native_buffers[3].offset)
                                              : nullptr),
                   "scalars, optional identity and full-width byte arguments survive exactly");
            expect(
                session.acknowledge_result(accepted.acceptance.receipt) && !session.poll().retired,
                "host ACK cannot replace the physical event");
            {
                CudaSessionDnnReduction query;
                query.handle = handle.id;
                for (std::size_t i = 0; i != ids.size(); ++i)
                    query.descriptors[i] = *metadata.borrow(ids[i], kinds[i]);
                for (const auto kind :
                     {CudaDnnReductionQueryKind::kIndices, CudaDnnReductionQueryKind::kWorkspace}) {
                    if ((missing == 2 && kind == CudaDnnReductionQueryKind::kIndices) ||
                        (missing == 3 && kind == CudaDnnReductionQueryKind::kWorkspace))
                        continue;
                    const auto result = session.query_dnn_reduction(query, kind);
                    success(result);
                    expect(result.value == (kind == CudaDnnReductionQueryKind::kWorkspace
                                                ? fixture::required_workspace.load()
                                            : fixture::needs_indices(value.reduction) ? 16U
                                                                                      : 0U),
                           "original size query completes while native event remains pending");
                }
                expect(!foundation::control.can_stop.load() && !session.poll().retired &&
                           !foundation::control.context_synchronizes.load() &&
                           fixture::state.computes == 1,
                       "pending queries do not drain, reissue or erase the physical frontier");
            }
            for (const auto& allocation : allocations)
                expect(
                    session.release(allocation.id).disposition == CudaSessionDisposition::kDeferred,
                    "all four allocation owners survive a pending original event");
            expect(metadata.destroy(ids[1]).disposition == CudaDnnDisposition::kDeferred,
                   "reduction descriptor remains immutable after native return");
            foundation::control.can_stop.store(true);
            expect(session.poll().retired == 1, "physical event retires reduction exactly once");
        }
    }
    std::array<std::vector<std::byte>, 4> actual = images;
    for (std::size_t i = 0; i != actual.size(); ++i)
        success(session.download({allocations[i].id, 0}, actual[i]));
    expect(actual[0] == input, "entire strided source and guards remain unchanged");
    const bool unchanged = skipped || cancel || native_status != CUDNN_STATUS_SUCCESS;
    fixture::verify(value, actual[1], initial, unchanged);
    fixture::verify_auxiliary(actual[2], actual[3], value, unchanged);
    for (const auto& allocation : allocations) success(session.release(allocation.id));
    success(metadata.shutdown());
    success(session.shutdown());
    expect(fixture::state.creates == 1 && fixture::state.destroys == 1 &&
               base::state.destroys == 2 && base::state.handle_destroys == 1 &&
               foundation::driver.frees == 4 && foundation::driver.contexts.empty() &&
               foundation::driver.events.empty() && foundation::driver.streams.empty(),
           "reduction native resources retire exactly once after proof");
}
}  // namespace
int main() {
    try {
        for (bool is_double : {false, true})
            for (std::uint32_t operation = 0; operation <= CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
                 ++operation)
                exercise(fixture::make_case(operation, is_double));
        for (bool input_is_int8 : {false, true}) {
            auto mixed = fixture::make_case(CUDNN_REDUCE_TENSOR_ADD);
            (input_is_int8 ? mixed.x : mixed.y).type = CUDNN_DATA_INT8;
            mixed.alpha = 2;
            mixed.beta = 1;
            exercise(mixed);
        }
        for (auto mode : {Buffers::kNullIgnored, Buffers::kZeroInterior, Buffers::kZeroEnd,
                          Buffers::kShortWorkspace, Buffers::kNullWorkspace})
            exercise(fixture::make_case(CUDNN_REDUCE_TENSOR_ADD), mode);
        exercise(fixture::make_case(CUDNN_REDUCE_TENSOR_MIN), Buffers::kShortIndices);
        exercise(fixture::make_case(CUDNN_REDUCE_TENSOR_MAX), Buffers::kPositive, true);
        exercise(fixture::make_case(CUDNN_REDUCE_TENSOR_MAX, true), Buffers::kPositive, false,
                 true);
        for (int missing : {1, 2, 3})
            exercise(fixture::make_case(CUDNN_REDUCE_TENSOR_ADD), Buffers::kPositive, false, false,
                     missing);
        std::cout
            << "reduction CPU numerical, mixed storage, native query and lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
