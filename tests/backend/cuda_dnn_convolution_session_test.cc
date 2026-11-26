#include "dnn_convolution_fixture.h"

#include <iostream>
#include <source_location>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location location = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("convolution Driver failure at line " +
                                 std::to_string(location.line()));
}
void success(const CudaDnnResult& result,
             std::source_location location = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("convolution SDK failure at line " +
                                 std::to_string(location.line()));
}
void fixture_context_isolation() {
    fixture::reset();
    const auto sdk = fixture::session_dispatch();
    std::array<CUcontext, 2> contexts{};
    std::array<CUstream, 2> streams{};
    std::array<CUevent, 2> events{};
    for (std::size_t i = 0; i != contexts.size(); ++i) {
        expect(sdk.context_create(&contexts[i], 0, 0) == CUDA_SUCCESS &&
                   sdk.stream_create(&streams[i], CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
                   sdk.execution.event_create(&events[i], CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS,
               "fake contexts create independent resource identities");
    }
    expect(sdk.context_set_current(contexts[0]) == CUDA_SUCCESS &&
               sdk.context_destroy(contexts[0]) == CUDA_SUCCESS &&
               !fixture::driver.streams.contains(streams[0]) &&
               !fixture::driver.events.contains(events[0]) &&
               fixture::driver.streams.contains(streams[1]) &&
               fixture::driver.events.contains(events[1]) &&
               fixture::driver.contexts.contains(contexts[1]) && !fixture::driver.stream_destroys &&
               !fixture::driver.event_destroys,
           "implicit context teardown releases only its resources without SDK destroy calls");
    expect(sdk.context_set_current(contexts[1]) == CUDA_SUCCESS &&
               sdk.stream_destroy(streams[1]) == CUDA_SUCCESS &&
               sdk.execution.event_destroy(events[1]) == CUDA_SUCCESS &&
               sdk.context_destroy(contexts[1]) == CUDA_SUCCESS &&
               fixture::driver.streams.empty() && fixture::driver.events.empty() &&
               fixture::driver.contexts.empty() && fixture::driver.stream_destroys == 1 &&
               fixture::driver.event_destroys == 1 && fixture::control.context_destroys.load() == 2,
           "remaining context resources retain exactly-once explicit destruction");
}
void query_semantics(CudaExecutionSession& session, const CudaSessionDnnConvolution& problem) {
    CudaSessionDnnConvolution handle_only;
    handle_only.handle = problem.handle;
    auto maximum =
        session.query_dnn_convolution(handle_only, CudaDnnConvolutionQueryKind::kMaxCount);
    success(maximum.result);
    expect(maximum.result.value == 12, "native algorithm maximum is not the eight algorithm enums");
    auto workspace =
        session.query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kWorkspace);
    success(workspace.result);
    expect(workspace.result.value == fixture::state.workspace_required,
           "original workspace byte count is authoritative");
    auto heuristic =
        session.query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kHeuristic, 20);
    success(heuristic.result);
    expect(heuristic.count == 10, "heuristic supports more than eight native math variants");
    for (std::size_t i = 0; i < heuristic.count; ++i) {
        const auto& item = heuristic.performance[i];
        expect(item.algorithm == i % 8 &&
                   item.status == static_cast<std::uint32_t>(i == 2 ? CUDNN_STATUS_NOT_SUPPORTED
                                                                    : CUDNN_STATUS_SUCCESS) &&
                   item.time_bits == 0x3f000000U + i * 0x1000U &&
                   item.memory == (std::uint64_t{1} << 33) + i && item.determinism == i % 2 &&
                   item.math == i % 4 &&
                   item.reserved == std::array<std::int32_t, 3>{-7, static_cast<int>(i), 31337},
               "every original performance field survives without raw struct serialization");
    }
    fixture::state.heuristic_status = CUDNN_STATUS_NOT_SUPPORTED;
    heuristic = session.query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kHeuristic, 5);
    expect(heuristic.result.original_status == CUDNN_STATUS_NOT_SUPPORTED && !heuristic.count &&
               !heuristic.performance[0].memory && !session.snapshot().is_quarantined,
           "failed native heuristic never publishes partially written output");
    fixture::state.heuristic_status = CUDNN_STATUS_SUCCESS;
    heuristic = session.query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kHeuristic, 0);
    expect(heuristic.result.original_status == CUDNN_STATUS_BAD_PARAM && !heuristic.count &&
               !session.snapshot().is_quarantined,
           "zero requested capacity preserves original rejection without quarantine");
    fixture::state.query_status = CUDNN_STATUS_BAD_PARAM;
    workspace = session.query_dnn_convolution(problem, CudaDnnConvolutionQueryKind::kWorkspace);
    maximum = session.query_dnn_convolution(handle_only, CudaDnnConvolutionQueryKind::kMaxCount);
    expect(workspace.result.original_status == CUDNN_STATUS_BAD_PARAM && !workspace.result.value &&
               maximum.result.original_status == CUDNN_STATUS_BAD_PARAM && !maximum.result.value,
           "failed native workspace/max-count queries suppress touched outputs");
    fixture::state.query_status = CUDNN_STATUS_SUCCESS;
}
template <typename Scalar>
void exercise(unsigned int workspace_mode, bool fails, int corruption = -1) {
    fixture::reset();
    auto sdk = fixture::session_dispatch();
    CudaDnnRegistry metadata(*sdk.dnn);
    constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kFilter,
                               CudaDnnObjectKind::kConvolution, CudaDnnObjectKind::kTensor};
    std::array<std::uint64_t, 4> descriptors{};
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto created = metadata.create(kinds[i]);
        success(created);
        descriptors[i] = created.id;
    }
    constexpr auto type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    const CudaDnnTensor x{type, 0, 4, {1, 2, 4, 5}, {40, 20, 5, 1}};
    const CudaDnnTensor y{type, 0, 4, {1, 2, 3, 3}, {18, 9, 3, 1}};
    const CudaDnnFilter w{type, 0, 4, {2, 2, 2, 3}};
    const CudaDnnConvolutionState convolution{
        CUDNN_CROSS_CORRELATION, type, 2, {1, 0}, {2, 1}, {1, 1}};
    success(metadata.set_tensor(descriptors[0], CudaDnnTensorSetter::kNd, x));
    success(metadata.set_filter(descriptors[1], w, true));
    success(metadata.set_convolution(descriptors[2], convolution, true));
    success(metadata.set_tensor(descriptors[3], CudaDnnTensorSetter::kNd, y));
    expect(!fixture::control.context_creates.load(), "descriptors precede CUDA initialization");
    CudaExecutionSession session({91, 92}, 93, sdk);
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    CudaSessionResult stream;
    if (workspace_mode != 2) {
        stream = session.create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        success(session.set_dnn_stream(handle.id, stream.id));
    }
    const auto values = fixture::values<Scalar>();
    constexpr std::array<std::size_t, 4> offsets{16, 24, 32, 16};
    const std::array<std::size_t, 4> extents{sizeof(values.x), sizeof(values.w), sizeof(values.y),
                                             47};
    std::array<CudaSessionResult, 4> allocations;
    for (std::size_t i = 0; i < allocations.size(); ++i) {
        allocations[i] = session.allocate(offsets[i] + extents[i] + 16);
        success(allocations[i]);
    }
    success(session.upload({allocations[0].id, offsets[0]}, std::as_bytes(std::span(values.x))));
    success(session.upload({allocations[1].id, offsets[1]}, std::as_bytes(std::span(values.w))));
    success(session.upload({allocations[2].id, offsets[2]}, std::as_bytes(std::span(values.y))));
    CudaSessionDnnConvolution problem;
    problem.handle = handle.id;
    problem.algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    for (std::size_t i = 0; i < 4; ++i) {
        problem.descriptors[i] = *metadata.borrow(descriptors[i], kinds[i]);
        problem.buffers[i] = {allocations[i].id, offsets[i]};
    }
    problem.workspace_bytes = workspace_mode == 0 ? 47 : 0;
    if (workspace_mode == 2) problem.buffers[3] = {};
    const auto workspace_offset = workspace_mode == 3 ? offsets[3] + extents[3] + 16 : offsets[3];
    if (workspace_mode == 3) problem.buffers[3].offset = workspace_offset;
    fixture::state.workspace_required = workspace_mode == 0 ? 32 : 0;
    const Scalar alpha{1.5}, beta{-0.25};
    problem.scalar_bytes = sizeof(Scalar);
    std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
    std::memcpy(problem.beta.data(), &beta, sizeof(beta));
    query_semantics(session, problem);
    if (corruption >= 0) {
        fixture::state.maximum = corruption == 0 ? -1 : 12;
        fixture::state.should_overshoot_count = corruption == 1;
        fixture::state.should_corrupt_performance = corruption == 2;
        const auto result =
            session.query_dnn_convolution(problem,
                                          corruption == 0 ? CudaDnnConvolutionQueryKind::kMaxCount
                                                          : CudaDnnConvolutionQueryKind::kHeuristic,
                                          12);
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_UNKNOWN &&
                   result.result.original_status == CUDNN_STATUS_SUCCESS &&
                   session.snapshot().is_quarantined && !result.count && !result.result.value &&
                   std::ranges::all_of(result.performance,
                                       [](const auto& item) {
                                           return !item.algorithm && !item.status &&
                                                  !item.time_bits && !item.memory &&
                                                  !item.determinism && !item.math &&
                                                  item.reserved == std::array<std::int32_t, 3>{};
                                       }),
               "malformed native success quarantines with provenance and no partial output");
        problem = {};
        success(metadata.shutdown());
        success(session.shutdown());
        return;  // Read-only corruption does not invalidate proven native ownership.
    }
    auto invalid = problem;
    invalid.buffers[0].offset += 17;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "native input extent cannot exceed owned range");
    invalid = problem;
    invalid.buffers[1].offset += 17;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "native filter extent cannot exceed owned range");
    invalid = problem;
    invalid.buffers[2].offset += 17;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "native output extent cannot exceed owned range");
    invalid = problem;
    invalid.buffers[3] = {allocations[3].id, offsets[3]};
    invalid.workspace_bytes = 64;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "full declared workspace, not only native minimum, must fit its allocation");
    invalid = problem;
    invalid.buffers[3] = {};
    invalid.workspace_bytes = 1;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "nonnull declared workspace bytes cannot discard pointer ownership");
    invalid = problem;
    invalid.scalar_bytes = sizeof(Scalar) == 8 ? 4 : 8;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "native tensor type determines coefficient width independently of caller");
    invalid = problem;
    invalid.descriptors[1].lifetime.reset();
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "native filter pointer alone cannot authorize convolution");
    invalid = problem;
    invalid.algorithm = 16;
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "unrepresentable algorithm bits cannot become a native SDK enum");
    invalid = problem;
    invalid.workspace_bytes = 0;
    invalid.buffers[3] = {allocations[3].id, offsets[3] + extents[3] + 17};
    expect(!session.accept_dnn_convolution(1, invalid).acceptance.accepted(),
           "zero-size workspace cannot exceed even the owned one-past-end identity");
    invalid = {};
    base::state.get_status = CUDNN_STATUS_BAD_PARAM;
    const auto bad_query = session.accept_dnn_convolution(1, problem);
    expect(!bad_query.acceptance.accepted() && bad_query.original_status == CUDNN_STATUS_BAD_PARAM,
           "original metadata preparation failure admits no GPU command");
    base::state.get_status = CUDNN_STATUS_SUCCESS;
    const auto accepted = session.accept_dnn_convolution(1, problem);
    expect(accepted.acceptance.accepted() && !fixture::state.forwards,
           "acceptance retains arguments without executing the SDK");
    problem = {};
    expect(metadata.set_filter(descriptors[1], w).disposition == CudaDnnDisposition::kDeferred &&
               metadata.destroy(descriptors[2]).disposition == CudaDnnDisposition::kDeferred &&
               session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
           "queued convolution retains filter, convolution descriptor and native handle");
    fixture::control.can_stop.store(false);
    fixture::state.forward_status = fails ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS;
    const auto issued = session.issue_next();
    const auto host = session.host_result(accepted.acceptance.receipt);
    expect(issued.state == CudaExecutionIssueState::kIssued && fixture::state.forwards == 1 &&
               issued.library_status == fixture::state.forward_status && host &&
               host->status == fixture::state.forward_status &&
               host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk,
           "exact original forward status is distinct from host admission or physical stop");
    const auto& call = fixture::state.last;
    expect(call.alpha == alpha && call.beta == beta && call.algorithm == 7 &&
               call.x == reinterpret_cast<const void*>(allocations[0].address + offsets[0]) &&
               call.w == reinterpret_cast<const void*>(allocations[1].address + offsets[1]) &&
               call.y == reinterpret_cast<void*>(allocations[2].address + offsets[2]) &&
               call.workspace ==
                   (workspace_mode == 2
                        ? nullptr
                        : reinterpret_cast<void*>(allocations[3].address + workspace_offset)) &&
               call.workspace_bytes == (workspace_mode == 0 ? 47U : 0U) &&
               (workspace_mode == 2 ? call.stream == nullptr : call.stream != nullptr),
           "all native addresses, offsets, full workspace, coefficients and stream survive issue");
    expect(session.acknowledge_result(accepted.acceptance.receipt),
           "acknowledge actual SDK result");
    expect(
        !session.poll().retired &&
            metadata.set_convolution(descriptors[2], convolution).disposition ==
                CudaDnnDisposition::kDeferred &&
            session.release(allocations[0].id).disposition == CudaSessionDisposition::kDeferred &&
            session.release(allocations[1].id).disposition == CudaSessionDisposition::kDeferred &&
            session.release(allocations[2].id).disposition == CudaSessionDisposition::kDeferred &&
            session.set_dnn_stream(handle.id, 0).disposition == CudaDnnDisposition::kDeferred,
        "host ACK does not retire descriptors, buffers or bound stream");
    if (workspace_mode != 2)
        expect(session.release(allocations[3].id).disposition == CudaSessionDisposition::kDeferred,
               "even a nonnull zero-byte workspace retains its exact allocation identity");
    fixture::control.can_stop.store(true);
    expect(session.poll().retired == 1, "independent physical event releases the complete command");
    for (std::size_t allocation = 0; allocation < allocations.size(); ++allocation) {
        std::vector<std::byte> output(offsets[allocation] + extents[allocation] + 16);
        success(session.download({allocations[allocation].id, 0}, output));
        for (std::size_t i = 0; i < output.size(); ++i)
            if (i < offsets[allocation] || i >= offsets[allocation] + extents[allocation])
                expect(output[i] == std::byte{0xcd}, "both allocation guards remain unchanged");
        if (allocation == 0)
            expect(std::memcmp(output.data() + offsets[0], values.x.data(), sizeof(values.x)) == 0,
                   "convolution input is read-only");
        if (allocation == 1)
            expect(std::memcmp(output.data() + offsets[1], values.w.data(), sizeof(values.w)) == 0,
                   "convolution filter is read-only");
        if (allocation == 2)
            for (std::size_t i = 0; i < values.y.size(); ++i) {
                Scalar actual{};
                std::memcpy(&actual, output.data() + offsets[2] + i * sizeof(Scalar),
                            sizeof(Scalar));
                const auto expected =
                    fails ? values.y[i]
                          : alpha * static_cast<Scalar>(fixture::kProduct[i]) + beta * values.y[i];
                expect(actual == expected,
                       "every CPU convolution output matches independent oracle");
            }
        success(session.release(allocations[allocation].id));
    }
    success(metadata.shutdown());
    success(session.shutdown());
    if (fixture::state.creates != 2 || fixture::state.destroys != 2 || base::state.creates != 2 ||
        base::state.destroys != 2 || base::state.handle_destroys != 1 ||
        fixture::driver.frees != 4 || !fixture::driver.events.empty() ||
        !fixture::driver.streams.empty() || !fixture::driver.contexts.empty() ||
        fixture::control.context_destroys.load() != 1) {
        std::cerr << "Ownership counters: scalar=" << sizeof(Scalar)
                  << " workspace_mode=" << workspace_mode << " fails=" << fails
                  << " descriptors=" << fixture::state.creates << '/' << fixture::state.destroys
                  << " tensors=" << base::state.creates << '/' << base::state.destroys
                  << " handles=" << base::state.handle_destroys
                  << " frees=" << fixture::driver.frees
                  << " events=" << fixture::driver.events.size()
                  << " streams=" << fixture::driver.streams.size()
                  << " contexts=" << fixture::driver.contexts.size()
                  << " context_destroys=" << fixture::control.context_destroys.load() << '\n';
    }
    expect(fixture::state.creates == 2 && fixture::state.destroys == 2 &&
               base::state.creates == 2 && base::state.destroys == 2 &&
               base::state.handle_destroys == 1 && fixture::driver.frees == 4 &&
               fixture::driver.events.empty() && fixture::driver.streams.empty() &&
               fixture::driver.contexts.empty() && fixture::control.context_destroys.load() == 1,
           "all original ownership retires once after physical proof");
}
}  // namespace

int main() {
    try {
        fixture_context_isolation();
        exercise<float>(0, false);
        exercise<double>(0, false);
        exercise<double>(1, false);
        exercise<double>(3, false);
        exercise<float>(2, false);
        exercise<double>(0, true);
        for (int corruption = 0; corruption != 3; ++corruption)
            exercise<float>(0, false, corruption);
        std::cout << "Owned convolution CPU oracle/session checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
