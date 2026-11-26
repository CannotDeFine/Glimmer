#include "dnn_convolution_fixture.h"

#include <iostream>
#include <limits>
#include <source_location>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           ("Find Driver failure at line " + std::to_string(where.line())).c_str());
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           ("Find SDK failure at line " + std::to_string(where.line())).c_str());
}
struct Owner {
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::array<std::uint64_t, 4> descriptors{};
    CudaSessionDnnConvolution problem;
    std::uint64_t handle{}, stream{};
    CUcontext context{};
    explicit Owner(unsigned int unavailable = 0, bool use_default = false) {
        fixture::reset();
        auto sdk = fixture::session_dispatch();
        auto dnn = fixture::dispatch();
        if (unavailable == 1) dnn.convolution_forward_find = nullptr;
        if (unavailable == 2) sdk.context_synchronize = nullptr;
        if (unavailable == 3) dnn.convolution_forward_max_count = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(dnn);
        metadata = std::make_unique<CudaDnnRegistry>(dnn);
        constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kFilter,
                                   CudaDnnObjectKind::kConvolution, CudaDnnObjectKind::kTensor};
        for (std::size_t i = 0; i != kinds.size(); ++i) {
            const auto created = metadata->create(kinds[i]);
            success(created);
            descriptors[i] = created.id;
        }
        success(metadata->set_tensor(descriptors[0], CudaDnnTensorSetter::kNd,
                                     {CUDNN_DATA_FLOAT, 0, 4, {1, 2, 4, 5}, {40, 20, 5, 1}}));
        success(metadata->set_filter(descriptors[1], {CUDNN_DATA_FLOAT, 0, 4, {2, 2, 2, 3}}, true));
        success(metadata->set_convolution(
            descriptors[2], {CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT, 2, {1, 0}, {2, 1}, {1, 1}},
            true));
        success(metadata->set_tensor(descriptors[3], CudaDnnTensorSetter::kNd,
                                     {CUDNN_DATA_FLOAT, 0, 4, {1, 2, 3, 3}, {18, 9, 3, 1}}));
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk);
        success(session->initialize(0));
        context = fixture::driver.current;
        const auto created = session->create_dnn();
        success(created);
        handle = problem.handle = created.id;
        if (!use_default) {
            const auto created_stream = session->create_stream(CU_STREAM_NON_BLOCKING);
            success(created_stream);
            stream = created_stream.id;
            success(session->set_dnn_stream(handle, stream));
        }
        for (std::size_t i = 0; i != kinds.size(); ++i)
            problem.descriptors[i] = *metadata->borrow(descriptors[i], kinds[i]);
    }
    void retained() {
        for (const auto id : descriptors)
            expect(metadata->destroy(id).disposition == CudaDnnDisposition::kDeferred,
                   "Find retains all four immutable descriptors");
        expect(session->destroy_dnn(handle).disposition == CudaDnnDisposition::kDeferred,
               "Find retains native handle independently of host publication");
        expect(metadata->set_filter(descriptors[1], {CUDNN_DATA_FLOAT, 0, 4, {2, 2, 2, 3}})
                       .disposition == CudaDnnDisposition::kDeferred,
               "Find retains immutable filter metadata");
    }
    void finish() {
        problem = {};
        success(metadata->shutdown());
        success(session->shutdown());
        expect(fixture::state.destroys == 2 && base::state.destroys == 2 &&
                   base::state.handle_destroys == 1 && fixture::driver.contexts.empty() &&
                   fixture::driver.events.empty() && fixture::driver.streams.empty() &&
                   fixture::control.context_destroys.load() == 1,
               "known stopped Find owners are cleaned exactly once");
    }
};
void empty_performance(const CudaDnnConvolutionQueryResult& result) {
    expect(!result.count && !result.result.value &&
               std::ranges::all_of(result.performance,
                                   [](const auto& item) {
                                       return !item.algorithm && !item.status && !item.time_bits &&
                                              !item.memory && !item.determinism && !item.math &&
                                              item.reserved == std::array<std::int32_t, 3>{};
                                   }),
           "failed Find publishes no count or partially validated record");
}
void performance(const CudaDnnConvolutionQueryResult& result, unsigned int count) {
    expect(result.count == count, "Find reports actual bounded variant count");
    for (std::size_t i = 0; i != count; ++i) {
        const auto& item = result.performance[i];
        expect(item.algorithm == i % 8 &&
                   item.status == static_cast<std::uint32_t>(i == 2 ? CUDNN_STATUS_NOT_SUPPORTED
                                                                    : CUDNN_STATUS_SUCCESS) &&
                   item.time_bits == 0x3f000000U + i * 0x1000U &&
                   item.memory == (std::uint64_t{1} << 33) + i && item.determinism == i % 2 &&
                   item.math == i % 4 &&
                   item.reserved == std::array<std::int32_t, 3>{-7, static_cast<int>(i), 31337},
               "actual Find preserves all native fields, including multiple math variants");
    }
}
void ordinary(int requested, bool native_error = false, bool block = false,
              bool use_default = false) {
    Owner owner(0, use_default);
    fixture::state.find_status =
        native_error ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS;
    const auto expected_status =
        requested <= 0 ? CUDNN_STATUS_BAD_PARAM : fixture::state.find_status;
    const auto accepted = owner.session->accept_dnn_find(1, owner.problem, requested);
    expect(accepted.native.acceptance.accepted() && accepted.result && !accepted.result->is_ready &&
               !fixture::state.finds && !fixture::control.context_synchronizes.load(),
           "Find admission creates only a pending result, never benchmarks on admission");
    const auto native_descriptors =
        std::array{owner.problem.descriptors[0].native, owner.problem.descriptors[1].native,
                   owner.problem.descriptors[2].native, owner.problem.descriptors[3].native};
    owner.problem = {};
    owner.retained();
    fixture::control.can_stop.store(false);
    fixture::control.should_block_find.store(block);
    fixture::control.can_finish_find.store(!block);
    std::atomic<bool> issue_finished{}, saw_block{};
    std::jthread observer;
    if (block)
        observer = std::jthread([&] {
            while (!fixture::control.has_find_entered.load())
                fixture::control.has_find_entered.wait(false);
            saw_block.store(!issue_finished.load() &&
                            !fixture::control.context_synchronizes.load());
            fixture::control.can_finish_find.store(true);
            fixture::control.can_finish_find.notify_all();
        });
    const auto issued = owner.session->issue_next();
    issue_finished.store(true);
    if (observer.joinable()) observer.join();
    const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
    expect((!block || saw_block.load()) && issued.state == CudaExecutionIssueState::kIssued &&
               issued.library_status == expected_status && host &&
               host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
               host->status == expected_status && accepted.result->is_ready &&
               fixture::state.finds == 1 && fixture::state.heuristics == 0 &&
               fixture::state.find_requested == std::min(requested, 12) &&
               fixture::state.last.descriptors == native_descriptors &&
               (use_default ? fixture::state.last.stream == nullptr
                            : fixture::state.last.stream != nullptr) &&
               fixture::control.context_synchronizes.load() == 1,
           "actual host-blocking Find executes once on issue and proves its owned context");
    const auto& output = accepted.result->query;
    expect(output.result.disposition == CudaDnnDisposition::kComplete &&
               output.result.status == CUDA_SUCCESS &&
               output.result.original_status == expected_status,
           "native Find status remains distinct from successful Driver completion");
    if (expected_status == CUDNN_STATUS_SUCCESS)
        performance(output, static_cast<unsigned int>(std::min(requested, 10)));
    else
        empty_performance(output);
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt) &&
               !owner.session->poll().retired,
           "Find result ACK does not authorize trailing-event retirement");
    owner.retained();
    fixture::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1 && fixture::state.finds == 1 &&
               fixture::control.context_synchronizes.load() == 1,
           "physical event releases Find once without rerunning benchmark or context proof");
    owner.finish();
}
void rejected_admission() {
    for (unsigned int unavailable = 1; unavailable != 4; ++unavailable) {
        Owner owner(unavailable);
        const auto result = owner.session->accept_dnn_find(1, owner.problem, 20);
        expect(!result.native.acceptance.accepted() && !result.result && !fixture::state.finds,
               "missing native Find/proof/max-count support rejects before benchmark");
        owner.finish();
    }
    Owner owner;
    auto invalid = owner.problem;
    invalid.descriptors[1].lifetime.reset();
    expect(!owner.session->accept_dnn_find(1, invalid, 20).native.acceptance.accepted() &&
               !fixture::state.finds,
           "raw filter identity without an ownership lease cannot authorize Find");
    invalid = {};
    fixture::state.query_status = CUDNN_STATUS_NOT_SUPPORTED;
    const auto failed_maximum = owner.session->accept_dnn_find(1, owner.problem, 20);
    expect(!failed_maximum.native.acceptance.accepted() && !failed_maximum.result &&
               failed_maximum.native.original_status == CUDNN_STATUS_NOT_SUPPORTED &&
               !fixture::state.finds && !owner.session->snapshot().is_quarantined,
           "failed original maximum rejects admission with exact native status and no benchmark");
    fixture::state.query_status = CUDNN_STATUS_SUCCESS;
    for (const auto maximum : {65, -1}) {
        fixture::state.maximum = maximum;
        const auto result = owner.session->accept_dnn_find(1, owner.problem, 20);
        expect(!result.native.acceptance.accepted() && !result.result && !fixture::state.finds,
               "unbounded or malformed original maximum cannot start Find");
        if (owner.session->snapshot().is_quarantined) break;
    }
    owner.finish();
}
void malformed(bool wrong_count) {
    Owner owner;
    fixture::state.should_overshoot_count = wrong_count;
    fixture::state.should_corrupt_performance = !wrong_count;
    const auto accepted = owner.session->accept_dnn_find(1, owner.problem, 20);
    expect(accepted.native.acceptance.accepted(), "admit original Find before corrupt output");
    owner.problem = {};
    const auto issued = owner.session->issue_next();
    expect(issued.library_status == CUDNN_STATUS_SUCCESS && accepted.result->is_ready &&
               accepted.result->query.result.disposition == CudaDnnDisposition::kRejected &&
               accepted.result->query.result.status == CUDA_ERROR_UNKNOWN &&
               accepted.result->query.result.original_status == CUDNN_STATUS_SUCCESS &&
               owner.session->snapshot().is_quarantined,
           "malformed native Find success preserves provenance and quarantines execution");
    empty_performance(accepted.result->query);
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt) &&
               owner.session->poll().retired == 1 &&
               fixture::control.context_synchronizes.load() == 1,
           "successful physical proof permits known-owner cleanup after malformed results");
    owner.finish();
}
void cancelled_before_issue() {
    Owner owner;
    const auto accepted = owner.session->accept_dnn_find(1, owner.problem, 20);
    expect(accepted.native.acceptance.accepted() && accepted.result,
           "queued Find owns a pending result before cancellation");
    owner.problem = {};
    owner.retained();
    expect(owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
           "reject exactly one accepted but unissued Find command");
    const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
    expect(host && host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
               host->status == CUDA_ERROR_INVALID_VALUE && !accepted.result->is_ready &&
               !accepted.result->query.result.original_status && !fixture::state.finds &&
               !fixture::control.context_synchronizes.load() &&
               owner.session->issue_next().state == CudaExecutionIssueState::kIdle &&
               owner.session->acknowledge_result(accepted.native.acceptance.receipt),
           "cancelled Find has only a rejection receipt, never a fabricated native result");
    empty_performance(accepted.result->query);
    owner.finish();
}
void uncertain(unsigned int mode) {
    Owner owner;
    const auto accepted = owner.session->accept_dnn_find(1, owner.problem, 20);
    expect(accepted.native.acceptance.accepted(), "admit Find before proof uncertainty");
    owner.problem = {};
    fixture::control.synchronize_status.store(mode == 0 ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS);
    fixture::control.should_change_find_context.store(mode == 1);
    fixture::control.should_change_sync_context.store(mode == 2);
    const auto issued = owner.session->issue_next();
    const auto expected = mode == 0 ? CUDA_ERROR_LAUNCH_FAILED : CUDA_ERROR_INVALID_CONTEXT;
    const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
    expect(issued.library_status == CUDNN_STATUS_SUCCESS && accepted.result->is_ready && host &&
               host->status == CUDNN_STATUS_SUCCESS &&
               accepted.result->query.result.status == expected &&
               accepted.result->query.result.disposition == CudaDnnDisposition::kRejected &&
               accepted.result->query.result.original_status == CUDNN_STATUS_SUCCESS &&
               fixture::control.context_synchronizes.load() == (mode == 1 ? 0U : 1U),
           "failed proof preserves native success and never synchronizes a changed context");
    empty_performance(accepted.result->query);
    // Restoring current context and making every later event/sync successful
    // cannot retroactively authorize cleanup after the one uncertain proof.
    fixture::driver.current = owner.context;
    fixture::control.can_stop.store(true);
    fixture::control.synchronize_status.store(CUDA_SUCCESS);
    fixture::control.should_change_find_context.store(false);
    fixture::control.should_change_sync_context.store(false);
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt),
           "uncertain Find still has one actual native result to acknowledge");
    expect(!owner.session->poll().retired && !owner.session->drain_issued().retired &&
               owner.session->shutdown().status != CUDA_SUCCESS &&
               owner.metadata->shutdown().disposition != CudaDnnDisposition::kComplete &&
               owner.session->snapshot().is_quarantined && fixture::state.finds == 1 &&
               fixture::control.context_synchronizes.load() == (mode == 1 ? 0U : 1U) &&
               !fixture::state.destroys && !base::state.destroys && !base::state.handle_destroys &&
               !fixture::control.context_destroys.load(),
           "uncertain Find is retained despite ACK/event/drain and never retries native cleanup");
    owner.session.reset();
    owner.metadata.reset();
    expect(!fixture::state.destroys && !base::state.destroys && !base::state.handle_destroys &&
               !fixture::control.context_destroys.load(),
           "owner destructors cannot erase uncertain Find lifetimes");
}
void isolated(unsigned int mode) {
    const auto child = ::fork();
    expect(child >= 0, "fork isolated uncertain Find ownership");
    if (!child) {
        try {
            uncertain(mode);
            ::_exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Find uncertainty mode " << mode << ": " << error.what() << '\n';
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated uncertain Find regression succeeds without releasing retained ownership");
}
}  // namespace
int main() {
    try {
        ordinary(20, false, true);
        ordinary(5, false, false, true);
        ordinary(std::numeric_limits<int>::max());
        ordinary(0);
        ordinary(-1);
        ordinary(20, true);
        rejected_admission();
        cancelled_before_issue();
        malformed(false);
        malformed(true);
        for (unsigned int mode = 0; mode != 3; ++mode) isolated(mode);
        std::cout << "Actual Find CPU session and proof-retention checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.can_finish_find.store(true);
        fixture::control.can_finish_find.notify_all();
        fixture::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
