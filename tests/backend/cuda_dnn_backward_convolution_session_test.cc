#include "dnn_backward_convolution_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_backward_convolution;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Kind = CudaDnnBackwardConvolutionQueryKind;
constexpr std::array kKinds{Kind::kDataMaxCount, Kind::kFilterMaxCount, Kind::kDataWorkspace};
constexpr std::array kDescriptors{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kFilter,
                                  CudaDnnObjectKind::kConvolution, CudaDnnObjectKind::kTensor};
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned backward-query Driver operation succeeds");
}
void native(const CudaDnnResult& result, cudnnStatus_t status, std::uint64_t value = 0) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && result.value == value,
           "original query status and successful-only output are exact");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.value,
           "pre-issue rejection carries neither native provenance nor output");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::array<std::uint64_t, 4> descriptors{};
    std::uint64_t handle{};
    CUcontext context{};
    explicit Owner(unsigned int missing = 0) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        for (std::size_t i = 0; i != descriptors.size(); ++i) {
            const auto created = metadata->create(kDescriptors[i]);
            native(created, CUDNN_STATUS_SUCCESS);
            descriptors[i] = created.id;
        }
        native(metadata->set_tensor(descriptors[0], CudaDnnTensorSetter::kNd, fixture::kDy),
               CUDNN_STATUS_SUCCESS);
        native(metadata->set_filter(descriptors[1], fixture::kFilter), CUDNN_STATUS_SUCCESS);
        native(metadata->set_convolution(descriptors[2], fixture::kConvolution),
               CUDNN_STATUS_SUCCESS);
        native(metadata->set_tensor(descriptors[3], CudaDnnTensorSetter::kNd, fixture::kDx),
               CUDNN_STATUS_SUCCESS);
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{71, 72}, 73, sdk);
        success(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created, CUDNN_STATUS_SUCCESS);
        handle = created.id;
    }
    CudaSessionDnnConvolution problem(bool has_descriptors = true) {
        CudaSessionDnnConvolution result;
        result.handle = handle;
        result.algorithm = 5;
        if (has_descriptors)
            for (std::size_t i = 0; i != descriptors.size(); ++i)
                result.descriptors[i] = *metadata->borrow(descriptors[i], kDescriptors[i]);
        return result;
    }
    CudaDnnResult query(Kind kind, bool has_output = true) {
        return session->query_dnn_backward_convolution(problem(kind == Kind::kDataWorkspace), kind,
                                                       has_output);
    }
    void close() {
        if (!session->snapshot().is_quarantined)
            native(session->destroy_dnn(handle), CUDNN_STATUS_SUCCESS);
        native(metadata->shutdown(), CUDNN_STATUS_SUCCESS);
        success(session->shutdown());
        expect(metadata->snapshot().is_closed && base::state.creates == 2 &&
                   base::state.destroys == 2 && foundation::state.creates == 2 &&
                   foundation::state.destroys == 2 && base::state.handle_creates == 1 &&
                   base::state.handle_destroys == 1 && foundation::driver.events.empty() &&
                   foundation::driver.memory.empty() && foundation::driver.streams.empty() &&
                   foundation::driver.contexts.empty(),
               "every proven descriptor/handle/context retires exactly once");
    }
};
void ordinary() {
    Owner owner;
    for (std::size_t direction = 0; direction != 2; ++direction) {
        for (const int value : {0, 1, 65, INT32_MAX}) {
            fixture::control.counts[direction].store(value);
            native(owner.query(kKinds[direction]), CUDNN_STATUS_SUCCESS,
                   static_cast<std::uint64_t>(value));
        }
        const auto calls = fixture::calls();
        rejected(owner.query(kKinds[direction], false), CUDA_ERROR_INVALID_VALUE);
        expect(fixture::calls() == calls, "null MaxCount is rejected before unsafe native access");
    }
    for (const auto bytes :
         {std::size_t{0}, (std::size_t{1} << 40) + 17, std::numeric_limits<std::size_t>::max()}) {
        fixture::control.bytes.store(bytes);
        native(owner.query(Kind::kDataWorkspace), CUDNN_STATUS_SUCCESS, bytes);
    }
    native(owner.query(Kind::kDataWorkspace, false), CUDNN_STATUS_BAD_PARAM);
    expect(fixture::control.null_workspace.load() == 1, "null workspace output reaches native");
    auto problem = owner.problem();
    expect(fixture::last.descriptors ==
               std::array<void*, 4>{problem.descriptors[1].native, problem.descriptors[0].native,
                                    problem.descriptors[2].native, problem.descriptors[3].native},
           "backend tensor/filter order explicitly converts to native filter/dy/convolution/dx");
    for (const auto algorithm : {6U, 7U}) {
        problem.algorithm = algorithm;
        native(owner.session->query_dnn_backward_convolution(problem, Kind::kDataWorkspace),
               CUDNN_STATUS_NOT_SUPPORTED);
        expect(fixture::control.algorithm.load() == algorithm, "representable enum reaches native");
    }
    const auto before = fixture::calls();
    problem.algorithm = 8;
    rejected(owner.session->query_dnn_backward_convolution(problem, Kind::kDataWorkspace),
             CUDA_ERROR_INVALID_VALUE);
    problem.algorithm = 0;
    for (std::size_t i = 0; i != 4; ++i) {
        auto invalid = problem;
        invalid.descriptors[i].lifetime.reset();
        rejected(owner.session->query_dnn_backward_convolution(invalid, Kind::kDataWorkspace),
                 CUDA_ERROR_INVALID_HANDLE);
        invalid = problem;
        invalid.descriptors[i].kind = CudaDnnObjectKind::kActivation;
        rejected(owner.session->query_dnn_backward_convolution(invalid, Kind::kDataWorkspace),
                 CUDA_ERROR_INVALID_HANDLE);
    }
    auto invalid = problem;
    invalid.handle = UINT64_MAX;
    rejected(owner.session->query_dnn_backward_convolution(invalid, Kind::kDataMaxCount),
             CUDA_ERROR_INVALID_HANDLE);
    rejected(owner.session->query_dnn_backward_convolution(problem, static_cast<Kind>(0)),
             CUDA_ERROR_INVALID_VALUE);
    expect(owner.sdk.context_set_current(nullptr) == CUDA_SUCCESS, "detach original context");
    rejected(owner.query(Kind::kFilterMaxCount), CUDA_ERROR_INVALID_CONTEXT);
    expect(owner.sdk.context_set_current(owner.context) == CUDA_SUCCESS, "restore exact context");
    auto foreign_thread =
        std::async(std::launch::async, [&] { return owner.query(Kind::kDataMaxCount); });
    rejected(foreign_thread.get(), CUDA_ERROR_UNKNOWN);
    expect(fixture::calls() == before, "invalid ownership and enum bits invoke no native query");
    for (std::size_t index = 0; index != kKinds.size(); ++index) {
        fixture::control.statuses[index].store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        native(owner.query(kKinds[index]), CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        fixture::control.statuses[index].store(CUDNN_STATUS_SUCCESS);
    }
    problem = {};
    invalid = {};
    expect(!fixture::control.synchronizes.load() && !fixture::control.event_queries.load() &&
               foundation::driver.events.empty() && !owner.session->snapshot().is_quarantined,
           "read-only errors neither synchronize nor poison healthy ownership");
    owner.close();
}
void pending() {
    Owner owner;
    auto forward = owner.problem();
    std::swap(forward.descriptors[0], forward.descriptors[3]);  // dx,w,conv,dy is forward order.
    std::array<CudaSessionResult, 4> buffers;
    constexpr std::array<std::size_t, 4> sizes{192, 144, 120, 64};
    for (std::size_t i = 0; i != 4; ++i) {
        buffers[i] = owner.session->allocate(sizes[i]);
        success(buffers[i]);
        forward.buffers[i] = {buffers[i].id, 0};
    }
    const float alpha = 1, beta = 0;
    std::memcpy(forward.alpha.data(), &alpha, sizeof(alpha));
    std::memcpy(forward.beta.data(), &beta, sizeof(beta));
    forward.scalar_bytes = 4;
    forward.workspace_bytes = 64;
    const auto accepted = owner.session->accept_dnn_convolution(1, forward);
    expect(accepted.acceptance.accepted(), "accept prior original work without issuing query");
    forward = {};
    foundation::control.can_stop.store(false);
    foundation::control.fail_forward.store(true);
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued,
           "issue one original command with a genuinely pending fake physical event");
    const auto receipt = owner.session->host_result(accepted.acceptance.receipt);
    expect(receipt && receipt->status == CUDNN_STATUS_EXECUTION_FAILED,
           "prior native failure still owns its physical leases");
    for (unsigned int phase = 0; phase != 2; ++phase) {
        const auto polls = fixture::control.event_queries.load();
        const auto events = foundation::driver.events.size();
        for (std::size_t index = 0; index != 3; ++index)
            native(owner.query(kKinds[index]), CUDNN_STATUS_SUCCESS,
                   index < 2 ? static_cast<std::uint64_t>(11 + 2 * index) : 47);
        expect(!foundation::control.can_stop.load() && !fixture::control.synchronizes.load() &&
                   fixture::control.event_queries.load() == polls &&
                   foundation::driver.events.size() == events && foundation::state.forwards == 1,
               "pending queries neither poll/drain/synchronize/issue nor retire prior work");
        if (!phase)
            expect(owner.session->acknowledge_result(accepted.acceptance.receipt),
                   "host acknowledgement is not physical completion");
    }
    for (const auto id : owner.descriptors)
        expect(owner.metadata->destroy(id).disposition == CudaDnnDisposition::kDeferred,
               "all four immutable descriptors survive query and host ACK");
    for (const auto& buffer : buffers)
        expect(owner.session->release(buffer.id).disposition == CudaSessionDisposition::kDeferred,
               "all prior original buffers remain physically owned");
    expect(owner.session->destroy_dnn(owner.handle).disposition == CudaDnnDisposition::kDeferred &&
               owner.session->set_dnn_stream(owner.handle, 0).disposition ==
                   CudaDnnDisposition::kDeferred,
           "original handle and bound stream remain protected after read-only queries");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "only actual successful event retires prior work");
    for (const auto& buffer : buffers) success(owner.session->release(buffer.id));
    owner.close();
    expect(foundation::driver.frees == 4, "four physical buffers free exactly once");
}
void missing() {
    for (unsigned int index = 0; index != 3; ++index) {
        Owner owner(index + 1);
        rejected(owner.query(kKinds[index]), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture::calls() && !owner.session->snapshot().is_quarantined,
               "missing callback cleanly rejects without original invocation");
        native(owner.query(kKinds[(index + 1) % 2]), CUDNN_STATUS_SUCCESS,
               (index + 1) % 2 == 0 ? 11 : 13);
        owner.close();
    }
}
void poisoned(unsigned int index) {
    Owner owner;
    const auto kind = kKinds[index % 3];
    const bool context_loss = index >= 2;
    if (context_loss) {
        fixture::control.changes_context.store(true);
        if (index >= 5) fixture::control.statuses[index % 3].store(CUDNN_STATUS_BAD_PARAM);
    } else {
        fixture::control.counts[index].store(-1);
    }
    const auto result = owner.query(kind);
    expect(result.disposition == CudaDnnDisposition::kRejected &&
               result.status == (context_loss ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_UNKNOWN) &&
               result.original_status ==
                   (index >= 5 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_SUCCESS) &&
               !result.value && owner.session->snapshot().is_quarantined,
           "negative count or postcallback context loss retains provenance without publishing");
    const auto calls = fixture::calls();
    expect(owner.query(kind).disposition == CudaDnnDisposition::kRejected &&
               fixture::calls() == calls && !fixture::control.synchronizes.load(),
           "quarantined query never retries native call or synchronization");
    if (context_loss) {
        expect(owner.session->shutdown().status == CUDA_ERROR_INVALID_CONTEXT &&
                   !base::state.handle_destroys && !foundation::control.context_destroys.load(),
               "lost current context never authorizes native handle/context cleanup");
        expect(owner.sdk.context_set_current(owner.context) == CUDA_SUCCESS,
               "restore proven owner only for safe isolated cleanup");
    }
    owner.close();
}
void isolated() {
    for (unsigned int index = 0; index != 8; ++index) {
        const auto child = ::fork();
        expect(child >= 0, "fork isolated backward-query corruption case");
        if (!child) {
            try {
                poisoned(index);
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated backward-query corruption case succeeds");
    }
}
}  // namespace
int main() {
    try {
        ordinary();
        pending();
        missing();
        isolated();
        std::cout << "Backward-convolution original query CPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        fixture::control.can_finish.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
