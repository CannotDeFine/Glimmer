#include "dnn_folded_convolution_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_folded_convolution;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void native(const CudaDnnResult& value, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(value.disposition == CudaDnnDisposition::kComplete && value.status == CUDA_SUCCESS &&
               value.original_status == status && !value.value,
           "exact once-only original Folded status has no synthetic output");
}
void rejected(const CudaDnnResult& value, CUresult status) {
    expect(value.disposition == CudaDnnDisposition::kRejected && value.status == status &&
               !value.original_status && !value.value,
           "preissue policy rejects without invented original Folded status");
}
void driver(const CudaSessionResult& value) {
    expect(value.disposition == CudaSessionDisposition::kComplete && value.status == CUDA_SUCCESS,
           "owned fake Driver lifecycle succeeds");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    CudaDnnFoldedDescriptors request;
    std::uint64_t handle{};
    CUcontext context{};
    explicit Owner(unsigned int missing = 0) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        for (std::size_t i = 0; i != 4; ++i) {
            const auto created = metadata->create(fixture::kInputs[i]);
            native(created);
            request.inputs[i] = created.id;
        }
        for (std::size_t i = 0; i != 8; ++i) {
            const auto created = metadata->create(fixture::kOutputs[i]);
            native(created);
            request.outputs[i] = created.id;
        }
        configure();
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk);
        driver(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
    }
    void configure() {
        native(metadata->set_filter(request.inputs[0], fixture::kFilter));
        native(metadata->set_tensor(request.inputs[1], CudaDnnTensorSetter::kNd, fixture::kDiff));
        native(metadata->set_convolution(request.inputs[2], fixture::kConvolution));
        native(metadata->set_tensor(request.inputs[3], CudaDnnTensorSetter::kNd, fixture::kGrad));
    }
    CudaDnnResult call() {
        return session->fold_dnn_backward_convolution(handle, *metadata, request);
    }
    void close() {
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(base::state.creates == base::state.destroys &&
                   foundation::state.creates == foundation::state.destroys &&
                   fixture::transform::state.creates == fixture::transform::state.destroys &&
                   base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
                   foundation::driver.events.empty() && foundation::driver.memory.empty() &&
                   foundation::driver.contexts.empty(),
               "all proven descriptor and context owners retire exactly once");
    }
};
void ordinary() {
    Owner owner;
    const auto initial = owner.request;
    native(owner.call());
    expect(
        fixture::last.requested == 2 && fixture::control.gets == 1 && fixture::control.folds == 1,
        "one authoritative Nd preflight and one original Folded call");
    for (std::size_t i = 0; i != 4; ++i)
        expect(owner.metadata->borrow(initial.inputs[i], fixture::kInputs[i])->native ==
                   fixture::last.inputs[i],
               "all original positional input identities survive dispatch");
    for (std::size_t i = 0; i != 8; ++i)
        expect(owner.metadata->borrow(initial.outputs[i], fixture::kOutputs[i])->native ==
                   fixture::last.outputs[i],
               "all positional output identities survive dispatch");
    expect(
        foundation::descriptor(fixture::last.outputs[0], fixture::Kind::kFilter).filter.sizes[0] ==
                101 &&
            base::object(fixture::last.outputs[1], fixture::Kind::kTensor).tensor.sizes[0] == 102 &&
            base::object(fixture::last.outputs[3], fixture::Kind::kTensor).tensor.sizes[0] == 103 &&
            fixture::transform::object(fixture::last.outputs[7], fixture::Kind::kTensorTransform)
                    .transform.dimensions == 108,
        "independent synthetic oracle observes all native mutation classes");
    owner.request.outputs[7] = 0;
    native(owner.call());
    expect(!fixture::last.outputs[7], "optional final native transform remains truly null");
    owner.request = initial;
    for (std::size_t i = 0; i != 4; ++i) owner.request.outputs[i] = initial.inputs[i];
    for (std::size_t i = 5; i != 8; ++i) owner.request.outputs[i] = initial.outputs[4];
    native(owner.call());
    expect(fixture::last.inputs ==
                   std::array<void*, 4>{fixture::last.outputs[0], fixture::last.outputs[1],
                                        fixture::last.outputs[2], fixture::last.outputs[3]} &&
               fixture::last.outputs[4] == fixture::last.outputs[7] &&
               fixture::last.final_source_stride == 7 &&
               fixture::transform::object(fixture::last.outputs[4], fixture::Kind::kTensorTransform)
                       .transform.dimensions == 114,
           "unique leases preserve source/output aliases and late read-after-write semantics");
    owner.request = initial;
    owner.configure();
    const auto calls = fixture::control.folds.load();
    for (std::size_t i = 0; i != 8; ++i) {
        auto held = owner.metadata->borrow(initial.outputs[i], fixture::kOutputs[i]);
        const auto gets = fixture::control.gets.load();
        expect(owner.call().disposition == CudaDnnDisposition::kDeferred &&
                   fixture::control.gets == gets && fixture::control.folds == calls,
               "every destination external lease defers before even native preflight");
    }
    for (std::size_t i = 0; i != 4; ++i) {
        auto held = owner.metadata->borrow(initial.inputs[i], fixture::kInputs[i]);
        native(owner.call());
    }
    for (std::size_t i = 0; i != 7; ++i) {
        owner.request.outputs[i] = 0;
        rejected(owner.call(), CUDA_ERROR_INVALID_HANDLE);
        owner.request = initial;
    }
    for (std::size_t i = 0; i != 4; ++i) {
        owner.request.inputs[i] = 0;
        rejected(owner.call(), CUDA_ERROR_INVALID_HANDLE);
        owner.request = initial;
    }
    owner.request.outputs[0] = initial.inputs[1];
    rejected(owner.call(), CUDA_ERROR_INVALID_HANDLE);
    owner.request = initial;
    owner.request.format = 4;
    rejected(owner.call(), CUDA_ERROR_INVALID_VALUE);
    owner.request.format = 3;
    native(owner.call());
    expect(fixture::last.format == 3, "all representable format bits reach the native callback");
    owner.request = initial;
    for (const int rank : {0, 1, 3, 6}) {
        fixture::control.rank_override = rank;
        rejected(owner.call(), CUDA_ERROR_INVALID_VALUE);
    }
    fixture::control.rank_override = 2;
    for (const int stride : {0, -1, INT32_MIN}) {
        fixture::control.stride_override = stride;
        rejected(owner.call(), CUDA_ERROR_INVALID_VALUE);
    }
    fixture::control.stride_override = 1;
    fixture::control.getter_status = CUDNN_STATUS_BAD_PARAM;
    rejected(owner.call(), CUDA_ERROR_UNKNOWN);
    fixture::control.getter_status = CUDNN_STATUS_SUCCESS;
    native(owner.call());
    expect(!owner.metadata->snapshot().is_quarantined &&
               !owner.session->snapshot().is_quarantined && !fixture::control.synchronizes &&
               !fixture::control.event_queries,
           "clean preflight rejection retains healthy owners without GPU progress");
    owner.close();
}
inline Owner* active_owner{};
inline unsigned int reentries{};
void reenter() {
    auto& owner = *active_owner;
    ++reentries;
    rejected(owner.metadata->get_tensor(owner.request.inputs[1], 4), CUDA_ERROR_NOT_READY);
    rejected(owner.metadata->create(fixture::Kind::kTensor), CUDA_ERROR_NOT_READY);
    rejected(owner.metadata->destroy(owner.request.outputs[0]), CUDA_ERROR_NOT_READY);
    rejected(owner.metadata->set_convolution(owner.request.inputs[2], fixture::kConvolution),
             CUDA_ERROR_NOT_READY);
    expect(
        !owner.metadata->borrow(owner.request.outputs[0], fixture::Kind::kFilter) &&
            owner.metadata->shutdown().disposition == CudaDnnDisposition::kDeferred &&
            owner.call().disposition == CudaDnnDisposition::kDeferred &&
            owner.session->destroy_dnn(owner.handle).disposition == CudaDnnDisposition::kDeferred &&
            owner.session->set_dnn_stream(owner.handle, 0).disposition ==
                CudaDnnDisposition::kDeferred &&
            owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred &&
            !base::state.handle_destroys && !foundation::control.context_destroys,
        "callback reentry cannot acquire new views, mutate or destroy active native owners");
}
void reentrancy() {
    Owner owner;
    active_owner = &owner;
    fixture::reenter_get = reenter;
    fixture::reenter_fold = reenter;
    native(owner.call());
    expect(reentries == 2, "both preflight and actual native callback execute the reentry probe");
    fixture::reenter_get = nullptr;
    fixture::reenter_fold = nullptr;
    native(owner.call());
    owner.close();
    active_owner = nullptr;
}
void close_during_callback() {
    Owner owner;
    active_owner = &owner;
    fixture::reenter_get = [] {
        active_owner->metadata->close();
        active_owner->session->close();
        expect(!base::state.handle_destroys && !base::state.destroys &&
                   !foundation::state.destroys && !foundation::control.context_destroys,
               "close marks owners without destroying the admitted synchronous call");
    };
    native(owner.call());
    expect(fixture::control.gets == 1 && fixture::control.folds == 1 &&
               owner.metadata->snapshot().is_closing,
           "accepted operation finishes exactly once despite callback close request");
    rejected(owner.call(), CUDA_ERROR_DEINITIALIZED);
    fixture::reenter_get = nullptr;
    owner.close();
    active_owner = nullptr;
}
void pending() {
    Owner owner;
    CudaSessionDnnConvolution work;
    work.handle = owner.handle;
    constexpr std::array<std::size_t, 4> order{3, 0, 2, 1};
    for (std::size_t i = 0; i != 4; ++i)
        work.descriptors[i] =
            *owner.metadata->borrow(owner.request.inputs[order[i]], fixture::kInputs[order[i]]);
    std::array<CudaSessionResult, 4> buffers;
    constexpr std::array<std::size_t, 4> sizes{192, 144, 120, 64};
    for (std::size_t i = 0; i != 4; ++i) {
        buffers[i] = owner.session->allocate(sizes[i]);
        driver(buffers[i]);
        work.buffers[i] = {buffers[i].id, 0};
    }
    const float alpha = 1, beta = 0;
    std::memcpy(work.alpha.data(), &alpha, 4);
    std::memcpy(work.beta.data(), &beta, 4);
    work.scalar_bytes = 4;
    work.workspace_bytes = 64;
    const auto accepted = owner.session->accept_dnn_convolution(1, work);
    expect(accepted.acceptance.accepted(), "prior original compute accepted");
    work = {};
    foundation::control.can_stop = false;
    foundation::control.fail_forward = true;
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued,
           "prior original owns a NOT_READY physical event");
    native(owner.call());  // Disjoint outputs must not drain unrelated pending input borrows.
    const auto saved = owner.request;
    owner.request.outputs[0] = owner.request.inputs[0];
    const auto gets = fixture::control.gets.load(), folds = fixture::control.folds.load();
    for (int phase = 0; phase != 2; ++phase) {
        expect(owner.call().disposition == CudaDnnDisposition::kDeferred &&
                   fixture::control.gets == gets && fixture::control.folds == folds &&
                   !fixture::control.synchronizes && !fixture::control.event_queries,
               "physical source-as-destination borrow blocks whole Folded without progress calls");
        if (!phase)
            expect(owner.session->acknowledge_result(accepted.acceptance.receipt),
                   "host ACK is independent of physical ownership");
    }
    foundation::control.can_stop = true;
    expect(owner.session->poll().retired == 1, "actual event completion releases immutable borrow");
    native(owner.call());
    owner.request = saved;
    for (const auto& buffer : buffers) driver(owner.session->release(buffer.id));
    owner.close();
    expect(foundation::driver.frees == 4, "all prior buffers retire once");
}
void poisoned(unsigned int mode) {
    Owner owner;
    if (mode == 0) fixture::control.fold_status = CUDNN_STATUS_BAD_PARAM;
    if (mode == 1 || mode == 2) fixture::control.change_after_get = true;
    if (mode == 2) fixture::control.getter_status = CUDNN_STATUS_BAD_PARAM;
    if (mode == 3 || mode == 4) fixture::control.change_after_fold = true;
    if (mode == 4) fixture::control.fold_status = CUDNN_STATUS_EXECUTION_FAILED;
    if (mode == 5) fixture::control.rank_override = -1;
    if (mode == 6) fixture::control.rank_override = 7;
    if (mode == 7) fixture::control.mode_override = UINT32_MAX;
    const auto result = owner.call();
    const bool issued = mode == 0 || mode == 3 || mode == 4;
    const bool lost = mode >= 1 && mode <= 4;
    if (mode == 0)
        native(result, CUDNN_STATUS_BAD_PARAM);
    else {
        expect(
            result.disposition == CudaDnnDisposition::kRejected &&
                result.status == (lost ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_UNKNOWN) &&
                result.original_status ==
                    (issued ? std::optional<cudnnStatus_t>{mode == 4 ? CUDNN_STATUS_EXECUTION_FAILED
                                                                     : CUDNN_STATUS_SUCCESS}
                            : std::nullopt),
            "context proof and malformed preflight retain only actual original provenance");
    }
    expect(owner.metadata->snapshot().is_quarantined == (mode == 0 || mode >= 3) &&
               owner.session->snapshot().is_quarantined == lost &&
               fixture::control.folds == (issued ? 1U : 0U),
           "metadata-only, context-only and combined fault scopes remain distinct");
    const auto gets = fixture::control.gets.load();
    expect(owner.call().disposition == CudaDnnDisposition::kRejected &&
               fixture::control.gets == gets && !fixture::control.synchronizes,
           "uncertain owner does not replay any native callback or synchronize");
    if (lost)
        expect(owner.session->shutdown().status == CUDA_ERROR_INVALID_CONTEXT &&
                   !base::state.handle_destroys && !foundation::control.context_destroys,
               "wrong current owner cannot authorize native cleanup");
    else {
        driver(owner.session->shutdown());
        expect(base::state.handle_destroys == 1,
               "metadata uncertainty alone retains proven handle cleanup ownership");
    }
    ::_exit(0);  // Uncertain native metadata intentionally retained in this isolated process.
}
}  // namespace
int main() {
    try {
        ordinary();
        reentrancy();
        close_during_callback();
        pending();
        for (unsigned int absent = 1; absent <= 2; ++absent) {
            Owner owner(absent);
            rejected(owner.call(), CUDA_ERROR_NOT_SUPPORTED);
            expect(!fixture::control.gets && !fixture::control.folds,
                   "missing capability rejects before any native preflight");
            owner.close();
        }
        for (unsigned int i = 0; i != 8; ++i) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated uncertain Folded ownership");
            if (!child) poisoned(i);
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated partial mutation/context/malformed case passed");
        }
        std::cout << "Folded original descriptor CPU session checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
