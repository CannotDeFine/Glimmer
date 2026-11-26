#include "dnn_rnn_query_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_rnn_query;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Kind = CudaDnnRnnQueryKind;
constexpr std::array kKinds{Kind::kWeightSpace, Kind::kTempSpace, Kind::kAttentionBuffers};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void native(const CudaDnnResult& result, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "exact original RNN query/metadata result");
}
void driver(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
void rejected(const CudaDnnRnnQueryResult& result, CUresult status) {
    expect(result.result.disposition == CudaDnnDisposition::kRejected &&
               result.result.status == status && !result.result.original_status &&
               result.sizes == std::array<std::uint64_t, 3>{},
           "preissue rejection publishes neither sizes nor requested-query provenance");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t rnn{}, attn{}, input{}, dropout{}, handle{};
    CUcontext context{};
    explicit Owner(unsigned int missing = 0) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        rnn = create(CudaDnnObjectKind::kRnn);
        attn = create(CudaDnnObjectKind::kAttn);
        input = create(CudaDnnObjectKind::kRnnData);
        dropout = create(CudaDnnObjectKind::kDropout);
        CudaDnnRnnState value;
        value.dropout = dropout;
        native(metadata->set_rnn(rnn, value));
        CudaDnnAttnState attention;
        attention.dropouts = {dropout, dropout};
        native(metadata->set_attn(attn, attention));
        CudaDnnRnnDataState data{.presence = 3,
                                 .padding_bytes = 4,
                                 .max_sequence_length = 3,
                                 .batch_size = 2,
                                 .vector_size = 4,
                                 .lengths = {3, 2}};
        native(metadata->set_rnn_data(input, data));
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk);
        driver(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
    }
    std::uint64_t create(CudaDnnObjectKind kind) {
        const auto result = metadata->create(kind);
        native(result);
        return result.id;
    }
    CudaDnnRnnQuery problem(Kind kind, std::uint32_t outputs) const {
        return {kind, kind == Kind::kAttentionBuffers ? attn : rnn,
                kind == Kind::kTempSpace ? input : 0, 0, outputs};
    }
    CudaDnnRnnQueryResult query(Kind kind, std::uint32_t outputs) {
        return session->query_dnn_rnn(handle, *metadata, problem(kind, outputs));
    }
    void close() {
        native(session->destroy_dnn(handle));
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(metadata->snapshot().is_closed && fixture::metadata::state.creates == 2 &&
                   fixture::metadata::state.destroys == 2 &&
                   fixture::sequence::state.destroys == 1 &&
                   fixture::dropout::state.destroys == 1 && base::state.handle_destroys == 1 &&
                   foundation::driver.contexts.empty() && foundation::driver.streams.empty() &&
                   foundation::driver.events.empty() && foundation::driver.memory.empty(),
               "all proven parents/dependencies/handle/context retire exactly once");
    }
};
void ordinary() {
    Owner owner;
    for (std::size_t i = 0; i != 3; ++i) {
        const auto allowed = i == 0 ? 1U : i == 1 ? 6U : 7U;
        for (unsigned int mask = 0; mask != 8; ++mask) {
            if (mask & ~allowed) continue;
            const auto gets = fixture::control.gets.load();
            const auto result = owner.query(kKinds[i], mask);
            const auto status = (i == 0 && !mask) || (i == 2 && (mask & 3U) != 3U)
                                    ? CUDNN_STATUS_BAD_PARAM
                                    : CUDNN_STATUS_SUCCESS;
            native(result.result, status);
            for (std::size_t slot = 0; slot != 3; ++slot)
                expect(
                    result.sizes[slot] == (status == CUDNN_STATUS_SUCCESS && (mask & (1U << slot))
                                               ? fixture::control.sizes[slot].load()
                                               : 0),
                    "only successful requested slots publish");
            expect(fixture::control.mask.load() == mask &&
                       fixture::control.gets.load() == gets + (i == 1 && (mask & 2) ? 1 : 0),
                   "independent native nulls and workspace-only original preflight are preserved");
        }
    }
    for (const auto bytes : {std::size_t{0}, (std::size_t{1} << 40) + 7, SIZE_MAX}) {
        for (auto& value : fixture::control.sizes) value.store(bytes);
        const auto result = owner.query(Kind::kAttentionBuffers, 7);
        native(result.result);
        expect(result.sizes == std::array<std::uint64_t, 3>{bytes, bytes, bytes},
               "all size_t bits survive");
    }
    for (std::size_t i = 0; i != 3; ++i) {
        fixture::control.statuses[i].store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        const auto result = owner.query(kKinds[i], i == 0 ? 1 : i == 1 ? 6 : 7);
        native(result.result, CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        expect(result.sizes == std::array<std::uint64_t, 3>{},
               "failed private writes are not published");
        fixture::control.statuses[i].store(CUDNN_STATUS_SUCCESS);
    }
    const auto before = fixture::calls();
    auto problem = owner.problem(Kind::kTempSpace, 2);
    problem.mode = 2;
    rejected(owner.session->query_dnn_rnn(owner.handle, *owner.metadata, problem),
             CUDA_ERROR_INVALID_VALUE);
    problem = owner.problem(Kind::kWeightSpace, 2);
    rejected(owner.session->query_dnn_rnn(owner.handle, *owner.metadata, problem),
             CUDA_ERROR_INVALID_VALUE);
    problem = owner.problem(Kind::kTempSpace, 2);
    problem.input = owner.attn;
    rejected(owner.session->query_dnn_rnn(owner.handle, *owner.metadata, problem),
             CUDA_ERROR_INVALID_HANDLE);
    rejected(owner.session->query_dnn_rnn(UINT64_MAX, *owner.metadata,
                                          owner.problem(Kind::kWeightSpace, 1)),
             CUDA_ERROR_INVALID_HANDLE);
    expect(fixture::calls() == before, "invalid enum/mask/authority never reaches original query");
    expect(owner.sdk.context_set_current(nullptr) == CUDA_SUCCESS, "detach current context");
    rejected(owner.query(Kind::kWeightSpace, 1), CUDA_ERROR_INVALID_CONTEXT);
    expect(owner.sdk.context_set_current(owner.context) == CUDA_SUCCESS, "restore exact context");
    auto foreign =
        std::async(std::launch::async, [&] { return owner.query(Kind::kWeightSpace, 1); });
    rejected(foreign.get(), CUDA_ERROR_UNKNOWN);
    for (const int bad : {3, 4}) {
        fixture::control.corrupt.store(bad);
        rejected(owner.query(Kind::kTempSpace, 2), CUDA_ERROR_INVALID_VALUE);
        native(owner.query(Kind::kTempSpace, 4).result);
        native(owner.query(Kind::kTempSpace, 0).result);
    }
    fixture::control.corrupt.store(0);
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
    rejected(owner.query(Kind::kTempSpace, 2), CUDA_ERROR_UNKNOWN);
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    native(owner.query(Kind::kTempSpace, 6).result);
    for (const auto range : {std::array<double, 2>{-INFINITY, INFINITY}, {-0.0, 0.0}, {-4, 9}}) {
        native(owner.metadata->set_rnn_clip(owner.rnn, {1, 1, range[0], range[1]}));
        expect(fixture::state.clip_bits ==
                   std::array<std::uint64_t, 2>{std::bit_cast<std::uint64_t>(range[0]),
                                                std::bit_cast<std::uint64_t>(range[1])},
               "Clip preserves raw double representations");
    }
    const auto clips = fixture::control.calls[3].load();
    expect(
        owner.metadata->set_rnn_clip(owner.rnn, {2, 0, 0, 1}).status == CUDA_ERROR_INVALID_VALUE &&
            fixture::control.calls[3].load() == clips,
        "Clip representation rejects before native");
    expect(owner.metadata->destroy(owner.dropout).status == CUDA_ERROR_INVALID_VALUE &&
               !owner.metadata->snapshot().is_quarantined &&
               !owner.session->snapshot().is_quarantined && !fixture::control.polls.load() &&
               !fixture::control.synchronizes.load(),
           "clean queries/Clip retain dependency edges without device drain or quarantine");
    owner.close();
}
void exclusion() {
    Owner owner;
    auto mutation = owner.metadata->begin_dropout_mutation(owner.dropout).mutation;
    expect(mutation != nullptr, "unbound dependency remains independently mutable");
    const auto before = fixture::calls();
    for (const auto kind : kKinds)
        expect(owner.query(kind, kind == Kind::kWeightSpace ? 1
                                 : kind == Kind::kTempSpace ? 6
                                                            : 7)
                       .result.disposition == CudaDnnDisposition::kDeferred,
               "pending dependency excludes every native parent logger");
    expect(owner.metadata->set_rnn_clip(owner.rnn, {1, 0, -1, 1}).disposition ==
                   CudaDnnDisposition::kDeferred &&
               fixture::calls() == before && !fixture::control.gets.load(),
           "pending mutation excludes Clip and preflight");
    mutation.reset();
    unsigned int observed{};
    fixture::state.callback = [&] {
        ++observed;
        expect(!owner.metadata->borrow(owner.rnn, CudaDnnObjectKind::kRnn) &&
                   !owner.metadata->begin_dropout_mutation(owner.dropout).mutation &&
                   owner.metadata->destroy(owner.input).status == CUDA_ERROR_NOT_READY &&
                   owner.metadata->shutdown().disposition == CudaDnnDisposition::kDeferred,
               "native callback cannot reenter metadata or invalidate composite owners");
        expect(
            owner.session->destroy_dnn(owner.handle).disposition == CudaDnnDisposition::kDeferred &&
                owner.session->set_dnn_stream(owner.handle, 0).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred &&
                owner.query(Kind::kWeightSpace, 1).result.disposition ==
                    CudaDnnDisposition::kDeferred,
            "query scoped guard rejects recursive handle/stream/session transitions");
        expect(owner.metadata->snapshot().borrowed >= 2,
               "parent and unique dropout are genuinely borrowed");
    };
    for (const auto kind : kKinds)
        native(owner.query(kind, kind == Kind::kWeightSpace ? 1
                                 : kind == Kind::kTempSpace ? 6
                                                            : 7)
                   .result);
    fixture::state.callback = {};
    expect(observed == 4, "preflight and every original are guarded independently");
    auto borrowed = owner.metadata->borrow(owner.rnn, CudaDnnObjectKind::kRnn);
    expect(owner.metadata->set_rnn_clip(owner.rnn, {1, 0, -1, 1}).disposition ==
               CudaDnnDisposition::kDeferred,
           "Clip waits for immutable parent use");
    borrowed.reset();
    fixture::state.callback = [&] {
        expect(!owner.metadata->begin_dropout_mutation(owner.dropout).mutation &&
                   owner.metadata->snapshot().borrowed >= 1,
               "Clip temporarily freezes dependent descriptor");
        owner.metadata->close();
        expect(!fixture::metadata::state.destroys && !fixture::dropout::state.destroys,
               "close during accepted Clip is non-destructive");
    };
    native(owner.metadata->set_rnn_clip(owner.rnn, {1, 0, -1, 1}));
    fixture::state.callback = {};
    expect(owner.metadata->set_rnn_clip(owner.rnn, {}).status == CUDA_ERROR_DEINITIALIZED,
           "close prevents subsequent metadata mutation");
    owner.close();
}
void pending() {
    Owner owner;
    const auto tensor = owner.create(CudaDnnObjectKind::kTensor);
    const CudaDnnTensor shape{CUDNN_DATA_FLOAT, 0, 4, {1, 1, 1, 1}, {1, 1, 1, 1}};
    native(owner.metadata->set_tensor(tensor, CudaDnnTensorSetter::kNd, shape));
    CudaSessionDnnCompute compute;
    compute.kind = CudaDnnComputeKind::kAddTensor;
    compute.handle = owner.handle;
    std::array<std::uint64_t, 2> allocations{};
    for (std::size_t i = 0; i != 2; ++i) {
        compute.tensors[i] = *owner.metadata->borrow(tensor, CudaDnnObjectKind::kTensor);
        const auto allocation = owner.session->allocate(4);
        driver(allocation);
        allocations[i] = allocation.id;
        compute.buffers[i] = {allocation.id, 0};
    }
    const float alpha = 1, beta = 0;
    std::memcpy(compute.alpha.data(), &alpha, 4);
    std::memcpy(compute.beta.data(), &beta, 4);
    compute.scalar_bytes = 4;
    const auto accepted = owner.session->accept_dnn_compute(1, compute);
    expect(accepted.acceptance.accepted(), "accept independent prior compute");
    compute = {};
    foundation::control.can_stop.store(false);
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued,
           "issue one event-held original");
    for (unsigned int phase = 0; phase != 2; ++phase) {
        const auto polls = fixture::control.polls.load();
        for (const auto kind : kKinds)
            native(
                owner.query(kind, kind == Kind::kWeightSpace ? 1
                                  : kind == Kind::kTempSpace ? 6
                                                             : 7)
                    .result);
        expect(fixture::control.polls.load() == polls && !fixture::control.synchronizes.load() &&
                   foundation::driver.events.size() == 1 && !foundation::control.can_stop.load(),
               "queries never poll/drain/synchronize/add events or retire NOT_READY work");
        if (!phase)
            expect(owner.session->acknowledge_result(accepted.acceptance.receipt),
                   "ACK is not physical retirement");
    }
    expect(
        owner.metadata->destroy(tensor).disposition == CudaDnnDisposition::kDeferred &&
            owner.session->destroy_dnn(owner.handle).disposition == CudaDnnDisposition::kDeferred,
        "prior descriptor and handle survive all queries and ACK");
    for (const auto allocation : allocations)
        expect(owner.session->release(allocation).disposition == CudaSessionDisposition::kDeferred,
               "prior allocation remains physically retained");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "only physical success retires prior call");
    for (const auto allocation : allocations) driver(owner.session->release(allocation));
    native(owner.metadata->destroy(tensor));
    owner.close();
    expect(foundation::driver.frees == 2, "each prior allocation frees once");
}
void accepted_close() {
    Owner owner;
    fixture::state.callback = [&] {
        owner.metadata->close();
        owner.session->close();
        expect(!fixture::metadata::state.destroys && !base::state.handle_destroys,
               "closing both owners during native query does not destroy accepted resources");
    };
    native(owner.query(Kind::kWeightSpace, 1).result);
    fixture::state.callback = {};
    rejected(owner.query(Kind::kWeightSpace, 1), CUDA_ERROR_DEINITIALIZED);
    native(owner.metadata->shutdown());
    driver(owner.session->shutdown());
    expect(base::state.handle_destroys == 1 && fixture::metadata::state.destroys == 2 &&
               fixture::dropout::state.destroys == 1,
           "closed owners clean exactly once after accepted query");
}
void missing() {
    for (unsigned int absent = 1; absent <= 5; ++absent) {
        Owner owner(absent);
        if (absent == 4) {
            const auto result = owner.metadata->set_rnn_clip(owner.rnn, {});
            expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status,
                   "missing Clip is clean policy");
        } else
            rejected(owner.query(absent == 1   ? Kind::kWeightSpace
                                 : absent == 3 ? Kind::kAttentionBuffers
                                               : Kind::kTempSpace,
                                 absent == 1   ? 1
                                 : absent == 3 ? 7
                                               : 2),
                     CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture::calls() && !fixture::control.gets.load() &&
                   !owner.metadata->snapshot().is_quarantined,
               "missing original capability invokes neither query nor preflight");
        native(owner
                   .query(absent == 1 ? Kind::kAttentionBuffers : Kind::kWeightSpace,
                          absent == 1 ? 7 : 1)
                   .result);
        owner.close();
    }
}
void poison(unsigned int which) {
    Owner owner;
    if (which < 2) {
        fixture::control.corrupt.store(static_cast<int>(which + 1));
        rejected(owner.query(Kind::kTempSpace, 2), CUDA_ERROR_UNKNOWN);
        expect(owner.metadata->snapshot().is_quarantined &&
                   !owner.session->snapshot().is_quarantined && !fixture::calls(),
               "corrupt successful preflight faults metadata, not proven execution");
        native(owner.session->destroy_dnn(owner.handle));
        driver(owner.session->shutdown());
    } else if (which < 6) {
        fixture::control.context_loss.store(which < 4 ? 1 : 2);
        if (which == 3) fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
        if (which == 5) fixture::control.statuses[1].store(CUDNN_STATUS_NOT_SUPPORTED);
        const auto result = owner.query(Kind::kTempSpace, 6);
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_INVALID_CONTEXT &&
                   result.sizes == std::array<std::uint64_t, 3>{} &&
                   result.result.original_status ==
                       (which < 4
                            ? std::optional<cudnnStatus_t>{}
                            : std::optional<cudnnStatus_t>{which == 5 ? CUDNN_STATUS_NOT_SUPPORTED
                                                                      : CUDNN_STATUS_SUCCESS}) &&
                   owner.session->snapshot().is_quarantined &&
                   !owner.metadata->snapshot().is_quarantined,
               "context proof preserves only the requested query's actual native provenance");
        expect(!base::state.handle_destroys && !foundation::control.context_destroys.load(),
               "lost-context cleanup never runs");
    } else {
        fixture::control.statuses[3].store(which == 6 ? CUDNN_STATUS_ALLOC_FAILED
                                                      : CUDNN_STATUS_SUCCESS);
        const double lower = which == 7   ? std::bit_cast<double>(UINT64_C(0x7ff8000000000042))
                             : which == 8 ? 3
                                          : -1;
        native(owner.metadata->set_rnn_clip(owner.rnn, {1, 0, lower, 1}),
               which == 6 ? CUDNN_STATUS_ALLOC_FAILED : CUDNN_STATUS_BAD_PARAM);
        expect(owner.metadata->snapshot().is_quarantined &&
                   !owner.session->snapshot().is_quarantined &&
                   fixture::metadata::object(fixture::state.descriptor, CudaDnnObjectKind::kRnn)
                           .rnn.dropout != nullptr,
               "issued Clip failure quarantines partial metadata without removing dropout binding");
        native(owner.session->destroy_dnn(owner.handle));
        driver(owner.session->shutdown());
    }
    expect(!fixture::control.synchronizes.load() && !fixture::dropout::state.destroys,
           "quarantine performs no hidden drain or dependency destruction");
    ::_exit(0);
}
}  // namespace
int main() {
    try {
        ordinary();
        exclusion();
        pending();
        accepted_close();
        missing();
        for (unsigned int which = 0; which != 9; ++which) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated original-error ownership case");
            if (!child) poison(which);
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated RNN query/Clip containment passes");
        }
        std::cout << "RNN size-query and Clip CPU ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
