#include "dnn_attn_weights_fixture.h"

#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_attn_weights;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
void expect(bool value, const char* text) {
    if (!value) throw std::runtime_error(text);
}
void native(const CudaDnnResult& result, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "exact original attention layout status");
}
void driver(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
void rejected(const CudaDnnAttnWeightsResult& result, CUresult status) {
    expect(result.result.disposition == CudaDnnDisposition::kRejected &&
               result.result.status == status && !result.result.original_status &&
               !result.non_null && !result.offset,
           "clean rejection has no requested-helper provenance or result pointer");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t parent{}, destination{}, handle{};
    std::array<std::uint64_t, 2> dropouts{};
    CudaDnnAttnState config;
    CUcontext context{};
    explicit Owner(unsigned int missing = 0) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        parent = create(CudaDnnObjectKind::kAttn);
        destination = create(CudaDnnObjectKind::kTensor);
        for (auto& id : dropouts) id = create(CudaDnnObjectKind::kDropout);
        config.heads = 1;
        config.mode = CUDNN_ATTN_ENABLE_PROJ_BIASES;
        config.query_size = config.key_size = config.value_size = 2;
        config.query_projection = config.key_projection = config.value_projection =
            config.output_projection = 2;
        config.max_query_sequence = 2;
        config.max_key_sequence = 3;
        config.max_batch = config.max_beam = 1;
        config.dropouts = dropouts;
        native(metadata->set_attn(parent, config));
        native(metadata->set_tensor(destination, CudaDnnTensorSetter::kNd,
                                    {CUDNN_DATA_FLOAT, 0, 3, {1, 1, 1}, {1, 1, 1}}));
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{281, 282}, 283, sdk);
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
    CudaDnnAttnWeights problem() const {
        return {parent, 0, true, 512, destination, true};
    }
    CudaDnnAttnWeightsResult query(CudaDnnAttnWeights problem) {
        return session->query_dnn_attn_weights(handle, *metadata, problem);
    }
    void unprojected() {
        config.query_projection = config.key_projection = config.value_projection =
            config.output_projection = 0;
        native(metadata->set_attn(parent, config));
    }
    void close() {
        native(session->destroy_dnn(handle));
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(metadata->snapshot().is_closed && fixture::metadata::state.destroys == 1 &&
                   fixture::dropout::state.destroys == 2 && base::state.handle_destroys == 1 &&
                   foundation::driver.contexts.empty() && foundation::driver.events.empty() &&
                   foundation::driver.memory.empty(),
               "all proven original owners retire exactly once");
    }
};
void ordinary() {
    Owner owner;
    for (unsigned int kind = 0; kind != 8; ++kind) {
        auto problem = owner.problem();
        problem.kind = kind;
        const auto result = owner.query(problem);
        native(result.result);
        const auto expected = kind < 4 ? 32U * kind : 192U + 16U * (kind - 4);
        expect(
            result.non_null && result.offset == expected && fixture::state.last.kind == kind &&
                fixture::state.last.base == reinterpret_cast<void*>(1) &&
                fixture::state.last.bytes == 512 && fixture::state.last.has_output,
            "all logical parameter kinds map relative to private sentinel, including offset zero");
        const auto shape = owner.metadata->get_tensor(owner.destination, 3);
        native(shape);
        expect(shape.tensor.sizes == std::array<int, 8>{1, 2, kind < 4 ? 2 : 1} &&
                   shape.tensor.strides == std::array<int, 8>{2, 1, 2},
               "original column-major metadata remains authoritative");
    }
    auto problem = owner.problem();
    problem.weight_bytes = (std::size_t{1} << 40) + 7;
    native(owner.query(problem).result);
    expect(fixture::state.last.bytes == problem.weight_bytes,
           "no capacity narrowing or allocation");
    owner.config.dropouts[1] = owner.dropouts[0];
    native(owner.metadata->set_attn(owner.parent, owner.config));
    native(owner.query(problem).result);
    owner.unprojected();
    for (const bool has_base : {false, true}) {
        problem = owner.problem();
        problem.has_base = has_base;
        problem.weight_bytes = 0;
        const auto result = owner.query(problem);
        native(result.result);
        expect(!result.non_null && !result.offset &&
                   fixture::state.last.base == (has_base ? reinterpret_cast<void*>(1) : nullptr),
               "absent original parameter is distinct from nonnull offset zero");
    }
    expect(!fixture::query::control.polls.load() && !fixture::query::control.synchronizes.load() &&
               foundation::driver.events.empty() && foundation::driver.memory.empty(),
           "layout helper has no queue, GPU allocation, event, drain or RNG state lease");
    owner.close();
}
void controls() {
    Owner owner;
    const auto calls = fixture::control.calls.load();
    auto problem = owner.problem();
    problem.weight_bytes = SIZE_MAX;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    expect(!fixture::control.versions.load(),
           "sentinel interval overflow rejects before version observation");
    problem = owner.problem();
    problem.kind = 8;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    problem = owner.problem();
    problem.destination = owner.dropouts[0];
    rejected(owner.query(problem), CUDA_ERROR_INVALID_HANDLE);
    problem = owner.problem();
    problem.descriptor = owner.destination;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_HANDLE);
    rejected(owner.session->query_dnn_attn_weights(UINT64_MAX, *owner.metadata, owner.problem()),
             CUDA_ERROR_INVALID_HANDLE);
    expect(fixture::control.calls.load() == calls, "invalid kind/authority cannot invoke helper");
    auto parent = owner.metadata->borrow(owner.parent, CudaDnnObjectKind::kAttn);
    native(owner.query(owner.problem()).result);
    parent.reset();
    auto output = owner.metadata->borrow(owner.destination, CudaDnnObjectKind::kTensor);
    expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred,
           "externally immutable destination defers without native mutation");
    output.reset();
    for (const auto dropout : owner.dropouts) {
        auto mutation = owner.metadata->begin_dropout_mutation(dropout);
        expect(bool(mutation.mutation), "reserve each distinct dependent dropout");
        const auto before = fixture::control.calls.load();
        expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred &&
                   fixture::control.calls.load() == before,
               "either pending dropout mutation defers whole-parent logging");
        mutation.mutation.reset();
    }
    native(owner.query(owner.problem()).result);
    owner.close();
}
void reentry() {
    Owner owner;
    unsigned int checks{};
    const auto check = [&] {
        ++checks;
        expect(
            owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred &&
                owner.session->activate().disposition == CudaSessionDisposition::kDeferred &&
                owner.session->deactivate().disposition == CudaSessionDisposition::kDeferred &&
                owner.session->get_dnn_stream(owner.handle).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->destroy_dnn(owner.handle).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred,
            "version/original callbacks cannot reenter execution owner or change current context");
        expect(!owner.metadata->borrow(owner.parent, CudaDnnObjectKind::kAttn) &&
                   !owner.metadata->begin_dropout_mutation(owner.dropouts[0]).mutation &&
                   !owner.metadata->begin_dropout_mutation(owner.dropouts[1]).mutation &&
                   owner.metadata->shutdown().disposition == CudaDnnDisposition::kDeferred,
               "callbacks cannot retire immutable descriptors or either transitive dependency");
    };
    fixture::state.version_callback = check;
    fixture::state.callback = check;
    native(owner.query(owner.problem()).result);
    expect(
        checks == 2 && fixture::control.calls.load() == 1 && fixture::control.versions.load() == 1,
        "one version observation and one actual original, without recursion");
    fixture::state.version_callback = {};
    fixture::state.callback = [&] {
        owner.metadata->close();
        owner.session->close();
        expect(!base::state.handle_destroys && !fixture::metadata::state.destroys,
               "close does not tear down an admitted original");
    };
    native(owner.query(owner.problem()).result);
    fixture::state.callback = {};
    rejected(owner.query(owner.problem()), CUDA_ERROR_DEINITIALIZED);
    native(owner.metadata->shutdown());
    driver(owner.session->shutdown());
}
void stale_rng() {
    Owner owner;
    const auto allocation = owner.session->allocate(fixture::dropout::kStateBytes);
    driver(allocation);
    CudaSessionDnnDropout problem;
    problem.kind = CudaDnnDropoutKind::kRestore;
    problem.handle = owner.handle;
    problem.mutation = owner.metadata->begin_dropout_mutation(owner.dropouts[0]).mutation;
    problem.states = {allocation.id, 0};
    problem.state_bytes = fixture::dropout::kStateBytes;
    problem.probability_bits = 0x3e800000U;
    problem.seed = 0xfedcba9876543210ULL;
    const auto accepted = owner.session->accept_dnn_dropout(1, problem);
    expect(accepted.acceptance.accepted(), "admit real Restore state binding");
    problem = {};
    expect(owner.session->issue_next().library_status == CUDNN_STATUS_SUCCESS,
           "original Restore issued");
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt), "Restore ACK");
    expect(owner.session->poll().retired == 1, "physical Restore event retires");
    driver(owner.session->release(allocation.id));
    auto descriptor = owner.metadata->borrow_dropout(owner.dropouts[0]);
    expect(descriptor && !owner.session->resolve_dnn_dropout(*descriptor),
           "state provenance is genuinely stale");
    const auto synchronizes = fixture::query::control.synchronizes.load();
    native(owner.query(owner.problem()).result);
    expect(fixture::query::control.synchronizes.load() == synchronizes &&
               foundation::driver.memory.empty(),
           "layout helper neither resolves freed RNG state nor introduces hidden allocation/sync");
    descriptor.reset();
    owner.close();
}
void pending() {
    Owner owner;
    CudaSessionDnnCompute call;
    call.kind = CudaDnnComputeKind::kAddTensor;
    call.handle = owner.handle;
    std::array<std::uint64_t, 2> allocations{};
    for (std::size_t i = 0; i != 2; ++i) {
        call.tensors[i] = *owner.metadata->borrow(owner.destination, CudaDnnObjectKind::kTensor);
        const auto memory = owner.session->allocate(4);
        driver(memory);
        allocations[i] = memory.id;
        call.buffers[i] = {memory.id, 0};
    }
    const float alpha = 1, beta = 0;
    std::memcpy(call.alpha.data(), &alpha, 4);
    std::memcpy(call.beta.data(), &beta, 4);
    call.scalar_bytes = 4;
    const auto accepted = owner.session->accept_dnn_compute(1, call);
    expect(accepted.acceptance.accepted(), "accept physically borrowed destination");
    call = {};
    foundation::control.can_stop.store(false);
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued,
           "issue original AddTensor");
    const auto polls = fixture::query::control.polls.load();
    expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred,
           "physical destination borrow blocks metadata mutation");
    auto problem = owner.problem();
    problem.destination = owner.create(CudaDnnObjectKind::kTensor);
    native(owner.query(problem).result);
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt),
           "ACK prior error receipt");
    expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred &&
               fixture::query::control.polls.load() == polls &&
               !fixture::query::control.synchronizes.load(),
           "independent query and ACK cannot drain or release NOT_READY physical work");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "only physical event success retires descriptor");
    native(owner.query(owner.problem()).result);
    for (const auto id : allocations) driver(owner.session->release(id));
    owner.close();
}
void poison(unsigned int index) {
    Owner owner;
    auto problem = owner.problem();
    if (index < 3) {
        problem.destination = index & 1 ? owner.destination : 0;
        problem.has_output = bool(index & 2);
    }
    if (index == 3) problem.has_base = false;
    if (index == 4) problem.weight_bytes = 0;
    if (index == 5) fixture::control.status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    if (index == 6 || index == 7) fixture::control.corrupt.store(index - 5);
    if (index == 8) fixture::control.context_loss.store(true);
    if (index == 9) fixture::control.version_context_loss.store(true);
    if (index >= 10) {
        fixture::control.impossible_success.store(true);
        if (index == 10) problem.destination = 0;
        if (index == 11) problem.has_output = 0;
        if (index == 12) problem.has_base = 0;
    }
    const auto result = owner.query(problem);
    expect(!result.non_null && !result.offset, "failed private result never publishes an offset");
    if (index < 6)
        native(result.result,
               index == 5 ? CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE : CUDNN_STATUS_BAD_PARAM);
    else if (index < 8 || index >= 10)
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_UNKNOWN &&
                   result.result.original_status == CUDNN_STATUS_SUCCESS,
               "corrupt mapping or impossible nullable success retains provenance and faults "
               "metadata");
    else
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_INVALID_CONTEXT && result.has_context_loss &&
                   result.result.original_status ==
                       (index == 8 ? std::optional<cudnnStatus_t>{CUDNN_STATUS_SUCCESS}
                                   : std::nullopt),
               "version preflight loss differs from post-original context provenance");
    expect(fixture::control.calls.load() == (index == 9 ? 0U : 1U) &&
               owner.metadata->snapshot().is_quarantined == (index != 9),
           "preflight does not fabricate native mutation or metadata failure");
    if (index < 8 || index >= 10) {
        native(owner.session->destroy_dnn(owner.handle));
        driver(owner.session->shutdown());
    }
    ::_exit(0);
}
}  // namespace
int main() {
    try {
        ordinary();
        controls();
        reentry();
        stale_rng();
        pending();
        for (unsigned int missing = 1; missing != 5; ++missing) {
            Owner owner(missing < 3 ? missing : 0);
            if (missing >= 3) fixture::control.version.store(missing == 3 ? 91001 : 91003);
            rejected(owner.query(owner.problem()), CUDA_ERROR_NOT_SUPPORTED);
            expect(!fixture::control.calls.load() && !owner.metadata->snapshot().is_quarantined,
                   "missing callback/version and unqualified SDK identity reject cleanly");
            owner.close();
        }
        for (unsigned int index = 0; index != 13; ++index) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated metadata mutation failure");
            if (!child) poison(index);
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated attention-layout poison case");
        }
        std::cout << "Attention-layout CPU ownership and arithmetic checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
