#include "dnn_rnn_weight_params_fixture.h"

#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_rnn_weight_params;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
void expect(bool value, const char* text) {
    if (!value) throw std::runtime_error(text);
}
void native(const CudaDnnResult& result, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "exact original WeightParams/metadata status");
}
void driver(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
void rejected(const CudaDnnRnnWeightParamsResult& result, CUresult status) {
    expect(result.result.disposition == CudaDnnDisposition::kRejected &&
               result.result.status == status && !result.result.original_status &&
               !result.non_null && result.offsets == std::array<std::uint64_t, 2>{},
           "clean preissue rejection has no original provenance or published addresses");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t parent{}, dropout{}, handle{}, allocation{};
    std::array<std::uint64_t, 2> destinations{};
    CUdeviceptr address{};
    CUcontext context{};
    explicit Owner(bool missing = false) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        parent = create(CudaDnnObjectKind::kRnn);
        dropout = create(CudaDnnObjectKind::kDropout);
        CudaDnnRnnState rnn;
        rnn.cell_mode = CUDNN_LSTM;
        rnn.input_size = rnn.hidden_size = rnn.projection_size = 2;
        rnn.layers = 1;
        rnn.dropout = dropout;
        native(metadata->set_rnn(parent, rnn));
        for (auto& id : destinations) {
            id = create(CudaDnnObjectKind::kTensor);
            native(metadata->set_tensor(id, CudaDnnTensorSetter::kNd,
                                        {CUDNN_DATA_FLOAT, 0, 3, {1, 1, 1}, {1, 1, 1}}));
        }
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{181, 182}, 183, sdk);
        driver(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
        const auto memory = session->allocate(544);
        driver(memory);
        allocation = memory.id;
        address = memory.address;
        std::memset(reinterpret_cast<void*>(address), 0x6d, 544);
    }
    std::uint64_t create(CudaDnnObjectKind kind) {
        const auto result = metadata->create(kind);
        native(result);
        return result.id;
    }
    CudaSessionDnnRnnWeightParams problem() const {
        return {{parent, 0, 0, destinations, 3}, {allocation, 16}, 512};
    }
    CudaDnnRnnWeightParamsResult query(const CudaSessionDnnRnnWeightParams& value) {
        return session->query_dnn_rnn_weight_params(handle, *metadata, value);
    }
    void close() {
        driver(session->release(allocation));
        native(session->destroy_dnn(handle));
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(metadata->snapshot().is_closed && foundation::driver.memory.empty() &&
                   foundation::driver.events.empty() && foundation::driver.contexts.empty() &&
                   fixture::metadata::state.destroys == 1 &&
                   fixture::dropout::state.destroys == 1 && base::state.handle_destroys == 1,
               "all proven owners retire once");
    }
};
void ordinary() {
    Owner owner;
    for (unsigned int mask = 0; mask != 4; ++mask) {
        auto problem = owner.problem();
        problem.parameters.outputs = mask;
        for (std::size_t i = 0; i != 2; ++i)
            if (!(mask & (1U << i))) problem.parameters.destinations[i] = 0;
        const auto result = owner.query(problem);
        native(result.result);
        expect(result.non_null == mask && result.offsets[0] == 0 &&
                   result.offsets[1] == (mask & 2 ? 256U : 0U) &&
                   fixture::state.last.outputs == mask &&
                   fixture::state.last.weights == reinterpret_cast<void*>(owner.address + 16) &&
                   fixture::state.last.bytes == 512,
               "independent optional pairs and owned base-relative offsets remain exact");
    }
    auto problem = owner.problem();
    problem.weights = {};
    auto result = owner.query(problem);
    native(result.result);
    expect(result.non_null == 2 && result.offsets == std::array<std::uint64_t, 2>{0, 256},
           "raw null-base first matrix offset zero preserves original ambiguity");
    for (int linear = 1; linear != 8; ++linear) {
        problem.parameters.linear_layer = linear;
        result = owner.query(problem);
        native(result.result);
        expect(result.non_null == 3 && result.offsets[0] == 32U * linear &&
                   result.offsets[1] == 256U + 16U * linear,
               "every logical slot is positional, not a fabricated allocation");
    }
    problem.parameters.linear_layer = 8;
    problem.weight_bytes = 0;
    result = owner.query(problem);
    native(result.result);
    expect(!result.non_null && result.offsets == std::array<std::uint64_t, 2>{},
           "absent projection and bias preserve native NULL and zero capacity");
    problem.weights = {owner.allocation, 544};
    native(owner.query(problem).result);
    expect(fixture::state.last.weights == reinterpret_cast<void*>(owner.address + 544),
           "owned zero-byte endpoint forwarded without fabricated minimum extent");
    problem = owner.problem();
    problem.parameters.destinations[1] = problem.parameters.destinations[0];
    result = owner.query(problem);
    native(result.result);
    const auto tensor = owner.metadata->get_tensor(owner.destinations[0], 3);
    native(tensor);
    expect(tensor.tensor.sizes[2] == 1 && result.non_null == 3 && result.offsets[1] == 256 &&
               fixture::state.last.destinations[0] == fixture::state.last.destinations[1],
           "shared destination identity is one owner, original bias mutation wins");
    expect(std::all_of(reinterpret_cast<const std::byte*>(owner.address),
                       reinterpret_cast<const std::byte*>(owner.address + 544),
                       [](std::byte value) { return value == std::byte{0x6d}; }) &&
               foundation::driver.events.empty() && !fixture::query::control.polls.load() &&
               !fixture::query::control.synchronizes.load(),
           "host query neither touches any weight/guard byte nor polls/drains/issues GPU work");
    owner.close();
}
void controls() {
    Owner owner;
    const auto calls = fixture::control.calls.load();
    auto problem = owner.problem();
    problem.parameters.outputs = 1;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    problem = owner.problem();
    problem.parameters.outputs = 4;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    problem = owner.problem();
    problem.parameters.destinations[1] = owner.dropout;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_HANDLE);
    problem = owner.problem();
    problem.parameters.descriptor = owner.destinations[0];
    rejected(owner.query(problem), CUDA_ERROR_INVALID_HANDLE);
    for (const auto range : {CudaAllocationRange{owner.allocation, 545},
                             CudaAllocationRange{UINT64_MAX, 0}, CudaAllocationRange{0, 1}}) {
        problem = owner.problem();
        problem.weights = range;
        rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    }
    problem = owner.problem();
    problem.weight_bytes = SIZE_MAX;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    rejected(
        owner.session->query_dnn_rnn_weight_params(UINT64_MAX, *owner.metadata, owner.problem()),
        CUDA_ERROR_INVALID_HANDLE);
    expect(fixture::control.calls.load() == calls, "all structural/authority errors precede SDK");
    auto parent = owner.metadata->borrow(owner.parent, CudaDnnObjectKind::kRnn);
    native(owner.query(owner.problem()).result);
    parent.reset();
    auto destination = owner.metadata->borrow(owner.destinations[0], CudaDnnObjectKind::kTensor);
    auto result = owner.query(owner.problem());
    expect(result.result.disposition == CudaDnnDisposition::kDeferred &&
               result.result.status == CUDA_ERROR_NOT_READY && !result.result.original_status,
           "external destination reader defers without fake original status");
    destination.reset();
    auto mutation = owner.metadata->begin_dropout_mutation(owner.dropout);
    expect(bool(mutation.mutation), "obtain pending dropout mutation");
    result = owner.query(owner.problem());
    expect(result.result.disposition == CudaDnnDisposition::kDeferred,
           "whole-parent logging dependencies defer while dropout mutates");
    mutation.mutation.reset();
    native(owner.query(owner.problem()).result);
    auto memory = owner.session->allocate(16);
    driver(memory);
    driver(owner.session->release(memory.id));
    problem = owner.problem();
    problem.weights = {memory.id, 0};
    problem.weight_bytes = 0;
    rejected(owner.query(problem), CUDA_ERROR_INVALID_VALUE);
    owner.close();
}
void reentry() {
    Owner owner;
    unsigned int entered{};
    fixture::state.callback = [&] {
        ++entered;
        expect(
            !owner.metadata->borrow(owner.parent, CudaDnnObjectKind::kRnn) &&
                !owner.metadata->begin_dropout_mutation(owner.dropout).mutation &&
                owner.metadata->destroy(owner.destinations[0]).status == CUDA_ERROR_NOT_READY &&
                owner.metadata->shutdown().disposition == CudaDnnDisposition::kDeferred &&
                owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred &&
                owner.session->release(owner.allocation).disposition ==
                    CudaSessionDisposition::kDeferred &&
                owner.session->destroy_dnn(owner.handle).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->set_dnn_stream(owner.handle, 0).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->activate().disposition == CudaSessionDisposition::kDeferred &&
                owner.session->deactivate().disposition == CudaSessionDisposition::kDeferred &&
                owner.session->get_dnn_stream(owner.handle).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred,
            "native callback cannot invalidate context, range, destination or composite parent");
    };
    native(owner.query(owner.problem()).result);
    expect(entered == 1, "exactly one original invocation, not speculative replay");
    fixture::state.callback = [&] {
        owner.metadata->close();
        owner.session->close();
        expect(!base::state.handle_destroys && !fixture::metadata::state.destroys,
               "close is non-destructive inside accepted native call");
    };
    native(owner.query(owner.problem()).result);
    fixture::state.callback = {};
    rejected(owner.query(owner.problem()), CUDA_ERROR_DEINITIALIZED);
    native(owner.metadata->shutdown());
    driver(owner.session->shutdown());
}
void pending() {
    Owner owner;
    CudaSessionDnnCompute compute;
    compute.kind = CudaDnnComputeKind::kAddTensor;
    compute.handle = owner.handle;
    std::array<std::uint64_t, 2> allocations{};
    for (std::size_t i = 0; i != 2; ++i) {
        compute.tensors[i] =
            *owner.metadata->borrow(owner.destinations[0], CudaDnnObjectKind::kTensor);
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
    expect(accepted.acceptance.accepted(), "accept prior physical work");
    compute = {};
    foundation::control.can_stop.store(false);
    expect(owner.session->issue_next().state == CudaExecutionIssueState::kIssued,
           "original failed compute still records physical event");
    const auto polls = fixture::query::control.polls.load();
    expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred,
           "physically borrowed destination cannot mutate");
    auto fresh = owner.problem();
    fresh.parameters.destinations[0] = fresh.parameters.destinations[1];
    native(owner.query(fresh).result);
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt), "ACK host receipt");
    expect(owner.query(owner.problem()).result.disposition == CudaDnnDisposition::kDeferred &&
               fixture::query::control.polls.load() == polls &&
               !fixture::query::control.synchronizes.load() &&
               foundation::driver.events.size() == 1,
           "ACK and successful independent query neither drain nor release physical owners");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "physical success alone retires old work");
    native(owner.query(owner.problem()).result);
    for (auto id : allocations) driver(owner.session->release(id));
    owner.close();
}
void poison(unsigned int index) {
    Owner owner;
    auto problem = owner.problem();
    if (index < 4) fixture::control.corrupt.store(index + 1);
    if (index == 4) fixture::control.status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    if (index == 5) problem.weight_bytes = 16;  // Matrix changes before bias capacity failure.
    if (index == 6) problem.parameters.destinations[1] = 0;
    if (index == 7) fixture::control.context_loss.store(true);
    if (index == 8) {
        expect(owner.sdk.context_set_current(nullptr) == CUDA_SUCCESS, "lose current before call");
    }
    if (index >= 9) {
        const auto value = index % 2 ? INT32_MIN : INT32_MAX;
        if (index < 11)
            problem.parameters.pseudo_layer = value;
        else
            problem.parameters.linear_layer = value;
    }
    const auto result = owner.query(problem);
    expect(!result.non_null && result.offsets == std::array<std::uint64_t, 2>{},
           "no successful prefix escapes any failure");
    if (index < 4) {
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_UNKNOWN &&
                   result.result.original_status == CUDNN_STATUS_SUCCESS,
               "out-of-range successful original retains exact provenance");
    } else if (index < 7 || index >= 9) {
        native(result.result,
               index == 4 ? CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE : CUDNN_STATUS_BAD_PARAM);
    } else {
        expect(result.result.disposition == CudaDnnDisposition::kRejected &&
                   result.result.status == CUDA_ERROR_INVALID_CONTEXT && result.has_context_loss &&
                   result.result.original_status ==
                       (index == 7 ? std::optional<cudnnStatus_t>{CUDNN_STATUS_SUCCESS}
                                   : std::nullopt),
               "postnative and preissue context failures have distinct provenance");
    }
    if (index >= 9)
        expect(fixture::state.last.pseudo_layer == problem.parameters.pseudo_layer &&
                   fixture::state.last.linear_layer == problem.parameters.linear_layer &&
                   fixture::state.last.outputs == 3 && !result.has_context_loss,
               "both signed extrema reach their exact original ABI slots without normalization");
    expect(owner.metadata->snapshot().is_quarantined == (index != 8) &&
               fixture::control.calls.load() == (index == 8 ? 0U : 1U),
           "issued partial mutation faults metadata; preissue context loss does not");
    if (index < 7 || index >= 9) {
        native(owner.session->destroy_dnn(owner.handle));
        driver(owner.session->release(owner.allocation));
        driver(owner.session->shutdown());
        expect(base::state.handle_destroys == 1,
               "metadata-only poison preserves execution cleanup");
    }
    ::_exit(0);  // Isolate deliberately retained, quarantined descriptor ownership.
}
}  // namespace
int main() {
    try {
        ordinary();
        controls();
        reentry();
        pending();
        {
            Owner owner(true);
            rejected(owner.query(owner.problem()), CUDA_ERROR_NOT_SUPPORTED);
            expect(!fixture::control.calls.load() && !owner.metadata->snapshot().is_quarantined,
                   "missing original callback is clean policy");
            owner.close();
        }
        for (unsigned int index = 0; index != 13; ++index) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated mutation failure");
            if (!child) poison(index);
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated WeightParams poison case");
        }
        std::cout << "WeightParams CPU ownership and address-mapping checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
