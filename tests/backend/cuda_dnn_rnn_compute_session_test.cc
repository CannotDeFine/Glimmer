#include "dnn_rnn_compute_fixture.h"

#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_rnn_compute;
namespace foundation = fixture::foundation;
using Kind = fixture::Kind;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void native(const CudaDnnResult& result, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "exact original RNN result");
}
void driver(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t rnn{}, dropout{}, handle{}, stream{}, next{1};
    std::array<std::uint64_t, 4> descriptors{};
    std::array<CudaAllocationRange, 17> buffers{};
    std::array<std::size_t, 17> bytes{};
    CudaDnnRnnDataState data;
    CudaDnnRnnState parent;
    std::size_t scalar{};
    CUcontext context{};
    std::map<std::uint64_t, fixture::Invocation> expected_invocations;
    Owner(unsigned int type = CUDNN_DATA_FLOAT, unsigned int layout = 0, unsigned int algorithm = 0,
          unsigned int missing = 0, bool default_stream = false) {
        fixture::reset();
        scalar = type == CUDNN_DATA_DOUBLE ? 8 : 4;
        sdk = fixture::session_dispatch(missing);
        auto sizing = std::make_shared<CudaDnnDispatch>(*sdk.dnn);
        sizing->tensor_size = [](cudnnTensorDescriptor_t descriptor, std::size_t* bytes) {
            const auto status = fixture::dispatch().tensor_size(descriptor, bytes);
            // The shared numerical fixture handles float/double. These ABI-only
            // safety cases also need the original HALF descriptor's exact hull.
            if (status == CUDNN_STATUS_SUCCESS &&
                fixture::base::object(descriptor, CudaDnnObjectKind::kTensor).tensor.type ==
                    CUDNN_DATA_HALF)
                *bytes /= 2;
            return status;
        };
        sdk.dnn = std::move(sizing);
        metadata = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        rnn = create(CudaDnnObjectKind::kRnn);
        dropout = create(CudaDnnObjectKind::kDropout);
        parent.algorithm = algorithm;
        parent.cell_mode = CUDNN_LSTM;
        parent.type = parent.math_precision = type;
        parent.input_size = parent.hidden_size = parent.projection_size = parent.layers = 1;
        parent.dropout = dropout;
        native(metadata->set_rnn(rnn, parent));
        data = {.type = type,
                .layout = layout,
                .presence = 3,
                .padding_bytes = static_cast<std::uint32_t>(scalar),
                .max_sequence_length = 3,
                .batch_size = 2,
                .vector_size = 1,
                .lengths = {3, 2}};
        for (std::size_t i = 0; i != 4; ++i) {
            descriptors[i] =
                create(i < 2 ? CudaDnnObjectKind::kRnnData : CudaDnnObjectKind::kTensor);
            if (i < 2)
                native(metadata->set_rnn_data(descriptors[i], data));
            else
                native(metadata->set_tensor(descriptors[i], CudaDnnTensorSetter::kNd,
                                            {type, 0, 3, {1, 2, 1}, {2, 1, 1}}));
        }
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk);
        driver(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
        if (!default_stream) {
            const auto result = session->create_stream(CU_STREAM_NON_BLOCKING);
            driver(result);
            stream = result.id;
            native(session->set_dnn_stream(handle, stream));
        }
        bytes = {8,
                 fixture::elements(data) * scalar,
                 fixture::elements(data) * scalar,
                 fixture::elements(data) * scalar,
                 fixture::elements(data) * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 2 * scalar,
                 fixture::kWeights * scalar,
                 32,
                 fixture::kReserve * scalar,
                 fixture::kWeights * scalar};
        for (std::size_t i = 0; i != buffers.size(); ++i) {
            const auto allocated = session->allocate(bytes[i] + 32);
            driver(allocated);
            buffers[i] = {allocated.id, 16};
            std::memset(reinterpret_cast<void*>(allocated.address + 16), 0, bytes[i]);
        }
        driver(session->upload(buffers[0], std::as_bytes(std::span(fixture::kLengths))));
        if (scalar == 8)
            initialize<double>();
        else
            initialize<float>();
    }
    std::uint64_t create(CudaDnnObjectKind kind) {
        const auto result = metadata->create(kind);
        native(result);
        return result.id;
    }
    template <typename T>
    T* memory(std::size_t slot) {
        const auto view = session->device_memory(buffers[slot], bytes[slot]);
        expect(view.has_value(), "fake device range is owned");
        return reinterpret_cast<T*>(view->address);
    }
    template <typename T>
    void initialize() {
        for (std::size_t t = 0; t != fixture::kSteps; ++t)
            for (std::size_t b = 0; b != fixture::kBatch; ++b) {
                if (data.layout == 1 && static_cast<int>(t) >= data.lengths[b]) continue;
                memory<T>(1)[fixture::index(data, t, b)] =
                    static_cast<T>(fixture::kInput[t * 2 + b]);
                memory<T>(3)[fixture::index(data, t, b)] = static_cast<T>(fixture::kDy[t * 2 + b]);
            }
        for (std::size_t i = 0; i != fixture::kWeights; ++i) {
            memory<T>(13)[i] = static_cast<T>(fixture::kWeight[i]);
            memory<T>(16)[i] = T{.125};
        }
        for (std::size_t i = 0; i != 2; ++i) {
            memory<T>(5)[i] = static_cast<T>(fixture::kHidden[i]);
            memory<T>(9)[i] = static_cast<T>(fixture::kCell[i]);
            memory<T>(7)[i] = static_cast<T>(fixture::kDhidden[i]);
            memory<T>(11)[i] = static_cast<T>(fixture::kDcell[i]);
        }
    }
    CudaSessionDnnRnn problem(Kind kind) {
        CudaSessionDnnRnn result;
        result.kind = kind;
        result.handle = handle;
        result.rnn = metadata->borrow_rnn(rnn).view;
        expect(result.rnn.object.native != nullptr, "atomic composite RNN lease");
        for (std::size_t i = 0; i != 4; ++i)
            if (i != 3 || kind != Kind::kBackwardWeights)
                result.descriptors[i] =
                    *metadata->borrow(descriptors[i], i < 2 ? CudaDnnObjectKind::kRnnData
                                                            : CudaDnnObjectKind::kTensor);
        std::copy_n(buffers.begin(), 16, result.buffers.begin());
        if (kind == Kind::kForward) {
            result.mode = CUDNN_FWD_MODE_TRAINING;
            for (const auto i : {3, 4, 7, 8, 11, 12}) result.buffers[i] = {};
        } else if (kind == Kind::kBackwardData) {
            for (const auto i : {1, 6, 10}) result.buffers[i] = {};
        } else {
            for (const auto i : {3, 4, 6, 7, 8, 9, 10, 11, 12}) result.buffers[i] = {};
            result.buffers[13] = buffers[16];
        }
        result.weight_bytes = bytes[13];
        result.workspace_bytes = bytes[14];
        result.reserve_bytes = bytes[15];
        return result;
    }
    CudaDnnRnnAcceptance accept(CudaSessionDnnRnn value) {
        fixture::Invocation expected;
        expected.kind = value.kind;
        expected.mode = value.mode;
        expected.descriptors[0] = value.rnn.object.native;
        for (std::size_t i = 0; i != value.descriptors.size(); ++i)
            expected.descriptors[i + 1] = value.descriptors[i].native;
        for (std::size_t i = 0; i != value.buffers.size(); ++i)
            if (value.buffers[i].allocation) {
                const auto memory = session->device_memory(value.buffers[i], 0);
                expect(memory.has_value(), "snapshot actual native ABI allocation identity");
                expected.buffers[i] = reinterpret_cast<const void*>(memory->address);
            }
        expected.bytes = {value.weight_bytes, value.workspace_bytes, value.reserve_bytes};
        const auto result = session->accept_dnn_rnn(next++, value);
        expect(result.native.acceptance.accepted() && result.result && !result.result->is_ready,
               "admission produces pending capsule without invoking native compute");
        expected_invocations.emplace(result.native.acceptance.receipt.sequence(), expected);
        return result;
    }
    void issue(const CudaDnnRnnAcceptance& accepted, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
        const auto result = session->issue_next();
        const auto host = session->host_result(accepted.native.acceptance.receipt);
        expect(result.state == CudaExecutionIssueState::kIssued &&
                   result.library_status == status && host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                   host->status == status && accepted.result->is_ready,
               "one original invocation publishes matching host status and completion capsule");
        const auto& expected =
            expected_invocations.at(accepted.native.acceptance.receipt.sequence());
        expect(fixture::state.last.kind == expected.kind &&
                   fixture::state.last.mode == expected.mode &&
                   fixture::state.last.descriptors == expected.descriptors &&
                   fixture::state.last.buffers == expected.buffers &&
                   fixture::state.last.bytes == expected.bytes,
               "every positional descriptor, buffer, unused null, mode and capacity reaches "
               "original ABI");
        expected_invocations.erase(accepted.native.acceptance.receipt.sequence());
        expect(session->acknowledge_result(accepted.native.acceptance.receipt), "host ACK");
    }
    void retire() {
        foundation::control.can_stop.store(true);
        expect(session->poll().retired == 1, "physical event retires one original invocation");
    }
    void run(Kind kind) {
        const auto accepted = accept(problem(kind));
        issue(accepted);
        native(accepted.result->result);
        retire();
    }
    void close(unsigned int expected_handles = 1) {
        fixture::state.callback = {};
        foundation::control.can_stop.store(true);
        for (const auto buffer : buffers)
            if (buffer.allocation) driver(session->release(buffer.allocation));
        native(metadata->shutdown());
        driver(session->shutdown());
        expect(foundation::driver.memory.empty() && foundation::driver.events.empty() &&
                   foundation::driver.streams.empty() && foundation::driver.contexts.empty() &&
                   fixture::base::state.handle_destroys == expected_handles,
               "all known original resources retire exactly once");
    }
    void guards() {
        for (std::size_t i = 0; i != buffers.size(); ++i) {
            const auto view = session->device_memory({buffers[i].allocation, 0}, bytes[i] + 32);
            const auto* p = reinterpret_cast<const std::byte*>(view->address);
            for (std::size_t j = 0; j != 16; ++j)
                expect(p[j] == std::byte{0xcd} && p[16 + bytes[i] + j] == std::byte{0xcd},
                       "complete allocation guards remain intact");
        }
    }
};
template <typename T>
void numerical(unsigned int layout, unsigned int algorithm, bool default_stream) {
    Owner owner(sizeof(T) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT, layout, algorithm, 0,
                default_stream);
    native(owner.session->build_dnn_rnn(owner.handle, *owner.metadata, owner.rnn, 2).result);
    owner.run(Kind::kForward);
    double loss{};
    for (std::size_t t = 0; t != 3; ++t)
        for (std::size_t b = 0; b != 2; ++b) {
            if (layout == 1 && static_cast<int>(t) >= fixture::kLengths[b]) continue;
            const auto y = owner.memory<T>(2)[fixture::index(owner.data, t, b)];
            if (static_cast<int>(t) >= fixture::kLengths[b]) expect(y == 0, "padding is zero");
            loss += y * fixture::kDy[t * 2 + b];
        }
    for (std::size_t b = 0; b != 2; ++b)
        loss += owner.memory<T>(6)[b] * fixture::kDhidden[b] +
                owner.memory<T>(10)[b] * fixture::kDcell[b];
    expect(std::abs(loss - fixture::loss(fixture::kInput, fixture::kWeight, fixture::kHidden,
                                         fixture::kCell)) < 2e-6,
           "forward full loss matches independent forward-only oracle");
    owner.run(Kind::kBackwardData);
    owner.run(Kind::kBackwardWeights);
    owner.run(Kind::kBackwardWeights);
    const auto tolerance = sizeof(T) == 8 ? 2e-8 : 3e-6;
    for (std::size_t t = 0; t != 3; ++t)
        for (std::size_t b = 0; b != 2; ++b) {
            if (layout == 1 && static_cast<int>(t) >= fixture::kLengths[b]) continue;
            expect(std::abs(owner.memory<T>(4)[fixture::index(owner.data, t, b)] -
                            fixture::derivative(0, t * 2 + b)) < tolerance,
                   "every dx including padding matches central difference");
        }
    for (std::size_t i = 0; i != 12; ++i)
        expect(
            std::abs(owner.memory<T>(16)[i] - (.125 + 2 * fixture::derivative(1, i))) < tolerance,
            "all twelve ADD weight gradients and initial values match finite differences");
    for (std::size_t i = 0; i != 2; ++i)
        expect(std::abs(owner.memory<T>(8)[i] - fixture::derivative(2, i)) < tolerance &&
                   std::abs(owner.memory<T>(12)[i] - fixture::derivative(3, i)) < tolerance,
               "complete hidden and cell gradients match finite differences");
    owner.guards();
    expect(!fixture::control.syncs.load() && fixture::control.records.load() == 4,
           "successful compute stays asynchronous and records exactly one event each");
    owner.close();
    expect(fixture::state.unloads == 1, "dynamic library ownership unloads once after leases");
}
void controls() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    auto value = owner.problem(Kind::kForward);
    value.buffers[0] = {};
    for (const auto slot : {5, 6, 9, 10}) value.buffers[slot] = {};
    const auto gets = fixture::control.tensor_gets.load();
    const auto accepted = owner.accept(value);
    value = {};
    owner.issue(accepted);
    expect(fixture::control.tensor_gets.load() == gets && !fixture::state.last.buffers[0] &&
               fixture::state.last.descriptors[3] && fixture::state.last.descriptors[4],
           "unused supplied h/c identities stay pinned without authoritative tensor getters");
    owner.retire();
    const auto before = fixture::calls();
    for (unsigned int bad = 0; bad != 4; ++bad) {
        value = owner.problem(Kind::kForward);
        if (bad == 0) value.buffers[0].offset += owner.bytes[0] + 31;
        if (bad == 1) value.weight_bytes = SIZE_MAX;
        if (bad == 2) value.mode = UINT32_MAX;
        if (bad == 3) value.buffers[3] = owner.buffers[3];
        const auto rejected = owner.session->accept_dnn_rnn(owner.next, value);
        expect(!rejected.native.acceptance.accepted() && !rejected.result &&
                   !rejected.native.original_status && fixture::calls() == before,
               "unowned extent, overflow, bad enum and noncanonical unused range reject preissue");
    }
    value = {};
    fixture::control.status.store(CUDNN_STATUS_EXECUTION_FAILED);
    const auto failed = owner.accept(owner.problem(Kind::kForward));
    owner.issue(failed, CUDNN_STATUS_EXECUTION_FAILED);
    native(failed.result->result, CUDNN_STATUS_EXECUTION_FAILED);
    expect(fixture::control.syncs.load() == 1 && !failed.result->is_cleanup_uncertain &&
               !owner.metadata->snapshot().is_quarantined &&
               !owner.session->snapshot().is_quarantined &&
               *owner.memory<std::byte>(14) == std::byte{0x5a},
           "issued partial error retains status and proves all streams without poisoning metadata");
    owner.retire();
    fixture::control.status.store(CUDNN_STATUS_SUCCESS);
    owner.run(Kind::kForward);
    owner.close();
}
void lifecycle() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    const auto first = owner.accept(owner.problem(Kind::kForward));
    const auto second = owner.accept(owner.problem(Kind::kForward));
    expect(owner.metadata->set_rnn(owner.rnn, owner.parent).disposition ==
                   CudaDnnDisposition::kDeferred &&
               !owner.metadata->begin_dropout_mutation(owner.dropout).mutation &&
               owner.session->build_dnn_rnn(owner.handle, *owner.metadata, owner.rnn, 2)
                       .result.disposition == CudaDnnDisposition::kDeferred,
           "queued composite views retain parent and dropout and exclude population");
    unsigned int reentries{};
    fixture::state.callback = [&](unsigned int phase) {
        if (phase != 2) return;
        ++reentries;
        const auto calls = fixture::calls();
        (void)owner.session->issue_next();
        expect(
            fixture::calls() == calls && owner.session->poll().retired == 0 &&
                owner.session->drain_issued().retired == 0 &&
                owner.session->destroy_dnn(owner.handle).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->set_dnn_stream(owner.handle, 0).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred &&
                owner.metadata->shutdown().disposition == CudaDnnDisposition::kDeferred,
            "native callback cannot recursively consume queued command or release physical owners");
    };
    foundation::control.can_stop.store(false);
    owner.issue(first);
    owner.issue(second);
    fixture::state.callback = {};
    expect(reentries == 2 && fixture::control.calls[0].load() == 2 &&
               owner.session->poll().retired == 0,
           "two host results do not prove physical retirement");
    expect(owner.session->release(owner.buffers[1].allocation).disposition ==
                   CudaSessionDisposition::kDeferred &&
               owner.metadata->destroy(owner.descriptors[0]).disposition ==
                   CudaDnnDisposition::kDeferred,
           "acknowledged original command still retains device allocation and immutable metadata");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 2, "both exact native events retire");
    for (const auto buffer : owner.buffers)
        if (buffer.allocation) driver(owner.session->release(buffer.allocation));
    native(owner.metadata->shutdown());
    driver(owner.session->shutdown());
}
void cancellation() {
    Owner owner;
    const auto accepted = owner.accept(owner.problem(Kind::kForward));
    expect(owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
           "cancel accepted unissued RNN");
    const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
    expect(host && host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
               !accepted.result->is_ready && !fixture::calls() && !fixture::control.syncs.load() &&
               !fixture::control.records.load(),
           "cancellation fabricates no original SDK status or proof");
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt), "cancel ACK");
    owner.close();
}
void restore_dropout(Owner& owner) {
    auto mutation = owner.metadata->begin_dropout_mutation(owner.dropout);
    expect(mutation.mutation != nullptr, "reserve dependent dropout mutation before RNN admission");
    CudaSessionDnnDropout problem;
    problem.kind = CudaDnnDropoutKind::kRestore;
    problem.handle = owner.handle;
    problem.mutation = std::move(mutation.mutation);
    // Forward never consumes the separate weight-gradient output. Use its owned
    // allocation for RNG state so retaining it cannot be explained by RNN's ABI.
    problem.states = owner.buffers[16];
    problem.state_bytes = fixture::dropout::kStateBytes;
    problem.probability_bits = 0x3e800000U;
    problem.seed = 0xfedcba9876543210ULL;
    const auto accepted = owner.session->accept_dnn_dropout(owner.next++, problem);
    expect(accepted.acceptance.accepted(), "admit real original Restore through queued mutation");
    problem = {};
    const auto issued = owner.session->issue_next();
    const auto host = owner.session->host_result(accepted.acceptance.receipt);
    expect(issued.library_status == CUDNN_STATUS_SUCCESS && host &&
               host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
               host->status == CUDNN_STATUS_SUCCESS && fixture::dropout::state.restores == 1 &&
               !fixture::dropout::state.initializations &&
               fixture::dropout::state.last.seed == 0xfedcba9876543210ULL,
           "Restore preserves seed and does not replay RNG initialization");
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt), "Restore ACK");
    owner.retire();
}
void dropout_lifetime() {
    Owner owner;
    restore_dropout(owner);
    const auto* state = owner.memory<std::byte>(16);
    const std::vector<std::byte> before(state, state + owner.bytes[16]);
    const auto accepted = owner.accept(owner.problem(Kind::kForward));
    foundation::control.can_stop.store(false);
    expect(owner.session->release(owner.buffers[16].allocation).disposition ==
               CudaSessionDisposition::kDeferred,
           "queued RNN retains transitive restored RNG allocation not present in ABI buffers");
    owner.issue(accepted);
    expect(
        !owner.session->poll().retired &&
            owner.session->release(owner.buffers[16].allocation).disposition ==
                CudaSessionDisposition::kDeferred &&
            !owner.metadata->begin_dropout_mutation(owner.dropout).mutation &&
            std::equal(before.begin(), before.end(), state),
        "host ACK retains RNG state and descriptor until physical event without reinitialization");
    owner.retire();
    driver(owner.session->release(owner.buffers[16].allocation));
    owner.buffers[16] = {};
    const auto calls = fixture::calls();
    const auto rejected = owner.session->accept_dnn_rnn(owner.next, owner.problem(Kind::kForward));
    expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status &&
               fixture::calls() == calls && !owner.metadata->snapshot().is_quarantined &&
               !owner.session->snapshot().is_quarantined,
           "freed restored binding rejects future RNN before SDK without poisoning healthy owners");
    owner.close();
}
void dropout_incarnation() {
    Owner owner;
    restore_dropout(owner);
    driver(owner.session->deactivate());
    CudaExecutionSession other({81, 82}, 83, owner.sdk);
    driver(other.initialize(0));
    const auto handle = other.create_dnn();
    native(handle);
    driver(other.create_stream(CU_STREAM_NON_BLOCKING));
    std::array<CudaAllocationRange, 17> ranges{};
    for (std::size_t i = 0; i != ranges.size(); ++i) {
        const auto allocation = other.allocate(owner.bytes[i] + 32);
        driver(allocation);
        expect(allocation.id == owner.buffers[i].allocation,
               "independent session deliberately reuses every public allocation number");
        ranges[i] = {allocation.id, 16};
    }
    auto problem = owner.problem(Kind::kForward);
    problem.handle = handle.id;
    const auto calls = fixture::calls();
    const auto rejected = other.accept_dnn_rnn(1, problem);
    problem = {};
    expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status &&
               fixture::calls() == calls && !other.snapshot().is_quarantined &&
               !owner.metadata->snapshot().is_quarantined,
           "identical public session/context/allocation IDs cannot impersonate RNG incarnation");
    for (const auto range : ranges) driver(other.release(range.allocation));
    driver(other.shutdown());
    driver(owner.session->activate());
    owner.run(Kind::kForward);
    owner.close(2);
}
auto sdk_observations() {
    return std::array{fixture::calls(),
                      fixture::control.gets.load(),
                      fixture::control.tensor_gets.load(),
                      fixture::control.sizes.load(),
                      fixture::control.records.load(),
                      fixture::control.polls.load(),
                      fixture::control.syncs.load(),
                      fixture::control.context_gets.load(),
                      fixture::control.context_sets.load(),
                      fixture::control.stream_gets.load(),
                      fixture::control.stream_syncs.load()};
}
void callback_reentry() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    std::array<unsigned int, 6> phases{};
    fixture::state.callback = [&](unsigned int phase) {
        expect(phase < phases.size(), "known native callback phase");
        ++phases[phase];
        const auto before = sdk_observations();
        const auto issue = owner.session->issue_next();
        const auto poll = owner.session->poll();
        const auto drain = owner.session->drain_issued();
        const auto activate = owner.session->activate();
        const auto deactivate = owner.session->deactivate();
        const auto stream = owner.session->get_dnn_stream(owner.handle);
        expect(issue.native_status == CUDA_ERROR_NOT_READY && poll.status == CUDA_ERROR_NOT_READY &&
                   !poll.retired && drain.status == CUDA_ERROR_NOT_READY && !drain.retired &&
                   activate.disposition == CudaSessionDisposition::kDeferred &&
                   activate.status == CUDA_ERROR_NOT_READY &&
                   deactivate.disposition == CudaSessionDisposition::kDeferred &&
                   deactivate.status == CUDA_ERROR_NOT_READY &&
                   stream.disposition == CudaDnnDisposition::kDeferred &&
                   stream.status == CUDA_ERROR_NOT_READY && !stream.original_status &&
                   sdk_observations() == before && foundation::current() == owner.context,
               "callback issue/poll/drain/context/stream reentry performs zero nested SDK calls");
    };
    native(owner.session->build_dnn_rnn(owner.handle, *owner.metadata, owner.rnn, 2).result);
    fixture::control.status.store(CUDNN_STATUS_EXECUTION_FAILED);
    const auto accepted = owner.accept(owner.problem(Kind::kForward));
    owner.issue(accepted, CUDNN_STATUS_EXECUTION_FAILED);
    fixture::state.callback = {};
    for (unsigned int phase = 1; phase != phases.size(); ++phase)
        expect(phases[phase] != 0, "exercise getter, compute, Build, stream and proof callbacks");
    owner.retire();
    owner.close();
}
void event_record_failure() {
    Owner owner;
    const auto accepted = owner.accept(owner.problem(Kind::kForward));
    fixture::control.event_status.store(CUDA_ERROR_LAUNCH_FAILED);
    owner.issue(accepted);
    native(accepted.result->result);
    const auto polls = fixture::control.polls.load();
    const auto frees = foundation::driver.frees;
    foundation::control.can_stop.store(true);
    const auto polled = owner.session->poll();
    expect(polled.retired == 0 && polled.retained == 1 &&
               polled.status == CUDA_ERROR_LAUNCH_FAILED &&
               fixture::control.polls.load() == polls && foundation::driver.frees == frees &&
               owner.session->release(owner.buffers[1].allocation).status != CUDA_SUCCESS &&
               owner.metadata->set_rnn(owner.rnn, owner.parent).disposition ==
                   CudaDnnDisposition::kDeferred,
           "failed trailing record cannot become physical proof through an unrecorded event");
    const auto drained = owner.session->drain_issued();
    expect(drained.retired == 1 && fixture::control.stream_syncs.load() == 1,
           "explicit successful exact-stream drain may recover record failure ownership");
    native(owner.metadata->shutdown());
    driver(owner.session->shutdown());
    expect(foundation::driver.memory.empty() && foundation::driver.contexts.empty(),
           "quarantined queue permits owned context teardown only after proven physical drain");
}
void overlapping_state_extents() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    fixture::control.status.store(CUDNN_STATUS_BAD_PARAM);
    // The independent native size fixture reports the strided byte hull: four
    // floats here. An RNN's native packed-state access may cover all eight
    // logical elements despite overlapping strides, so its owned bound is 32.
    const CudaDnnTensor overlapping{CUDNN_DATA_FLOAT, 0, 3, {2, 2, 2}, {1, 1, 1}};
    for (const auto kind : {Kind::kForward, Kind::kBackwardData, Kind::kBackwardWeights}) {
        for (const auto slot : {std::size_t{5}, std::size_t{9}}) {
            if (kind == Kind::kBackwardWeights && slot == 9) continue;
            const auto descriptor = owner.descriptors[slot == 5 ? 2 : 3];
            native(owner.metadata->set_tensor(descriptor, CudaDnnTensorSetter::kNd, overlapping));
            auto problem = owner.problem(kind);
            for (std::size_t i = 5; i != 13; ++i) problem.buffers[i] = {};
            problem.buffers[slot] = {owner.buffers[slot].allocation, owner.bytes[slot] + 32 - 16};
            const auto calls = fixture::calls();
            const auto rejected = owner.session->accept_dnn_rnn(owner.next, problem);
            expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status &&
                       rejected.native.rejection_status == CUDA_ERROR_INVALID_VALUE &&
                       fixture::calls() == calls && !owner.metadata->snapshot().is_quarantined,
                   "exact hull-sized state range cannot authorize a larger dense native footprint");
            const auto dense = owner.session->allocate(32);
            driver(dense);
            problem.buffers[slot] = {dense.id, 0};
            const auto accepted = owner.accept(problem);
            problem = {};
            owner.issue(accepted, CUDNN_STATUS_BAD_PARAM);
            native(accepted.result->result, CUDNN_STATUS_BAD_PARAM);
            expect(!accepted.result->is_cleanup_uncertain &&
                       !owner.metadata->snapshot().is_quarantined,
                   "adequately owned overlapping state preserves the native legality decision");
            owner.retire();
            driver(owner.session->release(dense.id));
        }
    }
    owner.close();
}
void state_rank_bounds() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    fixture::control.status.store(CUDNN_STATUS_BAD_PARAM);
    for (const auto slot : {std::size_t{5}, std::size_t{9}}) {
        const auto descriptor = owner.descriptors[slot == 5 ? 2 : 3];
        for (const int rank : {1, 2, 4}) {
            CudaDnnTensor tensor{CUDNN_DATA_FLOAT, 0, rank, {1, 1, 1, 1}, {1, 1, 1, 1}};
            native(owner.metadata->set_tensor(descriptor, CudaDnnTensorSetter::kNd, tensor));
            auto problem = owner.problem(Kind::kForward);
            for (std::size_t i = 5; i != 13; ++i) problem.buffers[i] = {};
            problem.buffers[slot] = owner.buffers[slot];
            if (rank < 3) {
                const auto calls = fixture::calls();
                const auto rejected = owner.session->accept_dnn_rnn(owner.next, problem);
                expect(!rejected.native.acceptance.accepted() &&
                           rejected.native.rejection_status == CUDA_ERROR_INVALID_VALUE &&
                           !rejected.native.original_status && fixture::calls() == calls &&
                           !owner.metadata->snapshot().is_quarantined,
                       "active rank-one/two state cannot reach native fixed three-axis reads");
                problem.buffers[slot] = {};
            }
            const auto gets = fixture::control.tensor_gets.load();
            const auto accepted = owner.accept(problem);
            problem = {};
            owner.issue(accepted, CUDNN_STATUS_BAD_PARAM);
            native(accepted.result->result, CUDNN_STATUS_BAD_PARAM);
            expect(
                fixture::control.tensor_gets.load() == gets + (rank >= 3 ? 1U : 0U),
                "unused low-rank identity skips getters; active rank-four reaches native legality");
            owner.retire();
        }
    }
    owner.close();
}
void mixed_state_widths() {
    for (unsigned int test = 0; test != 3; ++test) {
        Owner owner(test == 1 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT);
        fixture::control.arithmetic.store(false);
        fixture::control.status.store(CUDNN_STATUS_BAD_PARAM);
        CudaAllocationRange wide_y{};
        if (test == 2) {
            auto output = owner.data;
            output.type = CUDNN_DATA_DOUBLE;
            output.padding_bytes = sizeof(double);
            native(owner.metadata->set_rnn_data(owner.descriptors[1], output));
            const auto allocation = owner.session->allocate(fixture::kCells * sizeof(double));
            driver(allocation);
            wide_y = {allocation.id, 0};
        }
        const std::size_t required = test == 0 ? 8 : 16;
        for (const auto slot : {std::size_t{5}, std::size_t{9}}) {
            native(owner.metadata->set_tensor(owner.descriptors[slot == 5 ? 2 : 3],
                                              CudaDnnTensorSetter::kNd,
                                              {CUDNN_DATA_HALF, 0, 3, {1, 2, 1}, {2, 1, 1}}));
            auto problem = owner.problem(Kind::kForward);
            for (std::size_t i = 5; i != 13; ++i) problem.buffers[i] = {};
            problem.buffers[slot] = {owner.buffers[slot].allocation, owner.bytes[slot] + 32 - 4};
            if (wide_y.allocation) problem.buffers[2] = wide_y;
            const auto calls = fixture::calls();
            const auto rejected = owner.session->accept_dnn_rnn(owner.next, problem);
            expect(!rejected.native.acceptance.accepted() &&
                       rejected.native.rejection_status == CUDA_ERROR_INVALID_VALUE &&
                       !rejected.native.original_status && fixture::calls() == calls &&
                       !owner.metadata->snapshot().is_quarantined,
                   "HALF state native size cannot authorize larger x/y compute-width access");
            const auto allocation = owner.session->allocate(required);
            driver(allocation);
            problem.buffers[slot] = {allocation.id, 0};
            const auto accepted = owner.accept(problem);
            problem = {};
            owner.issue(accepted, CUDNN_STATUS_BAD_PARAM);
            native(accepted.result->result, CUDNN_STATUS_BAD_PARAM);
            expect(
                !owner.metadata->snapshot().is_quarantined,
                "HALF native hull is validated in HALF bytes; adequate compute bound reaches SDK");
            owner.retire();
            driver(owner.session->release(allocation.id));
        }
        if (wide_y.allocation) driver(owner.session->release(wide_y.allocation));
        owner.close();
    }
}
void state_dense_overflow() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    const int large = std::numeric_limits<int>::max();
    native(owner.metadata->set_tensor(owner.descriptors[2], CudaDnnTensorSetter::kNd,
                                      {CUDNN_DATA_FLOAT, 0, 3, {large, large, large}, {1, 1, 1}}));
    auto problem = owner.problem(Kind::kForward);
    for (std::size_t i = 5; i != 13; ++i) problem.buffers[i] = {};
    problem.buffers[5] = owner.buffers[5];
    const auto sizes = fixture::control.sizes.load();
    const auto rejected = owner.session->accept_dnn_rnn(owner.next, problem);
    problem = {};
    expect(!rejected.native.acceptance.accepted() &&
               rejected.native.rejection_status == CUDA_ERROR_INVALID_VALUE &&
               !rejected.native.original_status && !fixture::calls() &&
               fixture::control.sizes.load() == sizes && !owner.metadata->snapshot().is_quarantined,
           "overflowing dense product rejects before native size callback without wrapping");
    owner.close();
}
void missing() {
    for (unsigned int missing = 1; missing != 6; ++missing) {
        Owner owner(0, 0, 0, missing);
        if (missing == 4) {
            const auto result =
                owner.session->build_dnn_rnn(owner.handle, *owner.metadata, owner.rnn, 2);
            expect(
                result.result.status == CUDA_ERROR_NOT_SUPPORTED && !result.result.original_status,
                "missing Build is clean policy error");
        } else {
            const auto kind = missing < 4 ? static_cast<Kind>(missing) : Kind::kForward;
            const auto result = owner.session->accept_dnn_rnn(owner.next, owner.problem(kind));
            expect(!result.native.acceptance.accepted() && !result.result &&
                       result.native.rejection_status == CUDA_ERROR_NOT_SUPPORTED,
                   "missing actual callback or authoritative preflight rejects cleanly");
        }
        expect(!fixture::calls() && !owner.metadata->snapshot().is_quarantined,
               "unavailable callbacks never execute native or poison metadata");
        owner.close();
    }
}
void poison(unsigned int test) {
    Owner owner;
    fixture::control.arithmetic.store(false);
    if (test < 7) {
        const bool native_error = test == 1 || test == 4 || test == 6;
        fixture::control.status.store(native_error ? CUDNN_STATUS_EXECUTION_FAILED
                                                   : CUDNN_STATUS_SUCCESS);
        const auto accepted = owner.accept(owner.problem(Kind::kForward));
        if (test == 0) fixture::control.context_loss.store(2);
        if (test == 1) fixture::control.sync_status.store(CUDA_ERROR_LAUNCH_FAILED);
        if (test == 2 || test == 6) fixture::control.stream_change.store(1);
        if (test == 3) fixture::control.stream_status.store(CUDNN_STATUS_BAD_PARAM);
        if (test == 4) fixture::control.context_loss.store(5);
        if (test == 5) fixture::control.context_loss.store(4);
        owner.issue(accepted, native_error ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS);
        expect(
            accepted.result->result.disposition == CudaDnnDisposition::kRejected &&
                accepted.result->result.original_status ==
                    (native_error ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS) &&
                accepted.result->has_context_loss && accepted.result->is_cleanup_uncertain &&
                !accepted.result->has_metadata_quarantine && !fixture::control.records.load(),
            "failed physical/stream proof preserves provenance and permanently retains resources");
        foundation::control.can_stop.store(true);
        (void)owner.session->poll();
        (void)owner.session->drain_issued();
        (void)owner.session->shutdown();
        (void)owner.metadata->shutdown();
        expect(
            !foundation::driver.memory.empty() && !fixture::base::state.handle_destroys &&
                fixture::metadata::state.destroys == 0,
            "unrelated event success and drain cannot rescue failed proof or clean native owners");
    } else if (test < 9) {
        fixture::control.corrupt.store(static_cast<int>(test - 6));
        const auto result =
            owner.session->accept_dnn_rnn(owner.next, owner.problem(Kind::kForward));
        expect(!result.native.acceptance.accepted() && !fixture::calls() &&
                   owner.metadata->snapshot().is_quarantined && !result.native.original_status,
               "malformed authoritative getter cannot acquire invented original compute status");
    } else if (test < 11) {
        fixture::control.build_status.store(CUDNN_STATUS_EXECUTION_FAILED);
        if (test == 10) fixture::control.context_loss.store(3);
        const auto result =
            owner.session->build_dnn_rnn(owner.handle, *owner.metadata, owner.rnn, 2);
        expect(
            result.result.original_status == CUDNN_STATUS_EXECUTION_FAILED &&
                owner.metadata->snapshot().is_quarantined &&
                result.has_context_loss == (test == 10) && fixture::state.built.size() == 1,
            "partially populated failed Build quarantines metadata with exact native provenance");
    } else {
        auto problem = owner.problem(Kind::kForward);
        const auto lease = problem.rnn.lease;
        const auto accepted = owner.accept(std::move(problem));
        if (test == 11)
            lease->quarantine();
        else
            fixture::control.event_context_loss.store(true);
        const auto issued = owner.session->issue_next();
        const auto host = owner.session->host_result(accepted.native.acceptance.receipt);
        expect(issued.state == CudaExecutionIssueState::kRejected && !issued.library_status &&
                   host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   host->status != CUDA_SUCCESS && accepted.result->is_ready &&
                   !accepted.result->result.original_status &&
                   accepted.result->result.disposition == CudaDnnDisposition::kRejected &&
                   accepted.result->is_cleanup_uncertain && !fixture::calls() &&
                   !fixture::control.records.load() && !fixture::control.syncs.load(),
               "queued lease/context pre-invoke failure publishes preparation rejection, not SDK "
               "provenance");
        expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt),
               "acknowledge pre-invoke rejection without releasing uncertain owners");
        foundation::driver.current = owner.context;
        foundation::control.can_stop.store(true);
        expect(!owner.session->poll().retired && !owner.session->drain_issued().retired &&
                   !foundation::driver.memory.empty() && !fixture::base::state.handle_destroys,
               "later restored context/event success cannot guess away uncertain pre-invoke "
               "ownership");
    }
}
}  // namespace
int main() {
    try {
        for (unsigned int algorithm = 0; algorithm != 4; ++algorithm)
            for (unsigned int layout = 0; layout != 3; ++layout) {
                numerical<float>(layout, algorithm, false);
                numerical<double>(layout, algorithm, true);
            }
        controls();
        lifecycle();
        cancellation();
        missing();
        dropout_lifetime();
        dropout_incarnation();
        callback_reentry();
        event_record_failure();
        overlapping_state_extents();
        state_rank_bounds();
        mixed_state_widths();
        state_dense_overflow();
        for (unsigned int test = 0; test != 13; ++test) {
            const auto child = fork();
            expect(child >= 0, "isolated quarantine fork");
            if (!child) {
                try {
                    poison(test);
                    _exit(0);
                } catch (const std::exception& error) {
                    std::cerr << test << ": " << error.what() << '\n';
                    _exit(1);
                }
            }
            int status{};
            expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                   "isolated uncertain native ownership check");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
