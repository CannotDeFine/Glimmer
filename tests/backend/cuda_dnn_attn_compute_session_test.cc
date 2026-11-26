#include "dnn_attn_compute_fixture.h"

#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_attn_compute;
namespace foundation = fixture::foundation;
namespace oracle = fixture::oracle;
using Kind = fixture::Kind;
std::function<void()> on_event_cleanup;
decltype(CudaExecutionDispatch{}.event_destroy) original_event_destroy{};
void expect(bool value, const char* text) {
    if (!value) throw std::runtime_error(text);
}
void driver(const CudaSessionResult& value) {
    expect(value.disposition == CudaSessionDisposition::kComplete && value.status == CUDA_SUCCESS,
           "owned Driver result");
}
void native(const CudaDnnResult& value, cudnnStatus_t expected = CUDNN_STATUS_SUCCESS) {
    expect(value.disposition == CudaDnnDisposition::kComplete && value.status == CUDA_SUCCESS &&
               value.original_status == expected,
           "exact original attention status");
}
void near(double value, double expected, double tolerance) {
    expect(std::isfinite(value) && std::abs(value - expected) < tolerance,
           "complete projected-attention numerical output");
}
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> registry;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t parent{}, handle{}, stream{}, next{1};
    std::array<std::uint64_t, 2> dropouts{};
    std::array<std::uint64_t, 4> descriptors{};
    std::array<CudaDnnSeqDataState, 4> data;
    CudaDnnAttnState config;
    std::array<CudaAllocationRange, 15> buffers{};
    std::array<std::size_t, 15> bytes{};
    std::vector<std::uint64_t> extra;
    std::size_t scalar;
    CUcontext context{};
    explicit Owner(unsigned int type = CUDNN_DATA_FLOAT, unsigned int axes = 0,
                   unsigned int missing = 0, bool default_stream = false,
                   bool observe_cleanup = false)
        : scalar(type == CUDNN_DATA_DOUBLE ? 8 : 4) {
        fixture::reset();
        sdk = fixture::session_dispatch(missing);
        if (observe_cleanup) {
            original_event_destroy = sdk.execution.event_destroy;
            sdk.execution.event_destroy = +[](CUevent event) {
                if (on_event_cleanup) on_event_cleanup();
                return original_event_destroy(event);
            };
        }
        registry = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        parent = create(CudaDnnObjectKind::kAttn);
        for (auto& id : dropouts) id = create(CudaDnnObjectKind::kDropout);
        config.mode = CUDNN_ATTN_ENABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ONE_TO_ONE;
        config.heads = 1;
        config.scale = .75;
        config.type = config.precision = type;
        config.dropouts = dropouts;
        config.query_size = config.key_size = config.value_size = 2;
        config.query_projection = config.key_projection = config.value_projection =
            config.output_projection = 2;
        config.max_query_sequence = 4;
        config.max_key_sequence = 5;
        config.max_batch = config.max_beam = 3;
        native(registry->set_attn(parent, config));
        for (std::size_t i = 0; i != 4; ++i) {
            descriptors[i] = create(CudaDnnObjectKind::kSeqData);
            const auto time = i == 1 || i == 2 ? 3 : 2;
            data[i] = {.type = type,
                       .presence = 7,
                       .dimensions = 4,
                       .sizes = {time, 2, 2, 2},
                       .axes = fixture::kAxes[axes],
                       .sequence_count = 4,
                       .lengths = std::vector<int>(4, time)};
            native(registry->set_seq_data(descriptors[i], data[i]));
        }
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{291, 292}, 293, sdk);
        driver(session->initialize(0));
        context = foundation::current();
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
        if (!default_stream) {
            const auto created_stream = session->create_stream(CU_STREAM_NON_BLOCKING);
            driver(created_stream);
            stream = created_stream.id;
            native(session->set_dnn_stream(handle, stream));
        }
        bytes = {16,          16,          16 * scalar, 24 * scalar, 24 * scalar,
                 16 * scalar, 16 * scalar, 16 * scalar, 16 * scalar, 24 * scalar,
                 24 * scalar, 24 * scalar, 24 * scalar, 32,          fixture::kReserve * scalar};
        for (std::size_t i = 0; i != 15; ++i) {
            const auto allocation = session->allocate(bytes[i] + 32);
            driver(allocation);
            buffers[i] = {allocation.id, 16};
            std::memset(reinterpret_cast<void*>(allocation.address), 0xa5, bytes[i] + 32);
            std::memset(reinterpret_cast<void*>(allocation.address + 16), 0, bytes[i]);
        }
        if (scalar == 8)
            initialize<double>();
        else
            initialize<float>();
    }
    std::uint64_t create(CudaDnnObjectKind kind) {
        const auto value = registry->create(kind);
        native(value);
        return value.id;
    }
    template <typename T>
    T* memory(std::size_t i) {
        const auto value = session->device_memory(buffers[i], bytes[i]);
        expect(bool(value), "owned fake memory");
        return reinterpret_cast<T*>(value->address);
    }
    template <typename T>
    void initialize() {
        std::fill_n(memory<int>(0), 4, 2);
        std::fill_n(memory<int>(1), 4, 3);
        const auto weights = fixture::flatten(oracle::seeded_parameters());
        for (std::size_t i = 0; i != 24; ++i) {
            memory<T>(11)[i] = static_cast<T>(weights[i]);
            memory<T>(12)[i] = T{.125};
        }
        for (std::size_t b = 0; b != 2; ++b)
            for (std::size_t beam = 0; beam != 2; ++beam) {
                const auto input = fixture::inputs(b, beam);
                const auto derivative = fixture::upstream(b, beam);
                for (std::size_t t = 0; t != 2; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        memory<T>(2)[fixture::index(data[0], t, b, beam, c)] =
                            static_cast<T>(input.queries[2 * t + c]);
                        memory<T>(5)[fixture::index(data[0], t, b, beam, c)] =
                            static_cast<T>(input.residual[2 * t + c]);
                        memory<T>(7)[fixture::index(data[3], t, b, beam, c)] =
                            static_cast<T>(derivative.out[2 * t + c]);
                    }
                for (std::size_t t = 0; t != 3; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        memory<T>(3)[fixture::index(data[1], t, b, beam, c)] =
                            static_cast<T>(input.keys[2 * t + c]);
                        memory<T>(4)[fixture::index(data[2], t, b, beam, c)] =
                            static_cast<T>(input.values[2 * t + c]);
                    }
            }
    }
    CudaSessionDnnAttn problem(Kind kind, bool captured = true) {
        CudaSessionDnnAttn value;
        value.kind = kind;
        value.handle = handle;
        auto parent_view = registry->borrow_attn(parent);
        expect(parent_view.result.status == CUDA_SUCCESS, "borrow composite attention owner");
        value.attn = std::move(parent_view.view);
        for (std::size_t i = 0; i != 4; ++i)
            value.descriptors[i] = *registry->borrow(descriptors[i], CudaDnnObjectKind::kSeqData);
        value.buffers = buffers;
        value.weight_bytes = bytes[11];
        value.workspace_bytes = bytes[13];
        value.reserve_bytes = bytes[14];
        if (kind == Kind::kForward) {
            for (auto i : {7, 8, 9, 10, 12}) value.buffers[i] = {};
            value.current = -1;
        }
        if (kind == Kind::kBackwardData)
            for (auto i : {5, 6, 12}) value.buffers[i] = {};
        if (kind == Kind::kBackwardWeights)
            for (auto i : {0, 1, 5, 6, 8, 9, 10}) value.buffers[i] = {};
        else {
            value.window_presence = 3;
            if (captured) value.windows = {std::vector<int>(4, 0), std::vector<int>(4, 3)};
        }
        return value;
    }
    CudaDnnAttnAcceptance accept(CudaSessionDnnAttn value) {
        auto result = session->accept_dnn_attn(next, value);
        if (result.native.acceptance.accepted()) ++next;
        return result;
    }
    CudaDnnAttnAcceptance issue(CudaSessionDnnAttn value,
                                cudnnStatus_t expected = CUDNN_STATUS_SUCCESS, bool retire = true) {
        fixture::Invocation abi;
        abi.kind = value.kind;
        abi.descriptors[0] = value.attn.object.native;
        for (std::size_t i = 0; i != 4; ++i) abi.descriptors[i + 1] = value.descriptors[i].native;
        for (std::size_t i = 0; i != 15; ++i)
            if (value.buffers[i].allocation) {
                const auto memory = session->device_memory(value.buffers[i], 0);
                expect(bool(memory), "ABI pointer identity");
                abi.buffers[i] = reinterpret_cast<void*>(memory->address);
            }
        abi.bytes = {value.weight_bytes, value.workspace_bytes, value.reserve_bytes};
        abi.mode = value.mode;
        abi.current = value.current;
        const auto accepted = accept(std::move(value));
        expect(accepted.native.acceptance.accepted(), "admit owned attention command");
        const auto result = session->issue_next();
        expect(
            result.state == CudaExecutionIssueState::kIssued && result.library_status == expected,
            "one original SDK issue");
        expect(fixture::state.last.kind == abi.kind &&
                   fixture::state.last.descriptors == abi.descriptors &&
                   fixture::state.last.buffers == abi.buffers &&
                   fixture::state.last.bytes == abi.bytes && fixture::state.last.mode == abi.mode &&
                   fixture::state.last.current == abi.current,
               "all original positional ABI arguments preserved");
        expect(accepted.result && accepted.result->is_ready, "command completion capsule ready");
        native(accepted.result->result, expected);
        expect(session->acknowledge_result(accepted.native.acceptance.receipt), "host ACK");
        if (retire) expect(session->poll().retired == 1, "physical event retires one original");
        return accepted;
    }
    template <typename T>
    void verify(double accumulated, double initial) {
        std::array<double, 24> gradient{};
        const auto tolerance = scalar == 8 ? 1e-11 : 2e-5;
        for (std::size_t b = 0; b != 2; ++b)
            for (std::size_t beam = 0; beam != 2; ++beam) {
                const auto input = fixture::inputs(b, beam);
                const auto output = oracle::forward(oracle::seeded_parameters(), input);
                const auto result = oracle::backward(oracle::seeded_parameters(), input,
                                                     fixture::upstream(b, beam));
                const auto weights = fixture::flatten(result.parameters);
                for (std::size_t i = 0; i != 24; ++i) gradient[i] += weights[i];
                for (std::size_t t = 0; t != 2; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        near(memory<T>(6)[fixture::index(data[3], t, b, beam, c)],
                             output.out[2 * t + c], tolerance);
                        near(memory<T>(8)[fixture::index(data[0], t, b, beam, c)],
                             result.queries[2 * t + c], tolerance);
                    }
                for (std::size_t t = 0; t != 3; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        near(memory<T>(9)[fixture::index(data[1], t, b, beam, c)],
                             result.keys[2 * t + c], tolerance);
                        near(memory<T>(10)[fixture::index(data[2], t, b, beam, c)],
                             result.values[2 * t + c], tolerance);
                    }
            }
        for (std::size_t i = 0; i != 24; ++i)
            near(memory<T>(12)[i], initial + accumulated * gradient[i], tolerance);
    }
    void close() {
        for (std::size_t i = 0; i != 15; ++i) {
            const auto memory = session->device_memory({buffers[i].allocation, 0}, bytes[i] + 32);
            expect(bool(memory), "complete guarded owner");
            const auto* raw = reinterpret_cast<const std::byte*>(memory->address);
            for (std::size_t j = 0; j != 16; ++j)
                expect(raw[j] == std::byte{0xa5} && raw[16 + bytes[i] + j] == std::byte{0xa5},
                       "all allocation guards unchanged");
        }
        for (const auto buffer : buffers) driver(session->release(buffer.allocation));
        for (const auto id : extra) driver(session->release(id));
        native(session->destroy_dnn(handle));
        native(registry->shutdown());
        driver(session->shutdown());
        expect(fixture::metadata::state.destroys == 1 && fixture::dropout::state.destroys == 2 &&
                   fixture::sequence::state.destroys == fixture::sequence::state.creates &&
                   foundation::driver.memory.empty() && foundation::driver.events.empty() &&
                   foundation::driver.contexts.empty(),
               "all known original ownership retires exactly once");
    }
};
void numerical() {
    for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE})
        for (unsigned int layout = 0; layout != 6; ++layout) {
            Owner owner(type, layout, 0, layout == 0);
            for (int cycle = 0; cycle != 2; ++cycle) {
                auto prepare = owner.problem(Kind::kForward, false);
                const auto range = owner.session->prepare_dnn_attn(prepare);
                prepare = {};
                expect(range.result.disposition == CudaDnnDisposition::kComplete &&
                           range.result.status == CUDA_SUCCESS && !range.result.original_status &&
                           range.first == 0 && range.count == 4,
                       "full windows use parent maximum rather than actual two steps");
                owner.issue(owner.problem(Kind::kForward));
                owner.issue(owner.problem(Kind::kBackwardData));
                owner.issue(owner.problem(Kind::kBackwardWeights));
                if (type == CUDNN_DATA_DOUBLE)
                    owner.verify<double>(cycle + 1, .125);
                else
                    owner.verify<float>(cycle + 1, .125);
            }
            auto overwrite = owner.problem(Kind::kBackwardWeights);
            overwrite.mode = CUDNN_WGRAD_MODE_SET;
            owner.issue(std::move(overwrite));
            if (type == CUDNN_DATA_DOUBLE)
                owner.verify<double>(1, 0);
            else
                owner.verify<float>(1, 0);
            expect(!fixture::control.syncs.load(), "successful originals remain asynchronous");
            owner.close();
        }
}
void preparation() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    auto value = owner.problem(Kind::kForward, false);
    value.current = 1;
    value.buffers[14] = {};
    value.reserve_bytes = 0;
    auto result = owner.session->prepare_dnn_attn(value);
    expect(result.first == 1 && result.count == 1 && fixture::control.length_gets.load() == 1,
           "incremental singleton with bounded actual length observation");
    value.windows = {std::vector<int>{1}, std::vector<int>{2}};
    owner.issue(std::move(value));
    expect(fixture::state.observed_windows ==
               std::array<std::vector<int>, 2>{std::vector<int>{1}, std::vector<int>{2}},
           "reconstructed singleton preserves only selected caller element");
    for (unsigned int which = 0; which != 5; ++which) {
        value = owner.problem(Kind::kForward, false);
        if (which == 0) value.descriptors[1] = {};
        if (which == 1) value.buffers[2] = {};
        if (which == 2) value.window_presence = 1;
        if (which == 3) value.current = 2;
        if (which == 4) {
            value.buffers[14] = {};
            value.reserve_bytes = 16;
        }
        result = owner.session->prepare_dnn_attn(value);
        expect(result.result.disposition == CudaDnnDisposition::kComplete &&
                   !result.result.original_status && !result.first && !result.count,
               "proven early error captures no host window bytes");
        owner.issue(std::move(value), CUDNN_STATUS_BAD_PARAM);
    }
    value = owner.problem(Kind::kForward, false);
    value.buffers[14] = {};
    value.reserve_bytes = 0;
    value.current = 1;
    for (auto& data : owner.data)
        if (data.sizes[0] == 2) data.lengths.assign(4, 1);
    value = {};
    for (std::size_t i = 0; i != 4; ++i)
        native(owner.registry->set_seq_data(owner.descriptors[i], owner.data[i]));
    value = owner.problem(Kind::kForward, false);
    value.buffers[14] = {};
    value.reserve_bytes = 0;
    value.current = 1;
    result = owner.session->prepare_dnn_attn(value);
    expect(!result.count && !result.first,
           "actual maximum length yields native no-op without window reads");
    owner.issue(std::move(value));
    value = owner.problem(Kind::kForward);
    value.windows[0].pop_back();
    const auto rejected = owner.accept(std::move(value));
    expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status,
           "stale/truncated capture has no native compute provenance");
    owner.close();
}
void fresh_and_limits() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    const auto fresh = owner.create(CudaDnnObjectKind::kSeqData);
    auto value = owner.problem(Kind::kForward, false);
    value.descriptors[0] = *owner.registry->borrow(fresh, CudaDnnObjectKind::kSeqData);
    auto result = owner.session->prepare_dnn_attn(value);
    expect(result.result.status == CUDA_ERROR_INVALID_VALUE && !result.result.original_status &&
               !owner.registry->snapshot().is_quarantined && !fixture::calls(),
           "legitimate fresh zero SeqData is clean unusable extent, not corrupted metadata");
    value = {};
    native(owner.registry->destroy(fresh));
    owner.config.max_query_sequence = INT_MAX;
    native(owner.registry->set_attn(owner.parent, owner.config));
    value = owner.problem(Kind::kForward, false);
    result = owner.session->prepare_dnn_attn(value);
    expect(result.result.status == CUDA_ERROR_INVALID_VALUE && !result.result.original_status &&
               !fixture::calls(),
           "full-window quota fails before native compute or payload capture");
    value = {};
    owner.config.max_query_sequence = 4;
    native(owner.registry->set_attn(owner.parent, owner.config));
    value = owner.problem(Kind::kForward);
    value.current = INT_MIN;
    owner.issue(std::move(value));
    value = owner.problem(Kind::kForward);
    value.windows[0] = value.windows[1];
    owner.issue(
        std::move(value));  // Empty windows are a native ABI case, not this mathematical oracle.
    owner.close();
}
std::uint64_t restore(Owner& owner, std::size_t index) {
    const auto allocation = owner.session->allocate(fixture::dropout::kStateBytes);
    driver(allocation);
    owner.extra.push_back(allocation.id);
    CudaSessionDnnDropout problem;
    problem.kind = CudaDnnDropoutKind::kRestore;
    problem.handle = owner.handle;
    problem.mutation = owner.registry->begin_dropout_mutation(owner.dropouts[index]).mutation;
    problem.states = {allocation.id, 0};
    problem.state_bytes = fixture::dropout::kStateBytes;
    problem.probability_bits = 0x3e800000U;
    problem.seed = 0xfedcba9876543210ULL + index;
    const auto accepted = owner.session->accept_dnn_dropout(owner.next, problem);
    expect(accepted.acceptance.accepted(), "original Restore establishes live owned RNG binding");
    ++owner.next;
    problem = {};
    expect(owner.session->issue_next().library_status == CUDNN_STATUS_SUCCESS,
           "Restore issues once");
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt), "Restore host ACK");
    expect(owner.session->poll().retired == 1, "Restore physical completion");
    return allocation.id;
}
void rng_lifetime() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    const std::array states{restore(owner, 0), restore(owner, 1)};
    const auto syncs = fixture::control.syncs.load();
    foundation::control.can_stop.store(false);
    owner.issue(owner.problem(Kind::kForward), CUDNN_STATUS_SUCCESS, false);
    for (const auto id : states)
        expect(owner.session->release(id).disposition == CudaSessionDisposition::kDeferred,
               "both distinct RNG allocations are physically retained by Forward training");
    expect(fixture::control.syncs.load() == syncs && owner.session->poll().retired == 0,
           "Forward host ACK does not synchronize or release RNG states");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "only event proof releases training RNG leases");
    for (const auto id : states) {
        driver(owner.session->release(id));
        std::erase(owner.extra, id);
    }
    auto value = owner.problem(Kind::kForward);
    const auto rejected = owner.accept(std::move(value));
    expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status,
           "Forward rejects stale RNG allocation identities before original issue");
    value = owner.problem(Kind::kForward);
    value.buffers[14] = {};
    value.reserve_bytes = 0;
    owner.issue(std::move(value));  // Inference has no RNG payload lease.
    owner.issue(owner.problem(Kind::kBackwardData));
    owner.issue(
        owner.problem(Kind::kBackwardWeights));  // Backward uses reserve/rates, not old RNG bytes.
    value = owner.problem(Kind::kForward, false);
    value.window_presence = 0;
    owner.issue(std::move(value), CUDNN_STATUS_BAD_PARAM);
    value = owner.problem(Kind::kForward, false);
    value.current = 1;
    owner.issue(std::move(value),
                CUDNN_STATUS_BAD_PARAM);  // Incremental plus reserve fails before RNG.
    value = owner.problem(Kind::kForward);
    value.reserve_bytes = 0;
    const auto zero_bytes = owner.accept(std::move(value));
    expect(!zero_bytes.native.acceptance.accepted(),
           "nonnull zero-byte reserve still selects training RNG provenance");
    owner.config.dropouts = {};
    native(owner.registry->set_attn(owner.parent, owner.config));
    owner.issue(owner.problem(Kind::kForward));  // Original null dropouts disable their segments.
    owner.close();
}
void reentry() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    auto recursive = owner.problem(Kind::kForward, false);
    std::array<unsigned int, 7> phases{};
    fixture::state.callback = [&](unsigned int phase) {
        ++phases[phase];
        const auto sets = fixture::control.context_sets.load(),
                   streams = fixture::control.stream_gets.load();
        expect(owner.session->prepare_dnn_attn(recursive).result.status == CUDA_ERROR_NOT_READY &&
                   !owner.session->accept_dnn_attn(owner.next, recursive)
                        .native.acceptance.accepted() &&
                   owner.session->activate().status == CUDA_ERROR_NOT_READY &&
                   owner.session->deactivate().status == CUDA_ERROR_NOT_READY &&
                   owner.session->get_dnn_stream(owner.handle).status == CUDA_ERROR_NOT_READY &&
                   owner.session->destroy_dnn(owner.handle).status == CUDA_ERROR_NOT_READY &&
                   owner.session->shutdown().status == CUDA_ERROR_NOT_READY,
               "private/native/proof callbacks cannot recurse into session lifecycle");
        expect(!owner.registry->borrow(owner.parent, CudaDnnObjectKind::kAttn) &&
                   !owner.registry->begin_dropout_mutation(owner.dropouts[0]).mutation &&
                   owner.registry->shutdown().disposition == CudaDnnDisposition::kDeferred &&
                   fixture::control.context_sets.load() == sets &&
                   fixture::control.stream_gets.load() == streams,
               "reentry cannot reach extra Driver/native callbacks or release transitive metadata");
        if (phase == 3 || phase == 4 || phase == 5) {
            expect(owner.session->issue_next().native_status == CUDA_ERROR_NOT_READY &&
                       owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                       owner.session->drain_issued().status == CUDA_ERROR_NOT_READY,
                   "active native issue excludes recursive queue consumption and premature event "
                   "polling");
        }
    };
    owner.issue(owner.problem(Kind::kForward));
    fixture::control.native_status.store(CUDNN_STATUS_EXECUTION_FAILED);
    owner.issue(owner.problem(Kind::kForward), CUDNN_STATUS_EXECUTION_FAILED);
    fixture::state.callback = {};
    recursive = {};
    for (auto phase : {1, 2, 3, 4, 5, 6})
        expect(phases[phase] != 0,
               "all private/version/native/proof callback boundaries exercised");
    owner.close();
}
void retirement_reentry() {
    Owner owner(CUDNN_DATA_FLOAT, 0, 0, false, true);
    fixture::control.arithmetic.store(false);
    auto recursive = owner.problem(Kind::kForward, false);
    const auto first = owner.issue(owner.problem(Kind::kForward), CUDNN_STATUS_SUCCESS, false);
    const auto second = owner.accept(owner.problem(Kind::kForward));
    expect(second.native.acceptance.accepted(), "retain an unissued successor during cleanup");
    unsigned int callbacks{};
    on_event_cleanup = [&] {
        ++callbacks;
        const auto sets = fixture::control.context_sets.load();
        const auto streams = fixture::control.stream_gets.load();
        const auto calls = fixture::calls();
        expect(
            owner.session->activate().status == CUDA_ERROR_NOT_READY &&
                owner.session->deactivate().status == CUDA_ERROR_NOT_READY &&
                owner.session->get_dnn_stream(owner.handle).status == CUDA_ERROR_NOT_READY &&
                owner.session->set_dnn_stream(owner.handle, 0).status == CUDA_ERROR_NOT_READY &&
                owner.session->destroy_dnn(owner.handle).status == CUDA_ERROR_NOT_READY &&
                owner.session->prepare_dnn_attn(recursive).result.status == CUDA_ERROR_NOT_READY &&
                !owner.session->accept_dnn_attn(owner.next, recursive)
                     .native.acceptance.accepted() &&
                owner.session->issue_next().native_status == CUDA_ERROR_NOT_READY &&
                owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                owner.session->drain_issued().status == CUDA_ERROR_NOT_READY &&
                !owner.session->reject_queued(CUDA_ERROR_DEINITIALIZED) &&
                owner.session->shutdown().status == CUDA_ERROR_NOT_READY,
            "physical cleanup callbacks cannot mutate session/context/handle or consume queue "
            "state");
        expect(fixture::control.context_sets.load() == sets &&
                   fixture::control.stream_gets.load() == streams && fixture::calls() == calls,
               "cleanup reentry reaches no additional native execution or handle callbacks");
    };
    expect(owner.session->poll().retired == 1 && callbacks == 1 &&
               first.result->result.original_status == CUDNN_STATUS_SUCCESS,
           "physical cleanup retains original status and invokes guarded callback once");
    on_event_cleanup = {};
    expect(owner.session->issue_next().sequence == second.native.acceptance.receipt.sequence() &&
               owner.session->poll().retired == 1 &&
               owner.session->acknowledge_result(second.native.acceptance.receipt),
           "cleanup callback did not reject or consume the queued successor");
    recursive = {};
    owner.close();
}
void cross_session_rng() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    (void)restore(owner, 0);
    (void)restore(owner, 1);
    expect(owner.sdk.context_set_current(nullptr) == CUDA_SUCCESS,
           "detach before second owned session");
    CudaExecutionSession other({291, 292}, 294, owner.sdk);
    driver(other.initialize(0));
    const auto handle = other.create_dnn();
    native(handle);
    const auto stream = other.create_stream(CU_STREAM_NON_BLOCKING);
    driver(stream);
    native(other.set_dnn_stream(handle.id, stream.id));
    std::vector<std::uint64_t> allocations;
    for (std::size_t i = 0; i != 17; ++i) {
        const auto memory = other.allocate(256);
        driver(memory);
        allocations.push_back(memory.id);
    }
    for (std::size_t i = 0; i != owner.buffers.size(); ++i)
        expect(allocations[i] == owner.buffers[i].allocation,
               "matching resource creation order aligns every ordinary allocation ID");
    expect(allocations[15] == owner.extra[0] && allocations[16] == owner.extra[1],
           "distinct session deliberately reuses numeric RNG allocation IDs");
    auto value = owner.problem(Kind::kForward);
    value.handle = handle.id;
    const auto denied = other.accept_dnn_attn(1, value);
    value = {};
    expect(
        !denied.native.acceptance.accepted() && !denied.native.original_status && !fixture::calls(),
        "matching numeric allocation IDs cannot adopt RNG storage from another session "
        "incarnation");
    for (const auto id : allocations) driver(other.release(id));
    native(other.destroy_dnn(handle.id));
    driver(other.shutdown());
    expect(owner.sdk.context_set_current(owner.context) == CUDA_SUCCESS,
           "restore known first-session context");
    owner.close();
}
void bounds() {
    Owner owner(CUDNN_DATA_DOUBLE);
    fixture::control.arithmetic.store(false);
    for (std::size_t i = 0; i != 4; ++i) {
        owner.data[i].type = CUDNN_DATA_HALF;
        native(owner.registry->set_seq_data(owner.descriptors[i], owner.data[i]));
    }
    auto value = owner.problem(Kind::kForward);
    value.buffers[2].offset += owner.bytes[2] / 2;
    auto denied = owner.accept(std::move(value));
    expect(!denied.native.acceptance.accepted() && !denied.native.original_status,
           "parent DOUBLE storage cannot fit HALF-only owned suffix");
    owner.issue(owner.problem(Kind::kForward));
    for (std::size_t i = 0; i != 4; ++i) {
        owner.data[i].sizes[2] = (i == 1 || i == 2) ? 1 : 3;
        owner.data[i].sequence_count = 2 * owner.data[i].sizes[2];
        owner.data[i].lengths.assign(owner.data[i].sequence_count, owner.data[i].sizes[0]);
        native(owner.registry->set_seq_data(owner.descriptors[i], owner.data[i]));
    }
    value = owner.problem(Kind::kForward);
    value.buffers[0].offset = 32;
    denied = owner.accept(std::move(value));
    expect(!denied.native.acceptance.accepted(),
           "Q length array needs six ints independent from KV two");
    // Supply fresh sufficiently large ordinary ranges; no parent-max allocation inflation.
    value = owner.problem(Kind::kForward);
    for (auto slot : {0, 2, 5, 6}) {
        const auto allocation = owner.session->allocate(256);
        driver(allocation);
        owner.extra.push_back(allocation.id);
        value.buffers[slot] = {allocation.id, 0};
    }
    value.buffers[1].offset = 40;  // Exactly eight bytes remain in the original guarded allocation.
    owner.issue(std::move(value));
    owner.close();
}
void lifetime() {
    Owner owner;
    fixture::control.arithmetic.store(false);
    auto value = owner.problem(Kind::kForward);
    const auto accepted = owner.accept(std::move(value));
    expect(accepted.native.acceptance.accepted(), "queued command accepted");
    expect(owner.session->reject_queued(CUDA_ERROR_NOT_READY) == 1,
           "cancel exactly one unissued command");
    const auto receipt = owner.session->host_result(accepted.native.acceptance.receipt);
    expect(receipt &&
               receipt->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
               !accepted.result->is_ready && !fixture::calls(),
           "cancellation has no fake native result or proof");
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt), "cancel ACK");
    foundation::control.can_stop.store(false);
    const auto issued = owner.issue(owner.problem(Kind::kForward), CUDNN_STATUS_SUCCESS, false);
    expect(owner.registry->set_attn(owner.parent, owner.config).disposition ==
                   CudaDnnDisposition::kDeferred &&
               !owner.registry->begin_dropout_mutation(owner.dropouts[0]).mutation &&
               !owner.registry->begin_dropout_mutation(owner.dropouts[1]).mutation &&
               owner.registry->destroy(owner.descriptors[0]).disposition ==
                   CudaDnnDisposition::kDeferred &&
               owner.session->release(owner.buffers[2].allocation).disposition ==
                   CudaSessionDisposition::kDeferred,
           "ACK cannot retire parent, sequence, dropout or memory leases");
    expect(owner.session->poll().retired == 0, "NOT_READY event retains all ownership");
    foundation::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1, "physical event is sole retirement proof");
    native(owner.registry->set_attn(owner.parent, owner.config));
    expect(issued.result->is_ready,
           "result capsule survives retirement without owning descriptors");
    owner.close();
}
void poison(unsigned int which) {
    Owner owner;
    fixture::control.arithmetic.store(false);
    auto value = owner.problem(Kind::kForward);
    if (which < 5) {
        if (which == 0)
            fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
        else
            fixture::control.corruption.store(which);
        const auto result = owner.accept(std::move(value));
        expect(!result.native.acceptance.accepted() && !result.native.original_status &&
                   !fixture::calls(),
               "private getter error/corruption cannot manufacture native compute status");
        expect(owner.registry->snapshot().is_quarantined == (which != 0),
               "clean private error differs from malformed native observation");
        ::_exit(0);
    }
    if (which == 5) fixture::control.context_loss.store(3);
    if (which == 6 || which == 7) {
        fixture::control.native_status.store(CUDNN_STATUS_EXECUTION_FAILED);
        fixture::control.partial_write.store(true);
        if (which == 6)
            fixture::control.sync_status.store(CUDA_ERROR_UNKNOWN);
        else
            fixture::control.context_loss.store(5);
    }
    if (which == 8) fixture::control.stream_change.store(true);
    if (which == 9) fixture::control.stream_status.store(CUDNN_STATUS_BAD_PARAM);
    if (which == 10) fixture::control.event_status.store(CUDA_ERROR_UNKNOWN);
    const auto accepted = owner.accept(std::move(value));
    expect(accepted.native.acceptance.accepted(), "admit later failed physical proof");
    const auto allocations = foundation::driver.memory.size();
    const auto* partial_output = owner.memory<std::byte>(6);
    const auto result = owner.session->issue_next();
    expect(result.library_status ==
               (which == 6 || which == 7 ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS),
           "original status retained across proof failure");
    if (which == 6 || which == 7)
        expect(*partial_output == std::byte{0x67} && fixture::control.partial_writes.load() == 1 &&
                   fixture::control.syncs.load() == 1 && !fixture::control.records.load(),
               "partially written failed operation attempts one proof and no unsafe event");
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt),
           "failed receipt ACK");
    if (which == 10) {
        const auto polls = fixture::control.polls.load();
        const auto stream_syncs = fixture::query::control.synchronizes.load();
        const auto polled = owner.session->poll();
        expect(polled.retired == 0 && polled.retained == 1 && polled.status == CUDA_ERROR_UNKNOWN &&
                   fixture::control.polls.load() == polls &&
                   foundation::driver.memory.size() == allocations &&
                   !fixture::base::state.handle_destroys &&
                   owner.registry->set_attn(owner.parent, owner.config).disposition ==
                       CudaDnnDisposition::kDeferred,
               "a never-recorded event cannot prove completion or release metadata");
        foundation::control.can_stop.store(false);
        const auto pending = owner.session->drain_issued();
        expect(pending.retired == 0 && pending.retained == 1 &&
                   pending.status == CUDA_ERROR_NOT_READY &&
                   fixture::query::control.synchronizes.load() == stream_syncs + 1 &&
                   foundation::driver.memory.size() == allocations,
               "failed fallback stream proof still retains every physical owner");
        foundation::control.can_stop.store(true);
        const auto drained = owner.session->drain_issued();
        expect(drained.retired == 1 &&
                   fixture::query::control.synchronizes.load() == stream_syncs + 2 &&
                   fixture::calls() == 1,
               "explicit successful known-stream proof may recover trailing record failure");
        const auto retry = owner.accept(owner.problem(Kind::kForward));
        expect(!retry.native.acceptance.accepted() && !retry.native.original_status &&
                   fixture::calls() == 1,
               "recovered physical ownership does not reopen a quarantined submission queue");
        native(owner.registry->shutdown());
        driver(owner.session->shutdown());
        expect(foundation::driver.memory.empty() && foundation::driver.contexts.empty() &&
                   fixture::base::state.handle_destroys == 1 &&
                   fixture::metadata::state.destroys == 1,
               "proven recovery permits exact cleanup without retrying original compute");
        ::_exit(0);
    }
    foundation::control.can_stop.store(true);
    (void)owner.session->poll();
    (void)owner.session->drain_issued();
    (void)owner.session->shutdown();
    if (foundation::driver.memory.size() != allocations || fixture::base::state.handle_destroys ||
        fixture::metadata::state.destroys || fixture::calls() != 1)
        std::cerr << "attention poison case=" << which
                  << " allocations=" << foundation::driver.memory.size() << '/' << allocations
                  << " handles_destroyed=" << fixture::base::state.handle_destroys
                  << " metadata_destroyed=" << fixture::metadata::state.destroys
                  << " original_calls=" << fixture::calls()
                  << " records=" << fixture::control.records.load()
                  << " syncs=" << fixture::control.syncs.load()
                  << " issue_state=" << static_cast<unsigned int>(result.state)
                  << " driver_status=" << result.native_status << '\n';
    expect(foundation::driver.memory.size() == allocations &&
               !fixture::base::state.handle_destroys && !fixture::metadata::state.destroys &&
               fixture::calls() == 1,
           "unknown completion never releases or retries native ownership");
    ::_exit(0);
}
void partial_error() {
    for (const auto kind : {Kind::kForward, Kind::kBackwardData, Kind::kBackwardWeights}) {
        Owner owner;
        fixture::control.arithmetic.store(false);
        fixture::control.native_status.store(CUDNN_STATUS_EXECUTION_FAILED);
        fixture::control.partial_write.store(true);
        foundation::control.can_stop.store(false);
        const auto slot = kind == Kind::kForward ? 6 : kind == Kind::kBackwardData ? 8 : 12;
        const auto* output = owner.memory<std::byte>(slot);
        const auto allocations = foundation::driver.memory.size();
        owner.issue(owner.problem(kind), CUDNN_STATUS_EXECUTION_FAILED, false);
        expect(*output == std::byte{0x67} && fixture::control.partial_writes.load() == 1 &&
                   fixture::control.syncs.load() == 1 && fixture::control.records.load() == 1 &&
                   !owner.registry->snapshot().is_quarantined,
               "partial native failure preserves exact status after one context proof");
        expect(owner.session->poll().retired == 0 &&
                   owner.session->release(owner.buffers[slot].allocation).disposition ==
                       CudaSessionDisposition::kDeferred &&
                   foundation::driver.memory.size() == allocations,
               "partial write and host ACK cannot release in-flight allocation ownership");
        foundation::control.can_stop.store(true);
        expect(owner.session->poll().retired == 1,
               "proved failed operation retires at its trailing event");
        owner.close();
    }
}
}  // namespace
int main() {
    try {
        numerical();
        preparation();
        fresh_and_limits();
        bounds();
        lifetime();
        rng_lifetime();
        cross_session_rng();
        reentry();
        retirement_reentry();
        partial_error();
        for (unsigned int missing = 1; missing != 7; ++missing) {
            Owner owner(CUDNN_DATA_FLOAT, 0, missing);
            fixture::control.arithmetic.store(false);
            auto value = owner.problem(missing <= 3 ? static_cast<Kind>(missing) : Kind::kForward);
            const auto result = owner.accept(std::move(value));
            expect(!result.native.acceptance.accepted() && !result.native.original_status &&
                       !fixture::calls(),
                   "missing capability has no original compute provenance");
            owner.close();
        }
        for (unsigned int which = 0; which != 11; ++which) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated proof failure");
            if (!child) poison(which);
            int status{};
            if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status))
                throw std::runtime_error("isolated attention proof case " + std::to_string(which));
        }
        std::cout << "Attention CPU computation and ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
