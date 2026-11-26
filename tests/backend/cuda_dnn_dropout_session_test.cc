#include "dnn_dropout_fixture.h"

#include <iostream>
#include <source_location>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_dropout;
namespace foundation = fixture::driver;
namespace base = glimmer::tests::dnn;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           ("dropout Driver failure at " + std::to_string(where.line())).c_str());
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           ("dropout SDK failure at " + std::to_string(where.line())).c_str());
}
constexpr std::uint64_t kSeed = 0xfedcba9876543210ULL;
constexpr std::size_t kBytes = fixture::kStateBytes + 16;
std::vector<std::byte> initialized(std::uint64_t seed) {
    std::vector<std::byte> result(kBytes, std::byte{0xcd});
    const auto seed_bytes = std::bit_cast<std::array<std::byte, 8>>(seed);
    for (std::size_t i = 0; i != fixture::kStateBytes; ++i)
        result[i + 8] = seed_bytes[i % 8] ^ static_cast<std::byte>(0x5b + i);
    return result;
}
struct Owner {
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t descriptor{}, handle{}, allocation{}, next_sequence{1};
    CUcontext context{};
    CUdeviceptr address{};
    explicit Owner(unsigned int missing = 0, bool explicit_stream = false,
                   bool failed_event_record = false) {
        fixture::reset();
        auto sdk = fixture::session_dispatch();
        auto dnn = *sdk.dnn;
        if (missing == 1) dnn.dropout_set = nullptr;
        if (missing == 2) dnn.dropout_restore = nullptr;
        if (missing == 3) dnn.dropout_states_size = nullptr;
        if (missing == 4) sdk.context_synchronize = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(dnn);
        if (failed_event_record)
            sdk.execution.event_record = [](CUevent, CUstream) { return CUDA_ERROR_LAUNCH_FAILED; };
        metadata = std::make_unique<CudaDnnRegistry>(dnn);
        const auto created = metadata->create(CudaDnnObjectKind::kDropout);
        success(created);
        descriptor = created.id;
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{81, 82}, 83, sdk);
        success(session->initialize(0));
        context = foundation::driver.current;
        const auto native_handle = session->create_dnn();
        success(native_handle);
        handle = native_handle.id;
        if (explicit_stream) {
            const auto stream = session->create_stream(CU_STREAM_NON_BLOCKING);
            success(stream);
            success(session->set_dnn_stream(handle, stream.id));
        }
        const auto memory = session->allocate(kBytes);
        success(memory);
        allocation = memory.id;
        address = session->device_memory({allocation, 0}, kBytes)->address;
    }
    CudaSessionDnnDropout problem(CudaDnnDropoutKind kind = CudaDnnDropoutKind::kSet) {
        auto mutation = metadata->begin_dropout_mutation(descriptor);
        expect(mutation.mutation && mutation.result.status == CUDA_SUCCESS &&
                   !mutation.result.original_status,
               "mutation reservation is internal, not native SDK success");
        CudaSessionDnnDropout result;
        result.kind = kind;
        result.handle = handle;
        result.mutation = std::move(mutation.mutation);
        result.states = {allocation, 8};
        result.state_bytes = fixture::kStateBytes;
        result.probability_bits = 0x3e800000U;
        result.seed = kSeed;
        return result;
    }
    std::vector<std::byte> bytes() const {
        return foundation::driver.memory.at(address).bytes;
    }
    void retained() {
        expect(!metadata->borrow_dropout(descriptor) &&
                   metadata->begin_dropout_mutation(descriptor).result.disposition ==
                       CudaDnnDisposition::kDeferred &&
                   metadata->destroy(descriptor).disposition == CudaDnnDisposition::kDeferred &&
                   session->release(allocation).disposition == CudaSessionDisposition::kDeferred &&
                   session->destroy_dnn(handle).disposition == CudaDnnDisposition::kDeferred,
               "queued/pending configuration retains mutable descriptor, state and native handle");
    }
    void issue(CudaSessionDnnDropout problem, bool pending = false,
               cudnnStatus_t expected = CUDNN_STATUS_SUCCESS) {
        const auto accepted = session->accept_dnn_dropout(next_sequence++, problem);
        expect(accepted.acceptance.accepted(), "admit original stateful dropout call");
        problem = {};
        retained();
        foundation::control.can_stop.store(false);
        const auto issued = session->issue_next();
        const auto host = session->host_result(accepted.acceptance.receipt);
        expect(issued.state == CudaExecutionIssueState::kIssued && host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                   host->status == expected && issued.library_status == expected &&
                   session->acknowledge_result(accepted.acceptance.receipt) &&
                   !session->poll().retired,
               "native status and host ACK do not retire pending original work");
        if (expected == CUDNN_STATUS_SUCCESS) retained();
        if (pending) {
            const auto calls = fixture::control.callbacks.load();
            const auto queries = fixture::state.queries;
            const auto synchronizes = foundation::control.context_synchronizes.load();
            const auto result = session->query_dnn_dropout_states_size(handle);
            success(result);
            expect(result.value == fixture::kStateBytes && fixture::state.queries == queries + 1 &&
                       fixture::control.callbacks.load() == calls &&
                       !foundation::control.can_stop.load() &&
                       foundation::control.context_synchronizes.load() == synchronizes,
                   "native size query progresses without draining pending initialization");
        }
        foundation::control.can_stop.store(true);
        expect(session->poll().retired == 1, "original event retires one configuration");
    }
    void finish(bool freed = false) {
        if (!freed) success(session->release(allocation));
        success(metadata->destroy(descriptor));
        success(metadata->shutdown());
        success(session->shutdown());
        expect(fixture::state.destroys == 1 && base::state.handle_destroys == 1 &&
                   foundation::driver.memory.empty() && foundation::driver.contexts.empty() &&
                   foundation::driver.streams.empty() && foundation::driver.events.empty(),
               "proven ownership is destroyed exactly once without persistent allocation pins");
    }
};
void ordinary(bool explicit_stream) {
    Owner owner(0, explicit_stream);
    {
        const auto empty = owner.metadata->borrow_dropout(owner.descriptor);
        const auto resolved = owner.session->resolve_dnn_dropout(*empty);
        expect(resolved && !resolved->address && !resolved->bytes,
               "fresh unbound dropout descriptor resolves as an explicitly empty binding");
    }
    auto empty = owner.problem();
    empty.states = {};
    empty.state_bytes = std::numeric_limits<std::size_t>::max();
    empty.probability_bits = 0x80000000U;
    owner.issue(std::move(empty));
    expect(fixture::state.sets == 1 && !fixture::state.initializations &&
               fixture::state.last.probability_bits == 0x80000000U &&
               fixture::state.last.bytes == std::numeric_limits<std::size_t>::max() &&
               fixture::state.last.seed == kSeed,
           "null Set preserves raw negative-zero probability, full count and seed without RNG");
    owner.issue(owner.problem(), true);
    expect(owner.bytes() == initialized(kSeed) && fixture::state.initializations == 1 &&
               (explicit_stream ? fixture::state.last.stream != nullptr
                                : fixture::state.last.stream == nullptr),
           "Set initializes exactly the declared state once on the selected original stream");
    {
        const auto view = owner.metadata->borrow_dropout(owner.descriptor);
        const auto resolved = owner.session->resolve_dnn_dropout(*view);
        expect(resolved && resolved->address == owner.address + 8 &&
                   resolved->bytes == fixture::kStateBytes &&
                   view->binding.allocation == owner.allocation && view->binding.offset == 8,
               "successful initialization publishes exact range and revalidates live identity");
    }
    owner.issue(owner.problem(CudaDnnDropoutKind::kRestore));
    expect(owner.bytes() == initialized(kSeed) && fixture::state.initializations == 1 &&
               fixture::state.restores == 1,
           "Restore invokes its original callback once without rewriting saved RNG contents");
    for (const auto kind : {CudaDnnDropoutKind::kSet, CudaDnnDropoutKind::kRestore}) {
        auto problem = owner.problem(kind);
        problem.states = {};
        problem.state_bytes = 0x100000003ULL;
        problem.probability_bits = 0x3f000000U;
        problem.seed = ~kSeed;
        owner.issue(std::move(problem));
        const auto view = owner.metadata->borrow_dropout(owner.descriptor);
        expect(view && view->binding.allocation == owner.allocation &&
                   owner.session->resolve_dnn_dropout(*view)->address == owner.address + 8 &&
                   owner.bytes() == initialized(kSeed) && fixture::state.initializations == 1,
               "null Set/Restore preserves prior state provenance and contents");
    }
    const auto second = owner.session->allocate(kBytes);
    success(second);
    auto rebind = owner.problem();
    rebind.states.allocation = second.id;
    rebind.seed = ~kSeed;
    owner.issue(std::move(rebind));
    {
        const auto view = owner.metadata->borrow_dropout(owner.descriptor);
        expect(view && view->binding.allocation == second.id &&
                   owner.session->resolve_dnn_dropout(*view) && owner.bytes() == initialized(kSeed),
               "rebind changes exact provenance without modifying previous state allocation");
    }
    success(owner.session->release(second.id));
    {
        const auto stale = owner.metadata->borrow_dropout(owner.descriptor);
        expect(stale && !owner.session->resolve_dnn_dropout(*stale),
               "free-before-descriptor-destroy is legal but future state use rejects as stale");
    }
    auto stale_null = owner.problem();
    stale_null.states = {};
    stale_null.state_bytes = 77;
    owner.issue(std::move(stale_null));
    {
        const auto stale = owner.metadata->borrow_dropout(owner.descriptor);
        expect(stale && stale->binding.allocation == second.id &&
                   !owner.session->resolve_dnn_dropout(*stale),
               "null Set after caller free cannot silently detach or revive a stale binding");
    }
    owner.issue(owner.problem(CudaDnnDropoutKind::kRestore));
    {
        const auto rebound = owner.metadata->borrow_dropout(owner.descriptor);
        expect(rebound && rebound->binding.allocation == owner.allocation &&
                   owner.session->resolve_dnn_dropout(*rebound),
               "explicit Restore can replace stale provenance with another live owned allocation");
    }
    owner.finish();
    expect(foundation::driver.frees == 2 && fixture::state.sets == 5 &&
               fixture::state.restores == 3 && fixture::state.initializations == 2,
           "ordinary exact calls and both caller-owned frees are not replayed");
}
void cancelled_and_rejected() {
    Owner owner;
    owner.issue(owner.problem());
    auto problem = owner.problem();
    problem.seed = ~kSeed;
    const auto accepted = owner.session->accept_dnn_dropout(owner.next_sequence++, problem);
    expect(accepted.acceptance.accepted() &&
               !owner.session->accept_dnn_dropout(owner.next_sequence++, problem)
                    .acceptance.accepted(),
           "one exclusive mutation token cannot admit two commands");
    problem = {};
    expect(owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
           "cancel only accepted unissued configuration");
    const auto host = owner.session->host_result(accepted.acceptance.receipt);
    expect(host && host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
               host->status == CUDA_ERROR_INVALID_VALUE &&
               owner.session->acknowledge_result(accepted.acceptance.receipt) &&
               fixture::state.sets == 1 && owner.bytes() == initialized(kSeed),
           "queued cancellation has no fabricated SDK result, rewrite or lost old binding");
    auto invalid = owner.problem();
    invalid.states.offset = kBytes;
    invalid.state_bytes = 1;
    expect(
        !owner.session->accept_dnn_dropout(owner.next_sequence++, invalid).acceptance.accepted() &&
            fixture::state.sets == 1,
        "out-of-range state rejects before native initialization");
    invalid = {};
    auto view = owner.metadata->borrow_dropout(owner.descriptor);
    expect(view && owner.session->resolve_dnn_dropout(*view),
           "rejection preserves original binding");
    view.reset();
    owner.finish();
}
void queries_and_missing() {
    for (unsigned int missing = 1; missing != 5; ++missing) {
        Owner owner(missing);
        if (missing == 3) {
            const auto result = owner.session->query_dnn_dropout_states_size(owner.handle);
            expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                       !fixture::state.queries,
                   "missing state-size callback rejects before original query");
        } else {
            auto problem = owner.problem(missing == 1 ? CudaDnnDropoutKind::kSet
                                                      : CudaDnnDropoutKind::kRestore);
            const auto result = owner.session->accept_dnn_dropout(1, problem);
            expect(!result.acceptance.accepted() && !fixture::control.callbacks.load(),
                   "missing original Set/Restore callback rejects before initialization");
        }
        owner.finish();
    }
    Owner owner;
    fixture::state.reported_bytes = std::numeric_limits<std::size_t>::max() - 7;
    const auto full = owner.session->query_dnn_dropout_states_size(owner.handle);
    success(full);
    expect(full.value == fixture::state.reported_bytes, "native size query preserves full width");
    fixture::control.query_status.store(CUDNN_STATUS_NOT_SUPPORTED);
    const auto error = owner.session->query_dnn_dropout_states_size(owner.handle);
    expect(error.original_status == CUDNN_STATUS_NOT_SUPPORTED && !error.value,
           "failed original query publishes no partially written size");
    fixture::control.query_status.store(CUDNN_STATUS_SUCCESS);
    owner.finish();
}
void context_identity() {
    Owner owner;
    owner.issue(owner.problem());
    auto view = owner.metadata->borrow_dropout(owner.descriptor);
    expect(view && owner.session->resolve_dnn_dropout(*view), "first owner binding resolves");
    const auto sdk = fixture::session_dispatch();
    expect(sdk.context_set_current(nullptr) == CUDA_SUCCESS,
           "detach first owned context before constructing an independent owner");
    CudaExecutionSession other({81, 82}, 83, sdk);
    success(other.initialize(0));
    const auto second_context = foundation::driver.current;
    const auto handle = other.create_dnn();
    success(handle);
    const auto memory = other.allocate(kBytes);
    success(memory);
    expect(memory.id == owner.allocation && !other.resolve_dnn_dropout(*view),
           "same public session/context/allocation numbers cannot impersonate another incarnation");
    success(other.release(memory.id));
    success(other.shutdown());
    expect(!foundation::driver.contexts.contains(second_context) &&
               foundation::driver.contexts.contains(owner.context),
           "unrelated context cleanup preserves original state owner");
    expect(sdk.context_set_current(owner.context) == CUDA_SUCCESS,
           "restore exact first context before its once-only shutdown");
    success(owner.session->shutdown());
    expect(!owner.session->resolve_dnn_dropout(*view),
           "destroyed context invalidates long-lived binding despite retained CPU descriptor view");
    view.reset();
    success(owner.metadata->destroy(owner.descriptor));
    success(owner.metadata->shutdown());
    expect(fixture::state.destroys == 1 && base::state.handle_destroys == 2 &&
               foundation::driver.contexts.empty() && foundation::driver.memory.empty(),
           "host descriptor can be destroyed after both original contexts without deadlock");
}
void failed_native(bool restore, cudnnStatus_t status) {
    Owner owner;
    owner.issue(owner.problem());
    const auto previous = owner.bytes();
    auto problem = owner.problem(restore ? CudaDnnDropoutKind::kRestore : CudaDnnDropoutKind::kSet);
    problem.seed = ~kSeed;
    fixture::control.should_mutate_on_error.store(true);
    (restore ? fixture::control.restore_status : fixture::control.set_status).store(status);
    owner.issue(std::move(problem), false, status);
    expect(!owner.metadata->borrow_dropout(owner.descriptor) &&
               owner.metadata->snapshot().is_quarantined &&
               (restore ? owner.bytes() == previous : owner.bytes() == initialized(~kSeed)),
           "partially mutating original failure is returned exactly and never publishes usable "
           "binding");
    success(owner.session->release(owner.allocation));
    const auto destroys = fixture::state.destroys;
    expect(owner.metadata->shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.destroys == destroys,
           "proven physical stop permits caller state free but not uncertain descriptor cleanup");
    success(owner.session->shutdown());
}
void failed_event() {
    Owner owner(0, false, true);
    auto problem = owner.problem();
    const auto accepted = owner.session->accept_dnn_dropout(1, problem);
    expect(accepted.acceptance.accepted(), "admit before trailing event failure");
    problem = {};
    const auto issued = owner.session->issue_next();
    expect(issued.library_status == CUDNN_STATUS_SUCCESS &&
               issued.completion_status == CUDA_ERROR_LAUNCH_FAILED &&
               !owner.session->poll().retired && !owner.metadata->borrow_dropout(owner.descriptor),
           "successful native initialization does not excuse failed trailing physical record");
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt),
           "acknowledge original status without releasing physical obligation");
    // The existing dispatcher may recover a known owner by real stream drain;
    // an unrecorded event's synthetic SUCCESS is never accepted as proof.
    const auto drained = owner.session->drain_issued();
    expect(drained.retired == 1 && fixture::state.sets == 1 &&
               owner.metadata->borrow_dropout(owner.descriptor),
           "owned successful drain retires known work once without replaying Set");
    success(owner.metadata->shutdown());
    success(owner.session->shutdown());
}
void native_validation(unsigned int mode) {
    Owner owner;
    auto problem =
        owner.problem(mode == 0 ? CudaDnnDropoutKind::kRestore : CudaDnnDropoutKind::kSet);
    if (mode < 2) {
        problem.states.offset = mode == 0 ? kBytes : 8;
        problem.state_bytes = 0;
    } else if (mode == 2) {
        problem.state_bytes = fixture::kStateBytes - 1;
    } else {
        problem.probability_bits = 0x7fc12345U;
    }
    const auto offset = problem.states.offset, bytes = problem.state_bytes;
    const auto probability = problem.probability_bits;
    owner.issue(std::move(problem), false, CUDNN_STATUS_BAD_PARAM);
    expect(
        fixture::state.last.states == reinterpret_cast<void*>(owner.address + offset) &&
            fixture::state.last.bytes == bytes &&
            fixture::state.last.probability_bits == probability &&
            owner.bytes() == std::vector<std::byte>(kBytes, std::byte{0xcd}),
        "zero/end/short/NaN original validation preserves exact inputs and all allocation guards");
    success(owner.session->release(owner.allocation));
    success(owner.session->shutdown());
}
void uncertain_physical(unsigned int mode) {
    Owner owner;
    auto problem = owner.problem();
    const auto accepted = owner.session->accept_dnn_dropout(1, problem);
    expect(accepted.acceptance.accepted(), "admit before physical initialization uncertainty");
    problem = {};
    foundation::control.synchronize_status.store(mode == 0 ? CUDA_ERROR_LAUNCH_FAILED
                                                           : CUDA_SUCCESS);
    fixture::control.should_change_context.store(mode == 1);
    foundation::control.should_change_sync_context.store(mode == 2);
    const auto issued = owner.session->issue_next();
    expect(
        issued.library_status == CUDNN_STATUS_SUCCESS &&
            issued.completion_status ==
                (mode == 0 ? CUDA_ERROR_LAUNCH_FAILED : CUDA_ERROR_INVALID_CONTEXT) &&
            fixture::state.sets == 1 &&
            foundation::control.context_synchronizes.load() == (mode == 1 ? 0U : 1U),
        "failed initialization proof retains native SUCCESS and never synchronizes foreign owner");
    foundation::driver.current = owner.context;
    foundation::control.can_stop.store(true);
    foundation::control.synchronize_status.store(CUDA_SUCCESS);
    foundation::control.should_change_sync_context.store(false);
    expect(owner.session->acknowledge_result(accepted.acceptance.receipt) &&
               !owner.session->poll().retired && !owner.session->drain_issued().retired &&
               !owner.metadata->borrow_dropout(owner.descriptor) &&
               owner.session->shutdown().disposition != CudaSessionDisposition::kComplete &&
               owner.metadata->shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.sets == 1 && !fixture::state.destroys && !foundation::driver.frees &&
               !base::state.handle_destroys && !foundation::control.context_destroys.load(),
           "later fake event success cannot release uncertain state, retry Set, or destroy owners");
    owner.session.reset();
    owner.metadata.reset();
    expect(!fixture::state.destroys && !foundation::driver.frees &&
               !foundation::control.context_destroys.load(),
           "destructors retain uncertain physical ownership without SDK retries");
}
}  // namespace
int main() {
    try {
        ordinary(false);
        ordinary(true);
        cancelled_and_rejected();
        queries_and_missing();
        context_identity();
        failed_event();
        for (unsigned int mode = 0; mode != 4; ++mode) {
            const auto child = ::fork();
            expect(child >= 0, "fork native range/probability validation");
            if (!child) {
                try {
                    native_validation(mode);
                    ::_exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "dropout validation " << mode << ": " << error.what() << '\n';
                    ::_exit(1);
                }
            }
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated native argument validation returns original error");
        }
        for (unsigned int mode = 0; mode != 3; ++mode) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated failed physical proof");
            if (!child) {
                try {
                    uncertain_physical(mode);
                    ::_exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "dropout physical mode " << mode << ": " << error.what() << '\n';
                    ::_exit(1);
                }
            }
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated lost-context/fence proof retains ownership");
        }
        for (bool restore : {false, true})
            for (auto status :
                 {CUDNN_STATUS_BAD_PARAM, CUDNN_STATUS_ALLOC_FAILED, CUDNN_STATUS_INTERNAL_ERROR}) {
                const auto child = ::fork();
                expect(child >= 0, "fork intentional failed-mutation quarantine");
                if (!child) {
                    try {
                        failed_native(restore, status);
                        ::_exit(0);
                    } catch (const std::exception& error) {
                        std::cerr << "dropout SDK failure fixture: " << error.what() << '\n';
                        ::_exit(1);
                    }
                }
                int child_status{};
                expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                           !WEXITSTATUS(child_status),
                       "isolated original-error partial-mutation checks pass");
            }
        std::cout
            << "Dropout state initialization, provenance and physical lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
