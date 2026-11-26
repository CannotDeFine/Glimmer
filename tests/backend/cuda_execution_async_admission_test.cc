#include "cuda_execution_domain_fixture.h"
#include "internal/execution_submission_queue.h"
#include "internal/cuda_profile_cache.h"

#include <thread>

namespace {
thread_local bool counts_process_checks{};
thread_local unsigned int process_checks{};
}  // namespace
// GNU ld --wrap requires these exact native ABI names.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" pid_t __real_getpid() noexcept;
extern "C" pid_t __wrap_getpid() noexcept {
    if (counts_process_checks) ++process_checks;
    return __real_getpid();
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;

bool preparation_demand(const CudaExecutionSession& session) {
    const auto light = session.scheduling_snapshot();
    const auto full = session.snapshot();
    expect(light.has_pending_preparation == full.has_pending_preparation &&
               light.submissions.paused == full.submissions.paused &&
               light.is_quarantined == full.is_quarantined,
           "lightweight preparation, pause and fault observations retain full-view semantics");
    return light.has_pending_preparation;
}

unsigned int occupancy_calls{};
CUresult occupancy_failure{CUDA_SUCCESS};
CUresult next_event_failure{CUDA_SUCCESS};
CudaExecutionSession* preparation_owner{};
bool excludes_preparation_reentry{};
alignas(64) thread_local std::array<std::byte, 192> source_cut_host{};
thread_local bool is_source_cut_host_live{};
thread_local CUcontext source_cut_host_owner{};
CUresult CUDAAPI source_cut_host_allocate(void** output, std::size_t bytes, unsigned int flags) {
    expect(bytes == source_cut_host.size() && flags == CU_MEMHOSTALLOC_DEVICEMAP &&
               driver.current && !is_source_cut_host_live,
           "transformed source-cut control owns exactly three aligned HostSlots");
    source_cut_host_owner = driver.current;
    is_source_cut_host_live = true;
    *output = source_cut_host.data();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI source_cut_host_release(void* pointer) {
    expect(pointer == source_cut_host.data() && is_source_cut_host_live &&
               source_cut_host_owner == driver.current,
           "transformed source-cut host memory is released exactly once by its owner");
    is_source_cut_host_live = false;
    source_cut_host_owner = nullptr;
    ++driver.host_frees;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI source_cut_context_destroy(CUcontext context) {
    expect(!is_source_cut_host_live,
           "context destruction cannot substitute for explicit source-cut host cleanup");
    return context_destroy(context);
}
CUresult CUDAAPI observed_event_create(CUevent* output, unsigned int flags) {
    const auto failure = std::exchange(next_event_failure, CUDA_SUCCESS);
    return failure == CUDA_SUCCESS ? event_create(output, flags) : failure;
}
CUresult CUDAAPI observed_occupancy(int* blocks, CUfunction function, int threads,
                                    std::size_t bytes) {
    ++occupancy_calls;
    if (preparation_owner)
        excludes_preparation_reentry =
            !preparation_demand(*preparation_owner) &&
            preparation_owner->pending_preparation().status == CUDA_ERROR_NOT_READY;
    const auto failure = std::exchange(occupancy_failure, CUDA_SUCCESS);
    return failure == CUDA_SUCCESS ? occupancy(blocks, function, threads, bytes) : failure;
}
CudaSessionDispatch dispatch() {
    auto result = session_dispatch();
    result.occupancy = observed_occupancy;
    result.execution.event_create = observed_event_create;
    return result;
}
void reset_driver() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    driver.next_claimed = 8;
    occupancy_calls = 0;
    occupancy_failure = CUDA_SUCCESS;
    next_event_failure = CUDA_SUCCESS;
    preparation_owner = nullptr;
    excludes_preparation_reentry = false;
}
struct Resources {
    CudaSessionResult module, function, buffer;
};
Resources initialize(CudaExecutionSession& session) {
    success(session.initialize(0));
    const auto module = session.load_ptx(kSource);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    const auto buffer = session.allocate(32);
    success(function);
    success(buffer);
    return {module, function, buffer};
}
void cleanup(CudaExecutionSession& session, const Resources& resources) {
    success(session.release(resources.buffer.id));
    success(session.unload_module(resources.module.id));
    success(session.shutdown());
    expect(!driver.current && driver.context_destroys == driver.context_creates &&
               !driver.live_resources(),
           "healthy async cases explicitly retire every exact native resource");
}
void expect_values(const Resources& resources, std::uint32_t expected) {
    for (unsigned int block = 0; block != 8; ++block) {
        std::uint32_t actual{};
        std::memcpy(&actual,
                    device_bytes(resources.buffer.address + block * sizeof(actual), sizeof(actual)),
                    sizeof(actual));
        expect(actual == expected, "every non-idempotent result uses the original copied argument");
    }
}

void queue_precharges_final_abi_without_publishing_a_result() {
    ExecutionSubmissionQueue queue({1, 2}, {2, 12});
    ExecutionSubmissionQueue foreign({2, 2}, {2, 12});
    expect(queue.bind_executor() && foreign.bind_executor(), "bind exact independent queue owners");
    std::array input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    auto lease = std::make_shared<int>(7);
    std::weak_ptr<const void> retained = lease;
    ExecutionRequest request{.operation = 1,
                             .context = 9,
                             .parameters = input,
                             .lifetime = lease,
                             .parameter_capacity_bytes = 12};
    const auto accepted = queue.accept(1, request);
    const auto other = foreign.accept(1, request);
    expect(
        accepted.accepted() && other.accepted() &&
            queue.snapshot().retained_parameter_bytes == 12 &&
            queue.accept(2, request).status == ExecutionAcceptStatus::kFull &&
            !queue.host_result(accepted.receipt) && !queue.acknowledge_result(accepted.receipt),
        "full private ABI capacity is reserved before acceptance, without fabricating SDK reply");
    input.fill(std::byte{99});
    std::array<std::byte, 4> copied{};
    expect(queue.copy_unissued_parameters(accepted.receipt, copied) && copied[0] == std::byte{1} &&
               !queue.copy_unissued_parameters(other.receipt, copied),
           "CPU preparation reads an immutable, receipt-authenticated original ABI");
    std::array<std::byte, 13> too_large{};
    expect(!queue.replace_unissued_parameters(accepted.receipt, too_large) &&
               queue.copy_unissued_parameters(accepted.receipt, copied) &&
               copied[3] == std::byte{4},
           "over-capacity replacement is atomic and leaves original bytes intact");
    std::array<std::byte, 12> prepared{};
    std::ranges::copy(copied, prepared.begin());
    prepared.back() = std::byte{17};
    bool wrong_thread{};
    std::thread observer(
        [&] { wrong_thread = !queue.replace_unissued_parameters(accepted.receipt, prepared); });
    observer.join();
    expect(wrong_thread && queue.replace_unissued_parameters(accepted.receipt, prepared),
           "only the bound nonrecycled executor publishes the prepared ABI");
    request.lifetime.reset();
    lease.reset();
    const auto issue = queue.take_next(accepted.receipt);
    expect(issue && issue->parameters.size() == 12 && issue->parameters.back() == std::byte{17} &&
               !queue.replace_unissued_parameters(accepted.receipt, prepared) &&
               queue.publish_host_result(issue->token, 0) &&
               queue.acknowledge_result(accepted.receipt) && !retained.expired(),
           "host result and its ACK neither permit another rewrite nor release physical input");
    expect(queue.complete_after_physical_stop(issue->token) &&
               !queue.copy_unissued_parameters(accepted.receipt, copied) &&
               queue.snapshot().retained_parameter_bytes == 0,
           "only final physical retirement releases the full reserved payload");
    expect(foreign.reject_unissued(other.receipt, 7) && foreign.acknowledge_result(other.receipt) &&
               retained.expired() && foreign.shutdown(),
           "independent ownership is explicitly retired too");
    request.lifetime = std::make_shared<int>(8);
    const auto final = queue.accept(2, request);
    expect(final.accepted(), "a quota failure did not consume sequence two");
    expect(queue.reject_unissued(final.receipt, 7) && queue.acknowledge_result(final.receipt) &&
               queue.shutdown(),
           "reject and acknowledge the exact never-issued final queue record before shutdown");
}

void original_admission_preserves_sources_and_leases_without_preparation() {
    reset_driver();
    auto bounds = limits(1000);
    bounds.submissions.max_commands = 3;
    CudaExecutionSession session({1, 99}, 101, dispatch(), bounds);
    auto resources = initialize(session);
    success(session.release(resources.buffer.id));
    resources.buffer = session.allocate(48);
    success(resources.buffer);
    std::array<std::uint32_t, 12> words{};
    constexpr std::uint32_t guard = 0xdeadbeefU;
    words[0] = words[1] = words[10] = words[11] = guard;
    success(session.upload({resources.buffer.id, 0}, std::as_bytes(std::span{words})));
    KernelInput first({resources.function.id, 0}, resources.buffer.address + 8);
    KernelInput second({resources.function.id, 0}, resources.buffer.address + 8);
    auto first_pin = std::make_shared<int>(1);
    auto second_pin = std::make_shared<int>(2);
    const std::weak_ptr<const void> first_retained = first_pin;
    const std::weak_ptr<const void> second_retained = second_pin;
    first.kernel.lifetime = first_pin;
    second.kernel.lifetime = second_pin;
    success(session.deactivate());
    driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
    auto invalid = first.kernel;
    invalid.function = UINT64_MAX;
    bool requires_preparation = true;
    expect(!session.accept_async_kernel(1, invalid, &requires_preparation).accepted() &&
               !requires_preparation,
           "unknown original function rejects and resets preparation provenance");
    invalid = first.kernel;
    invalid.grid[0] = 0;
    expect(!session.accept_async_kernel(1, invalid).accepted(), "zero original grid rejects");
    invalid = first.kernel;
    invalid.parameters = std::span{first.parameters}.first(1);
    expect(!session.accept_async_kernel(1, invalid).accepted(), "wrong original ABI rejects");
    requires_preparation = true;
    const auto one = session.accept_async_kernel(1, first.kernel, &requires_preparation);
    expect(!requires_preparation, "direct-original async admission requires no preparation");
    const auto prefix = session.capture_command_frontier(1);
    const auto barrier = session.accept_external_barrier(0, std::make_shared<int>(3));
    const auto two = session.accept_async_kernel(2, second.kernel);
    const auto suffix = session.capture_command_frontier(2);
    const auto preparation = session.pending_preparation(1);
    const auto pending = session.pending_command(1);
    const bool has_direct_admission = !preparation.command && !preparation_demand(session) &&
                                      pending.status == CUDA_SUCCESS && pending.command &&
                                      !pending.command.description().command.requires_preparation;
    expect(
        one.accepted() && two.accepted() && prefix.status == CUDA_SUCCESS &&
            suffix.status == CUDA_SUCCESS && barrier.status == CUDA_SUCCESS &&
            session.accept_async_kernel(3, first.kernel).status == ExecutionAcceptStatus::kFull &&
            !session.accept_async_kernel(2, first.kernel).accepted() && !driver.launches &&
            !session.host_result(one.receipt) && !session.host_result(two.receipt) &&
            session.snapshot().submissions.retained_parameter_bytes == 24 &&
            driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
        "original admission validates and charges copied inputs without native context work");
    // Do not retain another caller-owned lifetime in the local invalid carrier.
    invalid.lifetime.reset();
    first_pin.reset();
    second_pin.reset();
    first.kernel.lifetime.reset();
    second.kernel.lifetime.reset();
    first.parameters.fill(std::byte{0xa5});
    second.parameters.fill(std::byte{0x5a});
    bool wrong_thread{};
    std::thread observer([&] {
        wrong_thread = !session.accept_async_kernel(3, first.kernel).accepted() &&
                       session.pending_command().status == CUDA_ERROR_INVALID_CONTEXT;
    });
    observer.join();
    expect(wrong_thread && !first_retained.expired() && !second_retained.expired(),
           "foreign callers cannot issue or discard original inputs");
    driver.next_context_query_status = CUDA_SUCCESS;
    success(session.activate());
    session.close();
    expect(session.accept_async_kernel(3, first.kernel).status == ExecutionAcceptStatus::kClosed,
           "closing seals new original admission without discarding accepted commands");
    // The old preparation fallback permits a clean negative, never a direct-path pass.
    if (preparation.command)
        expect(session.prepare_pending_kernel(preparation.command).status == CUDA_SUCCESS,
               "legacy original preparation remains available for complete negative cleanup");
    const auto cut = session.inspect_frontier_dependency(prefix.frontier);
    expect(cut.status == CUDA_ERROR_NOT_READY && cut.has_progress && !cut.preparation_sequence,
           "ready original source-cut progress needs no preparation authority");
    bool excludes_reentry = true;
    unsigned int reentries{};
    driver.on_launch = [&] {
        ++reentries;
        excludes_reentry = excludes_reentry &&
                           !session.accept_async_kernel(3, first.kernel).accepted() &&
                           session.pending_command().status == CUDA_ERROR_NOT_READY;
    };
    driver.query_status = CUDA_ERROR_NOT_READY;
    expect(
        session.progress_frontier_dependency(prefix.frontier).state ==
                CudaExecutionIssueState::kIssued &&
            driver.launches == 1 && session.acknowledge_result(one.receipt) &&
            session.query_frontier(prefix.frontier) == CUDA_ERROR_NOT_READY &&
            !first_retained.expired() && !second_retained.expired() &&
            session.release(resources.buffer.id).disposition == CudaSessionDisposition::kDeferred,
        "host return and ACK retain physical input and resource ownership");
    driver.query_status = CUDA_SUCCESS;
    expect(session.poll().status == CUDA_SUCCESS && first_retained.expired() &&
               session.query_frontier(prefix.frontier) == CUDA_SUCCESS &&
               session.query_frontier(suffix.frontier) == CUDA_ERROR_NOT_READY,
           "actual prefix retirement releases only its exact input lease");
    const auto suffix_preparation = session.pending_preparation(2);
    if (suffix_preparation.command)
        expect(session.prepare_pending_kernel(suffix_preparation.command).status == CUDA_SUCCESS,
               "legacy suffix preparation preserves the still-held callback cut");
    expect(session.issue_next().state != CudaExecutionIssueState::kIssued && driver.launches == 1 &&
               !second_retained.expired(),
           "ready original suffix cannot bypass an unstarted CPU callback barrier");
    expect(session.cancel_external_barrier(barrier.barrier) == CUDA_SUCCESS &&
               session.progress_frontier_dependency(suffix.frontier).state ==
                   CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(two.receipt) && session.poll().status == CUDA_SUCCESS &&
               second_retained.expired() && reentries == 2 && excludes_reentry &&
               driver.launches == 2,
           "known boundary settlement permits exactly one original suffix and actual retirement");
    driver.on_launch = {};
    success(session.download({resources.buffer.id, 0}, std::as_writable_bytes(std::span{words})));
    for (std::size_t index = 0; index != words.size(); ++index)
        expect(words[index] == (index < 2 || index >= 10 ? guard : 14U),
               "both original commands use copied arguments once and preserve every guard");
    cleanup(session, resources);
    expect(has_direct_admission,
           "already-original async commands must not require a preparation pass");
}

void original_async_native_error_survives_ack_and_physical_retirement() {
    reset_driver();
    CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000));
    const auto resources = initialize(session);
    KernelInput input({resources.function.id, 0}, resources.buffer.address);
    auto pin = std::make_shared<int>(7);
    const std::weak_ptr<const void> retained = pin;
    input.kernel.lifetime = pin;
    const auto accepted = session.accept_async_kernel(1, input.kernel);
    const auto frontier = session.capture_command_frontier(1);
    const auto preparation = session.pending_preparation();
    if (preparation.command)
        expect(session.prepare_pending_kernel(preparation.command).status == CUDA_SUCCESS,
               "legacy error control prepares before exercising original SDK failure");
    pin.reset();
    input.kernel.lifetime.reset();
    constexpr auto failure = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_launch_status = failure;
    const auto issued = session.issue(session.pending_command().command);
    const auto reply = session.host_result(accepted.receipt);
    expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
               issued.native_status == failure && reply && reply->status == failure &&
               reply->origin == ExecutionReplyOrigin::kNativeSdk && driver.launches == 1 &&
               session.query_frontier(frontier.frontier) == failure &&
               !frontier.frontier.retirement().is_retired &&
               session.acknowledge_result(accepted.receipt) && !retained.expired(),
           "late original SDK error is visible immediately without inventing physical stop");
    expect(session.issue_next().state != CudaExecutionIssueState::kIssued && driver.launches == 1 &&
               session.snapshot().submissions.retained_parameter_bytes == 12,
           "failed original cannot replay or lose its charged input after ACK");
    driver.query_status = CUDA_SUCCESS;
    (void)session.poll();
    const auto retired = frontier.frontier.retirement();
    expect(retired.is_retired && retired.status == failure && retained.expired() &&
               session.query_frontier(frontier.frontier) == failure &&
               session.inspect_frontier_retirement(frontier.frontier).status == CUDA_SUCCESS,
           "actual physical retirement releases input but preserves the exact asynchronous error");
    expect_values(resources, 0);
    cleanup(session, resources);
    expect(!preparation.command,
           "original async error reporting must not depend on deferred preparation");
}

void original_profile_modes_retain_preparation_and_measurement() {
    for (const bool is_learning : {false, true}) {
        reset_driver();
        driver.has_profile_function_metadata = true;
        CudaProfileConfig settings;
        settings.workers = {320};
        settings.warmup_calls = 0;
        settings.samples_per_candidate = 1;
        auto profiles = std::make_shared<CudaProfileCache>(settings, pair_clock, is_learning);
        CudaExecutionSession session({1, 99}, 101, pair_dispatch(), limits(1000), {}, {}, profiles);
        const auto resources = initialize(session);
        KernelInput input({resources.function.id, 0}, resources.buffer.address);
        bool requires_preparation{};
        const auto accepted = session.accept_async_kernel(1, input.kernel, &requires_preparation);
        const auto pending = session.pending_preparation();
        expect(accepted.accepted() && requires_preparation && pending.command &&
                   preparation_demand(session) && !session.pending_command().command &&
                   !session.host_result(accepted.receipt) && !driver.launches,
               "readonly and learning original profiles retain their exact preparation stage");
        input.parameters.fill(std::byte{0x7e});
        const auto prepared = session.prepare_pending_kernel(pending.command);
        expect(prepared.status == CUDA_SUCCESS && prepared.command,
               "profile original preparation preserves the same copied command");
        pair_now_ns = 1000;
        expect(session.issue(prepared.command).state == CudaExecutionIssueState::kIssued &&
                   driver.launches == 1 && session.acknowledge_result(accepted.receipt),
               "profiled original is actually issued once before any completion claim");
        pair_now_ns = 2000;
        expect(session.poll().status == CUDA_SUCCESS,
               "profile observation terminates only after actual event retirement");
        expect_values(resources, 7);
        cleanup(session, resources);
        expect(profiles->snapshot().samples == (is_learning ? 1U : 0U),
               "learning retains its exact original sample and readonly invents no sample");
    }
}

void deferred_admission_is_cpu_only_bounded_and_exact() {
    reset_driver();
    auto bounds = limits(1000);
    bounds.submissions.max_parameter_bytes = 80;
    CudaExecutionSession session({1, 99}, 101, dispatch(), bounds, {320});
    const auto resources = initialize(session);
    const auto blocking = session.create_stream(CU_STREAM_DEFAULT);
    const auto nonblocking = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(blocking);
    success(nonblocking);
    expect(session.stream_is_nonblocking(0) == false &&
               session.stream_is_nonblocking(blocking.id) == false &&
               session.stream_is_nonblocking(nonblocking.id) == true &&
               !session.stream_is_nonblocking(UINT64_MAX).has_value(),
           "CPU stream policy uses logical flags and rejects unknown IDs");
    KernelInput first({resources.function.id, 0}, resources.buffer.address);
    KernelInput second({resources.function.id, 0}, resources.buffer.address);
    const auto occupancy_before = occupancy_calls;
    const auto context_calls = driver.context_control_calls;
    success(session.deactivate());
    driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
    const auto reservation = session.reserve_command_frontier(1);
    bool requires_preparation{};
    const auto accepted = session.accept_async_kernel(1, first.kernel, &requires_preparation);
    expect(requires_preparation, "successful deferred admission reports exact provenance");
    const auto two = session.accept_async_kernel(2, second.kernel);
    const auto full = session.accept_async_kernel(3, first.kernel, &requires_preparation);
    expect(!requires_preparation, "failed deferred admission resets a stale preparation flag");
    const auto pending = session.pending_preparation(1);
    process_checks = 0;
    counts_process_checks = true;
    const auto composed = session.snapshot();
    const auto legacy_preparation = session.pending_preparation();
    counts_process_checks = false;
    const auto composed_checks = process_checks;
    process_checks = 0;
    counts_process_checks = true;
    const auto projected = session.scheduling_snapshot();
    counts_process_checks = false;
    const auto projected_checks = process_checks;
    const auto exact = session.reserved_command_frontier(reservation.reservation);
    expect(reservation.status == CUDA_SUCCESS && accepted.accepted() && two.accepted() &&
               full.status == ExecutionAcceptStatus::kFull && pending.command &&
               composed.has_pending_preparation && projected.has_pending_preparation &&
               legacy_preparation.command &&
               pending.command.description().command.requires_preparation &&
               exact.status == CUDA_SUCCESS &&
               session.query_frontier(exact.frontier) == CUDA_ERROR_NOT_READY &&
               !session.host_result(accepted.receipt) &&
               !session.acknowledge_result(accepted.receipt) &&
               session.snapshot().submissions.retained_parameter_bytes == 80 &&
               !session.snapshot().has_ready_work && !driver.launches &&
               occupancy_calls == occupancy_before &&
               driver.context_control_calls == context_calls &&
               driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
           "two cold-context admissions reserve exact frontiers and maximum ABI without SDK work");
    driver.next_context_query_status = CUDA_SUCCESS;
    first.parameters.fill(std::byte{0xa5});
    second.parameters.fill(std::byte{0x5a});
    bool wrong_thread{};
    std::thread observer([&] {
        wrong_thread =
            !preparation_demand(session) &&
            session.prepare_pending_kernel(pending.command).status == CUDA_ERROR_INVALID_CONTEXT;
    });
    observer.join();
    expect(wrong_thread, "opaque preparation authority remains bound to its original owner thread");
    const auto child = fork();
    expect(child >= 0, "fork inherited preparation-demand observation");
    if (!child)
        _exit(!preparation_demand(session) &&
                      session.pending_preparation().status == CUDA_ERROR_INVALID_CONTEXT
                  ? 0
                  : 1);
    int child_status{};
    expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "forked observers cannot publish preparation demand from inherited owners");
    success(session.activate());
    preparation_owner = &session;
    const auto prepared = session.prepare_pending_kernel(pending.command);
    preparation_owner = nullptr;
    expect(
        prepared.status == CUDA_SUCCESS &&
            prepared.disposition == CudaSessionKernelPreparationDisposition::kPrepared &&
            prepared.command && excludes_preparation_reentry &&
            !prepared.command.description().command.requires_preparation &&
            prepared.command.description().command.geometry_epoch >
                pending.command.description().command.geometry_epoch &&
            session.prepare_pending_kernel(pending.command).status != CUDA_SUCCESS &&
            !session.host_result(accepted.receipt) && !driver.launches,
        "in-place preparation refreshes only the exact command and never publishes a host result");
    driver.query_status = CUDA_ERROR_NOT_READY;
    expect(session.issue(prepared.command).state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(accepted.receipt) &&
               session.query_frontier(exact.frontier) == CUDA_ERROR_NOT_READY,
           "even a real SDK success and ACK do not complete the retained physical frontier");
    const auto second_pending = session.pending_preparation(2);
    expect(second_pending.command &&
               session.prepare_pending_kernel(second_pending.command).status == CUDA_SUCCESS,
           "later exact preparation remains possible while an earlier physical event is pending");
    expect(!preparation_demand(session) && !session.pending_preparation().command,
           "prepared physical commands do not remain preparation demand");
    expect(session.release(resources.buffer.id).disposition == CudaSessionDisposition::kDeferred,
           "free cannot bypass admitted or physically active command ownership");
    driver.query_status = CUDA_SUCCESS;
    expect(session.poll().status == CUDA_SUCCESS &&
               session.query_frontier(exact.frontier) == CUDA_SUCCESS,
           "first real physical retirement settles only its exact receipt frontier");
    expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(two.receipt) && session.poll().status == CUDA_SUCCESS,
           "second same-stream command issues once after its retained predecessor");
    expect_values(resources, 14);
    KernelInput third({resources.function.id, 0}, resources.buffer.address);
    const auto next = session.accept_async_kernel(3, third.kernel);
    session.close();
    expect(next.accepted() && preparation_demand(session) &&
               session.pending_preparation().command &&
               session.reject_command(next.receipt, CUDA_ERROR_INVALID_VALUE) &&
               session.acknowledge_result(next.receipt),
           "closing preserves accepted preparation demand; pressure consumed no sequence or slot");
    success(session.destroy_stream(blocking.id));
    success(session.destroy_stream(nonblocking.id));
    cleanup(session, resources);
    expect(!preparation_demand(session) &&
               session.pending_preparation().status == CUDA_ERROR_INVALID_CONTEXT,
           "shutdown owners cannot publish preparation demand");
    expect(projected_checks >= 1 && projected_checks <= 3 && composed_checks > projected_checks,
           "preparation demand must not repeat the legacy Session/dispatcher process checks");
}

void known_preparation_failure_and_initial_native_failure_remain_observable() {
    for (const bool preparation_failure : {true, false}) {
        reset_driver();
        CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), {320});
        const auto resources = initialize(session);
        KernelInput input({resources.function.id, 0}, resources.buffer.address);
        const auto accepted = session.accept_async_kernel(1, input.kernel);
        const auto frontier = session.capture_command_frontier(1);
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
               "failed async request is already reliably accepted with its own immutable frontier");
        constexpr auto failure = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        if (preparation_failure) occupancy_failure = failure;
        const auto prepared = session.prepare_pending_kernel(session.pending_preparation().command);
        if (preparation_failure) {
            const auto host = session.host_result(accepted.receipt);
            expect(prepared.status == failure &&
                       prepared.disposition == CudaSessionKernelPreparationDisposition::kRejected &&
                       host && host->status == failure &&
                       host->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       !driver.launches && !session.snapshot().is_quarantined,
                   "real occupancy failure is an exact known-before-issue rejection, not fallback");
        } else {
            expect(prepared.status == CUDA_SUCCESS && prepared.command,
                   "initial-native-error case first completes exact preparation");
            driver.query_status = CUDA_ERROR_NOT_READY;
            driver.next_launch_status = failure;
            const auto issued = session.issue(prepared.command);
            const auto host = session.host_result(accepted.receipt);
            expect(
                issued.native_status == failure && host && host->status == failure &&
                    host->origin == ExecutionReplyOrigin::kNativeSdk &&
                    session.snapshot().submissions.awaiting_physical_stop == 1,
                "initial asynchronous SDK error retains its actual tail and original host status");
            expect(session.acknowledge_result(accepted.receipt) &&
                       session.snapshot().submissions.retained_parameter_bytes == 40,
                   "error receipt ACK cannot release a still-unproved native tail");
            driver.query_status = CUDA_SUCCESS;
            (void)session.poll();
        }
        const auto retired = frontier.frontier.retirement();
        expect(session.query_frontier(frontier.frontier) == failure && retired.status == failure &&
                   retired.is_retired &&
                   session.inspect_frontier_dependency(frontier.frontier).has_retired_failure &&
                   session.inspect_frontier_retirement(frontier.frontier).status == CUDA_SUCCESS,
               "the exact async frontier retains the original error after proven retirement");
        if (preparation_failure)
            expect(session.acknowledge_result(accepted.receipt), "ACK actual known rejection once");
        expect_values(resources, 0);
        cleanup(session, resources);
    }
}

void source_cut_can_prepare_only_its_deferred_prefix() {
    reset_driver();
    auto bounds = limits(1000);
    bounds.submissions.max_commands = 3;
    // Keep actual transformable preparation separate from the direct-original
    // source-cut case. The external boundary is the third retained record.
    auto source_dispatch = dispatch();
    source_dispatch.host_allocate = source_cut_host_allocate;
    source_dispatch.host_release = source_cut_host_release;
    source_dispatch.context_destroy = source_cut_context_destroy;
    CudaExecutionSession session({1, 99}, 101, source_dispatch, bounds, {320});
    const auto resources = initialize(session);
    KernelInput input({resources.function.id, 0}, resources.buffer.address);
    const auto first = session.accept_async_kernel(1, input.kernel);
    const auto prefix = session.capture_command_frontier(1);
    const auto barrier = session.accept_external_barrier(0, std::make_shared<int>(7));
    const auto second = session.accept_async_kernel(2, input.kernel);
    const auto suffix = session.capture_command_frontier(2);
    expect(first.accepted() && second.accepted() && prefix.status == CUDA_SUCCESS &&
               suffix.status == CUDA_SUCCESS && barrier.status == CUDA_SUCCESS,
           "source fence separates actual retained async prefix from ordinary suffix");
    const auto cut = session.inspect_frontier_dependency(prefix.frontier);
    expect(cut.status == CUDA_ERROR_NOT_READY && cut.has_progress &&
               cut.preparation_sequence == 1 && !driver.launches,
           "dependency observation exposes only exact closure-local preparation authority");
    (void)session.progress_frontier_dependency(prefix.frontier);
    expect(!driver.launches && !session.host_result(first.receipt),
           "dependency progress cannot accidentally issue an unprepared original kernel");
    const auto prepared = session.prepare_pending_kernel(session.pending_preparation(1).command);
    expect(prepared.status == CUDA_SUCCESS &&
               !session.inspect_frontier_dependency(prefix.frontier).preparation_sequence &&
               session.progress_frontier_dependency(prefix.frontier).state ==
                   CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(first.receipt) && session.poll().status == CUDA_SUCCESS,
           "fresh source-cut preparation enables a later exact issue and actual prefix retirement");
    expect(session.query_frontier(prefix.frontier) == CUDA_SUCCESS &&
               session.query_frontier(suffix.frontier) == CUDA_ERROR_NOT_READY &&
               driver.launches == 1,
           "settled prefix does not manufacture callback or suffix completion");
    const auto prepared_suffix =
        session.prepare_pending_kernel(session.pending_preparation(2).command);
    expect(prepared_suffix.status == CUDA_SUCCESS && !prepared_suffix.command &&
               session.issue_next().state != CudaExecutionIssueState::kIssued &&
               driver.launches == 1,
           "preparation behind a CPU callback boundary grants no native issue permission");
    expect(session.cancel_external_barrier(barrier.barrier) == CUDA_SUCCESS &&
               session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(second.receipt) && session.poll().status == CUDA_SUCCESS,
           "only explicit known CPU-boundary settlement releases the accepted suffix");
    expect_values(resources, 14);
    cleanup(session, resources);
    expect(!is_source_cut_host_live && driver.host_frees == 1,
           "the separate transformed source control releases its full three-slot arena once");
}

void continuation_prelaunch_failure_is_retained_in_a_child() {
    const auto child = fork();
    expect(child >= 0, "fork incomplete asynchronous continuation retention control");
    if (!child) {
        try {
            reset_driver();
            driver.next_claimed = 3;
            CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), {320});
            const auto resources = initialize(session);
            KernelInput input({resources.function.id, 0}, resources.buffer.address);
            const auto accepted = session.accept_async_kernel(1, input.kernel);
            const auto frontier = session.capture_command_frontier(1);
            const auto prepared =
                session.prepare_pending_kernel(session.pending_preparation().command);
            expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
                       prepared.status == CUDA_SUCCESS && prepared.command &&
                       session.issue(prepared.command).state == CudaExecutionIssueState::kIssued &&
                       session.request_retreat(1) == CUDA_SUCCESS &&
                       session.poll().status == CUDA_SUCCESS,
                   "a real first physical generation stops after only three logical blocks");
            const auto progress = session.persistent_progress(1);
            const auto original = session.host_result(accepted.receipt);
            expect(progress && progress->completed_blocks == 3 &&
                       progress->physical_launches == 1 && progress->physical_stops == 1 &&
                       original && original->status == CUDA_SUCCESS &&
                       original->origin == ExecutionReplyOrigin::kNativeSdk &&
                       session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY &&
                       session.snapshot().submissions.paused == 1 && !preparation_demand(session) &&
                       !session.pending_preparation().command,
                   "SDK success and physical pause leave the original logical command incomplete");
            const auto cleanup_before = driver.cleanup_calls();
            const auto events_before = driver.next_event;
            next_event_failure = CUDA_ERROR_OUT_OF_MEMORY;
            const auto failed = session.resume(1);
            const auto host = session.host_result(accepted.receipt);
            expect(
                failed.state == CudaExecutionIssueState::kRejected &&
                    failed.native_status == CUDA_ERROR_OUT_OF_MEMORY && host &&
                    host->status == original->status && host->origin == original->origin &&
                    session.query_frontier(frontier.frontier) == CUDA_ERROR_OUT_OF_MEMORY &&
                    !frontier.frontier.retirement().is_retired &&
                    session.snapshot().is_quarantined &&
                    session.snapshot().submissions.retained_parameter_bytes == 40 &&
                    session.snapshot().submissions.paused == 1 && driver.launches == 1 &&
                    driver.next_event == events_before && driver.cleanup_calls() == cleanup_before,
                "continuation event OOM persists exact failure without overwriting host result or "
                "inventing final retirement");
            const auto retry = session.resume(1);
            expect(
                retry.state == CudaExecutionIssueState::kError && driver.launches == 1 &&
                    session.acknowledge_result(accepted.receipt) &&
                    session.snapshot().submissions.retained_parameter_bytes == 40,
                "neither direct retry nor ACK can execute or release the retained failed suffix");
            for (unsigned int block = 0; block != 8; ++block) {
                std::uint32_t value{};
                std::memcpy(
                    &value,
                    device_bytes(resources.buffer.address + block * sizeof(value), sizeof(value)),
                    sizeof(value));
                expect(
                    value == (block < 3 ? 7U : 0U),
                    "failed continuation neither replays the completed prefix nor runs the suffix");
            }
            // Failure retention is intentional; no successful shutdown is claimed.
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Async continuation child: " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "continuation error preserves exact observation and incomplete ownership");
}

void wrong_context_preparation_is_retained_in_a_child() {
    const auto child = fork();
    expect(child >= 0, "fork intentional uncertain-context retention control");
    if (!child) {
        try {
            reset_driver();
            CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), {320});
            const auto resources = initialize(session);
            KernelInput input({resources.function.id, 0}, resources.buffer.address);
            const auto accepted = session.accept_async_kernel(1, input.kernel);
            const auto frontier = session.capture_command_frontier(1);
            const auto pending = session.pending_preparation();
            success(session.deactivate());
            const auto cleanup_before = driver.cleanup_calls();
            const auto failed = session.prepare_pending_kernel(pending.command);
            expect(accepted.accepted() && failed.status == CUDA_ERROR_INVALID_CONTEXT &&
                       failed.disposition == CudaSessionKernelPreparationDisposition::kUncertain &&
                       session.snapshot().is_quarantined && !preparation_demand(session) &&
                       session.pending_preparation().status == CUDA_ERROR_UNKNOWN &&
                       !session.host_result(accepted.receipt) &&
                       session.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_CONTEXT &&
                       session.snapshot().submissions.retained_parameter_bytes == 40 &&
                       !driver.launches && driver.cleanup_calls() == cleanup_before,
                   "unproved context ownership records a failure but cannot settle or refund work");
            // Intentionally retained uncertain ownership is not a healthy-cleanup claim.
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Async uncertainty child: " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "intentional uncertainty control retains rather than inventing physical retirement");
}
}  // namespace

int main() {
    try {
        queue_precharges_final_abi_without_publishing_a_result();
        original_admission_preserves_sources_and_leases_without_preparation();
        original_async_native_error_survives_ack_and_physical_retirement();
        original_profile_modes_retain_preparation_and_measurement();
        deferred_admission_is_cpu_only_bounded_and_exact();
        known_preparation_failure_and_initial_native_failure_remain_observable();
        source_cut_can_prepare_only_its_deferred_prefix();
        continuation_prelaunch_failure_is_retained_in_a_child();
        wrong_context_preparation_is_retained_in_a_child();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Async admission backend: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
