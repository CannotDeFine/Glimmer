#include "glimmer/control/cooperative_launch_gate.h"
#include "glimmer/control/async_cooperative_launch_gate.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"

#include <unistd.h>

#include <chrono>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
template <typename T>
decltype(auto) checked(T&& value) noexcept {
    if (!value.has_value()) std::abort();
    return *value;
}

using namespace glimmer::control;

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void handoff(CooperativeLaunchGate& gate) {
    const auto acquired = gate.submit_and_claim();
    require(acquired.task_id != 0 && acquired.claim.status == CooperativeClaimStatus::kGranted &&
                acquired.claim.execution.task_id == acquired.task_id &&
                acquired.claim.execution.generation != 0,
            "combined submission must return a generation-bound first grant");
    const auto denied = gate.submit_and_claim();
    require(denied.task_id != 0 && denied.task_id != acquired.task_id &&
                denied.claim.status == CooperativeClaimStatus::kPending &&
                !gate.cancel_pending(acquired.task_id) && gate.cancel_pending(denied.task_id) &&
                gate.complete(acquired.claim.execution),
            "denied first claim must retain pending identity, not steal running capacity");
    require(gate.submit_and_claim({.work_units = 0}).task_id == 0,
            "invalid combined submission must reject");
    const auto low = gate.submit({.relative_priority = 1});
    require(low.has_value(), "submit must accept low without implicit claim");
    const auto running = gate.claim(checked(low));
    require(running.status == CooperativeClaimStatus::kGranted &&
                running.execution.task_id == checked(low) && running.execution.generation != 0,
            "claim must grant an execution generation");
    require(gate.claim(checked(low)).status != CooperativeClaimStatus::kGranted,
            "a published physical execution grant must not be issued twice");
    const auto high = gate.submit({.relative_priority = 9});
    require(high && gate.claim(*high).status == CooperativeClaimStatus::kPending &&
                gate.retreat_requested(running.execution) == true,
            "high arrival requests retreat but cannot steal physical capacity");
    const TaskExecutionIdentity stale{checked(low), running.execution.generation + 1};
    require(!gate.suspend(stale) && !gate.complete(stale) && !gate.retreat_requested(stale) &&
                gate.heartbeat(running.execution),
            "stale reports cannot release running capacity");
    // The backend would supply a physical-stop event before this report.
    require(gate.suspend(running.execution), "proven stop must return low to pending");
    require(gate.claim(checked(low)).status == CooperativeClaimStatus::kPending,
            "suspension alone cannot bypass the high priority task");
    const auto priority = gate.claim(checked(high));
    require(
        priority.status == CooperativeClaimStatus::kGranted && gate.complete(priority.execution),
        "priority task must complete first");
    const auto resumed = gate.claim(checked(low));
    require(resumed.status == CooperativeClaimStatus::kGranted &&
                resumed.execution.generation != running.execution.generation &&
                !gate.complete(running.execution) && !gate.suspend(running.execution) &&
                gate.complete(resumed.execution),
            "resume must preserve task but reject old physical generation");
    const auto pending = gate.submit();
    require(pending && gate.cancel_pending(*pending) && !gate.cancel_pending(*pending),
            "never-issued pending work may cancel once");
    const auto failed = gate.submit();
    require(failed.has_value(), "failure fixture must submit");
    const auto failed_claim = gate.claim(checked(failed));
    require(failed_claim.status == CooperativeClaimStatus::kGranted &&
                !gate.cancel_pending(checked(failed)) &&
                !gate.cancel_stopped(failed_claim.execution) && gate.fail(failed_claim.execution),
            "running cancel must reject until physical ownership is stopped");
    const auto cancelled = gate.submit();
    require(cancelled.has_value(), "stopped cancellation fixture must submit");
    const auto stopped = gate.claim(checked(cancelled));
    require(stopped.status == CooperativeClaimStatus::kGranted && gate.suspend(stopped.execution) &&
                gate.heartbeat(stopped.execution) && !gate.cancel_pending(checked(cancelled)) &&
                gate.cancel_stopped(stopped.execution),
            "suspended work requires generation-bound cancellation");
    require(
        gate.claim(0).status == CooperativeClaimStatus::kFailed && !gate.submit({.work_units = 0}),
        "invalid ids and work units must reject");
}

void combined_completion(CooperativeLaunchGate& gate) {
    const auto first = gate.submit_and_claim();
    require(first.claim.status == CooperativeClaimStatus::kGranted, "handoff must start granted");
    const auto stale =
        gate.complete_and_submit_and_claim({first.task_id, first.claim.execution.generation + 1});
    require(!stale.is_completed && stale.submission.task_id == 0 &&
                gate.heartbeat(first.claim.execution),
            "stale completion cannot admit new work or release old capacity");
    const auto second = gate.complete_and_submit_and_claim(first.claim.execution);
    require(second.is_completed && second.submission.task_id != 0 &&
                second.submission.task_id != first.task_id &&
                second.submission.claim.status == CooperativeClaimStatus::kGranted &&
                second.submission.claim.execution.task_id == second.submission.task_id &&
                second.submission.claim.execution.generation != 0,
            "combined report must acknowledge old identity and grant a distinct task");
    const auto replay = gate.complete_and_submit_and_claim(first.claim.execution);
    require(!replay.is_completed && replay.submission.task_id == 0,
            "replayed completion cannot duplicate new admission");
    const auto high = gate.submit({.relative_priority = 9});
    const auto pending = gate.complete_and_submit_and_claim(second.submission.claim.execution);
    require(high && pending.is_completed && pending.submission.task_id != 0 &&
                pending.submission.claim.status == CooperativeClaimStatus::kPending &&
                pending.submission.claim.execution.generation == 0,
            "combined transport cannot bypass higher priority pending work");
    require(gate.cancel_pending(checked(high)), "unissued priority fixture can cancel");
    const auto last = gate.claim(pending.submission.task_id);
    require(last.status == CooperativeClaimStatus::kGranted,
            "denied new task retains claim identity");
    const auto rejected = gate.complete_and_submit_and_claim(last.execution, {.work_units = 0});
    require(rejected.is_completed && rejected.submission.task_id == 0 &&
                rejected.submission.claim.status == CooperativeClaimStatus::kFailed,
            "locally invalid new work cannot discard the valid standalone completion");
}

class RemoteService final {
   public:
    std::atomic<bool> is_registration_blocked{false};
    std::atomic<bool> is_registration_entered{false};
    TaskExecutionEndpoint endpoint;
    std::string directory;
    std::string socket;
    UnixSocketTaskExecutionServer server;
    std::jthread acceptor;
    explicit RemoteService(bool is_bound = true, bool should_block_registration = false)
        : is_registration_blocked(should_block_registration),
          endpoint(std::make_unique<TaskExecutionAdmissionService>(
                       64,
                       glimmer::core::SchedulerOptions{
                           .scheduling_policy = glimmer::core::SchedulingPolicy::kPriority},
                       std::chrono::milliseconds{0}, is_bound),
                   registered, rollback, this),
          directory(make_directory()),
          socket(directory + "/control.sock"),
          server(endpoint, {.socket_path = socket, .allowed_uid = std::nullopt}) {
        require(server.start(), "remote service must bind");
        acceptor = std::jthread([this](const std::stop_token& stop) {
            while (!stop.stop_requested()) {
                const auto status = server.serve_one_for(10);
                if (status == UnixSocketServeStatus::kError) {
                    return;
                }
            }
        });
    }
    ~RemoteService() {
        release_registration();
        acceptor.request_stop();
        acceptor.join();
        server.stop();
        if (::rmdir(directory.c_str()) != 0) {
            std::terminate();
        }
    }
    RemoteService(const RemoteService&) = delete;
    RemoteService& operator=(const RemoteService&) = delete;
    RemoteService(RemoteService&&) = delete;
    RemoteService& operator=(RemoteService&&) = delete;
    void release_registration() noexcept {
        is_registration_blocked.store(false);
        is_registration_blocked.notify_all();
    }
    bool wait_for_registration() const {
        for (int i = 0; i < 3000 && !is_registration_entered.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        return is_registration_entered.load();
    }

   private:
    static bool registered(void* context, glimmer::core::TaskId) noexcept {
        auto& service = *static_cast<RemoteService*>(context);
        service.is_registration_entered.store(true);
        while (service.is_registration_blocked.load()) service.is_registration_blocked.wait(true);
        return true;
    }
    static void rollback(void*, glimmer::core::TaskId) noexcept {}
    static std::string make_directory() {
        char path[] = "/tmp/glimmer-cooperative-gate-XXXXXX";
        require(::mkdtemp(path) != nullptr, "remote fixture directory must create");
        return path;
    }
};

void async_handoff() {
    LaunchGateOptions options;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    AsyncCooperativeLaunchGate gate(options);
    const auto acquired = gate.submit_and_claim({.relative_priority = 1}).get();
    const auto low = acquired.task_id;
    require(low != 0, "asynchronous combined submission must resolve");
    const auto running = acquired.claim;
    const auto high = gate.submit({.relative_priority = 9}).get();
    require(high && running.status == CooperativeClaimStatus::kGranted &&
                gate.claim(*high).get().status == CooperativeClaimStatus::kPending &&
                gate.retreat_requested(running.execution).get() == true &&
                gate.suspend(running.execution).get(),
            "typed futures must preserve physical-stop handoff ordering");
    const auto priority = gate.claim(checked(high)).get();
    require(priority.status == CooperativeClaimStatus::kGranted &&
                gate.complete(priority.execution).get(),
            "async high task must complete first");
    const auto resumed = gate.claim(low).get();
    require(resumed.status == CooperativeClaimStatus::kGranted &&
                resumed.execution.generation != running.execution.generation,
            "async resume must return new physical generation");
    auto terminal = gate.complete(resumed.execution);
    gate.shutdown();
    require(terminal.valid() && terminal.get() && !gate.submit().valid() &&
                !gate.claim(1).valid() && !gate.submit_and_claim().valid(),
            "close must finish accepted reports and reject new mutations");
}

void measured_claims_preserve_results() {
    AsyncCooperativeLaunchGate gate;
    const auto unmeasured = gate.submit_and_claim().get();
    require(unmeasured.task_id != 0 && unmeasured.claim.timing.enqueued_nanoseconds == 0 &&
                unmeasured.claim.timing.started_nanoseconds == 0 &&
                unmeasured.claim.timing.finished_nanoseconds == 0 &&
                gate.complete(unmeasured.claim.execution).get(),
            "disabled timing has zero metadata and unchanged grant cleanup");
    const auto granted = gate.submit_and_claim({.should_measure_timing = true}).get();
    const auto pending = gate.submit_and_claim({.should_measure_timing = true}).get();
    const auto denied = gate.claim(pending.task_id, true).get();
    for (const auto& claim : {granted.claim, pending.claim, denied}) {
        require(claim.timing.enqueued_nanoseconds > 0 &&
                    claim.timing.enqueued_nanoseconds <= claim.timing.started_nanoseconds &&
                    claim.timing.started_nanoseconds <= claim.timing.finished_nanoseconds,
                "timing follows actual mailbox and operation order");
    }
    require(granted.claim.status == CooperativeClaimStatus::kGranted &&
                pending.claim.status == CooperativeClaimStatus::kPending &&
                denied.status == CooperativeClaimStatus::kPending &&
                gate.cancel_pending(pending.task_id).get(),
            "measured pending requests retain identity and running capacity");
    auto terminal = gate.complete(granted.claim.execution);
    gate.shutdown();
    require(terminal.valid() && terminal.get() && !gate.claim(1, true).valid() &&
                !gate.submit_and_claim({.should_measure_timing = true}).valid(),
            "measurement cannot change accepted drain or closed rejection");
}

void delayed_remote_mailbox() {
    RemoteService service(true, true);
    AsyncCooperativeLaunchGate gate({.control_socket = service.socket}, 1);
    auto first = gate.submit_and_claim({.should_measure_timing = true});
    require(first.valid(), "delayed remote submit must be accepted");
    const bool has_entered = service.wait_for_registration();
    auto second = gate.claim(99);
    auto rejected = gate.claim(100);
    const bool has_bounded_mailbox =
        has_entered && second.valid() && !rejected.valid() &&
        first.wait_for(std::chrono::milliseconds::zero()) == std::future_status::timeout;
    const auto released_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
    service.release_registration();
    const auto task = first.get();
    const auto nonexistent = second.valid() ? second.get() : CooperativeClaim{};
    require(has_bounded_mailbox && task.task_id != 0 &&
                task.claim.status == CooperativeClaimStatus::kGranted &&
                task.claim.timing.enqueued_nanoseconds <= task.claim.timing.started_nanoseconds &&
                task.claim.timing.started_nanoseconds <=
                    static_cast<std::uint64_t>(released_nanoseconds) &&
                task.claim.timing.finished_nanoseconds >=
                    static_cast<std::uint64_t>(released_nanoseconds) &&
                nonexistent.status == CooperativeClaimStatus::kFailed,
            "submitter must return while remote request is blocked; mailbox must reject overflow");
    // No CUDA work was issued by this fixture; the granted operation is stopped.
    auto cancellation = gate.fail(task.claim.execution);
    gate.shutdown();
    const auto remaining = service.endpoint.lease_status();
    require(cancellation.valid() && cancellation.get() && remaining &&
                remaining->queued_tasks == 0 && remaining->running_tasks == 0,
            "accepted asynchronous cancellation must drain during shutdown");
}

void progress_notifications() {
    AsyncCooperativeLaunchGate::ProgressSignal signal;
    const auto immediate = std::chrono::milliseconds::zero();
    require(!signal.wait_for(immediate), "an empty signal must not invent progress");
    signal.notify();
    signal.notify();
    require(signal.wait_for(immediate) && !signal.wait_for(immediate),
            "notifications before wait persist and coalesce without stale wakeups");
    require(!signal.wait_for(std::chrono::microseconds{100}),
            "a submillisecond timeout is accepted without inventing progress");
    signal.notify();
    require(signal.wait_for(std::chrono::microseconds{100}) && !signal.wait_for(immediate),
            "submillisecond waits consume each coalesced hint once");

    AsyncCooperativeLaunchGate bounded({}, 2);
    auto first = bounded.subscribe_progress();
    const auto second = bounded.subscribe_progress();
    require(first && second && !bounded.subscribe_progress(),
            "each bounded subscriber has independent owned wakeup state");
    first.reset();
    const auto replacement = bounded.subscribe_progress();
    require(replacement && !bounded.subscribe_progress(),
            "expired subscriptions are reclaimed without expanding the bound");
    auto result = bounded.claim(999);
    require(second->wait_for(std::chrono::seconds{3}) &&
                replacement->wait_for(std::chrono::seconds{3}) &&
                result.wait_for(immediate) == std::future_status::ready &&
                result.get().status == CooperativeClaimStatus::kFailed,
            "all observers see result publication, including rejected control operations");
    bounded.shutdown();
    require(!bounded.subscribe_progress() && !bounded.claim(999).valid() &&
                !second->wait_for(immediate) && !replacement->wait_for(immediate),
            "closed rejection cannot publish a fictitious executed request");
    replacement->notify();
    require(replacement->wait_for(immediate), "signals remain CPU-only after gate shutdown");

    RemoteService service(true, true);
    AsyncCooperativeLaunchGate remote({.control_socket = service.socket});
    const auto observer = remote.subscribe_progress();
    auto submission = remote.submit();
    const bool has_entered = service.wait_for_registration();
    const bool was_unpublished = observer && !observer->wait_for(immediate) &&
                                 submission.wait_for(immediate) == std::future_status::timeout;
    service.release_registration();
    require(has_entered && was_unpublished && observer->wait_for(std::chrono::seconds{3}) &&
                submission.wait_for(immediate) == std::future_status::ready,
            "a delayed control operation wakes only after its future is published");
    const auto task = submission.get();
    require(task.has_value(), "delayed submission must return its actual task");
    auto cancellation = remote.cancel_pending(checked(task));
    remote.shutdown();
    require(cancellation.get() && observer->wait_for(immediate) && !observer->wait_for(immediate),
            "shutdown drains and publishes the accepted terminal report before joining");
}
}  // namespace

int main() {
    try {
        LaunchGateOptions local_options;
        local_options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
        CooperativeLaunchGate gate(local_options);
        require(!gate.remote_mode(), "local gate must remain local");
        handoff(gate);
        combined_completion(gate);
        for (unsigned int count = 0; count < 2000; ++count) {
            const auto task = gate.submit();
            require(task.has_value(), "long history task must submit");
            require(gate.complete(gate.claim(checked(task)).execution),
                    "long history task must complete");
        }
        const auto stats = gate.local_stats();
        require(stats && stats->total_task_count == 0 && stats->quota.allocated_bytes == 0 &&
                    stats->quota.reserved_bytes == 0,
                "terminal local history and quota must remain bounded");
        auto invalid_options = local_options;
        invalid_options.tenant_weight = 0;
        CooperativeLaunchGate invalid(invalid_options);
        require(!invalid.submit(), "invalid trusted weight must reject");
        auto saturated_options = local_options;
        saturated_options.task_priority = std::numeric_limits<std::uint32_t>::max();
        CooperativeLaunchGate saturated(saturated_options);
        const auto task =
            saturated.submit({.relative_priority = std::numeric_limits<std::uint32_t>::max()});
        require(task && saturated.complete(saturated.claim(*task).execution),
                "priority arithmetic must not overflow");
        {
            RemoteService service;
            CooperativeLaunchGate remote({.control_socket = service.socket});
            require(remote.remote_mode() && !remote.local_stats(),
                    "remote mode cannot fake local stats");
            handoff(remote);
            combined_completion(remote);
            const auto snapshot = service.endpoint.lease_status();
            require(snapshot && snapshot->running_tasks == 0 && snapshot->queued_tasks == 0 &&
                        snapshot->pending_cooperative == 0 && snapshot->active_cooperative == 0,
                    "remote handoff must release only terminal or proven stopped capacity");
        }
        {
            RemoteService legacy(false);
            CooperativeLaunchGate remote({.control_socket = legacy.socket});
            require(remote.submit_and_claim().task_id == 0 && legacy.endpoint.lease_status() &&
                        checked(legacy.endpoint.lease_status()).queued_tasks == 0,
                    "unsupported combined operation must not retry legacy submission");
            const auto legacy_task = remote.submit();
            require(!legacy_task, "an unbound service cannot silently grant cooperative execution");
        }
        CooperativeLaunchGate unavailable(
            {.control_socket = "/nonexistent/glimmer-cooperative-gate.sock"});
        require(!unavailable.submit() && unavailable.submit_and_claim().task_id == 0 &&
                    unavailable.claim(1).status == CooperativeClaimStatus::kFailed &&
                    !unavailable.complete({1, 1}) && unavailable.remote_mode(),
                "transport failure must not switch policy or retry locally");
        async_handoff();
        measured_claims_preserve_results();
        delayed_remote_mailbox();
        progress_notifications();
        std::cout << "Cooperative launch gate checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
