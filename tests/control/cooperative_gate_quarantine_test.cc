#include "glimmer/control/cooperative_launch_gate.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#include "internal/task_execution_endpoint_test_hooks.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer::control;

void expect(bool value, const char* message) {
    if (!value) {
        std::cerr << "Test failed: " << message << '\n';
        std::abort();
    }
}

struct Resources {
    std::atomic<bool> blocked{false};
    std::atomic<bool> entered{false};
    std::atomic<bool> mismatch_epoch{false};
    std::atomic<bool> error_epoch{false};
    std::atomic<unsigned> registered{0};
    std::atomic<unsigned> rolled_back{0};
};

bool register_resource(void* context, glimmer::core::TaskId) noexcept {
    auto& resources = *static_cast<Resources*>(context);
    resources.registered.fetch_add(1);
    resources.entered.store(true);
    while (resources.blocked.load()) resources.blocked.wait(true);
    if (resources.mismatch_epoch.exchange(false)) {
        testing::mismatch_next_epoch_grant_extent();
    }
    if (resources.error_epoch.exchange(false)) {
        testing::error_next_epoch_grant();
    }
    return true;
}

void rollback_resource(void* context, glimmer::core::TaskId) noexcept {
    static_cast<Resources*>(context)->rolled_back.fetch_add(1);
}

void wait_for_registrar(Resources& resources) {
    for (int i = 0; i < 3000 && !resources.entered.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    expect(resources.entered.load(), "request reaches registrar");
}

void release_registrar(Resources& resources) {
    resources.blocked.store(false);
    resources.blocked.notify_all();
}

void exercise(bool acquire) {
    char directory[] = "/tmp/glimmer-gate-quarantine-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create private directory");
    const std::string path = std::string(directory) + "/server.sock";
    Resources resources;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), true);
    TaskExecutionEndpoint endpoint(std::move(admission), register_resource, rollback_resource,
                                   &resources);
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 2000});
    expect(server.start(), "start execution server");
    std::atomic<bool> running{true};
    std::thread acceptor([&] {
        while (running.load()) {
            expect(server.serve_one_for(20) != UnixSocketServeStatus::kError, "serve client");
        }
    });
    {
        CooperativeLaunchGate gate({.control_socket = path});
        const auto normal = gate.submit_and_claim();
        expect(normal.claim.status == CooperativeClaimStatus::kGranted &&
                   gate.complete(normal.claim.execution),
               "normal versioned round trip");
        const auto queued = gate.submit();
        expect(queued.has_value(), "ordinary versioned submission accepted");
        const auto claimed = gate.claim(queued.value_or(0));
        expect(
            claimed.status == CooperativeClaimStatus::kGranted && gate.complete(claimed.execution),
            "ordinary submission claims and completes");
        resources.blocked.store(true);
        resources.entered.store(false);
        auto result = std::async(std::launch::async, [&] {
            return acquire ? gate.submit_and_claim().task_id : gate.submit().value_or(0);
        });
        wait_for_registrar(resources);
        expect(result.wait_for(std::chrono::seconds{4}) == std::future_status::ready &&
                   result.get() == 0 && gate.remote_outcome_uncertain(),
               "lost reply is quarantined, not treated as clean rejection");
        release_registrar(resources);
        // Wait for the server-side operation to publish before inspecting it.
        for (int i = 0; i < 200; ++i) {
            const auto status = endpoint.lease_status();
            if (status && status->queued_tasks + status->running_tasks == 1) break;
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        const auto before = endpoint.lease_status();
        expect(before && before->queued_tasks + before->running_tasks == 1 &&
                   resources.registered.load() == 3 && resources.rolled_back.load() == 0,
               "server retains the one possibly published task and its resource");
        expect(!gate.submit() && gate.submit_and_claim().task_id == 0 &&
                   resources.registered.load() == 3,
               "quarantined gate cannot submit or replay");
        expect(endpoint.begin_shutdown(), "begin shutdown");
        const auto remaining = endpoint.lease_status();
        expect(remaining && remaining->queued_tasks + remaining->running_tasks == 1,
               "shutdown cannot falsely report a drained ledger");
    }
    running.store(false);
    acceptor.join();
    server.stop();
    // Registrar has no external resource; no backend work was ever issued.
    expect(::rmdir(directory) == 0, "remove private directory");
}

void native_bad_reply_quarantines_concurrent_submit(bool error_reply) {
    char directory[] = "/tmp/glimmer-gate-mismatch-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create mismatch directory");
    const std::string path = std::string(directory) + "/server.sock";
    const NativeEpochExtent extent{2, 1, 1, 1000000};
    Resources resources;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        100, glimmer::core::SchedulerOptions{}, std::chrono::milliseconds::zero(), true, nullptr,
        nullptr, nullptr, nullptr, extent);
    TaskExecutionEndpoint endpoint(std::move(admission), register_resource, rollback_resource,
                                   &resources);
    UnixSocketTaskExecutionServer server(endpoint, {.socket_path = path, .io_timeout_ms = 2000});
    expect(server.start(), "start mismatch server");
    std::atomic<bool> running{true};
    std::thread acceptor([&] {
        while (running.load()) {
            expect(server.serve_one_for(20) != UnixSocketServeStatus::kError, "serve client");
        }
    });
    {
        CooperativeLaunchGate gate({.control_socket = path}, extent);
        const auto first = gate.acquire_native_epoch({.extent = extent});
        const auto waiting = gate.acquire_native_epoch({.extent = extent});
        expect(first.status == NativeEpochClaimStatus::kGranted &&
                   waiting.status == NativeEpochClaimStatus::kPending &&
                   gate.finish_native_epoch(first.grant, 0, false),
               "native grant and pending admission round trip");
        const auto next = gate.claim_native_epoch(waiting.grant);
        expect(next.status == NativeEpochClaimStatus::kGranted &&
                   gate.finish_native_epoch(next.grant, 0, false),
               "pending native admission can claim and complete");
        resources.blocked.store(true);
        resources.entered.store(false);
        resources.mismatch_epoch.store(!error_reply);
        resources.error_epoch.store(error_reply);
        auto native = std::async(std::launch::async,
                                 [&] { return gate.acquire_native_epoch({.extent = extent}); });
        wait_for_registrar(resources);
        auto concurrent = std::async(std::launch::async, [&] { return gate.submit(); });
        release_registrar(resources);
        expect(native.get().status == NativeEpochClaimStatus::kUncertain && !concurrent.get() &&
                   gate.remote_outcome_uncertain(),
               "mismatched native reply quarantines a concurrent ordinary request");
        expect(resources.registered.load() == 3 && resources.rolled_back.load() == 0,
               "mismatch preserves the sole accepted resource");
        const auto status = endpoint.lease_status();
        expect(status && status->active_native == 1, "native ownership remains active");
        expect(endpoint.begin_shutdown(), "stop new work");
    }
    running.store(false);
    acceptor.join();
    server.stop();
    // The fake registrar owns no external resource; no backend work was issued.
    expect(::rmdir(directory) == 0, "remove mismatch directory");
}
}  // namespace

int main() {
    exercise(false);
    exercise(true);
    native_bad_reply_quarantines_concurrent_submit(false);
    native_bad_reply_quarantines_concurrent_submit(true);
}
