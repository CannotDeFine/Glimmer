#include "internal/allocation_registry.h"

#include <cstdlib>
#include <array>
#include <iostream>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using glimmer::interceptor::AllocationIdentity;
using glimmer::interceptor::AllocationProvenance;
using glimmer::interceptor::AllocationRegistry;
using glimmer::interceptor::AllocationScope;
using glimmer::interceptor::AsyncStreamIdentity;
using glimmer::interceptor::VmmAllocationIdentity;

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }

    std::cerr << message << '\n';
    return false;
}

bool test_independent_allocation_provenance() {
    AllocationRegistry registry;
    const AllocationIdentity key{0x1000, reinterpret_cast<CUcontext>(0x3000), 0};
    bool passed = expect(registry.record(key, 128, AllocationScope::kContextBound,
                                         AllocationProvenance::kIndependentDevice),
                         "ordinary record failed");
    passed &= expect(registry.has_live_independent_allocation(key, 128) &&
                         registry.has_live_independent_allocation(key, 1) &&
                         !registry.has_live_independent_allocation(key, 0) &&
                         !registry.has_live_independent_allocation(key, 129),
                     "exact allocation extent not enforced");
    passed &= expect(!registry.has_live_independent_allocation({0x1001, key.context, 0}, 1) &&
                         !registry.has_live_independent_allocation({0x1000, nullptr, 0}, 1) &&
                         !registry.has_live_independent_allocation({0x1000, key.context, 1}, 1),
                     "allocation identity not enforced");
    const auto summaries = registry.erase_context_by_device(key.context);
    passed &=
        expect(summaries && summaries->size() == 1 && summaries->front().memory_bytes == 128 &&
                   !registry.has_live_independent_allocation(key, 1),
               "context retirement must preserve accounting and remove provenance");
    const AllocationIdentity wrapped{~CUdeviceptr{0} - 15, key.context, 0};
    passed &= expect(registry.record(wrapped, 32, AllocationScope::kContextBound,
                                     AllocationProvenance::kIndependentDevice) &&
                         !registry.has_live_independent_allocation(wrapped, 1),
                     "wrapped recorded allocation cannot establish provenance");
    return passed;
}

bool test_registry_reinitializes_after_fork() {
    AllocationRegistry registry;
    const AllocationIdentity parent_key{
        .device_pointer = 0x1000, .context = reinterpret_cast<CUcontext>(0x3000), .device = 0};
    const AllocationIdentity child_key{
        .device_pointer = 0x2000, .context = reinterpret_cast<CUcontext>(0x4000), .device = 0};
    if (!expect(registry.record(parent_key, 16), "fork parent allocation was not recorded")) {
        return false;
    }

    const pid_t child_pid = ::fork();
    if (!expect(child_pid >= 0, "allocation registry fork failed")) {
        return false;
    }
    if (child_pid == 0) {
        const bool child_recorded = registry.record(child_key, 32);
        const auto [parent_status, parent_ticket] = registry.begin_release(parent_key);
        static_cast<void>(parent_ticket);
        _exit(child_recorded && parent_status == AllocationRegistry::ReleaseStatus::kUnknown
                  ? EXIT_SUCCESS
                  : EXIT_FAILURE);
    }

    int child_status = 0;
    if (!expect(::waitpid(child_pid, &child_status, 0) == child_pid,
                "allocation registry child wait failed")) {
        return false;
    }
    return expect(WIFEXITED(child_status) && WEXITSTATUS(child_status) == EXIT_SUCCESS,
                  "forked allocation registry retained parent metadata");
}

bool test_submitted_context_snapshots() {
    AllocationRegistry registry;
    const auto context = reinterpret_cast<CUcontext>(0x8000);
    const auto foreign = reinterpret_cast<CUcontext>(0x8100);
    const AsyncStreamIdentity legacy{.context = context};
    const AsyncStreamIdentity detached{.stream = reinterpret_cast<CUstream>(0x9000),
                                       .context = context};
    std::thread other([] {});
    const AsyncStreamIdentity thread_stream{
        .context = context, .per_thread_default_stream = true, .thread_id = other.get_id()};
    other.join();
    const AsyncStreamIdentity foreign_stream{.context = foreign};
    const AllocationIdentity native{0x100, context, 0}, cooperative{0x200, context, 0},
        detached_key{0x300, context, 1}, foreign_allocation{0x400, foreign, 1},
        late{0x500, context, 0}, foreign_release{0x600, foreign, 0};
    for (const auto& [key, bytes] :
         std::array{std::pair{native, 16U}, std::pair{cooperative, 32U},
                    std::pair{detached_key, 64U}, std::pair{foreign_allocation, 128U},
                    std::pair{late, 256U}, std::pair{foreign_release, 512U}}) {
        if (!registry.record(key, bytes, AllocationScope::kContextIndependent)) {
            return false;
        }
    }
    const auto first = registry.begin_async_release(native, legacy).second;
    const auto second = registry.begin_async_release(cooperative, legacy).second;
    const auto third = registry.begin_async_release(detached_key, detached).second;
    const auto fourth = registry.begin_async_release(foreign_allocation, thread_stream).second;
    const auto fifth = registry.begin_async_release(late, legacy).second;
    const auto sixth = registry.begin_async_release(foreign_release, foreign_stream).second;
    if (!first || !second || !third || !fourth || !fifth || !sixth ||
        !registry.prepare_cooperative_release(*second) ||
        !registry.prepare_cooperative_release(*third)) {
        return false;
    }
    for (const auto& ticket : {*first, *second, *third, *fourth, *sixth}) {
        if (!registry.commit_async_release(ticket)) {
            return false;
        }
    }
    if (!registry.detach_async_releases_for_stream(detached)) {
        return false;
    }
    const auto snapshot = registry.submitted_async_releases(true);
    const auto stream_snapshot = registry.submitted_async_releases();
    if (!snapshot || snapshot->size() != 5 || !stream_snapshot || stream_snapshot->size() != 4 ||
        !registry.commit_async_release(*fifth)) {
        return false;
    }
    const auto has_bytes = [](const auto& result, std::array<glimmer::core::MemoryBytes, 2> bytes) {
        if (!result) {
            return false;
        }
        for (const auto& summary : *result) {
            if (summary.device < 0 || summary.device > 1 ||
                bytes[static_cast<std::size_t>(summary.device)] != summary.memory_bytes) {
                return false;
            }
            bytes[static_cast<std::size_t>(summary.device)] = 0;
        }
        return bytes == std::array<glimmer::core::MemoryBytes, 2>{};
    };
    bool passed = expect(
        has_bytes(registry.complete_submitted_context_releases(*snapshot, foreign), {512, 0}) &&
            has_bytes(registry.complete_submitted_context_releases(*snapshot, context), {16, 128}),
        "context proof uses release context, all PTDS callers and detached ownership, not late or "
        "unconfirmed frees");
    passed &= expect(
        registry.confirm_cooperative_release(*second) &&
            registry.confirm_cooperative_release(*third) &&
            has_bytes(registry.complete_submitted_context_releases(*snapshot, context), {32, 64}),
        "captured and detached context frees require exact physical confirmation");
    if (!registry.record(native, 16)) {
        return false;
    }
    const auto reused = registry.begin_async_release(native, legacy).second;
    if (!reused || !registry.commit_async_release(*reused)) {
        return false;
    }
    passed &= expect(
        has_bytes(registry.complete_submitted_context_releases(*snapshot, context), {0, 0}) &&
            !registry.complete_submitted_context_releases(*snapshot, nullptr) &&
            !registry.is_accounting_degraded(),
        "stale context snapshot and missing context retire no ownership");
    const auto fresh = registry.submitted_async_releases(true);
    passed &= expect(
        fresh && has_bytes(registry.complete_submitted_context_releases(*fresh, context), {272, 0}),
        "fresh context proof retires exact reused and late generations once");

    AllocationRegistry duplicate;
    if (!duplicate.record(native, 16)) {
        return false;
    }
    const auto ticket = duplicate.begin_async_release(native, legacy).second;
    if (!ticket || !duplicate.commit_async_release(*ticket)) {
        return false;
    }
    const std::array duplicated{*ticket, *ticket};
    passed &= expect(
        !duplicate.complete_submitted_context_releases(duplicated, context) &&
            duplicate.is_accounting_degraded() &&
            duplicate.begin_release(native).first == AllocationRegistry::ReleaseStatus::kInProgress,
        "duplicate context proof fails closed before erasing any allocation");
    return passed;
}

bool test_submitted_release_snapshots() {
    AllocationRegistry registry;
    const auto context = reinterpret_cast<CUcontext>(0x8000);
    const AsyncStreamIdentity stream{.stream = reinterpret_cast<CUstream>(0x9000),
                                     .context = context};
    const AllocationIdentity cooperative{0x100, context, 0};
    const AllocationIdentity native{0x200, context, 0};
    const AllocationIdentity late{0x300, context, 0};
    for (const auto identity : {cooperative, native, late}) {
        if (!registry.record(identity, identity.device_pointer)) {
            return false;
        }
    }
    const auto first = registry.begin_async_release(cooperative, stream).second;
    const auto second = registry.begin_async_release(native, stream).second;
    const auto third = registry.begin_async_release(late, stream).second;
    if (!first || !second || !third || !registry.prepare_cooperative_release(*first) ||
        !registry.commit_async_release(*first) || !registry.commit_async_release(*second)) {
        return false;
    }
    const auto snapshot = registry.submitted_async_releases();
    if (!snapshot || snapshot->size() != 2 || !registry.commit_async_release(*third)) {
        return false;
    }
    const auto has_bytes = [](const auto& result, glimmer::core::MemoryBytes bytes) {
        return result && (bytes ? result->size() == 1 && result->front().memory_bytes == bytes
                                : result->empty());
    };
    bool passed = expect(has_bytes(registry.complete_submitted_async_releases(*snapshot, stream),
                                   native.device_pointer),
                         "snapshot excludes unconfirmed physical free and late submission");
    passed &= expect(registry.confirm_cooperative_release(*first) &&
                         has_bytes(registry.complete_submitted_async_releases(*snapshot, stream),
                                   cooperative.device_pointer),
                     "captured cooperative release retires only after exact physical proof");
    if (!registry.record(native, native.device_pointer)) {
        return false;
    }
    const auto reused = registry.begin_async_release(native, stream).second;
    if (!reused || !registry.commit_async_release(*reused)) {
        return false;
    }
    passed &= expect(has_bytes(registry.complete_submitted_async_releases(*snapshot, stream), 0),
                     "old snapshot cannot retire reused allocation or later release generation");
    const auto fresh = registry.submitted_async_releases();
    if (!fresh) {
        return false;
    }
    passed &= expect(has_bytes(registry.complete_submitted_async_releases(*fresh, stream),
                               native.device_pointer + late.device_pointer),
                     "fresh proof retires exactly its matching submitted generations");
    const AllocationIdentity detached{0x400, context, 0};
    if (!registry.record(detached, 16)) {
        return false;
    }
    const auto detached_ticket = registry.begin_async_release(detached, stream).second;
    if (!detached_ticket || !registry.commit_async_release(*detached_ticket)) {
        return false;
    }
    const auto before_detach = registry.submitted_async_releases();
    passed &=
        expect(before_detach && registry.detach_async_releases_for_stream(stream) &&
                   has_bytes(registry.complete_submitted_async_releases(*before_detach, stream), 0),
               "detached release cannot inherit proof through an old stream snapshot");
    passed &= expect(has_bytes(registry.erase_context_by_device(context), 16),
                     "only checked context retirement can erase detached ownership");

    const AsyncStreamIdentity legacy{.context = context};
    AsyncStreamIdentity per_thread{.context = context,
                                   .per_thread_default_stream = true,
                                   .thread_id = std::this_thread::get_id()};
    if (!registry.record(cooperative, 16) || !registry.record(native, 32)) {
        return false;
    }
    const auto legacy_ticket = registry.begin_async_release(cooperative, legacy).second;
    const auto thread_ticket = registry.begin_async_release(native, per_thread).second;
    if (!legacy_ticket || !thread_ticket || !registry.commit_async_release(*legacy_ticket) ||
        !registry.commit_async_release(*thread_ticket)) {
        return false;
    }
    const auto defaults = registry.submitted_async_releases();
    if (!defaults) {
        return false;
    }
    passed &= expect(has_bytes(registry.complete_submitted_async_releases(*defaults, legacy), 16),
                     "legacy completion does not retire the caller PTDS namespace");
    const auto owner_thread = per_thread.thread_id;
    std::thread other([] {});
    per_thread.thread_id = other.get_id();
    other.join();
    passed &=
        expect(has_bytes(registry.complete_submitted_async_releases(*defaults, per_thread), 0),
               "another caller cannot retire a PTDS ticket");
    per_thread.thread_id = owner_thread;
    passed &=
        expect(has_bytes(registry.complete_submitted_async_releases(*defaults, per_thread), 32),
               "exact PTDS namespace proof retires its captured ticket once");

    AllocationRegistry duplicate;
    if (!duplicate.record(native, 32)) {
        return false;
    }
    const auto duplicate_ticket = duplicate.begin_async_release(native, stream).second;
    if (!duplicate_ticket || !duplicate.commit_async_release(*duplicate_ticket)) {
        return false;
    }
    const std::array duplicated{*duplicate_ticket, *duplicate_ticket};
    passed &= expect(
        !duplicate.complete_submitted_async_releases(duplicated, stream) &&
            duplicate.is_accounting_degraded() &&
            duplicate.begin_release(native).first == AllocationRegistry::ReleaseStatus::kInProgress,
        "duplicate snapshot proof fails accounting without erasing ownership");
    return passed;
}

}  // namespace

namespace {
bool test_cooperative_release_proof() {
    AllocationRegistry registry;
    const AllocationIdentity identity{
        .device_pointer = 0x9000, .context = reinterpret_cast<CUcontext>(0x3000), .device = 0};
    const AsyncStreamIdentity stream{.stream = reinterpret_cast<CUstream>(0x4000),
                                     .context = identity.context};
    bool passed = true;
    for (bool should_confirm_first : {false, true}) {
        passed &= expect(registry.record(identity, 32), "record cooperative allocation");
        const auto [status, ticket] = registry.begin_async_release(identity, stream);
        if (!expect(status == AllocationRegistry::ReleaseStatus::kStarted && ticket.has_value(),
                    "start generation-bound cooperative release")) {
            return false;
        }
        if (!ticket) return false;
        passed &= expect(!registry.confirm_cooperative_release(*ticket) &&
                             registry.prepare_cooperative_release(*ticket) &&
                             !registry.prepare_cooperative_release(*ticket),
                         "only one prepared cooperative ticket permits completion proof");
        auto stale = *ticket;
        ++stale.async_release_generation;
        passed &= expect(
            !registry.confirm_cooperative_release(stale) && !registry.commit_async_release(stale),
            "wrong generation cannot confirm or commit a release");
        auto wrong_stream = *ticket;
        wrong_stream.stream.stream = reinterpret_cast<CUstream>(0x5000);
        passed &= expect(!registry.confirm_cooperative_release(wrong_stream),
                         "wrong stream cannot supply another release's proof");
        if (should_confirm_first) {
            passed &= expect(registry.confirm_cooperative_release(*ticket),
                             "physical confirmation may precede Runtime return/commit");
        }
        passed &=
            expect(registry.commit_async_release(*ticket), "commit accepted cooperative free");
        if (!should_confirm_first) {
            const auto premature_stream =
                registry.complete_async_releases_for_stream_by_device(stream);
            const auto premature_context =
                registry.complete_async_releases_for_context_by_device(identity.context);
            passed &=
                expect(premature_stream && premature_stream->empty() && premature_context &&
                           premature_context->empty() &&
                           registry.complete_async_releases_for_context(identity.context, 0) == 0,
                       "original stream/context completion cannot release unissued private free");
            passed &=
                expect(registry.confirm_cooperative_release(*ticket), "confirm actual queued free");
        }
        passed &= expect(!registry.confirm_cooperative_release(*ticket),
                         "physical proof is consumed once");
        const auto completed = registry.complete_async_releases_for_stream_by_device(stream);
        passed &=
            expect(completed && completed->size() == 1 && completed->front().memory_bytes == 32,
                   "only submitted plus physically confirmed release returns bytes once");
        passed &= expect(!registry.confirm_cooperative_release(*ticket),
                         "erased ticket cannot be confirmed again");
    }
    passed &= expect(registry.record(identity, 32), "record cancellable allocation");
    const auto first = registry.begin_async_release(identity, stream).second;
    if (!first) {
        return false;
    }
    passed &= expect(registry.prepare_cooperative_release(*first), "prepare cancellable release");
    registry.cancel_release(*first);
    const auto second = registry.begin_async_release(identity, stream).second;
    if (!second) {
        return false;
    }
    passed &= expect(first->async_release_generation != second->async_release_generation &&
                         !registry.confirm_cooperative_release(*first) &&
                         !registry.commit_async_release(*first),
                     "cancelled ticket cannot affect a new release of the same allocation");
    registry.cancel_release(*first);
    passed &= expect(registry.commit_async_release(*second),
                     "stale cancel must not cancel the new ticket");
    const auto ordinary = registry.complete_async_releases_for_context_by_device(identity.context);
    passed &= expect(ordinary && ordinary->size() == 1 && ordinary->front().memory_bytes == 32,
                     "cancelled cooperative marker must not change ordinary native completion");
    passed &= expect(!registry.is_accounting_degraded(),
                     "invalid proof attempts do not corrupt registry");
    return passed;
}
}  // namespace

int main() {
    const CUcontext first_context = reinterpret_cast<CUcontext>(0x1000);
    const CUcontext second_context = reinterpret_cast<CUcontext>(0x2000);
    const AllocationIdentity key{.device_pointer = 0x42, .context = first_context, .device = 0};
    const AllocationIdentity same_pointer_different_context{
        .device_pointer = 0x42, .context = second_context, .device = 0};
    const AllocationIdentity second_key{
        .device_pointer = 0x84, .context = first_context, .device = 0};
    AllocationRegistry registry;
    bool all_passed = true;

    all_passed &= expect(registry.record(key, 64), "initial allocation was not recorded");
    all_passed &= expect(registry.record(same_pointer_different_context, 32),
                         "same pointer in another context was not recorded");

    auto [release_status, release_ticket] = registry.begin_release(key);
    all_passed &= expect(release_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "release did not enter the releasing state");
    all_passed &= expect(release_ticket.has_value(), "release did not return a ticket");
    if (!release_ticket.has_value()) {
        return EXIT_FAILURE;
    }

    const auto [duplicate_release_status, duplicate_release_ticket] = registry.begin_release(key);
    all_passed &= expect(duplicate_release_status == AllocationRegistry::ReleaseStatus::kInProgress,
                         "duplicate release was not identified as in progress");
    all_passed &= expect(!duplicate_release_ticket.has_value(),
                         "duplicate release unexpectedly returned a ticket");

    all_passed &= expect(!registry.record(key, 128),
                         "a pointer under release was recorded before release completed");
    all_passed &= expect(registry.complete_release(*release_ticket), "release did not complete");
    all_passed &= expect(registry.record(key, 128),
                         "allocation could not be recorded after release completed");

    auto [second_release_status, second_release_ticket] = registry.begin_release(key);
    all_passed &= expect(second_release_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "recorded allocation could not be released");
    all_passed &= expect(second_release_ticket.has_value(), "second release had no ticket");
    if (second_release_ticket.has_value()) {
        registry.cancel_release(*second_release_ticket);
    }

    auto [cancelled_release_status, cancelled_release_ticket] = registry.begin_release(key);
    all_passed &= expect(cancelled_release_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "cancelled release did not restore the allocation");
    if (cancelled_release_ticket.has_value()) {
        all_passed &= expect(registry.complete_release(*cancelled_release_ticket),
                             "cancelled allocation could not be released later");
    }

    const auto [unknown_release_status, unknown_release_ticket] =
        registry.begin_release(second_key);
    all_passed &= expect(unknown_release_status == AllocationRegistry::ReleaseStatus::kUnknown,
                         "unknown release was not identified");
    all_passed &= expect(!unknown_release_ticket.has_value(), "unknown release returned a ticket");

    registry.mark_accounting_degraded();
    all_passed &=
        expect(registry.is_accounting_degraded(), "accounting degradation was not recorded");
    all_passed &=
        expect(!registry.record(second_key, 16), "degraded accounting accepted a new allocation");

    AllocationRegistry vmm_registry;
    const VmmAllocationIdentity vmm_key{.handle = 0x1234, .device = 1};
    all_passed &=
        expect(vmm_registry.record_vmm(vmm_key, 256), "VMM allocation handle was not recorded");
    all_passed &= expect(!vmm_registry.record_vmm(vmm_key, 256),
                         "duplicate VMM allocation handle was recorded");
    auto [vmm_status, vmm_ticket] = vmm_registry.begin_vmm_release_by_handle(vmm_key.handle);
    all_passed &=
        expect(vmm_status == AllocationRegistry::ReleaseStatus::kStarted && vmm_ticket.has_value(),
               "VMM release did not enter the releasing state");
    const auto [vmm_duplicate_status, vmm_duplicate_ticket] =
        vmm_registry.begin_vmm_release_by_handle(vmm_key.handle);
    all_passed &= expect(vmm_duplicate_status == AllocationRegistry::ReleaseStatus::kInProgress &&
                             !vmm_duplicate_ticket.has_value(),
                         "duplicate VMM release was not identified as in progress");
    if (vmm_ticket.has_value()) {
        vmm_registry.cancel_vmm_release(*vmm_ticket);
    }
    auto [vmm_retry_status, vmm_retry_ticket] = vmm_registry.begin_vmm_release(vmm_key);
    all_passed &= expect(vmm_retry_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             vmm_retry_ticket.has_value(),
                         "cancelled VMM release did not restore the handle");
    if (vmm_retry_ticket.has_value()) {
        all_passed &= expect(vmm_registry.complete_vmm_release(*vmm_retry_ticket),
                             "VMM release did not complete");
    }
    const auto [vmm_unknown_status, vmm_unknown_ticket] =
        vmm_registry.begin_vmm_release_by_handle(vmm_key.handle);
    all_passed &= expect(vmm_unknown_status == AllocationRegistry::ReleaseStatus::kUnknown &&
                             !vmm_unknown_ticket.has_value(),
                         "released VMM handle remained in the registry");

    AllocationRegistry retained_vmm_registry;
    const VmmAllocationIdentity retained_vmm_key{.handle = 0x5678, .device = 0};
    all_passed &= expect(!retained_vmm_registry.retain_vmm_handle(retained_vmm_key.handle),
                         "unknown VMM handle was retained");
    all_passed &= expect(retained_vmm_registry.record_vmm(retained_vmm_key, 512),
                         "retained VMM allocation was not recorded");
    all_passed &= expect(retained_vmm_registry.retain_vmm_handle(retained_vmm_key.handle),
                         "VMM handle retain was not recorded");
    auto [retained_release_status, retained_release_ticket] =
        retained_vmm_registry.begin_vmm_release_by_handle(retained_vmm_key.handle);
    all_passed &= expect(retained_release_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             retained_release_ticket.has_value() &&
                             !retained_release_ticket->is_last_reference,
                         "retained VMM release was marked as final");
    if (retained_release_ticket.has_value()) {
        all_passed &= expect(retained_vmm_registry.complete_vmm_release(*retained_release_ticket),
                             "retained VMM release did not decrement its reference count");
    }
    auto [final_release_status, final_release_ticket] =
        retained_vmm_registry.begin_vmm_release_by_handle(retained_vmm_key.handle);
    all_passed &=
        expect(final_release_status == AllocationRegistry::ReleaseStatus::kStarted &&
                   final_release_ticket.has_value() && final_release_ticket->is_last_reference,
               "final VMM release was not marked as final");
    if (final_release_ticket.has_value()) {
        all_passed &= expect(retained_vmm_registry.complete_vmm_release(*final_release_ticket),
                             "final retained VMM release did not complete");
    }

    AllocationRegistry context_registry;
    all_passed &= expect(context_registry.record(key, 64), "context allocation was not recorded");
    all_passed &= expect(context_registry.record(same_pointer_different_context, 32),
                         "second context allocation was not recorded");
    all_passed &= expect(context_registry.erase_context(first_context) == 64,
                         "context cleanup returned the wrong byte count");
    const auto [remaining_status, remaining_ticket] =
        context_registry.begin_release(same_pointer_different_context);
    all_passed &= expect(remaining_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "context cleanup removed an allocation from another context");
    all_passed &=
        expect(remaining_ticket.has_value(), "remaining allocation had no release ticket");

    AllocationRegistry fallback_registry;
    all_passed &= expect(fallback_registry.record(key, 64), "fallback allocation was not recorded");
    const auto [fallback_status, fallback_ticket] =
        fallback_registry.begin_release_by_pointer(key.device_pointer, key.device);
    all_passed &= expect(fallback_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "unique pointer fallback did not start release");
    all_passed &= expect(fallback_ticket.has_value(), "fallback release had no ticket");
    if (fallback_ticket.has_value()) {
        all_passed &= expect(fallback_registry.complete_release(*fallback_ticket),
                             "fallback release did not complete");
    }

    AllocationRegistry async_registry;
    const AsyncStreamIdentity first_context_default_stream{
        .stream = nullptr,
        .context = first_context,
        .per_thread_default_stream = false,
        .thread_id = std::this_thread::get_id(),
    };
    all_passed &= expect(async_registry.record(key, 128, AllocationScope::kContextIndependent),
                         "async allocation was not recorded");
    const auto [async_release_status, async_release_ticket] =
        async_registry.begin_async_release(key, first_context_default_stream);
    all_passed &= expect(async_release_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "async release did not enter the pending state");
    all_passed &= expect(async_release_ticket.has_value(), "async release had no ticket");
    const auto [pending_sync_status, pending_sync_ticket] = async_registry.begin_release(key);
    all_passed &= expect(pending_sync_status == AllocationRegistry::ReleaseStatus::kInProgress,
                         "pending async release was not protected from synchronous release");
    all_passed &= expect(!pending_sync_ticket.has_value(),
                         "pending async release returned a synchronous ticket");
    all_passed &= expect(async_registry.erase_context(first_context) == 0,
                         "context cleanup removed an async allocation");
    const auto unsubmitted_summaries =
        async_registry.complete_async_releases_for_stream_by_device(first_context_default_stream);
    all_passed &= expect(unsubmitted_summaries.has_value() && unsubmitted_summaries->empty(),
                         "unsubmitted async release was completed");
    all_passed &= expect(async_release_ticket.has_value() &&
                             async_registry.commit_async_release(*async_release_ticket),
                         "async release submission was not committed");
    all_passed &=
        expect(async_registry.detach_async_releases_for_stream(first_context_default_stream),
               "stream detachment failed");
    all_passed &= expect(async_registry.complete_async_releases_for_context(second_context, 0) == 0,
                         "wrong context completed an async release");
    all_passed &=
        expect(async_registry.complete_async_releases_for_context(first_context, 0) == 128,
               "context completion did not release async accounting");

    AllocationRegistry default_stream_registry;
    const AllocationIdentity legacy_default_allocation{
        .device_pointer = 0x90, .context = first_context, .device = 0};
    const AllocationIdentity other_context_default_allocation{
        .device_pointer = 0x91, .context = second_context, .device = 0};
    const AllocationIdentity per_thread_default_allocation{
        .device_pointer = 0x92, .context = first_context, .device = 0};
    all_passed &= expect(default_stream_registry.record(legacy_default_allocation, 16,
                                                        AllocationScope::kContextIndependent),
                         "legacy default-stream allocation was not recorded");
    all_passed &= expect(default_stream_registry.record(other_context_default_allocation, 32,
                                                        AllocationScope::kContextIndependent),
                         "other-context default-stream allocation was not recorded");
    all_passed &= expect(default_stream_registry.record(per_thread_default_allocation, 64,
                                                        AllocationScope::kContextIndependent),
                         "PTDS allocation was not recorded");
    const AsyncStreamIdentity first_context_legacy_stream{
        .stream = nullptr,
        .context = first_context,
        .per_thread_default_stream = false,
        .thread_id = std::this_thread::get_id(),
    };
    const AsyncStreamIdentity second_context_legacy_stream{
        .stream = nullptr,
        .context = second_context,
        .per_thread_default_stream = false,
        .thread_id = std::this_thread::get_id(),
    };
    const AsyncStreamIdentity first_context_ptds_stream{
        .stream = nullptr,
        .context = first_context,
        .per_thread_default_stream = true,
        .thread_id = std::this_thread::get_id(),
    };
    const auto [legacy_status, legacy_ticket] = default_stream_registry.begin_async_release(
        legacy_default_allocation, first_context_legacy_stream);
    const auto [other_context_status, other_context_ticket] =
        default_stream_registry.begin_async_release(other_context_default_allocation,
                                                    second_context_legacy_stream);
    const auto [ptds_status, ptds_ticket] = default_stream_registry.begin_async_release(
        per_thread_default_allocation, first_context_ptds_stream);
    all_passed &= expect(legacy_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             other_context_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             ptds_status == AllocationRegistry::ReleaseStatus::kStarted,
                         "default-stream releases did not start");
    if (legacy_ticket.has_value()) {
        all_passed &= expect(default_stream_registry.commit_async_release(*legacy_ticket),
                             "legacy default-stream release did not commit");
    }
    if (other_context_ticket.has_value()) {
        all_passed &= expect(default_stream_registry.commit_async_release(*other_context_ticket),
                             "other-context default-stream release did not commit");
    }
    if (ptds_ticket.has_value()) {
        all_passed &= expect(default_stream_registry.commit_async_release(*ptds_ticket),
                             "PTDS release did not commit");
    }
    const auto legacy_summaries =
        default_stream_registry.complete_async_releases_for_stream_by_device(
            first_context_legacy_stream);
    all_passed &= expect(legacy_summaries.has_value() && legacy_summaries->size() == 1 &&
                             legacy_summaries->front().memory_bytes == 16,
                         "legacy default-stream completion crossed context or PTDS boundaries");
    const auto ptds_summaries =
        default_stream_registry.complete_async_releases_for_stream_by_device(
            first_context_ptds_stream);
    all_passed &= expect(ptds_summaries.has_value() && ptds_summaries->size() == 1 &&
                             ptds_summaries->front().memory_bytes == 64,
                         "PTDS completion crossed the legacy default-stream boundary");
    const auto other_context_summaries =
        default_stream_registry.complete_async_releases_for_stream_by_device(
            second_context_legacy_stream);
    all_passed &=
        expect(other_context_summaries.has_value() && other_context_summaries->size() == 1 &&
                   other_context_summaries->front().memory_bytes == 32,
               "default-stream completion crossed context boundaries");

    AllocationRegistry grouped_registry;
    const CUstream grouped_stream = reinterpret_cast<CUstream>(0x55);
    const CUstream context_stream = reinterpret_cast<CUstream>(0x66);
    const AllocationIdentity grouped_first{
        .device_pointer = 0x100, .context = first_context, .device = 0};
    const AllocationIdentity grouped_second{
        .device_pointer = 0x101, .context = first_context, .device = 1};
    const AllocationIdentity grouped_other_context{
        .device_pointer = 0x102, .context = second_context, .device = 1};
    const AsyncStreamIdentity grouped_stream_identity{
        .stream = grouped_stream,
        .context = first_context,
    };
    const AsyncStreamIdentity context_stream_identity{
        .stream = context_stream,
        .context = second_context,
    };
    all_passed &= expect(grouped_registry.record(grouped_first, 16),
                         "first grouped allocation was not recorded");
    all_passed &= expect(grouped_registry.record(grouped_second, 32),
                         "second grouped allocation was not recorded");
    all_passed &= expect(grouped_registry.record(grouped_other_context, 64),
                         "other-context grouped allocation was not recorded");

    const auto [grouped_first_status, grouped_first_ticket] =
        grouped_registry.begin_async_release(grouped_first, grouped_stream_identity);
    const auto [grouped_second_status, grouped_second_ticket] =
        grouped_registry.begin_async_release(grouped_second, grouped_stream_identity);
    const auto [grouped_other_status, grouped_other_ticket] =
        grouped_registry.begin_async_release(grouped_other_context, context_stream_identity);
    all_passed &= expect(grouped_first_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             grouped_first_ticket.has_value(),
                         "first grouped release did not start");
    all_passed &= expect(grouped_second_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             grouped_second_ticket.has_value(),
                         "second grouped release did not start");
    all_passed &= expect(grouped_other_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             grouped_other_ticket.has_value(),
                         "other-context grouped release did not start");
    if (grouped_first_ticket.has_value()) {
        all_passed &= expect(grouped_registry.commit_async_release(*grouped_first_ticket),
                             "first grouped release submission was not committed");
    }
    if (grouped_second_ticket.has_value()) {
        all_passed &= expect(grouped_registry.commit_async_release(*grouped_second_ticket),
                             "second grouped release submission was not committed");
    }
    if (grouped_other_ticket.has_value()) {
        all_passed &= expect(grouped_registry.commit_async_release(*grouped_other_ticket),
                             "other-context grouped release submission was not committed");
    }

    const auto stream_summaries =
        grouped_registry.complete_async_releases_for_stream_by_device(grouped_stream_identity);
    all_passed &= expect(stream_summaries.has_value(), "stream grouping unexpectedly failed");
    if (stream_summaries.has_value()) {
        all_passed &= expect(stream_summaries->size() == 2,
                             "stream grouping did not preserve device separation");
        bool found_device_zero = false;
        bool found_device_one = false;
        for (const auto& summary : *stream_summaries) {
            found_device_zero =
                found_device_zero || (summary.device == 0 && summary.memory_bytes == 16);
            found_device_one =
                found_device_one || (summary.device == 1 && summary.memory_bytes == 32);
        }
        all_passed &= expect(found_device_zero && found_device_one,
                             "stream grouping returned incorrect device totals");
    }

    const auto context_summaries =
        grouped_registry.complete_async_releases_for_context_by_device(second_context);
    all_passed &= expect(context_summaries.has_value(), "context grouping unexpectedly failed");
    if (context_summaries.has_value()) {
        all_passed &=
            expect(context_summaries->size() == 1 && context_summaries->front().device == 1 &&
                       context_summaries->front().memory_bytes == 64,
                   "context grouping returned incorrect device totals");
    }

    AllocationRegistry cleanup_registry;
    all_passed &= expect(cleanup_registry.record(grouped_first, 16),
                         "context cleanup allocation was not recorded");
    all_passed &= expect(cleanup_registry.record(grouped_second, 32),
                         "second context cleanup allocation was not recorded");
    const AllocationIdentity context_independent_key{
        .device_pointer = 0x103, .context = first_context, .device = 0};
    all_passed &= expect(
        cleanup_registry.record(context_independent_key, 8, AllocationScope::kContextIndependent),
        "context-independent allocation was not recorded");
    const auto cleanup_summaries = cleanup_registry.erase_context_by_device(first_context);
    all_passed &= expect(cleanup_summaries.has_value(), "context cleanup grouping failed");
    if (cleanup_summaries.has_value()) {
        all_passed &= expect(cleanup_summaries->size() == 2,
                             "context cleanup did not preserve device separation");
    }
    const auto [independent_status, independent_ticket] =
        cleanup_registry.begin_release(context_independent_key);
    all_passed &= expect(independent_status == AllocationRegistry::ReleaseStatus::kStarted &&
                             independent_ticket.has_value(),
                         "context cleanup removed a context-independent allocation");
    if (independent_ticket.has_value()) {
        all_passed &= expect(cleanup_registry.complete_release(*independent_ticket),
                             "context-independent allocation could not be released");
    }

    all_passed &= test_registry_reinitializes_after_fork();
    all_passed &= test_independent_allocation_provenance();
    all_passed &= test_cooperative_release_proof();
    all_passed &= test_submitted_release_snapshots();
    all_passed &= test_submitted_context_snapshots();

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
