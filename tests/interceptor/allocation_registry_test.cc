#include "internal/allocation_registry.h"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using glimmer::interceptor::AllocationIdentity;
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

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
