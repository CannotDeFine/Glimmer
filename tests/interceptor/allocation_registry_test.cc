#include "internal/allocation_registry.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using glimmer::interceptor::AllocationIdentity;
using glimmer::interceptor::AllocationRegistry;
using glimmer::interceptor::AllocationScope;

bool expect(bool condition, std::string_view message) {
    if (condition) {
        return true;
    }

    std::cerr << message << '\n';
    return false;
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
    all_passed &= expect(async_registry.record(key, 128, AllocationScope::kContextIndependent),
                         "async allocation was not recorded");
    const auto [async_release_status, async_release_ticket] =
        async_registry.begin_async_release(key, nullptr);
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
    all_passed &= expect(async_registry.complete_async_releases_for_stream(nullptr) == 0,
                         "unsubmitted async release was completed");
    all_passed &= expect(async_release_ticket.has_value() &&
                             async_registry.commit_async_release(*async_release_ticket),
                         "async release submission was not committed");
    all_passed &= expect(async_registry.complete_async_releases_for_context(second_context, 0) == 0,
                         "wrong context completed an async release");
    all_passed &=
        expect(async_registry.complete_async_releases_for_context(first_context, 0) == 128,
               "context completion did not release async accounting");

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
