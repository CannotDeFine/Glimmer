#include "glimmer/control/task_admission.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using glimmer::control::TaskAdmissionRequest;
using glimmer::control::TaskAdmissionService;
using glimmer::control::TaskResourceRegistrar;
using glimmer::core::Scheduler;
using glimmer::core::SubmitStatus;
using glimmer::core::TaskId;

struct RegistrarState {
    bool should_register = true;
    int calls = 0;
    TaskId last_task_id = 0;
};

bool register_resource(void* context, TaskId task_id) noexcept {
    if (context == nullptr) {
        return false;
    }
    auto* state = static_cast<RegistrarState*>(context);
    ++state->calls;
    state->last_task_id = task_id;
    return state->should_register;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_admission_registers_and_cancels() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const auto admission = service.submit(
        TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20, .weight = 2},
        register_resource, &state);
    expect(admission.accepted(), "valid request should be admitted");
    expect(state.calls == 1 && state.last_task_id == admission.task_id,
           "registrar should receive the accepted task id");
    expect(service.find(admission.task_id).has_value(), "admitted task should be observable");
    expect(service.cancel(admission.task_id), "queued task should be cancellable");
    expect(!service.cancel_queued(admission.task_id),
           "a terminal task should not be cancellable through the queued-only path");
    expect(scheduler.usage().used_bytes() == 0, "cancellation should release reservation");
}

void test_registration_failure_rolls_back() {
    Scheduler scheduler(100);
    TaskAdmissionService service(scheduler);
    RegistrarState state{.should_register = false};
    const auto admission =
        service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 20},
                       register_resource, &state);
    expect(admission.status == SubmitStatus::kInternalError,
           "registration failure should be reported");
    expect(admission.task_id != 0, "rollback result should preserve task identity");
    expect(state.calls == 1, "failed registration should be attempted once");
    const auto snapshot = service.find(admission.task_id);
    expect(snapshot.has_value(), "rolled-back task should remain inspectable");
    if (snapshot.has_value()) {
        expect(snapshot->state == glimmer::core::TaskState::kCancelled,
               "rolled-back task should be cancelled");
    }
    expect(scheduler.usage().used_bytes() == 0, "registration failure should release reservation");
}

void test_admission_rejects_invalid_or_unavailable_requests() {
    Scheduler scheduler(10);
    TaskAdmissionService service(scheduler);
    RegistrarState state;
    const TaskAdmissionRequest invalid_request{.tenant_id = "tenant-a", .memory_bytes = 0};
    expect(service.submit(invalid_request, register_resource, &state).status ==
               SubmitStatus::kInvalidTask,
           "invalid request should be rejected by scheduler validation");
    expect(service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 1},
                          static_cast<TaskResourceRegistrar>(nullptr), nullptr)
                   .status == SubmitStatus::kInternalError,
           "missing registrar should be rejected");
    expect(service.submit(TaskAdmissionRequest{.tenant_id = "tenant-a", .memory_bytes = 11},
                          register_resource, &state)
                   .status == SubmitStatus::kQuotaExceeded,
           "request beyond quota should be rejected");
    expect(state.calls == 0, "rejected requests must not invoke registrar");
}

}  // namespace

int main() {
    test_admission_registers_and_cancels();
    test_registration_failure_rolls_back();
    test_admission_rejects_invalid_or_unavailable_requests();
    return EXIT_SUCCESS;
}
