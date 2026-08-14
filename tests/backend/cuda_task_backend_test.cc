#include "internal/cuda_task_backend.h"

#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using glimmer::backend::ExecutorStepStatus;
using glimmer::backend::cuda::CudaFunctionTable;
using glimmer::backend::cuda::CudaKernelLaunch;
using glimmer::backend::cuda::CudaTaskBackend;
using glimmer::backend::cuda::CudaTaskController;
using glimmer::core::Scheduler;
using glimmer::core::SubmitStatus;
using glimmer::core::TaskSpec;

struct FakeDriverState {
    CUresult launch_result = CUDA_SUCCESS;
    CUresult event_create_result = CUDA_SUCCESS;
    CUresult event_record_result = CUDA_SUCCESS;
    CUresult event_query_result = CUDA_SUCCESS;
    CUresult event_destroy_result = CUDA_SUCCESS;
    int launch_calls = 0;
    int event_create_calls = 0;
    int event_record_calls = 0;
    int event_query_calls = 0;
    int event_destroy_calls = 0;
    unsigned char event_storage = 0;
};

FakeDriverState& fake_state() {
    static FakeDriverState state;
    return state;
}

void reset_fake_state() {
    fake_state() = FakeDriverState{};
}

CUresult fake_launch_kernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    ++fake_state().launch_calls;
    return fake_state().launch_result;
}

CUresult fake_event_create(CUevent* event, unsigned int flags) {
    static_cast<void>(flags);
    ++fake_state().event_create_calls;
    if (fake_state().event_create_result != CUDA_SUCCESS) {
        return fake_state().event_create_result;
    }
    *event = reinterpret_cast<CUevent>(&fake_state().event_storage);
    return CUDA_SUCCESS;
}

CUresult fake_event_record(CUevent event, CUstream stream) {
    static_cast<void>(event);
    static_cast<void>(stream);
    ++fake_state().event_record_calls;
    return fake_state().event_record_result;
}

CUresult fake_event_query(CUevent event) {
    static_cast<void>(event);
    ++fake_state().event_query_calls;
    return fake_state().event_query_result;
}

CUresult fake_event_destroy(CUevent event) {
    static_cast<void>(event);
    ++fake_state().event_destroy_calls;
    return fake_state().event_destroy_result;
}

CudaFunctionTable make_function_table() {
    return {.launch_kernel = fake_launch_kernel,
            .event_create = fake_event_create,
            .event_record = fake_event_record,
            .event_query = fake_event_query,
            .event_destroy = fake_event_destroy};
}

CudaKernelLaunch make_launch() {
    static unsigned char function_storage = 0;
    return {.function = reinterpret_cast<CUfunction>(&function_storage),
            .grid_dim_x = 1,
            .grid_dim_y = 1,
            .grid_dim_z = 1,
            .block_dim_x = 1,
            .block_dim_y = 1,
            .block_dim_z = 1};
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_controller_drives_pending_and_completion() {
    reset_fake_state();
    Scheduler scheduler(64);
    CudaTaskBackend backend(make_function_table());
    CudaTaskController controller(scheduler, backend);
    const auto admission = controller.submit(
        TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16, .work_units = 2}, make_launch());
    expect(admission.status == SubmitStatus::kAccepted, "valid CUDA task should be admitted");

    expect(controller.step().status == ExecutorStepStatus::kSubmitted,
           "controller should submit a registered launch");
    expect(fake_state().launch_calls == 1, "launch should be submitted once");
    expect(fake_state().event_create_calls == 1, "completion event should be created");
    expect(fake_state().event_record_calls == 1, "completion event should be recorded");

    fake_state().event_query_result = CUDA_ERROR_NOT_READY;
    expect(controller.step().status == ExecutorStepStatus::kPending,
           "incomplete event should keep the task pending");
    fake_state().event_query_result = CUDA_SUCCESS;
    expect(controller.step().status == ExecutorStepStatus::kCompleted,
           "completed event should complete the task");
    expect(fake_state().event_destroy_calls == 1, "terminal event should be destroyed");
    expect(scheduler.usage().used_bytes() == 0, "completion should release task quota");
    expect(!controller.active_task().has_value(), "completed task should not remain active");
}

void test_controller_releases_after_launch_failure() {
    reset_fake_state();
    fake_state().launch_result = CUDA_ERROR_LAUNCH_FAILED;
    Scheduler scheduler(64);
    CudaTaskBackend backend(make_function_table());
    CudaTaskController controller(scheduler, backend);
    const auto admission =
        controller.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16}, make_launch());
    expect(admission.accepted(), "task should be admitted before launch failure");
    expect(controller.step().status == ExecutorStepStatus::kBackendError,
           "launch failure should be reported as a backend error");
    expect(fake_state().event_destroy_calls == 1, "failed launch event should be destroyed");
    expect(scheduler.usage().used_bytes() == 0, "launch failure should release task quota");
    expect(backend.register_launch(admission.task_id, make_launch()),
           "failed submission should discard its launch descriptor");
}

void test_controller_rejects_invalid_backend_registration() {
    reset_fake_state();
    Scheduler scheduler(64);
    CudaTaskBackend backend(CudaFunctionTable{});
    CudaTaskController controller(scheduler, backend);
    const auto admission =
        controller.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16}, make_launch());
    expect(admission.status == SubmitStatus::kInternalError,
           "unavailable CUDA functions should reject registration");
    expect(scheduler.usage().used_bytes() == 0, "registration rejection should release task quota");
    expect(controller.step().status == ExecutorStepStatus::kIdle,
           "unavailable backend should stay idle");
    expect(!controller.active_task().has_value(), "unavailable backend should have no active task");
}

void test_backend_reports_event_failure_and_unsupported_cancel() {
    reset_fake_state();
    Scheduler scheduler(64);
    CudaTaskBackend backend(make_function_table());
    CudaTaskController controller(scheduler, backend);
    const auto admission =
        controller.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16}, make_launch());
    expect(admission.accepted(), "task should be admitted before event failure");
    expect(controller.step().status == ExecutorStepStatus::kSubmitted,
           "task should be submitted before event failure");
    fake_state().event_query_result = CUDA_ERROR_INVALID_HANDLE;
    expect(controller.step().status == ExecutorStepStatus::kFailed,
           "event failure should map to a failed task");
    expect(!backend.cancel(admission.task_id), "running CUDA kernels cannot be preempted");
    expect(scheduler.usage().used_bytes() == 0, "event failure should release task quota");
}

void test_backend_fails_closed_on_event_cleanup_failure() {
    reset_fake_state();
    fake_state().event_destroy_result = CUDA_ERROR_UNKNOWN;
    Scheduler scheduler(64);
    CudaTaskBackend backend(make_function_table());
    CudaTaskController controller(scheduler, backend);
    const auto admission =
        controller.submit(TaskSpec{.tenant_id = "tenant-a", .memory_bytes = 16}, make_launch());
    expect(admission.accepted(), "task should be admitted before cleanup failure");
    expect(controller.step().status == ExecutorStepStatus::kSubmitted,
           "task should be submitted before cleanup failure");
    expect(controller.step().status == ExecutorStepStatus::kFailed,
           "event cleanup failure should fail the task closed");
    expect(!backend.is_healthy(), "event cleanup failure should mark backend unhealthy");
    expect(scheduler.usage().used_bytes() == 0, "cleanup failure should release task quota");
}

}  // namespace

int main() {
    test_controller_drives_pending_and_completion();
    test_controller_releases_after_launch_failure();
    test_controller_rejects_invalid_backend_registration();
    test_backend_reports_event_failure_and_unsupported_cancel();
    test_backend_fails_closed_on_event_cleanup_failure();
    return EXIT_SUCCESS;
}
