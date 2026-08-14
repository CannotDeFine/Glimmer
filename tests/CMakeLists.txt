add_executable(glimmer_core_quota_ledger_test core/quota_ledger_test.cc)

target_link_libraries(glimmer_core_quota_ledger_test
    PRIVATE
        Glimmer::core
)

glimmer_enable_clang_tidy(glimmer_core_quota_ledger_test)

add_test(NAME glimmer_core_quota_ledger_test COMMAND glimmer_core_quota_ledger_test)

add_executable(glimmer_core_scheduler_test core/scheduler_test.cc)

target_link_libraries(glimmer_core_scheduler_test
    PRIVATE
        Glimmer::core
)

glimmer_enable_clang_tidy(glimmer_core_scheduler_test)

add_test(NAME glimmer_core_scheduler_test COMMAND glimmer_core_scheduler_test)

add_executable(glimmer_backend_simulated_test backend/simulated_backend_test.cc)

target_link_libraries(glimmer_backend_simulated_test
    PRIVATE
        Glimmer::backend
)

glimmer_enable_clang_tidy(glimmer_backend_simulated_test)

add_test(NAME glimmer_backend_simulated_test COMMAND glimmer_backend_simulated_test)

add_executable(glimmer_backend_task_executor_test backend/task_executor_test.cc)

target_link_libraries(glimmer_backend_task_executor_test
    PRIVATE
        Glimmer::backend
)

glimmer_enable_clang_tidy(glimmer_backend_task_executor_test)

add_test(NAME glimmer_backend_task_executor_test COMMAND glimmer_backend_task_executor_test)

add_executable(glimmer_control_process_memory_quota_test control/process_memory_quota_test.cc)

target_link_libraries(glimmer_control_process_memory_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_process_memory_quota_test)

add_test(NAME glimmer_control_process_memory_quota_test
    COMMAND glimmer_control_process_memory_quota_test)

add_executable(glimmer_control_composite_quota_test control/composite_quota_test.cc)

target_link_libraries(glimmer_control_composite_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_composite_quota_test)

add_test(NAME glimmer_control_composite_quota_test
    COMMAND glimmer_control_composite_quota_test)

add_executable(glimmer_control_device_capacity_quota_test
    control/device_capacity_quota_test.cc
)

target_link_libraries(glimmer_control_device_capacity_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_device_capacity_quota_test)

add_test(NAME glimmer_control_device_capacity_quota_test
    COMMAND glimmer_control_device_capacity_quota_test
)

add_executable(glimmer_control_shared_memory_quota_test
    control/shared_memory_quota_test.cc
)

target_link_libraries(glimmer_control_shared_memory_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_shared_memory_quota_test)

add_test(NAME glimmer_control_shared_memory_quota_test
    COMMAND glimmer_control_shared_memory_quota_test)

add_executable(glimmer_control_task_admission_test control/task_admission_test.cc)

target_link_libraries(glimmer_control_task_admission_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_task_admission_test)

add_test(NAME glimmer_control_task_admission_test
    COMMAND glimmer_control_task_admission_test)

add_executable(glimmer_control_task_endpoint_test control/task_endpoint_test.cc)

target_link_libraries(glimmer_control_task_endpoint_test
    PRIVATE
        Glimmer::backend
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_task_endpoint_test)

add_test(NAME glimmer_control_task_endpoint_test
    COMMAND glimmer_control_task_endpoint_test)

add_executable(glimmer_control_unix_socket_server_test
    control/unix_socket_server_test.cc
)

target_link_libraries(glimmer_control_unix_socket_server_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_unix_socket_server_test)

add_test(NAME glimmer_control_unix_socket_server_test
    COMMAND glimmer_control_unix_socket_server_test)

add_executable(glimmer_control_service_process_test control/service_process_test.cc)

glimmer_enable_clang_tidy(glimmer_control_service_process_test)

add_test(NAME glimmer_control_service_process_test
    COMMAND glimmer_control_service_process_test)
set_tests_properties(glimmer_control_service_process_test PROPERTIES
    SKIP_RETURN_CODE 77
)

add_executable(glimmer_control_remote_lease_gpu_test control/remote_lease_gpu_test.cc)

target_link_libraries(glimmer_control_remote_lease_gpu_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_remote_lease_gpu_test)

if(GLIMMER_ENABLE_CUDA_GPU_TESTS AND GLIMMER_BUILD_CUDA_BACKEND AND GLIMMER_BUILD_CUDA_WORKLOAD)
    add_test(NAME glimmer_control_remote_lease_gpu_test
        COMMAND glimmer_control_remote_lease_gpu_test
    )
    set_tests_properties(glimmer_control_remote_lease_gpu_test PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60
    )
endif()

add_executable(glimmer_control_task_protocol_test control/task_protocol_test.cc)

target_link_libraries(glimmer_control_task_protocol_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_task_protocol_test)

add_test(NAME glimmer_control_task_protocol_test
    COMMAND glimmer_control_task_protocol_test)
