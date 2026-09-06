find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME glimmer_pytorch_measurements_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/pytorch_measurements_test.py
    )
    add_test(NAME glimmer_pytorch_overhead_baseline_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/pytorch_overhead_baseline_test.py
    )
    add_test(NAME glimmer_pytorch_launch_profile_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/pytorch_launch_profile_test.py
    )
endif()

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

add_executable(glimmer_core_adaptive_slo_test core/adaptive_slo_test.cc)

target_link_libraries(glimmer_core_adaptive_slo_test
    PRIVATE
        Glimmer::core
)

glimmer_enable_clang_tidy(glimmer_core_adaptive_slo_test)

add_test(NAME glimmer_core_adaptive_slo_test COMMAND glimmer_core_adaptive_slo_test)

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

add_executable(glimmer_control_task_admission_concurrency_test
    control/task_admission_concurrency_test.cc
)
target_link_libraries(glimmer_control_task_admission_concurrency_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_task_admission_concurrency_test)
add_test(NAME glimmer_control_task_admission_concurrency_test
    COMMAND glimmer_control_task_admission_concurrency_test
)
set_tests_properties(glimmer_control_task_admission_concurrency_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_launch_gate_test control/launch_gate_test.cc)

target_link_libraries(glimmer_control_launch_gate_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_launch_gate_test)

add_test(NAME glimmer_control_launch_gate_test
    COMMAND glimmer_control_launch_gate_test)

add_executable(glimmer_control_launch_gate_remote_process_test
    control/launch_gate_remote_process_test.cc
)

target_link_libraries(glimmer_control_launch_gate_remote_process_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_launch_gate_remote_process_test)

add_test(NAME glimmer_control_launch_gate_remote_process_test
    COMMAND glimmer_control_launch_gate_remote_process_test)
set_tests_properties(glimmer_control_launch_gate_remote_process_test PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 10
)

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
set_tests_properties(glimmer_control_unix_socket_server_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

add_executable(glimmer_control_socket_line_reader_test control/socket_line_reader_test.cc)
target_include_directories(glimmer_control_socket_line_reader_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_socket_line_reader_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_socket_line_reader_test)
add_test(NAME glimmer_control_socket_line_reader_test
    COMMAND glimmer_control_socket_line_reader_test)
set_tests_properties(glimmer_control_socket_line_reader_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

add_executable(glimmer_control_unix_socket_client_test control/unix_socket_client_test.cc)
target_link_libraries(glimmer_control_unix_socket_client_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_unix_socket_client_test)
add_test(NAME glimmer_control_unix_socket_client_test
    COMMAND glimmer_control_unix_socket_client_test)
set_tests_properties(glimmer_control_unix_socket_client_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

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
