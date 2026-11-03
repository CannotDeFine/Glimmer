add_executable(glimmer_control_owner_control_channel_test control/owner_control_channel_test.cc)
target_include_directories(glimmer_control_owner_control_channel_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_owner_control_channel_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_owner_control_channel_test)
add_test(NAME glimmer_control_owner_control_channel_test COMMAND glimmer_control_owner_control_channel_test)
set_tests_properties(glimmer_control_owner_control_channel_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

add_executable(glimmer_control_shared_control_channel_test control/shared_control_channel_test.cc)
target_include_directories(glimmer_control_shared_control_channel_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_shared_control_channel_test PRIVATE Glimmer::control)
target_link_options(glimmer_control_shared_control_channel_test PRIVATE "LINKER:--wrap=poll")
glimmer_enable_clang_tidy(glimmer_control_shared_control_channel_test)
add_test(NAME glimmer_control_shared_control_channel_test COMMAND glimmer_control_shared_control_channel_test)
set_tests_properties(glimmer_control_shared_control_channel_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

add_executable(glimmer_control_execution_socket_test control/execution_socket_test.cc)
target_include_directories(glimmer_control_execution_socket_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_execution_socket_test PRIVATE Glimmer::control)
target_link_options(glimmer_control_execution_socket_test PRIVATE
    "LINKER:--wrap=send" "LINKER:--wrap=sendmsg" "LINKER:--wrap=recvmsg")
glimmer_enable_clang_tidy(glimmer_control_execution_socket_test)
add_test(NAME glimmer_control_execution_socket_test COMMAND glimmer_control_execution_socket_test)
set_tests_properties(glimmer_control_execution_socket_test PROPERTIES TIMEOUT 30)
add_test(NAME glimmer_control_execution_preparation_deadline_test
    COMMAND glimmer_control_execution_socket_test --preparation-deadlines)
set_tests_properties(glimmer_control_execution_preparation_deadline_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_receive_test control/execution_socket_receive_test.cc)
target_include_directories(glimmer_control_execution_receive_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_execution_receive_test PRIVATE Glimmer::control)
target_link_options(glimmer_control_execution_receive_test PRIVATE
    "LINKER:--wrap=recvmsg" "LINKER:--wrap=eventfd" "LINKER:--wrap=read" "LINKER:--wrap=write"
    "LINKER:--wrap=ppoll")
glimmer_enable_clang_tidy(glimmer_control_execution_receive_test)
add_test(NAME glimmer_control_execution_receive_test COMMAND glimmer_control_execution_receive_test)
set_tests_properties(glimmer_control_execution_receive_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_send_test control/execution_socket_send_test.cc)
target_include_directories(glimmer_control_execution_send_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_execution_send_test PRIVATE Glimmer::control)
target_link_options(glimmer_control_execution_send_test PRIVATE
    "LINKER:--wrap=send" "LINKER:--wrap=sendmsg")
glimmer_enable_clang_tidy(glimmer_control_execution_send_test)
add_test(NAME glimmer_control_execution_send_test COMMAND glimmer_control_execution_send_test)
set_tests_properties(glimmer_control_execution_send_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_legacy_protocol_test
    control/execution_dnn_legacy_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_legacy_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_legacy_protocol_test)
add_test(NAME glimmer_control_execution_dnn_legacy_protocol_test
    COMMAND glimmer_control_execution_dnn_legacy_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_legacy_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_protocol_test control/execution_dnn_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_protocol_test)
add_test(NAME glimmer_control_execution_dnn_protocol_test
    COMMAND glimmer_control_execution_dnn_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_profile_protocol_test
    control/execution_dnn_profile_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_profile_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_profile_protocol_test)
add_test(NAME glimmer_control_execution_dnn_profile_protocol_test
    COMMAND glimmer_control_execution_dnn_profile_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_profile_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_backend_protocol_test
    control/execution_dnn_backend_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_backend_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_backend_protocol_test)
add_test(NAME glimmer_control_execution_dnn_backend_protocol_test
    COMMAND glimmer_control_execution_dnn_backend_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_backend_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_backend_execution_protocol_test
    control/execution_dnn_backend_execution_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_backend_execution_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_backend_execution_protocol_test)
add_test(NAME glimmer_control_execution_dnn_backend_execution_protocol_test
    COMMAND glimmer_control_execution_dnn_backend_execution_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_backend_execution_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_nccl_protocol_test
    control/execution_nccl_protocol_test.cc)
target_link_libraries(glimmer_control_execution_nccl_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_nccl_protocol_test)
add_test(NAME glimmer_control_execution_nccl_protocol_test
    COMMAND glimmer_control_execution_nccl_protocol_test)
set_tests_properties(glimmer_control_execution_nccl_protocol_test PROPERTIES TIMEOUT 45)
add_executable(glimmer_control_execution_blas_protocol_test control/execution_blas_protocol_test.cc)
target_link_libraries(glimmer_control_execution_blas_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_blas_protocol_test)
add_test(NAME glimmer_control_execution_blas_protocol_test
    COMMAND glimmer_control_execution_blas_protocol_test)
set_tests_properties(glimmer_control_execution_blas_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_blas_extension_protocol_test
    control/execution_blas_extension_protocol_test.cc)
target_link_libraries(glimmer_control_execution_blas_extension_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_blas_extension_protocol_test)
add_test(NAME glimmer_control_execution_blas_extension_protocol_test
    COMMAND glimmer_control_execution_blas_extension_protocol_test)
set_tests_properties(glimmer_control_execution_blas_extension_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_function_protocol_test
    control/execution_function_protocol_test.cc)
target_link_libraries(glimmer_control_execution_function_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_function_protocol_test)
add_test(NAME glimmer_control_execution_function_protocol_test
    COMMAND glimmer_control_execution_function_protocol_test)
set_tests_properties(glimmer_control_execution_function_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_jit_protocol_test control/execution_jit_protocol_test.cc)
target_link_libraries(glimmer_control_execution_jit_protocol_test PRIVATE Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_jit_protocol_test)
add_test(NAME glimmer_control_execution_jit_protocol_test COMMAND glimmer_control_execution_jit_protocol_test)
set_tests_properties(glimmer_control_execution_jit_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_process_memory_protocol_test
    control/execution_process_memory_protocol_test.cc)
target_link_libraries(glimmer_control_execution_process_memory_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_process_memory_protocol_test)
add_test(NAME glimmer_control_execution_process_memory_protocol_test
    COMMAND glimmer_control_execution_process_memory_protocol_test)
set_tests_properties(glimmer_control_execution_process_memory_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_pointer_protocol_test
    control/execution_pointer_protocol_test.cc)
target_link_libraries(glimmer_control_execution_pointer_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_pointer_protocol_test)
add_test(NAME glimmer_control_execution_pointer_protocol_test
    COMMAND glimmer_control_execution_pointer_protocol_test)
set_tests_properties(glimmer_control_execution_pointer_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_graph_protocol_test
    control/execution_graph_protocol_test.cc)
target_link_libraries(glimmer_control_execution_graph_protocol_test PRIVATE Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_graph_protocol_test)
add_test(NAME glimmer_control_execution_graph_protocol_test COMMAND glimmer_control_execution_graph_protocol_test)
set_tests_properties(glimmer_control_execution_graph_protocol_test PROPERTIES TIMEOUT 30)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME glimmer_check_script_test
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/examples/check_script_test.py
    )
    set_tests_properties(glimmer_check_script_test PROPERTIES TIMEOUT 120)
    add_test(NAME glimmer_artifact_paths_test
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/build/artifact_paths_test.py
    )
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

add_executable(glimmer_ptx_uniform_loop_test transform/ptx_uniform_loop_test.cc)
target_link_libraries(glimmer_ptx_uniform_loop_test PRIVATE
    Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_ptx_uniform_loop_test)
add_test(NAME glimmer_ptx_uniform_loop_test COMMAND glimmer_ptx_uniform_loop_test)
set_tests_properties(glimmer_ptx_uniform_loop_test PROPERTIES TIMEOUT 30)
if(CUDAToolkit_NVCC_EXECUTABLE)
    add_test(NAME glimmer_ptx_uniform_loop_compiler_test COMMAND "${CMAKE_COMMAND}"
        "-DTRANSFORMER=$<TARGET_FILE:glimmer_ptx_uniform_loop_test>"
        "-DNVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ptx-uniform-loop"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/transform/check_uniform_loop_ptx.cmake")
    set_tests_properties(glimmer_ptx_uniform_loop_compiler_test PROPERTIES TIMEOUT 120)
endif()
add_executable(glimmer_transform_ptx_static_storage_test transform/ptx_static_storage_test.cc)
target_link_libraries(glimmer_transform_ptx_static_storage_test PRIVATE
    Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_transform_ptx_static_storage_test)
add_test(NAME glimmer_transform_ptx_static_storage_test COMMAND glimmer_transform_ptx_static_storage_test)
set_tests_properties(glimmer_transform_ptx_static_storage_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_transform_cuda_module_image_test transform/cuda_module_image_test.cc)
target_link_libraries(glimmer_transform_cuda_module_image_test PRIVATE Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_transform_cuda_module_image_test)
add_test(NAME glimmer_transform_cuda_module_image_test COMMAND glimmer_transform_cuda_module_image_test)
add_executable(glimmer_ptx_transform_test
    interceptor/ptx_transform_test.cc
)
target_include_directories(glimmer_ptx_transform_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_ptx_transform_test PRIVATE Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_ptx_transform_test)
add_test(NAME glimmer_ptx_transform_test COMMAND glimmer_ptx_transform_test)
add_test(NAME glimmer_ptx_aggregate_parameters_test
    COMMAND glimmer_ptx_transform_test --aggregate-parameters)
set_tests_properties(glimmer_ptx_aggregate_parameters_test PROPERTIES TIMEOUT 30)
if(CUDAToolkit_NVCC_EXECUTABLE)
    add_test(NAME glimmer_ptx_compiler_test
        COMMAND "${CMAKE_COMMAND}"
            "-DTRANSFORMER=$<TARGET_FILE:glimmer_ptx_transform_test>"
            "-DNVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ptx-compiler"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/transform/check_compiler_ptx.cmake")
    set_tests_properties(glimmer_ptx_compiler_test PROPERTIES TIMEOUT 120)
endif()

add_executable(glimmer_core_execution_priority_test core/execution_priority_test.cc)
target_link_libraries(glimmer_core_execution_priority_test PRIVATE Glimmer::core Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_core_execution_priority_test)
add_test(NAME glimmer_core_execution_priority_test COMMAND glimmer_core_execution_priority_test)

add_executable(glimmer_backend_execution_submission_queue_test
    backend/execution_submission_queue_test.cc
)
target_include_directories(glimmer_backend_execution_submission_queue_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/backend"
)
target_link_libraries(glimmer_backend_execution_submission_queue_test
    PRIVATE Glimmer::backend
)
glimmer_enable_clang_tidy(glimmer_backend_execution_submission_queue_test)
add_test(NAME glimmer_backend_execution_submission_queue_test
    COMMAND glimmer_backend_execution_submission_queue_test
)
set_tests_properties(glimmer_backend_execution_submission_queue_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_execution_protocol_test control/execution_protocol_test.cc)
target_link_libraries(glimmer_control_execution_protocol_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_execution_protocol_test)
add_test(NAME glimmer_control_execution_protocol_test COMMAND glimmer_control_execution_protocol_test)
