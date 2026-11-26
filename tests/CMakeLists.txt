add_executable(glimmer_control_task_recovery_endpoint_test
    control/task_recovery_endpoint_test.cc)
target_include_directories(glimmer_control_task_recovery_endpoint_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/core")
target_link_libraries(glimmer_control_task_recovery_endpoint_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_recovery_endpoint_test)
add_test(NAME glimmer_control_task_recovery_endpoint_test
    COMMAND glimmer_control_task_recovery_endpoint_test)
set_tests_properties(glimmer_control_task_recovery_endpoint_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_task_recovery_protocol_test
    control/task_recovery_protocol_test.cc)
target_include_directories(glimmer_control_task_recovery_protocol_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_task_recovery_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_recovery_protocol_test)
add_test(NAME glimmer_control_task_recovery_protocol_test
    COMMAND glimmer_control_task_recovery_protocol_test)
set_tests_properties(glimmer_control_task_recovery_protocol_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_task_recovery_journal_test
    control/task_recovery_journal_test.cc)
target_include_directories(glimmer_control_task_recovery_journal_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_task_recovery_journal_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_recovery_journal_test)
add_test(NAME glimmer_control_task_recovery_journal_test
    COMMAND glimmer_control_task_recovery_journal_test)
set_tests_properties(glimmer_control_task_recovery_journal_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_unix_socket_task_execution_server_test
    control/unix_socket_task_execution_server_test.cc
    "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc")
target_include_directories(glimmer_control_unix_socket_task_execution_server_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_compile_definitions(glimmer_control_unix_socket_task_execution_server_test PRIVATE
    GLIMMER_ENDPOINT_TESTING)
target_link_libraries(glimmer_control_unix_socket_task_execution_server_test PRIVATE
    Glimmer::control Glimmer::project_options)
target_link_options(glimmer_control_unix_socket_task_execution_server_test PRIVATE
    "LINKER:--wrap=lstat" "LINKER:--wrap=listen" "LINKER:--wrap=getrandom")
glimmer_enable_clang_tidy(glimmer_control_unix_socket_task_execution_server_test)
add_test(NAME glimmer_control_unix_socket_task_execution_server_test
    COMMAND glimmer_control_unix_socket_task_execution_server_test)
set_tests_properties(glimmer_control_unix_socket_task_execution_server_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_shared_task_control_client_test
    control/shared_task_control_client_test.cc)
target_include_directories(glimmer_control_shared_task_control_client_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_shared_task_control_client_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_shared_task_control_client_test)
add_test(NAME glimmer_control_shared_task_control_client_test
    COMMAND glimmer_control_shared_task_control_client_test)
set_tests_properties(glimmer_control_shared_task_control_client_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_task_execution_endpoint_test
    control/task_execution_endpoint_test.cc
    "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc")
target_include_directories(glimmer_control_task_execution_endpoint_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_compile_definitions(glimmer_control_task_execution_endpoint_test PRIVATE
    GLIMMER_ENDPOINT_TESTING)
target_link_libraries(glimmer_control_task_execution_endpoint_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_execution_endpoint_test)
add_test(NAME glimmer_control_task_execution_endpoint_test
    COMMAND glimmer_control_task_execution_endpoint_test)
set_tests_properties(glimmer_control_task_execution_endpoint_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_versioned_remote_launch_gate_test
    control/versioned_remote_launch_gate_test.cc)
target_include_directories(glimmer_control_versioned_remote_launch_gate_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_versioned_remote_launch_gate_test PRIVATE Glimmer::control)
target_compile_definitions(glimmer_control_versioned_remote_launch_gate_test PRIVATE
    GLIMMER_ENABLE_TEST_HOOKS)
glimmer_enable_clang_tidy(glimmer_control_versioned_remote_launch_gate_test)
add_test(NAME glimmer_control_versioned_remote_launch_gate_test
    COMMAND glimmer_control_versioned_remote_launch_gate_test)
set_tests_properties(glimmer_control_versioned_remote_launch_gate_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_task_execution_admission_test
    control/task_execution_admission_test.cc)
target_include_directories(glimmer_control_task_execution_admission_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/core")
target_link_libraries(glimmer_control_task_execution_admission_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_execution_admission_test)
add_test(NAME glimmer_control_task_execution_admission_test
    COMMAND glimmer_control_task_execution_admission_test)
set_tests_properties(glimmer_control_task_execution_admission_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_task_execution_protocol_test
    control/task_execution_protocol_test.cc)
target_link_libraries(glimmer_control_task_execution_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_task_execution_protocol_test)
add_test(NAME glimmer_control_task_execution_protocol_test
    COMMAND glimmer_control_task_execution_protocol_test)
set_tests_properties(glimmer_control_task_execution_protocol_test PROPERTIES TIMEOUT 30)

add_library(glimmer_control_abi_v2_consumer OBJECT control/abi_v2/consumer.cc)
target_include_directories(glimmer_control_abi_v2_consumer BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/control/abi_v2/include"
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(glimmer_control_abi_v2_consumer PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_abi_v2_consumer)
add_executable(glimmer_control_abi_v2_test control/abi_v2/provider.cc
    $<TARGET_OBJECTS:glimmer_control_abi_v2_consumer>)
set_target_properties(glimmer_control_abi_v2_consumer glimmer_control_abi_v2_test
    PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_link_libraries(glimmer_control_abi_v2_test PRIVATE Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_abi_v2_test)
add_test(NAME glimmer_control_abi_v2_test COMMAND glimmer_control_abi_v2_test)
set_tests_properties(glimmer_control_abi_v2_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_native_epoch_protocol_test
    control/native_epoch_protocol_test.cc)
target_link_libraries(glimmer_control_native_epoch_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_native_epoch_protocol_test)
add_test(NAME glimmer_control_native_epoch_protocol_test
    COMMAND glimmer_control_native_epoch_protocol_test)
set_tests_properties(glimmer_control_native_epoch_protocol_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_execution_graph_protocol_test
    control/execution_graph_protocol_test.cc)
target_link_libraries(glimmer_control_execution_graph_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_graph_protocol_test)
add_test(NAME glimmer_control_execution_graph_protocol_test
    COMMAND glimmer_control_execution_graph_protocol_test)
set_tests_properties(glimmer_control_execution_graph_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_process_memory_protocol_test
    control/execution_process_memory_protocol_test.cc)
target_link_libraries(glimmer_control_execution_process_memory_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_process_memory_protocol_test)
add_test(NAME glimmer_control_execution_process_memory_protocol_test
    COMMAND glimmer_control_execution_process_memory_protocol_test)
set_tests_properties(glimmer_control_execution_process_memory_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_nccl_protocol_test
    control/execution_nccl_protocol_test.cc)
target_link_libraries(glimmer_control_execution_nccl_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_nccl_protocol_test)
add_test(NAME glimmer_control_execution_nccl_protocol_test
    COMMAND glimmer_control_execution_nccl_protocol_test)
set_tests_properties(glimmer_control_execution_nccl_protocol_test PROPERTIES TIMEOUT 45)
add_executable(glimmer_control_execution_pointer_protocol_test
    control/execution_pointer_protocol_test.cc)
target_link_libraries(glimmer_control_execution_pointer_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_pointer_protocol_test)
add_test(NAME glimmer_control_execution_pointer_protocol_test
    COMMAND glimmer_control_execution_pointer_protocol_test)
set_tests_properties(glimmer_control_execution_pointer_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_function_protocol_test
    control/execution_function_protocol_test.cc)
target_link_libraries(glimmer_control_execution_function_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_function_protocol_test)
add_test(NAME glimmer_control_execution_function_protocol_test
    COMMAND glimmer_control_execution_function_protocol_test)
set_tests_properties(glimmer_control_execution_function_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_attention_oracle_test interceptor/attention_oracle_test.cc)
add_executable(glimmer_control_execution_jit_protocol_test control/execution_jit_protocol_test.cc)
target_link_libraries(glimmer_control_execution_jit_protocol_test PRIVATE Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_jit_protocol_test)
add_test(NAME glimmer_control_execution_jit_protocol_test COMMAND glimmer_control_execution_jit_protocol_test)
set_tests_properties(glimmer_control_execution_jit_protocol_test PROPERTIES TIMEOUT 30)
target_link_libraries(glimmer_attention_oracle_test PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_attention_oracle_test)
add_test(NAME glimmer_attention_oracle_test COMMAND glimmer_attention_oracle_test)
set_tests_properties(glimmer_attention_oracle_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_rnn_lstm_oracle_test interceptor/rnn_lstm_oracle_test.cc)
target_link_libraries(glimmer_rnn_lstm_oracle_test PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_rnn_lstm_oracle_test)
add_test(NAME glimmer_rnn_lstm_oracle_test COMMAND glimmer_rnn_lstm_oracle_test)
set_tests_properties(glimmer_rnn_lstm_oracle_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_protocol_test control/execution_dnn_protocol_test.cc)
add_executable(glimmer_control_execution_dnn_profile_protocol_test
    control/execution_dnn_profile_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_profile_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_profile_protocol_test)
add_test(NAME glimmer_control_execution_dnn_profile_protocol_test
    COMMAND glimmer_control_execution_dnn_profile_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_profile_protocol_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_legacy_protocol_test
    control/execution_dnn_legacy_protocol_test.cc)
target_link_libraries(glimmer_control_execution_dnn_legacy_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_legacy_protocol_test)
add_test(NAME glimmer_control_execution_dnn_legacy_protocol_test
    COMMAND glimmer_control_execution_dnn_legacy_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_legacy_protocol_test PROPERTIES TIMEOUT 30)
target_link_libraries(glimmer_control_execution_dnn_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_protocol_test)
add_test(NAME glimmer_control_execution_dnn_protocol_test
    COMMAND glimmer_control_execution_dnn_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_protocol_test PROPERTIES TIMEOUT 30)
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
if(TARGET glimmer_execution_binary_images)
    add_dependencies(glimmer_transform_cuda_module_image_test glimmer_execution_binary_images)
    get_property(glimmer_binary_compiler_inputs GLOBAL PROPERTY GLIMMER_BINARY_COMPILER_INPUTS)
    add_test(NAME glimmer_transform_cuda_compiler_image_test
        COMMAND glimmer_transform_cuda_module_image_test ${glimmer_binary_compiler_inputs})
endif()
add_executable(glimmer_core_execution_priority_test core/execution_priority_test.cc)
target_link_libraries(glimmer_core_execution_priority_test PRIVATE Glimmer::core Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_core_execution_priority_test)
add_test(NAME glimmer_core_execution_priority_test COMMAND glimmer_core_execution_priority_test)
add_executable(glimmer_native_execution_epoch_test
    interceptor/native_execution_epoch_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/native_execution_epoch.cc")
target_include_directories(glimmer_native_execution_epoch_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_native_execution_epoch_test PRIVATE
    Glimmer::control Glimmer::project_options Threads::Threads)
glimmer_enable_clang_tidy(glimmer_native_execution_epoch_test)
add_test(NAME glimmer_native_execution_epoch_test COMMAND glimmer_native_execution_epoch_test)
set_tests_properties(glimmer_native_execution_epoch_test PROPERTIES TIMEOUT 20)
add_library(glimmer_loaded_image_path_fixture SHARED interceptor/loaded_image_path_fixture.cc)
target_link_libraries(glimmer_loaded_image_path_fixture PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_loaded_image_path_fixture)
add_executable(glimmer_loaded_image_path_test
    interceptor/loaded_image_path_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/loaded_image_path.cc")
target_include_directories(glimmer_loaded_image_path_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_loaded_image_path_test PRIVATE
    Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
glimmer_enable_clang_tidy(glimmer_loaded_image_path_test)
if(GLIMMER_ENABLE_SANITIZERS)
    target_compile_definitions(glimmer_loaded_image_path_test PRIVATE GLIMMER_SANITIZED_IMAGE_TEST)
endif()
add_dependencies(glimmer_loaded_image_path_test glimmer_loaded_image_path_fixture)
add_test(NAME glimmer_loaded_image_path_test
    COMMAND glimmer_loaded_image_path_test $<TARGET_FILE:glimmer_loaded_image_path_fixture>)
set_tests_properties(glimmer_loaded_image_path_test PROPERTIES TIMEOUT 20)
add_executable(glimmer_native_execution_heartbeat_test
    interceptor/native_execution_heartbeat_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/native_execution_heartbeat.cc")
target_include_directories(glimmer_native_execution_heartbeat_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_native_execution_heartbeat_test PRIVATE
    Glimmer::control Glimmer::project_options Threads::Threads)
glimmer_enable_clang_tidy(glimmer_native_execution_heartbeat_test)
add_test(NAME glimmer_native_execution_heartbeat_test COMMAND glimmer_native_execution_heartbeat_test)
set_tests_properties(glimmer_native_execution_heartbeat_test PROPERTIES TIMEOUT 20)
add_executable(glimmer_process_call_gate_test
    interceptor/process_call_gate_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/process_call_gate.cc")
target_include_directories(glimmer_process_call_gate_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_process_call_gate_test PRIVATE
    Glimmer::project_options Threads::Threads)
glimmer_enable_clang_tidy(glimmer_process_call_gate_test)
add_test(NAME glimmer_process_call_gate_test COMMAND glimmer_process_call_gate_test)
set_tests_properties(glimmer_process_call_gate_test PROPERTIES TIMEOUT 20)
add_executable(glimmer_operation_order_test
    interceptor/operation_order_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/operation_order.cc")
target_include_directories(glimmer_operation_order_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_operation_order_test PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_operation_order_test)
add_test(NAME glimmer_operation_order_test COMMAND glimmer_operation_order_test)
add_executable(glimmer_owned_kernel_arguments_test
    interceptor/owned_kernel_arguments_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/owned_kernel_arguments.cc")
target_include_directories(glimmer_owned_kernel_arguments_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_owned_kernel_arguments_test PRIVATE Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_owned_kernel_arguments_test)
add_test(NAME glimmer_owned_kernel_arguments_test COMMAND glimmer_owned_kernel_arguments_test)
add_executable(glimmer_ptx_dump_fixture interceptor/ptx_dump_fixture.cc)
target_link_libraries(glimmer_ptx_dump_fixture PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_ptx_dump_fixture)
add_executable(glimmer_embedded_ptx_test
    interceptor/embedded_ptx_test.cc
    "${PROJECT_SOURCE_DIR}/src/interceptor/internal/embedded_ptx.cc")
target_include_directories(glimmer_embedded_ptx_test PRIVATE "${PROJECT_SOURCE_DIR}/src/interceptor")
target_link_libraries(glimmer_embedded_ptx_test PRIVATE Glimmer::transform Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_embedded_ptx_test)
add_test(NAME glimmer_embedded_ptx_test
    COMMAND glimmer_embedded_ptx_test $<TARGET_FILE:glimmer_ptx_dump_fixture>)
set_tests_properties(glimmer_embedded_ptx_test PROPERTIES
    ENVIRONMENT "GLIMMER_PERSISTENT_RUNTIME=1" TIMEOUT 20)
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

if(Python3_Interpreter_FOUND)
    if(GLIMMER_BUILD_CUDA_WORKLOAD AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CUDAToolkit_NVCC_EXECUTABLE)
        add_test(NAME glimmer_cuda_compiler_selection_test
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build/cuda_compiler_selection_test.py"
                --cmake "${CMAKE_COMMAND}" --toolkit-root "${CUDAToolkit_BIN_DIR}/.."
                --expected-nvcc "${CUDAToolkit_NVCC_EXECUTABLE}")
        set_tests_properties(glimmer_cuda_compiler_selection_test PROPERTIES TIMEOUT 120)
    endif()
    add_test(NAME glimmer_session_scheduling_boundary_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/backend/session_scheduling_boundary_test.py")
    set_tests_properties(glimmer_session_scheduling_boundary_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_cudnn8_legacy_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/cudnn8_legacy_audit_test.py")
    set_tests_properties(glimmer_cudnn8_legacy_audit_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_rnn_v8_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/rnn_v8_audit_test.py")
    set_tests_properties(glimmer_rnn_v8_audit_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_artifact_paths_test
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build/artifact_paths_test.py")
    set_tests_properties(glimmer_artifact_paths_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_test_runtime_build_test
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build/execution_test_runtime_test.py")
    set_tests_properties(glimmer_execution_test_runtime_build_test PROPERTIES
        TIMEOUT 120 LABELS no-instrumented-code)
    add_test(NAME glimmer_execution_host_optimization_build_test
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/build/execution_host_optimization_test.py")
    set_tests_properties(glimmer_execution_host_optimization_build_test PROPERTIES
        TIMEOUT 60 LABELS no-instrumented-code)
    add_test(NAME glimmer_execution_cycle_witness_parser_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/control/execution_cycle_witness_test.py")
    set_tests_properties(glimmer_execution_cycle_witness_parser_test PROPERTIES
        TIMEOUT 180 LABELS no-instrumented-code)
    if(TARGET glimmer_execution_service)
        add_test(NAME glimmer_execution_sharded_signal_cli_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/control/execution_sharded_signal_cli_test.py"
                "$<TARGET_FILE:glimmer_execution_service>" "${GLIMMER_BUILD_CUDA_PERSISTENT_GEMM}")
        set_tests_properties(glimmer_execution_sharded_signal_cli_test PROPERTIES TIMEOUT 30)
        add_test(NAME glimmer_execution_priority_space_share_cli_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/control/execution_priority_space_share_cli_test.py"
                "$<TARGET_FILE:glimmer_execution_service>")
        set_tests_properties(glimmer_execution_priority_space_share_cli_test PROPERTIES TIMEOUT 30)
    endif()
    add_test(NAME glimmer_execution_unified_copy_numerical_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/unified_copy_gpu_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_execution_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/unified_copy_execution_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_h2h_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/h2h_copy_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_h2h_order_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/h2h_order_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_h2d_order_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/h2d_order_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_per_thread_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/unified_copy_per_thread_receipt_test.py")
    add_test(NAME glimmer_execution_unified_copy_per_thread_order_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/h2h_per_thread_order_receipt_test.py")
    add_test(NAME glimmer_execution_lt_scalar_bias_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/interceptor/lt_scalar_bias_receipt_test.py")
    # These receipt validators use only Python; Debug already runs them.
    set_tests_properties(glimmer_execution_unified_copy_numerical_receipt_test
        glimmer_execution_unified_copy_execution_receipt_test
        glimmer_execution_unified_copy_h2h_receipt_test
        glimmer_execution_unified_copy_h2h_order_receipt_test
        glimmer_execution_unified_copy_h2d_order_receipt_test
        glimmer_execution_unified_copy_per_thread_receipt_test
        glimmer_execution_unified_copy_per_thread_order_receipt_test
        glimmer_execution_lt_scalar_bias_receipt_test PROPERTIES
        TIMEOUT 30 LABELS no-instrumented-code)
    add_test(NAME glimmer_nccl_dependency_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/build/nccl_dependency_test.py"
            --cmake "${CMAKE_COMMAND}")
    set_tests_properties(glimmer_nccl_dependency_test PROPERTIES
        TIMEOUT 45 LABELS no-instrumented-code)
    add_test(NAME glimmer_execution_priority_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_priority_audit_test.py")
    add_test(NAME glimmer_execution_copy_audit_parser_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_copy_audit_test.py")
    set_tests_properties(glimmer_execution_copy_audit_parser_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_arrival_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_arrival_audit_test.py")
    set_tests_properties(glimmer_execution_arrival_audit_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_cohort_exec_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_cohort_exec_test.py")
    set_tests_properties(glimmer_execution_cohort_exec_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_application_priority_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/application_priority_audit_test.py")
    add_test(NAME glimmer_bert_candidate_validation_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_candidate_validation_test.py")
    set_tests_properties(glimmer_bert_candidate_validation_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_sgemm_coverage_candidates_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_sgemm_coverage_candidates_test.py")
    set_tests_properties(glimmer_bert_sgemm_coverage_candidates_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_sgemm_descriptor_probe_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_sgemm_descriptor_probe_test.py")
    set_tests_properties(glimmer_bert_sgemm_descriptor_probe_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_sgemm_ncu_process_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_sgemm_ncu_process_test.py")
    set_tests_properties(glimmer_bert_sgemm_ncu_process_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_ncu_client_broker_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_ncu_client_broker_test.py")
    set_tests_properties(glimmer_bert_ncu_client_broker_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_ncu_init_probe_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_ncu_init_probe_test.py")
    set_tests_properties(glimmer_execution_ncu_init_probe_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_ncu_replay_exec_test
        COMMAND "${Python3_EXECUTABLE}" -B -W error
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_ncu_replay_exec_test.py")
    set_tests_properties(glimmer_bert_ncu_replay_exec_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_ncu_service_proxy_test
        COMMAND "${Python3_EXECUTABLE}" -B -W error
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_ncu_service_proxy_test.py")
    set_tests_properties(glimmer_bert_ncu_service_proxy_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_ncu_relay_test
        COMMAND "${Python3_EXECUTABLE}" -B -W error
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_ncu_relay_test.py")
    set_tests_properties(glimmer_bert_ncu_relay_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_ncu_sibling_test
        COMMAND "${Python3_EXECUTABLE}" -B -W error
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_ncu_sibling_test.py")
    set_tests_properties(glimmer_bert_ncu_sibling_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_aggregate_sm_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/aggregate_sm_audit_test.py")
    set_tests_properties(glimmer_aggregate_sm_audit_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_device_observer_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_device_observer_test.py")
    set_tests_properties(glimmer_bert_device_observer_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_bert_device_capture_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_device_capture_test.py")
    set_tests_properties(glimmer_bert_device_capture_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_application_block_completion_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/application_block_completion_audit_test.py")
    add_test(NAME glimmer_execution_semantic_audit_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_semantic_audit_test.py")
    add_test(NAME glimmer_execution_application_receipt_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_application_receipt_test.py")
    if(TARGET glimmer_execution_nccl_service_test)
        add_test(NAME glimmer_execution_semantic_service_audit_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/execution_semantic_service_test.py"
                "$<TARGET_FILE:glimmer_execution_nccl_service_test>")
        set_tests_properties(glimmer_execution_semantic_service_audit_test PROPERTIES TIMEOUT 240)
    endif()
    if(NOT GLIMMER_BUILD_GEMM_COMPARISON)
        add_test(NAME glimmer_cutlass_dependency_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/build/cutlass_dependency_test.py"
                --cmake "${CMAKE_COMMAND}"
        )
        set_tests_properties(glimmer_cutlass_dependency_test PROPERTIES LABELS no-instrumented-code)
    endif()
    add_test(NAME glimmer_bert_colocation_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_colocation_test.py
    )
    add_test(NAME glimmer_check_script_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/check_script_test.py
    )
    set_tests_properties(glimmer_check_script_test PROPERTIES LABELS no-instrumented-code)
    add_test(NAME glimmer_bert_handoff_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_handoff_test.py
    )
    add_test(NAME glimmer_bert_workload_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_workload_test.py
    )
    add_test(NAME glimmer_bert_start_barrier_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/bert_start_barrier_test.py
    )
    set_tests_properties(glimmer_bert_start_barrier_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_pytorch_environment_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/pytorch_environment_test.py
    )
    add_test(NAME glimmer_pytorch_measurements_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/pytorch_measurements_test.py
    )
    add_test(NAME glimmer_persistent_mlp_handoff_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/persistent_mlp_handoff_test.py
    )
    add_test(NAME glimmer_persistent_mlp_acceptance_test
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/persistent_mlp_acceptance_test.py
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

add_executable(glimmer_control_execution_protocol_test control/execution_protocol_test.cc)
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
target_link_libraries(glimmer_control_execution_protocol_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_execution_protocol_test)
add_test(NAME glimmer_control_execution_protocol_test COMMAND glimmer_control_execution_protocol_test)

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

add_executable(glimmer_control_native_epoch_admission_test control/native_epoch_admission_test.cc)
target_link_libraries(glimmer_control_native_epoch_admission_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_native_epoch_admission_test)
add_test(NAME glimmer_control_native_epoch_admission_test
    COMMAND glimmer_control_native_epoch_admission_test)
add_test(NAME glimmer_control_native_epoch_socket_test
    COMMAND glimmer_control_native_epoch_admission_test --socket)
set_tests_properties(glimmer_control_native_epoch_admission_test PROPERTIES TIMEOUT 20)
set_tests_properties(glimmer_control_native_epoch_socket_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)

add_executable(glimmer_control_cooperative_admission_test control/cooperative_admission_test.cc)
add_executable(glimmer_control_cooperative_gate_quarantine_test
    control/cooperative_gate_quarantine_test.cc
    "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc")
target_include_directories(glimmer_control_cooperative_gate_quarantine_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_compile_definitions(glimmer_control_cooperative_gate_quarantine_test PRIVATE
    GLIMMER_ENDPOINT_TESTING)
target_link_libraries(glimmer_control_cooperative_gate_quarantine_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_cooperative_gate_quarantine_test)
add_test(NAME glimmer_control_cooperative_gate_quarantine_test
    COMMAND glimmer_control_cooperative_gate_quarantine_test)
set_tests_properties(glimmer_control_cooperative_gate_quarantine_test PROPERTIES TIMEOUT 20)
add_executable(glimmer_control_cooperative_launch_gate_test control/cooperative_launch_gate_test.cc)
add_executable(glimmer_control_native_epoch_launch_gate_test control/native_epoch_launch_gate_test.cc)
target_include_directories(glimmer_control_native_epoch_launch_gate_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_native_epoch_launch_gate_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_native_epoch_launch_gate_test)
add_test(NAME glimmer_control_native_epoch_launch_gate_test COMMAND glimmer_control_native_epoch_launch_gate_test)
add_test(NAME glimmer_control_native_epoch_launch_gate_socket_test COMMAND glimmer_control_native_epoch_launch_gate_test --socket)
set_tests_properties(glimmer_control_native_epoch_launch_gate_test PROPERTIES TIMEOUT 30)
set_tests_properties(glimmer_control_native_epoch_launch_gate_socket_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
add_test(NAME glimmer_control_native_epoch_service_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/control/native_epoch_service_test.py $<TARGET_FILE:glimmer_control_service>)
set_tests_properties(glimmer_control_native_epoch_service_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
target_link_libraries(glimmer_control_cooperative_launch_gate_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_cooperative_launch_gate_test)
add_test(NAME glimmer_control_cooperative_launch_gate_test COMMAND glimmer_control_cooperative_launch_gate_test)
set_tests_properties(glimmer_control_cooperative_launch_gate_test PROPERTIES TIMEOUT 20)
target_link_libraries(glimmer_control_cooperative_admission_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_cooperative_admission_test)
add_test(NAME glimmer_control_cooperative_admission_test
    COMMAND glimmer_control_cooperative_admission_test)
set_tests_properties(glimmer_control_cooperative_admission_test PROPERTIES TIMEOUT 10)

add_executable(glimmer_control_cooperative_remote_process_test
    control/cooperative_remote_process_test.cc)
target_link_libraries(glimmer_control_cooperative_remote_process_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_cooperative_remote_process_test)
add_test(NAME glimmer_control_cooperative_remote_process_test
    COMMAND glimmer_control_cooperative_remote_process_test
        $<TARGET_FILE:glimmer_task_execution_service>)
set_tests_properties(glimmer_control_cooperative_remote_process_test PROPERTIES TIMEOUT 20)

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
add_executable(glimmer_control_shared_control_channel_test control/shared_control_channel_test.cc)
target_include_directories(glimmer_control_shared_control_channel_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_shared_control_channel_test PRIVATE Glimmer::control)
target_link_options(glimmer_control_shared_control_channel_test PRIVATE "LINKER:--wrap=poll")
glimmer_enable_clang_tidy(glimmer_control_shared_control_channel_test)
add_test(NAME glimmer_control_shared_control_channel_test
    COMMAND glimmer_control_shared_control_channel_test)
set_tests_properties(glimmer_control_shared_control_channel_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
add_executable(glimmer_control_owner_control_channel_test control/owner_control_channel_test.cc)
target_include_directories(glimmer_control_owner_control_channel_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_owner_control_channel_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_owner_control_channel_test)
add_test(NAME glimmer_control_owner_control_channel_test
    COMMAND glimmer_control_owner_control_channel_test)
set_tests_properties(glimmer_control_owner_control_channel_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
add_test(NAME glimmer_control_cooperative_shared_memory_test
    COMMAND glimmer_control_cooperative_launch_gate_test)
add_executable(glimmer_control_owned_control_exchange_test control/owned_control_exchange_test.cc)
target_include_directories(glimmer_control_owned_control_exchange_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_owned_control_exchange_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_owned_control_exchange_test)
add_test(NAME glimmer_control_owned_control_exchange_test
    COMMAND glimmer_control_owned_control_exchange_test)
set_tests_properties(glimmer_control_owned_control_exchange_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
add_executable(glimmer_control_remote_control_operation_test control/remote_control_operation_test.cc)
target_include_directories(glimmer_control_remote_control_operation_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_remote_control_operation_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_remote_control_operation_test)
add_test(NAME glimmer_control_remote_control_operation_test
    COMMAND glimmer_control_remote_control_operation_test)
set_tests_properties(glimmer_control_remote_control_operation_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_process_control_mailbox_test control/process_control_mailbox_test.cc)
target_include_directories(glimmer_control_process_control_mailbox_test
    PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_process_control_mailbox_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_process_control_mailbox_test)
add_test(NAME glimmer_control_process_control_mailbox_test
    COMMAND glimmer_control_process_control_mailbox_test)
set_tests_properties(glimmer_control_process_control_mailbox_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_local_control_wake_test control/local_control_wake_test.cc)
target_include_directories(glimmer_control_local_control_wake_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_local_control_wake_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_local_control_wake_test)
add_test(NAME glimmer_control_local_control_wake_test COMMAND glimmer_control_local_control_wake_test)
set_tests_properties(glimmer_control_local_control_wake_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_prepared_control_future_test control/prepared_control_future_test.cc)
target_include_directories(glimmer_control_prepared_control_future_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_prepared_control_future_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_prepared_control_future_test)
add_test(NAME glimmer_control_prepared_control_future_test COMMAND glimmer_control_prepared_control_future_test)
set_tests_properties(glimmer_control_prepared_control_future_test PROPERTIES TIMEOUT 30)
add_executable(glimmer_control_adjacent_control_handoff_test control/adjacent_control_handoff_test.cc)
target_include_directories(glimmer_control_adjacent_control_handoff_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_adjacent_control_handoff_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_adjacent_control_handoff_test)
add_test(NAME glimmer_control_adjacent_control_handoff_test COMMAND glimmer_control_adjacent_control_handoff_test)
set_tests_properties(glimmer_control_adjacent_control_handoff_test PROPERTIES TIMEOUT 30)

add_executable(glimmer_control_owner_pumped_control_gate_test control/owner_pumped_control_gate_test.cc)
target_include_directories(glimmer_control_owner_pumped_control_gate_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_control_owner_pumped_control_gate_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_owner_pumped_control_gate_test)
add_test(NAME glimmer_control_owner_pumped_control_gate_test COMMAND glimmer_control_owner_pumped_control_gate_test)
set_tests_properties(glimmer_control_owner_pumped_control_gate_test PROPERTIES TIMEOUT 30)
add_test(NAME glimmer_control_cooperative_shared_memory_process_test
    COMMAND glimmer_control_cooperative_remote_process_test
        $<TARGET_FILE:glimmer_task_execution_service>)
add_test(NAME glimmer_control_native_epoch_shared_memory_test
    COMMAND glimmer_control_native_epoch_launch_gate_test --shared-socket)
set_tests_properties(glimmer_control_cooperative_shared_memory_test
    glimmer_control_cooperative_shared_memory_process_test
    glimmer_control_native_epoch_shared_memory_test PROPERTIES
    ENVIRONMENT "GLIMMER_CONTROL_TRANSPORT=shared_memory" TIMEOUT 60)
target_link_libraries(glimmer_control_unix_socket_client_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_unix_socket_client_test)
add_test(NAME glimmer_control_unix_socket_client_test
    COMMAND glimmer_control_unix_socket_client_test)
set_tests_properties(glimmer_control_unix_socket_client_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

add_executable(glimmer_control_service_process_test control/service_process_test.cc)
target_link_libraries(glimmer_control_service_process_test PRIVATE Glimmer::project_options)

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
