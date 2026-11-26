glimmer_add_test_program(glimmer_control_task_recovery_endpoint_test
    SOURCES control/task_recovery_endpoint_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/core"
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_task_recovery_protocol_test
    SOURCES control/task_recovery_protocol_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_task_recovery_journal_test
    SOURCES control/task_recovery_journal_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_unix_socket_task_execution_server_test
    SOURCES control/unix_socket_task_execution_server_test.cc
        "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    DEFINITIONS GLIMMER_ENDPOINT_TESTING
    LIBRARIES Glimmer::control Glimmer::project_options
    LINK_OPTIONS "LINKER:--wrap=lstat" "LINKER:--wrap=listen" "LINKER:--wrap=getrandom"
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_shared_task_control_client_test
    SOURCES control/shared_task_control_client_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_task_execution_endpoint_test
    SOURCES control/task_execution_endpoint_test.cc
        "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    DEFINITIONS GLIMMER_ENDPOINT_TESTING
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_versioned_remote_launch_gate_test
    SOURCES control/versioned_remote_launch_gate_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    DEFINITIONS GLIMMER_ENABLE_TEST_HOOKS
    LIBRARIES Glimmer::control
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_task_execution_admission_test
    SOURCES control/task_execution_admission_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/core"
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_task_execution_protocol_test
    SOURCES control/task_execution_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

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

glimmer_add_test_program(glimmer_control_native_epoch_protocol_test
    SOURCES control/native_epoch_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_execution_graph_protocol_test
    SOURCES control/execution_graph_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_process_memory_protocol_test
    SOURCES control/execution_process_memory_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_nccl_protocol_test
    SOURCES control/execution_nccl_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 45)
glimmer_add_test_program(glimmer_control_execution_pointer_protocol_test
    SOURCES control/execution_pointer_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_function_protocol_test
    SOURCES control/execution_function_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
add_executable(glimmer_attention_oracle_test interceptor/attention_oracle_test.cc)
glimmer_add_test_program(glimmer_control_execution_jit_protocol_test
    SOURCES control/execution_jit_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
target_link_libraries(glimmer_attention_oracle_test PRIVATE Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_attention_oracle_test)
add_test(NAME glimmer_attention_oracle_test COMMAND glimmer_attention_oracle_test)
set_tests_properties(glimmer_attention_oracle_test PROPERTIES TIMEOUT 30)
glimmer_add_test_program(glimmer_rnn_lstm_oracle_test
    SOURCES interceptor/rnn_lstm_oracle_test.cc
    LIBRARIES Glimmer::project_options
    TIMEOUT 30)
add_executable(glimmer_control_execution_dnn_protocol_test control/execution_dnn_protocol_test.cc)
glimmer_add_test_program(glimmer_control_execution_dnn_profile_protocol_test
    SOURCES control/execution_dnn_profile_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_dnn_legacy_protocol_test
    SOURCES control/execution_dnn_legacy_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
target_link_libraries(glimmer_control_execution_dnn_protocol_test PRIVATE
    Glimmer::control Glimmer::project_options)
glimmer_enable_clang_tidy(glimmer_control_execution_dnn_protocol_test)
add_test(NAME glimmer_control_execution_dnn_protocol_test
    COMMAND glimmer_control_execution_dnn_protocol_test)
set_tests_properties(glimmer_control_execution_dnn_protocol_test PROPERTIES TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_dnn_backend_protocol_test
    SOURCES control/execution_dnn_backend_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_dnn_backend_execution_protocol_test
    SOURCES control/execution_dnn_backend_execution_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_ptx_uniform_loop_test
    SOURCES transform/ptx_uniform_loop_test.cc
    LIBRARIES Glimmer::transform Glimmer::project_options
    TIMEOUT 30)
if(CUDAToolkit_NVCC_EXECUTABLE)
    add_test(NAME glimmer_ptx_uniform_loop_compiler_test COMMAND "${CMAKE_COMMAND}"
        "-DTRANSFORMER=$<TARGET_FILE:glimmer_ptx_uniform_loop_test>"
        "-DNVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ptx-uniform-loop"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/transform/check_uniform_loop_ptx.cmake")
    set_tests_properties(glimmer_ptx_uniform_loop_compiler_test PROPERTIES TIMEOUT 120)
endif()
glimmer_add_test_program(glimmer_transform_ptx_static_storage_test
    SOURCES transform/ptx_static_storage_test.cc
    LIBRARIES Glimmer::transform Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_transform_cuda_module_image_test
    SOURCES transform/cuda_module_image_test.cc
    LIBRARIES Glimmer::transform Glimmer::project_options)
if(TARGET glimmer_execution_binary_images)
    add_dependencies(glimmer_transform_cuda_module_image_test glimmer_execution_binary_images)
    get_property(glimmer_binary_compiler_inputs GLOBAL PROPERTY GLIMMER_BINARY_COMPILER_INPUTS)
    add_test(NAME glimmer_transform_cuda_compiler_image_test
        COMMAND glimmer_transform_cuda_module_image_test ${glimmer_binary_compiler_inputs})
endif()
glimmer_add_test_program(glimmer_core_execution_priority_test
    SOURCES core/execution_priority_test.cc
    LIBRARIES Glimmer::core Glimmer::project_options)
glimmer_add_test_program(glimmer_native_execution_epoch_test
    SOURCES interceptor/native_execution_epoch_test.cc
        "${PROJECT_SOURCE_DIR}/src/interceptor/internal/native_execution_epoch.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
    TIMEOUT 20)
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
glimmer_add_test_program(glimmer_native_execution_heartbeat_test
    SOURCES interceptor/native_execution_heartbeat_test.cc
        "${PROJECT_SOURCE_DIR}/src/interceptor/internal/native_execution_heartbeat.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
    TIMEOUT 20)
glimmer_add_test_program(glimmer_process_call_gate_test
    SOURCES interceptor/process_call_gate_test.cc
        "${PROJECT_SOURCE_DIR}/src/interceptor/internal/process_call_gate.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::project_options Threads::Threads
    TIMEOUT 20)
glimmer_add_test_program(glimmer_operation_order_test
    SOURCES interceptor/operation_order_test.cc
        "${PROJECT_SOURCE_DIR}/src/interceptor/internal/operation_order.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::project_options)
glimmer_add_test_program(glimmer_owned_kernel_arguments_test
    SOURCES interceptor/owned_kernel_arguments_test.cc
        "${PROJECT_SOURCE_DIR}/src/interceptor/internal/owned_kernel_arguments.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::transform Glimmer::project_options)
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
glimmer_add_test_program(glimmer_ptx_transform_test
    SOURCES interceptor/ptx_transform_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor"
    LIBRARIES Glimmer::transform Glimmer::project_options)
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

glimmer_add_test_program(glimmer_core_quota_ledger_test
    SOURCES core/quota_ledger_test.cc
    LIBRARIES Glimmer::core)

glimmer_add_test_program(glimmer_core_scheduler_test
    SOURCES core/scheduler_test.cc
    LIBRARIES Glimmer::core)

glimmer_add_test_program(glimmer_core_adaptive_slo_test
    SOURCES core/adaptive_slo_test.cc
    LIBRARIES Glimmer::core)

add_executable(glimmer_backend_simulated_test backend/simulated_backend_test.cc)

add_executable(glimmer_control_execution_protocol_test control/execution_protocol_test.cc)
glimmer_add_test_program(glimmer_control_execution_blas_protocol_test
    SOURCES control/execution_blas_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_execution_blas_extension_protocol_test
    SOURCES control/execution_blas_extension_protocol_test.cc
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 30)
target_link_libraries(glimmer_control_execution_protocol_test PRIVATE Glimmer::control)
glimmer_enable_clang_tidy(glimmer_control_execution_protocol_test)
add_test(NAME glimmer_control_execution_protocol_test COMMAND glimmer_control_execution_protocol_test)

glimmer_add_test_program(glimmer_control_execution_socket_test
    SOURCES control/execution_socket_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    LINK_OPTIONS "LINKER:--wrap=send" "LINKER:--wrap=sendmsg" "LINKER:--wrap=recvmsg"
    TIMEOUT 30)
add_test(NAME glimmer_control_execution_preparation_deadline_test
    COMMAND glimmer_control_execution_socket_test --preparation-deadlines)
set_tests_properties(glimmer_control_execution_preparation_deadline_test PROPERTIES TIMEOUT 30)

glimmer_add_test_program(glimmer_control_execution_receive_test
    SOURCES control/execution_socket_receive_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    LINK_OPTIONS "LINKER:--wrap=recvmsg" "LINKER:--wrap=eventfd" "LINKER:--wrap=read"
        "LINKER:--wrap=write" "LINKER:--wrap=ppoll"
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_execution_send_test
    SOURCES control/execution_socket_send_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    LINK_OPTIONS "LINKER:--wrap=send" "LINKER:--wrap=sendmsg"
    TIMEOUT 30)

target_link_libraries(glimmer_backend_simulated_test
    PRIVATE
        Glimmer::backend
)

glimmer_enable_clang_tidy(glimmer_backend_simulated_test)

add_test(NAME glimmer_backend_simulated_test COMMAND glimmer_backend_simulated_test)

glimmer_add_test_program(glimmer_backend_task_executor_test
    SOURCES backend/task_executor_test.cc
    LIBRARIES Glimmer::backend)

glimmer_add_test_program(glimmer_backend_execution_submission_queue_test
    SOURCES backend/execution_submission_queue_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/backend"
    LIBRARIES Glimmer::backend
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_process_memory_quota_test
    SOURCES control/process_memory_quota_test.cc
    LIBRARIES Glimmer::control)

glimmer_add_test_program(glimmer_control_composite_quota_test
    SOURCES control/composite_quota_test.cc
    LIBRARIES Glimmer::control)

glimmer_add_test_program(glimmer_control_device_capacity_quota_test
    SOURCES control/device_capacity_quota_test.cc
    LIBRARIES Glimmer::control)

glimmer_add_test_program(glimmer_control_shared_memory_quota_test
    SOURCES control/shared_memory_quota_test.cc
    LIBRARIES Glimmer::control)

add_executable(glimmer_control_task_admission_test control/task_admission_test.cc)

glimmer_add_test_program(glimmer_control_native_epoch_admission_test
    SOURCES control/native_epoch_admission_test.cc
    LIBRARIES Glimmer::control)
add_test(NAME glimmer_control_native_epoch_socket_test
    COMMAND glimmer_control_native_epoch_admission_test --socket)
set_tests_properties(glimmer_control_native_epoch_admission_test PROPERTIES TIMEOUT 20)
set_tests_properties(glimmer_control_native_epoch_socket_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)

add_executable(glimmer_control_cooperative_admission_test control/cooperative_admission_test.cc)
glimmer_add_test_program(glimmer_control_cooperative_gate_quarantine_test
    SOURCES control/cooperative_gate_quarantine_test.cc
        "${PROJECT_SOURCE_DIR}/src/control/task_execution_endpoint.cc"
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    DEFINITIONS GLIMMER_ENDPOINT_TESTING
    LIBRARIES Glimmer::control Glimmer::project_options
    TIMEOUT 20)
add_executable(glimmer_control_cooperative_launch_gate_test control/cooperative_launch_gate_test.cc)
glimmer_add_test_program(glimmer_control_native_epoch_launch_gate_test
    SOURCES control/native_epoch_launch_gate_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control)
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

glimmer_add_test_program(glimmer_control_task_admission_concurrency_test
    SOURCES control/task_admission_concurrency_test.cc
    LIBRARIES Glimmer::control
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_launch_gate_test
    SOURCES control/launch_gate_test.cc
    LIBRARIES Glimmer::control)

glimmer_add_test_program(glimmer_control_launch_gate_remote_process_test
    SOURCES control/launch_gate_remote_process_test.cc
    LIBRARIES Glimmer::control)
set_tests_properties(glimmer_control_launch_gate_remote_process_test PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 10
)

glimmer_add_test_program(glimmer_control_task_endpoint_test
    SOURCES control/task_endpoint_test.cc
    LIBRARIES Glimmer::backend Glimmer::control)

glimmer_add_test_program(glimmer_control_unix_socket_server_test
    SOURCES control/unix_socket_server_test.cc
    LIBRARIES Glimmer::control)
set_tests_properties(glimmer_control_unix_socket_server_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

glimmer_add_test_program(glimmer_control_socket_line_reader_test
    SOURCES control/socket_line_reader_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control)
set_tests_properties(glimmer_control_socket_line_reader_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 15)

add_executable(glimmer_control_unix_socket_client_test control/unix_socket_client_test.cc)
glimmer_add_test_program(glimmer_control_shared_control_channel_test
    SOURCES control/shared_control_channel_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    LINK_OPTIONS "LINKER:--wrap=poll")
set_tests_properties(glimmer_control_shared_control_channel_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
glimmer_add_test_program(glimmer_control_owner_control_channel_test
    SOURCES control/owner_control_channel_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control)
set_tests_properties(glimmer_control_owner_control_channel_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
add_test(NAME glimmer_control_cooperative_shared_memory_test
    COMMAND glimmer_control_cooperative_launch_gate_test)
glimmer_add_test_program(glimmer_control_owned_control_exchange_test
    SOURCES control/owned_control_exchange_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control)
set_tests_properties(glimmer_control_owned_control_exchange_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 30)
glimmer_add_test_program(glimmer_control_remote_control_operation_test
    SOURCES control/remote_control_operation_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_process_control_mailbox_test
    SOURCES control/process_control_mailbox_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_local_control_wake_test
    SOURCES control/local_control_wake_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_prepared_control_future_test
    SOURCES control/prepared_control_future_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)
glimmer_add_test_program(glimmer_control_adjacent_control_handoff_test
    SOURCES control/adjacent_control_handoff_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)

glimmer_add_test_program(glimmer_control_owner_pumped_control_gate_test
    SOURCES control/owner_pumped_control_gate_test.cc
    INCLUDES "${PROJECT_SOURCE_DIR}/src/control"
    LIBRARIES Glimmer::control
    TIMEOUT 30)
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

glimmer_add_test_program(glimmer_control_service_process_test
    SOURCES control/service_process_test.cc
    LIBRARIES Glimmer::project_options)
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

glimmer_add_test_program(glimmer_control_task_protocol_test
    SOURCES control/task_protocol_test.cc
    LIBRARIES Glimmer::control)
