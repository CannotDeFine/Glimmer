add_library(glimmer_execution_service_runtime runtime.cc host_owner.cc profile_archive.cc async_audit_log.cc
    host_stage_metrics.cc)
target_include_directories(glimmer_execution_service_runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/src/backend/cuda"
    "${PROJECT_SOURCE_DIR}/src/backend"
    "${CUDAToolkit_INCLUDE_DIRS}"
)
target_link_libraries(glimmer_execution_service_runtime PRIVATE
    Glimmer::cuda_backend Glimmer::backend Glimmer::core Glimmer::control Glimmer::project_options spdlog::spdlog
)
glimmer_enable_clang_tidy(glimmer_execution_service_runtime)
if(TARGET Glimmer::cudnn_headers)
    target_link_libraries(glimmer_execution_service_runtime PRIVATE Glimmer::cudnn_headers)
endif()
if(TARGET Glimmer::nccl_headers)
    # Injectable actor support alone does not bind a vendor dispatch. Public
    # activation additionally requires the explicit option and cold startup flag.
    target_sources(glimmer_execution_service_runtime PRIVATE nccl_actor.cc nccl_output_effects.cc)
    target_link_libraries(glimmer_execution_service_runtime PRIVATE Glimmer::nccl_headers Threads::Threads)
endif()

add_executable(glimmer_execution_service main.cc)
set_target_properties(glimmer_execution_service PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
target_include_directories(glimmer_execution_service PRIVATE
    "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
    "${PROJECT_SOURCE_DIR}/src/control"
)
target_link_libraries(glimmer_execution_service PRIVATE
    glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend Glimmer::control
    CUDA::cuda_driver CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::curand Glimmer::project_options
    spdlog::spdlog "${CMAKE_DL_LIBS}"
)
glimmer_enable_clang_tidy(glimmer_execution_service)
if(TARGET Glimmer::cudnn)
    target_link_libraries(glimmer_execution_service PRIVATE Glimmer::cudnn)
endif()
if(GLIMMER_ENABLE_NCCL_EXECUTION)
    target_link_libraries(glimmer_execution_service PRIVATE Glimmer::nccl)
    target_compile_definitions(glimmer_execution_service PRIVATE GLIMMER_ENABLE_NCCL_EXECUTION=1)
endif()
if(BUILD_TESTING)
    glimmer_add_test_program(glimmer_execution_host_stage_metrics_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_host_stage_metrics_test.cc"
            host_stage_metrics.cc
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARIES Glimmer::project_options
        TIMEOUT 15)
    glimmer_create_execution_test_runtime()
    glimmer_add_test_program(glimmer_execution_sgemm_range_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_sgemm_range_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
        LIBRARIES Glimmer::project_options
        TIMEOUT 15)
    glimmer_add_test_program(glimmer_execution_graph_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_graph_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_process_memory_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_process_memory_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_execution_pair_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_pair_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options Threads::Threads spdlog::spdlog
        TIMEOUT 90)
    add_test(NAME glimmer_execution_priority_space_share_test
        COMMAND glimmer_execution_pair_service_test --priority-space-share)
    set_tests_properties(glimmer_execution_priority_space_share_test PROPERTIES TIMEOUT 30)
    foreach(scenario drain departure arrival)
        add_test(NAME glimmer_execution_priority_space_share_${scenario}_test
            COMMAND glimmer_execution_pair_service_test --priority-space-share-${scenario})
        set_tests_properties(glimmer_execution_priority_space_share_${scenario}_test PROPERTIES TIMEOUT 30)
    endforeach()
    foreach(scenario mixed mixed-flags mixed-drain mixed-departure mixed-arrival)
        string(REPLACE "-" "_" test_suffix "${scenario}")
        add_test(NAME glimmer_execution_priority_space_share_${test_suffix}_test
            COMMAND glimmer_execution_pair_service_test --priority-space-share-${scenario})
        set_tests_properties(glimmer_execution_priority_space_share_${test_suffix}_test PROPERTIES TIMEOUT 30)
    endforeach()
    add_test(NAME glimmer_execution_global_pair_service_test
        COMMAND glimmer_execution_pair_service_test --global-pair-selection)
    set_tests_properties(glimmer_execution_global_pair_service_test PROPERTIES TIMEOUT 90)
    add_test(NAME glimmer_execution_global_pair_capacity_test
        COMMAND glimmer_execution_pair_service_test --global-pair-capacity)
    set_tests_properties(glimmer_execution_global_pair_capacity_test PROPERTIES TIMEOUT 90)
    if(GLIMMER_BUILD_CUDA_PERSISTENT_GEMM)
        add_test(NAME glimmer_execution_mixed_pair_candidates_test
            COMMAND glimmer_execution_pair_service_test --mixed-pair-candidates)
        set_tests_properties(glimmer_execution_mixed_pair_candidates_test PROPERTIES TIMEOUT 90)
    endif()
    add_test(NAME glimmer_execution_arrival_rebalance_test
        COMMAND glimmer_execution_pair_service_test --arrival-rebalance)
    set_tests_properties(glimmer_execution_arrival_rebalance_test PROPERTIES TIMEOUT 90)
    add_test(NAME glimmer_execution_isolated_next_partner_test
        COMMAND glimmer_execution_pair_service_test --isolated-next-partner)
    set_tests_properties(glimmer_execution_isolated_next_partner_test PROPERTIES TIMEOUT 90)
    add_test(NAME glimmer_execution_workload_aware_startup_test
        COMMAND glimmer_execution_pair_service_test --workload-aware-startup)
    add_test(NAME glimmer_execution_workload_aware_pairs_test
        COMMAND glimmer_execution_pair_service_test --workload-aware-pairs)
    set_tests_properties(glimmer_execution_workload_aware_startup_test
        glimmer_execution_workload_aware_pairs_test PROPERTIES TIMEOUT 90)
    if(TARGET Glimmer::nccl_headers)
        glimmer_add_test_program(glimmer_execution_nccl_actor_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_nccl_actor_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::nccl_headers Glimmer::project_options Threads::Threads
            TIMEOUT 90)
        glimmer_add_test_program(glimmer_execution_nccl_service_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_nccl_service_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::nccl_headers Glimmer::project_options Threads::Threads
            TIMEOUT 90)
        add_test(NAME glimmer_execution_nccl_async_source_cut_test
            COMMAND glimmer_execution_nccl_service_test --semantic async-source-cut)
        set_tests_properties(glimmer_execution_nccl_async_source_cut_test PROPERTIES TIMEOUT 30)
        glimmer_add_test_program(glimmer_execution_nccl_output_effects_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_nccl_output_effects_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::nccl_headers Glimmer::project_options
            TIMEOUT 30)
    endif()
    add_test(NAME glimmer_execution_pointer_metadata_service_test
        COMMAND glimmer_execution_service_test --pointer-metadata)
    set_tests_properties(glimmer_execution_pointer_metadata_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_unified_copy_service_test
        COMMAND glimmer_execution_service_test --unified-copy)
    set_tests_properties(glimmer_execution_unified_copy_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_unified_copy_oom_service_test
        COMMAND glimmer_execution_service_test --unified-copy-oom)
    set_tests_properties(glimmer_execution_unified_copy_oom_service_test PROPERTIES TIMEOUT 30)
    if(TARGET Glimmer::cudnn_headers)
        glimmer_add_test_program(glimmer_execution_dnn_profile_service_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_dnn_profile_service_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::project_options Glimmer::cudnn_headers
            TIMEOUT 30)
        glimmer_add_test_program(glimmer_execution_dnn_service_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_dnn_service_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::project_options Glimmer::cudnn_headers
            TIMEOUT 30)
        glimmer_add_test_program(glimmer_execution_dnn_backend_graph_service_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_dnn_backend_graph_service_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::project_options Glimmer::cudnn_headers
            TIMEOUT 30)
        glimmer_add_test_program(glimmer_execution_dnn_backend_execution_service_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_dnn_backend_execution_service_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::project_options Glimmer::cudnn_headers
            TIMEOUT 45)
        foreach(_backend_execution_mode disconnect foreign-pending queued-demand profile-rebind-pending)
            string(REPLACE "-" "_" _backend_execution_test "${_backend_execution_mode}")
            add_test(NAME glimmer_execution_dnn_backend_execution_${_backend_execution_test}_test
                COMMAND glimmer_execution_dnn_backend_execution_service_test
                    "--${_backend_execution_mode}")
            set_tests_properties(glimmer_execution_dnn_backend_execution_${_backend_execution_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
    endif()
    glimmer_add_test_program(glimmer_profile_archive_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/profile_archive_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::control
            Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_host_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_host_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_host_function_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_host_function_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options
        TIMEOUT 60)
    glimmer_add_test_program(glimmer_execution_priority_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_priority_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options spdlog::spdlog
        TIMEOUT 30)
    add_test(NAME glimmer_execution_priority_service_intake_test
        COMMAND glimmer_execution_priority_service_test --intake-race)
    set_tests_properties(glimmer_execution_priority_service_intake_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_priority_retired_retreat_test
        COMMAND glimmer_execution_priority_service_test --retired-retreat)
    set_tests_properties(glimmer_execution_priority_retired_retreat_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_priority_idle_test
        COMMAND glimmer_execution_priority_service_test --priority-idle)
    set_tests_properties(glimmer_execution_priority_idle_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_client_kernels_test
        COMMAND glimmer_execution_priority_service_test --client-kernels)
    set_tests_properties(glimmer_execution_client_kernels_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_workload_agnostic_service_test
        COMMAND glimmer_execution_priority_service_test --workload-agnostic)
    set_tests_properties(glimmer_execution_workload_agnostic_service_test PROPERTIES TIMEOUT 45)
    add_test(NAME glimmer_execution_unissued_priority_audit_test
        COMMAND glimmer_execution_priority_service_test --unissued-audit)
    set_tests_properties(glimmer_execution_unissued_priority_audit_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options spdlog::spdlog
        LINK_OPTIONS "LINKER:--wrap=ppoll" "LINKER:--wrap=__ppoll_chk")
    add_test(NAME glimmer_execution_sharded_signal_config_test
        COMMAND glimmer_execution_service_test --sharded-signal-config)
    add_test(NAME glimmer_execution_async_intake_boundaries_test
        COMMAND glimmer_execution_service_test --async-intake-boundaries)
    add_test(NAME glimmer_execution_async_intake_chain_test
        COMMAND glimmer_execution_service_test --async-intake check on)
    set_tests_properties(glimmer_execution_async_intake_boundaries_test
        glimmer_execution_async_intake_chain_test PROPERTIES TIMEOUT 90)
    add_test(NAME glimmer_execution_issue_depth_test COMMAND glimmer_execution_service_test --issue-depth)
    set_tests_properties(glimmer_execution_issue_depth_test PROPERTIES TIMEOUT 30)
    set_tests_properties(glimmer_execution_service_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_async_frontier_service_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_async_frontier_service_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    add_test(NAME glimmer_execution_context_purpose_service_test
        COMMAND glimmer_execution_service_test --context-purpose)
    set_tests_properties(glimmer_execution_context_purpose_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_runtime_function_service_test
        COMMAND glimmer_execution_service_test --runtime-attributes)
    set_tests_properties(glimmer_execution_runtime_function_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_geam_service_test
        COMMAND glimmer_execution_service_test --geam)
    set_tests_properties(glimmer_execution_geam_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_blas_extension_service_test
        COMMAND glimmer_execution_service_test --blas-extension)
    set_tests_properties(glimmer_execution_blas_extension_service_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_lt_query_context_test
        COMMAND glimmer_execution_service_test --lt-query-context)
    set_tests_properties(glimmer_execution_lt_query_context_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_preparation_audit_test
        COMMAND glimmer_execution_service_test --preparation-audit)
    set_tests_properties(glimmer_execution_preparation_audit_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_service_help COMMAND glimmer_execution_service --help)
    set_tests_properties(glimmer_execution_service_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")
    add_test(NAME glimmer_execution_service_modes_test COMMAND "${CMAKE_COMMAND}"
        "-DSERVICE=$<TARGET_FILE:glimmer_execution_service>"
        "-DNCCL_ENABLED=${GLIMMER_ENABLE_NCCL_EXECUTION}"
        -P "${PROJECT_SOURCE_DIR}/tests/control/check_execution_service_modes.cmake")
    set_tests_properties(glimmer_execution_service_modes_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_service_workload_aware_modes_test COMMAND "${CMAKE_COMMAND}"
        "-DSERVICE=$<TARGET_FILE:glimmer_execution_service>"
        "-DCUDNN_ROOT=${GLIMMER_CUDNN_ROOT}"
        -P "${PROJECT_SOURCE_DIR}/tests/control/check_execution_workload_aware_modes.cmake")
    set_tests_properties(glimmer_execution_service_workload_aware_modes_test PROPERTIES TIMEOUT 30)
    add_executable(glimmer_execution_public_shared_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/control/execution_public_shared_gpu_test.cc")
    target_include_directories(glimmer_execution_public_shared_gpu_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_public_shared_gpu_test PRIVATE
        glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
        Glimmer::control CUDA::cuda_driver CUDA::cudart CUDA::cublas Glimmer::project_options
        Threads::Threads)
    glimmer_enable_clang_tidy(glimmer_execution_public_shared_gpu_test)
    add_test(NAME glimmer_execution_public_shared_help
        COMMAND glimmer_execution_public_shared_gpu_test --help)
    add_test(NAME glimmer_execution_public_shared_phase_test
        COMMAND glimmer_execution_public_shared_gpu_test --phase-protocol)
    add_test(NAME glimmer_execution_public_shared_priority_path_test
        COMMAND glimmer_execution_public_shared_gpu_test --priority-path-protocol)
    set_tests_properties(glimmer_execution_public_shared_priority_path_test PROPERTIES
        TIMEOUT 15)
    set_tests_properties(glimmer_execution_public_shared_help
        glimmer_execution_public_shared_phase_test PROPERTIES TIMEOUT 15)
    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_executable(glimmer_execution_gemm_priority_gpu_client
            "${PROJECT_SOURCE_DIR}/tests/control/execution_gemm_priority_gpu_client.cc")
        target_link_libraries(glimmer_execution_gemm_priority_gpu_client PRIVATE
            CUDA::cudart CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_gemm_priority_gpu_client)
        glimmer_add_test_program(glimmer_execution_pair_profile_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/control/execution_pair_profile_gpu_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/execution_service"
                "${PROJECT_SOURCE_DIR}/src/control" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control CUDA::cuda_driver Glimmer::project_options Threads::Threads)
        set_tests_properties(glimmer_execution_pair_profile_gpu_test PROPERTIES
            LABELS "cuda-gpu" TIMEOUT 120 RUN_SERIAL TRUE)
        add_executable(glimmer_execution_host_gpu_client
            "${PROJECT_SOURCE_DIR}/tests/control/execution_host_gpu_client.cc")
        target_link_libraries(glimmer_execution_host_gpu_client PRIVATE
            Glimmer::control Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_host_gpu_client)
        # The client deliberately has no CUDA SDK headers or Driver linkage.
        add_executable(glimmer_execution_service_gpu_client
            "${PROJECT_SOURCE_DIR}/tests/control/execution_service_gpu_client.cc")
        target_link_libraries(glimmer_execution_service_gpu_client PRIVATE
            Glimmer::control Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_service_gpu_client)
    endif()
endif()
