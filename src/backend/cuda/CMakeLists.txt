add_library(glimmer_cuda_backend
    cuda_execution_dispatcher.cc
    cuda_execution_domain.cc
    cuda_execution_session.cc
    cuda_execution_session_blas.cc
    cuda_execution_session_dnn.cc
    cuda_execution_graph.cc
    cuda_external_execution_bridge.cc
    cuda_native_execution_lanes.cc
    cuda_native_output_cell.cc
    cuda_process_memory_owner.cc
    cuda_process_memory_backend.cc
    cuda_module_jit.cc
    cuda_nccl_primary_domain.cc
    cuda_ptx_cache.cc
    cuda_profile_cache.cc
    cuda_pair_profile_cache.cc
    cuda_lt_registry.cc
    cuda_sgemm_image.cc
    cuda_task_backend.cc
)
add_library(Glimmer::cuda_backend ALIAS glimmer_cuda_backend)

set_target_properties(glimmer_cuda_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glimmer_cuda_backend
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_backend
    PRIVATE
        Glimmer::backend
        Glimmer::transform
        Glimmer::project_options
        spdlog::spdlog
        Threads::Threads
)

glimmer_enable_clang_tidy(glimmer_cuda_backend)

if(TARGET Glimmer::nccl_headers)
    target_sources(glimmer_cuda_backend PRIVATE cuda_nccl_native_call.cc cuda_nccl_coordinator.cc)
    target_link_libraries(glimmer_cuda_backend PRIVATE Glimmer::nccl_headers)
endif()

if(TARGET Glimmer::cudnn_headers)
    target_sources(glimmer_cuda_backend PRIVATE cuda_dnn_registry.cc cuda_dnn_execution.cc
        cuda_dnn_backend_execution.cc cuda_dnn_legacy_execution.cc)
    target_link_libraries(glimmer_cuda_backend PRIVATE Glimmer::cudnn_headers)
endif()

if(GLIMMER_BUILD_CUDA_PERSISTENT_GEMM)
    include("${PROJECT_SOURCE_DIR}/cmake/CutlassDependency.cmake")
    glimmer_add_cutlass_headers("${PROJECT_SOURCE_DIR}/3rdparty/cutlass")
    if(NOT CUDAToolkit_NVCC_EXECUTABLE)
        message(FATAL_ERROR "The private CUTLASS provider requires installed nvcc")
    endif()
    foreach(gemm_tile_side IN ITEMS 64 32)
        if(gemm_tile_side EQUAL 64)
            set(gemm_stem "cuda_sgemm")
        else()
            set(gemm_stem "cuda_sgemm_tile32")
        endif()
        foreach(gemm_signal_layout IN ITEMS 0 1)
            if(gemm_signal_layout AND NOT gemm_tile_side EQUAL 32)
                continue()
            endif()
            if(gemm_signal_layout)
                set(gemm_image_stem "${gemm_stem}_sharded")
            else()
                set(gemm_image_stem "${gemm_stem}")
            endif()
            add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}_ptx.inc"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17
                    --gpu-architecture=compute_70 --expt-relaxed-constexpr
                    "-DGLIMMER_GEMM_TILE_SIDE=${gemm_tile_side}"
                    "-DGLIMMER_GEMM_SHARDED_SIGNAL=${gemm_signal_layout}"
                    -I "${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include"
                    -o "${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}.ptx"
                    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_persistent_sgemm.cu"
                COMMAND "${CMAKE_COMMAND}"
                    "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}.ptx"
                    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}_ptx.inc"
                    -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
                DEPENDS cuda_persistent_sgemm.cu internal/cuda_gemm_block_completion.cuh
                    internal/cuda_gemm_geometry.h internal/cuda_gemm_signal_layout.h
                    "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
                    glimmer_verify_cutlass
                BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}.ptx"
                VERBATIM)
            target_sources(glimmer_cuda_backend PRIVATE
                "${CMAKE_CURRENT_BINARY_DIR}/${gemm_image_stem}_ptx.inc")
        endforeach()
    endforeach()
    target_include_directories(glimmer_cuda_backend PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(glimmer_cuda_backend PRIVATE GLIMMER_HAS_CUTLASS_SGEMM=1)
endif()

if(BUILD_TESTING)
    glimmer_add_test_program(glimmer_backend_cuda_execution_graph_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_graph_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::core Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_process_memory_backend_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_process_memory_backend_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::core Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_process_memory_session_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_process_memory_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::core Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_stream_scheduling_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_stream_scheduling_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_module_jit_session_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_module_jit_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_module_jit_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_module_jit_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::control Glimmer::project_options
        LINK_OPTIONS "LINKER:--wrap=_Znwm"
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_module_factory_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_module_factory_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_execution_domain_configuration_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_domain_configuration_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_execution_domain_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_domain_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_execution_domain_session_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_domain_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_execution_async_admission_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_async_admission_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        LINK_OPTIONS "LINKER:--wrap=getpid"
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_block_completion_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_block_completion_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_gemm_block_completion_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_block_completion_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_native_output_cell_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_native_output_cell_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    add_executable(glimmer_backend_cuda_source_enqueue_session_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_source_enqueue_session_test.cc")
    target_include_directories(glimmer_backend_cuda_source_enqueue_session_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_backend_cuda_source_enqueue_session_test PRIVATE
        Glimmer::cuda_backend Glimmer::backend Glimmer::project_options Threads::Threads)
    if(TARGET Glimmer::cudnn_headers)
        target_link_libraries(glimmer_backend_cuda_source_enqueue_session_test
            PRIVATE Glimmer::cudnn_headers)
    endif()
    glimmer_enable_clang_tidy(glimmer_backend_cuda_source_enqueue_session_test)
    add_test(NAME glimmer_backend_cuda_source_enqueue_session_test
        COMMAND glimmer_backend_cuda_source_enqueue_session_test)
    set_tests_properties(glimmer_backend_cuda_source_enqueue_session_test PROPERTIES TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_nccl_primary_domain_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_nccl_primary_domain_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 45)
    if(TARGET Glimmer::nccl_headers)
        glimmer_add_test_program(glimmer_backend_cuda_nccl_context_coordinator_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_nccl_context_coordinator_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
                Glimmer::nccl_headers Threads::Threads
            TIMEOUT 45)
        glimmer_add_test_program(glimmer_backend_cuda_nccl_coordinator_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_nccl_coordinator_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
                Glimmer::nccl_headers Threads::Threads
            TIMEOUT 45)
        glimmer_add_test_program(glimmer_backend_cuda_nccl_native_call_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_nccl_native_call_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
                Glimmer::nccl_headers Threads::Threads
            TIMEOUT 45)
    endif()
    glimmer_add_test_program(glimmer_backend_cuda_native_execution_lanes_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_native_execution_lanes_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    add_executable(glimmer_backend_cuda_external_execution_bridge_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_external_execution_bridge_test.cc")
    target_include_directories(glimmer_backend_cuda_external_execution_bridge_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_backend_cuda_external_execution_bridge_test PRIVATE
        Glimmer::cuda_backend Glimmer::backend Glimmer::project_options)
    if(TARGET Glimmer::cudnn_headers)
        target_link_libraries(glimmer_backend_cuda_external_execution_bridge_test PRIVATE
            Glimmer::cudnn_headers)
    endif()
    glimmer_enable_clang_tidy(glimmer_backend_cuda_external_execution_bridge_test)
    add_test(NAME glimmer_backend_cuda_external_execution_bridge_test
        COMMAND glimmer_backend_cuda_external_execution_bridge_test)
    set_tests_properties(glimmer_backend_cuda_external_execution_bridge_test PROPERTIES TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_external_barrier_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_external_barrier_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 45)
    add_executable(glimmer_backend_cuda_external_lease_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_external_lease_session_test.cc")
    target_include_directories(glimmer_backend_cuda_external_lease_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_backend_cuda_external_lease_test PRIVATE
        Glimmer::cuda_backend Glimmer::backend Glimmer::project_options)
    if(TARGET Glimmer::cudnn_headers)
        target_link_libraries(glimmer_backend_cuda_external_lease_test PRIVATE Glimmer::cudnn_headers)
    endif()
    glimmer_enable_clang_tidy(glimmer_backend_cuda_external_lease_test)
    add_test(NAME glimmer_backend_cuda_external_lease_test
        COMMAND glimmer_backend_cuda_external_lease_test)
    set_tests_properties(glimmer_backend_cuda_external_lease_test PROPERTIES TIMEOUT 45)
    add_executable(glimmer_backend_cuda_pointer_metadata_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_pointer_metadata_session_test.cc")
    target_include_directories(glimmer_backend_cuda_pointer_metadata_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_backend_cuda_pointer_metadata_test PRIVATE
        Glimmer::cuda_backend Glimmer::backend Glimmer::project_options)
    if(TARGET Glimmer::cudnn_headers)
        target_link_libraries(glimmer_backend_cuda_pointer_metadata_test PRIVATE Glimmer::cudnn_headers)
    endif()
    glimmer_enable_clang_tidy(glimmer_backend_cuda_pointer_metadata_test)
    add_test(NAME glimmer_backend_cuda_pointer_metadata_test
        COMMAND glimmer_backend_cuda_pointer_metadata_test)
    set_tests_properties(glimmer_backend_cuda_pointer_metadata_test PROPERTIES TIMEOUT 45)
    if(TARGET Glimmer::cudnn_headers)
        foreach(_dnn_test sdk_profile legacy registry session convolution_registry convolution_session find_session
                pooling_registry pooling_session lrn_registry lrn_session
                reduction_registry reduction_session batch_norm_session
                dropout_registry dropout_session op_tensor_registry reorder_session
                library_info_registry backward_convolution_session folded_convolution_session
                sequence_registry rnn_attention_registry rnn_query_session rnn_compute_session
                rnn_weight_params_session attn_weights_session attn_compute_session
                backend_graph_registry backend_execution backend_planning backend_heuristics
                backend_engine_properties)
            glimmer_add_test_program(glimmer_backend_cuda_dnn_${_dnn_test}_test
                SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_dnn_${_dnn_test}_test.cc"
                INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda"
                    "${PROJECT_SOURCE_DIR}/src/backend" "${CUDAToolkit_INCLUDE_DIRS}"
                LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::cudnn_headers
                    Glimmer::project_options
                TIMEOUT 30)
        endforeach()
        add_test(NAME glimmer_backend_cuda_dnn_registry_quarantine_test
            COMMAND glimmer_backend_cuda_dnn_registry_test --quarantine)
        set_tests_properties(glimmer_backend_cuda_dnn_registry_quarantine_test PROPERTIES TIMEOUT 30)
        glimmer_add_test_program(glimmer_backend_cuda_lt_reentry_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_lt_reentry_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::cudnn_headers
                Glimmer::project_options
            TIMEOUT 30)
    endif()
    glimmer_add_test_program(glimmer_backend_cuda_profile_cache_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_profile_cache_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    add_test(NAME glimmer_backend_cuda_workload_agnostic_policy_test
        COMMAND glimmer_backend_cuda_profile_cache_test --sharing)
    set_tests_properties(glimmer_backend_cuda_workload_agnostic_policy_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_profile_original_measurement_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_profile_original_measurement_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_pair_profile_cache_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_pair_profile_cache_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads
        TIMEOUT 45)
    glimmer_add_test_program(glimmer_backend_cuda_ptx_cache_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_ptx_cache_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        LIBRARIES Glimmer::cuda_backend Glimmer::transform Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_lt_registry_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_lt_registry_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::project_options
        TIMEOUT 30)

    glimmer_add_test_program(glimmer_backend_cuda_execution_session_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend
        TIMEOUT 30)
    add_test(NAME glimmer_backend_cuda_workload_agnostic_session_test
        COMMAND glimmer_backend_cuda_execution_session_test --workload-agnostic)
    set_tests_properties(glimmer_backend_cuda_workload_agnostic_session_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_geam_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_geam_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_blas_extension_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_blas_extension_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_unified_copy_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_unified_copy_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 30)
    add_test(NAME glimmer_backend_cuda_large_image_policy_test
        COMMAND glimmer_backend_cuda_execution_session_test --large-image-policy)
    set_tests_properties(glimmer_backend_cuda_large_image_policy_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_backend_cuda_preparation_observer_test
        COMMAND glimmer_backend_cuda_execution_session_test --preparation-observer)
    set_tests_properties(glimmer_backend_cuda_preparation_observer_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_function_control_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_function_control_session_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend Glimmer::project_options
        TIMEOUT 30)
    add_test(NAME glimmer_backend_cuda_runtime_function_attributes_test
        COMMAND glimmer_backend_cuda_function_control_test --runtime-attributes)
    set_tests_properties(glimmer_backend_cuda_runtime_function_attributes_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_backend_cuda_execution_dispatcher_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_dispatcher_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::backend
        LINK_OPTIONS "LINKER:--wrap=getpid"
        TIMEOUT 30)

    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        if(GLIMMER_BUILD_CUDA_PERSISTENT_GEMM)
            set(completion_fixture "${CMAKE_CURRENT_BINARY_DIR}/gemm_completion_election.ptx")
            add_custom_command(OUTPUT "${completion_fixture}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17
                    --gpu-architecture=compute_70 -I "${CMAKE_CURRENT_SOURCE_DIR}"
                    -o "${completion_fixture}"
                    "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_completion_election_fixture.cu"
                DEPENDS internal/cuda_gemm_block_completion.cuh
                    "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_completion_election_fixture.cu"
                    "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_completion_election_fixture.h"
                VERBATIM)
            add_custom_target(glimmer_gemm_completion_election_image DEPENDS "${completion_fixture}")
            add_executable(glimmer_backend_cuda_gemm_completion_election_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_completion_election_gpu_test.cc")
            add_dependencies(glimmer_backend_cuda_gemm_completion_election_gpu_test
                glimmer_gemm_completion_election_image)
            target_link_libraries(glimmer_backend_cuda_gemm_completion_election_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_backend_cuda_gemm_completion_election_gpu_test)
            add_test(NAME glimmer_backend_cuda_gemm_completion_election_help
                COMMAND glimmer_backend_cuda_gemm_completion_election_gpu_test --help)
            add_test(NAME glimmer_backend_cuda_gemm_completion_election_gpu_test
                COMMAND glimmer_backend_cuda_gemm_completion_election_gpu_test "${completion_fixture}")
            set_tests_properties(glimmer_backend_cuda_gemm_completion_election_gpu_test PROPERTIES
                TIMEOUT 30 LABELS "cuda-gpu")
            add_executable(glimmer_backend_cuda_gemm_block_completion_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/backend/cuda_gemm_block_completion_gpu_test.cc")
            target_include_directories(glimmer_backend_cuda_gemm_block_completion_gpu_test PRIVATE
                "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend")
            target_link_libraries(glimmer_backend_cuda_gemm_block_completion_gpu_test PRIVATE
                Glimmer::cuda_backend Glimmer::backend CUDA::cuda_driver CUDA::cublas CUDA::cublasLt)
            glimmer_enable_clang_tidy(glimmer_backend_cuda_gemm_block_completion_gpu_test)
            add_test(NAME glimmer_backend_cuda_gemm_block_completion_gpu_test
                COMMAND glimmer_backend_cuda_gemm_block_completion_gpu_test --observed)
            add_test(NAME glimmer_backend_cuda_gemm_sharded_gpu_test
                COMMAND glimmer_backend_cuda_gemm_block_completion_gpu_test
                    --observed --gemm-tile-side 32 --sharded-retreat)
            add_test(NAME glimmer_backend_cuda_gemm_sharded_shared_gpu_test
                COMMAND glimmer_backend_cuda_gemm_block_completion_gpu_test --shared-sharded)
            set_tests_properties(glimmer_backend_cuda_gemm_block_completion_gpu_test
                glimmer_backend_cuda_gemm_sharded_gpu_test
                glimmer_backend_cuda_gemm_sharded_shared_gpu_test PROPERTIES TIMEOUT 150)
        endif()
        glimmer_add_test_program(glimmer_backend_cuda_execution_blas_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_blas_gpu_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend CUDA::cuda_driver CUDA::cublas)
        set_tests_properties(glimmer_backend_cuda_execution_blas_gpu_test PROPERTIES
            TIMEOUT 90 SKIP_RETURN_CODE 77)
        add_executable(glimmer_backend_cuda_execution_persistent_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_persistent_gpu_test.cc"
        )
        # Compile both real intake forms now; execution remains a guarded GPU gate.
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/module_state_ptx.inc"
                "${CMAKE_CURRENT_BINARY_DIR}/module_state_fatbin.inc"
                "${CMAKE_CURRENT_BINARY_DIR}/module_state_companion_fatbin.inc"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_module_state.ptx"
                "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/module_state_ptx.inc"
                -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --fatbin
                --generate-code=arch=compute_70,code=compute_70
                -o "${CMAKE_CURRENT_BINARY_DIR}/module_state.fatbin"
                "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_module_state.ptx"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/module_state.fatbin"
                "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/module_state_fatbin.inc"
                -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --fatbin
                --generate-code=arch=compute_70,code=compute_70
                --generate-code=arch=compute_70,code=sm_70
                -o "${CMAKE_CURRENT_BINARY_DIR}/module_state_companion.fatbin"
                "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_module_state.ptx"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/module_state_companion.fatbin"
                "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/module_state_companion_fatbin.inc"
                -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
            DEPENDS "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_module_state.ptx"
                "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaPtx.cmake"
            BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/module_state.fatbin"
                "${CMAKE_CURRENT_BINARY_DIR}/module_state_companion.fatbin"
            VERBATIM)
        target_sources(glimmer_backend_cuda_execution_persistent_gpu_test PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/module_state_ptx.inc"
            "${CMAKE_CURRENT_BINARY_DIR}/module_state_fatbin.inc"
            "${CMAKE_CURRENT_BINARY_DIR}/module_state_companion_fatbin.inc")
        target_include_directories(glimmer_backend_cuda_execution_persistent_gpu_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend" "${CMAKE_CURRENT_BINARY_DIR}"
        )
        target_link_libraries(glimmer_backend_cuda_execution_persistent_gpu_test PRIVATE
            Glimmer::cuda_backend Glimmer::backend Glimmer::transform CUDA::cuda_driver
        )
        glimmer_enable_clang_tidy(glimmer_backend_cuda_execution_persistent_gpu_test)
        add_test(NAME glimmer_backend_cuda_execution_persistent_gpu_test
            COMMAND glimmer_backend_cuda_execution_persistent_gpu_test)
        set_tests_properties(glimmer_backend_cuda_execution_persistent_gpu_test PROPERTIES
            TIMEOUT 60 SKIP_RETURN_CODE 77)
        add_test(NAME glimmer_backend_cuda_execution_module_state_input_test
            COMMAND glimmer_backend_cuda_execution_persistent_gpu_test --inspect-state)
        set_tests_properties(glimmer_backend_cuda_execution_module_state_input_test PROPERTIES
            TIMEOUT 10)
        add_test(NAME glimmer_backend_cuda_execution_compiler_input_test
            COMMAND glimmer_backend_cuda_execution_persistent_gpu_test --inspect-compiler)
        set_tests_properties(glimmer_backend_cuda_execution_compiler_input_test PROPERTIES
            TIMEOUT 10)
        add_test(NAME glimmer_backend_cuda_execution_aggregate_input_test
            COMMAND glimmer_backend_cuda_execution_persistent_gpu_test --inspect-aggregate)
        set_tests_properties(glimmer_backend_cuda_execution_aggregate_input_test PROPERTIES
            TIMEOUT 10)
        add_test(NAME glimmer_backend_cuda_execution_shared_domain_input_test
            COMMAND glimmer_backend_cuda_execution_persistent_gpu_test --inspect-shared-domain)
        set_tests_properties(glimmer_backend_cuda_execution_shared_domain_input_test PROPERTIES
            TIMEOUT 10)
        glimmer_add_test_program(glimmer_backend_cuda_execution_session_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_session_gpu_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend CUDA::cuda_driver)
        set_tests_properties(glimmer_backend_cuda_execution_session_gpu_test PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
        glimmer_add_test_program(glimmer_backend_cuda_execution_dispatcher_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_dispatcher_gpu_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
            LIBRARIES Glimmer::cuda_backend Glimmer::backend CUDA::cuda_driver)
        set_tests_properties(glimmer_backend_cuda_execution_dispatcher_gpu_test PROPERTIES
            TIMEOUT 60 SKIP_RETURN_CODE 77
        )
    endif()

    glimmer_add_test_program(glimmer_backend_cuda_task_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/backend/cuda_task_backend_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::cuda_backend Glimmer::project_options)
endif()
