enable_language(CUDA)

add_executable(glimmer_cuda_workload
    baseline/cuda_workload.cu
)

target_link_libraries(glimmer_cuda_workload
    PRIVATE
        Glimmer::project_options
        CUDA::cudart
)

target_compile_features(glimmer_cuda_workload PRIVATE cxx_std_20)
set_target_properties(glimmer_cuda_workload PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
glimmer_enable_clang_tidy(glimmer_cuda_workload)

if(BUILD_TESTING AND GLIMMER_BUILD_CUDA_INTERCEPTOR AND GLIMMER_ENABLE_CUDA_GPU_TESTS)
    add_test(NAME glimmer_cuda_workload_gpu_test
        COMMAND glimmer_cuda_workload
            --bytes 1048576
            --iterations 10
    )
    set_tests_properties(glimmer_cuda_workload_gpu_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_TRACE_MEMORY_INFO=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
        PASS_REGULAR_EXPRESSION
            "cuda_memory_kernel,0,1048576,8388608,8388608,7340032,8388608,[0-9.]+,0,0,0,0,0,0,0,0,0"
        TIMEOUT 60
    )
endif()

function(glimmer_add_cuda_workload target_name source_file workload_name)
    add_executable(${target_name}
        "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/workload_common.h"
    )

    target_include_directories(${target_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${target_name}
        PRIVATE
            Glimmer::project_options
            CUDA::cudart
    )

    target_compile_features(${target_name} PRIVATE cxx_std_20)
    set_target_properties(${target_name} PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    glimmer_enable_clang_tidy(${target_name})

    if(BUILD_TESTING AND GLIMMER_BUILD_CUDA_INTERCEPTOR AND GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_test(NAME ${workload_name}_gpu_test COMMAND ${target_name})
        set_tests_properties(${workload_name}_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_TRACE_MEMORY_INFO=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            PASS_REGULAR_EXPRESSION
                "workload=${workload_name} status=ok.*validation=1"
            TIMEOUT 60
        )
    endif()
endfunction()

glimmer_add_cuda_workload(
    glimmer_cuda_async_workload
    async/cuda_async_workload.cu
    glimmer_cuda_async_workload
)

glimmer_add_cuda_workload(
    glimmer_cuda_managed_workload
    managed/cuda_managed_workload.cu
    glimmer_cuda_managed_workload
)

glimmer_add_cuda_workload(
    glimmer_cuda_pitched_workload
    pitched/cuda_pitched_workload.cu
    glimmer_cuda_pitched_workload
)

glimmer_add_cuda_workload(
    glimmer_cuda_multistream_workload
    multistream/cuda_multistream_workload.cu
    glimmer_cuda_multistream_workload
)
