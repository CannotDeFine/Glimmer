enable_language(CUDA)

add_executable(glimmer_cuda_workload
    cuda_workload.cu
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
