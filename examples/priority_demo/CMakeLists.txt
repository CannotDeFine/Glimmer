enable_language(CUDA)

add_executable(glimmer_cuda_priority_workload
    priority_workload.cu
)

target_link_libraries(glimmer_cuda_priority_workload
    PRIVATE
        Glimmer::project_options
        CUDA::cudart
        CUDA::cublas
)

target_compile_features(glimmer_cuda_priority_workload PRIVATE cxx_std_20)
glimmer_enable_clang_tidy(glimmer_cuda_priority_workload)

set_target_properties(glimmer_cuda_priority_workload PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

add_custom_target(glimmer_cuda_priority_demo
    DEPENDS
        glimmer_cuda_priority_workload
        glimmer_control_service
    COMMENT "Building the transparent priority scheduling demo"
)

if(BUILD_TESTING AND GLIMMER_BUILD_CUDA_INTERCEPTOR AND GLIMMER_ENABLE_CUDA_GPU_TESTS)
    add_test(NAME glimmer_cuda_priority_workload_gpu_test
        COMMAND glimmer_cuda_priority_workload
            --role inference
            --kernel model
            --matrix-size 256
            --iterations 1
            --work-units 1
    )
    set_tests_properties(glimmer_cuda_priority_workload_gpu_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_SCHEDULER_MODE=observe;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
        PASS_REGULAR_EXPRESSION
            "priority_workload role=inference kernel=model status=ok iterations=1"
        TIMEOUT 60
    )
endif()
