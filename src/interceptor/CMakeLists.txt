add_library(glimmer_cuda_interceptor SHARED driver_api_interceptor.cc)

set_target_properties(glimmer_cuda_interceptor PROPERTIES
    OUTPUT_NAME glimmer_cuda_interceptor
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
)

target_include_directories(glimmer_cuda_interceptor
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_interceptor
    PRIVATE
        Glimmer::control
        Glimmer::project_options
        "${CMAKE_DL_LIBS}"
)

glimmer_enable_clang_tidy(glimmer_cuda_interceptor)

if(BUILD_TESTING)
    add_executable(glimmer_cuda_interceptor_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_api_interceptor_gpu_test.cc"
    )

    target_link_libraries(glimmer_cuda_interceptor_gpu_test
        PRIVATE
            Glimmer::project_options
            CUDA::cuda_driver
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_gpu_test)

    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_test(NAME glimmer_cuda_interceptor_gpu_test
            COMMAND glimmer_cuda_interceptor_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
        )
    endif()
endif()
