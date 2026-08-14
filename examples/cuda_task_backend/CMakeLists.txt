add_executable(glimmer_cuda_task_backend_demo cuda_task_backend_demo.cc)

target_include_directories(glimmer_cuda_task_backend_demo
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_task_backend_demo
    PRIVATE
        Glimmer::cuda_backend
        Glimmer::project_options
        CUDA::cuda_driver
)

glimmer_enable_clang_tidy(glimmer_cuda_task_backend_demo)

add_executable(glimmer_cuda_multi_tenant_demo cuda_multi_tenant_demo.cc)

target_include_directories(glimmer_cuda_multi_tenant_demo
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_multi_tenant_demo
    PRIVATE
        Glimmer::control
        Glimmer::cuda_backend
        Glimmer::project_options
        CUDA::cuda_driver
)

glimmer_enable_clang_tidy(glimmer_cuda_multi_tenant_demo)

add_executable(glimmer_cuda_lease_worker cuda_lease_worker.cc)

target_include_directories(glimmer_cuda_lease_worker
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_lease_worker
    PRIVATE
        Glimmer::control
        Glimmer::project_options
        CUDA::cuda_driver
)

glimmer_enable_clang_tidy(glimmer_cuda_lease_worker)

if(BUILD_TESTING AND GLIMMER_ENABLE_CUDA_GPU_TESTS)
    add_test(NAME glimmer_cuda_task_backend_gpu_test
        COMMAND glimmer_cuda_task_backend_demo
            --quota-bytes 8388608
            --task-memory-bytes 1048576
            --tasks 2
    )
    set_tests_properties(glimmer_cuda_task_backend_gpu_test PROPERTIES
        PASS_REGULAR_EXPRESSION
            "explicit_cuda_task_demo status=ok tasks=2 completed=2"
        TIMEOUT 60
    )

    add_test(NAME glimmer_cuda_multi_tenant_gpu_test
        COMMAND glimmer_cuda_multi_tenant_demo
    )
    set_tests_properties(glimmer_cuda_multi_tenant_gpu_test PROPERTIES
        PASS_REGULAR_EXPRESSION
            "explicit_cuda_multi_tenant_demo status=ok dispatch_order=tenant-a1,tenant-a2,tenant-b1,tenant-a3,tenant-b2"
        TIMEOUT 60
    )
endif()
