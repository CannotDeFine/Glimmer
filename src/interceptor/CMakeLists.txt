add_library(glimmer_cuda_interceptor SHARED
    internal/allocation_registry.cc
    internal/allocation_registry.h
    internal/diagnostics.cc
    internal/diagnostics.h
    driver_dispatch.cc
    internal/driver_dispatch.h
    nvml_dispatch.cc
    internal/nvml_dispatch.h
    driver_api_interceptor.cc
    internal/driver_api_interceptor.h
    driver_api_wrappers.cc
    nvml_api_wrappers.cc
    runtime_api_interceptor.cc
    symbol_interceptor.cc
    internal/runtime_api_bridge.h
    symbol_registry.cc
    internal/symbol_registry.h
)

set_target_properties(glimmer_cuda_interceptor PROPERTIES
    OUTPUT_NAME glimmer_cuda_interceptor
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
)

target_include_directories(glimmer_cuda_interceptor
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/interceptor"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_interceptor
    PRIVATE
        Glimmer::control
        Glimmer::project_options
        "${CMAKE_DL_LIBS}"
        Threads::Threads
)

glimmer_enable_clang_tidy(glimmer_cuda_interceptor)

if(BUILD_TESTING)
    add_executable(glimmer_cuda_interceptor_dlsym_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/dlsym_interceptor_test.cc"
    )

    target_link_libraries(glimmer_cuda_interceptor_dlsym_test
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_dlsym_test)

    add_test(NAME glimmer_cuda_interceptor_dlsym_test
        COMMAND glimmer_cuda_interceptor_dlsym_test
    )
    set_tests_properties(glimmer_cuda_interceptor_dlsym_test PROPERTIES
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
    )

    add_executable(glimmer_cuda_interceptor_allocation_registry_test
        internal/allocation_registry.cc
        internal/diagnostics.cc
        "${PROJECT_SOURCE_DIR}/tests/interceptor/allocation_registry_test.cc"
    )

    target_include_directories(glimmer_cuda_interceptor_allocation_registry_test
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src/interceptor"
            "${CUDAToolkit_INCLUDE_DIRS}"
    )

    target_link_libraries(glimmer_cuda_interceptor_allocation_registry_test
        PRIVATE
            Glimmer::control
            Glimmer::project_options
            Threads::Threads
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_allocation_registry_test)

    add_test(NAME glimmer_cuda_interceptor_allocation_registry_test
        COMMAND glimmer_cuda_interceptor_allocation_registry_test
    )

    add_executable(glimmer_cuda_interceptor_dispatch_test
        driver_dispatch.cc
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_dispatch_test.cc"
    )

    target_include_directories(glimmer_cuda_interceptor_dispatch_test
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src/interceptor"
            "${CUDAToolkit_INCLUDE_DIRS}"
    )

    target_link_libraries(glimmer_cuda_interceptor_dispatch_test
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_dispatch_test)

    add_test(NAME glimmer_cuda_interceptor_dispatch_test
        COMMAND glimmer_cuda_interceptor_dispatch_test
    )

    add_library(glimmer_fake_cuda_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_driver.cc"
    )
    set_target_properties(glimmer_fake_cuda_driver PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME cuda
        SUFFIX ".so.1"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda"
    )
    target_include_directories(glimmer_fake_cuda_driver
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_fake_cuda_driver PRIVATE Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_fake_cuda_driver)

    add_library(glimmer_fake_cuda_runtime SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_runtime.cc"
    )
    set_target_properties(glimmer_fake_cuda_runtime PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME cudart
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda"
    )
    target_include_directories(glimmer_fake_cuda_runtime
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_fake_cuda_runtime
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )
    glimmer_enable_clang_tidy(glimmer_fake_cuda_runtime)

    add_library(glimmer_fake_nvml SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_nvml.cc"
    )
    set_target_properties(glimmer_fake_nvml PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME nvidia-ml
        SUFFIX ".so.1"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda"
    )
    target_include_directories(glimmer_fake_nvml
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_fake_nvml PRIVATE Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_fake_nvml)

    add_executable(glimmer_cuda_interceptor_fake_driver_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/interceptor_fake_driver_test.cc"
    )
    target_include_directories(glimmer_cuda_interceptor_fake_driver_test
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_cuda_interceptor_fake_driver_test
        PRIVATE
            Glimmer::project_options
            Glimmer::control
            "${CMAKE_DL_LIBS}"
    )
    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_fake_driver_test)
    add_dependencies(glimmer_cuda_interceptor_fake_driver_test
        glimmer_fake_cuda_driver
        glimmer_fake_cuda_runtime
        glimmer_fake_nvml
        glimmer_cuda_interceptor
    )
    add_test(NAME glimmer_cuda_interceptor_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_physical_capacity_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_physical_capacity_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_FAKE_PHYSICAL_CAP=1;GLIMMER_MEMORY_LIMIT_BYTES=134217728;GLIMMER_TRACE_MEMORY_INFO=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "physical_total_bytes=67108864 physical_free_bytes=67108864"
    )

    add_test(NAME glimmer_cuda_interceptor_nvml_only_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_nvml_only_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_FAKE_NVML_ONLY=1;GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_observe_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_observe_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_SCHEDULER_MODE=observe;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "observed CUDA kernel launch #1 api=cuLaunchKernel grid="
        FAIL_REGULAR_EXPRESSION
            "observed CUDA kernel launch #2"
    )

    add_test(NAME glimmer_cuda_interceptor_trace_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_trace_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_SCHEDULER_MODE=observe;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "observed CUDA kernel launch #6 api=__cudaLaunchKernel_ptsz grid="
    )

    add_test(NAME glimmer_cuda_interceptor_trace_memory_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_trace_memory_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_TRACE_MEMORY_INFO=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "memory info api=cudaMemGetInfo device=0 total_bytes=4096 used_bytes=1024 free_bytes=3072"
    )

    add_executable(glimmer_cuda_interceptor_nvml_passthrough_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/nvml_passthrough_test.cc"
    )
    target_include_directories(glimmer_cuda_interceptor_nvml_passthrough_test
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_cuda_interceptor_nvml_passthrough_test
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )
    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_nvml_passthrough_test)
    add_dependencies(glimmer_cuda_interceptor_nvml_passthrough_test
        glimmer_fake_nvml
        glimmer_cuda_interceptor
    )
    add_test(NAME glimmer_cuda_interceptor_nvml_passthrough_test
        COMMAND glimmer_cuda_interceptor_nvml_passthrough_test
    )
    set_tests_properties(glimmer_cuda_interceptor_nvml_passthrough_test PROPERTIES
        ENVIRONMENT
            "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_nvml>"
    )

    add_executable(glimmer_cuda_interceptor_cuda_import_passthrough_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_import_passthrough_test.cc"
    )
    target_include_directories(glimmer_cuda_interceptor_cuda_import_passthrough_test
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_cuda_interceptor_cuda_import_passthrough_test
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )
    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_cuda_import_passthrough_test)
    add_dependencies(glimmer_cuda_interceptor_cuda_import_passthrough_test
        glimmer_fake_cuda_driver
        glimmer_cuda_interceptor
    )
    add_test(NAME glimmer_cuda_interceptor_cuda_import_passthrough_test
        COMMAND glimmer_cuda_interceptor_cuda_import_passthrough_test
    )
    set_tests_properties(glimmer_cuda_interceptor_cuda_import_passthrough_test PROPERTIES
        ENVIRONMENT
            "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_runtime_untracked_initialization_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_runtime_untracked_initialization_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_FAKE_RUNTIME_UNTRACKED_FIRST=1;GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_duplicate_record_failure_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_duplicate_record_failure_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_FAKE_DUPLICATE_POINTER=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_driver_async_null_pointer_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_driver_async_null_pointer_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_QUOTA_MODE=shared;GLIMMER_QUOTA_TENANT_ID=glimmer-cuda-interceptor-driver-null;GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_FAKE_NULL_SUCCESS_POINTER=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_runtime_async_null_pointer_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_runtime_async_null_pointer_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_QUOTA_MODE=shared;GLIMMER_QUOTA_TENANT_ID=glimmer-cuda-interceptor-runtime-null;GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_FAKE_NULL_SUCCESS_POINTER=1;GLIMMER_NULL_SUCCESS_RUNTIME=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_shared_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_shared_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_QUOTA_MODE=shared;GLIMMER_QUOTA_TENANT_ID=glimmer-cuda-interceptor-shared;GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_task_memory_limit_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_task_memory_limit_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_QUOTA_MODE=shared;GLIMMER_QUOTA_TENANT_ID=glimmer-cuda-interceptor-task-limit;GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_TASK_MEMORY_LIMIT_BYTES=1024;GLIMMER_FAKE_TASK_LIMIT=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_executable(glimmer_cuda_interceptor_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_api_interceptor_gpu_test.cc"
    )

    target_link_libraries(glimmer_cuda_interceptor_gpu_test
        PRIVATE
            Glimmer::project_options
            CUDA::cuda_driver
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_gpu_test)

    add_executable(glimmer_cuda_interceptor_kernel_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_kernel_gpu_test.cc"
    )

    target_link_libraries(glimmer_cuda_interceptor_kernel_gpu_test
        PRIVATE
            Glimmer::project_options
            CUDA::cuda_driver
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_kernel_gpu_test)

    add_executable(glimmer_cuda_interceptor_runtime_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_runtime_gpu_test.cc"
    )

    add_executable(glimmer_cuda_interceptor_nvml_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/nvml_interceptor_gpu_test.cc"
    )

    target_include_directories(glimmer_cuda_interceptor_nvml_gpu_test
        PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}"
    )

    target_link_libraries(glimmer_cuda_interceptor_nvml_gpu_test
        PRIVATE
            Glimmer::project_options
            "${CMAKE_DL_LIBS}"
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_nvml_gpu_test)

    target_link_libraries(glimmer_cuda_interceptor_runtime_gpu_test
        PRIVATE
            Glimmer::project_options
            CUDA::cudart
    )
    target_compile_definitions(glimmer_cuda_interceptor_runtime_gpu_test
        PRIVATE
            CUDA_API_PER_THREAD_DEFAULT_STREAM
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_runtime_gpu_test)

    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        enable_language(CUDA)

        add_executable(glimmer_cuda_interceptor_runtime_kernel_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_kernel_gpu_test.cu"
        )

        add_executable(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_shared_quota_gpu_test.cu"
        )

        target_compile_features(glimmer_cuda_interceptor_runtime_kernel_gpu_test
            PRIVATE
                cxx_std_20
        )
        target_compile_features(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            PRIVATE
                cxx_std_20
        )
        target_link_libraries(glimmer_cuda_interceptor_runtime_kernel_gpu_test
            PRIVATE
                CUDA::cudart
        )
        target_link_libraries(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            PRIVATE
                "$<LINK_ONLY:Glimmer::control>"
                CUDA::cudart
        )
        target_include_directories(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            PRIVATE
                "${PROJECT_SOURCE_DIR}/include"
        )
        target_compile_definitions(glimmer_cuda_interceptor_runtime_kernel_gpu_test
            PRIVATE
                CUDA_API_PER_THREAD_DEFAULT_STREAM
        )
        target_compile_definitions(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            PRIVATE
                CUDA_API_PER_THREAD_DEFAULT_STREAM
        )
        set_target_properties(glimmer_cuda_interceptor_runtime_kernel_gpu_test PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        set_target_properties(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
    endif()

    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_test(NAME glimmer_cuda_interceptor_gpu_test
            COMMAND glimmer_cuda_interceptor_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_kernel_gpu_test
            COMMAND glimmer_cuda_interceptor_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_kernel_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=observe;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_kernel_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_kernel_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=observe;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_shared_quota_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_shared_quota_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_QUOTA_MODE=shared;GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_nvml_gpu_test
            COMMAND glimmer_cuda_interceptor_nvml_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_nvml_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )
    endif()
endif()
