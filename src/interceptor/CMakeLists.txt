add_library(glimmer_cuda_interceptor SHARED
    internal/allocation_registry.cc
    internal/allocation_registry.h
    internal/diagnostics.cc
    internal/diagnostics.h
    driver_dispatch.cc
    internal/driver_dispatch.h
    driver_api_interceptor.cc
    runtime_api_interceptor.cc
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
            "${CMAKE_DL_LIBS}"
    )
    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_fake_driver_test)
    add_dependencies(glimmer_cuda_interceptor_fake_driver_test
        glimmer_fake_cuda_driver
        glimmer_fake_cuda_runtime
        glimmer_cuda_interceptor
    )
    add_test(NAME glimmer_cuda_interceptor_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
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

    add_executable(glimmer_cuda_interceptor_runtime_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_runtime_gpu_test.cc"
    )

    target_link_libraries(glimmer_cuda_interceptor_runtime_gpu_test
        PRIVATE
            Glimmer::project_options
            CUDA::cudart
    )

    glimmer_enable_clang_tidy(glimmer_cuda_interceptor_runtime_gpu_test)

    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_test(NAME glimmer_cuda_interceptor_gpu_test
            COMMAND glimmer_cuda_interceptor_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
        )
    endif()
endif()
