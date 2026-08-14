add_library(glimmer_cuda_backend cuda_task_backend.cc)
add_library(Glimmer::cuda_backend ALIAS glimmer_cuda_backend)

set_target_properties(glimmer_cuda_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glimmer_cuda_backend
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_backend
    PRIVATE
        Glimmer::backend
        Glimmer::project_options
)

glimmer_enable_clang_tidy(glimmer_cuda_backend)

if(BUILD_TESTING)
    add_executable(glimmer_backend_cuda_task_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_task_backend_test.cc"
    )

    target_include_directories(glimmer_backend_cuda_task_test
        PRIVATE
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${CUDAToolkit_INCLUDE_DIRS}"
    )

    target_link_libraries(glimmer_backend_cuda_task_test
        PRIVATE
            Glimmer::cuda_backend
            Glimmer::project_options
    )

    glimmer_enable_clang_tidy(glimmer_backend_cuda_task_test)

    add_test(NAME glimmer_backend_cuda_task_test
        COMMAND glimmer_backend_cuda_task_test
    )
endif()
