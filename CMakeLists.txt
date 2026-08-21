cmake_minimum_required(VERSION 3.20)

project(Glimmer VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Glimmer supports Linux only.")
endif()

include(cmake/ProjectOptions.cmake)
include(cmake/ProjectTesting.cmake)

option(GLIMMER_BUILD_CUDA_INTERCEPTOR "Build the optional CUDA Driver API preload library" OFF)
option(GLIMMER_BUILD_CUDA_BACKEND "Build the optional explicit CUDA task backend" OFF)
option(GLIMMER_ENABLE_CUDA_GPU_TESTS "Enable tests that require a CUDA-capable GPU" OFF)
option(GLIMMER_BUILD_CUDA_WORKLOAD "Build the real-GPU CUDA workload harnesses" OFF)

find_package(Threads REQUIRED)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    add_custom_target(glimmer_compile_commands ALL
        COMMAND ${CMAKE_COMMAND} -E rm -f "${PROJECT_SOURCE_DIR}/compile_commands.json"
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${PROJECT_SOURCE_DIR}/compile_commands.json"
        COMMENT "Updating root compile_commands.json link"
        VERBATIM
    )
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)

if(GLIMMER_BUILD_CUDA_INTERCEPTOR OR GLIMMER_BUILD_CUDA_BACKEND OR GLIMMER_BUILD_CUDA_WORKLOAD)
    find_package(CUDAToolkit REQUIRED)
endif()

if(GLIMMER_BUILD_CUDA_INTERCEPTOR)
    add_subdirectory(src/interceptor)
endif()

if(GLIMMER_BUILD_CUDA_BACKEND)
    add_subdirectory(src/backend/cuda)
endif()

if(GLIMMER_BUILD_CUDA_WORKLOAD)
    add_subdirectory(examples/cuda_workloads)
    add_subdirectory(examples/priority_demo)
    if(GLIMMER_BUILD_CUDA_BACKEND)
        add_subdirectory(examples/cuda_task_backend)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

find_program(GLIMMER_CLANG_FORMAT NAMES clang-format)

if(GLIMMER_CLANG_FORMAT)
    file(GLOB_RECURSE GLIMMER_FORMAT_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/examples/*.h"
        "${PROJECT_SOURCE_DIR}/examples/*.hpp"
        "${PROJECT_SOURCE_DIR}/examples/*.cc"
        "${PROJECT_SOURCE_DIR}/examples/*.cpp"
        "${PROJECT_SOURCE_DIR}/examples/*.cu"
        "${PROJECT_SOURCE_DIR}/tests/*.cc"
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cu"
    )

    # Examples may contain ignored framework environments and generated output.
    # They are external artifacts, not Glimmer sources, and must not be linted.
    list(FILTER GLIMMER_FORMAT_SOURCES EXCLUDE REGEX "/(\\.venv|output|build)/")

    if(GLIMMER_FORMAT_SOURCES)
        add_custom_target(format
            COMMAND ${GLIMMER_CLANG_FORMAT} -i ${GLIMMER_FORMAT_SOURCES}
            COMMENT "Formatting Glimmer source files"
            VERBATIM
        )

        add_custom_target(format-check
            COMMAND ${GLIMMER_CLANG_FORMAT} --dry-run --Werror ${GLIMMER_FORMAT_SOURCES}
            COMMENT "Checking Glimmer source formatting"
            VERBATIM
        )
    endif()
else()
    message(STATUS "clang-format not found; format targets are unavailable")
endif()
