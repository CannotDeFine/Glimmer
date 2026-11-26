cmake_minimum_required(VERSION 3.20)

include(cmake/BuildStorage.cmake)
glimmer_check_build_storage()

project(Glimmer VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Glimmer supports Linux only.")
endif()

include(cmake/ProjectOptions.cmake)
include(cmake/ProjectTesting.cmake)
if(BUILD_TESTING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
endif()

option(GLIMMER_BUILD_CUDA_INTERCEPTOR "Build the optional CUDA Driver API preload library" OFF)
option(GLIMMER_BUILD_CUDA_BACKEND "Build the optional explicit CUDA task backend" OFF)
option(GLIMMER_BUILD_CUDA_EXECUTION_SERVICE "Build the optional authenticated native execution service" OFF)
option(GLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR "Build the experimental centralized Driver frontend" OFF)
option(GLIMMER_ENABLE_NCCL_EXECUTION "Enable optional NCCL execution service and frontend routes" OFF)
set(GLIMMER_CUDNN_ROOT "" CACHE PATH "Optional explicit installed cuDNN SDK root")
set(GLIMMER_NCCL_ROOT "" CACHE PATH "Optional explicit installed NCCL SDK root")
if(GLIMMER_ENABLE_NCCL_EXECUTION AND
        (NOT GLIMMER_NCCL_ROOT OR NOT GLIMMER_BUILD_CUDA_EXECUTION_SERVICE OR
         NOT GLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR))
    message(FATAL_ERROR "NCCL execution requires GLIMMER_NCCL_ROOT and both CUDA execution targets")
endif()
if(GLIMMER_NCCL_ROOT AND NOT GLIMMER_BUILD_CUDA_BACKEND)
    message(FATAL_ERROR "The native NCCL packets require GLIMMER_BUILD_CUDA_BACKEND=ON")
endif()
if(GLIMMER_CUDNN_ROOT AND NOT GLIMMER_BUILD_CUDA_BACKEND)
    message(FATAL_ERROR "The owned cuDNN adapter requires GLIMMER_BUILD_CUDA_BACKEND=ON")
endif()
if(GLIMMER_BUILD_CUDA_EXECUTION_SERVICE AND NOT GLIMMER_BUILD_CUDA_BACKEND)
    message(FATAL_ERROR "The CUDA execution service requires GLIMMER_BUILD_CUDA_BACKEND=ON")
endif()
option(GLIMMER_ENABLE_CUDA_GPU_TESTS "Enable tests that require a CUDA-capable GPU" OFF)
option(GLIMMER_BUILD_CUDA_WORKLOAD "Build the real-GPU CUDA workload harnesses" OFF)
option(GLIMMER_BUILD_GEMM_COMPARISON "Build the optional CUTLASS GEMM comparison" OFF)
option(GLIMMER_BUILD_CUDA_PERSISTENT_GEMM "Build the private Sm70 CUTLASS execution provider" OFF)
if(GLIMMER_BUILD_CUDA_PERSISTENT_GEMM AND NOT GLIMMER_BUILD_CUDA_BACKEND)
    message(FATAL_ERROR "The persistent GEMM provider requires the CUDA backend")
endif()

find_package(Threads REQUIRED)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    add_custom_target(glimmer_compile_commands ALL
        COMMAND ${CMAKE_COMMAND}
            "-DGLIMMER_COMPILE_DATABASE=${CMAKE_BINARY_DIR}/compile_commands.json"
            "-DGLIMMER_COMPILE_LINK=${PROJECT_SOURCE_DIR}/compile_commands.json"
            -P "${PROJECT_SOURCE_DIR}/cmake/UpdateCompileCommands.cmake"
        COMMENT "Updating root compile_commands.json link"
        VERBATIM
    )
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/spdlog/CMakeLists.txt")
    message(FATAL_ERROR
        "spdlog is missing. Run: git submodule update --init --recursive -- 3rdparty/spdlog")
endif()
add_subdirectory(3rdparty)
if(GLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR)
    # Existing approved static logger is linked into the optional frontend DSO.
    # Configure the consumer requirement here; never patch third-party source.
    set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_subdirectory(src)

if(GLIMMER_BUILD_CUDA_INTERCEPTOR OR GLIMMER_BUILD_CUDA_BACKEND OR GLIMMER_BUILD_CUDA_WORKLOAD OR
        GLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR OR GLIMMER_BUILD_GEMM_COMPARISON)
    find_package(CUDAToolkit REQUIRED)
endif()

if(GLIMMER_BUILD_CUDA_WORKLOAD OR GLIMMER_BUILD_GEMM_COMPARISON)
    # Match automatic CUDA language selection to the toolkit just discovered.
    # An explicit compiler or CUDACXX remains the caller's choice.
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "" AND CUDAToolkit_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler")
    endif()
    enable_language(CUDA)
    include(cmake/CudaToolchain.cmake)
    glimmer_validate_cuda_toolchain()
endif()

if(GLIMMER_BUILD_CUDA_INTERCEPTOR)
    add_subdirectory(src/interceptor)
endif()

if(GLIMMER_BUILD_CUDA_BACKEND)
    if(GLIMMER_NCCL_ROOT)
        include(cmake/NcclDependency.cmake)
        glimmer_add_nccl_sdk("${GLIMMER_NCCL_ROOT}")
    endif()
    if(GLIMMER_CUDNN_ROOT)
        include(cmake/CudnnDependency.cmake)
        glimmer_add_cudnn_sdk("${GLIMMER_CUDNN_ROOT}")
    endif()
    add_subdirectory(src/backend/cuda)
endif()
if(GLIMMER_BUILD_CUDA_EXECUTION_SERVICE)
    add_subdirectory(src/execution_service)
endif()
if(GLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR)
    add_subdirectory(src/interceptor/execution)
endif()

include(cmake/ExecutionHostOptimization.cmake)
glimmer_configure_execution_host_optimization()

if(GLIMMER_BUILD_CUDA_WORKLOAD)
    add_subdirectory(examples/cuda_workloads)
    add_subdirectory(examples/priority_demo)
    if(GLIMMER_BUILD_CUDA_BACKEND)
        add_subdirectory(examples/cuda_task_backend)
    endif()
endif()

if(GLIMMER_BUILD_GEMM_COMPARISON)
    add_subdirectory(examples/gemm_comparison)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

find_program(GLIMMER_CLANG_FORMAT NAMES clang-format)

if(GLIMMER_CLANG_FORMAT)
    file(GLOB_RECURSE GLIMMER_FORMAT_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/include/*.cuh"
        "${PROJECT_SOURCE_DIR}/src/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cuh"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/examples/*.h"
        "${PROJECT_SOURCE_DIR}/examples/*.hpp"
        "${PROJECT_SOURCE_DIR}/examples/*.cuh"
        "${PROJECT_SOURCE_DIR}/examples/*.cc"
        "${PROJECT_SOURCE_DIR}/examples/*.cpp"
        "${PROJECT_SOURCE_DIR}/examples/*.cu"
        "${PROJECT_SOURCE_DIR}/tests/*.h"
        "${PROJECT_SOURCE_DIR}/tests/*.hpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cuh"
        "${PROJECT_SOURCE_DIR}/tests/*.cc"
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cu"
    )

    # Examples may contain ignored framework environments and generated output.
    # They are external artifacts, not Glimmer sources, and must not be linted.
    list(FILTER GLIMMER_FORMAT_SOURCES EXCLUDE REGEX "(^|/)(\\.venv|output|build)/")
    list(TRANSFORM GLIMMER_FORMAT_SOURCES PREPEND "${PROJECT_SOURCE_DIR}/")

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
