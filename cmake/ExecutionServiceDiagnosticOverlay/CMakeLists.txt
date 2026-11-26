cmake_minimum_required(VERSION 3.20)
get_property(is_transport_wrapper GLOBAL PROPERTY GLIMMER_TRANSPORT_OBJECT_WRAPPER)
if(NOT is_transport_wrapper OR NOT GLIMMER_TRANSPORT_DIAGNOSTICS)
    message(FATAL_ERROR
        "Current request headers require ExecutionFrontendDiagnosticOverlay with GLIMMER_TRANSPORT_DIAGNOSTICS=ON; an imported transport archive alone is ABI-incompatible")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../BuildStorage.cmake")
glimmer_check_build_storage()
project(GlimmerServiceDiagnostic LANGUAGES CXX)

# A service-only candidate may reuse an immutable, ABI-compatible working
# baseline. Its real source target remains authoritative for dependencies.
if(NOT IS_ABSOLUTE "${GLIMMER_BASELINE_BUILD}" OR
   NOT EXISTS "${GLIMMER_BASELINE_BUILD}/CMakeCache.txt")
    message(FATAL_ERROR "An explicit existing baseline build is required")
endif()
get_filename_component(GLIMMER_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." REALPATH)
set(PROJECT_SOURCE_DIR "${GLIMMER_SOURCE_ROOT}")
load_cache("${GLIMMER_BASELINE_BUILD}" READ_WITH_PREFIX BASE_
    CMAKE_BUILD_TYPE CMAKE_CXX_COMPILER CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG
    CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_DEBUG
    GLIMMER_ENABLE_SANITIZERS GLIMMER_ENABLE_CLANG_TIDY GLIMMER_WARNINGS_AS_ERRORS CUDAToolkit_ROOT
    GLIMMER_CUDNN_ROOT GLIMMER_NCCL_ROOT GLIMMER_CUDNN_ABI_PROFILE GLIMMER_ENABLE_NCCL_EXECUTION)
if(NOT "${BASE_CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
   NOT "${CMAKE_CXX_COMPILER}" STREQUAL "${BASE_CMAKE_CXX_COMPILER}" OR
   NOT "${BASE_CMAKE_CXX_FLAGS}" STREQUAL "" OR NOT "${BASE_GLIMMER_CUDNN_ABI_PROFILE}" STREQUAL "9")
    message(FATAL_ERROR "Only the matching explicit Debug compiler/SDK baseline is supported")
endif()
foreach(flag CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_DEBUG)
    if(NOT "${${flag}}" STREQUAL "${BASE_${flag}}")
        message(FATAL_ERROR "Overlay ${flag} differs from the baseline")
    endif()
endforeach()
foreach(option GLIMMER_ENABLE_SANITIZERS GLIMMER_ENABLE_CLANG_TIDY GLIMMER_WARNINGS_AS_ERRORS)
    if(DEFINED ${option} AND NOT "${${option}}" STREQUAL "${BASE_${option}}")
        message(FATAL_ERROR "Overlay ${option} differs from the baseline")
    endif()
    set(${option} "${BASE_${option}}")
endforeach()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDAToolkit_ROOT "${BASE_CUDAToolkit_ROOT}")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
set(GLIMMER_ENABLE_NCCL_EXECUTION "${BASE_GLIMMER_ENABLE_NCCL_EXECUTION}")
include("${PROJECT_SOURCE_DIR}/cmake/ProjectOptions.cmake")

function(baseline_archive target relative)
    if(NOT EXISTS "${GLIMMER_BASELINE_BUILD}/${relative}")
        message(FATAL_ERROR "Missing baseline archive: ${relative}")
    endif()
    add_library(${target} STATIC IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES
        IMPORTED_LOCATION "${GLIMMER_BASELINE_BUILD}/${relative}"
        IMPORTED_NO_SYSTEM TRUE
        INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include")
endfunction()
baseline_archive(Glimmer::core src/core/libglimmer_core.a)
baseline_archive(Glimmer::control src/control/libglimmer_control.a)
baseline_archive(Glimmer::backend src/backend/libglimmer_backend.a)
baseline_archive(Glimmer::transform src/transform/libglimmer_transform.a)
baseline_archive(Glimmer::cuda_backend src/backend/cuda/libglimmer_cuda_backend.a)
baseline_archive(spdlog::spdlog 3rdparty/spdlog/libspdlogd.a)
set_target_properties(spdlog::spdlog PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/3rdparty/spdlog/include"
    INTERFACE_COMPILE_DEFINITIONS SPDLOG_COMPILED_LIB)
set_target_properties(Glimmer::backend PROPERTIES INTERFACE_LINK_LIBRARIES Glimmer::core)
set_target_properties(Glimmer::control PROPERTIES INTERFACE_LINK_LIBRARIES "Glimmer::core;Threads::Threads")
set_target_properties(Glimmer::cuda_backend PROPERTIES INTERFACE_LINK_LIBRARIES
    "Glimmer::backend;Glimmer::transform;Glimmer::project_options;spdlog::spdlog;Threads::Threads")

foreach(sdk cudnn nccl)
    string(TOUPPER "${sdk}" upper)
    add_library(glimmer_${sdk}_headers INTERFACE)
    add_library(Glimmer::${sdk}_headers ALIAS glimmer_${sdk}_headers)
    target_include_directories(glimmer_${sdk}_headers SYSTEM INTERFACE "${BASE_GLIMMER_${upper}_ROOT}/include")
    target_compile_definitions(glimmer_${sdk}_headers INTERFACE GLIMMER_HAS_${upper}=1)
    add_library(Glimmer::${sdk} SHARED IMPORTED GLOBAL)
    if(sdk STREQUAL "cudnn")
        set(soname libcudnn.so.9)
        target_include_directories(glimmer_cudnn_headers INTERFACE
            "${GLIMMER_BASELINE_BUILD}/generated/cudnn-production")
    else()
        set(soname libnccl.so.2)
    endif()
    set_target_properties(Glimmer::${sdk} PROPERTIES
        IMPORTED_LOCATION "${BASE_GLIMMER_${upper}_ROOT}/lib/${soname}"
        INTERFACE_LINK_LIBRARIES Glimmer::${sdk}_headers)
endforeach()

# Build the real service target, including all of its runtime translation units.
set(BUILD_TESTING OFF)
add_subdirectory("${PROJECT_SOURCE_DIR}/src/execution_service" service)
set_target_properties(glimmer_execution_service_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_executable(glimmer_execution_host_stage_metrics_test
    "${PROJECT_SOURCE_DIR}/tests/control/execution_host_stage_metrics_test.cc"
    "${PROJECT_SOURCE_DIR}/src/execution_service/host_stage_metrics.cc")
target_include_directories(glimmer_execution_host_stage_metrics_test PRIVATE "${PROJECT_SOURCE_DIR}/src/execution_service")
target_link_libraries(glimmer_execution_host_stage_metrics_test PRIVATE Glimmer::project_options)
add_test(NAME glimmer_execution_host_stage_metrics_test COMMAND glimmer_execution_host_stage_metrics_test)
set_tests_properties(glimmer_execution_host_stage_metrics_test PROPERTIES TIMEOUT 15)
