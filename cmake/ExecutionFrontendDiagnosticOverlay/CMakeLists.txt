cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/../BuildStorage.cmake")
glimmer_check_build_storage()
project(GlimmerFrontendDiagnostic LANGUAGES CXX)
option(GLIMMER_TRANSPORT_DIAGNOSTICS "Rebuild the transport for matching diagnostic request headers" OFF)
if(NOT GLIMMER_TRANSPORT_DIAGNOSTICS)
    message(FATAL_ERROR "Current request headers require GLIMMER_TRANSPORT_DIAGNOSTICS=ON")
endif()

# Only this wrapper wires the matching transport object into every final target.
set_property(GLOBAL PROPERTY GLIMMER_TRANSPORT_OBJECT_WRAPPER TRUE)
include("${CMAKE_CURRENT_LIST_DIR}/../ExecutionServiceDiagnosticOverlay/CMakeLists.txt")
set_property(GLOBAL PROPERTY GLIMMER_TRANSPORT_OBJECT_WRAPPER FALSE)

# Reuse the same immutable Debug archives for the real execution frontend.
# Its original control dependency links spdlog without exporting compile flags.
set_target_properties(Glimmer::control PROPERTIES
    INTERFACE_LINK_LIBRARIES "Glimmer::core;Threads::Threads;$<LINK_ONLY:spdlog::spdlog>")
set(BUILD_TESTING OFF)
add_subdirectory("${PROJECT_SOURCE_DIR}/src/interceptor/execution" frontend)
option(GLIMMER_FRONTEND_OPTIMIZE "Optimize only the diagnostic frontend C++ translation units" OFF)
if(GLIMMER_FRONTEND_OPTIMIZE)
    target_compile_options(glimmer_execution_interceptor PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-O2>")
endif()

add_executable(glimmer_execution_copy_audit_test
    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_copy_audit_test.cc")
target_include_directories(glimmer_execution_copy_audit_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/interceptor" "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(glimmer_execution_copy_audit_test PRIVATE Glimmer::project_options Threads::Threads)
target_link_options(glimmer_execution_copy_audit_test PRIVATE
    "LINKER:--wrap=clock_gettime" "LINKER:--wrap=write")
add_test(NAME glimmer_execution_copy_audit_test COMMAND glimmer_execution_copy_audit_test)
set_tests_properties(glimmer_execution_copy_audit_test PROPERTIES TIMEOUT 30)

# Exercise the actual frontend against the existing metadata-only SDK fixtures.
foreach(provider memory_driver blas_provider lt_provider)
    add_library(glimmer_execution_${provider} SHARED IMPORTED)
endforeach()
set_target_properties(glimmer_execution_memory_driver PROPERTIES
    IMPORTED_LOCATION "${GLIMMER_BASELINE_BUILD}/src/interceptor/execution/libcuda.so.execution-memory-fixture.so")
set_target_properties(glimmer_execution_blas_provider PROPERTIES
    IMPORTED_LOCATION "${GLIMMER_BASELINE_BUILD}/src/interceptor/execution/libcublas.so.execution-fixture.so")
set_target_properties(glimmer_execution_lt_provider PROPERTIES
    IMPORTED_LOCATION "${GLIMMER_BASELINE_BUILD}/src/interceptor/execution/libcublasLt.so.execution-fixture.so")
add_executable(glimmer_execution_blas_test
    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_test.cc")
target_include_directories(glimmer_execution_blas_test PRIVATE
    "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
target_link_libraries(glimmer_execution_blas_test PRIVATE glimmer_execution_interceptor
    glimmer_execution_memory_driver glimmer_execution_blas_provider glimmer_execution_lt_provider
    Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
add_test(NAME glimmer_execution_blas_test COMMAND glimmer_execution_blas_test
    "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>")
add_test(NAME glimmer_execution_blas_quarantine_test COMMAND glimmer_execution_blas_test
    "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>" --quarantine)
set_tests_properties(glimmer_execution_blas_test glimmer_execution_blas_quarantine_test PROPERTIES TIMEOUT 30)

if(GLIMMER_TRANSPORT_DIAGNOSTICS)
    # An explicit object precedes imported archives in every selected caller.
    # Never rely on replacement-archive extraction order for the request ABI.
    add_library(glimmer_execution_transport_diagnostic OBJECT
        "${PROJECT_SOURCE_DIR}/src/control/execution_socket.cc")
    set_target_properties(glimmer_execution_transport_diagnostic PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(glimmer_execution_transport_diagnostic PRIVATE
        Glimmer::core Glimmer::project_options spdlog::spdlog Threads::Threads)
    foreach(target glimmer_execution_service glimmer_execution_interceptor glimmer_execution_blas_test)
        target_sources(${target} PRIVATE $<TARGET_OBJECTS:glimmer_execution_transport_diagnostic>)
        target_link_options(${target} PRIVATE "LINKER:-Map,${CMAKE_BINARY_DIR}/${target}.map")
    endforeach()

    add_executable(glimmer_control_execution_socket_test
        "${PROJECT_SOURCE_DIR}/tests/control/execution_socket_test.cc"
        $<TARGET_OBJECTS:glimmer_execution_transport_diagnostic>)
    target_include_directories(glimmer_control_execution_socket_test PRIVATE "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_control_execution_socket_test PRIVATE Glimmer::control Glimmer::project_options)
    target_link_options(glimmer_control_execution_socket_test PRIVATE
        "LINKER:--wrap=send" "LINKER:--wrap=sendmsg" "LINKER:--wrap=recvmsg")
    add_test(NAME glimmer_control_execution_socket_test COMMAND glimmer_control_execution_socket_test)
    set_tests_properties(glimmer_control_execution_socket_test PROPERTIES TIMEOUT 30)

    add_executable(glimmer_execution_service_test
        "${PROJECT_SOURCE_DIR}/tests/control/execution_service_test.cc"
        $<TARGET_OBJECTS:glimmer_execution_transport_diagnostic>)
    target_include_directories(glimmer_execution_service_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_service_test PRIVATE
        glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
        Glimmer::control Glimmer::project_options Threads::Threads spdlog::spdlog)
    target_link_options(glimmer_execution_service_test PRIVATE
        "LINKER:--wrap=ppoll" "LINKER:--wrap=__ppoll_chk")
    add_test(NAME glimmer_execution_service_context_test COMMAND glimmer_execution_service_test --context-purpose)
    set_tests_properties(glimmer_execution_service_context_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_issue_depth_test COMMAND glimmer_execution_service_test --issue-depth)
    set_tests_properties(glimmer_execution_issue_depth_test PROPERTIES TIMEOUT 30)

    # Reuse the existing host-actor fixture with the current runtime/request ABI.
    add_executable(glimmer_execution_host_test
        "${PROJECT_SOURCE_DIR}/tests/control/execution_host_test.cc"
        $<TARGET_OBJECTS:glimmer_execution_transport_diagnostic>)
    target_include_directories(glimmer_execution_host_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/control"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_host_test PRIVATE
        glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
        Glimmer::control Glimmer::project_options Threads::Threads spdlog::spdlog)
    foreach(mode on off)
        add_test(NAME glimmer_execution_host_plans_${mode}_test
            COMMAND glimmer_execution_host_test --helper-plans-${mode})
        set_tests_properties(glimmer_execution_host_plans_${mode}_test PROPERTIES TIMEOUT 30)
    endforeach()
endif()

option(GLIMMER_SESSION_ADMISSION_DIAGNOSTICS "Rebuild the exact async admission provenance boundary" OFF)
if(NOT GLIMMER_SESSION_ADMISSION_DIAGNOSTICS)
    message(FATAL_ERROR "Current admission diagnostics require GLIMMER_SESSION_ADMISSION_DIAGNOSTICS=ON")
endif()
if(GLIMMER_SESSION_ADMISSION_DIAGNOSTICS)
    # Rebuild every session translation unit so no baseline session object is linked.
    add_library(glimmer_execution_session_diagnostic OBJECT
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_execution_session.cc"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_execution_session_blas.cc"
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/cuda_execution_session_dnn.cc")
    set_target_properties(glimmer_execution_session_diagnostic PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(glimmer_execution_session_diagnostic PRIVATE
        "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}" "${GLIMMER_BASELINE_BUILD}/src/backend/cuda")
    target_compile_definitions(glimmer_execution_session_diagnostic PRIVATE GLIMMER_HAS_CUTLASS_SGEMM=1)
    target_link_libraries(glimmer_execution_session_diagnostic PRIVATE
        Glimmer::cuda_backend Glimmer::nccl_headers Glimmer::cudnn_headers)
    foreach(target glimmer_execution_service glimmer_execution_service_test glimmer_execution_host_test)
        target_sources(${target} PRIVATE $<TARGET_OBJECTS:glimmer_execution_session_diagnostic>)
        target_link_options(${target} PRIVATE "LINKER:-Map,${CMAKE_BINARY_DIR}/${target}.map")
    endforeach()
    add_executable(glimmer_backend_cuda_execution_async_admission_test
        "${PROJECT_SOURCE_DIR}/tests/backend/cuda_execution_async_admission_test.cc"
        $<TARGET_OBJECTS:glimmer_execution_session_diagnostic>)
    target_include_directories(glimmer_backend_cuda_execution_async_admission_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/backend/cuda" "${PROJECT_SOURCE_DIR}/src/backend"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_backend_cuda_execution_async_admission_test PRIVATE
        Glimmer::cuda_backend Glimmer::transform Glimmer::project_options Threads::Threads)
    target_link_options(glimmer_backend_cuda_execution_async_admission_test PRIVATE
        "LINKER:--wrap=getpid"
        "LINKER:-Map,${CMAKE_BINARY_DIR}/glimmer_backend_cuda_execution_async_admission_test.map")
    add_test(NAME glimmer_backend_cuda_execution_async_admission_test
        COMMAND glimmer_backend_cuda_execution_async_admission_test)
    set_tests_properties(glimmer_backend_cuda_execution_async_admission_test PROPERTIES TIMEOUT 45)
endif()
