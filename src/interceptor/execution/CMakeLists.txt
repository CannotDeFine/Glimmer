function(glimmer_add_dnn_fixture_test)
    cmake_parse_arguments(_dnn_test "" "NAME" "COMMAND" ${ARGN})
    list(POP_FRONT _dnn_test_COMMAND _dnn_application)
    if(NOT _dnn_test_NAME OR NOT TARGET "${_dnn_application}")
        message(FATAL_ERROR "DNN fixture tests require an explicit application target")
    endif()
    glimmer_share_execution_fixture("${_dnn_application}")
    # Ambient empty path components prefer CWD over the fixture's exact RUNPATH.
    # Explicit wrong-provider cases still construct their own dlopen/dlmopen input.
    add_test(NAME "${_dnn_test_NAME}" COMMAND "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH "$<TARGET_FILE:${_dnn_application}>" ${_dnn_test_COMMAND})
endfunction()

add_library(glimmer_execution_interceptor SHARED driver_frontend.cc driver_lookup.cc)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    enable_language(ASM)
    target_sources(glimmer_execution_interceptor PRIVATE ../dlsym_entry_x86_64.S)
endif()

set_target_properties(glimmer_execution_interceptor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
target_include_directories(glimmer_execution_interceptor PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(glimmer_execution_interceptor PRIVATE
    Glimmer::control Glimmer::transform Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
glimmer_enable_clang_tidy(glimmer_execution_interceptor)
if(TARGET Glimmer::cudnn_headers)
    target_link_libraries(glimmer_execution_interceptor PRIVATE Glimmer::cudnn_headers)
endif()
if(GLIMMER_ENABLE_NCCL_EXECUTION)
    target_link_libraries(glimmer_execution_interceptor PRIVATE Glimmer::nccl_headers)
endif()
target_link_options(glimmer_execution_interceptor PRIVATE
    "LINKER:-Bsymbolic-functions"
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_property(TARGET glimmer_execution_interceptor APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
if(BUILD_TESTING)
    add_library(glimmer_execution_unified_copy_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_unified_copy_provider.cc")
    set_target_properties(glimmer_execution_unified_copy_provider PROPERTIES
        OUTPUT_NAME cuda SUFFIX .so.unified-copy-fixture)
    target_include_directories(glimmer_execution_unified_copy_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_unified_copy_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_unified_copy_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_unified_copy_provider)
    add_executable(glimmer_execution_unified_copy_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_unified_copy_test.cc")
    target_include_directories(glimmer_execution_unified_copy_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_unified_copy_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_unified_copy_provider
        Glimmer::control Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    set_target_properties(glimmer_execution_unified_copy_test PROPERTIES ENABLE_EXPORTS ON)
    glimmer_enable_clang_tidy(glimmer_execution_unified_copy_test)
    add_test(NAME glimmer_execution_unified_copy_test COMMAND glimmer_execution_unified_copy_test
        "$<TARGET_FILE:glimmer_execution_unified_copy_provider>")
    set_tests_properties(glimmer_execution_unified_copy_test PROPERTIES TIMEOUT 30)
    foreach(glimmer_unified_case RANGE 1 14)
        add_test(NAME glimmer_execution_unified_copy_receipt_${glimmer_unified_case}_test
            COMMAND glimmer_execution_unified_copy_test
            "$<TARGET_FILE:glimmer_execution_unified_copy_provider>" ${glimmer_unified_case})
        set_tests_properties(glimmer_execution_unified_copy_receipt_${glimmer_unified_case}_test
            PROPERTIES TIMEOUT 30)
    endforeach()
    glimmer_add_test_program(glimmer_execution_copy_audit_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_copy_audit_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor" "${PROJECT_SOURCE_DIR}/include"
        LIBRARIES Glimmer::project_options Threads::Threads
        LINK_OPTIONS "LINKER:--wrap=clock_gettime" "LINKER:--wrap=write"
        TIMEOUT 30)
    if(TARGET Glimmer::nccl_headers)
        if(TARGET Glimmer::nccl)
            # Identical ordinary SDK input for supervised native/public-owned
            # pairs. Never run its device path in the no-GPU CTest suite.
            add_executable(glimmer_execution_nccl_async_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_nccl_async_gpu_test.cc")
            target_link_libraries(glimmer_execution_nccl_async_gpu_test PRIVATE
                Glimmer::nccl CUDA::cuda_driver CUDA::cudart
                Glimmer::project_options Threads::Threads)
            glimmer_enable_clang_tidy(glimmer_execution_nccl_async_gpu_test)
            add_test(NAME glimmer_execution_nccl_async_arguments_test
                COMMAND glimmer_execution_nccl_async_gpu_test --help)
            set_tests_properties(glimmer_execution_nccl_async_arguments_test PROPERTIES TIMEOUT 15)
            add_test(NAME glimmer_execution_nccl_async_missing_arguments_test
                COMMAND glimmer_execution_nccl_async_gpu_test)
            add_test(NAME glimmer_execution_nccl_async_invalid_placement_test
                COMMAND glimmer_execution_nccl_async_gpu_test invalid direct destroy)
            add_test(NAME glimmer_execution_nccl_async_invalid_mode_test
                COMMAND glimmer_execution_nccl_async_gpu_test primary-after invalid destroy)
            add_test(NAME glimmer_execution_nccl_async_invalid_cleanup_test
                COMMAND glimmer_execution_nccl_async_gpu_test primary-after direct invalid)
            add_test(NAME glimmer_execution_nccl_async_extra_arguments_test
                COMMAND glimmer_execution_nccl_async_gpu_test primary-after direct destroy extra)
            set_tests_properties(
                glimmer_execution_nccl_async_missing_arguments_test
                glimmer_execution_nccl_async_invalid_placement_test
                glimmer_execution_nccl_async_invalid_mode_test
                glimmer_execution_nccl_async_invalid_cleanup_test
                glimmer_execution_nccl_async_extra_arguments_test
                PROPERTIES TIMEOUT 15 WILL_FAIL TRUE)
        endif()
        # This independently enabled DSO keeps the isolated socket fixture usable
        # even when production NCCL activation is disabled.
        add_library(glimmer_execution_nccl_test_interceptor SHARED driver_frontend.cc driver_lookup.cc)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
            target_sources(glimmer_execution_nccl_test_interceptor PRIVATE ../dlsym_entry_x86_64.S)
        endif()
        target_include_directories(glimmer_execution_nccl_test_interceptor PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(glimmer_execution_nccl_test_interceptor PRIVATE
            Glimmer::control Glimmer::transform Glimmer::project_options Glimmer::nccl_headers
            Threads::Threads "${CMAKE_DL_LIBS}")
        if(TARGET Glimmer::cudnn_headers)
            target_link_libraries(glimmer_execution_nccl_test_interceptor PRIVATE Glimmer::cudnn_headers)
        endif()
        target_link_options(glimmer_execution_nccl_test_interceptor PRIVATE
            "LINKER:-Bsymbolic-functions"
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
        set_property(TARGET glimmer_execution_nccl_test_interceptor APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
        glimmer_enable_clang_tidy(glimmer_execution_nccl_test_interceptor)
        add_library(glimmer_execution_nccl_frontend_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_nccl_frontend_provider.cc")
        set_target_properties(glimmer_execution_nccl_frontend_provider PROPERTIES OUTPUT_NAME nccl SOVERSION 99)
        target_link_libraries(glimmer_execution_nccl_frontend_provider PRIVATE
            Glimmer::nccl_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_nccl_frontend_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_nccl_frontend_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_nccl_frontend_provider)
        add_executable(glimmer_execution_nccl_frontend_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_nccl_frontend_test.cc")
        target_include_directories(glimmer_execution_nccl_frontend_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src/control" "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(glimmer_execution_nccl_frontend_test PRIVATE
            glimmer_execution_nccl_test_interceptor glimmer_execution_nccl_frontend_provider
            Glimmer::control Glimmer::nccl_headers Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_nccl_frontend_test)
        add_test(NAME glimmer_execution_nccl_frontend_test
            COMMAND glimmer_execution_nccl_frontend_test "$<TARGET_FILE:glimmer_execution_nccl_frontend_provider>")
        set_tests_properties(glimmer_execution_nccl_frontend_test PROPERTIES TIMEOUT 90)
    endif()
    add_library(glimmer_execution_pointer_metadata_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pointer_metadata_test.cc")
    set_target_properties(glimmer_execution_pointer_metadata_provider PROPERTIES
        OUTPUT_NAME "cudart" SUFFIX ".so.pointer-metadata-fixture")
    target_compile_definitions(glimmer_execution_pointer_metadata_provider PRIVATE
        GLIMMER_POINTER_METADATA_PROVIDER)
    target_include_directories(glimmer_execution_pointer_metadata_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_pointer_metadata_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_pointer_metadata_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_pointer_metadata_provider)
    add_library(glimmer_execution_pointer_metadata_driver_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pointer_metadata_test.cc")
    set_target_properties(glimmer_execution_pointer_metadata_driver_provider PROPERTIES
        OUTPUT_NAME "cuda" SUFFIX ".so.pointer-metadata-fixture")
    target_compile_definitions(glimmer_execution_pointer_metadata_driver_provider PRIVATE
        GLIMMER_POINTER_METADATA_PROVIDER)
    target_include_directories(glimmer_execution_pointer_metadata_driver_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_pointer_metadata_driver_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_pointer_metadata_driver_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_pointer_metadata_driver_provider)
    add_executable(glimmer_execution_pointer_metadata_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pointer_metadata_test.cc")
    target_include_directories(glimmer_execution_pointer_metadata_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_pointer_metadata_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_pointer_metadata_provider
        glimmer_execution_pointer_metadata_driver_provider
        Glimmer::control Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_pointer_metadata_test)
    add_test(NAME glimmer_execution_pointer_metadata_test
        COMMAND glimmer_execution_pointer_metadata_test
            "$<TARGET_FILE:glimmer_execution_pointer_metadata_provider>"
            "$<TARGET_FILE:glimmer_execution_pointer_metadata_driver_provider>")
    set_tests_properties(glimmer_execution_pointer_metadata_test PROPERTIES TIMEOUT 60)
    add_library(glimmer_execution_runtime_function_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_function_test.cc")
    set_target_properties(glimmer_execution_runtime_function_provider PROPERTIES
        OUTPUT_NAME "cudart" SUFFIX ".so.runtime-function-fixture")
    target_compile_definitions(glimmer_execution_runtime_function_provider PRIVATE
        GLIMMER_RUNTIME_FUNCTION_PROVIDER)
    target_include_directories(glimmer_execution_runtime_function_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_runtime_function_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_runtime_function_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_runtime_function_provider)
    add_executable(glimmer_execution_runtime_function_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_function_test.cc")
    target_include_directories(glimmer_execution_runtime_function_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_runtime_function_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_runtime_function_provider
        Glimmer::control Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_runtime_function_test)
    add_test(NAME glimmer_execution_runtime_function_test
        COMMAND glimmer_execution_runtime_function_test
            "$<TARGET_FILE:glimmer_execution_runtime_function_provider>")
    set_tests_properties(glimmer_execution_runtime_function_test PROPERTIES TIMEOUT 60)
    add_library(glimmer_execution_function_control_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_function_control_test.cc")
    set_target_properties(glimmer_execution_function_control_provider PROPERTIES
        OUTPUT_NAME "cuda" SUFFIX ".so.function-control-fixture")
    target_compile_definitions(glimmer_execution_function_control_provider PRIVATE
        GLIMMER_FUNCTION_CONTROL_PROVIDER)
    target_include_directories(glimmer_execution_function_control_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_function_control_provider PRIVATE
        Glimmer::project_options)
    target_link_options(glimmer_execution_function_control_provider PRIVATE
        "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_function_control_provider)
    add_executable(glimmer_execution_function_control_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_function_control_test.cc")
    target_include_directories(glimmer_execution_function_control_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_function_control_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_function_control_provider
        Glimmer::control Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_function_control_test)
    add_test(NAME glimmer_execution_function_control_test
        COMMAND glimmer_execution_function_control_test
            "$<TARGET_FILE:glimmer_execution_function_control_provider>")
    set_tests_properties(glimmer_execution_function_control_test PROPERTIES TIMEOUT 45)
    add_library(glimmer_execution_module_factory_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_module_factory_test.cc")
    set_target_properties(glimmer_execution_module_factory_provider PROPERTIES
        OUTPUT_NAME "cuda" SUFFIX ".so.module-factory-fixture")
    target_compile_definitions(glimmer_execution_module_factory_provider PRIVATE
        GLIMMER_MODULE_FACTORY_PROVIDER)
    target_include_directories(glimmer_execution_module_factory_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_module_factory_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_module_factory_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_module_factory_provider)
    add_executable(glimmer_execution_module_factory_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_module_factory_test.cc")
    target_include_directories(glimmer_execution_module_factory_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_module_factory_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_module_factory_provider
        Glimmer::control Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_module_factory_test)
    foreach(api RANGE 0 2)
        foreach(scenario RANGE 0 39)
            if((scenario GREATER 0 AND scenario LESS 19) OR api EQUAL 0)
                add_test(NAME "glimmer_execution_module_factory_${api}_${scenario}_test"
                    COMMAND glimmer_execution_module_factory_test
                        "$<TARGET_FILE:glimmer_execution_module_factory_provider>" "${api}" "${scenario}")
                set_tests_properties("glimmer_execution_module_factory_${api}_${scenario}_test"
                    PROPERTIES TIMEOUT 20)
            endif()
        endforeach()
    endforeach()
    add_library(glimmer_execution_constructor_guard_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_constructor_guards_test.cc")
    set_target_properties(glimmer_execution_constructor_guard_provider PROPERTIES
        OUTPUT_NAME "cuda" SUFFIX ".so.constructor-guard-fixture")
    target_compile_definitions(glimmer_execution_constructor_guard_provider PRIVATE
        GLIMMER_CONSTRUCTOR_GUARD_PROVIDER)
    target_include_directories(glimmer_execution_constructor_guard_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_constructor_guard_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_constructor_guard_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_constructor_guard_provider)
    add_executable(glimmer_execution_constructor_guards_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_constructor_guards_test.cc")
    target_include_directories(glimmer_execution_constructor_guards_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_constructor_guards_test PRIVATE
        glimmer_execution_interceptor glimmer_execution_constructor_guard_provider
        Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_constructor_guards_test)
    add_test(NAME glimmer_execution_constructor_guards_test
        COMMAND glimmer_execution_constructor_guards_test
            "$<TARGET_FILE:glimmer_execution_constructor_guard_provider>")
    set_tests_properties(glimmer_execution_constructor_guards_test PROPERTIES TIMEOUT 45)
    if(TARGET Glimmer::cudnn_headers AND TARGET glimmer_execution_service_runtime)
        add_library(glimmer_execution_dnn_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_provider.cc")
        set_target_properties(glimmer_execution_dnn_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.execution-fixture")
        target_link_libraries(glimmer_execution_dnn_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_provider)
        add_executable(glimmer_execution_dnn_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_test.cc")
        target_include_directories(glimmer_execution_dnn_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_test COMMAND glimmer_execution_dnn_test
            "$<TARGET_FILE:glimmer_execution_dnn_provider>")
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_quarantine_test COMMAND glimmer_execution_dnn_test
            "$<TARGET_FILE:glimmer_execution_dnn_provider>" --quarantine)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_malformed_reply_test COMMAND glimmer_execution_dnn_test
            $<TARGET_FILE:glimmer_execution_dnn_provider> --malformed-replies)
        set_tests_properties(glimmer_execution_dnn_malformed_reply_test PROPERTIES TIMEOUT 45)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_profile_test COMMAND glimmer_execution_dnn_test
            "$<TARGET_FILE:glimmer_execution_dnn_provider>" --profile)
        set_tests_properties(glimmer_execution_dnn_test glimmer_execution_dnn_quarantine_test
            glimmer_execution_dnn_profile_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_convolution_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_convolution_provider.cc")
        set_target_properties(glimmer_execution_dnn_convolution_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.convolution-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_convolution_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_convolution_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_convolution_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_convolution_provider)
        add_executable(glimmer_execution_dnn_convolution_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_convolution_test.cc")
        target_include_directories(glimmer_execution_dnn_convolution_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_convolution_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_convolution_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_convolution_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_convolution_test COMMAND glimmer_execution_dnn_convolution_test
            "$<TARGET_FILE:glimmer_execution_dnn_convolution_provider>")
        set_tests_properties(glimmer_execution_dnn_convolution_test PROPERTIES TIMEOUT 45)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_convolution_malformed_replies_test
            COMMAND glimmer_execution_dnn_convolution_test
                "$<TARGET_FILE:glimmer_execution_dnn_convolution_provider>" "--malformed-replies")
        set_tests_properties(glimmer_execution_dnn_convolution_malformed_replies_test
            PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_pooling_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_pooling_provider.cc")
        set_target_properties(glimmer_execution_dnn_pooling_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.pooling-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_pooling_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_pooling_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_pooling_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_pooling_provider)
        add_executable(glimmer_execution_dnn_pooling_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_pooling_test.cc")
        target_include_directories(glimmer_execution_dnn_pooling_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_pooling_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_pooling_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_pooling_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_pooling_test COMMAND glimmer_execution_dnn_pooling_test
            "$<TARGET_FILE:glimmer_execution_dnn_pooling_provider>")
        set_tests_properties(glimmer_execution_dnn_pooling_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_lrn_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_lrn_provider.cc")
        set_target_properties(glimmer_execution_dnn_lrn_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.lrn-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_lrn_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_lrn_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_lrn_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_lrn_provider)
        add_executable(glimmer_execution_dnn_lrn_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_lrn_test.cc")
        target_include_directories(glimmer_execution_dnn_lrn_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_lrn_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_lrn_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_lrn_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_lrn_test COMMAND glimmer_execution_dnn_lrn_test
            "$<TARGET_FILE:glimmer_execution_dnn_lrn_provider>")
        set_tests_properties(glimmer_execution_dnn_lrn_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_reduction_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reduction_provider.cc")
        set_target_properties(glimmer_execution_dnn_reduction_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.reduction-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_reduction_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_reduction_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_reduction_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_reduction_provider)
        add_executable(glimmer_execution_dnn_reduction_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reduction_test.cc")
        target_include_directories(glimmer_execution_dnn_reduction_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_reduction_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_reduction_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_reduction_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_reduction_test COMMAND glimmer_execution_dnn_reduction_test
            "$<TARGET_FILE:glimmer_execution_dnn_reduction_provider>")
        set_tests_properties(glimmer_execution_dnn_reduction_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_batch_norm_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_batch_norm_provider.cc")
        set_target_properties(glimmer_execution_dnn_batch_norm_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.batch-norm-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_batch_norm_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_batch_norm_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_batch_norm_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_batch_norm_provider)
        add_executable(glimmer_execution_dnn_batch_norm_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_batch_norm_test.cc")
        target_include_directories(glimmer_execution_dnn_batch_norm_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_batch_norm_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_batch_norm_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_batch_norm_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_batch_norm_test COMMAND glimmer_execution_dnn_batch_norm_test
            "$<TARGET_FILE:glimmer_execution_dnn_batch_norm_provider>")
        set_tests_properties(glimmer_execution_dnn_batch_norm_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_dropout_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_dropout_provider.cc")
        set_target_properties(glimmer_execution_dnn_dropout_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.dropout-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_dropout_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_dropout_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_dropout_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_dropout_provider)
        add_executable(glimmer_execution_dnn_dropout_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_dropout_test.cc")
        target_include_directories(glimmer_execution_dnn_dropout_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_dropout_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_dropout_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_dropout_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_dropout_test COMMAND glimmer_execution_dnn_dropout_test
            "$<TARGET_FILE:glimmer_execution_dnn_dropout_provider>")
        set_tests_properties(glimmer_execution_dnn_dropout_test PROPERTIES TIMEOUT 45)
        foreach(_dropout_mode quarantine proof-failure)
            string(REPLACE "-" "_" _dropout_test "${_dropout_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_dropout_${_dropout_test}_test
                COMMAND glimmer_execution_dnn_dropout_test
                    "$<TARGET_FILE:glimmer_execution_dnn_dropout_provider>" "--${_dropout_mode}")
            set_tests_properties(glimmer_execution_dnn_dropout_${_dropout_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_op_tensor_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_op_tensor_provider.cc")
        set_target_properties(glimmer_execution_dnn_op_tensor_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.op-tensor-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_op_tensor_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_op_tensor_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_op_tensor_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_op_tensor_provider)
        add_executable(glimmer_execution_dnn_op_tensor_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_op_tensor_test.cc")
        target_include_directories(glimmer_execution_dnn_op_tensor_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_op_tensor_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_op_tensor_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_op_tensor_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_op_tensor_test COMMAND glimmer_execution_dnn_op_tensor_test
            "$<TARGET_FILE:glimmer_execution_dnn_op_tensor_provider>")
        set_tests_properties(glimmer_execution_dnn_op_tensor_test PROPERTIES TIMEOUT 45)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_op_tensor_profile_test
            COMMAND glimmer_execution_dnn_op_tensor_test
                "$<TARGET_FILE:glimmer_execution_dnn_op_tensor_provider>" --profile)
        set_tests_properties(glimmer_execution_dnn_op_tensor_profile_test PROPERTIES TIMEOUT 45)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_op_tensor_quarantine_test
            COMMAND glimmer_execution_dnn_op_tensor_test
                "$<TARGET_FILE:glimmer_execution_dnn_op_tensor_provider>" --quarantine)
        set_tests_properties(glimmer_execution_dnn_op_tensor_quarantine_test PROPERTIES TIMEOUT 45)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_op_tensor_malformed_replies_test
            COMMAND glimmer_execution_dnn_op_tensor_test
                "$<TARGET_FILE:glimmer_execution_dnn_op_tensor_provider>" --malformed-replies)
        set_tests_properties(glimmer_execution_dnn_op_tensor_malformed_replies_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_reorder_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reorder_provider.cc")
        set_target_properties(glimmer_execution_dnn_reorder_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.reorder-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_reorder_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_reorder_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_reorder_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_reorder_provider)
        add_executable(glimmer_execution_dnn_reorder_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reorder_test.cc")
        target_include_directories(glimmer_execution_dnn_reorder_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_reorder_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_reorder_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_reorder_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_reorder_test COMMAND glimmer_execution_dnn_reorder_test
            "$<TARGET_FILE:glimmer_execution_dnn_reorder_provider>")
        set_tests_properties(glimmer_execution_dnn_reorder_test PROPERTIES TIMEOUT 45)
        add_library(glimmer_execution_dnn_library_info_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_library_info_provider.cc")
        set_target_properties(glimmer_execution_dnn_library_info_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.library-info-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_library_info_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_library_info_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_library_info_provider PRIVATE "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_library_info_provider)
        add_executable(glimmer_execution_dnn_library_info_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_library_info_test.cc")
        target_include_directories(glimmer_execution_dnn_library_info_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_library_info_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_library_info_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_library_info_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_library_info_test
            COMMAND glimmer_execution_dnn_library_info_test
                "$<TARGET_FILE:glimmer_execution_dnn_library_info_provider>")
        set_tests_properties(glimmer_execution_dnn_library_info_test PROPERTIES TIMEOUT 45)
        foreach(_info_mode profile priority missing-callback malformed-replies)
            string(REPLACE "-" "_" _info_test "${_info_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_library_info_${_info_test}_test
                COMMAND glimmer_execution_dnn_library_info_test
                    "$<TARGET_FILE:glimmer_execution_dnn_library_info_provider>" "--${_info_mode}")
            set_tests_properties(glimmer_execution_dnn_library_info_${_info_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_backward_convolution_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backward_convolution_provider.cc")
        set_target_properties(glimmer_execution_dnn_backward_convolution_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.backward-convolution-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_backward_convolution_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_backward_convolution_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_backward_convolution_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backward_convolution_provider)
        add_executable(glimmer_execution_dnn_backward_convolution_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backward_convolution_test.cc")
        target_include_directories(glimmer_execution_dnn_backward_convolution_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_backward_convolution_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_backward_convolution_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backward_convolution_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backward_convolution_test
            COMMAND glimmer_execution_dnn_backward_convolution_test
                "$<TARGET_FILE:glimmer_execution_dnn_backward_convolution_provider>")
        set_tests_properties(glimmer_execution_dnn_backward_convolution_test PROPERTIES TIMEOUT 45)
        foreach(_backward_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _backward_test "${_backward_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backward_convolution_${_backward_test}_test
                COMMAND glimmer_execution_dnn_backward_convolution_test
                    "$<TARGET_FILE:glimmer_execution_dnn_backward_convolution_provider>"
                    "--${_backward_mode}")
            set_tests_properties(glimmer_execution_dnn_backward_convolution_${_backward_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_folded_convolution_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_folded_convolution_provider.cc")
        set_target_properties(glimmer_execution_dnn_folded_convolution_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.folded-convolution-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_folded_convolution_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_folded_convolution_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_folded_convolution_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_folded_convolution_provider)
        add_executable(glimmer_execution_dnn_folded_convolution_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_folded_convolution_test.cc")
        target_include_directories(glimmer_execution_dnn_folded_convolution_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_folded_convolution_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_folded_convolution_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_folded_convolution_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_folded_convolution_test
            COMMAND glimmer_execution_dnn_folded_convolution_test
                "$<TARGET_FILE:glimmer_execution_dnn_folded_convolution_provider>")
        set_tests_properties(glimmer_execution_dnn_folded_convolution_test PROPERTIES TIMEOUT 45)
        foreach(_folded_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _folded_test "${_folded_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_folded_convolution_${_folded_test}_test
                COMMAND glimmer_execution_dnn_folded_convolution_test
                    "$<TARGET_FILE:glimmer_execution_dnn_folded_convolution_provider>"
                    "--${_folded_mode}")
            set_tests_properties(glimmer_execution_dnn_folded_convolution_${_folded_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_sequence_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_sequence_provider.cc")
        set_target_properties(glimmer_execution_dnn_sequence_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.sequence-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_sequence_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_sequence_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_sequence_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_sequence_provider)
        add_executable(glimmer_execution_dnn_sequence_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_sequence_test.cc")
        target_include_directories(glimmer_execution_dnn_sequence_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_sequence_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_sequence_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_sequence_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_sequence_test
            COMMAND glimmer_execution_dnn_sequence_test
                "$<TARGET_FILE:glimmer_execution_dnn_sequence_provider>")
        set_tests_properties(glimmer_execution_dnn_sequence_test PROPERTIES TIMEOUT 45)
        foreach(_sequence_mode profile priority poison malformed-replies)
            string(REPLACE "-" "_" _sequence_test "${_sequence_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_sequence_${_sequence_test}_test
                COMMAND glimmer_execution_dnn_sequence_test
                    "$<TARGET_FILE:glimmer_execution_dnn_sequence_provider>"
                    "--${_sequence_mode}")
            set_tests_properties(glimmer_execution_dnn_sequence_${_sequence_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_rnn_attention_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_attention_provider.cc")
        set_target_properties(glimmer_execution_dnn_rnn_attention_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.rnn-attention-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_rnn_attention_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_rnn_attention_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_rnn_attention_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_attention_provider)
        add_executable(glimmer_execution_dnn_rnn_attention_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_attention_test.cc")
        target_include_directories(glimmer_execution_dnn_rnn_attention_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_rnn_attention_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_rnn_attention_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_attention_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_attention_test
            COMMAND glimmer_execution_dnn_rnn_attention_test
                "$<TARGET_FILE:glimmer_execution_dnn_rnn_attention_provider>")
        set_tests_properties(glimmer_execution_dnn_rnn_attention_test PROPERTIES TIMEOUT 45)
        foreach(_rnn_attention_mode profile priority poison malformed-replies)
            string(REPLACE "-" "_" _rnn_attention_test "${_rnn_attention_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_attention_${_rnn_attention_test}_test
                COMMAND glimmer_execution_dnn_rnn_attention_test
                    "$<TARGET_FILE:glimmer_execution_dnn_rnn_attention_provider>"
                    "--${_rnn_attention_mode}")
            set_tests_properties(glimmer_execution_dnn_rnn_attention_${_rnn_attention_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_rnn_query_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_query_provider.cc")
        set_target_properties(glimmer_execution_dnn_rnn_query_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.rnn-query-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_rnn_query_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_rnn_query_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_rnn_query_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_query_provider)
        add_executable(glimmer_execution_dnn_rnn_query_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_query_test.cc")
        target_include_directories(glimmer_execution_dnn_rnn_query_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_rnn_query_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_rnn_query_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_query_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_query_test
            COMMAND glimmer_execution_dnn_rnn_query_test
                "$<TARGET_FILE:glimmer_execution_dnn_rnn_query_provider>")
        set_tests_properties(glimmer_execution_dnn_rnn_query_test PROPERTIES TIMEOUT 45)
        foreach(_rnn_query_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _rnn_query_test "${_rnn_query_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_query_${_rnn_query_test}_test
                COMMAND glimmer_execution_dnn_rnn_query_test
                    "$<TARGET_FILE:glimmer_execution_dnn_rnn_query_provider>"
                    "--${_rnn_query_mode}")
            set_tests_properties(glimmer_execution_dnn_rnn_query_${_rnn_query_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_rnn_compute_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_compute_provider.cc")
        set_target_properties(glimmer_execution_dnn_rnn_compute_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.rnn-compute-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_rnn_compute_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_rnn_compute_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_rnn_compute_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_compute_provider)
        add_executable(glimmer_execution_dnn_rnn_compute_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_compute_test.cc")
        target_include_directories(glimmer_execution_dnn_rnn_compute_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_rnn_compute_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_rnn_compute_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_compute_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_compute_test
            COMMAND glimmer_execution_dnn_rnn_compute_test
                "$<TARGET_FILE:glimmer_execution_dnn_rnn_compute_provider>")
        set_tests_properties(glimmer_execution_dnn_rnn_compute_test PROPERTIES TIMEOUT 45)
        foreach(_rnn_compute_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _rnn_compute_test "${_rnn_compute_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_compute_${_rnn_compute_test}_test
                COMMAND glimmer_execution_dnn_rnn_compute_test
                    "$<TARGET_FILE:glimmer_execution_dnn_rnn_compute_provider>"
                    "--${_rnn_compute_mode}")
            set_tests_properties(glimmer_execution_dnn_rnn_compute_${_rnn_compute_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_rnn_weight_params_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_weight_params_provider.cc")
        set_target_properties(glimmer_execution_dnn_rnn_weight_params_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.rnn-weight-params-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_rnn_weight_params_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_rnn_weight_params_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_rnn_weight_params_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_weight_params_provider)
        add_executable(glimmer_execution_dnn_rnn_weight_params_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_weight_params_test.cc")
        target_include_directories(glimmer_execution_dnn_rnn_weight_params_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_rnn_weight_params_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_rnn_weight_params_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_weight_params_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_weight_params_test
            COMMAND glimmer_execution_dnn_rnn_weight_params_test
                "$<TARGET_FILE:glimmer_execution_dnn_rnn_weight_params_provider>")
        set_tests_properties(glimmer_execution_dnn_rnn_weight_params_test PROPERTIES TIMEOUT 45)
        foreach(_rnn_weight_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _rnn_weight_test "${_rnn_weight_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_rnn_weight_params_${_rnn_weight_test}_test
                COMMAND glimmer_execution_dnn_rnn_weight_params_test
                    "$<TARGET_FILE:glimmer_execution_dnn_rnn_weight_params_provider>"
                    "--${_rnn_weight_mode}")
            set_tests_properties(glimmer_execution_dnn_rnn_weight_params_${_rnn_weight_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_attn_weights_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attn_weights_provider.cc")
        set_target_properties(glimmer_execution_dnn_attn_weights_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.attn-weights-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_attn_weights_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_attn_weights_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_attn_weights_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_attn_weights_provider)
        add_executable(glimmer_execution_dnn_attn_weights_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attn_weights_test.cc")
        target_include_directories(glimmer_execution_dnn_attn_weights_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_attn_weights_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_attn_weights_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_attn_weights_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_attn_weights_test
            COMMAND glimmer_execution_dnn_attn_weights_test
                "$<TARGET_FILE:glimmer_execution_dnn_attn_weights_provider>")
        set_tests_properties(glimmer_execution_dnn_attn_weights_test PROPERTIES TIMEOUT 45)
        foreach(_attn_weights_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _attn_weights_test "${_attn_weights_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_attn_weights_${_attn_weights_test}_test
                COMMAND glimmer_execution_dnn_attn_weights_test
                    "$<TARGET_FILE:glimmer_execution_dnn_attn_weights_provider>"
                    "--${_attn_weights_mode}")
            set_tests_properties(glimmer_execution_dnn_attn_weights_${_attn_weights_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_attn_compute_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attn_compute_provider.cc")
        set_target_properties(glimmer_execution_dnn_attn_compute_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.attn-compute-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_attn_compute_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_attn_compute_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_attn_compute_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_attn_compute_provider)
        add_executable(glimmer_execution_dnn_attn_compute_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attn_compute_test.cc")
        target_include_directories(glimmer_execution_dnn_attn_compute_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_attn_compute_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_attn_compute_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_attn_compute_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_attn_compute_test
            COMMAND glimmer_execution_dnn_attn_compute_test
                "$<TARGET_FILE:glimmer_execution_dnn_attn_compute_provider>")
        set_tests_properties(glimmer_execution_dnn_attn_compute_test PROPERTIES TIMEOUT 45)
        foreach(_attn_compute_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _attn_compute_test "${_attn_compute_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_attn_compute_${_attn_compute_test}_test
                COMMAND glimmer_execution_dnn_attn_compute_test
                    "$<TARGET_FILE:glimmer_execution_dnn_attn_compute_provider>"
                    "--${_attn_compute_mode}")
            set_tests_properties(glimmer_execution_dnn_attn_compute_${_attn_compute_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        add_library(glimmer_execution_dnn_backend_graph_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_graph_provider.cc")
        set_target_properties(glimmer_execution_dnn_backend_graph_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.backend-graph-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_backend_graph_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_include_directories(glimmer_execution_dnn_backend_graph_provider PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_options(glimmer_execution_dnn_backend_graph_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_graph_provider)
        add_executable(glimmer_execution_dnn_backend_graph_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_graph_test.cc")
        target_include_directories(glimmer_execution_dnn_backend_graph_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend")
        target_link_libraries(glimmer_execution_dnn_backend_graph_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_backend_graph_provider
            glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_graph_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backend_graph_test
            COMMAND glimmer_execution_dnn_backend_graph_test
                "$<TARGET_FILE:glimmer_execution_dnn_backend_graph_provider>")
        set_tests_properties(glimmer_execution_dnn_backend_graph_test PROPERTIES TIMEOUT 45)
        add_executable(glimmer_execution_dnn_backend_frontend_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_frontend_test.cc")
        target_include_directories(glimmer_execution_dnn_backend_frontend_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
        target_link_libraries(glimmer_execution_dnn_backend_frontend_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_backend_graph_provider
            glimmer_execution_dnn_provider Glimmer::cudnn_headers Glimmer::control
            Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_frontend_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backend_frontend_test
            COMMAND glimmer_execution_dnn_backend_frontend_test
                "$<TARGET_FILE:glimmer_execution_dnn_backend_graph_provider>")
        set_tests_properties(glimmer_execution_dnn_backend_frontend_test PROPERTIES TIMEOUT 45)
        add_executable(glimmer_execution_dnn_backend_heuristics_frontend_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_heuristics_frontend_test.cc")
        target_include_directories(glimmer_execution_dnn_backend_heuristics_frontend_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
        target_link_libraries(glimmer_execution_dnn_backend_heuristics_frontend_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_backend_graph_provider
            Glimmer::cudnn_headers Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_heuristics_frontend_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backend_heuristics_frontend_test
            COMMAND glimmer_execution_dnn_backend_heuristics_frontend_test
                "$<TARGET_FILE:glimmer_execution_dnn_backend_graph_provider>")
        set_tests_properties(glimmer_execution_dnn_backend_heuristics_frontend_test PROPERTIES TIMEOUT 45)
        add_executable(glimmer_execution_dnn_backend_properties_frontend_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_properties_frontend_test.cc")
        target_include_directories(glimmer_execution_dnn_backend_properties_frontend_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
        target_link_libraries(glimmer_execution_dnn_backend_properties_frontend_test PRIVATE
            glimmer_execution_interceptor glimmer_execution_dnn_backend_graph_provider
            glimmer_execution_dnn_provider Glimmer::cudnn_headers Glimmer::control
            Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_properties_frontend_test)
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backend_properties_frontend_test
            COMMAND glimmer_execution_dnn_backend_properties_frontend_test
                "$<TARGET_FILE:glimmer_execution_dnn_backend_graph_provider>")
        set_tests_properties(glimmer_execution_dnn_backend_properties_frontend_test PROPERTIES TIMEOUT 45)
        foreach(_backend_graph_mode profile priority missing-callback poison malformed-replies)
            string(REPLACE "-" "_" _backend_graph_test "${_backend_graph_mode}")
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_backend_graph_${_backend_graph_test}_test
                COMMAND glimmer_execution_dnn_backend_graph_test
                    "$<TARGET_FILE:glimmer_execution_dnn_backend_graph_provider>"
                    "--${_backend_graph_mode}")
            set_tests_properties(glimmer_execution_dnn_backend_graph_${_backend_graph_test}_test
                PROPERTIES TIMEOUT 45)
        endforeach()
        if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
            # Ordinary fixed input: bounded manual native/profile/sharing runs only.
            add_executable(glimmer_execution_workload_agnostic_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_workload_agnostic_gpu_test.cc")
            target_link_libraries(glimmer_execution_workload_agnostic_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_workload_agnostic_gpu_test)
            # Ordinary SDK graph input for bounded manual runs, never an automatic CTest.
            add_executable(glimmer_execution_dnn_backend_execution_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_execution_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_backend_execution_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_execution_gpu_test)
            # Explicit ordinary SDK metadata input; no context or compute calls.
            # Do not register vendor-dependent probes as automatic CPU CTests.
            add_executable(glimmer_execution_dnn_backend_metadata_sdk_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backend_metadata_sdk_test.cc")
            target_link_libraries(glimmer_execution_dnn_backend_metadata_sdk_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_backend_metadata_sdk_test)
            add_executable(glimmer_execution_dnn_attention_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attention_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_attention_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_attention_gpu_test)
            # Compile-only ordinary SDK inputs, never automatic GPU CTests.
            add_executable(glimmer_execution_dnn_attn_weights_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_attn_weights_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_attn_weights_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_attn_weights_gpu_test)
            add_executable(glimmer_execution_dnn_rnn_weight_params_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_weight_params_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_rnn_weight_params_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options ${CMAKE_DL_LIBS})
            glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_weight_params_gpu_test)
            add_executable(glimmer_execution_dnn_rnn_compute_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_compute_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_rnn_compute_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_compute_gpu_test)
            # Context-owned ordinary SDK input; never run automatically by CTest.
            add_executable(glimmer_execution_dnn_rnn_query_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_query_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_rnn_query_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_query_gpu_test)
            # Ordinary SDK input, compiled but never registered as an unguarded GPU CTest.
            add_executable(glimmer_execution_dnn_rnn_attention_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_rnn_attention_gpu_test.cc")
            target_include_directories(glimmer_execution_dnn_rnn_attention_gpu_test PRIVATE
                "${CUDAToolkit_INCLUDE_DIRS}")
            target_link_libraries(glimmer_execution_dnn_rnn_attention_gpu_test PRIVATE
                Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_rnn_attention_gpu_test)
            add_executable(glimmer_execution_dnn_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_gpu_test)
            add_executable(glimmer_execution_dnn_convolution_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_convolution_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_convolution_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_convolution_gpu_test)
            add_executable(glimmer_execution_dnn_pooling_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_pooling_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_pooling_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_pooling_gpu_test)
            add_executable(glimmer_execution_dnn_lrn_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_lrn_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_lrn_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_lrn_gpu_test)
            add_executable(glimmer_execution_dnn_reduction_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reduction_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_reduction_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_reduction_gpu_test)
            add_executable(glimmer_execution_dnn_batch_norm_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_batch_norm_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_batch_norm_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_batch_norm_gpu_test)
            add_executable(glimmer_execution_dnn_dropout_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_dropout_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_dropout_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_dropout_gpu_test)
            add_executable(glimmer_execution_dnn_op_tensor_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_op_tensor_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_op_tensor_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_op_tensor_gpu_test)
            add_executable(glimmer_execution_dnn_reorder_filter_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_reorder_filter_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_reorder_filter_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_reorder_filter_gpu_test)
            add_executable(glimmer_execution_dnn_library_info_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_library_info_gpu_test.cc")
            target_include_directories(glimmer_execution_dnn_library_info_gpu_test PRIVATE
                "${CUDAToolkit_INCLUDE_DIRS}")
            target_link_libraries(glimmer_execution_dnn_library_info_gpu_test PRIVATE
                Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_library_info_gpu_test)
            add_executable(glimmer_execution_dnn_backward_convolution_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_backward_convolution_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_backward_convolution_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_backward_convolution_gpu_test)
            add_executable(glimmer_execution_dnn_folded_convolution_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_folded_convolution_gpu_test.cc")
            target_link_libraries(glimmer_execution_dnn_folded_convolution_gpu_test PRIVATE
                CUDA::cuda_driver Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_folded_convolution_gpu_test)
            add_executable(glimmer_execution_dnn_sequence_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_sequence_gpu_test.cc")
            target_include_directories(glimmer_execution_dnn_sequence_gpu_test PRIVATE
                "${CUDAToolkit_INCLUDE_DIRS}")
            target_link_libraries(glimmer_execution_dnn_sequence_gpu_test PRIVATE
                Glimmer::cudnn Glimmer::project_options)
            glimmer_enable_clang_tidy(glimmer_execution_dnn_sequence_gpu_test)
        endif()
    endif()
    if(CUDAToolkit_NVCC_EXECUTABLE)
    set(glimmer_binary_cubin "${PROJECT_BINARY_DIR}/tests/binary-images/kernel.cubin")
    set(glimmer_binary_fatbin "${PROJECT_BINARY_DIR}/tests/binary-images/kernel.fatbin")
    set(glimmer_binary_raw_fatbin "${PROJECT_BINARY_DIR}/tests/binary-images/kernel-uncompressed.fatbin")
    add_custom_command(OUTPUT "${glimmer_binary_cubin}" "${glimmer_binary_fatbin}" "${glimmer_binary_raw_fatbin}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_BINARY_DIR}/tests/binary-images"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --cubin -arch=sm_70 -o "${glimmer_binary_cubin}"
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --fatbin -gencode arch=compute_70,code=sm_70
            -gencode arch=compute_70,code=compute_70 -o "${glimmer_binary_fatbin}"
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --fatbin --no-compress -gencode arch=compute_70,code=sm_70
            -gencode arch=compute_70,code=compute_70 -o "${glimmer_binary_raw_fatbin}"
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
        DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu" VERBATIM)
    add_custom_target(glimmer_execution_binary_images DEPENDS
        "${glimmer_binary_cubin}" "${glimmer_binary_fatbin}" "${glimmer_binary_raw_fatbin}")
    set_property(GLOBAL PROPERTY GLIMMER_BINARY_COMPILER_INPUTS
        "${glimmer_binary_cubin};${glimmer_binary_fatbin};${glimmer_binary_raw_fatbin}")
    endif()
    # No project_options: this app must not initialize ASan before its preload.
    add_executable(glimmer_execution_lookup_bootstrap_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lookup_bootstrap_test.cc")
    target_compile_features(glimmer_execution_lookup_bootstrap_test PRIVATE cxx_std_20)
    target_compile_options(glimmer_execution_lookup_bootstrap_test PRIVATE -Wall -Wextra -Wpedantic)
    if(GLIMMER_WARNINGS_AS_ERRORS)
        target_compile_options(glimmer_execution_lookup_bootstrap_test PRIVATE -Werror)
    endif()
    target_link_libraries(glimmer_execution_lookup_bootstrap_test PRIVATE "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_lookup_bootstrap_test)
    set(glimmer_execution_preload "$<TARGET_FILE:glimmer_execution_interceptor>")
    if(GLIMMER_ENABLE_SANITIZERS)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libasan.so
            RESULT_VARIABLE glimmer_execution_asan_result OUTPUT_VARIABLE glimmer_execution_asan
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT glimmer_execution_asan_result EQUAL 0 OR NOT IS_ABSOLUTE "${glimmer_execution_asan}"
           OR NOT EXISTS "${glimmer_execution_asan}")
            message(FATAL_ERROR "Execution lookup cold-bootstrap test requires the shared ASan runtime")
        endif()
        set(glimmer_execution_preload "${glimmer_execution_asan}:${glimmer_execution_preload}")
    endif()
    add_test(NAME glimmer_execution_lookup_bootstrap_test COMMAND glimmer_execution_lookup_bootstrap_test)
    set_tests_properties(glimmer_execution_lookup_bootstrap_test PROPERTIES
        ENVIRONMENT "LD_PRELOAD=${glimmer_execution_preload};ASAN_OPTIONS=detect_leaks=1:abort_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
        TIMEOUT 20)
    add_library(glimmer_execution_lookup_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lookup_driver.cc")
    set_target_properties(glimmer_execution_lookup_driver PROPERTIES OUTPUT_NAME "cuda.so.execution-lookup-fixture")
    target_include_directories(glimmer_execution_lookup_driver PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_lookup_driver PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    target_link_options(glimmer_execution_lookup_driver PRIVATE "LINKER:-Bsymbolic-functions")
    target_compile_options(glimmer_execution_lookup_driver PRIVATE -fno-optimize-sibling-calls)
    glimmer_enable_clang_tidy(glimmer_execution_lookup_driver)
    add_library(glimmer_execution_lookup_other SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lookup_other.cc")
    target_include_directories(glimmer_execution_lookup_other PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_lookup_other PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    target_compile_options(glimmer_execution_lookup_other PRIVATE -fno-optimize-sibling-calls)
    glimmer_enable_clang_tidy(glimmer_execution_lookup_other)
    add_library(glimmer_execution_lookup_runtime SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lookup_runtime.cc")
    set_target_properties(glimmer_execution_lookup_runtime PROPERTIES OUTPUT_NAME "cudart.so.execution-lookup-fixture")
    target_include_directories(glimmer_execution_lookup_runtime PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_lookup_runtime PRIVATE
        glimmer_execution_lookup_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
    target_link_options(glimmer_execution_lookup_runtime PRIVATE "LINKER:-Bsymbolic-functions")
    target_compile_options(glimmer_execution_lookup_runtime PRIVATE -fno-optimize-sibling-calls)
    glimmer_enable_clang_tidy(glimmer_execution_lookup_runtime)
    add_executable(glimmer_execution_lookup_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lookup_test.cc")
    target_include_directories(glimmer_execution_lookup_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_lookup_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_lookup_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_execution_lookup_test glimmer_execution_lookup_other)
    glimmer_enable_clang_tidy(glimmer_execution_lookup_test)
    add_test(NAME glimmer_execution_lookup_test COMMAND glimmer_execution_lookup_test
        $<TARGET_FILE:glimmer_execution_lookup_driver> $<TARGET_FILE:glimmer_execution_lookup_other>)
    set_tests_properties(glimmer_execution_lookup_test PROPERTIES TIMEOUT 30)
    add_library(glimmer_execution_initialization_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_initialization_driver.cc")
    set_target_properties(glimmer_execution_initialization_driver PROPERTIES OUTPUT_NAME "cuda.so.initialization-fixture")
    target_include_directories(glimmer_execution_initialization_driver PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_initialization_driver PRIVATE Glimmer::project_options Threads::Threads)
    target_link_options(glimmer_execution_initialization_driver PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_initialization_driver)
    add_executable(glimmer_execution_initialization_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_initialization_test.cc")
    target_include_directories(glimmer_execution_initialization_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_initialization_test PRIVATE glimmer_execution_interceptor
        Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_execution_initialization_test glimmer_execution_initialization_driver)
    glimmer_enable_clang_tidy(glimmer_execution_initialization_test)
    add_test(NAME glimmer_execution_initialization_test COMMAND glimmer_execution_initialization_test
        "$<TARGET_FILE:glimmer_execution_initialization_driver>")
    set_tests_properties(glimmer_execution_initialization_test PROPERTIES TIMEOUT 20)
    add_executable(glimmer_execution_runtime_lookup_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_lookup_test.cc")
    target_include_directories(glimmer_execution_runtime_lookup_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(glimmer_execution_runtime_lookup_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_lookup_runtime glimmer_execution_lookup_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_execution_runtime_lookup_test glimmer_execution_lookup_other)
    glimmer_enable_clang_tidy(glimmer_execution_runtime_lookup_test)
    add_test(NAME glimmer_execution_runtime_lookup_test COMMAND glimmer_execution_runtime_lookup_test
        $<TARGET_FILE:glimmer_execution_lookup_runtime> $<TARGET_FILE:glimmer_execution_lookup_other>)
    set_tests_properties(glimmer_execution_runtime_lookup_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_runtime_lookup_concurrent_test COMMAND glimmer_execution_runtime_lookup_test
        $<TARGET_FILE:glimmer_execution_lookup_runtime> $<TARGET_FILE:glimmer_execution_lookup_other> --concurrent-first)
    set_tests_properties(glimmer_execution_runtime_lookup_concurrent_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_primary_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_primary_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_lookup_driver Glimmer::control
            Glimmer::project_options
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_execution_runtime_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_lookup_driver Glimmer::control
            Glimmer::project_options
        TIMEOUT 20)
    add_executable(glimmer_execution_async_receipt_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_async_receipt_test.cc")
    target_include_directories(glimmer_execution_async_receipt_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_async_receipt_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_lookup_driver Glimmer::control Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_execution_async_receipt_test)
    foreach(mode IN ITEMS complete replacement error-complete error-original error-id error-address
            error-extent error-arguments error-data duplicate-sequence duplicate-frontier clean-rejection
            runtime-complete batch-missing batch-reordered batch-duplicate batch-sequence
            batch-partial-cache
            batch-retirement batch-status batch-envelope batch-query-lost batch-release-missing
            batch-release-reordered batch-release-lost batch-release-error batch-mixed batch-capacity
            batch-success-release busy-retry busy-wrong-operation busy-status busy-original
            busy-id busy-address busy-extent busy-arguments busy-data)
        add_test(NAME "glimmer_execution_async_receipt_${mode}"
            COMMAND glimmer_execution_async_receipt_test "${mode}")
        set_tests_properties("glimmer_execution_async_receipt_${mode}" PROPERTIES TIMEOUT 15)
    endforeach()
    if(TARGET glimmer_execution_service_runtime)
        glimmer_add_test_program(glimmer_execution_bootstrap_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_bootstrap_test.cc"
            INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
                "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend"
                "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            LIBRARIES glimmer_execution_interceptor glimmer_execution_lookup_driver
                glimmer_execution_service_runtime Glimmer::cuda_backend Glimmer::backend
                Glimmer::control Glimmer::project_options
            TIMEOUT 30)
        add_executable(glimmer_execution_async_launch_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_async_launch_test.cc")
        target_include_directories(glimmer_execution_async_launch_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend"
            "${PROJECT_SOURCE_DIR}/src/backend/cuda")
        target_link_libraries(glimmer_execution_async_launch_test PRIVATE glimmer_execution_interceptor
            glimmer_execution_lookup_driver glimmer_execution_service_runtime Glimmer::cuda_backend
            Glimmer::backend Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
        target_link_options(glimmer_execution_async_launch_test PRIVATE "-Wl,--wrap=send")
        set_target_properties(glimmer_execution_async_launch_test PROPERTIES ENABLE_EXPORTS ON)
        glimmer_enable_clang_tidy(glimmer_execution_async_launch_test)
        add_test(NAME glimmer_execution_async_launch_test COMMAND glimmer_execution_async_launch_test)
        set_tests_properties(glimmer_execution_async_launch_test PROPERTIES TIMEOUT 60)
    endif()
    glimmer_add_test_program(glimmer_execution_frontier_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_frontier_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor Glimmer::control Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_submission_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_submission_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor Glimmer::control Glimmer::project_options
        TIMEOUT 30)
    add_executable(glimmer_execution_frontend_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_frontend_test.cc")
    set_target_properties(glimmer_execution_frontend_test PROPERTIES ENABLE_EXPORTS ON)
    target_include_directories(glimmer_execution_frontend_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_frontend_test PRIVATE
        glimmer_execution_interceptor Glimmer::control Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_execution_frontend_test)
    add_test(NAME glimmer_execution_frontend_test COMMAND glimmer_execution_frontend_test)
    set_tests_properties(glimmer_execution_frontend_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_graph_frontend_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_graph_frontend_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor Glimmer::control Glimmer::project_options
            Threads::Threads)
    add_test(NAME glimmer_execution_graph_malformed_test
        COMMAND glimmer_execution_graph_frontend_test --malformed)
    add_test(NAME glimmer_execution_graph_partial_output_test
        COMMAND glimmer_execution_graph_frontend_test --partial-output)
    add_test(NAME glimmer_execution_graph_launch_failure_test
        COMMAND glimmer_execution_graph_frontend_test --launch-failure)
    add_test(NAME glimmer_execution_graph_malformed_quarantine_test
        COMMAND glimmer_execution_graph_frontend_test --malformed-quarantine)
    set_tests_properties(glimmer_execution_graph_frontend_test glimmer_execution_graph_malformed_test
        glimmer_execution_graph_partial_output_test glimmer_execution_graph_launch_failure_test
        glimmer_execution_graph_malformed_quarantine_test PROPERTIES TIMEOUT 30)
    add_executable(glimmer_execution_ready_reply_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_ready_reply_test.cc")
    target_include_directories(glimmer_execution_ready_reply_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_ready_reply_test PRIVATE
        glimmer_execution_interceptor Glimmer::control Glimmer::project_options)
    set_target_properties(glimmer_execution_ready_reply_test PROPERTIES ENABLE_EXPORTS ON)
    glimmer_enable_clang_tidy(glimmer_execution_ready_reply_test)
    add_test(NAME glimmer_execution_ready_reply_test COMMAND glimmer_execution_ready_reply_test)
    set_tests_properties(glimmer_execution_ready_reply_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_reply_readiness_interrupt_test
        COMMAND glimmer_execution_ready_reply_test --interrupt-wait)
    set_tests_properties(glimmer_execution_reply_readiness_interrupt_test PROPERTIES TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_async_status_progress_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_async_status_progress_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_lookup_driver Glimmer::control
            Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_preparation_config_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_preparation_config_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor Glimmer::control Glimmer::project_options
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_execution_event_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_event_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
            "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
            "${PROJECT_SOURCE_DIR}/src/backend"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_service_runtime
            Glimmer::cuda_backend Glimmer::backend Glimmer::control Glimmer::project_options
            "${CMAKE_DL_LIBS}"
        TIMEOUT 30)
    add_library(glimmer_execution_process_memory_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_process_memory_driver.cc")
    set_target_properties(glimmer_execution_process_memory_driver PROPERTIES
        OUTPUT_NAME "cuda.so.execution-process-memory-fixture")
    target_include_directories(glimmer_execution_process_memory_driver PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_process_memory_driver PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_process_memory_driver PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_process_memory_driver)
    add_executable(glimmer_execution_process_memory_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_process_memory_test.cc")
    target_include_directories(glimmer_execution_process_memory_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_process_memory_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_process_memory_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_process_memory_test)
    add_test(NAME glimmer_execution_process_memory_test COMMAND glimmer_execution_process_memory_test
        "$<TARGET_FILE:glimmer_execution_process_memory_driver>")
    set_tests_properties(glimmer_execution_process_memory_test PROPERTIES TIMEOUT 30)
    foreach(process_memory_mode IN ITEMS malformed native-output-error pool-output-error)
        add_test(NAME glimmer_execution_process_memory_${process_memory_mode}_test
            COMMAND glimmer_execution_process_memory_test
            "$<TARGET_FILE:glimmer_execution_process_memory_driver>" "--${process_memory_mode}")
        set_tests_properties(glimmer_execution_process_memory_${process_memory_mode}_test PROPERTIES TIMEOUT 30)
    endforeach()
    add_library(glimmer_execution_memory_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_memory_driver.cc")
    set_target_properties(glimmer_execution_memory_driver PROPERTIES OUTPUT_NAME "cuda.so.execution-memory-fixture")
    target_include_directories(glimmer_execution_memory_driver PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_memory_driver PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_memory_driver PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_memory_driver)
    add_executable(glimmer_execution_memory_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_memory_test.cc")
    target_include_directories(glimmer_execution_memory_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_memory_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_memory_test)
    add_test(NAME glimmer_execution_memory_test COMMAND glimmer_execution_memory_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>")
    set_tests_properties(glimmer_execution_memory_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_memory_endpoint_failure_test
        COMMAND glimmer_execution_memory_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --endpoint-failure)
    set_tests_properties(glimmer_execution_memory_endpoint_failure_test PROPERTIES TIMEOUT 10)
    add_executable(glimmer_execution_host_memory_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_memory_test.cc")
    target_include_directories(glimmer_execution_host_memory_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_host_memory_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_host_memory_test)
    add_test(NAME glimmer_execution_host_memory_test COMMAND glimmer_execution_host_memory_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>")
    set_tests_properties(glimmer_execution_host_memory_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_host_memory_recycling_test COMMAND glimmer_execution_host_memory_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --recycle)
    set_tests_properties(glimmer_execution_host_memory_recycling_test PROPERTIES TIMEOUT 180)
    add_executable(glimmer_execution_pageable_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pageable_test.cc")
    target_link_options(glimmer_execution_pageable_test PRIVATE "LINKER:--export-dynamic-symbol=poll")
    add_library(glimmer_execution_host_function_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_function_provider.cc")
    target_include_directories(glimmer_execution_host_function_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_host_function_provider PRIVATE Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_execution_host_function_provider)
    glimmer_add_test_program(glimmer_execution_host_function_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_function_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_host_function_provider
            glimmer_execution_memory_driver Glimmer::control Glimmer::project_options
            "${CMAKE_DL_LIBS}"
        TIMEOUT 30)
    add_test(NAME glimmer_execution_host_function_saturated_test
        COMMAND glimmer_execution_host_function_test --saturated)
    set_tests_properties(glimmer_execution_host_function_saturated_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_host_function_not_run_test
        COMMAND glimmer_execution_host_function_test --not-run)
    add_test(NAME glimmer_execution_host_function_bad_query_test
        COMMAND glimmer_execution_host_function_test --bad-query)
    set_tests_properties(glimmer_execution_host_function_not_run_test
        glimmer_execution_host_function_bad_query_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_host_function_primary_reset_test
        COMMAND glimmer_execution_host_function_test --primary-reset)
    set_tests_properties(glimmer_execution_host_function_primary_reset_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_host_function_per_thread_test
        COMMAND glimmer_execution_host_function_test --per-thread)
    set_tests_properties(glimmer_execution_host_function_per_thread_test PROPERTIES TIMEOUT 30)
    foreach(glimmer_host_option IN ITEMS saturated not-run bad-query primary-reset)
        string(REPLACE "-" "_" glimmer_host_suffix "${glimmer_host_option}")
        add_test(NAME glimmer_execution_host_function_per_thread_${glimmer_host_suffix}_test
            COMMAND glimmer_execution_host_function_test --${glimmer_host_option} --per-thread)
        set_tests_properties(glimmer_execution_host_function_per_thread_${glimmer_host_suffix}_test
            PROPERTIES TIMEOUT 30)
    endforeach()
    foreach(glimmer_host_route IN ITEMS null explicit legacy special)
        add_test(NAME glimmer_execution_host_function_ptsz_${glimmer_host_route}_test
            COMMAND glimmer_execution_host_function_test --ptsz-${glimmer_host_route})
        set_tests_properties(glimmer_execution_host_function_ptsz_${glimmer_host_route}_test
            PROPERTIES TIMEOUT 30)
        foreach(glimmer_host_option IN ITEMS saturated not-run bad-query primary-reset)
            string(REPLACE "-" "_" glimmer_host_suffix "${glimmer_host_option}")
            add_test(NAME glimmer_execution_host_function_ptsz_${glimmer_host_route}_${glimmer_host_suffix}_test
                COMMAND glimmer_execution_host_function_test --${glimmer_host_option}
                    --ptsz-${glimmer_host_route})
            set_tests_properties(glimmer_execution_host_function_ptsz_${glimmer_host_route}_${glimmer_host_suffix}_test
                PROPERTIES TIMEOUT 30)
        endforeach()
    endforeach()
    target_include_directories(glimmer_execution_pageable_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_pageable_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_pageable_test)
    add_test(NAME glimmer_execution_pageable_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>")
    set_tests_properties(glimmer_execution_pageable_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_staging_reuse_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --reuse)
    set_tests_properties(glimmer_execution_staging_reuse_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_staging_pressure_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --pressure)
    set_tests_properties(glimmer_execution_staging_pressure_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_staging_idle_error_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --idle-error)
    add_test(NAME glimmer_execution_staging_invalid_cache_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --invalid-cache)
    set_tests_properties(glimmer_execution_staging_idle_error_test
        glimmer_execution_staging_invalid_cache_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_pageable_copy_audit_test COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:glimmer_execution_pageable_test>"
        "-DDRIVER=$<TARGET_FILE:glimmer_execution_memory_driver>"
        -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_execution_copy_audit.cmake")
    set_tests_properties(glimmer_execution_pageable_copy_audit_test PROPERTIES TIMEOUT 30
        ENVIRONMENT "GLIMMER_EXECUTION_COPY_AUDIT=1")
    add_test(NAME glimmer_execution_pageable_quarantine_test COMMAND glimmer_execution_pageable_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" --quarantine)
    set_tests_properties(glimmer_execution_pageable_quarantine_test PROPERTIES TIMEOUT 30)
    add_library(glimmer_execution_blas_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_provider.cc")
    set_target_properties(glimmer_execution_blas_provider PROPERTIES OUTPUT_NAME "cublas.so.execution-fixture")
    target_include_directories(glimmer_execution_blas_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_blas_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_blas_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_blas_provider)
    add_executable(glimmer_execution_blas_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_test.cc")
    target_include_directories(glimmer_execution_blas_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_blas_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver glimmer_execution_blas_provider Glimmer::control
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_blas_test)
    add_test(NAME glimmer_execution_blas_test COMMAND glimmer_execution_blas_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>")
    add_test(NAME glimmer_execution_blas_quarantine_test COMMAND glimmer_execution_blas_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>" --quarantine)
    set_tests_properties(glimmer_execution_blas_test glimmer_execution_blas_quarantine_test PROPERTIES TIMEOUT 30)
    add_library(glimmer_execution_geam_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_geam_provider.cc")
    set_target_properties(glimmer_execution_geam_provider PROPERTIES
        OUTPUT_NAME "cublas.so.geam-fixture")
    target_include_directories(glimmer_execution_geam_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_geam_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_geam_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_geam_provider)
    add_executable(glimmer_execution_geam_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_geam_test.cc")
    target_include_directories(glimmer_execution_geam_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_geam_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver glimmer_execution_blas_provider glimmer_execution_geam_provider
        Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}" Threads::Threads)
    glimmer_enable_clang_tidy(glimmer_execution_geam_test)
    add_test(NAME glimmer_execution_geam_test COMMAND glimmer_execution_geam_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>"
        "$<TARGET_FILE:glimmer_execution_geam_provider>")
    add_test(NAME glimmer_execution_geam_receipt_test COMMAND glimmer_execution_geam_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>"
        "$<TARGET_FILE:glimmer_execution_geam_provider>" --malformed-replies)
    set_tests_properties(glimmer_execution_geam_test glimmer_execution_geam_receipt_test
        PROPERTIES TIMEOUT 45 ENVIRONMENT "LD_PRELOAD=${glimmer_execution_preload}")
    add_library(glimmer_execution_blas_extension_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_extension_provider.cc")
    set_target_properties(glimmer_execution_blas_extension_provider PROPERTIES
        OUTPUT_NAME "cublas.so.blas-extension-fixture")
    target_include_directories(glimmer_execution_blas_extension_provider PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_blas_extension_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_blas_extension_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_blas_extension_provider)
    add_executable(glimmer_execution_blas_extension_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_extension_test.cc")
    target_include_directories(glimmer_execution_blas_extension_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_blas_extension_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_memory_driver glimmer_execution_blas_provider
        glimmer_execution_blas_extension_provider Glimmer::control Glimmer::project_options
        "${CMAKE_DL_LIBS}" Threads::Threads)
    glimmer_enable_clang_tidy(glimmer_execution_blas_extension_test)
    add_test(NAME glimmer_execution_blas_extension_test COMMAND glimmer_execution_blas_extension_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>"
        "$<TARGET_FILE:glimmer_execution_blas_extension_provider>")
    add_test(NAME glimmer_execution_blas_extension_receipt_test COMMAND glimmer_execution_blas_extension_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>"
        "$<TARGET_FILE:glimmer_execution_blas_extension_provider>" --malformed-replies)
    add_test(NAME glimmer_execution_blas_extension_boundary_test COMMAND glimmer_execution_blas_extension_test
        "$<TARGET_FILE:glimmer_execution_memory_driver>" "$<TARGET_FILE:glimmer_execution_blas_provider>"
        "$<TARGET_FILE:glimmer_execution_blas_extension_provider>" --boundary)
    set_tests_properties(glimmer_execution_blas_extension_test glimmer_execution_blas_extension_receipt_test
        glimmer_execution_blas_extension_boundary_test
        PROPERTIES TIMEOUT 45 ENVIRONMENT "LD_PRELOAD=${glimmer_execution_preload}")
    add_library(glimmer_execution_lt_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_provider.cc")
    set_target_properties(glimmer_execution_lt_provider PROPERTIES OUTPUT_NAME "cublasLt.so.execution-fixture")
    target_include_directories(glimmer_execution_lt_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_lt_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_lt_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_lt_provider)
    target_link_libraries(glimmer_execution_blas_test PRIVATE glimmer_execution_lt_provider)
    add_library(glimmer_execution_random_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_random_provider.cc")
    set_target_properties(glimmer_execution_random_provider PROPERTIES OUTPUT_NAME "curand.so.execution-fixture")
    target_include_directories(glimmer_execution_random_provider PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_random_provider PRIVATE Glimmer::project_options)
    target_link_options(glimmer_execution_random_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_execution_random_provider)
    add_executable(glimmer_execution_lt_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_test.cc")
    target_include_directories(glimmer_execution_lt_test PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        "${PROJECT_SOURCE_DIR}/src/execution_service" "${PROJECT_SOURCE_DIR}/src/backend/cuda"
        "${PROJECT_SOURCE_DIR}/src/backend")
    target_link_libraries(glimmer_execution_lt_test PRIVATE glimmer_execution_interceptor
        glimmer_execution_lt_provider glimmer_execution_random_provider glimmer_execution_service_runtime Glimmer::cuda_backend
        Glimmer::backend Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_lt_test)
    glimmer_add_test_program(glimmer_execution_lt_capability_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_capability_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control"
        LIBRARIES glimmer_execution_interceptor glimmer_execution_lt_provider Glimmer::control
            Glimmer::project_options "${CMAKE_DL_LIBS}"
        TIMEOUT 45)
    add_test(NAME glimmer_execution_lt_metadata_retirement_test COMMAND glimmer_execution_lt_test
        $<TARGET_FILE:glimmer_execution_lt_provider> --metadata-retirement)
    set_tests_properties(glimmer_execution_lt_metadata_retirement_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_random_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_random_provider>" --random)
    add_test(NAME glimmer_execution_random_quarantine_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_random_provider>" --random-quarantine)
    set_tests_properties(glimmer_execution_random_test glimmer_execution_random_quarantine_test PROPERTIES TIMEOUT 30)
    add_dependencies(glimmer_execution_lt_test glimmer_execution_lookup_driver glimmer_execution_lookup_runtime)
    add_test(NAME glimmer_execution_runtime_metadata_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>" --runtime-metadata
        "$<TARGET_FILE:glimmer_execution_lookup_driver>" "$<TARGET_FILE:glimmer_execution_lookup_runtime>")
    set_tests_properties(glimmer_execution_runtime_metadata_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_context_control_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>" --context-control
        "$<TARGET_FILE:glimmer_execution_lookup_driver>" "$<TARGET_FILE:glimmer_execution_lookup_runtime>")
    set_tests_properties(glimmer_execution_context_control_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_lt_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>")
    add_test(NAME glimmer_execution_lt_quarantine_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>" --quarantine)
    add_test(NAME glimmer_execution_lt_chain_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>" --chain)
    set_tests_properties(glimmer_execution_lt_chain_test PROPERTIES TIMEOUT 30)
    add_test(NAME glimmer_execution_lt_blas_chain_test COMMAND glimmer_execution_lt_test
        "$<TARGET_FILE:glimmer_execution_lt_provider>" --blas-chain)
    set_tests_properties(glimmer_execution_lt_blas_chain_test PROPERTIES TIMEOUT 30)
    set_tests_properties(glimmer_execution_lt_test PROPERTIES TIMEOUT 90)
    set_tests_properties(glimmer_execution_lt_quarantine_test PROPERTIES TIMEOUT 30)
    # This SDK-only fixture enters CUDA only with an explicit output directory.
    # Its ordinary CTest entries check argument/oracle logic without a GPU.
    add_executable(glimmer_execution_lt_scalar_bias_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_scalar_bias_gpu_test.cc")
    target_link_libraries(glimmer_execution_lt_scalar_bias_gpu_test PRIVATE
        CUDA::cuda_driver CUDA::cublasLt CUDA::cublas Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_execution_lt_scalar_bias_gpu_test)
    add_test(NAME glimmer_execution_lt_scalar_bias_oracle_test
        COMMAND glimmer_execution_lt_scalar_bias_gpu_test --self-test)
    add_test(NAME glimmer_execution_lt_scalar_bias_arguments_test
        COMMAND glimmer_execution_lt_scalar_bias_gpu_test --help)
    add_test(NAME glimmer_execution_lt_scalar_bias_missing_arguments_test
        COMMAND glimmer_execution_lt_scalar_bias_gpu_test)
    add_test(NAME glimmer_execution_lt_scalar_bias_unknown_arguments_test
        COMMAND glimmer_execution_lt_scalar_bias_gpu_test --unknown)
    set_tests_properties(glimmer_execution_lt_scalar_bias_missing_arguments_test
        glimmer_execution_lt_scalar_bias_unknown_arguments_test PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(glimmer_execution_lt_scalar_bias_oracle_test
        glimmer_execution_lt_scalar_bias_arguments_test
        glimmer_execution_lt_scalar_bias_missing_arguments_test
        glimmer_execution_lt_scalar_bias_unknown_arguments_test PROPERTIES TIMEOUT 30)
    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        add_executable(glimmer_execution_context_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_context_gpu_test.cc")
        target_link_libraries(glimmer_execution_context_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cudart Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_context_gpu_test)
        add_executable(glimmer_execution_lt_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_gpu_test.cc")
        target_include_directories(glimmer_execution_lt_gpu_test PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(glimmer_execution_lt_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublasLt CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_lt_gpu_test)
        add_executable(glimmer_execution_lt_priority_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_lt_priority_gpu_test.cc")
        target_link_libraries(glimmer_execution_lt_priority_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublasLt CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_lt_priority_gpu_test)
        add_executable(glimmer_execution_sgemm_priority_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_sgemm_priority_gpu_test.cc")
        target_link_libraries(glimmer_execution_sgemm_priority_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_sgemm_priority_gpu_test)
        add_executable(glimmer_execution_gemm_ex_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_gemm_ex_gpu_test.cc")
        target_link_libraries(glimmer_execution_gemm_ex_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_gemm_ex_gpu_test)
        add_executable(glimmer_execution_bert_gemm_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_bert_gemm_gpu_test.cc")
        target_link_libraries(glimmer_execution_bert_gemm_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_bert_gemm_gpu_test)
        add_executable(glimmer_execution_gemm_ex_priority_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_gemm_ex_priority_gpu_test.cc")
        target_link_libraries(glimmer_execution_gemm_ex_priority_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_gemm_ex_priority_gpu_test)
        add_executable(glimmer_execution_random_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_random_gpu_test.cc")
        target_link_libraries(glimmer_execution_random_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::curand Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_random_gpu_test)
        add_executable(glimmer_execution_blas_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_gpu_test.cc")
        target_link_libraries(glimmer_execution_blas_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_blas_gpu_test)
        add_executable(glimmer_execution_geam_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_geam_gpu_test.cc")
        target_link_libraries(glimmer_execution_geam_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_geam_gpu_test)
        add_executable(glimmer_execution_blas_extension_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_extension_gpu_test.cc")
        target_link_libraries(glimmer_execution_blas_extension_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_blas_extension_gpu_test)
        add_executable(glimmer_execution_blas_boundary_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_blas_boundary_gpu_test.cc")
        target_link_libraries(glimmer_execution_blas_boundary_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cublas Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_blas_boundary_gpu_test)
        add_executable(glimmer_execution_unified_copy_probe
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_unified_copy_probe.cc")
        target_link_libraries(glimmer_execution_unified_copy_probe PRIVATE
            CUDA::cuda_driver Threads::Threads Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_unified_copy_probe)
        add_executable(glimmer_execution_unified_copy_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_unified_copy_gpu_test.cc")
        target_link_libraries(glimmer_execution_unified_copy_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_unified_copy_gpu_test)
        add_executable(glimmer_execution_h2h_copy_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_h2h_copy_gpu_test.cc")
        target_link_libraries(glimmer_execution_h2h_copy_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cudart Threads::Threads Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_h2h_copy_gpu_test)
        add_executable(glimmer_execution_host_memory_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_memory_gpu_test.cc")
        target_link_libraries(glimmer_execution_host_memory_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cudart Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_host_memory_gpu_test)
        if(CUDAToolkit_NVCC_EXECUTABLE)
            set(glimmer_async_launch_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_async_launch_gpu_test")
            add_custom_command(OUTPUT "${glimmer_async_launch_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_async_launch_gpu_test.cu"
                    -ldl -o "${glimmer_async_launch_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_async_launch_gpu_test.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_async_launch_application
                DEPENDS "${glimmer_async_launch_application}")
            set(glimmer_host_function_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_host_function_gpu_test")
            add_custom_command(OUTPUT "${glimmer_host_function_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_function_gpu_test.cu"
                    -ldl -o "${glimmer_host_function_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_host_function_gpu_test.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_host_function_application
                DEPENDS "${glimmer_host_function_application}")
            set(glimmer_shared_lifecycle_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_shared_lifecycle_gpu_test")
            add_custom_command(OUTPUT "${glimmer_shared_lifecycle_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_shared_lifecycle_gpu_test.cu"
                    -lcuda -o "${glimmer_shared_lifecycle_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_shared_lifecycle_gpu_test.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_shared_lifecycle_application
                DEPENDS "${glimmer_shared_lifecycle_application}")
            set(glimmer_runtime_function_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_runtime_function_gpu_test")
            add_custom_command(OUTPUT "${glimmer_runtime_function_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared
                    -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_function_gpu_test.cu"
                    -o "${glimmer_runtime_function_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_function_gpu_test.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_runtime_function_application
                DEPENDS "${glimmer_runtime_function_application}")
            set(glimmer_pointer_metadata_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_pointer_metadata_gpu_test")
            add_custom_command(OUTPUT "${glimmer_pointer_metadata_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pointer_metadata_gpu_test.cu"
                    -lcuda -o "${glimmer_pointer_metadata_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_pointer_metadata_gpu_test.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_pointer_metadata_application
                DEPENDS "${glimmer_pointer_metadata_application}")
            set(glimmer_runtime_application "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_runtime_gpu_test")
            add_custom_command(OUTPUT "${glimmer_runtime_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared
                    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_gpu_test.cu"
                    -o "${glimmer_runtime_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_gpu_test.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cached_runtime_fixture.h"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_runtime_application ALL DEPENDS "${glimmer_runtime_application}")
            set(glimmer_graph_application "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_graph_sdk_test")
            add_custom_command(OUTPUT "${glimmer_graph_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_graph_sdk_test.cu"
                    -lcuda -ldl -o "${glimmer_graph_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_graph_sdk_test.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cached_runtime_fixture.h"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_graph_sdk_application ALL
                DEPENDS "${glimmer_graph_application}")
            set(glimmer_context_sharing_application
                "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_context_sharing_gpu_test")
            add_custom_command(OUTPUT "${glimmer_context_sharing_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++20 --cudart shared
                    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_context_sharing_gpu_test.cu"
                    -lcuda -ldl -o "${glimmer_context_sharing_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_context_sharing_gpu_test.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cached_driver_fixture.h"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cached_runtime_fixture.h"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_context_sharing_application ALL
                DEPENDS "${glimmer_context_sharing_application}")
            add_test(NAME glimmer_execution_context_sharing_help
                COMMAND "${glimmer_context_sharing_application}" --help)
            set_tests_properties(glimmer_execution_context_sharing_help PROPERTIES TIMEOUT 10)
            set(glimmer_cycle_witness_image "${CMAKE_CURRENT_BINARY_DIR}/cycle_witness.cubin")
            add_custom_command(OUTPUT "${glimmer_cycle_witness_image}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cubin -arch=sm_70
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cycle_witness_kernel.cu"
                    -o "${glimmer_cycle_witness_image}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cycle_witness_kernel.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cycle_witness_fixture.h"
                VERBATIM)
            add_custom_target(glimmer_execution_cycle_witness_image DEPENDS "${glimmer_cycle_witness_image}")
            add_executable(glimmer_execution_cycle_witness_gpu_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cycle_witness_gpu_test.cc")
            target_link_libraries(glimmer_execution_cycle_witness_gpu_test PRIVATE
                CUDA::cuda_driver CUDA::cudart ${CMAKE_DL_LIBS} Glimmer::project_options)
            add_dependencies(glimmer_execution_cycle_witness_gpu_test glimmer_execution_cycle_witness_image)
            glimmer_enable_clang_tidy(glimmer_execution_cycle_witness_gpu_test)
            add_test(NAME glimmer_execution_cycle_witness_help
                COMMAND glimmer_execution_cycle_witness_gpu_test --help)
            set_tests_properties(glimmer_execution_cycle_witness_help PROPERTIES TIMEOUT 10)
            set(glimmer_event_application "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_event_gpu_test")
            set(glimmer_event_loader_options)
            foreach(glimmer_event_loader_library IN LISTS CMAKE_DL_LIBS)
                list(APPEND glimmer_event_loader_options "-l${glimmer_event_loader_library}")
            endforeach()
            add_custom_command(OUTPUT "${glimmer_event_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared
                    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_event_gpu_test.cu"
                    ${glimmer_event_loader_options}
                    -o "${glimmer_event_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_event_gpu_test.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_event_application ALL DEPENDS "${glimmer_event_application}")
            set(glimmer_linked_runtime_application "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_runtime_linked_gpu_test")
            add_custom_command(OUTPUT "${glimmer_linked_runtime_application}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared -rdc=true
                    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                    "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                    -Xcompiler=-Wall,-Wextra,-Werror
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_gpu_test.cu"
                    -o "${glimmer_linked_runtime_application}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_runtime_gpu_test.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_cached_runtime_fixture.h"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_binary_kernel_fixture.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_runtime_linked_application ALL
                DEPENDS "${glimmer_linked_runtime_application}")
            foreach(glimmer_symbol_linkage IN ITEMS direct linked)
                set(glimmer_symbol_application "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_symbol_${glimmer_symbol_linkage}_gpu_test")
                set(glimmer_symbol_rdc false)
                set(glimmer_symbol_rdc_definition 0)
                if(glimmer_symbol_linkage STREQUAL "linked")
                    set(glimmer_symbol_rdc true)
                    set(glimmer_symbol_rdc_definition 1)
                endif()
                add_custom_command(OUTPUT "${glimmer_symbol_application}"
                    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared
                        "-rdc=${glimmer_symbol_rdc}"
                        "-DGLIMMER_SYMBOL_RDC=${glimmer_symbol_rdc_definition}"
                        -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
                        "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Release>:-O3>"
                        -Xcompiler=-Wall,-Wextra,-Werror
                        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_gpu_test.cu"
                        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_coalesced_kernel_second.cu"
                        -o "${glimmer_symbol_application}"
                    DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_gpu_test.cu"
                        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_coalesced_kernel_second.cu"
                        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_coalesced_kernel_fixture.cuh"
                    COMMAND_EXPAND_LISTS VERBATIM)
                add_custom_target("glimmer_execution_symbol_${glimmer_symbol_linkage}_application" ALL
                    DEPENDS "${glimmer_symbol_application}")
            endforeach()
            set(glimmer_symbol_alias_probe "${CMAKE_CURRENT_BINARY_DIR}/glimmer_execution_symbol_alias_probe")
            add_custom_command(OUTPUT "${glimmer_symbol_alias_probe}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --std=c++17 --cudart shared
                    -gencode arch=compute_70,code=sm_70
                    -Xcompiler=-Wall,-Wextra,-Werror -ldl
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_alias_probe.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_alias_second.cu"
                    -o "${glimmer_symbol_alias_probe}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_alias_probe.cu"
                    "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_symbol_alias_second.cu"
                COMMAND_EXPAND_LISTS VERBATIM)
            add_custom_target(glimmer_execution_symbol_alias_probe_application ALL
                DEPENDS "${glimmer_symbol_alias_probe}")
        endif()
        add_executable(glimmer_execution_priority_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_priority_gpu_test.cc")
        target_link_libraries(glimmer_execution_priority_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_priority_gpu_test)
        # The same ordinary Driver binary is used natively and under preload.
        # It has no Glimmer headers, scheduler calls or IPC/test adapter.
        add_executable(glimmer_execution_frontend_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_frontend_gpu_test.cc")
        target_link_libraries(glimmer_execution_frontend_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_frontend_gpu_test)
        add_executable(glimmer_execution_submission_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_submission_gpu_test.cc")
        target_link_libraries(glimmer_execution_submission_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_execution_submission_gpu_test)
        add_executable(glimmer_execution_memory_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_memory_gpu_test.cc")
        target_link_libraries(glimmer_execution_memory_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cudart Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_memory_gpu_test)
        add_executable(glimmer_execution_process_memory_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_process_memory_gpu_test.cc")
        target_link_libraries(glimmer_execution_process_memory_gpu_test PRIVATE
            CUDA::cuda_driver CUDA::cudart Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_process_memory_gpu_test)
        add_executable(glimmer_execution_function_control_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_function_control_gpu_test.cc")
        target_link_libraries(glimmer_execution_function_control_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_function_control_gpu_test)
        add_executable(glimmer_execution_module_factory_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_module_factory_gpu_test.cc")
        target_link_libraries(glimmer_execution_module_factory_gpu_test PRIVATE
            CUDA::cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
        add_dependencies(glimmer_execution_module_factory_gpu_test glimmer_execution_binary_images)
        glimmer_enable_clang_tidy(glimmer_execution_module_factory_gpu_test)
    endif()
endif()

# Production trusts only configured SDK ELFs. CPU fixtures use a separately
# compiled frontend with an exact test-only manifest, never a production bypass.
if(BUILD_TESTING AND TARGET Glimmer::cudnn_headers)
    # Manually supervised SDK input, never part of the CPU-only automatic suite.
    if(TARGET Glimmer::cudnn)
        add_executable(glimmer_execution_dnn_profile_sdk_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_profile_sdk_test.cc")
        target_link_libraries(glimmer_execution_dnn_profile_sdk_test PRIVATE
            Glimmer::cudnn Glimmer::project_options "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_profile_sdk_test)
    endif()
    foreach(_profile_variant IN ITEMS approved wrong)
        add_library(glimmer_dnn_profile_${_profile_variant}_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_profile_provider.cc")
        set_target_properties(glimmer_dnn_profile_${_profile_variant}_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.profile-${_profile_variant}-fixture")
        target_link_libraries(glimmer_dnn_profile_${_profile_variant}_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_link_options(glimmer_dnn_profile_${_profile_variant}_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_dnn_profile_${_profile_variant}_provider)
    endforeach()
    # Same SONAME/source ABI at an unapproved root must still fail identity proof.
    set_target_properties(glimmer_dnn_profile_wrong_provider PROPERTIES
        OUTPUT_NAME "cudnn.so.profile-approved-fixture"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/unapproved-dnn-profile")

    # A second loader namespace cannot host a second ASan runtime. Only this
    # inert SDK stub omits sanitizer linkage; the scanner, frontend, socket and
    # executable under test remain fully instrumented in sanitizer builds.
    add_library(glimmer_dnn_profile_namespace_provider SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_profile_provider.cc")
    set_target_properties(glimmer_dnn_profile_namespace_provider PROPERTIES
        OUTPUT_NAME "cudnn.so.profile-namespace-fixture")
    target_link_libraries(glimmer_dnn_profile_namespace_provider PRIVATE Glimmer::cudnn_headers)
    target_compile_features(glimmer_dnn_profile_namespace_provider PRIVATE cxx_std_20)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(glimmer_dnn_profile_namespace_provider PRIVATE -Wall -Wextra -Wpedantic)
        if(GLIMMER_WARNINGS_AS_ERRORS)
            target_compile_options(glimmer_dnn_profile_namespace_provider PRIVATE -Werror)
        endif()
    endif()
    target_link_options(glimmer_dnn_profile_namespace_provider PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_dnn_profile_namespace_provider)

    if(GLIMMER_CUDNN_ABI_PROFILE STREQUAL "8")
        if(TARGET Glimmer::cudnn)
            # Ordinary SDK input: supervised native-first GPU correctness only,
            # deliberately not registered in the automatic CPU CTest suite.
            add_executable(glimmer_execution_dnn_legacy_sdk_test
                "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_legacy_sdk_test.cc")
            target_link_libraries(glimmer_execution_dnn_legacy_sdk_test PRIVATE
                Glimmer::cudnn CUDA::cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
            glimmer_enable_clang_tidy(glimmer_execution_dnn_legacy_sdk_test)
        endif()
        add_library(glimmer_execution_dnn_legacy_provider SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_legacy_provider.cc")
        set_target_properties(glimmer_execution_dnn_legacy_provider PROPERTIES
            OUTPUT_NAME "cudnn.so.legacy-execution-fixture")
        target_link_libraries(glimmer_execution_dnn_legacy_provider PRIVATE
            Glimmer::cudnn_headers Glimmer::project_options)
        target_link_options(glimmer_execution_dnn_legacy_provider PRIVATE
            "LINKER:-Bsymbolic-functions")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_legacy_provider)
    endif()
    set(_dnn_test_providers glimmer_dnn_profile_approved_provider glimmer_dnn_profile_namespace_provider)
    if(TARGET glimmer_execution_dnn_legacy_provider)
        list(APPEND _dnn_test_providers glimmer_execution_dnn_legacy_provider)
    endif()
    get_property(_execution_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    if(GLIMMER_CUDNN_ABI_PROFILE STREQUAL "9")
        foreach(_execution_target IN LISTS _execution_targets)
            if(_execution_target MATCHES "^glimmer_execution_dnn_.*provider$")
                list(APPEND _dnn_test_providers "${_execution_target}")
            endif()
        endforeach()
    endif()
    set(_dnn_provider_inputs "")
    foreach(_dnn_provider IN LISTS _dnn_test_providers)
        list(APPEND _dnn_provider_inputs "$<TARGET_FILE:${_dnn_provider}>")
    endforeach()
    list(JOIN _dnn_provider_inputs "|" _dnn_provider_inputs)
    set(_dnn_test_identity_dir "${PROJECT_BINARY_DIR}/generated/cudnn-test")
    set(_dnn_test_identity_header "${_dnn_test_identity_dir}/cudnn_profile_providers.h")
    add_custom_command(OUTPUT "${_dnn_test_identity_header}"
        COMMAND "${CMAKE_COMMAND}"
            "-DGLIMMER_PROFILE_OUTPUT=${_dnn_test_identity_header}"
            "-DGLIMMER_PROFILE_INPUTS=${_dnn_provider_inputs}"
            "-DGLIMMER_PROFILE_READELF=${CMAKE_READELF}"
            -P "${PROJECT_SOURCE_DIR}/cmake/CudnnProfileProviders.cmake"
        DEPENDS ${_dnn_test_providers} "${PROJECT_SOURCE_DIR}/cmake/CudnnProfileProviders.cmake"
        VERBATIM)
    add_library(glimmer_execution_dnn_test_interceptor SHARED
        driver_frontend.cc driver_lookup.cc "${_dnn_test_identity_header}")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        target_sources(glimmer_execution_dnn_test_interceptor PRIVATE ../dlsym_entry_x86_64.S)
    endif()
    target_include_directories(glimmer_execution_dnn_test_interceptor BEFORE PRIVATE
        "${_dnn_test_identity_dir}" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_execution_dnn_test_interceptor PRIVATE
        Glimmer::control Glimmer::transform Glimmer::project_options Glimmer::cudnn_headers
        Threads::Threads "${CMAKE_DL_LIBS}")
    if(GLIMMER_ENABLE_NCCL_EXECUTION)
        target_link_libraries(glimmer_execution_dnn_test_interceptor PRIVATE Glimmer::nccl_headers)
    endif()
    target_link_options(glimmer_execution_dnn_test_interceptor PRIVATE
        "LINKER:-Bsymbolic-functions"
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    set_property(TARGET glimmer_execution_dnn_test_interceptor APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
    glimmer_enable_clang_tidy(glimmer_execution_dnn_test_interceptor)

    if(TARGET glimmer_execution_dnn_legacy_provider)
        add_executable(glimmer_execution_dnn_legacy_frontend_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_legacy_frontend_test.cc")
        target_include_directories(glimmer_execution_dnn_legacy_frontend_test PRIVATE
            "${CUDAToolkit_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/src/control")
        target_link_libraries(glimmer_execution_dnn_legacy_frontend_test PRIVATE
            glimmer_execution_dnn_test_interceptor glimmer_execution_dnn_legacy_provider
            Glimmer::control Glimmer::cudnn_headers Glimmer::project_options
            Threads::Threads "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_execution_dnn_legacy_frontend_test)
        foreach(_legacy_case RANGE 0 13)
            glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_legacy_${_legacy_case}_test
                COMMAND glimmer_execution_dnn_legacy_frontend_test
                    "$<TARGET_FILE:glimmer_execution_dnn_legacy_provider>" --case "${_legacy_case}")
            set_tests_properties(glimmer_execution_dnn_legacy_${_legacy_case}_test
                PROPERTIES TIMEOUT 20)
        endforeach()
    endif()

    # Preserve every existing oracle; substitute only the qualified CPU loader.
    foreach(_execution_target IN LISTS _execution_targets)
        if(_execution_target MATCHES "^glimmer_execution_dnn_")
            get_target_property(_dnn_links "${_execution_target}" LINK_LIBRARIES)
            if(_dnn_links)
                list(TRANSFORM _dnn_links REPLACE "^glimmer_execution_interceptor$"
                    "glimmer_execution_dnn_test_interceptor")
                set_property(TARGET "${_execution_target}" PROPERTY LINK_LIBRARIES "${_dnn_links}")
            endif()
        endif()
    endforeach()
    add_executable(glimmer_execution_dnn_profile_lookup_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/execution_dnn_profile_lookup_test.cc")
    target_include_directories(glimmer_execution_dnn_profile_lookup_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/control")
    target_link_libraries(glimmer_execution_dnn_profile_lookup_test PRIVATE
        glimmer_execution_dnn_test_interceptor Glimmer::control Glimmer::cudnn_headers
        Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_execution_dnn_profile_lookup_test)
    add_dependencies(glimmer_execution_dnn_profile_lookup_test glimmer_dnn_profile_wrong_provider)
    foreach(_profile_case IN ITEMS late-provider new-namespace binding-receipts)
        set(_profile_arguments)
        set(_profile_approved glimmer_dnn_profile_approved_provider)
        if(_profile_case STREQUAL "new-namespace")
            list(APPEND _profile_arguments --new-namespace)
            set(_profile_approved glimmer_dnn_profile_namespace_provider)
        elseif(_profile_case STREQUAL "binding-receipts")
            list(APPEND _profile_arguments --binding-receipts)
        endif()
        glimmer_add_dnn_fixture_test(NAME glimmer_execution_dnn_profile_${_profile_case}_test
            COMMAND glimmer_execution_dnn_profile_lookup_test
                "$<TARGET_FILE:${_profile_approved}>"
                "$<TARGET_FILE:glimmer_dnn_profile_wrong_provider>" ${_profile_arguments})
        set_tests_properties(glimmer_execution_dnn_profile_${_profile_case}_test PROPERTIES TIMEOUT 15)
    endforeach()
endif()
