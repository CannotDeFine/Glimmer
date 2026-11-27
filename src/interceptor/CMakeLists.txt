add_library(glimmer_cuda_interceptor SHARED
    internal/allocation_registry.cc
    internal/allocation_registry.h
    internal/diagnostics.cc
    internal/diagnostics.h
    internal/loaded_image_path.cc
    internal/loaded_image_path.h

    internal/ptx_transform.h
    internal/embedded_ptx.cc
    internal/embedded_ptx.h
    internal/persistent_kernel.cc
    internal/persistent_kernel.h
    internal/persistent_sgemm.cc
    internal/persistent_sgemm.h
    internal/cuda_operation_queue.cc
    internal/cuda_operation_queue.h
    internal/cuda_binary_slice.cc
    internal/cuda_gemm_allocation.cc
    internal/cuda_gemm_allocation.h
    internal/cuda_binary_slice.h
    internal/qualified_cubin.cc
    internal/qualified_cubin.h
    internal/qualified_cubin_source.cc
    internal/qualified_cubin_source.h
    internal/cuda_queue_worker.cc
    internal/cuda_queue_worker.h
    internal/owner_control_pump.h
    internal/cuda_admission_coordinator.cc
    internal/cuda_admission_coordinator.h
    internal/cuda_cooperative_pool.cc
    internal/cuda_cooperative_pool.h
    internal/cuda_cooperative_frontend.cc
    internal/cuda_native_execution_epoch.cc
    internal/native_execution_epoch.cc
    internal/cuda_cooperative_frontend.h
    internal/native_execution_heartbeat.cc
    internal/native_execution_heartbeat.h
    internal/driver_cooperative_router.cc
    internal/driver_cooperative_router.h
    internal/cuda_process_native_session.cc
    internal/cuda_process_native_session.h
    internal/cuda_capture_origin_registry.cc
    internal/cuda_capture_origin_registry.h
    internal/cuda_source_capture.h
    internal/cuda_paused_native_executor.cc
    internal/cuda_paused_native_executor.h
    internal/cuda_event_projection_registry.cc
    internal/cuda_event_projection_registry.h
    internal/cuda_deferred_event_binding.cc
    internal/cuda_deferred_event_binding.h
    internal/cuda_native_graph_effects.cc
    internal/cuda_native_graph_effects.h
    internal/cuda_native_graph_registry.cc
    internal/cuda_native_graph_registry.h
    internal/cuda_copy_range.cc
    internal/cuda_copy_range.h
    internal/process_call_gate.cc
    internal/process_call_gate.h
    internal/operation_order.cc
    internal/operation_order.h
    internal/owned_kernel_arguments.cc
    internal/owned_kernel_arguments.h
    driver_dispatch.cc
    internal/driver_dispatch.h
    nvml_dispatch.cc
    internal/nvml_dispatch.h
    driver_api_interceptor.cc
    driver_api_launch.cc
    driver_api_memory.cc
    driver_api_runtime_memory.cc
    internal/driver_api_interceptor.h
    internal/driver_api_interceptor_state.h
    driver_api_wrappers.cc
    blas_api_interceptor.cc
    driver_copy_wrappers.cc
    driver_graph_wrappers.cc
    internal/driver_graph_wrappers.h
    internal/driver_copy_wrappers.h
    launch_completion_tracker.cc
    internal/launch_completion_tracker.h
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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    enable_language(ASM)
    target_sources(glimmer_cuda_interceptor PRIVATE dlsym_entry_x86_64.S)
    target_compile_definitions(glimmer_cuda_interceptor PRIVATE GLIMMER_DLSYM_NATIVE_TRAMPOLINE)
else()
    message(STATUS "Caller-relative native dlsym trampoline is qualified only for Linux x86-64")
endif()

target_include_directories(glimmer_cuda_interceptor
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/interceptor"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(glimmer_cuda_interceptor
    PRIVATE Glimmer::transform
        Glimmer::control
        Glimmer::project_options
        "${CMAKE_DL_LIBS}"
        Threads::Threads
)

glimmer_enable_clang_tidy(glimmer_cuda_interceptor)
target_link_options(glimmer_cuda_interceptor PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runtime_graph_versions.map")
set_property(TARGET glimmer_cuda_interceptor APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime_graph_versions.map")

find_program(GLIMMER_CUOBJDUMP_EXECUTABLE cuobjdump HINTS "${CUDAToolkit_BIN_DIR}")
if(NOT GLIMMER_CUOBJDUMP_EXECUTABLE)
    set(GLIMMER_CUOBJDUMP_EXECUTABLE "")
endif()
target_compile_definitions(glimmer_cuda_interceptor PRIVATE
    GLIMMER_CUOBJDUMP_PATH="${GLIMMER_CUOBJDUMP_EXECUTABLE}")
if(GLIMMER_ENABLE_SANITIZERS)
    target_compile_definitions(glimmer_cuda_interceptor PRIVATE GLIMMER_SANITIZED_INTERCEPTOR)
endif()

if(BUILD_TESTING)
    glimmer_add_test_program(glimmer_persistent_sgemm_test
        SOURCES internal/persistent_sgemm.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_sgemm_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARIES Glimmer::project_options)
    glimmer_add_test_program(glimmer_cuda_native_execution_epoch_test
        SOURCES internal/native_execution_epoch.cc internal/cuda_native_execution_epoch.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_native_execution_epoch_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    glimmer_add_test_program(glimmer_cuda_native_graph_registry_test
        SOURCES internal/cuda_native_graph_effects.cc internal/cuda_native_graph_registry.cc
            internal/cuda_event_projection_registry.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_native_graph_registry_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_deferred_event_binding_test
        SOURCES internal/cuda_deferred_event_binding.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_deferred_event_binding_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_native_graph_effects_test
        SOURCES internal/cuda_native_graph_effects.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_native_graph_effects_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_event_projection_registry_test
        SOURCES internal/cuda_event_projection_registry.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_event_projection_registry_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_copy_range_test
        SOURCES internal/cuda_copy_range.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_copy_range_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options)
    # This ordinary application must not acquire an ASan preinit_array from
    # project_options: the injected library must survive cold sanitizer startup.
    add_executable(glimmer_cuda_loader_bootstrap_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/loader_bootstrap_test.cc")
    target_compile_features(glimmer_cuda_loader_bootstrap_test PRIVATE cxx_std_20)
    target_compile_options(glimmer_cuda_loader_bootstrap_test PRIVATE -Wall -Wextra -Wpedantic)
    if(GLIMMER_WARNINGS_AS_ERRORS)
        target_compile_options(glimmer_cuda_loader_bootstrap_test PRIVATE -Werror)
    endif()
    target_link_libraries(glimmer_cuda_loader_bootstrap_test PRIVATE "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_loader_bootstrap_test)
    set(glimmer_bootstrap_preload "$<TARGET_FILE:glimmer_cuda_interceptor>")
    set(glimmer_can_test_bootstrap TRUE)
    if(GLIMMER_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libasan.so
                RESULT_VARIABLE glimmer_asan_lookup_result
                OUTPUT_VARIABLE glimmer_asan_library OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(glimmer_asan_lookup_result EQUAL 0 AND IS_ABSOLUTE "${glimmer_asan_library}"
                AND EXISTS "${glimmer_asan_library}")
                set(glimmer_bootstrap_preload "${glimmer_asan_library}:${glimmer_bootstrap_preload}")
            else()
                set(glimmer_can_test_bootstrap FALSE)
                message(WARNING "Cold loader bootstrap test unavailable: shared libasan not found")
            endif()
        else()
            set(glimmer_can_test_bootstrap FALSE)
            message(WARNING "Cold loader bootstrap sanitizer test requires a discovered shared ASan runtime")
        endif()
    endif()
    if(glimmer_can_test_bootstrap)
        add_test(NAME glimmer_cuda_loader_bootstrap_test COMMAND glimmer_cuda_loader_bootstrap_test)
        set_tests_properties(glimmer_cuda_loader_bootstrap_test PROPERTIES
            ENVIRONMENT "LD_PRELOAD=${glimmer_bootstrap_preload};ASAN_OPTIONS=detect_leaks=1"
            TIMEOUT 20)
    endif()
    foreach(provider IN ITEMS first second)
        add_library(glimmer_driver_proxy_${provider} SHARED
            "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_proxy_fixture.cc")
        target_include_directories(glimmer_driver_proxy_${provider} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(glimmer_driver_proxy_${provider} PRIVATE Glimmer::project_options)
        set_target_properties(glimmer_driver_proxy_${provider} PROPERTIES
            OUTPUT_NAME glimmer_driver_proxy
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/driver-proxy-${provider}")
        glimmer_enable_clang_tidy(glimmer_driver_proxy_${provider})
    endforeach()
    add_library(glimmer_driver_proxy_runtime_caller SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_proxy_runtime_caller.cc")
    target_link_libraries(glimmer_driver_proxy_runtime_caller PRIVATE
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    target_compile_options(glimmer_driver_proxy_runtime_caller PRIVATE -fno-optimize-sibling-calls)
    set_target_properties(glimmer_driver_proxy_runtime_caller PROPERTIES
        OUTPUT_NAME cudart.so.glimmer_proxy_fixture)
    glimmer_enable_clang_tidy(glimmer_driver_proxy_runtime_caller)
    add_executable(glimmer_cuda_driver_proxy_lookup_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_proxy_lookup_test.cc")
    target_link_libraries(glimmer_cuda_driver_proxy_lookup_test PRIVATE
        Glimmer::project_options "${CMAKE_DL_LIBS}" Threads::Threads)
    add_dependencies(glimmer_cuda_driver_proxy_lookup_test
        glimmer_driver_proxy_first glimmer_driver_proxy_second glimmer_driver_proxy_runtime_caller)
    glimmer_enable_clang_tidy(glimmer_cuda_driver_proxy_lookup_test)
    if(glimmer_can_test_bootstrap)
        add_test(NAME glimmer_cuda_driver_proxy_lookup_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_proxy_lookup_test.py"
                --binary $<TARGET_FILE:glimmer_cuda_driver_proxy_lookup_test>
                --interceptor $<TARGET_FILE:glimmer_cuda_interceptor>
                --preload "${glimmer_bootstrap_preload}"
                --first $<TARGET_FILE:glimmer_driver_proxy_first>
                --second $<TARGET_FILE:glimmer_driver_proxy_second>
                --caller $<TARGET_FILE:glimmer_driver_proxy_runtime_caller>)
        set_tests_properties(glimmer_cuda_driver_proxy_lookup_test PROPERTIES TIMEOUT 120)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        foreach(provider IN ITEMS first second third local)
            if(provider STREQUAL "first")
                set(provider_id 1)
            elseif(provider STREQUAL "second")
                set(provider_id 2)
            elseif(provider STREQUAL "third")
                set(provider_id 3)
            else()
                set(provider_id 4)
            endif()
            add_library(glimmer_dlsym_chain_${provider} SHARED
                "${PROJECT_SOURCE_DIR}/tests/interceptor/dlsym_chain_fixture.cc")
            target_compile_features(glimmer_dlsym_chain_${provider} PRIVATE cxx_std_20)
            target_compile_options(glimmer_dlsym_chain_${provider} PRIVATE
                -Wall -Wextra -Wpedantic -fno-optimize-sibling-calls)
            if(GLIMMER_WARNINGS_AS_ERRORS)
                target_compile_options(glimmer_dlsym_chain_${provider} PRIVATE -Werror)
            endif()
            target_compile_definitions(glimmer_dlsym_chain_${provider} PRIVATE
                GLIMMER_LOOKUP_ID=${provider_id} GLIMMER_LOOKUP_ENTRY=glimmer_lookup_${provider})
            target_link_libraries(glimmer_dlsym_chain_${provider} PRIVATE "${CMAKE_DL_LIBS}")
            glimmer_enable_clang_tidy(glimmer_dlsym_chain_${provider})
        endforeach()
        add_executable(glimmer_cuda_dlsym_chain_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/dlsym_chain_test.cc")
        target_compile_features(glimmer_cuda_dlsym_chain_test PRIVATE cxx_std_20)
        target_compile_options(glimmer_cuda_dlsym_chain_test PRIVATE
            -Wall -Wextra -Wpedantic -fno-optimize-sibling-calls)
        if(GLIMMER_WARNINGS_AS_ERRORS)
            target_compile_options(glimmer_cuda_dlsym_chain_test PRIVATE -Werror)
        endif()
        target_link_libraries(glimmer_cuda_dlsym_chain_test PRIVATE "${CMAKE_DL_LIBS}" Threads::Threads)
        glimmer_enable_clang_tidy(glimmer_cuda_dlsym_chain_test)
        set(glimmer_chain_first "$<TARGET_FILE:glimmer_dlsym_chain_first>")
        set(glimmer_chain_second "$<TARGET_FILE:glimmer_dlsym_chain_second>")
        set(glimmer_chain_third "$<TARGET_FILE:glimmer_dlsym_chain_third>")
        set(glimmer_chain_local "$<TARGET_FILE:glimmer_dlsym_chain_local>")
        add_test(NAME glimmer_cuda_dlsym_chain_native_test COMMAND glimmer_cuda_dlsym_chain_test
            "${glimmer_chain_second}" "${glimmer_chain_third}" "${glimmer_chain_local}")
        set_tests_properties(glimmer_cuda_dlsym_chain_native_test PROPERTIES
            ENVIRONMENT "LD_PRELOAD=${glimmer_chain_first}:${glimmer_chain_second}:${glimmer_chain_third}"
            TIMEOUT 20)
        if(glimmer_can_test_bootstrap)
            set(glimmer_chain_sanitizer "")
            if(GLIMMER_ENABLE_SANITIZERS)
                set(glimmer_chain_sanitizer "${glimmer_asan_library}:")
            endif()
            foreach(position IN ITEMS before after)
                if(position STREQUAL "before")
                    set(glimmer_chain_preload "${glimmer_bootstrap_preload}:${glimmer_chain_first}:${glimmer_chain_second}:${glimmer_chain_third}")
                else()
                    set(glimmer_chain_preload "${glimmer_chain_sanitizer}${glimmer_chain_first}:${glimmer_chain_second}:$<TARGET_FILE:glimmer_cuda_interceptor>:${glimmer_chain_third}")
                endif()
                add_test(NAME glimmer_cuda_dlsym_chain_${position}_test COMMAND glimmer_cuda_dlsym_chain_test
                    "${glimmer_chain_second}" "${glimmer_chain_third}" "${glimmer_chain_local}")
                set_tests_properties(glimmer_cuda_dlsym_chain_${position}_test PROPERTIES
                    ENVIRONMENT "LD_PRELOAD=${glimmer_chain_preload};ASAN_OPTIONS=detect_leaks=1"
                    TIMEOUT 20)
            endforeach()
        endif()
    endif()
    glimmer_add_test_program(glimmer_cuda_paused_native_executor_test
        SOURCES internal/cuda_paused_native_executor.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_paused_native_executor_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_capture_origin_registry_test
        SOURCES internal/cuda_capture_origin_registry.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_capture_origin_registry_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_cooperative_frontend_test
        SOURCES internal/cuda_cooperative_frontend.cc internal/cuda_native_execution_epoch.cc
            internal/native_execution_epoch.cc internal/cuda_binary_slice.cc
            internal/native_execution_heartbeat.cc internal/cuda_process_native_session.cc
            internal/cuda_event_projection_registry.cc internal/cuda_cooperative_pool.cc
            internal/cuda_admission_coordinator.cc internal/cuda_queue_worker.cc
            internal/cuda_operation_queue.cc internal/operation_order.cc
            internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_cooperative_frontend_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_cooperative_pool_test
        SOURCES internal/cuda_cooperative_pool.cc internal/owned_kernel_arguments.cc
            internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_cooperative_pool_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options)
    glimmer_add_test_program(glimmer_cuda_binary_slice_test
        SOURCES internal/cuda_binary_slice.cc internal/cuda_operation_queue.cc
            internal/operation_order.cc internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_binary_slice_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_gemm_slice_test
        SOURCES internal/cuda_binary_slice.cc internal/cuda_gemm_allocation.cc
            internal/cuda_copy_range.cc internal/allocation_registry.cc
            internal/cuda_operation_queue.cc internal/operation_order.cc
            internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_gemm_slice_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::core Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_qualified_cubin_test
        SOURCES internal/qualified_cubin.cc internal/qualified_cubin_source.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/qualified_cubin_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARIES Glimmer::project_options
        TIMEOUT 120)
    glimmer_add_test_program(glimmer_qualified_gemm_test
        SOURCES internal/qualified_cubin.cc internal/qualified_cubin_source.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/qualified_gemm_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARIES Glimmer::project_options
        TIMEOUT 120)
    glimmer_add_test_program(glimmer_cuda_queue_worker_test
        SOURCES internal/cuda_queue_worker.cc internal/cuda_operation_queue.cc
            internal/operation_order.cc internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_queue_worker_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_operation_queue_test
        SOURCES internal/cuda_admission_coordinator.cc internal/cuda_operation_queue.cc
            internal/operation_order.cc internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_operation_queue_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 20)
    glimmer_add_test_program(glimmer_cuda_owner_control_actor_test
        SOURCES internal/cuda_admission_coordinator.cc internal/cuda_native_execution_epoch.cc
            internal/native_execution_epoch.cc internal/cuda_queue_worker.cc
            internal/cuda_paused_native_executor.cc internal/native_execution_heartbeat.cc
            internal/cuda_operation_queue.cc internal/operation_order.cc
            internal/owned_kernel_arguments.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_owner_control_actor_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/control"
            "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads
        TIMEOUT 30)
    add_executable(glimmer_cuda_persistent_kernel_test
        driver_dispatch.cc
        internal/qualified_cubin.cc
        internal/qualified_cubin_source.cc

        internal/embedded_ptx.cc
        internal/persistent_kernel.cc
        internal/persistent_sgemm.cc
        internal/diagnostics.cc
        "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_kernel_test.cc"
    )
    target_include_directories(glimmer_cuda_persistent_kernel_test PRIVATE
        "${PROJECT_SOURCE_DIR}/src/interceptor" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_persistent_kernel_test PRIVATE Glimmer::transform
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_persistent_kernel_test)
    add_test(NAME glimmer_cuda_persistent_kernel_test COMMAND glimmer_cuda_persistent_kernel_test
        $<TARGET_FILE:glimmer_ptx_dump_fixture>)

    glimmer_add_test_program(glimmer_cuda_launch_completion_tracker_test
        SOURCES driver_dispatch.cc launch_completion_tracker.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/launch_completion_tracker_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options "${CMAKE_DL_LIBS}" Threads::Threads
        TIMEOUT 20)

    glimmer_add_test_program(glimmer_cuda_interceptor_dlsym_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/dlsym_interceptor_test.cc"
        LIBRARIES Glimmer::project_options "${CMAKE_DL_LIBS}")
    set_tests_properties(glimmer_cuda_interceptor_dlsym_test PROPERTIES
        ENVIRONMENT "LD_PRELOAD=${glimmer_bootstrap_preload}"
    )

    glimmer_add_test_program(glimmer_cuda_interceptor_allocation_registry_test
        SOURCES internal/allocation_registry.cc internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/allocation_registry_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::control Glimmer::project_options Threads::Threads)

    glimmer_add_test_program(glimmer_cuda_interceptor_dispatch_test
        SOURCES driver_dispatch.cc "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_dispatch_test.cc"
        INCLUDES "${PROJECT_SOURCE_DIR}/src/interceptor" "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES Glimmer::project_options "${CMAKE_DL_LIBS}")

    add_library(glimmer_fake_cuda_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_driver.cc"
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_graph.cc"
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_memory.cc"
    )
    set_target_properties(glimmer_fake_cuda_driver PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME cuda
        SUFFIX ".so.1"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda"
    )
    target_include_directories(glimmer_fake_cuda_driver
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CUDAToolkit_INCLUDE_DIRS}"
    )
    target_link_libraries(glimmer_fake_cuda_driver PRIVATE Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_fake_cuda_driver)
    add_library(glimmer_fake_owner_control_startup_driver SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_owner_control_startup_driver.cc")
    set_target_properties(glimmer_fake_owner_control_startup_driver PROPERTIES
        OUTPUT_NAME cuda SUFFIX ".so.1"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_owner_startup")
    target_include_directories(glimmer_fake_owner_control_startup_driver PRIVATE
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_fake_owner_control_startup_driver PRIVATE Glimmer::project_options)
    target_link_options(glimmer_fake_owner_control_startup_driver PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_fake_owner_control_startup_driver)
    add_executable(glimmer_cuda_owner_control_startup_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_owner_control_startup_test.cc")
    target_include_directories(glimmer_cuda_owner_control_startup_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/control"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_owner_control_startup_test PRIVATE
        Glimmer::control glimmer_fake_owner_control_startup_driver Glimmer::project_options Threads::Threads)
    glimmer_enable_clang_tidy(glimmer_cuda_owner_control_startup_test)
    add_dependencies(glimmer_cuda_owner_control_startup_test glimmer_cuda_interceptor)
    if(glimmer_can_test_bootstrap)
        add_test(NAME glimmer_cuda_owner_control_startup_test
            COMMAND glimmer_cuda_owner_control_startup_test)
        set_tests_properties(glimmer_cuda_owner_control_startup_test PROPERTIES
            ENVIRONMENT "LD_PRELOAD=${glimmer_bootstrap_preload};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_owner_control_startup_driver>"
            TIMEOUT 30)
    endif()
    add_executable(glimmer_cuda_driver_graph_routing_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_graph_routing_test.cc")
    target_include_directories(glimmer_cuda_driver_graph_routing_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_driver_graph_routing_test PRIVATE
        glimmer_fake_cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_driver_graph_routing_test)
    add_dependencies(glimmer_cuda_driver_graph_routing_test glimmer_cuda_interceptor)
    if(glimmer_can_test_bootstrap)
        add_test(NAME glimmer_cuda_driver_graph_routing_test COMMAND glimmer_cuda_driver_graph_routing_test)
        set_tests_properties(glimmer_cuda_driver_graph_routing_test PROPERTIES
            ENVIRONMENT "LD_PRELOAD=${glimmer_bootstrap_preload};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>" TIMEOUT 20)
    else()
        message(WARNING "Cold graph ABI sanitizer test unavailable: shared ASan runtime not found")
    endif()
    glimmer_add_test_program(glimmer_cuda_primary_release_routing_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/primary_release_routing_test.cc"
        INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include"
        LIBRARIES glimmer_cuda_interceptor Glimmer::project_options CUDA::cuda_driver
        TIMEOUT 20)
    add_executable(glimmer_cuda_driver_copy_routing_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_copy_routing_test.cc")
    target_include_directories(glimmer_cuda_driver_copy_routing_test PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_driver_copy_routing_test PRIVATE
        glimmer_fake_cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_driver_copy_routing_test)
    add_dependencies(glimmer_cuda_driver_copy_routing_test glimmer_cuda_interceptor)
    if(glimmer_can_test_bootstrap)
        add_test(NAME glimmer_cuda_driver_copy_routing_test COMMAND glimmer_cuda_driver_copy_routing_test)
        set_tests_properties(glimmer_cuda_driver_copy_routing_test PROPERTIES
            ENVIRONMENT "LD_PRELOAD=${glimmer_bootstrap_preload};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
            TIMEOUT 20)
        add_test(NAME glimmer_cuda_driver_cooperative_rejection_test COMMAND glimmer_cuda_driver_copy_routing_test --reject)
        set_tests_properties(glimmer_cuda_driver_cooperative_rejection_test PROPERTIES
            ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;LD_PRELOAD=${glimmer_bootstrap_preload};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
            TIMEOUT 20)
        add_test(NAME glimmer_cuda_driver_copy_recursive_lookup_test
            COMMAND glimmer_cuda_driver_copy_routing_test --self)
        set_tests_properties(glimmer_cuda_driver_copy_recursive_lookup_test PROPERTIES
            ENVIRONMENT "GLIMMER_FAKE_SELF_COPY_LOOKUP=1;LD_PRELOAD=${glimmer_bootstrap_preload};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
            TIMEOUT 20)
    else()
        message(WARNING "Cold copy/capture ABI sanitizer tests unavailable: shared ASan runtime not found")
    endif()

    add_library(glimmer_cuda_cooperative_accounting_fixture SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cooperative_accounting_test.cc")
    target_include_directories(glimmer_cuda_cooperative_accounting_fixture PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_cooperative_accounting_fixture PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_cuda_cooperative_accounting_fixture)
    add_executable(glimmer_cuda_cooperative_accounting_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/cooperative_accounting_runner.cc")
    target_link_libraries(glimmer_cuda_cooperative_accounting_test PRIVATE
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_cuda_cooperative_accounting_test
        glimmer_cuda_cooperative_accounting_fixture glimmer_fake_cuda_driver)
    glimmer_enable_clang_tidy(glimmer_cuda_cooperative_accounting_test)
    foreach(progress_case IN ITEMS normal rollback released-accounting-error uncertain-allocation)
        add_test(NAME glimmer_cuda_cooperative_accounting_${progress_case}_test
            COMMAND glimmer_cuda_cooperative_accounting_test
                $<TARGET_FILE:glimmer_cuda_cooperative_accounting_fixture> "--${progress_case}")
        set_tests_properties(glimmer_cuda_cooperative_accounting_${progress_case}_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=4096;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
            TIMEOUT 20)
    endforeach()

    glimmer_add_test_program(glimmer_cuda_persistent_preload_test
        SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_preload_test.cc"
        INCLUDES "${CUDAToolkit_INCLUDE_DIRS}"
        LIBRARIES glimmer_fake_cuda_driver Glimmer::project_options "${CMAKE_DL_LIBS}")
    set_tests_properties(glimmer_cuda_persistent_preload_test PROPERTIES
        ENVIRONMENT "GLIMMER_PERSISTENT_WORKER_BLOCKS=3;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        TIMEOUT 20)

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
    target_compile_definitions(glimmer_fake_cuda_runtime PRIVATE GLIMMER_FAKE_GRAPH_ABI_VERSIONS)
    target_link_options(glimmer_fake_cuda_runtime PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runtime_graph_versions.map")
    set_property(TARGET glimmer_fake_cuda_runtime APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime_graph_versions.map")
    # Isolate owner-routing originals without DEEPBIND (ASan forbids it).
    # Other fake Driver fixtures keep their existing interposition behavior.
    get_target_property(glimmer_graph_owner_driver_sources glimmer_fake_cuda_driver SOURCES)
    add_library(glimmer_fake_cuda_graph_owner_driver SHARED ${glimmer_graph_owner_driver_sources})
    set_target_properties(glimmer_fake_cuda_graph_owner_driver PROPERTIES
        OUTPUT_NAME cuda SUFFIX ".so.1"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda_graph_owner")
    target_include_directories(glimmer_fake_cuda_graph_owner_driver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_fake_cuda_graph_owner_driver PRIVATE Glimmer::project_options)
    target_link_options(glimmer_fake_cuda_graph_owner_driver PRIVATE "LINKER:-Bsymbolic-functions")
    glimmer_enable_clang_tidy(glimmer_fake_cuda_graph_owner_driver)
    add_executable(glimmer_cuda_public_call_gate_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/public_call_gate_test.cc")
    target_include_directories(glimmer_cuda_public_call_gate_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_public_call_gate_test PRIVATE
        glimmer_cuda_interceptor glimmer_fake_cuda_runtime glimmer_fake_cuda_driver
        Glimmer::project_options Threads::Threads "${CMAKE_DL_LIBS}")
    # Main exports the CUDA ABIs too; retain both fake originals for RTLD_NEXT.
    target_link_options(glimmer_cuda_public_call_gate_test PRIVATE "LINKER:--no-as-needed")
    glimmer_enable_clang_tidy(glimmer_cuda_public_call_gate_test)
    add_executable(glimmer_cuda_runtime_graph_routing_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_graph_routing_test.cc")
    add_library(glimmer_fake_runtime_graph_unversioned SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_runtime_graph_unversioned.cc")
    set_target_properties(glimmer_fake_runtime_graph_unversioned PROPERTIES OUTPUT_NAME cudart
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda_graph_unversioned")
    target_include_directories(glimmer_fake_runtime_graph_unversioned PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_fake_runtime_graph_unversioned PRIVATE Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_fake_runtime_graph_unversioned)
    target_include_directories(glimmer_cuda_runtime_graph_routing_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_runtime_graph_routing_test PRIVATE
        glimmer_cuda_interceptor glimmer_fake_cuda_runtime
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_cuda_runtime_graph_routing_test glimmer_fake_cuda_graph_owner_driver)
    add_dependencies(glimmer_cuda_runtime_graph_routing_test glimmer_fake_runtime_graph_unversioned)
    target_compile_definitions(glimmer_cuda_runtime_graph_routing_test PRIVATE
        GLIMMER_UNVERSIONED_RUNTIME_LIBRARY="$<TARGET_FILE:glimmer_fake_runtime_graph_unversioned>")
    glimmer_enable_clang_tidy(glimmer_cuda_runtime_graph_routing_test)
    add_test(NAME glimmer_cuda_runtime_graph_routing_test COMMAND glimmer_cuda_runtime_graph_routing_test)
    add_test(NAME glimmer_cuda_runtime_graph_context_failure_test
        COMMAND glimmer_cuda_runtime_graph_routing_test --lost-context)
    add_test(NAME glimmer_cuda_runtime_graph_unknown_native_test
        COMMAND glimmer_cuda_runtime_graph_routing_test --unknown-native)
    add_test(NAME glimmer_cuda_runtime_graph_unknown_preview_test
        COMMAND glimmer_cuda_runtime_graph_routing_test --unknown-preview)
    set_tests_properties(glimmer_cuda_runtime_graph_unknown_native_test
        glimmer_cuda_runtime_graph_unknown_preview_test PROPERTIES TIMEOUT 20)
    set_tests_properties(glimmer_cuda_runtime_graph_unknown_native_test PROPERTIES
        ENVIRONMENT "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>")
    set_tests_properties(glimmer_cuda_runtime_graph_unknown_preview_test PROPERTIES
        ENVIRONMENT "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>")
    set_tests_properties(glimmer_cuda_runtime_graph_routing_test
        glimmer_cuda_runtime_graph_context_failure_test PROPERTIES
        ENVIRONMENT "GLIMMER_FAKE_GRAPH_OWNER_DISPATCH=1;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_graph_owner_driver>:$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>"
        TIMEOUT 20)
    add_executable(glimmer_cuda_runtime_capacity_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_capacity_test.cc")
    target_include_directories(glimmer_cuda_runtime_capacity_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_runtime_capacity_test PRIVATE
        glimmer_cuda_interceptor glimmer_fake_cuda_runtime glimmer_fake_cuda_driver Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_cuda_runtime_capacity_test)
    foreach(release_mode IN ITEMS null unknown)
        add_test(NAME glimmer_cuda_runtime_${release_mode}_free_test
            COMMAND glimmer_cuda_runtime_capacity_test --${release_mode}-free)
        set_tests_properties(glimmer_cuda_runtime_${release_mode}_free_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>:$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>"
            TIMEOUT 20)
    endforeach()
    foreach(capacity_mode IN ITEMS inactive failure unknown)
        add_test(NAME glimmer_cuda_runtime_capacity_${capacity_mode}_test
            COMMAND glimmer_cuda_runtime_capacity_test)
        set(capacity_fault_environment "")
        if(capacity_mode STREQUAL "failure")
            set(capacity_fault_environment "GLIMMER_FAKE_COLD_PRIMARY_FAILURE=1")
        elseif(capacity_mode STREQUAL "unknown")
            set(capacity_fault_environment "GLIMMER_FAKE_COLD_PRIMARY_UNKNOWN=1")
        endif()
        set_tests_properties(glimmer_cuda_runtime_capacity_${capacity_mode}_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0;GLIMMER_FAKE_COLD_RUNTIME_CAPACITY=1;${capacity_fault_environment};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>:$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>"
            TIMEOUT 20)
    endforeach()
    add_executable(glimmer_cuda_runtime_capture_dispatch_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_capture_dispatch_test.cc")
    target_include_directories(glimmer_cuda_runtime_capture_dispatch_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_runtime_capture_dispatch_test PRIVATE
        glimmer_cuda_interceptor glimmer_fake_cuda_runtime glimmer_fake_cuda_driver Glimmer::project_options)
    target_link_options(glimmer_cuda_runtime_capture_dispatch_test PRIVATE "LINKER:--no-as-needed")
    add_dependencies(glimmer_cuda_runtime_capture_dispatch_test glimmer_fake_cuda_graph_owner_driver)
    glimmer_enable_clang_tidy(glimmer_cuda_runtime_capture_dispatch_test)
    add_test(NAME glimmer_cuda_runtime_capture_dispatch_test
        COMMAND glimmer_cuda_runtime_capture_dispatch_test)
    set_tests_properties(glimmer_cuda_runtime_capture_dispatch_test PROPERTIES
        ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        TIMEOUT 20)
    add_executable(glimmer_cuda_runtime_profiling_boundary_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_profiling_boundary_test.cc")
    target_include_directories(glimmer_cuda_runtime_profiling_boundary_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_runtime_profiling_boundary_test PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_cuda_runtime_profiling_boundary_test)
    foreach(profile_mode IN ITEMS unset zero one empty unknown whitespace)
        if(profile_mode STREQUAL "unset" OR profile_mode STREQUAL "zero")
            set(profile_expected disabled)
        elseif(profile_mode STREQUAL "one")
            set(profile_expected enabled)
        else()
            set(profile_expected invalid)
        endif()
        if(profile_mode STREQUAL "unset")
            set(profile_env_args --unset=GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY)
        elseif(profile_mode STREQUAL "zero")
            set(profile_env_args GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY=0)
        elseif(profile_mode STREQUAL "one")
            set(profile_env_args GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY=1)
        elseif(profile_mode STREQUAL "empty")
            set(profile_env_args GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY=)
        elseif(profile_mode STREQUAL "whitespace")
            set(profile_env_args "GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY= 1")
        else()
            set(profile_env_args GLIMMER_PROFILE_RUNTIME_NATIVE_BOUNDARY=unknown)
        endif()
        add_test(NAME glimmer_cuda_runtime_profiling_boundary_${profile_mode}_test
            COMMAND ${CMAKE_COMMAND} -E env ${profile_env_args}
                $<TARGET_FILE:glimmer_cuda_runtime_profiling_boundary_test>
                ${profile_expected})
        set_tests_properties(glimmer_cuda_runtime_profiling_boundary_${profile_mode}_test PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>" TIMEOUT 20)
    endforeach()
    add_test(NAME glimmer_cuda_process_epoch_startup_test
        COMMAND glimmer_cuda_runtime_capture_dispatch_test --epoch-startup)
    foreach(epoch_environment_mode IN ITEMS valid default partial empty negative plus whitespace trailing overflow zero call-bound prefix-width prefix-bound time-bound runtime-disabled mode-disabled batch quota-missing quota-invalid quota-mode private-conflict)
        string(REPLACE "-" "_" epoch_environment_test_name "${epoch_environment_mode}")
        add_test(NAME glimmer_cuda_epoch_environment_${epoch_environment_test_name}_test
            COMMAND glimmer_cuda_runtime_capture_dispatch_test --epoch-environment-${epoch_environment_mode})
        set_tests_properties(glimmer_cuda_epoch_environment_${epoch_environment_test_name}_test PROPERTIES
            ENVIRONMENT "GLIMMER_FAKE_GRAPH_OWNER_DISPATCH=1;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_graph_owner_driver>:$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>"
            TIMEOUT 20)
    endforeach()
    add_test(NAME glimmer_cuda_process_epoch_late_startup_test
        COMMAND glimmer_cuda_runtime_capture_dispatch_test --epoch-startup-late)
    set_tests_properties(glimmer_cuda_process_epoch_startup_test glimmer_cuda_process_epoch_late_startup_test PROPERTIES
        ENVIRONMENT "GLIMMER_FAKE_GRAPH_OWNER_DISPATCH=1;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_graph_owner_driver>:$<TARGET_FILE_DIR:glimmer_fake_cuda_runtime>"
        TIMEOUT 20)
    add_test(NAME glimmer_cuda_private_runtime_kernel_bypass_test
        COMMAND glimmer_cuda_runtime_capture_dispatch_test --private-kernel-bypass)
    set_tests_properties(glimmer_cuda_private_runtime_kernel_bypass_test PROPERTIES
        ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        TIMEOUT 20)
    add_test(NAME glimmer_cuda_runtime_cold_capacity_test
        COMMAND glimmer_cuda_runtime_capture_dispatch_test --cold-capacity)
    set_tests_properties(glimmer_cuda_runtime_cold_capacity_test PROPERTIES
        ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_FAKE_UNBOUND_CAPACITY=1;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        TIMEOUT 20)
    foreach(mode IN ITEMS driver runtime disabled)
        if(mode STREQUAL "disabled")
            add_test(NAME glimmer_cuda_public_call_gate_${mode}_test
                COMMAND glimmer_cuda_public_call_gate_test --disabled)
            set(preview_environment "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0")
        else()
            add_test(NAME glimmer_cuda_public_call_gate_${mode}_test
                COMMAND glimmer_cuda_public_call_gate_test)
            if(mode STREQUAL "driver")
                set(preview_environment "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0")
            else()
                set(preview_environment "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1")
            endif()
        endif()
        set_tests_properties(glimmer_cuda_public_call_gate_${mode}_test PROPERTIES
            ENVIRONMENT "${preview_environment};LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
            TIMEOUT 20)
    endforeach()
    add_executable(glimmer_cuda_event_routing_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/event_routing_test.cc")
    target_include_directories(glimmer_cuda_event_routing_test PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_event_routing_test PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    target_compile_definitions(glimmer_cuda_event_routing_test PRIVATE
        GLIMMER_EVENT_INTERCEPTOR_PATH="$<TARGET_FILE:glimmer_cuda_interceptor>"
        GLIMMER_EVENT_MISSING_RUNTIME_PATH="$<TARGET_FILE:glimmer_fake_cuda_runtime_no_event_ptds>")
    add_library(glimmer_fake_cuda_runtime_no_event_ptds SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_runtime.cc")
    set_target_properties(glimmer_fake_cuda_runtime_no_event_ptds PROPERTIES
        OUTPUT_NAME cudart LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_cuda_missing_event_ptds")
    target_compile_definitions(glimmer_fake_cuda_runtime_no_event_ptds PRIVATE GLIMMER_FAKE_NO_EVENT_PTDS)
    target_include_directories(glimmer_fake_cuda_runtime_no_event_ptds PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_fake_cuda_runtime_no_event_ptds PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_fake_cuda_runtime_no_event_ptds)
    add_dependencies(glimmer_cuda_event_routing_test glimmer_fake_cuda_driver
        glimmer_fake_cuda_runtime glimmer_fake_cuda_runtime_no_event_ptds glimmer_cuda_interceptor)
    glimmer_enable_clang_tidy(glimmer_cuda_event_routing_test)
    set(event_routing_arguments)
    set(event_routing_environment "GLIMMER_FAKE_EVENT_AUDIT=1"
        "GLIMMER_SCHEDULER_MODE=enforce" "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>")
    if(GLIMMER_ENABLE_SANITIZERS)
        list(APPEND event_routing_arguments --driver-only)
    else()
        list(APPEND event_routing_environment "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>")
    endif()
    add_test(NAME glimmer_cuda_event_routing_test COMMAND glimmer_cuda_event_routing_test ${event_routing_arguments})
    set_tests_properties(glimmer_cuda_event_routing_test PROPERTIES TIMEOUT 20
        ENVIRONMENT "${event_routing_environment}")
    if(NOT GLIMMER_ENABLE_SANITIZERS)
        add_test(NAME glimmer_cuda_event_missing_ptds_test
            COMMAND glimmer_cuda_event_routing_test --missing-runtime-ptds)
        set_tests_properties(glimmer_cuda_event_missing_ptds_test PROPERTIES TIMEOUT 20
            ENVIRONMENT "${event_routing_environment}")
    endif()

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
    add_executable(glimmer_cuda_extended_launch_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/extended_launch_test.cc")
    target_include_directories(glimmer_cuda_extended_launch_test PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_extended_launch_test PRIVATE
        Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_dependencies(glimmer_cuda_extended_launch_test glimmer_fake_cuda_driver
        glimmer_fake_cuda_runtime glimmer_cuda_interceptor)
    glimmer_enable_clang_tidy(glimmer_cuda_extended_launch_test)
    add_test(NAME glimmer_cuda_extended_launch_test COMMAND glimmer_cuda_extended_launch_test)
    add_test(NAME glimmer_cuda_extended_launch_missing_runtime_test
        COMMAND glimmer_cuda_extended_launch_test --missing-runtime)
    set_tests_properties(glimmer_cuda_extended_launch_missing_runtime_test PROPERTIES TIMEOUT 20
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>")
    set_tests_properties(glimmer_cuda_extended_launch_test PROPERTIES
        TIMEOUT 20
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>")
    add_executable(glimmer_cuda_capture_admission_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/capture_admission_test.cc")
    target_include_directories(glimmer_cuda_capture_admission_test PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_cuda_capture_admission_test PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    add_library(glimmer_fake_cuda_runtime_no_capture SHARED
        "${PROJECT_SOURCE_DIR}/tests/interceptor/fake_cuda_runtime.cc")
    set_target_properties(glimmer_fake_cuda_runtime_no_capture PROPERTIES
        OUTPUT_NAME cudart
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/fake_missing_capture")
    target_compile_definitions(glimmer_fake_cuda_runtime_no_capture PRIVATE GLIMMER_FAKE_NO_CAPTURE)
    target_include_directories(glimmer_fake_cuda_runtime_no_capture PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_link_libraries(glimmer_fake_cuda_runtime_no_capture PRIVATE Glimmer::project_options "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_fake_cuda_runtime_no_capture)
    target_compile_definitions(glimmer_cuda_capture_admission_test PRIVATE
        GLIMMER_FAKE_MISSING_CAPTURE_RUNTIME_PATH="$<TARGET_FILE:glimmer_fake_cuda_runtime_no_capture>"
        GLIMMER_CAPTURE_INTERCEPTOR_PATH="$<TARGET_FILE:glimmer_cuda_interceptor>")
    add_dependencies(glimmer_cuda_capture_admission_test glimmer_fake_cuda_driver
        glimmer_fake_cuda_runtime glimmer_fake_cuda_runtime_no_capture glimmer_cuda_interceptor)
    glimmer_enable_clang_tidy(glimmer_cuda_capture_admission_test)
    set(capture_admission_arguments)
    set(capture_admission_environment
        "GLIMMER_SCHEDULER_MODE=enforce" "GLIMMER_MAX_CONCURRENT_KERNELS=1"
        "GLIMMER_TRACE_LAUNCH_TIMINGS=1" "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>")
    if(GLIMMER_ENABLE_SANITIZERS)
        # Validate Driver wrappers after sanitizer startup. Runtime's RTLD_NEXT
        # scope and preload startup remain explicit instrumentation exceptions.
        list(APPEND capture_admission_arguments --load-interceptor)
    else()
        list(APPEND capture_admission_environment "LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>")
    endif()
    add_test(NAME glimmer_cuda_capture_admission_test
        COMMAND glimmer_cuda_capture_admission_test ${capture_admission_arguments})
    set_tests_properties(glimmer_cuda_capture_admission_test PROPERTIES TIMEOUT 20
        ENVIRONMENT "${capture_admission_environment}"
        FAIL_REGULAR_EXPRESSION "capture admission regression;launch timing task=2")
    add_test(NAME glimmer_cuda_capture_missing_runtime_test
        COMMAND glimmer_cuda_capture_admission_test ${capture_admission_arguments} --missing-runtime-capture)
    set_tests_properties(glimmer_cuda_capture_missing_runtime_test PROPERTIES TIMEOUT 20
        ENVIRONMENT "${capture_admission_environment}")
    if(GLIMMER_ENABLE_SANITIZERS)
        set_tests_properties(glimmer_cuda_capture_missing_runtime_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
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

    add_test(NAME glimmer_cuda_interceptor_enforce_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_enforce_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
    )

    add_test(NAME glimmer_cuda_interceptor_batched_enforce_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_batched_enforce_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_BATCH_SIZE=2;GLIMMER_TRACE_LAUNCH_TIMINGS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "lease_reused=1"
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
            "observed CUDA graph launch #3 api=cuGraphLaunch stream="
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

    add_test(NAME glimmer_cuda_interceptor_trace_launch_timing_fake_driver_test
        COMMAND glimmer_cuda_interceptor_fake_driver_test
    )
    set_tests_properties(glimmer_cuda_interceptor_trace_launch_timing_fake_driver_test PROPERTIES
        ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=4096;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_TRACE_LAUNCH_TIMINGS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:glimmer_fake_cuda_driver>"
        PASS_REGULAR_EXPRESSION
            "launch timing task=1 remote=0 .*cuda_status=0 event_tracking=1"
    )

    # PASS_REGULAR_EXPRESSION ignores exit status. A matched diagnostic must
    # never hide a later failed assertion; preserve existing failure patterns.
    set_property(TEST
        glimmer_cuda_interceptor_physical_capacity_test
        glimmer_cuda_interceptor_batched_enforce_fake_driver_test
        glimmer_cuda_interceptor_observe_fake_driver_test
        glimmer_cuda_interceptor_trace_fake_driver_test
        glimmer_cuda_interceptor_trace_memory_fake_driver_test
        glimmer_cuda_interceptor_trace_launch_timing_fake_driver_test
        APPEND PROPERTY FAIL_REGULAR_EXPRESSION "test assertion failed:")

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

    add_executable(glimmer_cuda_persistent_kernel_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_kernel_gpu_test.cc")
    target_link_libraries(glimmer_cuda_persistent_kernel_gpu_test PRIVATE
        Glimmer::project_options CUDA::cuda_driver "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_persistent_kernel_gpu_test)

    add_executable(glimmer_cuda_driver_cooperative_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_cooperative_gpu_test.cc")
    target_link_libraries(glimmer_cuda_driver_cooperative_gpu_test PRIVATE
        Glimmer::project_options CUDA::cuda_driver Threads::Threads "${CMAKE_DL_LIBS}")
    glimmer_enable_clang_tidy(glimmer_cuda_driver_cooperative_gpu_test)

    add_executable(glimmer_cuda_process_native_session_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/process_native_session_gpu_test.cc")
    target_include_directories(glimmer_cuda_process_native_session_gpu_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(glimmer_cuda_process_native_session_gpu_test PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options CUDA::cuda_driver)
    glimmer_enable_clang_tidy(glimmer_cuda_process_native_session_gpu_test)
    add_executable(glimmer_cuda_runtime_capture_owner_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_capture_owner_gpu_test.cc")
    target_include_directories(glimmer_cuda_runtime_capture_owner_gpu_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(glimmer_cuda_runtime_capture_owner_gpu_test PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options CUDA::cuda_driver CUDA::cudart)
    glimmer_enable_clang_tidy(glimmer_cuda_runtime_capture_owner_gpu_test)
    add_executable(glimmer_cuda_primary_release_owner_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/primary_release_owner_gpu_test.cc")
    target_include_directories(glimmer_cuda_primary_release_owner_gpu_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(glimmer_cuda_primary_release_owner_gpu_test PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options CUDA::cuda_driver)
    glimmer_enable_clang_tidy(glimmer_cuda_primary_release_owner_gpu_test)
    add_executable(glimmer_cuda_public_primary_release_gpu_test
        "${PROJECT_SOURCE_DIR}/tests/interceptor/public_primary_release_gpu_test.cc")
    target_include_directories(glimmer_cuda_public_primary_release_gpu_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(glimmer_cuda_public_primary_release_gpu_test PRIVATE
        glimmer_cuda_interceptor Glimmer::project_options CUDA::cuda_driver CUDA::cudart)
    glimmer_enable_clang_tidy(glimmer_cuda_public_primary_release_gpu_test)
    find_library(GLIMMER_LEGACY_CUDART_LIBRARY NAMES libcudart.so.11.0)
    if(GLIMMER_LEGACY_CUDART_LIBRARY)
        add_executable(glimmer_cuda_runtime_graph_legacy_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_graph_legacy_gpu_test.cc")
        target_include_directories(glimmer_cuda_runtime_graph_legacy_gpu_test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(glimmer_cuda_runtime_graph_legacy_gpu_test PRIVATE
            glimmer_cuda_interceptor "${GLIMMER_LEGACY_CUDART_LIBRARY}" Glimmer::project_options)
        glimmer_enable_clang_tidy(glimmer_cuda_runtime_graph_legacy_gpu_test)
        if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
            foreach(legacy_graph_mode IN ITEMS native owned cold-capacity cold-owned nonprimary-capacity update-native update-owned)
                string(REPLACE "-" "_" legacy_graph_test_name "${legacy_graph_mode}")
                add_test(NAME glimmer_cuda_runtime_graph_legacy_${legacy_graph_test_name}_gpu_test
                    COMMAND glimmer_cuda_runtime_graph_legacy_gpu_test --${legacy_graph_mode})
                set_tests_properties(glimmer_cuda_runtime_graph_legacy_${legacy_graph_test_name}_gpu_test PROPERTIES
                    ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
                    TIMEOUT 60)
            endforeach()
        endif()
    else()
        message(STATUS "Legacy Runtime graph hardware ABI test unavailable: CUDA 11 shared Runtime not installed")
    endif()
    if(GLIMMER_ENABLE_CUDA_GPU_TESTS)
        glimmer_add_test_program(glimmer_persistent_sgemm_gpu_test
            SOURCES internal/persistent_sgemm.cc internal/cuda_cooperative_pool.cc
                internal/owned_kernel_arguments.cc internal/diagnostics.cc
                "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_sgemm_gpu_test.cc"
            INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
            LIBRARIES Glimmer::project_options CUDA::cuda_driver CUDA::cublas Threads::Threads
            TIMEOUT 120)
        add_executable(glimmer_persistent_sgemm_public_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_sgemm_public_gpu_test.cc")
        target_link_libraries(glimmer_persistent_sgemm_public_gpu_test PRIVATE
            Glimmer::project_options CUDA::cuda_driver CUDA::cublas "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_persistent_sgemm_public_gpu_test)
        add_test(NAME glimmer_persistent_sgemm_native_gpu_test
            COMMAND glimmer_persistent_sgemm_public_gpu_test)
        if(glimmer_can_test_bootstrap)
            add_test(NAME glimmer_persistent_sgemm_public_gpu_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DINTERCEPTOR=${glimmer_bootstrap_preload}"
                    "-DAPPLICATION=$<TARGET_FILE:glimmer_persistent_sgemm_public_gpu_test>"
                    -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_sgemm.cmake")
            set_tests_properties(glimmer_persistent_sgemm_public_gpu_test PROPERTIES TIMEOUT 130)
        endif()
        set_tests_properties(glimmer_persistent_sgemm_native_gpu_test PROPERTIES TIMEOUT 130)
        add_executable(glimmer_persistent_lt_public_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_lt_public_gpu_test.cc")
        target_link_libraries(glimmer_persistent_lt_public_gpu_test PRIVATE
            Glimmer::project_options CUDA::cuda_driver CUDA::cublas CUDA::cublasLt "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_persistent_lt_public_gpu_test)
        add_test(NAME glimmer_persistent_lt_native_gpu_test
            COMMAND glimmer_persistent_lt_public_gpu_test)
        if(glimmer_can_test_bootstrap)
            add_test(NAME glimmer_persistent_lt_public_gpu_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DINTERCEPTOR=${glimmer_bootstrap_preload}"
                    "-DAPPLICATION=$<TARGET_FILE:glimmer_persistent_lt_public_gpu_test>"
                    -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_lt.cmake")
            set_tests_properties(glimmer_persistent_lt_public_gpu_test PROPERTIES TIMEOUT 130)
        endif()
        set_tests_properties(glimmer_persistent_lt_native_gpu_test PROPERTIES TIMEOUT 130)
        add_test(NAME glimmer_cuda_runtime_capture_owner_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test)
        add_test(NAME glimmer_cuda_normal_runtime_epoch_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-epochs)
        set_tests_properties(glimmer_cuda_normal_runtime_epoch_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_runtime_epoch_recovery_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-epoch-recovery)
        set_tests_properties(glimmer_cuda_runtime_epoch_recovery_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_source_event_graph_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-event-graph)
        foreach(public_epoch_gpu_mode IN ITEMS runtime-lifecycle graph-dependency detached-capture-end cold-capture source-event-graph)
            string(REPLACE "-" "_" public_epoch_gpu_test_name "${public_epoch_gpu_mode}")
            if(public_epoch_gpu_mode STREQUAL "source-event-graph")
                set(public_epoch_gpu_argument "--source-event-graph")
            else()
                set(public_epoch_gpu_argument "--environment-${public_epoch_gpu_mode}")
            endif()
            add_test(NAME glimmer_cuda_public_epoch_environment_${public_epoch_gpu_test_name}_gpu_test
                COMMAND glimmer_cuda_runtime_capture_owner_gpu_test ${public_epoch_gpu_argument})
            set_tests_properties(glimmer_cuda_public_epoch_environment_${public_epoch_gpu_test_name}_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_NATIVE_EPOCH_MAX_CALLS=8;GLIMMER_NATIVE_EPOCH_CALLS_PER_PREFIX=2;GLIMMER_NATIVE_EPOCH_PENDING_PREFIXES=2;GLIMMER_NATIVE_EPOCH_ACCEPTANCE_US=1000000"
                TIMEOUT 90)
        endforeach()
        set_tests_properties(glimmer_cuda_source_event_graph_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 90)
        add_test(NAME glimmer_cuda_public_graph_dependency_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --public-graph-dependency)
        set_tests_properties(glimmer_cuda_public_graph_dependency_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 90)
        add_test(NAME glimmer_cuda_detached_capture_end_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --detached-capture-end)
        set_tests_properties(glimmer_cuda_detached_capture_end_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 90)
        add_test(NAME glimmer_cuda_cached_capture_overlap_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --cached-capture-overlap)
        set_tests_properties(glimmer_cuda_cached_capture_overlap_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_cold_epoch_capture_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --process-cold-epoch-capture)
        set_tests_properties(glimmer_cuda_process_cold_epoch_capture_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_epoch_capture_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --process-epoch-capture)
        set_tests_properties(glimmer_cuda_process_epoch_capture_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_runtime_lifecycle_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --process-runtime-lifecycle)
        set_tests_properties(glimmer_cuda_process_runtime_lifecycle_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_unified_epoch_recovery_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --process-epoch-recovery)
        set_tests_properties(glimmer_cuda_unified_epoch_recovery_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_runtime_null_free_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-null-free)
        set_tests_properties(glimmer_cuda_runtime_null_free_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        set_tests_properties(glimmer_cuda_runtime_capture_owner_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_source_module_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-module-probe)
        set_tests_properties(glimmer_cuda_source_module_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_source_module_retirement_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-module-retirement-probe)
        set_tests_properties(glimmer_cuda_source_module_retirement_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_runtime_code_ready_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-code-ready-probe)
        add_test(NAME glimmer_cuda_source_event_retirement_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-event-retirement-probe)
        add_test(NAME glimmer_cuda_runtime_source_event_destroy_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-source-event-destroy-probe)
        add_test(NAME glimmer_cuda_runtime_source_event_create_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-source-event-create-probe)
        add_test(NAME glimmer_cuda_source_capture_join_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-capture-join-probe)
        add_test(NAME glimmer_cuda_source_driver_stream_destroy_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-driver-stream-destroy-probe)
        add_test(NAME glimmer_cuda_source_runtime_stream_destroy_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --source-runtime-stream-destroy-probe)
        add_test(NAME glimmer_cuda_normal_runtime_event_destroy_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-event-destroy-probe)
        add_test(NAME glimmer_cuda_normal_runtime_stream_destroy_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-stream-destroy-probe)
        add_test(NAME glimmer_cuda_normal_runtime_event_completion_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-pending-completion-probe)
        add_test(NAME glimmer_cuda_normal_runtime_consumer_record_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-consumer-record-probe)
        add_test(NAME glimmer_cuda_normal_runtime_consumer_wait_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-consumer-wait-probe)
        add_test(NAME glimmer_cuda_normal_runtime_stream_completion_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-stream-completion-probe)
        add_test(NAME glimmer_cuda_normal_runtime_event_creation_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-event-creation-probe)
        add_test(NAME glimmer_cuda_normal_runtime_event_timing_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-event-timing-probe)
        add_test(NAME glimmer_cuda_normal_runtime_timing_serialization_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-timing-serialization-probe)
        add_test(NAME glimmer_cuda_normal_runtime_device_completion_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-device-sync-probe)
        add_test(NAME glimmer_cuda_normal_runtime_nonprimary_reset_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-nonprimary-reset-probe)
        add_test(NAME glimmer_cuda_normal_runtime_unbound_reset_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-unbound-reset-probe)
        add_test(NAME glimmer_cuda_normal_runtime_cached_reset_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-cached-reset-probe)
        add_test(NAME glimmer_cuda_normal_runtime_failed_reset_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --normal-runtime-failed-reset-probe)
        add_test(NAME glimmer_cuda_reversible_context_checkpoint_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --reversible-context-checkpoint)
        add_test(NAME glimmer_cuda_native_primary_release_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --native-primary-release-probe)
        foreach(release_version IN ITEMS legacy modern)
            add_test(NAME glimmer_cuda_primary_release_${release_version}_owner_gpu_test
                COMMAND glimmer_cuda_primary_release_owner_gpu_test --${release_version})
            set_tests_properties(glimmer_cuda_primary_release_${release_version}_owner_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
                TIMEOUT 180)
        endforeach()
        add_test(NAME glimmer_cuda_public_primary_release_gpu_test
            COMMAND glimmer_cuda_public_primary_release_gpu_test)
        set_tests_properties(glimmer_cuda_public_primary_release_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 300)
        add_test(NAME glimmer_cuda_native_caller_semantics_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --native-caller-semantics-probe)
        set_tests_properties(glimmer_cuda_native_caller_semantics_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 180)
        add_test(NAME glimmer_cuda_owned_primary_rejection_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --owned-primary-rejection-probe)
        set_tests_properties(glimmer_cuda_owned_primary_rejection_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 180)
        add_test(NAME glimmer_cuda_owned_primary_legacy_rejection_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --owned-primary-legacy-rejection-probe)
        set_tests_properties(glimmer_cuda_owned_primary_legacy_rejection_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 180)
        add_test(NAME glimmer_cuda_owned_user_lifecycle_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --owned-user-lifecycle-probe)
        set_tests_properties(glimmer_cuda_owned_user_lifecycle_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 300)
        add_test(NAME glimmer_cuda_owned_user_quarantine_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --owned-user-quarantine-probe)
        set_tests_properties(glimmer_cuda_owned_user_quarantine_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 180)
        add_test(NAME glimmer_cuda_owned_runtime_graph_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --owned-runtime-graph-probe)
        add_test(NAME glimmer_cuda_runtime_graph_constructor_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-graph-constructor-probe)
        add_test(NAME glimmer_cuda_runtime_graph_event_update_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-graph-event-update-probe)
        add_test(NAME glimmer_cuda_runtime_graph_node_update_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-graph-node-update-probe)
        add_test(NAME glimmer_cuda_runtime_graph_update_gpu_test
            COMMAND glimmer_cuda_runtime_capture_owner_gpu_test --runtime-graph-update-probe)
        set_tests_properties(glimmer_cuda_runtime_source_event_destroy_gpu_test
            glimmer_cuda_runtime_source_event_create_gpu_test
            glimmer_cuda_source_capture_join_gpu_test
            glimmer_cuda_source_driver_stream_destroy_gpu_test
            glimmer_cuda_source_runtime_stream_destroy_gpu_test
            glimmer_cuda_normal_runtime_event_destroy_gpu_test
            glimmer_cuda_normal_runtime_stream_destroy_gpu_test
            glimmer_cuda_normal_runtime_event_completion_gpu_test
            glimmer_cuda_normal_runtime_consumer_record_gpu_test
            glimmer_cuda_normal_runtime_consumer_wait_gpu_test
            glimmer_cuda_normal_runtime_stream_completion_gpu_test
            glimmer_cuda_normal_runtime_event_creation_gpu_test
            glimmer_cuda_normal_runtime_event_timing_gpu_test
            glimmer_cuda_normal_runtime_timing_serialization_gpu_test
            glimmer_cuda_normal_runtime_device_completion_gpu_test
            glimmer_cuda_normal_runtime_nonprimary_reset_gpu_test
            glimmer_cuda_normal_runtime_unbound_reset_gpu_test
            glimmer_cuda_normal_runtime_cached_reset_gpu_test
            glimmer_cuda_normal_runtime_failed_reset_gpu_test
            glimmer_cuda_owned_runtime_graph_gpu_test
            glimmer_cuda_runtime_graph_constructor_gpu_test
            glimmer_cuda_runtime_graph_event_update_gpu_test
            glimmer_cuda_runtime_graph_node_update_gpu_test
            glimmer_cuda_runtime_graph_update_gpu_test
            glimmer_cuda_native_primary_release_gpu_test
            glimmer_cuda_reversible_context_checkpoint_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        set_tests_properties(glimmer_cuda_reversible_context_checkpoint_gpu_test PROPERTIES
            TIMEOUT 180)
        set_tests_properties(glimmer_cuda_owned_runtime_graph_gpu_test PROPERTIES TIMEOUT 180)
        set_tests_properties(glimmer_cuda_runtime_graph_constructor_gpu_test PROPERTIES TIMEOUT 180)
        set_tests_properties(glimmer_cuda_runtime_graph_event_update_gpu_test PROPERTIES TIMEOUT 180)
        set_tests_properties(glimmer_cuda_runtime_graph_node_update_gpu_test PROPERTIES TIMEOUT 180)
        set_tests_properties(glimmer_cuda_runtime_graph_update_gpu_test PROPERTIES TIMEOUT 180)
        set_tests_properties(glimmer_cuda_source_event_retirement_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority"
            TIMEOUT 60)
        set_tests_properties(glimmer_cuda_runtime_code_ready_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_native_session_gpu_test
            COMMAND glimmer_cuda_process_native_session_gpu_test)
        set_tests_properties(glimmer_cuda_process_native_session_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_native_epoch_gpu_test
            COMMAND glimmer_cuda_process_native_session_gpu_test --native-epochs)
        set_tests_properties(glimmer_cuda_process_native_epoch_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_process_epoch_recovery_gpu_test
            COMMAND glimmer_cuda_process_native_session_gpu_test --epoch-recovery)
        set_tests_properties(glimmer_cuda_process_epoch_recovery_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_COOPERATIVE_DRIVER_PREVIEW=0;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=0"
            TIMEOUT 60)
    endif()

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
        glimmer_add_test_program(glimmer_cuda_cooperative_accounting_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/cooperative_accounting_gpu_test.cc"
            INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include"
            LIBRARIES glimmer_cuda_interceptor CUDA::cuda_driver Glimmer::project_options)
        set_tests_properties(glimmer_cuda_cooperative_accounting_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_MEMORY_LIMIT_BYTES=4096" TIMEOUT 30)
        add_executable(glimmer_cuda_operation_queue_gpu_test
            internal/qualified_cubin.cc
            internal/qualified_cubin_source.cc
            internal/cuda_native_graph_registry.cc
            internal/cuda_native_graph_effects.cc
            internal/cuda_event_projection_registry.cc
            internal/cuda_cooperative_frontend.cc
            internal/cuda_native_execution_epoch.cc
            internal/native_execution_epoch.cc
            internal/native_execution_heartbeat.cc
            internal/cuda_cooperative_pool.cc
            internal/cuda_admission_coordinator.cc
            driver_dispatch.cc
            internal/persistent_kernel.cc
            internal/persistent_sgemm.cc
            internal/embedded_ptx.cc
            internal/cuda_queue_worker.cc
            internal/cuda_operation_queue.cc
            internal/operation_order.cc
            internal/owned_kernel_arguments.cc

            internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_operation_queue_gpu_test.cc")
        glimmer_add_test_program(glimmer_cuda_native_execution_epoch_gpu_test
            SOURCES internal/native_execution_epoch.cc internal/cuda_native_execution_epoch.cc
                "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_native_execution_epoch_gpu_test.cc"
            INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
            LIBRARIES Glimmer::control Glimmer::project_options CUDA::cuda_driver Threads::Threads
            TIMEOUT 30)
        add_test(NAME glimmer_cuda_native_epoch_defaults_gpu_test
            COMMAND glimmer_cuda_native_execution_epoch_gpu_test --default-streams)
        set_tests_properties(glimmer_cuda_native_epoch_defaults_gpu_test PROPERTIES TIMEOUT 30)
        add_test(NAME glimmer_cuda_native_epoch_recovery_gpu_test
            COMMAND glimmer_cuda_native_execution_epoch_gpu_test --recovery)
        set_tests_properties(glimmer_cuda_native_epoch_recovery_gpu_test PROPERTIES TIMEOUT 30)
        glimmer_add_test_program(glimmer_cuda_binary_slice_gpu_test
            SOURCES driver_dispatch.cc internal/persistent_kernel.cc internal/persistent_sgemm.cc
                internal/embedded_ptx.cc internal/qualified_cubin_source.cc
                internal/cuda_binary_slice.cc internal/qualified_cubin.cc
                internal/cuda_operation_queue.cc internal/operation_order.cc
                internal/owned_kernel_arguments.cc internal/diagnostics.cc
                "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_binary_slice_gpu_test.cc"
            INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
            LIBRARIES Glimmer::transform Glimmer::project_options CUDA::cuda_driver Threads::Threads
                "${CMAKE_DL_LIBS}"
            TIMEOUT 30)
        add_executable(glimmer_cuda_gemm_slice_gpu_test
            internal/cuda_gemm_allocation.cc
            internal/cuda_copy_range.cc
            internal/allocation_registry.cc
            driver_dispatch.cc
            internal/persistent_kernel.cc
            internal/persistent_sgemm.cc
            internal/embedded_ptx.cc

            internal/cuda_binary_slice.cc
            internal/qualified_cubin.cc
            internal/qualified_cubin_source.cc
            internal/cuda_operation_queue.cc
            internal/operation_order.cc
            internal/owned_kernel_arguments.cc
            internal/diagnostics.cc
            "${PROJECT_SOURCE_DIR}/tests/interceptor/cuda_gemm_slice_gpu_test.cc")
        target_include_directories(glimmer_cuda_gemm_slice_gpu_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(glimmer_cuda_gemm_slice_gpu_test PRIVATE Glimmer::transform
            Glimmer::core Glimmer::project_options CUDA::cuda_driver Threads::Threads "${CMAKE_DL_LIBS}")
        if(GLIMMER_ENABLE_SANITIZERS)
            # Reuse the production sanitizer loader policy: ASan rejects DEEPBIND.
            target_compile_definitions(glimmer_cuda_gemm_slice_gpu_test PRIVATE GLIMMER_SANITIZED_INTERCEPTOR)
        endif()
        glimmer_enable_clang_tidy(glimmer_cuda_gemm_slice_gpu_test)
        add_test(NAME glimmer_cuda_gemm_slice_gpu_test COMMAND glimmer_cuda_gemm_slice_gpu_test)
        set_tests_properties(glimmer_cuda_gemm_slice_gpu_test PROPERTIES TIMEOUT 120)
        target_include_directories(glimmer_cuda_operation_queue_gpu_test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(glimmer_cuda_operation_queue_gpu_test PRIVATE Glimmer::transform
            Glimmer::project_options Glimmer::control CUDA::cuda_driver Threads::Threads "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_cuda_operation_queue_gpu_test)
        add_test(NAME glimmer_cuda_operation_queue_gpu_test
            COMMAND glimmer_cuda_operation_queue_gpu_test)
        set_tests_properties(glimmer_cuda_operation_queue_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_native_epoch_frontend_gpu_test
            COMMAND glimmer_cuda_operation_queue_gpu_test --native-epochs)
        set_tests_properties(glimmer_cuda_native_epoch_frontend_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_native_epoch_default_frontend_gpu_test
            COMMAND glimmer_cuda_operation_queue_gpu_test --default-epochs)
        set_tests_properties(glimmer_cuda_native_epoch_default_frontend_gpu_test PROPERTIES TIMEOUT 60)
        glimmer_add_test_program(glimmer_cuda_driver_graph_registration_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/driver_graph_registration_gpu_test.cc"
            INCLUDES "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CUDAToolkit_INCLUDE_DIRS}"
            LIBRARIES glimmer_cuda_interceptor CUDA::cuda_driver Glimmer::project_options
                "${CMAKE_DL_LIBS}")
        set_tests_properties(glimmer_cuda_driver_graph_registration_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MEMORY_LIMIT_BYTES=8388608"
            TIMEOUT 60)
        # Ordinary CUDA ABI compatibility; no operation-queue activation.
        add_executable(glimmer_cuda_event_routing_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/event_routing_gpu_test.cc")
        target_link_libraries(glimmer_cuda_event_routing_gpu_test PRIVATE
            Glimmer::project_options CUDA::cuda_driver CUDA::cudart Threads::Threads "${CMAKE_DL_LIBS}")
        glimmer_enable_clang_tidy(glimmer_cuda_event_routing_gpu_test)
        foreach(event_mode IN ITEMS native observe enforce)
            add_test(NAME glimmer_cuda_event_routing_${event_mode}_gpu_test COMMAND glimmer_cuda_event_routing_gpu_test)
            set_tests_properties(glimmer_cuda_event_routing_${event_mode}_gpu_test PROPERTIES TIMEOUT 30)
            if(NOT event_mode STREQUAL "native")
                set_tests_properties(glimmer_cuda_event_routing_${event_mode}_gpu_test PROPERTIES
                    ENVIRONMENT "GLIMMER_SCHEDULER_MODE=${event_mode};LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>")
            endif()
        endforeach()
        glimmer_add_test_program(glimmer_cuda_cooperative_kernel_gpu_test
            SOURCES "${PROJECT_SOURCE_DIR}/tests/interceptor/cooperative_kernel_gpu_test.cc"
                internal/owned_kernel_arguments.cc
            INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}"
            LIBRARIES Glimmer::transform Glimmer::project_options Glimmer::core CUDA::cuda_driver
            TIMEOUT 60)

        add_executable(glimmer_cuda_cooperative_remote_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/control/cooperative_remote_process_test.cc"

            internal/owned_kernel_arguments.cc)
        target_compile_definitions(glimmer_cuda_cooperative_remote_gpu_test PRIVATE
            GLIMMER_TEST_COOPERATIVE_GPU=1)
        target_include_directories(glimmer_cuda_cooperative_remote_gpu_test PRIVATE
            "${PROJECT_SOURCE_DIR}/src/interceptor"
            "${PROJECT_SOURCE_DIR}/tests/interceptor")
        target_link_libraries(glimmer_cuda_cooperative_remote_gpu_test PRIVATE Glimmer::transform
            Glimmer::project_options Glimmer::control CUDA::cuda_driver)
        glimmer_enable_clang_tidy(glimmer_cuda_cooperative_remote_gpu_test)
        add_test(NAME glimmer_cuda_cooperative_remote_gpu_test
            COMMAND glimmer_cuda_cooperative_remote_gpu_test
                $<TARGET_FILE:glimmer_task_execution_service>)
        set_tests_properties(glimmer_cuda_cooperative_remote_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_cooperative_shared_memory_remote_gpu_test
            COMMAND glimmer_cuda_cooperative_remote_gpu_test
                $<TARGET_FILE:glimmer_task_execution_service>)
        set_tests_properties(glimmer_cuda_cooperative_shared_memory_remote_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_CONTROL_TRANSPORT=shared_memory" TIMEOUT 60)
        add_test(NAME glimmer_cuda_persistent_kernel_native_gpu_test
            COMMAND glimmer_cuda_persistent_kernel_gpu_test)
        set_tests_properties(glimmer_cuda_persistent_kernel_native_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_driver_cooperative_native_gpu_test
            COMMAND glimmer_cuda_driver_cooperative_gpu_test)
        set_tests_properties(glimmer_cuda_driver_cooperative_native_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_driver_cooperative_preload_gpu_test
            COMMAND ${CMAKE_COMMAND}
                -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_driver_cooperative_gpu_test>
                -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_driver.cmake")
        set_tests_properties(glimmer_cuda_driver_cooperative_preload_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60)
        foreach(cooperative_case IN ITEMS proc dynamic ptds cooperative-high release cross-context async-release async-release-ptds)
            add_test(NAME glimmer_cuda_driver_cooperative_${cooperative_case}_gpu_test
                COMMAND ${CMAKE_COMMAND}
                    -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_driver_cooperative_gpu_test>
                    -DAPPLICATION_ARGS=--${cooperative_case}
                    -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_driver.cmake")
            set_tests_properties(glimmer_cuda_driver_cooperative_${cooperative_case}_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
                TIMEOUT 60)
        endforeach()
        add_test(NAME glimmer_cuda_persistent_capture_native_gpu_test
            COMMAND glimmer_cuda_persistent_kernel_gpu_test --capture)
        set_tests_properties(glimmer_cuda_persistent_capture_native_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_persistent_capture_gpu_test
            COMMAND ${CMAKE_COMMAND}
                -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_persistent_kernel_gpu_test>
                -DSHOULD_CAPTURE=ON
                -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_driver.cmake")
        set_tests_properties(glimmer_cuda_persistent_capture_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_persistent_kernel_gpu_test
            COMMAND ${CMAKE_COMMAND}
                -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_persistent_kernel_gpu_test>
                -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_driver.cmake")
        set_tests_properties(glimmer_cuda_persistent_kernel_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_persistent_kernel_enforce_gpu_test
            COMMAND ${CMAKE_COMMAND}
                -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_persistent_kernel_gpu_test>
                -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_driver.cmake")
        set_tests_properties(glimmer_cuda_persistent_kernel_enforce_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=2;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60)
        add_test(NAME glimmer_cuda_persistent_kernel_disabled_gpu_test
            COMMAND glimmer_cuda_persistent_kernel_gpu_test)
        set_tests_properties(glimmer_cuda_persistent_kernel_disabled_gpu_test PROPERTIES
            ENVIRONMENT "GLIMMER_PERSISTENT_WORKER_BLOCKS=invalid;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60)
        enable_language(CUDA)

        target_sources(glimmer_cuda_runtime_capture_owner_gpu_test PRIVATE
            "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_capture_kernel_fixture.cu")
        target_compile_definitions(glimmer_cuda_runtime_capture_owner_gpu_test PRIVATE
            GLIMMER_HAS_RUNTIME_CAPTURE_KERNEL
            GLIMMER_CUOBJDUMP_PATH="${GLIMMER_CUOBJDUMP_EXECUTABLE}")
        set_target_properties(glimmer_cuda_runtime_capture_owner_gpu_test PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared
            CUDA_ARCHITECTURES "70-real;70-virtual" LINKER_LANGUAGE CXX)

        add_executable(glimmer_cuda_runtime_cooperative_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/runtime_cooperative_gpu_test.cu")
        target_link_libraries(glimmer_cuda_runtime_cooperative_gpu_test PRIVATE
            CUDA::cudart Threads::Threads "${CMAKE_DL_LIBS}")
        set_target_properties(glimmer_cuda_runtime_cooperative_gpu_test PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared
            CUDA_ARCHITECTURES "70-real;70-virtual")
        add_test(NAME glimmer_cuda_runtime_cooperative_native_gpu_test
            COMMAND glimmer_cuda_runtime_cooperative_gpu_test)
        set_tests_properties(glimmer_cuda_runtime_cooperative_native_gpu_test PROPERTIES TIMEOUT 60)
        add_test(NAME glimmer_cuda_runtime_cooperative_fresh_function_native_gpu_test
            COMMAND glimmer_cuda_runtime_cooperative_gpu_test --fresh-function)
        set_tests_properties(glimmer_cuda_runtime_cooperative_fresh_function_native_gpu_test PROPERTIES TIMEOUT 60)
        foreach(graph_case IN ITEMS graph graph-ptds graph-driver graph-driver-ptds graph-null-ptds graph-flags graph-driver-flags)
            add_test(NAME glimmer_cuda_runtime_cooperative_${graph_case}_native_gpu_test
                COMMAND glimmer_cuda_runtime_cooperative_gpu_test --${graph_case})
            set_tests_properties(glimmer_cuda_runtime_cooperative_${graph_case}_native_gpu_test
                PROPERTIES TIMEOUT 60)
        endforeach()
        foreach(runtime_case IN ITEMS preload ptds native-high release compiler thread async-release async-release-ptds fresh-function graph graph-ptds graph-driver graph-driver-ptds graph-null-ptds graph-flags graph-driver-flags)
            add_test(NAME glimmer_cuda_runtime_cooperative_${runtime_case}_gpu_test
                COMMAND ${CMAKE_COMMAND}
                    -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_runtime_cooperative_gpu_test>
                    -DAPPLICATION_ARGS=--${runtime_case}
                    -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_runtime.cmake")
            set_tests_properties(glimmer_cuda_runtime_cooperative_${runtime_case}_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
                TIMEOUT 60)
        endforeach()

        foreach(api_layer IN ITEMS driver runtime)
            foreach(release_case IN ITEMS async-release async-release-ptds)
                add_test(NAME glimmer_cuda_${api_layer}_cooperative_${release_case}_native_gpu_test
                    COMMAND glimmer_cuda_${api_layer}_cooperative_gpu_test --${release_case})
                set_tests_properties(glimmer_cuda_${api_layer}_cooperative_${release_case}_native_gpu_test
                    PROPERTIES TIMEOUT 60)
            endforeach()
        endforeach()

        find_program(GLIMMER_COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer
            HINTS "${CUDAToolkit_BIN_DIR}")
        if(GLIMMER_COMPUTE_SANITIZER_EXECUTABLE)
            foreach(api_layer IN ITEMS driver runtime)
                foreach(device_tool IN ITEMS memcheck synccheck)
                    if(api_layer STREQUAL "runtime")
                        add_test(NAME glimmer_cuda_runtime_cooperative_fresh_function_${device_tool}_gpu_test
                            COMMAND ${CMAKE_COMMAND}
                                -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_runtime_cooperative_gpu_test>
                                -DAPPLICATION_ARGS=--fresh-function
                                -DSANITIZER=${GLIMMER_COMPUTE_SANITIZER_EXECUTABLE}
                                -DINTERCEPTOR=$<TARGET_FILE:glimmer_cuda_interceptor>
                                -DTOOL=${device_tool} -DAPI_LAYER=runtime
                                -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_tool.cmake")
                        set_tests_properties(glimmer_cuda_runtime_cooperative_fresh_function_${device_tool}_gpu_test PROPERTIES
                            ENVIRONMENT "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608"
                            TIMEOUT 60)
                    endif()
                    add_test(NAME glimmer_cuda_${api_layer}_cooperative_${device_tool}_gpu_test
                        COMMAND ${CMAKE_COMMAND}
                            -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_${api_layer}_cooperative_gpu_test>
                            -DSANITIZER=${GLIMMER_COMPUTE_SANITIZER_EXECUTABLE}
                            -DINTERCEPTOR=$<TARGET_FILE:glimmer_cuda_interceptor>
                            -DTOOL=${device_tool} -DAPI_LAYER=${api_layer}
                            -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_tool.cmake")
                    set_tests_properties(glimmer_cuda_${api_layer}_cooperative_${device_tool}_gpu_test PROPERTIES
                        ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=$<IF:$<STREQUAL:${api_layer},runtime>,1,0>;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608"
                        TIMEOUT 60)
                    add_test(NAME glimmer_cuda_${api_layer}_cooperative_release_${device_tool}_gpu_test
                        COMMAND ${CMAKE_COMMAND}
                            -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_${api_layer}_cooperative_gpu_test>
                            -DAPPLICATION_ARGS=--async-release
                            -DSANITIZER=${GLIMMER_COMPUTE_SANITIZER_EXECUTABLE}
                            -DINTERCEPTOR=$<TARGET_FILE:glimmer_cuda_interceptor>
                            -DTOOL=${device_tool} -DAPI_LAYER=${api_layer}
                            -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_cooperative_tool.cmake")
                    set_tests_properties(glimmer_cuda_${api_layer}_cooperative_release_${device_tool}_gpu_test PROPERTIES
                        ENVIRONMENT "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1;GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=$<IF:$<STREQUAL:${api_layer},runtime>,1,0>;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_SCHEDULER_POLICY=priority;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608"
                        TIMEOUT 60)
                endforeach()
            endforeach()
        else()
            message(STATUS "Public cooperative device checks unavailable: compute-sanitizer not found")
        endif()

        add_executable(glimmer_cuda_persistent_runtime_gpu_test
            "${PROJECT_SOURCE_DIR}/tests/interceptor/persistent_runtime_gpu_test.cu")
        target_link_libraries(glimmer_cuda_persistent_runtime_gpu_test PRIVATE CUDA::cudart Threads::Threads "${CMAKE_DL_LIBS}")
        target_compile_definitions(glimmer_cuda_persistent_runtime_gpu_test PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
        set_target_properties(glimmer_cuda_persistent_runtime_gpu_test PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared
            CUDA_ARCHITECTURES "70-real;70-virtual")
        add_test(NAME glimmer_cuda_persistent_runtime_native_gpu_test
            COMMAND glimmer_cuda_persistent_runtime_gpu_test --capture)
        set_tests_properties(glimmer_cuda_persistent_runtime_native_gpu_test PROPERTIES TIMEOUT 60)
        foreach(configuration enabled enforce capture enforce_capture)
            add_test(NAME glimmer_cuda_persistent_runtime_${configuration}_gpu_test
                COMMAND ${CMAKE_COMMAND}
                    -DAPPLICATION=$<TARGET_FILE:glimmer_cuda_persistent_runtime_gpu_test>
                    -DSHOULD_CAPTURE=$<OR:$<STREQUAL:${configuration},capture>,$<STREQUAL:${configuration},enforce_capture>>
                    -P "${PROJECT_SOURCE_DIR}/tests/interceptor/check_persistent_runtime.cmake")
            set_tests_properties(glimmer_cuda_persistent_runtime_${configuration}_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_PERSISTENT_RUNTIME=1;GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
                FAIL_REGULAR_EXPRESSION "cleanup failed;output mismatch" TIMEOUT 60)
        endforeach()
        set_property(TEST glimmer_cuda_persistent_runtime_enforce_gpu_test APPEND PROPERTY ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=8388608" "GLIMMER_SCHEDULER_MODE=enforce" "GLIMMER_MAX_CONCURRENT_KERNELS=2")
        set_property(TEST glimmer_cuda_persistent_runtime_enforce_capture_gpu_test APPEND PROPERTY ENVIRONMENT
            "GLIMMER_MEMORY_LIMIT_BYTES=8388608" "GLIMMER_SCHEDULER_MODE=enforce" "GLIMMER_MAX_CONCURRENT_KERNELS=2")
        foreach(setting 0 invalid)
            add_test(NAME glimmer_cuda_persistent_runtime_${setting}_gpu_test
                COMMAND glimmer_cuda_persistent_runtime_gpu_test --capture)
            set_tests_properties(glimmer_cuda_persistent_runtime_${setting}_gpu_test PROPERTIES
                ENVIRONMENT "GLIMMER_PERSISTENT_RUNTIME=${setting};GLIMMER_PERSISTENT_WORKER_BLOCKS=3;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
                FAIL_REGULAR_EXPRESSION "persistent_runtime logical_blocks;cleanup failed" TIMEOUT 60)
        endforeach()

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

        add_test(NAME glimmer_cuda_interceptor_kernel_enforce_gpu_test
            COMMAND glimmer_cuda_interceptor_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_kernel_enforce_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
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
        target_link_libraries(glimmer_cuda_interceptor_runtime_kernel_gpu_test PRIVATE "${CMAKE_DL_LIBS}")
        add_test(NAME glimmer_cuda_extended_launch_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_kernel_gpu_test)
        set_tests_properties(glimmer_cuda_extended_launch_gpu_test PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "GLIMMER_TEST_EXTENDED=1;GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_TRACE_KERNEL_LAUNCHES=1;GLIMMER_MAX_CONCURRENT_KERNELS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            PASS_REGULAR_EXPRESSION "api=cudaLaunchKernelExC_ptsz"
            FAIL_REGULAR_EXPRESSION "test assertion failed:")
        set_tests_properties(glimmer_cuda_interceptor_runtime_kernel_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=observe;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_kernel_enforce_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_kernel_enforce_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_graph_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_graph_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=observe;GLIMMER_TRACE_KERNEL_LAUNCHES=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
            PASS_REGULAR_EXPRESSION
                "observed CUDA graph launch #2 api=cudaGraphLaunch stream="
            FAIL_REGULAR_EXPRESSION "test assertion failed:"
            TIMEOUT 60
        )

        add_test(NAME glimmer_cuda_interceptor_runtime_kernel_batched_enforce_gpu_test
            COMMAND glimmer_cuda_interceptor_runtime_kernel_gpu_test
        )
        set_tests_properties(glimmer_cuda_interceptor_runtime_kernel_batched_enforce_gpu_test PROPERTIES
            ENVIRONMENT
                "GLIMMER_MEMORY_LIMIT_BYTES=8388608;GLIMMER_SCHEDULER_MODE=enforce;GLIMMER_MAX_CONCURRENT_KERNELS=1;GLIMMER_SCHEDULER_BATCH_SIZE=2;GLIMMER_TEST_BATCHED=1;LD_PRELOAD=$<TARGET_FILE:glimmer_cuda_interceptor>"
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
