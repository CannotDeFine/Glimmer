include_guard(GLOBAL)

option(GLIMMER_OPTIMIZE_EXECUTION_HOST
    "Compile execution host targets with -O2 while retaining Debug assertions" OFF)

function(glimmer_configure_execution_host_optimization)
    if(NOT GLIMMER_OPTIMIZE_EXECUTION_HOST)
        return()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "GLIMMER_OPTIMIZE_EXECUTION_HOST requires GCC or Clang")
    endif()
    if(CMAKE_CONFIGURATION_TYPES OR NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "GLIMMER_OPTIMIZE_EXECUTION_HOST requires a single-configuration Debug build")
    endif()
    if(NOT TARGET glimmer_execution_service AND NOT TARGET glimmer_execution_interceptor)
        message(FATAL_ERROR "GLIMMER_OPTIMIZE_EXECUTION_HOST requires an execution service or frontend target")
    endif()

    # PRIVATE options leave consumers, SDK targets and device compilation alone.
    # Compile shared production dependencies consistently, not just the entry points.
    foreach(target IN ITEMS glimmer_core glimmer_control glimmer_backend glimmer_transform
                            glimmer_cuda_backend glimmer_execution_service_runtime
                            glimmer_execution_service glimmer_execution_interceptor)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-O2>")
        endif()
    endforeach()
endfunction()
