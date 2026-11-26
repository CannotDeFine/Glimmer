include_guard(GLOBAL)

function(glimmer_validate_cuda_toolchain)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        return()
    endif()

    if(CMAKE_VERSION VERSION_LESS "3.25.2" OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR "CUDA C++20 with NVIDIA nvcc requires CMake 3.25.2+ and nvcc 12.0+.")
    endif()

    string(REGEX MATCH "^[0-9]+\\.[0-9]+" compiler_series "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" toolkit_series "${CUDAToolkit_VERSION}")
    if(NOT compiler_series OR NOT toolkit_series)
        message(FATAL_ERROR "Cannot compare NVIDIA CUDA compiler and Toolkit versions.")
    elseif(NOT compiler_series STREQUAL toolkit_series)
        message(FATAL_ERROR
            "CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION} and Toolkit ${CUDAToolkit_VERSION} do not match. "
            "Select compatible CMAKE_CUDA_COMPILER and CUDAToolkit_ROOT in a fresh build directory.")
    endif()
endfunction()
