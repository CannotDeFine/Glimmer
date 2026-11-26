cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/BuildStorage.cmake")
    glimmer_check_build_storage()
    project(GlimmerGemmComparison LANGUAGES CXX CUDA)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The GEMM experiment supports Linux only")
endif()

set(GLIMMER_EXPERIMENT_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass"
    CACHE PATH "Pinned CUTLASS checkout; standalone historical input override")
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CutlassDependency.cmake")
glimmer_add_cutlass_headers("${GLIMMER_EXPERIMENT_CUTLASS_ROOT}")
set(cutlass_revision "${GLIMMER_CUTLASS_REVISION}")

find_package(CUDAToolkit REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CudaToolchain.cmake")
glimmer_validate_cuda_toolchain()
add_executable(glimmer_gemm_comparison gemm_comparison.cu)
target_link_libraries(glimmer_gemm_comparison PRIVATE
    Glimmer::cutlass_headers CUDA::cudart CUDA::cublas)
target_compile_options(glimmer_gemm_comparison PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
option(GLIMMER_GEMM_ENABLE_SANITIZERS "Instrument the standalone host code" OFF)
if(GLIMMER_GEMM_ENABLE_SANITIZERS)
    target_compile_options(glimmer_gemm_comparison PRIVATE
        "-Xcompiler=-fsanitize=address,-fsanitize=undefined,-fno-omit-frame-pointer,-fno-pie"
    )
    target_link_options(glimmer_gemm_comparison PRIVATE
        "-fsanitize=address,undefined" "-no-pie"
    )
endif()
set_target_properties(glimmer_gemm_comparison PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 70
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/gemm_comparison.cu" source_sha256)
target_compile_definitions(glimmer_gemm_comparison PRIVATE
    "GLIMMER_EXPERIMENT_SOURCE_SHA256=\"${source_sha256}\""
)
set(host_sanitizers_json false)
if(GLIMMER_GEMM_ENABLE_SANITIZERS)
    set(host_sanitizers_json true)
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-inputs.json"
    "{\"source_sha256\":\"${source_sha256}\",\"cutlass_revision\":\"${cutlass_revision}\",\"host_sanitizers\":${host_sanitizers_json}}\n"
)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CTestDefaults.cmake")
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CUDAToolkit_NVCC_EXECUTABLE)
        add_test(NAME glimmer_gemm_cuda_compiler_selection_test
            COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/build/cuda_compiler_selection_test.py"
                --cmake "${CMAKE_COMMAND}" --toolkit-root "${CUDAToolkit_BIN_DIR}/.."
                --expected-nvcc "${CUDAToolkit_NVCC_EXECUTABLE}"
                --source standalone --cutlass-root "${GLIMMER_EXPERIMENT_CUTLASS_ROOT}")
        set_tests_properties(glimmer_gemm_cuda_compiler_selection_test PROPERTIES TIMEOUT 120)
    endif()
    add_test(NAME glimmer_gemm_comparison_evidence_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/examples/gemm_comparison_test.py"
    )
    add_test(NAME glimmer_gemm_utilization_analysis_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/examples/gemm_utilization_analysis_test.py"
    )
    add_test(NAME glimmer_gemm_comparison_invalid_option
        COMMAND glimmer_gemm_comparison --invalid
    )
    set_tests_properties(glimmer_gemm_comparison_invalid_option PROPERTIES WILL_FAIL TRUE)
    add_test(NAME glimmer_cutlass_dependency_test
        COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/build/cutlass_dependency_test.py"
            --cmake "${CMAKE_COMMAND}"
    )
endif()
