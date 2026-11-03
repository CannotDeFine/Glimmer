if(NOT EXISTS "${TRANSFORMER}" OR NOT EXISTS "${NVCC}" OR NOT IS_ABSOLUTE "${OUTPUT_DIR}")
    message(FATAL_ERROR "Uniform-loop test requires built transformer, nvcc and absolute output")
endif()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
execute_process(
    COMMAND "${TRANSFORMER}" --emit "${OUTPUT_DIR}/uniform-loop.ptx"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Shared uniform-loop transformation failed: ${result}\n${output}\n${error}")
endif()
execute_process(
    COMMAND "${TRANSFORMER}" --emit-reconvergent "${OUTPUT_DIR}/reconvergent-loop.ptx"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Shared reconvergent-loop transformation failed: ${result}\n${output}\n${error}")
endif()
execute_process(
    COMMAND "${NVCC}" --ptx --gpu-architecture=compute_70 --generate-line-info
        "${CMAKE_CURRENT_LIST_DIR}/compiler_uniform_loop_input.cu"
        -o "${OUTPUT_DIR}/nvcc-uniform-loop.ptx"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Actual nvcc loop compilation failed: ${result}\n${output}\n${error}")
endif()
file(READ "${OUTPUT_DIR}/nvcc-uniform-loop.ptx" source)
foreach(required IN ITEMS "bar.sync" "shfl.sync.bfly" "ld.param.u32" "bra" "compiler_uniform_loop")
    string(FIND "${source}" "${required}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "nvcc uniform-loop input lost required construct: ${required}")
    endif()
endforeach()
if(NOT source MATCHES "(atom|red)\\.global\\.add\\.u32")
    message(FATAL_ERROR "nvcc uniform-loop input lost its observable atomic update")
endif()
execute_process(
    COMMAND "${TRANSFORMER}" --transform "${OUTPUT_DIR}/nvcc-uniform-loop.ptx"
        "${OUTPUT_DIR}/nvcc-uniform-loop-transformed.ptx"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Actual nvcc loop transformation failed: ${result}\n${output}\n${error}")
endif()
foreach(input IN ITEMS uniform-loop reconvergent-loop nvcc-uniform-loop-transformed)
    execute_process(
        COMMAND "${NVCC}" --cubin --gpu-architecture=sm_70
            "${OUTPUT_DIR}/${input}.ptx" -o "${OUTPUT_DIR}/${input}.cubin"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "${input} original/cooperative assembly failed: ${result}\n${output}\n${error}")
    endif()
endforeach()
message(STATUS "Three repeated-collective inputs and four cooperative families assembled; no GPU execution")
