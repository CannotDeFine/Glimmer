if(NOT EXISTS "${TRANSFORMER}" OR NOT EXISTS "${NVCC}" OR NOT IS_ABSOLUTE "${OUTPUT_DIR}")
    message(FATAL_ERROR "PTX compiler test requires built transformer, nvcc and an absolute output")
endif()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
foreach(fixture IN ITEMS compiler calls atomic warp aggregate)
    execute_process(
        COMMAND "${TRANSFORMER}" "--emit-${fixture}" "${OUTPUT_DIR}/${fixture}.ptx"
        RESULT_VARIABLE generated
        OUTPUT_VARIABLE generator_output ERROR_VARIABLE generator_error
        TIMEOUT 10)
    if(NOT generated STREQUAL "0")
        message(FATAL_ERROR "${fixture} transformation failed: ${generated}\n${generator_output}\n${generator_error}")
    endif()
    execute_process(
        COMMAND "${NVCC}" --cubin --gpu-architecture=sm_70
            "${OUTPUT_DIR}/${fixture}.ptx" -o "${OUTPUT_DIR}/${fixture}.cubin"
        RESULT_VARIABLE compiled
        OUTPUT_VARIABLE compiler_output ERROR_VARIABLE compiler_error
        TIMEOUT 20)
    if(NOT compiled STREQUAL "0")
        message(FATAL_ERROR "${fixture} native/eligible-family assembly failed: ${compiled}\n${compiler_output}\n${compiler_error}")
    endif()
endforeach()
execute_process(
    COMMAND "${NVCC}" --ptx --gpu-architecture=compute_70 --generate-line-info
        "${CMAKE_CURRENT_LIST_DIR}/compiler_ptx_input.cu" -o "${OUTPUT_DIR}/nvcc.ptx"
    RESULT_VARIABLE compiled
    OUTPUT_VARIABLE compiler_output ERROR_VARIABLE compiler_error
    TIMEOUT 30)
if(NOT compiled STREQUAL "0")
    message(FATAL_ERROR "nvcc compiler input failed: ${compiled}\n${compiler_output}\n${compiler_error}")
endif()
file(READ "${OUTPUT_DIR}/nvcc.ptx" native_source)
foreach(required IN ITEMS ".local" "st.local." "ld.local." ".loc" ".file" ".maxntid"
        ".minnctapersm" ".func" "call.uni" "compiler_leaf" "compiler_middle")
    string(FIND "${native_source}" "${required}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "nvcc input no longer exercises required structure: ${required}")
    endif()
endforeach()
execute_process(
    COMMAND "${TRANSFORMER}" --transform-compiler "${OUTPUT_DIR}/nvcc.ptx"
        "${OUTPUT_DIR}/nvcc-transformed.ptx"
    RESULT_VARIABLE generated
    OUTPUT_VARIABLE generator_output ERROR_VARIABLE generator_error
    TIMEOUT 10)
if(NOT generated STREQUAL "0")
    message(FATAL_ERROR "Actual nvcc PTX failed transformation: ${generated}\n${generator_output}\n${generator_error}")
endif()
execute_process(
    COMMAND "${NVCC}" --cubin --gpu-architecture=sm_70
        "${OUTPUT_DIR}/nvcc-transformed.ptx" -o "${OUTPUT_DIR}/nvcc-transformed.cubin"
    RESULT_VARIABLE compiled
    OUTPUT_VARIABLE compiler_output ERROR_VARIABLE compiler_error
    TIMEOUT 30)
if(NOT compiled STREQUAL "0")
    message(FATAL_ERROR "Transformed nvcc input failed assembly: ${compiled}\n${compiler_output}\n${compiler_error}")
endif()
execute_process(
    COMMAND "${NVCC}" --ptx --gpu-architecture=compute_70 --generate-line-info
        "${CMAKE_CURRENT_LIST_DIR}/compiler_warp_input.cu" -o "${OUTPUT_DIR}/nvcc-warp.ptx"
    RESULT_VARIABLE compiled
    OUTPUT_VARIABLE compiler_output ERROR_VARIABLE compiler_error
    TIMEOUT 30)
if(NOT compiled STREQUAL "0")
    message(FATAL_ERROR "nvcc warp input failed: ${compiled}\n${compiler_output}\n${compiler_error}")
endif()
file(READ "${OUTPUT_DIR}/nvcc-warp.ptx" native_source)
foreach(required IN ITEMS ".shared" "bar.sync" "shfl.sync" "vote.sync" "match.any.sync" "popc.")
    string(FIND "${native_source}" "${required}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "nvcc warp input no longer exercises required instruction: ${required}")
    endif()
endforeach()
foreach(space IN ITEMS shared global)
    if(NOT native_source MATCHES "(atom|red)\\.${space}\\.add\\.u32")
        message(FATAL_ERROR "nvcc warp input no longer exercises ${space} atomic addition")
    endif()
endforeach()
execute_process(
    COMMAND "${TRANSFORMER}" --transform-warp "${OUTPUT_DIR}/nvcc-warp.ptx"
        "${OUTPUT_DIR}/nvcc-warp-transformed.ptx"
    RESULT_VARIABLE generated
    OUTPUT_VARIABLE generator_output ERROR_VARIABLE generator_error
    TIMEOUT 10)
if(NOT generated STREQUAL "0")
    message(FATAL_ERROR "Actual nvcc warp PTX failed transformation: ${generated}\n${generator_output}\n${generator_error}")
endif()
execute_process(
    COMMAND "${NVCC}" --cubin --gpu-architecture=sm_70
        "${OUTPUT_DIR}/nvcc-warp-transformed.ptx" -o "${OUTPUT_DIR}/nvcc-warp-transformed.cubin"
    RESULT_VARIABLE compiled
    OUTPUT_VARIABLE compiler_output ERROR_VARIABLE compiler_error
    TIMEOUT 30)
if(NOT compiled STREQUAL "0")
    message(FATAL_ERROR "Transformed nvcc warp input failed assembly: ${compiled}\n${compiler_output}\n${compiler_error}")
endif()
message(STATUS "Seven native inputs and eligible families assembled; warp inputs reject static; GPU execution was not performed")
