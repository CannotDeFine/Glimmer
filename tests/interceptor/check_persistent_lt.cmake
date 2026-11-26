execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "LD_PRELOAD=${INTERCEPTOR}"
    GLIMMER_PERSISTENT_BLAS=1
    GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1
    GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1
    GLIMMER_SCHEDULER_MODE=enforce
    GLIMMER_SCHEDULER_POLICY=priority
    GLIMMER_MEMORY_LIMIT_BYTES=4294967296
    GLIMMER_TRACE_KERNEL_LAUNCHES=1
    "${APPLICATION}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 120)
message("${output}${errors}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Public persistent Lt failed: ${result}")
endif()
string(REGEX MATCHALL "persistent_lt_bias accepted=1" accepted "${errors}")
list(LENGTH accepted count)
if(NOT count EQUAL 11 OR NOT output MATCHES "public_lt cleanup=1")
    message(FATAL_ERROR "Expected 11 accepted Lt GEMMs, nine native fallbacks and checked cleanup")
endif()
string(REPLACE "\n" ";" lines "${errors}")
set(inside FALSE)
set(cases 0)
foreach(line IN LISTS lines)
    if(line MATCHES "^public_lt_begin expected=([01])$")
        if(inside)
            message(FATAL_ERROR "Nested public Lt case")
        endif()
        set(inside TRUE)
        set(expected "${CMAKE_MATCH_1}")
        set(actual 0)
    elseif(line MATCHES "persistent_lt_bias accepted=1")
        if(NOT inside)
            message(FATAL_ERROR "Lt replacement outside a public call")
        endif()
        math(EXPR actual "${actual} + 1")
    elseif(line STREQUAL "public_lt_end")
        if(NOT inside OR NOT actual EQUAL expected)
            message(FATAL_ERROR "Public Lt case selected ${actual}; expected ${expected}")
        endif()
        set(inside FALSE)
        math(EXPR cases "${cases} + 1")
    endif()
endforeach()
if(inside OR NOT cases EQUAL 20)
    message(FATAL_ERROR "Incomplete per-call Lt selection evidence")
endif()
