foreach(required IN ITEMS APPLICATION SANITIZER INTERCEPTOR)
    if(NOT DEFINED ${required} OR NOT EXISTS "${${required}}")
        message(FATAL_ERROR "Public device-check fixture unavailable: ${required}")
    endif()
endforeach()
if(NOT TOOL STREQUAL "memcheck" AND NOT TOOL STREQUAL "synccheck")
    message(FATAL_ERROR "Unsupported device-check tool")
endif()
if(API_LAYER STREQUAL "runtime")
    set(expected_rounds 2)
elseif(API_LAYER STREQUAL "driver")
    set(expected_rounds 1)
else()
    message(FATAL_ERROR "Unknown ordinary application API layer")
endif()
execute_process(COMMAND "${SANITIZER}" --tool "${TOOL}" --report-api-errors no
    --target-processes application-only --launch-timeout 10
    --preload-library "${INTERCEPTOR}" --error-exitcode 1 "${APPLICATION}" ${APPLICATION_ARGS}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE diagnostic TIMEOUT 45)
set(all_output "${output}${diagnostic}")
message("${all_output}")
if(NOT result STREQUAL "0" OR
   NOT all_output MATCHES "${API_LAYER}_cooperative_transparency status=ok" OR
   NOT all_output MATCHES "ERROR SUMMARY: 0 errors")
    message(FATAL_ERROR "Public ${API_LAYER} ${TOOL} failed: ${result}")
endif()
string(REGEX MATCHALL "cooperative_context grants=3 retreat=1 suspended=1 resumed=1 completed=2 failed=0"
    handoffs "${all_output}")
list(LENGTH handoffs handoff_count)
if(NOT handoff_count EQUAL expected_rounds)
    message(FATAL_ERROR "Actual stop/resume was not proven under ${TOOL}")
endif()
