if(NOT DEFINED EXECUTABLE OR NOT DEFINED DRIVER)
    message(FATAL_ERROR "Explicit existing CPU fixture and driver are required")
endif()
execute_process(COMMAND "${EXECUTABLE}" "${DRIVER}" RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE diagnostics TIMEOUT 25)
if(NOT status EQUAL 0 OR
        NOT output MATCHES "Public pageable staging lifetime and asynchronous cleanup checks passed" OR
        NOT diagnostics MATCHES "Glimmer execution copy audit: v=1, pid=[0-9]+" OR
        NOT diagnostics MATCHES "operation=22, session=[0-9]+, request=[0-9]+" OR
        NOT diagnostics MATCHES "operation=25, session=[0-9]+, request=[0-9]+")
    message(FATAL_ERROR "Actual opt-in copy diagnostics or original CPU/API checks failed: ${status}: ${output}${diagnostics}")
endif()
