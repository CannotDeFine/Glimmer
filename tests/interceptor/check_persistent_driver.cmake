if(NOT DEFINED APPLICATION OR NOT EXISTS "${APPLICATION}")
    message(FATAL_ERROR "Driver fixture is unavailable")
endif()
set(application_args)
if(SHOULD_CAPTURE)
    list(APPEND application_args --capture)
endif()
execute_process(
    COMMAND "${APPLICATION}" ${application_args}
    RESULT_VARIABLE application_result
    OUTPUT_VARIABLE application_stdout
    ERROR_VARIABLE application_stderr
    TIMEOUT 45)
message("${application_stdout}${application_stderr}")
if(NOT application_result STREQUAL "0" OR
   NOT application_stdout MATCHES "persistent_transparency status=ok")
    message(FATAL_ERROR "Driver fixture failed: ${application_result}")
endif()
if(NOT application_stderr MATCHES "persistent_kernel logical_blocks=257 worker_blocks=3" OR
   NOT application_stderr MATCHES "persistent_kernel logical_blocks=105 worker_blocks=3")
    message(FATAL_ERROR "Driver fixture stayed native: independent and shared 3D PTB workers are required")
endif()
