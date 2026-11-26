if(NOT DEFINED APPLICATION OR NOT EXISTS "${APPLICATION}")
    message(FATAL_ERROR "Runtime fixture is unavailable")
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
if(NOT application_result STREQUAL "0")
    message(FATAL_ERROR "Runtime fixture failed: ${application_result}")
endif()
if(NOT application_stdout MATCHES "persistent_runtime_transparency status=ok")
    message(FATAL_ERROR "Runtime fixture did not report successful validation")
endif()
if(NOT application_stderr MATCHES "persistent_runtime logical_blocks=257 worker_blocks=3")
    message(FATAL_ERROR "Runtime fixture stayed native: reduced PTB workers were not observed")
endif()
if(NOT application_stderr MATCHES "persistent_runtime logical_blocks=105 worker_blocks=3")
    message(FATAL_ERROR "Shared 3D Runtime fixture stayed native: block-static workers were not observed")
endif()
if(application_stderr MATCHES "cleanup failed")
    message(FATAL_ERROR "Private Runtime library cleanup failed")
endif()
