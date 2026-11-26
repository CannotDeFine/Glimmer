include("${CMAKE_CURRENT_LIST_DIR}/CTestDefaults.cmake")
option(GLIMMER_SHARE_EXECUTION_TEST_RUNTIME
    "Share the private SDK9 CPU fixture runtime within one build configuration" ON)
include("${CMAKE_CURRENT_LIST_DIR}/ExecutionTestRuntime.cmake")

# glimmer_add_test_program(<name>
#     SOURCES <source>...
#     LIBRARIES <library>...
#     [INCLUDES <directory>...]
#     [DEFINITIONS <definition>...]
#     [LINK_OPTIONS <option>...]
#     [TIMEOUT <seconds>])
#
# Builds one test executable with clang-tidy enabled and registers a CTest test
# of the same name that runs it without arguments. All usage requirements are
# PRIVATE. Register tests that need arguments, environments or other properties
# with add_test and set_tests_properties after this call.
function(glimmer_add_test_program name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "TIMEOUT"
        "SOURCES;LIBRARIES;INCLUDES;DEFINITIONS;LINK_OPTIONS")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_SOURCES OR NOT arg_LIBRARIES)
        message(FATAL_ERROR "glimmer_add_test_program(${name}) requires SOURCES and LIBRARIES; "
            "unexpected: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    add_executable(${name} ${arg_SOURCES})
    if(arg_INCLUDES)
        target_include_directories(${name} PRIVATE ${arg_INCLUDES})
    endif()
    if(arg_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${arg_DEFINITIONS})
    endif()
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
    if(arg_LINK_OPTIONS)
        target_link_options(${name} PRIVATE ${arg_LINK_OPTIONS})
    endif()
    glimmer_enable_clang_tidy(${name})
    add_test(NAME ${name} COMMAND ${name})
    if(DEFINED arg_TIMEOUT)
        set_tests_properties(${name} PROPERTIES TIMEOUT ${arg_TIMEOUT})
    endif()
endfunction()
