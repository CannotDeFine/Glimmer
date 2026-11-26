include("${CMAKE_CURRENT_LIST_DIR}/CTestDefaults.cmake")
option(GLIMMER_SHARE_EXECUTION_TEST_RUNTIME
    "Share the private SDK9 CPU fixture runtime within one build configuration" ON)
include("${CMAKE_CURRENT_LIST_DIR}/ExecutionTestRuntime.cmake")
