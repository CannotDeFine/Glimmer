# This is deliberately not a production library or a cross-build object cache.
function(glimmer_create_execution_test_runtime)
    if(NOT BUILD_TESTING OR NOT GLIMMER_SHARE_EXECUTION_TEST_RUNTIME
       OR NOT GLIMMER_CUDNN_ABI_PROFILE STREQUAL "9")
        return()
    endif()
    set_target_properties(glimmer_execution_service_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(glimmer_execution_test_runtime SHARED
        "${PROJECT_SOURCE_DIR}/tests/support/execution_test_runtime_anchor.cc")
    set_target_properties(glimmer_execution_test_runtime PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/test-support")
    # Ordered flags work with the project's CMake3.20 minimum. Whole-archive is
    # scoped to the existing runtime; its private dependencies resolve normally.
    target_link_libraries(glimmer_execution_test_runtime PRIVATE
        "-Wl,--whole-archive" glimmer_execution_service_runtime "-Wl,--no-whole-archive"
        Glimmer::project_options)
    glimmer_enable_clang_tidy(glimmer_execution_test_runtime)
endfunction()

function(glimmer_share_execution_fixture application)
    # Explicitly reviewed fixtures only. DSO-internal calls cannot honor an
    # executable's --wrap options. Never infer eligibility from naming alone.
    set(_eligible
        glimmer_execution_dnn_convolution_test
        glimmer_execution_dnn_pooling_test
        glimmer_execution_dnn_lrn_test
        glimmer_execution_dnn_reduction_test
        glimmer_execution_dnn_batch_norm_test
        glimmer_execution_dnn_dropout_test
        glimmer_execution_dnn_op_tensor_test
        glimmer_execution_dnn_reorder_test
        glimmer_execution_dnn_library_info_test
        glimmer_execution_dnn_backward_convolution_test
        glimmer_execution_dnn_folded_convolution_test
        glimmer_execution_dnn_sequence_test
        glimmer_execution_dnn_rnn_attention_test
        glimmer_execution_dnn_rnn_query_test
        glimmer_execution_dnn_rnn_compute_test
        glimmer_execution_dnn_rnn_weight_params_test
        glimmer_execution_dnn_attn_weights_test
        glimmer_execution_dnn_attn_compute_test
        glimmer_execution_dnn_backend_graph_test)
    if(NOT application IN_LIST _eligible)
        return()
    endif()
    # One aggregate target avoids Make rewalking the common dependency graph
    # separately for nineteen --target arguments. It is not an extra build tree.
    if(NOT TARGET glimmer_execution_dnn_cpu_fixtures)
        add_custom_target(glimmer_execution_dnn_cpu_fixtures)
    endif()
    add_dependencies(glimmer_execution_dnn_cpu_fixtures "${application}")
    if(NOT TARGET glimmer_execution_test_runtime)
        return()
    endif()
    get_target_property(_options "${application}" LINK_OPTIONS)
    get_target_property(_libraries "${application}" LINK_LIBRARIES)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
    set(_flags "${_options};${_libraries};${CMAKE_EXE_LINKER_FLAGS};${CMAKE_EXE_LINKER_FLAGS_${_configuration}}")
    if(_flags MATCHES "--wrap|--defsym|-Bsymbolic")
        message(FATAL_ERROR "Shared fixture ${application} acquired symbol-rewriting link options")
    endif()
    list(FIND _libraries glimmer_execution_service_runtime _runtime_index)
    if(_runtime_index EQUAL -1)
        return() # Multiple test modes register the same application.
    endif()
    list(REMOVE_AT _libraries ${_runtime_index})
    list(INSERT _libraries ${_runtime_index} glimmer_execution_test_runtime)
    set_property(TARGET "${application}" PROPERTY LINK_LIBRARIES "${_libraries}")
endfunction()
