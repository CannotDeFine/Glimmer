add_executable(glimmer main.cc)

set_target_properties(glimmer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(glimmer
    PRIVATE
        Glimmer::project_options
        spdlog::spdlog
)

glimmer_enable_clang_tidy(glimmer)

add_executable(glimmer_control_service control_service_main.cc)

set_target_properties(glimmer_control_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(glimmer_control_service
    PRIVATE
        Glimmer::backend
        Glimmer::control
        Glimmer::project_options
)

glimmer_enable_clang_tidy(glimmer_control_service)

add_executable(glimmer_control_client control_client_main.cc)

set_target_properties(glimmer_control_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(glimmer_control_client
    PRIVATE
        Glimmer::control
        Glimmer::project_options
)

glimmer_enable_clang_tidy(glimmer_control_client)

if(BUILD_TESTING)
    add_test(NAME glimmer_control_service_help COMMAND glimmer_control_service --help)
    set_tests_properties(glimmer_control_service_help PROPERTIES
        PASS_REGULAR_EXPRESSION "Usage:"
    )

    add_test(NAME glimmer_control_client_help COMMAND glimmer_control_client --help)
    set_tests_properties(glimmer_control_client_help PROPERTIES
        PASS_REGULAR_EXPRESSION "Usage:"
    )
endif()

if(BUILD_TESTING)
    add_test(NAME glimmer_starts COMMAND glimmer)
    set_tests_properties(glimmer_starts PROPERTIES
        PASS_REGULAR_EXPRESSION "Glimmer started"
    )
endif()

add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(control)
