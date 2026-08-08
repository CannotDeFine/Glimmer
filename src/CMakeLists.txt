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

if(BUILD_TESTING)
    add_test(NAME glimmer_starts COMMAND glimmer)
    set_tests_properties(glimmer_starts PROPERTIES
        PASS_REGULAR_EXPRESSION "Glimmer started"
    )
endif()

add_subdirectory(core)
add_subdirectory(control)
