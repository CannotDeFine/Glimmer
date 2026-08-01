add_executable(glimmer main.cc)

set_target_properties(glimmer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

target_link_libraries(glimmer
    PRIVATE
        Glimmer::project_options
        spdlog::spdlog
)
